"""
hydrology_erosion.py
水文与侵蚀模块

输入：经海拔模块处理过的 World（elevation 已生成）。
输出：同一个 World，新增岩石硬度/湿度/河网/沉积图层，
      elevation 被侵蚀过程修改，land_mask 相应更新。

工作流：
    1.  阈值法更新海陆掩膜（唯一参数 sea_level；暂假设所有水体均为淡水，
        不做小水体清除，不做连通域分析）
    2.  岩石硬度图层（高频二维柏林噪声 → 0~255 整数；
        造山边缘叠加高斯场加硬，保护山脊不被侵蚀）
    3.  气候带图层（模拟 0~70°N：地图底部=赤道、顶部=70°N，
        依纬度分六带——赤道低压(0~10)/信风(10~25)/副热带高压
        (25~35)/地中海(35~42)/西风(42~55)/副极地低压(55~70)，
        仅纬度相关）
    4.  湿度图层（四级：全湿润/半湿润/半干旱/干旱，水体完全
        湿润、陆地默认干旱。气候带给基底，海岸带（距离变换+
        一维柏林噪声扰动边缘）、山脉雨影（靠海侧多升、靠内陆
        侧少升）、西风带（自西缘衰减）与季风（东缘×25°N
        双距离，季风区内不受气候带限制、四级皆可出现）提供
        增益；最后强制相邻区域等级差≤1，缺失中间等级自动补带）
    5.  河流生成（默认空间殖民算法 SCA 简化版：非山脉陆地随机散布
        吸引点、裂谷加撒吸引点使其有较大机会成为河道、
        海岸河口放置根节点，纯吸引方向生长，河道平直；
        反平行间距控制生成叶脉状分叉；唯一海拔约束为单步下切
        容差；山脉（山脊中线邻近区域，距离限制可调）标记为
        mountain_mask 图层，河流暂时终止于其边缘；可选旧版 DLA 算法。
        流量 = 上游汇流计数，离入海口越近越大）
    6.  水力侵蚀（湿度 × 坡度 × 可蚀性，部分向邻域再沉积）
    7.  风力侵蚀（干燥度 × 可蚀性，磨蚀降低海拔）
    8.  河流侵蚀与沉积（河流功率定律下切；
        坡度突变处形成山前冲积扇；河口形成三角洲，
        三角洲避让河道格，不改变入海点）
    9.  海拔增量钳制、应用，重算海陆掩膜，输出报告

随机数约定：全部来自 world.rng（世界种子派生的统一生成器），
本模块不自行播种。
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation, distance_transform_edt, gaussian_filter, uniform_filter,
)
from scipy.spatial import cKDTree

from world_core import World, PerlinNoise1D, PerlinNoise2D, grow_dla

# 八邻域偏移与对应距离
_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# 侵蚀基准面余量：任何侵蚀不得把陆地切到 海平面 + 该值 以下
_BASE_MARGIN = 0.05


# ============================================================
# 功能函数 1：阈值法更新海陆掩膜
# ============================================================
def _update_water(world: World, sea_level: Optional[float]) -> None:
    """
    阈值法划分海陆：elevation > sea_level 为陆地，反之为水体。
    本项目暂假设所有水体均为淡水；不做小水体清除与连通域分析。

    sea_level 为 None 时使用 world.sea_level；否则写回 world.sea_level。
    """
    if sea_level is not None:
        world.sea_level = float(sea_level)
    world.land_mask[...] = world.elevation > world.sea_level


# ============================================================
# 功能函数 2：岩石硬度图层（高频柏林噪声 + 造山边缘高斯加硬）
# ============================================================
def _generate_rock_hardness(
    world: World,
    rock_freq: float,
    rock_octaves: int,
    rock_lacunarity: float,
    rock_hardness_max: int,
    mountain_hardness_boost: float,
    mountain_boost_sigma: float,
) -> Dict[str, float]:
    """
    生成岩石硬度图层（0~rock_hardness_max 的整数）。

    高频二维柏林噪声归一化后映射到整数硬度；随后对所有
    plate_boundaries == 4 的造山边缘（无论高山低山）叠加高斯场，
    使山脊周围岩石显著加硬，避免侵蚀把山脊削平。
    """
    h, w = world.shape
    perlin = PerlinNoise2D(world.rng)
    yv, xv = np.mgrid[0:h, 0:w]
    nx = xv * rock_freq / w
    ny = yv * rock_freq / h
    noise = perlin.octave_noise(nx, ny, rock_octaves, rock_lacunarity, 0.5)

    nmin, nmax = float(noise.min()), float(noise.max())
    if nmax > nmin:
        norm = (noise - nmin) / (nmax - nmin)
    else:
        norm = np.zeros_like(noise)
    hardness = norm.astype(np.float64) * rock_hardness_max

    # ---- 造山边缘高斯加硬 ----
    if mountain_hardness_boost > 0 and mountain_boost_sigma > 0:
        edge_ys, edge_xs = np.nonzero(world.plate_boundaries == 4)
        if len(edge_xs) > 0:
            boost_field = np.zeros((h, w), dtype=np.float64)
            sigma = float(mountain_boost_sigma)
            cutoff = max(1.0, sigma * 3)
            inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
            for py, px in zip(edge_ys, edge_xs):
                x_min = max(0, int(px - cutoff))
                x_max = min(w, int(px + cutoff) + 1)
                y_min = max(0, int(py - cutoff))
                y_max = min(h, int(py + cutoff) + 1)
                xs = np.arange(x_min, x_max)
                ys = np.arange(y_min, y_max)
                mxv, myv = np.meshgrid(xs, ys)
                dist_sq = (mxv - px) ** 2 + (myv - py) ** 2
                # 取最大值而非叠加：避免密集边缘处高斯场堆叠饱和
                patch = boost_field[y_min:y_max, x_min:x_max]
                np.maximum(
                    patch,
                    mountain_hardness_boost * np.exp(-dist_sq * inv_two_sigma2),
                    out=patch,
                )
            hardness = hardness + boost_field

    hardness = np.clip(hardness, 0, rock_hardness_max)
    world.rock_hardness[...] = np.rint(hardness).astype(np.int16)

    rh = world.rock_hardness
    return {
        "min": float(rh.min()),
        "max": float(rh.max()),
        "mean": float(rh.mean()),
    }


# ============================================================
# 功能函数 3：气压带/气候带图层（仅纬度相关，北半球六带）
# ============================================================
# 气候带编码
_BELT_EQUATORIAL = 0        # 赤道低气压带（0~10°N）
_BELT_TRADE = 1             # 信风带（10~25°N）
_BELT_SUBTROPICAL_HIGH = 2  # 副热带高气压带（25~35°N）
_BELT_MEDITERRANEAN = 3     # 地中海带（35~42°N）
_BELT_WESTERLIES = 4        # 西风带（42~55°N）
_BELT_SUBPOLAR = 5          # 副极地低气压带（55~70°N）

_BELT_NAMES = ("equatorial_low", "trade_wind", "subtropical_high",
               "mediterranean", "westerlies", "subpolar_low")

# 各带北界（°N）：地图底部 = 0°（赤道），顶部 = 70°N
_BELT_BOUNDARIES = (10.0, 25.0, 35.0, 42.0, 55.0)
_MAP_TOP_LAT = 70.0

# 各气候带陆地湿度基底与海岸增益（带内基底等级, 海岸带增益等级）
# 等级：0=干旱 1=半干旱 2=半湿润 3=全湿润
_BELT_HUMIDITY_BASE = (
    (2.0, 1.0),   # 赤道低压：只有湿润/半湿润——内陆半湿润，沿海全湿润
    (0.0, 1.0),   # 信风带：内陆干旱，沿海半干旱
    (0.0, 1.0),   # 副热带高压：内陆干旱，沿海半干旱
    (1.0, 1.0),   # 地中海带：内陆半干旱，沿海半湿润
    (2.0, 0.0),   # 西风带：基底半湿润（全湿润由西风距离机制贡献）
    (1.0, 2.0),   # 副极地低压：内陆半干旱，沿海可经过渡带达全湿润
)

# 四个湿度等级的代表值（0~100）：干旱/半干旱/半湿润/全湿润；水体 = 100
_HUMIDITY_LEVEL_VALUES = (15.0, 40.0, 65.0, 90.0)


def _generate_pressure_belts(world: World) -> Dict[str, Any]:
    """
    生成气压带/气候带图层（int8，仅与纬度相关）。

    模拟范围为 0°~70°N：地图底部 = 赤道（0°），顶部 = 70°N。
    自赤道向高纬依次为：赤道低气压带(0) → 信风带(1) →
    副热带高气压带(2) → 地中海带(3) → 西风带(4) →
    副极地低气压带(5)。边界固定于 10/25/35/42/55°N。
    """
    h, w = world.shape
    # 行号 → 纬度：顶部 y=0 为 70°N，底部 y=h-1 为 0°
    lat = _MAP_TOP_LAT * (1.0 - np.arange(h, dtype=np.float64) / max(h - 1, 1))
    belt_row = np.digitize(lat, _BELT_BOUNDARIES).astype(np.int8)
    world.pressure_belt[...] = np.broadcast_to(belt_row[:, None], (h, w))

    belts = world.pressure_belt
    report: Dict[str, Any] = {"top_latitude": _MAP_TOP_LAT}
    for b, name in enumerate(_BELT_NAMES):
        report[f"cells_{name}"] = int((belts == b).sum())
    return report


# ============================================================
# 功能函数 4：湿度图层（气候带基底 + 海岸/地形/西风/季风增益
#             + 相邻等级过渡强制）
# ============================================================
def _enforce_adjacent_levels(level: np.ndarray, band: float) -> np.ndarray:
    """
    相邻等级过渡强制：任意相邻（八邻域）像素的湿度等级差不得
    超过 1；不满足时在低等级一侧自动补出中间等级带。

    实现为"灰度形态学膨胀"：等级 × band 作为初始高度做多源
    BFS，每向外传播一格高度减 1、逐格取最大值，最后除回等级。
    传播斜率保证相邻像素等级差 ≤ 1；高等級区域（含水体）向外
    投出逐级递减的过渡环——例如全湿润水体旁的干旱陆地会依次
    补出半湿润环与半干旱环。只升不降：低等级侧被抬升补带，
    高等级区域本身不变。band 为每级过渡带的宽度（格，≥1）。
    """
    h, w = level.shape
    scale = max(1, int(round(band)))
    M = level.astype(np.int32) * scale
    for _ in range(3 * scale):  # 最高 3 级，影响最多传播 3×scale 格
        padded = np.pad(M, 1, mode="edge")
        nb = np.full_like(M, -1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                np.maximum(
                    nb,
                    padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w],
                    out=nb,
                )
        np.maximum(M, nb - 1, out=M)
    return (M // scale).astype(np.int8)


def _compute_humidity(
    world: World,
    coastal_humidity_width: float,
    coastal_noise_amp: float,
    coastal_noise_freq: float,
    mountain_min_elev: float,
    mountain_effect_radius: float,
    mountain_sea_boost: float,
    mountain_inland_boost: float,
    westerly_reach_frac: float,
    westerly_boost: float,
    monsoon_lat: float,
    monsoon_lat_range: float,
    monsoon_reach_frac: float,
    monsoon_boost: float,
    transition_band: float,
) -> Dict[str, Any]:
    """
    生成湿度图层（四级：干旱 0 / 半干旱 1 / 半湿润 2 / 全湿润 3，
    再映射为代表值写入 humidity；水体 = 100）。

    计算流程（除过渡强制外均为连续"等级单位"加减，最后取整）：
        1. 气候带基底：陆地默认干旱，按所在气候带取基底等级
           （_BELT_HUMIDITY_BASE 表）；
        2. 海岸增益：距水距离 ≤ coastal_humidity_width 的陆地
           按所在带增益表升湿——带宽用一维柏林噪声逐对角线
           （x+y 索引）扰动，避免距离变换产生的生硬边缘；
        3. 雨影效应：海拔 ≥ mountain_min_elev 的陆地计为山体；
           每个陆地像素找最近山体格，比它与山体谁离水更近——
           更近（靠海一侧）加 mountain_sea_boost，更远（靠内陆
           一侧）加 mountain_inland_boost，作用范围
           mountain_effect_radius；
        4. 西风带额外湿度：西风带像素按到地图西缘的距离线性
           衰减加成（范围 westerly_reach_frac × 图宽）；
        5. 季风：f1 = 到 monsoon_lat° 纬线的距离衰减
           （monsoon_lat_range 为度宽），f2 = 到地图东缘的距离
           衰减（monsoon_reach_frac × 图宽）；季风区（f1×f2>0）
           内以季风等级（monsoon_boost × f1 × f2）替换气候带
           基底——不受气候带限制，四级湿度都可出现；海岸与
           地形增益照常叠加；
        6. 相邻过渡强制（_enforce_adjacent_levels）：任意相邻
           像素等级差 ≤ 1，缺失的中间等级自动补带（水体参与，
           海岸因此总有 全湿润→半湿润→半干旱 的过渡环）。
    """
    h, w = world.shape
    land = world.land_mask
    water = ~land
    elev = world.elevation.astype(np.float64)
    belts = world.pressure_belt
    yv, xv = np.mgrid[0:h, 0:w]

    # ---- 距水距离（距离变换；海岸线即距水 1 格内的陆地）----
    dist_water = distance_transform_edt(land)

    # ---- 1. 气候带基底 ----
    base_tab = np.asarray(_BELT_HUMIDITY_BASE, dtype=np.float64)  # (6, 2)
    belt_base = base_tab[belts, 0]
    belt_coastal_gain = base_tab[belts, 1]

    # ---- 2. 海岸增益（带宽经一维柏林噪声扰动）----
    # 噪声表按对角线（x+y）索引：只需 h+w 次标量求值，
    # 扰动后的海岸带边缘呈有机起伏而非生硬直线
    perlin1d = PerlinNoise1D(world.rng)
    k = np.arange(h + w, dtype=np.float64)
    noise_tbl = np.array([
        perlin1d.octave_noise(t * coastal_noise_freq / max(h + w, 1),
                              3, 2.0, 0.5)
        for t in k
    ])
    coastal_threshold = (float(coastal_humidity_width)
                         + float(coastal_noise_amp) * noise_tbl[xv + yv])
    coastal_mask = land & (dist_water <= np.maximum(coastal_threshold, 0.0))
    coastal_term = belt_coastal_gain * coastal_mask

    # ---- 3. 雨影效应（山脉靠海侧多升、靠内陆侧少升）----
    mtn = land & (elev >= float(mountain_min_elev))
    mtn_boost = np.zeros((h, w), dtype=np.float64)
    if mountain_effect_radius > 0 and mtn.any():
        dist_mtn, idx = distance_transform_edt(~mtn, return_indices=True)
        within = land & (dist_mtn <= float(mountain_effect_radius))
        mtn_dw = dist_water[idx[0], idx[1]]
        sea_side = dist_water <= mtn_dw  # 比山体离水更近 = 靠海一侧
        mtn_boost[within] = np.where(
            sea_side, float(mountain_sea_boost),
            float(mountain_inland_boost))[within]

    # ---- 4. 西风带额外湿度（自西缘向东线性衰减）----
    reach_west = max(float(westerly_reach_frac) * w, 1e-6)
    f_west = np.clip(1.0 - xv / reach_west, 0.0, 1.0)
    westerly_extra = np.where(belts == _BELT_WESTERLIES,
                              float(westerly_boost) * f_west, 0.0)

    # ---- 5. 季风（东侧；双距离共同决定）----
    rows_per_deg = max(h - 1, 1) / _MAP_TOP_LAT
    y_mono = (1.0 - float(monsoon_lat) / _MAP_TOP_LAT) * (h - 1)
    f1 = np.clip(1.0 - np.abs(yv - y_mono)
                 / max(float(monsoon_lat_range) * rows_per_deg, 1e-6),
                 0.0, 1.0)
    reach_east = max(float(monsoon_reach_frac) * w, 1e-6)
    f2 = np.clip(1.0 - ((w - 1) - xv) / reach_east, 0.0, 1.0)
    s_monsoon = f1 * f2                      # 季风强度 0~1
    monsoon_level = float(monsoon_boost) * s_monsoon
    monsoon_region = land & (s_monsoon > 0)
    # 季风区以季风等级替换气候带基底（四级皆可出现）
    base = np.where(monsoon_region, monsoon_level, belt_base)

    # ---- 合计并取整为离散等级 ----
    total = base + coastal_term + westerly_extra + mtn_boost
    level = np.clip(np.rint(total), 0, 3).astype(np.int8)
    level[water] = 3  # 水体完全湿润

    # ---- 6. 相邻等级过渡强制（缺失的中间等级自动补带）----
    level = _enforce_adjacent_levels(level, transition_band)

    # ---- 等级 → 代表值写入 ----
    values = np.asarray(_HUMIDITY_LEVEL_VALUES, dtype=np.float64)
    hum = values[level]
    hum[water] = 100.0
    world.humidity[...] = hum.astype(np.float32)

    hu = world.humidity
    report: Dict[str, Any] = {
        "min": float(hu.min()),
        "max": float(hu.max()),
        "mean": float(hu.mean()),
        "coastal_cells": int(coastal_mask.sum()),
        "mountain_cells": int(mtn.sum()),
        "monsoon_cells": int(monsoon_region.sum()),
    }
    for lv in range(4):
        report[f"land_level_{lv}"] = int(((level == lv) & land).sum())
    for b, name in enumerate(_BELT_NAMES):
        mask = (belts == b) & land
        report[f"mean_{name}"] = float(hu[mask].mean()) if mask.any() else 0.0
    return report


# ============================================================
# 功能函数 5 内部工具：河口播种（最小间距拒绝抽样）
# ============================================================
def _select_outlets(
    world: World,
    water: np.ndarray,
    num_outlets: int,
    outlet_min_spacing: float,
    outlet_coverage_spacing: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    在与水相邻的陆地像素（海岸线）中抽取河口种子。

    outlet_coverage_spacing 为 None 或 ≤ 0 时用随机模式（旧行为）：
    随机顺序 + 最小间距拒绝抽样，最多 num_outlets 个——
    河口分布不均，可能出现大片无河区域。

    否则用覆盖模式（最远点抽样 + 抖动）：首个河口随机，
    之后每轮在"距已选河口最远的海岸格"的最远 10% 候选带内随机
    取一个（抖动避免确定性格式），直到任意海岸格距最近河口
    不超过 outlet_coverage_spacing，或达到 num_outlets 上限。
    保证整条海岸线的河口覆盖，不会因河口稀少而出现无河区域；
    此模式下河口间距自然 ≥ outlet_coverage_spacing，
    outlet_min_spacing 不参与。
    """
    land = world.land_mask
    coast = land & binary_dilation(water)
    candidates = np.argwhere(coast)  # (y, x)
    if len(candidates) == 0 or num_outlets <= 0:
        return []

    rng = world.rng

    if outlet_coverage_spacing is None or outlet_coverage_spacing <= 0:
        # ---- 随机模式（旧行为）----
        order = rng.permutation(len(candidates))
        min_d2 = float(outlet_min_spacing) ** 2
        selected: List[Tuple[int, int]] = []
        for idx in order:
            y, x = int(candidates[idx][0]), int(candidates[idx][1])
            ok = True
            for sy, sx in selected:
                if (y - sy) ** 2 + (x - sx) ** 2 < min_d2:
                    ok = False
                    break
            if ok:
                selected.append((y, x))
                if len(selected) >= num_outlets:
                    break
        return selected

    # ---- 覆盖模式（最远点抽样 + 抖动）----
    cy = candidates[:, 0].astype(np.float64)
    cx = candidates[:, 1].astype(np.float64)
    first = int(rng.integers(len(candidates)))
    selected_idx = [first]
    # 每个海岸格到最近已选河口的距离（增量维护，每新选一个 O(n_coast)）
    dist = np.hypot(cy - cy[first], cx - cx[first])
    while len(selected_idx) < num_outlets:
        max_d = float(dist.max())
        if max_d <= float(outlet_coverage_spacing):
            break  # 海岸已被覆盖
        band = np.nonzero(dist >= max_d * 0.9)[0]  # 最远 10% 候选带
        i = int(band[rng.integers(len(band))])
        selected_idx.append(i)
        np.minimum(dist, np.hypot(cy - cy[i], cx - cx[i]), out=dist)
    return [(int(candidates[i][0]), int(candidates[i][1]))
            for i in selected_idx]


# ============================================================
# 功能函数 4 内部工具：圆形沉积偏移表
# ============================================================
def _disc_offsets(radius: float) -> List[Tuple[int, int, float]]:
    """
    生成半径 radius 内的 (dy, dx, 高斯权重) 偏移表（不含中心格）。
    用于冲积扇/三角洲的沉积分摊。
    """
    r = int(math.ceil(radius))
    sigma2 = max(radius * 0.5, 1e-6) ** 2
    offs: List[Tuple[int, int, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dy * dy + dx * dx
            if d2 == 0 or d2 > radius * radius:
                continue
            wgt = math.exp(-d2 / (2.0 * sigma2))
            offs.append((dy, dx, wgt))
    return offs


# ============================================================
# 功能函数 4 内部工具：Bresenham 直线像素序列
# ============================================================
def _line_pixels(y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
    """返回 (y0, x0) 到 (y1, x1) 的 Bresenham 直线像素（含两端）。"""
    pixels: List[Tuple[int, int]] = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        pixels.append((y, x))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pixels


# ============================================================
# 功能函数 4：DLA 河流生成
# ============================================================
def _grow_rivers_dla(
    world: World,
    num_outlets: int,
    outlet_min_spacing: float,
    outlet_coverage_spacing: Optional[float],
    spawn_radius: float,
    spawn_elevation_bias: float,
    walk_elevation_bias: float,
    max_river_neighbors: int,
    max_particles: int,
    max_walk_steps: int,
    coastal_buffer: int,
    pool_rebuild_interval: int,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int, int]]]:
    """
    DLA（扩散限制聚集）河流生成。

    河流特有逻辑留在本函数：海岸线河口播种（最小间距拒绝抽样）、
    可行域构造（陆地，可选扣除 coastal_buffer 近水带）、
    流量累加（附着逆序一趟，离入海口越近越大）。
    生长机制本身（投放池、投放/运动倾向、附着可行域控制）
    由 world_core.grow_dla 通用引擎完成，与山脉等模块共用。

    返回 (统计字典, 附着日志)。附着日志为 [(y, x, py, px), ...]，
    按附着先后顺序排列，(py, px) 为父格（下游），河口为 (-1, -1)。
    """
    h, w = world.shape
    water = ~world.land_mask

    world.river_mask[...] = False
    world.river_strength[...] = 0.0

    # ---- 可行域：陆地（可选扣除近水带）----
    if coastal_buffer > 0:
        near_water = binary_dilation(water, iterations=coastal_buffer)
    else:
        near_water = np.zeros((h, w), dtype=bool)
    feasible = world.land_mask & ~near_water

    # ---- 1. 河口播种 ----
    outlets = _select_outlets(world, water, num_outlets, outlet_min_spacing,
                              outlet_coverage_spacing)

    # ---- 2. 通用 DLA 生长 ----
    result = grow_dla(
        world.rng, world.elevation, outlets, feasible,
        base_level=world.sea_level,
        spawn_radius=spawn_radius,
        spawn_elevation_bias=spawn_elevation_bias,
        walk_elevation_bias=walk_elevation_bias,
        max_neighbors=max_river_neighbors,
        max_particles=max_particles,
        max_walk_steps=max_walk_steps,
        pool_rebuild_interval=pool_rebuild_interval,
    )
    attach_log = result.attach_log
    world.river_mask[...] = result.cluster

    # ---- 3. 流量：附着逆序一趟累加 ----
    strength = np.zeros((h, w), dtype=np.float64)
    for y, x, _, _ in attach_log:
        strength[y, x] = 1.0
    for y, x, py, px in reversed(attach_log):
        if py >= 0:
            strength[py, px] += strength[y, x]
    world.river_strength[...] = strength.astype(np.float32)

    stats = dict(result.stats)
    stats["outlet_mode"] = ("coverage"
                            if outlet_coverage_spacing and outlet_coverage_spacing > 0
                            else "random")
    stats["num_outlets"] = len(outlets)
    stats["river_cells"] = len(attach_log)
    if attach_log:
        stats["total_strength"] = float(strength[result.cluster].sum())
        stats["max_strength"] = float(strength.max())
    else:
        stats["total_strength"] = 0.0
        stats["max_strength"] = 0.0
    return stats, attach_log


# ============================================================
# 功能函数 4b：空间殖民算法（SCA，简化版）河流生成
# ============================================================
def _grow_rivers_sca(
    world: World,
    num_outlets: int,
    outlet_min_spacing: float,
    outlet_coverage_spacing: Optional[float],
    num_attraction: int,
    d_i: float,
    d_k: float,
    D: float,
    L_min: float,
    max_step_drop: float,
    min_channel_spacing: float,
    parallel_angle_deg: float,
    spacing_exempt_generations: int,
    mountain_max_distance: float,
    mountain_buffer: int,
    rift_attraction: int,
    node_retries: int,
    max_iterations: int,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int, int]]]:
    """
    空间殖民算法（Space Colonization Algorithm, Runions et al. 2007）
    河流生成 —— 简化版。

    基础算法：
        初始化：非山脉陆地内均匀随机散布吸引点 M，并在裂谷像素
        （plate_boundaries == 5；fallback ridge_id < 0）上加撒
        rift_attraction 个吸引点——裂谷是板块离散形成的构造低地，
        加撒吸引点把河网拉进裂谷，使其有较大机会成为河道；
        海岸线河口放置根节点（每条河一个，共享同一吸引点场 →
        河流自然争夺流域、互不交叉）。
        迭代：
            1. 关联：每个活跃吸引点 m 找到距离最近的节点 n，
               若 distance(m, n) < d_i 则加入 S(n)；
            2. 生长：对每个 S(n) 非空的节点 n，
               dir = unit(Σ unit(m − n))（纯吸引方向——
               无动量、无转角钳制、无梯度导向、无候选格择优），
               n_new = n + D·dir，取整落格，加边加节点；
            3. 删除：每个新节点诞生时，删除其周围 d_k 内的吸引点
               （与逐轮全局删除等价：旧节点诞生时已清理过邻域）。
        终止：无活跃吸引点 / 无生长前沿 / 连续数轮停滞 /
              达到迭代上限。
        后处理 Trimming：自每个叶节点回溯，累计边长直至根或首个
        交汇点（≥2 个存活子节点），该段总长 < L_min 则整段剪除
        （回溯到根的情形即删除整条短河）。

    河道形态控制：
        · 平直：生长方向完全由吸引点平均方向决定，无动量项、
          无正弦插值、无择优函数，河道相对平直；
        · 反平行（叶脉状）：新节点距任何已有节点（父链上最近
          spacing_exempt_generations 代祖先豁免）小于
          min_channel_spacing、且走向与之平行（方向夹角 <
          parallel_angle，头对头逆行也算平行）时，该分支永久
          终止——河道不近距离平行；近距但成角度的分流不受限，
          支流自干流叉出呈叶脉状；建议 min_channel_spacing ≥ d_k；
        · 唯一海拔约束：新格海拔 < 本格海拔 − max_step_drop 不可行
          （河道自河口向上游生长应大致爬升，容差用于越过噪声凹坑），
          其余海拔启发（梯度导向/关联容差/源头截止）全部取消；
        · 候选格被其他河道占用时本轮受阻（可重试）→ 河道互不交叉；
        · 其他不可行情形（越界/下水/进山/下坡）本轮受阻，连续
          node_retries 轮受阻的末梢退出生长前沿。

    山脉边界（山区河流暂未建模）：
        山脉 = 旧版 DLA 梳齿纹样覆盖区（plate_boundaries == 4；
        缺失则用 ridge_elevation > 0）与山脊中线邻近带
        （ridge_line_mask 自定义图层，碰撞山脊中线本体，不含梳齿
        与裂谷，周围 mountain_max_distance 格以内）的合取——
        梳齿纹样里只有离山脊线足够近的部分才算山脉，远端不再
        封山。ridge_line_mask 图层缺失时无法施加距离限制，退回
        旧行为（整个梳齿覆盖区）。山脉标记限于陆地（水下部分
        对河流无意义）。生长前写入 mountain_mask 自定义图层。
        吸引点不撒入山脉（外扩 mountain_buffer 格），节点不可
        进入，河流暂时终止于其边缘。

    返回 (统计字典, 附着日志)，附着日志与 DLA 版同构：
    [(y, x, py, px), ...]，(py, px) 为下游父格，河口为 (-1, -1)，
    父格必先于子格出现（供侵蚀模块逆序做上游→下游拓扑遍历）。
    """
    h, w = world.shape
    rng = world.rng
    land = world.land_mask
    water = ~land
    elev = world.elevation.astype(np.float64)
    eps = 1e-9

    world.river_mask[...] = False
    world.river_strength[...] = 0.0

    # ---- 0. 山脉标记：旧版梳齿覆盖区 ∩ 山脊中线邻近带 的合取 ----
    # 旧版山脉：DLA 梳齿纹样覆盖区（plate_boundaries == 4，含原初
    # 山脊与贴印梳齿；fallback 用自定义图层 ridge_elevation > 0）
    tooth = world.plate_boundaries == 4
    if not tooth.any():
        ridge_elev = world.get_layer("ridge_elevation")
        if ridge_elev is not None:
            tooth = ridge_elev > 0
        else:
            tooth = np.zeros((h, w), dtype=bool)
    # 距离限制带：山脊中线（ridge_line_mask 图层）周围
    # mountain_max_distance 格以内
    ridge_line = world.get_layer("ridge_line_mask")
    if ridge_line is not None and ridge_line.any():
        if mountain_max_distance > 0:
            near_line = (distance_transform_edt(~ridge_line)
                         <= float(mountain_max_distance))
        else:
            near_line = ridge_line.astype(bool)
        # 合取：只有既是梳齿覆盖区、又离山脊线足够近的像素才算山脉
        mountain = tooth & near_line
    else:
        # ridge_line_mask 图层缺失：无法施加距离限制，退回旧行为
        mountain = tooth
    # 山脉标记限于陆地：水下部分对河流无意义，也避免查看器在海里画山
    mountain &= land
    # 写入自定义图层，供查看器/下游模块使用
    if world.get_layer("mountain_mask") is None:
        world.add_layer("mountain_mask", np.bool_, False)
    world.get_layer("mountain_mask")[...] = mountain

    # 生长可行域：非山脉的陆地。吸引点只撒在这里，节点也只能进入
    # 这里——河流到达山脉边缘即自然终止（山区河流暂未建模）
    if mountain_buffer > 0:
        blocked = binary_dilation(mountain, iterations=int(mountain_buffer))
    else:
        blocked = mountain
    river_land = land & ~blocked

    # ---- 1. 初始化：非山脉陆地内均匀随机散布吸引点 M；
    # 裂谷像素上额外加撒 rift_attraction 个吸引点 ----
    land_cells = np.argwhere(river_land)
    n_uniform = int(min(max(num_attraction, 0), len(land_cells)))
    if n_uniform > 0:
        pick = rng.choice(len(land_cells), size=n_uniform, replace=False)
        att_y = land_cells[pick, 0].astype(np.float64)
        att_x = land_cells[pick, 1].astype(np.float64)
    else:
        att_y = np.zeros(0)
        att_x = np.zeros(0)

    # 裂谷 = 板块离散形成的构造低地（plate_boundaries == 5，含裂谷
    # 中线与窄梳齿；fallback 用 ridge_id < 0）。在裂谷像素上加撒
    # 吸引点，把河网拉进裂谷——裂谷有较大机会成为河道
    rift = world.plate_boundaries == 5
    if not rift.any():
        rift = world.ridge_id < 0
    rift_cells = np.argwhere(rift & river_land)
    n_rift = int(min(max(rift_attraction, 0), len(rift_cells)))
    if n_rift > 0:
        pick = rng.choice(len(rift_cells), size=n_rift, replace=False)
        att_y = np.concatenate(
            [att_y, rift_cells[pick, 0].astype(np.float64)])
        att_x = np.concatenate(
            [att_x, rift_cells[pick, 1].astype(np.float64)])
    n_att = n_uniform + n_rift
    att_active = np.ones(n_att, dtype=bool)

    # 吸引点静态桶（删除步骤的范围查询；桶格 ≥ d_k 保证 3×3 邻域覆盖）
    pcell = max(float(d_k), 1.0)
    pbuckets: Dict[Tuple[int, int], List[int]] = {}
    for mi in range(n_att):
        key = (int(att_y[mi] // pcell), int(att_x[mi] // pcell))
        pbuckets.setdefault(key, []).append(mi)

    def kill_around(fy: float, fx: float) -> int:
        """删除 (fy, fx) 周围 d_k 内的活跃吸引点，返回删除数。"""
        killed = 0
        cy, cx = int(fy // pcell), int(fx // pcell)
        d_k2 = float(d_k) ** 2
        for by in (cy - 1, cy, cy + 1):
            for bx in (cx - 1, cx, cx + 1):
                for mi in pbuckets.get((by, bx), ()):
                    if not att_active[mi]:
                        continue
                    dy = att_y[mi] - fy
                    dx = att_x[mi] - fx
                    if dy * dy + dx * dx < d_k2:
                        att_active[mi] = False
                        killed += 1
        return killed

    # ---- 2. 初始化：河口根节点 N ----
    outlets = _select_outlets(world, water, num_outlets, outlet_min_spacing,
                              outlet_coverage_spacing)
    pos_y: List[float] = []         # 浮点位置（生长运动学）
    pos_x: List[float] = []
    pix_y: List[int] = []           # 占据的像素格（海拔约束/避让/光栅化）
    pix_x: List[int] = []
    parent: List[int] = []          # 父节点索引，-1 = 根
    children: List[List[int]] = []
    alive: List[bool] = []
    retries: List[int] = []         # 连续受阻轮数（超限退出生长前沿）
    terminated: List[bool] = []     # 反平行间距终止（永久，不重试）
    node_at: Dict[Tuple[int, int], int] = {}

    # 节点空间桶（反平行间距查询；桶格 ≥ min_channel_spacing 保证
    # 3×3 邻域覆盖所有可能的近距离节点）
    ncell = max(float(min_channel_spacing), 1.0)
    nbuckets: Dict[Tuple[int, int], List[int]] = {}

    def add_node_bucket(idx: int) -> None:
        key = (int(pos_y[idx] // ncell), int(pos_x[idx] // ncell))
        nbuckets.setdefault(key, []).append(idx)

    def chain_ancestors(idx: int) -> set:
        """本节点及父链上最近 spacing_exempt_generations 代祖先
        （自身河道刚经过的路径，豁免反平行检查）。"""
        chain = set()
        cur = idx
        for _ in range(int(spacing_exempt_generations) + 1):
            if cur < 0 or cur in chain:
                break
            chain.add(cur)
            cur = parent[cur]
        return chain

    # 平行判定阈值：方向夹角 < parallel_angle（或 > 180°−parallel_angle，
    # 即头对头逆行）视为平行
    par_cos = math.cos(math.radians(float(parallel_angle_deg)))

    def too_close(idx: int, fy: float, fx: float, vy: float, vx: float) -> bool:
        """反平行检查：候选位置距任何非豁免节点小于 min_channel_spacing、
        且候选走向 (vy, vx) 与该节点处的河道走向平行则为真。
        近距但成角度（如支流自干流叉出）的分流不受限。"""
        exempt = chain_ancestors(idx)
        sp2 = float(min_channel_spacing) ** 2
        cy, cx = int(fy // ncell), int(fx // ncell)
        for by in (cy - 1, cy, cy + 1):
            for bx in (cx - 1, cx, cx + 1):
                for j in nbuckets.get((by, bx), ()):
                    if j in exempt:
                        continue
                    ddy = pos_y[j] - fy
                    ddx = pos_x[j] - fx
                    if ddy * ddy + ddx * ddx >= sp2:
                        continue
                    # 邻近节点 j 处的河道局部走向
                    if parent[j] >= 0:
                        jy = pos_y[j] - pos_y[parent[j]]
                        jx = pos_x[j] - pos_x[parent[j]]
                    elif children[j]:
                        jy = pos_y[children[j][0]] - pos_y[j]
                        jx = pos_x[children[j][0]] - pos_x[j]
                    else:
                        return True  # 尚无走向的孤立根：近距即冲突
                    jn = math.hypot(jy, jx)
                    if jn < eps:
                        return True
                    if abs(vy * jy + vx * jx) / jn >= par_cos:
                        return True
        return False

    for oy, ox in outlets:
        idx = len(pos_y)
        pos_y.append(float(oy))
        pos_x.append(float(ox))
        pix_y.append(oy)
        pix_x.append(ox)
        parent.append(-1)
        children.append([])
        alive.append(True)
        retries.append(0)
        terminated.append(False)
        node_at[(oy, ox)] = idx
        add_node_bucket(idx)
        kill_around(float(oy), float(ox))

    total_killed = int(n_att - att_active.sum())
    growth_rejected = 0
    spacing_stopped = 0
    reject_causes = [0, 0, 0, 0, 0]  # 越界/下水/进山/下坡/碰撞
    iterations = 0

    # ---- 3. 迭代生长（纯吸引方向：无动量、无择优、无地形启发）----
    stall = 0
    for it in range(max_iterations):
        iterations = it + 1
        active_idx = np.nonzero(att_active)[0]
        if len(active_idx) == 0:
            break

        # 生长前沿：未被间距终止、且连续受阻未超限的节点
        # （卡死末梢退出竞争，其附近的吸引点改由其他分支响应）
        front = [i for i in range(len(pos_y))
                 if not terminated[i] and retries[i] < node_retries]
        if not front:
            break

        # 3.1 关联：矢量化最近节点查询（cKDTree，每轮重建快照）
        fy_arr = np.fromiter((pos_y[i] for i in front), dtype=np.float64, count=len(front))
        fx_arr = np.fromiter((pos_x[i] for i in front), dtype=np.float64, count=len(front))
        tree = cKDTree(np.column_stack((fy_arr, fx_arr)))
        coords = np.column_stack((att_y[active_idx], att_x[active_idx]))
        dist, tidx = tree.query(coords, k=1, distance_upper_bound=float(d_i))
        S: Dict[int, List[int]] = {}
        for ai, d, ti in zip(active_idx, dist, tidx):
            if ti < len(front) and d <= float(d_i):
                S.setdefault(front[int(ti)], []).append(int(ai))

        # 3.2 生长（纯吸引方向，新格直接取整落格，无候选择优）
        grew: List[int] = []
        for n_idx, members in S.items():
            ny_, nx_ = pos_y[n_idx], pos_x[n_idx]
            # dir = unit(Σ unit(m − n))（等权平均方向）
            vy = vx = 0.0
            for mi in members:
                dy = att_y[mi] - ny_
                dx = att_x[mi] - nx_
                dd = math.hypot(dy, dx)
                if dd < eps:
                    continue
                vy += dy / dd
                vx += dx / dd
            nrm = math.hypot(vy, vx)
            if nrm < eps:
                continue
            vy /= nrm
            vx /= nrm

            fy = ny_ + D * vy
            fx = nx_ + D * vx
            qy, qx = int(round(fy)), int(round(fx))
            py_, px_ = pix_y[n_idx], pix_x[n_idx]

            if qy < 0 or qy >= h or qx < 0 or qx >= w:
                reject_causes[0] += 1  # 越界
            elif not land[qy, qx]:
                reject_causes[1] += 1  # 不下水
            elif blocked[qy, qx]:
                reject_causes[2] += 1  # 不进山：河流终止于山脉边缘
            elif elev[qy, qx] < elev[py_, px_] - max_step_drop:
                reject_causes[3] += 1  # 唯一海拔约束：不得明显下坡
            elif (qy, qx) in node_at:
                reject_causes[4] += 1  # 与其他河道相撞则避让
            elif too_close(n_idx, fy, fx, vy, vx):
                # 反平行：距其他河道过近且走向平行 → 该分支永久终止
                # （不重试），河道互不近距离平行，支流呈叶脉状分叉
                terminated[n_idx] = True
                spacing_stopped += 1
                continue
            else:
                new_idx = len(pos_y)
                pos_y.append(fy)
                pos_x.append(fx)
                pix_y.append(qy)
                pix_x.append(qx)
                parent.append(n_idx)
                children.append([])
                alive.append(True)
                retries.append(0)
                terminated.append(False)
                children[n_idx].append(new_idx)
                node_at[(qy, qx)] = new_idx
                add_node_bucket(new_idx)
                retries[n_idx] = 0
                grew.append(new_idx)
                continue
            # 走到这里 = 本轮受阻（可重试；连续超限则退出生长前沿）
            retries[n_idx] += 1
            growth_rejected += 1

        # 3.3 删除：新节点诞生处 d_k 内的吸引点
        killed = 0
        for ni in grew:
            killed += kill_around(pos_y[ni], pos_x[ni])
        total_killed += killed

        # 3.4 停滞检测
        if not grew and killed == 0:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0

    nodes_total = len(pos_y)

    # ---- 4. 后处理：Trimming（叶节点回溯修剪）----
    pruned_branches = 0
    for leaf in range(nodes_total):
        if not alive[leaf]:
            continue
        if any(alive[c] for c in children[leaf]):
            continue  # 非叶节点
        path = [leaf]
        total = 0.0
        cur = leaf
        while True:
            p = parent[cur]
            if p < 0:
                break  # 回溯到根：整段=整条河
            total += math.hypot(pos_y[cur] - pos_y[p], pos_x[cur] - pos_x[p])
            if sum(1 for c in children[p] if alive[c]) >= 2:
                break  # p 为交汇点：保留 p，只剪边缘枝条
            path.append(p)
            cur = p
        if total < L_min:
            for i in path:
                alive[i] = False
            pruned_branches += 1

    nodes_pruned = int(sum(1 for a in alive if not a))

    # ---- 5. 光栅化：BFS 沿存活树把边画成像素链 ----
    river_mask = world.river_mask
    attach_log: List[Tuple[int, int, int, int]] = []
    visited = set()
    dq: deque = deque()
    for i in range(nodes_total):
        if alive[i] and parent[i] < 0:
            p = (pix_y[i], pix_x[i])
            if p not in visited:
                visited.add(p)
                river_mask[p] = True
                attach_log.append((p[0], p[1], -1, -1))
            dq.append(i)
    while dq:
        n = dq.popleft()
        np_ = (pix_y[n], pix_x[n])
        for c in children[n]:
            if not alive[c]:
                continue
            cp = (pix_y[c], pix_x[c])
            prev = np_
            for p in _line_pixels(np_[0], np_[1], cp[0], cp[1])[1:]:
                if p not in visited:
                    visited.add(p)
                    river_mask[p] = True
                    attach_log.append((p[0], p[1], prev[0], prev[1]))
                prev = p
            dq.append(c)

    # ---- 6. 流量：附着逆序一趟累加（离入海口越近越大，同 DLA 版语义）----
    strength = np.zeros((h, w), dtype=np.float64)
    for y, x, _, _ in attach_log:
        strength[y, x] = 1.0
    for y, x, py, px in reversed(attach_log):
        if py >= 0:
            strength[py, px] += strength[y, x]
    world.river_strength[...] = strength.astype(np.float32)

    stats: Dict[str, Any] = {
        "algorithm": "sca",
        "outlet_mode": ("coverage"
                        if outlet_coverage_spacing and outlet_coverage_spacing > 0
                        else "random"),
        "num_outlets": len(outlets),
        "mountain_cells": int(mountain.sum()),
        "river_land_cells": int(river_land.sum()),
        "attraction_points": n_att,
        "attraction_uniform": n_uniform,
        "rift_cells": int(len(rift_cells)),
        "rift_attraction_points": n_rift,
        "attraction_killed": total_killed,
        "attraction_left": int(att_active.sum()),
        "iterations": iterations,
        "nodes_total": nodes_total,
        "nodes_pruned": nodes_pruned,
        "nodes_dormant": int(sum(
            1 for i in range(nodes_total)
            if not terminated[i] and retries[i] >= node_retries
        )),
        "branches_spacing_stopped": spacing_stopped,
        "pruned_branches": pruned_branches,
        "growth_rejected": growth_rejected,
        "reject_causes": {
            "bounds": reject_causes[0],
            "water": reject_causes[1],
            "mountain": reject_causes[2],
            "step_drop": reject_causes[3],
            "collision": reject_causes[4],
        },
        "river_cells": len(attach_log),
        "total_strength": float(strength[world.river_mask].sum()) if attach_log else 0.0,
        "max_strength": float(strength.max()) if attach_log else 0.0,
    }
    return stats, attach_log


# ============================================================
# 功能函数 5：水力侵蚀
# ============================================================
def _apply_hydraulic_erosion(
    world: World,
    elev: np.ndarray,
    hydraulic_K: float,
    hydraulic_iterations: int,
    hydraulic_deposit_ratio: float,
) -> Dict[str, float]:
    """
    水力侵蚀（片蚀/坡面过程）。

    每轮对每个陆地像素：
        侵蚀量 E = hydraulic_K × 湿度比 × 可蚀性 × 与最低邻居的高差
    其中 可蚀性 = 1 - 硬度/255。E 从本格扣除，其中
    hydraulic_deposit_ratio 比例以 3×3 邻域均摊方式再沉积
    （近似坡面短距离搬运），其余视为被水流带走。
    侵蚀不得切穿基准面（海平面 + 余量）。
    """
    if hydraulic_K <= 0 or hydraulic_iterations <= 0:
        return {"total_eroded": 0.0, "iterations": 0}

    land = world.land_mask
    sl = world.sea_level
    base = sl + _BASE_MARGIN
    hum = (world.humidity.astype(np.float64) / 100.0)
    erod = 1.0 - world.rock_hardness.astype(np.float64) / 255.0
    total_eroded = 0.0

    for _ in range(hydraulic_iterations):
        # 与最低邻居的高差（3×3 最小值含自身，故 drop >= 0）
        padded = np.pad(elev, 1, mode="edge")
        min_nb = np.full_like(elev, np.inf)
        for dy, dx in _D8:
            shifted = padded[1 + dy: 1 + dy + elev.shape[0],
                             1 + dx: 1 + dx + elev.shape[1]]
            np.minimum(min_nb, shifted, out=min_nb)
        drop = np.maximum(elev - min_nb, 0.0)

        E = hydraulic_K * hum * erod * drop
        E *= land
        # 不切穿基准面
        E = np.minimum(E, np.maximum(elev - base, 0.0))

        deposit = uniform_filter(E, size=3, mode="nearest") * hydraulic_deposit_ratio
        deposit *= land
        elev -= E
        elev += deposit
        total_eroded += float(E.sum()) - float(deposit.sum())

    return {"total_eroded": total_eroded, "iterations": hydraulic_iterations}


# ============================================================
# 功能函数 6：风力侵蚀
# ============================================================
def _apply_wind_erosion(
    world: World,
    elev: np.ndarray,
    wind_K: float,
    wind_iterations: int,
) -> Dict[str, float]:
    """
    风力侵蚀（风蚀磨蚀）。

    每轮对每个陆地像素：
        侵蚀量 E = wind_K × 干燥度 × 可蚀性
    干燥度 = 1 - 湿度比，即越干旱、岩性越软的地区风蚀越强。
    风蚀物质视为粉尘输出（不再沉积）。不切穿基准面。
    """
    if wind_K <= 0 or wind_iterations <= 0:
        return {"total_eroded": 0.0, "iterations": 0}

    land = world.land_mask
    sl = world.sea_level
    base = sl + _BASE_MARGIN
    dry = 1.0 - world.humidity.astype(np.float64) / 100.0
    erod = 1.0 - world.rock_hardness.astype(np.float64) / 255.0
    total_eroded = 0.0

    E_once = wind_K * dry * erod * land
    for _ in range(wind_iterations):
        E = np.minimum(E_once, np.maximum(elev - base, 0.0))
        elev -= E
        total_eroded += float(E.sum())

    return {"total_eroded": total_eroded, "iterations": wind_iterations}


# ============================================================
# 功能函数 7：河流侵蚀与沉积（下切 + 冲积扇 + 三角洲）
# ============================================================
def _apply_river_erosion(
    world: World,
    elev: np.ndarray,
    attach_log: List[Tuple[int, int, int, int]],
    river_K: float,
    river_m: float,
    river_n: float,
    fan_slope_threshold: float,
    fan_slope_drop_ratio: float,
    fan_min_strength: float,
    fan_deposition_ratio: float,
    fan_radius: int,
    delta_deposition_ratio: float,
    delta_radius: int,
) -> Dict[str, Any]:
    """
    河流侵蚀与沉积。

    下切（河流功率定律）：
        侵蚀量 E = river_K × strength^river_m × 坡度^river_n × 可蚀性
    坡度取本格到父格（下游）的高差/距离。侵蚀产物累加为泥沙通量，
    沿河向下游传递。下切不穿基准面（海平面 + 余量）。

    山前冲积扇（依赖海拔/坡度变化）：
        当某河道格满足 坡度突变 —— 到父格的坡度 < fan_slope_threshold，
        且其最陡子格来流坡度 > fan_slope_threshold × fan_slope_drop_ratio，
        且 strength >= fan_min_strength —— 判定为出山口扇顶，
        将当前泥沙通量的 fan_deposition_ratio 按高斯权重分摊到
        扇顶周围 fan_radius 内“不高于扇顶”的陆地格（避让河道格），
        deposition_type = 1。

    河口三角洲：
        河流到达水体（河口格）时，将剩余泥沙的 delta_deposition_ratio
        按高斯权重分摊到周围 delta_radius 的水下/贴水格（避让河道格，
        留出汊道空隙，因此三角洲不改变入海点），deposition_type = 2。
        沉积可抬升海床形成新陆地。

    未沉积的泥沙入海消失（不做质量守恒）。
    """
    stats: Dict[str, Any] = {
        "total_eroded": 0.0,
        "num_fans": 0,
        "fan_volume": 0.0,
        "num_deltas": 0,
        "delta_volume": 0.0,
    }
    if not attach_log or river_K < 0:
        return stats

    h, w = world.shape
    sl = world.sea_level
    base = sl + _BASE_MARGIN
    river = world.river_mask
    strength = world.river_strength.astype(np.float64)
    erod = 1.0 - world.rock_hardness.astype(np.float64) / 255.0

    sediment = np.zeros((h, w), dtype=np.float64)
    max_child_slope = np.zeros((h, w), dtype=np.float64)

    depo_type = world.deposition_type
    depo_th = world.deposition_thickness
    depo_type[...] = 0
    depo_th[...] = 0.0

    fan_offs = _disc_offsets(fan_radius) if fan_radius > 0 else []
    delta_offs = _disc_offsets(delta_radius) if delta_radius > 0 else []

    def deposit(cy: int, cx: int, amount: float,
                offs: List[Tuple[int, int, float]],
                mode: str, type_code: int) -> float:
        """按高斯权重把 amount 分摊到有效格，返回实际沉积量。"""
        if amount <= 0 or not offs:
            return 0.0
        apex_e = elev[cy, cx]
        cells: List[Tuple[int, int, float]] = []
        wsum = 0.0
        for dy, dx, wgt in offs:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if river[ny, nx]:
                continue  # 避让河道格（三角洲因此不改变入海点）
            if mode == "fan":
                # 只沉积在不高于扇顶的陆地上 → 自然形成下游方向扇面
                if elev[ny, nx] <= sl or elev[ny, nx] > apex_e + 1e-9:
                    continue
            else:  # delta：只沉积在水下或贴水带的低洼处
                if elev[ny, nx] > sl + 1.0:
                    continue
            cells.append((ny, nx, wgt))
            wsum += wgt
        if wsum <= 0:
            return 0.0
        per = amount / wsum
        deposited = 0.0
        for ny, nx, wgt in cells:
            add = per * wgt
            elev[ny, nx] += add
            depo_th[ny, nx] += add
            depo_type[ny, nx] = type_code
            deposited += add
        return deposited

    # 附着逆序 = 上游 → 下游 的拓扑序
    for y, x, py, px in reversed(attach_log):
        if py >= 0:
            dist = math.hypot(y - py, x - px) or 1.0
            slope = max(0.0, float(elev[y, x] - elev[py, px])) / dist
        else:
            slope = max(0.0, float(elev[y, x]) - sl)

        if py >= 0 and slope > max_child_slope[py, px]:
            max_child_slope[py, px] = slope

        # ---- 下切 ----
        E = river_K * (strength[y, x] ** river_m) * (slope ** river_n) * erod[y, x]
        E = min(E, max(0.0, float(elev[y, x]) - base))
        if E > 0:
            elev[y, x] -= E
            sediment[y, x] += E
            stats["total_eroded"] += E

        S = sediment[y, x]
        if S <= 0:
            continue

        # ---- 山前冲积扇 ----
        is_fan = (
            py >= 0
            and slope < fan_slope_threshold
            and max_child_slope[y, x] > fan_slope_threshold * fan_slope_drop_ratio
            and strength[y, x] >= fan_min_strength
        )
        if is_fan:
            deposited = deposit(y, x, S * fan_deposition_ratio, fan_offs, "fan", 1)
            if deposited > 0:
                stats["num_fans"] += 1
                stats["fan_volume"] += deposited
                S -= deposited

        # ---- 向下游传递 / 河口三角洲 ----
        if py >= 0:
            sediment[py, px] += S
        else:
            deposited = deposit(y, x, S * delta_deposition_ratio, delta_offs, "delta", 2)
            if deposited > 0:
                stats["num_deltas"] += 1
                stats["delta_volume"] += deposited
            # 剩余泥沙入海消失

    return stats


# ============================================================
# 主要进程函数：水文与侵蚀
# ============================================================
def generate_hydrology_erosion(
    world: World,
    sea_level: Optional[float] = None,
    # ── 岩石硬度 ──
    rock_freq: float = 12.0,
    rock_octaves: int = 4,
    rock_lacunarity: float = 2.0,
    rock_hardness_max: int = 255,
    mountain_hardness_boost: float = 150.0,
    mountain_boost_sigma: float = 5.0,
    # ── 气候带（0~70°N 六带，边界固定 10/25/35/42/55°N）──
    # ── 湿度：海岸增益 ──
    coastal_humidity_width: float = 8.0,   # 海岸湿度提升带宽度（格，默认较小）：距水距离 ≤ 带宽的陆地按所在气候带增益表升湿
    coastal_noise_amp: float = 8.0,        # 海岸带边缘一维柏林噪声扰动幅度（格）：避免距离变换产生生硬边缘
    coastal_noise_freq: float = 2.0,       # 扰动噪声频率（越大边缘起伏越细密）
    # ── 湿度：山脉雨影效应 ──
    mountain_min_elev: float = 60.0,       # 计入雨影效应的山体最低海拔（m）
    mountain_effect_radius: float = 10.0,  # 山体湿度影响半径（格）
    mountain_sea_boost: float = 1.0,       # 靠海一侧湿度增益（等级，0~3 连续）
    mountain_inland_boost: float = 0.3,    # 靠内陆一侧湿度增益（等级，应小于靠海侧）
    # ── 湿度：西风带 ──
    westerly_reach_frac: float = 0.5,      # 西风影响范围（占图宽比例）：默认 0.5 = 影响半张地图
    westerly_boost: float = 1.0,           # 西风最大额外湿度（等级）：使西风带西半侧升至全湿润
    # ── 湿度：季风（地图东侧，双距离共同决定）──
    monsoon_lat: float = 25.0,             # 季风中心纬度（°N）：到该纬线的距离参与衰减
    monsoon_lat_range: float = 15.0,       # 季风纬向影响半宽（°）
    monsoon_reach_frac: float = 0.333,     # 季风影响范围（占图宽比例）：默认 1/3 张地图
    monsoon_boost: float = 3.0,            # 季风最大湿度（等级）：季风区内替换气候带基底、四级皆可出现
    # ── 湿度：相邻等级过渡 ──
    humidity_transition_band: float = 4.0,  # 每级过渡带宽度（格）：相邻区域等级差强制 ≤1，缺失中间等级自动补带
    # ── 河流生成 ──
    num_outlets: int = 60,          # 河口（根节点）数量上限：覆盖模式下为上限，随机模式下为精确目标数；实际受海岸线长度制约
    outlet_min_spacing: float = 20.0,  # 河口最小间距（格）：仅随机模式生效，越小河口越密集
    outlet_coverage_spacing: Optional[float] = 40.0,  # 河口覆盖间距（格）：覆盖模式（默认）——任意海岸格距最近河口不超过该值，受 num_outlets 上限约束；建议 25~60，None/0=旧随机模式
    # 河流算法："sca"（空间殖民算法简化版，默认：纯吸引方向生长、河道平直，
    # 反平行间距控制生成叶脉状分叉，河流终止于山脉边缘）
    # 或 "dla"（旧版扩散限制聚集，河道呈灌木状分叉）
    river_algorithm: str = "sca",
    # ── SCA 河流（river_algorithm="sca" 时生效）──
    sca_num_attraction: int = 800,  # 吸引点数量：越多河网越密、支流越多；256² 建议 800~2000，512² 建议 3000~6000
    sca_d_i: float = 18.0,           # 影响半径（格）：节点只响应此距离内的吸引点；大→河道顺直平滑，小→蜿蜒扭曲
    sca_d_k: float = 5.0,            # 删除半径（格）：河道经过即清除附近吸引点；大→支流稀疏、河网开阔，小→支流密集
    sca_D: float = 1.5,              # 生长步长（格）：建议 1~3；小→河道细腻但节点多、速度慢
    sca_L_min: float = 30.0,         # 修剪阈值 L_min（格）：长度不足的边缘支流（或整条短河）被剪除
    sca_max_step_drop: float = 5.0,  # 唯一海拔约束：单步允许的最大海拔下降（米），容许越过噪声凹坑但不准逆坡下行；0=严格单调
    sca_min_channel_spacing: float = 6.0,  # 反平行间距（格）：新节点距其他河道近于该值且走向平行则分支永久终止，生成叶脉状分叉；建议 ≥ sca_d_k
    sca_parallel_angle: float = 30.0,      # 平行判定夹角（度）：近距河道方向夹角小于该值（或头对头）视为平行；0≈关闭间距控制，90=近距即终止
    sca_spacing_exempt: Optional[int] = None,  # 间距检查豁免的父链祖先代数（自身河道刚经过的路径）；None=按 spacing/D+6 自动
    sca_mountain_max_distance: float = 2.0,  # 山脉距离限制（格）：梳齿覆盖区 ∩ 山脊中线（ridge_line_mask 图层）邻近带 才算山脉（限陆地）；建议 2~8，0=仅中线上的梳齿
    sca_mountain_buffer: int = 2,    # 山脉禁入缓冲（格）：山脉区域外扩该格数，河流终止于其边缘；0=仅山脉本体
    sca_rift_attraction: int = 200,  # 裂谷加撒吸引点数：在裂谷（plate_boundaries == 5）像素上额外均匀撒点，裂谷有较大机会成为河道；0=关闭
    sca_max_iterations: Optional[int] = None,  # 迭代上限（性能兜底）：None=按 3.5×max(宽,高) 自动；正常在吸引点耗尽或停滞时提前结束
    sca_node_retries: int = 5,       # 节点连续受阻多少轮后退出生长前沿（卡死末梢放弃，吸引点让给其他分支）
    # ── DLA 河流（river_algorithm="dla" 时生效）──
    spawn_radius: float = 25.0,
    spawn_elevation_bias: float = 2.0,
    walk_elevation_bias: float = 1.0,
    max_river_neighbors: int = 3,
    max_particles: int = 20000,
    max_walk_steps: int = 400,
    coastal_buffer: int = 0,
    pool_rebuild_interval: int = 256,
    # ── 水力侵蚀 ──
    enable_hydraulic_erosion: bool = True,
    hydraulic_K: float = 0.02,
    hydraulic_iterations: int = 3,
    hydraulic_deposit_ratio: float = 0.5,
    # ── 风力侵蚀 ──
    enable_wind_erosion: bool = False,
    wind_K: float = 0.03,
    wind_iterations: int = 2,
    # ── 河流侵蚀与沉积 ──
    enable_river_erosion: bool = True,
    river_K: float = 0.02,
    river_m: float = 0.5,
    river_n: float = 1.0,
    fan_slope_threshold: float = 0.5,
    fan_slope_drop_ratio: float = 1.5,
    fan_min_strength: float = 20.0,
    fan_deposition_ratio: float = 0.5,
    fan_radius: int = 6,
    delta_deposition_ratio: float = 0.7,
    delta_radius: int = 4,
    # ── 稳定性 ──
    max_delta_per_cell: float = 5.0,
) -> Tuple[World, Dict[str, Any]]:
    """
    水文与侵蚀主函数：在海拔模块的结果上生成水文图层并施加侵蚀。

    Parameters
    ----------
    world : World
        经海拔模块处理过的世界实例（elevation 已生成）。
    sea_level : float, optional
        海平面。None 时使用 world.sea_level；否则写回 world.sea_level。
    rock_freq : float
        岩石硬度柏林噪声频率（高频 → 图样细碎）。
    rock_hardness_max : int
        硬度上限（默认 255，硬度范围 0~255 的整数）。
    mountain_hardness_boost / mountain_boost_sigma : float
        造山边缘高斯加硬的峰值与半径。
    气候带：模拟 0~70°N（底=赤道、顶=70°N）六带，边界固定
        10/25/35/42/55°N，无缩放参数。
    coastal_humidity_width / coastal_noise_amp / coastal_noise_freq :
        海岸湿度提升带宽度（格，默认较小）及其一维柏林噪声边缘
        扰动的幅度与频率。
    mountain_min_elev / mountain_effect_radius /
    mountain_sea_boost / mountain_inland_boost :
        雨影效应——海拔 ≥ mountain_min_elev 的陆地计为山体，
        半径内靠海一侧湿度增益 mountain_sea_boost、靠内陆一侧
        mountain_inland_boost（均为连续等级单位 0~3）。
    westerly_reach_frac / westerly_boost :
        西风带按到地图西缘距离线性衰减的额外湿度；默认影响
        半张地图（0.5×图宽）。
    monsoon_lat / monsoon_lat_range / monsoon_reach_frac /
    monsoon_boost :
        季风（东侧）——到 monsoon_lat°N 纬线与到地图东缘的
        双距离共同决定强度，默认影响 1/3 张地图；季风区内
        替换气候带基底，四级湿度皆可出现。
    humidity_transition_band : float
        相邻湿度区域等级差强制 ≤1 时每级过渡带的宽度（格）。
    num_outlets / outlet_min_spacing / outlet_coverage_spacing :
        河口（根节点）控制。覆盖模式（outlet_coverage_spacing > 0，
        默认）：最远点抽样，任意海岸格距最近河口不超过
        outlet_coverage_spacing，num_outlets 仅作上限——避免河口
        稀少导致大片区域无河；随机模式（None/0，旧行为）：
        随机顺序 + outlet_min_spacing 拒绝抽样，最多 num_outlets 个。
    river_algorithm : str
        河流生成算法："sca"（空间殖民算法，默认）或 "dla"（旧版）。
    sca_num_attraction / sca_d_i / sca_d_k / sca_D :
        SCA 基本参数：吸引点数量、影响半径、删除半径、生长步长。
    sca_L_min : float
        修剪阈值：长度不足的边缘支流或整条短河被剪除。
    sca_max_step_drop : float
        唯一海拔约束：单步允许的最大海拔下降（米），0 = 严格单调。
    sca_min_channel_spacing / sca_parallel_angle / sca_spacing_exempt :
        反平行参数：河道最小间距（格）、平行判定夹角（度）——近距且
        走向平行则分支永久终止，成角度的分流不受限，生成叶脉状分叉；
        以及间距检查豁免的父链祖先代数（None=自动）。
    sca_mountain_max_distance / sca_mountain_buffer :
        山脉范围参数：山脉 = 梳齿纹样覆盖区 ∩ 山脊中线邻近带
        （ridge_line_mask 图层周围 sca_mountain_max_distance 格
        以内，限陆地），建议 2~8（ridge_line_mask 缺失时不做
        距离限制，退回整个梳齿覆盖区）；
        sca_mountain_buffer 为山脉外扩禁入缓冲（格），
        河流终止于山脉边缘。
    sca_rift_attraction : int
        裂谷加撒吸引点数：在裂谷（plate_boundaries == 5，
        板块离散形成的构造低地）像素上额外均匀撒点，
        裂谷有较大机会成为河道；0 = 关闭。
    spawn_radius / spawn_elevation_bias : float
        投放池范围（流域范围）与高海拔投放偏好指数。
        注意：spawn_radius 不宜超过约 sqrt(max_walk_steps)（扩散可达距离），
        否则远处粒子难以走回河网附着，附着率会骤降。
    walk_elevation_bias : float
        粒子游走的高程启发强度（倾向走向高海拔）。
    max_river_neighbors : int
        可行域控制：八邻居中河流数达到该值的像素永久禁入。
    max_particles / max_walk_steps : int
        粒子总数与单粒子步数上限（性能兜底）。
    coastal_buffer : int
        距水该格数以内的陆地不可成为河道（0 = 关闭）。
    pool_rebuild_interval : int
        投放池快照每附着多少格重建一次。
    enable_hydraulic_erosion / hydraulic_K / hydraulic_iterations /
    hydraulic_deposit_ratio :
        水力侵蚀开关、强度、轮数与邻域再沉积比例。
    enable_wind_erosion / wind_K / wind_iterations :
        风力侵蚀开关、强度与轮数。
    enable_river_erosion / river_K / river_m / river_n :
        河流侵蚀开关与功率定律参数（E = K·A^m·S^n·可蚀性）。
    fan_slope_threshold / fan_slope_drop_ratio / fan_min_strength /
    fan_deposition_ratio / fan_radius :
        山前冲积扇的坡度突变判据、最小流量、沉积比例与半径。
    delta_deposition_ratio / delta_radius :
        河口三角洲的沉积比例与半径。
    max_delta_per_cell : float
        单格海拔总变化量钳制（±米）。

    Returns
    -------
    (World, dict)
        同一个世界实例与统计报告。
    """
    report: Dict[str, Any] = {}

    # ---------- 1. 阈值法海陆掩膜 ----------
    _update_water(world, sea_level)
    report["sea_level"] = world.sea_level

    # ---------- 2. 岩石硬度 ----------
    report["rock_hardness"] = _generate_rock_hardness(
        world, rock_freq, rock_octaves, rock_lacunarity,
        rock_hardness_max, mountain_hardness_boost, mountain_boost_sigma,
    )

    # ---------- 3. 气候带（0~70°N 六带）----------
    report["pressure_belts"] = _generate_pressure_belts(world)

    # ---------- 4. 湿度（气候带基底+海岸/雨影/西风/季风增益+相邻过渡强制）----------
    report["humidity"] = _compute_humidity(
        world,
        coastal_humidity_width, coastal_noise_amp, coastal_noise_freq,
        mountain_min_elev, mountain_effect_radius,
        mountain_sea_boost, mountain_inland_boost,
        westerly_reach_frac, westerly_boost,
        monsoon_lat, monsoon_lat_range, monsoon_reach_frac, monsoon_boost,
        humidity_transition_band,
    )

    # ---------- 5. 河流生成（SCA / DLA）----------
    if river_algorithm == "sca":
        if sca_max_iterations is None:
            sca_max_iterations = int(3.5 * max(world.shape))
        if sca_spacing_exempt is None:
            # 自动豁免代数：覆盖自身河道在间距半径内刚经过的路径
            # （间距 ÷ 步长 ≈ 路径格数，+6 余量供弯道使用）
            sca_spacing_exempt = int(sca_min_channel_spacing / max(sca_D, 1e-6)) + 6
        river_stats, attach_log = _grow_rivers_sca(
            world, num_outlets, outlet_min_spacing, outlet_coverage_spacing,
            sca_num_attraction, sca_d_i, sca_d_k, sca_D, sca_L_min,
            sca_max_step_drop, sca_min_channel_spacing, sca_parallel_angle,
            sca_spacing_exempt, sca_mountain_max_distance, sca_mountain_buffer,
            sca_rift_attraction, sca_node_retries, sca_max_iterations,
        )
    elif river_algorithm == "dla":
        river_stats, attach_log = _grow_rivers_dla(
            world, num_outlets, outlet_min_spacing, outlet_coverage_spacing,
            spawn_radius,
            spawn_elevation_bias, walk_elevation_bias, max_river_neighbors,
            max_particles, max_walk_steps, coastal_buffer, pool_rebuild_interval,
        )
    else:
        raise ValueError(f"未知河流算法: {river_algorithm!r}（可选 'sca' / 'dla'）")
    report["rivers"] = river_stats

    # ---------- 6~8. 侵蚀（在 float64 工作副本上累计）----------
    elev0 = world.elevation.astype(np.float64)
    elev = elev0.copy()

    if enable_hydraulic_erosion:
        report["hydraulic_erosion"] = _apply_hydraulic_erosion(
            world, elev, hydraulic_K, hydraulic_iterations, hydraulic_deposit_ratio,
        )
    else:
        report["hydraulic_erosion"] = {"total_eroded": 0.0, "iterations": 0}

    if enable_wind_erosion:
        report["wind_erosion"] = _apply_wind_erosion(
            world, elev, wind_K, wind_iterations,
        )
    else:
        report["wind_erosion"] = {"total_eroded": 0.0, "iterations": 0}

    if enable_river_erosion:
        report["river_erosion"] = _apply_river_erosion(
            world, elev, attach_log, river_K, river_m, river_n,
            fan_slope_threshold, fan_slope_drop_ratio, fan_min_strength,
            fan_deposition_ratio, fan_radius,
            delta_deposition_ratio, delta_radius,
        )
    else:
        report["river_erosion"] = {
            "total_eroded": 0.0, "num_fans": 0, "fan_volume": 0.0,
            "num_deltas": 0, "delta_volume": 0.0,
        }

    # ---------- 9. 增量钳制、应用、重算海陆 ----------
    delta = np.clip(elev - elev0, -max_delta_per_cell, max_delta_per_cell)
    new_elev = elev0 + delta
    old_land = world.land_mask.copy()
    world.elevation[...] = new_elev.astype(np.float32)
    world.land_mask[...] = new_elev > world.sea_level

    report["max_abs_delta"] = float(np.abs(delta).max()) if delta.size else 0.0
    report["new_land_cells"] = int((~old_land & world.land_mask).sum())
    report["new_water_cells"] = int((old_land & ~world.land_mask).sum())

    return world, report