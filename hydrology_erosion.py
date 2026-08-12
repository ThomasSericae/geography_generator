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
        元胞自动机扰动边缘）、山脉雨影（靠海侧多升、靠内陆
        侧少升）、西风（自西缘纬度45°N源点向东南北扩散的
        拉长椭圆风场，向东衰减慢、北向衰减快、南向探入地
        中海带西部，地中海带受小西风）与
        季风（东南→西北线性风，不作用于赤道低压带，区内
        最差半干旱）提供
        增益；带边界经元胞自动机扰动呈有机曲线；最后强制
        相邻区域等级差≤1，缺失中间等级自动补带；信风带与
        副热带高压带终检剔除海洋带来的升湿，带际过渡与
        季风带来的半湿润/湿润保留）
    5.  河流生成（默认空间殖民算法 SCA 简化版：非山脉陆地随机散布
        吸引点、裂谷加撒吸引点使其有较大机会成为河道、
        海岸河口放置根节点，纯吸引方向生长，河道平直；
        反平行间距控制生成叶脉状分叉；唯一海拔约束为单步下切
        容差；山脉（山脊中线邻近区域，距离限制可调）标记为
        mountain_mask 图层，河流暂时终止于其边缘；海岸线有
        河流规避距离（近岸不撒点、带内只许奔向本树河口）；
        吸引点按湿度加权（干旱不撒、半干旱稀化），河网密度
        随湿度变化；河口在几格宽的宽海岸带内随机撒点；
        可选旧版 DLA 算法。
        流量 = 上游汇流计数，离入海口越近越大；水流量单独
        成层 river_discharge——源头注入基础流量，湿润/临高山
        源头有加成，向下游累加，每个河道格有自己的水流量值）
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
    binary_dilation, distance_transform_edt, gaussian_filter,
    uniform_filter,
)
from scipy.spatial import cKDTree

from world_core import World, PerlinNoise2D, grow_dla, ca_edge_perturb

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
    (3.0, 0.0),   # 赤道低压：只有全湿润（且不受季风影响）
    (0.0, 1.0),   # 信风带：内陆干旱，沿海半干旱
    (0.0, 1.0),   # 副热带高压：内陆干旱，沿海半干旱
    (1.0, 1.0),   # 地中海带：内陆半干旱，沿海半湿润（另受小西风影响）
    (1.0, 0.0),   # 西风带：基底半干旱（最低湿度半干旱），其余由西风决定
    (1.0, 2.0),   # 副极地低压：内陆半干旱，沿海可经过渡带达全湿润
)

# 四个湿度等级的代表值（0~100）：干旱/半干旱/半湿润/全湿润；水体 = 100
_HUMIDITY_LEVEL_VALUES = (15.0, 40.0, 65.0, 90.0)


def _generate_pressure_belts(
    world: World,
    belt_edge_zone_width: float = 8.0,
    belt_edge_noise_prob: float = 0.20,
    belt_edge_iterations: int = 3,
) -> Dict[str, Any]:
    """
    生成气压带/气候带图层（int8，仅与纬度相关）。

    模拟范围为 0°~70°N：地图底部 = 赤道（0°），顶部 = 70°N。
    自赤道向高纬依次为：赤道低气压带(0) → 信风带(1) →
    副热带高气压带(2) → 地中海带(3) → 西风带(4) →
    副极地低气压带(5)。边界固定于 10/25/35/42/55°N。

    带边界扰动：纬度直线边界不自然，故对每条边界两侧
    ±belt_edge_zone_width 格的区域专门执行元胞自动机边缘
    扰动（world_core.ca_edge_perturb）——北侧向南凸、南侧向
    北凸双向撒噪点，多数规则平滑后边界呈有机曲线，跨界像素
    改标为邻侧气候带。belt_edge_zone_width / belt_edge_noise_prob
    任一 ≤ 0 时跳过（恢复直线边界）。
    """
    h, w = world.shape
    # 行号 → 纬度：顶部 y=0 为 70°N，底部 y=h-1 为 0°
    lat = _MAP_TOP_LAT * (1.0 - np.arange(h, dtype=np.float64) / max(h - 1, 1))
    belt_row = np.digitize(lat, _BELT_BOUNDARIES).astype(np.int8)
    belt_arr = np.broadcast_to(belt_row[:, None], (h, w)).copy()

    if belt_edge_zone_width > 0 and belt_edge_noise_prob > 0:
        yv = np.arange(h, dtype=np.float64)[:, None]
        for b, lat_b in enumerate(_BELT_BOUNDARIES):
            # 边界 b：belt b（南）与 b+1（北）之间，行号 y_b
            y_b = (1.0 - lat_b / _MAP_TOP_LAT) * (h - 1)
            zone = np.broadcast_to(
                np.abs(yv - y_b) <= float(belt_edge_zone_width), (h, w))
            orig = belt_arr
            north = orig > b
            # 双向扰动：北侧掩膜向南凸 + 南侧掩膜向北凸
            north = ca_edge_perturb(
                north, world.rng, noise_prob=belt_edge_noise_prob,
                expand_zone=zone, iterations=int(belt_edge_iterations))
            south = ca_edge_perturb(
                ~north, world.rng, noise_prob=belt_edge_noise_prob,
                expand_zone=zone, iterations=int(belt_edge_iterations))
            north = ~south
            belt_arr = orig.copy()
            # 只允许 b ↔ b+1 互换，区域内其他带的像素不受影响
            belt_arr[north & (orig == b)] = b + 1        # 跨入北侧 → 邻带
            belt_arr[(~north) & (orig == b + 1)] = b     # 跨入南侧 → 邻带

    world.pressure_belt[...] = belt_arr

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


def _wind_gradient_ca(
    proj: np.ndarray,
    reach: float,
    band_axis: int,
    edge_zone: float,
    noise_prob: float,
    iterations: int,
    rng,
) -> np.ndarray:
    """
    线性风基础强度场（0~1，未加 falloff 指数），截止线经元胞
    自动机扰动（world_core.ca_edge_perturb）而不再是一条直线。

    proj : 沿风向的投影距离场（风源处 = 0，背风侧递增）。
    reach : 直线截止距离（格）。
    band_axis : 取最大投影的条带轴——1 = 逐行条带（西风，风向
        沿 x，对每行取扰动区域的最大投影为有效截止），
        0 = 逐列条带（季风，风向沿对角线）。
    edge_zone / noise_prob / iterations : 截止线 CA 扰动参数
        （扰动范围 / 噪点概率 / 平滑迭代）；任一 ≤0 时退回
        直线截止。

    做法：直线截止区域 proj ≤ reach 的边缘条带内撒噪点、多数
    规则平滑，得到有机边界；再对每条垂直风向的条带取扰动后
    区域的最大投影作为该条带的有效截止，强度自风源向有效
    截止线性衰减——凸出的舌状区域同样获得完整梯度。
    """
    if edge_zone > 0 and noise_prob > 0:
        region = proj <= float(reach)
        zone = np.abs(proj - float(reach)) <= float(edge_zone)
        region = ca_edge_perturb(region, rng, noise_prob=noise_prob,
                                 expand_zone=zone,
                                 iterations=int(iterations))
        band_max = np.where(region, proj, -1.0).max(axis=band_axis)
    else:
        n = proj.shape[1] if band_axis == 0 else proj.shape[0]
        band_max = np.full(n, float(reach))
    bm_raw = band_max[None, :] if band_axis == 0 else band_max[:, None]
    bm = np.maximum(bm_raw + 1.0, 1e-6)
    f = np.clip(1.0 - proj / bm, 0.0, 1.0)
    # 该条带无扰动区域（CA 把区域削光的情形）→ 强度 0
    f = np.where(bm_raw < 0, 0.0, f)
    return f


def _elliptical_gradient_ca(
    proj: np.ndarray,
    source_y: float,
    edge_zone: float,
    noise_prob: float,
    iterations: int,
    rng,
) -> np.ndarray:
    """
    点源各向异性（拉长椭圆）风基础强度场（0~1，未加 falloff
    指数），截止线经元胞自动机扰动（world_core.ca_edge_perturb）
    而不再是一条光滑曲线。

    proj : 以风源（屏幕左缘 source_y 行）为原点的椭圆归一化
        距离场（风源处 = 0，等值线为东西拉长的椭圆，光滑截止于
        proj = 1）。
    source_y : 风源所在行号（纬度 45°N）。
    edge_zone / noise_prob / iterations : 截止线 CA 扰动参数
        （扰动范围（proj 单位，由格数按椭圆平均半轴折算）/
        噪点概率 / 平滑迭代）；任一 ≤0 时退回光滑椭圆截止。

    做法：光滑椭圆截止区域 proj ≤ 1 的边缘条带内撒噪点、多数
    规则平滑，得到有机边界；再把扰动后区域按自风源出发的方位角
    分桶，取每个方向上的最大投影作为该方向的有效截止，强度自
    风源向有效截止线性衰减——凸出的舌状区域同样获得完整梯度，
    咬入的缺口则截断其外侧。
    """
    if edge_zone > 0 and noise_prob > 0:
        region = proj <= 1.0
        zone = np.abs(proj - 1.0) <= float(edge_zone)
        region = ca_edge_perturb(region, rng, noise_prob=noise_prob,
                                 expand_zone=zone,
                                 iterations=int(iterations))
        h, w = proj.shape
        yv, xv = np.mgrid[0:h, 0:w]
        nbins = 720
        ang = np.arctan2(yv - float(source_y), xv.astype(np.float64))
        bin_idx = np.clip(((ang + math.pi) / (2.0 * math.pi) * nbins)
                          .astype(np.int64), 0, nbins - 1)
        reach_bin = np.zeros(nbins)
        np.maximum.at(reach_bin, bin_idx[region], proj[region])
        # 填补空桶：只向内插值填缺口、不向两端外推，
        # 避免梯度被涂抹到区域未覆盖的方向上
        nz = reach_bin > 0
        if nz.any() and not nz.all():
            idx = np.arange(nbins)
            first, last = int(idx[nz][0]), int(idx[nz][-1])
            gap = (~nz) & (idx > first) & (idx < last)
            reach_bin[gap] = np.interp(idx[gap], idx[nz], reach_bin[nz])
        bm = reach_bin[bin_idx]
    else:
        bm = np.ones_like(proj)
    bm = np.maximum(bm, 1e-6)
    return np.clip(1.0 - proj / bm, 0.0, 1.0)


def _compute_humidity(
    world: World,
    coastal_humidity_width: float,
    coastal_ca_noise_prob: float,
    coastal_ca_noise_range: float,
    coastal_ca_iterations: int,
    mountain_min_elev: float,
    mountain_effect_radius: float,
    mountain_sea_boost: float,
    mountain_inland_boost: float,
    westerly_reach_frac: float,
    westerly_boost: float,
    westerly_falloff: float,
    westerly_source_lat: float,
    westerly_lat_ratio: float,
    westerly_south_ratio: float,
    mediterranean_westerly_boost: float,
    westerlies_min_level: int,
    monsoon_reach_frac: float,
    monsoon_boost: float,
    monsoon_falloff: float,
    monsoon_min_level: int,
    wind_edge_zone_width: float,
    wind_edge_noise_prob: float,
    wind_edge_iterations: int,
    transition_band: float,
) -> Dict[str, Any]:
    """
    生成湿度图层（四级：干旱 0 / 半干旱 1 / 半湿润 2 / 全湿润 3，
    再映射为代表值写入 humidity；水体 = 100）。

    计算流程（除过渡强制外均为连续"等级单位"加减，最后取整）：
        1. 气候带基底：陆地默认干旱，按所在气候带取基底等级
           （_BELT_HUMIDITY_BASE 表；赤道低压带只有全湿润，
           西风带基底半干旱）；
        2. 海岸增益：距水距离 ≤ coastal_humidity_width 的陆地
           按所在带增益表升湿——带边缘用元胞自动机扰动
           （向外撒噪点 + 多数规则平滑，见 world_core.
           ca_edge_perturb），避免距离变换产生的生硬边缘；
        3. 雨影效应：海拔 ≥ mountain_min_elev 的陆地计为山体；
           每个陆地像素找最近山体格，比它与山体谁离水更近——
           更近（靠海一侧）加 mountain_sea_boost，更远（靠内陆
           一侧）加 mountain_inland_boost，作用范围
           mountain_effect_radius；
        4. 西风（含地中海"小西风"）：自屏幕左缘纬度
           westerly_source_lat（默认 45°N，西风带中部）的源点
           向东南北三个方向扩散的拉长椭圆风场——东西向半轴 =
           westerly_reach_frac × 图宽（向东衰减慢）；纵向分南北
           两个半轴：北向 = 东西向半轴 × westerly_lat_ratio，
           南向 = 东西向半轴 × westerly_south_ratio（默认更大，
           使西风南缘探入地中海带西部，地中海带因此能实际受到
           "小西风"加成）；衰减指数 westerly_falloff（<1 衰减
           放缓、湿润深入内陆）。西风带全额 westerly_boost，
           地中海带较弱的 mediterranean_westerly_boost；椭圆
           截止线经元胞自动机扰动（wind_edge_*，见
           _elliptical_gradient_ca），不再是光滑曲线；西风带最低
           湿度钳制为 westerlies_min_level（默认半干旱）；
        5. 季风（东南→西北线性风，与西风同构；不作用于赤道
           低压带）：沿风向的投影距离（距东缘与距南缘的平均，
           东南角为 0）线性衰减，范围 monsoon_reach_frac ×
           东南—西北对角线半长，衰减指数 monsoon_falloff
           （<1 衰减放缓）；截止线经元胞自动机扰动
           （wind_edge_*，与西风共用）；季风区（s>0）内以季风
           等级（monsoon_boost × s）替换气候带基底——不受
           气候带限制（赤道低压带除外），但最低等级钳制为
           monsoon_min_level（默认半干旱，季风区内不存在
           干旱）；海岸与地形增益照常叠加；
        6. 相邻过渡强制（_enforce_adjacent_levels）：任意相邻
           像素等级差 ≤ 1，缺失的中间等级自动补带（水体参与，
           海岸因此总有 全湿润→半湿润→半干旱 的过渡环）；
        7. 信风带与副热带高压带终检：临近海洋不得把两带抬过
           半干旱——海岸增益已被增益表限制，水体经过渡强制投出
           的湿润过渡环则另跑一遍"水体按半干旱计"的过渡强制并
           取两版较低者予以剔除；气候带过渡（如地中海带旁的
           半湿润过渡带）与季风带来的升湿不受影响。
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

    # ---- 2. 海岸增益（带边缘经元胞自动机扰动）----
    # 基础带 = 距水距离 ≤ 带宽；向外 noise_range 格内撒随机噪点，
    # 多数规则迭代后噪点在边缘团聚成有机凸包/分叉，碎点被消除
    band = land & (dist_water <= float(coastal_humidity_width))
    zone = land & (dist_water <= float(coastal_humidity_width)
                   + float(coastal_ca_noise_range))
    coastal_mask = ca_edge_perturb(
        band, world.rng,
        noise_prob=coastal_ca_noise_prob,
        expand_zone=zone,
        iterations=int(coastal_ca_iterations),
        constrain=land,
    )
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

    # ---- 4. 西风（自左缘纬度 45°N 源点向东南北扩散的拉长椭圆
    # 风场，截止线经元胞自动机扰动）----
    # 等值线为以源点（屏幕左缘、纬度 westerly_source_lat，即西风带
    # 中部）为端点的东西拉长椭圆：向东衰减慢（东西向半轴 =
    # westerly_reach_frac × 图宽），纵向衰减快。纵向分南北两个
    # 半轴：北向半轴 = 东西向半轴 × westerly_lat_ratio，南向半轴 =
    # 东西向半轴 × westerly_south_ratio（更大，使西风南缘探入
    # 地中海带西部）。
    # 西风带全额加成，地中海带另受较弱的"小西风"加成（同一风场）
    y0_west = (1.0 - float(westerly_source_lat) / _MAP_TOP_LAT) * (h - 1)
    a_west = max(float(westerly_reach_frac) * w, 1e-6)      # 东西向半轴（长轴）
    b_north = max(a_west * float(westerly_lat_ratio), 1e-6)   # 北向半轴
    b_south = max(a_west * float(westerly_south_ratio), 1e-6)  # 南向半轴
    dy_west = yv.astype(np.float64) - y0_west  # >0 = 源点以南（低纬）
    b_west = np.where(dy_west > 0, b_south, b_north)
    proj_west = np.sqrt((xv.astype(np.float64) / a_west) ** 2
                        + (dy_west / b_west) ** 2)
    f_west = _elliptical_gradient_ca(
        proj_west, y0_west,
        float(wind_edge_zone_width)
        / (0.5 * (a_west + 0.5 * (b_north + b_south))),
        wind_edge_noise_prob, wind_edge_iterations,
        world.rng) ** float(westerly_falloff)
    westerly_extra = np.where(belts == _BELT_WESTERLIES,
                              float(westerly_boost) * f_west, 0.0)
    westerly_extra += np.where(belts == _BELT_MEDITERRANEAN,
                               float(mediterranean_westerly_boost) * f_west,
                               0.0)

    # ---- 5. 季风（东南→西北线性风，与西风同构；截止线经
    # 元胞自动机扰动；不作用于赤道低压带）----
    # 沿风向的投影距离 = 距东缘与距南缘的平均（等值线为
    # 东北—西南走向的直线）：东南角 = 0，越靠西北越大
    proj_se = (((w - 1) - xv) + ((h - 1) - yv)) / 2.0
    reach_mono = max(float(monsoon_reach_frac) * (w + h) / 2.0, 1e-6)
    # monsoon_falloff 为衰减指数：<1 衰减放缓（湿润深入西北），
    # >1 衰减加剧
    s_monsoon = _wind_gradient_ca(
        proj_se, reach_mono, 0,
        wind_edge_zone_width, wind_edge_noise_prob, wind_edge_iterations,
        world.rng) ** float(monsoon_falloff)          # 季风强度 0~1
    monsoon_level = float(monsoon_boost) * s_monsoon
    # 赤道低压带只有全湿润，不受季风影响
    monsoon_region = land & (s_monsoon > 0) & (belts != _BELT_EQUATORIAL)
    # 季风区以季风等级替换气候带基底（四级皆可出现）
    base = np.where(monsoon_region, monsoon_level, belt_base)

    # ---- 合计并取整为离散等级 ----
    total = base + coastal_term + westerly_extra + mtn_boost
    level = np.clip(np.rint(total), 0, 3).astype(np.int8)
    # 季风区内不存在干旱：最差也是半干旱
    if monsoon_min_level > 0:
        level[monsoon_region] = np.maximum(
            level[monsoon_region], int(monsoon_min_level))
    # 西风带最低湿度为半干旱
    if westerlies_min_level > 0:
        wmask = land & (belts == _BELT_WESTERLIES)
        level[wmask] = np.maximum(level[wmask], int(westerlies_min_level))
    level[water] = 3  # 水体完全湿润
    level_pre = level.copy()  # 过渡强制前的等级（供步骤 7 使用）

    # ---- 6. 相邻等级过渡强制（缺失的中间等级自动补带）----
    level = _enforce_adjacent_levels(level, transition_band)

    # ---- 7. 信风带/副热带高压带终检：临近海洋不得造就半湿润/
    # 湿润，但气候带过渡与季风可以 ----
    # 海岸增益本身已被增益表限制在两带 ≤ 半干旱；还需剔除的是
    # 水体经过渡强制投出的湿润过渡环。做法：另跑一遍"水体按
    # 半干旱（1）计"的过渡强制——该版本中水体不再抬升两带，
    # 而陆地气候带（如地中海带）的带际过渡与季风等级不受影响；
    # 两带取两版结果的较低者
    dry_belts = land & ((belts == _BELT_TRADE)
                        | (belts == _BELT_SUBTROPICAL_HIGH))
    if dry_belts.any():
        level_alt = level_pre.copy()
        level_alt[water] = 1
        level_alt = _enforce_adjacent_levels(level_alt, transition_band)
        level[dry_belts] = np.minimum(level[dry_belts],
                                      level_alt[dry_belts])

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
# 功能函数 5 内部工具：河口播种（宽海岸带内随机撒点）
# ============================================================
def _select_outlets(
    world: World,
    water: np.ndarray,
    num_outlets: int,
    outlet_coast_spacing: Optional[float] = None,
    outlet_coast_band: float = 3.0,
) -> List[Tuple[int, int]]:
    """
    河口种子：宽海岸带内随机撒点（替代旧版"连通域走查 + 弧长
    均分"——弧长对海岸曲折度过于敏感，峡湾海岸会吞掉大量配额，
    平直肥沃的海岸反而分不到河口）。

    步骤：
        1. 海岸线 = 与水相邻的陆地像素（8 邻域）；
        2. 膨胀 outlet_coast_band 格形成几格宽的宽海岸带，并限制
           在陆地上（河口根必须在陆侧）；≤0 时退化为 1 格宽海岸线；
        3. 在带内无放回随机抽点：均分模式（outlet_coast_spacing
           为 None/≤0）抽取 num_outlets 个；间距模式（数值）以
           原始海岸线格数近似海岸弧长，抽取数 ≈ 弧长/间距
           （仍不超过 num_outlets）；带内格数不足时全取；
        4. 纯随机撒点——只要点够多，总有落在理想位置的河口候选；
           每条海岸、每段岸线都按面积等概率获得根节点。
    """
    land = world.land_mask
    coast = land & binary_dilation(water)
    # 可观测性：把撒点可用区域与最终入海口写入自定义图层，
    # 供地图查看器叠加显示（即使提前返回也保证图层存在）
    if world.get_layer("outlet_band") is None:
        world.add_layer("outlet_band", np.bool_, False)
    if world.get_layer("outlet_mask") is None:
        world.add_layer("outlet_mask", np.bool_, False)
    band_layer = world.get_layer("outlet_band")
    mask_layer = world.get_layer("outlet_mask")
    band_layer[...] = False
    mask_layer[...] = False
    if not coast.any() or num_outlets <= 0:
        return []
    if outlet_coast_band > 0:
        band = binary_dilation(coast, iterations=int(outlet_coast_band)) & land
    else:
        band = coast
    band_layer[...] = band
    cells = np.argwhere(band)
    if len(cells) == 0:
        return []
    if outlet_coast_spacing is not None and outlet_coast_spacing > 0:
        # 间距模式：原始海岸格数 ≈ 海岸弧长，推出目标数量
        approx_len = float(coast.sum())
        n = max(1, min(int(num_outlets),
                       int(approx_len / float(outlet_coast_spacing))))
    else:
        n = int(num_outlets)
    n = min(n, len(cells))
    pick = world.rng.choice(len(cells), size=n, replace=False)
    outlets = [(int(cells[i, 0]), int(cells[i, 1])) for i in pick]
    for y, x in outlets:
        mask_layer[y, x] = True
    return outlets



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
    outlet_coast_spacing: Optional[float],
    outlet_coast_band: float,
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

    河流特有逻辑留在本函数：海岸线河口播种（宽海岸带内随机撒点）、
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
    outlets = _select_outlets(world, water, num_outlets,
                              outlet_coast_spacing, outlet_coast_band)

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
    stats["outlet_mode"] = ("band_random"
                            if not (outlet_coast_spacing and outlet_coast_spacing > 0)
                            else "band_random_spacing")
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
    outlet_coast_spacing: Optional[float],
    outlet_coast_band: float,
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
    coast_avoid_distance: float,
    arid_attraction_frac: float,
    semi_arid_attraction_frac: float,
    rift_attraction: int,
    node_retries: int,
    young_influence: float,
    young_steps: int,
    max_iterations: int,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int, int]]]:
    """
    空间殖民算法（Space Colonization Algorithm, Runions et al. 2007）
    河流生成 —— 简化版。

    基础算法：
        初始化：非山脉陆地内均匀随机散布吸引点 M，再按湿度加权
        稀化（干旱区不保留、半干旱区按比例），并在裂谷像素
        （plate_boundaries == 5；fallback ridge_id < 0）上加撒
        rift_attraction 个吸引点——裂谷是板块离散形成的构造低地，
        加撒吸引点把河网拉进裂谷，使其有较大机会成为河道；
        海岸线河口放置根节点（每条河一个：在几格宽的宽海岸带内
        随机撒点，见 _select_outlets；共享同一吸引点场 → 河流
        自然争夺流域、互不交叉）。
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
        · 海岸规避：近岸带（距水 coast_avoid_distance 格以内）不撒
          吸引点；带内候选格仅在"没有更靠近水、且比父节点更靠近
          本树河口（或尚在河口半径内）"时允许——只允许奔向自己
          的入海口或向内陆穿越，不允许沿河岸平行游走（与山脉的
          mountain_buffer 外扩禁入相对称）；0 = 关闭；
        · 幼龄豁免：深度（离河口的步数）≤ young_steps 的节点，
          吸引点影响半径放宽为 d_i × young_influence——刚离开
          河口的河段尚在近岸无点带内，普通半径内没有吸引点会
          直接夭折；放宽后能捕获内陆吸引点、撑过无点带。
          young_steps ≤ 0 或 young_influence ≤ 1 = 关闭；
        · 湿度加权吸引点：河网密度 ∝ 湿度——干旱区按
          arid_attraction_frac（默认 0 = 不撒点）、半干旱区按
          semi_arid_attraction_frac（默认 0.3）稀化吸引点，
          半湿润/全湿润全保留。干旱区没有吸引点就不会长出
          支流，河网自然向湿润区集中；
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

    # 海岸规避：与山脉的山体掩膜外扩对称——近岸带（距水
    # coast_avoid_distance 格以内）不撒吸引点，支流不再被拉向
    # 海岸；生长入近岸带的候选格仅在"离水没有更近、且比父节点
    # 更靠近本树河口（或尚在河口半径内）"时允许，即只允许奔向
    # 自己的入海口，不允许沿河岸平行游走
    dist_water = distance_transform_edt(land)
    near_coast = land & (dist_water <= float(coast_avoid_distance)) \
        if coast_avoid_distance > 0 else np.zeros((h, w), dtype=bool)

    # ---- 1. 初始化：非山脉陆地内均匀随机散布吸引点 M（避开近岸带），
    # 再按湿度加权稀化；裂谷像素上额外加撒 rift_attraction 个吸引点 ----
    # 河网密度 ∝ 湿度：干旱区不撒点（不生支流、干流也不会被拉进
    # 干旱区），半干旱区按比例稀化，半湿润/全湿润全保留。湿度等级
    # 由 humidity 代表值还原（0=干旱 1=半干旱 2=半湿润 3=全湿润）
    _hv = np.asarray(_HUMIDITY_LEVEL_VALUES, dtype=np.float64)
    _mids = (_hv[:-1] + _hv[1:]) / 2.0
    hum_lvl = np.digitize(world.humidity, _mids)
    keep_prob = np.array([float(arid_attraction_frac),
                          float(semi_arid_attraction_frac), 1.0, 1.0])
    land_cells = np.argwhere(river_land & ~near_coast)
    n_scatter = int(min(max(num_attraction, 0), len(land_cells)))
    if n_scatter > 0:
        pick = rng.choice(len(land_cells), size=n_scatter, replace=False)
        lv = hum_lvl[land_cells[pick, 0], land_cells[pick, 1]]
        keep = rng.random(n_scatter) < keep_prob[lv]
        pick = pick[keep]
        att_y = land_cells[pick, 0].astype(np.float64)
        att_x = land_cells[pick, 1].astype(np.float64)
    else:
        lv = np.zeros(0, dtype=np.int64)
        att_y = np.zeros(0)
        att_x = np.zeros(0)
    n_uniform = len(att_y)
    # 各湿度等级的吸引点统计（报告用：撒布数 / 稀化后保留数）
    att_by_level = [int((lv == k).sum()) for k in range(4)]
    lv_kept = lv[keep] if n_scatter > 0 else lv
    att_kept_by_level = [int((lv_kept == k).sum()) for k in range(4)]

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

    # ---- 2. 初始化：河口根节点 N（宽海岸带内随机撒点）----
    outlets = _select_outlets(world, water, num_outlets,
                              outlet_coast_spacing, outlet_coast_band)
    pos_y: List[float] = []         # 浮点位置（生长运动学）
    pos_x: List[float] = []
    pix_y: List[int] = []           # 占据的像素格（海拔约束/避让/光栅化）
    pix_x: List[int] = []
    parent: List[int] = []          # 父节点索引，-1 = 根
    depth: List[int] = []           # 距本树河口的步数（幼龄豁免用）
    root_py: List[int] = []         # 本节点所属树的河口像素（海岸规避用）
    root_px: List[int] = []
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
        depth.append(0)
        root_py.append(oy)
        root_px.append(ox)
        children.append([])
        alive.append(True)
        retries.append(0)
        terminated.append(False)
        node_at[(oy, ox)] = idx
        add_node_bucket(idx)
        kill_around(float(oy), float(ox))

    total_killed = int(n_att - att_active.sum())
    young_assoc = 0               # 豁免关联救活的吸引点累计
    growth_rejected = 0
    spacing_stopped = 0
    reject_causes = [0, 0, 0, 0, 0, 0]  # 越界/下水/进山/下坡/碰撞/海岸规避
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
        # assoc: 吸引点 -> (距离, 节点)；普通半径关联先入册
        assoc: Dict[int, Tuple[float, int]] = {}
        for ai, d, ti in zip(active_idx, dist, tidx):
            if ti < len(front) and d <= float(d_i):
                assoc[int(ai)] = (float(d), front[int(ti)])
        # 幼龄豁免：深度 ≤ young_steps 的前沿节点单独建树，
        # 用放宽后的半径再查一遍；仅在"原本无关联、或豁免关联
        # 更近"时改挂到幼龄节点名下（统计豁免救活的吸引点数）
        young_d_i = float(d_i) * float(young_influence)
        if young_influence > 1.0 and young_steps > 0 and young_d_i > float(d_i):
            young_pos = [k for k, i in enumerate(front)
                         if depth[i] <= young_steps]
            if young_pos:
                ytree = cKDTree(np.column_stack(
                    (fy_arr[young_pos], fx_arr[young_pos])))
                yd, yti = ytree.query(coords, k=1,
                                      distance_upper_bound=young_d_i)
                for ai, d, ti in zip(active_idx, yd, yti):
                    if ti >= len(young_pos):
                        continue
                    node_idx = front[young_pos[int(ti)]]
                    prev = assoc.get(int(ai))
                    if prev is None:
                        assoc[int(ai)] = (float(d), node_idx)
                        young_assoc += 1
                    elif float(d) < prev[0] and prev[1] != node_idx:
                        assoc[int(ai)] = (float(d), node_idx)
        S: Dict[int, List[int]] = {}
        for ai, (_d, ni) in assoc.items():
            S.setdefault(ni, []).append(ai)

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
            elif (near_coast[qy, qx]
                  and dist_water[qy, qx] <= dist_water[py_, px_]
                  and ((fy - root_py[n_idx]) ** 2 + (fx - root_px[n_idx]) ** 2
                       > float(coast_avoid_distance) ** 2)
                  and ((fy - root_py[n_idx]) ** 2 + (fx - root_px[n_idx]) ** 2
                       >= (ny_ - root_py[n_idx]) ** 2 + (nx_ - root_px[n_idx]) ** 2)):
                # 海岸规避：近岸带内不朝水走、超出河口半径、且并非
                # 靠近本树河口 → 拒绝（只允许奔向自己的入海口或垂
                # 直海岸向内陆走，不允许沿河岸平行游走）
                reject_causes[5] += 1
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
                depth.append(depth[n_idx] + 1)
                root_py.append(root_py[n_idx])
                root_px.append(root_px[n_idx])
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
        "outlet_mode": ("band_random"
                        if not (outlet_coast_spacing and outlet_coast_spacing > 0)
                        else "band_random_spacing"),
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
        "young_rescued_attractions": young_assoc,
        "pruned_branches": pruned_branches,
        "attraction_scatter_by_level": att_by_level,
        "attraction_kept_by_level": att_kept_by_level,
        "near_coast_cells": int(near_coast.sum()),
        "growth_rejected": growth_rejected,
        "reject_causes": {
            "bounds": reject_causes[0],
            "water": reject_causes[1],
            "mountain": reject_causes[2],
            "step_drop": reject_causes[3],
            "collision": reject_causes[4],
            "coast_avoid": reject_causes[5],
        },
        "river_cells": len(attach_log),
        "total_strength": float(strength[world.river_mask].sum()) if attach_log else 0.0,
        "max_strength": float(strength.max()) if attach_log else 0.0,
    }
    return stats, attach_log


# ============================================================
# 功能函数 5b：河流水流量（源头注入 + 向下游累加）
# ============================================================
def _compute_river_discharge(
    world: World,
    attach_log: List[Tuple[int, int, int, int]],
    mountain_min_elev: float,
    source_flow: float,
    humid_source_flow: float,
    mountain_source_flow: float,
    mountain_radius: float,
) -> Dict[str, Any]:
    """
    河流水流量图层（world.river_discharge，float32）。

    源头（无上游子格的河道末梢，不含河口根）注入水流量：
        注入量 = source_flow（基础，每个源头都有）
               + humid_source_flow（源头位于半湿润/全湿润地区时）
               + mountain_source_flow × (1 − 到最近山体的距离
                 / mountain_radius) ×（该山体海拔 / 全图山体最高海拔）
                 （源头 mountain_radius 格内有海拔 ≥ mountain_min_elev
                 的山体时；越近、山体越高，加成越大）
    水流量自源头向下游累加（附着逆序 = 上游→下游拓扑序），每个
    河道格的水流量 = 其全部上游源头注入量之和——源头越多、源头
    越湿润、源头越近高山，水流量越大，离入海口越近越大。
    非河道格为 0。
    """
    h, w = world.shape
    stats: Dict[str, Any] = {
        "num_sources": 0,
        "humid_sources": 0,
        "mountain_sources": 0,
        "max_discharge": 0.0,
        "mean_discharge": 0.0,
    }
    discharge = np.zeros((h, w), dtype=np.float64)
    if not attach_log:
        world.river_discharge[...] = 0.0
        return stats

    # ---- 1. 源头 = 不作为任何格父格的非根河道格 ----
    parents = {(py, px) for _, _, py, px in attach_log if py >= 0}
    sources = [(y, x) for y, x, py, px in attach_log
               if py >= 0 and (y, x) not in parents]
    stats["num_sources"] = len(sources)

    # ---- 2. 源头注入量 ----
    land = world.land_mask
    elev = world.elevation.astype(np.float64)
    # 湿度等级由代表值还原（0=干旱 1=半干旱 2=半湿润 3=全湿润）
    _hv = np.asarray(_HUMIDITY_LEVEL_VALUES, dtype=np.float64)
    _mids = (_hv[:-1] + _hv[1:]) / 2.0
    hum_lvl = np.digitize(world.humidity, _mids)

    # 高山邻近加成：最近山体的距离与海拔（海拔归一化到全图山体最高值）
    near_mtn = None
    mtn = land & (elev >= float(mountain_min_elev))
    if mountain_source_flow > 0 and mountain_radius > 0 and mtn.any():
        dist_mtn, idx = distance_transform_edt(~mtn, return_indices=True)
        near_mtn = (dist_mtn, elev[idx[0], idx[1]],
                    max(float(elev[mtn].max()), 1e-9))

    for y, x in sources:
        q = float(source_flow)
        if hum_lvl[y, x] >= 2:  # 半湿润/全湿润地区的源头
            q += float(humid_source_flow)
            stats["humid_sources"] += 1
        if near_mtn is not None:
            dist_mtn, mtn_elev, mtn_ref = near_mtn
            d = dist_mtn[y, x]
            if d <= float(mountain_radius):
                q += (float(mountain_source_flow)
                      * (1.0 - d / float(mountain_radius))
                      * (mtn_elev[y, x] / mtn_ref))
                stats["mountain_sources"] += 1
        discharge[y, x] = q

    # ---- 3. 向下游累加（离入海口越近，水流量越大）----
    for y, x, py, px in reversed(attach_log):
        if py >= 0:
            discharge[py, px] += discharge[y, x]

    world.river_discharge[...] = discharge.astype(np.float32)
    rmask = world.river_mask
    if rmask.any():
        stats["max_discharge"] = float(discharge[rmask].max())
        stats["mean_discharge"] = float(discharge[rmask].mean())
    return stats


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
    belt_edge_zone_width: float = 8.0,    # 带边界元胞自动机扰动范围（格，边界两侧各）：≤0 关闭扰动（直线边界）
    belt_edge_noise_prob: float = 0.20,   # 带边界扰动噪点概率：越大边界越曲折；≤0 关闭扰动
    belt_edge_iterations: int = 3,        # 带边界扰动平滑迭代次数：越大边界越圆润
    # ── 湿度：海岸增益 ──
    coastal_humidity_width: float = 8.0,   # 海岸湿度提升带宽度（格，默认较小）：距水距离 ≤ 带宽的陆地按所在气候带增益表升湿
    coastal_ca_noise_prob: float = 0.18,   # 海岸带边缘元胞自动机噪点概率（0~1）：越大边缘扰动越剧烈
    coastal_ca_noise_range: float = 5.0,   # 噪点散布范围（格，带宽再向外）：噪点只出现在基础带外侧该距离内
    coastal_ca_iterations: int = 3,        # 多数规则平滑迭代次数：越大边缘越圆润、细节越少
    # ── 湿度：山脉雨影效应 ──
    mountain_min_elev: float = 100.0,       # 计入雨影效应的山体最低海拔（m）
    mountain_effect_radius: float = 10.0,  # 山体湿度影响半径（格）
    mountain_sea_boost: float = 1.0,       # 靠海一侧湿度增益（等级，0~3 连续）
    mountain_inland_boost: float = 0.3,    # 靠内陆一侧湿度增益（等级，应小于靠海侧）
    # ── 湿度：西风带 ──
    westerly_reach_frac: float = 0.6,      # 西风椭圆风场东西向半轴（占图宽比例）：向东衰减慢，默认 0.6
    westerly_source_lat: float = 45.0,     # 西风源点纬度（°N，位于屏幕左缘）：默认 45 = 西风带中部
    westerly_lat_ratio: float = 0.5,       # 北向半轴与东西向半轴之比（0~1）：越小北向衰减越快
    westerly_south_ratio: float = 0.8,     # 南向半轴与东西向半轴之比（0~1）：默认大于北向，使西风南缘探入地中海带西部
    westerly_boost: float = 2.0,           # 西风最大额外湿度（等级）：使西风带西半侧升至全湿润
    westerly_falloff: float = 0.4,         # 西风衰减指数：<1 衰减放缓（湿润深入内陆），>1 衰减加剧；调强度主力参数
    mediterranean_westerly_boost: float = 0.4,  # 地中海带"小西风"最大额外湿度（等级）：与西风带同一风场但力度较小
    westerlies_min_level: int = 1,         # 西风带最低湿度等级：默认 1（半干旱）
    # ── 湿度：风场截止线（西风/季风共用）元胞自动机扰动 ──
    wind_edge_zone_width: float = 8.0,     # 截止线扰动范围（格，截止线两侧各）：≤0 关闭（直线截止）
    wind_edge_noise_prob: float = 0.20,    # 截止线扰动噪点概率：越大截止线越曲折；≤0 关闭
    wind_edge_iterations: int = 3,         # 截止线扰动平滑迭代次数：越大越圆润
    # ── 湿度：季风（东南→西北线性风，不作用于赤道低压带）──
    monsoon_reach_frac: float = 0.4,     # 季风影响范围（占东南—西北对角线半长比例）：默认 1/3 张地图
    monsoon_boost: float = 3.0,            # 季风最大湿度（等级）：季风区内替换气候带基底
    monsoon_falloff: float = 0.3,          # 季风衰减指数：<1 衰减放缓（湿润深入西北）、>1 衰减加剧；调强度主力参数
    monsoon_min_level: int = 1,            # 季风区最低湿度等级：默认 1（半干旱）——季风区内不存在干旱
    # ── 湿度：相邻等级过渡 ──
    humidity_transition_band: float = 4.0,  # 每级过渡带宽度（格）：相邻区域等级差强制 ≤1，缺失中间等级自动补带
    # ── 河流生成 ──
    num_outlets: int = 60,          # 河口（根节点）数量：随机撒点模式（默认）为确切目标数；指定 outlet_coast_spacing 时为上限
    outlet_coast_spacing: Optional[float] = None,  # 河口沿河岸间距（格）：None/0=直接随机撒 num_outlets 个；数值=以原始海岸格数近似弧长、按该间距推出撒点数（不超 num_outlets）
    outlet_coast_band: float = 3.0, # 宽海岸带宽度（格）：海岸线向陆地一侧膨胀该格数形成宽带，河口在带内随机撒点；≤0=仅在 1 格宽海岸线上撒
    # 河流算法："sca"（空间殖民算法简化版，默认：纯吸引方向生长、河道平直，
    # 反平行间距控制生成叶脉状分叉，河流终止于山脉边缘）
    # 或 "dla"（旧版扩散限制聚集，河道呈灌木状分叉）
    river_algorithm: str = "sca",
    # ── SCA 河流（river_algorithm="sca" 时生效）──
    sca_num_attraction: int = 5000,  # 吸引点数量：越多河网越密、支流越多；256² 建议 800~2000，512² 建议 3000~6000
    sca_d_i: float = 18.0,           # 影响半径（格）：节点只响应此距离内的吸引点；大→河道顺直平滑，小→蜿蜒扭曲
    sca_d_k: float = 2.0,            # 删除半径（格）：河道经过即清除附近吸引点；大→支流稀疏、河网开阔，小→支流密集
    sca_D: float = 1.5,              # 生长步长（格）：建议 1~3；小→河道细腻但节点多、速度慢
    sca_L_min: float = 10.0,         # 修剪阈值 L_min（格）：长度不足的边缘支流（或整条短河）被剪除
    sca_max_step_drop: float = 5.0,  # 唯一海拔约束：单步允许的最大海拔下降（米），容许越过噪声凹坑但不准逆坡下行；0=严格单调
    sca_min_channel_spacing: float = 2.0,  # 反平行间距（格）：新节点距其他河道近于该值且走向平行则分支永久终止，生成叶脉状分叉；建议 ≥ sca_d_k
    sca_parallel_angle: float = 10.0,      # 平行判定夹角（度）：近距河道方向夹角小于该值（或头对头）视为平行；0≈关闭间距控制，90=近距即终止
    sca_spacing_exempt: Optional[int] = None,  # 间距检查豁免的父链祖先代数（自身河道刚经过的路径）；None=按 spacing/D+6 自动
    sca_mountain_max_distance: float = 2.0,  # 山脉距离限制（格）：梳齿覆盖区 ∩ 山脊中线（ridge_line_mask 图层）邻近带 才算山脉（限陆地）；建议 2~8，0=仅中线上的梳齿
    sca_mountain_buffer: int = 1,    # 山脉禁入缓冲（格）：山脉区域外扩该格数，河流终止于其边缘；0=仅山脉本体
    sca_coast_avoid: float = 6.0,    # 海岸规避距离（格）：近岸带不撒吸引点，且带内只允许奔向本树河口或向内陆的生长，不允许沿河岸平行游走；0=关闭
    sca_arid_attraction: float = 0.2,      # 干旱区吸引点保留比例（0~1）：0=干旱区不撒点、不生支流；河网密度∝湿度的主控参数
    sca_semi_arid_attraction: float = 0.3, # 半干旱区吸引点保留比例（0~1）：越小半干旱区支流越稀疏
    sca_rift_attraction: int = 200,  # 裂谷加撒吸引点数：在裂谷（plate_boundaries == 5）像素上额外均匀撒点，裂谷有较大机会成为河道；0=关闭
    sca_max_iterations: Optional[int] = None,  # 迭代上限（性能兜底）：None=按 3.5×max(宽,高) 自动；正常在吸引点耗尽或停滞时提前结束
    sca_node_retries: int = 5,       # 节点连续受阻多少轮后退出生长前沿（卡死末梢放弃，吸引点让给其他分支）
    sca_young_influence: float = 2.0,  # 幼龄豁免：刚离河口 sca_young_steps 步内的河段，吸引点影响半径 = d_i × 本系数（≤1 关闭）
    sca_young_steps: int = 8,        # 幼龄豁免的步数窗口：离河口多少步以内算幼龄（≤0 关闭）
    # ── DLA 河流（river_algorithm="dla" 时生效）──
    spawn_radius: float = 25.0,
    spawn_elevation_bias: float = 2.0,
    walk_elevation_bias: float = 1.0,
    max_river_neighbors: int = 3,
    max_particles: int = 20000,
    max_walk_steps: int = 400,
    coastal_buffer: int = 0,
    pool_rebuild_interval: int = 256,
    # ── 河流水流量 ──
    river_source_flow: float = 1.0,          # 每个源头的基础水流量：源头越多，下游水流量越大
    river_humid_source_flow: float = 1.0,    # 湿润地区（半湿润/全湿润）源头的额外水流量
    river_mountain_source_flow: float = 2.0, # 临近高山源头的最大额外水流量（随距离衰减、随山体海拔缩放）
    river_mountain_radius: float = 20.0,     # 源头的高山邻近半径（格）：≤0 关闭高山加成
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
    belt_edge_zone_width / belt_edge_noise_prob / belt_edge_iterations :
        带边界（纬度直线）的元胞自动机扰动：扰动范围 / 噪点概率 /
        平滑迭代次数；扰动后带边界呈有机曲线。
    coastal_humidity_width / coastal_ca_noise_prob /
    coastal_ca_noise_range / coastal_ca_iterations :
        海岸湿度提升带宽度（格，默认较小）及其边缘的元胞自动机
        扰动（噪点概率 / 外扩范围 / 平滑迭代次数）。
    mountain_min_elev / mountain_effect_radius /
    mountain_sea_boost / mountain_inland_boost :
        雨影效应——海拔 ≥ mountain_min_elev 的陆地计为山体，
        半径内靠海一侧湿度增益 mountain_sea_boost、靠内陆一侧
        mountain_inland_boost（均为连续等级单位 0~3）。
    westerly_reach_frac / westerly_source_lat / westerly_lat_ratio /
    westerly_south_ratio / westerly_boost / westerly_falloff /
    mediterranean_westerly_boost / westerlies_min_level :
        西风（含地中海"小西风"）：自屏幕左缘纬度 westerly_source_lat
        （默认 45°N，西风带中部）的源点向东南北三个方向扩散的
        拉长椭圆风场的额外湿度。reach_frac 为东西向半轴占图宽
        比例（向东衰减慢，默认 0.6）；纵向分南北两个半轴——
        westerly_lat_ratio 为北向半轴之比（默认 0.5），
        westerly_south_ratio 为南向半轴之比（默认 0.8，更大，
        使西风南缘探入地中海带西部）。falloff 为衰减指数，<1
        衰减放缓、西风更强，是调西风强度的主力参数；地中海带
        受同一风场但力度较小（小西风 boost）；westerlies_min_level
        为西风带最低湿度等级（默认 1 = 半干旱）。
    wind_edge_zone_width / wind_edge_noise_prob / wind_edge_iterations :
        西风与季风截止线的元胞自动机扰动（共用）：扰动范围 /
        噪点概率 / 平滑迭代；扰动后截止线呈有机曲线而非直线，
        ≤0 关闭。
    monsoon_reach_frac / monsoon_boost / monsoon_falloff /
    monsoon_min_level :
        季风（东南→西北线性风，与西风同构；不作用于赤道低压
        带——赤道低压带只有全湿润）——沿风向投影距离（东南角
        为 0）线性衰减，reach_frac 占对角线半长比例；季风区内
        替换气候带基底。falloff 为衰减指数（<1 衰减放缓、季风
        更强），是调季风强度的主力参数；min_level 为季风区最低
        湿度等级（默认 1 = 半干旱，季风区内不存在干旱）。
    humidity_transition_band : float
        相邻湿度区域等级差强制 ≤1 时每级过渡带的宽度（格）。
    num_outlets / outlet_coast_spacing / outlet_coast_band :
        河口（根节点）控制。与水相邻的海岸像素向外膨胀
        outlet_coast_band 格形成几格宽的宽海岸带（限陆侧），
        河口在带内无放回随机撒点——点够多时总有落在理想位置
        的候选，每段海岸按面积等概率获得根节点。均分模式
        （outlet_coast_spacing 为 None/0，默认）：撒 num_outlets
        个；间距模式（数值）：以原始海岸格数近似弧长，按该间距
        推出撒点数（不超 num_outlets）。
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
    sca_coast_avoid : float
        海岸规避距离（格）：近岸带不撒吸引点；带内生长仅允许
        奔向本树河口或向内陆穿越，不允许沿河岸平行游走；
        与山脉的 sca_mountain_buffer 对称；0 = 关闭。
    sca_arid_attraction / sca_semi_arid_attraction : float
        湿度加权吸引点保留比例（0~1）：河网密度 ∝ 湿度——干旱区
        （默认 0 = 不撒点、不生支流）与半干旱区（默认 0.3，
        支流稀疏）的吸引点按比例稀化，半湿润/全湿润全保留。
    sca_rift_attraction : int
        裂谷加撒吸引点数：在裂谷（plate_boundaries == 5，
        板块离散形成的构造低地）像素上额外均匀撒点，
        裂谷有较大机会成为河道；0 = 关闭。
    sca_young_influence / sca_young_steps :
        幼龄豁免：离河口 sca_young_steps 步以内的河段，吸引点
        影响半径放宽为 d_i × sca_young_influence——刚发育几步的
        河流尚在近岸无点带内，普通半径捕获不到吸引点会夭折；
        放宽后撑过无点带。≤1 或 ≤0 = 关闭。
    river_source_flow / river_humid_source_flow /
    river_mountain_source_flow / river_mountain_radius :
        河流水流量（river_discharge 图层）：每个源头注入
        river_source_flow 基础流量（源头越多下游越大）；位于半湿润/
        全湿润地区的源头额外 +river_humid_source_flow；源头
        river_mountain_radius 格内有山体（海拔 ≥ mountain_min_elev）
        时再额外 +river_mountain_source_flow × 距离衰减 × 山体海拔
        归一化（海拔影响：越近、山越高加成越大）。水流量向下游
        累加，每个河道格有自己的水流量值。
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

    # ---------- 3. 气候带（0~70°N 六带，边界经元胞自动机扰动）----------
    report["pressure_belts"] = _generate_pressure_belts(
        world, belt_edge_zone_width, belt_edge_noise_prob, belt_edge_iterations)

    # ---------- 4. 湿度（气候带基底+海岸/雨影/西风/季风增益+相邻过渡强制）----------
    report["humidity"] = _compute_humidity(
        world,
        coastal_humidity_width,
        coastal_ca_noise_prob, coastal_ca_noise_range, coastal_ca_iterations,
        mountain_min_elev, mountain_effect_radius,
        mountain_sea_boost, mountain_inland_boost,
        westerly_reach_frac, westerly_boost, westerly_falloff,
        westerly_source_lat, westerly_lat_ratio, westerly_south_ratio,
        mediterranean_westerly_boost, westerlies_min_level,
        monsoon_reach_frac, monsoon_boost,
        monsoon_falloff, monsoon_min_level,
        wind_edge_zone_width, wind_edge_noise_prob, wind_edge_iterations,
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
            world, num_outlets, outlet_coast_spacing, outlet_coast_band,
            sca_num_attraction, sca_d_i, sca_d_k, sca_D, sca_L_min,
            sca_max_step_drop, sca_min_channel_spacing, sca_parallel_angle,
            sca_spacing_exempt, sca_mountain_max_distance, sca_mountain_buffer,
            sca_coast_avoid, sca_arid_attraction, sca_semi_arid_attraction,
            sca_rift_attraction, sca_node_retries,
            sca_young_influence, sca_young_steps, sca_max_iterations,
        )
    elif river_algorithm == "dla":
        river_stats, attach_log = _grow_rivers_dla(
            world, num_outlets, outlet_coast_spacing, outlet_coast_band,
            spawn_radius,
            spawn_elevation_bias, walk_elevation_bias, max_river_neighbors,
            max_particles, max_walk_steps, coastal_buffer, pool_rebuild_interval,
        )
    else:
        raise ValueError(f"未知河流算法: {river_algorithm!r}（可选 'sca' / 'dla'）")
    report["rivers"] = river_stats

    # ---------- 5b. 河流水流量（源头注入，向下游累加）----------
    report["river_discharge"] = _compute_river_discharge(
        world, attach_log, mountain_min_elev,
        river_source_flow, river_humid_source_flow,
        river_mountain_source_flow, river_mountain_radius,
    )

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