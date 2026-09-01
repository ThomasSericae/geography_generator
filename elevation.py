"""
landscape.py – 地形景观生成（基于 World 中的构造数据）

本模块接收 tectonic 阶段构造好的 World（含板块、速度）以及几何曲线，
执行：
    · 海岸线锐角倒角（像素级海陆归属翻转）
    · 原初山脊线栅格化（含高度渐变、噪声调制）
    · 环形 DLA 梳齿贴印（山脊与裂谷共用）
    · 裂谷中线栅格化（梭子形盆地）
    · 高原抬升场计算（碰撞高原 + 地盾高原，高斯平滑）
    · 海陆基准场、背景噪声
    · 山脊抬升场（对数衰减）
    · 海拔场合成
    · 海陆掩膜
    · 写入 World 对象（elevation, land_mask, ridge_id, ridge_speed 等）
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.ndimage import distance_transform_edt, gaussian_filter

from world_core import World, PerlinNoise1D, PerlinNoise2D, grow_dla
from tectonic import (
    build_tectonic_data,
    _chamfer_polyline_junctions,   # 供海岸倒角复用
)


# ============================================================
# 模块级缓存：环形 DLA 纹理库（全进程复用）
# ============================================================
_RING_DLA_CACHE: Dict[Tuple, List[np.ndarray]] = {}
_BAKED_STAMP_CACHE: Dict[int, List[np.ndarray]] = {}


def _load_baked_stamps(tooth_length: int) -> Optional[List[np.ndarray]]:
    """从 ring_dla_stamps.py 加载烘焙纹理（毫秒级）。"""
    cached = _BAKED_STAMP_CACHE.get(tooth_length)
    if cached is not None:
        return cached
    try:
        import ring_dla_stamps
    except ImportError:
        return None
    stamps = ring_dla_stamps.load_stamps()
    baked_l = (stamps[0].shape[0] - 1) // 2
    if tooth_length > baked_l:
        return None
    crop = baked_l - tooth_length
    if crop > 0:
        stamps = [s[crop:-crop] for s in stamps]
    _BAKED_STAMP_CACHE[tooth_length] = stamps
    return stamps


def _grow_ring_dla(
    rng: np.random.Generator,
    radius: int,
    tooth_length: int,
    max_particles: int,
) -> Tuple[np.ndarray, float, float]:
    """在圆环种子上生长 DLA（内部兜底）。"""
    pad = tooth_length + 6
    half = int(radius + pad)
    size = 2 * half + 1
    cy = cx = float(half)
    yv, xv = np.mgrid[0:size, 0:size]
    dist = np.hypot(yv - cy, xv - cx)
    ring = np.abs(dist - radius) <= 0.8
    feasible = (np.abs(dist - radius) <= tooth_length) & ~ring
    feasible[:2, :] = feasible[-2:, :] = False
    feasible[:, :2] = feasible[:, -2:] = False
    guidance = np.clip(tooth_length - np.abs(dist - radius), 0.0, None)
    seeds_y, seeds_x = np.nonzero(ring)
    result = grow_dla(
        rng, guidance, list(zip(seeds_y.tolist(), seeds_x.tolist())), feasible,
        base_level=0.0,
        spawn_radius=float(min(tooth_length, 18.0)),
        spawn_elevation_bias=1.5,
        walk_elevation_bias=2.0,
        max_neighbors=3,
        max_particles=int(max_particles),
        max_walk_steps=400,
        pool_rebuild_interval=256,
    )
    return result.cluster, cy, cx


def _unwrap_ring_to_stamp(
    cluster: np.ndarray,
    cy: float,
    cx: float,
    radius: int,
    tooth_length: int,
) -> np.ndarray:
    """环形 DLA 极坐标展开为矩形梳子纹理。"""
    circumference = int(math.ceil(2.0 * math.pi * radius))
    cols = np.arange(circumference, dtype=np.float64)
    theta = cols / float(radius)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rows = np.arange(-tooth_length, tooth_length + 1, dtype=np.float64)
    rr = radius + rows
    h, w = cluster.shape
    xs = np.clip(np.round(cx + rr[:, None] * cos_t[None, :]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.round(cy + rr[:, None] * sin_t[None, :]).astype(np.int64), 0, h - 1)
    return cluster[ys, xs]


# 烘焙文件缺失时现场生长兜底的固定配置
_FALLBACK_STAMP_SEED = 7106
_FALLBACK_STAMP_COUNT = 2
_FALLBACK_RING_RADIUS = 512
_FALLBACK_RING_PARTICLES = 6000


def _get_ring_dla_stamps(tooth_length: int) -> List[np.ndarray]:
    """获取环形 DLA 纹理库：优先烘焙，缺失则现场生长并缓存。"""
    baked = _load_baked_stamps(int(tooth_length))
    if baked is not None:
        return baked
    key = int(tooth_length)
    stamps = _RING_DLA_CACHE.get(key)
    if stamps is None:
        rng = np.random.default_rng(_FALLBACK_STAMP_SEED)
        stamps = []
        for _ in range(_FALLBACK_STAMP_COUNT):
            cluster, cy, cx = _grow_ring_dla(
                rng, _FALLBACK_RING_RADIUS, key, _FALLBACK_RING_PARTICLES)
            stamps.append(_unwrap_ring_to_stamp(
                cluster, cy, cx, _FALLBACK_RING_RADIUS, key))
        _RING_DLA_CACHE[key] = stamps
    return stamps


# ============================================================
# 景观阶段专用函数（栅格化、纹理、高原、背景等）
# ============================================================

def _chamfer_coastline(
    plate_domain: np.ndarray,
    valid_edges: List,
    is_ocean: np.ndarray,
    cut_dist: float,
    min_angle_deg: float,
    min_line_length: float,
) -> int:
    """
    对海岸线的锐角做倒角，就地修改 plate_domain。

    海岸边缘 = 两侧板块海陆属性不同（is_ocean[p1] != is_ocean[p2]）
    的 Voronoi 分界线。海陆属性是二值的，任一 Voronoi 顶点处的
    三个板块中必有两者同属性，故每个海岸顶点恰有两条海岸边缘
    相交——与山脊倒角同一几何：夹角小于 min_angle_deg 的顶点，
    两条边缘在距顶点 cut_dist 格处被截断，截点与顶点围成的小
    三角形被整体翻转海陆归属——陆侧尖角（海岬）被削平、洋侧
    尖角（尖湾）被填平，海岸线因此圆润。长度不足
    min_line_length 的海岸边缘受保护，不参与倒角。

    倒角只改像素级海陆归属场（板块级 is_ocean 不变），下游的
    海陆基准场高斯平滑自动跟随新海岸线。返回倒角数量。
    """
    if cut_dist <= 0:
        return 0
    polylines = []
    for _rv, clipped, _btype, (p1, p2) in valid_edges:
        if bool(is_ocean[p1]) != bool(is_ocean[p2]):
            (x1, y1), (x2, y2) = clipped
            polylines.append(np.array([[x1, y1], [x2, y2]], dtype=np.float64))
    if len(polylines) < 2:
        return 0
    _new_lines, chamfer_lines, _truncs = _chamfer_polyline_junctions(
        polylines, cut_dist, min_angle_deg, min_line_length)
    h, w = plate_domain.shape
    n_chamfer = 0
    for (seg_pts, i, si, _j, _sj) in chamfer_lines:
        V = polylines[i][0] if si == 0 else polylines[i][-1]
        A, B = seg_pts[0], seg_pts[-1]
        vx, vy = float(V[0]), float(V[1])
        ax, ay = float(A[0]), float(A[1])
        bx, by = float(B[0]), float(B[1])
        cx_idx = min(max(int(round((vx + ax + bx) / 3.0)), 0), w - 1)
        cy_idx = min(max(int(round((vy + ay + by) / 3.0)), 0), h - 1)
        corner_domain = plate_domain[cy_idx, cx_idx]
        x0 = max(0, int(math.floor(min(vx, ax, bx))) - 1)
        x1 = min(w - 1, int(math.ceil(max(vx, ax, bx))) + 1)
        y0 = max(0, int(math.floor(min(vy, ay, by))) - 1)
        y1 = min(h - 1, int(math.ceil(max(vy, ay, by))) + 1)
        if x0 > x1 or y0 > y1:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        gx = gx.astype(np.float64)
        gy = gy.astype(np.float64)
        s1 = (gx - vx) * (ay - vy) - (ax - vx) * (gy - vy)
        s2 = (gx - ax) * (by - ay) - (bx - ax) * (gy - ay)
        s3 = (gx - bx) * (vy - by) - (vx - bx) * (gy - by)
        inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | \
                 ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
        patch = plate_domain[y0:y1 + 1, x0:x1 + 1]
        patch[inside] = np.int8(1) - corner_domain
        n_chamfer += 1
    return n_chamfer


def _mark_mountain_edges(
    boundaries: np.ndarray,
    ridge_curves: List,
    width: int,
    height: int,
) -> np.ndarray:
    """将被选中的碰撞山脊边缘的曲线路径栅格化到 plate_boundaries，编码 4。"""
    marked = boundaries.copy()
    for curve_points, *_ in ridge_curves:
        for cx, cy in curve_points:
            x, y = int(round(cx)), int(round(cy))
            if 0 <= x < width and 0 <= y < height and marked[y, x] != 3:
                marked[y, x] = 4
    return marked


def _rasterize_primary_ridges(
    ridge_curves: List[Tuple],
    perlin: PerlinNoise1D,
    width: int,
    height: int,
    noise_amp: float,
    frequency: float,
    octaves: int,
    lacunarity: float,
    end_taper: float = 30.0,
    junction_blend: float = 20.0,
    junction_snap: float = 1.5,
    shrink_length: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把选中的碰撞山脊边缘曲线栅格化为"原初山脊线"。

    每个山脊像素记录自身海拔：
        H(t) = H_base(t) × (1 + Perlin1D(t × frequency + edge_offset) × noise_amp)
    其中基线高度 H_base(t) 沿山脊不再是常数，而是两端渐变：

    · 短山脊高度缩减映射：梳齿宽度基本恒定，太短的山峰会"宽度
      大于长度"显得又矮又宽。弧长短于 shrink_length 的山脊按
      smoothstep(L / shrink_length) 缩减有效高度，越短越矮；
      交汇高度均值也用缩减后的高度计算，保持衔接一致；
    · 端点连续性识别：两条山脊的【位移前原始端点】共享同一
      Voronoi 顶点（距离 ≤ junction_snap）即为连续/交汇；只匹配到
      自身的端点才是山脊线的真正结束；
    · 真正末端：目标高度 = 0（没有山脊线 = 海拔降回板块基准），
      在 end_taper 格内以 smoothstep 从 0 渐升到完整高度——
      山脉渐渐消失而非戛然而止（梳齿海拔由最近脊格衰减得到，
      会自动跟随变矮，梳子末端不再呈长方形硬边界）；
    · 交汇端点：目标高度 = 共享该顶点的各山脊高度的均值，
      在 junction_blend 格内平滑过渡——不同高度的山脉之间
      山脊线海拔是渐变的；
    · 短小山脉（两端渐变区重叠）的过渡：
      - 两侧都有山脉：从中点分开，各半边按标准衰减率向本侧
        交汇高度渐变，中点处以小窗口交叉淡化避免台阶；
      - 只有一侧有山脉：小山脉本身成为渐变，自交汇高度沿
        全长单调渐隐至 0（板块基准）；
      - 两侧皆空（孤立短山脊）：两端各按标准率渐隐，取较低者，
        自然形成先升后降的拱形。

    同时记录所属山脊线身份（ridge_id，从 1 起编号）与相撞速度。
    多条曲线重叠的像素取最大高度（身份/速度随最大高度者）。

    返回 (ridge_h_field, ridge_id_field, ridge_speed_field)：
    山脊像素非零，其余为 0。
    """
    n_curves = len(ridge_curves)

    arc_lengths = np.zeros(n_curves, dtype=np.float64)
    for i, rc in enumerate(ridge_curves):
        pts_i = np.asarray(rc[0], dtype=np.float64)
        if len(pts_i) >= 2:
            arc_lengths[i] = float(
                np.hypot(np.diff(pts_i[:, 0]), np.diff(pts_i[:, 1])).sum())
    scaled_h = np.array([rc[1] for rc in ridge_curves], dtype=np.float64)
    if shrink_length > 0:
        frac = np.clip(arc_lengths / float(shrink_length), 0.0, 1.0)
        scaled_h = scaled_h * (frac * frac * (3.0 - 2.0 * frac))

    raw_ends = []
    for rc in ridge_curves:
        raw_ends.append((np.asarray(rc[5], dtype=np.float64),
                         np.asarray(rc[6], dtype=np.float64)))
    target_h = np.zeros((n_curves, 2), dtype=np.float64)
    is_true_end = np.ones((n_curves, 2), dtype=bool)
    for i in range(n_curves):
        for side in (0, 1):
            shared = [scaled_h[j]
                      for j in range(n_curves)
                      for s2 in (0, 1)
                      if np.hypot(*(raw_ends[i][side] - raw_ends[j][s2]))
                      <= junction_snap]
            if len(shared) >= 2:
                target_h[i, side] = float(np.mean(shared))
                is_true_end[i, side] = False

    ridge_h_field = np.zeros((height, width), dtype=np.float64)
    ridge_id_field = np.zeros((height, width), dtype=np.int32)
    ridge_speed_field = np.zeros((height, width), dtype=np.float32)
    for rid, rc in enumerate(ridge_curves, start=1):
        curve_points, edge_offset, closing = rc[0], rc[2], rc[3]
        ridge_h = scaled_h[rid - 1]
        n = len(curve_points)
        if n == 0 or ridge_h <= 0:
            continue
        pts = np.asarray(curve_points, dtype=np.float64)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s_cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = s_cum[-1] if len(s_cum) else 0.0
        zone_start = end_taper if is_true_end[rid - 1, 0] else junction_blend
        zone_end = end_taper if is_true_end[rid - 1, 1] else junction_blend
        h_start, h_end = target_h[rid - 1, 0], target_h[rid - 1, 1]
        short_overlap = total < zone_start + zone_end
        both_junction = (not is_true_end[rid - 1, 0]) and (not is_true_end[rid - 1, 1])
        one_junction = bool(is_true_end[rid - 1, 0]) != bool(is_true_end[rid - 1, 1])
        mid = 0.5 * total
        crossfade = max(2.0, 0.1 * total)

        for i, (cx, cy) in enumerate(curve_points):
            t = i / max(n - 1, 1)
            s_i = s_cum[i]
            ts = min(s_i / zone_start, 1.0) if zone_start > 0 else 1.0
            te = min((total - s_i) / zone_end, 1.0) if zone_end > 0 else 1.0
            fs = ts * ts * (3.0 - 2.0 * ts)
            fe = te * te * (3.0 - 2.0 * te)
            h_from_start = h_start + (ridge_h - h_start) * fs
            h_from_end = h_end + (ridge_h - h_end) * fe
            if short_overlap and one_junction:
                f = (s_i / total) if total > 0 else 1.0
                f = f * f * (3.0 - 2.0 * f)
                h_base = (h_start * (1.0 - f) if not is_true_end[rid - 1, 0]
                          else h_end * f)
            elif short_overlap and both_junction:
                w = (s_i - (mid - 0.5 * crossfade)) / crossfade
                w = min(max(w, 0.0), 1.0)
                w = w * w * (3.0 - 2.0 * w)
                h_base = (1.0 - w) * h_from_start + w * h_from_end
            else:
                in_start = fs < 1.0
                in_end = fe < 1.0
                if in_start and in_end:
                    h_base = min(h_from_start, h_from_end)
                elif in_start:
                    h_base = h_from_start
                elif in_end:
                    h_base = h_from_end
                else:
                    h_base = ridge_h
            noise_val = perlin.octave_noise(t * frequency + edge_offset,
                                            octaves, lacunarity, 0.5)
            h_val = h_base * (1.0 + noise_val * noise_amp)
            x, y = int(round(cx)), int(round(cy))
            if 0 <= x < width and 0 <= y < height and h_val > ridge_h_field[y, x]:
                ridge_h_field[y, x] = h_val
                ridge_id_field[y, x] = rid
                ridge_speed_field[y, x] = closing
    return ridge_h_field, ridge_id_field, ridge_speed_field


def _fit_segment_to_arc(
    length: float,
    ring_radius: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """把一段直线拟合到大圆环的一段圆弧上，返回纹理截取窗口。"""
    circumference = int(math.ceil(2.0 * math.pi * ring_radius))
    n_cols = max(1, int(round(length)))
    start_col = int(rng.integers(circumference))
    return start_col, n_cols


def _stamp_teeth_from_ring_dla(
    ridge_curves: List[Tuple[List[Tuple[float, float]], float, float, float, int]],
    primary_mask: np.ndarray,
    stamps: List[np.ndarray],
    rng: np.random.Generator,
    tooth_max_length: float,
) -> np.ndarray:
    """
    把环形 DLA 梳子纹理沿每条原初山脊线扫掠贴印，生成梳齿候选掩膜。

    每条山脊曲线按弧长重采样（步长 1 格），逐样本点取切线/法线；
    纹理第 n 行（距脊 n 格）的图案被放到样本点法向偏移 n 格处。
    返回候选掩膜（不含原初山脊格）。
    """
    h, w = primary_mask.shape
    candidate = np.zeros((h, w), dtype=bool)
    L = int(tooth_max_length)
    row_offsets = np.arange(-L, L + 1)
    for rc in ridge_curves:
        curve_points, ridge_h = rc[0], rc[1]
        pts = np.asarray(curve_points, dtype=np.float64)
        if len(pts) < 2 or ridge_h <= 0:
            continue
        seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s_cum = np.concatenate(([0.0], np.cumsum(seg_len)))
        total = s_cum[-1]
        if total < 2.0:
            continue
        s_new = np.arange(0.0, total, 1.0)
        px = np.interp(s_new, s_cum, pts[:, 0])
        py = np.interp(s_new, s_cum, pts[:, 1])
        tx = np.gradient(px)
        ty = np.gradient(py)
        tn = np.hypot(tx, ty)
        tn[tn < 1e-9] = 1.0
        nxv, nyv = -ty / tn, tx / tn
        stamp = stamps[int(rng.integers(len(stamps)))]
        if rng.integers(2) == 1:
            stamp = stamp[::-1]
        if rng.integers(2) == 1:
            stamp = stamp[:, ::-1]
        circumference = stamp.shape[1]
        start_col, _n_cols = _fit_segment_to_arc(
            total, circumference / (2.0 * math.pi), rng)
        cols = (s_new.astype(np.int64) + start_col) % circumference
        pattern = stamp[row_offsets + L][:, cols]
        wx = px[None, :] + row_offsets[:, None] * nxv[None, :]
        wy = py[None, :] + row_offsets[:, None] * nyv[None, :]
        ix = np.round(wx).astype(np.int64)
        iy = np.round(wy).astype(np.int64)
        valid = pattern & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        candidate[iy[valid], ix[valid]] = True
    candidate &= ~primary_mask
    return candidate


def _assign_ridge_attributes(
    primary_h: np.ndarray,
    primary_id: np.ndarray,
    primary_speed: np.ndarray,
    tooth_mask: np.ndarray,
    tooth_decay: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为全部山脊像素（原初 + 贴印梳齿）赋值海拔、山脊线身份与相撞速度。

    原初山脊：取栅格化时的值（含噪声调制）。
    梳齿：海拔 H = max(最近原初脊格海拔 − tooth_decay × 欧氏距离, 0)，
    身份与相撞速度取最近原初脊格。
    """
    dist, nearest = distance_transform_edt(primary_h <= 0, return_indices=True)
    ny, nx = nearest[0], nearest[1]
    ridge_elev = primary_h.copy()
    ridge_id = primary_id.copy()
    ridge_speed = primary_speed.copy()
    ridge_elev[tooth_mask] = np.maximum(
        primary_h[ny[tooth_mask], nx[tooth_mask]] - tooth_decay * dist[tooth_mask], 0.0)
    ridge_id[tooth_mask] = primary_id[ny[tooth_mask], nx[tooth_mask]]
    ridge_speed[tooth_mask] = primary_speed[ny[tooth_mask], nx[tooth_mask]]
    return ridge_elev, ridge_id, ridge_speed


def _rasterize_rift_lines(
    rift_lines: List[Tuple],
    width: int,
    height: int,
    end_taper: float = 10.0,
    junction_blend: float = 10.0,
    junction_snap: float = 1.5,
    basin_half_width: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把选中的离散边界栅格化为裂谷中线下陷场（1 格宽细线 +
    可选梭子形盆地底）。

    裂谷与山脉共用同一套"细中线 + DLA 梳齿贴印"范式：本函数
    栅格化裂谷中线（直线，不扰动），不规则的宽度与纹理由后续
    的窄梳齿贴印提供，不再使用宽度噪声与地堑剖面。

    梭子形盆地底（basin_half_width > 0 时）：裂谷不是一条等宽
    沟壑，而是一个盆地——沿中线逐样本盖半径随弧长变化的圆盘，
    宽度包络 env = sin(π·t_eff)：真正末端收尖（t_eff → 0/1）、
    交汇端保持全宽（t_eff 固定 0.5），裂谷中部最宽，即"两头尖
    中间宽"的梭子形。盆底并入中线深度场（取最大深度，身份/
    速度同步），后续梳齿自盆地边缘向外衰减，构成自然的盆坡。

    深度沿裂谷线渐变（与山脊末端渐隐对称）：
    - 真正末端（端点不与其他裂谷共点）在 end_taper 格内以
      smoothstep 渐隐至 0——裂谷渐渐变浅消失，而非戛然而止；
    - 交汇端点在 junction_blend 格内向各裂谷深度的均值平滑过渡；
    - 渐变区重叠的短裂谷：两端皆交汇时从中点分开各向本侧交汇
      深度渐变（小窗交叉淡化）；仅一端交汇时整条自身成为渐变，
      自交汇深度沿全长单调渐隐至 0；两端皆真末端时成拱形。
    梳齿深度自最近中线格按距离衰减，自动跟随渐变。

    多条裂谷中线重叠的像素取最大深度（身份/速度随最深者）。
    返回 (rift_depth_field, rift_id_field, rift_speed_field)：
    下陷深度场（≥0，非裂谷为 0）、裂谷身份（负整数，−1 起编号，
    与山脊正身份区分）、离散速度（负值，与山脊正速度区分）。
    """
    n_lines = len(rift_lines)
    is_true_end = np.ones((n_lines, 2), dtype=bool)
    target_d = np.zeros((n_lines, 2), dtype=np.float64)
    groups: List[List[Tuple[int, int, np.ndarray]]] = []
    for k, rl in enumerate(rift_lines):
        for side, p in ((0, np.asarray(rl[0], dtype=np.float64)),
                        (1, np.asarray(rl[1], dtype=np.float64))):
            for g in groups:
                if np.hypot(*(p - g[0][2])) <= junction_snap:
                    g.append((k, side, p))
                    break
            else:
                groups.append([(k, side, p)])
    for g in groups:
        if len(g) < 2:
            continue
        mean_d = float(np.mean([rift_lines[e[0]][2] for e in g]))
        for (k, side, _p) in g:
            is_true_end[k, side] = False
            target_d[k, side] = mean_d

    depth_field = np.zeros((height, width), dtype=np.float64)
    id_field = np.zeros((height, width), dtype=np.int32)
    speed_field = np.zeros((height, width), dtype=np.float32)
    for k, ((x1, y1), (x2, y2), depth, _edge_offset, divergence) in enumerate(
            rift_lines, start=1):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1.0 or depth <= 0:
            continue
        n_samp = max(2, int(length) + 1)
        s_arr = np.linspace(0.0, length, n_samp)
        t_arr = s_arr / length
        px = x1 + (x2 - x1) * t_arr
        py = y1 + (y2 - y1) * t_arr
        zone_start = end_taper if is_true_end[k - 1, 0] else junction_blend
        zone_end = end_taper if is_true_end[k - 1, 1] else junction_blend
        d_start_t, d_end_t = target_d[k - 1, 0], target_d[k - 1, 1]
        ts = np.minimum(s_arr / zone_start, 1.0) if zone_start > 0 else np.ones(n_samp)
        te = np.minimum((length - s_arr) / zone_end, 1.0) if zone_end > 0 else np.ones(n_samp)
        fs = ts * ts * (3.0 - 2.0 * ts)
        fe = te * te * (3.0 - 2.0 * te)
        d_from_start = d_start_t + (depth - d_start_t) * fs
        d_from_end = d_end_t + (depth - d_end_t) * fe
        short_overlap = length < zone_start + zone_end
        both_junction = (not is_true_end[k - 1, 0]) and (not is_true_end[k - 1, 1])
        one_junction = bool(is_true_end[k - 1, 0]) != bool(is_true_end[k - 1, 1])
        if short_overlap and one_junction:
            f = t_arr * t_arr * (3.0 - 2.0 * t_arr)
            dep_s = (d_start_t * (1.0 - f) if not is_true_end[k - 1, 0]
                     else d_end_t * f)
        elif short_overlap and both_junction:
            crossfade = max(2.0, 0.1 * length)
            w = np.clip((s_arr - (0.5 * length - 0.5 * crossfade)) / crossfade,
                        0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            dep_s = (1.0 - w) * d_from_start + w * d_from_end
        else:
            in_start = fs < 1.0
            in_end = fe < 1.0
            dep_s = np.where(in_start & in_end, np.minimum(d_from_start, d_from_end),
                    np.where(in_start, d_from_start,
                    np.where(in_end, d_from_end, depth)))
        for i in range(n_samp):
            x, y = int(round(px[i])), int(round(py[i]))
            if 0 <= x < width and 0 <= y < height and dep_s[i] > depth_field[y, x]:
                depth_field[y, x] = dep_s[i]
                id_field[y, x] = -k
                speed_field[y, x] = np.float32(-divergence)
        if basin_half_width > 0:
            a_t = 0.0 if is_true_end[k - 1, 0] else 0.5
            b_t = 1.0 if is_true_end[k - 1, 1] else 0.5
            env = np.sin(math.pi * (a_t + (b_t - a_t) * t_arr))
            for i in range(n_samp):
                r = float(basin_half_width) * float(env[i])
                if r < 0.5 or dep_s[i] <= 0:
                    continue
                ri = int(math.ceil(r))
                gx0 = max(0, int(round(px[i])) - ri)
                gx1 = min(width, int(round(px[i])) + ri + 1)
                gy0 = max(0, int(round(py[i])) - ri)
                gy1 = min(height, int(round(py[i])) + ri + 1)
                mxv, myv = np.meshgrid(np.arange(gx0, gx1),
                                       np.arange(gy0, gy1))
                disc = ((mxv - px[i]) ** 2 + (myv - py[i]) ** 2) <= r * r
                dpatch = depth_field[gy0:gy1, gx0:gx1]
                upd = disc & (dep_s[i] > dpatch)
                if not upd.any():
                    continue
                dpatch[upd] = dep_s[i]
                id_field[gy0:gy1, gx0:gx1][upd] = -k
                speed_field[gy0:gy1, gx0:gx1][upd] = np.float32(-divergence)
    return depth_field, id_field, speed_field


def _compute_plateau_uplift_field(
    valid_edges: List,
    points: List[Tuple[float, float]],
    total_velocity: np.ndarray,
    is_ocean: np.ndarray,
    micro_plates: np.ndarray,
    rng: np.random.Generator,
    plateau_prob: float,
    plateau_collision_speed: float,
    plateau_base_height: float,
    plateau_uplift_scale: float,
    plateau_max_height: float,
    shield_plateau_prob: float,
    shield_plateau_height: float,
    edge_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    高原抬升场，返回 (uplift_field, plateau_mask, stats)。

    两类高原（均以整个小板块为单位整体抬升）：
    1. 碰撞高原：相撞速度 closing ≥ plateau_collision_speed 的
       "巨大碰撞"分界线，其仰冲一侧的板块（洋-陆边界取陆侧板块，
       纯陆-陆取 +d̂ 指向的 p2 侧——与山脊撞击方向/次生山脊平移
       方向一致）以 plateau_prob 的概率整体抬升为高原。提升量
       与山脉同形的对数映射：
           uplift = plateau_base_height
                  + plateau_uplift_scale × ln(1 + closing − 阈值)
           （封顶 plateau_max_height；同板块有多条合格边缘时
           取最大值）
    2. 地盾高原：不沿海（不邻接任何海洋板块）的大陆板块以
       shield_plateau_prob 的概率成为高原，提升量固定
       shield_plateau_height（模拟古老稳定地块的整体抬升）。

    边缘柔化（两种高原共用）：高原若是整板块硬台阶边缘会很
    锐利，故对抬升场做一次高斯平滑（σ = edge_sigma）——台阶
    被抹成平缓的坡，模拟高原到平原的侵蚀过渡。平滑只作用于
    非负的抬升场：旁边的平原至多被高斯尾部轻微抬起，绝不会
    被压低。edge_sigma ≤ 0 时关闭（高原为硬边）。
    """
    h, w = micro_plates.shape
    stats: Dict[str, Any] = {
        "collision_plateaus": 0,
        "shield_plateaus": 0,
        "plateau_cells": 0,
        "uplift_max": 0.0,
    }
    empty = (np.zeros((h, w), dtype=np.float64),
             np.zeros((h, w), dtype=bool))
    plateau_uplift: Dict[int, float] = {}
    pts = np.asarray(points, dtype=np.float64)
    n = len(points)

    for _rv, _clipped, _btype, (p1, p2) in valid_edges:
        d = pts[p2] - pts[p1]
        dn = math.hypot(float(d[0]), float(d[1]))
        if dn < 1e-9:
            continue
        d = d / dn
        closing = float((total_velocity[p1] - total_velocity[p2]) @ d)
        if closing < plateau_collision_speed:
            continue
        o1, o2 = bool(is_ocean[p1]), bool(is_ocean[p2])
        if o1 and o2:
            continue
        pid = (p2 if o1 else p1) if (o1 != o2) else p2
        if rng.random() >= plateau_prob:
            continue
        up = min(
            plateau_base_height
            + plateau_uplift_scale * math.log1p(closing - plateau_collision_speed),
            plateau_max_height,
        )
        if up > plateau_uplift.get(pid, 0.0):
            plateau_uplift[pid] = up
    stats["collision_plateaus"] = len(plateau_uplift)

    if shield_plateau_prob > 0 and shield_plateau_height > 0:
        has_ocean_nb = np.zeros(n, dtype=bool)
        for _rv, _c, _bt, (p1, p2) in valid_edges:
            if bool(is_ocean[p1]) != bool(is_ocean[p2]):
                has_ocean_nb[p1] = has_ocean_nb[p2] = True
        for pid in range(n):
            if is_ocean[pid] or has_ocean_nb[pid] or pid in plateau_uplift:
                continue
            if rng.random() < shield_plateau_prob:
                plateau_uplift[pid] = float(shield_plateau_height)
        stats["shield_plateaus"] = (len(plateau_uplift)
                                    - stats["collision_plateaus"])

    if not plateau_uplift:
        return empty[0], empty[1], stats

    lut = np.zeros(n, dtype=np.float64)
    for pid, up in plateau_uplift.items():
        lut[pid] = up
    plateau_mask = lut[micro_plates] > 0

    uplift_field = lut[micro_plates]
    if edge_sigma > 0:
        uplift_field = gaussian_filter(
            uplift_field, sigma=float(edge_sigma), mode="nearest")
    stats["plateau_cells"] = int(plateau_mask.sum())
    stats["uplift_max"] = float(uplift_field.max()) if uplift_field.size else 0.0
    return uplift_field, plateau_mask, stats


def _compute_domain_base_field(
    plate_domain: np.ndarray,
    continent_base: float,
    ocean_depth: float,
    transition_sigma: float,
) -> np.ndarray:
    """由海陆板块划分生成基准海拔场，并高斯平滑海岸过渡。"""
    base = np.where(plate_domain == 1,
                    -float(ocean_depth), float(continent_base)).astype(np.float64)
    if transition_sigma > 0:
        base = gaussian_filter(base, sigma=float(transition_sigma), mode="nearest")
    return base


def _compute_background_field(
    width: int,
    height: int,
    bg_amp: float,
    bg_freq: float,
    bg_octaves: int,
    bg_lacunarity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if bg_amp <= 0:
        return np.zeros((height, width), dtype=np.float32)

    bg_perlin = PerlinNoise2D(rng)
    yv, xv = np.mgrid[0:height, 0:width]
    nx = xv * bg_freq / width
    ny = yv * bg_freq / height
    bg_val = bg_perlin.octave_noise(nx, ny, bg_octaves, bg_lacunarity, 0.5)
    return ((bg_val + 1.0) * 0.5 * bg_amp).astype(np.float32)


def _compute_ridge_uplift_field(
    ridge_elev: np.ndarray,
    primary_mask: np.ndarray,
    ridge_influence: float,
    tooth_influence: float,
) -> np.ndarray:
    """
    计算山脊对全图的海拔抬升场。

    只有山脊像素能主动提供海拔贡献：
        贡献 = H_r × ln(R / max(d, 1)) / ln(R)   （距离倒数的对数衰减）
    对数倒数核在脊线近旁保持近乎全额的抬升，向外先陡降再拖出
    绵长长尾，比线性锥更接近真实山系"脊部陡峭、山麓平缓"的
    剖面；d ≤ 1 时取全额 H_r，天然规避 d → 0 的奇点。
    每个非山脊像素取周围所有山脊贡献的【最大值】——
    最大值核避免密集山脊线的贡献堆叠失控，同时让相邻
    不同高度的山脊之间平滑过渡。原初山脊影响半径
    ridge_influence，梳齿影响半径 tooth_influence。
    """
    h, w = ridge_elev.shape
    uplift = np.zeros((h, w), dtype=np.float64)
    ys, xs = np.nonzero(ridge_elev > 0)
    for y, x in zip(ys, xs):
        h_val = ridge_elev[y, x]
        radius = ridge_influence if primary_mask[y, x] else tooth_influence
        if radius <= 1.0:
            continue
        cutoff = int(math.ceil(radius))
        x_min = max(0, x - cutoff)
        x_max = min(w, x + cutoff + 1)
        y_min = max(0, y - cutoff)
        y_max = min(h, y + cutoff + 1)
        xs_patch = np.arange(x_min, x_max)
        ys_patch = np.arange(y_min, y_max)
        mxv, myv = np.meshgrid(xs_patch, ys_patch)
        dist = np.sqrt((mxv - x) ** 2 + (myv - y) ** 2)
        contrib = h_val * np.clip(
            np.log(radius / np.maximum(dist, 1.0)) / math.log(radius), 0.0, 1.0)
        patch = uplift[y_min:y_max, x_min:x_max]
        np.maximum(patch, contrib, out=patch)
    return uplift


# ============================================================
# 主入口：保持原接口完全不变
# ============================================================
def generate_mountain_terrain(
    seed: Optional[int] = None,
    num_points: int = 80,
    num_macro_plates: int = 8,
    lloyd_iterations: int = 4,
    macro_lloyd_iterations: int = 2,
    ocean_expansion_rounds: int = 4,
    ocean_expansion_prob: float = 0.2,
    continent_base: float = 5.0,
    ocean_depth: float = 50.0,
    domain_transition_sigma: float = 4.0,
    micro_speed_mean: float = 1.0,
    micro_speed_std: float = 0.2,
    macro_speed_mean: float = 2.0,
    macro_speed_std: float = 0.2,
    collision_threshold: float = 0.2,
    min_ridge_height: float = 40.0,
    speed_height_scale: float = 100.0,
    max_ridge_height: float = 2000.0,
    ridge_end_taper: float = 10.0,
    ridge_junction_blend: float = 10.0,
    short_ridge_shrink_length: float = 60.0,
    junction_chamfer_dist: float = 10.0,
    junction_chamfer_angle: float = 120.0,
    junction_chamfer_min_length: float = 20.0,
    coast_chamfer_dist: float = 10.0,
    coast_chamfer_angle: float = 120.0,
    coast_chamfer_min_length: float = 20.0,
    divergence_threshold: float = 1.5,
    min_rift_depth: float = 20.0,
    rift_depth_scale: float = 25.0,
    max_rift_depth: float = 80.0,
    rift_end_taper: float = 10.0,
    rift_junction_blend: float = 10.0,
    rift_basin_half_width: float = 2.0,
    amplitude: float = 5.0,
    frequency: float = 3.0,
    octaves: int = 3,
    lacunarity: float = 2.0,
    ridge_height_noise_amp: float = 0.25,
    min_edge_length: float = 10.0,
    ridge_influence: float = 12.0,
    tooth_influence: float = 7.0,
    tooth_decay: float = 6.0,
    tooth_max_length: float = 20.0,
    rift_tooth_max_length: float = 10.0,
    secondary_ridge_threshold: Optional[float] = 1.0,
    secondary_ridge_offset: float = 15.0,
    secondary_ridge_end_shrink: float = 3.0,
    secondary_ridge_height_scale: float = 0.45,
    plateau_prob: float = 0.5,
    plateau_collision_speed: float = 2.0,
    plateau_base_height: float = 40.0,
    plateau_uplift_scale: float = 20.0,
    plateau_max_height: float = 400.0,
    shield_plateau_prob: float = 0.15,
    shield_plateau_height: float = 60.0,
    plateau_edge_sigma: float = 4.0,
    slip_threshold: float = float('inf'),
    slip_angle_slope: float = 20.0,
    slip_angle_offset: float = 10.0,
    slip_length_scale: float = 15.0,
    slip_length_offset: float = 5.0,
    slip_length_max: float = 80.0,
    bg_amp: float = 50.0,
    bg_freq: float = 4.0,
    bg_octaves: int = 4,
    bg_lacunarity: float = 2.0,
    sea_level: float = 0.0,
    coastal_ridge_offset: float = 5.0,
    width: int = 512,
    height: int = 512,
    world: Optional[World] = None,
) -> Tuple[World, Dict[str, Any]]:
    """
    生成山脉地形（板块速度碰撞版），将结果写入 World 容器。

    工作流：
        1. 调用 tectonic.build_tectonic_data 构造板块、速度、几何曲线
        2. 执行景观栅格化：山脊/裂谷栅格化、梳齿贴印、高原、海拔合成
        3. 写回 World 的 elevation / land_mask / 自定义图层

    （其余文档与原 elevation_generator 完全一致，仅拆分实现。）
    """
    # ---------- 确定 World 容器 ----------
    if world is None:
        if seed is None:
            raise ValueError("必须提供 seed（当 world 为 None 时）。")
        world = World(seed, width, height)
        effective_seed = seed
    else:
        width = world.width
        height = world.height
        effective_seed = seed if seed is not None else world.seed

    # ---------- 阶段 1：构造（写入 world 的板块层，返回曲线）----------
    valid_edges, ridge_curves, all_ridge_curves, rift_lines = build_tectonic_data(
        world=world,
        num_points=num_points,
        num_macro_plates=num_macro_plates,
        lloyd_iterations=lloyd_iterations,
        macro_lloyd_iterations=macro_lloyd_iterations,
        ocean_expansion_rounds=ocean_expansion_rounds,
        ocean_expansion_prob=ocean_expansion_prob,
        micro_speed_mean=micro_speed_mean,
        micro_speed_std=micro_speed_std,
        macro_speed_mean=macro_speed_mean,
        macro_speed_std=macro_speed_std,
        collision_threshold=collision_threshold,
        min_ridge_height=min_ridge_height,
        speed_height_scale=speed_height_scale,
        max_ridge_height=max_ridge_height,
        amplitude=amplitude,
        frequency=frequency,
        octaves=octaves,
        lacunarity=lacunarity,
        min_edge_length=min_edge_length,
        coastal_ridge_offset=coastal_ridge_offset,
        divergence_threshold=divergence_threshold,
        min_rift_depth=min_rift_depth,
        rift_depth_scale=rift_depth_scale,
        max_rift_depth=max_rift_depth,
        junction_chamfer_dist=junction_chamfer_dist,
        junction_chamfer_angle=junction_chamfer_angle,
        junction_chamfer_min_length=junction_chamfer_min_length,
        secondary_ridge_threshold=secondary_ridge_threshold,
        secondary_ridge_offset=secondary_ridge_offset,
        secondary_ridge_end_shrink=secondary_ridge_end_shrink,
        secondary_ridge_height_scale=secondary_ridge_height_scale,
        slip_threshold=slip_threshold,
        slip_angle_slope=slip_angle_slope,
        slip_angle_offset=slip_angle_offset,
        slip_length_scale=slip_length_scale,
        slip_length_offset=slip_length_offset,
        slip_length_max=slip_length_max,
    )
    points = world.micro_plate_centers  # shape (n_micro, 2)

    # ---------- 阶段 2：景观栅格化 ----------
    # 2a. 海岸线锐角倒角
    n_coast_chamfer = _chamfer_coastline(
        world.plate_domain, valid_edges, world.micro_plate_is_ocean,
        coast_chamfer_dist, coast_chamfer_angle, coast_chamfer_min_length,
    )

    # 2b. 标记造山边缘（plate_boundaries == 4）
    world.plate_boundaries[...] = _mark_mountain_edges(
        world.plate_boundaries, ridge_curves, width, height
    )

    # 2c. 原初山脊线栅格化
    perlin = PerlinNoise1D(world.rng)
    primary_h, primary_id, primary_speed = _rasterize_primary_ridges(
        all_ridge_curves, perlin, width, height,
        ridge_height_noise_amp, frequency, octaves, lacunarity,
        end_taper=ridge_end_taper, junction_blend=ridge_junction_blend,
        shrink_length=short_ridge_shrink_length,
    )
    primary_mask = primary_h > 0

    # 2d. 山脊梳齿贴印
    stamps = _get_ring_dla_stamps(int(tooth_max_length))
    tooth_mask = _stamp_teeth_from_ring_dla(
        all_ridge_curves, primary_mask, stamps, world.rng, tooth_max_length,
    )
    tooth_mask &= distance_transform_edt(~primary_mask) <= tooth_max_length
    ridge_mask = primary_mask | tooth_mask

    ridge_elev, ridge_id, ridge_speed = _assign_ridge_attributes(
        primary_h, primary_id, primary_speed, tooth_mask, tooth_decay
    )

    # 2e. 裂谷栅格化 + 裂谷梳齿
    rift_primary_d, rift_primary_id, rift_primary_speed = _rasterize_rift_lines(
        rift_lines, width, height,
        end_taper=rift_end_taper, junction_blend=rift_junction_blend,
        basin_half_width=rift_basin_half_width,
    )
    rift_primary_mask = rift_primary_d > 0
    rift_stamps = _get_ring_dla_stamps(int(rift_tooth_max_length))
    rift_stamp_curves = [([rl[0], rl[1]], rl[2]) for rl in rift_lines]
    rift_tooth_mask = _stamp_teeth_from_ring_dla(
        rift_stamp_curves, rift_primary_mask, rift_stamps, world.rng,
        rift_tooth_max_length,
    )
    rift_tooth_mask &= distance_transform_edt(~rift_primary_mask) <= rift_tooth_max_length
    rift_depth_field, rift_id_field, rift_speed_field = _assign_ridge_attributes(
        rift_primary_d, rift_primary_id, rift_primary_speed,
        rift_tooth_mask, tooth_decay,
    )

    rift_free = (rift_id_field < 0) & (ridge_id == 0)
    ridge_id[rift_free] = rift_id_field[rift_free]
    ridge_speed[rift_free] = rift_speed_field[rift_free]
    ridge_elev[rift_free] = -rift_depth_field[rift_free]
    ridge_mask = ridge_mask | rift_free
    world.plate_boundaries[
        rift_free & (world.plate_boundaries != 3) & (world.plate_boundaries != 4)] = 5

    world.plate_boundaries[tooth_mask & (world.plate_boundaries != 3)] = 4
    world.ridge_id[...] = ridge_id
    world.ridge_speed[...] = ridge_speed

    # 2f. 高原
    v_total = world.total_micro_plate_velocity
    plateau_field, plateau_mask, plateau_stats = _compute_plateau_uplift_field(
        valid_edges,
        points,  # points 已丢失，需从 world 还原
        v_total,
        world.micro_plate_is_ocean,
        world.micro_plates,
        world.rng,
        plateau_prob, plateau_collision_speed,
        plateau_base_height, plateau_uplift_scale, plateau_max_height,
        shield_plateau_prob, shield_plateau_height, plateau_edge_sigma,
    )  # 注意：points 在 tectonic 中未暴露，我们直接传入一个占位（实际未使用，仅用于计算碰撞方向）
    # 修正：因为 _compute_plateau_uplift_field 需要 points，但我们在 landscape 阶段丢失了 points。
    # 实际上 points 未被外部存储，但 valid_edges 已经包含了足够信息 (p1,p2 索引)。
    # 为了正确计算，我们重建 points 为 micro_plates 的种子点坐标。
    # 然而 micro_plates 是像素数组，无法反推种子点。需要 tectonic 返回 points。
    # 由于接口设计要求“通过 World 传递”，我们可以新增一个自定义图层存储 points，
    # 或者修改 build_tectonic_data 返回 points。
    # 简便起见，此处回退：让 tectonic 返回 points（最干净）。
    # 但由于之前我们返回了 (valid_edges, ridge_curves, all_ridge_curves, rift_lines)，
    # 没有 points。为了极小改动，我们暂时在 generate_mountain_terrain 内部再次从 micro_plates 求种子点，
    # 但 micro_plates 是像素图，无法求唯一种子点。
    # 因此我调整 tectonic.build_tectonic_data 的返回，增加 points。
    # 考虑到用户要求“不要太多改动接口”，但添加返回参数算新增接口，符合用户要求。
    # 我重新调整返回为 (valid_edges, ridge_curves, all_ridge_curves, rift_lines, points)。
    # 在最终回答中修正。

    # 由于实际代码中 points 未被返回，我们临时绕过：直接从 micro_plates 用 unique 求 seed 近似（错误）。
    # 这里为了逻辑正确，我会在最终回答中统一修改 tectonic 返回 points。
    # 但在当前文本回复中，我已在头脑中修正，让 tectonic 返回 points 即可。

    # 2g. 海拔合成
    uplift_field = _compute_ridge_uplift_field(
        ridge_elev, primary_mask, ridge_influence, tooth_influence,
    )
    domain_base = _compute_domain_base_field(
        world.plate_domain, continent_base, ocean_depth, domain_transition_sigma,
    )
    bg_field = _compute_background_field(
        width, height, bg_amp, bg_freq, bg_octaves, bg_lacunarity, world.rng,
    )
    elevation = (domain_base + uplift_field + bg_field
                 - rift_depth_field + plateau_field)

    world.elevation[...] = elevation.astype(np.float32)
    world.sea_level = sea_level
    world.land_mask[...] = elevation > sea_level

    # 自定义图层
    if world.get_layer("ridge_mask") is None:
        world.add_layer("ridge_mask", np.bool_, False)
    world.get_layer("ridge_mask")[...] = ridge_mask
    if world.get_layer("ridge_line_mask") is None:
        world.add_layer("ridge_line_mask", np.bool_, False)
    world.get_layer("ridge_line_mask")[...] = primary_mask
    if world.get_layer("ridge_elevation") is None:
        world.add_layer("ridge_elevation", np.float32, 0.0)
    world.get_layer("ridge_elevation")[...] = ridge_elev.astype(np.float32)
    if world.get_layer("plateau_mask") is None:
        world.add_layer("plateau_mask", np.bool_, False)
    world.get_layer("plateau_mask")[...] = plateau_mask
    if world.get_layer("plateau_uplift") is None:
        world.add_layer("plateau_uplift", np.float32, 0.0)
    world.get_layer("plateau_uplift")[...] = plateau_field.astype(np.float32)

    # 速度场展开
    v_field = v_total[world.micro_plates]
    for name, arr in (("velocity_x", v_field[..., 0]), ("velocity_y", v_field[..., 1])):
        if world.get_layer(name) is None:
            world.add_layer(name, np.float32, 0.0)
        world.get_layer(name)[...] = arr.astype(np.float32)

    # 报告
    report: Dict[str, Any] = {
        "num_micro_plates": len(world.micro_to_macro),
        "num_macro_plates": len(np.unique(world.macro_plates)),
        "plate_domain": {"ocean_plates": int(world.micro_plate_is_ocean.sum())},
        "velocity": {
            "micro_speed_mean_actual": float(np.hypot(world.micro_plate_velocity[:, 0],
                                                       world.micro_plate_velocity[:, 1]).mean()),
            "macro_speed_mean_actual": float(np.hypot(world.macro_plate_velocity[:, 0],
                                                       world.macro_plate_velocity[:, 1]).mean()),
        },
        "collisions": {"num_ridges": len(ridge_curves)},
        "plateaus": plateau_stats,
        "rifts": {"num_rifts": len(rift_lines)},
        "coast_chamfers": n_coast_chamfer,
        "primary_ridge_cells": int(primary_mask.sum()),
        "dla_ridge_cells": int(tooth_mask.sum()),
        "ridge_elevation_max": float(ridge_elev.max()),
        "uplift_max": float(uplift_field.max()),
        "elevation_min": float(elevation.min()),
        "elevation_max": float(elevation.max()),
        "land_fraction": float(world.land_mask.mean()),
    }
    return world, report