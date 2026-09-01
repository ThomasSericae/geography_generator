"""
tectonic.py – 板块构造与山脊/裂谷线几何生成（写入 World 的板块属性）

本模块负责：
    · 泊松采样 + Lloyd 松弛 → 点集
    · Voronoi 图构建
    · 小板块聚类为大板块（含 Lloyd 松弛）
    · 海陆板块划分（边缘海洋 + 概率扩张）→ 写入 plate_domain
    · 板块速度赋值（个体 + 集体）→ 写入 velocity 属性
    · 碰撞检测 → 山脊线几何曲线（含次生脊、倒角、侧滑分段）
    · 离散检测 → 裂谷线几何曲线（含倒角）

所有板块级数据（micro_plates / macro_plates / boundaries / domain / velocity）
均直接写入传入的 World 对象。返回纯几何曲线，供 landscape 栅格化使用。

通信约定：tectonic 依赖 world.rng 作为唯一随机源，不自行播种。
"""

import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import Voronoi, cKDTree

from world_core import World, PerlinNoise1D


# ============================================================
# 几何工具
# ============================================================
def _clip_line_segment(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    u1 = 0.0
    u2 = 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    if u1 > u2:
        return None
    nx1 = x1 + u1 * dx
    ny1 = y1 + u1 * dy
    nx2 = x1 + u2 * dx
    ny2 = y1 + u2 * dy
    return ((nx1, ny1), (nx2, ny2))


def _bridson_poisson_disc(width: float, height: float, r: float, k: int = 30) -> List[Tuple[float, float]]:
    cell_size = r / math.sqrt(2)
    grid_w = int(math.ceil(width / cell_size))
    grid_h = int(math.ceil(height / cell_size))
    grid = [[None for _ in range(grid_h)] for _ in range(grid_w)]

    def grid_coords(x, y):
        return int(x / cell_size), int(y / cell_size)

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def is_valid(x, y):
        if not in_bounds(x, y):
            return False
        gx, gy = grid_coords(x, y)
        for i in range(max(0, gx - 2), min(grid_w, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_h, gy + 3)):
                if grid[i][j] is not None:
                    px, py = grid[i][j]
                    if (x - px) ** 2 + (y - py) ** 2 < r * r:
                        return False
        return True

    x0 = random.uniform(0, width)
    y0 = random.uniform(0, height)
    points = [(x0, y0)]
    gx, gy = grid_coords(x0, y0)
    grid[gx][gy] = (x0, y0)
    active = [0]

    while active:
        idx = random.randrange(len(active))
        i = active[idx]
        x, y = points[i]
        found = False
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(r, 2 * r)
            nx = x + dist * math.cos(angle)
            ny = y + dist * math.sin(angle)
            if is_valid(nx, ny):
                points.append((nx, ny))
                gx, gy = grid_coords(nx, ny)
                grid[gx][gy] = (nx, ny)
                active.append(len(points) - 1)
                found = True
                break
        if not found:
            active.pop(idx)

    return points


def _poisson_disc_sample(
    width: float, height: float, target_count: int, seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)
    area = width * height
    r = math.sqrt(0.5 * area / (math.pi * target_count))
    for _ in range(20):
        points = _bridson_poisson_disc(width, height, r)
        if len(points) >= target_count:
            return random.sample(points, target_count)
        r *= 0.9
    points = list(points)
    while len(points) < target_count:
        points.append((random.uniform(0, width), random.uniform(0, height)))
    return points


def _lloyd_relaxation(
    points: List[Tuple[float, float]], width: float, height: float, iterations: int
) -> List[Tuple[float, float]]:
    points_arr = np.array(points, dtype=float)
    n = len(points_arr)
    margin = max(width, height) * 10
    for _ in range(iterations):
        boundary = np.array([
            [-margin, -margin],
            [width + margin, -margin],
            [width + margin, height + margin],
            [-margin, height + margin]
        ])
        all_points = np.vstack([points_arr, boundary])
        vor = Voronoi(all_points)
        new_points = []
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region:
                new_points.append(points_arr[i])
                continue
            vertices = np.array([vor.vertices[j] for j in region])
            m = len(vertices)
            area = 0.0
            cx = 0.0
            cy = 0.0
            for j in range(m):
                x1, y1 = vertices[j]
                x2, y2 = vertices[(j + 1) % m]
                cross = x1 * y2 - x2 * y1
                area += cross
                cx += (x1 + x2) * cross
                cy += (y1 + y2) * cross
            area *= 0.5
            if abs(area) > 1e-10:
                cx /= (6 * area)
                cy /= (6 * area)
                cx = max(0, min(width, cx))
                cy = max(0, min(height, cy))
                new_points.append([cx, cy])
            else:
                new_points.append(points_arr[i])
        points_arr = np.array(new_points)
    return points_arr.tolist()


def _build_voronoi(
    points: List[Tuple[float, float]], width: int, height: int
) -> Tuple[Voronoi, int]:
    n = len(points)
    margin = max(width, height) * 10
    boundary = np.array([
        [-margin, -margin],
        [width + margin, -margin],
        [width + margin, height + margin],
        [-margin, height + margin]
    ])
    all_points = np.vstack([np.array(points), boundary])
    vor = Voronoi(all_points)
    return vor, n


def _cluster_micro_to_macro(
    points: List[Tuple[float, float]],
    num_macro_plates: int,
    seed: int,
    lloyd_iterations: int = 2,
) -> np.ndarray:
    """
    在小板块中心点上进行 K-Means 式的 Lloyd 松弛迭代，
    使大板块中心趋向均匀分布，从而让大板块边界更平滑、更规则。
    """
    rng = np.random.RandomState(seed)
    n = len(points)
    num_macro_plates = min(num_macro_plates, n)

    center_indices = rng.choice(n, num_macro_plates, replace=False)
    centers = np.array(points, dtype=float)[center_indices].copy()
    points_arr = np.array(points, dtype=float)

    for _ in range(lloyd_iterations):
        tree = cKDTree(centers)
        _, macro_ids = tree.query(points_arr)

        new_centers = np.zeros_like(centers)
        for m in range(num_macro_plates):
            mask = macro_ids == m
            if np.any(mask):
                new_centers[m] = points_arr[mask].mean(axis=0)
            else:
                new_centers[m] = points_arr[rng.randint(n)]
        centers = new_centers

    tree = cKDTree(centers)
    _, macro_ids = tree.query(points_arr)
    return macro_ids.astype(np.int32)


def _assign_micro_plates(
    points: List[Tuple[float, float]], width: int, height: int
) -> np.ndarray:
    tree = cKDTree(points)
    yv, xv = np.mgrid[0:height, 0:width]
    coords = np.column_stack([xv.ravel(), yv.ravel()])
    _, indices = tree.query(coords)
    return indices.reshape(height, width).astype(np.int32)


def _assign_macro_plates(
    micro_plates: np.ndarray, macro_id_per_micro: np.ndarray
) -> np.ndarray:
    return macro_id_per_micro[micro_plates]


def _detect_pixel_boundaries(
    micro_plates: np.ndarray,
    macro_plates: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    boundaries = np.zeros((height, width), dtype=np.int32)

    h_micro = micro_plates[:, :-1] != micro_plates[:, 1:]
    h_macro = macro_plates[:, :-1] != macro_plates[:, 1:]
    boundaries[:, :-1][h_micro] = 1
    boundaries[:, :-1][h_macro] = 2

    v_micro = micro_plates[:-1, :] != micro_plates[1:, :]
    v_macro = macro_plates[:-1, :] != macro_plates[1:, :]
    boundaries[:-1, :][v_micro] = 1
    boundaries[:-1, :][v_macro] = 2

    boundaries[0, :] = 3
    boundaries[-1, :] = 3
    boundaries[:, 0] = 3
    boundaries[:, -1] = 3

    return boundaries


def _classify_voronoi_edges(
    vor: Voronoi,
    n: int,
    macro_id_per_micro: np.ndarray,
    width: int,
    height: int,
) -> List[Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]], int, Tuple[int, int]]]:
    """
    返回 [(ridge_vertices, clipped_endpoints, btype, (p1, p2)), ...]。

    btype：2 = 跨大板块边界，1 = 大板块内部的小板块边界；
    (p1, p2) 为分界线两侧的小板块编号（碰撞检测与速度查询用）。
    """
    valid_edges = []
    for ridge_vertices, ridge_points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 in ridge_vertices:
            continue
        if not any(p < n for p in ridge_points):
            continue

        p1, p2 = ridge_points
        if p1 >= n or p2 >= n:
            continue

        macro1 = macro_id_per_micro[p1]
        macro2 = macro_id_per_micro[p2]

        if macro1 != macro2:
            btype = 2
        else:
            btype = 1

        v1 = vor.vertices[ridge_vertices[0]]
        v2 = vor.vertices[ridge_vertices[1]]
        clipped = _clip_line_segment(v1[0], v1[1], v2[0], v2[1], 0, 0, width, height)
        if not clipped:
            continue

        valid_edges.append((ridge_vertices, clipped, btype, (int(p1), int(p2))))

    return valid_edges


def _assign_plate_velocity(
    rng: np.random.Generator,
    n_micro: int,
    macro_id_per_micro: np.ndarray,
    n_macro: int,
    micro_speed_mean: float,
    micro_speed_std: float,
    macro_speed_mean: float,
    macro_speed_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为每个小板块与大板块赋予速度向量。

    方向：均匀随机 [0, 2π)。
    大小：正态分布 N(mean, std²)，负值截断为 0。
    小板块总速度 = 个体速度 + 所属大板块集体速度。

    返回 (v_micro_individual, v_macro_collective, v_micro_total)，
    均为 (n, 2) float64，列向量为 (vx, vy)。
    """
    def _sample(count: int, mean: float, std: float) -> np.ndarray:
        speed = np.maximum(rng.normal(mean, std, size=count), 0.0)
        angle = rng.uniform(0.0, 2.0 * math.pi, size=count)
        return np.column_stack((speed * np.cos(angle), speed * np.sin(angle)))

    v_macro = _sample(n_macro, macro_speed_mean, macro_speed_std)
    v_micro = _sample(n_micro, micro_speed_mean, micro_speed_std)
    v_total = v_micro + v_macro[macro_id_per_micro]
    return v_micro, v_macro, v_total


def _classify_land_ocean_plates(
    micro_plates: np.ndarray,
    vor: Voronoi,
    n: int,
    rng: np.random.Generator,
    expansion_rounds: int,
    expansion_prob: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    划分海陆板块，返回 (is_ocean (n,) bool, 统计字典)。

    1. 所有与地图边缘相接的小板块一律划为海洋板块；
    2. 扩张：每轮找出所有与现有海洋板块相邻（Voronoi 边共享）
       的大陆板块，各以 expansion_prob 概率转化为海洋板块，
       重复 expansion_rounds 轮。轮内同步翻转，保证扩张一轮
       只向外推进一层板块，海洋因此连续，形成"海中大洲"。
    """
    h, w = micro_plates.shape
    is_ocean = np.zeros(n, dtype=bool)

    border_ids = np.unique(np.concatenate([
        micro_plates[0, :], micro_plates[-1, :],
        micro_plates[:, 0], micro_plates[:, -1],
    ]))
    is_ocean[border_ids] = True

    adjacency: List[set] = [set() for _ in range(n)]
    for p1, p2 in vor.ridge_points:
        if p1 < n and p2 < n:
            adjacency[p1].add(p2)
            adjacency[p2].add(p1)

    stats: Dict[str, Any] = {
        "edge_ocean_plates": int(border_ids.size),
        "expansion_rounds": [],
    }

    for _round in range(max(int(expansion_rounds), 0)):
        candidates = [
            p for p in range(n)
            if not is_ocean[p] and any(is_ocean[q] for q in adjacency[p])
        ]
        converted = 0
        for p in candidates:
            if rng.random() < expansion_prob:
                is_ocean[p] = True
                converted += 1
        stats["expansion_rounds"].append({
            "candidates": len(candidates),
            "converted": converted,
        })

    stats["ocean_plates"] = int(is_ocean.sum())
    stats["land_plates"] = int(n - is_ocean.sum())
    return is_ocean, stats


def _select_collision_ridges(
    valid_edges: List,
    points: List[Tuple[float, float]],
    total_velocity: np.ndarray,
    collision_threshold: float,
    min_ridge_height: float,
    speed_height_scale: float,
    max_ridge_height: float,
    perlin: PerlinNoise1D,
    amplitude: float,
    frequency: float,
    octaves: int,
    lacunarity: float,
    min_edge_length: float,
    is_ocean: np.ndarray,
    coastal_ridge_offset: float,
) -> Tuple[List[Tuple[List[Tuple[float, float]], float, float, float, int]], Dict[str, Any]]:
    """
    按板块速度碰撞检测选取山脊线。

    对每条板块分界线（两侧小板块 p1、p2）：
        d̂      = unit(P2 − P1)（板块中心连线方向）
        closing = (V_p1 − V_p2) · d̂   （相撞速度；>0 表示相互接近）
    closing ≥ collision_threshold 的边界被选为山脊线，其高度（对数映射）：
        ridge_h = min_ridge_height + speed_height_scale × ln(1 + closing − threshold)
        （封顶 max_ridge_height）
    对数压缩右尾：相撞速度很大时高度增长趋缓，山脊高度分布更均匀。
    因大板块集体速度均值更大，跨大板块边界（btype=2）的 closing
    通常显著大于大板块内部边界（btype=1），山系自然分级。

    海岸山脉位移：洋-陆碰撞边界（恰有一侧为海洋板块）的山脊线
    沿碰撞矢量向陆地一侧平移 coastal_ridge_offset 格——洋板块
    俯冲于陆板块之下，造山带形成于陆侧、距海沟一段距离处。
    Voronoi 边是两板块中心连线的中垂线，故边界法向即 d̂ 方向，
    位移方向取指向陆地一侧的 ±d̂。纯陆-陆碰撞不位移。

    返回 (ridge_curves, stats)。ridge_curves 元素为
    (curve_points, ridge_h, edge_offset, closing, btype,
     raw_start, raw_end, shift_dir)；raw_start/raw_end 为位移前的
    原始边缘端点，供山脊连续性（共享 Voronoi 顶点）判定使用；
    shift_dir 为单位撞击方向（洋-陆边界指向陆侧，纯陆-陆取 +d̂），
    供次生山脊平移使用。
    """
    pts = np.asarray(points, dtype=np.float64)
    ridge_curves = []
    stats: Dict[str, Any] = {
        "edges_examined": 0,
        "edges_too_short": 0,
        "edges_below_threshold": 0,
        "macro_ridges": 0,
        "micro_ridges": 0,
        "coastal_ridges_offset": 0,
    }
    macro_closings: List[float] = []
    micro_closings: List[float] = []
    heights: List[float] = []

    for ridge_vertices, clipped, btype, (p1, p2) in valid_edges:
        stats["edges_examined"] += 1
        (x1, y1), (x2, y2) = clipped
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_edge_length:
            stats["edges_too_short"] += 1
            continue

        d = pts[p2] - pts[p1]
        dn = math.hypot(float(d[0]), float(d[1]))
        if dn < 1e-9:
            continue
        d = d / dn
        closing = float((total_velocity[p1] - total_velocity[p2]) @ d)
        if closing < collision_threshold:
            stats["edges_below_threshold"] += 1
            continue

        ridge_h = min(
            min_ridge_height + speed_height_scale * math.log1p(closing - collision_threshold),
            max_ridge_height,
        )

        shift_x = shift_y = 0.0
        sd_x, sd_y = float(d[0]), float(d[1])
        if is_ocean is not None:
            o1, o2 = bool(is_ocean[p1]), bool(is_ocean[p2])
            if o1 != o2:
                sign = 1.0 if o1 else -1.0
                sd_x, sd_y = sign * float(d[0]), sign * float(d[1])
                if coastal_ridge_offset > 0:
                    shift_x = sd_x * coastal_ridge_offset
                    shift_y = sd_y * coastal_ridge_offset
                    stats["coastal_ridges_offset"] += 1

        dx = x2 - x1
        dy = y2 - y1
        nx = -dy / length
        ny = dx / length

        edge_offset = ((x1 * 0.374761393) + (y1 * 0.668265263) +
                       (x2 * 0.171) + (y2 * 0.513)) * 3.7

        num_samples = max(12, min(300, int(length / 1.5)))
        curve_points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            bx = x1 + dx * t
            by = y1 + dy * t
            noise_val = perlin.octave_noise(t * frequency + edge_offset,
                                            octaves, lacunarity, 0.5)
            envelope = math.sin(math.pi * t)
            offset = noise_val * amplitude * envelope
            curve_points.append((bx + nx * offset + shift_x,
                                 by + ny * offset + shift_y))

        ridge_curves.append((curve_points, ridge_h, edge_offset, closing, btype,
                             (x1, y1), (x2, y2), (sd_x, sd_y)))
        heights.append(ridge_h)
        if btype == 2:
            stats["macro_ridges"] += 1
            macro_closings.append(closing)
        else:
            stats["micro_ridges"] += 1
            micro_closings.append(closing)

    all_closings = macro_closings + micro_closings
    stats["num_ridges"] = len(ridge_curves)
    stats["closing_speed_min"] = float(min(all_closings)) if all_closings else 0.0
    stats["closing_speed_max"] = float(max(all_closings)) if all_closings else 0.0
    stats["closing_speed_macro_mean"] = (
        float(np.mean(macro_closings)) if macro_closings else 0.0)
    stats["closing_speed_micro_mean"] = (
        float(np.mean(micro_closings)) if micro_closings else 0.0)
    stats["ridge_height_min"] = float(min(heights)) if heights else 0.0
    stats["ridge_height_max"] = float(max(heights)) if heights else 0.0
    return ridge_curves, stats


def _select_divergence_rifts(
    valid_edges: List,
    points: List[Tuple[float, float]],
    total_velocity: np.ndarray,
    divergence_threshold: float,
    min_rift_depth: float,
    rift_depth_scale: float,
    max_rift_depth: float,
    min_edge_length: float,
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    按板块速度离散检测选取裂谷线（与碰撞山脊逻辑对称）。

    对每条板块分界线（两侧小板块 p1、p2）：
        d̂      = unit(P2 − P1)（板块中心连线方向）
        closing = (V_p1 − V_p2) · d̂   （>0 相撞；<0 相互远离）
    closing ≤ −divergence_threshold 的边界被选为裂谷线，其深度
    （对数映射，与山脉同形但量级明显更小）：
        rift_depth = min_rift_depth + rift_depth_scale × ln(1 + |closing| − threshold)
        （封顶 max_rift_depth）

    返回 (rift_lines, stats)，rift_lines 元素为
    ((x1, y1), (x2, y2), rift_depth, edge_offset, |closing|)。
    """
    pts = np.asarray(points, dtype=np.float64)
    rift_lines = []
    stats: Dict[str, Any] = {
        "edges_examined": 0,
        "edges_too_short": 0,
        "edges_above_threshold": 0,
        "num_rifts": 0,
    }
    divergences: List[float] = []
    depths: List[float] = []

    for _ridge_vertices, clipped, _btype, (p1, p2) in valid_edges:
        stats["edges_examined"] += 1
        (x1, y1), (x2, y2) = clipped
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_edge_length:
            stats["edges_too_short"] += 1
            continue
        d = pts[p2] - pts[p1]
        dn = math.hypot(float(d[0]), float(d[1]))
        if dn < 1e-9:
            continue
        d = d / dn
        closing = float((total_velocity[p1] - total_velocity[p2]) @ d)
        if closing > -divergence_threshold:
            stats["edges_above_threshold"] += 1
            continue

        divergence = -closing
        rift_depth = min(
            min_rift_depth + rift_depth_scale * math.log1p(divergence - divergence_threshold),
            max_rift_depth,
        )
        edge_offset = ((x1 * 0.374761393) + (y1 * 0.668265263) +
                       (x2 * 0.171) + (y2 * 0.513)) * 3.7
        rift_lines.append(((x1, y1), (x2, y2), rift_depth, edge_offset, divergence))
        divergences.append(divergence)
        depths.append(rift_depth)

    stats["num_rifts"] = len(rift_lines)
    stats["divergence_speed_min"] = float(min(divergences)) if divergences else 0.0
    stats["divergence_speed_max"] = float(max(divergences)) if divergences else 0.0
    stats["rift_depth_min"] = float(min(depths)) if depths else 0.0
    stats["rift_depth_max"] = float(max(depths)) if depths else 0.0
    return rift_lines, stats


def _chamfer_polyline_junctions(
    polylines: List[np.ndarray],
    cut_dist: float,
    min_angle_deg: float,
    min_line_length: float,
    snap: float = 1.5,
) -> Tuple[List[np.ndarray], List[Tuple], Dict[Tuple[int, int], np.ndarray]]:
    """
    对共享端点的折线组做尖角倒角（山脊/裂谷共用的几何核心）。

    夹角小于 min_angle_deg 的交汇过于锐利：在两条线上距交点
    cut_dist 格处截断（删掉交点侧的线段），并新增一条连接两个
    截点的倒角短线。原尖角 α 被两个 (180°+α)/2 的钝角取代
    （如 90° → 两个 135°），线条变得圆润。长度不足
    min_line_length 的线受保护，不参与倒角。单次执行，不递归。

    返回 (新折线列表, 倒角线列表, 截断记录)。倒角线元素为
    (倒角折线, 父线i, 侧i, 父线j, 侧j)，供调用方继承属性；
    截断记录 {(线, 侧): 截点} 供调用方同步端点信息——截点坐标
    在父线与倒角线间完全一致，下游端点共点识别会视为交汇。
    """
    n = len(polylines)
    lengths = np.zeros(n)
    for i, pts in enumerate(polylines):
        if len(pts) >= 2:
            lengths[i] = float(
                np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())
    protected = (lengths < min_line_length) | (lengths < cut_dist + 1e-6)

    groups: List[List[Tuple[int, int, np.ndarray]]] = []
    for i, pts in enumerate(polylines):
        if len(pts) < 2:
            continue
        for side, ep in ((0, pts[0]), (1, pts[-1])):
            for g in groups:
                if np.hypot(*(ep - g[0][2])) <= snap:
                    g.append((i, side, ep))
                    break
            else:
                groups.append([(i, side, ep)])

    def cut_point(i: int, side: int) -> Optional[np.ndarray]:
        """沿折线从 side 端量 cut_dist 格处的点（线长不足则 None）。"""
        pts = polylines[i] if side == 0 else polylines[i][::-1]
        acc = 0.0
        for k in range(1, len(pts)):
            seg = float(np.hypot(*(pts[k] - pts[k - 1])))
            if acc + seg >= cut_dist:
                t = (cut_dist - acc) / seg if seg > 1e-12 else 0.0
                return pts[k - 1] + t * (pts[k] - pts[k - 1])
            acc += seg
        return None

    truncations: Dict[Tuple[int, int], np.ndarray] = {}
    chamfer_pairs: List[Tuple] = []
    for g in groups:
        if len(g) < 2:
            continue
        V = np.mean([e[2] for e in g], axis=0)
        rays = []
        for (i, side, _ep) in g:
            if protected[i]:
                continue
            cp = cut_point(i, side)
            if cp is None:
                continue
            dvec = cp - V
            dn = float(np.hypot(*dvec))
            if dn < 1e-9:
                continue
            rays.append((i, side, cp, dvec / dn))
        if len(rays) < 2:
            continue
        rays.sort(key=lambda r: math.atan2(r[3][1], r[3][0]))
        angs = [math.atan2(r[3][1], r[3][0]) for r in rays]
        m = len(rays)
        for a in range(m):
            b = (a + 1) % m
            sector = math.degrees((angs[b] - angs[a]) % (2.0 * math.pi))
            if sector >= min_angle_deg:
                continue
            i, si, ci = rays[a][0], rays[a][1], rays[a][2]
            j, sj, cj = rays[b][0], rays[b][1], rays[b][2]
            if i == j:
                continue
            truncations[(i, si)] = ci
            truncations[(j, sj)] = cj
            chamfer_pairs.append((i, si, ci, j, sj, cj))

    new_polylines: List[np.ndarray] = []
    for i, pts in enumerate(polylines):
        pts_new = pts
        for side in (0, 1):
            cp = truncations.get((i, side))
            if cp is None:
                continue
            arr = pts_new if side == 0 else pts_new[::-1]
            out = [cp]
            acc = 0.0
            for k in range(1, len(arr)):
                seg = float(np.hypot(*(arr[k] - arr[k - 1])))
                if acc + seg >= cut_dist - 1e-9:
                    out.extend(arr[k:])
                    break
                acc += seg
            pts_new = np.asarray(out[::-1] if side == 1 else out)
        new_polylines.append(pts_new)

    chamfer_lines: List[Tuple] = []
    for (i, si, ci, j, sj, cj) in chamfer_pairs:
        span = float(np.hypot(*(cj - ci)))
        npts = max(2, int(span / 1.5) + 1)
        seg_pts = np.linspace(ci, cj, npts)
        chamfer_lines.append((seg_pts, i, si, j, sj))
    return new_polylines, chamfer_lines, truncations


def _chamfer_ridge_junctions(
    ridge_curves: List[Tuple],
    cut_dist: float,
    min_angle_deg: float,
    min_line_length: float,
) -> Tuple[List[Tuple], int]:
    """
    山脊尖角倒角。倒角山脊线的高度/相撞速度取两父线均值，
    btype 取较大者；截点坐标在父线新端点与倒角线端点间完全
    一致，下游（原初山脊栅格化）的端点共点识别会把它们视为
    交汇，高度过渡由既有的交汇渐变机制自动完成。
    """
    if cut_dist <= 0 or len(ridge_curves) < 2:
        return ridge_curves, 0
    polylines = [np.asarray(rc[0], dtype=np.float64) for rc in ridge_curves]
    new_lines, chamfer_lines, truncations = _chamfer_polyline_junctions(
        polylines, cut_dist, min_angle_deg, min_line_length)
    if not chamfer_lines:
        return ridge_curves, 0
    new_curves: List[Tuple] = []
    for i, rc in enumerate(ridge_curves):
        rs, re_ = rc[5], rc[6]
        if (i, 0) in truncations:
            c = truncations[(i, 0)]
            rs = (float(c[0]), float(c[1]))
        if (i, 1) in truncations:
            c = truncations[(i, 1)]
            re_ = (float(c[0]), float(c[1]))
        new_curves.append(([(float(p[0]), float(p[1])) for p in new_lines[i]],
                           rc[1], rc[2], rc[3], rc[4], rs, re_, rc[7]))
    for (seg_pts, i, _si, j, _sj) in chamfer_lines:
        h_new = 0.5 * (ridge_curves[i][1] + ridge_curves[j][1])
        closing_new = 0.5 * (ridge_curves[i][3] + ridge_curves[j][3])
        btype_new = max(ridge_curves[i][4], ridge_curves[j][4])
        ax, ay = float(seg_pts[0][0]), float(seg_pts[0][1])
        bx, by = float(seg_pts[-1][0]), float(seg_pts[-1][1])
        eo = ((ax * 0.374761393) + (ay * 0.668265263) +
              (bx * 0.171) + (by * 0.513)) * 3.7
        sd = (np.asarray(ridge_curves[i][7], dtype=np.float64)
              + np.asarray(ridge_curves[j][7], dtype=np.float64))
        sd_n = math.hypot(float(sd[0]), float(sd[1]))
        sd_t = ((float(sd[0]) / sd_n, float(sd[1]) / sd_n)
                if sd_n > 1e-9 else (0.0, 0.0))
        new_curves.append(([(float(p[0]), float(p[1])) for p in seg_pts],
                           h_new, eo, closing_new, btype_new,
                           (ax, ay), (bx, by), sd_t))
    return new_curves, len(chamfer_lines)


def _chamfer_rift_junctions(
    rift_lines: List[Tuple],
    cut_dist: float,
    min_angle_deg: float,
    min_line_length: float,
) -> Tuple[List[Tuple], int]:
    """
    裂谷尖角倒角（与山脊同一几何规则）。倒角裂谷线的深度与
    离散速度取两父线均值；裂谷无高度渐变机制，交叉处下陷场
    取最大深度，衔接自然平滑。
    """
    if cut_dist <= 0 or len(rift_lines) < 2:
        return rift_lines, 0
    polylines = [np.asarray([rl[0], rl[1]], dtype=np.float64)
                 for rl in rift_lines]
    new_lines, chamfer_lines, truncations = _chamfer_polyline_junctions(
        polylines, cut_dist, min_angle_deg, min_line_length)
    if not chamfer_lines:
        return rift_lines, 0
    new_rifts: List[Tuple] = []
    for i, rl in enumerate(rift_lines):
        p0, p1 = rl[0], rl[1]
        if (i, 0) in truncations:
            c = truncations[(i, 0)]
            p0 = (float(c[0]), float(c[1]))
        if (i, 1) in truncations:
            c = truncations[(i, 1)]
            p1 = (float(c[0]), float(c[1]))
        new_rifts.append((p0, p1, rl[2], rl[3], rl[4]))
    for (seg_pts, i, _si, j, _sj) in chamfer_lines:
        depth_new = 0.5 * (rift_lines[i][2] + rift_lines[j][2])
        div_new = 0.5 * (rift_lines[i][4] + rift_lines[j][4])
        ax, ay = float(seg_pts[0][0]), float(seg_pts[0][1])
        bx, by = float(seg_pts[-1][0]), float(seg_pts[-1][1])
        eo = ((ax * 0.374761393) + (ay * 0.668265263) +
              (bx * 0.171) + (by * 0.513)) * 3.7
        new_rifts.append(((ax, ay), (bx, by), depth_new, eo, div_new))
    return new_rifts, len(chamfer_lines)


def _shrink_polyline_ends(pts: np.ndarray, shrink: float) -> np.ndarray:
    """
    沿弧长自折线两端各裁去 shrink 格（端点为精确切点，线性插值）。

    山脊曲线统一为弧长采样折线（源自单条 Voronoi 边 + 噪声扰动），
    无论近似直线还是明显弯曲，端部缩短都按折线弧长裁切。
    裁后不足两个点（剩余弧长 < 1 格）时返回空数组，由调用方放弃。
    """
    if len(pts) < 2 or shrink <= 0:
        return pts
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total <= 2.0 * shrink + 1.0:
        return pts[:0]
    lo, hi = float(shrink), total - float(shrink)
    s_new = np.concatenate(([lo], s[(s > lo) & (s < hi)], [hi]))
    nx = np.interp(s_new, s, pts[:, 0])
    ny = np.interp(s_new, s, pts[:, 1])
    return np.column_stack((nx, ny))


def _generate_secondary_ridges(
    ridge_curves: List[Tuple],
    threshold: Optional[float],
    offset: float,
    end_shrink: float,
    height_scale: float,
) -> List[Tuple]:
    """
    次生山脊（平行第二皱褶 / 前陆褶皱带）生成。

    自然地理对应：板块相撞速度很大时，主造山带后方（仰冲板块
    一侧）会发育一条低矮的平行褶皱带——它不位于板块分界线上。

    对 closing > threshold 的山脊曲线：
        1. 平移：整条曲线沿撞击方向 shift_dir（选取山脊时记录：
           洋-陆边界指向陆侧，纯陆-陆取 +d̂）平移 offset 格；
        2. 两端缩短：沿曲线弧长自两端各裁去 end_shrink 格
           （山脊曲线统一为弧长采样折线，直的弯的都按折线两端裁）；
        3. 压低：高度 × height_scale（低矮的第二皱褶）。

    返回次生山脊曲线列表（与山脊曲线同构的 8 元组；raw 端点取
    平移缩短后的实际端点——次生山脊不与主山脊共享 Voronoi
    顶点，不参与交汇渐变，两端按真正末端渐隐）。与主山脊共用
    下游栅格化与梳齿贴印通道，DLA 梳齿纹样并集贴印，允许少量
    重叠（重叠区海拔取最近脊格，两脊之间自然成谷）。
    """
    out: List[Tuple] = []
    if threshold is None or offset <= 0:
        return out
    for rc in ridge_curves:
        closing = rc[3]
        if closing <= threshold:
            continue
        sd_x, sd_y = rc[7]
        if sd_x == 0.0 and sd_y == 0.0:
            continue
        pts = np.asarray(rc[0], dtype=np.float64)
        if len(pts) < 2:
            continue
        pts2 = _shrink_polyline_ends(pts, float(end_shrink))
        if len(pts2) < 2:
            continue
        pts2 = pts2 + np.array([sd_x, sd_y]) * float(offset)
        h2 = rc[1] * float(height_scale)
        if h2 <= 0:
            continue
        new_pts = [(float(p[0]), float(p[1])) for p in pts2]
        out.append((new_pts, h2, rc[2], closing, rc[4],
                    (float(pts2[0][0]), float(pts2[0][1])),
                    (float(pts2[-1][0]), float(pts2[-1][1])),
                    rc[7]))
    return out


def _compute_slip_speed(p1: int, p2: int, points, v_total) -> float:
    """计算两个板块之间的侧滑速度（切向分量大小）。"""
    d = np.asarray(points[p2]) - np.asarray(points[p1])
    dn = np.linalg.norm(d)
    if dn < 1e-9:
        return 0.0
    d = d / dn
    v_rel = v_total[p1] - v_total[p2]
    closing = np.dot(v_rel, d)
    sliding_vec = v_rel - closing * d
    return float(np.linalg.norm(sliding_vec))


def _apply_slip_parallel_segments(
    ridge_curves: List[Tuple],
    valid_edges: List,
    points: List[Tuple[float, float]],
    v_total: np.ndarray,
    slip_threshold: float,
    angle_slope: float,
    angle_offset: float,
    length_scale: float,
    length_offset: float,
    length_max: float,
) -> List[Tuple]:
    """
    将侧滑速度超过阈值的山脊替换为一组平行短线。

    每条短线中点位于原山脊曲线（含噪声扰动）上，长度由对数函数
    决定，与山脊线切线的夹角由线性函数决定（角度取锐角，方向固定
    为切线左侧）。短线彼此平行（相对于各自局部的切线旋转相同角度，
    由于山脊线弯曲，全局并非严格平行，但视觉上近似平行）。
    """
    edge_map = {}
    for _rv, clipped, _btype, (p1, p2) in valid_edges:
        key = (tuple(np.round(clipped[0], 6)), tuple(np.round(clipped[1], 6)))
        edge_map[key] = (p1, p2)

    new_curves = []
    for rc in ridge_curves:
        curve_points, ridge_h, edge_offset, closing, btype, raw_start, raw_end, shift_dir = rc
        key = (tuple(np.round(raw_start, 6)), tuple(np.round(raw_end, 6)))
        if key not in edge_map:
            new_curves.append(rc)
            continue
        p1, p2 = edge_map[key]
        sliding = _compute_slip_speed(p1, p2, points, v_total)
        if sliding < slip_threshold:
            new_curves.append(rc)
            continue

        pts = np.asarray(curve_points, dtype=np.float64)
        if len(pts) < 2:
            new_curves.append(rc)
            continue
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s_cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = s_cum[-1]
        if total < 2.0:
            new_curves.append(rc)
            continue

        length = min(length_offset + length_scale * math.log1p(sliding - slip_threshold),
                     length_max)
        length = max(length, 2.0)

        angle_deg = angle_offset + angle_slope * (sliding - slip_threshold)
        angle_rad = math.radians(abs(angle_deg))

        num_segments = max(2, int(round(total / length)))
        sample_s = np.linspace(0.0, total, num_segments)
        cx = np.interp(sample_s, s_cum, pts[:, 0])
        cy = np.interp(sample_s, s_cum, pts[:, 1])

        tx = np.gradient(cx)
        ty = np.gradient(cy)
        tnorm = np.hypot(tx, ty)
        tnorm[tnorm < 1e-9] = 1.0
        tx /= tnorm
        ty /= tnorm

        nx = -ty
        ny = tx

        rot_angle = math.pi / 2 - angle_rad
        cos_r = math.cos(rot_angle)
        sin_r = math.sin(rot_angle)
        dirx = nx * cos_r - ny * sin_r
        diry = nx * sin_r + ny * cos_r
        dnorm = np.hypot(dirx, diry)
        dirx /= dnorm
        diry /= dnorm

        half_len = length / 2.0
        for i in range(num_segments):
            p0x = cx[i] - dirx[i] * half_len
            p0y = cy[i] - diry[i] * half_len
            p1x = cx[i] + dirx[i] * half_len
            p1y = cy[i] + diry[i] * half_len
            new_pts = [(float(p0x), float(p0y)), (float(p1x), float(p1y))]
            new_rc = (new_pts, ridge_h, edge_offset, closing, btype,
                      (float(p0x), float(p0y)), (float(p1x), float(p1y)),
                      shift_dir)
            new_curves.append(new_rc)

    return new_curves


# ============================================================
# 顶层接口：构造阶段，写入 World 的板块属性，返回几何曲线
# ============================================================
def build_tectonic_data(
    world: World,
    # 构造参数（与旧接口完全一致）
    num_points: int = 80,
    num_macro_plates: int = 8,
    lloyd_iterations: int = 4,
    macro_lloyd_iterations: int = 2,
    ocean_expansion_rounds: int = 4,
    ocean_expansion_prob: float = 0.2,
    micro_speed_mean: float = 1.0,
    micro_speed_std: float = 0.2,
    macro_speed_mean: float = 2.0,
    macro_speed_std: float = 0.2,
    collision_threshold: float = 0.2,
    min_ridge_height: float = 40.0,
    speed_height_scale: float = 100.0,
    max_ridge_height: float = 2000.0,
    amplitude: float = 5.0,
    frequency: float = 3.0,
    octaves: int = 3,
    lacunarity: float = 2.0,
    min_edge_length: float = 10.0,
    coastal_ridge_offset: float = 5.0,
    divergence_threshold: float = 1.5,
    min_rift_depth: float = 20.0,
    rift_depth_scale: float = 25.0,
    max_rift_depth: float = 80.0,
    junction_chamfer_dist: float = 10.0,
    junction_chamfer_angle: float = 120.0,
    junction_chamfer_min_length: float = 20.0,
    secondary_ridge_threshold: Optional[float] = 1.0,
    secondary_ridge_offset: float = 15.0,
    secondary_ridge_end_shrink: float = 3.0,
    secondary_ridge_height_scale: float = 0.45,
    slip_threshold: float = float('inf'),
    slip_angle_slope: float = 20.0,
    slip_angle_offset: float = 10.0,
    slip_length_scale: float = 15.0,
    slip_length_offset: float = 5.0,
    slip_length_max: float = 80.0,
) -> Tuple[List, List, List, List]:
    """
    执行构造阶段，直接写入 world 的板块构造属性，并返回几何曲线。

    写入 world 的内容：
        micro_plates, macro_plates, plate_boundaries,
        micro_to_macro, micro_plate_is_ocean, plate_domain,
        micro_plate_velocity, macro_plate_velocity。

    返回：
        valid_edges : 所有 Voronoi 分界线（供景观阶段海岸倒角/高原等使用）
        ridge_curves : 主山脊几何曲线（含倒角、侧滑变换后）
        all_ridge_curves : 主山脊 + 次生山脊的并集
        rift_lines : 裂谷几何曲线（含倒角）
    """
    width = world.width
    height = world.height
    rng = world.rng

    # 泊松采样与 Lloyd 松弛沿用旧版流程（使用 random/np.random 全局）
    random.seed(world.seed)
    np.random.seed(world.seed)

    points = _poisson_disc_sample(width, height, num_points)
    # ========== 新增：将 points 存入 World ==========
    world.micro_plate_centers = np.array(points, dtype=np.float64)
    if lloyd_iterations > 0:
        points = _lloyd_relaxation(points, width, height, lloyd_iterations)

    vor, n = _build_voronoi(points, width, height)

    macro_id_per_micro = _cluster_micro_to_macro(
        points, num_macro_plates, world.seed, macro_lloyd_iterations
    )
    n_macro = int(macro_id_per_micro.max()) + 1

    micro_plates = _assign_micro_plates(points, width, height)
    macro_plates = _assign_macro_plates(micro_plates, macro_id_per_micro)
    boundaries = _detect_pixel_boundaries(micro_plates, macro_plates, width, height)

    # ---- 写入 World ----
    world.micro_plates[...] = micro_plates
    world.macro_plates[...] = macro_plates
    world.plate_boundaries[...] = boundaries
    world.micro_to_macro = macro_id_per_micro

    # ---- 海陆划分 ----
    is_ocean, ocean_stats = _classify_land_ocean_plates(
        micro_plates, vor, n, rng,
        ocean_expansion_rounds, ocean_expansion_prob,
    )
    world.micro_plate_is_ocean = is_ocean
    plate_domain = is_ocean[micro_plates].astype(np.int8)
    world.plate_domain[...] = plate_domain

    # ---- 速度赋值 ----
    v_micro, v_macro, v_total = _assign_plate_velocity(
        rng, n, macro_id_per_micro, n_macro,
        micro_speed_mean, micro_speed_std,
        macro_speed_mean, macro_speed_std,
    )
    world.micro_plate_velocity = v_micro
    world.macro_plate_velocity = v_macro

    # ---- 分类 Voronoi 边 ----
    valid_edges = _classify_voronoi_edges(vor, n, macro_id_per_micro, width, height)

    # ---- 碰撞检测 → 山脊曲线 ----
    perlin = PerlinNoise1D(rng)
    ridge_curves, _collision_stats = _select_collision_ridges(
        valid_edges, points, v_total,
        collision_threshold, min_ridge_height, speed_height_scale, max_ridge_height,
        perlin, amplitude, frequency, octaves, lacunarity, min_edge_length,
        is_ocean, coastal_ridge_offset,
    )

    # 侧滑平行短线变换
    if slip_threshold is not None and slip_threshold < np.inf:
        ridge_curves = _apply_slip_parallel_segments(
            ridge_curves, valid_edges, points, v_total,
            slip_threshold,
            slip_angle_slope, slip_angle_offset,
            slip_length_scale, slip_length_offset, slip_length_max,
        )

    # 山脊倒角
    ridge_curves, _ = _chamfer_ridge_junctions(
        ridge_curves, junction_chamfer_dist, junction_chamfer_angle,
        junction_chamfer_min_length,
    )

    # 次生山脊
    secondary_curves = _generate_secondary_ridges(
        ridge_curves, secondary_ridge_threshold, secondary_ridge_offset,
        secondary_ridge_end_shrink, secondary_ridge_height_scale,
    )
    all_ridge_curves = ridge_curves + secondary_curves

    # ---- 裂谷选取与倒角 ----
    rift_lines, _ = _select_divergence_rifts(
        valid_edges, points, v_total,
        divergence_threshold, min_rift_depth, rift_depth_scale, max_rift_depth,
        min_edge_length,
    )
    rift_lines, _ = _chamfer_rift_junctions(
        rift_lines, junction_chamfer_dist, junction_chamfer_angle,
        junction_chamfer_min_length,
    )

    return valid_edges, ridge_curves, all_ridge_curves, rift_lines