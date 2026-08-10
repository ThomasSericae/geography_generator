"""
plate_tectonics.py
板块构造模块

负责：
    1. 泊松圆盘采样 + Lloyd 松弛 → 均匀点集
    2. 构建 Voronoi 图
    3. 像素级小板块分配（cKDTree 最近邻）
    4. 小板块聚类为大板块（随机种子 + Lloyd 松弛迭代）
    5. 像素级大板块分配
    6. 像素级边界检测（大板块边界 > 小板块边界 > 地图边缘）
    7. 海陆板块划分：地图边缘小板块一律划为海洋板块；随后对与
       现有海洋板块相邻的大陆板块按概率转化为海洋板块
    8. 板块速度赋值：每个小板块获得方向随机、大小服从正态分布的
       个体速度；每个大板块获得均值更大的集体速度
    9. Voronoi 几何边分类（携带两侧小板块编号）
    10. 海岸线锐角倒角

输出 PlateSystem 供海拔-地形生成模块使用。
"""

import random
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial import Voronoi, cKDTree

from world_core import World


# ============================================================
# 几何工具：线段裁剪
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


# ============================================================
# 泊松圆盘采样
# ============================================================
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


# ============================================================
# Lloyd 松弛（通用点集）
# ============================================================
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


# ============================================================
# 功能函数：构建 Voronoi 图
# ============================================================
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


# ============================================================
# 功能函数：像素级小板块分配
# ============================================================
def _assign_micro_plates(
    points: List[Tuple[float, float]], width: int, height: int
) -> np.ndarray:
    tree = cKDTree(points)
    yv, xv = np.mgrid[0:height, 0:width]
    coords = np.column_stack([xv.ravel(), yv.ravel()])
    _, indices = tree.query(coords)
    return indices.reshape(height, width).astype(np.int32)


# ============================================================
# 功能函数：小板块聚类为大板块（含 Lloyd 松弛迭代）
# ============================================================
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

    # 随机初始化大板块种子中心
    center_indices = rng.choice(n, num_macro_plates, replace=False)
    centers = np.array(points, dtype=float)[center_indices].copy()
    points_arr = np.array(points, dtype=float)

    for _ in range(lloyd_iterations):
        # 分配：每个小板块归属到最近的大板块中心
        tree = cKDTree(centers)
        _, macro_ids = tree.query(points_arr)

        # 松弛：重新计算每个大板块的质心
        new_centers = np.zeros_like(centers)
        for m in range(num_macro_plates):
            mask = macro_ids == m
            if np.any(mask):
                new_centers[m] = points_arr[mask].mean(axis=0)
            else:
                # 空簇：随机重选一个小板块中心
                new_centers[m] = points_arr[rng.randint(n)]
        centers = new_centers

    # 最终分配
    tree = cKDTree(centers)
    _, macro_ids = tree.query(points_arr)
    return macro_ids.astype(np.int32)


# ============================================================
# 功能函数：像素级大板块分配
# ============================================================
def _assign_macro_plates(
    micro_plates: np.ndarray, macro_id_per_micro: np.ndarray
) -> np.ndarray:
    return macro_id_per_micro[micro_plates]


# ============================================================
# 功能函数：像素级边界检测
# ============================================================
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


# ============================================================
# 功能函数：Voronoi 几何边分类（携带两侧小板块编号）
# ============================================================
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


# ============================================================
# 功能函数：板块速度赋值（个体正态 + 大板块集体正态）
# ============================================================
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


# ============================================================
# 功能函数：海陆板块划分（边缘海洋 + 概率扩张，海中大洲）
# ============================================================
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

    # ---- 1. 地图边缘小板块一律为海洋 ----
    border_ids = np.unique(np.concatenate([
        micro_plates[0, :], micro_plates[-1, :],
        micro_plates[:, 0], micro_plates[:, -1],
    ]))
    is_ocean[border_ids] = True

    # ---- 小板块邻接表（共享 Voronoi 边即为相邻）----
    adjacency: List[set] = [set() for _ in range(n)]
    for p1, p2 in vor.ridge_points:
        if p1 < n and p2 < n:
            adjacency[p1].add(p2)
            adjacency[p2].add(p1)

    stats: Dict[str, Any] = {
        "edge_ocean_plates": int(border_ids.size),
        "expansion_rounds": [],
    }

    # ---- 2. 概率扩张 ----
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


# ============================================================
# 功能函数：尖角倒角（截断锐角交汇 + 新增倒角短线）
# ============================================================
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

    # 端点分组：端点距离 ≤ snap 即共点交汇
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
            # 射线间的扇区角（0~360°），小于阈值才倒角
            sector = math.degrees((angs[b] - angs[a]) % (2.0 * math.pi))
            if sector >= min_angle_deg:
                continue
            i, si, ci = rays[a][0], rays[a][1], rays[a][2]
            j, sj, cj = rays[b][0], rays[b][1], rays[b][2]
            if i == j:      # 同一条线的两端共点（退化环），跳过
                continue
            truncations[(i, si)] = ci
            truncations[(j, sj)] = cj
            chamfer_pairs.append((i, si, ci, j, sj, cj))

    # 应用截断：删去交点侧线段，截点成为新端点
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

    # 倒角短线（~1.5 格间距插值，与其他曲线采样密度一致）
    chamfer_lines: List[Tuple] = []
    for (i, si, ci, j, sj, cj) in chamfer_pairs:
        span = float(np.hypot(*(cj - ci)))
        npts = max(2, int(span / 1.5) + 1)
        seg_pts = np.linspace(ci, cj, npts)
        chamfer_lines.append((seg_pts, i, si, j, sj))
    return new_polylines, chamfer_lines, truncations


# ============================================================
# 功能函数：海岸线锐角倒角（切角三角形翻转海陆归属）
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
    相交——与山脊倒角同一几何：夹角小于 min_angle_deg 的顶点，两条边缘
    在距顶点 cut_dist 格处被截断，截点与顶点围成的小三角形被整体
    翻转海陆归属——陆侧尖角（海岬）被削平、洋侧尖角（尖湾）被填平，
    海岸线圆润。长度不足 min_line_length 的海岸边缘受保护，不参与倒角。

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
    for (seg_pts, i, si, _j, _sj) in chamfer_lines:
        # 交点 V = 父线 i 被截侧的原始端点；切角三角形 (V, 截点i, 截点j)
        V = polylines[i][0] if si == 0 else polylines[i][-1]
        A, B = seg_pts[0], seg_pts[-1]
        vx, vy = float(V[0]), float(V[1])
        ax, ay = float(A[0]), float(A[1])
        bx, by = float(B[0]), float(B[1])
        # 三角形内的当前归属（以质心采样）决定翻转方向：
        # 陆岬三角在陆侧 → 翻成洋；尖湾三角在洋侧 → 翻成陆
        cx = min(max(int(round((vx + ax + bx) / 3.0)), 0), w - 1)
        cy = min(max(int(round((vy + ay + by) / 3.0)), 0), h - 1)
        corner_domain = plate_domain[cy, cx]
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
    return len(chamfer_lines)


# ============================================================
# PlateSystem 命名元组
# ============================================================
class PlateSystem(NamedTuple):
    points: List[Tuple[float, float]]
    valid_edges: List
    is_ocean: np.ndarray
    boundaries: np.ndarray
    ocean_stats: Dict[str, Any]
    coast_chamfers: int


# ============================================================
# 主函数：构建板块系统
# ============================================================
def build_plate_system(
    world: World,
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
    coast_chamfer_dist: float = 10.0,
    coast_chamfer_angle: float = 120.0,
    coast_chamfer_min_length: float = 20.0,
) -> PlateSystem:
    """
    构建板块系统，将结果写入 World 容器。

    工作流：
        1. 泊松圆盘采样 → 均匀点集
        2. Lloyd 松弛 → 规则化 Voronoi 细胞
        3. 构建 Voronoi 图
        4. 像素级小板块分配
        5. 小板块聚类为大板块（含 Lloyd 松弛迭代）
        6. 像素级大板块分配与边界检测
        7. 海陆板块划分（边缘海洋 + 概率扩张 → 海中大洲）
        8. 板块速度赋值（个体正态 + 大板块集体正态叠加）
        9. Voronoi 几何边分类
        10. 海岸线锐角倒角
    """
    width = world.width
    height = world.height
    effective_seed = world.seed

    # 初始化随机状态（泊松采样/大板块聚类沿用旧版流程）
    random.seed(effective_seed)
    np.random.seed(effective_seed)

    # 1. 泊松圆盘采样
    points = _poisson_disc_sample(width, height, num_points)

    # 2. Lloyd 松弛（小板块）
    if lloyd_iterations > 0:
        points = _lloyd_relaxation(points, width, height, lloyd_iterations)

    # 3. 构建 Voronoi 图
    vor, n = _build_voronoi(points, width, height)

    # 4. 像素级小板块分配
    micro_plates = _assign_micro_plates(points, width, height)
    world.micro_plates[...] = micro_plates

    # 5. 小板块聚类为大板块（含 Lloyd 松弛）
    macro_id_per_micro = _cluster_micro_to_macro(
        points, num_macro_plates, effective_seed, macro_lloyd_iterations
    )
    n_macro = int(macro_id_per_micro.max()) + 1

    # 6. 像素级大板块分配与边界检测
    macro_plates = _assign_macro_plates(micro_plates, macro_id_per_micro)
    world.macro_plates[...] = macro_plates
    boundaries = _detect_pixel_boundaries(micro_plates, macro_plates, width, height)
    world.plate_boundaries[...] = boundaries
    world.micro_to_macro = macro_id_per_micro

    # 7. 海陆板块划分（边缘海洋 + 概率扩张 → 海中大洲）
    is_ocean, ocean_stats = _classify_land_ocean_plates(
        micro_plates, vor, n, world.rng,
        ocean_expansion_rounds, ocean_expansion_prob,
    )
    world.micro_plate_is_ocean = is_ocean
    plate_domain = is_ocean[micro_plates].astype(np.int8)
    world.plate_domain[...] = plate_domain

    # 8. 板块速度赋值（个体正态 + 大板块集体正态叠加）
    v_micro, v_macro, v_total = _assign_plate_velocity(
        world.rng, n, macro_id_per_micro, n_macro,
        micro_speed_mean, micro_speed_std,
        macro_speed_mean, macro_speed_std,
    )
    world.micro_plate_velocity = v_micro
    world.macro_plate_velocity = v_macro

    # 9. Voronoi 几何边分类
    valid_edges = _classify_voronoi_edges(vor, n, macro_id_per_micro, width, height)

    # 10. 海岸线锐角倒角：切角三角形翻转海陆归属
    n_coast_chamfer = _chamfer_coastline(
        plate_domain, valid_edges, is_ocean,
        coast_chamfer_dist, coast_chamfer_angle, coast_chamfer_min_length,
    )
    if n_coast_chamfer:
        world.plate_domain[...] = plate_domain

    return PlateSystem(points, valid_edges, is_ocean, boundaries, ocean_stats, n_coast_chamfer)