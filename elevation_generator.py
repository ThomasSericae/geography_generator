"""
elevation_generator.py
山脉地形生成模块（板块速度碰撞大修版）

大修要点（相对 DLA 梳子状山脊版）：
    · 山脊线的选取不再随机：引入板块速度场，按板块分界线两侧板块的
      相撞速度（相对速度在板块中心连线方向上的分量）是否超过阈值
      来决定是否造山，相撞速度经对数变形后决定山脊高度。
    · 海陆分布不再由背景柏林噪声随缘决定：引入海陆板块划分——
      地图边缘小板块一律划为海洋板块并整体沉降，再沿海洋边缘
      按概率向外扩张若干轮，生成海洋连续的"海中大洲"格局。

工作流：
    1.  泊松圆盘采样 + Lloyd 松弛 → 均匀点集
    2.  构建 Voronoi 图
    3.  像素级小板块分配（cKDTree 最近邻）
    4.  小板块聚类为大板块（随机种子 + Lloyd 松弛迭代）
    5.  像素级大板块分配
    6.  像素级边界检测（大板块边界 > 小板块边界 > 地图边缘）
    7.  海陆板块划分：地图边缘小板块一律划为海洋板块；随后对与
        现有海洋板块相邻的大陆板块按 ocean_expansion_prob 概率转化
        为海洋板块，重复 ocean_expansion_rounds 轮（保证海洋连续，
        形成"海中大洲"）
    8.  板块速度赋值：每个小板块获得方向随机、大小服从正态分布的
        个体速度；每个大板块获得均值更大的集体速度；小板块总速度 =
        个体速度 + 所属大板块集体速度
    9.  Voronoi 几何边碰撞检测：对每条板块分界线计算相撞速度
        closing = (V_p1 − V_p2) · d̂（d̂ 为两板块中心连线单位向量）；
        closing ≥ collision_threshold 的边界被选为山脊线并记录 closing。
        同一大板块内部的小板块共享集体速度，相撞速度主要来自个体
        速度（较小）；跨大板块边界集体速度不同，相撞速度通常更大，
        因此大板块边界自然形成更高的山系
    9c. 海岸线锐角倒角：海陆属性二值使每个海岸顶点恰有两条海岸
        边缘相交；夹角小于 coast_chamfer_angle 的顶点被倒角——两
        边缘在距顶点 coast_chamfer_dist 处截断，截点与顶点围成的
        三角形整体翻转海陆归属（陆岬削平、尖湾填平），海岸线圆润；
        只改像素级海陆归属场，板块级 is_ocean 不变
    10. 相撞速度 → 山脊高度（对数映射）：ridge_h = min_ridge_height +
        speed_height_scale × ln(1 + closing − threshold)
        （封顶 max_ridge_height）；对数压缩使相撞速度很大时高度增长
        也趋于平缓，山脊高度因此更均匀；
        洋-陆碰撞边界（海岸山脉）的山脊线沿碰撞矢量向陆地一侧
        位移 coastal_ridge_offset 格——洋板块俯冲于陆板块之下，
        造山带实际形成于陆侧、距海沟一段距离处；纯陆-陆碰撞不位移；
        选中边缘沿用原流程噪声扰动采样为曲线，栅格化为原初山脊线
        （携带各自高度与小幅高度噪声）；山脊高度沿曲线两端渐变：
        真正末端（不与其他山脊共享 Voronoi 顶点的端点）渐隐至 0
        ——没有山脊线即海拔降回板块基准，山脉渐渐消失；交汇端点
        向各山脊高度的均值平滑过渡，异高山脉之间海拔渐变；
        过短的山脊按 smoothstep(L/L₀) 映射缩减高度（避免宽度大于
        长度），渐变区重叠的短山脉两侧有山时中点分开各自渐变、
        仅一侧有山时整条自身成为渐变；共点交汇处夹角小于
        junction_chamfer_angle 的锐角做倒角——两线在距交点
        junction_chamfer_dist 处截断并新增连接截点的短线，α 尖角
        被两个 (180°+α)/2 钝角取代（90° → 两个 135°），过短的线
        不执行；裂谷线共用同一倒角规则
    11. 梳齿贴印（替代逐粒子 DLA）：梳齿纹理由大半径圆环 DLA
        经极坐标展开而来——行是距脊偏移、列是弧长，环的周期性
        使纹理列向无缝。纹理离线烘焙并硬编码于 ring_dla_stamps.py
        （由 ring_dla_baker.py 一次性生成），运行时解码仅需毫秒；
        每条山脊经"直线拟合圆弧"（弧长→圆心角，随机起始角）
        取得截取窗口后沿曲线扫掠贴印，纹理跟随山脊弯曲，随机
        选纹理/起始角/双向镜像保证多样性。烘焙文件缺失时现场
        生长兜底（独立随机流、模块级缓存，不影响可复现性）；
        梳齿身份/相撞速度取最近原初脊格，海拔自最近脊格按
        欧氏距离衰减（同附着链衰减语义）
    11d. 裂谷（板块离散）：closing ≤ −divergence_threshold 的边界
        相互离散成谷；离散速度经同形对数映射为裂谷深度（量级
        明显小于山脉抬升）。裂谷与山脉共用"细中线 + DLA 梳齿"
        范式：直线中线（Voronoi 边缘本身）栅格化后，沿中线扫掠
        自同一烘焙纹理按 rift_tooth_max_length 裁短的窄梳齿——
        裂谷比山脉窄，不规则宽度由梳齿纹理天然提供；深度沿线
        渐变：真正末端在 rift_end_taper 内渐隐至 0，交汇端点在
        rift_junction_blend 内向各裂谷深度的均值过渡（短裂谷重叠
        区的处理与山脊对称），梳齿深度自最近中线格按距离衰减，
        自动跟随渐变。裂谷与山脊存入同一组图层：ridge_id /
        ridge_speed / ridge_elevation 取负值（负身份/负速度/负深度）
        与山脊正值区分，plate_boundaries 编码 5 = 裂谷，供未来
        水文等模块使用
    12. 海拔场合成：海陆基准场（大陆板块 +continent_base / 海洋板块
        −ocean_depth，板块边界处高斯平滑过渡）+ 山脊抬升场
        （距离倒数对数衰减，最大值核）− 裂谷下陷场 + 背景噪声；
        海岸山脉不再被压入水下，
        其齿间谷地低于海平面时自然形成峡湾状溺谷海岸
    13. 海陆掩膜（elevation > sea_level），结果写回 World

写回 World 的内容：
    elevation / sea_level / land_mask / micro_plates / macro_plates /
    plate_boundaries（4 = 碰撞山脊，5 = 离散裂谷）/ plate_domain /
    ridge_id（正 = 山脊，负 = 裂谷）/ ridge_speed（正 = 相撞速度，
    负 = 离散速度）/ micro_to_macro / micro_plate_is_ocean /
    micro_plate_velocity / macro_plate_velocity，
    以及自定义图层 ridge_mask（含裂谷格）/ ridge_line_mask
    （仅碰撞山脊中线本体，不含梳齿与裂谷）/ ridge_elevation
    （负值 = 裂谷深度）/ velocity_x / velocity_y。

移除的旧参数：high_prob / low_prob / adjacent_boost_prob /
    high_height / low_height（随机选边与固定两档高度已被
    速度碰撞模型取代）。

随机数约定：板块速度、海洋扩张、柏林噪声、DLA 均取自 world.rng
（世界种子派生的统一生成器）；泊松采样与大板块聚类沿用旧版
random / np.random 全局种子流程，未改动。
"""

import random
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter

from world_core import World, PerlinNoise1D, PerlinNoise2D, grow_dla


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
# 功能函数：海陆基准场（大陆抬升 / 海洋沉降，边界平滑过渡）
# ============================================================
def _compute_domain_base_field(
    plate_domain: np.ndarray,
    continent_base: float,
    ocean_depth: float,
    transition_sigma: float,
) -> np.ndarray:
    """
    由海陆板块划分生成基准海拔场：
        大陆板块 = +continent_base，海洋板块 = −ocean_depth，
    随后以 transition_sigma 高斯平滑，使海岸带在板块边界两侧
    渐变过渡（形成陆架/陆坡），避免像素级硬台阶。
    """
    base = np.where(plate_domain == 1,
                    -float(ocean_depth), float(continent_base)).astype(np.float64)
    if transition_sigma > 0:
        base = gaussian_filter(base, sigma=float(transition_sigma), mode="nearest")
    return base


# ============================================================
# 功能函数：碰撞山脊选取（速度比较 + 相撞速度 → 高度）
# ============================================================
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

        # 对数映射：相撞速度很大时高度增长也趋于平缓，山脊高度更均匀
        ridge_h = min(
            min_ridge_height + speed_height_scale * math.log1p(closing - collision_threshold),
            max_ridge_height,
        )

        # ---- 海岸山脉：根据碰撞矢量向陆地一侧位移 ----
        # 洋-陆碰撞时洋板块俯冲于陆板块之下，造山带形成于陆侧
        # 距海沟一定距离处。Voronoi 边是两板块中心连线的中垂线，
        # 边界法向即板块中心连线方向 d̂，位移取指向陆地一侧的 ±d̂。
        # 撞击方向 sd（次生山脊平移方向）：洋-陆边界指向陆侧
        # （与海岸位移同向），纯陆-陆边界取 +d̂（p1 → p2）
        shift_x = shift_y = 0.0
        sd_x, sd_y = float(d[0]), float(d[1])
        if is_ocean is not None:
            o1, o2 = bool(is_ocean[p1]), bool(is_ocean[p2])
            if o1 != o2:  # 恰有一侧为海洋 → 洋-陆（海岸）边界
                sign = 1.0 if o1 else -1.0  # o1 时 d̂ 由洋指向陆
                sd_x, sd_y = sign * float(d[0]), sign * float(d[1])
                if coastal_ridge_offset > 0:
                    shift_x = sd_x * coastal_ridge_offset
                    shift_y = sd_y * coastal_ridge_offset
                    stats["coastal_ridges_offset"] += 1

        # ---- 沿边采样曲线（与原流程一致的噪声扰动）----
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

        # 同时记录位移前的原始边缘端点：两条山脊是否共享 Voronoi
        # 顶点（连续/交汇）只能靠位移前端点精确判定——海岸山脊的
        # 陆侧位移会使共享顶点的端点错开
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


# ============================================================
# 功能函数：离散裂谷选取（速度比较 + 离散速度 → 深度）
# ============================================================
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
        "edges_above_threshold": 0,   # 未达离散阈值（含相撞者）
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

        divergence = -closing   # 离散速率（正值）
        # 对数映射：与山脉同形，但系数与封顶都小得多（降低量比较少）
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


# ============================================================
# 功能函数：裂谷线栅格化（细中线 + 末端渐隐；梳齿贴印提供宽度）
# ============================================================
def _rasterize_rift_lines(
    rift_lines: List[Tuple],
    width: int,
    height: int,
    end_taper: float = 10.0,
    junction_blend: float = 10.0,
    junction_snap: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把选中的离散边界栅格化为裂谷中线下陷场（1 格宽细线）。

    裂谷与山脉共用同一套"细中线 + DLA 梳齿贴印"范式：本函数只
    栅格化裂谷中线（直线，不扰动），不规则的宽度与纹理由后续
    的窄梳齿贴印提供，不再使用宽度噪声与地堑剖面。

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
    # 端点共点检测：区分真正末端与交汇端（与山脊同一机制）
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
        # 深度沿线渐变（两端各自的 smoothstep 进度，标准衰减率）
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
            # 仅一端交汇：整条成为渐变，自交汇深度单调渐隐至 0
            f = t_arr * t_arr * (3.0 - 2.0 * t_arr)
            dep_s = (d_start_t * (1.0 - f) if not is_true_end[k - 1, 0]
                     else d_end_t * f)
        elif short_overlap and both_junction:
            # 两端皆交汇：中点分开各向本侧交汇深度渐变，小窗交叉淡化
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
                id_field[y, x] = -k                   # 裂谷身份为负
                speed_field[y, x] = np.float32(-divergence)  # 裂谷速度为负
    return depth_field, id_field, speed_field


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
        # 倒角短线的撞击方向取两父线合成（归一化；抵消则为零向量，
        # 下游次生山脊生成会跳过它）
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
# ============================================================
# 功能函数：次生山脊（强碰撞的平行第二皱褶）生成
# ============================================================
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


# ============================================================
# 功能函数：标记造山边缘（写入 plate_boundaries == 4）
# ============================================================
def _mark_mountain_edges(
    boundaries: np.ndarray,
    ridge_curves: List,
    width: int,
    height: int,
) -> np.ndarray:
    """
    将被选中的碰撞山脊边缘的曲线路径栅格化到
    plate_boundaries，编码为 4。不覆盖地图边缘（编码 3）。
    水文与侵蚀模块依赖该编码对山脊周围岩石加硬。
    """
    marked = boundaries.copy()
    for curve_points, *_ in ridge_curves:
        for cx, cy in curve_points:
            x, y = int(round(cx)), int(round(cy))
            if 0 <= x < width and 0 <= y < height and marked[y, x] != 3:
                marked[y, x] = 4
    return marked


# ============================================================
# 功能函数：原初山脊线栅格化（高度噪声 + 身份/相撞速度记录）
# ============================================================
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

    # ---- 短山脊高度缩减映射：太短的山峰宽度大于长度，按弧长缩减 ----
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

    # ---- 端点连续性识别：共享 Voronoi 顶点 = 连续/交汇 ----
    raw_ends = []  # 每条曲线的位移前原始端点 [(x0,y0), (x1,y1)]
    for rc in ridge_curves:
        raw_ends.append((np.asarray(rc[5], dtype=np.float64),
                         np.asarray(rc[6], dtype=np.float64)))
    target_h = np.zeros((n_curves, 2), dtype=np.float64)   # 各端点目标高度
    is_true_end = np.ones((n_curves, 2), dtype=bool)       # 是否真正结束
    for i in range(n_curves):
        for side in (0, 1):
            shared = [scaled_h[j]
                      for j in range(n_curves)
                      for s2 in (0, 1)
                      if np.hypot(*(raw_ends[i][side] - raw_ends[j][s2]))
                      <= junction_snap]
            if len(shared) >= 2:        # 除自身外还有别的山脊共点
                target_h[i, side] = float(np.mean(shared))
                is_true_end[i, side] = False
            # 否则目标高度保持 0：真正末端，渐隐至板块基准

    ridge_h_field = np.zeros((height, width), dtype=np.float64)
    ridge_id_field = np.zeros((height, width), dtype=np.int32)
    ridge_speed_field = np.zeros((height, width), dtype=np.float32)
    for rid, rc in enumerate(ridge_curves, start=1):
        curve_points, edge_offset, closing = rc[0], rc[2], rc[3]
        ridge_h = scaled_h[rid - 1]     # 缩减映射后的有效高度
        n = len(curve_points)
        if n == 0 or ridge_h <= 0:
            continue
        pts = np.asarray(curve_points, dtype=np.float64)
        # 沿曲线弧长（渐变按真实距离计算）
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s_cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = s_cum[-1] if len(s_cum) else 0.0
        # 两端渐变区长度：真正末端用 end_taper，交汇处用 junction_blend
        zone_start = end_taper if is_true_end[rid - 1, 0] else junction_blend
        zone_end = end_taper if is_true_end[rid - 1, 1] else junction_blend
        h_start, h_end = target_h[rid - 1, 0], target_h[rid - 1, 1]
        # 短小山脉（两端渐变区重叠）的过渡方式
        short_overlap = total < zone_start + zone_end
        both_junction = (not is_true_end[rid - 1, 0]) and (not is_true_end[rid - 1, 1])
        one_junction = bool(is_true_end[rid - 1, 0]) != bool(is_true_end[rid - 1, 1])
        mid = 0.5 * total
        crossfade = max(2.0, 0.1 * total)   # 中点交叉淡化窗口，避免台阶

        for i, (cx, cy) in enumerate(curve_points):
            t = i / max(n - 1, 1)
            s_i = s_cum[i]
            # 两端各自的 smoothstep 渐变进度（标准衰减率）
            ts = min(s_i / zone_start, 1.0) if zone_start > 0 else 1.0
            te = min((total - s_i) / zone_end, 1.0) if zone_end > 0 else 1.0
            fs = ts * ts * (3.0 - 2.0 * ts)
            fe = te * te * (3.0 - 2.0 * te)
            h_from_start = h_start + (ridge_h - h_start) * fs
            h_from_end = h_end + (ridge_h - h_end) * fe
            if short_overlap and one_junction:
                # 只有一侧有山脉：小山脉本身成为渐变，自交汇高度
                # 沿全长单调渐隐至 0（板块基准）
                f = (s_i / total) if total > 0 else 1.0
                f = f * f * (3.0 - 2.0 * f)
                h_base = (h_start * (1.0 - f) if not is_true_end[rid - 1, 0]
                          else h_end * f)
            elif short_overlap and both_junction:
                # 两侧都有山脉：从中点分开，各半边按标准衰减率向
                # 本侧交汇高度渐变；中点小窗交叉淡化避免台阶
                w = (s_i - (mid - 0.5 * crossfade)) / crossfade
                w = min(max(w, 0.0), 1.0)
                w = w * w * (3.0 - 2.0 * w)
                h_base = (1.0 - w) * h_from_start + w * h_from_end
            else:
                # 常规（渐变区不重叠）：身处哪端的渐变区就跟随哪端
                # 的渐变剖面（交汇高度可能高于自身高度，此时 min
                # 会把渐变削成台阶，故按区域选择而非取小）；两端
                # 皆在（孤立短山脊）取较低者成拱形；都不在则满高
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


# ============================================================
# 功能函数：环形 DLA 纹理库（烘焙文件优先；缺失时现场生成兜底）
# ============================================================
# 模块级缓存：纹理库全进程复用，后续世界生成立即命中
_RING_DLA_CACHE: Dict[Tuple, List[np.ndarray]] = {}
_BAKED_STAMP_CACHE: Dict[int, List[np.ndarray]] = {}


def _load_baked_stamps(tooth_length: int) -> Optional[List[np.ndarray]]:
    """
    从 ring_dla_stamps.py（ring_dla_baker.py 离线烘焙的硬编码数据
    文件）加载梳子纹理，毫秒级完成——运行时不再进行任何 DLA
    随机游走。烘焙齿长有余量时按请求的 tooth_length 裁剪行；
    文件缺失或烘焙齿长不足时返回 None（调用方回退现场生成）。
    """
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
    """
    在圆环种子上生长 DLA（粒子自环内外两侧投放）。

    圆环上的枝杈处处沿径向（垂直于环），正好对应"梳齿垂直于
    山脊线"；两侧投放使环的内外两侧都长出枝杈，展开后齿朝两侧。
    返回 (聚集体画布, 环心 y, 环心 x)。
    """
    pad = tooth_length + 6
    half = int(radius + pad)
    size = 2 * half + 1
    cy = cx = float(half)
    yv, xv = np.mgrid[0:size, 0:size]
    dist = np.hypot(yv - cy, xv - cx)
    ring = np.abs(dist - radius) <= 0.8                      # 环形种子带
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
    """
    把环形 DLA 沿极坐标展开为 (2L+1, 周长) 的矩形梳子纹理。

    行 = 相对环半径的偏移 −L..+L（齿朝两侧），列 = 角度 × 半径
    （即弧长）。环是周期的，纹理在列方向天然无缝，可按任意
    起始角/弧长截取而不出现接缝。环半径远大于齿长时，极坐标
    展开的几何畸变（齿尖/齿根宽度差 (R±L)/R）可以忽略。
    """
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


# 烘焙文件缺失时现场生长兜底的固定配置（与 ring_dla_baker 默认一致；
# 正常使用永远命中烘焙文件，这些常量不参与主流程）
_FALLBACK_STAMP_SEED = 7106     # 独立随机流种子（不占世界随机流）
_FALLBACK_STAMP_COUNT = 2
_FALLBACK_RING_RADIUS = 512
_FALLBACK_RING_PARTICLES = 6000


def _get_ring_dla_stamps(tooth_length: int) -> List[np.ndarray]:
    """
    获取环形 DLA 梳子纹理库：优先解码烘焙文件（毫秒级，零 DLA
    成本）；文件缺失时按固定的兜底配置现场生长（一次性，模块级
    缓存；独立随机流，不占用世界随机流，缓存命中与否都不影响
    "同世界种子 → 同世界"的可复现性）。
    """
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
# 功能函数：把一段直线拟合到大圆环的一段圆弧上
# ============================================================
def _fit_segment_to_arc(
    length: float,
    ring_radius: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    把一段（近似直的）山脊线拟合到半径 ring_radius 的大圆环的
    一段圆弧上，返回纹理截取窗口 (起始列, 列数)。

    拟合关系：弧长 = 山脊长度 ⇒ 圆心角跨度 θ = length / R；
    起始角在 [0, 2π) 内均匀随机（不同山脊截取环的不同弧段）。
    由于纹理按"列 = 弧长"组织，窗口列数 = 弧长取整、起始列 =
    起始角 × R。大半径下圆弧与其弦几乎重合（矢高 ≈ length²/8R，
    R=512、length=300 时约 22 格，且本模块并不刚性搬运圆弧
    几何，只按弧长索引纹理后沿山脊扫掠，几何误差为零），
    因此"以弧代直"完全成立。
    """
    circumference = int(math.ceil(2.0 * math.pi * ring_radius))
    n_cols = max(1, int(round(length)))
    start_col = int(rng.integers(circumference))
    return start_col, n_cols


# ============================================================
# 功能函数：沿山脊扫掠贴印梳齿（替代逐粒子 DLA）
# ============================================================
def _stamp_teeth_from_ring_dla(
    ridge_curves: List[Tuple[List[Tuple[float, float]], float, float, float, int]],
    primary_mask: np.ndarray,
    stamps: List[np.ndarray],
    rng: np.random.Generator,
    tooth_max_length: float,
) -> np.ndarray:
    """
    把环形 DLA 梳子纹理沿每条原初山脊线扫掠贴印，生成梳齿候选掩膜。

    每条山脊曲线按弧长重采样（步长 1 格，与纹理列同尺度），逐样本
    点取切线/法线；纹理第 n 行（距脊 n 格）的图案被放到样本点法向
    偏移 n 格处——纹理跟随山脊弯曲，弯山脊的齿自然转弯，这等价于
    "按山脊角度与长度在环上截取"，且对任意弯曲山脊都成立。
    每条山脊随机选一张纹理、随机起始列（截取角）与随机镜像以保证
    多样性；纹理列向无缝，山脊比环周长更长时回绕也不留接缝。

    返回候选掩膜（不含原初山脊格）；身份/海拔由调用方按最近脊格解析。
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
        nxv, nyv = -ty / tn, tx / tn                       # 单位法线
        stamp = stamps[int(rng.integers(len(stamps)))]
        if rng.integers(2) == 1:
            stamp = stamp[::-1]                            # 跨脊镜像
        if rng.integers(2) == 1:
            stamp = stamp[:, ::-1]                         # 沿脊镜像
        circumference = stamp.shape[1]
        # 把这段山脊拟合到圆环的一段圆弧上，取得纹理截取窗口
        start_col, _n_cols = _fit_segment_to_arc(
            total, circumference / (2.0 * math.pi), rng)
        cols = (s_new.astype(np.int64) + start_col) % circumference
        pattern = stamp[row_offsets + L][:, cols]          # (2L+1, n_samples)
        wx = px[None, :] + row_offsets[:, None] * nxv[None, :]
        wy = py[None, :] + row_offsets[:, None] * nyv[None, :]
        ix = np.round(wx).astype(np.int64)
        iy = np.round(wy).astype(np.int64)
        valid = pattern & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        candidate[iy[valid], ix[valid]] = True
    candidate &= ~primary_mask
    return candidate


# ============================================================
# 功能函数：山脊属性赋值（原初 + 贴印梳齿衰减/继承）
# ============================================================
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
    与旧附着链"每格衰减 tooth_decay"语义一致；身份与相撞速度取最近
    原初脊格——不同山脊的梳子在中轴处自然分界，不会互相错认。
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


# ============================================================
# 功能函数：山脊抬升场（距离倒数对数衰减，最大值核）
# ============================================================
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
        if radius <= 1.0:               # ln(R) 需为正
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
        # 距离倒数的对数衰减：ln(R/d)/ln(R)，d≤1 处为全额 H_r
        contrib = h_val * np.clip(
            np.log(radius / np.maximum(dist, 1.0)) / math.log(radius), 0.0, 1.0)
        patch = uplift[y_min:y_max, x_min:x_max]
        np.maximum(patch, contrib, out=patch)
    return uplift


# ============================================================
# 功能函数：背景柏林噪声场（向量化）
# ============================================================
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


# ============================================================
# 主要进程函数：山脉地形生成（板块速度碰撞版）
# ============================================================
def generate_mountain_terrain(
    seed: Optional[int] = None,          # 世界种子；world 为 None 时必填
    num_points: int = 80,                # 小板块(Voronoi)点数；↑板块更碎 [30~200]
    num_macro_plates: int = 8,           # 大板块数；↑山系更多 [4~16]
    lloyd_iterations: int = 4,           # 小板块Lloyd松弛；0=不规则，↑细胞更均匀 [0~10]
    macro_lloyd_iterations: int = 2,     # 大板块聚类松弛；↑大板块边界更平滑 [0~6]
    # ── 海陆板块划分（海中大洲）──
    ocean_expansion_rounds: int = 4,     # 海洋向内陆扩张轮数；0=仅边缘海洋 [0~5]
    ocean_expansion_prob: float = 0.2,   # 每轮邻洋大陆板块转为海洋的概率 [0~1]
    continent_base: float = 5.0,        # 大陆板块基准海拔(米)
    ocean_depth: float = 50.0,          # 海洋板块基准沉降深度(米)；应大于山脊抬升量级
    domain_transition_sigma: float = 2.0,# 海陆基准场高斯平滑σ(格)；↑海岸过渡更宽 [0~15]
    # ── 板块速度（方向随机，大小正态）──
    micro_speed_mean: float = 1.0,       # 小板块个体速度大小均值(速度单位)
    micro_speed_std: float = 0.2,        # 小板块个体速度大小标准差
    macro_speed_mean: float = 2.0,       # 大板块集体速度大小均值(应大于小板块)
    macro_speed_std: float = 0.2,        # 大板块集体速度大小标准差
    # ── 碰撞 → 山脊 ──
    collision_threshold: float = 0.2,    # 相撞速度阈值；↓山脉更多 [0.5~3]
    min_ridge_height: float = 40.0,      # 刚达阈值的山脊基准高度(米)
    speed_height_scale: float = 100.0,   # 对数高度系数(米)：ridge_h = 基准 + 系数×ln(1+超出速度)
    max_ridge_height: float = 2000.0,     # 山脊高度封顶(米)
    ridge_end_taper: float = 10.0,       # 山脊真正末端的高度渐隐长度(格)；↑收尾更缓 [0~80]
    ridge_junction_blend: float = 10.0,  # 交汇山脊的高度过渡长度(格) [0~60]
    short_ridge_shrink_length: float = 60.0,  # 短山脊高度缩减参考长度(格)；0=不缩减 [0~120]
    junction_chamfer_dist: float = 10.0,   # 尖角倒角截距(格)：截点距交点距离；0=不倒角 [0~40]
    junction_chamfer_angle: float = 120.0, # 触发倒角的夹角阈值(度)；小于它才倒角 [60~179]
    junction_chamfer_min_length: float = 20.0,  # 参与倒角的最短线长(格)；更短的脊/谷线不执行 [0~80]
    coast_chamfer_dist: float = 10.0,      # 海岸线锐角倒角截距(格)；0=不倒角 [0~40]
    coast_chamfer_angle: float = 120.0,    # 海岸线触发倒角的夹角阈值(度) [60~179]
    coast_chamfer_min_length: float = 20.0,# 参与倒角的最短海岸边缘长(格) [0~80]
    # ── 裂谷（板块离散，细中线 + 窄梳齿贴印，与山脉同范式）──
    divergence_threshold: float = 1.5,   # 离散速度阈值；↓裂谷更多 [0.5~3]
    min_rift_depth: float = 20.0,        # 刚达阈值的裂谷基准深度(米)
    rift_depth_scale: float = 25.0,      # 对数深度系数(米)：depth = 基准 + 系数×ln(1+超出速度)
    max_rift_depth: float = 80.0,        # 裂谷深度封顶(米)；应明显小于山脉抬升
    rift_end_taper: float = 10.0,        # 裂谷真正末端的深度渐隐长度(格) [0~80]
    rift_junction_blend: float = 10.0,   # 交汇裂谷的深度过渡长度(格) [0~60]
    # ── 山脊曲线与栅格化 ──
    amplitude: float = 5.0,             # 山脊曲线摆动幅度(格)；↑边缘更弯 [0~40]
    frequency: float = 3.0,              # 曲线/高度一维噪声频率；↑变化更密 [1~10]
    octaves: int = 3,                    # 一维噪声分频数 [1~6]
    lacunarity: float = 2.0,             # 一维噪声间隙度 [1.5~3]
    ridge_height_noise_amp: float = 0.25,# 原初山脊高度噪声相对振幅；0=一样高 [0~0.5]
    min_edge_length: float = 10.0,       # 造山边缘最小长度(格)
    # ── 山脊形态（梳齿：环形DLA纹理贴印）──
    ridge_influence: float = 12.0,       # 原初山脊对数衰减抬升半径(格)；↑山更宽 [4~30]
    tooth_influence: float = 7.0,        # 梳齿抬升半径(格)；宜小于原初 [2~15]
    tooth_decay: float = 6.0,            # 梳齿海拔每格衰减(米)；↑齿更矮 [1~20]
    tooth_max_length: float = 20.0,      # 山脊梳齿最大长度(格) [5~60]
    rift_tooth_max_length: float = 10.0, # 裂谷梳齿最大长度(格)；宜小于山脊梳齿 [3~40]
    # ── 次生山脊（强碰撞的平行第二皱褶/前陆褶皱带）──
    secondary_ridge_threshold: Optional[float] = 1.0,  # 次生山脊相撞速度阈值：超过才生成，应明显高于碰撞阈值；None=关闭
    secondary_ridge_offset: float = 15.0,       # 次生山脊沿撞击方向的平移距离(格)；小于梳齿长度则两脊梳齿少量重叠 [8~30]
    secondary_ridge_end_shrink: float = 12.0,   # 次生山脊两端各缩短的弧长(格) [0~40]
    secondary_ridge_height_scale: float = 0.45, # 次生山脊高度倍率（低矮的第二皱褶）[0.2~0.7]
    # 梳齿纹理解码自 ring_dla_stamps.py（ring_dla_baker.py 离线烘焙
    # 的硬编码数据，毫秒级，零DLA成本）；烘焙文件缺失时按固定的
    # 兜底配置现场生长（一次性，模块级缓存，不占世界随机流）
    # ── 背景起伏 ──
    bg_amp: float = 50.0,                # 背景起伏振幅(米)；0=纯板块+山脊地形 [0~100]
    bg_freq: float = 4.0,                # 背景噪声频率 [1~10]
    bg_octaves: int = 4,                 # 背景噪声分频数 [1~6]
    bg_lacunarity: float = 2.0,          # 背景噪声间隙度 [1.5~3]
    # ── 海陆 ──
    sea_level: float = 0.0,              # 海平面(米)
    coastal_ridge_offset: float = 5.0,   # 洋-陆边界山脊线向陆侧位移(格)；0=不位移 [0~15]
    # ── 画布与容器 ──
    width: int = 512,                    # 地图宽(格)
    height: int = 512,                   # 地图高(格)
    world: Optional[World] = None,       # 复用已有 World；None 则新建
) -> Tuple[World, Dict[str, Any]]:
    """
    生成山脉地形（板块速度碰撞版），将结果写入 World 容器。

    工作流：
        1. 泊松圆盘采样 → 均匀点集
        2. Lloyd 松弛 → 规则化 Voronoi 细胞
        3. 构建 Voronoi 图
        4. 像素级小板块分配
        5. 小板块聚类为大板块（含 Lloyd 松弛迭代）
        6. 像素级大板块分配与边界检测
        7. 海陆板块划分（边缘海洋 + 概率扩张 → 海中大洲）
        8. 板块速度赋值（个体正态 + 大板块集体正态叠加）
        9. 分界线相撞速度检测，超阈值者选为山脊线并记录相撞速度
        10. 相撞速度 → 山脊高度，曲线采样并栅格化原初山脊线；
            相撞速度超过次生山脊阈值者，沿撞击方向平移、两端
            缩短并压低，额外生成不位于板块分界线上的次生山脊
            （平行第二皱褶），与主山脊共用梳齿贴印通道（允许
            少量重叠）
        11. 梳齿贴印：沿山脊扫掠预生成的环形 DLA 梳子纹理
        11d. 裂谷：离散边界栅格化细中线并贴印窄梳齿（与山脉同范式）
        12. 海拔场合成（海陆基准场 + 山脊抬升场 − 裂谷下陷场 + 背景噪声）；
            海岸山脉完整出露，齿间谷地低于海平面即形成峡湾
        13. 海陆掩膜（elevation > sea_level），写入 World

    Parameters
    ----------
    seed : int, optional
        随机种子。
    num_points : int
        小板块（Voronoi 种子点）数量。
    num_macro_plates : int
        大板块数量。
    ocean_expansion_rounds / ocean_expansion_prob : int / float
        海洋扩张轮数与每轮邻洋大陆板块转化为海洋的概率。
        两者共同决定海洋占比与连续性；0 轮 = 仅地图边缘为海。
    continent_base / ocean_depth : float
        大陆板块基准海拔 / 海洋板块基准沉降深度（米）。
    domain_transition_sigma : float
        海陆基准场在板块边界处的高斯平滑 σ（格），形成陆架过渡。
    micro_speed_mean / micro_speed_std / macro_speed_mean / macro_speed_std :
        板块速度大小的正态分布参数（速度单位）；方向均匀随机。
        大板块集体速度应大于小板块个体速度，使大板块边界相撞更剧烈。
    collision_threshold : float
        相撞速度阈值：closing = (V_p1 − V_p2)·d̂ ≥ 该值才造山。
    min_ridge_height / speed_height_scale / max_ridge_height : float
        相撞速度 → 山脊高度的对数变形：基准高度 + 系数 × ln(1 + 超出速度)，封顶。
    ridge_end_taper / ridge_junction_blend : float
        山脊高度沿曲线的渐变（格）：真正末端（位移前原始端点不与任何
        其他山脊共享 Voronoi 顶点）在 ridge_end_taper 内渐隐至 0——
        没有山脊线即海拔降回板块基准，山脉渐渐消失而非戛然而止；
        交汇端点在 ridge_junction_blend 内向各山脊高度的均值平滑
        过渡，不同高度的山脉之间海拔是渐变的。梳齿海拔由最近脊格
        衰减得到，自动跟随渐变。渐变区重叠的短小山脉：两侧都有
        山脉时从中点分开、各半边按标准衰减率向本侧交汇高度渐变
        （中点小窗交叉淡化）；只有一侧有山脉时，小山脉本身成为
        渐变，自交汇高度沿全长单调渐隐至 0。
    short_ridge_shrink_length : float
        短山脊高度缩减映射的参考长度（格）：梳齿宽度基本恒定，
        太短的山峰会"宽度大于长度"；弧长短于该值的山脊按
        smoothstep(L / L₀) 缩减有效高度，越短越矮。0 = 不缩减。
    junction_chamfer_dist / junction_chamfer_angle / junction_chamfer_min_length : float
        尖角倒角（山脊与裂谷共用）：自然界山脉与裂谷鲜有锐角
        转折。共点交汇处凡夹角小于 junction_chamfer_angle（度）
        且两侧线长均不短于 junction_chamfer_min_length（格）的，
        两线在距交点 junction_chamfer_dist（格）处截断，新增一条
        连接两截点的倒角短线——α 尖角被两个 (180°+α)/2 钝角
        取代（如 90° → 两个 135°），线条更圆润。倒角山脊线的
        高度/相撞速度取两父线均值（裂谷线为深度/离散速度均值），
        截点坐标在父线新端点与倒角线端点间完全一致，下游的端点
        共点识别视为交汇，高度由既有交汇渐变机制平滑过渡。
        junction_chamfer_dist = 0 时整体关闭。单次执行，不递归。
    coast_chamfer_dist / coast_chamfer_angle / coast_chamfer_min_length : float
        海岸线锐角倒角：海陆属性二值使每个海岸顶点恰有两条海岸
        边缘相交。夹角小于 coast_chamfer_angle 且两边缘均不短于
        coast_chamfer_min_length 的顶点被倒角——两边缘在距顶点
        coast_chamfer_dist 处截断，截点与顶点围成的三角形整体
        翻转海陆归属：陆侧锐角（海岬）削平、洋侧锐角（尖湾）
        填平，海岸线圆润。只改像素级海陆归属场（板块级 is_ocean
        不变），海陆基准场的高斯平滑自动跟随新海岸线。
        coast_chamfer_dist = 0 时整体关闭。
    divergence_threshold / min_rift_depth / rift_depth_scale / max_rift_depth : float
        裂谷（板块离散）：closing ≤ −divergence_threshold 的边界成谷；
        深度对数映射 = min_rift_depth + rift_depth_scale × ln(1 + 超出
        速度)，封顶 max_rift_depth，量级明显小于山脉抬升。裂谷与山脉
        共用"细中线 + DLA 梳齿贴印"范式：直线中线栅格化后，沿中线
        扫掠更窄的梳齿纹理（rift_tooth_max_length），不规则的裂谷
        宽度由梳齿天然提供。裂谷与山脊存入同一组图层：ridge_id /
        ridge_speed / ridge_elevation 取负值与山脊区分，
        plate_boundaries 编码 5。
    rift_end_taper / rift_junction_blend : float
        裂谷深度沿线的渐变（格），与山脊末端渐隐对称：真正末端
        在 rift_end_taper 内渐隐至 0——裂谷渐渐变浅消失；交汇端点
        在 rift_junction_blend 内向各裂谷深度的均值平滑过渡；渐变区
        重叠的短裂谷：两端皆交汇时中点分开各自渐变、仅一端交汇时
        整条自身成为渐变、两端皆真末端时成拱形。梳齿深度自最近
        中线格按距离衰减，自动跟随渐变。
    ridge_influence / tooth_influence / tooth_decay / tooth_max_length : float
        原初山脊抬升半径 / 梳齿抬升半径 / 梳齿每格海拔(深度)衰减 /
        山脊梳齿最大长度。裂谷梳齿复用同一衰减率 tooth_decay。
    rift_tooth_max_length : float
        裂谷梳齿最大长度（格）：裂谷纹理自同一份烘焙环形 DLA 按
        更短的齿长裁剪，裂谷因此比山脉窄；宜明显小于
        tooth_max_length。梳齿纹理由大半径圆环 DLA 极坐标展开而来
        （列向无缝），运行时解码 ring_dla_stamps.py（ring_dla_baker.py
        离线烘焙的硬编码数据，毫秒级，零 DLA 成本）；烘焙文件缺失
        时按固定兜底配置现场生长（独立随机流、模块级缓存，不影响
        世界可复现性）。梳齿贴印替代逐粒子随机游走，提速近一个
        数量级。
    coastal_ridge_offset : float
        洋-陆碰撞边界（海岸山脉）的山脊线沿碰撞矢量向陆地一侧
        位移的格数：洋板块俯冲于陆板块之下，造山带实际形成于
        陆侧、距海沟一段距离处；纯陆-陆碰撞的山脊线不位移。
        海岸山脉不再被压入水下，其齿间谷地低于海平面时自然
        形成峡湾状溺谷海岸。
    secondary_ridge_threshold / secondary_ridge_offset /
    secondary_ridge_end_shrink / secondary_ridge_height_scale :
        次生山脊（平行第二皱褶/前陆褶皱带）参数：相撞速度阈值
        （None=关闭）、沿撞击方向平移距离（格，小于梳齿长度则
        两脊梳齿少量重叠）、两端各缩短弧长（格）、高度倍率。
        次生山脊不位于板块分界线上，与主山脊共用梳齿贴印通道。
    sea_level : float
        海平面海拔。elevation > sea_level 为陆地，反之为水域。

    Returns
    -------
    (World, dict)
        同一个世界实例与统计报告。
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

    # ---------- 初始化随机状态（泊松采样/大板块聚类沿用旧版流程）----------
    random.seed(effective_seed)
    np.random.seed(effective_seed)

    # ---------- 1. 泊松圆盘采样 ----------
    points = _poisson_disc_sample(width, height, num_points)

    # ---------- 2. Lloyd 松弛（小板块）----------
    if lloyd_iterations > 0:
        points = _lloyd_relaxation(points, width, height, lloyd_iterations)

    # ---------- 3. 构建 Voronoi 图 ----------
    vor, n = _build_voronoi(points, width, height)

    # ---------- 4. 像素级小板块分配 ----------
    micro_plates = _assign_micro_plates(points, width, height)
    world.micro_plates[...] = micro_plates

    # ---------- 5. 小板块聚类为大板块（含 Lloyd 松弛）----------
    macro_id_per_micro = _cluster_micro_to_macro(
        points, num_macro_plates, effective_seed, macro_lloyd_iterations
    )
    n_macro = int(macro_id_per_micro.max()) + 1

    # ---------- 6. 像素级大板块分配与边界检测 ----------
    macro_plates = _assign_macro_plates(micro_plates, macro_id_per_micro)
    world.macro_plates[...] = macro_plates
    boundaries = _detect_pixel_boundaries(micro_plates, macro_plates, width, height)
    world.plate_boundaries[...] = boundaries
    world.micro_to_macro = macro_id_per_micro

    # ---------- 7. 海陆板块划分（边缘海洋 + 概率扩张 → 海中大洲）----------
    is_ocean, ocean_stats = _classify_land_ocean_plates(
        micro_plates, vor, n, world.rng,
        ocean_expansion_rounds, ocean_expansion_prob,
    )
    world.micro_plate_is_ocean = is_ocean
    plate_domain = is_ocean[micro_plates].astype(np.int8)
    world.plate_domain[...] = plate_domain

    # ---------- 8. 板块速度赋值（个体正态 + 大板块集体正态叠加）----------
    v_micro, v_macro, v_total = _assign_plate_velocity(
        world.rng, n, macro_id_per_micro, n_macro,
        micro_speed_mean, micro_speed_std,
        macro_speed_mean, macro_speed_std,
    )
    world.micro_plate_velocity = v_micro
    world.macro_plate_velocity = v_macro

    # ---------- 9~10. 碰撞检测 → 山脊选取与曲线采样 ----------
    perlin = PerlinNoise1D(world.rng)
    valid_edges = _classify_voronoi_edges(vor, n, macro_id_per_micro, width, height)

    # ---------- 9c. 海岸线锐角倒角：切角三角形翻转海陆归属 ----------
    # 与山脊/裂谷同一倒角几何：海岸顶点处夹角过锐时，两条海岸边缘
    # 在距顶点 coast_chamfer_dist 处被截，截点与顶点围成的三角形
    # 整体翻转海陆归属——陆岬削平、尖湾填平；板块级 is_ocean 不变
    n_coast_chamfer = _chamfer_coastline(
        plate_domain, valid_edges, is_ocean,
        coast_chamfer_dist, coast_chamfer_angle, coast_chamfer_min_length,
    )
    if n_coast_chamfer:
        world.plate_domain[...] = plate_domain

    ridge_curves, collision_stats = _select_collision_ridges(
        valid_edges, points, v_total,
        collision_threshold, min_ridge_height, speed_height_scale, max_ridge_height,
        perlin, amplitude, frequency, octaves, lacunarity, min_edge_length,
        is_ocean, coastal_ridge_offset,
    )

    # ---------- 10a. 尖角倒角：截断锐角交汇处，新增倒角短线 ----------
    # 夹角 < junction_chamfer_angle 的交汇过于锐利：两线在距交点
    # junction_chamfer_dist 处截断，新增连接两截点的短线，α 尖角
    # 被两个 (180°+α)/2 钝角取代（90° → 两个 135°）；过短的线不执行
    ridge_curves, n_ridge_chamfer = _chamfer_ridge_junctions(
        ridge_curves, junction_chamfer_dist, junction_chamfer_angle,
        junction_chamfer_min_length,
    )
    collision_stats["corner_chamfers"] = n_ridge_chamfer

    # ---------- 10a-bis. 次生山脊：强碰撞山脊的平行第二皱褶 ----------
    # 相撞速度超过 secondary_ridge_threshold 的山脊，沿撞击方向平移
    # secondary_ridge_offset 格、两端各缩短 secondary_ridge_end_shrink
    # 格、高度压低为 secondary_ridge_height_scale 倍，生成不位于板块
    # 分界线上的低矮次生山脊（前陆褶皱带）。与主山脊共用栅格化与
    # 梳齿贴印通道——DLA 梳齿纹样并集贴印，允许少量重叠（重叠区
    # 海拔取最近脊格，两脊之间自然成谷）
    secondary_curves = _generate_secondary_ridges(
        ridge_curves, secondary_ridge_threshold, secondary_ridge_offset,
        secondary_ridge_end_shrink, secondary_ridge_height_scale,
    )
    collision_stats["secondary_ridges"] = len(secondary_curves)
    all_ridge_curves = ridge_curves + secondary_curves

    # ---------- 10b. 标记造山边缘（plate_boundaries == 4）----------
    world.plate_boundaries[...] = _mark_mountain_edges(
        boundaries, ridge_curves, width, height
    )

    # ---------- 10c. 原初山脊线栅格化（高度 + 身份 + 相撞速度）----------
    # 高度沿山脊渐变：真正末端在 ridge_end_taper 内渐隐至 0（= 板块
    # 基准），交汇处在 ridge_junction_blend 内向交汇高度平滑过渡
    primary_h, primary_id, primary_speed = _rasterize_primary_ridges(
        all_ridge_curves, perlin, width, height,
        ridge_height_noise_amp, frequency, octaves, lacunarity,
        end_taper=ridge_end_taper, junction_blend=ridge_junction_blend,
        shrink_length=short_ridge_shrink_length,
    )
    primary_mask = primary_h > 0

    # ---------- 11. 梳齿贴印：沿山脊扫掠环形 DLA 纹理（替代逐粒子 DLA）----------
    # 纹理库一次性预生成并模块级缓存（独立随机流，不占世界随机流）；
    # 贴印是纯数组查找，相对逐粒子随机游走提速一个数量级
    stamps = _get_ring_dla_stamps(int(tooth_max_length))
    tooth_mask = _stamp_teeth_from_ring_dla(
        all_ridge_curves, primary_mask, stamps, world.rng, tooth_max_length,
    )
    # 距离兜底：弯曲处扫掠映射可能把齿放到欧氏距离越界处，裁掉
    tooth_mask &= distance_transform_edt(~primary_mask) <= tooth_max_length
    ridge_mask = primary_mask | tooth_mask

    # 梳齿海拔自最近原初脊格按欧氏距离衰减；身份与相撞速度取最近脊格
    ridge_elev, ridge_id, ridge_speed = _assign_ridge_attributes(
        primary_h, primary_id, primary_speed, tooth_mask, tooth_decay
    )

    # ---------- 11d. 裂谷：板块离散边界 → 裂谷中线 + 窄梳齿贴印 ----------
    # 与山脉逻辑对称：closing ≤ −divergence_threshold 的边界离散成谷，
    # 深度经对数映射（量级明显小于山脉抬升）；与山脉共用"细中线 +
    # DLA 梳齿"范式，但梳齿纹理按 rift_tooth_max_length 裁得更窄，
    # 不规则的裂谷宽度由梳齿纹理天然提供
    rift_lines, rift_stats = _select_divergence_rifts(
        valid_edges, points, v_total,
        divergence_threshold, min_rift_depth, rift_depth_scale, max_rift_depth,
        min_edge_length,
    )
    # 裂谷与山脊共用同一倒角规则，锐角交汇同样截断并新增倒角短线
    rift_lines, n_rift_chamfer = _chamfer_rift_junctions(
        rift_lines, junction_chamfer_dist, junction_chamfer_angle,
        junction_chamfer_min_length,
    )
    rift_stats["corner_chamfers"] = n_rift_chamfer
    # 裂谷中线栅格化（深度 + 身份 + 离散速度；末端渐隐/交汇过渡）
    rift_primary_d, rift_primary_id, rift_primary_speed = _rasterize_rift_lines(
        rift_lines, width, height,
        end_taper=rift_end_taper, junction_blend=rift_junction_blend,
    )
    rift_primary_mask = rift_primary_d > 0
    # 沿裂谷中线扫掠窄梳齿（比山脊梳齿短的纹理），宽度/纹理同山脉
    rift_stamps = _get_ring_dla_stamps(int(rift_tooth_max_length))
    rift_stamp_curves = [([rl[0], rl[1]], rl[2]) for rl in rift_lines]
    rift_tooth_mask = _stamp_teeth_from_ring_dla(
        rift_stamp_curves, rift_primary_mask, rift_stamps, world.rng,
        rift_tooth_max_length,
    )
    rift_tooth_mask &= (
        distance_transform_edt(~rift_primary_mask) <= rift_tooth_max_length)
    # 梳齿深度自最近中线格按欧氏距离衰减（与山脊同一衰减率）；
    # 身份与离散速度取最近中线格
    rift_depth_field, rift_id_field, rift_speed_field = _assign_ridge_attributes(
        rift_primary_d, rift_primary_id, rift_primary_speed,
        rift_tooth_mask, tooth_decay,
    )
    # 裂谷写入保存山脊线的各图层（负身份/负速度/负海拔，与山脊正值
    # 区分；不覆盖已有山脊格），供未来水文等模块使用
    rift_free = (rift_id_field < 0) & (ridge_id == 0)
    ridge_id[rift_free] = rift_id_field[rift_free]
    ridge_speed[rift_free] = rift_speed_field[rift_free]
    ridge_elev[rift_free] = -rift_depth_field[rift_free]
    ridge_mask = ridge_mask | rift_free
    # plate_boundaries：5 = 裂谷（不覆盖地图边缘 3 与造山 4）
    world.plate_boundaries[
        rift_free & (world.plate_boundaries != 3) & (world.plate_boundaries != 4)] = 5

    # DLA 山脊同样标记为造山区域（不覆盖地图边缘 3）
    world.plate_boundaries[tooth_mask & (world.plate_boundaries != 3)] = 4
    world.ridge_id[...] = ridge_id
    world.ridge_speed[...] = ridge_speed

    # ---------- 12. 海拔场合成（海陆基准场 + 山脊抬升 − 裂谷下陷 + 背景噪声）----------
    uplift_field = _compute_ridge_uplift_field(
        ridge_elev, primary_mask, ridge_influence, tooth_influence,
    )
    domain_base = _compute_domain_base_field(
        plate_domain, continent_base, ocean_depth, domain_transition_sigma,
    )
    bg_field = _compute_background_field(
        width, height, bg_amp, bg_freq, bg_octaves, bg_lacunarity, world.rng,
    )
    elevation = domain_base + uplift_field + bg_field - rift_depth_field

    # ---------- 13. 海陆掩膜、图层写回与报告 ----------
    world.elevation[...] = elevation.astype(np.float32)
    world.sea_level = sea_level
    world.land_mask[...] = elevation > sea_level

    if world.get_layer("ridge_mask") is None:
        world.add_layer("ridge_mask", np.bool_, False)
    world.get_layer("ridge_mask")[...] = ridge_mask
    # 碰撞山脊中线本体（不含梳齿、不含裂谷），供水文模块按距离
    # 限制划定山脉范围使用
    if world.get_layer("ridge_line_mask") is None:
        world.add_layer("ridge_line_mask", np.bool_, False)
    world.get_layer("ridge_line_mask")[...] = primary_mask
    if world.get_layer("ridge_elevation") is None:
        world.add_layer("ridge_elevation", np.float32, 0.0)
    world.get_layer("ridge_elevation")[...] = ridge_elev.astype(np.float32)

    # 像素级速度场（小板块总速度按像素展开，供可视化）
    v_field = v_total[micro_plates]  # (h, w, 2)
    for name, arr in (("velocity_x", v_field[..., 0]), ("velocity_y", v_field[..., 1])):
        if world.get_layer(name) is None:
            world.add_layer(name, np.float32, 0.0)
        world.get_layer(name)[...] = arr.astype(np.float32)

    v_micro_speed = np.hypot(v_micro[:, 0], v_micro[:, 1])
    v_macro_speed = np.hypot(v_macro[:, 0], v_macro[:, 1])
    v_total_speed = np.hypot(v_total[:, 0], v_total[:, 1])

    report: Dict[str, Any] = {
        "num_micro_plates": n,
        "num_macro_plates": n_macro,
        "plate_domain": ocean_stats,
        "velocity": {
            "micro_speed_mean_actual": float(v_micro_speed.mean()),
            "micro_speed_max_actual": float(v_micro_speed.max()),
            "macro_speed_mean_actual": float(v_macro_speed.mean()),
            "macro_speed_max_actual": float(v_macro_speed.max()),
            "total_speed_mean": float(v_total_speed.mean()),
            "total_speed_max": float(v_total_speed.max()),
        },
        "collisions": collision_stats,
        "rifts": rift_stats,
        "rift_cells": int((rift_id_field < 0).sum()),
        "coast_chamfers": n_coast_chamfer,
        "primary_ridge_cells": int(primary_mask.sum()),
        "dla_ridge_cells": int(tooth_mask.sum()),
        "ring_dla": {
            "stamp_count": len(stamps),
            "stamp_shape": list(stamps[0].shape),
            "stamp_fill": [round(float(s.mean()), 4) for s in stamps],
            "tooth_cells": int(tooth_mask.sum()),
            "rift_tooth_cells": int(rift_tooth_mask.sum()),
        },
        "ridge_elevation_max": float(ridge_elev.max()) if ridge_elev.size else 0.0,
        "uplift_max": float(uplift_field.max()) if uplift_field.size else 0.0,
        "elevation_min": float(elevation.min()),
        "elevation_max": float(elevation.max()),
        "land_fraction": float(world.land_mask.mean()),
    }

    return world, report