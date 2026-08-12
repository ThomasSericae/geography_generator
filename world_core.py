"""
world_core.py
架空世界生成器 - 世界数据容器模块（重写版）

提供 World 类，用于存储和管理架空世界的各个图层数据。
所有图层均使用 NumPy 数组实现，便于后续向量化生成与处理。

重写要点：
    1. 一维/二维柏林噪声提升为公共类 PerlinNoise1D / PerlinNoise2D。
       构造函数不再接收 int 种子，而是接收 numpy.random.Generator ——
       全项目统一使用世界种子派生的同一个随机数生成器（world.rng），
       任何模块不得自行播种。
    2. 新增 sea_level 属性（海平面），由海拔/水文模块统一读写。
    3. 水文与侵蚀模块的图层正式化为一等公民：
       rock_hardness / pressure_belt / humidity / river_mask /
       river_strength / river_discharge / deposition_type /
       deposition_thickness。
    4. 通用 DLA 生长引擎 grow_dla 与噪声并列，供河流、山脉等
       多个模块共用（投放范围/投放倾向/运动倾向/附着可行域均为参数）。
    5. 板块速度碰撞大修：新增海陆板块划分层 plate_domain、山脊线
       身份/相撞速度层 ridge_id / ridge_speed，以及板块级数据结构
       micro_to_macro / micro_plate_is_ocean / micro_plate_velocity /
       macro_plate_velocity（长度随板块数变化，由海拔模块整块赋值）。
"""

import math
import numpy as np
from numpy.random import SeedSequence, Generator, PCG64
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, NamedTuple, Optional, Tuple, Any, Union

# 八邻域偏移（DLA 引擎内部使用）
_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ============================================================
# 公共噪声工具：一维柏林噪声
# ============================================================
class PerlinNoise1D:
    """
    一维柏林噪声（标量接口）。

    Parameters
    ----------
    rng : numpy.random.Generator
        随机数生成器（通常传 world.rng）。排列表由 rng.permutation 生成，
        本类不会自行播种。
    """

    def __init__(self, rng: Generator):
        p = rng.permutation(256).astype(np.int64)
        self.p = np.concatenate([p, p])

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _grad(self, hash_val: int) -> float:
        return 1.0 if (hash_val & 1) == 0 else -1.0

    def noise(self, x: float) -> float:
        X = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = self._fade(xf)
        g0 = self._grad(int(self.p[X])) * xf
        g1 = self._grad(int(self.p[X + 1])) * (xf - 1.0)
        return self._lerp(g0, g1, u)

    def octave_noise(self, x: float, octaves: int, lacunarity: float, persistence: float = 0.5) -> float:
        total = 0.0
        freq = 1.0
        amp = 1.0
        max_val = 0.0
        for _ in range(octaves):
            total += self.noise(x * freq) * amp
            max_val += amp
            amp *= persistence
            freq *= lacunarity
        return total / max_val if max_val > 0 else 0.0


# ============================================================
# 公共噪声工具：二维柏林噪声（向量化实现）
# ============================================================
class PerlinNoise2D:
    """
    二维柏林噪声（向量化接口，x/y 为同形状数组）。

    Parameters
    ----------
    rng : numpy.random.Generator
        随机数生成器（通常传 world.rng）。排列表由 rng.permutation 生成，
        本类不会自行播种。
    """

    def __init__(self, rng: Generator):
        p = rng.permutation(256).astype(np.int32)
        self.p = np.concatenate([p, p])

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        return a + (b - a) * t

    def _grad(self, hash_val: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = hash_val & 3
        result = np.zeros_like(x, dtype=np.float64)
        mask0 = h == 0
        mask1 = h == 1
        mask2 = h == 2
        mask3 = h == 3
        result[mask0] = (x + y)[mask0]
        result[mask1] = (-x + y)[mask1]
        result[mask2] = (x - y)[mask2]
        result[mask3] = (-x - y)[mask3]
        return result

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.floor(x).astype(np.int32) & 255
        Y = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)

        p = self.p
        aa = p[p[X] + Y]
        ab = p[p[X] + Y + 1]
        ba = p[p[X + 1] + Y]
        bb = p[p[X + 1] + Y + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1.0, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1.0), self._grad(bb, xf - 1.0, yf - 1.0), u)
        return self._lerp(x1, x2, v)

    def octave_noise(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int,
        lacunarity: float,
        persistence: float = 0.5,
    ) -> np.ndarray:
        total = np.zeros_like(x, dtype=np.float64)
        freq = 1.0
        amp = 1.0
        max_val = 0.0
        for _ in range(octaves):
            total += self.noise(x * freq, y * freq) * amp
            max_val += amp
            amp *= persistence
            freq *= lacunarity
        if max_val > 0:
            return (total / max_val).astype(np.float32)
        return np.zeros_like(x, dtype=np.float32)


# ============================================================
# 公共形态工具：二值元胞自动机（边缘扰动 / 平滑）
# ============================================================
def ca_smooth_binary(mask: np.ndarray, iterations: int = 2,
                     threshold: int = 5) -> np.ndarray:
    """
    洞穴生成式二值元胞自动机平滑（多数规则）。

    每次迭代统计每个像素九格（八邻域 + 自身）中 True 的数量，
    ≥ threshold 则置 True，否则置 False。迭代数次后边界圆润、
    孤立噪点被消除、细碎缺口被填补。地图外视为 False。

    Parameters
    ----------
    mask : (h, w) bool 数组。
    iterations : 迭代次数（每次迭代每像素 9 次加法，O(K·N)）。
    threshold : 九格多数阈值，1~9；默认 5（严格多数）。
        调小 → 掩膜趋于扩张；调大 → 掩膜趋于收缩。

    Returns
    -------
    (h, w) bool 新数组。
    """
    m = np.ascontiguousarray(mask.astype(np.int8))
    for _ in range(max(int(iterations), 0)):
        p = np.pad(m, 1, mode="constant")
        cnt = (p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
               p[1:-1, 0:-2] + m               + p[1:-1, 2:] +
               p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:])
        m = (cnt >= threshold).astype(np.int8)
    return m.astype(bool)


def ca_edge_perturb(mask: np.ndarray, rng: Generator,
                    noise_prob: float = 0.15,
                    expand_zone: Optional[np.ndarray] = None,
                    iterations: int = 3,
                    threshold: int = 5,
                    constrain: Optional[np.ndarray] = None) -> np.ndarray:
    """
    用元胞自动机扰动二值掩膜的边缘（替代噪声抖动边界）。

    步骤：
        1. 在 expand_zone（候选扩张区，None 时全域）内按
           noise_prob 撒随机噪点，并入掩膜；
        2. 运行 ca_smooth_binary 多数规则迭代——噪点在原掩膜
           边缘团聚成有机的凸包/分叉，远离主体的碎点被消除；
        3. constrain（如陆地掩膜）非 None 时把结果限制在其内。

    渐近复杂度与噪声抖动同为 O(N)（K 次迭代 × 9 邻域计数，
    仅常数因子略大）。

    Parameters
    ----------
    mask : (h, w) bool 原掩膜。
    rng : numpy.random.Generator（通常传 world.rng），不自行播种。
    noise_prob : 噪点概率（0~1），越大边缘扰动越剧烈。
    expand_zone : (h, w) bool，允许撒噪点的区域；None = 全域。
    iterations / threshold : 见 ca_smooth_binary。
    constrain : (h, w) bool，结果的允许范围；None = 不限制。

    Returns
    -------
    (h, w) bool 新数组。
    """
    m = np.asarray(mask, dtype=bool)
    if noise_prob > 0:
        zone = np.ones_like(m) if expand_zone is None else np.asarray(expand_zone, dtype=bool)
        m = m | (zone & (rng.random(m.shape) < float(noise_prob)))
    m = ca_smooth_binary(m, iterations=iterations, threshold=threshold)
    if constrain is not None:
        m &= np.asarray(constrain, dtype=bool)
    return m


# ============================================================
# 通用 DLA 生长引擎（扩散限制聚集）
# ============================================================
class DLAResult(NamedTuple):
    """
    DLA 生长结果。

    Attributes
    ----------
    cluster : np.ndarray, bool, (h, w)
        聚集体掩膜（河流/山脉等）。
    parent_flat : np.ndarray, int64, (h, w)
        每个聚集体格的父格扁平索引（py * w + px），种子格为 -1。
        父格即“先存在”方向（河流即下游方向）。
    attach_log : list of (y, x, py, px)
        附着日志，按附着先后排列；逆序遍历即 枝叶 → 根 的拓扑序。
    stats : dict
        统计字典（粒子投放/附着/死亡等计数）。
    """
    cluster: np.ndarray
    parent_flat: np.ndarray
    attach_log: List[Tuple[int, int, int, int]]
    stats: Dict[str, Any]


def grow_dla(
    rng: Generator,
    guidance: np.ndarray,
    seeds: List[Tuple[int, int]],
    feasible: np.ndarray,
    base_level: float = 0.0,
    spawn_radius: float = 25.0,
    spawn_elevation_bias: float = 2.0,
    walk_elevation_bias: float = 1.0,
    max_neighbors: int = 3,
    max_particles: int = 20000,
    max_walk_steps: int = 400,
    pool_rebuild_interval: int = 256,
    seed_ids: Optional[List[int]] = None,
) -> DLAResult:
    """
    通用 DLA 生长引擎，供河流、山脉等多个模块共用。

    机制：
        1. 种子格构成初始聚集体；
        2. 维护“投放池”：距聚集体不超过 spawn_radius 的可行像素，
           按 ((guidance - base_level)/span + 0.02)^spawn_elevation_bias
           加权抽样 —— 投放范围 + 投放倾向；
        3. 粒子做八邻域随机游走，步进权重
           1 + walk_elevation_bias × 归一化高度 —— 粒子运动倾向；
        4. 粒子与聚集体相邻时附着：当前格加入聚集体，父格取
           guidance 最低的相邻聚集体格（保证“下游”方向合理）；
        5. 可行域控制：八邻居中聚集体数 >= max_neighbors 的像素
           永久禁入，保证聚集体细窄不成团；
        6. 归一化跨度 span = max(guidance) - base_level，
           使偏好强度与 guidance 的绝对取值范围解耦。

    注意：spawn_radius 不宜超过约 sqrt(max_walk_steps)（扩散可达距离），
    否则远处粒子难以走回聚集体附着，附着率会骤降。

    Parameters
    ----------
    rng : numpy.random.Generator
        随机数生成器（通常传 world.rng），本函数不自行播种。
    guidance : np.ndarray, (h, w)
        引导标量场（如海拔）。投放与游走的偏好都基于
        guidance - base_level 的归一化值。
    seeds : list of (y, x)
        初始聚集体格（河流模块传河口，山脉模块可传山脊种子）。
    feasible : np.ndarray, bool, (h, w)
        初始可行域（会被复制，不会修改调用方的数组）。
    base_level : float
        归一化基准（河流用海平面）。
    spawn_radius : float
        投放池半径：粒子只在距聚集体该距离内投放。
    spawn_elevation_bias : float
        投放倾向指数，越大越偏向高 guidance 处投放。
    walk_elevation_bias : float
        运动倾向强度（加性），正值倾向走向高 guidance。
    max_neighbors : int
        附着可行域控制：邻居聚集体数达到该值的像素永久禁入。
    max_particles / max_walk_steps : int
        粒子总数与单粒子步数上限（性能兜底）。
    pool_rebuild_interval : int
        投放池快照每附着多少格重建一次。
    seed_ids : list of int, optional
        与 seeds 等长的种子身份标识（如同一山系编号）。
        提供时启用“桥接终止”：粒子若同时与两个及以上不同身份的
        聚集体相邻（即将成为连接异源聚集体的桥），立即终止而不附着；
        附着成功后子格继承父格身份。不提供则行为与旧版一致。

    Returns
    -------
    DLAResult
    """
    h, w = guidance.shape
    guide = np.asarray(guidance, dtype=np.float64)
    has_ids = seed_ids is not None
    if has_ids and len(seed_ids) != len(seeds):
        raise ValueError("seed_ids 必须与 seeds 等长。")

    cluster = np.zeros((h, w), dtype=bool)
    cluster_id = np.full((h, w), -1, dtype=np.int32)
    parent_flat = np.full((h, w), -1, dtype=np.int64)
    attach_log: List[Tuple[int, int, int, int]] = []
    stats: Dict[str, Any] = {
        "num_seeds": 0,
        "particles_launched": 0,
        "particles_attached": 0,
        "particles_dead": 0,
        "particles_bridge_killed": 0,
        "particles_exhausted": 0,
        "invalid_spawns": 0,
        "cluster_cells": 0,
    }

    feasible = feasible.copy()
    cluster_nb_count = np.zeros((h, w), dtype=np.int16)

    def mark_cluster(y: int, x: int, py: int, px: int, cid: int = -1) -> None:
        cluster[y, x] = True
        feasible[y, x] = False
        parent_flat[y, x] = (py * w + px) if py >= 0 else -1
        if has_ids:
            cluster_id[y, x] = cid
        attach_log.append((y, x, py, px))
        for dy, dx in _D8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not cluster[ny, nx]:
                cluster_nb_count[ny, nx] += 1
                if cluster_nb_count[ny, nx] >= max_neighbors:
                    feasible[ny, nx] = False

    # ---- 种子构成初始聚集体 ----
    stats["num_seeds"] = len(seeds)
    for i, (y, x) in enumerate(seeds):
        mark_cluster(int(y), int(x), -1, -1,
                     seed_ids[i] if has_ids else -1)
    if not seeds:
        return DLAResult(cluster, parent_flat, attach_log, stats)

    # ---- 投放池（快照 + 定期重建）----
    span = max(float(guide.max()) - base_level, 1e-6)

    pool_ys: Optional[np.ndarray] = None
    pool_xs: Optional[np.ndarray] = None
    pool_cdf: Optional[np.ndarray] = None

    def rebuild_pool() -> None:
        nonlocal pool_ys, pool_xs, pool_cdf
        dist_to_cluster = distance_transform_edt(~cluster)
        pool_mask = feasible & (dist_to_cluster <= spawn_radius)
        ys, xs = np.nonzero(pool_mask)
        if len(ys) == 0:
            pool_ys, pool_xs, pool_cdf = None, None, None
            return
        norm = np.maximum(guide[ys, xs] - base_level, 0.0) / span
        weights = np.power(norm + 0.02, spawn_elevation_bias)
        pool_ys, pool_xs = ys, xs
        pool_cdf = np.cumsum(weights)

    def sample_spawn() -> Optional[Tuple[int, int]]:
        if pool_cdf is None or len(pool_cdf) == 0:
            return None
        total = pool_cdf[-1]
        last = len(pool_cdf) - 1
        for _ in range(8):
            r = rng.random() * total
            i = int(np.searchsorted(pool_cdf, r))
            if i > last:
                i = last
            y, x = int(pool_ys[i]), int(pool_xs[i])
            if feasible[y, x] and not cluster[y, x]:
                return y, x
        return None

    # ---- 粒子游走与附着 ----
    def launch_particle(sy: int, sx: int) -> Tuple[str, int, int, int, int]:
        y, x = sy, sx
        for _step in range(max_walk_steps):
            best_py, best_px = -1, -1
            best_pe = math.inf
            contact_id_count = 0
            last_contact_id = -1
            nbs: List[Tuple[int, int]] = []
            nbs_w: List[float] = []
            for dy, dx in _D8:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if cluster[ny, nx]:
                    e2 = guide[ny, nx]
                    if e2 < best_pe:
                        best_pe = e2
                        best_py, best_px = ny, nx
                    if has_ids:
                        cid = cluster_id[ny, nx]
                        if cid != last_contact_id:
                            contact_id_count += 1
                            last_contact_id = cid
                elif feasible[ny, nx]:
                    nbs.append((ny, nx))
                    norm_n = max(guide[ny, nx] - base_level, 0.0) / span
                    nbs_w.append(1.0 + walk_elevation_bias * norm_n)
            if best_py >= 0:
                # 桥接终止：同时接触两个及以上异源聚集体 → 终止不附着
                if has_ids and contact_id_count > 1:
                    return ("bridge", y, x, -1, -1)
                return ("attach", y, x, best_py, best_px)
            if not nbs:
                return ("dead", y, x, -1, -1)
            # 偏好抽样（手写累积扫描，避免 numpy 开销）
            total_w = 0.0
            for wt in nbs_w:
                total_w += wt
            r = rng.random() * total_w
            acc = 0.0
            pick = len(nbs) - 1
            for i, wt in enumerate(nbs_w):
                acc += wt
                if r <= acc:
                    pick = i
                    break
            y, x = nbs[pick]
        return ("exhausted", y, x, -1, -1)

    # ---- 主循环 ----
    rebuild_pool()
    since_rebuild = 0
    for _ in range(max_particles):
        if pool_cdf is None:
            break  # 池已空，无可投放区域
        if since_rebuild >= pool_rebuild_interval:
            rebuild_pool()
            since_rebuild = 0
            if pool_cdf is None:
                break
        spawn = sample_spawn()
        if spawn is None:
            stats["invalid_spawns"] += 1
            rebuild_pool()  # 快照过期，强制重建后继续
            since_rebuild = 0
            if pool_cdf is None:
                break
            continue
        stats["particles_launched"] += 1
        status, y, x, py, px = launch_particle(*spawn)
        if status == "attach":
            mark_cluster(y, x, py, px, cluster_id[py, px] if has_ids else -1)
            stats["particles_attached"] += 1
            since_rebuild += 1
        elif status == "bridge":
            stats["particles_bridge_killed"] += 1
        elif status == "exhausted":
            stats["particles_exhausted"] += 1
        else:
            stats["particles_dead"] += 1

    stats["cluster_cells"] = len(attach_log)
    return DLAResult(cluster, parent_flat, attach_log, stats)


# ============================================================
# 世界数据容器
# ============================================================
class World:
    """
    架空世界的核心数据容器。

    使用 NumPy 数组存储世界的各个图层，包括海拔、海陆分布、生物群系、
    板块构造、岩石硬度、湿度、河网与沉积等。同时维护一个基于世界种子的
    随机数生成器（rng），供所有生成模块统一使用——任何模块不得自行播种。

    Attributes
    ----------
    seed : int
        世界种子，决定伪随机序列，保证世界可复现。
    width : int
        世界地图宽度（像素/格数）。
    height : int
        世界地图高度（像素/格数）。
    sea_level : float
        海平面海拔。elevation > sea_level 为陆地，反之为水体。
        本项目暂假设所有水体均为淡水。
    """

    def __init__(self, seed: int, width: int = 512, height: int = 512):
        """
        初始化一个世界实例。

        Parameters
        ----------
        seed : int
            世界种子。使用相同的种子可以生成完全相同的世界。
        width : int, optional
            世界宽度，默认 512。
        height : int, optional
            世界高度，默认 512。
        """
        self.seed: int = int(seed)
        self.width: int = int(width)
        self.height: int = int(height)

        # 海平面（默认 0.0，由海拔/水文模块读写）
        self._sea_level: float = 0.0

        # 基于种子的随机数生成器（全项目唯一随机源）
        self._rng: Generator = Generator(PCG64(SeedSequence(self.seed)))

        # ---------- 核心图层 ----------
        # 海拔层：单精度浮点，单位建议为米（可正可负，负值表示海床）
        self._elevation: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # 海陆分布层：布尔型，True = 陆地，False = 水体（淡水）
        self._land_mask: np.ndarray = np.zeros((self.height, self.width), dtype=np.bool_)

        # 生物群系层：整数型，每个整数值映射到一种生物群系
        self._biome: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # ---------- 板块构造图层 ----------
        # 小板块层：每个像素所属的小板块（Voronoi 细胞）ID
        self._micro_plates: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # 大板块层：每个像素所属的大板块（聚类后）ID
        self._macro_plates: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # 板块边界层：标记各类板块边界
        # 0 = 无边界
        # 1 = 小板块边界
        # 2 = 大板块边界
        # 3 = 地图边界边缘（与地图边界相连的边缘）
        # 4 = 山脉生成边缘（被选中的造山边缘，无论高山低山）
        self._plate_boundaries: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # ---------- 板块速度碰撞体系图层 ----------
        # 海陆板块划分层：int8。0 = 大陆板块，1 = 海洋板块。
        # （地图边缘小板块一律为海洋板块，扩张转化见海拔模块）
        self._plate_domain: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)

        # 山脊线身份层：int32。0 = 非山脊；1..k = 第 k 条碰撞山脊线
        # （DLA 梳齿继承父格身份）。
        self._ridge_id: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)

        # 山脊相撞速度层：float32。山脊像素记录所属分界线的相撞速度
        # （速度单位见海拔模块；非山脊为 0）。
        self._ridge_speed: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # ---------- 板块级数据（非像素图层，长度随板块数变化）----------
        # 小板块 → 大板块 归属映射，int32，(n_micro,)
        self._micro_to_macro: np.ndarray = np.zeros(0, dtype=np.int32)
        # 小板块是否为海洋板块，bool，(n_micro,)
        self._micro_plate_is_ocean: np.ndarray = np.zeros(0, dtype=np.bool_)
        # 小板块个体速度向量 (vx, vy)，float64，(n_micro, 2)
        self._micro_plate_velocity: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        # 大板块集体速度向量 (vx, vy)，float64，(n_macro, 2)
        self._macro_plate_velocity: np.ndarray = np.zeros((0, 2), dtype=np.float64)

        # ---------- 水文与侵蚀图层 ----------
        # 岩石硬度层：int16，整数，范围 0~255。越高越抗蚀。
        self._rock_hardness: np.ndarray = np.zeros((self.height, self.width), dtype=np.int16)

        # 气压带层：int8。仅与纬度相关（地图顶部 = 70°N，底部 = 0° 赤道）。
        # 0 = 赤道低气压带（0~10°N）
        # 1 = 信风带（10~25°N）
        # 2 = 副热带高气压带（25~35°N）
        # 3 = 地中海带（35~42°N）
        # 4 = 西风带（42~55°N）
        # 5 = 副极地低气压带（55~70°N）
        self._pressure_belt: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)

        # 湿度层：float32，范围 0~100。水体 = 100，
        # 陆地按气压带各自的距水距离宽度划分为湿润/半湿润/半干旱/干旱四级。
        self._humidity: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # 河流掩膜层：bool，True = 河道像素。
        self._river_mask: np.ndarray = np.zeros((self.height, self.width), dtype=np.bool_)

        # 河流强度层：float32，上游汇流计数（流量代理）。
        # 离入海口越近数值越大，源头为 1。
        self._river_strength: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # 河流水流量层：float32。源头注入水流量（基础 + 湿润地区加成 +
        # 临近高山加成）并向下游累加，每个河道格有自己的水流量值，
        # 离入海口越近越大。
        self._river_discharge: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # 沉积类型层：int8。
        # 0 = 无沉积
        # 1 = 山前/盆地冲积扇
        # 2 = 河口三角洲
        self._deposition_type: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)

        # 沉积厚度层：float32，本模块沉积作用造成的累计抬升量（米）。
        self._deposition_thickness: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)

        # ---------- 扩展图层 ----------
        # 允许用户或后续模块动态添加自定义图层
        self._layers: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """返回世界形状 (height, width)。"""
        return (self.height, self.width)

    @property
    def rng(self) -> Generator:
        """基于世界种子的 NumPy 随机数生成器（全项目唯一随机源）。"""
        return self._rng

    @property
    def elevation(self) -> np.ndarray:
        """海拔图层，float32。"""
        return self._elevation

    @property
    def land_mask(self) -> np.ndarray:
        """海陆分布图层，bool。True = 陆地，False = 水体（淡水）。"""
        return self._land_mask

    @property
    def biome(self) -> np.ndarray:
        """生物群系图层，int32。"""
        return self._biome

    @property
    def micro_plates(self) -> np.ndarray:
        """小板块图层，int32。每个像素所属的 Voronoi 细胞 ID。"""
        return self._micro_plates

    @property
    def macro_plates(self) -> np.ndarray:
        """大板块图层，int32。每个像素所属的大板块 ID。"""
        return self._macro_plates

    @property
    def plate_boundaries(self) -> np.ndarray:
        """
        板块边界图层，int32。

        编码：
        0 = 无边界
        1 = 小板块边界
        2 = 大板块边界
        3 = 地图边界边缘
        4 = 山脉生成边缘（无论高山低山）
        """
        return self._plate_boundaries

    @property
    def plate_domain(self) -> np.ndarray:
        """海陆板块划分图层，int8。0 = 大陆板块，1 = 海洋板块。"""
        return self._plate_domain

    @property
    def ridge_id(self) -> np.ndarray:
        """山脊线身份图层，int32。0 = 非山脊；1..k = 第 k 条碰撞山脊线。"""
        return self._ridge_id

    @property
    def ridge_speed(self) -> np.ndarray:
        """山脊相撞速度图层，float32。山脊像素记录所属分界线的相撞速度。"""
        return self._ridge_speed

    # ------------------------------------------------------------------
    # 板块级数据（长度随板块数变化，可整块赋值）
    # ------------------------------------------------------------------

    @property
    def micro_to_macro(self) -> np.ndarray:
        """小板块 → 大板块归属映射，int32，(n_micro,)。"""
        return self._micro_to_macro

    @micro_to_macro.setter
    def micro_to_macro(self, value: np.ndarray) -> None:
        self._micro_to_macro = np.asarray(value, dtype=np.int32)

    @property
    def micro_plate_is_ocean(self) -> np.ndarray:
        """小板块是否为海洋板块，bool，(n_micro,)。"""
        return self._micro_plate_is_ocean

    @micro_plate_is_ocean.setter
    def micro_plate_is_ocean(self, value: np.ndarray) -> None:
        self._micro_plate_is_ocean = np.asarray(value, dtype=np.bool_)

    @property
    def micro_plate_velocity(self) -> np.ndarray:
        """小板块个体速度向量，float64，(n_micro, 2)。方向随机，大小服从正态分布。"""
        return self._micro_plate_velocity

    @micro_plate_velocity.setter
    def micro_plate_velocity(self, value: np.ndarray) -> None:
        self._micro_plate_velocity = np.asarray(value, dtype=np.float64)

    @property
    def macro_plate_velocity(self) -> np.ndarray:
        """大板块集体速度向量，float64，(n_macro, 2)。幅值普遍大于小板块个体速度。"""
        return self._macro_plate_velocity

    @macro_plate_velocity.setter
    def macro_plate_velocity(self, value: np.ndarray) -> None:
        self._macro_plate_velocity = np.asarray(value, dtype=np.float64)

    @property
    def total_micro_plate_velocity(self) -> np.ndarray:
        """小板块总速度 = 个体速度 + 所属大板块集体速度，float64，(n_micro, 2)。"""
        if (self._micro_plate_velocity.shape[0] == 0
                or self._macro_plate_velocity.shape[0] == 0
                or self._micro_to_macro.shape[0] != self._micro_plate_velocity.shape[0]):
            return self._micro_plate_velocity.copy()
        return self._micro_plate_velocity + self._macro_plate_velocity[self._micro_to_macro]

    @property
    def rock_hardness(self) -> np.ndarray:
        """岩石硬度图层，int16，范围 0~255，越高越抗蚀。"""
        return self._rock_hardness

    @property
    def pressure_belt(self) -> np.ndarray:
        """
        气压带图层，int8。仅与纬度相关（顶部 = 70°N，底部 = 0° 赤道）。

        编码：
        0 = 赤道低气压带（0~10°N）
        1 = 信风带（10~25°N）
        2 = 副热带高气压带（25~35°N）
        3 = 地中海带（35~42°N）
        4 = 西风带（42~55°N）
        5 = 副极地低气压带（55~70°N）
        """
        return self._pressure_belt

    @property
    def humidity(self) -> np.ndarray:
        """湿度图层，float32，范围 0~100。水体 = 100。"""
        return self._humidity

    @property
    def river_mask(self) -> np.ndarray:
        """河流掩膜图层，bool。True = 河道像素。"""
        return self._river_mask

    @property
    def river_strength(self) -> np.ndarray:
        """河流强度图层，float32。上游汇流计数，离入海口越近越大。"""
        return self._river_strength

    @property
    def river_discharge(self) -> np.ndarray:
        """河流水流量图层，float32。源头注入（含湿润/临高山加成）向下游累加，离入海口越近越大。"""
        return self._river_discharge

    @property
    def deposition_type(self) -> np.ndarray:
        """
        沉积类型图层，int8。

        编码：
        0 = 无沉积
        1 = 山前/盆地冲积扇
        2 = 河口三角洲
        """
        return self._deposition_type

    @property
    def deposition_thickness(self) -> np.ndarray:
        """沉积厚度图层，float32。沉积作用造成的累计抬升量（米）。"""
        return self._deposition_thickness

    # ------------------------------------------------------------------
    # 海平面（可读写）
    # ------------------------------------------------------------------

    @property
    def sea_level(self) -> float:
        """海平面海拔。elevation > sea_level 为陆地。"""
        return self._sea_level

    @sea_level.setter
    def sea_level(self, value: float) -> None:
        self._sea_level = float(value)

    # ------------------------------------------------------------------
    # 扩展图层管理
    # ------------------------------------------------------------------

    def add_layer(
        self,
        name: str,
        dtype: Union[np.dtype, str],
        default_value: Any = 0,
    ) -> np.ndarray:
        """动态添加一个自定义图层。"""
        if name in self._layers:
            raise KeyError(f'图层 "{name}" 已存在。如需覆盖，请先调用 remove_layer("{name}")。')
        layer = np.full(self.shape, default_value, dtype=dtype)
        self._layers[name] = layer
        return layer

    def get_layer(self, name: str) -> Optional[np.ndarray]:
        """获取自定义图层。"""
        return self._layers.get(name)

    def remove_layer(self, name: str) -> bool:
        """删除自定义图层。"""
        if name in self._layers:
            del self._layers[name]
            return True
        return False

    def list_layers(self) -> Tuple[str, ...]:
        """返回所有自定义图层的名称列表。"""
        return tuple(self._layers.keys())

    # ------------------------------------------------------------------
    # 实用方法
    # ------------------------------------------------------------------

    def reset_core_layers(self) -> None:
        """将核心图层（海拔、海陆、生物群系）重置为默认值。"""
        self._elevation.fill(0.0)
        self._land_mask.fill(False)
        self._biome.fill(0)

    def reset_plate_layers(self) -> None:
        """将板块构造图层（含速度碰撞体系与板块级数据）重置为默认值。"""
        self._micro_plates.fill(0)
        self._macro_plates.fill(0)
        self._plate_boundaries.fill(0)
        self._plate_domain.fill(0)
        self._ridge_id.fill(0)
        self._ridge_speed.fill(0.0)
        self._micro_to_macro = np.zeros(0, dtype=np.int32)
        self._micro_plate_is_ocean = np.zeros(0, dtype=np.bool_)
        self._micro_plate_velocity = np.zeros((0, 2), dtype=np.float64)
        self._macro_plate_velocity = np.zeros((0, 2), dtype=np.float64)

    def reset_hydro_layers(self) -> None:
        """将水文与侵蚀图层重置为默认值。"""
        self._rock_hardness.fill(0)
        self._pressure_belt.fill(0)
        self._humidity.fill(0.0)
        self._river_mask.fill(False)
        self._river_strength.fill(0.0)
        self._deposition_type.fill(0)
        self._deposition_thickness.fill(0.0)

    def reset_all_layers(self) -> None:
        """重置所有图层（核心 + 板块 + 水文 + 自定义）。"""
        self.reset_core_layers()
        self.reset_plate_layers()
        self.reset_hydro_layers()
        for layer in self._layers.values():
            layer.fill(0)

    def copy(self) -> "World":
        """创建世界的深拷贝。"""
        new_world = World(self.seed, self.width, self.height)
        new_world._sea_level = self._sea_level
        new_world._elevation[...] = self._elevation
        new_world._land_mask[...] = self._land_mask
        new_world._biome[...] = self._biome
        new_world._micro_plates[...] = self._micro_plates
        new_world._macro_plates[...] = self._macro_plates
        new_world._plate_boundaries[...] = self._plate_boundaries
        new_world._plate_domain[...] = self._plate_domain
        new_world._ridge_id[...] = self._ridge_id
        new_world._ridge_speed[...] = self._ridge_speed
        new_world._micro_to_macro = self._micro_to_macro.copy()
        new_world._micro_plate_is_ocean = self._micro_plate_is_ocean.copy()
        new_world._micro_plate_velocity = self._micro_plate_velocity.copy()
        new_world._macro_plate_velocity = self._macro_plate_velocity.copy()
        new_world._rock_hardness[...] = self._rock_hardness
        new_world._pressure_belt[...] = self._pressure_belt
        new_world._humidity[...] = self._humidity
        new_world._river_mask[...] = self._river_mask
        new_world._river_strength[...] = self._river_strength
        new_world._deposition_type[...] = self._deposition_type
        new_world._deposition_thickness[...] = self._deposition_thickness
        for name, arr in self._layers.items():
            new_world.add_layer(name, arr.dtype, 0)
            new_world._layers[name][...] = arr
        return new_world

    def __repr__(self) -> str:
        return (
            f"World(seed={self.seed}, width={self.width}, height={self.height}, "
            f"sea_level={self._sea_level}, layers={len(self._layers)})"
        )

    def __eq__(self, other: object) -> bool:
        """判断两个世界是否相等。"""
        if not isinstance(other, World):
            return NotImplemented
        return (
            self.seed == other.seed
            and self.width == other.width
            and self.height == other.height
            and self._sea_level == other._sea_level
            and np.array_equal(self._elevation, other._elevation)
            and np.array_equal(self._land_mask, other._land_mask)
            and np.array_equal(self._biome, other._biome)
            and np.array_equal(self._micro_plates, other._micro_plates)
            and np.array_equal(self._macro_plates, other._macro_plates)
            and np.array_equal(self._plate_boundaries, other._plate_boundaries)
            and np.array_equal(self._plate_domain, other._plate_domain)
            and np.array_equal(self._ridge_id, other._ridge_id)
            and np.array_equal(self._ridge_speed, other._ridge_speed)
            and np.array_equal(self._micro_to_macro, other._micro_to_macro)
            and np.array_equal(self._micro_plate_is_ocean, other._micro_plate_is_ocean)
            and np.array_equal(self._micro_plate_velocity, other._micro_plate_velocity)
            and np.array_equal(self._macro_plate_velocity, other._macro_plate_velocity)
            and np.array_equal(self._rock_hardness, other._rock_hardness)
            and np.array_equal(self._pressure_belt, other._pressure_belt)
            and np.array_equal(self._humidity, other._humidity)
            and np.array_equal(self._river_mask, other._river_mask)
            and np.array_equal(self._river_strength, other._river_strength)
            and np.array_equal(self._deposition_type, other._deposition_type)
            and np.array_equal(self._deposition_thickness, other._deposition_thickness)
        )