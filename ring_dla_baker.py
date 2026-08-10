"""
ring_dla_baker.py
环形 DLA 纹理烘焙工具（一次性离线运行的小模块）

在一个半径很大的圆环上运行 DLA（粒子自环内外两侧投放，枝杈沿
径向生长），把聚集体沿极坐标展开为矩形梳子纹理（行 = 距环偏移、
列 = 弧长），压缩后以 base85 文本硬编码进 ring_dla_stamps.py。

海拔生成模块运行时直接从烘焙文件解码（毫秒级），完全不跑 DLA；
纹理列向无缝（环是周期的），山脊按弧长截取任意窗口都没有接缝。

半径选择：512 见方地图的山脊线一般不超过约 400 格，取 R = 512
时环周长 2πR ≈ 3217 列，任何山脊都能无回绕截取；且 R ≫ 齿长，
极坐标展开的几何畸变（齿尖/齿根宽度差 (R±L)/R ≈ ±6%）可忽略。

用法（在包含 world_core.py 的项目目录下运行）：
    python3 ring_dla_baker.py                      # 默认参数
    python3 ring_dla_baker.py --radius 512 --tooth-length 30 \
        --count 2 --target-fill 0.40 --output ring_dla_stamps.py
"""

import argparse
import base64
import math
import time
import zlib
from typing import List, Tuple

import numpy as np

from world_core import grow_dla


# ------------------------------------------------------------
# 大圆环 DLA 生长（分轮进行，直至环带填充率达标）
# ------------------------------------------------------------
def grow_ring_dla(
    rng: np.random.Generator,
    radius: int,
    tooth_length: int,
    target_fill: float,
    particles_per_round: int,
    max_rounds: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    在半径 radius 的圆环种子上生长 DLA。

    粒子自环内外两侧投放（投放池限于聚集体附近），枝杈处处沿
    径向——展开后即"梳齿垂直于山脊线"的两侧梳子。分轮调用
    通用 grow_dla 引擎：每轮以当前聚集体为种子、以环带剩余
    部分为可行域，直到环带填充率达到 target_fill。

    返回 (聚集体画布, 环心 y, 环心 x, 实际填充率)。
    """
    pad = tooth_length + 6
    half = int(radius + pad)
    size = 2 * half + 1
    cy = cx = float(half)
    yv, xv = np.mgrid[0:size, 0:size]
    dist = np.hypot(yv - cy, xv - cx)
    ring = np.abs(dist - radius) <= 0.8                      # 环形种子带
    band = np.abs(dist - radius) <= tooth_length             # 齿可及的环带
    band[:2, :] = band[-2:, :] = False
    band[:, :2] = band[:, -2:] = False
    tooth_area = int((band & ~ring).sum())
    guidance = np.clip(tooth_length - np.abs(dist - radius), 0.0, None)

    cluster = ring.copy()
    attach_order: List[Tuple[int, int]] = []   # 全局附着顺序（跨轮拼接）
    for rd in range(1, max_rounds + 1):
        fill = float((cluster & band & ~ring).sum()) / max(tooth_area, 1)
        if fill >= target_fill:
            break
        feasible = band & ~cluster
        seeds_y, seeds_x = np.nonzero(cluster)
        result = grow_dla(
            rng, guidance,
            list(zip(seeds_y.tolist(), seeds_x.tolist())), feasible,
            base_level=0.0,
            spawn_radius=18.0,          # ≤ √max_walk_steps，保证附着率
            spawn_elevation_bias=1.5,
            walk_elevation_bias=2.0,
            max_neighbors=2,
            max_particles=particles_per_round,
            max_walk_steps=400,
            pool_rebuild_interval=512,
        )
        cluster |= result.cluster
        # attach_log 中 py=-1 的是种子格（后续轮种子为整个聚集体，
        # 数量巨大），只记录真正的新附着格
        attach_order.extend((y, x) for y, x, py, _px in result.attach_log
                            if py >= 0)
        print("  第 %d 轮：附着 %d 格，填充率 %.3f" % (
            rd, int(result.cluster.sum()),
            float((cluster & band & ~ring).sum()) / max(tooth_area, 1)))

    # 按全局附着顺序裁尾，精确控制填充率：父格先于子格附着，
    # 保留前 K 格不会破坏枝杈与环的连通性
    final_fill = float((cluster & band & ~ring).sum()) / max(tooth_area, 1)
    if final_fill > target_fill and attach_order:
        keep = int(target_fill * tooth_area)
        cluster = ring.copy()
        for y, x in attach_order[:keep]:
            cluster[y, x] = True
        final_fill = float((cluster & band & ~ring).sum()) / max(tooth_area, 1)
        print("  裁尾至目标填充率：保留 %d 格，填充率 %.3f" % (keep, final_fill))
    return cluster, cy, cx, final_fill


# ------------------------------------------------------------
# 极坐标展开：圆环 → 矩形梳子纹理（列向无缝）
# ------------------------------------------------------------
def unwrap_ring_to_stamp(
    cluster: np.ndarray,
    cy: float,
    cx: float,
    radius: int,
    tooth_length: int,
) -> np.ndarray:
    """
    行 = 相对环半径的偏移 −L..+L（齿朝两侧），列 = 角度 × 半径
    （即弧长，一周共 ⌈2πR⌉ 列）。环的周期性使纹理列向无缝。
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


# ------------------------------------------------------------
# 压缩编码并写出硬编码 Python 数据文件
# ------------------------------------------------------------
_FILE_TEMPLATE = '''"""
ring_dla_stamps.py    【本文件由 ring_dla_baker.py 自动生成，请勿手改】

环形 DLA 梳子纹理（硬编码数据）：大半径圆环 DLA 经极坐标展开得到，
行 = 距脊偏移 −L..+L（齿朝两侧），列 = 弧长（列向无缝，可任意截取）。
海拔生成模块通过 load_stamps() 解码使用，毫秒级完成，运行时不再
进行任何 DLA 随机游走。

烘焙参数：{meta_line}
"""

import base64
import zlib

import numpy as np

META = {meta!r}

_DATA = [
{data_block}
]


def load_stamps():
    """解码全部梳子纹理，返回 bool 数组列表 [(rows, cols), ...]。"""
    rows, cols = META["rows"], META["cols"]
    stamps = []
    for blob in _DATA:
        raw = zlib.decompress(base64.b85decode(blob.encode("ascii")))
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=rows * cols)
        stamps.append(bits.reshape(rows, cols).astype(bool))
    return stamps
'''


def bake(
    stamps: List[np.ndarray],
    meta: dict,
    output: str,
) -> None:
    """np.packbits → zlib → base85，把纹理硬编码进 Python 文件。"""
    blobs = []
    for stamp in stamps:
        packed = np.packbits(stamp.astype(np.uint8))
        comp = zlib.compress(packed.tobytes(), 9)
        blobs.append(base64.b85encode(comp).decode("ascii"))
    data_block = ",\n".join(
        '    "%s"' % blob for blob in blobs
    )
    meta_line = ", ".join("%s=%r" % (k, v) for k, v in sorted(meta.items()))
    with open(output, "w", encoding="utf-8") as f:
        f.write(_FILE_TEMPLATE.format(meta=meta, meta_line=meta_line,
                                      data_block=data_block))
    total_chars = sum(len(b) for b in blobs)
    print("已写出 %s：%d 张纹理，编码文本共 %.1f KB" % (
        output, len(blobs), total_chars / 1024.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="环形 DLA 纹理烘焙工具")
    ap.add_argument("--radius", type=int, default=512,
                    help="圆环半径(格)；周长 2πR 应不小于最长山脊线 [默认 512]")
    ap.add_argument("--tooth-length", type=int, default=30,
                    help="梳齿烘焙长度(格)；运行时可裁剪到更小值 [默认 30]")
    ap.add_argument("--count", type=int, default=2,
                    help="烘焙纹理张数；↑齿形更多样 [默认 2]")
    ap.add_argument("--target-fill", type=float, default=0.40,
                    help="环带目标填充率；↑梳齿更密 [默认 0.40]")
    ap.add_argument("--particles-per-round", type=int, default=30000,
                    help="每轮 DLA 粒子上限 [默认 30000]")
    ap.add_argument("--max-rounds", type=int, default=10,
                    help="生长轮数上限 [默认 10]")
    ap.add_argument("--seed", type=int, default=7106,
                    help="烘焙随机种子 [默认 7106]")
    ap.add_argument("--output", type=str, default="ring_dla_stamps.py",
                    help="输出文件 [默认 ring_dla_stamps.py]")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    stamps = []
    t0 = time.perf_counter()
    for i in range(args.count):
        print("生长环形 DLA %d/%d (R=%d, L=%d)..." % (
            i + 1, args.count, args.radius, args.tooth_length))
        cluster, cy, cx, fill = grow_ring_dla(
            rng, args.radius, args.tooth_length,
            args.target_fill, args.particles_per_round, args.max_rounds,
        )
        stamp = unwrap_ring_to_stamp(cluster, cy, cx, args.radius, args.tooth_length)
        stamps.append(stamp)
        print("  展开完成：形状 %s，纹理填充率 %.3f" % (stamp.shape, stamp.mean()))
    meta = {
        "radius": args.radius,
        "tooth_length": args.tooth_length,
        "count": args.count,
        "seed": args.seed,
        "rows": int(stamps[0].shape[0]),
        "cols": int(stamps[0].shape[1]),
    }
    bake(stamps, meta, args.output)
    print("总耗时 %.1f s（一次性离线成本）" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()