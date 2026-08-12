"""
world_map_viewer.py
World Map Viewer 2D (pure tkinter + PIL, no matplotlib)

A flat top-down map viewer for the generated world, kept fully
independent from the 3D viewer (world_viewer.py):

    - Color layers (radio buttons):
        * Elevation     - continuous terrain-style gradient
                          (flat blue water; green -> khaki -> brown -> white land)
        * Humidity      - blue gradient by humidity
        * Rock Hardness - grayscale
    - Overlay layers (check buttons):
        * Rivers      - river cells in cyan
        * Deposition  - alluvial fans (yellow) / deltas (orange), blended
        * Mountains   - mountain_mask (DLA ridge-stamp coverage), blended red;
                        rivers currently terminate at its edge
        * Outlets     - outlet scatter debug view: dilated coast band
                        (scatter-available area, magenta tint) + seeded
                        outlet points (bright magenta)
    - Single seed entry + "Generate" button (no console input).

All UI text is English (target environment has no CJK font).
World generation runs in a worker thread so the UI stays responsive.
"""

import colorsys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from world_core import World
from elevation_generator import generate_mountain_terrain
from hydrology_erosion import generate_hydrology_erosion


# ============================================================
# Tunable constants (not exposed in the GUI)
# ============================================================
WORLD_WIDTH = 256
WORLD_HEIGHT = 256
DISPLAY_SCALE = 3           # integer upscale for the on-screen map
SEA_LEVEL = 0.0             # 固定海平面（板块速度碰撞版的海陆由板块划分物理决定，
                          # 不再用海拔分位数反推——双峰海拔分布下分位数会失效）
NUM_OUTLETS = 18
MAX_PARTICLES = 8000
DEFAULT_SEED = 42

# Elevation colouring: continuous terrain-style gradient
# (water flat blue; land green -> khaki -> brown -> white by relief).
_WATER_RGB = np.array([0.10, 0.20, 0.55])
_TERRAIN_STOPS = [
    (0.00, np.array([0.28, 0.60, 0.28])),   # lowland green
    (0.35, np.array([0.62, 0.78, 0.38])),   # khaki green
    (0.60, np.array([0.55, 0.42, 0.25])),   # brown
    (0.80, np.array([0.62, 0.60, 0.58])),   # grey rock
    (1.00, np.array([0.97, 0.97, 0.97])),   # snow white
]

_RIVER_RGB = np.array([102, 178, 255], dtype=np.float64)
_FAN_RGB = np.array([255, 213, 74], dtype=np.float64)
_DELTA_RGB = np.array([255, 112, 67], dtype=np.float64)
_DEPOSITION_BLEND = 0.65    # 沉积区颜色与地形的混合比
_MOUNTAIN_RGB = np.array([192, 57, 43], dtype=np.float64)
_MOUNTAIN_BLEND = 0.55      # 山脉（DLA 梳齿纹样覆盖区）颜色与底色的混合比

# Outlet scatter debug overlay
_OUTLET_BAND_RGB = np.array([255, 64, 200], dtype=np.float64)
_OUTLET_BAND_BLEND = 0.45   # 宽海岸带（撒点可用区）颜色与底色的混合比
_OUTLET_RGB = np.array([255, 0, 255], dtype=np.float64)

# Plates 板块视图：色相按大板块分组；鲜艳明亮 = 大陆板块，灰暗低沉 = 海洋板块
_VELOCITY_ARROW_SCALE = 8.0   # 速度箭头长度（基础分辨率像素 / 速度单位）


# ============================================================
# World generation
# ============================================================
def build_world(seed: int):
    """Run the full pipeline: elevation -> hydrology & erosion."""
    world, _ = generate_mountain_terrain(
        seed=seed, width=WORLD_WIDTH, height=WORLD_HEIGHT,
        num_points=80, num_macro_plates=8, lloyd_iterations=6,
        sea_level=SEA_LEVEL,
    )
    world, report = generate_hydrology_erosion(
        world, sea_level=world.sea_level,
        num_outlets=NUM_OUTLETS, max_particles=MAX_PARTICLES,
    )
    return world, report


# ============================================================
# Color schemes (all return float RGB in 0..1)
# ============================================================
def _elevation_colors(world: World) -> np.ndarray:
    """
    连续地形渐变设色（取代旧版 12 级区间）：
    水体为单一深蓝；陆地按自身起伏归一化后，
    在 绿→黄绿→棕→灰岩→雪白 锚点间分段线性插值。
    返回 (h, w, 3) float RGB（0..1）。
    """
    elev = world.elevation.astype(np.float64)
    land = world.land_mask
    h, w = elev.shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    rgb[~land] = _WATER_RGB
    if np.any(land):
        e_land = elev[land]
        lo, hi = float(e_land.min()), float(e_land.max())
        t = np.clip((elev - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        stops_t = np.array([s[0] for s in _TERRAIN_STOPS])
        stops_c = np.array([s[1] for s in _TERRAIN_STOPS])
        t_land = t[land]
        for ch in range(3):
            channel = np.interp(t_land, stops_t, stops_c[:, ch])
            rgb[..., ch][land] = channel
    return rgb


def _humidity_colors(hum: np.ndarray) -> np.ndarray:
    """Humidity 0-100 -> light blue-grey to deep blue."""
    t = np.clip(hum.astype(np.float64) / 100.0, 0.0, 1.0)[..., None]
    light = np.array([0.93, 0.95, 0.98])
    dark = np.array([0.02, 0.17, 0.45])
    return light + (dark - light) * t


def _hardness_colors(hard: np.ndarray) -> np.ndarray:
    """Hardness 0-255 -> grayscale."""
    v = np.clip(hard.astype(np.float64) / 255.0, 0.0, 1.0)[..., None]
    return np.concatenate([v, v, v], axis=-1)


# ============================================================
# Plates mode: micro plates (macro grouping + land/ocean tone)
# ============================================================
def _plate_colors(world: World) -> np.ndarray:
    """
    板块视图底色：
        色相按大板块分组（黄金角序列取色相，组内小板块色相轻微抖动），
        大陆板块鲜艳明亮，海洋板块灰暗低沉，
        每个小板块另有确定性的明度/饱和度微差以便区分相邻板块。
    返回 (h, w, 3) float RGB（0..1）。
    """
    h, w = world.shape
    micro = world.micro_plates
    n_micro = int(micro.max()) + 1 if micro.size else 0
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    if n_micro == 0:
        return rgb
    macro_of = world.micro_to_macro
    is_ocean = world.micro_plate_is_ocean
    for p in range(n_micro):
        m = int(macro_of[p]) if p < macro_of.shape[0] else p
        ocean = bool(is_ocean[p]) if p < is_ocean.shape[0] else False
        hue = (m * 0.6180339887498949
               + (((p * 0.7548776662466927) % 1.0) - 0.5) * 0.09) % 1.0
        frac = ((p * 2654435761) % 1024) / 1024.0
        if ocean:
            s, v = 0.30 + 0.15 * frac, 0.42 + 0.16 * frac
        else:
            s, v = 0.50 + 0.25 * frac, 0.72 + 0.22 * frac
        rgb[micro == p] = colorsys.hsv_to_rgb(hue, s, v)
    return rgb


def _plate_boundary_masks(world: World):
    """
    由板块图层即时计算边界掩膜（不受 plate_boundaries 山脊标记覆盖影响）：
    小板块边界细线（1 像素），大板块边界粗线（四邻域膨胀一次，约 3 像素）。
    返回 (micro_edge, thick_macro_edge) 两个 bool 掩膜。
    """
    micro = world.micro_plates
    macro = world.macro_plates
    micro_edge = np.zeros(micro.shape, dtype=bool)
    micro_edge[:, :-1] |= micro[:, :-1] != micro[:, 1:]
    micro_edge[:-1, :] |= micro[:-1, :] != micro[1:, :]
    macro_edge = np.zeros(macro.shape, dtype=bool)
    macro_edge[:, :-1] |= macro[:, :-1] != macro[:, 1:]
    macro_edge[:-1, :] |= macro[:-1, :] != macro[1:, :]
    thick = macro_edge.copy()
    thick[:, :-1] |= macro_edge[:, 1:]
    thick[:, 1:] |= macro_edge[:, :-1]
    thick[:-1, :] |= macro_edge[1:, :]
    thick[1:, :] |= macro_edge[:-1, :]
    return micro_edge, thick


def _plate_centroids(labels: np.ndarray, count: int):
    """每个标签区域的质心 (cy, cx)，bincount 实现。"""
    h, w = labels.shape
    flat = labels.ravel()
    idx = np.arange(flat.size)
    cnt = np.bincount(flat, minlength=count).astype(np.float64)
    cnt[cnt == 0] = 1.0
    cy = np.bincount(flat, weights=(idx // w).astype(np.float64), minlength=count) / cnt
    cx = np.bincount(flat, weights=(idx % w).astype(np.float64), minlength=count) / cnt
    return cy, cx


# ============================================================
# Pure renderer (unit-testable without a display)
# ============================================================
def render_map_rgb(
    world: World,
    color_mode: str,
    show_rivers: bool,
    show_deposition: bool,
    show_mountains: bool = True,
    show_outlets: bool = False,
) -> np.ndarray:
    """Compose the final map as a (h, w, 3) uint8 array."""
    if color_mode == "Elevation":
        base = _elevation_colors(world)
    elif color_mode == "Humidity":
        base = _humidity_colors(world.humidity)
    elif color_mode == "Plates":
        base = _plate_colors(world)
    else:  # "Rock Hardness"
        base = _hardness_colors(world.rock_hardness)

    rgb = base * 255.0

    if color_mode == "Plates":
        micro_edge, macro_edge = _plate_boundary_masks(world)
        rgb[micro_edge] *= 0.45          # 小板块边界：压暗的细线
        rgb[macro_edge] = (18, 18, 18)   # 大板块边界：近黑粗线

    if show_mountains:
        # 山脉 = DLA 梳齿纹样覆盖区（水文模块写入 mountain_mask 自定义
        # 图层；未经水文模块时回退 plate_boundaries == 4）
        mountain = world.get_layer("mountain_mask")
        if mountain is None:
            mountain = world.plate_boundaries == 4
        if np.any(mountain):
            rgb[mountain] = (rgb[mountain] * (1.0 - _MOUNTAIN_BLEND)
                             + _MOUNTAIN_RGB * _MOUNTAIN_BLEND)

    if show_deposition:
        depo = world.deposition_type
        for code, color in ((1, _FAN_RGB), (2, _DELTA_RGB)):
            mask = depo == code
            if np.any(mask):
                rgb[mask] = (rgb[mask] * (1.0 - _DEPOSITION_BLEND)
                             + color * _DEPOSITION_BLEND)

    if show_rivers:
        rgb[world.river_mask] = _RIVER_RGB

    if show_outlets:
        # 河口撒点调试：宽海岸带（可用撒点区）半透明品红，
        # 成功播撒的入海口实心亮品红（压在河流之上，便于核对）
        band = world.get_layer("outlet_band")
        if band is not None and np.any(band):
            rgb[band] = (rgb[band] * (1.0 - _OUTLET_BAND_BLEND)
                         + _OUTLET_BAND_RGB * _OUTLET_BAND_BLEND)
        outlet = world.get_layer("outlet_mask")
        if outlet is not None and np.any(outlet):
            rgb[outlet] = _OUTLET_RGB

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================
# tkinter application
# ============================================================
class WorldMapApp(tk.Tk):
    COLOR_MODES = ("Elevation", "Humidity", "Rock Hardness", "Plates")

    def __init__(self):
        super().__init__()
        self.title("World Map Viewer 2D")
        self.resizable(False, False)

        self.world = None
        self._photo = None
        self.color_mode = tk.StringVar(value="Elevation")
        self.show_rivers = tk.BooleanVar(value=True)
        self.show_deposition = tk.BooleanVar(value=True)
        self.show_mountains = tk.BooleanVar(value=True)
        self.show_outlets = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        self.after(100, self._on_generate)

    # ---------------- UI ----------------
    def _build_ui(self):
        map_size = WORLD_WIDTH * DISPLAY_SCALE
        self.canvas = tk.Canvas(self, width=map_size, height=map_size,
                                bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        panel = ttk.Frame(self, width=230)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        panel.pack_propagate(False)

        seed_frame = ttk.LabelFrame(panel, text="World Seed", padding=6)
        seed_frame.pack(fill=tk.X, pady=4)
        ttk.Entry(seed_frame, textvariable=self.seed_var).pack(fill=tk.X)
        self.gen_button = ttk.Button(seed_frame, text="Generate World",
                                     command=self._on_generate)
        self.gen_button.pack(fill=tk.X, pady=(6, 0))

        mode_frame = ttk.LabelFrame(panel, text="Color Layer", padding=6)
        mode_frame.pack(fill=tk.X, pady=4)
        for name in self.COLOR_MODES:
            ttk.Radiobutton(mode_frame, text=name, value=name,
                            variable=self.color_mode,
                            command=self._redraw).pack(anchor=tk.W)

        overlay_frame = ttk.LabelFrame(panel, text="Overlay Layers", padding=6)
        overlay_frame.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(overlay_frame, text="Rivers",
                        variable=self.show_rivers,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Deposition (fans / deltas)",
                        variable=self.show_deposition,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Mountains (DLA ridges)",
                        variable=self.show_mountains,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Outlet scatter (band + seeds)",
                        variable=self.show_outlets,
                        command=self._redraw).pack(anchor=tk.W)

        legend = ttk.LabelFrame(panel, text="Legend", padding=6)
        legend.pack(fill=tk.X, pady=4)
        ttk.Label(legend, text="Cyan   : rivers", foreground="#1a8fd1").pack(anchor=tk.W)
        ttk.Label(legend, text="Yellow : alluvial fans", foreground="#b8960f").pack(anchor=tk.W)
        ttk.Label(legend, text="Orange : deltas", foreground="#c55a11").pack(anchor=tk.W)
        ttk.Label(legend, text="Red    : mountains (ridges)", foreground="#c0392b").pack(anchor=tk.W)
        ttk.Label(legend, text="Pink   : outlet scatter band", foreground="#d030a0").pack(anchor=tk.W)
        ttk.Label(legend, text="Magenta: seeded outlets", foreground="#c000c0").pack(anchor=tk.W)
        ttk.Separator(legend, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(legend, text="Plates mode:").pack(anchor=tk.W)
        ttk.Label(legend, text="  hue group  = macro plate").pack(anchor=tk.W)
        ttk.Label(legend, text="  vivid / muted = land / ocean").pack(anchor=tk.W)
        ttk.Label(legend, text="  arrows = plate velocity").pack(anchor=tk.W)
        ttk.Label(legend, text="  Mx label = macro plate ID").pack(anchor=tk.W)

        status = ttk.Label(self, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- generation (worker thread) ----------------
    def _on_generate(self):
        text = self.seed_var.get().strip()
        try:
            seed = int(float(text))
        except ValueError:
            self.status_var.set(f"Invalid seed: {text!r}")
            return
        self.gen_button.config(state=tk.DISABLED)
        self.status_var.set(f"Generating world (seed={seed}) ... ~15 s")

        def work():
            try:
                result = build_world(seed)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._on_build_failed, exc)
                return
            self.after(0, self._on_build_done, seed, result)

        threading.Thread(target=work, daemon=True).start()

    def _on_build_failed(self, exc):
        self.gen_button.config(state=tk.NORMAL)
        self.status_var.set(f"Generation failed: {exc}")

    def _on_build_done(self, seed, result):
        self.world, report = result
        self.gen_button.config(state=tk.NORMAL)
        rivers = report.get("rivers", {})
        band = self.world.get_layer("outlet_band")
        outlet = self.world.get_layer("outlet_mask")
        n_band = int(band.sum()) if band is not None else 0
        n_outlet = int(outlet.sum()) if outlet is not None else 0
        self.status_var.set(
            f"seed={seed} | sea level {self.world.sea_level:.0f} m | "
            f"outlets {n_outlet}/{NUM_OUTLETS} (band {n_band}) | "
            f"river cells {rivers.get('river_cells', 0)} | "
            f"fans {report['river_erosion']['num_fans']} | "
            f"deltas {report['river_erosion']['num_deltas']}"
        )
        self._redraw()

    # ---------------- rendering ----------------
    def _redraw(self, *_args):
        if self.world is None:
            return
        rgb = render_map_rgb(
            self.world,
            color_mode=self.color_mode.get(),
            show_rivers=self.show_rivers.get(),
            show_deposition=self.show_deposition.get(),
            show_mountains=self.show_mountains.get(),
            show_outlets=self.show_outlets.get(),
        )
        img = Image.fromarray(rgb, mode="RGB")
        if DISPLAY_SCALE > 1:
            img = img.resize((img.width * DISPLAY_SCALE,
                              img.height * DISPLAY_SCALE), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        if self.color_mode.get() == "Plates":
            self._draw_plate_overlay()

    # ---------------- plates overlay (velocity arrows + macro labels) ----------------
    def _draw_plate_overlay(self):
        """
        板块视图画布覆盖层：
        每个小板块质心绘制总速度箭头（白描边 + 黑箭体，长度 ∝ 速度），
        每个大板块质心标注 M<id>。
        """
        world = self.world
        if world.micro_plate_velocity.shape[0] == 0:
            return
        s = DISPLAY_SCALE
        micro = world.micro_plates
        n_micro = int(micro.max()) + 1
        cy, cx = _plate_centroids(micro, n_micro)
        vel = world.total_micro_plate_velocity
        k = _VELOCITY_ARROW_SCALE * s
        for p in range(n_micro):
            vx, vy = float(vel[p, 0]), float(vel[p, 1])
            if vx * vx + vy * vy < 1e-6:
                continue
            x1, y1 = float(cx[p]) * s, float(cy[p]) * s
            x2, y2 = x1 + vx * k, y1 + vy * k
            self.canvas.create_line(x1, y1, x2, y2, fill="white", width=4,
                                    arrow=tk.LAST, arrowshape=(10, 12, 5))
            self.canvas.create_line(x1, y1, x2, y2, fill="#101010", width=2,
                                    arrow=tk.LAST, arrowshape=(10, 12, 5))
        if world.macro_plates.size and world.macro_plate_velocity.shape[0] > 0:
            n_macro = int(world.macro_plates.max()) + 1
            my, mx = _plate_centroids(world.macro_plates, n_macro)
            for m in range(n_macro):
                tx, ty = float(mx[m]) * s, float(my[m]) * s
                self.canvas.create_text(tx + 1, ty + 1, text=f"M{m}",
                                        fill="black",
                                        font=("TkDefaultFont", 11, "bold"))
                self.canvas.create_text(tx, ty, text=f"M{m}",
                                        fill="white",
                                        font=("TkDefaultFont", 11, "bold"))


# ============================================================
# Entry point
# ============================================================
def main():
    app = WorldMapApp()
    app.mainloop()


if __name__ == "__main__":
    main()