"""
world_viewer.py
World Viewer 3D (tkinter + matplotlib continuous surface)

A 3D viewer for the generated world:
    - Continuous elevation surface (plot_surface), NOT discrete bars;
    - Color layers (radio buttons):
        * Elevation     - contour-band hypsometric tinting (等高线分层设色)
                          with hillshade relief shading
        * Humidity      - whole surface shaded in blue by humidity
        * Rock Hardness - grayscale
      (all color modes are multiplied by the same hillshade for relief)
    - Overlay layers (check buttons):
        * Rivers      - polylines draped on the surface
        * Deposition  - alluvial fan / delta markers
    - View rotation is locked to the vertical axis (fixed tilt angle);
      scroll zoom still works.
    - Single seed entry + "Generate" button (no console input at all).

All UI text is English because the target matplotlib environment
has no CJK font configured.

Rendering resolution: the surface is block-downsampled to about
SURF_RES x SURF_RES facets for interactivity; rivers and deposition
markers are still mapped from full-resolution world coordinates.
"""

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from world_core import World
from elevation_generator import generate_mountain_terrain
from hydrology_erosion import generate_hydrology_erosion


# ============================================================
# Tunable constants (not exposed in the GUI)
# ============================================================
WORLD_WIDTH = 256
WORLD_HEIGHT = 256
SURF_RES = 128              # max facets per side on the 3D surface
SEA_LEVEL = 0.0             # 固定海平面（板块速度碰撞版的海陆由板块划分物理决定，
                          # 不再用海拔分位数反推——双峰海拔分布下分位数会失效）
NUM_OUTLETS = 18
MAX_PARTICLES = 8000
DEFAULT_SEED = 42

# View: rotation locked to the vertical axis (only azimuth changes)
FIXED_VIEW_ELEV = 55.0        # 固定俯仰角（度）
FIXED_VIEW_AZIM = -60.0       # 初始方位角（度）

# Contour-band hypsometric tinting (等高线分层设色)
N_CONTOUR_BANDS = 14          # 陆地/海洋各自的等高带数量
_CONTOUR_LINE_DARKEN = 0.45   # 带边界（等高线）压暗系数
_SHADE_FLOOR = 0.55           # 阴影最低亮度（0=全黑阴影，1=无阴影）

# Terrain gradient anchor colors for land bands (low -> high)
_TERRAIN_STOPS = [
    (0.00, (0.28, 0.60, 0.28)),   # lowland green
    (0.35, (0.62, 0.78, 0.38)),   # khaki green
    (0.60, (0.55, 0.42, 0.25)),   # brown
    (0.80, (0.62, 0.60, 0.58)),   # grey rock
    (1.00, (0.97, 0.97, 0.97)),   # snow white
]
# Water bands: shoreline -> deep
_WATER_SHALLOW = (0.45, 0.65, 0.85)
_WATER_DEEP = (0.05, 0.15, 0.45)

_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ============================================================
# World generation
# ============================================================
def build_world(seed: int):
    """Run the full pipeline: elevation -> hydrology & erosion."""
    world, _ = generate_mountain_terrain(
        seed=seed, width=WORLD_WIDTH, height=WORLD_HEIGHT,
        num_points=50, num_macro_plates=6, lloyd_iterations=2,
        sea_level=SEA_LEVEL,
    )
    world, report = generate_hydrology_erosion(
        world, sea_level=world.sea_level,
        num_outlets=NUM_OUTLETS, max_particles=MAX_PARTICLES,
    )
    return world, report


# ============================================================
# View preprocessing: downsampling + vertical scaling
# ============================================================
def _block_reduce(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    nh, nw = h // factor, w // factor
    cropped = arr[:nh * factor, :nw * factor].astype(np.float64)
    return cropped.reshape(nh, factor, nw, factor).mean(axis=(1, 3))


class PreparedView:
    """Pre-compute everything the 3D renderer needs from a World."""

    def __init__(self, world: World):
        self.world = world
        f = max(1, -(-max(world.width, world.height) // SURF_RES))  # ceil
        self.factor = f

        self.elev = _block_reduce(world.elevation, f)
        self.hum = _block_reduce(world.humidity, f)
        self.hard = _block_reduce(world.rock_hardness.astype(np.float32), f)
        self.sea_level = world.sea_level

        self.z0 = float(self.elev.min())
        z_span = max(float(self.elev.max()) - self.z0, 1e-6)
        nh, nw = self.elev.shape
        # vertical exaggeration: tallest peak ~ 45% of map span
        self.z_scale = 0.45 * max(nh, nw) / z_span
        self.surf_z = (self.elev - self.z0) * self.z_scale
        self.z_top = float(self.surf_z.max())

        # hillshade（山体阴影）：固定光源西北 315°、仰角 45°，
        # 0~1 亮度场，渲染时统一乘到各色彩模式上
        ls = LightSource(azdeg=315, altdeg=45)
        self.shade = ls.hillshade(self.surf_z, vert_exag=1.0, dx=1.0, dy=1.0)

    def surface_point(self, y: float, x: float, lift: float = 0.6):
        """Full-res pixel coords -> surface 3D coords (slightly lifted)."""
        nh, nw = self.elev.shape
        ry = min(int(y / self.factor), nh - 1)
        rx = min(int(x / self.factor), nw - 1)
        z = self.surf_z[ry, rx] + lift
        return (rx, ry, z)


# ============================================================
# Color schemes
# ============================================================
def _elevation_contour_colors(elev: np.ndarray, sea_level: float) -> np.ndarray:
    """
    等高线分层设色（取代旧版 12 级区间表）：
    以海平面为基准，陆地 0..最高 分 N_CONTOUR_BANDS 个等高带，
    带内取 绿→黄绿→棕→灰岩→雪白 锚点渐变色；水体按深度带在
    浅蓝→深蓝间取色。相邻带索引不同的像素（带边界，含海岸线）
    压暗形成等高线。返回 (h, w, 3) float RGB（0..1）。
    """
    h, w = elev.shape
    rel = elev - sea_level
    land = rel > 0
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    band_l = np.zeros((h, w), dtype=np.int32)
    band_w = np.zeros((h, w), dtype=np.int32)
    n = N_CONTOUR_BANDS

    if np.any(land):
        step = max(float(rel[land].max()) / n, 1e-6)
        band_l = np.clip((rel / step).astype(np.int32), 0, n - 1)
        t_land = (band_l[land] + 0.5) / n
        stops_t = np.array([s[0] for s in _TERRAIN_STOPS])
        stops_c = np.array([s[1] for s in _TERRAIN_STOPS])
        for ch in range(3):
            rgb[..., ch][land] = np.interp(t_land, stops_t, stops_c[:, ch])

    if np.any(~land):
        step_w = max(float(-rel[~land].min()) / n, 1e-6)
        band_w = np.clip(((-rel) / step_w).astype(np.int32), 0, n - 1)
        t_w = (band_w[~land] + 0.5) / n  # 0 = 海面, 1 = 最深
        for ch in range(3):
            rgb[..., ch][~land] = (_WATER_SHALLOW[ch]
                                   + (_WATER_DEEP[ch] - _WATER_SHALLOW[ch]) * t_w)

    # 全局带索引（水体取负避免与陆地带撞号），边界即等高线
    band_all = np.where(land, band_l, -band_w - 1)
    contour = np.zeros((h, w), dtype=bool)
    contour[:, :-1] |= band_all[:, :-1] != band_all[:, 1:]
    contour[:-1, :] |= band_all[:-1, :] != band_all[1:, :]
    rgb[contour] *= _CONTOUR_LINE_DARKEN
    return rgb


def _humidity_colors(hum: np.ndarray) -> np.ndarray:
    """Humidity 0-100 -> light blue-grey to deep blue."""
    t = np.clip(hum / 100.0, 0.0, 1.0)[..., None]
    light = np.array([0.93, 0.95, 0.98])
    dark = np.array([0.02, 0.17, 0.45])
    return light + (dark - light) * t


def _hardness_colors(hard: np.ndarray) -> np.ndarray:
    """Hardness 0-255 -> grayscale."""
    v = np.clip(hard / 255.0, 0.0, 1.0)[..., None]
    return np.concatenate([v, v, v], axis=-1)


# ============================================================
# Overlays: river polylines / deposition markers
# ============================================================
def _river_segments(world: World, view: PreparedView):
    """
    Rebuild river polylines: each river cell connects to the
    8-neighbour with higher river_strength (strength grows
    downstream toward the outlet).
    """
    rm = world.river_mask
    st = world.river_strength
    h, w = rm.shape
    segs = []
    ys, xs = np.nonzero(rm)
    for y, x in zip(ys, xs):
        best = None
        best_s = st[y, x]
        for dy, dx in _D8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and rm[ny, nx] and st[ny, nx] > best_s:
                best = (ny, nx)
                best_s = st[ny, nx]
        if best is not None:
            segs.append([
                view.surface_point(y, x),
                view.surface_point(best[0], best[1]),
            ])
    return segs


def _deposition_points(world: World, view: PreparedView, max_points: int = 1500):
    """Return (fan_points, delta_points) as Nx3 arrays."""
    out = []
    for code in (1, 2):
        cells = np.argwhere(world.deposition_type == code)
        if len(cells) == 0:
            out.append(np.zeros((0, 3)))
            continue
        step = max(1, len(cells) // max_points)
        pts = np.array([view.surface_point(y, x) for y, x in cells[::step]])
        out.append(pts)
    return out


# ============================================================
# Pure renderer (backend-independent, unit-testable)
# ============================================================
def render_scene(ax3d, world: World, view: PreparedView,
                 color_mode: str, show_rivers: bool, show_deposition: bool):
    """Draw the whole 3D scene onto ax3d."""
    # 保留用户当前的方位角（绕纵轴的旋转角），切换图层重绘时不丢失；
    # 俯仰角始终锁定为 FIXED_VIEW_ELEV
    prev_azim = getattr(ax3d, "azim", None)
    if prev_azim is None:
        prev_azim = FIXED_VIEW_AZIM
    ax3d.cla()
    ax3d.set_axis_off()

    nh, nw = view.elev.shape
    xv, yv = np.meshgrid(np.arange(nw), np.arange(nh))

    if color_mode == "Elevation":
        colors = _elevation_contour_colors(view.elev, view.sea_level)
    elif color_mode == "Humidity":
        colors = _humidity_colors(view.hum)
    else:  # "Rock Hardness"
        colors = _hardness_colors(view.hard)

    # 统一乘山体阴影（hillshade）：亮度在 _SHADE_FLOOR~1.0 之间
    colors = colors * (_SHADE_FLOOR + (1.0 - _SHADE_FLOOR) * view.shade[..., None])

    alpha = np.ones((*colors.shape[:2], 1))
    facecolors = np.concatenate([colors, alpha], axis=-1)[:-1, :-1, :]

    ax3d.plot_surface(
        xv, yv, view.surf_z,
        facecolors=facecolors, shade=False,
        rstride=1, cstride=1, antialiased=False,
    )

    if show_rivers:
        segs = _river_segments(world, view)
        if segs:
            lc = Line3DCollection(segs, colors=[(0.2, 0.75, 1.0, 1.0)], linewidths=1.2)
            ax3d.add_collection3d(lc)

    if show_deposition:
        fans, deltas = _deposition_points(world, view)
        if len(fans):
            ax3d.scatter(fans[:, 0], fans[:, 1], fans[:, 2],
                         s=5, c="#ffd54a", depthshade=False)
        if len(deltas):
            ax3d.scatter(deltas[:, 0], deltas[:, 1], deltas[:, 2],
                         s=9, c="#ff7043", depthshade=False)

    ax3d.set_xlim(0, nw - 1)
    ax3d.set_ylim(0, nh - 1)
    ax3d.set_zlim(0, view.z_top * 1.05)
    ax3d.set_box_aspect((nw, nh, view.z_top * 1.05))
    ax3d.view_init(elev=FIXED_VIEW_ELEV, azim=prev_azim)


# ============================================================
# tkinter application
# ============================================================
class WorldViewerApp(tk.Tk):
    COLOR_MODES = ("Elevation", "Humidity", "Rock Hardness")

    def __init__(self):
        super().__init__()
        self.title("World Viewer 3D")
        self.geometry("1280x860")

        self.world = None
        self.view = None
        self.color_mode = tk.StringVar(value="Elevation")
        self.show_rivers = tk.BooleanVar(value=True)
        self.show_deposition = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        # generate the default world right after the window appears
        self.after(100, self._on_generate)

    # ---------------- UI ----------------
    def _build_ui(self):
        panel = ttk.Frame(self, width=230)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        panel.pack_propagate(False)

        # seed
        seed_frame = ttk.LabelFrame(panel, text="World Seed", padding=6)
        seed_frame.pack(fill=tk.X, pady=4)
        ttk.Entry(seed_frame, textvariable=self.seed_var).pack(fill=tk.X)
        self.gen_button = ttk.Button(seed_frame, text="Generate World",
                                     command=self._on_generate)
        self.gen_button.pack(fill=tk.X, pady=(6, 0))

        # color layers
        mode_frame = ttk.LabelFrame(panel, text="Color Layer", padding=6)
        mode_frame.pack(fill=tk.X, pady=4)
        for name in self.COLOR_MODES:
            ttk.Radiobutton(mode_frame, text=name, value=name,
                            variable=self.color_mode,
                            command=self._redraw).pack(anchor=tk.W)

        # overlay layers
        overlay_frame = ttk.LabelFrame(panel, text="Overlay Layers", padding=6)
        overlay_frame.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(overlay_frame, text="Rivers",
                        variable=self.show_rivers,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Deposition (fans / deltas)",
                        variable=self.show_deposition,
                        command=self._redraw).pack(anchor=tk.W)

        # legend
        legend = ttk.LabelFrame(panel, text="Legend", padding=6)
        legend.pack(fill=tk.X, pady=4)
        ttk.Label(legend, text="Cyan lines  : rivers", foreground="#1a8fd1").pack(anchor=tk.W)
        ttk.Label(legend, text="Yellow dots : alluvial fans", foreground="#b8960f").pack(anchor=tk.W)
        ttk.Label(legend, text="Orange dots : deltas", foreground="#c55a11").pack(anchor=tk.W)

        # 3D canvas
        self.fig = plt.figure(figsize=(10, 8), facecolor="#202124")
        self.ax3d = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 旋转锁定：拖动时强制俯仰角回到固定值，只允许绕纵轴（方位角）旋转
        self.canvas.mpl_connect("motion_notify_event", self._lock_view_elev)

        # status bar
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
        self.view = PreparedView(self.world)
        self.gen_button.config(state=tk.NORMAL)
        rivers = report.get("rivers", {})
        self.status_var.set(
            f"seed={seed} | sea level {self.world.sea_level:.0f} m | "
            f"river cells {rivers.get('river_cells', 0)} | "
            f"fans {report['river_erosion']['num_fans']} | "
            f"deltas {report['river_erosion']['num_deltas']}"
        )
        self._redraw()

    # ---------------- rendering ----------------
    def _lock_view_elev(self, event):
        """鼠标拖动旋转时锁定俯仰角：只允许绕纵轴（方位角）旋转。"""
        if event.button == 1 and event.inaxes is self.ax3d:
            if abs(self.ax3d.elev - FIXED_VIEW_ELEV) > 1e-6:
                self.ax3d.elev = FIXED_VIEW_ELEV
                self.canvas.draw_idle()

    def _redraw(self, *_args):
        if self.view is None:
            return
        render_scene(
            self.ax3d, self.world, self.view,
            color_mode=self.color_mode.get(),
            show_rivers=self.show_rivers.get(),
            show_deposition=self.show_deposition.get(),
        )
        self.canvas.draw_idle()


# ============================================================
# Entry point
# ============================================================
def main():
    app = WorldViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()