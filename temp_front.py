"""
world_visualizer_gl.py
World 三维地形可视化模块（基于 ModernGL + moderngl_window）

使用全局变量传递 World 对象和参数，支持垂直比例尺调整、
高度设色、河流流量颜色映射。

依赖安装：
    pip install moderngl moderngl_window[pyglet] numpy
"""

import numpy as np
import moderngl
import moderngl_window as mglw

# 全局变量，用于从 visualize_world 传递参数到 WorldViewer 实例
_VIEWER_WORLD = None
_VIEWER_Z_SCALE = 1.0

# 地形着色器
TERRAIN_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

TERRAIN_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

# 河流点着色器
POINT_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = 6.0;
    v_color = in_color;
}
"""

POINT_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


def elevation_to_color(elevation, sea_level):
    """将海拔映射到地形颜色（简单蓝-绿-棕-白渐变）"""
    e = np.clip(elevation, sea_level - 50, sea_level + 500)
    below = e < sea_level
    above = ~below
    color = np.zeros((e.size, 3), dtype=np.float32)
    if below.any():
        depth = (sea_level - e[below]) / 50.0
        color[below, 0] = 0.2 + 0.3 * (1 - depth)
        color[below, 1] = 0.4 + 0.4 * (1 - depth)
        color[below, 2] = 0.8 + 0.2 * (1 - depth)
    if above.any():
        height = (e[above] - sea_level) / 500.0
        color[above, 0] = 0.3 + 0.5 * height
        color[above, 1] = 0.5 + 0.4 * height
        color[above, 2] = 0.2 + 0.6 * height
    return color.reshape(-1, 3)


class WorldViewer(mglw.WindowConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = _VIEWER_WORLD
        self.z_scale = _VIEWER_Z_SCALE
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # 默认相机由 WindowConfig 提供（OrbitCamera），直接使用
        self.camera.target = (self.world.width / 2, self.world.height / 2, 0)
        self.camera.distance = max(self.world.width, self.world.height) * 1.5

        self.build_terrain()
        self.build_rivers()

        self.terrain_prog = self.ctx.program(
            vertex_shader=TERRAIN_VERT,
            fragment_shader=TERRAIN_FRAG,
        )
        self.point_prog = self.ctx.program(
            vertex_shader=POINT_VERT,
            fragment_shader=POINT_FRAG,
        )

        self.mvp_loc_terrain = self.terrain_prog["mvp"]
        self.mvp_loc_point = self.point_prog["mvp"]

    def build_terrain(self):
        H, W = self.world.elevation.shape
        elevation = self.world.elevation * self.z_scale
        x = np.arange(W, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        Z = elevation
        vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        colors = elevation_to_color(self.world.elevation, self.world.sea_level).astype(np.float32)

        indices = []
        idx_grid = np.arange(H * W, dtype=np.uint32).reshape(H, W)
        for i in range(H - 1):
            for j in range(W - 1):
                idx00 = idx_grid[i, j]
                idx01 = idx_grid[i, j + 1]
                idx10 = idx_grid[i + 1, j]
                idx11 = idx_grid[i + 1, j + 1]
                indices.extend([idx00, idx10, idx01, idx10, idx11, idx01])
        indices = np.array(indices, dtype=np.uint32)

        vbo_vertices = self.ctx.buffer(vertices.tobytes())
        vbo_colors = self.ctx.buffer(colors.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())

        self.terrain_vao = self.ctx.vertex_array(
            self.terrain_prog,
            [
                (vbo_vertices, '3f', 'in_position'),
                (vbo_colors, '3f', 'in_color'),
            ],
            index_buffer=ibo,
        )

    def build_rivers(self):
        if not hasattr(self.world, 'river_mask') or self.world.river_mask is None:
            self.river_vao = None
            return
        river_mask = self.world.river_mask
        discharge = self.world.river_discharge if self.world.river_discharge is not None else np.zeros_like(river_mask)
        rows, cols = np.nonzero(river_mask)
        if len(rows) == 0:
            self.river_vao = None
            return
        x = cols.astype(np.float32)
        y = rows.astype(np.float32)
        z = self.world.elevation[rows, cols] * self.z_scale + 0.5
        points = np.stack([x, y, z], axis=1).astype(np.float32)

        norm_discharge = (discharge - discharge.min()) / (discharge.max() - discharge.min() + 1e-8)
        r = 0.267 + 0.533 * norm_discharge
        g = 0.004 + 0.650 * norm_discharge
        b = 0.329 + 0.241 * norm_discharge
        colors = np.stack([r, g, b], axis=1).astype(np.float32)

        vbo_points = self.ctx.buffer(points.tobytes())
        vbo_colors = self.ctx.buffer(colors.tobytes())
        self.river_vao = self.ctx.vertex_array(
            self.point_prog,
            [
                (vbo_points, '3f', 'in_position'),
                (vbo_colors, '3f', 'in_color'),
            ],
        )

    def render(self, time, frametime):
        self.ctx.clear(0.9, 0.9, 0.9, 1.0)
        proj = self.camera.projection_matrix
        view = self.camera.view_matrix
        mvp = (proj @ view).astype('f4').tobytes()

        self.mvp_loc_terrain.write(mvp)
        self.terrain_vao.render()

        if self.river_vao is not None:
            self.mvp_loc_point.write(mvp)
            self.river_vao.render(moderngl.POINTS)

    def mouse_drag_event(self, x, y, dx, dy, button):
        self.camera.rotate(dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.camera.zoom(y_offset)


def visualize_world(world, z_scale=1.0, window_size=(1280, 800)):
    """启动三维可视化窗口。"""
    global _VIEWER_WORLD, _VIEWER_Z_SCALE
    _VIEWER_WORLD = world
    _VIEWER_Z_SCALE = z_scale
    # 设置窗口大小
    WorldViewer.window_size = window_size
    WorldViewer.run()


if __name__ == "__main__":
    from world_core import World
    import numpy as np

    w = World(seed=42, width=100, height=100)
    x = np.linspace(-3, 3, w.width)
    y = np.linspace(-3, 3, w.height)
    X, Y = np.meshgrid(x, y)
    w._elevation = (np.sin(X) * np.cos(Y) * 100 + 50).astype(np.float32)
    w.sea_level = 50.0
    w._land_mask = w.elevation > w.sea_level

    w._river_mask = np.zeros((w.height, w.width), dtype=bool)
    for i in range(min(w.height, w.width)):
        w._river_mask[i, i] = True
    w._river_discharge = np.linspace(1, 10, w.width * w.height).reshape(w.height, w.width)

    visualize_world(w, z_scale=0.3)