"""
param_input.py
架空世界生成器 - 参数输入模块（tkinter GUI）

以滑条 + 数字输入框（双向联动）的形式收集三大模块的用户参数：
    1. 山脉地形生成  generate_mountain_terrain
    2. 水文与侵蚀    generate_hydrology_erosion
    3. 气候分类      generate_climate

用法（阻塞式调用）：
    from param_input import collect_params
    params = collect_params()
    # params == {"mountain": {...}, "hydrology": {...}, "climate": {...}} 或 None（用户取消）
    # 字典键名与各主函数签名一致，可直接解包调用：
    world, report = generate_mountain_terrain(**params["mountain"])
    world, report = generate_hydrology_erosion(world, **params["hydrology"])
    generate_climate(world, config=params["climate"])

也支持非阻塞式嵌入：
    app = ParamInputApp(master, on_confirm=callback)
    callback(params) 在用户点击“确认”时被调用。

约定：
    - Optional 参数带“自动(None)”勾选框，勾选后控件禁用、返回 None；
    - 布尔参数用勾选框；枚举参数用下拉框；
    - “保存预设 / 载入预设”按钮读写 JSON 文件，便于复用整套参数。
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================
# 参数声明表
#   每行: (键名, 中文标签, 类型, 最小, 最大, 默认值, 说明)
#   类型: "int" / "float" / "bool" / "choice" / "oint" / "ofloat"
#         (o 开头 = Optional，带“自动(None)”勾选)
#   choice 类型时 最小/最大 位置放 (选项元组,)
# ============================================================

MOUNTAIN_GROUPS = [
    ("基础与画布", [
        ("seed", "世界种子", "oint", 0, 999999999, None, "None=随机；world 复用时忽略"),
        ("width", "地图宽(格)", "int", 128, 2048, 512, ""),
        ("height", "地图高(格)", "int", 128, 2048, 512, ""),
        ("sea_level", "海平面(米)", "float", -100, 100, 0.0, "elevation>海平面为陆地"),
    ]),
    ("板块划分", [
        ("num_points", "小板块点数", "int", 30, 200, 80, "↑板块更碎"),
        ("num_macro_plates", "大板块数", "int", 4, 16, 8, "↑山系更多"),
        ("lloyd_iterations", "小板块Lloyd松弛", "int", 0, 10, 4, "0=不规则，↑细胞更均匀"),
        ("macro_lloyd_iterations", "大板块聚类松弛", "int", 0, 6, 2, "↑大板块边界更平滑"),
        ("ocean_expansion_rounds", "海洋扩张轮数", "int", 0, 5, 4, "0=仅边缘海洋"),
        ("ocean_expansion_prob", "海洋扩张概率", "float", 0, 1, 0.2, "每轮邻洋大陆转海洋概率"),
        ("continent_base", "大陆基准海拔(米)", "float", 0, 100, 5.0, ""),
        ("ocean_depth", "海洋基准沉降(米)", "float", 0, 200, 50.0, "应大于山脊抬升量级"),
        ("domain_transition_sigma", "海陆过渡平滑σ(格)", "float", 0, 15, 4.0, "↑海岸过渡更宽"),
    ]),
    ("板块速度", [
        ("micro_speed_mean", "小板块速度均值", "float", 0, 5, 1.0, ""),
        ("micro_speed_std", "小板块速度标准差", "float", 0, 1, 0.2, ""),
        ("macro_speed_mean", "大板块速度均值", "float", 0, 5, 2.0, "应大于小板块"),
        ("macro_speed_std", "大板块速度标准差", "float", 0, 1, 0.2, ""),
    ]),
    ("碰撞→山脊", [
        ("collision_threshold", "相撞速度阈值", "float", 0.5, 3, 0.2, "↓山脉更多"),
        ("min_ridge_height", "山脊基准高度(米)", "float", 0, 200, 40.0, ""),
        ("speed_height_scale", "对数高度系数(米)", "float", 0, 300, 100.0, "基准+系数×ln(1+超出)"),
        ("max_ridge_height", "山脊高度封顶(米)", "float", 0, 5000, 2000.0, ""),
        ("ridge_end_taper", "山脊末端渐隐(格)", "float", 0, 80, 10.0, "↑收尾更缓"),
        ("ridge_junction_blend", "交汇山脊过渡(格)", "float", 0, 60, 10.0, ""),
        ("short_ridge_shrink_length", "短山脊缩减参考长(格)", "float", 0, 120, 60.0, "0=不缩减"),
        ("junction_chamfer_dist", "尖角倒角截距(格)", "float", 0, 40, 10.0, "0=不倒角"),
        ("junction_chamfer_angle", "倒角夹角阈值(度)", "float", 60, 179, 120.0, "小于它才倒角"),
        ("junction_chamfer_min_length", "倒角最短线长(格)", "float", 0, 80, 20.0, ""),
        ("coast_chamfer_dist", "海岸倒角截距(格)", "float", 0, 40, 10.0, "0=不倒角"),
        ("coast_chamfer_angle", "海岸倒角夹角阈值(度)", "float", 60, 179, 120.0, ""),
        ("coast_chamfer_min_length", "海岸倒角最短边长(格)", "float", 0, 80, 20.0, ""),
    ]),
    ("裂谷", [
        ("divergence_threshold", "离散速度阈值", "float", 0.5, 3, 1.5, "↓裂谷更多"),
        ("min_rift_depth", "裂谷基准深度(米)", "float", 0, 100, 20.0, ""),
        ("rift_depth_scale", "对数深度系数(米)", "float", 0, 100, 25.0, ""),
        ("max_rift_depth", "裂谷深度封顶(米)", "float", 0, 200, 80.0, "应明显小于山脉抬升"),
        ("rift_end_taper", "裂谷末端渐隐(格)", "float", 0, 80, 10.0, ""),
        ("rift_junction_blend", "交汇裂谷过渡(格)", "float", 0, 60, 10.0, ""),
        ("rift_basin_half_width", "裂谷盆底最大半宽(格)", "float", 0, 30, 2.0, "0=保持等宽沟壑"),
    ]),
    ("山脊曲线与形态", [
        ("amplitude", "山脊曲线摆动幅度(格)", "float", 0, 40, 5.0, "↑边缘更弯"),
        ("frequency", "一维噪声频率", "float", 1, 10, 3.0, "↑变化更密"),
        ("octaves", "一维噪声分频数", "int", 1, 6, 3, ""),
        ("lacunarity", "一维噪声间隙度", "float", 1.5, 3, 2.0, ""),
        ("ridge_height_noise_amp", "高度噪声相对振幅", "float", 0, 0.5, 0.25, "0=一样高"),
        ("min_edge_length", "造山边缘最小长度(格)", "float", 0, 50, 10.0, ""),
        ("ridge_influence", "原初山脊抬升半径(格)", "float", 4, 30, 12.0, "↑山更宽"),
        ("tooth_influence", "梳齿抬升半径(格)", "float", 2, 15, 7.0, "宜小于原初"),
        ("tooth_decay", "梳齿每格衰减(米)", "float", 1, 20, 6.0, "↑齿更矮"),
        ("tooth_max_length", "山脊梳齿最大长度(格)", "float", 5, 60, 20.0, ""),
        ("rift_tooth_max_length", "裂谷梳齿最大长度(格)", "float", 3, 40, 10.0, "宜小于山脊梳齿"),
    ]),
    ("次生山脊", [
        ("secondary_ridge_threshold", "次生山脊速度阈值", "ofloat", 0.5, 5, 1.0, "None=关闭"),
        ("secondary_ridge_offset", "次生山脊平移距离(格)", "float", 8, 30, 15.0, ""),
        ("secondary_ridge_end_shrink", "两端缩短弧长(格)", "float", 0, 40, 3.0, ""),
        ("secondary_ridge_height_scale", "次生山脊高度倍率", "float", 0.2, 0.7, 0.45, "低矮第二皱褶"),
    ]),
    ("高原", [
        ("plateau_prob", "碰撞高原概率", "float", 0, 1, 0.5, ""),
        ("plateau_collision_speed", "巨大碰撞速度阈值", "float", 0.5, 5, 2.0, ""),
        ("plateau_base_height", "碰撞高原基准提升(米)", "float", 0, 200, 40.0, ""),
        ("plateau_uplift_scale", "对数提升系数(米)", "float", 0, 100, 20.0, ""),
        ("plateau_max_height", "碰撞高原封顶(米)", "float", 0, 1000, 400.0, ""),
        ("shield_plateau_prob", "地盾高原概率", "float", 0, 1, 0.15, ""),
        ("shield_plateau_height", "地盾高原提升(米)", "float", 0, 200, 60.0, ""),
        ("plateau_edge_sigma", "高原边缘平滑σ(格)", "float", 0, 15, 4.0, "≤0=硬边"),
    ]),
    ("侧滑分段", [
        ("slip_threshold", "侧滑速度阈值", "float", 0, 10, 5.0, "超过则替换为平行短线"),
        ("slip_angle_slope", "角度斜率(度/速度)", "float", 0, 60, 20.0, ""),
        ("slip_angle_offset", "角度截距(度)", "float", 0, 90, 10.0, ""),
        ("slip_length_scale", "长度对数缩放(格)", "float", 0, 50, 15.0, ""),
        ("slip_length_offset", "长度偏移(格)", "float", 0, 50, 5.0, ""),
        ("slip_length_max", "短线最大长度(格)", "float", 0, 200, 80.0, ""),
    ]),
    ("背景起伏与海陆", [
        ("bg_amp", "背景起伏振幅(米)", "float", 0, 100, 50.0, "0=纯板块+山脊地形"),
        ("bg_freq", "背景噪声频率", "float", 1, 10, 4.0, ""),
        ("bg_octaves", "背景噪声分频数", "int", 1, 6, 4, ""),
        ("bg_lacunarity", "背景噪声间隙度", "float", 1.5, 3, 2.0, ""),
        ("coastal_ridge_offset", "海岸山脊向陆位移(格)", "float", 0, 15, 5.0, "0=不位移"),
    ]),
]

HYDROLOGY_GROUPS = [
    ("基础与岩石硬度", [
        ("sea_level", "海平面覆盖(米)", "ofloat", -100, 100, None, "None=沿用 world.sea_level"),
        ("rock_freq", "岩石硬度噪声频率", "float", 1, 50, 12.0, "高频→图样细碎"),
        ("rock_octaves", "岩石硬度分频数", "int", 1, 8, 4, ""),
        ("rock_lacunarity", "岩石硬度间隙度", "float", 1.5, 3, 2.0, ""),
        ("rock_hardness_max", "硬度上限", "int", 0, 255, 255, "硬度范围0~255"),
        ("mountain_hardness_boost", "造山加硬峰值", "float", 0, 300, 150.0, ""),
        ("mountain_boost_sigma", "造山加硬半径σ", "float", 1, 20, 5.0, ""),
    ]),
    ("气候带边界", [
        ("belt_edge_zone_width", "带边界扰动范围(格)", "float", 0, 30, 8.0, "≤0=直线边界"),
        ("belt_edge_noise_prob", "带边界噪点概率", "float", 0, 1, 0.20, "越大越曲折"),
        ("belt_edge_iterations", "带边界平滑迭代", "int", 0, 10, 3, "越大越圆润"),
    ]),
    ("湿度：海岸增益", [
        ("coastal_humidity_width", "海岸增湿带宽度(格)", "float", 0, 30, 8.0, ""),
        ("coastal_ca_noise_prob", "海岸边缘噪点概率", "float", 0, 1, 0.18, ""),
        ("coastal_ca_noise_range", "噪点外扩范围(格)", "float", 0, 20, 5.0, ""),
        ("coastal_ca_iterations", "海岸边缘平滑迭代", "int", 0, 10, 3, ""),
    ]),
    ("湿度：雨影效应", [
        ("mountain_min_elev", "山体最低海拔(米)", "float", 0, 1000, 100.0, "计入雨影的山体"),
        ("mountain_effect_radius", "山体影响半径(格)", "float", 0, 50, 10.0, ""),
        ("mountain_sea_boost", "靠海侧湿度增益(级)", "float", 0, 3, 1.0, ""),
        ("mountain_inland_boost", "靠内陆侧湿度增益(级)", "float", 0, 3, 0.3, "应小于靠海侧"),
    ]),
    ("湿度：西风带", [
        ("westerly_reach_frac", "西风东西向半轴(图宽比)", "float", 0, 1.5, 0.6, ""),
        ("westerly_source_lat", "西风源点纬度(°N)", "float", 0, 70, 45.0, "屏幕左缘"),
        ("westerly_lat_ratio", "北向半轴之比", "float", 0, 1, 0.5, "越小北向衰减越快"),
        ("westerly_south_ratio", "南向半轴之比", "float", 0, 1, 0.8, ""),
        ("westerly_boost", "西风最大额外湿度(级)", "float", 0, 3, 2.0, ""),
        ("westerly_falloff", "西风衰减指数", "float", 0.1, 3, 0.4, "<1更强，调强度主力"),
        ("mediterranean_westerly_boost", "地中海小西风湿度(级)", "float", 0, 3, 0.4, ""),
        ("westerlies_min_level", "西风带最低湿度等级", "int", 0, 3, 1, "1=半干旱"),
    ]),
    ("湿度：风场截止线扰动", [
        ("wind_edge_zone_width", "截止线扰动范围(格)", "float", 0, 30, 8.0, "≤0=直线截止"),
        ("wind_edge_noise_prob", "截止线噪点概率", "float", 0, 1, 0.20, ""),
        ("wind_edge_iterations", "截止线平滑迭代", "int", 0, 10, 3, ""),
    ]),
    ("湿度：季风", [
        ("monsoon_reach_frac", "季风影响范围(对角线比)", "float", 0, 1.5, 0.4, ""),
        ("monsoon_boost", "季风最大湿度(级)", "float", 0, 3, 3.0, "季风区内替换基底"),
        ("monsoon_falloff", "季风衰减指数", "float", 0.1, 3, 0.3, "<1更强，调强度主力"),
        ("monsoon_min_level", "季风区最低湿度等级", "int", 0, 3, 1, "1=半干旱"),
        ("humidity_transition_band", "湿度过渡带宽度(格)", "float", 0, 20, 4.0, "相邻等级差≤1"),
    ]),
    ("河口与河流算法", [
        ("num_outlets", "河口数量", "int", 0, 300, 60, "随机撒点模式的目标数"),
        ("outlet_coast_spacing", "河口沿岸间距(格)", "ofloat", 0, 50, None, "None/0=直接随机撒点"),
        ("outlet_coast_band", "宽海岸带宽度(格)", "float", 0, 10, 3.0, "≤0=仅1格宽海岸撒点"),
        ("river_algorithm", "河流算法", "choice", ("sca", "dla"), None, "sca", "sca=叶脉状(默认)/dla=灌木状"),
    ]),
    ("SCA 河流", [
        ("sca_num_attraction", "吸引点数量", "int", 100, 20000, 5000, "512²建议3000~6000"),
        ("sca_d_i", "影响半径(格)", "float", 1, 50, 18.0, "大→顺直，小→蜿蜒"),
        ("sca_d_k", "删除半径(格)", "float", 0.5, 10, 2.0, "大→支流稀疏"),
        ("sca_D", "生长步长(格)", "float", 0.5, 5, 1.5, "建议1~3"),
        ("sca_L_min", "修剪阈值(格)", "float", 0, 50, 10.0, "剪除短支流"),
        ("sca_max_step_drop", "单步最大海拔下降(米)", "float", 0, 20, 5.0, "0=严格单调"),
        ("sca_min_channel_spacing", "河道最小间距(格)", "float", 0, 10, 2.0, "建议≥删除半径"),
        ("sca_parallel_angle", "平行判定夹角(度)", "float", 0, 90, 10.0, "0≈关闭间距控制"),
        ("sca_spacing_exempt", "间距豁免祖先代数", "oint", 0, 50, None, "None=自动"),
        ("sca_mountain_max_distance", "山脉距离限制(格)", "float", 0, 20, 2.0, "建议2~8，0=仅中线"),
        ("sca_mountain_buffer", "山脉禁入缓冲(格)", "int", 0, 10, 1, "河流止于山脉边缘"),
        ("sca_coast_avoid", "海岸规避距离(格)", "float", 0, 20, 6.0, "0=关闭"),
        ("sca_arid_attraction", "干旱区吸引点保留比例", "float", 0, 1, 0.2, "河网密度∝湿度"),
        ("sca_semi_arid_attraction", "半干旱区吸引点保留比例", "float", 0, 1, 0.3, ""),
        ("sca_rift_attraction", "裂谷加撒吸引点数", "int", 0, 2000, 200, "0=关闭"),
        ("sca_max_iterations", "迭代上限", "oint", 100, 10000, None, "None=自动(性能兜底)"),
        ("sca_node_retries", "节点受阻重试轮数", "int", 1, 20, 5, ""),
        ("sca_young_influence", "幼龄豁免半径系数", "float", 0, 5, 2.0, "≤1关闭"),
        ("sca_young_steps", "幼龄豁免步数窗口", "int", 0, 30, 8, "≤0关闭"),
    ]),
    ("DLA 河流", [
        ("spawn_radius", "投放池半径(格)", "float", 5, 100, 25.0, "勿超√max_walk_steps"),
        ("spawn_elevation_bias", "投放高程偏好指数", "float", 0, 5, 2.0, ""),
        ("walk_elevation_bias", "游走高程启发强度", "float", 0, 5, 1.0, ""),
        ("max_river_neighbors", "河流邻居上限", "int", 1, 8, 3, "达到则永久禁入"),
        ("max_particles", "粒子总数", "int", 1000, 100000, 20000, "性能兜底"),
        ("max_walk_steps", "单粒子步数上限", "int", 50, 2000, 400, ""),
        ("coastal_buffer", "海岸缓冲(格)", "int", 0, 10, 0, "0=关闭"),
        ("pool_rebuild_interval", "投放池重建间隔", "int", 32, 2048, 256, ""),
    ]),
    ("河流水流量", [
        ("river_source_flow", "源头基础水流量", "float", 0, 10, 1.0, ""),
        ("river_humid_source_flow", "湿润源头额外流量", "float", 0, 10, 1.0, ""),
        ("river_mountain_source_flow", "高山源头最大额外流量", "float", 0, 10, 2.0, ""),
        ("river_mountain_radius", "高山邻近半径(格)", "float", 0, 50, 20.0, "≤0关闭高山加成"),
    ]),
    ("侵蚀与沉积", [
        ("enable_hydraulic_erosion", "启用水力侵蚀", "bool", None, None, True, ""),
        ("hydraulic_K", "水力侵蚀强度K", "float", 0, 0.2, 0.02, ""),
        ("hydraulic_iterations", "水力侵蚀轮数", "int", 0, 10, 3, ""),
        ("hydraulic_deposit_ratio", "邻域再沉积比例", "float", 0, 1, 0.5, ""),
        ("enable_wind_erosion", "启用风力侵蚀", "bool", None, None, False, ""),
        ("wind_K", "风力侵蚀强度K", "float", 0, 0.2, 0.03, ""),
        ("wind_iterations", "风力侵蚀轮数", "int", 0, 10, 2, ""),
        ("enable_river_erosion", "启用河流侵蚀", "bool", None, None, True, ""),
        ("river_K", "河流侵蚀系数K", "float", 0, 0.2, 0.02, "E=K·A^m·S^n"),
        ("river_m", "功率指数m", "float", 0, 2, 0.5, ""),
        ("river_n", "功率指数n", "float", 0, 2, 1.0, ""),
        ("fan_slope_threshold", "冲积扇坡度阈值", "float", 0, 5, 0.5, ""),
        ("fan_slope_drop_ratio", "冲积扇坡降比", "float", 0, 5, 1.5, ""),
        ("fan_min_strength", "冲积扇最小流量", "float", 0, 100, 20.0, ""),
        ("fan_deposition_ratio", "冲积扇沉积比例", "float", 0, 1, 0.5, ""),
        ("fan_radius", "冲积扇半径(格)", "int", 1, 20, 6, ""),
        ("delta_deposition_ratio", "三角洲沉积比例", "float", 0, 1, 0.7, ""),
        ("delta_radius", "三角洲半径(格)", "int", 1, 20, 4, ""),
        ("max_delta_per_cell", "单格海拔变化钳制(米)", "float", 0.5, 20, 5.0, "稳定性"),
    ]),
]

CLIMATE_GROUPS = [
    ("湿度阈值", [
        ("arid_threshold", "干旱阈值(0~100)", "float", 0, 100, 20.0, "低于此值为干旱"),
        ("semiarid_threshold", "半干旱阈值", "float", 0, 100, 40.0, ""),
        ("semihumid_threshold", "半湿润阈值", "float", 0, 100, 60.0, "高于此值为湿润"),
    ]),
    ("高山与季风", [
        ("high_altitude_threshold", "高山海拔阈值(米)", "float", 200, 4000, 1500.0, "高于此值划为高山气候"),
        ("monsoon_start_lat", "季风起始纬度(°N)", "float", 0, 70, 15.0, "地图右缘起、45°向西北"),
    ]),
    ("绿洲", [
        ("oasis_density", "绿洲泊松密度", "float", 0, 0.01, 0.001, "每像素随机概率"),
        ("river_oasis_threshold", "河流绿洲流量阈值", "float", 0, 1000, 100.0, "流量超过则成绿洲"),
    ]),
]

MODULES = [
    ("mountain", "地形生成", MOUNTAIN_GROUPS),
    ("hydrology", "水文与侵蚀", HYDROLOGY_GROUPS),
    ("climate", "气候分类", CLIMATE_GROUPS),
]


# ============================================================
# 参数控件
# ============================================================

class ParamRow:
    """
    一行参数控件：标签 + (滑条+输入框 双向联动) / 勾选框 / 下拉框。
    Optional 参数额外带“自动(None)”勾选框，勾选后禁用控件并返回 None。
    """

    def __init__(self, parent, key, label, ptype, vmin, vmax, default, tooltip=""):
        self.key = key
        self.ptype = ptype
        self.optional = ptype.startswith("o")
        self.is_int = ptype in ("int", "oint")

        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=6, pady=2)

        text = label + (f"  [{tooltip}]" if tooltip else "")
        ttk.Label(frame, text=text, width=34, anchor="w").pack(side="left")

        self.none_var = None
        if self.optional:
            self.none_var = tk.BooleanVar(value=(default is None))
            ttk.Checkbutton(frame, text="自动(None)", variable=self.none_var,
                            command=self._on_none_toggle).pack(side="right")

        if ptype in ("int", "oint"):
            self.var = tk.IntVar(value=default if default is not None else int(vmin))
        elif ptype in ("float", "ofloat"):
            self.var = tk.DoubleVar(value=default if default is not None else float(vmin))
        elif ptype == "bool":
            self.var = tk.BooleanVar(value=bool(default))
        else:  # choice
            self.var = tk.StringVar(value=str(default))

        if ptype in ("int", "float", "oint", "ofloat"):
            if self.is_int:
                self.var.trace_add("write", lambda *a: self._round_int())
            res = 1 if self.is_int else max((vmax - vmin) / 400.0, 1e-6)
            self.scale = ttk.Scale(frame, from_=vmin, to=vmax, variable=self.var,
                                   orient="horizontal", length=260)
            self.scale.pack(side="left", padx=(6, 4), fill="x", expand=True)
            # 不设置 format 选项（不同平台的 Tk 对它的格式串要求不一致，
            # "%.6g"/"%g" 都可能被拒绝）；不设时直接显示变量当前值即可。
            self.spin = tk.Spinbox(frame, textvariable=self.var, from_=vmin, to=vmax,
                                   increment=res, width=9)
            self.spin.pack(side="left")
        elif ptype == "bool":
            ttk.Checkbutton(frame, variable=self.var).pack(side="left", padx=6)
        else:
            self.scale = None
            self.combo = ttk.Combobox(frame, textvariable=self.var,
                                      values=list(vmin), state="readonly", width=10)
            self.combo.pack(side="left", padx=6)

        if self.optional:
            self._on_none_toggle()

    def _on_none_toggle(self):
        state = "disabled" if self.none_var.get() else "normal"
        for w in (getattr(self, "scale", None), getattr(self, "spin", None)):
            if w is not None:
                w.configure(state=state)

    def _round_int(self):
        try:
            self.var.set(round(self.var.get()))
        except (tk.TclError, ValueError):
            pass  # 输入框为中间状态（如空串/负号）时忽略

    def get(self):
        """返回当前参数值（Python 原生类型；Optional 勾选自动时返回 None）。"""
        if self.optional and self.none_var.get():
            return None
        v = self.var.get()
        if self.is_int:
            return int(v)
        if self.ptype == "bool":
            return bool(v)
        return v

    def set(self, value):
        """从外部（载入预设）写入参数值。"""
        if self.optional:
            self.none_var.set(value is None)
            self._on_none_toggle()
            if value is None:
                return
        if self.ptype == "choice":
            self.var.set(str(value))
        else:
            self.var.set(value)


# ============================================================
# 滚动容器
# ============================================================

class ScrollFrame(ttk.Frame):
    """带垂直滚动条的框架（用于放大量参数行）。"""

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        # 鼠标滚轮
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")


# ============================================================
# 主窗口
# ============================================================

class ParamInputApp:
    """
    参数输入窗口。

    Parameters
    ----------
    master : tk.Tk / tk.Toplevel, optional
        父窗口；None 时自行创建 Tk 根窗口。
    on_confirm : callable, optional
        用户点击“确认生成参数”时的回调，签名为 on_confirm(params: dict)。
        不提供时窗口关闭后可通过 .result 读取（阻塞模式）。
    initial : dict, optional
        初始参数（{"mountain": {...}, ...}），覆盖默认值。
    """

    def __init__(self, master=None, on_confirm=None, initial=None):
        self._own_root = master is None
        self.master = master if master is not None else tk.Tk()
        self.on_confirm = on_confirm
        self.result = None

        self.master.title("架空世界生成器 — 参数输入")
        self.master.geometry("860x640")

        self.rows = {}  # {module_key: {param_key: ParamRow}}

        notebook = ttk.Notebook(self.master)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        for mod_key, mod_title, groups in MODULES:
            scroll = ScrollFrame(notebook)
            notebook.add(scroll, text=f" {mod_title} ")
            self.rows[mod_key] = {}
            for group_title, params in groups:
                box = ttk.LabelFrame(scroll.inner, text=group_title)
                box.pack(fill="x", padx=8, pady=6)
                for spec in params:
                    row = ParamRow(box, *spec)
                    self.rows[mod_key][spec[0]] = row

        # ---- 底部按钮区 ----
        bar = ttk.Frame(self.master)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(bar, text="恢复默认值", command=self._reset_defaults).pack(side="left")
        ttk.Button(bar, text="载入预设(JSON)…", command=self._load_preset).pack(side="left", padx=6)
        ttk.Button(bar, text="保存预设(JSON)…", command=self._save_preset).pack(side="left")
        ttk.Button(bar, text="取消", command=self._on_cancel).pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="确认生成参数", command=self._on_confirm).pack(side="right")

        self._defaults = {m: {k: r.get() for k, r in rs.items()}
                          for (m, _, _), (m, rs) in zip(MODULES, self.rows.items())}
        if initial:
            self.set_params(initial)

    # ---------------- 数据打包 ----------------

    def get_params(self):
        """收集当前所有参数，返回 {"mountain": {...}, "hydrology": {...}, "climate": {...}}。"""
        return {mod: {k: row.get() for k, row in rows.items()}
                for mod, rows in self.rows.items()}

    def set_params(self, params):
        """从字典回填参数（缺失键保持现状）。"""
        for mod, values in params.items():
            if mod not in self.rows or not isinstance(values, dict):
                continue
            for k, v in values.items():
                if k in self.rows[mod]:
                    self.rows[mod][k].set(v)

    # ---------------- 按钮动作 ----------------

    def _reset_defaults(self):
        self.set_params(self._defaults)

    def _save_preset(self):
        path = filedialog.asksaveasfilename(
            parent=self.master, defaultextension=".json",
            filetypes=[("JSON", "*.json")], title="保存参数预设")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_params(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            messagebox.showerror("保存失败", str(e), parent=self.master)

    def _load_preset(self):
        path = filedialog.askopenfilename(
            parent=self.master, filetypes=[("JSON", "*.json")], title="载入参数预设")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.set_params(data)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("载入失败", str(e), parent=self.master)

    def _on_confirm(self):
        self.result = self.get_params()
        if self.on_confirm is not None:
            self.on_confirm(self.result)
        if self._own_root:
            self.master.destroy()

    def _on_cancel(self):
        self.result = None
        if self._own_root:
            self.master.destroy()


# ============================================================
# 阻塞式入口
# ============================================================

def collect_params(initial=None):
    """
    弹出参数输入窗口（阻塞），返回打包好的参数字典：
        {"mountain": {...}, "hydrology": {...}, "climate": {...}}
    键名与各模块主函数签名一致，可直接 ** 解包传入。
    用户点“取消”时返回 None。
    """
    app = ParamInputApp(initial=initial)
    app.master.mainloop()
    return app.result


if __name__ == "__main__":
    params = collect_params()
    if params is None:
        print("用户取消。")
    else:
        print(json.dumps(params, ensure_ascii=False, indent=2))