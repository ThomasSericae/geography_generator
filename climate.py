"""
climate_generator.py
架空世界气候分类模块

依据气压带（pressure_belt）、湿度（humidity）、海拔（elevation）、
季风区掩膜、河流流量（river_discharge）生成生物群系（biome）图层。

规则完全依照用户描述实现，所有随机源统一使用 world.rng。
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import IntEnum

# 假设 world_core 在同一包下，可根据实际导入路径调整
from world_core import World


class Biome(IntEnum):
    """气候/生物群系类型枚举（写入 world.biome 的整数值）"""
    UNKNOWN = 0

    # 热带
    TROPICAL_RAINFOREST = 1
    TROPICAL_MONSOON = 2
    TROPICAL_SAVANNA = 3
    TROPICAL_DESERT = 4

    # 温带（通用）
    TEMPERATE_OCEANIC = 5          # 暂未使用，见注释

    # 季风系列
    TEMPERATE_MONSOON_HUMID = 6
    TEMPERATE_MONSOON_SEMIHUMID = 7
    TEMPERATE_MONSOON_SEMIARID = 8
    SUBTROPICAL_MONSOON_HUMID = 9
    SUBTROPICAL_MONSOON_SEMIHUMID = 10

    # 温带大陆
    HUMID_CONTINENTAL = 11
    SEMIHUMID_CONTINENTAL = 12
    SEMIARID_CONTINENTAL = 13

    # 亚寒带
    TAIGA = 14
    SUBARCTIC_STEPPE = 15

    # 地中海
    MEDITERRANEAN = 16

    # 高山
    HIGHLAND = 17
    TROPICAL_HIGHLAND = 18

    # 特殊（沙漠灌溉区/绿洲）
    IRRIGATED_OASIS = 19


def classify_humidity(
    humidity: np.ndarray,
    arid_threshold: float = 20.0,
    semiarid_threshold: float = 40.0,
    semihumid_threshold: float = 60.0,
) -> np.ndarray:
    """
    将湿度（0~100）分为四类，返回 int 数组：
        0 = 干旱 (arid)
        1 = 半干旱 (semiarid)
        2 = 半湿润 (semihumid)
        3 = 湿润 (humid)
    """
    result = np.zeros_like(humidity, dtype=np.int8)
    # 注意：使用浮点数比较，阈值可配置
    result[humidity < arid_threshold] = 0
    result[(humidity >= arid_threshold) & (humidity < semiarid_threshold)] = 1
    result[(humidity >= semiarid_threshold) & (humidity < semihumid_threshold)] = 2
    result[humidity >= semihumid_threshold] = 3
    return result


def create_monsoon_mask(
    height: int,
    width: int,
    start_lat_deg: float = 15.0,
    max_lat_deg: float = 70.0,
) -> np.ndarray:
    """
    生成季风区掩膜 (bool, height, width)。

    规则：从地图右侧的北纬 start_lat_deg 度，斜向 45° 向西北，
    占地图面积 1/3（右侧/南侧区域）。

    地图坐标：y=0 为 70°N（顶部），y=height-1 为 0°（底部）。
    """
    # 计算起始行（北纬15°对应的像素 y）
    # 纬度 = max_lat_deg * (1 - y / height)  =>  y = height * (1 - lat / max_lat_deg)
    y_start = int(height * (1.0 - start_lat_deg / max_lat_deg))

    # 斜向 45° 向西北：在图像坐标中，x减小，y减小，直线斜率为 1 (dy/dx = 1)
    # 直线方程：y = y_start + (x - (width - 1))
    # 直线右侧（南侧）的区域为 y > y_start + x - (width - 1)
    x_coords = np.arange(width)
    y_coords = np.arange(height).reshape(-1, 1)
    # 计算直线边界
    line_y = y_start + (x_coords - (width - 1))
    # 季风区掩膜：像素位于直线下方（y更大）
    monsoon = y_coords > line_y

    # 可选：强制面积恰好为 1/3（如果需要微调偏移量）
    # 但这里严格遵循“右侧北纬15度”起点，面积可能是近似1/3，
    # 如果你需要精确1/3，可以取消下面注释并调整 y_start
    # target_area = height * width / 3
    # current_area = np.sum(monsoon)
    # if abs(current_area - target_area) / target_area > 0.05:
    #     print(f"Warning: Monsoon area {current_area} vs target {target_area}")

    return monsoon


def generate_climate(
    world,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    主入口：根据 world 中的图层和配置生成气候，填充 world.biome。

    Parameters
    ----------
    world : World 实例
        必须包含 elevation, pressure_belt, humidity, river_discharge, rng。
    config : dict, optional
        可调参数，支持以下键（默认值见下方）：
            arid_threshold : float = 20.0
            semiarid_threshold : float = 40.0
            semihumid_threshold : float = 60.0
            high_altitude_threshold : float = 1500.0   # 米
            oasis_density : float = 0.001              # 泊松密度（每像素概率）
            river_oasis_threshold : float = 100.0      # 河流流量阈值
            monsoon_start_lat : float = 15.0           # 季风起始纬度
    """
    # ---------- 默认配置 ----------
    defaults = {
        "arid_threshold": 20.0,
        "semiarid_threshold": 40.0,
        "semihumid_threshold": 60.0,
        "high_altitude_threshold": 1500.0,
        "oasis_density": 0.001,
        "river_oasis_threshold": 100.0,
        "monsoon_start_lat": 15.0,
    }
    if config is None:
        config = {}
    cfg = {**defaults, **config}

    h, w = world.height, world.width
    elev = world.elevation
    pressure = world.pressure_belt          # 0~5
    humidity = world.humidity               # 0~100
    discharge = world.river_discharge
    rng = world.rng

    # 1. 湿度分类
    hum_class = classify_humidity(
        humidity,
        arid_threshold=cfg["arid_threshold"],
        semiarid_threshold=cfg["semiarid_threshold"],
        semihumid_threshold=cfg["semihumid_threshold"],
    )  # 0~3

    # 2. 季风区掩膜
    monsoon_mask = create_monsoon_mask(
        h, w, start_lat_deg=cfg["monsoon_start_lat"]
    )

    # 3. 初始化生物群系为 UNKNOWN
    biome = np.full((h, w), Biome.UNKNOWN, dtype=np.int32)

    # 4. 高海拔覆盖（优先级最高）
    high_mask = elev > cfg["high_altitude_threshold"]
    # 热带高山：非季风区 + 气压带 0,1,2 (0~35°)
    tropical_high_mask = (
        high_mask
        & (~monsoon_mask)
        & (pressure >= 0) & (pressure <= 2)
    )
    other_high_mask = high_mask & (~tropical_high_mask)

    biome[tropical_high_mask] = Biome.TROPICAL_HIGHLAND
    biome[other_high_mask] = Biome.HIGHLAND

    # 5. 低海拔区域分类（~high_mask）
    low_mask = ~high_mask

    # 5a. 季风区（低海拔）
    monsoon_low = low_mask & monsoon_mask

    # 气压带 0,1 (0~25°) -> 热带季风
    m1 = monsoon_low & (pressure <= 1)
    biome[m1] = Biome.TROPICAL_MONSOON

    # 气压带 2 (25~35°) -> 亚热带季风
    m2 = monsoon_low & (pressure == 2)
    # 湿润 vs 半湿润（用户指定只分这两种）
    m2_humid = m2 & (hum_class == 3)   # 湿润
    m2_semihumid = m2 & (hum_class == 2)  # 半湿润
    # 其他湿度（干旱/半干旱）在季风区如何处理？用户未明确，暂归为半湿润的亚热带季风（或可设默认）
    m2_other = m2 & (~m2_humid) & (~m2_semihumid)
    biome[m2_humid] = Biome.SUBTROPICAL_MONSOON_HUMID
    biome[m2_semihumid] = Biome.SUBTROPICAL_MONSOON_SEMIHUMID
    # 对于季风区里亚热带出现干旱/半干旱，按常见逻辑可算作半湿润变体，这里我统一归为半湿润
    biome[m2_other] = Biome.SUBTROPICAL_MONSOON_SEMIHUMID

    # 气压带 3,4 (35~60°) -> 温带季风
    m3 = monsoon_low & (pressure >= 3) & (pressure <= 4)
    m3_humid = m3 & (hum_class == 3)
    m3_semihumid = m3 & (hum_class == 2)
    m3_semiarid = m3 & (hum_class == 1)
    m3_arid = m3 & (hum_class == 0)  # 用户未提干旱温带季风，归为半干旱
    biome[m3_humid] = Biome.TEMPERATE_MONSOON_HUMID
    biome[m3_semihumid] = Biome.TEMPERATE_MONSOON_SEMIHUMID
    biome[m3_semiarid | m3_arid] = Biome.TEMPERATE_MONSOON_SEMIARID

    # 气压带 5 (60~70°) 季风用户没提，不处理，留待非季风规则

    # 5b. 非季风区（低海拔）
    nonmonsoon_low = low_mask & (~monsoon_mask)

    # ----- 气压带 0,1,2 (赤道/信风/副高) 0~35° -----
    trop_mask = nonmonsoon_low & (pressure <= 2)
    # 湿润 -> 雨林
    trop_rain = trop_mask & (hum_class == 3)
    # 半湿润 -> 草原
    trop_savanna = trop_mask & (hum_class == 2)
    # 半干旱/干旱 -> 沙漠（先全置沙漠，后续绿洲覆盖）
    trop_desert = trop_mask & (hum_class <= 1)

    biome[trop_rain] = Biome.TROPICAL_RAINFOREST
    biome[trop_savanna] = Biome.TROPICAL_SAVANNA
    biome[trop_desert] = Biome.TROPICAL_DESERT

    # ----- 气压带 3 (地中海带 35~42°) -----
    med_mask = nonmonsoon_low & (pressure == 3)
    med_humid_semi = med_mask & (hum_class >= 2)   # 湿润/半湿润
    med_arid_semi = med_mask & (hum_class <= 1)    # 半干旱/干旱
    biome[med_humid_semi] = Biome.MEDITERRANEAN
    biome[med_arid_semi] = Biome.SEMIARID_CONTINENTAL

    # ----- 气压带 4 (西风带 42~55°) -----
    westerly_mask = nonmonsoon_low & (pressure == 4)
    w_humid = westerly_mask & (hum_class == 3)
    w_semihumid = westerly_mask & (hum_class == 2)
    w_semiarid_arid = westerly_mask & (hum_class <= 1)
    biome[w_humid] = Biome.HUMID_CONTINENTAL
    biome[w_semihumid] = Biome.SEMIHUMID_CONTINENTAL
    biome[w_semiarid_arid] = Biome.SEMIARID_CONTINENTAL

    # ----- 气压带 5 (副极地 55~70°) -----
    polar_mask = nonmonsoon_low & (pressure == 5)
    p_humid_semi = polar_mask & (hum_class >= 2)
    p_arid_semi = polar_mask & (hum_class <= 1)
    biome[p_humid_semi] = Biome.TAIGA
    biome[p_arid_semi] = Biome.SUBARCTIC_STEPPE

    # 6. 特殊：绿洲（沙漠灌溉区）覆盖
    # 作用范围：非季风区 + 热带沙漠（气压带 0,1,2）+ 半干旱（hum_class == 1）
    oasis_base_mask = (
        nonmonsoon_low
        & (pressure <= 2)
        & (hum_class == 1)          # 半干旱
        & (biome == Biome.TROPICAL_DESERT)   # 当前是沙漠
    )

    # 6a. 泊松随机点（密度 cfg["oasis_density"]）
    rand_vals = rng.random((h, w))
    oasis_random = oasis_base_mask & (rand_vals < cfg["oasis_density"])

    # 6b. 高流量河流
    oasis_river = oasis_base_mask & (discharge > cfg["river_oasis_threshold"])

    oasis_mask = oasis_random | oasis_river
    biome[oasis_mask] = Biome.IRRIGATED_OASIS

    # 7. 将结果写回 world
    # 注意：world.biome 是只读属性，但它是返回视图，直接通过索引赋值即可
    # 因为 world._biome 是内部数组，world.biome 返回的是该数组的引用
    # 安全做法：获取数组引用并赋值
    biome_arr = world.biome
    biome_arr[:] = biome


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 假设 world 已创建
    # from world_core import World
    # w = World(seed=42, width=512, height=512)
    # 此处需先运行海拔、湿度、气压带等生成模块（略）
    # generate_climate(w)
    pass