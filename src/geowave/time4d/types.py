from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from geowave.time4d.config import CrossConfig

# 可靠性码（reliability map 取值）
REL_OK = 0            # 两期都有效，所有校正均成功
REL_MISSING = 1       # 至少一期缺覆盖（不可插值、不可解释为变化）
REL_SUPPORT = 2       # 校正算子支撑区落入缺覆盖，结果不可靠
REL_LOW_ALIGN = 3     # 局部对齐相关度过低
REL_LOW_ENERGY = 4    # 时窗内能量过低，NRMS 无意义


@dataclass
class Survey4D:
    """同一规则网格上的一期三维数据体。

    data 形状为 (iline, xline, time)；valid 为布尔覆盖掩码。
    data 在 valid=False 处约定为 NaN，但所有流程只信任 valid。
    dt 为时间采样率（秒），origin 为网格原点 (iline, xline, t0)，
    spacing 为网格间距，随结果一起落盘，保证回放时可核对几何。
    """

    name: str
    data: np.ndarray
    valid: np.ndarray
    dt: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.data.shape != self.valid.shape or self.data.ndim != 3:
            raise ValueError('data 与 valid 必须同为三维且形状一致')
        if self.dt <= 0:
            raise ValueError('dt 必须为正')
        # 契约：valid=True 的体元必须是有限值；边界重采样产生的 NaN
        # 必须由调用方并入缺覆盖掩码，而不是带进均衡流程。
        if np.any(self.valid & ~np.isfinite(self.data)):
            raise ValueError('valid=True 处存在 NaN/Inf；请先将其并入缺覆盖掩码')
        self.data = np.where(self.valid, self.data, np.nan)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def ntime(self) -> int:
        return self.data.shape[2]


@dataclass
class CrossResult:
    """交叉均衡产物。全部体元与输入同网格，缺覆盖处为 NaN。"""

    base_matched: np.ndarray          # 覆盖匹配后的基准体
    monitor_matched: np.ndarray       # 全部校正后的监测体（回放链作用结果）
    difference: np.ndarray            # base_matched - monitor_matched
    difference_raw: np.ndarray        # 覆盖匹配后、均衡前的差异（仅几何/相位残差）
    nrms: np.ndarray                  # NRMS 重复性图（%）
    predictability: np.ndarray        # 归一化相关系数重复性图
    reliability: np.ndarray           # uint8，取值见 REL_* 常量
    valid_common: np.ndarray          # 两期原始覆盖交集（侵蚀前）

    bulk_shift_s: float               # 整体时移（秒），监测体被平移量
    bulk_score: float                 # 整体时移归一化相关
    complex_scalar: complex           # 联合相位/振幅算子 g（b_corr = g*m）
    filter_taps: np.ndarray           # 零相位整形滤波器（可能长度为 0 表示禁用）
    amp_map: np.ndarray               # 空变振幅增益（2D）
    lag_map: np.ndarray               # 局部时移场（秒，监测被校正量）
    lag_anchor: np.ndarray            # 局部时移锚点（秒，未做空洞填充）
    lag_corr: np.ndarray               # 局部时移归一化相关（2D）

    stages: list[str] = field(default_factory=list)  # 实际执行的阶段名
    operators: dict[str, Any] = field(default_factory=dict)  # 回放链（算子与增益/时移场）
    gate: tuple[int, int] = (0, 0)
    config: 'CrossConfig | None' = None
