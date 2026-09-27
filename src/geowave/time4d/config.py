from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class CrossConfig:
    """交叉均衡参数。任何一项变化都会进入运行摘要，回放结果必须逐位一致。"""

    # 覆盖匹配
    coverage_erode: tuple[int, int, int] = (1, 1, 0)  # 交集掩码三维侵蚀半径
    support_margin_s: float = 0.05                    # 整形滤波器单侧支撑（秒）
    gate_s: tuple[float, float] | None = None         # 参数估计分析时窗（秒）；None 用全程

    # 整体时移
    max_bulk_shift_s: float = 0.04                    # 搜索半窗（秒）
    robust_quantile: float = 0.90                     # 逐道 lag 相关分位合成门槛

    # 联合相位/振幅（解析信号最小二乘）
    phase_max_deg: float = 180.0
    spatial_phase: bool = False                       # 是否估计缓空变 g（锚点网格）

    # 整形滤波器（监测 -> 基准，匹配频带与子波）
    # 默认关闭：仅当两期确有子波/频带差异、且滤波后相关确实提升时才启用；
    # 在已高相关的数据上强行整形会过拟合并反相。
    filter_length_s: float = 0.0                     # 单侧长度；<=0 关闭
    filter_prewhitening: float = 0.05                # 相对自谱峰值的白噪
    filter_min_corr_gain: float = 0.005              # 相关提升低于此则拒绝滤波器
    filter_min_energy: float = 1e-6                  # 监测能量过低则跳过

    # 空变振幅平衡（锚点网格 + 归一化卷积）
    amp_anchor_il: int = 4                            # 锚点间距（线），<=0 关闭
    amp_anchor_xl: int = 4
    amp_smooth_il: float = 1.5                        # 锚点间平滑尺度（单位：锚点距）
    amp_smooth_xl: float = 1.5
    amp_gain_clip: float = 4.0                        # 增益截断，防止在弱区放大噪声
    amp_min_energy: float = 1e-6

    # 局部形变（滑窗归一化互相关，锚点网格）
    warp_anchor_il: int = 3
    warp_anchor_xl: int = 3
    warp_window_s: float = 0.20                       # 相关窗半长
    warp_search_s: float = 0.02                       # 在整体时移之后的局部搜索半窗
    warp_upsample: int = 16                           # 局部 lag 频域上采样倍数（1/up 采样分辨率）
    warp_smooth_il: float = 0.8
    warp_smooth_xl: float = 0.8
    warp_smooth_t: float = 6.0                        # 时间方向平滑尺度（样本/锚点窗步）
    warp_min_corr: float = 0.30                       # 锚点相关门槛
    warp_min_support: float = 0.35                    # 平滑核内有效锚点权重占比门槛
    warp_strong_corr: float = 0.90                    # 高于此认为峰唯一，免锐度判据
    warp_peak_prominence: float = 0.02                # 低相关区主峰须压过次峰（lag 唯一性）
    warp_max_jump_s: float = 0.012                    # 相邻锚点跳变上限，超限标记不可靠

    # 重复性图
    nrms_window_s: float = 0.10                       # NRMS/相关图窗半长（沿时间）
    nrms_min_rms: float = 1e-6                        # 两期 RMS 均低于此则 REL_LOW_ENERGY

    def to_dict(self) -> dict:
        d = asdict(self)
        d['coverage_erode'] = list(self.coverage_erode)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> 'CrossConfig':
        d = dict(d)
        if 'coverage_erode' in d:
            d['coverage_erode'] = tuple(d['coverage_erode'])
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})
