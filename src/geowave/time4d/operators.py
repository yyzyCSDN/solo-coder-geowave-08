"""均衡算子：全部为纯函数，输入相同则逐位可复现。

约定 shift>0 / lag>0 表示监测体相对基准在时间上偏晚（需向早校正）。
"""
from __future__ import annotations

import numpy as np

from geowave.time4d.grid import resample_shift, resample_warp


def apply_bulk_shift(monitor: np.ndarray, valid_m: np.ndarray, lag_samples: float):
    """整体时移校正。

    约定 lag_samples>0 表示监测事件偏晚；校正后 y[i] = m[i + lag]，
    即重采样位移为 -lag。数据与掩码一起重采样，越界样本变成缺覆盖。
    """
    m = np.where(valid_m, monitor, np.nan)
    y = resample_shift(m, -lag_samples)
    v = resample_shift(valid_m.astype(np.float64), -lag_samples)
    return y, v > 0.999


def apply_complex_scalar(monitor: np.ndarray, g: complex) -> np.ndarray:
    """解析信号域复数算子 g：等价于增益 |g| + 相位旋转 angle(g)。

    对实信号 x，H 为希尔伯特变换，Re(g*(x + iHx)) 给出旋转结果，
    用与 scipy.signal.hilbert 相同的 FFT 方式实现。
    """
    n = monitor.shape[-1]
    X = np.fft.fft(np.where(np.isnan(monitor), 0.0, monitor), axis=-1)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(X * h, axis=-1)
    return np.real(g * analytic)


def apply_filter(monitor: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """施加零相位整形滤波器（边界为 NaN，不做补零欺骗）。"""
    if taps.size <= 1:
        return monitor.copy()
    pad = taps.size // 2
    # 频域线性卷积，再裁回原长度，两端各 pad 点支撑不足 -> NaN
    nfft = monitor.shape[-1] + taps.size - 1
    X = np.fft.rfft(np.where(np.isnan(monitor), 0.0, monitor), n=nfft, axis=-1)
    H = np.fft.rfft(taps, n=nfft)
    full = np.fft.irfft(X * H, n=nfft, axis=-1)
    y = full[..., pad:pad + monitor.shape[-1]].copy()
    y[..., :pad] = np.nan
    y[..., monitor.shape[-1] - pad:] = np.nan
    return y


def apply_amp_map(monitor: np.ndarray, gain2d: np.ndarray) -> np.ndarray:
    """逐道空变振幅增益。"""
    return monitor * gain2d[..., None]


def apply_local_warp(monitor: np.ndarray, lag3d_samples: np.ndarray) -> np.ndarray:
    """局部形变校正：lag>0 表示该处监测事件偏晚，y[i]=m[i+lag]。"""
    ni, nx, nt = monitor.shape
    m = monitor.reshape(ni * nx, nt)
    # resample_warp 实现为 y[i]=x[i-lag]，故取负号
    lag = (-lag3d_samples).reshape(ni * nx, nt)
    y = resample_warp(m, lag)
    return y.reshape(ni, nx, nt)
