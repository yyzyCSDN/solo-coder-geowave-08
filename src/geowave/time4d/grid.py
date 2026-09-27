"""与业务无关的数值原语：掩码侵蚀、加权平滑、亚采样重采样、滑窗能量。

约定：时间轴恒为最后轴；缺覆盖一律 NaN，绝不补零参与估计。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

LANCZOS_A = 4


def erode(mask: np.ndarray, radius: tuple[int, int, int]) -> np.ndarray:
    """三维各向异性二值侵蚀；半径 0 的轴不处理。边界按无效处理（保守）。"""
    out = np.asarray(mask, dtype=bool).copy()
    for axis, r in enumerate(radius):
        if r <= 0:
            continue
        struct = np.zeros([3, 3, 3], dtype=bool)
        sl = [slice(1, 2)] * 3
        sl[axis] = slice(None)
        struct[tuple(sl)] = True
        out = binary_erosion(out, structure=struct, iterations=int(r), border_value=0)
    return out


def weighted_gaussian(
    field: np.ndarray, weight: np.ndarray, sigma: tuple[float, ...], min_weight: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """归一化高斯平滑（支持缺失锚点）。

    返回 (平滑场, 有效性)；有效性 = 有效权重占核内总权重的比例，
    低于 min_weight 处平滑场为 NaN——缺失锚点不会被静默填成正常值。
    """
    w = np.where(np.isfinite(field) & (weight > 0), weight, 0.0)
    z = np.where(w > 0, field, 0.0)
    num = gaussian_filter(z * w, sigma=sigma, mode='constant', cval=0.0, truncate=4.0)
    den = gaussian_filter(w, sigma=sigma, mode='constant', cval=0.0, truncate=4.0)
    mass = gaussian_filter(np.ones_like(w), sigma=sigma, mode='constant', cval=0.0, truncate=4.0)
    support = np.divide(den, mass, out=np.zeros_like(den), where=mass > 0)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    out[support < min_weight] = np.nan
    return out, support


def _lanczos_taps(frac: np.ndarray, a: int = LANCZOS_A) -> tuple[np.ndarray, np.ndarray]:
    """给定小数部分 frac（pos - floor(pos)），返回相对整数位移与权重。"""
    k = np.arange(-a + 1, a + 1)  # 2a 个抽头
    u = frac[..., None] - k  # y[pos] = sum_k x[floor(pos)+k] * L(frac - k)
    pi_u = np.pi * u

    def _sinc(v: np.ndarray) -> np.ndarray:
        return np.where(np.abs(v) < 1e-12, 1.0, np.sin(v) / np.where(v == 0, 1.0, v))

    w = _sinc(pi_u) * _sinc(pi_u / a)
    w = np.where(np.abs(u) < a, w, 0.0)
    w /= w.sum(axis=-1, keepdims=True)
    return k, w


def resample_shift(
    x: np.ndarray, shift_samples: float, a: int = LANCZOS_A
) -> np.ndarray:
    """沿最后轴整体平移并 Lanczos 重采样：y[i] = x[i - shift]。

    shift>0 表示信号向下（时间增大方向）移动。越过边界的输出为 NaN。
    """
    nt = x.shape[-1]
    i = np.arange(nt)
    pos = i - shift_samples
    j0 = np.floor(pos).astype(np.int64)
    frac = pos - j0
    k, w = _lanczos_taps(frac, a)
    idx = j0[..., None] + k  # (nt, 2a)
    lead = x.shape[:-1]
    xf = x.reshape(-1, nt)
    idx2 = np.broadcast_to(idx, (xf.shape[0],) + idx.shape)
    valid_k = (idx2 >= 0) & (idx2 < nt)
    idx_c = np.clip(idx2, 0, nt - 1)
    xg = np.take_along_axis(xf, idx_c.reshape(xf.shape[0], -1), axis=1)
    xg = xg.reshape(xf.shape[0], nt, -1)
    wv = np.where(valid_k, w, 0.0)
    y = (xg * wv).sum(axis=-1)
    y[(~valid_k).all(axis=-1) | (wv.sum(axis=-1) < 0.999)] = np.nan
    return y.reshape(lead + (nt,))


def resample_warp(x2d: np.ndarray, lag_samples: np.ndarray, a: int = LANCZOS_A) -> np.ndarray:
    """逐道（行=道，列=时间）施加空变 lag：y[r, i] = x[r, i - lag[r, i]]。"""
    nr, nt = x2d.shape
    i = np.broadcast_to(np.arange(nt), (nr, nt))
    pos = i - lag_samples
    j0 = np.floor(pos).astype(np.int64)
    frac = pos - j0
    k, w = _lanczos_taps(frac, a)  # (nr, nt, 2a)
    idx = j0[..., None] + k
    valid_k = (idx >= 0) & (idx < nt)
    idx_c = np.clip(idx, 0, nt - 1)
    xg = np.take_along_axis(x2d, idx_c.reshape(nr, -1), axis=1).reshape(nr, nt, -1)
    wv = np.where(valid_k, w, 0.0)
    y = (xg * wv).sum(axis=-1)
    no_support = (lag_samples != lag_samples) | (~valid_k).all(axis=-1) | (wv.sum(axis=-1) < 0.999)
    y[no_support] = np.nan
    return y


def convolve_support(mask_float: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """把滤波器作用于浮点掩码，返回每点获得的 |tap| 权重占比（支撑度）。

    用零填充线性卷积（绝不周期回绕）；边界处支撑度自然下降到 <1，
    调用方据此把缺支撑体元标成 REL_SUPPORT。
    """
    k = taps / (np.abs(taps).sum() + 1e-30)
    pad = k.size // 2
    nt = mask_float.shape[-1]
    nfft = nt + k.size - 1
    X = np.fft.rfft(mask_float, n=nfft, axis=-1)
    K = np.fft.rfft(np.abs(k), n=nfft)
    full = np.fft.irfft(X * K, n=nfft, axis=-1)
    return full[..., pad:pad + nt]
