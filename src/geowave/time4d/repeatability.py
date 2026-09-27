"""重复性图：NRMS 与可预测性（归一化相关）。

NRMS = 200 * RMS(b - m) / (RMS(b) + RMS(m))，单位 %。
窗内含任一无效样本（缺覆盖/支撑不足）则输出 NaN，绝不补零参与统计。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _window_sum(x, half: int) -> np.ndarray:
    """最后轴长度 2*half+1 的窗和；边界外按 0 计。"""
    size = (1,) * (x.ndim - 1) + (2 * half + 1,)
    win = uniform_filter(x, size=size, mode='constant', cval=0.0) * (2 * half + 1)
    return win


def nrms_map(b: np.ndarray, m: np.ndarray, valid: np.ndarray,
             half_samples: int, min_rms: float):
    """返回 (nrms[%], predictability, low_energy)。

    要求窗内全部样本可靠（全 1 窗）；否则该体元 NaN——宁可空白也不把
    缺覆盖边缘统计成高重复性或强差异。
    """
    w = valid.astype(np.float64)
    # 只信任 valid：输入在无效处可能是 NaN，先整体清零避免 0*NaN 污染窗和
    b = np.where(valid, np.nan_to_num(b, nan=0.0), 0.0)
    m = np.where(valid, np.nan_to_num(m, nan=0.0), 0.0)
    bw, mw = b, m
    count = _window_sum(w, half_samples)
    need = 2 * half_samples + 1
    full = count >= 0.999 * need

    b2 = np.divide(_window_sum(bw * bw, half_samples), count,
                   out=np.full_like(bw, np.nan), where=count > 0)
    m2 = np.divide(_window_sum(mw * mw, half_samples), count,
                   out=np.full_like(mw, np.nan), where=count > 0)
    d2 = np.divide(_window_sum((bw - mw) ** 2, half_samples), count,
                   out=np.full_like(bw, np.nan), where=count > 0)
    bm = np.divide(_window_sum(bw * mw, half_samples), count,
                   out=np.full_like(bw, np.nan), where=count > 0)
    rb = np.sqrt(np.maximum(b2, 0))
    rm = np.sqrt(np.maximum(m2, 0))
    denom = rb + rm
    nrms = np.divide(200.0 * np.sqrt(np.maximum(d2, 0)), denom,
                     out=np.full_like(denom, np.nan), where=denom > 0)
    pred = np.divide(bm, rb * rm, out=np.full_like(bm, np.nan),
                     where=rb * rm > 0)
    pred = np.clip(pred, -1.0, 1.0)
    low_energy = full & ((rb < min_rms) | (rm < min_rms))
    nrms = np.where(full & ~low_energy, nrms, np.nan)
    pred = np.where(full, pred, np.nan)
    return nrms, pred, low_energy
