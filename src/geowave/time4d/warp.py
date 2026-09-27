"""局部形变估计：滑窗包络/波形归一化互相关 + 锚点网格正则化。

锚点上逐道逐时窗估计 lag（lag>0 表示监测事件晚），再做加权高斯平滑、
相邻跳变检查，最后插值回全网格。空间方向只在锚点包络内插值；
时间方向用首尾锚点值向记录两端延伸（静态边界假设），并随可靠性图披露。
任何锚点空洞只能标记不可靠，不允许平滑/插值后冒充真实变化。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

from geowave.time4d.grid import weighted_gaussian


@dataclass
class LocalWarp:
    lag_full: np.ndarray   # (ni, nx, nt) 样本数，监测被校正量；不可靠处 NaN
    lag_anchor: np.ndarray # (nu, nv, nw) 平滑后的锚点 lag（样本）
    lag_raw: np.ndarray    # 平滑前锚点 lag（NaN = 未检出）
    lag_weight: np.ndarray # 平滑前锚点权重（相关度）
    corr: np.ndarray       # (ni, nx) 各道最大窗相关
    low_align: np.ndarray  # (ni, nx, nt) 低相关/跳变/锚点包络外标记


_MIN_VALID_FRACTION = 0.9  # 窗内（含 lag 位移后）有效样本比例门槛


def _masked_normcorr(b: np.ndarray, m: np.ndarray, wb: np.ndarray, wm: np.ndarray):
    """掩码加权归一化互相关；有效权重比例不足返回 nan。"""
    n = b.size
    if n == 0:
        return np.nan
    w = (wb & wm).astype(np.float64)
    if w.sum() < _MIN_VALID_FRACTION * n:
        return np.nan
    x = w * (b - np.average(b, weights=w))
    y = w * (m - np.average(m, weights=w))
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 0 else np.nan


def _masked_complex_corr(zb: np.ndarray, zm: np.ndarray,
                         wb: np.ndarray, wm: np.ndarray) -> float:
    """复数解析信号掩码互相关取模；对恒定相位不敏感且半采样有唯一峰。"""
    w = (wb & wm).astype(np.float64)
    if w.sum() < _MIN_VALID_FRACTION * w.size:
        return np.nan
    x = (zb - np.average(zb, weights=w)) * w
    y = (zm - np.average(zm, weights=w)) * w
    den = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)
    return float(abs(np.vdot(x, y)) / den) if den > 0 else np.nan


def _window_corr(b, m, w, center: int, half_win: int, maxlag: int, signals):
    """以 center 为心的窗内掩码 NCC，corr[j]=corr(b[t], m[t+lag])。

    返回 (lags, 复数相关模, 包络相关, 波形相关)：复数模用于定峰（抗相位、
    半采样唯一），包络/波形相关用于相似度质控。
    """
    c0 = center - half_win
    c1 = center + half_win + 1
    if c0 - maxlag < 0 or c1 + maxlag > b.size:
        return None
    lags = np.arange(-maxlag, maxlag + 1)
    cz = np.full(lags.size, np.nan)
    ce = np.full(lags.size, np.nan)
    cw = np.full(lags.size, np.nan)
    zb, zm, eb, em = signals
    for j, lag in enumerate(lags):
        cw[j] = _masked_normcorr(b[c0:c1], m[c0 + lag:c1 + lag],
                                 w[c0:c1], w[c0 + lag:c1 + lag])
        ce[j] = _masked_normcorr(eb[c0:c1], em[c0 + lag:c1 + lag],
                                 w[c0:c1], w[c0 + lag:c1 + lag])
        cz[j] = _masked_complex_corr(zb[c0:c1], zm[c0 + lag:c1 + lag],
                                     w[c0:c1], w[c0 + lag:c1 + lag])
    return lags, cz, ce, cw


def _parabolic(corr: np.ndarray, lags: np.ndarray):
    j = int(np.nanargmax(corr))
    lag, c = float(lags[j]), float(corr[j])
    if 0 < j < corr.size - 1 and np.isfinite(corr[j - 1]) and np.isfinite(corr[j + 1]):
        y0, y1, y2 = corr[j - 1], corr[j], corr[j + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-12:
            p = 0.5 * (y0 - y2) / d
            if abs(p) <= 1:
                lag += float(p)
                c = float(y1 - 0.25 * (y0 - y2) * p)
    return lag, c


def _window_upsampled_lag(zb, zm, wtr, center: int, half_win: int, k0: int,
                          up: int = 16) -> float:
    """完整有效窗内复互谱上采样，在整数峰 k0 附近取亚采样 lag。

    用整道解析信号按窗切片；频点放到 N 网格 k*up 位置（Nyquist 不重复），
    否则存在固定偏置。
    """
    c0 = center - half_win
    c1 = center + half_win + 1
    if not (wtr[c0:c1].all() and wtr[c0 + k0:c1 + k0].all()):
        return float(k0)
    x = zb[c0:c1]
    y = zm[c0 + k0:c1 + k0]
    x = x - x.mean()
    y = y - y.mean()
    n = x.size
    N = up * n
    S = np.conj(np.fft.fft(x)) * np.fft.fft(y)  # 峰在 +lag（监测晚）
    Rs = np.zeros(N, dtype=complex)
    idx = np.concatenate([np.arange(0, n // 2 + 1), np.arange(n // 2 + 1, n)])
    pos = np.concatenate([np.arange(0, n // 2 + 1),
                          N - np.arange(n // 2 + 1, n)[::-1]])
    Rs[pos] = S[idx]
    cc = np.abs(np.fft.ifft(Rs))
    lags = np.arange(N) / up
    lags[N // 2:] -= n
    sel = np.abs(lags) <= 1.0 + 1.0 / up
    j = int(np.argmax(cc[sel]))
    return float(k0 + lags[sel][j])


def estimate_local_warp(base, monitor, common, cfg, dt: float) -> LocalWarp:
    ni, nx, nt = base.shape
    half_win = max(2, int(round(cfg.warp_window_s / dt)))
    maxlag = max(1, int(round(cfg.warp_search_s / dt)))
    step_t = max(2, half_win)
    sil, sxl = max(1, cfg.warp_anchor_il), max(1, cfg.warp_anchor_xl)
    ils = np.arange(sil // 2, ni, sil)
    xls = np.arange(sxl // 2, nx, sxl)
    tks = np.arange(half_win + maxlag, nt - half_win - maxlag, step_t)
    lag0 = np.full((ils.size, xls.size, tks.size), np.nan)
    wgt = np.zeros_like(lag0)
    corr_anchor = np.zeros((ils.size, xls.size))

    eb_all = np.abs(hilbert(np.nan_to_num(base, nan=0.0), axis=-1))
    em_all = np.abs(hilbert(np.nan_to_num(monitor, nan=0.0), axis=-1))
    zb_all = hilbert(np.nan_to_num(base, nan=0.0), axis=-1)
    zm_all = hilbert(np.nan_to_num(monitor, nan=0.0), axis=-1)

    for ui, il in enumerate(ils):
        for ux, xl in enumerate(xls):
            btr, mtr, wtr = base[il, xl], monitor[il, xl], common[il, xl]
            if wtr.sum() < 16:
                continue
            signals = (zb_all[il, xl], zm_all[il, xl],
                       eb_all[il, xl], em_all[il, xl])
            best = 0.0
            for ut, tk in enumerate(tks):
                res = _window_corr(btr, mtr, wtr, int(tk), half_win, maxlag, signals)
                if res is None or np.isfinite(res[1]).sum() < 3:
                    continue
                lags, c_cx, c_env, c_wave = res
                k0 = int(lags[int(np.nanargmax(c_cx))])
                lag = _window_upsampled_lag(
                    signals[0], signals[1], wtr, int(tk), half_win, k0,
                    cfg.warp_upsample)
                c = float(np.nanmax(c_cx))
                wv = np.nanmax(c_wave) if np.isfinite(c_wave).any() else 0.0
                best = max(best, c, wv)
                ordered = np.sort(c_cx[np.isfinite(c_cx)])
                second = ordered[-2] if ordered.size >= 2 else 0.0
                # 高相关时半采样峰天然与邻峰接近，不能按锐度拒绝；
                # 只在低相关（lag 易歧义）区要求主峰显著压过次峰。
                ambiguous = (c < cfg.warp_strong_corr
                             and c - second < cfg.warp_peak_prominence)
                if (c < cfg.warp_min_corr or abs(lag) > maxlag + 0.5
                        or ambiguous):
                    continue
                lag0[ui, ux, ut] = lag
                wgt[ui, ux, ut] = max(c, 1e-6)
            corr_anchor[ui, ux] = best

    II, XX = np.meshgrid(np.arange(ni), np.arange(nx), indexing='ij')
    rgi_c = RegularGridInterpolator((ils, xls), corr_anchor, bounds_error=False,
                                    fill_value=0.0)
    corr_full = rgi_c(np.stack([II.ravel(), XX.ravel()], 1)).reshape(ni, nx)

    smooth, support = weighted_gaussian(
        lag0, wgt,
        sigma=(cfg.warp_smooth_il, cfg.warp_smooth_xl, cfg.warp_smooth_t),
        min_weight=cfg.warp_min_support)
    low = support < cfg.warp_min_support
    for axis in range(3):
        bad = np.abs(np.diff(smooth, axis=axis)) > cfg.warp_max_jump_s / dt
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None)
        low[tuple(sl)] |= bad
    lag_ok = np.where(low, np.nan, smooth)

    # 时间方向：逐锚点道用最外侧可信锚点向两端延伸（静态边界）
    for u in range(lag_ok.shape[0]):
        for v in range(lag_ok.shape[1]):
            col = lag_ok[u, v]
            good = np.isfinite(col)
            if not good.any():
                continue
            first = int(np.argmax(good))
            last = len(good) - 1 - int(np.argmax(good[::-1]))
            col[:first] = col[first]
            col[last + 1:] = col[last]
            lag_ok[u, v] = col

    rgi = RegularGridInterpolator((ils, xls, tks), np.nan_to_num(lag_ok, nan=0.0),
                                  bounds_error=False, fill_value=np.nan)
    pts = np.stack(np.meshgrid(np.arange(ni), np.arange(nx), np.arange(nt),
                               indexing='ij'), -1)
    lag_full = rgi(pts.reshape(-1, 3)).reshape(ni, nx, nt)
    # 空间锚点包络之外保持 NaN；包络内（含时间延伸）接受
    inside_xy = ((pts[..., 0] >= ils[0]) & (pts[..., 0] <= ils[-1]) &
                 (pts[..., 1] >= xls[0]) & (pts[..., 1] <= xls[-1]))
    had_anchor = np.zeros((ils.size, xls.size), dtype=bool)
    for u in range(lag0.shape[0]):
        for v in range(lag0.shape[1]):
            had_anchor[u, v] = np.isfinite(lag0[u, v]).any()
    rgi_h = RegularGridInterpolator((ils, xls), had_anchor.astype(float),
                                    bounds_error=False, fill_value=0.0)
    supported = rgi_h(np.stack([II.ravel(), XX.ravel()], 1)).reshape(ni, nx) > 0.5
    lag_full = np.where(inside_xy & supported[..., None], lag_full, np.nan)

    return LocalWarp(lag_full, lag_ok, lag0, wgt, corr_full,
                     ~np.isfinite(lag_full))
