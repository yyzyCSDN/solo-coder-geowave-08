"""参数估计：整体时移、联合相位/振幅、整形滤波器、空变振幅。

只在两期共同有效样本上统计；所有估计函数不修改输入。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from geowave.time4d.grid import weighted_gaussian


def _gate(gate, nt):
    return gate if gate is not None else (0, nt)


def _overlap(b: np.ndarray, m: np.ndarray, lag: int):
    if lag >= 0:
        return b[:b.size - lag or None], m[lag:]
    return b[-lag:], m[:m.size + lag]


def _complex_coarse_lag(b: np.ndarray, m: np.ndarray, maxlag: int) -> tuple[int, float]:
    """复数解析信号互相关取模定整数峰（相位不变性）。"""
    zb = _hilbert_1d(b)
    zm = _hilbert_1d(m)
    best_k, best_c = 0, -1.0
    for lag in range(-maxlag, maxlag + 1):
        x, y = _overlap(zb, zm, lag)
        den = np.linalg.norm(x) * np.linalg.norm(y)
        c = abs(np.vdot(x, y)) / den if den > 0 else np.nan
        if c > best_c:
            best_k, best_c = lag, c
    return best_k, float(best_c)


def _hilbert_1d(x: np.ndarray) -> np.ndarray:
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def _complex_upsampled_lag(b: np.ndarray, m: np.ndarray, k0: int,
                           up: int = 16, half: int = 1) -> float:
    """在整数峰 k0±half 范围内，用复数互谱频域升采样取亚采样 lag。

    关键：补零升采样必须把原谱频点放到 N 网格的 k*up 位置，
    不能简单前缀复制（Nyquist 两侧位置不同，否则引入固定偏置）。
    """
    bb, mm = _overlap(b, m, k0)
    n = bb.size
    N = up * n
    Zb = np.fft.fft(_hilbert_1d(bb))
    Zm = np.fft.fft(_hilbert_1d(mm))
    S = np.conj(Zb) * Zm  # C[lag]=Σ z_b*[t] z_m[t+lag]，峰在 +lag（m 晚）
    Rs = np.zeros(N, dtype=complex)
    idx = np.concatenate([np.arange(0, n // 2 + 1), np.arange(n // 2 + 1, n)])
    pos = np.concatenate([np.arange(0, n // 2 + 1),
                          N - np.arange(n // 2 + 1, n)[::-1]])
    Rs[pos] = S[idx] * up
    cc = np.abs(np.fft.ifft(Rs))
    lags = np.arange(N) / up
    lags[N // 2:] -= n
    sel = np.abs(lags) <= half + 1.0 / up
    j = int(np.argmax(cc[sel]))
    return float(k0 + lags[sel][j])


def estimate_bulk_shift(base, monitor, common, gate, max_lag_samples: int,
                        robust_quantile: float) -> tuple[float, float]:
    """逐道复数解析信号定整体时移：整数峰取模（抗相位）+ 频域上采样细化。"""
    lags, scores = [], []
    ni, nx, nt = base.shape
    i0, i1 = _gate(gate, nt)
    for il in range(ni):
        for xl in range(nx):
            w = common[il, xl, i0:i1]
            if w.sum() < 16 or not w.all():
                continue
            b = base[il, xl, i0:i1]
            m = monitor[il, xl, i0:i1]
            maxlag = min(max_lag_samples, b.size // 4)
            if maxlag < 1:
                continue
            k0, c = _complex_coarse_lag(b, m, maxlag)
            lag = _complex_upsampled_lag(b, m, k0)
            lags.append(lag)
            scores.append(c)
    if not lags:
        return 0.0, 0.0
    lags = np.asarray(lags)
    scores = np.asarray(scores)
    # 稳健合成：先按相关度去底（保留上半），再按 |lag-中位| 的 MAD 去离群，
    # 不能只挑高相关道——高相关与 lag 一致性不是一回事。
    keep = scores >= np.quantile(scores, 0.5)
    med = np.median(lags[keep])
    mad = np.median(np.abs(lags[keep] - med)) + 1e-9
    inlier = keep & (np.abs(lags - med) < 3.0 * 1.4826 * mad)
    if inlier.sum() < 3:
        inlier = keep
    return float(np.median(lags[inlier])), float(np.median(scores[inlier]))


def _hilbert(x: np.ndarray) -> np.ndarray:
    """最后轴解析信号；输入 NaN 处置零，调用方自行限制有效样本。"""
    n = x.shape[-1]
    X = np.fft.fft(np.where(np.isnan(x), 0.0, x), axis=-1)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h, axis=-1)


def _clip_phase(g: complex, max_deg: float) -> complex:
    mag = abs(g)
    ang = np.angle(g)
    limit = np.deg2rad(max_deg)
    if abs(ang) > limit:
        ang = np.sign(ang) * limit
    return mag * np.exp(1j * ang)


def estimate_complex_scalar(base, monitor, common, gate, phase_max_deg: float) -> complex:
    """解析信号域最小二乘 g，使 Re(g*analytic(m)) ≈ b（联合相位+振幅）。"""
    nt = base.shape[2]
    i0, i1 = _gate(gate, nt)
    w = common[:, :, i0:i1]
    am = _hilbert(monitor)
    z = am[:, :, i0:i1][w]
    b = base[:, :, i0:i1][w]
    denom = np.vdot(z, z).real
    g = np.vdot(z, b) / denom if denom > 0 else 1 + 0j
    return _clip_phase(complex(g), phase_max_deg)


def estimate_shaping_filter(base, monitor, common, gate, half_taps: int,
                            prewhitening: float, min_energy: float) -> np.ndarray:
    """零相位整形滤波器（监测->基准），对逐道相关做集成平均后求解。

    不能把所有道拼成一条长序列直接解：门内稀疏反射会使自相关阵近乎奇异，
    得到振荡甚至反相的滤波器。改为逐道（共同有效段）计算自相关/互相关再
    平均（标准 4D 整形法），零相位约束 h[k]=h[-k] 不引入残余时移；
    相对尺度不归一化，交给后续空变振幅平衡。
    返回居中零相位抽头；关闭或能量不足时返回长度 0 数组。
    """
    if half_taps < 1:
        return np.zeros(0)
    nt = base.shape[2]
    i0, i1 = _gate(gate, nt)
    L2 = half_taps + 1
    rmm_sum = np.zeros(L2)
    rbm_sum = np.zeros(L2)
    used = 0
    ni, nx = base.shape[0], base.shape[1]
    for il in range(ni):
        for xl in range(nx):
            w = common[il, xl, i0:i1]
            if not w.all():
                continue
            b = base[il, xl, i0:i1]
            m = monitor[il, xl, i0:i1]
            nfft = 1
            while nfft < m.size:
                nfft *= 2
            B = np.fft.rfft(b, n=nfft)
            M = np.fft.rfft(m, n=nfft)
            scale = m.size
            rmm_sum += np.fft.irfft(M * np.conj(M), n=nfft)[:L2].real / scale
            cbm = np.fft.irfft(np.conj(M) * B, n=nfft)
            rbm_sum[0] += cbm[0].real / scale
            rbm_sum[1:] += 0.5 * (cbm[1:L2].real
                                  + cbm[nfft - half_taps:nfft].real) / scale
            used += 1
    if used < 4 or rmm_sum[0] < min_energy:
        return np.zeros(0)
    rmm = rmm_sum / used
    rbm = rbm_sum / used
    from scipy.linalg import toeplitz
    eps = prewhitening * rmm[0]
    try:
        h = np.linalg.solve(toeplitz(rmm) + eps * np.eye(L2), rbm)
    except np.linalg.LinAlgError:
        return np.zeros(0)
    taps = np.concatenate([h[:0:-1], h])  # 居中零相位
    taps *= np.hanning(taps.size + 2)[1:-1]
    return taps


def estimate_amp_map(base, monitor, common, gate, step_il: int, step_xl: int,
                     smooth_il: float, smooth_xl: float, gain_clip: float,
                     min_energy: float) -> np.ndarray:
    """锚点 RMS 比 -> 归一化高斯平滑 -> 全线双线性增益图。

    比值只在锚点块内实际共同有效的样本上统计，权重 = 块内覆盖比例；
    归一化平滑允许局部锚点空洞（不补值），但平滑核覆盖率低于一半的位置
    输出 NaN；全线插值落在锚点包络外同样为 NaN。
    """
    ni, nx, nt = base.shape
    if step_il <= 0 or step_xl <= 0:
        return np.ones((ni, nx))
    i0, i1 = _gate(gate, nt)
    w = common[:, :, i0:i1]
    block_n = step_il * step_xl * (i1 - i0)
    b2_all = np.where(w, base[:, :, i0:i1] ** 2, 0.0)
    m2_all = np.where(w, monitor[:, :, i0:i1] ** 2, 0.0)
    ils = np.arange(step_il // 2, ni, step_il)
    xls = np.arange(step_xl // 2, nx, step_xl)
    ratio = np.full((ils.size, xls.size), np.nan)
    wt = np.zeros((ils.size, xls.size))
    min_count = int(0.25 * (step_il * step_xl) * (i1 - i0))
    for u, il in enumerate(ils):
        for v, xl in enumerate(xls):
            sli = slice(max(0, il - step_il // 2), min(ni, il + step_il // 2 + 1))
            slx = slice(max(0, xl - step_xl // 2), min(nx, xl + step_xl // 2 + 1))
            count = w[sli, slx].sum()
            if count < min_count:
                continue
            mb = b2_all[sli, slx].sum() / count
            mm = m2_all[sli, slx].sum() / count
            if mm > min_energy and mb > 0:
                ratio[u, v] = np.sqrt(mb / mm)
                wt[u, v] = count / block_n
    r_s, _ = weighted_gaussian(np.clip(ratio, 1.0 / gain_clip, gain_clip),
                               np.maximum(wt, 1e-12),
                               sigma=(smooth_il, smooth_xl), min_weight=0.35)
    fill = np.nan_to_num(r_s, nan=1.0)
    rgi = RegularGridInterpolator((ils, xls), fill,
                                  bounds_error=False, fill_value=np.nan)
    II, XX = np.meshgrid(np.arange(ni), np.arange(nx), indexing='ij')
    gain = rgi(np.stack([II.ravel(), XX.ravel()], axis=1)).reshape(ni, nx)
    # RGI 在锚点外给 NaN；内部曾被 nan_to_num 成 1 的位置必须改回 NaN，
    # 不允许把“无估计”静默变成单位增益
    missing_anchor = ~np.isfinite(r_s)
    rgi_mask = RegularGridInterpolator((ils, xls), missing_anchor.astype(float),
                                       bounds_error=False, fill_value=1.0)
    inside_missing = rgi_mask(np.stack([II.ravel(), XX.ravel()], axis=1)
                              ).reshape(ni, nx) > 0.5
    gain[inside_missing] = np.nan
    return gain
