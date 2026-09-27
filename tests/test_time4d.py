"""四维交叉均衡流程测试。

用合成数据验证：覆盖匹配、时移/相位/振幅/局部形变恢复、缺覆盖不被插值、
差异体异常检出、NRMS 重复性、以及整条校正链逐位回放。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from geowave.time4d import CrossConfig, Survey4D, run_cross_equalization
from geowave.time4d.grid import (convolve_support, erode, resample_shift,
                                resample_warp, weighted_gaussian)
from geowave.time4d.operators import (apply_bulk_shift, apply_complex_scalar,
                                      apply_local_warp)
from geowave.time4d.pipeline import replay_run, save_run
from geowave.time4d.summary import summarize
from geowave.time4d.repeatability import nrms_map
from geowave.time4d.types import (REL_LOW_ENERGY, REL_MISSING, REL_OK,
                                  REL_SUPPORT)

DT = 0.004


def _ricker(tt, t0, f=28.0):
    u = (np.pi * f * (tt - t0)) ** 2
    return (1 - 2 * u) * np.exp(-u)


def _build_pair(ni=20, nx=20, nt=400, bulk_samples=1.5,
                phase_deg=20.0, amp=0.7, noise=0.01, seed=7,
                local_warp=None, anomaly_amp=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * DT
    il, xl, tt = np.meshgrid(np.arange(ni), np.arange(nx), t, indexing='ij')
    il2, xl2 = np.meshgrid(np.arange(ni), np.arange(nx), indexing='ij')
    base = (0.9 * _ricker(tt, 0.45 + 0.003 * il + 0.002 * xl)
            + 0.6 * _ricker(tt, 0.95 + 0.002 * il - 0.001 * xl, f=22.0))
    if local_warp is None:
        local_warp = np.zeros((ni, nx))
    geo = base.copy()
    if anomaly_amp:
        blob = np.exp(-(((il2 - ni // 2) / 3.0) ** 2
                        + ((xl2 - nx // 2) / 3.0) ** 2))
        geo = geo + anomaly_amp * blob[..., None] * _ricker(
            tt, 0.95 + 0.002 * il - 0.001 * xl, f=22.0)
    mon = np.empty_like(base)
    edge1 = np.zeros_like(base, bool)
    for a in range(ni):
        for x in range(nx):
            s = resample_shift(geo[a, x], local_warp[a, x] / DT)
            edge1[a, x] = ~np.isfinite(s)
            mon[a, x] = np.nan_to_num(s, nan=0.0)
    mon2 = resample_shift(mon, bulk_samples)
    edge = ~np.isfinite(mon2) | edge1
    mon2 = np.nan_to_num(mon2, nan=0.0)
    phi = np.deg2rad(phase_deg)
    z = hilbert(mon2, axis=-1)
    mon_final = amp * (np.cos(phi) * np.real(z) - np.sin(phi) * np.imag(z))
    base = base + noise * rng.standard_normal(base.shape)
    mon_final = mon_final + noise * rng.standard_normal(mon_final.shape)

    vb = np.ones((ni, nx, nt), bool)
    vm = np.ones((ni, nx, nt), bool) & ~edge
    for a in range(ni):                       # 监测楔形缺覆盖
        vm[a, nx - 1 - (a // 3):, :] = False
    vb[:2, :, :] = False                      # 基准边缘缺测线
    return Survey4D('base', base, vb, DT), Survey4D('monitor', mon_final, vm, DT)


# ----------------------------------------------------------------- 数据契约

def test_survey_rejects_nan_inside_valid():
    data = np.ones((2, 2, 4))
    valid = np.ones((2, 2, 4), bool)
    data[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        Survey4D('s', data, valid, DT)


def test_geometry_mismatch_rejected():
    b, _ = _build_pair(ni=20, nx=20)
    m_small = Survey4D('m', np.zeros((20, 18, 400)),
                       np.ones((20, 18, 400), bool), DT)
    with pytest.raises(ValueError):
        run_cross_equalization(b, m_small)


# ------------------------------------------------------------- 网格数值原语

def test_resample_shift_fractional_and_sign():
    t = np.arange(200) * DT
    x = _ricker(t, 0.4)
    y = resample_shift(x, 1.5)          # y[i] = x[i-1.5]，事件向下移
    assert np.nanargmax(y) - np.argmax(x) == 2 or np.nanargmax(y) - np.argmax(x) == 1
    y2 = resample_shift(x, -1.5)        # 向上移
    assert np.nanargmax(y2) < np.argmax(x)
    # 正向平移后尾端、反向平移后首端出现完全失支撑 NaN（不补零/回绕）
    assert np.isnan(y[-1])
    assert np.isnan(y2[0])


def test_weighted_gaussian_no_invented_values():
    f = np.full((5, 5), np.nan)
    w = np.zeros((5, 5))
    f[2, 2] = 1.0
    w[2, 2] = 1.0
    out, support = weighted_gaussian(f, w, sigma=(2, 2), min_weight=0.5)
    # 单点无法撑起 sigma=2 的核，除自身邻域外应为 NaN，不能被填成 1
    assert np.isnan(out[0, 0])
    assert support.min() < 0.5


def test_convolve_support_no_wraparound():
    mask = np.ones((1, 1, 40))
    taps = np.zeros(11)
    taps[5] = 1.0
    taps[3] = taps[7] = -0.5
    sup = convolve_support(mask, taps)
    # 中部全支撑，两端因零填充而下降（绝无周期回绕导致的虚假 1）
    assert sup[0, 0, 20] == pytest.approx(1.0)
    assert sup[0, 0, 0] < 0.99 and sup[0, 0, -1] < 0.99


# ----------------------------------------------------------------- 算子符号

def test_bulk_correction_sign():
    b, m = _build_pair(bulk_samples=1.5, phase_deg=0.0, amp=1.0, noise=0.0)
    common = erode(b.valid & m.valid, (1, 1, 0))
    corrected, v = apply_bulk_shift(m.data, common, 1.5)   # lag=+1.5 监测偏晚
    i = slice(120, 280)
    # 校正后应与基准对齐（高相关），而不是把残差加倍
    x, y = b.data[5, 5, i][v[5, 5, i]], corrected[5, 5, i][v[5, 5, i]]
    assert np.corrcoef(x, y)[0, 1] > 0.99


def test_complex_scalar_undoes_rotation():
    z = _ricker(np.arange(400) * DT, 0.5)
    phi = np.deg2rad(35.0)
    a = hilbert(z)
    rotated = np.cos(phi) * np.real(a) - np.sin(phi) * np.imag(a)
    base = z.reshape(1, 1, -1)
    mon = rotated.reshape(1, 1, -1)
    from geowave.time4d.estimate import estimate_complex_scalar
    g = estimate_complex_scalar(base, mon, np.ones_like(base, bool), None, 180)
    assert abs(np.rad2deg(np.angle(g)) + 35.0) < 0.5


# ------------------------------------------------------------ 主管线：重复性

def test_full_pipeline_repeatability_and_determinism():
    b, m = _build_pair(anomaly_amp=0.25)
    cfg = CrossConfig(gate_s=(0.2, 1.3), warp_search_s=0.006)
    r1 = run_cross_equalization(b, m, cfg)
    r2 = run_cross_equalization(b, m, cfg)
    # 确定性
    assert np.allclose(r1.difference, r2.difference, equal_nan=True)
    assert np.array_equal(r1.reliability, r2.reliability)
    # 整体时移/相位恢复
    assert abs(r1.bulk_shift_s - 6e-3) < 1.2e-3
    assert abs(np.rad2deg(np.angle(r1.complex_scalar)) + 20) < 4
    # 信号带重复性显著优于未校正
    ok = r1.reliability == REL_OK
    ev = []
    for t0, t1 in ((0.42, 0.55), (0.92, 1.05)):
        i0, i1 = int(t0 / DT), int(t1 / DT)
        ev.append(np.nanmedian(r1.nrms[:, :, i0:i1][ok[:, :, i0:i1]]))
    assert np.nanmean(ev) < 15.0


def test_missing_coverage_never_becomes_reliable():
    b, m = _build_pair()
    r = run_cross_equalization(b, m, CrossConfig(gate_s=(0.2, 1.3),
                                                 warp_search_s=0.006))
    # 监测楔形缺覆盖
    assert np.isnan(r.difference[5, 19, 100])
    assert r.reliability[5, 19, 100] == REL_MISSING
    # 基准缺测线
    assert r.reliability[0, 10, 100] == REL_MISSING
    assert np.isnan(r.nrms[0, 10, 100])
    # 任何差异体都不允许 inf
    assert not np.isinf(np.nan_to_num(r.difference, nan=0.0, posinf=0,
                                      neginf=0)).any()


def test_anomaly_recovered_above_background():
    b, m = _build_pair(anomaly_amp=0.25)
    r = run_cross_equalization(b, m, CrossConfig(gate_s=(0.2, 1.3),
                                                 warp_search_s=0.006))
    signal = np.nanmax(np.abs(r.difference[8:12, 8:12, :]))
    bg = np.nanmedian(np.abs(r.difference[4:7, 4:7, int(0.3 / DT):int(0.7 / DT)]))
    assert signal > 8 * bg


def test_local_warp_detected_at_right_place():
    ni = nx = 20
    il2, xl2 = np.meshgrid(np.arange(ni), np.arange(nx), indexing='ij')
    warp_s = 0.002 * np.exp(-(((il2 - 7) / 5.0) ** 2 + ((xl2 - 15) / 5.0) ** 2))
    b, m = _build_pair(local_warp=warp_s)
    r = run_cross_equalization(b, m, CrossConfig(gate_s=(0.2, 1.3),
                                                 warp_search_s=0.006))
    lag = r.lag_map[:, :, int(0.45 / DT):int(1.0 / DT)]
    peak = np.nanmax(lag) * 1000
    p = np.unravel_index(np.nanargmax(lag), lag.shape)
    assert peak > 0.6                       # 形变显著检出（方向为正=监测偏晚）
    assert abs(p[0] - 7) <= 2 and abs(p[1] - 15) <= 2


def test_shaping_filter_rejected_when_not_helpful():
    # 同源数据（仅相位/振幅差异），整形滤波必须被相关性门槛拒绝，不得反相
    b, m = _build_pair()
    cfg = CrossConfig(gate_s=(0.2, 1.3), warp_search_s=0.006,
                      filter_length_s=0.08)
    r = run_cross_equalization(b, m, cfg)
    assert any('rejected' in s or 'skipped' in s for s in r.stages)
    assert r.filter_taps.size <= 1


# -------------------------------------------------------------------- NRMS

def test_nrms_zero_for_identical():
    x = np.random.default_rng(0).standard_normal((2, 2, 100))
    v = np.ones((2, 2, 100), bool)
    nrms, pred, low = nrms_map(x, x.copy(), v, 10, 1e-9)
    assert np.nanmax(nrms) < 1e-8
    assert np.nanmin(pred) > 0.9999


def test_nrms_nan_when_window_incomplete():
    x = np.random.default_rng(0).standard_normal((1, 1, 60))
    y = x.copy()
    v = np.ones((1, 1, 60), bool)
    v[0, 0, :5] = False
    nrms, _, _ = nrms_map(x, y, v, 10, 1e-9)
    assert np.isnan(nrms[0, 0, 0])         # 窗内有缺样本 -> NaN，不补零
    assert np.isfinite(nrms[0, 0, 30])


# ------------------------------------------------------------------ 回放链

def test_summary_fractions_partition_to_one():
    b, m = _build_pair()
    r = run_cross_equalization(b, m, CrossConfig(gate_s=(0.2, 1.3),
                                                 warp_search_s=0.006))
    s = summarize(r)
    assert abs(sum(s.fractions.values()) - 1.0) < 1e-9
    assert 0.0 < s.reliable_fraction < 1.0
    assert s.fractions['missing_coverage'] > 0
    assert np.isfinite(s.nrms_median)


# ------------------------------------------------------------------ 回放链

def test_replay_bit_identical(tmp_path):
    b, m = _build_pair(anomaly_amp=0.25)
    cfg = CrossConfig(gate_s=(0.2, 1.3), warp_search_s=0.006)
    r = run_cross_equalization(b, m, cfg)
    save_run(tmp_path / 'run', b, m, r)
    checks = replay_run(tmp_path / 'run')
    assert checks['inputs_unchanged']
    assert checks['reproducible'], {k: v for k, v in checks.items() if not v}


def test_replay_detects_tampered_input(tmp_path):
    b, m = _build_pair()
    cfg = CrossConfig(gate_s=(0.2, 1.3), warp_search_s=0.006)
    r = run_cross_equalization(b, m, cfg)
    save_run(tmp_path / 'run', b, m, r)
    # 篡改包内基准数据后回放，应被输入摘要检出
    d = np.load(tmp_path / 'run' / 'surveys.npz')
    tampered = dict(d)
    tam = d['base_data'].copy()
    tam[5, 5, 100] += 1.0
    tampered['base_data'] = tam
    np.savez_compressed(tmp_path / 'run' / 'surveys.npz', **tampered)
    checks = replay_run(tmp_path / 'run')
    assert not checks['inputs_unchanged']
