"""交叉均衡主管线与回放。

阶段顺序：
1. coverage  覆盖交集 + 边界侵蚀，共同区之外一律缺覆盖
2. bulk      整体时移（整数+亚采样 sinc）
3. phase     联合相位/振幅复数标量
4. filter    零相位整形滤波器（频带/子波匹配）
5. amp       空变振幅平衡（锚点 RMS 比）
6. warp      局部形变（滑窗相关 + 锚点平滑 + sinc 重采样）
7. products  差异体、NRMS、可预测性、可靠性图

每个阶段更新 (数据, 有效掩码)；所有算子记录在 operators 里，
replay 时从原始监测体重放整条链并逐位比对，确保校正可回放。
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from geowave.time4d.config import CrossConfig
from geowave.time4d.estimate import (estimate_amp_map, estimate_bulk_shift,
                                     estimate_complex_scalar,
                                     estimate_shaping_filter)
from geowave.time4d.grid import convolve_support, erode
from geowave.time4d.operators import (apply_amp_map, apply_bulk_shift,
                                      apply_complex_scalar, apply_filter,
                                      apply_local_warp)
from geowave.time4d.repeatability import nrms_map
from geowave.time4d.summary import summarize
from geowave.time4d.types import (REL_LOW_ALIGN, REL_LOW_ENERGY, REL_MISSING,
                                  REL_OK, REL_SUPPORT, CrossResult, Survey4D)
from geowave.time4d.warp import estimate_local_warp


def _gate_indices(cfg: CrossConfig, nt: int, dt: float, t0: float) -> tuple[int, int]:
    if cfg.gate_s is None:
        return 0, nt
    g0 = int(round((cfg.gate_s[0] - t0) / dt))
    g1 = int(round((cfg.gate_s[1] - t0) / dt))
    return max(0, g0), min(nt, g1)


def _gate_corr(base: np.ndarray, mon: np.ndarray, valid: np.ndarray,
               gate: tuple[int, int]) -> float:
    """时窗内两期体的整体归一化相关（整形滤波器接受/拒绝的判据）。"""
    i0, i1 = gate
    w = valid[:, :, i0:i1]
    b = base[:, :, i0:i1][w]
    m = mon[:, :, i0:i1][w]
    if b.size < 16:
        return 0.0
    b = b - b.mean()
    m = m - m.mean()
    den = np.linalg.norm(b) * np.linalg.norm(m)
    return float(np.dot(b, m) / den) if den > 0 else 0.0


def _check_geometry(base: Survey4D, monitor: Survey4D) -> None:
    if base.shape != monitor.shape:
        raise ValueError(f'两期网格形状不一致: {base.shape} vs {monitor.shape}')
    if abs(base.dt - monitor.dt) > 1e-12 or base.origin != monitor.origin \
            or base.spacing != monitor.spacing:
        raise ValueError('两期网格几何（dt/origin/spacing）不一致，需先规则化到同一网格')


def run_cross_equalization(base: Survey4D, monitor: Survey4D,
                           cfg: CrossConfig | None = None) -> CrossResult:
    cfg = cfg or CrossConfig()
    _check_geometry(base, monitor)
    ni, nx, nt = base.shape
    dt = base.dt
    gate = _gate_indices(cfg, nt, dt, base.origin[2])
    stages: list[str] = []
    operators: dict = {}

    # 1 ---- 覆盖匹配：原始交集（用于归档）+ 保守侵蚀（用于全部估计与产物）
    common_raw = base.valid & monitor.valid
    common = erode(common_raw, cfg.coverage_erode)
    b = np.where(common, base.data, np.nan)
    m = np.where(common, monitor.data, np.nan)
    valid = common.copy()
    stages.append('coverage')

    # 2 ---- 整体时移
    max_lag = int(round(cfg.max_bulk_shift_s / dt))
    bulk_samples, bulk_score = estimate_bulk_shift(
        base.data, monitor.data, common, gate, max_lag, cfg.robust_quantile)
    bulk_shift_s = bulk_samples * dt
    m, valid = apply_bulk_shift(monitor.data, valid, bulk_samples)
    stages.append('bulk')
    operators['bulk_shift_samples'] = float(bulk_samples)

    # 3 ---- 联合相位/振幅
    g = estimate_complex_scalar(base.data, m, valid, gate, cfg.phase_max_deg)
    m = apply_complex_scalar(np.where(valid, m, np.nan), g)
    m = np.where(valid, m, np.nan)
    stages.append('phase')
    operators['complex_scalar'] = {'real': float(g.real), 'imag': float(g.imag)}

    # 4 ---- 整形滤波器（零相位）
    half_taps = int(round(cfg.filter_length_s / dt))
    taps = estimate_shaping_filter(
        base.data, m, valid, gate, half_taps,
        cfg.filter_prewhitening, cfg.filter_min_energy)
    applied_taps = taps
    if taps.size > 1:
        # 先试应用：整形滤波器必须真正提升与基准的相关，否则拒绝并原样回滚，
        # 绝不能在已经高相关的数据上套一个过拟合/反相滤波器。
        trial = apply_filter(np.where(valid, m, np.nan), taps)
        w = valid.copy()
        w[..., :taps.size // 2] = False
        w[..., nt - taps.size // 2:] = False
        c_before = _gate_corr(base.data, m, w, gate)
        c_after = _gate_corr(base.data, trial, w, gate)
        if c_after - c_before >= cfg.filter_min_corr_gain:
            support = convolve_support(valid.astype(np.float64), taps)
            valid = valid & (support > 0.999)
            m = np.where(valid, trial, np.nan)
            stages.append(f'filter(corr {c_before:.3f}->{c_after:.3f})')
            operators['filter_taps'] = taps.tolist()
        else:
            # 拒绝：结果链里不得残留该滤波器（落盘 taps 也置空，保证可回放）
            applied_taps = np.zeros(0)
            stages.append(
                f'filter(rejected:corr {c_before:.3f}->{c_after:.3f})')
    else:
        stages.append('filter(skipped:disabled/low-energy)')

    # 5 ---- 空变振幅
    gain = estimate_amp_map(base.data, m, valid, gate,
                            cfg.amp_anchor_il, cfg.amp_anchor_xl,
                            cfg.amp_smooth_il, cfg.amp_smooth_xl,
                            cfg.amp_gain_clip, cfg.amp_min_energy)
    amp_ok = np.isfinite(gain)
    valid3 = valid & amp_ok[..., None]
    m = np.where(valid3, apply_amp_map(m, np.where(amp_ok, gain, 1.0)), np.nan)
    valid = valid3
    stages.append('amp')
    operators['amp_gain_map'] = gain  # 与结果体一起落盘

    # 6 ---- 局部形变
    v_pre_warp = valid.copy()
    warp = estimate_local_warp(np.where(v_pre_warp, b, np.nan),
                               np.where(v_pre_warp, m, np.nan), v_pre_warp, cfg, dt)
    lag_samples = warp.lag_full
    valid = v_pre_warp & ~warp.low_align & np.isfinite(lag_samples)
    m_corr = apply_local_warp(np.where(valid, m, np.nan),
                              np.where(valid, lag_samples, 0.0))
    m = np.where(valid, m_corr, np.nan)
    stages.append('warp')
    operators['lag_map_samples'] = lag_samples

    # 7 ---- 差异体与重复性图
    diff = np.where(valid, b - m, np.nan)
    diff_raw = np.where(common_raw, base.data - monitor.data, np.nan)

    half_nrms = max(1, int(round(cfg.nrms_window_s / dt)))
    nrms, pred, low_energy = nrms_map(b, m, valid, half_nrms, cfg.nrms_min_rms)

    # ---- 可靠性图（优先级：缺覆盖 > 支撑 > 低相关 > 低能量）
    reliability = np.full((ni, nx, nt), REL_OK, dtype=np.uint8)
    reliability[~common_raw] = REL_MISSING
    reliability[common_raw & ~v_pre_warp] = REL_SUPPORT
    reliability[v_pre_warp & warp.low_align] = REL_LOW_ALIGN
    reliability[valid & low_energy] = REL_LOW_ENERGY
    ok = reliability == REL_OK
    nrms = np.where(ok, nrms, np.nan)
    pred = np.where(ok, pred, np.nan)
    diff = np.where(ok, diff, np.nan)
    stages.append('products')

    result = CrossResult(
        base_matched=np.where(common_raw, base.data, np.nan),
        monitor_matched=np.where(ok, m, np.nan),
        difference=diff,
        difference_raw=diff_raw,
        nrms=nrms,
        predictability=pred,
        reliability=reliability,
        valid_common=common_raw,
        bulk_shift_s=float(bulk_shift_s),
        bulk_score=float(bulk_score),
        complex_scalar=g,
        filter_taps=applied_taps,
        amp_map=gain,
        lag_map=np.where(np.isfinite(lag_samples), lag_samples * dt, np.nan),
        lag_anchor=warp.lag_anchor * dt,
        lag_corr=warp.corr,
        stages=stages,
    )
    result.operators = operators
    result.gate = gate
    result.config = cfg
    return result


# ---------------------------------------------------------------- 落盘/回放

def _array_digest(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save_run(path: str | Path, base: Survey4D, monitor: Survey4D,
             result: CrossResult) -> Path:
    """保存可回放包：原始两期、全部算子、全部产物、配置、输入摘要。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg: CrossConfig = result.config
    manifest = {
        'config': cfg.to_dict(),
        'stages': result.stages,
        'gate_samples': list(result.gate),
        'geometry': {
            'dt': base.dt,
            'origin': list(base.origin),
            'spacing': list(base.spacing),
            'shape': list(base.shape),
        },
        'scalars': {
            'bulk_shift_s': result.bulk_shift_s,
            'bulk_score': result.bulk_score,
            'complex_scalar': {'real': result.complex_scalar.real,
                               'imag': result.complex_scalar.imag},
        },
        'summary': summarize(result).as_dict(),
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
        },
        'inputs_digest': {
            'base_data': _array_digest(base.data),
            'base_valid': _array_digest(base.valid),
            'monitor_data': _array_digest(monitor.data),
            'monitor_valid': _array_digest(monitor.valid),
        },
    }
    (path / 'manifest.json').write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2))
    np.savez_compressed(path / 'surveys.npz',
                        base_data=base.data, base_valid=base.valid,
                        monitor_data=monitor.data, monitor_valid=monitor.valid)
    np.savez_compressed(path / 'operators.npz',
                        filter_taps=result.filter_taps,
                        amp_map=result.amp_map,
                        lag_map_samples=result.operators['lag_map_samples'],
                        bulk_shift_samples=result.operators['bulk_shift_samples'])
    np.savez_compressed(
        path / 'products.npz',
        base_matched=result.base_matched, monitor_matched=result.monitor_matched,
        difference=result.difference, difference_raw=result.difference_raw,
        nrms=result.nrms, predictability=result.predictability,
        reliability=result.reliability, valid_common=result.valid_common,
        lag_map=result.lag_map, lag_anchor=result.lag_anchor, lag_corr=result.lag_corr)
    return path


def _load_surveys(path: Path):
    d = np.load(path / 'surveys.npz')
    meta = json.loads((path / 'manifest.json').read_text())['geometry']
    base = Survey4D('base', d['base_data'], d['base_valid'], meta['dt'],
                    tuple(meta['origin']), tuple(meta['spacing']))
    mon = Survey4D('monitor', d['monitor_data'], d['monitor_valid'], meta['dt'],
                   tuple(meta['origin']), tuple(meta['spacing']))
    return base, mon


def replay_run(path: str | Path) -> dict:
    """从包内原始两期与配置重跑整条链，逐位比对全部产物与算子。

    返回 {名称: bool}；任何一项 False 都说明该次校正不能忠实回放。
    """
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    cfg = CrossConfig.from_dict(manifest['config'])
    base, mon = _load_surveys(path)
    actual_inputs = {
        'base_data': _array_digest(base.data),
        'base_valid': _array_digest(base.valid),
        'monitor_data': _array_digest(mon.data),
        'monitor_valid': _array_digest(mon.valid),
    }
    checks: dict = {}
    checks['inputs_unchanged'] = actual_inputs == manifest['inputs_digest']
    rerun = run_cross_equalization(base, mon, cfg)
    stored = np.load(path / 'products.npz')
    stored_op = np.load(path / 'operators.npz')
    pairs = {
        'difference': (rerun.difference, stored['difference']),
        'monitor_matched': (rerun.monitor_matched, stored['monitor_matched']),
        'nrms': (rerun.nrms, stored['nrms']),
        'predictability': (rerun.predictability, stored['predictability']),
        'reliability': (rerun.reliability, stored['reliability']),
        'lag_map': (rerun.lag_map, stored['lag_map']),
        'amp_map': (rerun.amp_map, stored_op['amp_map']),
        'filter_taps': (rerun.filter_taps.astype(np.float64)
                        if rerun.filter_taps.size else np.zeros(0),
                        stored_op['filter_taps'].astype(np.float64)
                        if stored_op['filter_taps'].size else np.zeros(0)),
    }
    nan_same = lambda a, b: (a.shape == b.shape and
                             np.array_equal(np.isnan(a) if np.issubdtype(a.dtype, np.floating)
                                            else np.zeros(a.shape, bool),
                                            np.isnan(b) if np.issubdtype(b.dtype, np.floating)
                                            else np.zeros(b.shape, bool)))
    for name, (a, b) in pairs.items():
        if a.shape != b.shape:
            checks[name] = False
            continue
        if np.issubdtype(a.dtype, np.floating):
            ok = nan_same(a, b) and np.allclose(a, b, rtol=1e-12, atol=1e-12,
                                                equal_nan=True)
        else:
            ok = np.array_equal(a, b)
        checks[name] = bool(ok)
    checks['scalars'] = (rerun.bulk_shift_s == manifest['scalars']['bulk_shift_s']
                         and abs(rerun.bulk_score
                                 - manifest['scalars']['bulk_score']) < 1e-15)
    checks['reproducible'] = all(checks.values())
    return checks
