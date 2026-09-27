from __future__ import annotations

import numpy as np

from geowave.time_lapse.cross_equalization import (
    CrossEqualizationConfig,
    FourDVolume,
    apply_manifest,
    cross_equalize,
    load_manifest,
    match_coverage,
    replay_cross_equalization,
    save_manifest,
)
from geowave.time_lapse.workflow import four_d_workflow, replay_four_d_workflow


def _synthetic_pair():
    rng = np.random.default_rng(8241)
    nil, nxl, nt = 10, 8, 180
    dt = 0.004
    t = np.arange(nt) * dt
    inline = np.arange(nil, dtype=float)
    crossline = np.arange(nxl, dtype=float)
    tt, yy = np.meshgrid(t, np.arange(nxl), indexing='ij')
    trace_shape = (nil, nxl, nt)

    base = np.zeros(trace_shape)
    monitor = np.zeros(trace_shape)
    for ix in range(nil):
        center = 0.30 + 0.003 * (ix - 4) + 0.002 * (yy - 3)
        envelope = np.exp(-0.5 * ((tt.T - center.T) / 0.018) ** 2)
        phase_true = 2.0 * np.pi * 25.0 * (tt.T - center.T)
        signal = np.cos(phase_true) * envelope
        base[ix] = signal
        shifted_center = center + 0.010 + 0.004 * np.sin(np.pi * ix / nil) * (yy / nxl)
        phase_mon = 2.0 * np.pi * 25.0 * (tt.T - shifted_center.T - np.radians(20.0) / (2.0 * np.pi * 25.0))
        monitor[ix] = 1.40 * np.cos(phase_mon) * envelope

    base += 0.01 * rng.standard_normal(trace_shape)
    monitor += 0.01 * rng.standard_normal(trace_shape)
    ref_footprint = np.ones((nil, nxl), dtype=bool)
    mon_footprint = np.ones((nil, nxl), dtype=bool)
    mon_footprint[0, :] = False
    mon_footprint[-2:, -1] = False
    monitor[:, ~mon_footprint, :] = np.nan
    return (
        FourDVolume(inline, crossline, t, base, ref_footprint, 'baseline'),
        FourDVolume(inline, crossline, t, monitor, mon_footprint, 'monitor'),
        mon_footprint,
    )


def test_coverage_match_preserves_missing_traces_without_interpolation():
    reference, monitor, mon_footprint = _synthetic_pair()
    matched = match_coverage(reference, monitor)

    assert matched.common_mask.shape == reference.data.shape
    assert not matched.common_mask[:, ~mon_footprint, :].any()
    assert matched.reference_only_mask[:, 0, :].all()
    assert np.isnan(matched.monitor.data[:, 0, :]).all()
    assert matched.common_fraction < 1.0
    assert np.isfinite(matched.monitor.data[:, 1, :]).all()


def test_cross_equalization_recovers_corrections_and_reduces_nrms():
    reference, monitor, _ = _synthetic_pair()
    config = CrossEqualizationConfig(
        stable_start_s=0.18,
        stable_end_s=0.46,
        max_bulk_shift_s=0.03,
        warp_step_s=0.02,
        warp_smooth=(3, 3, 5),
        max_local_shift_s=0.008,
    )
    result = cross_equalize(reference, monitor, config)

    assert abs(result.time_shift_s - 0.010) < 0.003
    assert abs(result.phase_shift_deg - 20.0) < 5.0
    assert abs(result.amplitude_scale - 1.0 / 1.4) < 0.08
    assert result.median_nrms_after < result.median_nrms_before
    assert np.isnan(result.difference.data[:, 0, :]).all()
    assert np.isfinite(result.difference.data[4:7, 1:5, 60:100]).all()
    assert result.nrms_after.shape == (reference.inline.size, reference.crossline.size)
    assert np.isnan(result.nrms_after[:, 0]).all()
    assert result.manifest.qc['median_nrms_after'] == result.median_nrms_after


def test_manifest_replay_is_deterministic_and_detects_changed_input(tmp_path):
    reference, monitor, _ = _synthetic_pair()
    config = CrossEqualizationConfig(
        stable_start_s=0.18,
        stable_end_s=0.46,
        max_bulk_shift_s=0.03,
        warp_step_s=0.02,
        warp_smooth=(3, 3, 5),
        max_local_shift_s=0.008,
    )
    result = cross_equalize(reference, monitor, config)
    path = tmp_path / 'xequalization.npz'
    save_manifest(result.manifest, path)
    restored = load_manifest(path)

    replayed = replay_cross_equalization(reference, monitor, restored)
    np.testing.assert_allclose(
        replayed.corrected_monitor.data,
        result.corrected_monitor.data,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        replayed.difference.data,
        result.difference.data,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(replayed.reliable_mask, result.reliable_mask)

    changed = FourDVolume(
        reference.inline,
        reference.crossline,
        reference.samples,
        reference.data + 1.0,
        reference.footprint,
        reference.name,
    )
    try:
        apply_manifest(changed, monitor, restored, verify_digests=True)
    except ValueError as exc:
        assert 'reference input digest differs' in str(exc)
    else:
        raise AssertionError('changed reference was accepted during replay')


def test_geometry_resampling_does_not_extrapolate_missing_coverage():
    reference, _, _ = _synthetic_pair()
    _, monitor, mon_footprint = _synthetic_pair()
    shifted_monitor = FourDVolume(
        reference.inline + 0.0,
        reference.crossline + 0.25,
        reference.samples + 0.001,
        monitor.data,
        monitor.footprint,
        'monitor-shifted',
    )

    matched = match_coverage(reference, shifted_monitor)
    assert matched.resampled
    assert not matched.common_mask[:, 0, :].any()
    assert np.isnan(matched.monitor.data[:, 0, :]).all()
    # Output samples requiring any missing source corner remain invalid.
    edge_missing = ~mon_footprint
    assert np.isnan(matched.monitor.data[:, edge_missing, :]).all()
    assert matched.common_mask[:, 1:-2, 1:-1].any()


def test_high_level_workflow_writes_and_replays_manifest(tmp_path):
    reference, monitor, _ = _synthetic_pair()
    config = CrossEqualizationConfig(
        stable_start_s=0.18,
        stable_end_s=0.46,
        max_bulk_shift_s=0.03,
        warp_step_s=0.02,
        warp_smooth=(3, 3, 5),
        max_local_shift_s=0.008,
    )
    manifest_path = tmp_path / 'fourd.npz'
    output = four_d_workflow(reference, monitor, config, manifest_path=manifest_path)

    assert manifest_path.exists()
    replayed = replay_four_d_workflow(reference, monitor, manifest_path)
    np.testing.assert_allclose(
        output.difference.data,
        replayed.difference.data,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
