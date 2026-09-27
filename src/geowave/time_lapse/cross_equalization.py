"""Deterministic 4D baseline/monitor cross-equalization.

The workflow is intentionally conservative about acquisition footprint:
coverage is matched before any scalar or wavelet correction is estimated, and
positions outside the common reliable coverage remain NaN rather than being
turned into interpolated 4D differences.  Every estimated operator is stored
in a manifest and can be replayed from the original volumes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

__all__ = [
    'FourDVolume',
    'CoverageMatch',
    'CrossEqualizationConfig',
    'CrossEqualizationResult',
    'CorrectionManifest',
    'match_coverage',
    'cross_equalize',
    'apply_manifest',
    'replay_cross_equalization',
    'save_manifest',
    'load_manifest',
    'nrms_map',
]


@dataclass
class FourDVolume:
    """3D volume indexed by inline, crossline and time/depth sample."""

    inline: np.ndarray
    crossline: np.ndarray
    samples: np.ndarray
    data: np.ndarray
    footprint: np.ndarray | None = None
    name: str = ''
    domain: str = 'time'

    def __post_init__(self) -> None:
        self.inline = np.asarray(self.inline, dtype=float)
        self.crossline = np.asarray(self.crossline, dtype=float)
        self.samples = np.asarray(self.samples, dtype=float)
        self.data = np.asarray(self.data, dtype=float)
        if self.domain != 'time':
            raise ValueError('four-dimensional cross-equalization requires a time axis')
        for axis, name in ((self.inline, 'inline'), (self.crossline, 'crossline'), (self.samples, 'samples')):
            if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f'{name} must be a strictly increasing 1-D array')
        expected = (self.inline.size, self.crossline.size, self.samples.size)
        if self.data.shape != expected:
            raise ValueError(f'data shape {self.data.shape} does not match axes {expected}')
        if self.footprint is not None:
            self.footprint = np.asarray(self.footprint, dtype=bool)
            if self.footprint.shape == expected[:2]:
                pass
            elif self.footprint.shape != expected:
                raise ValueError('footprint must be a 2-D trace mask or a 3-D sample mask')

    def valid_mask(self) -> np.ndarray:
        mask = np.isfinite(self.data)
        if self.footprint is None:
            return mask
        footprint = self.footprint
        if footprint.ndim == 2:
            footprint = footprint[:, :, None]
        return mask & footprint


@dataclass
class CoverageMatch:
    reference: FourDVolume
    monitor: FourDVolume
    common_mask: np.ndarray
    reference_only_mask: np.ndarray
    monitor_only_mask: np.ndarray
    resampled: bool

    @property
    def common_fraction(self) -> float:
        return float(np.mean(self.common_mask))


@dataclass(frozen=True)
class CrossEqualizationConfig:
    """Parameters for fitting and replaying a cross-equalization operator."""

    stable_start_s: float | None = None
    stable_end_s: float | None = None
    nrms_start_s: float | None = None
    nrms_end_s: float | None = None
    axis_tolerance: float = 1.0e-6
    allow_monitor_resample: bool = True
    max_bulk_shift_s: float = 0.050
    min_correlation_samples: int = 16
    estimate_local_warp: bool = True
    warp_gate_s: float = 0.120
    warp_step_s: float = 0.040
    max_local_shift_s: float = 0.012
    warp_smooth: tuple[int, int, int] = (3, 3, 5)
    warp_min_correlation: float = 0.50
    warp_apply_correlation: float = 0.50
    warp_min_observations: int = 3

    def validate(self) -> None:
        if self.max_bulk_shift_s <= 0:
            raise ValueError('max_bulk_shift_s must be positive')
        if self.warp_gate_s <= 0 or self.warp_step_s <= 0 or self.max_local_shift_s <= 0:
            raise ValueError('warp gates and maximum shift must be positive')
        if len(self.warp_smooth) != 3 or any(size < 1 or size % 2 == 0 for size in self.warp_smooth):
            raise ValueError('warp_smooth must contain three positive odd sizes')
        if not 0.0 <= self.warp_min_correlation <= 1.0:
            raise ValueError('warp_min_correlation must be between 0 and 1')
        if not 0.0 <= self.warp_apply_correlation <= 1.0:
            raise ValueError('warp_apply_correlation must be between 0 and 1')


@dataclass(frozen=True)
class CrossEqualizationResult:
    coverage: CoverageMatch
    corrected_monitor: FourDVolume
    difference: FourDVolume
    reliable_mask: np.ndarray
    nrms_before: np.ndarray
    nrms_after: np.ndarray
    time_shift_s: float
    phase_shift_deg: float
    amplitude_scale: float
    local_warp_s: np.ndarray
    warp_confidence: np.ndarray
    manifest: 'CorrectionManifest'
    warnings: tuple[str, ...]

    @property
    def median_nrms_before(self) -> float:
        return _finite_median(self.nrms_before)

    @property
    def median_nrms_after(self) -> float:
        return _finite_median(self.nrms_after)


@dataclass(frozen=True)
class CorrectionManifest:
    """Serialized, replayable scalar corrections and local warp operator."""

    parameters: dict
    time_shift_s: float
    phase_shift_deg: float
    amplitude_scale: float
    warp_shift_s: np.ndarray
    warp_confidence: np.ndarray
    common_mask: np.ndarray
    fit_mask: np.ndarray | None
    input_reference_digest: str
    input_monitor_digest: str
    output_digests: dict
    provenance: tuple
    qc: dict
    workflow_name: str = 'four_d_cross_equalization'
    workflow_version: str = '1.0'
    version: str = '1.0'

    def digest(self) -> str:
        payload = {
            'workflow_name': self.workflow_name,
            'workflow_version': self.workflow_version,
            'version': self.version,
            'parameters': self.parameters,
            'time_shift_s': self.time_shift_s,
            'phase_shift_deg': self.phase_shift_deg,
            'amplitude_scale': self.amplitude_scale,
            'warp_shift_s': _array_digest(self.warp_shift_s),
            'warp_confidence': _array_digest(self.warp_confidence),
            'common_mask': _array_digest(self.common_mask),
            'fit_mask': _array_digest(self.fit_mask) if self.fit_mask is not None else None,
            'input_reference_digest': self.input_reference_digest,
            'input_monitor_digest': self.input_monitor_digest,
            'output_digests': self.output_digests,
            'provenance': self.provenance,
            'qc': self.qc,
        }
        return _json_digest(payload)


def match_coverage(
    reference: FourDVolume,
    monitor: FourDVolume,
    axis_tolerance: float = 1.0e-6,
    allow_monitor_resample: bool = True,
) -> CoverageMatch:
    """Map both surveys to the reference grid and retain only common coverage.

    Monitor resampling is geometric only.  A resampled monitor sample is valid
    only when every required source sample is valid; there is no extrapolation
    and no interpolation across a footprint hole.
    """

    same_axes = (
        _same_axis(reference.inline, monitor.inline, axis_tolerance)
        and _same_axis(reference.crossline, monitor.crossline, axis_tolerance)
        and _same_axis(reference.samples, monitor.samples, axis_tolerance)
    )
    resampled = False
    if same_axes:
        ref_mask = reference.valid_mask()
        mon_mask = monitor.valid_mask()
        mon_data = np.where(mon_mask, monitor.data, np.nan)
        matched_monitor = FourDVolume(
            reference.inline.copy(),
            reference.crossline.copy(),
            reference.samples.copy(),
            mon_data,
            None,
            monitor.name + ':matched',
        )
    else:
        if not allow_monitor_resample:
            raise ValueError('monitor axes do not match the reference grid and resampling is disabled')
        matched_monitor, mon_mask = _resample_monitor(reference, monitor)
        ref_mask = reference.valid_mask()
        resampled = True

    ref_data = np.where(ref_mask, reference.data, np.nan)
    matched_reference = FourDVolume(
        reference.inline.copy(),
        reference.crossline.copy(),
        reference.samples.copy(),
        ref_data,
        None,
        reference.name + ':matched',
    )
    common = ref_mask & mon_mask
    ref_only = ref_mask & ~mon_mask
    mon_only = mon_mask & ~ref_mask
    if not common.any():
        raise ValueError('no common reliable coverage between baseline and monitor')
    return CoverageMatch(matched_reference, matched_monitor, common, ref_only, mon_only, resampled)


def cross_equalize(
    reference: FourDVolume,
    monitor: FourDVolume,
    config: CrossEqualizationConfig | None = None,
    fit_mask: np.ndarray | None = None,
) -> CrossEqualizationResult:
    """Run coverage matching, alignment, phase/amplitude balance and warping."""

    config = config or CrossEqualizationConfig()
    config.validate()
    ref_digest = volume_digest(reference)
    mon_digest = volume_digest(monitor)
    provenance: list[dict] = []
    warnings: list[str] = []

    coverage = match_coverage(
        reference,
        monitor,
        config.axis_tolerance,
        config.allow_monitor_resample,
    )
    common = coverage.common_mask
    ref = coverage.reference.data
    mon_raw = coverage.monitor.data
    provenance.append(
        _step(
            'coverage_match',
            _json_digest({'reference': ref_digest, 'monitor': mon_digest}),
            _json_digest(
                {
                    'reference': _array_digest(ref),
                    'monitor': _array_digest(mon_raw),
                    'common': _array_digest(common),
                }
            ),
            {'resampled': coverage.resampled, 'common_fraction': coverage.common_fraction},
        )
    )

    analysis_mask = _build_fit_mask(coverage, config, fit_mask)
    if analysis_mask.sum() < max(8, config.min_correlation_samples):
        raise ValueError('insufficient stable/common samples for correction estimation')

    dt = _uniform_dt(coverage.reference)
    bulk_shift, bulk_correlation = estimate_bulk_shift(
        ref, mon_raw, analysis_mask, dt, config.max_bulk_shift_s, config.min_correlation_samples
    )
    if abs(bulk_shift) >= config.max_bulk_shift_s * 0.95:
        warnings.append('bulk shift is at the configured search boundary')
    mon_time = apply_time_shift(mon_raw, common, bulk_shift, dt)
    provenance.append(
        _step(
            'bulk_time_shift',
            _array_digest(mon_raw),
            _array_digest(mon_time),
            {'time_shift_s': bulk_shift, 'correlation': bulk_correlation},
        )
    )

    phase_deg = estimate_phase_shift(ref, mon_time, analysis_mask)
    mon_phase = apply_phase_shift(mon_time, common, phase_deg)
    provenance.append(
        _step(
            'constant_phase',
            _array_digest(mon_time),
            _array_digest(mon_phase),
            {'phase_shift_deg': phase_deg},
        )
    )

    amplitude = estimate_amplitude(ref, mon_phase, analysis_mask)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError(f'non-physical amplitude scale estimated: {amplitude}')
    mon_amp = mon_phase * amplitude
    provenance.append(
        _step(
            'amplitude_scale',
            _array_digest(mon_phase),
            _array_digest(mon_amp),
            {'amplitude_scale': amplitude},
        )
    )

    qc = {
        'bulk_shift_correlation': float(bulk_correlation),
        'common_sample_fraction': coverage.common_fraction,
        'reference_only_fraction': float(np.mean(coverage.reference_only_mask)),
        'monitor_only_fraction': float(np.mean(coverage.monitor_only_mask)),
    }

    warp = np.zeros(ref.shape, dtype=float)
    warp_confidence = np.zeros(ref.shape, dtype=float)
    corrected = mon_amp
    if config.estimate_local_warp:
        warp, warp_confidence, warp_qc = estimate_local_warp(
            ref,
            mon_amp,
            analysis_mask,
            dt,
            config.warp_gate_s,
            config.warp_step_s,
            config.max_local_shift_s,
            tuple(config.warp_smooth),
            config.warp_min_correlation,
            config.warp_min_observations,
        )
        qc.update(warp_qc)
        if qc['local_warp_accepted_fraction'] < 0.01:
            raise ValueError('local warp estimation failed for nearly all traces')
        if qc['local_warp_accepted_fraction'] < 0.25:
            warnings.append('local warp accepted for fewer than 25% of common trace/gate positions')
        corrected = apply_local_warp(
            mon_amp, common, warp, warp_confidence, dt, config.warp_apply_correlation
        )
        provenance.append(
            _step(
                'local_vertical_warp',
                _array_digest(mon_amp),
                _json_digest({'data': _array_digest(corrected), 'warp': _array_digest(warp)}),
                {
                    'gate_s': config.warp_gate_s,
                    'step_s': config.warp_step_s,
                    'max_shift_s': config.max_local_shift_s,
                    'smooth': tuple(config.warp_smooth),
                    'median_correlation': qc.get('local_warp_median_correlation', 0.0),
                },
            )
        )

    reliable = common & np.isfinite(ref) & np.isfinite(corrected)
    corrected = np.where(reliable, corrected, np.nan)
    difference = np.where(reliable, ref - corrected, np.nan)

    nrms_before = nrms_map(ref, mon_raw, _nrms_mask(common, coverage.reference, config))
    nrms_after = nrms_map(ref, corrected, _nrms_mask(reliable, coverage.reference, config))
    qc.update(
        {
            'median_nrms_before': _finite_median(nrms_before),
            'median_nrms_after': _finite_median(nrms_after),
            'reliable_sample_fraction': float(np.mean(reliable)),
        }
    )

    corrected_volume = FourDVolume(
        coverage.reference.inline,
        coverage.reference.crossline,
        coverage.reference.samples,
        corrected,
        None,
        monitor.name + ':xequalized',
    )
    difference_volume = FourDVolume(
        coverage.reference.inline,
        coverage.reference.crossline,
        coverage.reference.samples,
        difference,
        None,
        monitor.name + ':minus_baseline',
    )

    params = asdict(config)
    params['warp_smooth'] = tuple(params['warp_smooth'])
    output_digests = {
        'corrected_monitor': _array_digest(corrected),
        'difference': _array_digest(difference),
        'nrms_before': _array_digest(nrms_before),
        'nrms_after': _array_digest(nrms_after),
        'warp_shift': _array_digest(warp),
        'warp_confidence': _array_digest(warp_confidence),
        'common_mask': _array_digest(common),
    }
    manifest = CorrectionManifest(
        parameters=params,
        time_shift_s=float(bulk_shift),
        phase_shift_deg=float(phase_deg),
        amplitude_scale=float(amplitude),
        warp_shift_s=warp,
        warp_confidence=warp_confidence,
        common_mask=common,
        fit_mask=None if fit_mask is None else np.asarray(fit_mask, dtype=bool),
        input_reference_digest=ref_digest,
        input_monitor_digest=mon_digest,
        output_digests=output_digests,
        provenance=tuple(provenance),
        qc=qc,
    )
    return CrossEqualizationResult(
        coverage=coverage,
        corrected_monitor=corrected_volume,
        difference=difference_volume,
        reliable_mask=reliable,
        nrms_before=nrms_before,
        nrms_after=nrms_after,
        time_shift_s=float(bulk_shift),
        phase_shift_deg=float(phase_deg),
        amplitude_scale=float(amplitude),
        local_warp_s=warp,
        warp_confidence=warp_confidence,
        manifest=manifest,
        warnings=tuple(warnings),
    )


def apply_manifest(
    reference: FourDVolume,
    monitor: FourDVolume,
    manifest: CorrectionManifest,
    verify_digests: bool = True,
) -> CrossEqualizationResult:
    """Reapply a stored operator to the original baseline and monitor."""

    ref_digest = volume_digest(reference)
    mon_digest = volume_digest(monitor)
    if verify_digests:
        if ref_digest != manifest.input_reference_digest:
            raise ValueError('reference input digest differs from manifest')
        if mon_digest != manifest.input_monitor_digest:
            raise ValueError('monitor input digest differs from manifest')

    config = _config_from_dict(manifest.parameters)
    coverage = match_coverage(
        reference,
        monitor,
        config.axis_tolerance,
        config.allow_monitor_resample,
    )
    common = coverage.common_mask
    if _array_digest(common) != _array_digest(manifest.common_mask):
        raise ValueError('replayed coverage mask differs from stored mask')

    ref = coverage.reference.data
    mon = coverage.monitor.data
    dt = _uniform_dt(coverage.reference)
    mon = apply_time_shift(mon, common, manifest.time_shift_s, dt)
    mon = apply_phase_shift(mon, common, manifest.phase_shift_deg)
    mon = mon * manifest.amplitude_scale
    if config.estimate_local_warp:
        mon = apply_local_warp(
            mon,
            common,
            manifest.warp_shift_s,
            manifest.warp_confidence,
            dt,
            config.warp_apply_correlation,
        )

    reliable = common & np.isfinite(ref) & np.isfinite(mon)
    mon = np.where(reliable, mon, np.nan)
    difference = np.where(reliable, ref - mon, np.nan)
    nrms_before = nrms_map(ref, coverage.monitor.data, _nrms_mask(common, coverage.reference, config))
    nrms_after = nrms_map(ref, mon, _nrms_mask(reliable, coverage.reference, config))

    replayed = {
        'corrected_monitor': _array_digest(mon),
        'difference': _array_digest(difference),
        'nrms_before': _array_digest(nrms_before),
        'nrms_after': _array_digest(nrms_after),
        'warp_shift': _array_digest(manifest.warp_shift_s),
        'warp_confidence': _array_digest(manifest.warp_confidence),
        'common_mask': _array_digest(common),
    }
    if verify_digests and replayed != manifest.output_digests:
        raise RuntimeError('replayed output digest differs from manifest output digest')

    return CrossEqualizationResult(
        coverage=coverage,
        corrected_monitor=FourDVolume(
            coverage.reference.inline,
            coverage.reference.crossline,
            coverage.reference.samples,
            mon,
            None,
            monitor.name + ':xequalized:replayed',
        ),
        difference=FourDVolume(
            coverage.reference.inline,
            coverage.reference.crossline,
            coverage.reference.samples,
            difference,
            None,
            monitor.name + ':minus_baseline:replayed',
        ),
        reliable_mask=reliable,
        nrms_before=nrms_before,
        nrms_after=nrms_after,
        time_shift_s=manifest.time_shift_s,
        phase_shift_deg=manifest.phase_shift_deg,
        amplitude_scale=manifest.amplitude_scale,
        local_warp_s=manifest.warp_shift_s,
        warp_confidence=manifest.warp_confidence,
        manifest=manifest,
        warnings=tuple(),
    )


def replay_cross_equalization(
    reference: FourDVolume,
    monitor: FourDVolume,
    manifest: CorrectionManifest,
) -> CrossEqualizationResult:
    """Convenience wrapper for digest-verified manifest replay."""

    return apply_manifest(reference, monitor, manifest, verify_digests=True)


def estimate_bulk_shift(
    reference: np.ndarray,
    monitor: np.ndarray,
    mask: np.ndarray,
    dt_s: float,
    max_shift_s: float,
    min_samples: int,
) -> tuple[float, float]:
    """Estimate a bulk monitor delay using masked normalized cross-correlation."""

    nt = reference.shape[-1]
    max_lag = min(nt - 1, int(np.floor(max_shift_s / dt_s)))
    if max_lag < 1:
        raise ValueError('maximum bulk shift is smaller than one sample')
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    correlations = np.full(lags.shape, np.nan)
    weights = np.zeros(lags.shape, dtype=float)

    for i, lag in enumerate(lags):
        if lag >= 0:
            r = reference[..., : nt - lag]
            m = monitor[..., lag:]
            valid = mask[..., lag:] & mask[..., : nt - lag]
        else:
            p = -lag
            r = reference[..., p:]
            m = monitor[..., : nt - p]
            valid = mask[..., p:] & mask[..., : nt - p]
        corr, count = _pearson_fields(r, m, valid)
        good = np.isfinite(corr) & (count >= min_samples)
        if good.any():
            correlations[i] = np.average(corr[good], weights=count[good])
            weights[i] = float(np.sum(count[good]))

    if not np.isfinite(correlations).any():
        raise ValueError('insufficient overlap for bulk time-shift estimation')
    best = int(np.nanargmax(correlations))
    peak = float(correlations[best])
    offset = 0.0
    if 0 < best < len(lags) - 1:
        y0, y1, y2 = correlations[best - 1], correlations[best], correlations[best + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1.0e-12:
            offset = 0.5 * (y0 - y2) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
    shift_samples = float(lags[best] + offset)
    return shift_samples * dt_s, peak


def apply_time_shift(data: np.ndarray, mask: np.ndarray, shift_s: float, dt_s: float) -> np.ndarray:
    """Shift monitor along the time axis without interpolating invalid samples."""

    nil, nxl, nt = data.shape
    shift = shift_s / dt_s
    ii = np.arange(nil)[:, None]
    jj = np.arange(nxl)[None, :]
    kk = np.arange(nt)[None, None, :]
    source = kk + shift
    i0 = np.floor(source).astype(int)
    fraction = source - i0
    i1 = i0 + 1
    exact = fraction <= 1.0e-10
    i0_clipped = np.clip(i0, 0, nt - 1)
    i1_clipped = np.clip(i1, 0, nt - 1)

    v0 = data[ii[:, :, None], jj[:, :, None], i0_clipped]
    v1 = data[ii[:, :, None], jj[:, :, None], i1_clipped]
    m0 = mask[ii[:, :, None], jj[:, :, None], i0_clipped]
    m1 = mask[ii[:, :, None], jj[:, :, None], i1_clipped]
    exact_valid = exact & (i0 >= 0) & (i0 < nt) & m0
    fraction_valid = (~exact) & (i0 >= 0) & (i1 < nt) & m0 & m1
    value = (1.0 - fraction) * v0 + fraction * v1
    selected = np.where(exact_valid, v0, np.where(fraction_valid, value, np.nan))
    return np.where(np.isfinite(selected), selected, np.nan)


def estimate_phase_shift(reference: np.ndarray, monitor: np.ndarray, mask: np.ndarray) -> float:
    """Estimate a constant phase rotation in degrees from a stable window."""

    valid = mask & np.isfinite(reference) & np.isfinite(monitor)
    if valid.sum() < 8:
        raise ValueError('insufficient samples for phase estimation')
    analytic_imag = _masked_hilbert(monitor, valid)
    r = reference[valid]
    m = monitor[valid]
    h = analytic_imag[valid]
    r = r - np.mean(r)
    m = m - np.mean(m)
    h = h - np.mean(h)
    matrix = np.column_stack((m, h))
    coefficients, *_ = np.linalg.lstsq(matrix, r, rcond=None)
    phase = float(np.arctan2(coefficients[1], coefficients[0]))
    return float(np.degrees(phase))


def apply_phase_shift(data: np.ndarray, mask: np.ndarray, phase_deg: float) -> np.ndarray:
    phase = np.radians(phase_deg)
    valid = mask & np.isfinite(data)
    hilbert = _masked_hilbert(data, valid)
    rotated = np.cos(phase) * data + np.sin(phase) * hilbert
    return np.where(valid, rotated, np.nan)


def estimate_amplitude(reference: np.ndarray, monitor: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares scalar amplitude in the stable window."""

    valid = mask & np.isfinite(reference) & np.isfinite(monitor)
    if valid.sum() < 8:
        raise ValueError('insufficient samples for amplitude estimation')
    r = reference[valid]
    m = monitor[valid]
    r = r - np.mean(r)
    m = m - np.mean(m)
    denominator = float(np.dot(m, m))
    if denominator <= 1.0e-20:
        raise ValueError('monitor has negligible energy in the amplitude window')
    return float(np.dot(r, m) / denominator)


def estimate_local_warp(
    reference: np.ndarray,
    monitor: np.ndarray,
    mask: np.ndarray,
    dt_s: float,
    gate_s: float,
    step_s: float,
    max_shift_s: float,
    smooth: tuple[int, int, int],
    min_correlation: float,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate a smooth residual vertical deformation field.

    Correlations are measured in local time gates and then spatially/temporally
    smoothed.  The confidence map records the normalized correlation and amount
    of support used for each smoothed warp estimate.
    """

    nil, nxl, nt = reference.shape
    half = max(2, int(round(gate_s / (2.0 * dt_s))))
    step = max(1, int(round(step_s / dt_s)))
    max_lag = min(nt - 1, int(np.floor(max_shift_s / dt_s)))
    if max_lag < 1:
        raise ValueError('maximum local shift is smaller than one sample')
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    valid_time = np.where(mask.any(axis=(0, 1)))[0]
    first, last = int(valid_time[0]), int(valid_time[-1])
    first_center = first + half + max_lag
    last_center = last - half - max_lag
    centers = [c for c in range(first_center, last_center + 1, step)]
    if not centers:
        raise ValueError('stable window is too short for local warp estimation')

    observations = np.full(reference.shape, np.nan, dtype=float)
    weights = np.zeros(reference.shape, dtype=float)
    accepted = 0
    candidates = 0
    peak_correlations: list[float] = []
    seg_base = np.arange(-half, half + 1)

    for center in centers:
        seg = center + seg_base
        r_seg = reference[:, :, seg]
        correlations = np.full((nil, nxl, len(lags)), np.nan)
        counts = np.zeros((nil, nxl, len(lags)), dtype=int)
        for li, lag in enumerate(lags):
            shifted_seg = seg + int(lag)
            m_seg = monitor[:, :, shifted_seg]
            valid = mask[:, :, seg] & mask[:, :, shifted_seg]
            correlations[:, :, li], counts[:, :, li] = _pearson_fields(r_seg, m_seg, valid)

        has_finite = np.isfinite(correlations).any(axis=-1)
        correlations_for_argmax = np.where(has_finite[:, :, None], correlations, -np.inf)
        best_li = np.argmax(correlations_for_argmax, axis=-1)
        best_corr = np.take_along_axis(correlations, best_li[:, :, None], axis=-1)[:, :, 0]
        ii = np.arange(nil)[:, None]
        jj = np.arange(nxl)[None, :]
        usable = (counts.sum(axis=-1) > 0) & np.isfinite(best_corr)
        candidates += int(usable.sum())
        interior = (best_li > 0) & (best_li < len(lags) - 1) & usable
        c0 = correlations[ii, jj, np.clip(best_li - 1, 0, len(lags) - 1)]
        c1 = best_corr
        c2 = correlations[ii, jj, np.clip(best_li + 1, 0, len(lags) - 1)]
        denominator = c0 - 2.0 * c1 + c2
        offset = np.zeros_like(best_corr)
        good_denominator = np.abs(denominator) > 1.0e-12
        offset[good_denominator] = 0.5 * (c0[good_denominator] - c2[good_denominator]) / denominator[good_denominator]
        offset = np.clip(offset, -1.0, 1.0)
        accepted_mask = interior & (best_corr >= min_correlation)
        shift_samples = lags[best_li] + offset
        observations[:, :, center] = np.where(accepted_mask, shift_samples * dt_s, np.nan)
        weights[:, :, center] = np.where(accepted_mask, best_corr, 0.0)
        accepted += int(accepted_mask.sum())
        peak_correlations.extend(best_corr[accepted_mask].tolist())

    warp, confidence, support = _weighted_boxcar(observations, weights, smooth)
    warp = np.clip(warp, -max_shift_s, max_shift_s)
    support_factor = np.minimum(1.0, support / float(max(1, min_observations)))
    confidence = confidence * support_factor
    confidence = np.where(np.isfinite(warp), confidence, 0.0)
    warp = np.where(confidence > 0.0, warp, 0.0)
    qc = {
        'local_warp_accepted_fraction': float(accepted / max(1, candidates)),
        'local_warp_median_correlation': _finite_median(np.asarray(peak_correlations, dtype=float)),
        'local_warp_gate_count': float(len(centers)),
        'local_warp_max_abs_shift_s': float(np.nanmax(np.abs(warp))),
    }
    return warp, confidence, qc


def apply_local_warp(
    data: np.ndarray,
    mask: np.ndarray,
    warp_s: np.ndarray,
    confidence: np.ndarray,
    dt_s: float,
    apply_correlation: float,
) -> np.ndarray:
    """Apply a vertical warp only where its stored confidence is sufficient."""

    nil, nxl, nt = data.shape
    effective = np.where(confidence >= apply_correlation, warp_s, 0.0)
    shift = effective / dt_s
    ii = np.arange(nil)[:, None]
    jj = np.arange(nxl)[None, :]
    out = np.full(data.shape, np.nan, dtype=float)

    for k in range(nt):
        source = k + shift[:, :, k]
        i0 = np.floor(source).astype(int)
        fraction = source - i0
        i1 = i0 + 1
        exact = fraction <= 1.0e-10
        i0_clipped = np.clip(i0, 0, nt - 1)
        i1_clipped = np.clip(i1, 0, nt - 1)
        exact_valid = exact & (i0 >= 0) & (i0 < nt) & mask[ii, jj, i0_clipped]
        fraction_valid = (
            (~exact)
            & (i0 >= 0)
            & (i1 < nt)
            & mask[ii, jj, i0]
            & mask[ii, jj, i1]
        )
        v0 = data[ii, jj, i0_clipped]
        v1 = data[ii, jj, i1_clipped]
        value = (1.0 - fraction) * v0 + fraction * v1
        selected = np.where(exact_valid, v0, np.where(fraction_valid, value, np.nan))
        out[:, :, k] = selected
    return out


def nrms_map(
    reference: np.ndarray,
    monitor: np.ndarray,
    mask: np.ndarray,
    min_samples: int = 8,
) -> np.ndarray:
    """Trace-wise NRMS percentage over valid samples, shape (inline, crossline)."""

    valid = mask & np.isfinite(reference) & np.isfinite(monitor)
    count = valid.sum(axis=-1)
    r = np.where(valid, reference, 0.0)
    m = np.where(valid, monitor, 0.0)
    d = np.where(valid, reference - monitor, 0.0)
    rms_d = np.sqrt(np.sum(d * d, axis=-1) / np.maximum(count, 1))
    rms_r = np.sqrt(np.sum(r * r, axis=-1) / np.maximum(count, 1))
    rms_m = np.sqrt(np.sum(m * m, axis=-1) / np.maximum(count, 1))
    denominator = rms_r + rms_m
    result = np.full(count.shape, np.nan, dtype=float)
    good = (count >= min_samples) & (denominator > 1.0e-20)
    result[good] = 200.0 * rms_d[good] / denominator[good]
    return result


def save_manifest(manifest: CorrectionManifest, path: str | Path) -> None:
    path = Path(path)
    metadata = {
        'workflow_name': manifest.workflow_name,
        'workflow_version': manifest.workflow_version,
        'version': manifest.version,
        'parameters': manifest.parameters,
        'time_shift_s': manifest.time_shift_s,
        'phase_shift_deg': manifest.phase_shift_deg,
        'amplitude_scale': manifest.amplitude_scale,
        'input_reference_digest': manifest.input_reference_digest,
        'input_monitor_digest': manifest.input_monitor_digest,
        'output_digests': manifest.output_digests,
        'provenance': manifest.provenance,
        'qc': manifest.qc,
    }
    arrays = dict(
        warp_shift_s=manifest.warp_shift_s,
        warp_confidence=manifest.warp_confidence,
        common_mask=manifest.common_mask,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=_json_default)),
    )
    if manifest.fit_mask is not None:
        arrays['fit_mask'] = manifest.fit_mask
    np.savez_compressed(path, **arrays)


def load_manifest(path: str | Path) -> CorrectionManifest:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        fit_mask = archive['fit_mask'] if 'fit_mask' in archive.files else None
        return CorrectionManifest(
            parameters=metadata['parameters'],
            time_shift_s=float(metadata['time_shift_s']),
            phase_shift_deg=float(metadata['phase_shift_deg']),
            amplitude_scale=float(metadata['amplitude_scale']),
            warp_shift_s=np.asarray(archive['warp_shift_s'], dtype=float),
            warp_confidence=np.asarray(archive['warp_confidence'], dtype=float),
            common_mask=np.asarray(archive['common_mask'], dtype=bool),
            fit_mask=None if fit_mask is None else np.asarray(fit_mask, dtype=bool),
            input_reference_digest=metadata['input_reference_digest'],
            input_monitor_digest=metadata['input_monitor_digest'],
            output_digests=dict(metadata['output_digests']),
            provenance=tuple(metadata['provenance']),
            qc=dict(metadata['qc']),
            workflow_name=metadata.get('workflow_name', 'four_d_cross_equalization'),
            workflow_version=metadata.get('workflow_version', '1.0'),
            version=metadata['version'],
        )


def volume_digest(volume: FourDVolume) -> str:
    return _json_digest(
        {
            'name': volume.name,
            'domain': volume.domain,
            'inline': _array_digest(volume.inline),
            'crossline': _array_digest(volume.crossline),
            'samples': _array_digest(volume.samples),
            'data': _array_digest(volume.data),
            'footprint': _array_digest(volume.footprint) if volume.footprint is not None else None,
        }
    )


def _resample_monitor(reference: FourDVolume, monitor: FourDVolume) -> tuple[FourDVolume, np.ndarray]:
    ix0, ix1, wx0, x_valid = _axis_weights(reference.inline, monitor.inline)
    iy0, iy1, wy0, y_valid = _axis_weights(reference.crossline, monitor.crossline)
    it0, it1, wt0, t_valid = _axis_weights(reference.samples, monitor.samples)
    source_mask = monitor.valid_mask()
    output = np.zeros(reference.data.shape, dtype=float)
    coordinate_valid = x_valid[:, None, None] & y_valid[None, :, None] & t_valid[None, None, :]
    valid = coordinate_valid.copy()

    ix = (ix0, ix1)
    iy = (iy0, iy1)
    it = (it0, it1)
    wx = (wx0, 1.0 - wx0)
    wy = (wy0, 1.0 - wy0)
    wt = (wt0, 1.0 - wt0)
    for x_index, x_weight in zip(ix, wx, strict=True):
        for y_index, y_weight in zip(iy, wy, strict=True):
            for t_index, t_weight in zip(it, wt, strict=True):
                xi = x_index[:, None, None]
                yi = y_index[None, :, None]
                ti = t_index[None, None, :]
                corner = np.nan_to_num(monitor.data[xi, yi, ti], nan=0.0)
                corner_mask = source_mask[xi, yi, ti]
                weight = (x_weight[:, None, None] * y_weight[None, :, None] * t_weight[None, None, :])
                output += weight * corner
                valid &= corner_mask

    valid &= coordinate_valid
    output = np.where(valid, output, np.nan)
    return (
        FourDVolume(
            reference.inline.copy(),
            reference.crossline.copy(),
            reference.samples.copy(),
            output,
            None,
            monitor.name + ':resampled',
        ),
        valid,
    )


def _axis_weights(output_axis: np.ndarray, source_axis: np.ndarray):
    i0 = np.zeros(output_axis.size, dtype=int)
    i1 = np.zeros(output_axis.size, dtype=int)
    w0 = np.zeros(output_axis.size, dtype=float)
    valid = np.zeros(output_axis.size, dtype=bool)
    for i, value in enumerate(output_axis):
        q = int(np.searchsorted(source_axis, value, side='left'))
        if q > 0 and q <= source_axis.size and np.isclose(source_axis[q - 1], value):
            index = q - 1
            i0[i] = i1[i] = index
            w0[i] = 1.0
            valid[i] = True
        elif 1 <= q <= source_axis.size - 1:
            left = q - 1
            right = q
            i0[i] = left
            i1[i] = right
            w0[i] = (source_axis[right] - value) / (source_axis[right] - source_axis[left])
            valid[i] = True
    return i0, i1, w0, valid


def _same_axis(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=tolerance)


def _uniform_dt(volume: FourDVolume) -> float:
    differences = np.diff(volume.samples)
    dt = float(np.median(differences))
    if not np.allclose(differences, dt, rtol=0.0, atol=1.0e-8 * max(1.0, abs(dt))):
        raise ValueError('sample axis must be uniformly sampled for time-shift estimation')
    return dt


def _build_fit_mask(
    coverage: CoverageMatch,
    config: CrossEqualizationConfig,
    supplied: np.ndarray | None,
) -> np.ndarray:
    mask = coverage.common_mask.copy()
    time_axis = coverage.reference.samples
    if config.stable_start_s is not None:
        mask &= (time_axis[None, None, :] >= config.stable_start_s)
    if config.stable_end_s is not None:
        mask &= (time_axis[None, None, :] <= config.stable_end_s)
    if supplied is not None:
        supplied = np.asarray(supplied, dtype=bool)
        if supplied.shape != mask.shape:
            raise ValueError(f'fit_mask shape {supplied.shape} does not match {mask.shape}')
        mask &= supplied
    return mask


def _nrms_mask(mask: np.ndarray, volume: FourDVolume, config: CrossEqualizationConfig) -> np.ndarray:
    result = mask.copy()
    if config.nrms_start_s is not None:
        result &= volume.samples[None, None, :] >= config.nrms_start_s
    if config.nrms_end_s is not None:
        result &= volume.samples[None, None, :] <= config.nrms_end_s
    return result


def _pearson_fields(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = valid.sum(axis=-1)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    sx = xv.sum(axis=-1)
    sy = yv.sum(axis=-1)
    sxx = np.sum(xv * xv, axis=-1)
    syy = np.sum(yv * yv, axis=-1)
    sxy = np.sum(xv * yv, axis=-1)
    numerator = count * sxy - sx * sy
    denominator = np.sqrt(np.maximum(0.0, count * sxx - sx * sx) * np.maximum(0.0, count * syy - sy * sy))
    corr = np.full(count.shape, np.nan, dtype=float)
    good = (count > 2) & (denominator > 1.0e-20)
    corr[good] = numerator[good] / denominator[good]
    return corr, count


def _hilbert(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    spectrum = np.fft.fft(x, axis=-1)
    analytic = np.zeros_like(spectrum)
    if n % 2 == 0:
        analytic[..., 0] = spectrum[..., 0]
        analytic[..., 1 : n // 2] = 2.0 * spectrum[..., 1 : n // 2]
        analytic[..., n // 2] = spectrum[..., n // 2]
    else:
        analytic[..., 0] = spectrum[..., 0]
        analytic[..., 1 : (n + 1) // 2] = 2.0 * spectrum[..., 1 : (n + 1) // 2]
    return np.fft.ifft(analytic, axis=-1).imag


def _masked_hilbert(data: np.ndarray, mask: np.ndarray, taper_fraction: float = 0.08) -> np.ndarray:
    """Hilbert transform with a local taper at valid-window boundaries."""

    nt = data.shape[-1]
    any_valid = mask.any(axis=-1)
    first = np.full(data.shape[:-1], nt, dtype=int)
    last = np.full(data.shape[:-1], -1, dtype=int)
    if any_valid.any():
        indices = np.broadcast_to(np.arange(nt), data.shape)
        first[any_valid] = np.where(mask, indices, nt)[any_valid].min(axis=-1)
        last[any_valid] = np.where(mask, indices, -1)[any_valid].max(axis=-1)

    radius = max(1, int(round(nt * taper_fraction)))
    distance = np.minimum(
        np.arange(nt)[None, None, :] - first[:, :, None],
        last[:, :, None] - np.arange(nt)[None, None, :],
    )
    taper = np.clip((distance + 1.0) / float(radius + 1), 0.0, 1.0)
    window = np.sin(0.5 * np.pi * taper) ** 2
    windowed = np.where(mask, data, 0.0) * window
    return _hilbert(windowed)


def _weighted_boxcar(
    values: np.ndarray,
    weights: np.ndarray,
    size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & (weights > 0.0)
    weighted_values = np.where(finite, values * weights, 0.0)
    weight_sum = np.where(finite, weights, 0.0)
    support = finite.astype(float)
    for axis, window in enumerate(size):
        weighted_values = _box_sum(weighted_values, window, axis)
        weight_sum = _box_sum(weight_sum, window, axis)
        support = _box_sum(support, window, axis)
    mean = np.divide(
        weighted_values,
        weight_sum,
        out=np.full_like(weighted_values, np.nan),
        where=weight_sum > 0.0,
    )
    confidence = np.divide(
        weight_sum,
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 0.0,
    )
    return mean, confidence, support


def _box_sum(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    radius = (size - 1) // 2
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(array, padding, mode='constant', constant_values=0.0)
    cumulative = np.cumsum(padded, axis=axis)
    zero = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate((zero, cumulative), axis=axis)
    end = np.arange(array.shape[axis]) + 2 * radius + 1
    start = np.arange(array.shape[axis])
    return np.take(cumulative, end, axis=axis) - np.take(cumulative, start, axis=axis)


def _config_from_dict(values: dict) -> CrossEqualizationConfig:
    params = dict(values)
    if 'warp_smooth' in params:
        params['warp_smooth'] = tuple(int(v) for v in params['warp_smooth'])
    allowed = CrossEqualizationConfig.__dataclass_fields__
    return CrossEqualizationConfig(**{k: v for k, v in params.items() if k in allowed})


def _step(name: str, input_digest: str, output_digest: str, parameters: dict) -> dict:
    return {
        'operation': name,
        'input_digest': input_digest,
        'output_digest': output_digest,
        'parameters': _json_safe(parameters),
    }


def _array_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    a = np.asarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(a.shape).encode())
    hasher.update(str(a.dtype).encode())
    hasher.update(a.tobytes())
    return hasher.hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return {'__ndarray__': True, 'digest': _array_digest(value)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'object of type {type(value)} is not JSON serializable')


def _json_safe(value):
    return json.loads(json.dumps(value, sort_keys=True, default=_json_default))


def _json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float('nan')
