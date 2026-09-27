"""交叉均衡质量汇总：把差异体/NRMS/可靠性图聚合成可签收的指标。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geowave.time4d.types import (REL_LOW_ALIGN, REL_LOW_ENERGY, REL_MISSING,
                                  REL_OK, REL_SUPPORT, CrossResult)

REL_NAMES = {
    REL_OK: 'ok',
    REL_MISSING: 'missing_coverage',
    REL_SUPPORT: 'support_loss',
    REL_LOW_ALIGN: 'low_alignment',
    REL_LOW_ENERGY: 'low_energy',
}


@dataclass(frozen=True)
class CrossSummary:
    total_voxels: int
    fractions: dict[str, float]
    nrms_median: float
    nrms_p10: float
    nrms_p90: float
    predictability_median: float
    difference_rms: float
    reliable_fraction: float

    def as_dict(self) -> dict:
        d = {
            'total_voxels': self.total_voxels,
            'fractions': self.fractions,
            'reliable_fraction': self.reliable_fraction,
            'nrms_median_pct': self.nrms_median,
            'nrms_p10_pct': self.nrms_p10,
            'nrms_p90_pct': self.nrms_p90,
            'predictability_median': self.predictability_median,
            'difference_rms': self.difference_rms,
        }
        return d


def summarize(result: CrossResult, gate_s: tuple[float, float] | None = None,
              dt: float | None = None) -> CrossSummary:
    """按可靠性码分层统计。可选只统计某时间窗（用于只评价目的层）。"""
    rel = result.reliability
    total = rel.size
    frac = {}
    for code, name in REL_NAMES.items():
        frac[name] = float((rel == code).sum()) / total

    nrms = result.nrms
    pred = result.predictability
    diff = result.difference
    if gate_s is not None and dt is not None:
        i0 = max(0, int(gate_s[0] / dt))
        i1 = min(rel.shape[2], int(gate_s[1] / dt))
        nrms = nrms[..., i0:i1]
        pred = pred[..., i0:i1]
        diff = diff[..., i0:i1]

    finite_n = nrms[np.isfinite(nrms)]
    finite_p = pred[np.isfinite(pred)]
    finite_d = diff[np.isfinite(diff)]
    return CrossSummary(
        total_voxels=total,
        fractions=frac,
        nrms_median=float(np.median(finite_n)) if finite_n.size else float('nan'),
        nrms_p10=float(np.percentile(finite_n, 10)) if finite_n.size else float('nan'),
        nrms_p90=float(np.percentile(finite_n, 90)) if finite_n.size else float('nan'),
        predictability_median=float(np.median(finite_p)) if finite_p.size else float('nan'),
        difference_rms=float(np.sqrt(np.mean(finite_d ** 2))) if finite_d.size else float('nan'),
        reliable_fraction=frac[REL_NAMES[REL_OK]],
    )
