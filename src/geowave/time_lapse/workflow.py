"""High-level 4D cross-equalization workflow.

This module documents and enforces the operational order used by the library:

1. Grid and coverage matching.
2. Operator estimation exclusively inside common stable coverage.
3. Deterministic replay of the complete operator from the original surveys.

It is a thin façade over :mod:`geowave.time_lapse.cross_equalization`, kept
separate so processing code can depend on the workflow name rather than an
individual numerical function.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_equalization import (
    CorrectionManifest,
    CrossEqualizationConfig,
    CrossEqualizationResult,
    FourDVolume,
    apply_manifest,
    cross_equalize,
    load_manifest,
    save_manifest,
)

__all__ = ['FourDWorkflow', 'FourDWorkflowOutput', 'four_d_workflow', 'replay_four_d_workflow']


@dataclass(frozen=True)
class FourDWorkflowOutput:
    result: CrossEqualizationResult
    manifest_path: Path | None = None

    @property
    def difference(self) -> FourDVolume:
        return self.result.difference

    @property
    def nrms_after(self) -> np.ndarray:
        return self.result.nrms_after


class FourDWorkflow:
    """Callable, serializable baseline-to-monitor cross-equalization flow."""

    name = 'four_d_cross_equalization'
    version = '1.0'

    def __init__(self, config: CrossEqualizationConfig | None = None):
        self.config = config or CrossEqualizationConfig()

    def run(
        self,
        baseline: FourDVolume,
        monitor: FourDVolume,
        fit_mask: np.ndarray | None = None,
        manifest_path: str | Path | None = None,
    ) -> FourDWorkflowOutput:
        result = cross_equalize(baseline, monitor, self.config, fit_mask)
        path = None if manifest_path is None else Path(manifest_path)
        if path is not None:
            save_manifest(result.manifest, path)
        return FourDWorkflowOutput(result, path)


def four_d_workflow(
    baseline: FourDVolume,
    monitor: FourDVolume,
    config: CrossEqualizationConfig | None = None,
    fit_mask: np.ndarray | None = None,
    manifest_path: str | Path | None = None,
) -> FourDWorkflowOutput:
    return FourDWorkflow(config).run(baseline, monitor, fit_mask, manifest_path)


def replay_four_d_workflow(
    baseline: FourDVolume,
    monitor: FourDVolume,
    manifest: str | Path | CorrectionManifest,
) -> CrossEqualizationResult:
    restored = load_manifest(manifest) if not isinstance(manifest, CorrectionManifest) else manifest
    return apply_manifest(baseline, monitor, restored, verify_digests=True)
