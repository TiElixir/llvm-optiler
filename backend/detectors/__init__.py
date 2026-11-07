"""Detectors D1-D3. ``run_all`` returns findings ranked by estimated impact."""
from __future__ import annotations

from backend.schemas import Finding

from . import d1_unfused, d2_copies, d3_unvectorized
from .base import Context, MIN_RATIO

DETECTORS = [d1_unfused, d2_copies, d3_unvectorized]


def run_all(ctx: Context) -> list[Finding]:
    found: list[Finding] = []
    for d in DETECTORS:
        try:
            found += d.detect(ctx)
        except Exception as e:  # noqa: BLE001 - a broken detector must not fail the run
            import sys
            print(f"detector {d.__name__} failed: {type(e).__name__}: {e}", file=sys.stderr)
    found = [f for f in found if f.impact_est.ratio >= MIN_RATIO]
    return sorted(found, key=lambda f: -f.impact_est.ratio)
