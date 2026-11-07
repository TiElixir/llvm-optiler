"""Apply a fix to a pipeline: pure functions, no I/O (the transform script is written by the job)."""
from __future__ import annotations

from backend.pipeline.config import PassSpec, PipelineConfig
from backend.schemas import Fix, Impact, Metrics

MIN_RATIO = 1.05


def apply_patch(cfg: PipelineConfig, fix: Fix) -> PipelineConfig:
    """Return a patched copy of ``cfg``; the input is never modified."""
    cfg = cfg.model_copy(deep=True)
    for p in fix.patch:
        stages = [s for s in cfg.stages if p.stage in (None, s.stage)]
        if p.op == "remove":
            for s in cfg.stages:
                s.passes = [x for x in s.passes if p.target not in x.text()]
        elif p.op in ("insert_after", "insert_before"):
            if not stages:
                raise ValueError(f"fix {fix.id}: unknown stage {p.stage!r}")
            s = stages[0]
            new = PassSpec(**{"pass": p.value})
            if any(x.text() == new.text() for x in s.passes):
                continue  # already there: applying twice is harmless
            if p.target is None:
                at = len(s.passes) if p.op == "insert_after" else 0
            else:
                idx = next((i for i, x in enumerate(s.passes) if p.target in x.text()), None)
                if idx is None:
                    raise ValueError(f"fix {fix.id}: no pass matching {p.target!r} in stage {s.stage}")
                at = idx + (1 if p.op == "insert_after" else 0)
            s.passes.insert(at, new)
        elif p.op == "replace":
            hit = False
            for s in cfg.stages:
                for i, x in enumerate(s.passes):
                    if p.target in x.text():
                        s.passes[i], hit = PassSpec(**{"pass": p.value}), True
            if not hit:
                raise ValueError(f"fix {fix.id}: no pass matching {p.target!r}")
        elif p.op == "set_option":
            for s in cfg.stages:
                for x in s.passes:
                    if p.target in x.text():
                        x.options = {**(x.options or {}), p.key: p.value}
    return cfg


def verdict(parent: Metrics, child: Metrics) -> Impact:
    """Measured outcome of a fix (decision 5): verified only when the runs do not overlap and it is >= 1.05x."""
    if child.correct is not True:
        return Impact(ratio=1.0, kind="broken", note="the patched build no longer matches PyTorch eager")
    if not parent.runtime_ms_median or not child.runtime_ms_median:
        return Impact(ratio=1.0, kind="none", note="no timing to compare")
    ratio = parent.runtime_ms_median / child.runtime_ms_median
    if child.runtime_ms_p90 < parent.runtime_ms_p10 and ratio >= MIN_RATIO:
        return Impact(ratio=round(ratio, 3), kind="verified",
                      note=f"{parent.runtime_ms_median:.3f} ms to {child.runtime_ms_median:.3f} ms (median of {child.iters} runs)")
    if child.runtime_ms_p10 > parent.runtime_ms_p90 and ratio <= 1 / MIN_RATIO:
        return Impact(ratio=round(ratio, 3), kind="slower",
                      note=f"{parent.runtime_ms_median:.3f} ms to {child.runtime_ms_median:.3f} ms")
    return Impact(ratio=round(ratio, 3), kind="none", note="the timing ranges overlap or the change is under 5%")
