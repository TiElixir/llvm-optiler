"""D2: a temporary materialized and then copied.

Certain only for ``memref.copy`` ops at ``linalg_buffer`` whose *source is an allocation* (a computed temporary
copied into its final place; copying a function argument is unavoidable). No automatic fix is offered:
``eliminate-empty-tensors`` was tried on the `copy_cat` test model and did not remove the copy.
"""
from __future__ import annotations

from backend.schemas import Finding

from .base import Context, MIN_RATIO, est, memory_ms

ID = "d2-temporary-copy"


def detect(ctx: Context) -> list[Finding]:
    ops = ctx.stage_ops("linalg_buffer")
    avoidable = [c for c in ctx.copies if c["src_is_alloc"] and c["bytes"] and c["idx"] < len(ops)]
    if not avoidable or ctx.runtime_ms <= 0:
        return []
    saved = sum(memory_ms(c["bytes"], ctx.calib) for c in avoidable)
    ratio = ctx.runtime_ms / max(ctx.runtime_ms - saved, ctx.runtime_ms * 0.05)
    if ratio < MIN_RATIO:
        return []
    lines = sorted({ln for c in avoidable for ln in ops[c["idx"]].source_lines})
    return [Finding(
        id=ID, detector="D2",
        title=f"{len(avoidable)} temporary buffer{'s are' if len(avoidable) > 1 else ' is'} copied: "
              + (", ".join(f"L{ln}" for ln in lines) or "no source line"),
        op_ids=[ops[c["idx"]].id for c in avoidable],
        evidence={
            "copies": [{"op": ops[c["idx"]].id, "bytes": c["bytes"], "lines": ops[c["idx"]].source_lines} for c in avoidable],
            "copy_bytes": sum(c["bytes"] for c in avoidable), "lines": lines,
            "memory_bandwidth_gbs": ctx.calib.get("bandwidth_gbs"),
            "bufferization_conflict_report": "not captured: the copy comes from a temporary, not from an in-place conflict",
            "why": "the result of a computation is written to a temporary and then copied into the buffer that holds it",
        },
        impact_est=est(ratio, f"{saved:.3f} ms to copy {sum(c['bytes'] for c in avoidable)} bytes out of {ctx.runtime_ms:.3f} ms"),
        fix_ids=[],
    )]
