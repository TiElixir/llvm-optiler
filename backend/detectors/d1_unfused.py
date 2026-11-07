"""D1: an elementwise producer/consumer chain that was not fused, so its intermediate lives in memory.

Certain only when, at the end of ``linalg_tensor``, an all-parallel ``linalg.generic`` has a single user that is a
linalg op over the same domain AND the intermediate shows up as a ``memref.alloc`` at ``linalg_buffer``.
"""
from __future__ import annotations

from backend.schemas import Finding

from .base import Context, MIN_RATIO, est, memory_ms, type_bytes

ID = "d1-unfused-elementwise"


def detect(ctx: Context) -> list[Finding]:
    by_idx = {x["idx"]: x for x in ctx.linalg}
    lt = ctx.stage_ops("linalg_tensor")
    allocs = [o for o in ctx.stage_ops("linalg_buffer") if o.name == "memref.alloc"]
    pairs = []
    for a in ctx.linalg:
        if a["name"] != "linalg.generic" or not a["iterators"] or set(a["iterators"]) != {"parallel"} or len(a["users"]) != 1:
            continue
        b = by_idx.get(a["users"][0])
        if b is None or b["result"] != a["result"] or a["idx"] >= len(lt):
            continue
        nbytes = type_bytes(a["result"])
        lines = lt[a["idx"]].source_lines
        mem = a["result"].replace("tensor<", "memref<") if a["result"] else None
        if not nbytes or not lines or not any(o.result_types[:1] == [mem] and set(o.source_lines) & set(lines) for o in allocs):
            continue  # not materialized later (e.g. loop fusion removed it)
        pairs.append({"producer_op": f"linalg_tensor:{lt[a['idx']].snapshot_idx}:{a['idx']}",
                      "consumer_op": f"linalg_tensor:{lt[a['idx']].snapshot_idx}:{b['idx']}",
                      "lines": lines, "intermediate": a["result"], "bytes": nbytes})
    if not pairs or ctx.runtime_ms <= 0:
        return []
    saved = sum(memory_ms(p["bytes"], ctx.calib) for p in pairs)
    ratio = ctx.runtime_ms / max(ctx.runtime_ms - saved, ctx.runtime_ms * 0.05)
    if ratio < MIN_RATIO:
        return []
    lines = sorted({ln for p in pairs for ln in p["lines"]})
    return [Finding(
        id=ID, detector="D1",
        title=f"Unfused elementwise chain materializes {len(pairs)} intermediate{'s' if len(pairs) > 1 else ''}: "
              + ", ".join(f"L{ln}" for ln in lines),
        op_ids=[x for p in pairs for x in (p["producer_op"], p["consumer_op"])],
        evidence={
            "chains": pairs, "lines": lines,
            "intermediate_bytes": sum(p["bytes"] for p in pairs),
            "linalg_fuse_elementwise_in_pipeline": "linalg-fuse-elementwise-ops" in ctx.pipeline_yaml,
            "memory_bandwidth_gbs": ctx.calib.get("bandwidth_gbs"),
            "why": "each intermediate is written once and read once; fusing the two ops removes both passes over memory",
        },
        impact_est=est(ratio, f"{saved:.3f} ms of memory traffic (2x bytes / measured bandwidth) out of {ctx.runtime_ms:.3f} ms"),
        fix_ids=["enable-elementwise-fusion"],
    )]
