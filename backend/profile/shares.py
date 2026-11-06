"""Turn attribution (measured or estimated) into per-line / per-op shares on the stage tables."""
from __future__ import annotations

from collections import defaultdict

from backend.schemas import FxNode, Metrics, StageOps

from .perf import Attribution, nearest_instruction


def _split(share: float, lines: list[int], into: dict[int, float]) -> None:
    for ln in lines:
        into[ln] += share / len(lines)


def apply_perf(att: Attribution, stages: dict[str, StageOps]) -> dict:
    """Shares from perf samples. Sum of per-line shares + ``other`` == 1 (samples inside model.so)."""
    total = max(att.total, 1)
    per_line: dict[int, float] = defaultdict(float)
    other = att.other_samples / total

    dial = stages.get("llvm_dialect")
    if dial:
        by_text = {o.text_line: o for o in dial.ops if o.text_line is not None}
        for anchor, n in att.by_anchor_line.items():
            op = by_text.get(anchor)
            share = n / total
            if op is not None and op.displayable and op.source_lines:
                op.share, op.share_kind = (op.share or 0.0) + share, "perf"
                _split(share, op.source_lines, per_line)
            else:  # prologue, container ops, instructions without a source line
                other += share

    asm = stages.get("asm")
    if asm:
        ops = [o for o in asm.ops if o.address is not None]
        addrs = [o.address for o in ops]  # objdump order == ascending address
        for addr, n in att.by_addr.items():
            i = nearest_instruction(addrs, addr)
            if i is not None:
                ops[i].share, ops[i].share_kind = (ops[i].share or 0.0) + n / total, "perf"

    return {"per_line": dict(per_line), "other": other}


def apply_est(per_line: dict[int, float], other: float = 0.0) -> dict:
    return {"per_line": dict(per_line), "other": other}


def fill_metrics(m: Metrics, result: dict, kind: str, fx_nodes: list[FxNode], notice: str | None = None) -> None:
    per_line = result["per_line"]
    m.per_line_share = {str(k): round(v, 5) for k, v in sorted(per_line.items())}
    m.attribution, m.attribution_notice, m.attribution_other = kind, notice, round(result["other"], 5)
    if per_line:
        line, share = max(per_line.items(), key=lambda kv: kv[1])
        m.hottest_line, m.hottest_line_share = line, round(share, 5)
        node = next((n for n in fx_nodes if n.line == line and n.code), None)
        m.hottest_line_code = node.code if node else None
