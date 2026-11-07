"""D3: a hot matmul loop that LLVM could not vectorize.

Certain only when all of these hold: a source line has >= 10% of the runtime, a ``linalg.matmul`` sits on it,
``loop-vectorize`` reported a missed vectorization (``CantReorderFPOps``: the reduction cannot be reordered, or
``UnsafeDep``) at an LLVM-dialect op of that line, and the instructions that burn that line's time are scalar.
(Other loops sharing the line, e.g. a transpose, may well be vectorized; they are not the hot ones.)
"""
from __future__ import annotations

from backend.schemas import Finding

from .base import Context, MIN_RATIO, amdahl, est, parse_remarks

MIN_SHARE = 0.10
BLOCKERS = ("CantReorderFPOps", "UnsafeDep")
HOT_COVERAGE = 0.6  # the instructions covering this share of a line's samples are "the hot ones"
ID = "d3-unvectorized-hot-loop"


def _packed(mnemonic: str) -> bool:
    return mnemonic.endswith(("ps", "pd")) and mnemonic.startswith(("v", "mul", "add", "sub", "div", "fmadd", "movu", "mova"))


def detect(ctx: Context) -> list[Finding]:
    dialect = {o.text_line: o for o in ctx.stage_ops("llvm_dialect") if o.text_line is not None}
    remarks = [r for r in parse_remarks(ctx.run_dir / "reports" / "remarks.yaml")
               if r["pass"] == "loop-vectorize" and r["name"] in BLOCKERS and r["line"] in dialect]
    remark_lines: dict[int, str] = {}
    for r in remarks:
        for ln in dialect[r["line"]].source_lines:
            remark_lines.setdefault(ln, r["message"])
    matmul_lines = {ln for o in ctx.stage_ops("linalg_tensor") if o.name == "linalg.matmul" for ln in o.source_lines}
    asm = [o for o in ctx.stage_ops("asm") if o.displayable]

    lines = []
    for ln in sorted(remark_lines):
        if ln not in matmul_lines or ctx.share(ln) < MIN_SHARE:
            continue
        on_line = [o for o in asm if ln in o.source_lines]
        measured = sorted((o for o in on_line if o.share), key=lambda o: -o.share)
        if measured:  # perf: look only at the instructions that actually run hot
            need, acc, hot_ops = HOT_COVERAGE * sum(o.share for o in measured), 0.0, []
            for o in measured:
                hot_ops.append(o); acc += o.share
                if acc >= need:
                    break
            if any(_packed(o.name) for o in hot_ops):
                continue
        elif any(_packed(o.name) for o in on_line):
            continue  # no samples to tell the hot loop apart: any packed code on the line vetoes
        lines.append(ln)
    if not lines:
        return []

    shares = {ln: ctx.share(ln) for ln in lines}
    total = sum(shares.values())
    width = max(2, min(ctx.lanes, 8) // 2)  # conservative: half of the vector width, at most 4x
    ratio = amdahl(total * (1 - 1 / width))
    if ratio < MIN_RATIO:
        return []

    hot = sorted((o for o in asm if set(o.source_lines) & set(lines) and o.share), key=lambda o: -(o.share or 0))[:6]
    asm_text = (ctx.run_dir / "asm" / "model.s").read_text().split("\n") if (ctx.run_dir / "asm" / "model.s").exists() else []
    excerpt = [asm_text[o.text_line - 1].strip() for o in hot if o.text_line and o.text_line <= len(asm_text)]
    op_ids = [o.id for st in ("linalg_tensor", "loops", "asm") for o in ctx.stage_ops(st)
              if o.displayable and set(o.source_lines) & set(lines)][:300]
    return [Finding(
        id=ID, detector="D3",
        title="Hot matmul loop is not vectorized: " + ", ".join(f"L{ln}" for ln in lines),
        op_ids=op_ids,
        evidence={
            "lines": lines,
            "code": {str(ln): ctx.code(ln) for ln in lines},
            "runtime_share": {str(ln): round(s, 4) for ln, s in shares.items()},
            "attribution": ctx.metrics.attribution,
            "llvm_remark": sorted(set(remark_lines[ln] for ln in lines)),
            "remark_source": "opt -pass-remarks-missed=loop-vectorize (reports/remarks.yaml)",
            "hot_instructions_scalar": True,
            "hot_asm": excerpt,
            "why": "the innermost loop is the floating-point reduction; LLVM may not reorder FP adds, so it stays scalar",
        },
        impact_est=est(ratio, f"Amdahl with a conservative {width}x vector width over {total:.0%} of runtime"),
        fix_ids=["vectorize-reduction-loops"],
    )]
