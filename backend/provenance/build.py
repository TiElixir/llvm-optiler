"""Assemble per-stage op tables into API objects: links between stages and coverage."""
from __future__ import annotations

import json
from pathlib import Path

from backend.schemas import FxNode, Link, Op, StageOps

from .model import OpRec, StageProv

ORDER = ["source", "torch", "linalg_tensor", "linalg_buffer", "loops", "llvm_dialect", "llvm_ir", "asm"]
MAX_LINKS_PER_OP = 4


def source_stage(fx_nodes: list[FxNode] | list[dict]) -> StageProv:
    """The root of every lineage: one op per FX call node, identified by its source line."""
    prov = StageProv(stage="source", snapshot_idx=0)
    for n in fx_nodes:
        n = n if isinstance(n, dict) else n.model_dump()
        if n["op"] != "call_function" or n["line"] is None:
            continue
        prov.ops.append(OpRec(idx=len(prov.ops), name=n["target"].replace("torch.ops.", ""),
                              lines=[n["line"]], loc=n.get("code")))
    return prov


def op_id(p: StageProv, o: OpRec) -> str:
    return f"{p.stage}:{p.snapshot_idx}:{o.idx}"


def _links(prev: StageProv, cur: StageProv) -> list[Link]:
    """Shared-origin links. Exact when both ends have exact lines; ``approximate`` otherwise.

    llvm_ir and asm instructions link through their anchor line (precise); everything else links
    through shared source lines. Each op gets at most MAX_LINKS_PER_OP links; a dense group's full
    relation is carried by ``source_lines`` instead of enumerating every pair.
    """
    by_line: dict[int, list[OpRec]] = {}
    by_anchor: dict[int, list[OpRec]] = {}
    for o in prev.displayable():
        for ln in o.lines:
            by_line.setdefault(ln, []).append(o)
        if o.anchor_line is not None:
            by_anchor.setdefault(o.anchor_line, []).append(o)
    if cur.stage == "llvm_ir":  # previous stage is the dialect: its op *is* the anchor line
        by_anchor = {o.text_line: [o] for o in prev.ops if o.text_line is not None and o.displayable}

    links: list[Link] = []
    for o in cur.displayable():
        cands: list[OpRec] = []
        if cur.stage in ("llvm_ir", "asm") and o.anchor_line is not None:
            cands = by_anchor.get(o.anchor_line, [])
        if not cands:
            seen: set[int] = set()
            for ln in o.lines:
                for p in by_line.get(ln, []):
                    if p.idx not in seen:
                        seen.add(p.idx)
                        cands.append(p)
        for p in cands[:MAX_LINKS_PER_OP]:
            links.append(Link(from_op=op_id(prev, p), to_op=op_id(cur, o),
                              approximate=o.approximate or p.approximate))
    return links


def stage_ops(cur: StageProv, prev: StageProv | None) -> StageOps:
    ops = [Op(
        id=op_id(cur, o), snapshot_idx=cur.snapshot_idx, stage=cur.stage, name=o.name, loc=o.loc,
        source_line=o.lines[0] if o.lines else None, source_lines=o.lines, approximate=o.approximate,
        displayable=o.displayable, result_types=o.result_types, text_line=o.text_line,
    ) for o in cur.ops]
    return StageOps(stage=cur.stage, snapshot_idx=None if cur.stage == "source" else cur.snapshot_idx, ops=ops,
                    links=_links(prev, cur) if prev else [], exact_lineage_pct=cur.exact_pct())


def build_all(provs: dict[str, StageProv]) -> dict[str, StageOps]:
    out: dict[str, StageOps] = {}
    prev: StageProv | None = None
    for s in ORDER:
        p = provs.get(s)
        if p is None:
            continue
        out[s] = stage_ops(p, prev)
        prev = p
    return out


def write_all(run_dir: Path, stage_ops_by_name: dict[str, StageOps]) -> None:
    d = Path(run_dir) / "provenance"
    d.mkdir(exist_ok=True)
    for name, so in stage_ops_by_name.items():
        (d / f"{name}.json").write_text(so.model_dump_json())


def read_stage(run_dir: Path, stage: str) -> StageOps | None:
    p = Path(run_dir) / "provenance" / f"{stage}.json"
    return StageOps.model_validate(json.loads(p.read_text())) if p.exists() else None
