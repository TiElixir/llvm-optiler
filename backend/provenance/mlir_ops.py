"""Op tables for stage-final MLIR modules.

For every stage the module is walked in print order, each op's location is resolved to
lines of the user's source (callsite/fused/name unwrapped), and the module is printed once
more by ``snapshot-op-locations`` on a clone so every op knows its *exact* line in the stored
snapshot text. That text (constants elided) replaces the snapshot file, so highlighted lines
in the inspector are exact rather than guessed.
"""
from __future__ import annotations

import re
from pathlib import Path

from torch_mlir import ir
from torch_mlir.passmanager import PassManager

from .locs import lines_of, parse_loc
from .model import OpRec, StageProv

CONTAINERS = {"builtin.module", "func.func", "llvm.func", "memref.global", "llvm.mlir.global"}
CONSTANTS = {"arith.constant", "llvm.mlir.constant", "llvm.mlir.undef", "llvm.mlir.poison",
             "torch.constant.int", "torch.constant.float", "torch.constant.bool", "torch.constant.str",
             "torch.constant.none", "torch.constant.device", "torch.vtensor.literal", "torch.prim.ListConstruct",
             "ub.poison", "memref.get_global"}
TERMINATORS = {"func.return", "llvm.return", "cf.br", "cf.cond_br", "llvm.br", "llvm.cond_br", "scf.yield",
               "affine.yield", "linalg.yield", "tm_tensor.yield", "llvm.unreachable", "return"}
HIDDEN = CONTAINERS | CONSTANTS | TERMINATORS


def walk(op: ir.Operation):
    """Pre-order walk (print order): yields (operation, depth, parent_walk_idx, in_linalg_region)."""
    out: list[tuple[ir.Operation, int, int, bool]] = []

    def rec(o: ir.Operation, depth: int, parent: int, payload: bool) -> None:
        idx = len(out)
        out.append((o, depth, parent, payload))
        inner = payload or o.name.startswith(("linalg.", "tm_tensor."))
        for region in o.regions:
            for block in region.blocks:
                for child in block.operations:
                    rec(child.operation, depth + 1, idx, inner)

    rec(op, 0, -1, False)
    return out


_DENSE = re.compile(r'dense<(?:"0x[0-9A-Fa-f]{16,}"|\[[^>]{48,}\])>')


def elide_mlir_text(text: str, max_line: int = 400) -> str:
    """Shorten big constants and drop the dialect-resource blob section; line numbers of ops stay."""
    cut = text.find("\n{-#")
    if cut != -1:
        text = text[:cut] + "\n"
    out = []
    for line in text.split("\n"):
        if len(line) > max_line:
            line = _DENSE.sub('dense<"__elided__">', line)
            if len(line) > max_line:
                line = line[:max_line] + " …"
        out.append(line)
    return "\n".join(out)


def _loc_line(loc_text: str) -> int | None:
    node = parse_loc(loc_text)
    return node[2] if node[0] == "file" else None


def collect_stage(module: ir.Module, stage: str, snapshot_idx: int, anchor_path: Path,
                  snapshot_path: Path, source: str = "source/model.py") -> StageProv:
    """Build the op table of ``module`` (state at the end of ``stage``) and rewrite its snapshot text."""
    ctx = module.context
    orig = walk(module.operation)

    clone = module.operation.clone()
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    with ctx:
        PassManager.parse(
            f"builtin.module(snapshot-op-locations{{filename={anchor_path} print-debuginfo=true}})"
        ).run(clone)
    anchored = walk(clone)
    assert len(anchored) == len(orig), "clone walk differs from the original walk"

    text = anchor_path.read_text()
    snapshot_path.write_text(elide_mlir_text(text))  # exact line numbers == text_line below

    recs: list[OpRec] = []
    index_of: dict[int, int] = {}
    for i, ((op, _depth, _parent, payload), (aop, _, _, _)) in enumerate(zip(orig, anchored)):
        loc_text = str(op.location)
        name = op.name
        displayable = name not in HIDDEN and not payload
        recs.append(OpRec(
            idx=i, name=name, lines=lines_of(loc_text, source), displayable=displayable,
            result_types=[str(r.type) for r in op.results], loc=loc_text[:200],
            text_line=_loc_line(str(aop.location)),
        ))
        index_of[hash(op)] = i

    _infer_missing(orig, recs, index_of)
    return StageProv(stage=stage, snapshot_idx=snapshot_idx, ops=recs)


def _owner_idx(value, index_of: dict[int, int]) -> int | None:
    owner = getattr(value, "owner", None)
    op = getattr(owner, "operation", None)
    if op is None:
        return None
    return index_of.get(hash(op))


def _infer_missing(orig, recs: list[OpRec], index_of: dict[int, int]) -> None:
    """Heuristic lines for ops whose location carries no user line. Always ``approximate``.

    Order: lines of operand producers, then of result users, then the nearest preceding
    sibling (or the enclosing op). Never presented as exact.
    """
    def fill(i: int, lines: list[int]) -> bool:
        if lines and not recs[i].lines:
            recs[i].lines, recs[i].approximate = sorted(set(lines)), True
            return True
        return False

    # 1. producers (forward pass so chains propagate)
    for i, (op, _d, _p, _pl) in enumerate(orig):
        if recs[i].lines or not recs[i].displayable:
            continue
        lines: list[int] = []
        for v in op.operands:
            j = _owner_idx(v, index_of)
            if j is not None:
                lines += recs[j].lines
        fill(i, lines)
    # 2. users (backward pass)
    for i in range(len(orig) - 1, -1, -1):
        op = orig[i][0]
        if recs[i].lines or not recs[i].displayable:
            continue
        lines = []
        for r in op.results:
            for use in r.uses:
                j = index_of.get(hash(use.owner.operation)) if getattr(use.owner, "operation", None) is not None else None
                if j is not None:
                    lines += recs[j].lines
        fill(i, lines)
    # 3. nearest preceding sibling with lines, else the parent's lines
    last_in_parent: dict[int, int] = {}
    for i, (_op, _d, parent, _pl) in enumerate(orig):
        if recs[i].lines or not recs[i].displayable:
            if recs[i].lines:
                last_in_parent[parent] = i
            continue
        j = last_in_parent.get(parent)
        if j is not None:
            fill(i, recs[j].lines)
        elif parent >= 0:
            fill(i, recs[parent].lines)
        if recs[i].lines:
            last_in_parent[parent] = i
