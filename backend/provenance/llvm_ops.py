"""Op tables for the two stages below the LLVM dialect.

Both map through the *anchor file* (``anchor/llvm_dialect.mlir``): MLIR locations were rewritten to
anchor lines, so ``!dbg`` locations in LLVM IR and ``; …/llvm_dialect.mlir:N`` markers in
``llvm-objdump -l`` output name an LLVM-dialect op, which carries the user's source lines.
"""
from __future__ import annotations

import re

from .model import OpRec, StageProv

_DILOC = re.compile(r"^!(\d+) = (?:distinct )?!DILocation\(line: (\d+)")
_DBG = re.compile(r"!dbg !(\d+)")
_INSTR = re.compile(r"^  (?:%[^\s=]+ = )?(?:tail |musttail |notail )?([a-z_][a-z0-9_]*)\b")
_CALLEE = re.compile(r"@([\w.$]+)\(")
_ASM_MARK = re.compile(r"^;\s+\S*llvm_dialect\.mlir:(\d+)\s*$")
_ASM_SYM = re.compile(r"^[0-9a-f]{8,16} <([^>]+)>:")
_ASM_INSN = re.compile(r"^\s+([0-9a-f]+):\s+(\S+)")

LL_HIDDEN = {"br", "ret", "unreachable", "switch"}
ASM_HIDDEN = {"nop", "nopw", "nopl", "endbr64", "int3", "ud2", "cs", "data16"}
MODEL_SYMBOLS = {"forward", "_mlir_ciface_forward"}


def dialect_index(dialect: StageProv) -> dict[int, OpRec]:
    """anchor line -> the LLVM-dialect op printed there (displayable ops win)."""
    by_line: dict[int, OpRec] = {}
    for o in dialect.ops:
        if o.text_line is None:
            continue
        cur = by_line.get(o.text_line)
        if cur is None or (not cur.displayable and o.displayable):
            by_line[o.text_line] = o
    return by_line


def _inherit(rec: OpRec, parent: OpRec | None, anchor_line: int | None) -> None:
    rec.anchor_line = anchor_line
    if parent is not None and parent.displayable and parent.lines:
        rec.lines, rec.approximate = list(parent.lines), parent.approximate


def parse_llvm_ir(text: str, dialect: StageProv, snapshot_idx: int) -> StageProv:
    """``text`` is the final optimized module; line numbers match the stored (elided) snapshot."""
    diloc = {int(m.group(1)): int(m.group(2)) for line in text.split("\n") if (m := _DILOC.match(line))}
    by_line = dialect_index(dialect)
    prov = StageProv(stage="llvm_ir", snapshot_idx=snapshot_idx)
    in_func = False
    for n, line in enumerate(text.split("\n"), start=1):
        if line.startswith("define "):
            in_func = True
            continue
        if in_func and line.startswith("}"):
            in_func = False
            continue
        if not in_func:
            continue
        m = _INSTR.match(line)
        if not m:
            continue
        op = m.group(1)
        if op == "call" and (c := _CALLEE.search(line)):
            op = f"call {c.group(1)}"
        rec = OpRec(idx=len(prov.ops), name=op, text_line=n, displayable=m.group(1) not in LL_HIDDEN)
        d = _DBG.search(line)
        anchor = diloc.get(int(d.group(1))) if d else None
        _inherit(rec, by_line.get(anchor) if anchor else None, anchor)
        prov.ops.append(rec)
    return prov


def parse_asm(text: str, dialect: StageProv, snapshot_idx: int) -> StageProv:
    by_line = dialect_index(dialect)
    prov = StageProv(stage="asm", snapshot_idx=snapshot_idx)
    symbol, anchor = None, None
    for n, line in enumerate(text.split("\n"), start=1):
        if (s := _ASM_SYM.match(line)):
            symbol, anchor = s.group(1), None
            continue
        if symbol not in MODEL_SYMBOLS:
            continue
        if (m := _ASM_MARK.match(line)):
            anchor = int(m.group(1))
            continue
        m = _ASM_INSN.match(line)
        if not m:
            continue
        mnemonic = m.group(2)
        rec = OpRec(idx=len(prov.ops), name=mnemonic, text_line=n, displayable=mnemonic not in ASM_HIDDEN,
                    addr=int(m.group(1), 16))
        _inherit(rec, by_line.get(anchor) if anchor else None, anchor)
        prov.ops.append(rec)
    return prov
