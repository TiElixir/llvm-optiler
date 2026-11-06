"""Shared record types for provenance."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpRec:
    idx: int                      # walk / ordinal index within the stage table
    name: str
    lines: list[int] = field(default_factory=list)   # user source lines
    approximate: bool = False
    displayable: bool = True
    result_types: list[str] = field(default_factory=list)
    loc: str | None = None
    text_line: int | None = None  # 1-based line in the displayed snapshot text
    anchor_line: int | None = None  # LLVM dialect only: line in anchor/llvm_dialect.mlir (== text_line)


@dataclass
class StageProv:
    stage: str
    snapshot_idx: int
    ops: list[OpRec] = field(default_factory=list)
    # llvm_ir only: ops[i] came from the dialect op at this anchor line (exact, via !dbg)
    exact_parent_lines: dict[int, int] = field(default_factory=dict)

    def displayable(self) -> list[OpRec]:
        return [o for o in self.ops if o.displayable]

    def exact_pct(self) -> float | None:
        d = self.displayable()
        if not d:
            return None
        return sum(1 for o in d if o.lines and not o.approximate) / len(d)
