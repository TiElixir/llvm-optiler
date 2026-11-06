"""Pipeline config: YAML model, textual-pipeline splitter and composite expansion.

A pipeline entry is a textual MLIR pass pipeline (e.g. ``func.func(canonicalize)``).
Runs execute one leaf pass at a time so every pass gets its own snapshot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field

STAGES = [
    "source", "torch", "linalg_tensor", "linalg_buffer",
    "loops", "llvm_dialect", "llvm_ir", "asm",
]
MLIR_STAGES = ["torch", "linalg_tensor", "linalg_buffer", "loops", "llvm_dialect"]

DEFAULT_PIPELINE = Path(__file__).resolve().parent.parent / "pipelines" / "default.yaml"


class PassSpec(BaseModel):
    pass_: str = Field(alias="pass")
    options: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}

    def text(self) -> str:
        if not self.options:
            return self.pass_
        opts = " ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in self.options.items())
        if self.pass_.endswith("}"):
            return self.pass_[:-1] + " " + opts + "}"
        return f"{self.pass_}{{{opts}}}"


class StageSpec(BaseModel):
    stage: str
    passes: list[PassSpec]


class PipelineConfig(BaseModel):
    stages: list[StageSpec]

    @classmethod
    def from_yaml(cls, text: str) -> "PipelineConfig":
        raw = yaml.safe_load(text)
        if isinstance(raw, dict):
            raw = raw.get("stages", [])
        stages = []
        for entry in raw:
            passes = [
                PassSpec(**{"pass": p}) if isinstance(p, str) else PassSpec(**p)
                for p in entry["passes"]
            ]
            if entry["stage"] not in MLIR_STAGES:
                raise ValueError(f"unknown stage {entry['stage']!r}; expected one of {MLIR_STAGES}")
            stages.append(StageSpec(stage=entry["stage"], passes=passes))
        return cls(stages=stages)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PIPELINE) -> "PipelineConfig":
        return cls.from_yaml(Path(path).read_text())

    def to_yaml(self) -> str:
        out = []
        for s in self.stages:
            out.append({"stage": s.stage, "passes": [p.text() for p in s.passes]})
        return yaml.safe_dump(out, sort_keys=False, width=200)

    def all_pass_texts(self) -> list[tuple[str, str]]:
        return [(s.stage, p.text()) for s in self.stages for p in s.passes]


# ---------------------------------------------------------------- splitter

def _split_top(s: str) -> list[str]:
    """Split on commas that are outside any ( ) or { }."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] in "({":
            depth += 1
        elif s[i] in ")}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced parentheses in pipeline: {s!r}")


def split_pipeline(text: str) -> list[str]:
    """Flatten a textual pipeline into anchored single-pass strings.

    ``builtin.module(a,func.func(b,c))`` -> ``['a', 'func.func(b)', 'func.func(c)']``.
    ``builtin.module`` is a transparent anchor (PassManager.parse adds it back).
    """
    leaves: list[str] = []

    def walk(s: str, anchors: list[str]) -> None:
        for item in _split_top(s):
            idx = [i for i in (item.find("("), item.find("{")) if i != -1]
            first = min(idx) if idx else -1
            if first != -1 and item[first] == "(":
                end = _matching_paren(item, first)
                anchor = item[:first].strip()
                inner = item[first + 1:end]
                walk(inner, anchors if anchor == "builtin.module" else anchors + [anchor])
            else:
                leaf = item
                for a in reversed(anchors):
                    leaf = f"{a}({leaf})"
                leaves.append(leaf)

    walk(text, [])
    return leaves


def pass_name(leaf: str) -> str:
    """Display name of an anchored leaf: ``func.func(canonicalize{...})`` -> ``canonicalize``."""
    s = leaf
    while True:
        p, b = s.find("("), s.find("{")
        if p != -1 and (b == -1 or p < b):
            s = s[p + 1:s.rfind(")")]
        else:
            break
    b = s.find("{")
    return (s[:b] if b != -1 else s).strip()


def expand(spec: str, parse_and_print) -> tuple[list[str], bool]:
    """Expand one YAML entry into leaf passes.

    ``parse_and_print`` turns a pipeline string into the textual form MLIR prints
    after parsing it (``str(PassManager.parse(f"builtin.module({spec})"))``).
    Returns ``(leaves, composite)``; a plain pass keeps the user's own spelling.
    """
    printed = parse_and_print(spec)
    leaves = split_pipeline(printed)
    if len(leaves) == 1 and pass_name(leaves[0]) == pass_name(spec):
        return [spec], False
    return leaves, True
