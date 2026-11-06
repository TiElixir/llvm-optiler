#!/usr/bin/env python
"""Rule 1 gate: every pass in a pipeline config must exist in the pinned toolchain.

    .venv/bin/python scripts/check_passes.py [pipeline.yaml ...] [--expand]

Exit code 1 if any entry fails to parse. With --expand, prints the leaf passes each
entry expands to (composites are split one pass at a time at run time).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch_mlir import ir  # noqa: E402
from torch_mlir.dialects import torch as torch_d  # noqa: E402
from torch_mlir.passmanager import PassManager  # noqa: E402

from backend.pipeline.config import DEFAULT_PIPELINE, PipelineConfig, expand, pass_name  # noqa: E402


def make_context() -> ir.Context:
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    return ctx


def check_text(pipeline_yaml: str, expand_leaves: bool = False) -> list[tuple[str, str, str]]:
    """Return [(stage, entry, error)] for entries that fail to parse; empty means all verified."""
    ctx = make_context()
    cfg = PipelineConfig.from_yaml(pipeline_yaml)
    bad = []

    def to_text(spec: str) -> str:
        with ctx:
            return str(PassManager.parse(f"builtin.module({spec})"))

    for stage, spec in cfg.all_pass_texts():
        try:
            leaves, composite = expand(spec, to_text)
        except Exception as e:  # noqa: BLE001
            bad.append((stage, spec, str(e).strip().splitlines()[0]))
            continue
        if expand_leaves:
            tag = "composite" if composite else "pass"
            print(f"  [{stage}] {spec}  ({tag}: {len(leaves)})")
            if composite:
                for leaf in leaves:
                    print(f"      - {pass_name(leaf)}")
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = [Path(a) for a in args] or [DEFAULT_PIPELINE]
    rc = 0
    for p in paths:
        print(f"checking {p}")
        bad = check_text(p.read_text(), expand_leaves="--expand" in sys.argv)
        for stage, spec, err in bad:
            print(f"  FAIL [{stage}] {spec}\n       {err}")
        if bad:
            rc = 1
        else:
            print("  all passes verified")
    return rc


if __name__ == "__main__":
    sys.exit(main())
