"""Rule 1 as code: every pass in a pipeline must exist in the pinned toolchain."""
from __future__ import annotations

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.passmanager import PassManager

from .config import PipelineConfig, expand, pass_name


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
            print(f"  [{stage}] {spec}  ({'composite' if composite else 'pass'}: {len(leaves)})")
            if composite:
                for leaf in leaves:
                    print(f"      - {pass_name(leaf)}")
    return bad
