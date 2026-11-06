"""End-to-end: every shipped template compiles through the real pipeline and matches eager."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.pipeline.config import PipelineConfig
from backend.pipeline.run import compile_run, run_bench_subprocess

pytestmark = pytest.mark.pipeline

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
TEMPLATES = sorted(p.stem for p in TEMPLATES_DIR.glob("*.py") if not p.name.startswith("_"))
REQUIRED_STAGES = {"torch", "linalg_tensor", "linalg_buffer", "loops", "llvm_dialect", "llvm_ir", "asm"}
EXPECTED_TEMPLATES = {"matmul_relu", "mlp", "softmax", "layernorm", "attention_block"}


def test_all_expected_templates_present():
    assert EXPECTED_TEMPLATES <= set(TEMPLATES)


@pytest.mark.parametrize("name", sorted(EXPECTED_TEMPLATES))
def test_template_compiles_and_matches_eager(name, tmp_path):
    run_dir = tmp_path / name
    (run_dir / "source").mkdir(parents=True)
    shutil.copy(TEMPLATES_DIR / f"{name}.py", run_dir / "source" / "model.py")

    out = compile_run(run_dir, PipelineConfig.load(), benchmark=False)
    records = out["records"]

    assert len(records) > 40
    for r in records:
        assert (run_dir / r.path).is_file(), f"missing snapshot file for record {r.idx} ({r.stage}/{r.pass_name}): {r.path}"
    stages = {r.stage for r in records}
    assert REQUIRED_STAGES <= stages, f"missing stages: {sorted(REQUIRED_STAGES - stages)}"

    bench = run_bench_subprocess(run_dir, iters=5, warmup=1)
    assert bench["correct"], f"compiled output differs from eager (max_abs_err={bench['max_abs_err']})"
