"""Provenance end to end: source line -> every stage -> LLVM IR -> assembly."""
import json
import shutil
from pathlib import Path

import pytest
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

from backend.pipeline.config import PipelineConfig
from backend.pipeline.run import compile_run
from backend.provenance.build import ORDER, read_stage
from backend.provenance.mlir_ops import collect_stage, elide_mlir_text

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.pipeline


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("prov") / "run"
    (d / "source").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "templates" / "matmul_relu.py", d / "source" / "model.py")
    out = compile_run(d, PipelineConfig.load(), benchmark=False)
    return d, out


def stage(run, name):
    return read_stage(run[0], name)


def test_every_stage_has_a_table_and_coverage(run):
    for s in ORDER:
        so = stage(run, s)
        assert so is not None and so.ops, s
        assert 0.0 <= (so.exact_lineage_pct or 0.0) <= 1.0
    assert stage(run, "source").exact_lineage_pct == 1.0


def test_the_relu_line_reaches_assembly(run):
    """Known op -> lineage into asm: every stage has displayable ops for each source line."""
    lines = sorted({ln for o in stage(run, "source").ops for ln in o.source_lines})
    assert len(lines) >= 2
    for s in ORDER:
        for ln in lines:
            hits = [o for o in stage(run, s).ops if o.displayable and ln in o.source_lines]
            assert hits, f"no ops for source line {ln} in stage {s}"


def test_mlir_text_lines_are_exact(run):
    """Each op's text_line points at the line in the stored snapshot that prints that op."""
    d, out = run
    snaps = {r.idx: r for r in out["records"]}
    checked = 0
    for s in ["torch", "linalg_tensor", "linalg_buffer", "loops", "llvm_dialect"]:
        so = stage(run, s)
        text = (d / snaps[so.snapshot_idx].path).read_text().split("\n")
        for o in so.ops:
            if o.displayable:
                assert o.text_line is not None, (s, o.name)
                assert o.name in text[o.text_line - 1], (s, o.name, text[o.text_line - 1])
                checked += 1
    assert checked > 100


def test_llvm_and_asm_lines_point_at_their_instructions(run):
    d, out = run
    snaps = {r.idx: r for r in out["records"]}
    ir_text = (d / snaps[stage(run, "llvm_ir").snapshot_idx].path).read_text().split("\n")
    for o in stage(run, "llvm_ir").ops[:80]:
        assert o.name.split()[0] in ir_text[o.text_line - 1]
    asm_text = (d / snaps[stage(run, "asm").snapshot_idx].path).read_text().split("\n")
    for o in stage(run, "asm").ops[:80]:
        assert o.name in asm_text[o.text_line - 1]


def test_links_connect_consecutive_stages_and_flag_approximate(run):
    for prev, cur in zip(ORDER, ORDER[1:]):
        so, ps = stage(run, cur), stage(run, prev)
        ids = {o.id for o in ps.ops}
        assert so.links, cur
        for l in so.links:
            assert l.from_op in ids and l.to_op.startswith(cur + ":")
    # LLVM IR instructions link to their exact dialect op through the anchor line
    assert any(not l.approximate for l in stage(run, "llvm_ir").links)


def test_snapshot_files_stay_readable(run):
    d, out = run
    for r in out["records"]:
        assert (d / r.path).read_text().strip(), r.path


def test_unknown_locations_get_flagged_heuristic_lines(tmp_path):
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    with ctx:
        m = ir.Module.parse('''
module {
  func.func @f(%a: f32) -> f32 {
    %0 = arith.addf %a, %a : f32 loc("source/model.py":4:0)
    %1 = arith.mulf %0, %0 : f32 loc(unknown)
    %2 = arith.addf %1, %1 : f32 loc(unknown)
    return %2 : f32 loc("source/model.py":5:0)
  }
}''')
    snap = tmp_path / "s.mlir"
    prov = collect_stage(m, "loops", 3, tmp_path / "anchor.mlir", snap)
    byname = {o.name: o for o in prov.ops if o.displayable}
    adds = [o for o in prov.ops if o.name == "arith.addf"]
    mul = byname["arith.mulf"]
    assert adds[0].lines == [4] and not adds[0].approximate           # exact
    assert mul.lines == [4] and mul.approximate                        # inherited from its producer
    assert adds[1].lines == [4] and adds[1].approximate                # chain propagates
    assert prov.exact_pct() == pytest.approx(1 / 3)


def test_elision_keeps_line_numbers():
    text = "module {\n  %0 = arith.constant dense<[" + ", ".join(["1.0"] * 200) + "]> : tensor<200xf32> loc(#loc1)\n  %1 = foo\n}\n{-#\n  dialect_resources: { blob }\n#-}\n"
    out = elide_mlir_text(text)
    assert out.split("\n")[2].strip() == "%1 = foo"
    assert "__elided__" in out and "{-#" not in out
