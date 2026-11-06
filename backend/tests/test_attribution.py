import shutil
from collections import Counter
from pathlib import Path

import pytest

from backend.pipeline.config import PipelineConfig
from backend.pipeline.run import compile_run
from backend.profile import perf, shares
from backend.provenance.build import read_stage, ORDER
from backend.schemas import FxNode, Metrics, Op, StageOps
from backend.toolchain import perf_status

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
\t    7f00aa forward+0x20 (/x/bin/model.so)
\t    7f00bb _mlir_ciface_forward+0x10 (/x/bin/model.so)

\t    7f11aa [unknown] (/usr/lib/libm.so.6)
\t    7f00cc forward+0x40 (/x/bin/model.so)
\t    7f00dd _mlir_ciface_forward+0x10 (/x/bin/model.so)

\t    7f22aa PyEval_EvalFrame+0x10 (/usr/bin/python3)
"""


def test_parse_script_uses_first_model_frame_and_return_address_minus_one():
    addrs, outside = perf.parse_script(SCRIPT, "model.so", {"forward": 0x1000, "_mlir_ciface_forward": 0x2000})
    assert addrs == [0x1020, 0x1040 - 1]      # leaf sample exact; libm sample -> caller's return address - 1
    assert outside == 1                       # the interpreter-only sample


def op(i, name, lines, text_line=None, addr=None, displayable=True):
    return Op(id=f"s:0:{i}", snapshot_idx=0, stage="s", name=name, source_lines=lines, text_line=text_line,
              address=addr, displayable=displayable)


def test_apply_perf_shares_sum_to_one_and_split_fused_lines():
    stages = {
        "llvm_dialect": StageOps(stage="llvm_dialect", ops=[
            op(0, "a", [7], text_line=10), op(1, "b", [7, 8], text_line=11), op(2, "func", [], text_line=3, displayable=False)]),
        "asm": StageOps(stage="asm", ops=[op(0, "mov", [7], addr=0x100), op(1, "add", [8], addr=0x110)]),
    }
    att = perf.Attribution(mode="lbr", total=10, other_samples=1)
    att.by_anchor_line.update({10: 5, 11: 2, 3: 2, 99: 0})  # 2 samples land on a container op
    att.by_addr.update({0x100: 6, 0x118: 3})                # 0x118 falls inside the instruction at 0x110
    res = shares.apply_perf(att, stages)
    assert res["per_line"] == pytest.approx({7: 0.5 + 0.1, 8: 0.1})
    assert res["other"] == pytest.approx(0.1 + 0.2)
    assert sum(res["per_line"].values()) + res["other"] == pytest.approx(1.0)
    assert stages["asm"].ops[0].share == pytest.approx(0.6) and stages["asm"].ops[1].share == pytest.approx(0.3)
    assert stages["llvm_dialect"].ops[0].share_kind == "perf"


def test_fill_metrics_picks_hottest_line_and_labels_attribution():
    m = Metrics()
    fx = [FxNode(name="mm", op="call_function", target="aten.mm", line=20, code="scores = q @ k")]
    shares.fill_metrics(m, {"per_line": {20: 0.43, 22: 0.02}, "other": 0.01}, "perf", fx)
    assert (m.attribution, m.hottest_line, m.hottest_line_code) == ("perf", 20, "scores = q @ k")
    assert m.per_line_share == {"20": 0.43, "22": 0.02}
    shares.fill_metrics(m, {"per_line": {}, "other": 0}, "est", fx, notice="perf not used")
    assert m.attribution == "est" and m.attribution_notice == "perf not used"


@pytest.mark.pipeline
@pytest.mark.skipif(not perf_status()["available"], reason="perf not usable here")
def test_perf_attribution_on_a_real_run(tmp_path):
    d = tmp_path / "run"
    (d / "source").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "templates" / "matmul_relu.py", d / "source" / "model.py")
    out = compile_run(d, PipelineConfig.load(), benchmark=False)
    att = perf.attribute(d, seconds=1.0)
    assert att.total > 500 and att.mode in ("lbr", "dwarf", "none")
    stages = {s: read_stage(d, s) for s in ORDER}
    res = shares.apply_perf(att, stages)
    assert sum(res["per_line"].values()) + res["other"] == pytest.approx(1.0, abs=0.05)
    assert sum(res["per_line"].values()) > 0.9
    # the scalar 256^3 matmul on the first source line dominates the bias+relu
    lines = sorted(res["per_line"])
    assert max(res["per_line"], key=res["per_line"].get) == lines[0]
    asm_total = sum(o.share or 0 for o in stages["asm"].ops)
    assert asm_total == pytest.approx(1.0, abs=0.05)


def _run_job(tmp_path, monkeypatch, attribution):
    from backend.runner.job import run_job
    from backend.schemas import Run
    from backend.store.runs import RunStore
    monkeypatch.setenv("OPTILER_ATTRIBUTION", attribution)
    store = RunStore(tmp_path / "runs")
    src = (ROOT / "backend" / "templates" / "attention_block.py").read_text()
    run = store.create(Run(id="", created_at="", source_code=src, pipeline_yaml=(ROOT / "backend/pipelines/default.yaml").read_text(),
                           settings={"iters": 5, "warmup": 1, "threads": 1}))
    run_job(store, run.id)
    return store.get(run.id)


@pytest.mark.slow
@pytest.mark.skipif(not perf_status()["available"], reason="perf not usable here")
def test_job_measures_with_perf_and_reports_the_hottest_line(tmp_path, monkeypatch):
    run = _run_job(tmp_path, monkeypatch, "auto")
    m = run.metrics
    assert run.status == "done" and m.correct and m.attribution == "perf"
    assert sum(m.per_line_share.values()) + m.attribution_other == pytest.approx(1.0, abs=0.05)
    assert m.hottest_line is not None and m.hottest_line_code and m.hottest_line_share > 0.2


@pytest.mark.slow
def test_forced_estimate_is_labeled_est(tmp_path, monkeypatch):
    run = _run_job(tmp_path, monkeypatch, "est")
    m = run.metrics
    assert run.status == "done" and m.attribution == "est"
    assert "estimate" in m.attribution_notice
    assert sum(m.per_line_share.values()) == pytest.approx(1.0, abs=0.01)
    assert m.hottest_line is not None
