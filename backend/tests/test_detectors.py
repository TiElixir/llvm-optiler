"""Each detector has a positive and a negative model; a model with nothing to fix yields zero findings."""
import re
from pathlib import Path

import pytest

from backend.runner.job import run_job
from backend.schemas import Run
from backend.store.runs import RunStore

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "backend" / "tests" / "models"
pytestmark = [pytest.mark.pipeline, pytest.mark.slow]


def job(tmp_path, monkeypatch, source_path: Path, drop: tuple[str, ...] = (), attribution="auto"):
    monkeypatch.setenv("OPTILER_ATTRIBUTION", attribution)
    yaml = (ROOT / "backend/pipelines/default.yaml").read_text()
    for name in drop:  # remove a pass from the pipeline text
        yaml = "\n".join(l for l in yaml.splitlines() if name not in l)
    store = RunStore(tmp_path / "runs")
    run = store.create(Run(id="", created_at="", source_code=source_path.read_text(), pipeline_yaml=yaml,
                           settings={"iters": 20, "warmup": 3, "threads": 1}))
    run_job(store, run.id)
    r = store.get(run.id)
    assert r.status == "done" and r.metrics.correct, r.error
    return r


def ids(run):
    return {f.id for f in run.findings}


def test_d1_fires_when_fusion_is_missing_and_stays_quiet_when_fused(tmp_path, monkeypatch):
    unfused = job(tmp_path / "a", monkeypatch, MODELS / "elementwise_chain.py", drop=("linalg-fuse-elementwise-ops",))
    f = next(f for f in unfused.findings if f.id == "d1-unfused-elementwise")
    assert f.impact_est.kind == "est" and f.impact_est.ratio >= 1.05 and f.fix_ids == ["enable-elementwise-fusion"]
    assert f.evidence["chains"] and f.evidence["linalg_fuse_elementwise_in_pipeline"] is False
    fused = job(tmp_path / "b", monkeypatch, MODELS / "elementwise_chain.py")
    assert "d1-unfused-elementwise" not in ids(fused)


def test_d2_fires_on_a_temporary_copy_and_not_on_a_copy_of_the_input(tmp_path, monkeypatch):
    cat = job(tmp_path / "a", monkeypatch, MODELS / "copy_cat.py")
    f = next(f for f in cat.findings if f.id == "d2-temporary-copy")
    assert f.evidence["copy_bytes"] >= 512 * 512 * 4 and f.fix_ids == []
    assert len(f.evidence["copies"]) == 1  # only the copy of the computed temporary, not of the argument x


def test_d3_fires_on_the_scalar_matmul_and_not_on_models_without_one(tmp_path, monkeypatch):
    mm = job(tmp_path / "a", monkeypatch, ROOT / "backend/templates/matmul_relu.py")
    f = next(f for f in mm.findings if f.id == "d3-unvectorized-hot-loop")
    assert f.fix_ids == ["vectorize-reduction-loops"] and f.impact_est.ratio > 1.5
    assert f.evidence["llvm_remark"] and f.evidence["hot_instructions_scalar"] is True
    soft = job(tmp_path / "b", monkeypatch, ROOT / "backend/templates/softmax.py")
    assert "d3-unvectorized-hot-loop" not in ids(soft)


def test_a_model_with_nothing_to_fix_has_zero_findings(tmp_path, monkeypatch):
    assert job(tmp_path, monkeypatch, MODELS / "clean.py").findings == []


def test_findings_are_ranked_by_estimated_impact(tmp_path, monkeypatch):
    run = job(tmp_path, monkeypatch, ROOT / "backend/templates/attention_block.py")
    ratios = [f.impact_est.ratio for f in run.findings]
    assert ratios == sorted(ratios, reverse=True) and "d3-unvectorized-hot-loop" in ids(run)


def test_the_whole_loop_apply_the_top_finding_and_prove_it(tmp_path, monkeypatch):
    """PLAN.md M6: on attention_block, applying the top finding gives a measurably faster child, finding verified."""
    from backend.fixes.registry import FIXES
    from backend.fixes.apply import apply_patch
    from backend.pipeline.config import PipelineConfig
    parent = job(tmp_path, monkeypatch, ROOT / "backend/templates/attention_block.py")
    top = parent.findings[0]
    assert top.id == "d3-unvectorized-hot-loop" and top.impact_verified is None
    store = RunStore(tmp_path / "runs")
    fix = FIXES[top.fix_ids[0]]
    child = store.create(Run(id="", created_at="", source_code=parent.source_code, template=parent.template,
                             pipeline_yaml=apply_patch(PipelineConfig.from_yaml(parent.pipeline_yaml), fix).to_yaml(),
                             parent_id=parent.id, fix_id=fix.id, label=fix.label, settings=parent.settings))
    run_job(store, child.id)
    child = store.get(child.id)
    assert child.status == "done" and child.metrics.correct, child.error
    assert child.metrics.runtime_ms_median < parent.metrics.runtime_ms_median / 3
    reread = store.get(parent.id)
    f = next(f for f in reread.findings if f.id == top.id)
    assert f.impact_verified.kind == "verified" and f.impact_verified.ratio > 3 and f.child_run_id == child.id
    assert "est" == f.impact_est.kind and f.impact_verified.ratio > f.impact_est.ratio  # the estimate was conservative
