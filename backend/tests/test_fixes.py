import pytest

from backend.fixes.apply import apply_patch, verdict
from backend.fixes.registry import FIXES
from backend.pipeline.config import DEFAULT_PIPELINE, PipelineConfig
from backend.schemas import Fix, Metrics, PipelinePatchOp


def passes(cfg):
    return [(s.stage, p.text()) for s in cfg.stages for p in s.passes]


def test_d3_fix_removes_affine_fusion_and_adds_the_transform_after_linalg():
    base = PipelineConfig.load()
    child = apply_patch(base, FIXES["vectorize-reduction-loops"])
    b, c = passes(base), passes(child)
    assert any("affine-loop-fusion" in t for _, t in b) and not any("affine-loop-fusion" in t for _, t in c)
    added = [x for x in c if x not in b]
    assert [s for s, _ in added] == ["linalg_tensor", "linalg_tensor"]
    assert "transform-preload-library" in added[0][1] and "${RUN_DIR}/transforms/vectorize-reduction-loops.mlir" in added[0][1]
    assert added[1][1] == "transform-interpreter"
    # the child differs from the parent ONLY by the patch
    assert [x for x in b if x not in c] == [x for x in b if "affine-loop-fusion" in x[1]]


def test_apply_does_not_mutate_the_input_and_is_idempotent():
    base = PipelineConfig.load()
    snapshot = base.to_yaml()
    fix = FIXES["enable-elementwise-fusion"]
    once = apply_patch(base, fix)
    assert base.to_yaml() == snapshot
    twice = apply_patch(once, fix)
    assert passes(once) == passes(twice) or sum("linalg-fuse-elementwise-ops" in t for _, t in passes(twice)) == 2  # default already has one


def test_insert_before_after_replace_and_errors():
    cfg = PipelineConfig.load()
    f = lambda *ops: Fix(id="t", label="t", patch=list(ops))  # noqa: E731
    out = passes(apply_patch(cfg, f(PipelinePatchOp(op="insert_before", stage="linalg_buffer", target="one-shot-bufferize", value="eliminate-empty-tensors"))))
    i = [t for s, t in out if s == "linalg_buffer"]
    assert i[0] == "eliminate-empty-tensors" and "one-shot-bufferize" in i[1]
    out = passes(apply_patch(cfg, f(PipelinePatchOp(op="replace", target="lower-affine", value="lower-affine"))))
    assert ("loops", "lower-affine") in out
    with pytest.raises(ValueError):
        apply_patch(cfg, f(PipelinePatchOp(op="insert_after", stage="nope", value="cse")))
    with pytest.raises(ValueError):
        apply_patch(cfg, f(PipelinePatchOp(op="insert_before", stage="loops", target="does-not-exist", value="cse")))
    with pytest.raises(ValueError):
        apply_patch(cfg, f(PipelinePatchOp(op="replace", target="does-not-exist", value="cse")))


def m(median, p10, p90, correct=True):
    return Metrics(runtime_ms_median=median, runtime_ms_p10=p10, runtime_ms_p90=p90, correct=correct, iters=100)


def test_verdict_verified_needs_non_overlapping_ranges_and_5_percent():
    v = verdict(m(6.7, 6.4, 7.0), m(0.52, 0.50, 0.53))
    assert v.kind == "verified" and v.ratio == pytest.approx(12.88, abs=0.05)
    assert verdict(m(1.00, 0.9, 1.3), m(0.90, 0.8, 1.2)).kind == "none"          # ranges overlap
    assert verdict(m(1.00, 0.99, 1.01), m(0.97, 0.96, 0.98)).kind == "none"      # only 3% faster
    slow = verdict(m(1.0, 0.99, 1.01), m(2.0, 1.9, 2.1))
    assert slow.kind == "slower" and slow.ratio == pytest.approx(0.5)
    assert verdict(m(1.0, 0.9, 1.1), m(0.5, 0.4, 0.6, correct=False)).kind == "broken"
    assert verdict(Metrics(), m(1, 1, 1)).kind == "none"


def test_registry_fixes_are_well_formed():
    assert set(FIXES) == {"vectorize-reduction-loops", "enable-elementwise-fusion"}
    for f in FIXES.values():
        assert f.patch and f.label and f.description
    assert "transform.structured.interchange" in FIXES["vectorize-reduction-loops"].transform_script
    PipelineConfig.from_yaml(apply_patch(PipelineConfig.load(), FIXES["vectorize-reduction-loops"]).to_yaml())  # round-trips
