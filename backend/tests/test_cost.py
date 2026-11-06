"""Static cost model, bufferized allocation summary, estimated shares and calibration."""
import json
import shutil
from pathlib import Path

import pytest
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

from backend.pipeline.capture import SRC_REL, run_capture
from backend.pipeline.config import PipelineConfig
from backend.pipeline.mlir_runner import run_mlir_pipeline
from backend.profile import calibrate
from backend.profile.calibrate import est_line_shares, est_total_ms, load_or_run
from backend.profile.cost import compute_costs, static_alloc_summary
from backend.provenance.mlir_ops import walk
from backend.toolchain import ToolMissing

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ["matmul_relu", "mlp", "softmax", "layernorm", "attention_block"]
MATMUL_256_FLOPS = 2 * 256 * 256 * 256
pytestmark = pytest.mark.pipeline


def _lower(tmp: Path, template: str, stages: tuple[str, ...], on_stage: dict) -> None:
    """Capture a template and run the pipeline up to ``stages``; ``on_stage`` hooks run on live modules."""
    d = tmp / template
    (d / "source").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "templates" / f"{template}.py", d / SRC_REL)
    run_capture(d)
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    with ctx:
        module = ir.Module.parse((d / "capture" / "import.mlir").read_text())
    cfg = PipelineConfig.load()
    cfg.stages = [s for s in cfg.stages if s.stage in stages]

    def hook(stage, _rec, mod):
        if stage in on_stage:
            on_stage[stage](mod)

    run_mlir_pipeline(module, cfg, d, stage_hook=hook)


@pytest.fixture(scope="module")
def costs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cost")
    out, allocs = {}, {}
    for t in TEMPLATES:
        hooks = {"linalg_tensor": lambda mod, t=t: out.__setitem__(t, compute_costs(mod))}
        if t == "attention_block":
            hooks["linalg_buffer"] = lambda mod: (
                allocs.__setitem__("summary", static_alloc_summary(mod)),
                allocs.__setitem__("max_alloc", _largest_alloc(mod)),
            )
            _lower(tmp, t, ("torch", "linalg_tensor", "linalg_buffer"), hooks)
        else:
            _lower(tmp, t, ("torch", "linalg_tensor"), hooks)
    return out, allocs


def _largest_alloc(mod) -> int:
    best = 0
    for op, _d, _p, _pl in walk(mod.operation):
        if op.name != "memref.alloc":
            continue
        mt = ir.MemRefType(op.results[0].type)
        numel = 1
        for dim in mt.shape:
            numel *= int(dim)
        best = max(best, numel * 4)  # the templates are f32
    return best


def test_matmul_relu_matmul_flops(costs):
    per_op, _ = costs
    flops = [c["flops"] for c in per_op["matmul_relu"].values()]
    assert MATMUL_256_FLOPS in flops


def test_every_template_is_costed_without_error(costs):
    per_op, _ = costs
    assert set(per_op) == set(TEMPLATES)
    for t in TEMPLATES:
        assert any(c["flops"] > 0 for c in per_op[t].values()), t


def test_cost_fields_are_consistent(costs):
    per_op, _ = costs
    for t in TEMPLATES:
        for idx, c in per_op[t].items():
            assert set(c) == {"flops", "bytes_read", "bytes_written", "bytes", "intensity"}, (t, idx)
            assert c["bytes"] == pytest.approx(c["bytes_read"] + c["bytes_written"])
            assert c["intensity"] == (c["flops"] / c["bytes"] if c["bytes"] else 0.0)


def test_static_alloc_summary_on_bufferized_module(costs):
    _, allocs = costs
    summary = allocs["summary"]
    assert summary["alloc_count"] >= 1
    assert summary["peak_live_bytes"] >= allocs["max_alloc"] > 0
    assert summary["peak_live_bytes"] <= summary["alloc_bytes"]


def test_line_shares_sum_to_one():
    calib = {"scalar_gflops": 1.0, "bandwidth_gbs": 1.0}
    ops = [([1], {"flops": 3e9, "bytes": 0.0}), ([2], {"flops": 0.0, "bytes": 1e9}),
           ([2, 3], {"flops": 1e9, "bytes": 2e9})]
    shares = est_line_shares(ops, calib)
    assert sum(shares.values()) == pytest.approx(1.0)
    # times 3s, 1s and 2s (the last split over lines 2 and 3)
    assert shares == pytest.approx({1: 0.5, 2: 1 / 3, 3: 1 / 6})


def test_multi_line_op_splits_time_equally():
    calib = {"scalar_gflops": 1.0, "bandwidth_gbs": 1.0}
    shares = est_line_shares([([4, 9], {"flops": 2e9, "bytes": 0.0})], calib)
    assert shares == pytest.approx({4: 0.5, 9: 0.5})


def test_ops_without_lines_are_ignored():
    calib = {"scalar_gflops": 1.0, "bandwidth_gbs": 1.0}
    ops = [([], {"flops": 9e9, "bytes": 0.0}), ([5], {"flops": 1e9, "bytes": 0.0})]
    assert est_line_shares(ops, calib) == pytest.approx({5: 1.0})
    assert est_total_ms(ops, calib) == pytest.approx(10000.0)


def test_empty_or_zero_cost_input_gives_no_shares():
    calib = {"scalar_gflops": 1.0, "bandwidth_gbs": 1.0}
    assert est_line_shares([], calib) == {}
    assert est_line_shares([([1], {"flops": 0.0, "bytes": 0.0})], calib) == {}
    assert est_total_ms([], calib) == 0.0


def test_calibration_cache_and_wrong_key(tmp_path):
    path = tmp_path / "calibration.json"
    first = load_or_run(path)
    assert path.exists()
    second = load_or_run(path)
    assert second == first
    for k in ("bandwidth_gbs", "bandwidth_cache_gbs", "scalar_gflops", "vector_gflops", "seconds"):
        assert first[k] > 0, k
    assert first["vector_gflops"] >= first["scalar_gflops"] * 0.9

    stale = {**first, "cpu": "some other cpu", "bandwidth_gbs": 1.0}
    path.write_text(json.dumps(stale))
    refreshed = load_or_run(path)
    assert refreshed["cpu"] == first["cpu"]
    assert refreshed["bandwidth_gbs"] != 1.0


def test_missing_clang_raises(tmp_path, monkeypatch):
    def missing(_name):
        raise ToolMissing("clang not found")

    monkeypatch.setattr(calibrate, "require_tool", missing)
    with pytest.raises(ToolMissing):
        load_or_run(tmp_path / "calibration.json", force=True)
