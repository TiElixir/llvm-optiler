"""Pure-python tests for backend/pipeline/config.py (no compilation)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from backend.pipeline.config import (
    DEFAULT_PIPELINE,
    MLIR_STAGES,
    PassSpec,
    PipelineConfig,
    expand,
    pass_name,
    split_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- split_pipeline

def test_split_plain_passes():
    assert split_pipeline("canonicalize,cse") == ["canonicalize", "cse"]


def test_split_nested_func_func_distributes_anchor():
    assert split_pipeline("func.func(a,b)") == ["func.func(a)", "func.func(b)"]


def test_split_mixed_top_level_and_nested():
    assert split_pipeline("a,func.func(b,c),d") == ["a", "func.func(b)", "func.func(c)", "d"]


def test_split_builtin_module_is_transparent():
    assert split_pipeline("builtin.module(a,func.func(b,c))") == ["a", "func.func(b)", "func.func(c)"]


def test_split_deeply_nested_anchors_wrap_in_order():
    assert split_pipeline("builtin.module(func.func(affine.for(x)))") == ["func.func(affine.for(x))"]


def test_split_options_with_spaces_stay_attached():
    text = "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},cse"
    assert split_pipeline(text) == [
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "cse",
    ]


def test_split_commas_inside_option_braces_do_not_split():
    assert split_pipeline("a{x=1,2 y={3,4}},b") == ["a{x=1,2 y={3,4}}", "b"]


def test_split_nested_with_options():
    text = "func.func(canonicalize{max-iterations=3 test-convergence=true},cse)"
    assert split_pipeline(text) == [
        "func.func(canonicalize{max-iterations=3 test-convergence=true})",
        "func.func(cse)",
    ]


def test_split_unbalanced_raises():
    with pytest.raises(ValueError):
        split_pipeline("func.func(a")


# ---------------------------------------------------------------- pass_name

@pytest.mark.parametrize("leaf,expected", [
    ("cse", "cse"),
    ("canonicalize{max-iterations=3}", "canonicalize"),
    ("func.func(canonicalize)", "canonicalize"),
    ("func.func(canonicalize{region-simplify=normal})", "canonicalize"),
    ("builtin.module(func.func(affine-loop-fusion))", "affine-loop-fusion"),
    ("one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "one-shot-bufferize"),
])
def test_pass_name(leaf, expected):
    assert pass_name(leaf) == expected


# ---------------------------------------------------------------- expand

def test_expand_plain_pass_keeps_user_spelling():
    seen = []

    def fake(spec):
        seen.append(spec)
        return "canonicalize{ max-iterations=10 region-simplify=normal }"

    leaves, composite = expand("canonicalize", fake)
    assert (leaves, composite) == (["canonicalize"], False)
    assert seen == ["canonicalize"]


def test_expand_plain_pass_with_user_options_is_not_composite():
    leaves, composite = expand(
        "func.func(canonicalize{max-iterations=2})",
        lambda s: "func.func(canonicalize{max-iterations=2 region-simplify=normal})",
    )
    assert composite is False
    assert leaves == ["func.func(canonicalize{max-iterations=2})"]


def test_expand_composite_returns_printed_leaves():
    printed = "func.func(convert-a,convert-b),convert-c,func.func(cse)"
    leaves, composite = expand("some-backend-pipeline", lambda s: printed)
    assert composite is True
    assert leaves == ["func.func(convert-a)", "func.func(convert-b)", "convert-c", "func.func(cse)"]


def test_expand_pipeline_that_prints_as_one_different_pass_is_composite():
    leaves, composite = expand("alias-pipeline", lambda s: "real-pass")
    assert composite is True
    assert leaves == ["real-pass"]


# ---------------------------------------------------------------- PipelineConfig

YAML = """
- stage: torch
  passes:
    - "func.func(torch-match-quantized-custom-ops)"
    - pass: "torch-simplify-shape-calculations"
- stage: linalg_tensor
  passes:
    - pass: canonicalize
      options:
        max-iterations: 3
        region-simplify: normal
    - pass: "one-shot-bufferize{bufferize-function-boundaries}"
      options:
        function-boundary-type-conversion: identity-layout-map
    - cse
"""


def test_from_yaml_accepts_strings_and_dict_entries():
    cfg = PipelineConfig.from_yaml(YAML)
    assert [s.stage for s in cfg.stages] == ["torch", "linalg_tensor"]
    assert [p.pass_ for p in cfg.stages[0].passes] == [
        "func.func(torch-match-quantized-custom-ops)", "torch-simplify-shape-calculations"]
    assert cfg.stages[1].passes[0].options == {"max-iterations": 3, "region-simplify": "normal"}


def test_from_yaml_accepts_mapping_with_stages_key():
    cfg = PipelineConfig.from_yaml("stages:\n  - stage: loops\n    passes: [lower-affine]\n")
    assert cfg.all_pass_texts() == [("loops", "lower-affine")]


def test_pass_text_with_options():
    cfg = PipelineConfig.from_yaml(YAML)
    texts = [t for _, t in cfg.all_pass_texts()]
    assert texts == [
        "func.func(torch-match-quantized-custom-ops)",
        "torch-simplify-shape-calculations",
        "canonicalize{max-iterations=3 region-simplify=normal}",
        # options merge into an existing brace group
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "cse",
    ]


def test_pass_spec_text_without_options_is_the_pass():
    assert PassSpec(**{"pass": "cse"}).text() == "cse"
    assert PassSpec(**{"pass": "cse"}, options={}).text() == "cse"


def test_yaml_round_trip():
    cfg = PipelineConfig.from_yaml(YAML)
    again = PipelineConfig.from_yaml(cfg.to_yaml())
    assert again.all_pass_texts() == cfg.all_pass_texts()
    assert [s.stage for s in again.stages] == [s.stage for s in cfg.stages]


def test_round_trip_is_stable_after_one_pass():
    first = PipelineConfig.from_yaml(YAML).to_yaml()
    assert PipelineConfig.from_yaml(first).to_yaml() == first


def test_unknown_stage_raises_value_error():
    with pytest.raises(ValueError, match="unknown stage"):
        PipelineConfig.from_yaml("- stage: bogus\n  passes: [cse]\n")


def test_source_and_asm_are_not_mlir_stages():
    for stage in ("source", "llvm_ir", "asm"):
        assert stage not in MLIR_STAGES
        with pytest.raises(ValueError):
            PipelineConfig.from_yaml(f"- stage: {stage}\n  passes: [cse]\n")


def test_default_yaml_loads():
    cfg = PipelineConfig.load()
    assert DEFAULT_PIPELINE.is_file()
    assert [s.stage for s in cfg.stages] == MLIR_STAGES
    assert all(s.passes for s in cfg.stages)


def test_load_accepts_str_path(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("- stage: loops\n  passes:\n    - lower-affine\n")
    assert PipelineConfig.load(str(p)).all_pass_texts() == [("loops", "lower-affine")]


def test_default_yaml_round_trips():
    cfg = PipelineConfig.load()
    assert PipelineConfig.from_yaml(cfg.to_yaml()).all_pass_texts() == cfg.all_pass_texts()


# ---------------------------------------------------------------- Rule 1 gate

def test_pass_names_exist():
    """Every pass in default.yaml must parse against the pinned torch-mlir toolchain."""
    spec = importlib.util.spec_from_file_location("check_passes", ROOT / "scripts" / "check_passes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check_text(DEFAULT_PIPELINE.read_text()) == []
