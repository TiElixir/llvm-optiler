#!/usr/bin/env python
"""M0b: do pipeline patches make a template measurably faster? (de-risks the M6 acceptance check)

    .venv/bin/python scripts/viability.py [template ...] [--iters N] [--variants a,b]

Each variant is a hand-written pipeline patch compiled and benchmarked against the
default pipeline. Results are measurements, not claims: they decide template sizes and
which fixes the detectors may offer.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline.config import PassSpec, PipelineConfig  # noqa: E402
from backend.pipeline.run import compile_run  # noqa: E402

INTERCHANGE_KJ = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %g = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    %i = transform.structured.interchange %g iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""


def drop(cfg: PipelineConfig, *names: str) -> None:
    for s in cfg.stages:
        s.passes = [p for p in s.passes if not any(n in p.pass_ for n in names)]


def add_after(cfg: PipelineConfig, stage: str, passes: list[str]) -> None:
    for s in cfg.stages:
        if s.stage == stage:
            s.passes += [PassSpec(**{"pass": p}) for p in passes]


def with_transform(cfg: PipelineConfig, stage: str) -> None:
    add_after(cfg, stage, [
        "transform-preload-library{transform-library-paths=${RUN_DIR}/transforms/t.mlir}",
        "transform-interpreter"])


def v_baseline(cfg, d):
    pass


def v_nofusion(cfg, d):
    drop(cfg, "affine-loop-fusion")


def v_interchange_nofusion(cfg, d):
    (d / "transforms").mkdir(exist_ok=True)
    (d / "transforms" / "t.mlir").write_text(INTERCHANGE_KJ)
    with_transform(cfg, "linalg_tensor")
    drop(cfg, "affine-loop-fusion")


def v_no_elementwise_fusion(cfg, d):
    drop(cfg, "linalg-fuse-elementwise-ops")


VARIANTS = {
    "baseline": v_baseline,
    "nofusion": v_nofusion,
    "interchange_nofusion": v_interchange_nofusion,
    "no_elementwise_fusion": v_no_elementwise_fusion,
}


def run_variant(template: str, name: str, iters: int, root: Path) -> dict:
    d = root / template / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "source").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "templates" / f"{template}.py", d / "source" / "model.py")
    cfg = PipelineConfig.load()
    VARIANTS[name](cfg, d)
    t0 = time.perf_counter()
    try:
        out = compile_run(d, cfg, iters=iters)
    except Exception as e:  # noqa: BLE001
        return {"template": template, "variant": name, "error": str(e).strip().splitlines()[-1][:160]}
    b = out["bench"]
    return {"template": template, "variant": name, "correct": b["correct"],
            "compiled": b.get("compiled_ms", {}), "eager": b.get("eager_ms", {}),
            "secs": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("templates", nargs="*", default=["matmul_relu"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--root", default=str(ROOT / "runs" / "viability"))
    a = ap.parse_args()
    print(f"{'template':16} {'variant':24} {'ok':>3} {'median ms':>10} {'p10':>8} {'p90':>8} {'vs base':>8} {'eager ms':>9}")
    for t in a.templates:
        base = None
        for v in a.variants.split(","):
            r = run_variant(t, v, a.iters, Path(a.root))
            if "error" in r:
                print(f"{t:16} {v:24} ERR {r['error']}")
                continue
            c = r["compiled"]
            if v == "baseline":
                base = c.get("median")
            sp = f"{base / c['median']:.2f}x" if base and c else "-"
            print(f"{t:16} {v:24} {'Y' if r['correct'] else 'N':>3} {c.get('median', float('nan')):10.3f} "
                  f"{c.get('p10', float('nan')):8.3f} {c.get('p90', float('nan')):8.3f} {sp:>8} "
                  f"{r['eager'].get('median', float('nan')):9.3f}")


if __name__ == "__main__":
    main()
