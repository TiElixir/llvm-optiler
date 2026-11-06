"""Call the compiled model in a loop for a fixed time; run under ``perf record`` for attribution."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .harness import CompiledModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--so", default="bin/model.so")
    a = ap.parse_args()
    d = Path(a.run_dir)
    meta = json.loads((d / "capture" / "meta.json").read_text())
    z = np.load(d / "capture" / "inputs.npz")
    arrays = [z[f"arr_{i}"] for i in range(len(meta["inputs"]))]
    cm = CompiledModel(d / a.so, len(arrays), tuple(meta["output"]["shape"]))
    keep, _desc, ptrs = cm.prepare(arrays)
    for _ in range(5):
        cm.release(cm.call_raw(ptrs))
    end, n = time.perf_counter() + a.seconds, 0
    while time.perf_counter() < end:
        cm.release(cm.call_raw(ptrs))
        n += 1
    print(json.dumps({"calls": n}))


if __name__ == "__main__":
    main()
