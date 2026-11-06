"""Capture step: run the user's model code in a sandboxed subprocess.

Parent side: :func:`run_capture`. Child side: ``python -m backend.pipeline.capture <run_dir>``.

The child imports ``<run_dir>/source/model.py`` (relative path, so FX ``stack_trace``
frames and MLIR locations read ``source/model.py:LINE``), runs the model in eager
mode, exports it through ``torch.export`` + decompositions, and imports it into the
Torch dialect. Everything the rest of the pipeline needs is written to disk, so the
compile process never executes user code.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path

SRC_REL = "source/model.py"
SEED = 0


class CaptureError(RuntimeError):
    pass


def _limits(timeout_s: int, mem_gb: int):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_gb << 30, mem_gb << 30))
    return fn


def run_capture(run_dir: Path, timeout_s: int = 120, mem_gb: int = 12) -> dict:
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    try:
        r = subprocess.run(
            [sys.executable, "-P", "-m", "backend.pipeline.capture", str(run_dir.resolve())],
            cwd=run_dir, env=env, capture_output=True, text=True,
            timeout=timeout_s + 10, preexec_fn=_limits(timeout_s, mem_gb),
        )
    except subprocess.TimeoutExpired as e:
        raise CaptureError(f"model capture timed out after {timeout_s}s") from e
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-25:])
        raise CaptureError(tail or f"capture exited with code {r.returncode}")
    return json.loads((run_dir / "capture" / "meta.json").read_text())


# ----------------------------------------------------------------- child

def _child(run_dir: Path) -> None:
    import runpy

    import numpy as np
    import torch
    from torch_mlir import fx
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table

    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir / "source"))
    cap = run_dir / "capture"
    cap.mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    ns = runpy.run_path(SRC_REL)
    if "model" not in ns or "inputs" not in ns:
        raise SystemExit("source must define `model` and `inputs`")
    model, inputs = ns["model"], ns["inputs"]
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    inputs = tuple(inputs)
    for i, t in enumerate(inputs):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
            raise SystemExit(f"input {i} must be a float32 tensor (got {getattr(t, 'dtype', type(t))})")
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        eager = model(*inputs)
    if not isinstance(eager, torch.Tensor):
        raise SystemExit("model must return a single tensor (tuple outputs are not supported yet)")
    np.savez(cap / "inputs.npz", *[t.detach().numpy() for t in inputs])
    np.savez(cap / "eager_out.npz", eager.detach().numpy())

    prog = torch.export.export(model, inputs)
    prog = prog.run_decompositions(get_decomposition_table())
    (cap / "fx_graph.txt").write_text(str(prog.graph_module.graph))

    src_lines = Path(SRC_REL).read_text().splitlines()
    nodes = []
    for n in prog.graph_module.graph.nodes:
        line = None
        st = getattr(n, "stack_trace", None)
        if st:
            for ln in reversed(st.strip().splitlines()):
                ln = ln.strip()
                if ln.startswith("File ") and "source/model.py" in ln.replace("\\", "/"):
                    try:
                        line = int(ln.split("line ")[1].split(",")[0])
                    except (IndexError, ValueError):
                        pass
                    break
        nodes.append({
            "name": n.name, "op": n.op, "target": str(n.target), "line": line,
            "code": src_lines[line - 1].strip() if line and line <= len(src_lines) else None,
        })
    (cap / "fx_nodes.json").write_text(json.dumps(nodes, indent=1))

    module = fx.export_and_import(prog, output_type="raw", func_name="forward", decomposition_table={})
    (cap / "import.mlir").write_text(module.operation.get_asm(enable_debug_info=True))
    meta = {
        "inputs": [{"shape": list(t.shape), "dtype": "float32"} for t in inputs],
        "output": {"shape": list(eager.shape), "dtype": "float32"},
        "torch": torch.__version__, "seed": SEED,
    }
    (cap / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    _child(Path(sys.argv[1]).resolve())
