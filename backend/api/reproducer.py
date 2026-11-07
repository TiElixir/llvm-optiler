"""Reproducer zip (PLAN.md 3.2): everything needed to replay a run without the app."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from backend.pipeline.config import MLIR_STAGES
from backend.pipeline.llvm_steps import RT_SRC
from backend.schemas import Run

MLIR_PASSES_PY = '''"""Replays the MLIR half of the run: import.mlir -> each pass in passes.json -> LLVM IR (model.ll).

Needs the torch-mlir wheel from toolchain.json. The LLVM half is repro.sh.
"""
import json
import os

from torch_mlir import ir
from torch_mlir.dialects import llvm, torch as torch_d
from torch_mlir.passmanager import PassManager

os.makedirs("anchor", exist_ok=True)
tc = json.load(open("toolchain.json"))
ctx = ir.Context()
torch_d.register_dialect(ctx)
with ctx:
    module = ir.Module.parse(open("import.mlir").read())
    for p in json.load(open("passes.json")):
        PassManager.parse(f"builtin.module({p})").run(module.operation)
        print("ran", p[:90])
    module.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(tc.get("triple") or "x86_64-unknown-linux-gnu")
    open("model.ll", "w").write(llvm.translate_module_to_llvmir(module.operation))
print("wrote model.ll")
'''


def _rel(text: str, run_dir: Path) -> str:
    return text.replace(str(run_dir.resolve()) + "/", "./").replace(str(run_dir.resolve()), ".")


def build_reproducer(run: Run, run_dir: Path) -> bytes:
    run_dir = Path(run_dir)
    mlir_stage = set(MLIR_STAGES)
    passes = [_rel(s.pass_text, run_dir) for s in run.snapshots
              if s.kind != "import" and s.stage in mlir_stage]
    commands = json.loads((run_dir / "reports" / "build_commands.json").read_text()) \
        if (run_dir / "reports" / "build_commands.json").exists() else []

    lines = ["#!/usr/bin/env bash",
             f"# Reproduces run {run.id} ({run.label or 'unlabeled'}). Needs LLVM tools of the versions in toolchain.json.",
             "set -euo pipefail", "cd \"$(dirname \"$0\")\"",
             "python3 mlir_passes.py              # import.mlir -> passes.json -> model.ll",
             "mkdir -p llvm/full bin asm reports && cp model.ll llvm/full/input.ll"]
    for cmd in commands:
        c = [(Path(a).name if "/" in a and Path(a).name in ("opt", "llc", "clang", "clang-23", "llvm-objdump") else a) for a in cmd]
        text = " ".join(f"'{x}'" if (" " in x or "<" in x or ">" in x) else x for x in map(lambda a: _rel(a, run_dir), c))
        text = text.replace(str(RT_SRC), "optiler_rt.c")
        if cmd and "objdump" in cmd[0]:
            text += " > asm/model.s"
        lines.append(text)

    files: dict[str, str | bytes] = {
        "model.py": run.source_code,
        "pipeline.yaml": run.pipeline_yaml,
        "passes.json": json.dumps(passes, indent=1),
        "mlir_passes.py": MLIR_PASSES_PY,
        "repro.sh": "\n".join(lines) + "\n",
        "toolchain.json": json.dumps(run.toolchain, indent=1, default=str),
        "optiler_rt.c": RT_SRC.read_text(),
        "README.txt": (f"Run {run.id}\nmodel.py          the model (defines `model` and `inputs`)\nimport.mlir       torch dialect module imported "
                       "from it\npipeline.yaml     the pipeline config\npasses.json       every pass actually run, one at a time\n"
                       "mlir_passes.py    replays the MLIR passes (torch-mlir bindings) and writes model.ll\nrepro.sh          the LLVM "
                       "commands (opt, llc, link, objdump) exactly as run\ntoolchain.json    versions\n"),
    }
    if (run_dir / "capture" / "import.mlir").exists():
        files["import.mlir"] = (run_dir / "capture" / "import.mlir").read_text()
    for t in sorted((run_dir / "transforms").glob("*.mlir")) if (run_dir / "transforms").exists() else []:
        files[f"transforms/{t.name}"] = t.read_text()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if name.endswith((".sh", ".py")) else 0o644) << 16
            z.writestr(info, data)
    return buf.getvalue()
