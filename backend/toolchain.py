"""Tool discovery, versions and capability checks.

Search order: ``$OPTILER_LLVM_BIN``, the project-local ``toolchain/llvm/bin`` (a pinned
LLVM release tarball, see docs/toolchain.md), then ``PATH``.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_LLVM_BIN = ROOT / "toolchain" / "llvm" / "bin"
LLVM_TOOLS = ["clang", "opt", "llc", "llvm-objdump", "llvm-symbolizer", "llvm-mca"]


class ToolMissing(RuntimeError):
    pass


def find_tool(name: str) -> str | None:
    for d in (os.environ.get("OPTILER_LLVM_BIN"), str(LOCAL_LLVM_BIN)):
        if d and (Path(d) / name).exists():
            return str(Path(d) / name)
    return shutil.which(name)


def require_tool(name: str) -> str:
    p = find_tool(name)
    if not p:
        raise ToolMissing(
            f"`{name}` not found. Install LLVM (e.g. `sudo pacman -Syu llvm`) "
            "or point OPTILER_LLVM_BIN at an LLVM bin directory."
        )
    return p


def _run(cmd: list[str], timeout: float = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def torch_mlir_opt_path() -> str | None:
    try:
        import torch_mlir
        p = Path(list(torch_mlir.__path__)[0]) / "_mlir_libs" / "torch-mlir-opt"
        return str(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=8)
def llvm_version(tool: str = "opt") -> str | None:
    p = find_tool(tool)
    if not p:
        return None
    m = re.search(r"(?:LLVM|clang) version ([0-9][^\s]*)", _run([p, "--version"]))
    return m.group(1) if m else None


@functools.lru_cache(maxsize=1)
def bundled_mlir_llvm_version() -> str | None:
    p = torch_mlir_opt_path()
    if not p:
        return None
    m = re.search(r"LLVM version ([0-9][^\s]*)", _run([p, "--version"]))
    return m.group(1) if m else None


@functools.lru_cache(maxsize=1)
def target_triple() -> str:
    clang = find_tool("clang")
    return _run([clang, "-print-target-triple"]) if clang else "x86_64-pc-linux-gnu"


@functools.lru_cache(maxsize=1)
def host_cpu() -> str | None:
    llc = find_tool("llc")
    if not llc:
        return None
    m = re.search(r"Host CPU:\s*(\S+)", _run([llc, "--version"]))
    return m.group(1) if m else None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def vector_isa() -> dict:
    f = cpu_flags()
    if "avx512f" in f:
        return {"name": "avx512", "width_bits": 512, "f32_lanes": 16}
    if "avx2" in f:
        return {"name": "avx2", "width_bits": 256, "f32_lanes": 8}
    if "sse4_2" in f or "sse2" in f:
        return {"name": "sse", "width_bits": 128, "f32_lanes": 4}
    return {"name": "scalar", "width_bits": 32, "f32_lanes": 1}


def perf_status() -> dict:
    p = find_tool("perf")
    if not p:
        return {"available": False, "reason": "perf not installed"}
    try:
        para = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
    except Exception:  # noqa: BLE001
        para = None
    out = _run([p, "stat", "-e", "cycles:u", "--", "true"], timeout=20)
    ok = "cycles" in out and "not supported" not in out and "Permission" not in out and "denied" not in out
    return {"available": bool(ok), "paranoid": para, "path": p,
            "reason": None if ok else out.splitlines()[-1][:160] if out else "perf stat failed"}


def toolchain_report() -> dict:
    def ver(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    tools = {t: find_tool(t) for t in LLVM_TOOLS + ["taskset", "perf"]}
    sys_v = llvm_version("opt")
    bundled = bundled_mlir_llvm_version()
    warnings = []
    if sys_v and bundled and sys_v.split(".")[0] != bundled.split(".")[0]:
        warnings.append(
            f"MLIR in torch-mlir is LLVM {bundled} but system opt/llc is LLVM {sys_v}; "
            "textual LLVM IR is handed between them, see docs/toolchain.md")
    missing = [t for t in ("clang", "opt", "llc", "llvm-objdump") if not tools[t]]
    if missing:
        warnings.append(f"missing LLVM tools: {', '.join(missing)}")
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": ver("torch"), "torch_mlir": ver("torch-mlir"),
        "bundled_mlir_llvm": bundled, "system_llvm": sys_v,
        "clang": llvm_version("clang"), "triple": target_triple(),
        "cpu_model": cpu_model(), "host_cpu": host_cpu(), "vector_isa": vector_isa(),
        "cores": os.cpu_count(), "tools": tools, "perf": perf_status(),
        "warnings": warnings,
    }
