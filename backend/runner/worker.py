"""Single worker process: claims queued runs one at a time and supervises ``runner.job``.

One worker means no compile ever overlaps a benchmark, which keeps timings clean.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from backend.schemas import ProgressEvent
from backend.store.runs import DEFAULT_RUNS_DIR, ROOT, RunStore

JOB_TIMEOUT_S = int(os.environ.get("OPTILER_JOB_TIMEOUT", "1800"))


def supervise(store: RunStore, run_id: str) -> None:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
           "OMP_NUM_THREADS": "1"}
    msg = None
    try:
        r = subprocess.run(
            [sys.executable, "-P", "-m", "backend.runner.job", "--runs-dir", str(store.root), run_id],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=JOB_TIMEOUT_S)
        if r.returncode != 0:
            tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
            msg = f"compiler process crashed (exit code {r.returncode})" + (f":\n{tail}" if tail else "")
    except subprocess.TimeoutExpired:
        msg = f"compile job timed out after {JOB_TIMEOUT_S}s"
    run = store.get(run_id)
    if run.status in ("queued", "running"):  # the job died before reporting
        run.status, run.phase, run.error = "error", "error", msg or "job exited without a result"
        store.save(run)
        store.append_event(run_id, ProgressEvent(type="error", phase="error", message=run.error))
    store.finish_job(run_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--once", action="store_true", help="drain the queue and exit")
    a = ap.parse_args()
    store = RunStore(Path(a.runs_dir))
    store.requeue_stale()
    while True:
        rid = store.claim_next()
        if rid is None:
            if a.once:
                return
            time.sleep(0.3)
            continue
        supervise(store, rid)


if __name__ == "__main__":
    main()
