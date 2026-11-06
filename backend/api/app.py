"""FastAPI app: REST + SSE over the run store (PLAN.md section 11).

``uvicorn backend.api.app:app`` serves it; the lifespan owns the single compile worker.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import difflib
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from backend.pipeline.config import DEFAULT_PIPELINE, STAGES, PipelineConfig
from backend.schemas import (
    CreateRunRequest, CreateRunResponse, DiffResult, ProgressEvent, Run, RunSettings, RunSummary,
    SnapshotText, StageOps, TemplateInfo, TemplateSource,
)
from backend.provenance.build import read_stage
from backend.store.runs import DEFAULT_RUNS_DIR, ROOT, RunNotFound, RunStore
from backend.toolchain import toolchain_report

FRONTEND_DIR = ROOT / "frontend"

TEMPLATES_DIR = ROOT / "backend" / "templates"
EVENT_POLL_S = 0.2
FINISHED_GRACE_S = 1.0     # how long a finished run may still owe its terminal event
WORKER_CHECK_S = 1.0


# ------------------------------------------------------------------ templates

def _template_names() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.py") if not p.stem.startswith("_"))


def _template_path(name: str) -> Path:
    if name not in _template_names():  # also rules out any path component
        raise HTTPException(404, f"unknown template {name!r}")
    return TEMPLATES_DIR / f"{name}.py"


def _template_description(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text()))
    except (OSError, SyntaxError):
        return ""
    return doc.strip().splitlines()[0].strip() if doc and doc.strip() else ""


# ------------------------------------------------------------------ worker process

class WorkerProcess:
    """One ``backend.runner.worker`` subprocess, restarted if it dies."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        # own session so stopping the worker also stops its compile job and benchmark children
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "backend.runner.worker", "--runs-dir", str(self.runs_dir)],
            env=env, cwd=ROOT, start_new_session=True)

    def _killpg(self, sig: int) -> None:
        if self.proc is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, sig)

    def stop(self) -> None:
        if self.proc is None:
            return
        self._killpg(signal.SIGTERM)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._killpg(signal.SIGKILL)  # stragglers in the group (job, benchmark)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5)
        self.proc = None

    async def watch(self) -> None:
        backoff = 0.0
        while True:
            await asyncio.sleep(WORKER_CHECK_S + backoff)
            if self.proc is not None and self.proc.poll() is not None:
                sys.stderr.write(f"[api] worker exited with {self.proc.returncode}; restarting\n")
                self._killpg(signal.SIGKILL)
                backoff = min(backoff * 2 + 1.0, 15.0)
                self.start()
            else:
                backoff = 0.0


# ------------------------------------------------------------------ app

def create_app(runs_dir: Path | str | None = None, start_worker: bool = True) -> FastAPI:
    root = Path(runs_dir) if runs_dir is not None else DEFAULT_RUNS_DIR
    holder: dict[str, RunStore] = {}

    def get_store() -> RunStore:  # lazy: importing the module must not create ./runs
        if "store" not in holder:
            holder["store"] = RunStore(root)
        return holder["store"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker, watcher = None, None
        if start_worker:
            get_store()  # make sure the index exists before the worker opens it
            worker = WorkerProcess(root)
            worker.start()
            watcher = asyncio.create_task(worker.watch())
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            if worker is not None:
                await asyncio.to_thread(worker.stop)

    app = FastAPI(title="Optiler", lifespan=lifespan)
    app.state.get_store = get_store

    @app.exception_handler(RunNotFound)
    async def _run_not_found(_req: Request, exc: RunNotFound) -> JSONResponse:
        return JSONResponse({"detail": f"run not found: {exc.args[0] if exc.args else ''}"}, status_code=404)

    api = APIRouter(prefix="/api")

    # -------------------------------------------------------------- runs
    @api.post("/runs", response_model=CreateRunResponse)
    def create_run(req: CreateRunRequest) -> CreateRunResponse:
        store = get_store()
        parent = store.get(req.parent_id) if req.parent_id else None  # RunNotFound -> 404
        template_source = None
        if req.template:
            template_source = _template_path(req.template).read_text()  # 404 when unknown

        if req.source and req.source.strip():
            source, from_parent = req.source, False
        elif template_source is not None:
            source, from_parent = template_source, False
        elif parent is not None and parent.source_code.strip():
            source, from_parent = parent.source_code, True
        else:
            raise HTTPException(422, "no model source: give `source`, a `template` or a `parent_id`")

        pipeline_yaml = req.pipeline_yaml or (parent.pipeline_yaml if parent else "") \
            or DEFAULT_PIPELINE.read_text()
        try:
            PipelineConfig.from_yaml(pipeline_yaml)
        except Exception as e:  # noqa: BLE001 - yaml, key, type and pydantic errors all mean "bad pipeline"
            raise HTTPException(422, f"invalid pipeline: {type(e).__name__}: {e}") from e

        settings = RunSettings()
        if req.iters is not None:
            settings.iters = req.iters
        if req.warmup is not None:
            settings.warmup = req.warmup
        if req.threads is not None:
            settings.threads = req.threads
        if settings.iters < 1 or settings.threads < 1 or settings.warmup < 0:
            raise HTTPException(422, "iters and threads must be at least 1 and warmup at least 0")

        run = Run(
            id="", created_at="",  # assigned by store.create
            parent_id=parent.id if parent else None, fix_id=req.fix_id,
            label=req.label or req.template or "",
            source_code=source,
            template=req.template or (parent.template if parent and from_parent else None),
            pipeline_yaml=pipeline_yaml, settings=settings)
        return CreateRunResponse(run_id=store.create(run).id)

    @api.get("/runs", response_model=list[RunSummary])
    def list_runs() -> list[RunSummary]:
        return get_store().list()

    @api.get("/runs/{run_id}", response_model=Run)
    def get_run(run_id: str) -> Run:
        return get_store().get(run_id)

    @api.get("/runs/{run_id}/events")
    async def run_events(run_id: str) -> EventSourceResponse:
        store = get_store()
        store.get(run_id)  # 404 before the stream starts

        async def stream():
            offset, finished_at = 0, None
            loop = asyncio.get_running_loop()
            while True:
                events, offset = store.read_events(run_id, offset)
                for ev in events:
                    yield {"data": ev.model_dump_json()}
                    if ev.type in ("done", "error"):
                        return
                if events:
                    finished_at = None
                    continue
                run = store.get(run_id)
                if run.status in ("done", "error"):
                    # the terminal event is appended just after the status is saved: allow a short grace
                    finished_at = finished_at if finished_at is not None else loop.time()
                    if loop.time() - finished_at >= FINISHED_GRACE_S:
                        final = (ProgressEvent(type="done", phase="done") if run.status == "done"
                                 else ProgressEvent(type="error", phase="error", message=run.error))
                        yield {"data": final.model_dump_json()}
                        return
                await asyncio.sleep(EVENT_POLL_S)

        return EventSourceResponse(stream())

    def _snapshot_text(run: Run, store: RunStore, idx: int) -> tuple[SnapshotText, str]:
        snap = next((s for s in run.snapshots if s.idx == idx), None)
        if snap is None:
            raise HTTPException(404, f"run {run.id} has no snapshot {idx}")
        base = store.run_dir(run.id).resolve()
        path = (base / snap.path).resolve()
        if not path.is_relative_to(base):
            raise HTTPException(404, "snapshot path escapes the run directory")
        try:
            text = path.read_text()
        except OSError:
            raise HTTPException(404, f"snapshot file missing: {snap.path}") from None
        return SnapshotText(idx=snap.idx, stage=snap.stage, pass_name=snap.pass_name, text=text), snap.path

    @api.get("/runs/{run_id}/snapshots/{idx}", response_class=PlainTextResponse)
    def get_snapshot(run_id: str, idx: int) -> PlainTextResponse:
        store = get_store()
        return PlainTextResponse(_snapshot_text(store.get(run_id), store, idx)[0].text)

    @api.get("/runs/{run_id}/diff", response_model=DiffResult)
    def get_diff(run_id: str, a: int, b: int, b_run: str | None = None) -> DiffResult:
        store = get_store()
        run_a = store.get(run_id)
        run_b = store.get(b_run) if b_run else run_a
        sa, _ = _snapshot_text(run_a, store, a)
        sb, _ = _snapshot_text(run_b, store, b)

        def lines(t: str) -> list[str]:
            return (t if t.endswith("\n") or not t else t + "\n").splitlines(keepends=True)

        unified = "".join(difflib.unified_diff(
            lines(sa.text), lines(sb.text),
            fromfile=f"{run_a.id}/{sa.idx:03d}_{sa.pass_name}", tofile=f"{run_b.id}/{sb.idx:03d}_{sb.pass_name}"))
        return DiffResult(a=sa, b=sb, unified=unified)

    @api.get("/runs/{run_id}/stages/{stage}", response_model=StageOps)
    def get_stage(run_id: str, stage: str) -> StageOps:
        if stage not in STAGES:
            raise HTTPException(404, f"unknown stage {stage!r}")
        store = get_store()
        run = store.get(run_id)
        so = read_stage(store.run_dir(run_id), stage)  # written at the end of the compile (provenance/)
        if so is not None:
            return so
        idxs = [s.idx for s in run.snapshots if s.stage == stage]
        return StageOps(stage=stage, snapshot_idx=max(idxs) if idxs else None,
                        ops=[], links=[], exact_lineage_pct=None)

    @api.post("/runs/{run_id}/findings/{fid}/apply")
    def apply_finding(run_id: str, fid: str):
        # M6: create a child run from the finding's fix.
        raise HTTPException(501, "not implemented yet")

    @api.get("/runs/{run_id}/reproducer")
    def reproducer(run_id: str):
        # M7: zip of source, pipeline, snapshots and reports.
        raise HTTPException(501, "not implemented yet")

    # -------------------------------------------------------------- templates, toolchain
    @api.get("/templates", response_model=list[TemplateInfo])
    def list_templates() -> list[TemplateInfo]:
        return [TemplateInfo(name=n, description=_template_description(TEMPLATES_DIR / f"{n}.py"))
                for n in _template_names()]

    @api.get("/templates/{name}", response_model=TemplateSource)
    def get_template(name: str) -> TemplateSource:
        return TemplateSource(name=name, source=_template_path(name).read_text())

    @api.get("/toolchain")
    def get_toolchain() -> dict:
        return toolchain_report()

    app.include_router(api)
    if FRONTEND_DIR.is_dir():  # one process serves API and UI; mounted last so /api/* wins
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
