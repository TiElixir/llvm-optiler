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
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from backend.pipeline.config import DEFAULT_PIPELINE, STAGES, PipelineConfig
from backend.pipeline.validate import check_text
from backend.schemas import (
    ApplyFixRequest, CompareResult, CompareStage, CreateRunRequest, CreateRunResponse, DiffResult, ProgressEvent, Run,
    RunSettings, RunSummary, SnapshotText, StageOps, TemplateInfo, TemplateSource,
)
from backend.api.reproducer import build_reproducer
from backend.fixes.apply import apply_patch
from backend.fixes.registry import FIXES
from backend.provenance.build import read_stage
from backend.store.runs import DEFAULT_RUNS_DIR, ROOT, RunNotFound, RunStore
from backend.toolchain import toolchain_report

FRONTEND_DIR = ROOT / "frontend"


class RevalidatingStatic(StaticFiles):
    """The UI is plain files that change between versions: make browsers revalidate instead of guessing."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

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
        if req.pipeline_yaml:  # Rule 1: unknown passes are rejected before anything is queued
            bad = check_text(pipeline_yaml)
            if bad:
                raise HTTPException(422, "unknown pass: " + "; ".join(f"[{s}] {p}: {err}" for s, p, err in bad[:3]))

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
            pipeline_yaml=pipeline_yaml, settings=settings, transform_script=req.transform_script or None)
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

    @api.post("/runs/{run_id}/findings/{fid}/apply", response_model=CreateRunResponse)
    def apply_finding(run_id: str, fid: str, body: ApplyFixRequest | None = None) -> CreateRunResponse:
        """Apply a finding's fix: a new child run with the patched pipeline (the parent is never changed)."""
        store = get_store()
        parent = store.get(run_id)
        finding = next((f for f in parent.findings if f.id == fid), None)
        if finding is None:
            raise HTTPException(404, f"run {run_id} has no finding {fid!r}")
        fix_id = (body.fix_id if body and body.fix_id else None) or (finding.fix_ids[0] if finding.fix_ids else None)
        if fix_id is None or fix_id not in finding.fix_ids:
            raise HTTPException(422, "this finding has no automatic fix")
        fix = FIXES[fix_id]
        try:
            patched = apply_patch(PipelineConfig.from_yaml(parent.pipeline_yaml), fix)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        child = store.create(Run(
            id="", created_at="", source_code=parent.source_code, template=parent.template,
            pipeline_yaml=patched.to_yaml(), parent_id=parent.id, fix_id=fix.id, label=fix.label,
            settings=parent.settings))
        return CreateRunResponse(run_id=child.id)

    @api.get("/runs/{run_id}/compare", response_model=CompareResult)
    def compare(run_id: str, other: str) -> CompareResult:
        """Run vs run: metrics, op counts per stage and the pipeline config diff."""
        store = get_store()
        a, b = store.get(run_id), store.get(other)

        def finals(r: Run) -> dict[str, tuple[int, int]]:
            out: dict[str, tuple[int, int]] = {}
            for s in r.snapshots:
                if s.kind != "instrumentation":
                    out[s.stage] = (s.idx, s.total_ops)
            return out

        fa, fb = finals(a), finals(b)
        stages = [CompareStage(stage=s, a_snapshot_idx=fa.get(s, (None, None))[0], b_snapshot_idx=fb.get(s, (None, None))[0],
                               a_ops=fa.get(s, (None, None))[1], b_ops=fb.get(s, (None, None))[1])
                  for s in STAGES if s in fa or s in fb]
        diff = "".join(difflib.unified_diff(a.pipeline_yaml.splitlines(True), b.pipeline_yaml.splitlines(True),
                                            fromfile=f"{a.id} pipeline", tofile=f"{b.id} pipeline"))
        child = b if b.parent_id == a.id else a if a.parent_id == b.id else None
        summaries = {x.id: x for x in store.list()}
        return CompareResult(a=summaries[a.id], b=summaries[b.id], a_metrics=a.metrics, b_metrics=b.metrics, stages=stages,
                             config_diff=diff, fix=FIXES.get(child.fix_id) if child and child.fix_id else None)

    @api.get("/runs/{run_id}/reproducer")
    def reproducer(run_id: str) -> Response:
        store = get_store()
        run = store.get(run_id)
        if run.status != "done":
            raise HTTPException(409, "the run has not finished")
        return Response(build_reproducer(run, store.run_dir(run_id)), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="optiler-{run_id}.zip"'})

    # -------------------------------------------------------------- templates, toolchain
    @api.get("/templates", response_model=list[TemplateInfo])
    def list_templates() -> list[TemplateInfo]:
        return [TemplateInfo(name=n, description=_template_description(TEMPLATES_DIR / f"{n}.py"))
                for n in _template_names()]

    @api.get("/templates/{name}", response_model=TemplateSource)
    def get_template(name: str) -> TemplateSource:
        return TemplateSource(name=name, source=_template_path(name).read_text())

    @api.get("/pipeline")
    def default_pipeline() -> dict:
        return {"yaml": DEFAULT_PIPELINE.read_text()}

    @api.get("/toolchain")
    def get_toolchain() -> dict:
        return toolchain_report()

    app.include_router(api)
    if FRONTEND_DIR.is_dir():  # one process serves API and UI; mounted last so /api/* wins
        app.mount("/", RevalidatingStatic(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
