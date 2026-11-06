"""Run storage: ``runs/<id>/run.json`` + ``runs/index.db`` (SQLite index and job queue)."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.schemas import ProgressEvent, Run, RunSummary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = Path(os.environ.get("OPTILER_RUNS_DIR", ROOT / "runs"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, seq INTEGER NOT NULL, parent_id TEXT, status TEXT NOT NULL,
  fix_id TEXT, label TEXT, created_at TEXT NOT NULL, runtime_ms REAL, correct INTEGER
);
CREATE TABLE IF NOT EXISTS jobs (
  run_id TEXT PRIMARY KEY, state TEXT NOT NULL, enqueued_at REAL NOT NULL, claimed_at REAL
);
"""


class RunNotFound(KeyError):
    pass


class RunStore:
    def __init__(self, root: Path | str = DEFAULT_RUNS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------ paths
    def run_dir(self, run_id: str) -> Path:
        if not run_id.replace("_", "").isalnum():
            raise RunNotFound(run_id)
        return self.root / run_id

    # ------------------------------------------------------------ runs
    def create(self, run: Run | None = None, **fields) -> Run:
        """Create a new queued run (id and seq are assigned here) and enqueue its job."""
        rid = f"run_{secrets.token_hex(4)}"
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._db() as db:  # seq is allocated inside one write transaction
            db.execute("BEGIN IMMEDIATE")
            seq = db.execute("SELECT COALESCE(MAX(seq), 0) FROM runs").fetchone()[0] + 1
            run = (run.model_copy(update=dict(id=rid, seq=seq, created_at=created)) if run
                   else Run(id=rid, seq=seq, created_at=created, **fields))
            db.execute(
                "INSERT INTO runs (id, seq, parent_id, status, fix_id, label, created_at) VALUES (?,?,?,?,?,?,?)",
                (rid, seq, run.parent_id, run.status, run.fix_id, run.label, created))
            db.execute("INSERT INTO jobs (run_id, state, enqueued_at) VALUES (?, 'queued', ?)",
                       (rid, time.time()))
        self.run_dir(rid).mkdir(parents=True)
        self.save(run)
        self.append_event(rid, ProgressEvent(phase="queued"))
        return run

    def save(self, run: Run) -> None:
        d = self.run_dir(run.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "run.json.tmp"
        tmp.write_text(run.model_dump_json(by_alias=True, indent=1))
        os.replace(tmp, d / "run.json")
        with self._db() as db:
            db.execute(
                "INSERT INTO runs (id, seq, parent_id, status, fix_id, label, created_at, runtime_ms, correct) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "label=excluded.label, runtime_ms=excluded.runtime_ms, correct=excluded.correct",
                (run.id, run.seq, run.parent_id, run.status, run.fix_id, run.label, run.created_at,
                 run.metrics.runtime_ms_median,
                 None if run.metrics.correct is None else int(run.metrics.correct)))

    def get(self, run_id: str) -> Run:
        p = self.run_dir(run_id) / "run.json"
        if not p.exists():
            raise RunNotFound(run_id)
        return Run.model_validate_json(p.read_text())

    def list(self) -> list[RunSummary]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY seq").fetchall()
        return [RunSummary(
            id=r["id"], seq=r["seq"], parent_id=r["parent_id"], label=r["label"] or "",
            fix_id=r["fix_id"], status=r["status"], created_at=r["created_at"],
            runtime_ms_median=r["runtime_ms"],
            correct=None if r["correct"] is None else bool(r["correct"])) for r in rows]

    # ------------------------------------------------------------ events
    def append_event(self, run_id: str, ev: ProgressEvent) -> None:
        ev.t = time.time()
        with open(self.run_dir(run_id) / "events.jsonl", "a") as f:
            f.write(ev.model_dump_json() + "\n")

    def read_events(self, run_id: str, offset: int = 0) -> tuple[list[ProgressEvent], int]:
        """Return (events after byte offset, new offset). Only complete lines are consumed."""
        p = self.run_dir(run_id) / "events.jsonl"
        if not p.exists():
            return [], offset
        with open(p, "rb") as f:
            f.seek(offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        evs = [ProgressEvent.model_validate_json(l) for l in data[:end].splitlines() if l.strip()]
        return evs, offset + end

    # ------------------------------------------------------------ jobs
    def claim_next(self) -> str | None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT run_id FROM jobs WHERE state='queued' ORDER BY enqueued_at LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running', claimed_at=? WHERE run_id=?",
                       (time.time(), row["run_id"]))
            return row["run_id"]

    def finish_job(self, run_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET state='done' WHERE run_id=?", (run_id,))

    def requeue_stale(self) -> int:
        """After a crash: running jobs go back to queued (their runs restart from scratch)."""
        with self._db() as db:
            cur = db.execute("UPDATE jobs SET state='queued' WHERE state='running'")
            return cur.rowcount
