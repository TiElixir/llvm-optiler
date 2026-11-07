"""API tests: fast ones need no compile; one slow end-to-end run goes through run_job in-process."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import backend.api.app as app_module
from backend.api.app import create_app
from backend.pipeline.config import DEFAULT_PIPELINE
from backend.runner.job import run_job
from backend.schemas import ProgressEvent, Run, Snapshot
from backend.store.runs import RunStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(runs_dir=tmp_path, start_worker=False))


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path)


def snap(idx: int, stage: str, name: str, path: str) -> Snapshot:
    return Snapshot(idx=idx, stage=stage, **{"pass": name}, path=path)


def fabricate(store: RunStore, texts: dict[str, str], snapshots: list[Snapshot],
              status: str = "done") -> Run:
    """A saved run with hand-written snapshot files (no compile)."""
    run = store.create(Run(id="", created_at="", source_code="x = 1", pipeline_yaml="- stage: torch"))
    d = store.run_dir(run.id)
    for rel, text in texts.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    run.snapshots, run.status = snapshots, status
    store.save(run)
    (d / "events.jsonl").unlink(missing_ok=True)  # fabricated runs start with no events
    return run


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line[len("data:"):].strip()) for line in body.splitlines() if line.startswith("data:")]


def read_sse(client: TestClient, url: str) -> list[dict]:
    with client.stream("GET", url) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse_sse("".join(r.iter_text()))


# ------------------------------------------------------------------ templates, toolchain

def test_templates_list_and_get(client):
    r = client.get("/api/templates")
    assert r.status_code == 200
    items = r.json()
    names = [t["name"] for t in items]
    assert "matmul_relu" in names and names == sorted(names)
    assert all(set(t) == {"name", "description"} for t in items)

    r = client.get("/api/templates/matmul_relu")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "matmul_relu"
    assert body["source"] == (ROOT / "backend/templates/matmul_relu.py").read_text()


@pytest.mark.parametrize("name", ["nope", "..%2Fschemas", "%2e%2e%2fschemas", "__init__", "matmul_relu.py"])
def test_template_unknown_or_traversal_is_404(client, name):
    assert client.get(f"/api/templates/{name}").status_code == 404


def test_toolchain(client):
    r = client.get("/api/toolchain")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["tools"], dict) and "clang" in body["tools"]
    assert all(v is None or isinstance(v, str) for v in body["tools"].values())
    assert "torch_mlir" in body and isinstance(body["warnings"], list)



# ------------------------------------------------------------------ creating and reading runs

def test_list_runs_empty(client):
    r = client.get("/api/runs")
    assert r.status_code == 200 and r.json() == []


def test_create_run_from_template_is_queued(client, store):
    r = client.post("/api/runs", json={"template": "matmul_relu", "iters": 5, "threads": 2})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id.startswith("run_")

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "queued" and run["phase"] == "queued"
    assert run["label"] == "matmul_relu" and run["template"] == "matmul_relu"
    assert run["source_code"] == (ROOT / "backend/templates/matmul_relu.py").read_text()
    assert run["pipeline_yaml"] == DEFAULT_PIPELINE.read_text()
    assert run["settings"]["iters"] == 5 and run["settings"]["threads"] == 2
    assert run["seq"] == 1 and run["snapshots"] == []

    summaries = client.get("/api/runs").json()
    assert [s["id"] for s in summaries] == [run_id] and summaries[0]["status"] == "queued"
    assert store.claim_next() == run_id  # the job is enqueued for the worker


def test_create_run_with_source_and_label(client):
    r = client.post("/api/runs", json={"source": "print('hi')", "label": "mine",
                                       "template": "matmul_relu"})
    run = client.get(f"/api/runs/{r.json()['run_id']}").json()
    assert run["source_code"] == "print('hi')" and run["label"] == "mine"


def test_child_run_inherits_source_and_pipeline(client):
    custom = "- stage: torch\n  passes: [canonicalize]\n"
    parent_id = client.post("/api/runs", json={"source": "x = 1", "pipeline_yaml": custom}).json()["run_id"]
    child_id = client.post("/api/runs", json={"parent_id": parent_id, "fix_id": "fx1"}).json()["run_id"]
    child = client.get(f"/api/runs/{child_id}").json()
    assert child["parent_id"] == parent_id and child["fix_id"] == "fx1"
    assert child["source_code"] == "x = 1" and child["pipeline_yaml"] == custom
    assert child["seq"] == 2

    override = client.post("/api/runs", json={"parent_id": parent_id, "pipeline_yaml": DEFAULT_PIPELINE.read_text()})
    assert client.get(f"/api/runs/{override.json()['run_id']}").json()["pipeline_yaml"] == DEFAULT_PIPELINE.read_text()


@pytest.mark.parametrize("yaml_text", [
    "stages: [",                                   # not YAML
    "- stage: bogus\n  passes: [canonicalize]",     # unknown stage
    "- stage: torch",                               # missing passes
    "just a string",
])
def test_bad_pipeline_yaml_is_422(client, yaml_text):
    r = client.post("/api/runs", json={"source": "x = 1", "pipeline_yaml": yaml_text})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("invalid pipeline")
    assert client.get("/api/runs").json() == []  # nothing was created


def test_no_source_is_422(client):
    r = client.post("/api/runs", json={})
    assert r.status_code == 422
    assert client.post("/api/runs", json={"source": "   "}).status_code == 422
    assert client.post("/api/runs", json={"source": "x", "iters": 0}).status_code == 422


def test_unknown_parent_or_template_is_404(client):
    assert client.post("/api/runs", json={"source": "x", "parent_id": "run_deadbeef"}).status_code == 404
    assert client.post("/api/runs", json={"template": "nope"}).status_code == 404
    assert client.post("/api/runs", json={"source": "x", "template": "nope"}).status_code == 404


@pytest.mark.parametrize("path", [
    "", "/events", "/snapshots/0", "/diff?a=0&b=1", "/stages/torch",
])
def test_unknown_run_is_404(client, path):
    assert client.get(f"/api/runs/run_00000000{path}").status_code == 404


def test_malformed_run_id_is_404(client):
    assert client.get("/api/runs/..").status_code == 404
    assert client.get("/api/runs/a.b").status_code == 404


# ------------------------------------------------------------------ snapshots, diff, stages

def test_snapshot_text_and_missing(client, store):
    run = fabricate(store, {"snapshots/000_import.mlir": "module {\n}\n"},
                    [snap(0, "torch", "import", "snapshots/000_import.mlir"),
                     snap(1, "torch", "gone", "snapshots/001_gone.mlir")])
    r = client.get(f"/api/runs/{run.id}/snapshots/0")
    assert r.status_code == 200 and r.text == "module {\n}\n"
    assert r.headers["content-type"].startswith("text/plain")
    assert client.get(f"/api/runs/{run.id}/snapshots/1").status_code == 404   # file missing
    assert client.get(f"/api/runs/{run.id}/snapshots/7").status_code == 404   # no such snapshot
    assert client.get(f"/api/runs/{run.id}/snapshots/x").status_code == 422


def test_snapshot_path_escape_is_refused(client, store, tmp_path):
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    run = fabricate(store, {}, [snap(0, "torch", "up", "../secret.txt"),
                                snap(1, "torch", "abs", str(tmp_path / "secret.txt")),
                                snap(2, "torch", "sneaky", "snapshots/../../secret.txt")])
    for idx in range(3):
        r = client.get(f"/api/runs/{run.id}/snapshots/{idx}")
        assert r.status_code == 404 and "SECRET" not in r.text
    ok = fabricate(store, {"snapshots/a.mlir": "a\n"}, [snap(0, "torch", "a", "snapshots/a.mlir")])
    assert client.get(f"/api/runs/{run.id}/diff?a=0&b=1").status_code == 404
    assert client.get(f"/api/runs/{ok.id}/diff?a=0&b=0&b_run={run.id}").status_code == 404


def test_diff_within_and_across_runs(client, store):
    run = fabricate(store, {"snapshots/000_a.mlir": "one\ntwo\nthree\n", "snapshots/001_b.mlir": "one\n2\nthree\n"},
                    [snap(0, "torch", "import", "snapshots/000_a.mlir"),
                     snap(1, "torch", "canonicalize", "snapshots/001_b.mlir")])
    other = fabricate(store, {"snapshots/000_a.mlir": "one\ntwo\nthree\nfour"},
                      [snap(0, "torch", "import", "snapshots/000_a.mlir")])

    d = client.get(f"/api/runs/{run.id}/diff", params={"a": 0, "b": 1}).json()
    assert d["a"] == {"idx": 0, "stage": "torch", "pass_name": "import", "text": "one\ntwo\nthree\n"}
    assert d["b"]["pass_name"] == "canonicalize"
    lines = d["unified"].splitlines()
    assert lines[0].startswith("--- ") and lines[1].startswith("+++ ")
    assert "-two" in lines and "+2" in lines

    same = client.get(f"/api/runs/{run.id}/diff", params={"a": 1, "b": 1}).json()
    assert same["unified"] == ""

    cross = client.get(f"/api/runs/{run.id}/diff", params={"a": 0, "b": 0, "b_run": other.id}).json()
    assert "+four" in cross["unified"].splitlines()
    assert other.id in cross["unified"] and run.id in cross["unified"]

    assert client.get(f"/api/runs/{run.id}/diff", params={"a": 0, "b": 9}).status_code == 404
    assert client.get(f"/api/runs/{run.id}/diff", params={"a": 0, "b": 0, "b_run": "run_00000000"}).status_code == 404
    assert client.get(f"/api/runs/{run.id}/diff", params={"a": 0}).status_code == 422


def test_stage_endpoint(client, store):
    run = fabricate(store, {}, [snap(0, "torch", "import", "x"), snap(1, "torch", "canon", "y"),
                                snap(2, "linalg_tensor", "lower", "z")])
    r = client.get(f"/api/runs/{run.id}/stages/torch")
    assert r.status_code == 200
    assert r.json() == {"stage": "torch", "snapshot_idx": 1, "ops": [], "links": [], "exact_lineage_pct": None}
    assert client.get(f"/api/runs/{run.id}/stages/linalg_tensor").json()["snapshot_idx"] == 2
    assert client.get(f"/api/runs/{run.id}/stages/asm").json()["snapshot_idx"] is None
    assert client.get(f"/api/runs/{run.id}/stages/nonsense").status_code == 404


def test_run_json_uses_pass_alias(client, store):
    run = fabricate(store, {}, [snap(0, "torch", "import", "x")])
    s = client.get(f"/api/runs/{run.id}").json()["snapshots"][0]
    assert s["pass"] == "import" and "pass_name" not in s


def test_apply_validation_and_reproducer_not_implemented(client, store):
    run = fabricate(store, {}, [])
    assert client.post(f"/api/runs/{run.id}/findings/f1/apply").status_code == 404   # no such finding
    r = client.get(f"/api/runs/{run.id}/reproducer")
    assert r.status_code == 501 and r.json() == {"detail": "not implemented yet"}


# ------------------------------------------------------------------ SSE

def test_sse_replays_finished_run(client, store):
    run = fabricate(store, {}, [])
    for ev in [ProgressEvent(phase="capture"), ProgressEvent(phase="passes", n=1, total=2, name="canonicalize"),
               ProgressEvent(type="done", phase="done")]:
        store.append_event(run.id, ev)
    events = read_sse(client, f"/api/runs/{run.id}/events")
    assert [(e["type"], e["phase"]) for e in events] == [("progress", "capture"), ("progress", "passes"),
                                                         ("done", "done")]
    assert events[1]["n"] == 1 and events[1]["total"] == 2 and events[1]["name"] == "canonicalize"


def test_sse_stops_at_first_terminal_event(client, store):
    run = fabricate(store, {}, [], status="error")
    store.append_event(run.id, ProgressEvent(phase="capture"))
    store.append_event(run.id, ProgressEvent(type="error", phase="error", message="boom"))
    store.append_event(run.id, ProgressEvent(phase="capture"))  # never sent
    events = read_sse(client, f"/api/runs/{run.id}/events")
    assert [e["type"] for e in events] == ["progress", "error"] and events[-1]["message"] == "boom"


@pytest.mark.parametrize("status, kind", [("done", "done"), ("error", "error")])
def test_sse_finished_run_without_events_gets_terminal_event(client, store, monkeypatch, status, kind):
    monkeypatch.setattr(app_module, "FINISHED_GRACE_S", 0.1)
    run = fabricate(store, {}, [], status=status)
    run.error = "bad" if status == "error" else None
    store.save(run)
    events = read_sse(client, f"/api/runs/{run.id}/events")
    assert len(events) == 1 and events[0]["type"] == kind
    if status == "error":
        assert events[0]["message"] == "bad"


def test_sse_tails_a_running_run(client, store):
    run = fabricate(store, {}, [], status="running")
    store.append_event(run.id, ProgressEvent(phase="capture"))

    def finish():
        store.append_event(run.id, ProgressEvent(phase="passes", n=1, total=1, name="canonicalize"))
        done = store.get(run.id)
        done.status = "done"
        store.save(done)
        store.append_event(run.id, ProgressEvent(type="done", phase="done"))

    timer = threading.Timer(0.6, finish)
    timer.start()
    try:
        events = read_sse(client, f"/api/runs/{run.id}/events")
    finally:
        timer.join()
    assert [e["phase"] for e in events] == ["capture", "passes", "done"]


# ------------------------------------------------------------------ static frontend

def test_serves_frontend_and_api_side_by_side(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Optiler</title>" in r.text
    assert client.get("/js/main.js").status_code == 200
    assert client.get("/api/templates").status_code == 200


# ------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    runs = tmp_path_factory.mktemp("e2e_runs")
    client = TestClient(create_app(runs_dir=runs, start_worker=False))
    r = client.post("/api/runs", json={"template": "matmul_relu", "iters": 5})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    store = RunStore(runs)
    run_job(store, run_id)
    run = client.get(f"/api/runs/{run_id}").json()
    return SimpleNamespace(client=client, store=store, id=run_id, run=run)


@pytest.mark.slow
def test_e2e_run_finishes_correct(e2e):
    run = e2e.run
    assert run["status"] == "done", run["error"]
    assert run["phase"] == "done" and run["error"] is None
    assert run["metrics"]["correct"] is True
    assert run["metrics"]["runtime_ms_median"] > 0
    assert len(run["snapshots"]) > 40
    assert [s["idx"] for s in run["snapshots"]] == list(range(len(run["snapshots"])))
    assert all("pass" in s for s in run["snapshots"])
    summary = e2e.client.get("/api/runs").json()[0]
    assert summary["id"] == e2e.id and summary["status"] == "done" and summary["correct"] is True


@pytest.mark.slow
def test_e2e_every_snapshot_is_served(e2e):
    for s in e2e.run["snapshots"]:
        r = e2e.client.get(f"/api/runs/{e2e.id}/snapshots/{s['idx']}")
        assert r.status_code == 200, s
        assert r.text.strip(), f"snapshot {s['idx']} ({s['pass']}) is empty"


@pytest.mark.slow
def test_e2e_diff_between_consecutive_snapshots(e2e):
    snaps = [s for s in e2e.run["snapshots"] if s["path"].endswith(".mlir")]
    texts = {s["idx"]: e2e.client.get(f"/api/runs/{e2e.id}/snapshots/{s['idx']}").text for s in snaps}
    pairs = [(a, b) for a, b in zip(snaps, snaps[1:]) if b["idx"] == a["idx"] + 1]
    changed = [(a, b) for a, b in pairs if texts[a["idx"]] != texts[b["idx"]]]
    assert changed, "no pass changed the printed IR"
    a, b = changed[0]
    d = e2e.client.get(f"/api/runs/{e2e.id}/diff", params={"a": a["idx"], "b": b["idx"]}).json()
    assert d["a"]["text"] == texts[a["idx"]] and d["b"]["text"] == texts[b["idx"]]
    lines = d["unified"].splitlines()
    assert lines[0].startswith("---") and lines[1].startswith("+++")
    assert any(l.startswith("@@") for l in lines)
    assert any(l.startswith("+") and not l.startswith("+++") for l in lines)
    assert any(l.startswith("-") and not l.startswith("---") for l in lines)
    # a pass that left the module untouched produces an empty diff
    for a, b in pairs:
        if texts[a["idx"]] == texts[b["idx"]]:
            d = e2e.client.get(f"/api/runs/{e2e.id}/diff", params={"a": a["idx"], "b": b["idx"]}).json()
            assert d["unified"] == ""
            break


@pytest.mark.slow
def test_e2e_events_stream(e2e):
    events = read_sse(e2e.client, f"/api/runs/{e2e.id}/events")
    assert events[0]["type"] == "progress" and events[0]["phase"] == "queued"
    assert events[1]["phase"] == "capture"
    assert events[-1]["type"] == "done"
    phases = [e["phase"] for e in events]
    for p in ("import", "passes", "build", "provenance", "benchmark", "analyze"):
        assert p in phases
    passes = [e for e in events if e["phase"] == "passes"]
    assert passes[-1]["n"] == passes[-1]["total"] and passes[0]["name"]


@pytest.mark.slow
def test_e2e_stage_endpoint_points_at_last_snapshot(e2e):
    """The stage's op table belongs to its last real snapshot (instrumentation passes come after it)."""
    kinds = {s["idx"]: s["kind"] for s in e2e.run["snapshots"]}
    for st in e2e.run["stages"]:
        r = e2e.client.get(f"/api/runs/{e2e.id}/stages/{st['id']}").json()
        real = [i for i in st["snapshot_idxs"] if kinds[i] != "instrumentation"]
        assert r["snapshot_idx"] == (max(real) if real else None), st["id"]


@pytest.mark.slow
def test_e2e_stage_ops_links_and_coverage(e2e):
    run = e2e.client.get(f"/api/runs/{e2e.id}").json()
    prev_ids: set[str] = set()
    for st in run["stages"]:
        r = e2e.client.get(f"/api/runs/{e2e.id}/stages/{st['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["ops"], st["id"]
        assert body["exact_lineage_pct"] is not None
        assert st["exact_lineage_pct"] == pytest.approx(body["exact_lineage_pct"])
        for link in body["links"]:
            assert link["from_op"] in prev_ids
        prev_ids = {o["id"] for o in body["ops"]}
        assert all(o["source_lines"] == ([o["source_line"]] if o["source_line"] and len(o["source_lines"]) == 1
                                         else o["source_lines"]) for o in body["ops"])


@pytest.mark.slow
def test_e2e_apply_fix_creates_a_child_and_compare_shows_the_patch(e2e):
    findings = e2e.client.get(f"/api/runs/{e2e.id}").json()["findings"]
    f = next(f for f in findings if f["id"] == "d3-unvectorized-hot-loop")
    assert e2e.client.post(f"/api/runs/{e2e.id}/findings/{f['id']}/apply", json={"fix_id": "nope"}).status_code == 422
    r = e2e.client.post(f"/api/runs/{e2e.id}/findings/{f['id']}/apply")
    assert r.status_code == 200
    child_id = r.json()["run_id"]
    child = e2e.client.get(f"/api/runs/{child_id}").json()
    assert child["parent_id"] == e2e.id and child["fix_id"] == "vectorize-reduction-loops" and child["status"] == "queued"
    assert "affine-loop-fusion" not in child["pipeline_yaml"] and "transform-interpreter" in child["pipeline_yaml"]
    run_job(e2e.store, child_id)
    cmp = e2e.client.get(f"/api/runs/{e2e.id}/compare", params={"other": child_id}).json()
    assert cmp["a_metrics"]["correct"] and cmp["b_metrics"]["correct"]
    assert cmp["b_metrics"]["runtime_ms_median"] < cmp["a_metrics"]["runtime_ms_median"]
    assert "-" in cmp["config_diff"] and "affine-loop-fusion" in cmp["config_diff"] and "+" in cmp["config_diff"]
    assert cmp["fix"]["id"] == "vectorize-reduction-loops"
    assert [s["stage"] for s in cmp["stages"]][0] == "torch" and all(s["a_ops"] and s["b_ops"] for s in cmp["stages"])
    parent = e2e.client.get(f"/api/runs/{e2e.id}").json()
    done = next(f for f in parent["findings"] if f["id"] == f["id"] and f["id"] == "d3-unvectorized-hot-loop")
    assert done["impact_verified"]["kind"] == "verified" and done["child_run_id"] == child_id
