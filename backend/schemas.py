"""Data model (PLAN.md section 10). Mirrored to TypeScript by scripts/gen_types.py.

Honesty rule: every number is either measured or labeled ``est``; heuristic links carry
``approximate=True``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

StageId = Literal["source", "torch", "linalg_tensor", "linalg_buffer", "loops",
                  "llvm_dialect", "llvm_ir", "asm"]
RunStatus = Literal["queued", "running", "done", "error"]
Attribution = Literal["perf", "probe", "est"]
Phase = Literal["queued", "capture", "import", "passes", "build", "provenance", "benchmark", "analyze", "done", "error"]


class Snapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    idx: int
    stage: str
    pass_name: str = Field(alias="pass")
    pass_text: str = ""
    composite: bool = False
    changed: bool = False
    time_ms: float = 0.0
    op_counts: dict[str, int] = {}
    total_ops: int = 0
    path: str
    kind: Literal["import", "pass", "instrumentation"] = "pass"


class Stage(BaseModel):
    id: str
    name: str
    snapshot_idxs: list[int] = []
    exact_lineage_pct: Optional[float] = None


class Timing(BaseModel):
    median: float
    p10: float
    p90: float
    n: int


class Metrics(BaseModel):
    runtime_ms_median: Optional[float] = None
    runtime_ms_p10: Optional[float] = None
    runtime_ms_p90: Optional[float] = None
    eager_ms_median: Optional[float] = None
    eager_ms_p10: Optional[float] = None
    eager_ms_p90: Optional[float] = None
    ratio_vs_eager: Optional[float] = None
    threads: int = 1
    iters: int = 100
    correct: Optional[bool] = None
    max_abs_err: Optional[float] = None
    rtol: float = 1e-4
    atol: float = 1e-5
    peak_bytes: Optional[int] = None
    alloc_count: Optional[int] = None
    per_line_share: dict[str, float] = {}
    per_op_share: dict[str, float] = {}
    attribution: Optional[Attribution] = None
    attribution_notice: Optional[str] = None
    attribution_other: Optional[float] = None   # share of samples with no source line (prologue, runtime)
    hottest_line: Optional[int] = None
    hottest_line_code: Optional[str] = None
    hottest_line_share: Optional[float] = None


class Impact(BaseModel):
    ratio: float                       # 1.6 means "1.6x faster if fixed"
    kind: Literal["est", "verified", "none", "slower", "broken"] = "est"
    note: Optional[str] = None


class Finding(BaseModel):
    id: str
    detector: str
    title: str
    op_ids: list[str] = []
    evidence: dict[str, Any] = {}
    impact_est: Impact
    impact_verified: Optional[Impact] = None
    fix_ids: list[str] = []
    child_run_id: Optional[str] = None


class PipelinePatchOp(BaseModel):
    op: Literal["insert_before", "insert_after", "replace", "remove", "set_option"]
    target: Optional[str] = None       # pass text or name the op refers to
    stage: Optional[str] = None
    value: Optional[str] = None        # new pass text / option value
    key: Optional[str] = None


class Fix(BaseModel):
    id: str
    label: str
    description: str = ""
    patch: list[PipelinePatchOp] = []
    transform_script: Optional[str] = None


class Op(BaseModel):
    id: str                            # <stage>:<snapshot_idx>:<walk_idx>
    snapshot_idx: int
    stage: str
    name: str
    loc: Optional[str] = None
    source_line: Optional[int] = None  # first of source_lines
    source_lines: list[int] = []       # lines of the user's source this op stands for (fusion: several)
    approximate: bool = False          # lines came from a heuristic (dataflow neighbours), not a location
    displayable: bool = True           # false for containers, constants, terminators, payload ops
    result_types: list[str] = []
    text_line: Optional[int] = None    # 1-based line in the displayed snapshot text
    address: Optional[int] = None      # asm only: instruction address in model.so
    cost: Optional[dict[str, float]] = None
    share: Optional[float] = None
    share_kind: Optional[Attribution] = None


class Link(BaseModel):
    from_op: str
    to_op: str
    approximate: bool = False


class StageOps(BaseModel):
    stage: str
    snapshot_idx: Optional[int] = None
    ops: list[Op] = []
    links: list[Link] = []             # links from the previous stage's ops to these
    exact_lineage_pct: Optional[float] = None


class FxNode(BaseModel):
    name: str
    op: str
    target: str
    line: Optional[int] = None
    code: Optional[str] = None


class RunSettings(BaseModel):
    iters: int = 100
    warmup: int = 10
    threads: int = 1


class Run(BaseModel):
    id: str
    seq: int = 0
    parent_id: Optional[str] = None
    created_at: str
    label: str = ""
    source_code: str
    transform_script: Optional[str] = None   # user transform-dialect script, written to transforms/user.mlir
    template: Optional[str] = None
    inputs_spec: list[dict[str, Any]] = []
    output_spec: Optional[dict[str, Any]] = None
    pipeline_yaml: str = ""
    fix_id: Optional[str] = None
    settings: RunSettings = RunSettings()
    status: RunStatus = "queued"
    phase: Phase = "queued"
    error: Optional[str] = None
    error_stage: Optional[str] = None
    metrics: Metrics = Metrics()
    stages: list[Stage] = []
    snapshots: list[Snapshot] = []
    findings: list[Finding] = []
    fx_nodes: list[FxNode] = []
    toolchain: dict[str, Any] = {}


class RunSummary(BaseModel):
    id: str
    seq: int
    parent_id: Optional[str] = None
    label: str = ""
    fix_id: Optional[str] = None
    status: RunStatus
    created_at: str
    runtime_ms_median: Optional[float] = None
    correct: Optional[bool] = None


# ------------------------------------------------------------------ API I/O

class CreateRunRequest(BaseModel):
    source: Optional[str] = None           # model source; defaults to the parent's
    template: Optional[str] = None
    pipeline_yaml: Optional[str] = None    # defaults to the parent's, else default.yaml
    parent_id: Optional[str] = None
    fix_id: Optional[str] = None
    label: Optional[str] = None
    transform_script: Optional[str] = None
    iters: Optional[int] = None
    warmup: Optional[int] = None
    threads: Optional[int] = None


class ApplyFixRequest(BaseModel):
    fix_id: Optional[str] = None


class CreateRunResponse(BaseModel):
    run_id: str


class ProgressEvent(BaseModel):
    type: Literal["progress", "done", "error"] = "progress"
    phase: Phase
    n: Optional[int] = None
    total: Optional[int] = None
    name: Optional[str] = None
    message: Optional[str] = None
    t: float = 0.0


class SnapshotText(BaseModel):
    idx: int
    stage: str
    pass_name: str
    text: str


class DiffResult(BaseModel):
    a: SnapshotText
    b: SnapshotText
    unified: str


class TemplateInfo(BaseModel):
    name: str
    description: str = ""


class TemplateSource(BaseModel):
    name: str
    source: str


class CompareStage(BaseModel):
    stage: str
    a_snapshot_idx: Optional[int] = None
    b_snapshot_idx: Optional[int] = None
    a_ops: Optional[int] = None
    b_ops: Optional[int] = None


class CompareResult(BaseModel):
    a: RunSummary
    b: RunSummary
    a_metrics: Metrics
    b_metrics: Metrics
    stages: list[CompareStage] = []
    config_diff: str = ""
    fix: Optional[Fix] = None             # the patch that turns the parent into the child, if one is a fix child
