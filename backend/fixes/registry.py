"""Fixes are data: a pipeline patch plus an optional transform script (PLAN.md section 9).

Applying a fix (M6) means creating a child run with the patched pipeline; nothing here executes anything.
"""
from __future__ import annotations

from backend.schemas import Fix, PipelinePatchOp

TRANSFORM_PATH = "${RUN_DIR}/transforms/{fix_id}.mlir"

# Loop interchange so the reduction loop is not innermost, and keep affine-loop-fusion from re-nesting it:
# measured at 13-15x on the matmul templates (docs/toolchain.md).
INTERCHANGE_REDUCTION = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %g = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    %i = transform.structured.interchange %g iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""

FIXES: dict[str, Fix] = {
    "vectorize-reduction-loops": Fix(
        id="vectorize-reduction-loops",
        label="interchange matmul loops",
        description="Move the reduction loop out of the innermost position (transform interchange) and drop "
                    "affine-loop-fusion, which would put it back, so LLVM can vectorize the parallel inner loop.",
        patch=[
            PipelinePatchOp(op="remove", target="affine-loop-fusion"),
            PipelinePatchOp(op="insert_after", stage="linalg_tensor",
                            value="transform-preload-library{transform-library-paths=" + TRANSFORM_PATH.format(fix_id="vectorize-reduction-loops") + "}"),
            PipelinePatchOp(op="insert_after", stage="linalg_tensor", value="transform-interpreter"),
        ],
        transform_script=INTERCHANGE_REDUCTION,
    ),
    "enable-elementwise-fusion": Fix(
        id="enable-elementwise-fusion",
        label="fuse elementwise ops",
        description="Run linalg-fuse-elementwise-ops at the end of the linalg stage so producer/consumer "
                    "elementwise generics become one op and the intermediate buffer disappears.",
        patch=[PipelinePatchOp(op="insert_after", stage="linalg_tensor", value="func.func(linalg-fuse-elementwise-ops)")],
    ),
}
