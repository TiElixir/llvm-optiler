"""Static cost model for linalg-on-tensors ops, and a liveness summary for bufferized modules.

``compute_costs`` is keyed by the walk index of ``backend.provenance.mlir_ops.walk``. Bytes
are the tensor traffic of the op: the bytes of every ranked-tensor input plus every result
(the destination-passing ``outs`` are the results). An op whose shapes are dynamic, or whose
kind is unknown, has no entry; the function never raises on an unexpected op.
"""
from __future__ import annotations

import re

from torch_mlir import ir

from backend.provenance.mlir_ops import walk

ARITH_FLOPS = {"add": 1, "sub": 1, "mul": 1, "max": 1, "min": 1, "cmp": 1, "select": 1, "neg": 1,
               "div": 4}
MATH_FLOPS = {"exp": 10, "log": 10, "tanh": 10, "sqrt": 10, "rsqrt": 10}
NAMED_ELEMENTWISE = {"linalg.add": 1, "linalg.sub": 1, "linalg.mul": 1, "linalg.div": 4,
                     "linalg.exp": 10, "linalg.log": 10, "linalg.abs": 1, "linalg.negf": 1,
                     "linalg.select": 1}
ZERO_FLOP_NAMED = {"linalg.fill", "linalg.transpose", "linalg.broadcast", "linalg.copy"}
ALIAS_OPS = {"memref.expand_shape", "memref.collapse_shape", "memref.subview", "memref.cast",
             "memref.reinterpret_cast", "memref.reshape"}
RETURN_OPS = {"func.return", "return", "llvm.return"}
SKIP_OPS = {"linalg.yield", "linalg.index"}


class _Skip(Exception):
    """Raised for an op that cannot be costed; the op is then absent from the result."""


def _elt_bytes(elt) -> int:
    s = str(elt)
    if s == "index":
        return 8
    m = re.fullmatch(r"(?:bf|f|si|ui|i)(\d+)", s)
    if not m:
        raise _Skip(f"element type {s}")
    return max(1, (int(m.group(1)) + 7) // 8)


def _shaped(t):
    """(numel, element bytes) of a static ranked tensor or memref; None if not shaped."""
    if isinstance(t, ir.RankedTensorType) or isinstance(t, ir.MemRefType):
        if not t.has_static_shape:
            raise _Skip("dynamic shape")
        numel = 1
        for d in t.shape:
            numel *= int(d)
        return numel, _elt_bytes(t.element_type)
    return None


def _nbytes(t) -> int:
    s = _shaped(t)
    return 0 if s is None else s[0] * s[1]


def _payload_flops(op) -> float:
    """Flops of one iteration of a linalg.generic body (arith/math ops, nested regions included)."""
    total = 0.0
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                c = child.operation
                parts = c.name.split(".", 1)
                if len(parts) == 2 and parts[0] == "arith":
                    key = re.sub(r"(f|i|si|ui)$", "", parts[1])
                    if key.startswith(("max", "min")):  # maximumf, maxnumf, maxsi, ...
                        key = "max"
                    total += ARITH_FLOPS.get(key, 0)
                elif len(parts) == 2 and parts[0] == "math":
                    total += MATH_FLOPS.get(parts[1], 0)
                total += _payload_flops(c)
    return total


def _generic_domain(op, operand_types) -> int:
    """Iteration-domain size: each loop dim takes the extent of the first operand dim it indexes."""
    maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != len(operand_types):
        raise _Skip("indexing maps do not match operands")
    n_dims = maps[0].n_dims
    extents: list[int | None] = [None] * n_dims
    for amap, t in zip(maps, operand_types):
        s = _shaped(t)
        if s is None:
            continue
        shape = list(t.shape)
        for pos, expr in enumerate(amap.results):
            if isinstance(expr, ir.AffineDimExpr) and pos < len(shape):
                d = expr.position
                if extents[d] is None:
                    extents[d] = int(shape[pos])
    if any(e is None for e in extents):
        raise _Skip("loop dimension not indexed by any operand")
    size = 1
    for e in extents:
        size *= e
    return size


def _named_flops(name: str, ins, results) -> float:
    if name.startswith(("linalg.matmul", "linalg.batch_matmul")):
        lhs = ins[0]
        out_numel = _shaped(results[0])[0]
        k = lhs.shape[-2] if "transpose_a" in name else lhs.shape[-1]
        return 2.0 * out_numel * int(k)
    if name in ("linalg.matvec", "linalg.vecmat"):
        return 2.0 * _shaped(ins[0])[0]
    if name in ZERO_FLOP_NAMED:
        return 0.0
    if name in NAMED_ELEMENTWISE:
        return float(NAMED_ELEMENTWISE[name] * _shaped(results[0])[0])
    raise _Skip(f"no cost model for {name}")


def _op_cost(op, name: str) -> dict[str, float] | None:
    operands = [v.type for v in op.operands]
    results = [r.type for r in op.results]
    n_out = len(results)
    ins_types = operands[: len(operands) - n_out] if n_out else operands
    if name == "linalg.generic":
        flops = _generic_domain(op, operands) * _payload_flops(op)
    else:
        flops = _named_flops(name, ins_types, results)
    read = float(sum(_nbytes(t) for t in ins_types))
    written = float(sum(_nbytes(t) for t in results))
    total = read + written
    return {"flops": float(flops), "bytes_read": read, "bytes_written": written,
            "bytes": total, "intensity": float(flops) / total if total else 0.0}


def compute_costs(module: ir.Module) -> dict[int, dict[str, float]]:
    """Cost of every linalg-on-tensors op in ``module``, keyed by walk index (see module doc)."""
    out: dict[int, dict[str, float]] = {}
    for idx, (op, _depth, _parent, payload) in enumerate(walk(module.operation)):
        name = op.name
        if payload or not name.startswith("linalg.") or name in SKIP_OPS:
            continue
        try:
            cost = _op_cost(op, name)
        except Exception:  # noqa: BLE001 - an op that cannot be costed is simply absent
            continue
        if cost is not None:
            out[idx] = cost
    return out


def _extent(value, index_of: dict[int, int], n: int) -> int:
    """Last walk index at which ``value`` (or a view of it) is used; n-1 when it escapes."""
    end = -1
    for use in value.uses:
        owner = getattr(use.owner, "operation", None)
        if owner is None:
            end = n - 1
            continue
        if owner.name in RETURN_OPS:
            return n - 1
        j = index_of.get(hash(owner))
        if j is None:
            end = n - 1
            continue
        end = max(end, j)
        if owner.name in ALIAS_OPS and owner.results:
            end = max(end, _extent(owner.results[0], index_of, n))
    return end


def static_alloc_summary(module: ir.Module) -> dict[str, int]:
    """Static ``memref.alloc`` count/bytes and peak live bytes, for a bufferized module.

    An allocation is live from its walk index to its last use (a ``memref.dealloc`` or the last
    user, following views); a value returned from the function is live to the end.
    """
    walked = walk(module.operation)
    n = len(walked)
    index_of = {hash(op): i for i, (op, _d, _p, _pl) in enumerate(walked)}
    intervals: list[tuple[int, int, int]] = []
    count = total = 0
    for i, (op, _d, _p, _pl) in enumerate(walked):
        if op.name != "memref.alloc" or not op.results:
            continue
        try:
            s = _shaped(op.results[0].type)
        except _Skip:
            continue
        if s is None:
            continue
        size = s[0] * s[1]
        count += 1
        total += size
        intervals.append((i, _extent(op.results[0], index_of, n), size))
    peak = 0
    for t in range(n):
        peak = max(peak, sum(b for d, e, b in intervals if d <= t <= e))
    return {"alloc_count": count, "alloc_bytes": total, "peak_live_bytes": peak}

