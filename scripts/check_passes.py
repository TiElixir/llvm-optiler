#!/usr/bin/env python
"""Rule 1 gate: every pass in a pipeline config must exist in the pinned toolchain.

    .venv/bin/python scripts/check_passes.py [pipeline.yaml ...] [--expand]

Exit code 1 if any entry fails to parse. With --expand, prints the leaf passes each
entry expands to (composites are split one pass at a time at run time).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline.config import DEFAULT_PIPELINE  # noqa: E402
from backend.pipeline.validate import check_text  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = [Path(a) for a in args] or [DEFAULT_PIPELINE]
    rc = 0
    for p in paths:
        print(f"checking {p}")
        bad = check_text(p.read_text(), expand_leaves="--expand" in sys.argv)
        for stage, spec, err in bad:
            print(f"  FAIL [{stage}] {spec}\n       {err}")
        if bad:
            rc = 1
        else:
            print("  all passes verified")
    return rc


if __name__ == "__main__":
    sys.exit(main())
