#!/usr/bin/env -S uv run --locked python
"""Scan written STEP files for the shortest edge and the smallest face (issue #153).

Prints the minimum non-degenerate edge length and minimum face area of each
STEP file, so a caller can check that a reused curve-seam split has not left a
needle edge far shorter than any mesh feature. Exits 1 if any edge is below
``--floor`` (a whole-part faceted fallback is exempt; pass only the analytic
STEP files here)::

    uv run --locked harness/scripts/seam_scan.py [--floor 1e-3] part.step other.step
"""

from __future__ import annotations

import argparse
import sys

from unmesh._writer import occ
from unmesh_harness.metrics.edges import minimum_edge_length, minimum_face_area


def scan(path) -> tuple[float, float]:
    shape = occ.read_step(path)
    return minimum_edge_length(shape), minimum_face_area(shape)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, default=1e-3)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    bad = 0
    for path in args.paths:
        edge, face = scan(path)
        below = edge < args.floor
        bad += below
        print(
            f"[{'BAD' if below else 'ok '}] {path}: min_edge={edge:.6g} min_face_area={face:.6g}",
            flush=True,
        )
    print(f"edges below {args.floor:g} mm: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
