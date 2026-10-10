"""Time mesh conversion on the same parts in two checkouts, interleaved.

For each part the script runs a worker with checkout A's interpreter, then the
same part with checkout B's interpreter, so machine drift affects both equally.
Each worker tessellates the part, warms up once and reports the fastest of
``--repeat`` ``unmesh.convert`` calls. Reports the per-part time ratio B/A and
its median; exits 1 when the median exceeds 1.1.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess

WORKER = r"""
import sys
import time
import numpy as np
import unmesh
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import tessellate

family, part, repeat = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mesh = tessellate(generate(family, part).solid, 0.01, 0.2)
tris = np.asarray(mesh.tris)
unmesh.convert(tris)
best = float("inf")
for _ in range(repeat):
    start = time.perf_counter()
    unmesh.convert(tris)
    best = min(best, time.perf_counter() - start)
print(best)
"""

DEFAULT_PARTS = ",".join(
    [
        "straight_fillet:0",
        "straight_fillet:1",
        "straight_fillet:2",
        "straight_fillet:3",
        "straight_fillet:4",
        "circular_fillet:0",
        "circular_fillet:1",
        "circular_fillet:2",
        "circular_fillet:3",
        "corner_fillet:0",
        "corner_fillet:1",
        "corner_fillet:2",
        "corner_fillet:3",
        "bore_chamfer:0",
        "bore_chamfer:1",
        "bore_chamfer:2",
        "through_bore:0",
        "through_bore:1",
        "round_boss:0",
        "round_boss:1",
    ]
)


def time_one(python: str, part: str, repeat: int) -> float:
    family, seed = part.split(":")
    out = subprocess.run(
        [python, "-c", WORKER, family, seed, str(repeat)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-a", required=True, help="baseline interpreter")
    p.add_argument("--python-b", required=True, help="candidate interpreter")
    p.add_argument("--parts", default=DEFAULT_PARTS, help="comma-separated family:part")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args(argv)

    parts = args.parts.split(",")
    ratios = []
    print(f"{'part':26s} {'A ms':>9s} {'B ms':>9s} {'B/A':>7s}")
    for part in parts:
        a = time_one(args.python_a, part, args.repeat) * 1e3
        b = time_one(args.python_b, part, args.repeat) * 1e3
        ratios.append(b / a)
        print(f"{part:26s} {a:9.1f} {b:9.1f} {b / a:7.3f}", flush=True)
    print(f"\nmedian B/A = {statistics.median(ratios):.3f}")
    print(f"mean   B/A = {statistics.fmean(ratios):.3f}")
    return 0 if statistics.median(ratios) <= 1.1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
