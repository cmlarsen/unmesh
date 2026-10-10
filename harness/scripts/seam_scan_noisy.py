#!/usr/bin/env -S uv run --locked python
"""Scan noisy written parts for needle edges (issue #153).

Builds each part the way ``check_regions`` does (ground-truth solid ->
(0.01, 0.2) tessellation -> ``noise_normal`` 0.02 at the noise seed, against the
ground-truth part at ``--gt-seed``), converts and writes it, then reports the
shortest edge and smallest face of the written STEP. Exits 1 if any written edge
is shorter than ``--floor``, except for a whole-part faceted fallback::

    uv run --locked harness/scripts/seam_scan_noisy.py [--gt-seed 0]
        [--floor 1e-3] [--out DIR] [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

LIN, ANG = 0.01, 0.2
NOISE = [("noise_normal", 0.02)]
CASES = [("through_bore", s) for s in range(16)] + [
    (family, s) for family in ("round_boss", "counterbore", "blind_bore") for s in range(3)
]


def scan(job):
    out, family, gt_seed, seed = job
    import numpy as np

    import unmesh
    import unmesh.step as step
    from unmesh._writer import occ
    from unmesh_harness import degrade
    from unmesh_harness.groundtruth import generate
    from unmesh_harness.labels import tessellate
    from unmesh_harness.metrics.edges import minimum_edge_length, minimum_face_area

    mesh = tessellate(generate(family, gt_seed).solid, LIN, ANG)
    mesh = degrade.chain(mesh, NOISE, seed)
    tris = np.asarray(mesh.tris)
    ir, _ = unmesh.convert(tris)
    path = out / f"{family}-{gt_seed}-{seed}.step"
    report = step.write(ir, path, mesh=tris)
    shape = occ.read_step(path)
    return {
        "family": family,
        "gt_seed": gt_seed,
        "seed": seed,
        "vertex_merge": ir.tolerances.vertex_merge,
        "valid": report.valid,
        "fallback": report.fallback,
        "min_edge": minimum_edge_length(shape),
        "min_face_area": minimum_face_area(shape),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("seam-scan-noisy"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--gt-seed", type=int, default=0)
    parser.add_argument("--floor", type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.out, family, args.gt_seed, seed) for family, seed in CASES]
    rows = []
    with mp.Pool(args.jobs) as pool:
        for r in pool.imap_unordered(scan, jobs):
            rows.append(r)
            exempt = r["fallback"] is not None
            bad = not exempt and r["min_edge"] < args.floor
            edge = "skip" if exempt else ("ok  " if not bad else "BAD ")
            print(
                f"[{edge}] {r['family']}-{r['seed']}: min_edge={r['min_edge']:.6g}"
                f" min_face={r['min_face_area']:.6g} vertex_merge={r['vertex_merge']:.1g}"
                f" valid={r['valid']} fallback={r['fallback']}",
                flush=True,
            )
    rows.sort(key=lambda r: (r["family"], r["seed"]))
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    bad = [r for r in rows if r["fallback"] is None and r["min_edge"] < args.floor]
    print(f"edges below {args.floor:g} mm: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
