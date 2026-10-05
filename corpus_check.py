"""One-shot corpus oracle-IR comparison (before-change emulation vs new splitting).

BEFORE emulation: build the IR while forcing _split_runs to the whole-edge path,
which is exactly what the pre-change oracle did (it classified each adjacency edge
as a whole from its median dihedral). Labels fields feeding that path (median
dihedral, points, vertices, orientation) are computed by unchanged code, so the
emulated IR is byte-identical to the pre-change IR; a spot check against stashed
original code confirms this.

Writes one JSON object per (entry, deflection) line to OUT.
"""

import json
import math
import os
import sys
import traceback

from unmesh.ir import validate
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, tessellate
from unmesh_harness import oracle as oracle_mod
from unmesh_harness.oracle import _Seg, build_oracle_ir


def build_before(mesh):
    orig = oracle_mod._split_runs

    def whole(d, threshold_deg, pair):
        return [_Seg(d.points, d.dihedral, d.start, d.end)]

    oracle_mod._split_runs = whole
    try:
        return build_oracle_ir(mesh)
    finally:
        oracle_mod._split_runs = orig


def describe(mesh, ir):
    out = []
    for adj in mesh.adjacency:
        if not adj.dihedral_samples or len(adj.dihedral_samples) != len(adj.points):
            continue
        lo = math.degrees(min(adj.dihedral_samples))
        hi = math.degrees(max(adj.dihedral_samples))
        if lo < 3.0 < hi:
            out.append(
                {
                    "pair": [adj.face_a, adj.face_b],
                    "min_deg": lo,
                    "max_deg": hi,
                    "npts": len(adj.points),
                }
            )
    return {
        "kind_change": sum(1 for v in ir.vertices if v.role == "kind_change"),
        "crossed_edges": out,
    }


def main(out_path, shard, nshards):
    entries = select(load_manifest(), "standard")
    mine = [e for i, e in enumerate(entries) if i % nshards == shard]
    with open(out_path, "w") as fh:
        for n, e in enumerate(mine):
            try:
                gt = generate(e["family"], e["seed"])
                for lin, ang in DEFLECTION_SETTINGS:
                    mesh = tessellate(gt.solid, lin, ang)
                    after = build_oracle_ir(mesh)
                    before = build_before(mesh)
                    ta, tb = after.dumps(), before.dumps()
                    rec = {
                        "id": e["id"],
                        "family": e["family"],
                        "seed": e["seed"],
                        "deflection": [lin, ang],
                        "same": ta == tb,
                        "errors": validate(after),
                        "after": describe(mesh, after),
                    }
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[shard {shard}] {n + 1}/{len(mine)} {e['id']} done", flush=True)
            except Exception:
                fh.write(json.dumps({"id": e["id"], "failed": traceback.format_exc()[-2000:]}) + "\n")
                fh.flush()
                print(f"[shard {shard}] {n + 1}/{len(mine)} {e['id']} FAILED", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
