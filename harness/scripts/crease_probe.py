"""Probe tangency snapping on a real crease beside an R=2 fillet.

Builds a block whose top plane meets an R=2 fillet cylinder at a real crease of
1-5 degrees, then converts it clean and under 1 um normal noise for several
seeds.  For each case it prints whether the crease survives (the plane-cylinder
positional defect stays near R*(1 - cos alpha) rather than collapsing to zero),
the dihedral the IR reports at the crease, and the deviation to the true part.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    CenterArc,
    Line,
    Plane,
    extrude,
    make_face,
)

from unmesh_harness import degrade
from unmesh_harness.judge import judge
from unmesh_harness.labels import tessellate


def crease_part(
    alpha_deg: float,
    radius: float = 2.0,
    width: float = 20.0,
    height: float = 10.0,
    depth: float = 8.0,
):
    alpha = math.radians(alpha_deg)
    drop = (width - radius) * math.tan(alpha)
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Line((0, 0), (width, 0))
                Line((width, 0), (width, height - radius))
                CenterArc((width - radius, height - radius), radius, 0, 90)
                Line((width - radius, height), (0, height - drop))
                Line((0, height - drop), (0, 0))
            make_face()
        extrude(amount=depth)
    return bp.part


def crease_state(ir):
    """(kind, dihedral_deg, plane-cylinder positional defect) at the crease."""
    for adj in ir.adjacencies:
        a, b = adj.regions
        sa, sb = ir.regions[a].surface, ir.regions[b].surface
        if {sa.type, sb.type} != {"plane", "cylinder"}:
            continue
        plane = sa if sa.type == "plane" else sb
        cyl = sb if sa.type == "plane" else sa
        if abs(float(np.array(plane.normal)[2])) < 0.9:
            continue
        n = np.array(plane.normal)
        offset = float(np.dot(n, np.array(plane.origin)))
        sign = 1.0 if cyl.orientation == "same" else -1.0
        defect = abs(float(np.dot(n, np.array(cyl.origin))) + sign * cyl.radius - offset)
        return adj.boundaries[0].kind, adj.boundaries[0].dihedral_deg, defect
    return None


def run(args):
    import unmesh

    for alpha in args.angles:
        part = crease_part(alpha)
        mesh = tessellate(part, args.deflection[0], args.deflection[1])
        truth = tessellate(part, 0.001, 0.1)
        expected = 2.0 * (1.0 - math.cos(math.radians(alpha)))
        cases = [("clean", None, 0)] + [(args.kind, args.kind, s) for s in args.seeds]
        for label, kind, seed in cases:
            cell = mesh if kind is None else degrade.chain(mesh, [(kind, args.severity)], seed)
            ir, report = unmesh.convert(cell.tris)
            state = crease_state(ir)
            result = judge(ir, np.asarray(cell.tris), np.asarray(truth.tris), report)
            if state is None:
                print(f"alpha={alpha:>4} {label:>5} seed={seed}: crease missing")
                continue
            kind_s, dihedral, defect = state
            kept = defect > 1e-5
            print(
                f"alpha={alpha:>4} {label:>5} seed={seed}: "
                f"kept={kept} kind={kind_s:>10} dihedral={dihedral:6.3f}deg "
                f"defect={defect:.3e} (R(1-cos)={expected:.3e}) "
                f"truth={result.truth.max * 1e3:7.3f}um rep={report.max_deviation * 1e3:7.3f}um "
                f"under_report={result.input.max > report.max_deviation + 1e-9}"
            )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angles", default="1,2,2.5,2.9,3,5")
    p.add_argument("--kind", default="noise_normal")
    p.add_argument("--severity", type=float, default=0.02)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--deflection", default="0.01,0.2")
    args = p.parse_args(argv)
    args.angles = [float(a) for a in args.angles.split(",")]
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.deflection = tuple(float(x) for x in args.deflection.split(","))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
