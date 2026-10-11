"""Probe cylinder-vs-cone on shallow-draft bosses and bores (issue #137).

Builds a boss and a bore whose side is a cylinder plus a coaxial draft cone, at
draft angles 0.1-2 degrees, then converts each clean and under ~1 um noise for
several seeds.  For every run it reports the fitted kind and half-angle of the
region covering the drafted face, and the harness face-recovery F1.

The printed table is the acceptance evidence for the fix; the truth drafts are
the designed ones.  ``--shape`` selects boss/bore and ``--json`` writes the rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from build123d import Align, Box, Cone, Cylinder, Pos

import unmesh
from unmesh_harness import degrade
from unmesh_harness.degrade.core import to_original
from unmesh_harness.labels import tessellate
from unmesh_harness.runner.score import face_recovery


def boss(delta_deg: float, r: float, hc: float, hk: float):
    d = math.radians(delta_deg)
    rt = r - hk * math.tan(d)
    base = Cylinder(r, hc, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top = Pos(0, 0, hc) * Cone(r, rt, hk, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return base + top


def bore(delta_deg: float, r: float, hc: float, hk: float):
    d = math.radians(delta_deg)
    rt = r + hk * math.tan(d)
    hole = Cylinder(r, hc, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole += Pos(0, 0, hc) * Cone(r, rt, hk, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return Box(4 * r, 4 * r, hc + hk, align=(Align.CENTER, Align.CENTER, Align.MIN)) - hole


def drafted_face(mesh) -> int:
    best, err = None, math.inf
    for f in mesh.faces:
        if f.surface == "cone":
            e = abs(f.params["half_angle"])
            if e < err:
                best, err = f.id, e
    if best is None:
        raise RuntimeError("part has no conical face")
    return best


def covering_region(ir, face_id: np.ndarray, fid: int):
    sel = np.flatnonzero(face_id == fid)
    best, cover = None, 0
    for r in ir.regions:
        tris = np.asarray(r.triangles, dtype=np.int64)
        if tris.size == 0:
            continue
        n = int(np.isin(tris, sel).sum())
        if n > cover:
            best, cover = r, n
    return best, cover, int(sel.size)


def fitted_draft(region) -> tuple[str, float | None]:
    if region is None:
        return "-", None
    s = region.to_dict().get("surface") or {}
    if s.get("type") == "cone" and s.get("half_angle") is not None:
        return "cone", math.degrees(float(s["half_angle"]))
    if s.get("type") == "cylinder":
        return "cylinder", 0.0
    return str(s.get("type", "-")), None


def run(args):
    kinds = [(k, args.severity) if k != "clean" else (None, 0.0) for k in args.kinds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    lin, ang = (float(x) for x in args.deflection.split(","))
    make = boss if args.shape == "boss" else bore
    rows = []
    for delta in (float(x) for x in args.deltas.split(",")):
        part = make(delta, args.r, args.hc, args.hk)
        mesh = tessellate(part, lin, ang)
        fid = drafted_face(mesh)
        for kname, sev in kinds:
            for seed in seeds:
                cell = mesh if kname is None else degrade.chain(mesh, [(kname, sev)], seed)
                ir, _ = unmesh.convert(np.asarray(cell.tris))
                reg, cover, total = covering_region(ir, np.asarray(cell.face_id), fid)
                kind, draft = fitted_draft(reg)
                f1 = face_recovery(mesh, np.asarray(cell.face_id), ir, to_original(cell)).get("f1")
                rows.append(
                    {
                        "shape": args.shape,
                        "delta": delta,
                        "noise": kname or "clean",
                        "seed": seed,
                        "fitted": kind,
                        "draft": draft,
                        "cover": cover,
                        "total": total,
                        "f1": f1,
                    }
                )
    print("| shape | delta | noise | seed | fitted | draft_deg | cover | F1 |", flush=True)
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        d = "-" if r["draft"] is None else f"{r['draft']:.4f}"
        f1 = "-" if r["f1"] is None else f"{r['f1']:.3f}"
        print(
            f"| {r['shape']} | {r['delta']} | {r['noise']} | {r['seed']} | "
            f"{r['fitted']} | {d} | {r['cover']}/{r['total']} | {f1} |",
            flush=True,
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deltas", default="0.1,0.2,0.5,1,2")
    p.add_argument("--shape", default="boss")
    p.add_argument("--kinds", default="clean,noise_normal,noise_isotropic")
    p.add_argument("--severity", type=float, default=0.02)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--deflection", default="0.01,0.2")
    p.add_argument("--r", type=float, default=8.0)
    p.add_argument("--hc", type=float, default=4.0)
    p.add_argument("--hk", type=float, default=6.0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
