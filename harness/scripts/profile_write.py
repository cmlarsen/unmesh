#!/usr/bin/env -S uv run --locked python
"""Profile and compare the STEP write of noisy through_bore (issue #135).

Builds the part the way the acceptance grid does (ground-truth solid ->
(0.01, 0.2) tessellation -> ``noise_normal`` 0.02), converts it with
``unmesh.convert`` and times the STEP write::

    uv run --locked harness/scripts/profile_write.py --seeds 0 1 2 --gt-seed 0

Options:
  --family F       ground-truth family (default through_bore)
  --clean          write the clean tessellation, without the noise chain
  --sections       time the writer's own entry points with CPU timers (robust to
                   a loaded machine that makes wall time noisy)
  --no-verify      skip the OCCT read-back, so the timing is only solid building
  --dump DIR       write each STEP plus a JSON summary (face count, surface
                   types, volume, deviations) for comparing two checkouts
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path

DEFLECTION = (0.01, 0.2)
NOISE = ("noise_normal", 0.02)


def build_mesh(family: str, gt_seed: int, noise_seed: int | None, out_dir: Path):
    import numpy as np

    from unmesh import read_stl
    from unmesh_harness import degrade
    from unmesh_harness.groundtruth import generate
    from unmesh_harness.labels import tessellate

    clean = tessellate(generate(family, gt_seed).solid, *DEFLECTION)
    if noise_seed is None:
        mesh = clean
    else:
        mesh = degrade.chain(clean, [NOISE], noise_seed)
    path = out_dir / f"input-{family}-{gt_seed}-{noise_seed}.stl"
    mesh.write_stl(path)
    return np.asarray(read_stl(path), dtype=np.float64)


class _Sections:
    def __init__(self):
        self.times: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, obj, name: str, label: str):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            start = time.process_time()
            try:
                return original(*args, **kwargs)
            finally:
                self.times[label] = self.times.get(label, 0.0) + (time.process_time() - start)
                self.calls[label] = self.calls.get(label, 0) + 1

        setattr(obj, name, timed)

    def report(self):
        print("[sections] CPU time by writer entry point")
        for key in sorted(self.times, key=lambda k: -self.times[k]):
            print(
                f"[sections]   {self.times[key]:8.3f} s  {self.calls[key]:6d} calls  {key}",
                flush=True,
            )


def install_sections(sections: _Sections):
    from unmesh import step
    from unmesh._writer import curved, occ

    for name in (
        "_analytic",
        "_finish_group",
        "_check",
        "_check_patches",
        "_check_input_volume",
        "_check_face_fluxes",
    ):
        sections.wrap(step, name, f"step.{name}")
    for name in (
        "write_step",
        "fix_shape",
        "make_oriented_solid",
        "face_tolerances",
        "face_fluxes",
        "volume_of",
        "area_of",
        "is_valid",
    ):
        sections.wrap(occ, name, f"occ.{name}")
    for name in ("build_shell", "refine_vertices"):
        sections.wrap(curved, name, f"curved.{name}")
    for name in (
        "collect",
        "intersect",
        "place_poles",
        "settle_vertices",
        "choose_frames",
        "make_edges",
        "facets_seam",
        "curved_edge",
        "straight_edges",
        "check_side",
        "check_windings",
        "face",
        "facets_faces",
        "plane_face",
        "curved_face",
        "wire",
    ):
        sections.wrap(curved._Builder, name, f"_Builder.{name}")
    for name in ("pcurve_error", "sampled_pcurve", "corrected_pcurve"):
        sections.wrap(curved, name, f"curved.{name}")
    sections.wrap(curved._UVLoop, "__init__", "_UVLoop.__init__")


def step_summary(path: Path) -> dict:
    from collections import Counter

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from unmesh._writer import occ

    shape = occ.read_step(path)
    types: Counter[str] = Counter()
    count = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        count += 1
        adaptor = BRepAdaptor_Surface(TopoDS.Face_s(ex.Current()))
        types[str(adaptor.GetType())] += 1
        ex.Next()
    return {"step_faces": count, "step_surface_types": dict(sorted(types.items()))}


def report_summary(wr) -> dict:
    volumes = [s.volume for s in wr.shells if s.volume is not None]
    return {
        "valid": wr.valid,
        "fallback": wr.fallback,
        "faceted_regions": wr.faceted_regions,
        "faceted_faces": wr.faceted_faces,
        "faces": [
            {"region": f.region, "surface": f.surface_type, "tolerance": f.max_shape_tolerance}
            for f in sorted(wr.faces, key=lambda f: f.region)
        ],
        "max_vertex_displacement": wr.max_vertex_displacement,
        "max_boundary_deviation": wr.max_boundary_deviation,
        "max_shape_tolerance": wr.max_shape_tolerance,
        "volume": sum(volumes) if volumes else None,
        "issues": list(wr.issues),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="through_bore")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gt-seed", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args(argv)

    import unmesh
    from unmesh import step

    sections = _Sections() if args.sections else None
    if sections is not None:
        install_sections(sections)

    out_dir = Path(tempfile.mkdtemp(prefix="profile-write-"))
    dumps = []
    for seed in args.seeds:
        gt_seed = seed if args.gt_seed is None else args.gt_seed
        flavour = "clean" if args.clean else f"noise{seed}"
        print(f"[profile] {args.family}-{gt_seed} {flavour}: building mesh", flush=True)
        tris = build_mesh(args.family, gt_seed, None if args.clean else seed, out_dir)
        result = unmesh.convert(tris)
        ir = result.ir

        profile = cProfile.Profile()
        path = out_dir / f"{args.family}-{gt_seed}-{flavour}.step"
        wall = time.perf_counter()
        cpu = time.process_time()
        if not args.no_profile:
            profile.enable()
        wr = step.write(ir, path, mesh=tris, verify=not args.no_verify)
        if not args.no_profile:
            profile.disable()
        wall_s = time.perf_counter() - wall
        cpu_s = time.process_time() - cpu
        size = path.stat().st_size if path.exists() else 0
        summary = report_summary(wr)
        dumped = {
            "seed": seed,
            "gt_seed": gt_seed,
            "clean": args.clean,
            "write_s": wr.timings.get("write_s"),
            "readback_s": wr.timings.get("readback_s"),
            "wall_s": wall_s,
            "cpu_s": cpu_s,
            "bytes": size,
            "write": summary,
        }
        if args.dump is not None and path.exists():
            args.dump.mkdir(parents=True, exist_ok=True)
            target = args.dump / path.name
            target.write_bytes(path.read_bytes())
            dumped["step"] = step_summary(path)
            dumped["step_path"] = str(target)
        dumps.append(dumped)
        print(
            f"[profile] {args.family}-{gt_seed} {flavour} write: {wall_s:.2f} s wall,"
            f" {cpu_s:.2f} s cpu, write_s={dumped['write_s']} readback_s={dumped['readback_s']}"
            f" {size / 1024:.0f} KiB valid={summary['valid']} fallback={summary['fallback']}"
            f" faceted_regions={summary['faceted_regions']}"
            f" faceted_faces={summary['faceted_faces']}"
            f" step={dumped.get('step')}",
            flush=True,
        )
        for issue in summary["issues"]:
            print(f"[profile]   issue: {issue}", flush=True)
        if not args.no_profile:
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(args.top)
            print(stream.getvalue(), flush=True)
        for s in wr.seams:
            print(
                f"[profile]   seam {s.regions} {s.surface_type}: points={s.points}"
                f" max_gap={s.max_gap:.3g} chord_gap={s.chord_gap:.3g} inserted={s.inserted}",
                flush=True,
            )
    if sections is not None:
        sections.report()
    if args.dump is not None:
        tail = args.dump / f"{args.family}-{'clean' if args.clean else 'noisy'}.json"
        tail.write_text(json.dumps(dumps, indent=2) + "\n")
        print(f"[profile] wrote {tail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
