import time
from dataclasses import replace

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Sphere, Torus

from unmesh import _core
from unmesh.ir import Ir, Plane
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.judge import calibration, judge, sample_ir, under_reports
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir

from .cases import MESHER_CHORD_FACTOR, smoke_entries

LIN, ANG = 0.01, 0.2
FINE = (0.001, 0.1)

SMOKE = select(load_manifest(), "smoke")


def planar_only(mesh):
    return all(f.surface == "plane" for f in mesh.faces)


@pytest.mark.parametrize("entry", smoke_entries(curved_slow=True))
def test_oracle_scores_its_own_input(entry):
    shape = generate(entry["family"], entry["seed"]).solid
    mesh = tessellate(shape, LIN, ANG)
    fine = tessellate(shape, *FINE)
    ir = build_oracle_ir(mesh)
    result = judge(ir, mesh, fine, samples_per_mm2=2.0, seed=1)
    if planar_only(mesh):
        assert result.input.max < 1e-6
        assert result.truth.ir_to_mesh.max <= FINE[0] + 1e-9
        assert result.truth.mesh_to_ir.max <= LIN + 1e-9
    else:
        bound = max(LIN * MESHER_CHORD_FACTOR.get(f.surface, 1.0) for f in mesh.faces)
        fine_bound = max(FINE[0] * MESHER_CHORD_FACTOR.get(f.surface, 1.0) for f in fine.faces)
        assert result.input.max <= bound + 1e-9
        assert result.truth.ir_to_mesh.max <= fine_bound + 1e-9
        assert result.truth.mesh_to_ir.max <= LIN + 1e-9


@pytest.mark.parametrize(
    "make",
    [
        lambda: Box(10, 20, 30),
        lambda: Box(10, 10, 10) - Cylinder(2, 20),
    ],
)
def test_planar_corpus_is_exact(make):
    mesh = tessellate(make(), LIN, ANG)
    r = judge(build_oracle_ir(mesh), mesh, mesh)
    assert planar_only(mesh) or r.input.ir_to_mesh.max <= LIN + 1e-9
    assert r.input.mesh_to_ir.max <= LIN + 1e-9


def test_box_is_below_one_micron():
    mesh = tessellate(Box(10, 20, 30), LIN, ANG)
    r = judge(build_oracle_ir(mesh), mesh, mesh)
    assert r.input.max < 1e-6
    assert r.truth.max < 1e-6


@pytest.mark.parametrize(
    "make",
    [lambda: Cylinder(5, 10), lambda: Cone(5, 2, 10), lambda: Sphere(5), lambda: Torus(10, 2)],
)
def test_curved_primitives_report_the_deflection_bound(make):
    mesh = tessellate(make(), LIN, ANG)
    r = judge(build_oracle_ir(mesh), mesh, samples_per_mm2=20.0)
    assert 0.1 * LIN < r.input.ir_to_mesh.max <= LIN + 1e-9


def test_shifted_plane_reports_the_offset():
    mesh = tessellate(Box(20, 20, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    for d in (0.01, 0.2, 1.5):
        shifted = replace(ir, regions=list(ir.regions))
        region = shifted.regions[0]
        plane = region.surface
        origin = tuple(np.array(plane.origin) + d * np.array(plane.normal))
        shifted.regions[0] = replace(region, surface=Plane(origin, plane.normal))
        r = judge(shifted, mesh)
        assert r.input.max == pytest.approx(d, rel=0.05)
        assert r.region_max[0] == pytest.approx(d, rel=0.05)
        assert max(r.region_max[1:]) < 1e-6


def test_calibration_sign():
    mesh = tessellate(Box(20, 20, 20), LIN, ANG)
    r = judge(build_oracle_ir(mesh), mesh, report={"max_deviation": 0.0})
    assert abs(r.calibration) < 1e-6

    ir = build_oracle_ir(mesh)
    plane = ir.regions[0].surface
    origin = tuple(np.array(plane.origin) + 0.3 * np.array(plane.normal))
    ir.regions[0] = replace(ir.regions[0], surface=Plane(origin, plane.normal))
    r = judge(ir, mesh, report=0.1)
    assert r.calibration == pytest.approx(0.1 - 0.3, abs=1e-6)
    assert calibration(0.5, r.input) == pytest.approx(0.2, abs=1e-6)


def test_samples_lie_on_their_regions_and_are_deterministic():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    pts, owner = sample_ir(ir, mesh, samples_per_mm2=3.0, seed=5)
    again, _ = sample_ir(ir, mesh, samples_per_mm2=3.0, seed=5)
    assert np.array_equal(pts, again)
    assert len(pts) > 500
    from unmesh_harness.labels import distance_to_surface

    for face in mesh.faces:
        assert distance_to_surface(face, pts[owner == face.id]).max() < 1e-9


def test_independent_of_the_converter():
    import ast
    from pathlib import Path

    import unmesh_harness.judge as judge_module

    tree = ast.parse(Path(judge_module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    unmesh_imports = {name for name in imported if name.split(".")[0] == "unmesh"}
    assert unmesh_imports <= {"unmesh._core", "unmesh.ir.Ir"}


def noisy_box_ir_and_mesh(sigma, seed):
    import unmesh

    mesh = tessellate(Box(12, 9, 7) - Cylinder(1.5, 20), LIN, ANG)
    rng = np.random.default_rng(seed)
    flat = mesh.tris.reshape(-1, 3)
    keys, inverse = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    noisy = (keys + rng.normal(0, sigma, keys.shape))[inverse.ravel()].reshape(-1, 3, 3)
    return unmesh.convert(noisy), noisy


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_converged_on_noisy_planar_parts(seed):
    result, noisy = noisy_box_ir_and_mesh(0.004, seed)
    coarse = judge(result.ir, noisy, samples_per_mm2=10.0, seed=3)
    dense = judge(result.ir, noisy, samples_per_mm2=1000.0, seed=4)
    assert dense.input.max > 1e-3
    assert abs(coarse.input.max - dense.input.max) <= 0.005 * dense.input.max


def test_converged_on_a_coarse_sphere():
    mesh = tessellate(Sphere(5), 0.1, 0.5)
    ir = build_oracle_ir(mesh)
    coarse = judge(ir, mesh, samples_per_mm2=10.0, seed=3)
    dense = judge(ir, mesh, samples_per_mm2=1000.0, seed=4)
    assert abs(coarse.input.max - dense.input.max) <= 0.005 * dense.input.max


def test_under_report_tolerance():
    mesh = tessellate(Box(20, 20, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    plane = ir.regions[0].surface
    origin = tuple(np.array(plane.origin) + 0.3 * np.array(plane.normal))
    ir.regions[0] = replace(ir.regions[0], surface=Plane(origin, plane.normal))
    r = judge(ir, mesh)
    assert not under_reports(0.3, r.input)
    assert not under_reports(0.3 * (1 - 0.004), r.input)
    assert under_reports(0.3 * (1 - 0.006), r.input)


@pytest.mark.benchmark
def test_judge_ir_end_to_end_on_a_million_triangles():
    import json

    n = 708
    xs = np.linspace(0, 100, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    z = 0.002 * np.sin(gx * 0.3) * np.cos(gy * 0.3)
    verts = np.stack([gx, gy, z], axis=-1)
    a, b, c, d = verts[:-1, :-1], verts[1:, :-1], verts[1:, 1:], verts[:-1, 1:]
    tris = np.concatenate(
        [
            np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
            np.stack([a, c, d], axis=2).reshape(-1, 3, 3),
        ]
    )
    assert len(tris) >= 1_000_000
    ir = Ir.from_dict(
        {
            "ir_version": 0,
            "tolerances": {
                "linear": 0.001,
                "angular_snap_deg": 0.5,
                "tangent_threshold_deg": 3.0,
                "vertex_merge": 1e-6,
            },
            "source": {"triangle_count": len(tris), "vertex_count": (n + 1) ** 2},
            "shells": [{"closed": True, "role": "outer", "parent": None, "regions": [0]}],
            "regions": [
                {
                    "id": 0,
                    "surface": {
                        "type": "plane",
                        "origin": [0.0, 0.0, 0.0],
                        "normal": [0.0, 0.0, 1.0],
                    },
                    "triangles": list(range(len(tris))),
                    "residual": {"rms": 0.0, "max": 0.002},
                }
            ],
            "adjacencies": [],
            "vertices": [],
        }
    )
    text = ir.dumps()
    assert json.loads(text)["source"]["triangle_count"] == len(tris)
    t0 = time.perf_counter()
    raw = _core.judge_ir(text, tris, None, 100.0, 0, False)
    elapsed = time.perf_counter() - t0
    assert raw["input"]["ir_to_mesh"]["count"] >= 1_000_000
    assert raw["input"]["mesh_to_ir"]["count"] >= 1_000_000
    print(f"judge_ir end to end: {elapsed:.2f}s")
    assert elapsed < 2.0, elapsed
