import time
from dataclasses import replace

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Sphere, Torus

from unmesh import _core
from unmesh.ir import Plane
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.judge import calibration, judge, sample_ir
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir

LIN, ANG = 0.01, 0.2
FINE = (0.001, 0.1)

SMOKE = select(load_manifest(), "smoke")


def planar_only(mesh):
    return all(f.surface == "plane" for f in mesh.faces)


@pytest.mark.parametrize("entry", SMOKE, ids=lambda e: e["id"])
def test_oracle_scores_its_own_input(entry):
    shape = generate(entry["family"], entry["seed"]).solid
    mesh = tessellate(shape, LIN, ANG)
    ir = build_oracle_ir(mesh)
    result = judge(ir, mesh, tessellate(shape, *FINE), samples_per_mm2=2.0, seed=1)
    if planar_only(mesh):
        assert result.input.max < 1e-6
    else:
        assert result.input.max <= LIN + 1e-9
    assert result.truth.ir_to_mesh.max <= FINE[0] + 1e-9
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


@pytest.mark.benchmark
def test_one_million_samples_against_a_million_triangles():
    n = 708
    xs = np.linspace(0, 100, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    z = np.sin(gx * 0.3) * np.cos(gy * 0.3)
    verts = np.stack([gx, gy, z], axis=-1)
    a, b, c, d = verts[:-1, :-1], verts[1:, :-1], verts[1:, 1:], verts[:-1, 1:]
    tris = np.concatenate(
        [
            np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
            np.stack([a, c, d], axis=2).reshape(-1, 3, 3),
        ]
    )
    assert len(tris) >= 1_000_000
    rng = np.random.default_rng(0)
    pts = tris[rng.integers(0, len(tris), 1_000_000)].mean(axis=1) + rng.normal(
        0, 0.01, (1_000_000, 3)
    )
    t0 = time.perf_counter()
    d = _core.mesh_distances(tris, pts)
    elapsed = time.perf_counter() - t0
    assert len(d) == 1_000_000
    assert elapsed < 2.0, elapsed
