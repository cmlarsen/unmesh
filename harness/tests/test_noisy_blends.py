import math

import numpy as np
import pytest

import unmesh
from unmesh_harness import degrade
from unmesh_harness.groundtruth import generate
from unmesh_harness.judge import judge
from unmesh_harness.labels import tessellate

LIN, ANG = 0.01, 0.2
MAX_AMPLITUDE = 0.05
NOISE = [("noise_normal", 0.02)]
FAMILIES = ("corner_fillet", "bore_chamfer")
CASES = [(f, p, s) for f in FAMILIES for p in range(3) for s in range(3)]
FAST = [(f, 0, 0) for f in FAMILIES]


def noisy(family, part, seed):
    mesh = tessellate(generate(family, part).solid, LIN, ANG)
    return degrade.chain(mesh, NOISE, seed)


def check_regions(family, part, seed):
    mesh = noisy(family, part, seed)
    tris = np.asarray(mesh.tris)
    ir, report = unmesh.convert(tris)
    truth = len(set(np.asarray(mesh.face_id).tolist()))
    assert len(ir.regions) <= 1.5 * truth, (len(ir.regions), truth, report.region_counts)
    measured = judge(ir, tris, None, report).input.max
    assert measured <= report.max_deviation, (measured, report.max_deviation)
    return ir, tris


@pytest.mark.parametrize("family,part,seed", FAST)
def test_noisy_blends_keep_their_faces(family, part, seed):
    check_regions(family, part, seed)


@pytest.mark.slow
@pytest.mark.parametrize("family,part,seed", [c for c in CASES if c not in FAST])
def test_noisy_blends_keep_their_faces_slow(family, part, seed):
    check_regions(family, part, seed)


def draft_block(deg):
    from build123d import Plane, Polyline, extrude, make_face

    t = math.tan(math.radians(deg))
    profile = Polyline((0, 0), (40, 0), (40, 10), (20, 10), (0, 10 - 20 * t), close=True)
    return extrude(make_face(Plane.XZ * profile), amount=-30)


def draft_boss(deg, r=8.0, hc=4.0, hk=6.0):
    from build123d import Align, Cone, Cylinder, Pos

    t = math.tan(math.radians(deg))
    base = Cylinder(r, hc, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top = Pos(0, 0, hc) * Cone(r, r - hk * t, hk, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return base + top


@pytest.mark.slow
def test_noisy_shallow_draft_never_emits_a_wrong_cone():
    """A 0.2 degree boss draft under 1 um noise (seed 4) is below the noise
    resolution: growth must not lock in a wrong cone. Main kept a 0.0396 degree
    cone over 184 triangles here; the gate drops it."""
    part = draft_boss(0.2, r=12.0, hc=4.0, hk=12.0)
    mesh = tessellate(part, LIN, ANG)
    cell = degrade.chain(mesh, [("noise_normal", 0.02)], 4)
    ir, _ = unmesh.convert(np.asarray(cell.tris))
    for region in ir.regions:
        surface = region.to_dict().get("surface") or {}
        if surface.get("type") == "cone":
            angle = abs(math.degrees(surface["half_angle"]))
            assert angle >= 0.1, (angle, surface)


def draft_bore(deg, r=8.0, hc=4.0, hk=6.0):
    from build123d import Align, Box, Cone, Cylinder, Pos

    t = math.tan(math.radians(deg))
    hole = Cylinder(r, hc, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole += Pos(0, 0, hc) * Cone(r, r + hk * t, hk, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return Box(4 * r, 4 * r, hc + hk, align=(Align.CENTER, Align.CENTER, Align.MIN)) - hole


def draft_cone_angles(shape):
    mesh = tessellate(shape, LIN, ANG)
    ir, _ = unmesh.convert(np.asarray(mesh.tris))
    return [
        abs(math.degrees(surface["half_angle"]))
        for region in ir.regions
        if (surface := region.to_dict().get("surface") or {}).get("type") == "cone"
    ]


@pytest.mark.parametrize("shape", ["boss", "bore"])
def test_clean_shallow_draft_is_an_exact_cone(shape):
    """A clean 0.2 degree boss or bore draft is an exact cone, not facets: the
    individual facets all lie on a cylinder within tolerance, so the cone is
    only found by regressing the radius on the axis over enough of the face."""
    part = draft_boss(0.2) if shape == "boss" else draft_bore(0.2)
    angles = draft_cone_angles(part)
    assert any(abs(a - 0.2) <= 1e-6 for a in angles), angles


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["boss", "bore"])
def test_clean_shallow_draft_sweep(shape):
    """No clean draft in 0.1-0.5 degrees in steps of 0.01 falls through."""
    for i in range(41):
        delta = round(0.1 + 0.01 * i, 2)
        part = draft_boss(delta) if shape == "boss" else draft_bore(delta)
        angles = draft_cone_angles(part)
        assert any(abs(a - delta) <= 1e-6 for a in angles), (shape, delta, angles)


def test_clean_shallow_draft_keeps_floor_and_regions():
    tris = np.asarray(tessellate(draft_block(0.2), LIN, ANG).tris)
    verts = tris.reshape(-1, 3)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    floor = max(1e-6 * diag, 5e-7 * float(np.abs(verts).max()))
    ir, _ = unmesh.convert(tris)
    assert len(ir.regions) == 7, len(ir.regions)
    assert ir.tolerances.linear == floor, (ir.tolerances.linear, floor)


@pytest.mark.parametrize("seed", range(3))
def test_noisy_shallow_draft_keeps_floor_and_regions(seed):
    mesh = tessellate(draft_block(0.2), LIN, ANG)
    deg = degrade.chain(mesh, [("noise_normal", 0.02)], seed)
    ir, _ = unmesh.convert(np.asarray(deg.tris))
    truth = 2.0 * 0.02 * MAX_AMPLITUDE / math.sqrt(3.0)
    assert ir.tolerances.linear / 5.0 <= truth, (ir.tolerances.linear / 5.0, truth)
    assert len(ir.regions) == 7, len(ir.regions)


@pytest.mark.parametrize("seed", range(3))
def test_coarse_tangent_blend_isotropic_noise_within_two(seed):
    mesh = tessellate(generate("straight_fillet", 0).solid, 0.3, 1.0)
    deg = degrade.chain(mesh, [("noise_isotropic", 0.02)], seed)
    ir, _ = unmesh.convert(np.asarray(deg.tris))
    truth = 2.0 * 0.02 * MAX_AMPLITUDE / math.sqrt(5.0)
    assert ir.tolerances.linear / 5.0 <= truth, (ir.tolerances.linear / 5.0, truth)


@pytest.mark.slow
@pytest.mark.parametrize("part,seed", [(p, s) for p in range(3) for s in range(3)])
def test_noisy_bore_chamfer_writes_analytic(tmp_path, part, seed):
    pytest.importorskip("OCP")
    import unmesh.step

    ir, tris = check_regions("bore_chamfer", part, seed)
    written = unmesh.step.write(ir, tmp_path / "part.step", mesh=tris)
    assert written.valid and written.fallback is None, written.fallback_reason


@pytest.mark.slow
def test_noisy_through_bore_seed_11_writes_analytic(tmp_path):
    pytest.importorskip("OCP")
    import unmesh.step

    ir, tris = check_regions("through_bore", 0, 11)
    written = unmesh.step.write(ir, tmp_path / "part.step", mesh=tris)
    assert written.valid and written.fallback is None, written.fallback_reason
