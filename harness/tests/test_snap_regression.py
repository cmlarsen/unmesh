import math

import numpy as np
import pytest
from build123d import (
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    CenterArc,
    Cylinder,
    Keep,
    Line,
    Plane,
    Polyline,
    Pos,
    extrude,
    fillet,
    make_face,
    split,
)

import unmesh
from unmesh_harness import degrade
from unmesh_harness.groundtruth import generate
from unmesh_harness.judge import judge
from unmesh_harness.labels import tessellate
from unmesh_harness.metrics.recovery import score_recovery

LIN, ANG = 0.01, 0.2
REFINE_SEED = 0


@pytest.mark.parametrize("part_id", ["circular_fillet-0000", "round_slot_through-0000"])
def test_clean_curved_part_recovers_every_plane_face(part_id):
    family, seed = part_id.rsplit("-", 1)
    mesh = tessellate(generate(family, int(seed)).solid, LIN, ANG)
    ir, report = unmesh.convert(mesh.tris)
    result = score_recovery(mesh, mesh.face_id, ir)
    lost = [
        d["face"] for d in result["faces_detail"] if d["type"] == "plane" and not d["recovered"]
    ]
    assert lost == [], (part_id, lost)
    measured = judge(ir, np.asarray(mesh.tris), None, report).input.max
    assert measured <= report.max_deviation <= 1.01 * measured + 1e-9, (
        report.max_deviation,
        measured,
    )


def tilted_boss(width, tilt_deg):
    base = Pos(0, 0, 30) * Cylinder(30, 60)
    boss = Pos(0, 0, 62.5) * Box(width, 20, 5)
    solid = base + boss
    plane = Plane(
        origin=(0, 0, 64),
        z_dir=(math.sin(math.radians(tilt_deg)), 0, math.cos(math.radians(tilt_deg))),
    )
    return split(solid, bisect_by=plane, keep=Keep.BOTTOM)


def top_tilt_deg(ir, tris, width):
    best = None
    for region in ir.regions:
        if region.surface.type != "plane":
            continue
        centroid = tris[np.array(region.triangles)].mean(axis=(0, 1))
        normal = np.array(region.surface.normal, float)
        if centroid[2] > 63 and abs(centroid[0]) < width and normal[2] > 0.9:
            tilt = math.degrees(math.acos(min(1.0, abs(normal[2]) / np.linalg.norm(normal))))
            if best is None or len(region.triangles) > best[1]:
                best = (tilt, len(region.triangles))
    return None if best is None else best[0]


@pytest.mark.parametrize("width,tilt_deg", [(1.0, 0.8), (1.0, 1.5), (1.0, 2.5), (3.0, 0.8)])
def test_deliberate_tilt_on_curved_part_stays_unsnapped(width, tilt_deg):
    mesh = tessellate(tilted_boss(width, tilt_deg), LIN, ANG)
    ir, _ = unmesh.convert(mesh.tris)
    measured = top_tilt_deg(ir, np.asarray(mesh.tris), width)
    assert measured is not None, "tilted top face missing from the IR"
    assert measured > 0.5, f"tilt {tilt_deg} deg was snapped to the axis"
    assert measured == pytest.approx(tilt_deg, abs=0.3), measured


def test_clean_curved_part_recovers_planes_after_refine():
    mesh = tessellate(generate("circular_fillet", 0).solid, LIN, ANG)
    refined = degrade.apply("refine", mesh, 1.0, REFINE_SEED)
    ir, report = unmesh.convert(refined.tris)
    result = score_recovery(refined, refined.face_id, ir)
    lost = [
        d["face"] for d in result["faces_detail"] if d["type"] == "plane" and not d["recovered"]
    ]
    assert lost == [], ("circular_fillet-0000", lost)
    assert report.max_deviation <= 0.02, report.max_deviation


@pytest.mark.parametrize("width,tilt_deg", [(1.0, 0.8), (1.0, 1.5), (1.0, 2.5), (3.0, 0.8)])
def test_deliberate_tilt_on_curved_part_stays_unsnapped_after_refine(width, tilt_deg):
    mesh = tessellate(tilted_boss(width, tilt_deg), LIN, ANG)
    refined = degrade.apply("refine", mesh, 1.0, REFINE_SEED)
    ir, _ = unmesh.convert(refined.tris)
    measured = top_tilt_deg(ir, np.asarray(refined.tris), width)
    assert measured is not None, "tilted top face missing from the IR"
    assert measured > 0.5, f"tilt {tilt_deg} deg was snapped to the axis"
    assert measured == pytest.approx(tilt_deg, abs=0.05), measured


def lost_plane_faces(mesh, ir):
    result = score_recovery(mesh, mesh.face_id, ir, degrade.to_original(mesh))
    masked = {d["face"] for d in result["masked_faces"]}
    return [
        d["face"]
        for d in result["faces_detail"]
        if d["type"] == "plane" and d["face"] not in masked and not d["recovered"]
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("width,tilt_deg", [(1.0, 0.8), (1.0, 1.5), (3.0, 0.8)])
def test_deliberate_tilt_survives_refine_and_light_noise(width, tilt_deg, seed):
    mesh = tessellate(tilted_boss(width, tilt_deg), LIN, ANG)
    mesh = degrade.chain(mesh, [("refine", 1.0), ("noise_isotropic", 0.002)], seed)
    ir, report = unmesh.convert(mesh.tris)
    measured = top_tilt_deg(ir, np.asarray(mesh.tris), width)
    assert measured is not None, "tilted top face missing from the IR"
    assert measured == pytest.approx(tilt_deg, abs=0.05), (
        measured,
        ir.tolerances.linear,
        report.max_deviation,
    )


def test_exact_curved_strips_do_not_floor_the_tolerance_under_plane_noise():
    mesh = tessellate(generate("blind_bore", 0).solid, LIN, ANG)
    mesh = degrade.chain(mesh, [("refine", 0.15), ("noise_off_plane", 0.1)], 0)
    ir, _ = unmesh.convert(mesh.tris)
    assert lost_plane_faces(mesh, ir) == [], ir.tolerances.linear


@pytest.mark.parametrize("part", [0, 1, 2])
@pytest.mark.parametrize("severity", [0.02, 0.1])
def test_tolerance_tracks_isotropic_noise_on_curved_part(severity, part):
    mesh = tessellate(generate("circular_fillet", part).solid, LIN, ANG)
    mesh = degrade.chain(mesh, [("noise_isotropic", severity)], 0)
    ir, _ = unmesh.convert(mesh.tris)
    std = severity * 0.05 / math.sqrt(5.0)
    assert 4.0 * std <= ir.tolerances.linear <= 13.0 * std, ir.tolerances.linear / std


@pytest.mark.parametrize("part", [0, 1, 2])
def test_curved_part_keeps_every_plane_under_1um_noise(part):
    mesh = tessellate(generate("circular_fillet", part).solid, LIN, ANG)
    mesh = degrade.chain(mesh, [("noise_isotropic", 0.02)], 0)
    ir, _ = unmesh.convert(mesh.tris)
    assert lost_plane_faces(mesh, ir) == [], ir.tolerances.linear


def crease_part(alpha_deg, radius=2.0, width=20.0, height=10.0, depth=8.0):
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


def fillet_plane_defect(ir):
    for adj in ir.adjacencies:
        a, b = adj.regions
        sa, sb = ir.regions[a].surface, ir.regions[b].surface
        if {sa.type, sb.type} != {"plane", "cylinder"}:
            continue
        plane = sa if sa.type == "plane" else sb
        cyl = sb if sa.type == "plane" else sa
        if float(np.array(plane.normal)[2]) < 0.9:
            continue
        n = np.array(plane.normal)
        offset = float(np.dot(n, np.array(plane.origin)))
        sign = 1.0 if cyl.orientation == "same" else -1.0
        return abs(float(np.dot(n, np.array(cyl.origin))) + sign * cyl.radius - offset)
    return None


@pytest.mark.parametrize("kind", ["noise_normal", "noise_isotropic"])
@pytest.mark.parametrize("alpha_deg", [1.0, 2.0, 2.5, 2.9, 3.0, 5.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_crease_beside_a_fillet_is_kept_under_1um_noise(alpha_deg, kind, seed):
    mesh = tessellate(crease_part(alpha_deg), LIN, ANG)
    cell = degrade.chain(mesh, [(kind, 0.02)], seed)
    ir, report = unmesh.convert(cell.tris)
    defect = fillet_plane_defect(ir)
    expect = 2.0 * (1.0 - math.cos(math.radians(alpha_deg)))
    assert defect is not None, "the fillet-plane crease is missing from the IR"
    assert defect > 1e-5, (defect, expect, ir.tolerances.linear)
    measured = judge(ir, np.asarray(cell.tris), None, report).input.max
    assert report.max_deviation >= measured - 1e-9 - 0.005 * measured


def test_crease_snap_decision_is_translation_invariant():
    mesh = tessellate(crease_part(2.9), LIN, ANG)
    cell = degrade.chain(mesh, [("noise_isotropic", 0.02)], 0)
    shift = np.array([1000.0, -2000.0, 500.0])
    base, base_report = unmesh.convert(cell.tris)
    moved, moved_report = unmesh.convert(np.asarray(cell.tris) + shift)
    assert len(base.regions) == len(moved.regions)
    assert fillet_plane_defect(base) == pytest.approx(fillet_plane_defect(moved), abs=1e-6)
    assert fillet_plane_defect(base) > 1e-5
    assert abs(base_report.max_deviation - moved_report.max_deviation) < 1e-6


def zigzag_part(segments, amp=20.0, thick=30.0, depth=10.0, radius=3.0):
    half = max(200.0, 6.0 * segments)
    pts = [(-half, 0.0)]
    dx = 2 * half / segments
    for i in range(1, segments):
        pts.append((-half + i * dx, amp if i % 2 else 0.0))
    pts.append((half, 0.0))
    with BuildPart() as bp:
        with BuildSketch(Plane.XZ) as sk:
            with BuildLine():
                Polyline(*pts)
                Line(pts[-1], (half, -thick))
                Line((half, -thick), (-half, -thick))
                Line((-half, -thick), pts[0])
            make_face()
            fillet(sk.vertices(), radius)
        extrude(amount=depth)
    return bp.part


def test_large_tangent_cluster_is_reported():
    mesh = tessellate(zigzag_part(340), 0.05, 0.3)
    cell = degrade.chain(mesh, [("noise_normal", 0.02)], 0)
    _, report = unmesh.convert(cell.tris)
    assert "tangency_cluster_too_large" in {w.code for w in report.warnings}
