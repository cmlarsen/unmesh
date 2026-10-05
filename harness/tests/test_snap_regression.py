import math

import numpy as np
import pytest
from build123d import Box, Cylinder, Keep, Plane, Pos, split

import unmesh
from unmesh_harness import degrade
from unmesh_harness.groundtruth import generate
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
    assert report.max_deviation <= 1e-3, report.max_deviation


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
    for region in ir.regions:
        if region.surface.type != "plane":
            continue
        centroid = tris[np.array(region.triangles)].mean(axis=(0, 1))
        normal = np.array(region.surface.normal, float)
        if centroid[2] > 63 and abs(centroid[0]) < width and normal[2] > 0.9:
            return math.degrees(math.acos(min(1.0, abs(normal[2]) / np.linalg.norm(normal))))
    return None


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
