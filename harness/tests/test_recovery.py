import dataclasses
import json

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Sphere, Torus, fillet

from unmesh.ir import Region
from unmesh_harness.degrade import apply, to_original
from unmesh_harness.labels import tessellate
from unmesh_harness.metrics.recovery import aggregate, score_recovery
from unmesh_harness.oracle import build_oracle_ir
from unmesh_harness.runner.score import face_recovery

LIN, ANG = 0.01, 0.2

PRIMITIVES = {
    "box": lambda: Box(10, 10, 10),
    "cylinder": lambda: Cylinder(5, 10),
    "cone": lambda: Cone(5, 2, 10),
    "sphere": lambda: Sphere(5),
    "torus": lambda: Torus(10, 2),
    "drilled_box": lambda: Box(10, 10, 10) - Cylinder(2, 20),
    "filleted_box": lambda: fillet(Box(20, 20, 20).edges(), 3),
}


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_oracle_scores_f1_one(name):
    mesh = tessellate(PRIMITIVES[name](), LIN, ANG)
    result = score_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    assert result["matched"] == len(mesh.faces)
    assert result["f1"] == 1.0
    assert result["unsupported_faces"] == 0
    assert all(t["f1"] == 1.0 for t in result["per_type"].values())
    assert result["segmentation"]["over_segmented"] == 0
    assert result["segmentation"]["under_segmented"] == 0
    assert result["edge_error"]["missing"] == 0
    assert result["edge_error"]["max"] == pytest.approx(0.0)
    json.dumps(result)


def test_planar_legacy_keys_unchanged():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    result = face_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    for key in ("faces", "regions", "matched", "precision", "recall", "f1", "unsupported_faces"):
        assert key in result
    assert (result["faces"], result["matched"], result["f1"]) == (6, 6, 1.0)


def test_bore_radius_off_by_two_percent_is_not_recovered():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    bore = next(r for r in ir.regions if r.surface.type == "cylinder")
    assert bore.surface.orientation == "reversed"
    wrong = dataclasses.replace(bore.surface, radius=bore.surface.radius * 1.02)
    regions = [dataclasses.replace(r, surface=wrong) if r.id == bore.id else r for r in ir.regions]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    assert result["f1"] < 1.0
    detail = next(d for d in result["faces_detail"] if d["type"] == "cylinder")
    assert detail["recovered"] is False
    assert detail["errors"]["radius_rel"] == pytest.approx(0.02)


def test_cone_half_angle_off_by_half_a_degree_is_not_recovered():
    mesh = tessellate(Cone(5, 2, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    side = next(r for r in ir.regions if r.surface.type == "cone")
    wrong = dataclasses.replace(side.surface, half_angle=side.surface.half_angle + np.deg2rad(0.5))
    regions = [dataclasses.replace(r, surface=wrong) if r.id == side.id else r for r in ir.regions]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    detail = next(d for d in result["faces_detail"] if d["type"] == "cone")
    assert detail["errors"]["half_angle_deg"] == pytest.approx(0.5)


def test_split_region_counts_as_over_segmentation():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    first, rest = ir.regions[0], ir.regions[1:]
    half = len(first.triangles) // 2
    regions = [
        Region(0, first.surface, first.triangles[:half], first.residual),
        Region(1, first.surface, first.triangles[half:], first.residual),
        *[dataclasses.replace(r, id=r.id + 1) for r in rest],
    ]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["segmentation"]["over_segmented"] == 1
    assert result["segmentation"]["under_segmented"] == 0
    assert result["matched"] == len(mesh.faces) - 1


def test_merged_regions_count_as_under_segmentation():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    first, second, rest = ir.regions[0], ir.regions[1], ir.regions[2:]
    merged = Region(0, first.surface, first.triangles + second.triangles, first.residual)
    regions = [merged, *[dataclasses.replace(r, id=r.id - 1) for r in rest]]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["segmentation"]["under_segmented"] == 1
    assert result["segmentation"]["over_segmented"] == 0


def test_rotated_frame_compares_in_the_original_frame():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    moved = apply("rotation", mesh, 0.5, 7)
    ir = build_oracle_ir(moved)
    result = score_recovery(mesh, moved.face_id, ir, to_original(moved))
    assert result["f1"] == 1.0
    assert result["edge_error"]["max"] == pytest.approx(0.0)


def test_aggregate_groups_by_family_and_strata():
    records = [
        {
            "status": "ok",
            "family": "plate_pockets",
            "strata": {"category": "planar"},
            "faces": 10,
            "regions": 10,
            "matched": 10,
        },
        {
            "status": "ok",
            "family": "plate_pockets",
            "strata": {"category": "planar"},
            "faces": 10,
            "regions": 10,
            "matched": 8,
        },
        {
            "status": "error",
            "family": "plate_pockets",
            "strata": {"category": "planar"},
            "faces": 0,
            "regions": 0,
            "matched": 0,
        },
    ]
    by_family = aggregate(records, by="family")
    assert by_family["plate_pockets"]["cells"] == 2
    assert by_family["plate_pockets"]["matched"] == 18
    assert by_family["plate_pockets"]["f1"] == pytest.approx(2 * 0.9 * 0.9 / 1.8)
    by_strata = aggregate(records, by="strata")
    assert by_strata["planar"]["f1"] == by_family["plate_pockets"]["f1"]
