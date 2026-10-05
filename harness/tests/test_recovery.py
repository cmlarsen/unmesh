import dataclasses
import json

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Solid, Sphere, Torus, fillet

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
    if result["edge_error"]["evaluated"]:
        assert result["edge_error"]["max"] == pytest.approx(0.0)
    else:
        assert result["edge_error"]["max"] is None
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


def _perp(v):
    v = np.asarray(v, float)
    p = np.cross(v, [1, 0, 0])
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(v, [0, 1, 0])
    return p / np.linalg.norm(p)


def _area_centroid(tris):
    tris = np.asarray(tris, dtype=float)
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(cross, axis=1)
    return (tris.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()


def _rotate(vec, axis, angle_deg):
    vec = np.asarray(vec, float)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    return vec * np.cos(theta) + np.cross(axis, vec) * np.sin(theta)


def _with_surface(ir, pred, fn):
    regions = [dataclasses.replace(r, surface=fn(r.surface)) if pred(r) else r for r in ir.regions]
    return dataclasses.replace(ir, regions=regions)


def _detail(result, surface):
    return next(d for d in result["faces_detail"] if d["type"] == surface)


def test_bore_radius_plus_0_99_percent_is_recovered():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    bore = next(r for r in ir.regions if r.surface.type == "cylinder")
    assert bore.surface.radius == pytest.approx(2.0)
    bigger = dataclasses.replace(bore.surface, radius=bore.surface.radius * 1.0099)
    regions = [dataclasses.replace(r, surface=bigger) if r.id == bore.id else r for r in ir.regions]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["matched"] == len(mesh.faces)
    assert result["f1"] == 1.0
    detail = _detail(result, "cylinder")
    assert detail["recovered"] is True
    assert detail["errors"]["radius_rel"] == pytest.approx(0.0099)
    assert detail["errors"]["position_mm"] > 0.01


def test_bore_axis_tilted_0_19_deg_is_recovered():
    mesh = tessellate(Box(20, 20, 4) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_bore = lambda r: r.surface.type == "cylinder"  # noqa: E731
    face = next(f for f in mesh.faces if f.surface == "cylinder")
    centroid = _area_centroid(mesh.face_tris(face.id))
    gt_origin = np.array(face.params["origin"], float)
    gt_axis = np.array(face.params["axis"], float)
    gt_axis = gt_axis / np.linalg.norm(gt_axis)
    pivot = gt_origin + float((centroid - gt_origin) @ gt_axis) * gt_axis
    tilted = _rotate(gt_axis, _perp(gt_axis), 0.19)
    moved = _with_surface(
        ir, is_bore, lambda s: dataclasses.replace(s, origin=tuple(pivot), axis=tuple(tilted))
    )
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces)
    assert result["f1"] == 1.0
    detail = _detail(result, "cylinder")
    assert detail["recovered"] is True
    assert detail["errors"]["axis_deg"] == pytest.approx(0.19, abs=1e-6)
    assert detail["errors"]["axis_mm"] <= 0.01


@pytest.mark.parametrize(("length", "tilt"), [(10, 0.19), (100, 0.05)])
def test_long_bore_tilted_within_spec_about_its_centroid_is_recovered(length, tilt):
    mesh = tessellate(Box(20, 20, length) - Cylinder(2, length * 2), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_bore = lambda r: r.surface.type == "cylinder"  # noqa: E731
    face = next(f for f in mesh.faces if f.surface == "cylinder")
    centroid = _area_centroid(mesh.face_tris(face.id))
    gt_origin = np.array(face.params["origin"], float)
    gt_axis = np.array(face.params["axis"], float) / np.linalg.norm(face.params["axis"])
    pivot = gt_origin + float((centroid - gt_origin) @ gt_axis) * gt_axis
    tilted = _rotate(gt_axis, _perp(gt_axis), tilt)
    moved = _with_surface(
        ir, is_bore, lambda s: dataclasses.replace(s, origin=tuple(pivot), axis=tuple(tilted))
    )
    detail = _detail(score_recovery(mesh, mesh.face_id, moved, np.eye(4)), "cylinder")
    assert detail["recovered"] is True
    assert detail["errors"]["axis_mm"] <= 0.01


@pytest.mark.parametrize(("radii", "delta_deg"), [((5, 2), 0.1), ((5, 4.5), 0.05)])
def test_cone_half_angle_within_spec_at_the_face_is_recovered(radii, delta_deg):
    mesh = tessellate(Cone(*radii, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_cone = lambda r: r.surface.type == "cone"  # noqa: E731
    face = next(f for f in mesh.faces if f.surface == "cone")
    centroid = _area_centroid(mesh.face_tris(face.id))
    apex = np.array(face.params["apex"], float)
    axis = np.array(face.params["axis"], float) / np.linalg.norm(face.params["axis"])
    height = float((centroid - apex) @ axis)
    point = apex + height * axis
    radius = height * np.tan(face.params["half_angle"])
    half_angle = face.params["half_angle"] + np.deg2rad(delta_deg)
    new_apex = point - (radius / np.tan(half_angle)) * axis
    moved = _with_surface(
        ir,
        is_cone,
        lambda s: dataclasses.replace(s, apex=tuple(new_apex), half_angle=float(half_angle)),
    )
    detail = _detail(score_recovery(mesh, mesh.face_id, moved, np.eye(4)), "cone")
    assert detail["recovered"] is True
    assert detail["errors"]["apex_mm"] > 0.01


def test_plate_tilted_0_05_deg_about_centroid_is_recovered():
    mesh = tessellate(Box(100, 100, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    top = next(
        f for f in mesh.faces if f.surface == "plane" and np.array(f.params["normal"])[2] > 0.9
    )
    centroid = _area_centroid(mesh.face_tris(top.id))
    tilted = _rotate(top.params["normal"], _perp(top.params["normal"]), 0.05)
    moved = _with_surface(
        ir,
        lambda r: r.id == top.id,
        lambda s: dataclasses.replace(s, normal=tuple(tilted), origin=tuple(centroid)),
    )
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces)
    assert result["f1"] == 1.0
    detail = next(d for d in result["faces_detail"] if d["face"] == top.id)
    assert detail["recovered"] is True
    assert detail["errors"]["normal_deg"] == pytest.approx(0.05, abs=1e-6)
    assert detail["errors"]["offset_mm"] <= 0.01
    assert detail["errors"]["position_mm"] > 0.01


def test_cylinder_axis_line_shifted_two_mm_is_not_recovered():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_bore = lambda r: r.surface.type == "cylinder"  # noqa: E731
    bore = next(r for r in ir.regions if is_bore(r))
    shift = 2 * _perp(bore.surface.axis)
    moved = _with_surface(
        ir, is_bore, lambda s: dataclasses.replace(s, origin=tuple(np.array(s.origin) + shift))
    )
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    detail = _detail(result, "cylinder")
    assert detail["recovered"] is False
    assert detail["errors"]["position_mm"] == pytest.approx(2.0, abs=0.01)


@pytest.mark.parametrize("direction", ["along", "sideways"])
def test_cone_apex_moved_three_mm_is_not_recovered(direction):
    mesh = tessellate(Cone(5, 2, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_cone = lambda r: r.surface.type == "cone"  # noqa: E731
    side = next(r for r in ir.regions if is_cone(r))
    axis = np.array(side.surface.axis)
    shift = 3 * (axis if direction == "along" else _perp(axis))
    moved = _with_surface(
        ir, is_cone, lambda s: dataclasses.replace(s, apex=tuple(np.array(s.apex) + shift))
    )
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    detail = _detail(result, "cone")
    assert detail["recovered"] is False
    assert detail["errors"]["position_mm"] > 0.01


@pytest.mark.parametrize("direction", ["in_plane", "along_axis"])
def test_torus_center_moved_three_mm_is_not_recovered(direction):
    mesh = tessellate(Torus(10, 2), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_torus = lambda r: r.surface.type == "torus"  # noqa: E731
    ring = next(r for r in ir.regions if is_torus(r))
    axis = np.array(ring.surface.axis)
    shift = 3 * (axis if direction == "along_axis" else _perp(axis))
    moved = _with_surface(
        ir, is_torus, lambda s: dataclasses.replace(s, center=tuple(np.array(s.center) + shift))
    )
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == 0
    assert result["f1"] == 0.0


def test_flipped_plane_normal_is_not_recovered():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    first = next(r.id for r in ir.regions if r.surface.type == "plane")
    flipped = _with_surface(
        ir,
        lambda r: r.id == first,
        lambda s: dataclasses.replace(s, normal=tuple(-np.array(s.normal))),
    )
    result = score_recovery(mesh, mesh.face_id, flipped, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    detail = next(d for d in result["faces_detail"] if d["face"] == first)
    assert detail["recovered"] is False
    assert detail["errors"]["normal_deg"] == pytest.approx(180.0)


def _flipped_orientation(surface):
    other = "same" if surface.orientation == "reversed" else "reversed"
    return dataclasses.replace(surface, orientation=other)


def test_bore_orientation_flipped_to_boss_is_not_recovered():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_bore = lambda r: r.surface.type == "cylinder"  # noqa: E731
    assert next(r for r in ir.regions if is_bore(r)).surface.orientation == "reversed"
    moved = _with_surface(ir, is_bore, _flipped_orientation)
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    assert _detail(result, "cylinder")["recovered"] is False


def test_cone_orientation_flipped_is_not_recovered():
    mesh = tessellate(Cone(5, 2, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_cone = lambda r: r.surface.type == "cone"  # noqa: E731
    moved = _with_surface(ir, is_cone, _flipped_orientation)
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    assert _detail(result, "cone")["recovered"] is False


def test_sphere_made_cavity_is_not_recovered():
    mesh = tessellate(Sphere(5), LIN, ANG)
    ir = build_oracle_ir(mesh)
    assert ir.regions[0].surface.orientation == "same"
    moved = _with_surface(ir, lambda r: True, _flipped_orientation)
    result = score_recovery(mesh, mesh.face_id, moved, np.eye(4))
    assert result["matched"] == 0
    assert result["f1"] == 0.0


def test_split_gt_circles_score_zero_edge_error():
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosedEdges

    shape = Box(10, 10, 10) - Cylinder(2, 20)
    divided = ShapeUpgrade_ShapeDivideClosedEdges(shape.wrapped)
    divided.SetNbSplitPoints(1)
    divided.Perform()
    mesh = tessellate(Solid(divided.Result()), LIN, ANG)
    assert len(mesh.adjacency) > len(build_oracle_ir(mesh).adjacencies)
    result = score_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    assert result["f1"] == 1.0
    assert result["edge_error"]["missing"] == 0
    assert result["edge_error"]["spurious"] == 0
    assert result["edge_error"]["max"] == pytest.approx(0.0)


def test_empty_ir_and_missing_adjacencies_report_none_edge_error():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    empty = dataclasses.replace(ir, regions=[], adjacencies=[], shells=[], vertices=[])
    result = score_recovery(mesh, mesh.face_id, empty, np.eye(4))
    assert result["f1"] == 0.0
    assert result["edge_error"]["evaluated"] == 0
    assert result["edge_error"]["missing"] > 0
    assert result["edge_error"]["max"] is None
    assert result["edge_error"]["mean"] is None
    assert result["edge_error"]["sharp"]["max"] is None
    no_adj = dataclasses.replace(ir, adjacencies=[])
    result = score_recovery(mesh, mesh.face_id, no_adj, np.eye(4))
    assert result["f1"] == 1.0
    assert result["edge_error"]["evaluated"] == 0
    assert result["edge_error"]["missing"] > 0
    assert result["edge_error"]["max"] is None


def test_unrecovered_face_pairs_are_missing_not_evaluated():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    is_bore = lambda r: r.surface.type == "cylinder"  # noqa: E731
    bore = next(r for r in ir.regions if is_bore(r)).id
    pairs = {(min(a.face_a, a.face_b), max(a.face_a, a.face_b)) for a in mesh.adjacency}
    bore_pairs = {p for p in pairs if bore in p}
    assert len(bore_pairs) == 2
    wrong = _with_surface(ir, is_bore, lambda s: dataclasses.replace(s, radius=s.radius * 1.02))
    result = score_recovery(mesh, mesh.face_id, wrong, np.eye(4))
    assert result["matched"] == len(mesh.faces) - 1
    assert result["edge_error"]["missing"] == len(bore_pairs)
    assert result["edge_error"]["evaluated"] == len(pairs) - len(bore_pairs)


def test_spurious_ir_adjacency_is_counted():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    pairs = {(min(a.face_a, a.face_b), max(a.face_a, a.face_b)) for a in mesh.adjacency}
    hole = next((a, b) for a in range(7) for b in range(a + 1, 7) if (a, b) not in pairs)
    extra = dataclasses.replace(ir.adjacencies[0], regions=hole)
    result = score_recovery(
        mesh, mesh.face_id, dataclasses.replace(ir, adjacencies=ir.adjacencies + [extra]), np.eye(4)
    )
    assert result["f1"] == 1.0
    assert result["edge_error"]["spurious"] == 1
    assert result["edge_error"]["missing"] == 0
    assert result["edge_error"]["max"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [[-1], [10**6]])
def test_triangle_ids_out_of_range_raise(bad):
    from unmesh.ir import Region

    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    first = ir.regions[0]
    regions = [Region(0, first.surface, first.triangles + bad, first.residual)] + ir.regions[1:]
    with pytest.raises(ValueError, match="outside"):
        score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))


def test_runner_record_omits_details_but_function_keeps_them():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    full = score_recovery(mesh, mesh.face_id, ir, np.eye(4))
    assert full["faces_detail"]
    assert full["edge_error"]["details"]
    record = face_recovery(mesh, mesh.face_id, ir, np.eye(4))
    assert "faces_detail" not in record
    assert "details" not in record["edge_error"]
    assert record["edge_error"]["missing"] == 0
    assert record["f1"] == 1.0


def test_non_rigid_frame_raises():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    with pytest.raises(ValueError, match="rigid"):
        score_recovery(mesh, mesh.face_id, ir, np.diag([2.0, 1.0, 1.0, 1.0]))


def test_mirrored_oracle_still_scores_f1_one():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    moved = apply("mirror", mesh, 1.0, 3)
    moved = apply("rotation", moved, 0.3, 4)
    ir = build_oracle_ir(moved)
    result = score_recovery(mesh, moved.face_id, ir, to_original(moved))
    assert result["f1"] == 1.0
    assert result["edge_error"]["max"] == pytest.approx(0.0)


def _corner_fillet_mesh(seed=0):
    from unmesh_harness.groundtruth import generate

    gt = generate("corner_fillet", seed)
    mesh = tessellate(gt.solid, LIN, ANG)
    assert gt.face_tags is not None and len(gt.face_tags) == 8
    mesh.metadata["face_tags"] = dict(gt.face_tags)
    return mesh


def test_corner_blend_faces_are_masked_from_f1():
    mesh = _corner_fillet_mesh()
    result = score_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    assert len(result["masked_faces"]) == 8
    assert {d["face"] for d in result["masked_faces"]} == {
        int(k) for k in mesh.metadata["face_tags"]
    }
    assert all(d["type"] == "sphere" for d in result["masked_faces"])
    assert result["faces"] == len(mesh.faces) - 8
    assert result["regions"] == len(build_oracle_ir(mesh).regions) - 8
    assert result["matched"] == result["faces"]
    assert result["f1"] == 1.0
    assert "sphere" not in result["per_type"]
    json.dumps(result)


def test_corrupted_masked_face_does_not_lower_f1():
    mesh = _corner_fillet_mesh()
    ir = build_oracle_ir(mesh)
    scored = score_recovery(mesh, mesh.face_id, ir, np.eye(4))
    masked_id = next(d["face"] for d in scored["masked_faces"])
    bad = next(r for r in ir.regions if r.id == masked_id)
    assert bad.surface.type == "sphere"
    wrong = dataclasses.replace(bad.surface, radius=bad.surface.radius * 1.5)
    regions = [
        dataclasses.replace(r, surface=wrong) if r.id == masked_id else r for r in ir.regions
    ]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["f1"] == 1.0
    assert result["matched"] == result["faces"]
    detail = next(d for d in result["masked_faces"] if d["face"] == masked_id)
    assert detail["recovered"] is False


def test_merged_masked_region_counts_as_false_positive():
    mesh = _corner_fillet_mesh()
    ir = build_oracle_ir(mesh)
    centroids = {f.id: _area_centroid(mesh.face_tris(f.id)) for f in mesh.faces}
    cylinders = [f.id for f in mesh.faces if f.surface == "cylinder"]
    spheres = [f.id for f in mesh.faces if f.surface == "sphere"]
    targets: dict[int, list[int]] = {}
    for s in spheres:
        c = min(cylinders, key=lambda c: float(np.linalg.norm(centroids[c] - centroids[s])))
        targets.setdefault(c, []).append(s)
    by_id = {r.id: r for r in ir.regions}
    consumed = set(targets) | {s for group in targets.values() for s in group}
    merged = []
    for c, group in targets.items():
        tris = list(by_id[c].triangles)
        for s in group:
            tris += list(by_id[s].triangles)
        merged.append(dataclasses.replace(by_id[c], triangles=tris))
    regions = merged + [r for r in ir.regions if r.id not in consumed]
    regions = [dataclasses.replace(r, id=i) for i, r in enumerate(regions)]
    result = score_recovery(mesh, mesh.face_id, dataclasses.replace(ir, regions=regions), np.eye(4))
    assert result["regions"] == len(regions)
    assert result["precision"] < 1.0
    assert result["precision"] == pytest.approx(14 / 18)


def test_unmasked_mesh_reports_no_masked_faces():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    result = score_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    assert result["masked_faces"] == []
    assert (result["faces"], result["matched"], result["f1"]) == (6, 6, 1.0)


def test_other_tag_values_do_not_mask_faces():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    mesh.metadata["face_tags"] = {0: "anything_else"}
    result = score_recovery(mesh, mesh.face_id, build_oracle_ir(mesh), np.eye(4))
    assert result["masked_faces"] == []
    assert (result["faces"], result["matched"], result["f1"]) == (6, 6, 1.0)


def test_confidence_threshold_matches_degrade():
    from unmesh_harness.degrade.processing import MASK_THRESHOLD
    from unmesh_harness.metrics.recovery import CONFIDENCE_MASK

    assert CONFIDENCE_MASK == MASK_THRESHOLD == 0.9


def test_absent_and_all_one_confidence_score_identical():
    import json

    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    plain = score_recovery(mesh, mesh.face_id, ir, np.eye(4))
    assert "faces_unrecoverable" not in plain
    assert json.dumps(
        score_recovery(mesh, mesh.face_id, ir, np.eye(4), None), sort_keys=True
    ) == json.dumps(plain, sort_keys=True)
    ones = score_recovery(mesh, mesh.face_id, ir, np.eye(4), np.ones(len(mesh.face_id)))
    assert ones.pop("faces_unrecoverable") == 0
    assert json.dumps(ones, sort_keys=True) == json.dumps(plain, sort_keys=True)


def test_masked_mislabel_is_excluded_from_both_sides():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    assert len(mesh.face_id) == 12
    wrong = mesh.face_id.copy()
    wrong[0] = wrong[2]
    unmasked = score_recovery(mesh, wrong, ir, np.eye(4))
    assert unmasked["matched"] == len(mesh.faces) - 2
    conf = np.ones(len(wrong))
    conf[0] = 0.0
    masked = score_recovery(mesh, wrong, ir, np.eye(4), conf)
    assert masked["matched"] == len(mesh.faces)
    assert masked["f1"] == 1.0


def test_face_with_no_confident_triangles_leaves_both_denominators():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    conf = np.ones(len(mesh.face_id))
    conf[mesh.face_id == 0] = 0.0
    result = score_recovery(mesh, mesh.face_id, ir, np.eye(4), conf)
    assert result["faces_unrecoverable"] == 1
    assert (result["faces"], result["regions"], result["matched"]) == (5, 5, 5)
    assert result["f1"] == 1.0
    assert result["per_type"]["plane"]["faces"] == 5
    assert result["per_type"]["plane"]["regions"] == 5


def _without_face(mesh, face: int):
    keep = np.nonzero(mesh.face_id != face)[0]
    base = build_oracle_ir(mesh)
    regions = []
    for r in base.regions:
        if r.id == face:
            continue
        tris = np.nonzero(mesh.face_id[keep] == r.id)[0]
        regions.append(Region(len(regions), r.surface, tris.tolist(), None))
    return mesh.face_id[keep], dataclasses.replace(base, regions=regions, adjacencies=[])


def test_erased_face_is_unrecoverable_only_with_confidence():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    face_id, ir = _without_face(mesh, 0)
    legacy = score_recovery(mesh, face_id, ir, np.eye(4))
    assert (legacy["faces"], legacy["matched"]) == (6, 5)
    assert legacy["recall"] == pytest.approx(5 / 6)
    scored = score_recovery(mesh, face_id, ir, np.eye(4), np.ones(len(face_id)))
    assert scored["faces_unrecoverable"] == 1
    assert (scored["faces"], scored["matched"], scored["f1"]) == (5, 5, 1.0)


def test_confident_spurious_region_still_costs_precision():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    first = ir.regions[0]
    half = len(first.triangles) // 2
    regions = [dataclasses.replace(first, triangles=first.triangles[:half])]
    regions += list(ir.regions[1:])
    regions.append(Region(len(ir.regions), first.surface, first.triangles[half:], None))
    split = dataclasses.replace(ir, regions=regions)
    conf = np.ones(len(mesh.face_id))
    result = score_recovery(mesh, mesh.face_id, split, np.eye(4), conf)
    assert result["regions"] == 7
    assert result["precision"] < 1.0
    conf[first.triangles[half:]] = 0.0
    masked = score_recovery(mesh, mesh.face_id, split, np.eye(4), conf)
    assert masked["regions"] == 6
    assert masked["f1"] == 1.0


def test_nothing_confident_scores_vacuously():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    result = score_recovery(mesh, mesh.face_id, ir, np.eye(4), np.zeros(len(mesh.face_id)))
    assert (result["faces"], result["regions"], result["faces_unrecoverable"]) == (0, 0, 6)
    assert result["f1"] == 1.0


def test_confidence_length_mismatch_raises():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    with pytest.raises(ValueError, match="confidence"):
        score_recovery(mesh, mesh.face_id, ir, np.eye(4), np.ones(3))
