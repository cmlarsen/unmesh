import dataclasses
import json
import random

import pytest
from build123d import Box, Cylinder, Pos, Rot, Rotation

from unmesh.ir import Adjacency, Region, validate
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, tessellate
from unmesh_harness.metrics.structure import score_structure, score_topology, score_validity
from unmesh_harness.oracle import build_oracle_ir
from unmesh_harness.runner.converters import faceted_ir

LIN, ANG = 0.01, 0.2


def _write_box_step(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "box.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid and report.fallback is None
    return out, mesh


def test_validity_oracle_box_step_is_valid(tmp_path):
    out, _ = _write_box_step(tmp_path)
    result = score_validity(out)
    assert result["valid"], result["problems"]
    assert (result["solids"], result["shells"]) == (1, 1)
    assert result["volume"] == pytest.approx(1000.0)
    assert result["volume_positive"] is True
    assert result["brepcheck_valid"] is True
    assert result["max_shape_tolerance"] <= result["tolerance_bound"] == pytest.approx(1e-3)
    assert result["tolerance_ok"] is True
    assert result["fallback"] is False
    json.dumps(result)


def test_validity_cavity_expects_two_shells(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(30, 30, 30) - Box(10, 10, 10), 0.1, 0.5)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "cavity.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid, report.issues
    one_shell = score_validity(out)
    assert one_shell["valid"] is False
    assert any("shell" in p for p in one_shell["problems"])
    two_shells = score_validity(out, expected_solids=1, expected_shells=2)
    assert two_shells["valid"], two_shells["problems"]
    assert (two_shells["solids"], two_shells["shells"]) == (1, 2)
    json.dumps(two_shells)


def test_validity_missing_step_is_invalid(tmp_path):
    result = score_validity(tmp_path / "absent.step")
    assert result["valid"] is False
    assert result["problems"] == ["no STEP file written"]
    assert result["solids"] is None
    empty = tmp_path / "empty.step"
    empty.write_text("")
    assert score_validity(empty)["problems"] == ["no STEP file written"]
    json.dumps(result)


def test_validity_reports_fallback_without_failing(tmp_path):
    out, _ = _write_box_step(tmp_path)
    result = score_validity(out, write={"fallback": "faceted", "fallback_reason": "test"})
    assert result["fallback"] is True
    assert result["valid"] is True
    assert score_validity(out, write={"fallback": None})["fallback"] is False
    assert score_validity(out, write={"fallback": None}, fallback=True)["fallback"] is True


def test_validity_through_bore_counts_one_solid(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "bore.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid, report.issues
    result = score_validity(out)
    assert result["valid"], result["problems"]
    assert (result["solids"], result["shells"]) == (1, 1)
    assert result["volume"] == pytest.approx(1000 - 3.14159265 * 4 * 10, rel=1e-3)


def _drop_region(ir, rid):
    remap = {old: new for new, old in enumerate(r.id for r in ir.regions if r.id != rid)}
    regions = [dataclasses.replace(r, id=remap[r.id]) for r in ir.regions if r.id != rid]
    shells = [
        dataclasses.replace(s, regions=[remap[r] for r in s.regions if r != rid]) for s in ir.shells
    ]
    adjacencies = [
        Adjacency((remap[a], remap[b]), list(adj.boundaries))
        for adj in ir.adjacencies
        for a, b in [adj.regions]
        if rid not in adj.regions
    ]
    vertices = [
        dataclasses.replace(v, id=i, regions=[remap[r] for r in v.regions])
        for i, v in enumerate(v for v in ir.vertices if rid not in v.regions)
    ]
    return dataclasses.replace(
        ir, regions=regions, shells=shells, adjacencies=adjacencies, vertices=vertices
    )


def _merge_regions(ir, keep_id, drop_id):
    keep = next(r for r in ir.regions if r.id == keep_id)
    others = [r.id for r in ir.regions if r.id not in (keep_id, drop_id)]
    remap = {keep_id: 0, drop_id: 0}
    remap.update({old: new for new, old in enumerate(others, 1)})
    merged = Region(
        0,
        keep.surface,
        keep.triangles + next(r for r in ir.regions if r.id == drop_id).triangles,
        keep.residual,
    )
    regions = [merged] + [dataclasses.replace(ir.regions[o], id=remap[o]) for o in others]
    shells = [
        dataclasses.replace(s, regions=sorted({remap[r] for r in s.regions})) for s in ir.shells
    ]
    grouped: dict[tuple[int, int], list] = {}
    for adj in ir.adjacencies:
        a, b = (remap[r] for r in adj.regions)
        if a == b:
            continue
        grouped.setdefault((min(a, b), max(a, b)), []).extend(adj.boundaries)
    adjacencies = [Adjacency(pair, bounds) for pair, bounds in sorted(grouped.items())]
    vertices = [
        dataclasses.replace(v, id=i, regions=sorted({remap[r] for r in v.regions}))
        for i, v in enumerate(ir.vertices)
    ]
    return dataclasses.replace(
        ir, regions=regions, shells=shells, adjacencies=adjacencies, vertices=vertices
    )


PRIMITIVE_CASES = {
    "box": (lambda: Box(10, 10, 10), 0.0),
    "through_bore": (lambda: Box(10, 10, 10) - Cylinder(2, 20), 1.0),
    "solid_cylinder": (lambda: Cylinder(5, 10), 0.0),
}


@pytest.mark.parametrize("name", sorted(PRIMITIVE_CASES))
def test_oracle_topology_matches_primitives(name):
    make, holes = PRIMITIVE_CASES[name]
    mesh = tessellate(make(), LIN, ANG)
    result = score_topology(mesh, mesh.face_id, mesh.tris, build_oracle_ir(mesh))
    assert result["topology_match"], result
    assert result["through_holes"] == result["truth_through_holes"] == holes
    assert result["truth_faces"] == result["output_faces"] == len(mesh.faces)
    assert result["pairs_match"] and result["roles_match"] and result["shells_match"]
    json.dumps(result)


CURVED_FAMILIES = [
    "through_bore",
    "blind_bore",
    "round_boss",
    "counterbore",
    "countersink",
    "round_slot_through",
    "round_slot_blind",
    "revolved_cone",
    "revolved_dome",
    "revolved_torus",
]


@pytest.mark.parametrize("family", CURVED_FAMILIES)
def test_oracle_topology_matches_curved_families(family):
    mesh = tessellate(generate(family, 0).solid, LIN, ANG)
    result = score_topology(mesh, mesh.face_id, mesh.tris, build_oracle_ir(mesh))
    assert result["topology_match"], (family, result)
    assert result["through_holes"] == result["truth_through_holes"]
    json.dumps(result)


@pytest.mark.slow
def test_oracle_topology_matches_smoke_parts():
    for entry in select(load_manifest(), "smoke"):
        mesh = tessellate(generate(entry["family"], entry["seed"]).solid, 0.1, 0.5)
        result = score_topology(mesh, mesh.face_id, mesh.tris, build_oracle_ir(mesh))
        assert result["topology_match"], (entry["id"], result)


def _relabel_regions(ir, perm):
    regions = sorted(
        [dataclasses.replace(r, id=perm[r.id]) for r in ir.regions], key=lambda r: r.id
    )
    shells = [dataclasses.replace(s, regions=sorted(perm[r] for r in s.regions)) for s in ir.shells]
    adjacencies = []
    for adj in ir.adjacencies:
        x, y = (perm[r] for r in adj.regions)
        bounds = adj.boundaries
        if x > y:
            x, y = y, x
            bounds = [
                dataclasses.replace(
                    b,
                    points=b.points[::-1],
                    start_vertex=b.end_vertex,
                    end_vertex=b.start_vertex,
                )
                for b in bounds
            ]
        adjacencies.append(Adjacency((x, y), bounds))
    adjacencies.sort(key=lambda a: a.regions)
    vertices = [
        dataclasses.replace(v, regions=sorted(perm[r] for r in v.regions)) for v in ir.vertices
    ]
    return dataclasses.replace(
        ir, regions=regions, shells=shells, adjacencies=adjacencies, vertices=vertices
    )


def test_renumbered_oracle_regions_still_match():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    perm = list(range(len(ir.regions)))
    random.Random(1).shuffle(perm)
    relabeled = _relabel_regions(ir, perm)
    assert validate(relabeled) == []
    result = score_topology(mesh, mesh.face_id, mesh.tris, relabeled)
    assert result["topology_match"], result
    assert result["pairs_match"] and result["roles_match"] and result["shells_match"]
    json.dumps(result)


def test_swapped_shell_roles_do_not_match():
    mesh = tessellate(Box(30, 30, 30) - Box(10, 10, 10), 0.1, 0.5)
    ir = build_oracle_ir(mesh)
    assert [s.role for s in ir.shells] == ["outer", "cavity"]
    swapped = dataclasses.replace(
        ir,
        shells=[
            dataclasses.replace(ir.shells[0], role="cavity", parent=1),
            dataclasses.replace(ir.shells[1], role="outer", parent=None),
        ],
    )
    assert validate(swapped) == []
    result = score_topology(mesh, mesh.face_id, mesh.tris, swapped)
    assert result["topology_match"] is False
    assert result["shells_match"] is False
    assert result["pairs_match"] and result["roles_match"]
    json.dumps(result)


def test_rod_through_block_oracle_matches():
    solid = Box(10, 10, 10) + Pos(0, 0, 5) * Rot(0, 90, 0) * Cylinder(3, 20)
    mesh = tessellate(solid, LIN, ANG)
    result = score_topology(mesh, mesh.face_id, mesh.tris, build_oracle_ir(mesh))
    assert result["topology_match"], result
    assert result["pairs_match"] and result["edges_match"]
    json.dumps(result)


def _equal_tee():
    return Cylinder(5, 20) + (Pos(0, 0, 0) * Rotation(0, 90, 0) * Cylinder(5, 20))


def _tangent_boss():
    wall = Rotation(0, 90, 0) * Cylinder(5, 20)
    return wall + (Pos(0, 0, 7.5) * Cylinder(5, 15))


@pytest.mark.parametrize("make", [_equal_tee, _tangent_boss])
@pytest.mark.parametrize("lin,ang", DEFLECTION_SETTINGS)
def test_oracle_topology_matches_mid_edge_parts(make, lin, ang):
    mesh = tessellate(make(), lin, ang)
    result = score_topology(mesh, mesh.face_id, mesh.tris, build_oracle_ir(mesh))
    assert result["topology_match"], result
    assert result["pairs_match"] and result["edges_match"] and result["roles_match"]
    json.dumps(result)


@pytest.mark.parametrize("make", [_equal_tee, _tangent_boss])
def test_whole_edge_ir_misses_mid_edge_vertices(make):
    mesh = tessellate(make(), 0.001, 0.1)
    ir = build_oracle_ir(mesh)
    assert validate(ir) == []
    assert any(v.role == "kind_change" for v in ir.vertices)
    whole = build_oracle_ir(
        dataclasses.replace(
            mesh, adjacency=[dataclasses.replace(a, dihedral_samples=[]) for a in mesh.adjacency]
        )
    )
    assert validate(whole) == []
    assert not any(v.role == "kind_change" for v in whole.vertices)
    result = score_topology(mesh, mesh.face_id, mesh.tris, whole)
    assert result["topology_match"] is False
    assert result["roles_match"] is False
    assert result["edges_match"] is False
    json.dumps(result)


def test_dropped_bore_is_caught_by_through_holes():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    bore = next(r.id for r in ir.regions if r.surface.type == "cylinder")
    dropped = _drop_region(ir, bore)
    assert validate(dropped) == []
    assert len(dropped.regions) == len(ir.regions) - 1
    result = score_topology(mesh, mesh.face_id, mesh.tris, dropped)
    assert result["topology_match"] is False
    assert result["truth_faces"] == len(mesh.faces)
    assert result["output_faces"] == len(mesh.faces) - 1
    assert result["truth_through_holes"] == 1.0
    assert result["through_holes"] == 0.0
    assert result["holes_match"] is False
    json.dumps(result)


def test_merged_regions_are_caught_by_face_count():
    mesh = tessellate(Cylinder(5, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    side = next(r.id for r in ir.regions if r.surface.type == "cylinder")
    cap = next(r.id for r in ir.regions if r.surface.type == "plane")
    merged = _merge_regions(ir, side, cap)
    assert validate(merged) == []
    result = score_topology(mesh, mesh.face_id, mesh.tris, merged)
    assert result["topology_match"] is False
    assert result["truth_faces"] == 3
    assert result["output_faces"] == 2
    assert result["faces_match"] is False
    json.dumps(result)


def test_structure_oracle_is_fully_analytic():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    result = score_structure(build_oracle_ir(mesh), len(mesh.faces), mesh.tris)
    assert result["analytic_area_fraction"] == pytest.approx(1.0)
    assert result["face_count_ratio"] == pytest.approx(1.0)
    assert result["faceted_regions"] == 0
    assert result["analytic_area_mm2"] == pytest.approx(result["total_area_mm2"])
    json.dumps(result)


def test_structure_faceted_is_fully_faceted():
    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    result = score_structure(faceted_ir(mesh.tris), len(mesh.faces), mesh.tris)
    assert result["analytic_area_fraction"] == 0.0
    assert result["face_count_ratio"] == pytest.approx(1 / 6)
    assert result["analytic_regions"] == 0
    json.dumps(result)


def test_structure_dropped_bore_loses_area_and_faces():
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    full = score_structure(ir, len(mesh.faces), mesh.tris)
    assert full["analytic_area_fraction"] == pytest.approx(1.0)
    bore = next(r.id for r in ir.regions if r.surface.type == "cylinder")
    dropped = _drop_region(ir, bore)
    result = score_structure(dropped, len(mesh.faces), mesh.tris)
    assert result["face_count_ratio"] == pytest.approx(6 / 7)
    assert 0.0 < result["analytic_area_fraction"] < 1.0
    assert result["analytic_area_mm2"] < full["analytic_area_mm2"]
    json.dumps(result)
