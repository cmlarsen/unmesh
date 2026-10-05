import math
from collections import Counter

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Plane, Pos, Rotation, Sphere, Torus, fillet, mirror

from unmesh.ir import Ir, validate
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import (
    DEFLECTION_SETTINGS,
    TANGENT_THRESHOLD_DEG,
    EdgeAdjacency,
    LabeledMesh,
    tessellate,
)
from unmesh_harness.oracle import _Directed, _Split, _split_runs, build_oracle_ir

from .cases import entry_shape, smoke_by_deflection


def equal_tee():
    return Cylinder(5, 20) + (Pos(0, 0, 0) * Rotation(0, 90, 0) * Cylinder(5, 20))


def tangent_boss():
    wall = Rotation(0, 90, 0) * Cylinder(5, 20)
    return wall + (Pos(0, 0, 7.5) * Cylinder(5, 15))


def crossing_side(mesh, vertex):
    a, b = vertex.regions
    thr = math.radians(TANGENT_THRESHOLD_DEG)
    pos = tuple(vertex.position)
    for adj in mesh.adjacency:
        if (adj.face_a, adj.face_b) != (a, b) or len(adj.dihedral_samples) != len(adj.points):
            continue
        pts = [tuple(p) for p in adj.points]
        kinds = [s < thr for s in adj.dihedral_samples]
        if pos in pts:
            j = pts.index(pos)
            if any(kinds[k] != kinds[j] for k in (j - 1, j + 1) if 0 <= k < len(kinds)):
                return True
        for k in range(len(pts) - 1):
            on_segment = (
                abs(
                    math.dist(pos, pts[k])
                    + math.dist(pos, pts[k + 1])
                    - math.dist(pts[k], pts[k + 1])
                )
                < 1e-9
            )
            if on_segment and kinds[k] != kinds[k + 1]:
                return True
    return False


def surface_normals(surface, pts):
    s = surface
    if s.type == "plane":
        return np.tile(s.normal, (len(pts), 1))
    if s.type == "sphere":
        n = pts - np.array(s.center)
    elif s.type == "torus":
        axis, v = np.array(s.axis), pts - np.array(s.center)
        radial = v - np.outer(v @ axis, axis)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        n = pts - (np.array(s.center) + s.major_radius * radial)
    else:
        origin = np.array(s.apex if s.type == "cone" else s.origin)
        axis, v = np.array(s.axis), pts - origin
        n = v - np.outer(v @ axis, axis)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        if s.type == "cone":
            n = math.cos(s.half_angle) * n - math.sin(s.half_angle) * axis
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return -n if s.orientation == "reversed" else n


def check_normals_and_left_hand_rule(mesh, ir):
    for region in ir.regions:
        tris = mesh.tris[region.triangles]
        tn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        area = np.linalg.norm(tn, axis=1)
        keep = area > 1e-12
        tn = tn[keep] / area[keep, None]
        sn = surface_normals(region.surface, tris.mean(axis=1)[keep])
        assert (np.einsum("ij,ij->i", tn, sn) > 0.5).all(), (region.id, region.surface.type)

    directed = {}
    for region in ir.regions:
        for t in region.triangles:
            c = [tuple(p) for p in mesh.tris[t]]
            for i in range(3):
                directed[(c[i], c[(i + 1) % 3])] = region.id
    for adj in ir.adjacencies:
        a, b = adj.regions
        for bd in adj.boundaries:
            pts = [tuple(p) for p in bd.points]
            segs = list(zip(pts, pts[1:], strict=False))
            if bd.closed:
                segs.append((pts[-1], pts[0]))
            for p, q in segs:
                assert directed.get((p, q)) == a and directed.get((q, p)) == b


def check_oracle(shape, lin, ang):
    mesh = tessellate(shape, lin, ang)
    ir = build_oracle_ir(mesh)
    assert validate(ir) == []
    check_normals_and_left_hand_rule(mesh, ir)
    text = ir.dumps()
    back = Ir.loads(text)
    assert back.dumps() == text
    assert len(ir.regions) == len(shape.faces()) == len(mesh.faces)
    assert {a.regions for a in ir.adjacencies} == {(a.face_a, a.face_b) for a in mesh.adjacency}
    assert ir.source.triangle_count == len(mesh.tris)
    covered = sorted(t for r in ir.regions for t in r.triangles)
    assert len(set(covered)) == len(covered)
    missing = np.setdiff1d(np.arange(len(mesh.tris)), covered)
    areas = np.linalg.norm(
        np.cross(
            mesh.tris[missing, 1] - mesh.tris[missing, 0],
            mesh.tris[missing, 2] - mesh.tris[missing, 0],
        ),
        axis=1,
    )
    assert (areas < 1e-9).all()
    for adj in ir.adjacencies:
        assert adj.boundaries
    return mesh, ir


@pytest.mark.parametrize(("entry", "lin", "ang"), smoke_by_deflection())
def test_smoke_oracle_validates(entry, lin, ang):
    gt = generate(entry["family"], entry["seed"])
    mesh, ir = check_oracle(gt.solid, lin, ang)
    assert len(ir.shells) == gt.parameters.get("shells", 1)
    assert all(s.closed for s in ir.shells)
    assert all(v.role == "junction" for v in ir.vertices)
    assert all(not b.closed or len(b.points) >= 3 for a in ir.adjacencies for b in a.boundaries)


@pytest.mark.slow
@pytest.mark.parametrize(("lin", "ang"), DEFLECTION_SETTINGS)
@pytest.mark.parametrize(
    "entry",
    [e for e in select(load_manifest(), "standard") if e["tier"] == "generated"],
    ids=lambda e: e["id"],
)
def test_standard_oracle_validates(entry, lin, ang):
    check_oracle(entry_shape(entry), lin, ang)


def test_box_structure_and_orientation():
    mesh, ir = check_oracle(Box(10, 20, 30), 0.1, 0.5)
    assert len(ir.vertices) == 8
    assert len(ir.adjacencies) == 12
    assert all(
        b.kind == "transversal" and b.dihedral_deg == pytest.approx(90)
        for a in ir.adjacencies
        for b in a.boundaries
    )
    for adj in ir.adjacencies:
        a, b = adj.regions
        (bd,) = adj.boundaries
        pa, pb = ir.regions[a].surface, ir.regions[b].surface
        na, nb = np.array(pa.normal), np.array(pb.normal)
        t = np.array(bd.points[-1]) - np.array(bd.points[0])
        mid = (np.array(bd.points[0]) + np.array(bd.points[-1])) / 2
        centroid_a = mesh.face_tris(a).mean(axis=(0, 1))
        assert np.cross(na, t) @ (centroid_a - mid) > 0
        centroid_b = mesh.face_tris(b).mean(axis=(0, 1))
        assert np.cross(nb, -t) @ (centroid_b - mid) > 0


@pytest.mark.parametrize(
    "make",
    [
        lambda: fillet(Box(20, 20, 20).edges(), 3),
        lambda: mirror(Box(10, 10, 10) - Cylinder(2, 20), Plane.YZ),
        lambda: mirror(fillet(Box(20, 20, 20).edges(), 3), Plane.XY),
        lambda: mirror(Cone(5, 2, 10) + Torus(10, 2), Plane.XZ),
    ],
)
def test_orientation_survives_left_handed_surfaces(make):
    check_oracle(make(), 0.02, 0.3)


def test_fillet_all_edges_orientations():
    mesh, ir = check_oracle(fillet(Box(20, 20, 20).edges(), 3), 0.02, 0.3)
    assert all(r.surface.orientation == "same" for r in ir.regions if r.surface.type != "plane")


def test_retessellate_gives_requested_deflection_without_mutating_input():
    shape = Cylinder(5, 10)
    fine = tessellate(shape, 0.001, 0.1)
    coarse = tessellate(shape, 0.1, 0.5)
    again = tessellate(shape, 0.001, 0.1)
    assert len(coarse.tris) < len(fine.tris) == len(again.tris)
    assert np.array_equal(fine.tris, again.tris)


def test_degenerate_pole_triangles_are_in_no_region():
    mesh, ir = check_oracle(Sphere(5), 0.01, 0.2)
    covered = sum(len(r.triangles) for r in ir.regions)
    assert covered < len(mesh.tris)


def test_cylinder_loops_and_orientation():
    mesh, ir = check_oracle(Cylinder(5, 10), 0.01, 0.2)
    assert not ir.vertices
    assert all(b.closed for a in ir.adjacencies for b in a.boundaries)
    side = next(r for r in ir.regions if r.surface.type == "cylinder")
    assert side.surface.orientation == "same"


def test_bore_is_reversed():
    mesh, ir = check_oracle(Box(10, 10, 10) - Cylinder(2, 20), 0.01, 0.2)
    bore = next(r for r in ir.regions if r.surface.type == "cylinder")
    assert bore.surface.orientation == "reversed"
    assert Counter(b.closed for a in ir.adjacencies for b in a.boundaries)[True] == 2


@pytest.mark.parametrize("make", [lambda: Cone(5, 2, 10), lambda: Sphere(5), lambda: Torus(10, 2)])
def test_round_primitives(make):
    mesh, ir = check_oracle(make(), 0.01, 0.2)
    for r in ir.regions:
        if r.surface.type == "cone":
            assert r.surface.orientation == "same"
            assert 0 < r.surface.half_angle < math.pi / 2
        assert r.residual.max <= 0.01 + 1e-9


def test_fillet_boundaries_are_tangent():
    box = Box(20, 20, 20)
    edge = box.edges().filter_by(lambda e: abs(e.center().Z - 10) < 1e-6)[:1]
    mesh, ir = check_oracle(fillet(edge, 3), 0.05, 0.3)
    tangent = [(a.regions, b) for a in ir.adjacencies for b in a.boundaries if b.kind == "tangent"]
    assert len(tangent) == 2
    assert all(b.dihedral_deg < 3.0 and not b.closed for _, b in tangent)
    assert not any(v.role == "kind_change" for v in ir.vertices)


def test_cavity_shells():
    solid = Box(30, 30, 30) - Box(10, 10, 10)
    mesh, ir = check_oracle(solid, 0.1, 0.5)
    assert [s.role for s in ir.shells] == ["outer", "cavity"]
    assert ir.shells[1].parent == 0
    vol = {}
    for i, s in enumerate(ir.shells):
        ids = [t for r in s.regions for t in ir.regions[r].triangles]
        tris = mesh.tris[ids]
        vol[i] = float(
            np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
        )
    assert vol[0] == pytest.approx(27000) and vol[1] == pytest.approx(-1000)


def test_labeled_mesh_round_trip_builds_same_ir(tmp_path):
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), 0.01, 0.2)
    mesh.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert build_oracle_ir(back).dumps() == build_oracle_ir(mesh).dumps()


def check_mid_edge_splits(shape):
    mesh = tessellate(shape, 0.01, 0.2)
    ir = build_oracle_ir(mesh)
    assert validate(ir) == []
    changed = [v for v in ir.vertices if v.role == "kind_change"]
    assert changed
    assert all(crossing_side(mesh, v) for v in changed)
    for adj in ir.adjacencies:
        for b in adj.boundaries:
            assert (b.dihedral_deg < TANGENT_THRESHOLD_DEG) == (b.kind == "tangent")
    return mesh, ir


def test_equal_tee_mid_edge_kind_change():
    mesh, ir = check_mid_edge_splits(equal_tee())
    pairs = {(a.face_a, a.face_b) for a in mesh.adjacency}
    split_pairs = {tuple(sorted(v.regions)) for v in ir.vertices if v.role == "kind_change"}
    assert split_pairs
    assert split_pairs <= pairs
    assert any(b.kind == "tangent" for a in ir.adjacencies for b in a.boundaries if not b.closed)


def test_boss_part_loop_kind_change():
    mesh, ir = check_mid_edge_splits(tangent_boss())
    assert any(b.kind == "tangent" for a in ir.adjacencies for b in a.boundaries if not b.closed)


def make_directed(points, samples_deg):
    adj = EdgeAdjacency(
        edge_id=0,
        face_a=0,
        face_b=1,
        curve="line",
        tangent=False,
        dihedral=math.radians(45.0),
        dihedral_min=math.radians(min(samples_deg)),
        dihedral_max=math.radians(max(samples_deg)),
        points=[[float(x), 0.0, 0.0] for x in points],
        start_vertex=0,
        end_vertex=1,
        forward_in_a=True,
        dihedral_samples=[math.radians(s) for s in samples_deg],
    )
    return _Directed(adj)


def test_split_runs_inserts_far_crossing():
    d = make_directed([0.0, 10.0, 20.0], [1.0, 1.0, 10.0])
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert isinstance(segs[0].end, _Split) and segs[0].end is segs[1].start
    assert segs[0].points[-1] == segs[1].points[0] == segs[0].end.point
    assert segs[0].points[-1][0] == pytest.approx(10.0 + (3.0 - 1.0) / 9.0 * 10.0)
    assert segs[0].dihedral == pytest.approx(1.0)
    assert segs[1].dihedral == pytest.approx(10.0)


def test_split_runs_splits_at_near_node():
    d = make_directed([0.0, 10.0, 20.0], [1.0, 2.9, 10.0])
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert segs[0].points[-1] == segs[1].points[0] == (10.0, 0.0, 0.0)
    assert segs[0].dihedral == pytest.approx(1.95)
    assert segs[1].dihedral == pytest.approx(6.45)


def test_split_runs_interior_tangent_run():
    d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [10.0, 10.0, 1.0, 10.0, 10.0])
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 3
    assert [s.dihedral > 3.0 for s in (segs[0], segs[2])] == [True, True]
    assert segs[1].dihedral == pytest.approx(1.0)
    assert len(segs[1].points) == 3


def test_split_runs_no_crossing_keeps_whole_edge():
    d = make_directed([0.0, 10.0, 20.0], [10.0, 20.0, 30.0])
    (seg,) = _split_runs(d, 3.0, (0, 1))
    assert seg.points == d.points and seg.dihedral == d.dihedral
    assert (seg.start, seg.end) == (0, 1)
