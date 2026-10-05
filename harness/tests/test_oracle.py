import math
import random
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
        if pos not in pts:
            continue
        j = pts.index(pos)
        kinds = [s < thr for s in adj.dihedral_samples]
        if any(kinds[k] != kinds[j] for k in (j - 1, j + 1) if 0 <= k < len(kinds)):
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
    mesh = tessellate(shape, 0.001, 0.1)
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


def make_directed(points, samples_deg, forward=True, closed=False):
    adj = EdgeAdjacency(
        edge_id=0,
        face_a=0,
        face_b=1,
        curve="line",
        tangent=False,
        dihedral=math.radians(45.0),
        dihedral_min=math.radians(min(samples_deg)),
        dihedral_max=math.radians(max(samples_deg)),
        points=[[float(x), 0.0, 0.0] if isinstance(x, float) else list(x) for x in points],
        start_vertex=0,
        end_vertex=0 if closed else 1,
        forward_in_a=forward,
        dihedral_samples=[math.radians(s) for s in samples_deg],
    )
    return _Directed(adj)


def test_split_runs_snaps_to_nearest_node():
    d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 10.0, 10.0])
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert isinstance(segs[0].end, _Split) and segs[0].end is segs[1].start
    assert segs[0].points[-1] == segs[1].points[0] == segs[0].end.point == (20.0, 0.0, 0.0)
    assert segs[0].dihedral == pytest.approx(1.0)
    assert segs[1].dihedral == pytest.approx(10.0)


def test_split_runs_snaps_to_closer_side_of_segment():
    d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 2.9, 10.0, 10.0])
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert segs[0].points[-1] == segs[1].points[0] == (20.0, 0.0, 0.0)


def test_split_runs_short_spike_stays_whole():
    d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [10.0, 10.0, 1.0, 10.0, 10.0])
    (seg,) = _split_runs(d, 3.0, (0, 1))
    assert seg.points == d.points and seg.dihedral == d.dihedral
    assert (seg.start, seg.end) == (0, 1)


def test_split_runs_no_crossing_keeps_whole_edge():
    d = make_directed([0.0, 10.0, 20.0], [10.0, 20.0, 30.0])
    (seg,) = _split_runs(d, 3.0, (0, 1))
    assert seg.points == d.points and seg.dihedral == d.dihedral
    assert (seg.start, seg.end) == (0, 1)


def test_split_runs_touch_only_edge_stays_whole():
    for peak in (3.0, 3.0000001, 3.4):
        d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [1.0, 1.0, peak, 1.0, 1.0])
        (seg,) = _split_runs(d, 3.0, (0, 1))
        assert seg.points == d.points and seg.dihedral == d.dihedral
    for dip in (3.0, 2.9, 2.6):
        d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0], [10.0, 10.0, dip, 10.0, 10.0])
        (seg,) = _split_runs(d, 3.0, (0, 1))
        assert seg.points == d.points and seg.dihedral == d.dihedral


def test_split_runs_hovering_edge_stays_whole():
    d = make_directed(
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        [2.9, 3.1, 2.9, 3.1, 2.9, 3.1],
    )
    (seg,) = _split_runs(d, 3.0, (0, 1))
    assert seg.points == d.points and seg.dihedral == d.dihedral


def test_split_runs_closed_loop_with_two_crossings():
    points = [
        (10.0 * math.cos(i * math.pi / 4), 10.0 * math.sin(i * math.pi / 4), 0.0) for i in range(9)
    ]
    d = make_directed(points, [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0], closed=True)
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert segs[0].end is segs[1].start and segs[1].end is segs[0].start
    assert all(len(s.points) >= 3 for s in segs)
    assert (segs[0].dihedral < 3.0) != (segs[1].dihedral < 3.0)
    assert segs[0].dihedral == pytest.approx(1.0)
    assert segs[1].dihedral == pytest.approx(10.0)


def test_split_runs_reversed_edge_keeps_samples_on_points():
    d = make_directed([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 1.0, 1.0, 10.0, 10.0, 10.0], False)
    segs = _split_runs(d, 3.0, (0, 1))
    assert len(segs) == 2
    assert (segs[0].start, segs[1].end) == (1, 0)
    assert segs[0].dihedral == pytest.approx(10.0)
    assert segs[1].dihedral == pytest.approx(1.0)
    assert segs[0].end.point == segs[1].points[0] == (20.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "samples",
    [
        [1.0, 1.0, 2.8, 3.2, 10.0, 10.0],
        [1.0, 1.0, 2.7, 2.9, 3.1, 10.0, 10.0],
    ],
)
def test_split_runs_direction_independent_inside_band(samples):
    xs = [10.0 * i for i in range(len(samples))]
    fwd = _split_runs(make_directed(xs, samples, True), 3.0, (0, 1))
    rev = _split_runs(make_directed(xs, samples, False), 3.0, (0, 1))
    assert len(fwd) == len(rev) == 2

    def splits(segs):
        return sorted({v.point for s in segs for v in (s.start, s.end) if isinstance(v, _Split)})

    def runs(segs):
        return [(s.dihedral < 3.0, s.dihedral, sorted(s.points)) for s in segs]

    assert splits(fwd) and splits(fwd) == splits(rev)
    assert runs(fwd) == runs(rev)[::-1]


def split_points(segs):
    return sorted({v.point for s in segs for v in (s.start, s.end) if isinstance(v, _Split)})


def run_key(segs, threshold=3.0):
    return sorted(
        (s.dihedral < threshold, s.dihedral, sorted({tuple(p) for p in s.points})) for s in segs
    )


def test_split_runs_short_run_merge_direction_independent():
    samples = [10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 10.0]
    xs = [10.0 * i for i in range(len(samples))]
    fwd = _split_runs(make_directed(xs, samples, True), 3.0, (0, 1))
    rev = _split_runs(make_directed(xs, samples, False), 3.0, (0, 1))
    assert split_points(fwd) == split_points(rev)
    assert run_key(fwd) == run_key(rev)


def test_split_runs_direction_independent_fuzz():
    rng = random.Random(89)
    for trial in range(2000):
        closed = trial % 2 == 1
        n = rng.randint(4, 14)
        samples = []
        for _ in range(n):
            r = rng.random()
            if r < 0.45:
                samples.append(rng.uniform(5.0, 30.0))
            elif r < 0.9:
                samples.append(rng.uniform(0.2, 2.4))
            else:
                samples.append(rng.uniform(2.5, 3.5))
        if closed:
            samples[-1] = samples[0]
            m = n - 1
            cyc = [
                (10.0 * math.cos(2 * math.pi * i / m), 10.0 * math.sin(2 * math.pi * i / m), 0.0)
                for i in range(m)
            ]
            pts = cyc + [cyc[0]]
            fwd = _split_runs(make_directed(pts, samples, True, closed=True), 3.0, (0, 1))
            rev = _split_runs(make_directed(pts, samples, False, closed=True), 3.0, (0, 1))
            assert split_points(fwd) == split_points(rev), trial
            assert run_key(fwd) == run_key(rev), trial
            k = rng.randint(1, n - 2)
            cyc_samples = samples[:-1]
            rpts, rsamples = cyc[k:] + cyc[:k], cyc_samples[k:] + cyc_samples[:k]
            rot = _split_runs(
                make_directed(rpts + [rpts[0]], rsamples + [rsamples[0]], True, closed=True),
                3.0,
                (0, 1),
            )
            assert split_points(fwd) == split_points(rot), trial
            assert run_key(fwd) == run_key(rot), trial
        else:
            xs = [10.0 * i for i in range(n)]
            fwd = _split_runs(make_directed(xs, samples, True), 3.0, (0, 1))
            rev = _split_runs(make_directed(xs, samples, False), 3.0, (0, 1))
            assert split_points(fwd) == split_points(rev), trial
            assert run_key(fwd) == run_key(rev), trial


def test_directed_samples_follow_reversed_points():
    mesh = tessellate(mirror(equal_tee(), Plane.YZ), 0.01, 0.2)
    adj = next(
        a
        for a in mesh.adjacency
        if not a.forward_in_a
        and len(a.dihedral_samples) == len(a.points)
        and max(math.degrees(s) for s in a.dihedral_samples)
        - min(math.degrees(s) for s in a.dihedral_samples)
        > 5.0
    )
    d = _Directed(adj)
    assert [tuple(p) for p in d.points] == [tuple(p) for p in adj.points[::-1]]
    by_point = {
        tuple(p): math.degrees(s) for p, s in zip(adj.points, adj.dihedral_samples, strict=True)
    }
    assert len(by_point) == len(adj.points)
    for p, s in zip(d.points, d.samples, strict=True):
        assert s == pytest.approx(by_point[tuple(p)])


def revolved_spline():
    from build123d import (
        Axis,
        BuildLine,
        BuildPart,
        BuildSketch,
        Line,
        Spline,
        make_face,
        revolve,
    )

    with BuildPart() as part:
        with BuildSketch(Plane.XZ):
            with BuildLine():
                Line((0, 0), (8, 0))
                Spline((8, 0), (6, 5), (9, 10))
                Line((9, 10), (0, 10))
                Line((0, 10), (0, 0))
            make_face()
        revolve(axis=Axis.Z)
    return part.part


def test_non_analytic_face_becomes_facets_region():
    mesh = tessellate(revolved_spline(), 0.05, 0.3)
    kinds = [f.surface for f in mesh.faces]
    assert "revolution" in kinds
    ir = build_oracle_ir(mesh)
    assert validate(ir) == []
    for face, region in zip(mesh.faces, ir.regions, strict=True):
        expected = "facets" if face.surface == "revolution" else face.surface
        assert region.surface.type == expected
        assert (region.residual is None) == (expected == "facets")
    facets = ir.regions[kinds.index("revolution")]
    assert len(facets.surface.faces) == len(facets.triangles) > 0
