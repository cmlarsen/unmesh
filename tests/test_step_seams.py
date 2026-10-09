import numpy as np
import pytest

pytest.importorskip("OCP")

from build123d import Box, Cone, Cylinder, Pos, Sphere, Torus, fillet  # noqa: E402
from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE  # noqa: E402
from OCP.TopExp import TopExp  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # noqa: E402

import unmesh.step as step  # noqa: E402
from unmesh._writer import curved, occ, topology  # noqa: E402
from unmesh._writer import geometry as geo  # noqa: E402
from unmesh.ir import Cylinder as IrCylinder  # noqa: E402
from unmesh.ir import Facets  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402
from unmesh_harness.oracle import build_oracle_ir, force_facets  # noqa: E402
from unmesh_harness.oracle_fit import corpus_seeds  # noqa: E402

DEFLECTION = (0.02, 0.3)


def _top_plane(ir):
    planes = [r for r in ir.regions if r.surface.type == "plane" and r.surface.normal[2] > 0.99]
    return max(planes, key=lambda r: r.surface.origin[2]).id


def _bottom_plane(ir):
    planes = [r for r in ir.regions if r.surface.type == "plane" and r.surface.normal[2] < -0.99]
    return min(planes, key=lambda r: r.surface.origin[2]).id


def _of_type(kind):
    return lambda ir: next(r.id for r in ir.regions if r.surface.type == kind)


def _torus_part():
    body = Cylinder(6, 4)
    top = body.edges().sort_by(lambda e: e.center().Z)[-1]
    return fillet(top, 1.5)


CASES = {
    "plane": (lambda: Box(10, 10, 10), _top_plane, "plane"),
    "cylinder": (lambda: Box(20, 20, 6) - Cylinder(4, 10), _top_plane, "cylinder"),
    "cylinder_patch": (lambda: Box(20, 20, 6) - Cylinder(4, 10), _of_type("cylinder"), "plane"),
    "cone": (lambda: Cone(6, 3, 5), _top_plane, "cone"),
    "sphere": (lambda: Sphere(5) - Pos(0, 0, -5) * Box(20, 20, 10), _bottom_plane, "sphere"),
    "torus": (
        lambda: Torus(10, 3) - Pos(0, 0, -10) * Box(40, 40, 20) - Pos(0, 0, 11.5) * Box(40, 40, 20),
        _top_plane,
        "torus",
    ),
    "torus_tangent": (_torus_part, _top_plane, "torus"),
}


def _forced(name, choose=None):
    make, default, _ = CASES[name]
    choose = choose or default
    shape = make()
    mesh = tessellate(shape, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = build_oracle_ir(mesh)
    region = choose(ir)
    return shape, force_facets(ir, region, tris), tris, region


def _faces_per_edge(shape) -> set[int]:
    edges = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edges)
    counts = set()
    for i in range(1, edges.Extent() + 1):
        if not BRep_Tool.Degenerated_s(TopoDS.Edge_s(edges.FindKey(i))):
            counts.add(edges.FindFromIndex(i).Size())
    return counts


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_facets_patch_shares_its_seam_with_the_analytic_face(name, tmp_path):
    shape, ir, tris, region = _forced(name)
    path = tmp_path / f"{name}.step"
    report = step.write(ir, path, mesh=tris)
    assert report.valid and report.verified and report.readback.ok, report.issues
    assert report.fallback is None, report.fallback_reason
    assert report.faceted_regions == 1
    assert report.faceted_faces >= len(ir.regions[region].triangles)
    assert report.max_shape_tolerance <= 1e-3
    assert {f.region for f in report.faces} == {r.id for r in ir.regions}
    seams = [s for s in report.seams if s.surface_type == CASES[name][2]]
    assert seams
    assert all(region in s.regions for s in report.seams)
    if CASES[name][2] != "plane":
        assert all(s.max_gap <= 1e-4 and s.tolerance <= 1e-3 for s in seams)
        assert all((s.inserted > 0) == (s.chord_gap > 1e-4) for s in seams)
        split = any(s.inserted for s in seams)
        assert split == (name != "torus_tangent")
        assert (report.faceted_faces > len(ir.regions[region].triangles)) == split
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=2e-3)
    assert _faces_per_edge(occ.read_step(path)) == {2}


def test_a_coarser_seam_gap_inserts_fewer_points(tmp_path):
    _, ir, tris, _ = _forced("cylinder")
    fine = step.write(ir, tmp_path / "f.step", mesh=tris)
    coarse = step.write(ir, tmp_path / "c.step", step.WriteOptions(max_seam_gap=1e-3), mesh=tris)
    assert fine.valid and coarse.valid and coarse.fallback is None
    gap = max(s.max_gap for s in coarse.seams if s.surface_type == "cylinder")
    assert 1e-4 < gap <= 1e-3
    assert sum(s.inserted for s in coarse.seams) < sum(s.inserted for s in fine.seams)


def test_a_flipped_facets_patch_falls_back(tmp_path):
    _, ir, tris, region = _forced("cylinder")
    patch = ir.regions[region].surface
    ir.regions[region].surface = Facets(patch.vertices, [f[::-1] for f in patch.faces])
    report = step.write(ir, tmp_path / "flipped.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert f"region {region}: facets triangles wind against their neighbours" in (
        report.fallback_reason
    )


def _largest_patch(family: str, index: int, interior: bool = True):
    gt = generate(family, corpus_seeds([family])[family][index])
    mesh = tessellate(gt.solid, 0.01, 0.2)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = build_oracle_ir(mesh)
    curved_regions = [
        r
        for r in ir.regions
        if r.surface.type not in ("plane", "facets")
        and len(r.triangles) <= step.FACETED_SHARE * len(tris)
    ]
    boundary = {tuple(p) for adj in ir.adjacencies for bd in adj.boundaries for p in bd.points}
    for r in sorted(curved_regions, key=lambda r: -len(r.triangles)):
        corners = {tuple(map(float, p)) for t in r.triangles for p in tris[t]}
        if corners - boundary or not interior:
            return force_facets(ir, r.id, tris), tris, r.id
    raise AssertionError(f"{family} has no curved region with an interior vertex")


def _bulge(ir, region, distance):
    patch = ir.regions[region].surface
    boundary = {tuple(p) for adj in ir.adjacencies for bd in adj.boundaries for p in bd.points}
    corners = {tuple(v.position) for v in ir.vertices}
    inner = next(
        (i for i, p in enumerate(patch.vertices) if tuple(p) not in boundary),
        next(i for i, p in enumerate(patch.vertices) if tuple(p) not in corners),
    )
    normal = np.zeros(3)
    for f in patch.faces:
        if inner in f:
            a, b, c = (np.asarray(patch.vertices[i]) for i in f)
            normal += np.cross(b - a, c - a)
    patch.vertices[inner] = tuple(np.asarray(patch.vertices[inner]) + distance * geo.unit(normal))


@pytest.mark.parametrize("family", ["countersink", "corner_fillet"])
@pytest.mark.parametrize("distance", [2e-3, 2e-2, 2e-1])
def test_a_patch_vertex_off_the_mesh_is_measured_and_beyond_the_limit_falls_back(
    family, distance, tmp_path
):
    ir, tris, region = _largest_patch(family, 3, interior=family != "countersink")
    _bulge(ir, region, distance)
    report = step.write(ir, tmp_path / "bulge.step", mesh=tris)
    assert report.valid and report.verified
    limit = min(5 * ir.tolerances.linear, step.WriteOptions().max_deviation)
    if family == "countersink":
        assert report.fallback == "faceted"
        assert f"region {region}: boundary point" in report.fallback_reason
        assert "is not a vertex of the facets patch" in report.fallback_reason
        if distance > limit:
            assert f"region {region}: a facets patch vertex is" in report.fallback_reason
    elif distance <= limit:
        assert report.fallback is None, report.fallback_reason
        face = next(f for f in report.faces if f.region == region)
        assert face.max_vertex_displacement == pytest.approx(distance, rel=1e-6)
    else:
        assert report.fallback == "faceted"
        assert f"region {region}: a facets patch vertex is" in report.fallback_reason


def test_a_patch_missing_a_triangle_is_not_written_mixed(tmp_path):
    ir, tris, region = _largest_patch("countersink", 3, interior=False)
    patch = ir.regions[region]
    patch.triangles = patch.triangles[1:]
    patch.surface = Facets(patch.surface.vertices, patch.surface.faces[1:])
    report = step.write(ir, tmp_path / "hole.step", mesh=tris)
    assert not (report.valid and report.fallback is None)


def test_a_flipped_patch_on_a_planar_part_falls_back(tmp_path):
    shape = Box(10, 10, 10) - Pos(0, 0, 5) * Box(4, 4, 4)
    mesh = tessellate(shape, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = build_oracle_ir(mesh)
    region = _top_plane(ir)
    ir = force_facets(ir, region, tris)
    good = step.write(ir, tmp_path / "good.step", mesh=tris)
    assert good.valid and good.fallback is None and good.faceted_regions == 1
    patch = ir.regions[region].surface
    ir.regions[region].surface = Facets(patch.vertices, [f[::-1] for f in patch.faces])
    for mesh_arg in (tris, None):
        report = step.write(ir, tmp_path / "flipped.step", mesh=mesh_arg)
        assert report.fallback == "faceted" or not report.valid
        reason = report.fallback_reason or "; ".join(report.issues)
        assert f"region {region}: facets triangles wind against their neighbours" in reason


def test_patches_over_half_the_triangles_write_the_faceted_solid(tmp_path):
    _, ir, tris, sphere = _forced("sphere", _of_type("sphere"))
    share = len(ir.regions[sphere].triangles) / len(tris)
    assert share > step.FACETED_SHARE
    report = step.write(ir, tmp_path / "big.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert "over the 50% share above which the faceted solid is written" in report.fallback_reason
    _, ir, tris, plane = _forced("sphere")
    assert len(ir.regions[plane].triangles) / len(tris) < step.FACETED_SHARE
    small = step.write(ir, tmp_path / "small.step", mesh=tris)
    assert small.valid and small.fallback is None and small.faceted_regions == 1
    unchecked = step.write(_forced("sphere", _of_type("sphere"))[1], tmp_path / "n.step")
    assert unchecked.valid and unchecked.fallback is None


def test_a_seam_split_never_folds_a_triangle(tmp_path, monkeypatch):
    _, ir, tris, _ = _forced("cylinder")
    original = curved._subdivide

    def outward(surface, p, q, gap, depth=0):
        pts = original(surface, p, q, gap, depth)
        if isinstance(surface, IrCylinder) and pts:
            pts[0] = pts[0] + 50.0 * geo.closest(surface, pts[0])[1]
        return pts

    monkeypatch.setattr(curved, "_subdivide", outward)
    report = step.write(ir, tmp_path / "fold.step", mesh=tris)
    assert report.fallback == "faceted"
    assert "folds a facets triangle" in report.fallback_reason


def test_only_moves_the_ir_records_are_excused_at_patch_corners():
    ir, tris, region = _largest_patch("corner_fillet", 3, interior=True)
    patch = ir.regions[region].surface
    corners = np.asarray(patch.vertices, dtype=float)[np.asarray(patch.faces, dtype=np.int64)]
    vertex = next(v for v in ir.vertices if region in v.regions)
    shifted = np.asarray(vertex.position) + np.array([0.0, 0.0, 0.3])
    vertex.source_positions = [tuple(shifted)]
    claimed = step._recorded_moves(ir, region, corners)
    at_vertex = np.all(corners == np.asarray(vertex.position), axis=2)
    assert at_vertex.any()
    assert np.allclose(claimed[at_vertex], 0.3)
    surfaces = [
        ir.regions[a + b - region]
        for a, b in (adj.regions for adj in ir.adjacencies)
        if region in (a, b)
    ]
    bound = max(s.residual.max for s in surfaces if s.residual is not None)
    assert claimed[~at_vertex].max() <= bound


def _forged_boundary_move():
    _, ir, tris, region = _forced("cylinder")
    surface = ir.regions[region].surface
    cylinder = next(
        ir.regions[a + b - region]
        for a, b in (adj.regions for adj in ir.adjacencies)
        if region in (a, b) and ir.regions[a + b - region].surface.type == "cylinder"
    )
    positions = {tuple(v.position) for v in ir.vertices}
    index = next(
        i
        for i, p in enumerate(surface.vertices)
        if tuple(p) not in positions
        and abs(geo.distance(cylinder.surface, np.asarray(p, dtype=float)))
        <= ir.tolerances.vertex_merge
    )
    p = np.asarray(surface.vertices[index], dtype=float)
    normal = geo.closest(cylinder.surface, p)[1]
    moved = geo.closest(cylinder.surface, p + 0.3 * geo.perpendicular(normal))[0]
    old = tuple(round(float(x), 6) for x in p)
    surface.vertices[index] = tuple(float(x) for x in moved)
    for adj in ir.adjacencies:
        for boundary in adj.boundaries:
            for i, point in enumerate(boundary.points):
                if tuple(round(float(x), 6) for x in point) == old:
                    boundary.points[i] = tuple(float(x) for x in moved)
    cylinder.residual.max = 0.3
    ir.validate()
    return ir, tris, region


def test_a_move_the_ir_claims_beyond_ten_times_tolerance_falls_back(tmp_path, monkeypatch):
    ir, tris, region = _forged_boundary_move()
    report = step.write(ir, tmp_path / "forged.step", mesh=tris)
    assert report.fallback == "faceted", report.fallback_reason
    assert f"region {region}: a facets patch vertex is" in report.fallback_reason
    assert "beyond the IR's recorded move" in report.fallback_reason

    ir, tris, _ = _forged_boundary_move()
    monkeypatch.setattr(topology, "RECORDED_MOVE_FACTOR", 1e9)
    assert step.write(ir, tmp_path / "unclamped.step", mesh=tris).fallback is None
