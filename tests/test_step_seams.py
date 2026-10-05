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
from unmesh._writer import curved, occ  # noqa: E402
from unmesh._writer import geometry as geo  # noqa: E402
from unmesh.ir import Cylinder as IrCylinder  # noqa: E402
from unmesh.ir import Facets  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402
from unmesh_harness.oracle import build_oracle_ir, force_facets  # noqa: E402

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


def test_a_bulging_facets_patch_fails_verification(tmp_path):
    _, ir, tris, sphere = _forced("sphere", _of_type("sphere"))
    patch = ir.regions[sphere].surface
    boundary = {tuple(p) for adj in ir.adjacencies for bd in adj.boundaries for p in bd.points}
    inner = next(i for i, p in enumerate(patch.vertices) if tuple(p) not in boundary)
    p = np.asarray(patch.vertices[inner])
    patch.vertices[inner] = tuple(p + 2e-3 * geo.unit(p))
    report = step.write(ir, tmp_path / "bulge.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert "differs from mesh volume" in report.fallback_reason


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
