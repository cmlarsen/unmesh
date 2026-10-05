import copy
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("OCP")

from build123d import (  # noqa: E402
    Align,
    Axis,
    Box,
    Cone,
    Cylinder,
    Pos,
    Rot,
    Sphere,
    Torus,
    fillet,
)
from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.Geom import Geom_Circle, Geom_Ellipse  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

import unmesh.ir as uir  # noqa: E402
import unmesh.step as step  # noqa: E402
from unmesh._writer import curved, occ  # noqa: E402
from unmesh._writer import geometry as geo  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402
from unmesh_harness.metrics.edges import brep_counts, edge_hausdorff  # noqa: E402
from unmesh_harness.oracle import build_oracle_ir  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "ir"
Z = (0.0, 0.0, 1.0)


def plane(origin, normal):
    return uir.Plane(tuple(map(float, origin)), tuple(map(float, geo.unit(normal))))


def cylinder(origin, axis, r, orientation="same"):
    return uir.Cylinder(tuple(map(float, origin)), tuple(geo.unit(axis)), r, orientation)


def cone(apex, axis, half_angle):
    return uir.Cone(tuple(map(float, apex)), tuple(geo.unit(axis)), half_angle, "same")


def ring(center, axis, radius, n=48):
    axis = geo.unit(axis)
    x = geo.perpendicular(axis)
    y = np.cross(axis, x)
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return [np.asarray(center) + radius * (math.cos(a) * x + math.sin(a) * y) for a in t]


def on_both(curve, sa, sb, lo=None, hi=None, n=50):
    lo = curve.FirstParameter() if lo is None else lo
    hi = curve.LastParameter() if hi is None else hi
    pts = [curve.Value(float(t)).Coord() for t in np.linspace(lo, hi, n)]
    return max(max(geo.distance(sa, p), geo.distance(sb, p)) for p in pts)


def intersect(sa, sb):
    return curved.intersection_curves(curved.geom_surface(sa), curved.geom_surface(sb))


PAIRS = {
    "plane-cylinder": (
        plane((0, 0, 3), Z),
        cylinder((0, 0, 0), Z, 4.0),
        ring((0, 0, 3), Z, 4.0),
        Geom_Circle,
    ),
    "plane-cylinder-oblique": (
        plane((0, 0, 3), (0.0, math.sin(0.4), math.cos(0.4))),
        cylinder((0, 0, 0), Z, 4.0),
        [
            np.array([4 * math.cos(a), 4 * math.sin(a), 3 - 4 * math.sin(a) * math.tan(0.4)])
            for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)
        ],
        Geom_Ellipse,
    ),
    "plane-cone": (
        plane((0, 0, 5), Z),
        cone((0, 0, 0), Z, 0.5),
        ring((0, 0, 5), Z, 5 * math.tan(0.5)),
        Geom_Circle,
    ),
    "cone-cylinder-coaxial": (
        cone((1, 2, 0), Z, 0.7),
        cylinder((1, 2, -4), Z, 2.0),
        ring((1, 2, 2.0 / math.tan(0.7)), Z, 2.0),
        Geom_Circle,
    ),
    "plane-sphere": (
        plane((0, 0, 2), Z),
        uir.Sphere((0.0, 0.0, 0.0), 5.0, "same"),
        ring((0, 0, 2), Z, math.sqrt(21.0)),
        Geom_Circle,
    ),
    "plane-torus-outer": (
        plane((0, 0, 0), Z),
        uir.Torus((0.0, 0.0, 0.0), Z, 10.0, 3.0, "same"),
        ring((0, 0, 0), Z, 13.0),
        Geom_Circle,
    ),
    "plane-torus-inner": (
        plane((0, 0, 0), Z),
        uir.Torus((0.0, 0.0, 0.0), Z, 10.0, 3.0, "same"),
        ring((0, 0, 0), Z, 7.0),
        Geom_Circle,
    ),
}


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_conic_intersection_branch_matches_the_polyline(name):
    sa, sb, points, kind = PAIRS[name]
    curves = intersect(sa, sb)
    assert curves
    curve, dev = curved.select_branch(curves, points)
    assert dev < 1e-9
    assert isinstance(curve, kind)
    assert on_both(curve, sa, sb) < 1e-9


def test_branch_selection_picks_the_nearest_of_two_circles():
    sa, sb, outer, _ = PAIRS["plane-torus-outer"]
    inner = PAIRS["plane-torus-inner"][2]
    curves = intersect(sa, sb)
    assert len(curves) == 2
    a, _ = curved.select_branch(curves, outer)
    b, _ = curved.select_branch(curves, inner)
    assert a.Radius() == pytest.approx(13.0) and b.Radius() == pytest.approx(7.0)
    assert curved.select_branch(curves[::-1], outer)[0].Radius() == pytest.approx(13.0)


def branch_loop(r_main, r_branch, axis, n=64):
    axis = geo.unit(axis)
    x = geo.perpendicular(axis)
    y = np.cross(axis, x)
    out = []
    for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
        p = r_branch * (math.cos(a) * x + math.sin(a) * y)
        lo, hi = 0.0, 4 * r_main
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            q = p + mid * axis
            lo, hi = (mid, hi) if q[0] ** 2 + q[1] ** 2 < r_main**2 else (lo, mid)
        out.append(p + 0.5 * (lo + hi) * axis)
    return out


@pytest.mark.parametrize("tilt", [0.0, 0.5], ids=["perpendicular", "oblique"])
def test_cylinder_cylinder_branches_cover_the_polyline(tilt):
    axis = (math.cos(tilt), 0.0, math.sin(tilt))
    main = cylinder((0, 0, 0), Z, 5.0)
    branch = cylinder((0, 0, 0), axis, 2.0)
    points = branch_loop(5.0, 2.0, axis)
    assert max(max(geo.distance(main, p), geo.distance(branch, p)) for p in points) < 1e-9
    curves = intersect(main, branch)
    assert curves
    assert max(min(curved.nearest(c, p)[1] for c in curves) for p in points) < 1e-5
    assert all(on_both(c, main, branch) < 1e-5 for c in curves)


def test_trimming_follows_the_polyline_direction():
    sa, sb, points, _ = PAIRS["plane-cylinder"]
    curve, _ = curved.select_branch(intersect(sa, sb), points)
    arc = points[:13]
    t0, t1, _ = curved.trim_open(curve, arc[0], arc[-1], arc, False)
    assert abs(t1 - t0) == pytest.approx(math.pi / 2, abs=1e-9)
    s0, s1, _ = curved.trim_open(curve, arc[-1], arc[0], arc[::-1], False)
    assert abs(s1 - s0) == pytest.approx(math.pi / 2, abs=1e-9)
    assert (t1 - t0) * (s1 - s0) < 0
    assert curved._range_deviation(curve, min(t0, t1), max(t0, t1), arc) < 1e-9
    assert curved._range_deviation(curve, min(t0, t1), max(t0, t1), points[20:30]) > 1.0


SHAPES = {
    "hemisphere_up": lambda: Sphere(5) - Pos(0, 0, -10) * Box(30, 30, 20),
    "hemisphere_down": lambda: Sphere(5) - Pos(0, 0, 10) * Box(30, 30, 20),
    "sphere_cap": lambda: Sphere(5) - Pos(0, 0, -7) * Box(30, 30, 20),
    "sphere_band": lambda: (
        Sphere(5) - Pos(0, 0, 12) * Box(30, 30, 20) - Pos(0, 0, -11) * Box(30, 30, 20)
    ),
    "sphere_pocket": lambda: Box(20, 20, 10) - Pos(0, 0, 5) * Sphere(4),
    "cone_apex_up": lambda: Cone(5, 0, 8),
    "cone_apex_down": lambda: Cone(0, 5, 8),
    "half_torus": lambda: Torus(10, 3) - Pos(0, 0, -10) * Box(40, 40, 20),
    "torus_band": lambda: (
        Torus(10, 3) - Pos(0, 0, -10) * Box(40, 40, 20) - Pos(0, 0, 11.5) * Box(40, 40, 20)
    ),
    "oblique_boss": lambda: (
        Box(30, 30, 4)
        + Pos(0, 0, 2)
        * Rot(20, 0, 0)
        * Cylinder(3, 12, align=(Align.CENTER, Align.CENTER, Align.MIN))
    ),
    "cross_bores": lambda: (
        Box(30, 30, 30) - Rot(0, 90, 0) * Cylinder(6, 40) - Rot(90, 0, 0) * Cylinder(3, 40)
    ),
}


def oracle(shape):
    return build_oracle_ir(tessellate(shape, 0.01, 0.2))


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_seams_poles_and_bands_write_exactly(name, tmp_path):
    shape = SHAPES[name]()
    path = tmp_path / f"{name}.step"
    report = step.write(oracle(shape), path)
    assert report.valid and report.verified, report.issues
    assert all(e.kind == "interpolated" for e in report.edge_fallbacks)
    assert all(e.intersection_distance < 1e-5 for e in report.edge_fallbacks)
    assert bool(report.edge_fallbacks) == (name == "cross_bores")
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-6)
    written = occ.read_step(path)
    assert brep_counts(written) == brep_counts(shape)
    h = edge_hausdorff(written, shape)
    assert h["edges"] == h["matched"] and h["max"] < 1e-6


def curve_types(path):
    shape = occ.read_step(path)
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    kinds = set()
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        ex.Next()
        if not BRep_Tool.Degenerated_s(e):
            kinds.add(type(BRep_Tool.Curve_s(e, 0.0, 0.0)).__name__)
    return kinds


def test_circles_stay_circles(tmp_path):
    path = tmp_path / "t.step"
    report = step.write(oracle(SHAPES["half_torus"]()), path)
    assert report.valid
    assert curve_types(path) == {"Geom_Circle"}


def test_failed_intersection_is_projected_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(curved, "intersection_curves", lambda a, b: [])
    ir = uir.Ir.loads((FIXTURES / "plate_with_bore.json").read_text())
    report = step.write(ir, tmp_path / "p.step")
    assert report.valid and report.verified
    assert len(report.edge_fallbacks) == 2
    assert {e.regions for e in report.edge_fallbacks} == {(4, 6), (5, 6)}
    assert all(e.reason == "surface intersection failed" for e in report.edge_fallbacks)
    assert all(e.max_deviation < 1e-4 for e in report.edge_fallbacks)
    assert report.shells[0].volume == pytest.approx(30 * 20 * 6 - math.pi * 16 * 6, rel=1e-6)


def test_stray_branch_is_projected_and_recorded(tmp_path, monkeypatch):
    real = curved.intersection_curves

    def shifted(a, b):
        out = []
        for c in real(a, b):
            c = c.Copy()
            c.Translate(__import__("OCP.gp", fromlist=["gp_Vec"]).gp_Vec(0.0, 0.0, 0.5))
            out.append(c)
        return out

    monkeypatch.setattr(curved, "intersection_curves", shifted)
    ir = uir.Ir.loads((FIXTURES / "plate_with_bore.json").read_text())
    report = step.write(ir, tmp_path / "p.step")
    assert report.valid
    assert len(report.edge_fallbacks) == 2
    assert all("nearest intersection branch is 0.5" in e.reason for e in report.edge_fallbacks)


def test_an_ir_orientation_the_geometry_contradicts_is_not_written(tmp_path):
    ir = uir.Ir.loads((FIXTURES / "plate_with_bore.json").read_text())
    bad = copy.deepcopy(ir)
    bad.regions[6].surface.orientation = "same"
    report = step.write(bad, tmp_path / "bad.step")
    assert not report.valid
    assert not (tmp_path / "bad.step").exists()


def write_oracle(shape, tmp_path, deflection=(0.01, 0.2), name="t"):
    ir = build_oracle_ir(tessellate(shape, *deflection))
    path = tmp_path / f"{name}.step"
    return ir, step.write(ir, path), path


def assert_exact(shape, report, path, hausdorff=1e-6):
    assert report.valid and report.verified, report.issues
    assert report.fallback is None
    written = occ.read_step(path)
    assert brep_counts(written) == brep_counts(shape)
    h = edge_hausdorff(written, shape)
    assert h["edges"] == h["truth_edges"] == h["matched"]
    assert h["max"] < hausdorff


def tangent_pairs(ir, report):
    kinds = {}
    for t in report.tangent_edges:
        a, b = t.regions
        pair = tuple(sorted((ir.regions[a].surface.type, ir.regions[b].surface.type)))
        kinds.setdefault(pair, set()).add(t.curve)
    return kinds


def test_plane_cylinder_fillet_is_a_line(tmp_path):
    shape = fillet(Box(20, 16, 8).edges().filter_by(Axis.Z)[0], 3)
    ir, report, path = write_oracle(shape, tmp_path)
    assert_exact(shape, report, path)
    assert tangent_pairs(ir, report) == {("cylinder", "plane"): {"line"}}
    assert all(t.max_deviation < 1e-9 for t in report.tangent_edges)


def test_disc_fillet_circles_on_plane_torus_and_cylinder_torus(tmp_path):
    disc = Cylinder(12, 8)
    shape = fillet(disc.edges().group_by(Axis.Z)[-1], 2.5)
    ir, report, path = write_oracle(shape, tmp_path)
    assert_exact(shape, report, path)
    assert tangent_pairs(ir, report) == {
        ("plane", "torus"): {"circle"},
        ("cylinder", "torus"): {"circle"},
    }
    assert curve_types(path) == {"Geom_Circle", "Geom_Line"}


def test_tangent_edge_without_an_analytic_fit_is_a_bspline(tmp_path, monkeypatch):
    monkeypatch.setattr(curved, "fit_line", lambda points: None)
    monkeypatch.setattr(curved, "fit_circle", lambda points, closed: None)
    shape = fillet(Cylinder(12, 8).edges().group_by(Axis.Z)[-1], 2.5)
    ir, report, path = write_oracle(shape, tmp_path)
    assert_exact(shape, report, path, hausdorff=1e-4)
    assert {t.curve for t in report.tangent_edges} == {"bspline"}
    assert all(t.max_deviation < 1e-4 for t in report.tangent_edges)
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-6)


def test_corner_blend_sphere_closes_at_a_pole_on_its_vertex(tmp_path):
    shape = fillet(Box(20, 16, 12).edges(), 2)
    ir, report, path = write_oracle(shape, tmp_path)
    assert_exact(shape, report, path)
    assert tangent_pairs(ir, report) == {
        ("cylinder", "plane"): {"line"},
        ("cylinder", "sphere"): {"circle"},
    }


def tee():
    return Cylinder(5, 20) + Rot(0, 90, 0) * Cylinder(5, 20)


def boss_on_pipe():
    return Rot(0, 90, 0) * Cylinder(5, 20) + Pos(0, 0, 7.5) * Cylinder(5, 15)


def union_distance(written, truth) -> float:
    from unmesh_harness.metrics.edges import _distances, _sample, boundary_edges

    theirs = boundary_edges(truth.wrapped)
    return max(
        float(min(_distances(_sample(e, 200)[k : k + 1], t)[0] for t in theirs))
        for e in boundary_edges(written)
        for k in range(200)
    )


@pytest.mark.parametrize("make", [tee, boss_on_pipe], ids=["equal_tee", "boss_on_pipe"])
def test_kind_change_edges_follow_the_intersection(make, tmp_path):
    shape = make()
    ir, report, path = write_oracle(shape, tmp_path, (0.001, 0.1))
    assert any(v.role == "kind_change" for v in ir.vertices)
    assert report.valid and report.verified and report.fallback is None, report.issues
    assert {t.curve for t in report.tangent_edges} == {"bspline"}
    assert all(e.intersection_distance < 1e-6 for e in report.edge_fallbacks)
    assert report.max_shape_tolerance <= 1e-6
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-9)
    assert union_distance(occ.read_step(path), shape) < 1e-6


def test_interpolated_edges_are_recorded_with_their_distance_to_the_intersection(tmp_path):
    shape = boss_on_pipe()
    ir, report, path = write_oracle(shape, tmp_path)
    assert report.valid and report.verified, report.issues
    (edge,) = report.edge_fallbacks
    assert edge.kind == "interpolated"
    assert 0 < edge.intersection_distance < 1e-5
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-8)


def test_branch_distance_sees_a_bump_between_samples():
    sa, sb, points, _ = PAIRS["plane-cylinder"]
    circle, _ = curved.select_branch(intersect(sa, sb), points)
    pts = [np.asarray(circle.Value(t).Coord()) for t in np.linspace(0, 1, 9)]
    pts[4] = pts[4] + 0.01 * geo.unit(pts[4] - np.asarray(circle.Location().Coord()))
    bumped = curved._interpolate(pts, False)
    assert curved.branch_distance(bumped, [circle], 400) == pytest.approx(0.01, rel=0.05)


def test_an_inconsistent_orientation_on_a_tangent_boundary_is_not_written(tmp_path):
    shape = Cylinder(5, 10) + Pos(0, 0, 5) * Sphere(5)
    ir = oracle(shape)
    sphere = next(r for r in ir.regions if r.surface.type == "sphere")
    sphere.surface.orientation = "reversed"
    report = step.write(ir, tmp_path / "bad.step")
    assert not report.valid
    assert any("orientations meet at 180 degrees" in i for i in report.issues)


def notch():
    return Box(30, 30, 10) - Pos(15, 0, 0) * Cylinder(5, 20)


def test_a_wrong_trim_of_the_intersection_is_projected_and_recorded(tmp_path, monkeypatch):
    real = curved.trim_open

    def complement(curve, p0, p1, points, same_vertex):
        t0, t1, d = real(curve, p0, p1, points, same_vertex)
        if curve.IsPeriodic():
            t1 = t1 - curve.Period() if t1 > t0 else t1 + curve.Period()
        return t0, t1, d

    monkeypatch.setattr(curved, "trim_open", complement)
    shape = notch()
    ir, report, path = write_oracle(shape, tmp_path)
    assert report.valid and report.verified, report.issues
    assert report.edge_fallbacks
    assert all("trimmed intersection is" in e.reason for e in report.edge_fallbacks)
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-6)


def test_a_face_built_on_the_wrong_side_of_its_boundary_is_not_written(tmp_path, monkeypatch):
    real = curved._Builder.curved_face

    def flipped(self, r, s, loops):
        if s.type == "sphere":
            loops = [[curved._reverse_segment(seg) for seg in reversed(lp)] for lp in loops]
        return real(self, r, s, loops)

    monkeypatch.setattr(curved._Builder, "curved_face", flipped)
    report = step.write(oracle(Cylinder(5, 10) + Pos(0, 0, 5) * Sphere(5)), tmp_path / "w.step")
    assert not report.valid
    assert any("is not on the IR's side of its boundary" in i for i in report.issues)


def test_seam_vertex_is_snapped_onto_the_seam_when_refine_leaves_it_off(monkeypatch):
    shape = Cylinder(5, 30) - Pos(0, 0, 8) * Rot(25, 0, 0) * Box(30, 30, 20)
    ir = oracle(shape)
    limit = 5e-3
    from unmesh._writer import topology

    vpos, vmoved, bad = topology.vertex_positions(ir, limit)
    b = curved._Builder(ir, 0, *curved.refine_vertices(ir, vpos, vmoved, bad, limit), limit, None)
    b.collect()
    b.intersect()
    e = next(
        e
        for e in b.edges
        if e.closed and e.curve is not None and "Ellipse" in type(e.curve).__name__
    )
    r = next(r for r in (e.a, e.b) if ir.regions[r].surface.type == "cylinder")
    b.frames[r] = curved._Frame(geo.unit(ir.regions[r].surface.axis), geo.unit((0.6, 0.8, 0.0)))
    monkeypatch.setattr(curved.geo, "refine", lambda surfaces, p0: np.asarray(p0, dtype=float))
    p = b.seam_point(e, r)
    origin = geo.origin_of(ir.regions[r].surface)
    assert abs(geo.angle_about(p, origin, b.frames[r].axis, b.frames[r].xdir)) < 1e-12
    assert curved.nearest(e.curve, p)[1] < 1e-12


def flipped_pocket(size=20.0, radius=4.0):
    shape = Box(size, size, 10) - Pos(0, 0, 5) * Sphere(radius)
    mesh = tessellate(shape, 0.01, 0.2)
    ir = build_oracle_ir(mesh)
    next(r for r in ir.regions if r.surface.type == "sphere").surface.orientation = "same"
    return shape, ir, np.asarray(mesh.tris).reshape(-1, 3, 3)


def test_a_volume_the_mesh_contradicts_falls_back(tmp_path):
    shape, ir, tris = flipped_pocket()
    report = step.write(ir, tmp_path / "p.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert report.volume_checked_against_input
    assert "differs from mesh volume" in report.fallback_reason
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-3)


def test_without_a_mesh_the_volume_is_reported_unchecked(tmp_path):
    _, ir, _ = flipped_pocket()
    report = step.write(ir, tmp_path / "p.step")
    assert report.valid and report.fallback is None
    assert not report.volume_checked_against_input


def test_tangent_fit_prefers_a_line_on_a_straight_tangency():
    wall = plane((5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    fillet_ = cylinder((0.0, 0.0, 0.0), Z, 5.0)
    wobble = 1e-6 * (np.sin(np.arange(11)) + 0.3 * np.linspace(-1.0, 1.0, 11) ** 2)
    pts = [np.array([5.0, w, z]) for w, z in zip(wobble, np.linspace(0.0, 10.0, 11), strict=True)]
    circle = curved.fit_circle(pts, False)
    line_error = curved.curve_error(curved.fit_line(pts), [wall, fillet_], pts, False)[0]
    assert curved.curve_error(circle, [wall, fillet_], pts, False)[0] < line_error
    kind, curve, err, _ = curved.tangent_curve([wall, fillet_], pts, False, 5e-3)
    assert kind == "line" and err == pytest.approx(line_error)


def test_tangent_fit_prefers_a_circle_over_a_line_within_the_limit():
    top = plane((0.0, 0.0, 4.0), Z)
    tube = uir.Torus((0.0, 0.0, 2.0), Z, 20.0, 2.0, "same")
    pts = [np.array([20.0 * math.cos(a), 20.0 * math.sin(a), 4.0]) for a in np.linspace(0, 0.03, 7)]
    line = curved.fit_line(pts)
    assert curved.curve_error(line, [top, tube], pts, False)[0] < 5e-3
    kind, curve, err, _ = curved.tangent_curve([top, tube], pts, False, 5e-3)
    assert kind == "circle" and err < 1e-9
    assert curve.Radius() == pytest.approx(20.0, abs=1e-9)


def test_tangent_fit_gives_up_when_neither_fits():
    sa = cylinder((0.0, 0.0, 0.0), Z, 5.0)
    sb = cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5.0)
    pts = [
        np.array([5 * math.cos(a), 5 * math.sin(a), abs(5 * math.cos(a))])
        for a in np.linspace(1.2, 1.9, 9)
    ]
    assert curved.tangent_curve([sa, sb], pts, False, 1e-4) is None


@pytest.mark.parametrize(
    ("size", "radius"),
    [(100.0, 3.5), (20.0, 1.0), (40.0, 0.2), (40.0, 0.5), (40.0, 2.0)],
)
def test_a_small_flipped_pocket_is_caught_face_by_face(size, radius, tmp_path):
    shape, ir, tris = flipped_pocket(size, radius)
    report = step.write(ir, tmp_path / "p.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert "the written face's flux about its centre" in report.fallback_reason
    assert report.shells[0].volume == pytest.approx(shape.volume, rel=1e-3)
    unflipped = build_oracle_ir(tessellate(shape, 0.01, 0.2))
    good = step.write(unflipped, tmp_path / "g.step", mesh=tris)
    assert good.valid and good.fallback is None and good.volume_checked_against_input
