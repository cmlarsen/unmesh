import copy
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("OCP")

from build123d import Align, Box, Cone, Cylinder, Pos, Rot, Sphere, Torus  # noqa: E402
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
    assert not report.edge_fallbacks
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
