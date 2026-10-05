from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepLib import BRepLib
from OCP.Geom import (
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
    Geom_TrimmedCurve,
)
from OCP.Geom2d import Geom2d_Line
from OCP.GeomAPI import GeomAPI_Interpolate, GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
from OCP.GeomProjLib import GeomProjLib
from OCP.gp import gp_Ax3, gp_Dir, gp_Dir2d, gp_Pnt, gp_Pnt2d, gp_Vec2d
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TopAbs import TopAbs_FORWARD, TopAbs_IN, TopAbs_OUT, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Wire

from unmesh.ir import Cone, Cylinder, Facets, Ir, Plane, Sphere, Torus

from . import geometry as geo
from .topology import BuildError, SeamGap, _boundary_nodes, _signed_area

INTERSECTION_TOLERANCE = 1e-7
EDGE_TOLERANCE = 1e-7
SEAM_TOLERANCE = 1e-6
BRANCH_SAMPLES = 24
SEAM_ITERATIONS = 60
TWO_PI = 2.0 * math.pi
TANGENT_REASON = "tangent edges not supported yet (#33)"
FACETS_REASON = "facets regions next to curved surfaces are not supported yet (#34)"


@dataclass
class ProjectedEdge:
    regions: tuple[int, int]
    reason: str
    max_deviation: float


@dataclass
class CurvedShell:
    shell: TopoDS_Shell
    mapping: list
    seams: list[SeamGap] = field(default_factory=list)
    vertex_displacement: dict[int, float] = field(default_factory=dict)
    boundary_deviation: dict[int, float] = field(default_factory=dict)
    projected: list[ProjectedEdge] = field(default_factory=list)


def _pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(d) -> gp_Dir:
    return gp_Dir(float(d[0]), float(d[1]), float(d[2]))


def _xyz(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def refine_vertices(ir: Ir, vpos, vmoved, bad, limit: float):
    vpos = list(vpos)
    vmoved = list(vmoved)
    bad = dict(bad)
    for v in ir.vertices:
        surfaces = [ir.regions[r].surface for r in v.regions]
        if not any(geo.is_curved(s) for s in surfaces):
            continue
        analytic = [s for s in surfaces if geo.is_analytic(s)]
        p0 = np.asarray(v.position, dtype=float)
        x = geo.refine(analytic, p0) if len(analytic) >= 2 else p0
        gap = float(np.linalg.norm(x - p0))
        bad.pop(v.id, None)
        if gap > limit:
            bad[v.id] = (
                f"vertex {v.id}: surface intersection is {gap:.6g} from its IR position,"
                f" over the {limit:.6g} limit"
            )
        vpos[v.id] = x
        vmoved[v.id] = gap
    return vpos, vmoved, bad


def geom_surface(surface, axis=None, xdir=None):
    if isinstance(surface, Plane):
        return Geom_Plane(gp_Ax3(_pnt(surface.origin), _dir(geo.unit(surface.normal))))
    if axis is None:
        axis = geo.axis_of(surface)
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0])
    if xdir is None:
        xdir = geo.perpendicular(axis)
    origin = geo.origin_of(surface)
    ax3 = gp_Ax3(_pnt(origin), _dir(axis), _dir(xdir))
    if isinstance(surface, Cylinder):
        return Geom_CylindricalSurface(ax3, float(surface.radius))
    if isinstance(surface, Cone):
        return Geom_ConicalSurface(ax3, float(surface.half_angle), 0.0)
    if isinstance(surface, Sphere):
        return Geom_SphericalSurface(ax3, float(surface.radius))
    if isinstance(surface, Torus):
        return Geom_ToroidalSurface(ax3, float(surface.major_radius), float(surface.minor_radius))
    raise BuildError(f"no OCCT surface for {surface.type}")


def intersection_curves(s1, s2) -> list:
    try:
        inter = GeomAPI_IntSS(s1, s2, INTERSECTION_TOLERANCE)
    except Exception:
        return []
    if not inter.IsDone():
        return []
    out = []
    for k in range(1, inter.NbLines() + 1):
        c = inter.Line(k)
        if isinstance(c, Geom_TrimmedCurve) and c.BasisCurve().IsPeriodic():
            c = c.BasisCurve()
        out.append(c)
    return out


def _bounded(curve) -> bool:
    return not curve.IsPeriodic() and abs(curve.FirstParameter()) < 1e50


def nearest(curve, p) -> tuple[float, float]:
    best_t, best_d = None, math.inf
    try:
        proj = GeomAPI_ProjectPointOnCurve(_pnt(p), curve)
        if proj.NbPoints() > 0:
            best_t, best_d = proj.LowerDistanceParameter(), proj.LowerDistance()
    except Exception:
        pass
    if _bounded(curve):
        for t in (curve.FirstParameter(), curve.LastParameter()):
            d = float(np.linalg.norm(_xyz(curve.Value(t)) - p))
            if d < best_d:
                best_t, best_d = t, d
    return best_t, best_d


def _samples(points: list[np.ndarray], count: int = BRANCH_SAMPLES) -> list[np.ndarray]:
    if len(points) <= count:
        return points
    idx = np.unique(np.linspace(0, len(points) - 1, count).round().astype(int))
    return [points[i] for i in idx]


def select_branch(curves, points: list[np.ndarray]):
    best, best_dev = None, math.inf
    probe = _samples(points)
    for c in curves:
        dev = max(nearest(c, p)[1] for p in probe)
        if dev < best_dev:
            best, best_dev = c, dev
    return best, best_dev


def _tangent(curve, t) -> np.ndarray:
    v = curve.DN(t, 1)
    return np.array([v.X(), v.Y(), v.Z()])


def _wrap(t: float, lo: float, period: float) -> float:
    return lo + (t - lo) % period


def trim_open(curve, p0, p1, points, same_vertex: bool):
    t0, d0 = nearest(curve, p0)
    t1, d1 = nearest(curve, p1)
    if t0 is None or t1 is None:
        raise ValueError("vertices do not project onto the intersection curve")
    travel = points[1] - points[0]
    forward = float(_tangent(curve, t0) @ travel) > 0
    if curve.IsPeriodic():
        period = curve.Period()
        if same_vertex:
            t1 = t0 + period if forward else t0 - period
        elif forward:
            t1 = _wrap(t1, t0, period)
        else:
            t1 = _wrap(t1, t0 - period, period)
    elif same_vertex or (t1 > t0) != forward:
        raise ValueError("the intersection branch does not run along the boundary")
    return t0, t1, max(d0, d1)


def _range_deviation(curve, lo: float, hi: float, points) -> float:
    period = curve.Period() if curve.IsPeriodic() else None
    ends = [_xyz(curve.Value(lo)), _xyz(curve.Value(hi))]
    worst = 0.0
    for p in points:
        t, d = nearest(curve, p)
        inside = False
        if t is not None:
            if period is not None:
                inside = _wrap(t, lo, period) <= hi + 1e-12 or abs(hi - lo - period) < 1e-9
            else:
                inside = lo - 1e-12 <= t <= hi + 1e-12
        if not inside:
            d = min(float(np.linalg.norm(e - p)) for e in ends)
        worst = max(worst, d)
    return worst


def seam_parameter(curve, t: float, origin, axis, xdir) -> float:
    def f(s):
        return geo.angle_about(_xyz(curve.Value(s)), origin, axis, xdir)

    h = 1e-6 * (curve.Period() if curve.IsPeriodic() else 1.0)
    for _ in range(SEAM_ITERATIONS):
        a = f(t)
        if abs(a) < 1e-15:
            break
        slope = (f(t + h) - f(t - h)) / (2 * h)
        if slope == 0:
            break
        step = a / slope
        t -= step
        if abs(step) < 1e-15:
            break
    return t


def polyline_seam_point(points, origin, axis, xdir) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    ang = [geo.angle_about(p, origin, axis, xdir) for p in pts]
    for i in range(n):
        j = (i + 1) % n
        a, b = ang[i], ang[j]
        if a == 0.0:
            return pts[i]
        if (a < 0) != (b < 0) and abs(a - b) < math.pi:
            s = a / (a - b)
            return pts[i] + s * (pts[j] - pts[i])
    raise BuildError("closed boundary does not cross the seam")


def projected_curve(surfaces, nodes, closed: bool):
    inner = nodes[1:] if closed else nodes[1:-1]
    fixed = [nodes[0], *(geo.refine(surfaces, p) for p in inner)]
    if not closed:
        fixed.append(nodes[-1])
    ring = [*fixed, fixed[0]] if closed else fixed
    pts = []
    for p, q in zip(ring, ring[1:], strict=False):
        pts.extend([p, geo.refine(surfaces, 0.5 * (p + q))])
    if not closed:
        pts.append(fixed[-1])
    arr = TColgp_HArray1OfPnt(1, len(pts))
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, _pnt(p))
    interp = GeomAPI_Interpolate(arr, closed, 1e-9)
    interp.Perform()
    if not interp.IsDone():
        raise BuildError("could not fit a curve through the boundary polyline")
    curve = interp.Curve()
    first, last = curve.FirstParameter(), curve.LastParameter()
    dev = 0.0
    for s in np.linspace(first, last, 8 * len(pts) + 1):
        q = _xyz(curve.Value(float(s)))
        dev = max(dev, *(geo.distance(srf, q) for srf in surfaces))
    return curve, dev


@dataclass
class _Edge:
    a: int
    b: int
    closed: bool
    start: int | None
    end: int | None
    points: list[np.ndarray]
    curved: bool
    curve: object = None
    fallback: str | None = None
    approximate: bool = False
    seam_point: np.ndarray | None = None
    edges: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    deviation: float = 0.0


@dataclass
class _Segment:
    edges: list
    start: int | None
    end: int | None
    nodes: list


@dataclass
class _Frame:
    axis: np.ndarray
    xdir: np.ndarray


def _reverse_segment(s: _Segment) -> _Segment:
    return _Segment(
        [TopoDS.Edge_s(e.Reversed()) for e in reversed(s.edges)], s.end, s.start, s.nodes[::-1]
    )


def _chain(segments: list[_Segment], normal_at, region: int) -> list[list[_Segment]]:
    loops = []
    open_segs = []
    for s in segments:
        if s.start is None:
            loops.append([s])
        else:
            open_segs.append(s)
    by_start: dict[int, list[int]] = {}
    for i, s in enumerate(open_segs):
        by_start.setdefault(s.start, []).append(i)
    used = [False] * len(open_segs)
    for first in range(len(open_segs)):
        if used[first]:
            continue
        used[first] = True
        cur = open_segs[first]
        loop = [cur]
        while cur.end != open_segs[first].start:
            options = [i for i in by_start.get(cur.end, []) if not used[i]]
            if not options:
                raise BuildError(f"region {region}: boundary loop does not close")
            d_in = geo.unit(cur.nodes[-1] - cur.nodes[-2])
            normal = normal_at(cur.nodes[-1])

            def left_turn(i, d_in=d_in, normal=normal):
                d_out = geo.unit(open_segs[i].nodes[1] - open_segs[i].nodes[0])
                return math.atan2(float(normal @ np.cross(d_in, d_out)), float(d_in @ d_out))

            pick = max(options, key=left_turn)
            used[pick] = True
            cur = open_segs[pick]
            loop.append(cur)
        loops.append(loop)
    return loops


def _loop_nodes(loop: list[_Segment]) -> list[np.ndarray]:
    out = []
    for s in loop:
        out.extend(s.nodes[:-1] if s.start is not None else s.nodes)
    return out


class _Builder:
    def __init__(self, ir: Ir, index: int, vpos, vmoved, bad, limit: float, pool):
        self.ir = ir
        self.index = index
        self.shell = ir.shells[index]
        self.members = set(self.shell.regions)
        self.vpos = vpos
        self.vmoved = vmoved
        self.bad = bad
        self.limit = limit
        self.pool = pool
        self.builder = BRep_Builder()
        self.frames: dict[int, _Frame] = {}
        self.surfaces: dict[int, object] = {}
        self.edges: list[_Edge] = []
        self.loose: list = []
        self.out = CurvedShell(TopoDS_Shell(), [])

    def surface(self, r):
        return self.ir.regions[r].surface

    def run(self) -> CurvedShell:
        for r in self.shell.regions:
            self.out.vertex_displacement[r] = 0.0
            self.out.boundary_deviation[r] = 0.0
        for v, m in zip(self.ir.vertices, self.vmoved, strict=True):
            for r in v.regions:
                if r in self.members:
                    if v.id in self.bad:
                        raise BuildError(self.bad[v.id])
                    self.out.vertex_displacement[r] = max(self.out.vertex_displacement[r], m)
        self.collect()
        self.intersect()
        self.choose_frames()
        self.make_edges()
        faces = [(r, f) for r in self.shell.regions for f in self.face(r)]
        self.builder.MakeShell(self.out.shell)
        for _, f in faces:
            self.builder.Add(self.out.shell, f)
        self.out.shell.Closed(self.shell.closed)
        for edge in self.loose:
            BRepLib.SameParameter_s(edge, EDGE_TOLERANCE)
        self.out.mapping = faces
        for r, f in faces:
            self.check_side(r, f)
        return self.out

    def collect(self):
        for adj in self.ir.adjacencies:
            a, b = adj.regions
            if a not in self.members:
                continue
            sa, sb = self.surface(a), self.surface(b)
            curved = geo.is_curved(sa) or geo.is_curved(sb)
            facets = isinstance(sa, Facets) or isinstance(sb, Facets)
            if curved and facets:
                raise BuildError(f"regions {a}/{b}: {FACETS_REASON}")
            for bd in adj.boundaries:
                if curved and bd.kind == "tangent":
                    raise BuildError(f"regions {a}/{b}: {TANGENT_REASON}")
                pts = [np.asarray(p, dtype=float) for p in bd.points]
                e = _Edge(a, b, bd.closed, bd.start_vertex, bd.end_vertex, pts, curved)
                if facets:
                    analytic = sb if isinstance(sa, Facets) else sa
                    n = geo.unit(analytic.normal)
                    o = np.asarray(analytic.origin, dtype=float)
                    gap = max(abs(float(n @ (p - o))) for p in pts)
                    self.out.seams.append(SeamGap((a, b), "plane", len(pts), gap))
                self.edges.append(e)
                if not curved:
                    e.nodes, e.deviation = _boundary_nodes(self.ir, bd, a, b, self.vpos, self.limit)
                    if bd.closed:
                        e.nodes = [*e.nodes, e.nodes[0]]

    def occ_surface(self, r):
        s = self.surfaces.get(r)
        if s is None:
            frame = self.frames.get(r)
            surf = self.surface(r)
            if frame is None:
                s = geom_surface(surf)
            else:
                s = geom_surface(surf, frame.axis, frame.xdir)
            self.surfaces[r] = s
        return s

    def intersect(self):
        for e in self.edges:
            if not e.curved:
                continue
            sa = geom_surface(self.surface(e.a))
            sb = geom_surface(self.surface(e.b))
            curves = intersection_curves(sa, sb)
            if not curves:
                e.fallback = "surface intersection failed"
                continue
            curve, dev = select_branch(curves, e.points)
            if dev <= self.limit and (curve.IsPeriodic() or not e.closed):
                e.curve = curve
                continue
            cover = max(min(nearest(c, p)[1] for c in curves) for p in _samples(e.points))
            if cover <= self.limit:
                e.approximate = True
                continue
            e.fallback = (
                f"nearest intersection branch is {cover:.3g} from the boundary,"
                f" over the {self.limit:.3g} limit"
            )

    def axis_of(self, r) -> np.ndarray:
        frame = self.frames.get(r)
        if frame is not None:
            return frame.axis
        s = self.surface(r)
        axis = geo.axis_of(s)
        if axis is not None:
            return axis
        for e in self.edges:
            if e.closed and r in (e.a, e.b):
                return geo.plane_normal(e.points)
        return np.array([0.0, 0.0, 1.0])

    def winds(self, e: _Edge, r: int) -> bool:
        s = self.surface(r)
        if not geo.is_curved(s) or not e.closed:
            return False
        return (
            geo.winding(
                e.points, geo.origin_of(s), self.axis_of(r), geo.perpendicular(self.axis_of(r))
            )
            != 0
        )

    def initial_frame(self, r) -> _Frame:
        axis = self.axis_of(r)
        origin = geo.origin_of(self.surface(r))
        xdir = geo.perpendicular(axis)
        angles = []
        for e in self.edges:
            if r in (e.a, e.b) and not (e.closed and self.winds(e, r)):
                angles.extend(geo.angle_about(p, origin, axis, xdir) for p in e.points)
        if angles:
            ang = np.sort(np.mod(np.asarray(angles), TWO_PI))
            gaps = np.diff(np.append(ang, ang[0] + TWO_PI))
            k = int(np.argmax(gaps))
            mid = ang[k] + gaps[k] / 2
            y = np.cross(axis, xdir)
            xdir = geo.unit(math.cos(mid) * xdir + math.sin(mid) * y)
        return _Frame(axis, xdir)

    def frame_through(self, r, point) -> _Frame:
        axis = self.axis_of(r)
        q = point - geo.origin_of(self.surface(r))
        x = q - (q @ axis) * axis
        if np.linalg.norm(x) < 1e-12:
            raise BuildError(f"region {r}: seam vertex on the axis")
        return _Frame(axis, geo.unit(x))

    def seam_point(self, e: _Edge, r: int) -> np.ndarray:
        frame = self.frames[r]
        origin = geo.origin_of(self.surface(r))
        p = polyline_seam_point(e.points, origin, frame.axis, frame.xdir)
        half = Plane(tuple(origin), tuple(np.cross(frame.axis, frame.xdir)))
        p = geo.refine([self.surface(e.a), self.surface(e.b), half], p)
        if e.curve is not None:
            t, _ = nearest(e.curve, p)
            t = seam_parameter(e.curve, t, origin, frame.axis, frame.xdir)
            p = _xyz(e.curve.Value(t))
        return p

    def on_seam(self, r, p) -> bool:
        frame = self.frames[r]
        origin = geo.origin_of(self.surface(r))
        q = p - origin
        rho = float(np.linalg.norm(q - (q @ frame.axis) * frame.axis))
        ang = geo.angle_about(p, origin, frame.axis, frame.xdir)
        return abs(ang) * rho <= SEAM_TOLERANCE

    def anchor(self, r) -> np.ndarray | None:
        segs = []
        for e in self.edges:
            if r not in (e.a, e.b):
                continue
            seg = _Segment([], e.start, e.end, list(e.points))
            segs.append(seg if r == e.a else _reverse_segment(seg))
        s = self.surface(r)
        try:
            loops = _chain(segs, lambda p: geo.outward(s, p), r)
        except BuildError:
            return None
        origin, axis = geo.origin_of(s), self.axis_of(r)
        for lp in loops:
            starts = [seg.start for seg in lp if seg.start is not None]
            if not starts:
                continue
            if geo.winding(_loop_nodes(lp), origin, axis, geo.perpendicular(axis)) != 0:
                return np.asarray(self.vpos[min(starts)], dtype=float)
        return None

    def fit_anchor(self, r, anchors):
        if r in anchors and not self.on_seam(r, anchors[r]):
            raise BuildError(f"region {r}: the seam cannot pass through both of its loop vertices")

    def choose_frames(self):
        periodic = sorted(r for r in self.shell.regions if geo.is_curved(self.surface(r)))
        anchors = {r: a for r in periodic if (a := self.anchor(r)) is not None}
        for start in sorted(periodic, key=lambda r: (r not in anchors, r)):
            if start in self.frames:
                continue
            if start in anchors:
                self.frames[start] = self.frame_through(start, anchors[start])
            else:
                self.frames[start] = self.initial_frame(start)
            queue = [start]
            while queue:
                r = queue.pop(0)
                for e in self.edges:
                    if r not in (e.a, e.b) or not e.closed or not self.winds(e, r):
                        continue
                    if e.seam_point is None:
                        e.seam_point = self.seam_point(e, r)
                    elif not self.on_seam(r, e.seam_point):
                        raise BuildError(
                            f"regions {e.a}/{e.b}: the seams of the neighbouring faces"
                            " cannot meet at one vertex"
                        )
                    other = e.b if r == e.a else e.a
                    if self.winds(e, other):
                        if other not in self.frames:
                            self.frames[other] = self.frame_through(other, e.seam_point)
                            self.fit_anchor(other, anchors)
                            queue.append(other)
                        elif not self.on_seam(other, e.seam_point):
                            raise BuildError(
                                f"regions {e.a}/{e.b}: the seams of the neighbouring faces"
                                " cannot meet at one vertex"
                            )

    def make_edges(self):
        for e in self.edges:
            if e.curved:
                self.curved_edge(e)
            else:
                self.straight_edges(e)
            for r in (e.a, e.b):
                self.out.boundary_deviation[r] = max(self.out.boundary_deviation[r], e.deviation)

    def straight_edges(self, e: _Edge):
        nodes = e.nodes
        for i in range(len(nodes) - 1):
            edge = self.pool.edge(nodes[i], nodes[i + 1])
            if edge is not None:
                e.edges.append(edge)
        if not e.edges:
            raise BuildError(f"regions {e.a}/{e.b}: boundary collapsed to a point")

    def curved_edge(self, e: _Edge):
        surfaces = [self.surface(e.a), self.surface(e.b)]
        if e.closed:
            start = e.seam_point
            if start is None:
                start = geo.refine(surfaces, e.points[0])
                if e.curve is not None:
                    t, _ = nearest(e.curve, start)
                    start = _xyz(e.curve.Value(t))
        else:
            start, end = self.vpos[e.start], self.vpos[e.end]
        if e.curve is not None:
            try:
                if e.closed:
                    t0, _ = nearest(e.curve, start)
                    k = int(np.argmin([np.linalg.norm(p - start) for p in e.points]))
                    n = len(e.points)
                    travel = e.points[(k + 1) % n] - e.points[(k - 1) % n]
                    forward = float(_tangent(e.curve, t0) @ travel) > 0
                    t1 = t0 + e.curve.Period() if forward else t0 - e.curve.Period()
                    end = start
                else:
                    t0, t1, _ = trim_open(e.curve, start, end, e.points, e.start == e.end)
                lo, hi = min(t0, t1), max(t0, t1)
                dev = _range_deviation(e.curve, lo, hi, e.points)
                if dev > self.limit:
                    raise ValueError(
                        f"trimmed intersection is {dev:.3g} from the boundary,"
                        f" over the {self.limit:.3g} limit"
                    )
                e.deviation = dev
                v0 = self.pool.vertex(start)
                v1 = self.pool.vertex(end)
                if t0 <= t1:
                    mk = BRepBuilderAPI_MakeEdge(e.curve, v0, v1, t0, t1)
                    reverse = False
                else:
                    mk = BRepBuilderAPI_MakeEdge(e.curve, v1, v0, t1, t0)
                    reverse = True
                if not mk.IsDone():
                    raise ValueError(f"edge construction failed ({mk.Error()})")
                edge = mk.Edge()
                e.edges = [TopoDS.Edge_s(edge.Reversed()) if reverse else edge]
                e.nodes = [start, *e.points[1:-1], end] if not e.closed else [*e.points]
                return
            except ValueError as err:
                e.fallback = str(err)
        self.projected_edge(e, start, None if e.closed else end, surfaces)

    def projected_edge(self, e: _Edge, start, end, surfaces):
        if e.closed:
            k = int(np.argmin([np.linalg.norm(p - start) for p in e.points]))
            pts = e.points[k:] + e.points[:k]
            nodes = [start, *pts[1:]]
        else:
            nodes = [start, *e.points[1:-1], end]
        curve, dev = projected_curve(surfaces, nodes, e.closed)
        if dev > self.limit:
            raise BuildError(
                f"regions {e.a}/{e.b}: projected boundary curve is {dev:.3g} off the surfaces,"
                f" over the {self.limit:.3g} limit"
            )
        v0 = self.pool.vertex(start)
        v1 = v0 if e.closed else self.pool.vertex(end)
        mk = BRepBuilderAPI_MakeEdge(curve, v0, v1, curve.FirstParameter(), curve.LastParameter())
        if not mk.IsDone():
            raise BuildError(f"regions {e.a}/{e.b}: projected edge construction failed")
        edge = mk.Edge()
        tol = max(EDGE_TOLERANCE, 1.01 * dev)
        self.builder.UpdateEdge(edge, tol)
        self.builder.SameParameter(edge, False)
        self.loose.append(edge)
        for v in (v0, v1):
            self.builder.UpdateVertex(v, tol)
        e.edges = [edge]
        e.nodes = nodes + ([start] if e.closed else [])
        lo, hi = curve.FirstParameter(), curve.LastParameter()
        e.deviation = max(dev, _range_deviation(curve, lo, hi, e.points))
        if e.deviation > self.limit:
            raise BuildError(
                f"regions {e.a}/{e.b}: boundary points are {e.deviation:.3g} from the built"
                f" edge, over the {self.limit:.3g} limit"
            )
        if not e.approximate:
            self.out.projected.append(
                ProjectedEdge((e.a, e.b), e.fallback or "no intersection", float(e.deviation))
            )

    def segments(self, r) -> list[_Segment]:
        out = []
        for e in self.edges:
            if r not in (e.a, e.b):
                continue
            nodes = e.nodes if e.nodes else e.points
            seg = _Segment(list(e.edges), e.start, e.end, list(nodes))
            out.append(seg if r == e.a else _reverse_segment(seg))
        return out

    def face(self, r):
        s = self.surface(r)
        if isinstance(s, Facets):
            return self.facets_faces(r, s)
        segs = self.segments(r)
        loops = _chain(segs, lambda p: geo.outward(s, p), r)
        if isinstance(s, Plane):
            return [self.plane_face(r, s, loops)]
        return [self.curved_face(r, s, loops)]

    def wire(self, edges) -> TopoDS_Wire:
        w = TopoDS_Wire()
        self.builder.MakeWire(w)
        for e in edges:
            self.builder.Add(w, e)
        w.Closed(True)
        return w

    def plane_face(self, r, s: Plane, loops):
        normal = geo.unit(s.normal)
        if not loops:
            raise BuildError(f"region {r}: no boundary loops")
        areas = [_signed_area(_loop_nodes(lp), normal) for lp in loops]
        outer = max(range(len(loops)), key=lambda i: abs(areas[i]))
        if areas[outer] <= 0:
            raise BuildError(f"region {r}: outer loop winds against the outward normal")
        order = [outer] + [i for i in range(len(loops)) if i != outer]
        face = TopoDS_Face()
        self.builder.MakeFace(face, self.occ_surface(r), EDGE_TOLERANCE)
        for i in order:
            if i != outer and areas[i] >= 0:
                raise BuildError(f"region {r}: inner loop winds with the outward normal")
            self.builder.Add(face, self.wire([e for seg in loops[i] for e in seg.edges]))
        return face

    def facets_faces(self, r, s: Facets):
        snapped = {}
        for v, x in zip(self.ir.vertices, self.vpos, strict=True):
            for p in (v.position, *v.source_positions):
                snapped[tuple(np.round(np.asarray(p, dtype=float) * 1e7))] = x
        v = np.array(
            [
                snapped.get(tuple(np.round(np.asarray(p, dtype=float) * 1e7)), np.asarray(p))
                for p in s.vertices
            ],
            dtype=float,
        )
        faces = []
        for f in s.faces:
            a, b, c = v[list(f)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-14:
                raise BuildError(f"region {r}: degenerate facets triangle")
            face = TopoDS_Face()
            self.builder.MakeFace(face, Geom_Plane(gp_Ax3(_pnt(a), _dir(geo.unit(n)))), 1e-7)
            edges = [self.pool.edge(p, q) for p, q in ((a, b), (b, c), (c, a))]
            if any(e is None for e in edges):
                raise BuildError(f"region {r}: degenerate facets triangle")
            self.builder.Add(face, self.wire(edges))
            faces.append(face)
        return faces

    def curved_face(self, r, s, loops):
        surf = self.occ_surface(r)
        reversed_ = s.orientation == "reversed"
        if reversed_:
            loops = [[_reverse_segment(seg) for seg in reversed(lp)] for lp in loops]
        face = TopoDS_Face()
        self.builder.MakeFace(face, surf, EDGE_TOLERANCE)
        if not loops:
            if not isinstance(s, (Sphere, Torus)):
                raise BuildError(f"region {r}: an unbounded {s.type} has no boundary")
            mk = BRepBuilderAPI_MakeFace(surf, EDGE_TOLERANCE)
            if not mk.IsDone():
                raise BuildError(f"region {r}: natural {s.type} face construction failed")
            face = mk.Face()
            return TopoDS.Face_s(face.Reversed()) if reversed_ else face
        uv_loops = [_UVLoop(r, surf, [e for seg in lp for e in seg.edges], s) for lp in loops]
        wraps = [lp for lp in uv_loops if lp.du != 0]
        holes = [lp for lp in uv_loops if lp.du == 0]
        if any(lp.dv != 0 for lp in uv_loops):
            raise BuildError(f"region {r}: loops around the tube of a torus are not supported yet")
        wires = []
        if len(wraps) == 2:
            up, down = sorted(wraps, key=lambda lp: -lp.du)
            if up.du != 1 or down.du != -1:
                raise BuildError(f"region {r}: boundary loops wind the same way around the axis")
            if isinstance(s, Torus):
                down.shift_v(up.v0)
            if not down.v0 > up.v0:
                raise BuildError(
                    f"region {r}: boundary loops are in the wrong order along the axis"
                )
            seam = self.seam_edge(surf, face, up.start_vertex, down.start_vertex, up.v0, down.v0)
            wires.append([*up.edges, seam, *down.edges, TopoDS.Edge_s(seam.Reversed())])
        elif len(wraps) == 1:
            lp = wraps[0]
            pole_v, pole_p = self.pole(r, s, lp.du)
            if lp.du > 0:
                seam = self.seam_edge(surf, face, lp.start_vertex, None, lp.v0, pole_v, pole_p)
                pole = self.degenerate_edge(face, seam_vertex(seam, last=True), pole_v)
                wires.append(
                    [
                        *lp.edges,
                        seam,
                        TopoDS.Edge_s(pole.Reversed()),
                        TopoDS.Edge_s(seam.Reversed()),
                    ]
                )
            else:
                seam = self.seam_edge(surf, face, None, lp.start_vertex, pole_v, lp.v0, pole_p)
                pole = self.degenerate_edge(face, seam_vertex(seam, last=False), pole_v)
                wires.append([*lp.edges, TopoDS.Edge_s(seam.Reversed()), pole, seam])
        elif wraps:
            raise BuildError(f"region {r}: {len(wraps)} boundary loops wind around the axis")
        for lp in holes:
            lp.center_u()
            wires.append(lp.edges)
        for edges in wires:
            self.builder.Add(face, self.wire(edges))
        for lp in uv_loops:
            lp.commit(self.builder, face)
        return TopoDS.Face_s(face.Reversed()) if reversed_ else face

    def pole(self, r, s, du):
        if isinstance(s, Sphere):
            v = math.pi / 2 if du > 0 else -math.pi / 2
            frame = self.frames[r]
            p = np.asarray(s.center, dtype=float) + math.copysign(s.radius, du) * frame.axis
            return v, p
        if isinstance(s, Cone) and du < 0:
            return 0.0, np.asarray(s.apex, dtype=float)
        raise BuildError(f"region {r}: a single boundary loop around a {s.type} leaves it open")

    def seam_edge(self, surf, face, v_lo, v_hi, lo, hi, pole_point=None):
        iso = surf.UIso(0.0)
        if v_lo is None:
            v_lo = self.pool.vertex(pole_point)
        if v_hi is None:
            v_hi = self.pool.vertex(pole_point)
        mk = BRepBuilderAPI_MakeEdge(iso, v_lo, v_hi, lo, hi)
        if not mk.IsDone():
            raise BuildError(f"seam edge construction failed ({mk.Error()})")
        seam = mk.Edge()
        shift = BRep_Tool.Range_s(seam)[0] - lo
        right = Geom2d_Line(gp_Pnt2d(TWO_PI, -shift), gp_Dir2d(0.0, 1.0))
        left = Geom2d_Line(gp_Pnt2d(0.0, -shift), gp_Dir2d(0.0, 1.0))
        self.builder.UpdateEdge(seam, right, left, face, EDGE_TOLERANCE)
        return seam

    def degenerate_edge(self, face, vertex, v):
        edge = TopoDS_Edge()
        self.builder.MakeEdge(edge)
        self.builder.Add(edge, vertex.Oriented(TopAbs_FORWARD))
        self.builder.Add(edge, vertex.Oriented(TopAbs_REVERSED))
        self.builder.Degenerated(edge, True)
        line = Geom2d_Line(gp_Pnt2d(0.0, v), gp_Dir2d(1.0, 0.0))
        self.builder.UpdateEdge(edge, line, face, EDGE_TOLERANCE)
        self.builder.Range(edge, 0.0, TWO_PI)
        return edge

    def check_side(self, r, face):
        s = self.surface(r)
        if isinstance(s, Facets):
            return
        for e in self.edges:
            if r in (e.a, e.b):
                self.probe(r, s, face, e)

    def probe(self, r, s, face, e: _Edge):
        pts = e.points
        n = len(pts)
        segments = n if e.closed else n - 1
        scale = max(float(np.linalg.norm(np.ptp(np.asarray(pts), axis=0))), 1e-3)
        floor = 4.0 * e.deviation + 1e-6
        steps = sorted({max(1e-4 * scale, floor), max(1e-5 * scale, floor)}, reverse=True)
        pair = [self.surface(e.a), self.surface(e.b)]
        if not all(geo.is_analytic(x) for x in pair):
            pair = None
        for i in sorted({segments // 2, segments // 3, (2 * segments) // 3, 0}):
            a, b = pts[i], pts[(i + 1) % n]
            t = geo.unit(b - a) if r == e.a else geo.unit(a - b)
            p = 0.5 * (a + b)
            if pair:
                p = geo.refine(pair, p)
            p, _ = geo.closest(s, p)
            side = geo.unit(np.cross(geo.outward(s, p), t))
            for k, step in enumerate(steps):
                q, _ = geo.closest(s, p + step * side)
                state = BRepClass_FaceClassifier(face, _pnt(q), 1e-7).State()
                if state == TopAbs_IN:
                    return
                if state != TopAbs_OUT or k + 1 < len(steps):
                    continue
                raise BuildError(
                    f"region {r}: the built face is not on the IR's side of its boundary with"
                    f" region {e.b if r == e.a else e.a}"
                )
        raise BuildError(
            f"region {r}: could not confirm the built face's side of its boundary with"
            f" region {e.b if r == e.a else e.a}"
        )


def seam_vertex(seam, last: bool):
    return TopExp.LastVertex_s(seam) if last else TopExp.FirstVertex_s(seam)


class _UVLoop:
    def __init__(self, region, surf, edges, ir_surface):
        self.region = region
        self.surf = surf
        self.edges = list(edges)
        self.v_periodic = isinstance(ir_surface, Torus)
        self.pcurves = []
        uv = []
        for e in self.edges:
            curve = BRep_Tool.Curve_s(e, 0.0, 0.0)
            first, last = BRep_Tool.Range_s(e)
            pc = GeomProjLib.Curve2d_s(curve, first, last, surf)
            if pc is None:
                raise BuildError(f"region {region}: could not project an edge onto the surface")
            a, b = (first, last) if e.Orientation() == TopAbs_FORWARD else (last, first)
            uv.append([np.array(pc.Value(a).Coord()), np.array(pc.Value(b).Coord())])
            self.pcurves.append(pc)
        self.shifts = [np.zeros(2) for _ in self.edges]
        for i in range(1, len(self.edges)):
            prev_end = uv[i - 1][1] + self.shifts[i - 1]
            delta = prev_end - uv[i][0]
            self.shifts[i] = self._period_round(delta)
        last_end = uv[-1][1] + self.shifts[-1]
        total = last_end - uv[0][0]
        self.du = int(round(total[0] / TWO_PI))
        self.dv = int(round(total[1] / TWO_PI)) if self.v_periodic else 0
        self.uv = uv
        if self.du != 0:
            self._start_on_seam()
        self.v0 = float(self.uv[0][0][1] + self.shifts[0][1])

    @property
    def start_vertex(self):
        return TopExp.FirstVertex_s(self.edges[0], True)

    def _period_round(self, delta):
        du = TWO_PI * round(delta[0] / TWO_PI)
        dv = TWO_PI * round(delta[1] / TWO_PI) if self.v_periodic else 0.0
        return np.array([du, dv])

    def _start_on_seam(self):
        n = len(self.edges)
        for k in range(n):
            u = self.uv[k][0][0] + self.shifts[k][0]
            if abs(u - TWO_PI * round(u / TWO_PI)) < 1e-6:
                break
        else:
            us = [uv[0][0] + sh[0] for uv, sh in zip(self.uv, self.shifts, strict=True)]
            raise BuildError(
                f"region {self.region}: no vertex of a wrapping loop is on the seam {us}"
            )
        order = list(range(k, n)) + list(range(k))
        self.edges = [self.edges[i] for i in order]
        self.pcurves = [self.pcurves[i] for i in order]
        self.uv = [self.uv[i] for i in order]
        base = self.shifts[k].copy()
        shifts = [self.shifts[i] - base for i in order]
        for i in range(len(order)):
            if order[i] < k:
                shifts[i] = shifts[i] + np.array([TWO_PI * self.du, TWO_PI * self.dv])
        target = 0.0 if self.du > 0 else TWO_PI
        u0 = self.uv[0][0][0]
        shift0 = np.array([target - u0, 0.0])
        if self.v_periodic:
            shift0[1] = 0.0
        self.shifts = [s + shift0 for s in shifts]

    def shift_v(self, above: float):
        v = self.uv[0][0][1] + self.shifts[0][1]
        k = math.ceil((above - v) / TWO_PI + 1e-12)
        if v + TWO_PI * k - above > TWO_PI:
            k -= 1
        if v + TWO_PI * k <= above:
            k += 1
        self.shifts = [s + np.array([0.0, TWO_PI * k]) for s in self.shifts]
        self.v0 = float(self.uv[0][0][1] + self.shifts[0][1])

    def center_u(self):
        us = [uv[0][0] + s[0] for uv, s in zip(self.uv, self.shifts, strict=True)]
        us += [uv[1][0] + s[0] for uv, s in zip(self.uv, self.shifts, strict=True)]
        lo, hi = min(us), max(us)
        k = math.floor(lo / TWO_PI)
        self.shifts = [s - np.array([TWO_PI * k, 0.0]) for s in self.shifts]
        if hi - TWO_PI * k > TWO_PI + 1e-9:
            raise BuildError(f"region {self.region}: a boundary loop crosses the seam")

    def commit(self, builder, face):
        for e, pc, s in zip(self.edges, self.pcurves, self.shifts, strict=True):
            if np.any(s != 0):
                pc = pc.Copy()
                pc.Translate(gp_Vec2d(float(s[0]), float(s[1])))
            builder.UpdateEdge(e, pc, face, EDGE_TOLERANCE)


def build_shell(ir: Ir, index: int, vpos, vmoved, bad, limit: float, pool) -> CurvedShell:
    out = _Builder(ir, index, vpos, vmoved, bad, limit, pool).run()
    analyzer = BRepCheck_Analyzer(out.shell)
    if not analyzer.IsValid():
        raise BuildError(f"shell {index}: the assembled curved shell is invalid")
    return out
