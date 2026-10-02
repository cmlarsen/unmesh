from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from geometry import dihedral_deg, normal, project, sample_points

from unmesh.ir import (
    Adjacency,
    Boundary,
    Cone,
    Cylinder,
    Facets,
    Ir,
    Plane,
    Region,
    Residual,
    Shell,
    Source,
    Tolerances,
    Torus,
    Vertex,
)

TOL = Tolerances()
STEP = 0.1
DEPTH = 1e-3


def key(p) -> tuple[float, float, float]:
    return tuple(round(float(c), 9) + 0.0 for c in p)


class Builder:
    def __init__(self, inside: Callable[[np.ndarray], bool], tol: Tolerances = TOL):
        self.inside = inside
        self.tol = tol
        self.regions: list[Region] = []
        self.shells: list[list[int]] = []
        self.shell_meta: list[tuple[str, int | None]] = []
        self.edges: dict[tuple[int, int], list[Boundary]] = {}
        self.next_triangle = 0

    def shell(self, role: str = "outer", parent: int | None = None) -> None:
        self.shell_meta.append((role, parent))
        self.shells.append([])

    def region(self, surface, n_triangles: int = 2) -> int:
        rid = len(self.regions)
        ids = list(range(self.next_triangle, self.next_triangle + n_triangles))
        self.next_triangle += n_triangles
        residual = None if isinstance(surface, Facets) else Residual(2e-5, 6e-5)
        self.regions.append(Region(rid, surface, ids, residual))
        self.shells[-1].append(rid)
        return rid

    def plane(self, origin, n, n_triangles: int = 2) -> int:
        return self.region(Plane(tuple(map(float, origin)), tuple(map(float, n))), n_triangles)

    def edge(self, a: int, b: int, points, closed: bool = False) -> None:
        assert a < b
        pts = [tuple(map(float, p)) for p in points]
        if not self._a_on_left(a, pts, closed):
            pts = pts[::-1]
            assert self._a_on_left(a, pts, closed), f"orientation probe failed for ({a}, {b})"
        sa, sb = self.regions[a].surface, self.regions[b].surface
        deg = round(dihedral_deg(sa, sb, sample_points(pts, closed)), 6)
        kind = "tangent" if deg < self.tol.tangent_threshold_deg else "transversal"
        self.edges.setdefault((a, b), []).append(Boundary(kind, deg, closed, None, None, pts))

    def _a_on_left(self, a: int, pts, closed: bool) -> bool:
        if len(pts) == 2 and not closed:
            p = (np.asarray(pts[0]) + np.asarray(pts[1])) / 2
            t = np.asarray(pts[1]) - np.asarray(pts[0])
        else:
            m = len(pts) // 2
            p = np.asarray(pts[m])
            t = np.asarray(pts[(m + 1) % len(pts)]) - np.asarray(pts[m - 1])
        t = t / np.linalg.norm(t)
        surface = self.regions[a].surface
        w = np.cross(normal(surface, p), t)
        q = p + STEP * w
        if not isinstance(surface, Facets):
            q = project(surface, q)
        n = normal(surface, q)
        return self.inside(q - DEPTH * n) and not self.inside(q + DEPTH * n)

    def build(self) -> Ir:
        ends: dict[tuple, set[int]] = {}
        for (a, b), bds in self.edges.items():
            for bd in bds:
                if not bd.closed:
                    for p in (bd.points[0], bd.points[-1]):
                        ends.setdefault(key(p), set()).update((a, b))
        order = sorted(ends)
        index = {k: i for i, k in enumerate(order)}
        vertices = [
            Vertex(i, "junction" if len(ends[k]) > 2 else "kind_change", k, sorted(ends[k]), [k])
            for i, k in enumerate(order)
        ]
        adjacencies = []
        for a, b in sorted(self.edges):
            bds = self.edges[(a, b)]
            for bd in bds:
                if not bd.closed:
                    bd.start_vertex = index[key(bd.points[0])]
                    bd.end_vertex = index[key(bd.points[-1])]
            adjacencies.append(Adjacency((a, b), bds))
        n = self.next_triangle
        ir = Ir(
            self.tol,
            Source(n, n // 2 + 2),
            [
                Shell(True, role, parent, s)
                for (role, parent), s in zip(self.shell_meta, self.shells, strict=True)
            ],
            self.regions,
            adjacencies,
            vertices,
        )
        ir.validate()
        return ir


def arc(center, u, v, r, a0, a1, n):
    c, u, v = (np.asarray(x, dtype=float) for x in (center, u, v))
    return [tuple(c + r * (math.cos(t) * u + math.sin(t) * v)) for t in np.linspace(a0, a1, n)]


def circle(center, u, v, r, n=32):
    return arc(center, u, v, r, 0.0, 2 * math.pi, n + 1)[:-1]


def in_box(p, lo, hi) -> bool:
    return all(lo[i] < p[i] < hi[i] for i in range(3))


def add_box(
    b: Builder, lo, hi, skip: set[str] = frozenset(), inward: bool = False
) -> dict[str, int]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    specs = {
        "-x": ((x0, y0, z0), (-1, 0, 0)),
        "+x": ((x1, y0, z0), (1, 0, 0)),
        "-y": ((x0, y0, z0), (0, -1, 0)),
        "+y": ((x0, y1, z0), (0, 1, 0)),
        "-z": ((x0, y0, z0), (0, 0, -1)),
        "+z": ((x0, y0, z1), (0, 0, 1)),
    }
    sign = -1 if inward else 1
    return {
        k: b.plane(o, tuple(sign * c for c in n)) for k, (o, n) in specs.items() if k not in skip
    }


def box_edge(b: Builder, ids, f1: str, f2: str, p, q) -> None:
    a, c = sorted((ids[f1], ids[f2]))
    b.edge(a, c, [p, q])


def all_box_edges(b: Builder, ids, lo, hi, skip=()) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    edges = {
        ("-x", "-y"): ((x0, y0, z0), (x0, y0, z1)),
        ("-x", "+y"): ((x0, y1, z0), (x0, y1, z1)),
        ("+x", "-y"): ((x1, y0, z0), (x1, y0, z1)),
        ("+x", "+y"): ((x1, y1, z0), (x1, y1, z1)),
        ("-x", "-z"): ((x0, y0, z0), (x0, y1, z0)),
        ("-x", "+z"): ((x0, y0, z1), (x0, y1, z1)),
        ("+x", "-z"): ((x1, y0, z0), (x1, y1, z0)),
        ("+x", "+z"): ((x1, y0, z1), (x1, y1, z1)),
        ("-y", "-z"): ((x0, y0, z0), (x1, y0, z0)),
        ("-y", "+z"): ((x0, y0, z1), (x1, y0, z1)),
        ("+y", "-z"): ((x0, y1, z0), (x1, y1, z0)),
        ("+y", "+z"): ((x0, y1, z1), (x1, y1, z1)),
    }
    for k, (p, q) in edges.items():
        if k not in skip and k[0] in ids and k[1] in ids:
            box_edge(b, ids, k[0], k[1], p, q)


def fixture_box() -> Ir:
    lo, hi = (0, 0, 0), (20, 10, 5)

    def inside(p):
        return in_box(p, lo, hi)

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    all_box_edges(b, ids, lo, hi)
    return b.build(), inside


def fixture_plate_with_bore() -> Ir:
    lo, hi = (0, 0, 0), (30, 20, 6)
    c, r = (15.0, 10.0), 4.0

    def inside(p):
        return in_box(p, lo, hi) and math.hypot(p[0] - c[0], p[1] - c[1]) > r

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    bore = b.region(Cylinder((c[0], c[1], 0.0), (0.0, 0.0, 1.0), r, "reversed"), 16)
    all_box_edges(b, ids, lo, hi)
    for z, face in ((6.0, "+z"), (0.0, "-z")):
        b.edge(ids[face], bore, circle((c[0], c[1], z), (1, 0, 0), (0, 1, 0), r), closed=True)
    return b.build(), inside


def fixture_box_fillet() -> Ir:
    lo, hi, rad = (0, 0, 0), (20, 10, 5), 2.0
    cx, cy = 18.0, 8.0

    def inside(p):
        if not in_box(p, lo, hi):
            return False
        if p[0] > cx and p[1] > cy:
            return math.hypot(p[0] - cx, p[1] - cy) < rad
        return True

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    fil = b.region(Cylinder((cx, cy, 0.0), (0.0, 0.0, 1.0), rad, "same"), 8)
    all_box_edges(
        b, ids, lo, hi, skip={("+x", "+y"), ("+x", "-z"), ("+x", "+z"), ("+y", "-z"), ("+y", "+z")}
    )
    for face, z in (("-z", 0.0), ("+z", 5.0)):
        box_edge(b, ids, "+x", face, (20, 0, z), (20, cy, z))
        box_edge(b, ids, "+y", face, (0, 10, z), (cx, 10, z))
        a, c = sorted((ids[face], fil))
        b.edge(a, c, arc((cx, cy, z), (1, 0, 0), (0, 1, 0), rad, 0.0, math.pi / 2, 9))
    b.edge(ids["+x"], fil, [(20, cy, 0), (20, cy, 5)])
    b.edge(ids["+y"], fil, [(cx, 10, 0), (cx, 10, 5)])
    return b.build(), inside


def fixture_plate_chamfer() -> Ir:
    lo, hi = (0, 0, 0), (20, 10, 5)

    def inside(p):
        return in_box(p, lo, hi) and (p[0] - 19.0) + (p[2] - 5.0) < 0.0

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    s = 1 / math.sqrt(2)
    ch = b.plane((20, 0, 4), (s, 0, s))
    all_box_edges(
        b,
        ids,
        lo,
        hi,
        skip={("+x", "+z"), ("+x", "-y"), ("+x", "+y"), ("-y", "+z"), ("+y", "+z")},
    )
    box_edge(b, ids, "+x", "-y", (20, 0, 0), (20, 0, 4))
    box_edge(b, ids, "+x", "+y", (20, 10, 0), (20, 10, 4))
    box_edge(b, ids, "-y", "+z", (0, 0, 5), (19, 0, 5))
    box_edge(b, ids, "+y", "+z", (0, 10, 5), (19, 10, 5))
    b.edge(ids["+x"], ch, [(20, 0, 4), (20, 10, 4)])
    b.edge(ids["+z"], ch, [(19, 0, 5), (19, 10, 5)])
    b.edge(ids["-y"], ch, [(20, 0, 4), (19, 0, 5)])
    b.edge(ids["+y"], ch, [(20, 10, 4), (19, 10, 5)])
    return b.build(), inside


def fixture_countersink() -> Ir:
    lo, hi = (0, 0, 0), (20, 20, 8)
    cx, cy = 10.0, 10.0
    r_top, r_bore, z_cone = 4.0, 2.0, 6.0

    def inside(p):
        if not in_box(p, lo, hi):
            return False
        rho = math.hypot(p[0] - cx, p[1] - cy)
        if p[2] > z_cone:
            return rho > r_bore + (p[2] - z_cone)
        return rho > r_bore

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    cone = b.region(Cone((cx, cy, 4.0), (0.0, 0.0, 1.0), math.pi / 4, "reversed"), 16)
    bore = b.region(Cylinder((cx, cy, 0.0), (0.0, 0.0, 1.0), r_bore, "reversed"), 16)
    all_box_edges(b, ids, lo, hi)
    x, y = (1, 0, 0), (0, 1, 0)
    b.edge(ids["+z"], cone, circle((cx, cy, 8.0), x, y, r_top), closed=True)
    b.edge(cone, bore, circle((cx, cy, z_cone), x, y, r_bore), closed=True)
    b.edge(ids["-z"], bore, circle((cx, cy, 0.0), x, y, r_bore), closed=True)
    return b.build(), inside


def fixture_disc_fillet() -> Ir:
    radius, height, rf = 10.0, 4.0, 1.5
    zc = height - rf

    def inside(p):
        rho = math.hypot(p[0], p[1])
        if not (0 < p[2] < height and rho < radius):
            return False
        if p[2] > zc and rho > radius - rf:
            return math.hypot(rho - (radius - rf), p[2] - zc) < rf
        return True

    b = Builder(inside)
    b.shell()
    top = b.plane((0, 0, height), (0, 0, 1))
    bottom = b.plane((0, 0, 0), (0, 0, -1))
    side = b.region(Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), radius, "same"), 32)
    tor = b.region(Torus((0.0, 0.0, zc), (0.0, 0.0, 1.0), radius - rf, rf, "same"), 32)
    x, y = (1, 0, 0), (0, 1, 0)
    b.edge(top, tor, circle((0, 0, height), x, y, radius - rf), closed=True)
    b.edge(side, tor, circle((0, 0, zc), x, y, radius), closed=True)
    b.edge(bottom, side, circle((0, 0, 0), x, y, radius), closed=True)
    return b.build(), inside


def fixture_mixed_facets() -> Ir:
    lo, hi = (0, 0, 0), (20, 10, 5)
    apex = (22.5, 5.0, 2.5)

    def inside(p):
        if in_box(p, lo, hi):
            return True
        if p[0] < 20.0:
            return False
        k = (22.5 - p[0]) / 2.5
        return k > max(abs(p[1] - 5.0) / 5.0, abs(p[2] - 2.5) / 2.5)

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi, skip={"+x"})
    corners = [(20, 0, 0), (20, 10, 0), (20, 10, 5), (20, 0, 5)]
    verts = [tuple(map(float, c)) for c in corners] + [apex]
    faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    patch = b.region(Facets(verts, faces), 4)
    all_box_edges(b, ids, lo, hi)
    for face, p, q in (
        ("-y", corners[3], corners[0]),
        ("+y", corners[1], corners[2]),
        ("-z", corners[0], corners[1]),
        ("+z", corners[2], corners[3]),
    ):
        b.edge(ids[face], patch, [p, q])
    return b.build(), inside


def fixture_two_bodies() -> Ir:
    lo1, hi1 = (0, 0, 0), (10, 10, 5)
    lo2, hi2 = (20, 0, 0), (25, 5, 5)

    def inside(p):
        return in_box(p, lo1, hi1) or in_box(p, lo2, hi2)

    b = Builder(inside)
    for lo, hi in ((lo1, hi1), (lo2, hi2)):
        b.shell()
        ids = add_box(b, lo, hi)
        all_box_edges(b, ids, lo, hi)
    return b.build(), inside


def fixture_open_shell() -> Ir:
    def inside(p):
        return False

    b = Builder(inside)
    b.shell()
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    b.region(Facets(verts, [(0, 1, 2), (0, 2, 3)]), 2)
    ir = Ir(TOL, Source(2, 4), [Shell(False, "outer", None, [0])], b.regions, [], [])
    ir.validate()
    return ir, inside


def fixture_kind_change() -> tuple[Ir, Callable]:
    cx, cy, rad = 10.0, 5.0, 3.0
    alpha = math.radians(30.0)
    thr = 75.0
    slope = math.tan(alpha)

    def top_z(x):
        return 6.0 - slope * (x - cx)

    def inside(p):
        return 0 < p[2] < top_z(p[0]) and math.hypot(p[0] - cx, p[1] - cy) < rad

    tol = Tolerances(tangent_threshold_deg=thr)
    b = Builder(inside, tol)
    b.shell()
    bottom = b.plane((cx, cy, 0.0), (0, 0, -1))
    side = b.region(Cylinder((cx, cy, 0.0), (0.0, 0.0, 1.0), rad, "same"), 16)
    n = (math.sin(alpha), 0.0, math.cos(alpha))
    top = b.plane((cx, cy, 6.0), n, 4)
    theta0 = math.acos(math.cos(math.radians(thr)) / math.sin(alpha))

    def ring(t0, t1, count):
        return [
            (cx + rad * math.cos(t), cy + rad * math.sin(t), top_z(cx + rad * math.cos(t)))
            for t in np.linspace(t0, t1, count)
        ]

    b.edge(bottom, side, circle((cx, cy, 0.0), (1, 0, 0), (0, 1, 0), rad), closed=True)
    b.edge(side, top, ring(-theta0, theta0, 17))
    b.edge(side, top, ring(theta0, 2 * math.pi - theta0, 33))
    return b.build(), inside


def fixture_cavity() -> tuple[Ir, Callable]:
    lo, hi = (0, 0, 0), (20, 20, 20)
    vlo, vhi = (7, 7, 7), (13, 13, 13)

    def inside(p):
        return in_box(p, lo, hi) and not in_box(p, vlo, vhi)

    b = Builder(inside)
    b.shell()
    ids = add_box(b, lo, hi)
    all_box_edges(b, ids, lo, hi)
    b.shell("cavity", 0)
    ids = add_box(b, vlo, vhi, inward=True)
    all_box_edges(b, ids, vlo, vhi)
    return b.build(), inside


FIXTURES = {
    "box": fixture_box,
    "plate_with_bore": fixture_plate_with_bore,
    "box_fillet": fixture_box_fillet,
    "plate_chamfer": fixture_plate_chamfer,
    "countersink": fixture_countersink,
    "disc_fillet": fixture_disc_fillet,
    "mixed_facets": fixture_mixed_facets,
    "two_bodies": fixture_two_bodies,
    "open_shell": fixture_open_shell,
    "kind_change": fixture_kind_change,
    "cavity": fixture_cavity,
}
