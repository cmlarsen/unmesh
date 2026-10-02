from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from unmesh.ir import Facets, Ir, Plane

VERTEX_PULL = 1e-8
DEVIATION_FACTOR = 5.0


class BuildError(Exception):
    pass


@dataclass
class FacePlan:
    region: int
    surface_type: str
    origin: np.ndarray
    normal: np.ndarray
    loops: list[list[np.ndarray]]


@dataclass
class SeamGap:
    regions: tuple[int, int]
    surface_type: str
    points: int
    max_gap: float


@dataclass
class ShellPlan:
    index: int
    closed: bool
    role: str
    parent: int | None
    faces: list[FacePlan] = field(default_factory=list)
    seams: list[SeamGap] = field(default_factory=list)


@dataclass
class _Segment:
    nodes: list[np.ndarray]
    start: int | None
    end: int | None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise BuildError("zero-length normal")
    return v / n


def vertex_positions(ir: Ir) -> list[np.ndarray]:
    limit = DEVIATION_FACTOR * ir.tolerances.linear
    out = []
    for v in ir.vertices:
        p0 = np.asarray(v.position, dtype=float)
        surfaces = [ir.regions[r].surface for r in v.regions]
        if len(surfaces) >= 3 and all(isinstance(s, Plane) for s in surfaces):
            n = np.array([_unit(s.normal) for s in surfaces])
            d = np.array([n[i] @ np.asarray(s.origin, dtype=float) for i, s in enumerate(surfaces)])
            lam = VERTEX_PULL
            x = np.linalg.solve(n.T @ n + lam * np.eye(3), n.T @ d + lam * p0)
            gap = float(np.linalg.norm(x - p0))
            if gap > limit:
                raise BuildError(
                    f"vertex {v.id}: plane intersection is {gap:.6g} from its IR position"
                )
            out.append(x)
        else:
            out.append(p0)
    return out


def _boundary_nodes(ir, bd, a, b, vpos) -> list[np.ndarray]:
    pts = [np.asarray(p, dtype=float) for p in bd.points]
    sa, sb = ir.regions[a].surface, ir.regions[b].surface
    if bd.closed:
        return pts
    start, end = vpos[bd.start_vertex], vpos[bd.end_vertex]
    if isinstance(sa, Plane) and isinstance(sb, Plane):
        chord = end - start
        length = float(np.linalg.norm(chord))
        if length > 0:
            u = chord / length
            limit = DEVIATION_FACTOR * ir.tolerances.linear
            for p in pts:
                q = p - start
                dev = float(np.linalg.norm(q - (q @ u) * u))
                if dev > limit:
                    raise BuildError(
                        f"regions {a}/{b}: boundary point {dev:.6g} off the plane-plane edge"
                    )
        return [start, end]
    return [start, *pts[1:-1], end]


def _signed_area(loop: list[np.ndarray], normal: np.ndarray) -> float:
    s = np.zeros(3)
    for i in range(len(loop)):
        s += np.cross(loop[i], loop[(i + 1) % len(loop)])
    return 0.5 * float(normal @ s)


def _direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def _chain_loops(segments: list[_Segment], normal: np.ndarray, region: int):
    loops = []
    open_segs = []
    for s in segments:
        if s.start is None:
            loops.append(list(s.nodes))
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
        loop = list(cur.nodes[:-1])
        while cur.end != open_segs[first].start:
            options = [i for i in by_start.get(cur.end, []) if not used[i]]
            if not options:
                raise BuildError(f"region {region}: boundary loop does not close")
            d_in = _direction(cur.nodes[-2], cur.nodes[-1])

            def left_turn(i, d_in=d_in):
                d_out = _direction(open_segs[i].nodes[0], open_segs[i].nodes[1])
                return math.atan2(float(normal @ np.cross(d_in, d_out)), float(d_in @ d_out))

            pick = max(options, key=left_turn)
            used[pick] = True
            cur = open_segs[pick]
            loop.extend(cur.nodes[:-1])
        loops.append(loop)
    return loops


def _plane_face(ir, region, segments) -> FacePlan:
    plane = ir.regions[region].surface
    normal = _unit(plane.normal)
    loops = _chain_loops(segments, normal, region)
    if not loops:
        raise BuildError(f"region {region}: no boundary loops")
    for lp in loops:
        if len(lp) < 3:
            raise BuildError(f"region {region}: degenerate boundary loop")
    areas = [_signed_area(lp, normal) for lp in loops]
    outer = max(range(len(loops)), key=lambda i: abs(areas[i]))
    if areas[outer] <= 0:
        raise BuildError(f"region {region}: outer loop winds against the outward normal")
    ordered = [loops[outer]]
    for i, lp in enumerate(loops):
        if i == outer:
            continue
        if areas[i] >= 0:
            raise BuildError(f"region {region}: inner loop winds with the outward normal")
        ordered.append(lp)
    return FacePlan(region, "plane", np.asarray(plane.origin, dtype=float), normal, ordered)


def _facets_faces(region: int, surface: Facets) -> list[FacePlan]:
    v = np.asarray(surface.vertices, dtype=float)
    faces = []
    for f in surface.faces:
        a, b, c = v[list(f)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            raise BuildError(f"region {region}: degenerate facets triangle")
        faces.append(FacePlan(region, "facets", a, n / norm, [[a, b, c]]))
    return faces


def plan_shell(ir: Ir, index: int, vpos: list[np.ndarray]) -> ShellPlan:
    shell = ir.shells[index]
    plan = ShellPlan(index, shell.closed, shell.role, shell.parent)
    members = set(shell.regions)
    segments: dict[int, list[_Segment]] = {r: [] for r in shell.regions}
    for adj in ir.adjacencies:
        a, b = adj.regions
        if a not in members:
            continue
        for bd in adj.boundaries:
            nodes = _boundary_nodes(ir, bd, a, b, vpos)
            segments[a].append(_Segment(nodes, bd.start_vertex, bd.end_vertex))
            segments[b].append(_Segment(nodes[::-1], bd.end_vertex, bd.start_vertex))
            sa, sb = ir.regions[a].surface, ir.regions[b].surface
            if isinstance(sa, Facets) != isinstance(sb, Facets):
                analytic = sb if isinstance(sa, Facets) else sa
                if isinstance(analytic, Plane):
                    n = _unit(analytic.normal)
                    o = np.asarray(analytic.origin, dtype=float)
                    gap = max(abs(float(n @ (np.asarray(p, dtype=float) - o))) for p in bd.points)
                    plan.seams.append(SeamGap((a, b), "plane", len(bd.points), gap))
    for r in shell.regions:
        surface = ir.regions[r].surface
        if isinstance(surface, Plane):
            plan.faces.append(_plane_face(ir, r, segments[r]))
        elif isinstance(surface, Facets):
            plan.faces.extend(_facets_faces(r, surface))
        else:
            raise BuildError(f"region {r}: {surface.type} surfaces are not supported yet")
    return plan
