from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from unmesh.ir import Facets, Ir, Plane

VERTEX_PULL = 1e-8
DEVIATION_FACTOR = 5.0
KEY_SCALE = 1e7


def deviation_limit(ir: Ir, cap: float) -> float:
    return min(DEVIATION_FACTOR * ir.tolerances.linear, cap)


def _key(p) -> tuple[int, int, int]:
    return (round(p[0] * KEY_SCALE), round(p[1] * KEY_SCALE), round(p[2] * KEY_SCALE))


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
    chord_gap: float = 0.0
    inserted: int = 0
    tolerance: float = 0.0


@dataclass
class PatchWrite:
    corners: np.ndarray
    splits: list[tuple[int, int, np.ndarray, int]] = field(default_factory=list)


@dataclass
class ShellPlan:
    index: int
    closed: bool
    role: str
    parent: int | None
    faces: list[FacePlan] = field(default_factory=list)
    seams: list[SeamGap] = field(default_factory=list)
    vertex_displacement: dict[int, float] = field(default_factory=dict)
    boundary_deviation: dict[int, float] = field(default_factory=dict)
    patches: dict[int, PatchWrite] = field(default_factory=dict)


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


def vertex_positions(ir: Ir, limit: float):
    out = []
    moved = []
    bad: dict[int, str] = {}
    for v in ir.vertices:
        p0 = np.asarray(v.position, dtype=float)
        planes = [
            ir.regions[r].surface for r in v.regions if isinstance(ir.regions[r].surface, Plane)
        ]
        if len(planes) >= 2:
            n = np.array([_unit(s.normal) for s in planes])
            d = np.array([n[i] @ np.asarray(s.origin, dtype=float) for i, s in enumerate(planes)])
            x = np.linalg.solve(n.T @ n + VERTEX_PULL * np.eye(3), n.T @ d + VERTEX_PULL * p0)
        else:
            x = p0
        gap = float(np.linalg.norm(x - p0))
        if gap > limit:
            bad[v.id] = (
                f"vertex {v.id}: plane intersection is {gap:.6g} from its IR position,"
                f" over the {limit:.6g} limit"
            )
        out.append(x)
        moved.append(gap)
    return out, moved, bad


def _boundary_nodes(ir, bd, a, b, vpos, limit):
    pts = [np.asarray(p, dtype=float) for p in bd.points]
    sa, sb = ir.regions[a].surface, ir.regions[b].surface
    both_planes = isinstance(sa, Plane) and isinstance(sb, Plane)
    dev = 0.0
    if bd.closed:
        nodes = pts
        if both_planes:
            for plane in (sa, sb):
                n = _unit(plane.normal)
                o = np.asarray(plane.origin, dtype=float)
                dev = max(dev, max(abs(float(n @ (p - o))) for p in pts))
    else:
        start, end = vpos[bd.start_vertex], vpos[bd.end_vertex]
        if both_planes:
            nodes = [start, end]
            chord = end - start
            length = float(np.linalg.norm(chord))
            if length > 0:
                u = chord / length
                for p in pts:
                    q = p - start
                    dev = max(dev, float(np.linalg.norm(q - (q @ u) * u)))
        else:
            nodes = [start, *pts[1:-1], end]
    if dev > limit:
        raise BuildError(
            f"regions {a}/{b}: boundary point {dev:.6g} off the built edge,"
            f" over the {limit:.6g} limit"
        )
    return nodes, dev


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


def check_windings(faces: list[tuple[int, list]], facets: set[int]) -> None:
    owner: dict[tuple, tuple[int, int]] = {}
    for face, (region, loops) in enumerate(faces):
        for loop in loops:
            keys = [_key(p) for p in loop]
            for a, b in zip(keys, keys[1:] + keys[:1], strict=True):
                if a == b:
                    continue
                other_face, other = owner.setdefault((a, b), (face, region))
                if other_face != face and (region in facets or other in facets):
                    bad = region if region in facets else other
                    raise BuildError(
                        f"region {bad}: facets triangles wind against their neighbours"
                    )


def check_patch_boundary(ir: Ir, a: int, b: int, bd, cache: dict) -> None:
    for r in (a, b):
        surface = ir.regions[r].surface
        if not isinstance(surface, Facets):
            continue
        keys = cache.get(r)
        if keys is None:
            keys = cache[r] = {_key(p) for p in surface.vertices}
        for p in bd.points:
            if _key(p) not in keys:
                raise BuildError(
                    f"region {r}: boundary point {tuple(map(float, p))} with region"
                    f" {b if r == a else a} is not a vertex of the facets patch"
                )


def _facets_faces(region: int, surface: Facets, snapped) -> list[FacePlan]:
    v = np.array([snapped.get(_key(p), np.asarray(p, dtype=float)) for p in surface.vertices])
    faces = []
    for f in surface.faces:
        a, b, c = v[list(f)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            raise BuildError(f"region {region}: degenerate facets triangle")
        faces.append(FacePlan(region, "facets", a, n / norm, [[a, b, c]]))
    return faces


def plan_shell(ir: Ir, index: int, vpos, vmoved, bad, limit: float) -> ShellPlan:
    shell = ir.shells[index]
    plan = ShellPlan(index, shell.closed, shell.role, shell.parent)
    members = set(shell.regions)
    snapped = {}
    for v, x in zip(ir.vertices, vpos, strict=True):
        for p in (v.position, *v.source_positions):
            snapped[_key(p)] = x
    for r in shell.regions:
        plan.vertex_displacement[r] = 0.0
        plan.boundary_deviation[r] = 0.0
    for v, m in zip(ir.vertices, vmoved, strict=True):
        for r in v.regions:
            if r in members:
                if v.id in bad:
                    raise BuildError(bad[v.id])
                plan.vertex_displacement[r] = max(plan.vertex_displacement[r], m)
    segments: dict[int, list[_Segment]] = {r: [] for r in shell.regions}
    patch_keys: dict = {}
    for adj in ir.adjacencies:
        a, b = adj.regions
        if a not in members:
            continue
        for bd in adj.boundaries:
            check_patch_boundary(ir, a, b, bd, patch_keys)
            nodes, dev = _boundary_nodes(ir, bd, a, b, vpos, limit)
            for r in (a, b):
                plan.boundary_deviation[r] = max(plan.boundary_deviation[r], dev)
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
            plan.faces.extend(_facets_faces(r, surface, snapped))
            plan.patches[r] = PatchWrite(
                np.array(
                    [[f.loops[0][i] for i in range(3)] for f in plan.faces[-len(surface.faces) :]]
                )
            )
        else:
            raise BuildError(f"region {r}: {surface.type} surfaces are not supported yet")
    if shell.closed and plan.patches:
        check_windings([(f.region, f.loops) for f in plan.faces], set(plan.patches))
    return plan
