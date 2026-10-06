from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from unmesh.ir import Ir

MAX_TURN_DEG = 15.0
MAX_CREASE_DEG = 72.5
MAX_STRIP_CREASE_DEG = 60.0
MAX_FILLET_TURN_DEG = 175.0
MIN_RING_SIDES = 5
AXIS_PARALLEL_DEG = 1.0
FIT_FACTOR = 4.0

KINDS = {1: "chamfer_or_fillet", 2: "two_planes_or_fillet"}


@dataclass(frozen=True)
class Decision:
    kind: str
    regions: tuple[int, ...]
    supports: tuple[int, ...]
    chosen: str
    alternative: str
    reason: str
    segments: int
    radius: float
    axis: tuple[float, float, float]
    axis_point: tuple[float, float, float]
    alternative_deviation: float
    min_crease_deg: float
    max_crease_deg: float


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def _plane_edges(ir: Ir, normals: dict[int, np.ndarray]) -> list:
    edges = []
    for adj in ir.adjacencies:
        a, b = adj.regions
        if a not in normals or b not in normals:
            continue
        if any(bd.kind != "transversal" for bd in adj.boundaries):
            continue
        crease = _angle(normals[a], normals[b])
        if not ir.tolerances.tangent_threshold_deg <= crease <= MAX_CREASE_DEG:
            continue
        edges.append((a, b, crease, _unit(np.cross(normals[a], normals[b]))))
    return edges


def _clusters(edges) -> list[tuple[np.ndarray, list]]:
    cos = math.cos(math.radians(AXIS_PARALLEL_DEG))
    clusters: list[tuple[np.ndarray, list]] = []
    for e in edges:
        for axis, members in clusters:
            if abs(float(axis @ e[3])) >= cos:
                members.append(e)
                break
        else:
            clusters.append((e[3], [e]))
    return clusters


def _components(members) -> list[tuple[set[int], dict[int, list[int]], dict]]:
    nbr: dict[int, list[int]] = {}
    crease: dict[frozenset, float] = {}
    for a, b, c, _ in members:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
        crease[frozenset((a, b))] = c
    seen: set[int] = set()
    out = []
    for start in sorted(nbr):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(nbr[x])
        seen |= comp
        out.append((comp, {x: sorted(nbr[x]) for x in comp}, crease))
    return out


def _walk(comp: set[int], nbr: dict[int, list[int]]) -> tuple[list[int], bool] | None:
    if any(len(nbr[x]) > 2 for x in comp):
        return None
    ends = sorted(x for x in comp if len(nbr[x]) == 1)
    start = ends[0] if ends else min(comp)
    order, prev = [start], None
    while True:
        nxt = [y for y in nbr[order[-1]] if y != prev]
        if not nxt or nxt[0] == start:
            break
        prev = order[-1]
        order.append(nxt[0])
    return order, not ends and len(order) == len(comp)


class _Section:
    def __init__(self, axis: np.ndarray, normals, origins, ref: np.ndarray):
        e1 = _unit(np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]))
        self.axis, self.e1, self.e2, self.ref = axis, e1, np.cross(axis, e1), ref
        self.lines = {}
        for r, n in normals.items():
            m = np.array([n @ e1, n @ self.e2])
            length = float(np.linalg.norm(m))
            if length > 0:
                self.lines[r] = (m / length, float(n @ (origins[r] - ref)) / length)

    def meet(self, a: int, b: int) -> np.ndarray | None:
        (ma, da), (mb, db) = self.lines[a], self.lines[b]
        det = ma[0] * mb[1] - ma[1] * mb[0]
        if abs(det) < 1e-12:
            return None
        return np.array([(da * mb[1] - db * ma[1]) / det, (ma[0] * db - mb[0] * da) / det])

    def to_3d(self, p: np.ndarray) -> np.ndarray:
        return self.ref + p[0] * self.e1 + p[1] * self.e2


def _corners(sec: _Section, chain: list[int], closed: bool) -> list[np.ndarray] | None:
    pairs = (
        list(zip(chain, chain[1:] + chain[:1], strict=True))
        if closed
        else list(zip(chain, chain[1:], strict=False))
    )
    points = [sec.meet(a, b) for a, b in pairs]
    return None if any(p is None for p in points) else points


def _fillet(sec: _Section, chain: list[int], eps: float):
    points = _corners(sec, chain, False)
    if points is None:
        return None
    (ma, _), (mb, _) = sec.lines[chain[0]], sec.lines[chain[-1]]
    p0, pk = points[0], points[-1]
    m = np.column_stack([-ma, mb])
    if abs(np.linalg.det(m)) < 1e-12:
        return None
    s, t = np.linalg.solve(m, pk - p0)
    if s * t <= 0 or abs(s - t) > eps:
        return None
    radius = 0.5 * (abs(s) + abs(t))
    center = p0 - s * ma
    if any(abs(np.linalg.norm(p - center) - radius) > eps for p in points):
        return None
    return center, radius


def _ring(sec: _Section, chain: list[int], eps: float):
    points = _corners(sec, chain, True)
    if points is None:
        return None
    p = np.array(points)
    a = np.column_stack([2 * p, np.ones(len(p))])
    sol, *_ = np.linalg.lstsq(a, (p**2).sum(axis=1), rcond=None)
    center = sol[:2]
    radius = math.sqrt(max(float(sol[2] + center @ center), 0.0))
    if radius == 0 or np.abs(np.linalg.norm(p - center, axis=1) - radius).max() > eps:
        return None
    return center, radius


def _decision(kind, strips, supports, sec, center, radius, creases, segment_deg, reason):
    sag = radius * (1 - math.cos(math.radians(segment_deg) / 2))
    return Decision(
        kind=kind,
        regions=tuple(strips),
        supports=tuple(supports),
        chosen="planes",
        alternative="cylinder",
        reason=reason,
        segments=len(strips),
        radius=float(radius),
        axis=tuple(float(x) for x in sec.axis),
        axis_point=tuple(float(x) for x in sec.to_3d(center)),
        alternative_deviation=float(sag),
        min_crease_deg=float(min(creases)),
        max_crease_deg=float(max(creases)),
    )


def _reason(k: int, inner: list[float]) -> str:
    if k == 1:
        return "one_facet"
    if k == 2:
        return "two_facets"
    return "crease_at_least_15deg" if min(inner) >= MAX_TURN_DEG else "not_grouped"


def decisions(ir: Ir) -> list[Decision]:
    normals, origins = {}, {}
    for r in ir.regions:
        if r.surface.type == "plane":
            normals[r.id] = _unit(np.asarray(r.surface.normal, dtype=float))
            origins[r.id] = np.asarray(r.surface.origin, dtype=float)
    if len(normals) < 3:
        return []
    eps = FIT_FACTOR * ir.tolerances.linear
    out: list[Decision] = []
    for axis, members in _clusters(_plane_edges(ir, normals)):
        for comp, nbr, crease in _components(members):
            walked = _walk(comp, nbr)
            if walked is None:
                continue
            chain, closed = walked
            sec = _Section(axis, {r: normals[r] for r in chain}, origins, origins[chain[0]])
            if any(r not in sec.lines for r in chain):
                continue
            creases = [crease[frozenset(p)] for p in zip(chain, chain[1:], strict=False)]
            if closed:
                creases.append(crease[frozenset((chain[-1], chain[0]))])
                if len(chain) >= MIN_RING_SIDES and (fit := _ring(sec, chain, eps)):
                    out.append(
                        _decision(
                            "prism_or_cylinder",
                            sorted(chain),
                            [],
                            sec,
                            *fit,
                            creases,
                            360.0 / len(chain),
                            _reason(len(chain), creases),
                        )
                    )
                continue
            out.extend(_chain_decisions(sec, chain, creases, eps))
    return sorted(out, key=lambda d: d.regions)


def _chain_decisions(sec: _Section, chain: list[int], creases: list[float], eps: float):
    found = []
    m = len(chain) - 1
    for i in range(m + 1):
        for j in range(i + 2, m + 1):
            window = creases[i:j]
            k = j - i - 1
            inner = window[1:-1]
            if sum(window) > MAX_FILLET_TURN_DEG or max(window) > MAX_STRIP_CREASE_DEG:
                continue
            fit = _fillet(sec, chain[i : j + 1], eps)
            if fit is None:
                continue
            corners = np.array(_corners(sec, chain[i : j + 1], False))
            width = float(np.linalg.norm(np.diff(corners, axis=0), axis=1).sum())
            found.append((k, width, i, j, fit, window, inner))
    found.sort(key=lambda f: (-f[0], f[1], f[2]))
    used_strip: set[int] = set()
    used_support: set[int] = set()
    out = []
    for k, _, i, j, fit, window, inner in found:
        strips = chain[i + 1 : j]
        supports = [chain[i], chain[j]]
        if used_strip & (set(strips) | set(supports)) or used_support & set(strips):
            continue
        used_strip |= set(strips)
        used_support |= set(supports)
        kind = KINDS.get(k, "prism_or_cylinder")
        out.append(
            _decision(kind, strips, supports, sec, *fit, window, sum(window) / k, _reason(k, inner))
        )
    return out
