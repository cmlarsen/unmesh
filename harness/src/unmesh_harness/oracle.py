from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import unmesh
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
    Sphere,
    Tolerances,
    Torus,
    Vertex,
)

from .labels import TANGENT_THRESHOLD_DEG, EdgeAdjacency, FaceInfo, LabeledMesh, distance_to_surface


def _surface(face: FaceInfo):
    p = face.params
    orientation = "reversed" if face.reversed else "same"
    if face.surface == "plane":
        return Plane(tuple(p["origin"]), tuple(p["normal"]))
    if face.surface == "cylinder":
        return Cylinder(tuple(p["origin"]), tuple(p["axis"]), p["radius"], orientation)
    if face.surface == "cone":
        return Cone(tuple(p["apex"]), tuple(p["axis"]), p["half_angle"], orientation)
    if face.surface == "sphere":
        return Sphere(tuple(p["center"]), p["radius"], orientation)
    if face.surface == "torus":
        return Torus(
            tuple(p["center"]),
            tuple(p["axis"]),
            p["major_radius"],
            p["minor_radius"],
            orientation,
        )
    raise NotImplementedError(f"oracle IR has no region for surface type {face.surface!r}")


def _residual(face: FaceInfo, tris: np.ndarray) -> Residual:
    pts = np.concatenate([tris.reshape(-1, 3), tris.mean(axis=1)])
    d = distance_to_surface(face, pts)
    return Residual(float(np.sqrt(np.mean(d**2))), float(d.max()))


class _Directed:
    def __init__(self, adj: EdgeAdjacency):
        fwd = adj.forward_in_a
        self.points = adj.points if fwd else adj.points[::-1]
        self.start = adj.start_vertex if fwd else adj.end_vertex
        self.end = adj.end_vertex if fwd else adj.start_vertex
        self.dihedral = math.degrees(adj.dihedral)
        samples = [math.degrees(s) for s in adj.dihedral_samples]
        self.samples = samples if (fwd or not samples) else samples[::-1]
        if len(self.samples) != len(self.points):
            self.samples = []


class _Split:
    def __init__(self, point: tuple[float, ...], faces: tuple[int, int]):
        self.point = point
        self.faces = faces


class _Seg:
    def __init__(self, points, dihedral: float, start: int | _Split, end: int | _Split):
        self.points = points
        self.dihedral = dihedral
        self.start = start
        self.end = end


KIND_HYSTERESIS_DEG = 0.5
MIN_RUN_SEGMENTS = 2


def _span_nodes(span: tuple[int, int, bool], n: int) -> list[int]:
    lo, hi, wrap = span
    if not wrap:
        return list(range(lo, hi + 1))
    return list(range(lo, n)) + list(range(1, hi + 1))


def _span_len(span: tuple[int, int, bool], n: int) -> int:
    lo, hi, wrap = span
    return (n - 1 - lo + hi) if wrap else hi - lo


def _spans(joints: list[int], n: int, closed: bool) -> list[tuple[int, int, bool]]:
    if not closed:
        bounds = [0] + joints + [n - 1]
        return [(bounds[j], bounds[j + 1], False) for j in range(len(bounds) - 1)]
    return [
        (joints[j], joints[(j + 1) % len(joints)], j == len(joints) - 1) for j in range(len(joints))
    ]


def _split_runs(d: _Directed, threshold_deg: float, pair: tuple[int, int]) -> list[_Seg]:
    samples = d.samples
    n = len(d.points)
    if len(samples) != n or n < 2:
        return [_Seg(d.points, d.dihedral, d.start, d.end)]
    lo = threshold_deg - KIND_HYSTERESIS_DEG
    hi = threshold_deg + KIND_HYSTERESIS_DEG
    tangent = [s < lo for s in samples]
    transversal = [s > hi for s in samples]
    if not any(tangent) or not any(transversal):
        return [_Seg(d.points, d.dihedral, d.start, d.end)]
    closed = d.start == d.end
    pts = [tuple(p) for p in d.points]
    out = [i for i in range(n) if tangent[i] or transversal[i]]
    pairs = list(zip(out, out[1:], strict=False))
    if closed and len(out) > 1:
        pairs.append((out[-1], out[0]))
    joints = set()
    for a, b in pairs:
        if tangent[a] == tangent[b]:
            continue
        between = list(range(a + 1, b)) if a < b else list(range(a + 1, n)) + list(range(b))
        cands = between or [a, b]
        joints.add(min(cands, key=lambda i: (abs(samples[i] - threshold_deg), pts[i], i)))
    joints = sorted(joints)

    def drop_key(j: int) -> tuple[tuple[float, ...], float]:
        return (pts[j], samples[j])

    while True:
        spans = _spans(joints, n, closed)
        bad = [s for s in spans if _span_len(s, n) < MIN_RUN_SEGMENTS]
        if not bad:
            break
        if len(joints) < (2 if closed else 1):
            return [_Seg(d.points, d.dihedral, d.start, d.end)]
        span = min(bad, key=lambda s: (_span_len(s, n), sorted(pts[i] for i in _span_nodes(s, n))))
        joints.remove(min((b for b in span[:2] if b in joints), key=drop_key))
    if not joints or (closed and len(joints) < 2):
        return [_Seg(d.points, d.dihedral, d.start, d.end)]
    while True:
        spans = _spans(joints, n, closed)
        kinds = [
            float(np.median([samples[i] for i in _span_nodes(s, n)])) < threshold_deg for s in spans
        ]
        if closed:
            shared = [
                joints[(j + 1) % len(joints)]
                for j in range(len(spans))
                if kinds[j] == kinds[(j + 1) % len(spans)]
            ]
        else:
            shared = [joints[j] for j in range(len(spans) - 1) if kinds[j] == kinds[j + 1]]
        if not shared:
            break
        joints.remove(min(shared, key=drop_key))
        if not joints or (closed and len(joints) < 2):
            return [_Seg(d.points, d.dihedral, d.start, d.end)]
    splits = {j: _Split(pts[j], pair) for j in joints}
    segs = []
    for j, span in enumerate(_spans(joints, n, closed)):
        nodes = _span_nodes(span, n)
        if closed:
            start = splits[joints[j]]
            end = splits[joints[(j + 1) % len(joints)]]
        else:
            start = d.start if j == 0 else splits[span[0]]
            end = d.end if j == len(joints) else splits[span[1]]
        segs.append(
            _Seg([pts[i] for i in nodes], float(np.median([samples[i] for i in nodes])), start, end)
        )
    return segs


def _walk(first, outgoing, used, roles):
    chain = [first]
    used.add(id(first))
    while chain[-1].end not in roles:
        nxt = [e for e in outgoing[chain[-1].end] if id(e) not in used]
        if not nxt:
            break
        used.add(id(nxt[0]))
        chain.append(nxt[0])
    return chain


def _merge(chain):
    pts = list(chain[0].points)
    for e in chain[1:]:
        pts.extend(e.points[1:])
    return pts, float(np.median([e.dihedral for e in chain]))


def build_oracle_ir(mesh: LabeledMesh, tangent_threshold_deg: float = TANGENT_THRESHOLD_DEG) -> Ir:
    verts, _, kept_src, _ = unmesh.weld(mesh.tris, 1e-6)
    kept = np.zeros(len(mesh.tris), dtype=bool)
    kept[kept_src] = True
    order = np.argsort(mesh.face_id, kind="stable")
    bounds = np.searchsorted(mesh.face_id[order], np.arange(len(mesh.faces) + 1))
    regions = []
    for face in mesh.faces:
        ids = order[bounds[face.id] : bounds[face.id + 1]]
        ids = ids[kept[ids]]
        regions.append(
            Region(face.id, _surface(face), ids.tolist(), _residual(face, mesh.tris[ids]))
        )

    def kind_of(deg: float) -> str:
        return "tangent" if deg < tangent_threshold_deg else "transversal"

    by_pair: dict[tuple[int, int], list[_Seg]] = defaultdict(list)
    faces_at: dict[int | _Split, set[int]] = defaultdict(set)
    ends_at: dict[int | _Split, list[str]] = defaultdict(list)
    for adj in mesh.adjacency:
        pair = (adj.face_a, adj.face_b)
        for seg in _split_runs(_Directed(adj), tangent_threshold_deg, pair):
            by_pair[pair].append(seg)
            kind = kind_of(seg.dihedral)
            for v in (seg.start, seg.end):
                faces_at[v].update(pair)
                ends_at[v].append(kind)

    def position_of(v) -> tuple[float, ...]:
        return v.point if isinstance(v, _Split) else tuple(mesh.vertices[v])

    roles: dict[int | _Split, str] = {}
    for v, faces in faces_at.items():
        if len(faces) >= 3:
            roles[v] = "junction"
        elif len(faces) == 2 and sorted(ends_at[v]) == ["tangent", "transversal"]:
            roles[v] = "kind_change"

    order = sorted(roles, key=position_of)
    vertex_id = {v: i for i, v in enumerate(order)}
    vertices = [
        Vertex(
            vertex_id[v],
            roles[v],
            position_of(v),
            sorted(faces_at[v]),
            [position_of(v)],
        )
        for v in order
    ]

    adjacencies = []
    for pair in sorted(by_pair):
        edges = by_pair[pair]
        outgoing: dict[int | _Split, list[_Seg]] = defaultdict(list)
        for e in edges:
            outgoing[e.start].append(e)
        used: set[int] = set()
        boundaries = []

        for e in edges:
            if id(e) in used or e.start not in roles:
                continue
            chain = _walk(e, outgoing, used, roles)
            pts, deg = _merge(chain)
            boundaries.append(
                Boundary(
                    kind_of(deg),
                    deg,
                    False,
                    vertex_id[chain[0].start],
                    vertex_id[chain[-1].end],
                    [tuple(p) for p in pts],
                )
            )
        for e in edges:
            if id(e) in used:
                continue
            chain = _walk(e, outgoing, used, roles)
            pts, deg = _merge(chain)
            if pts[0] == pts[-1]:
                pts = pts[:-1]
            boundaries.append(
                Boundary(kind_of(deg), deg, True, None, None, [tuple(p) for p in pts])
            )
        boundaries.sort(key=lambda b: b.points[0])
        adjacencies.append(Adjacency(pair, boundaries))

    if mesh.shells:
        shells = [Shell(s.closed, s.role, s.parent, list(s.faces)) for s in mesh.shells]
    else:
        shells = [Shell(True, "outer", None, [f.id for f in mesh.faces])]

    return Ir(
        Tolerances(linear=mesh.linear_deflection, tangent_threshold_deg=tangent_threshold_deg),
        Source(len(mesh.tris), len(verts)),
        shells,
        regions,
        adjacencies,
        vertices,
    )


def _facets_dihedral(ir: Ir, analytic: int, bd: Boundary, normal_of) -> float:
    from unmesh._writer import geometry as geo

    surface = ir.regions[analytic].surface
    pts = [np.asarray(p, dtype=float) for p in bd.points]
    n = len(pts)
    chords = [(i, (i + 1) % n) for i in range(n if bd.closed else n - 1)]
    samples = [(pts[i], pts[j], 0.5 * (pts[i] + pts[j])) for i, j in chords]
    if not bd.closed and n > 2:
        samples = [(pts[max(i - 1, 0)], pts[i], pts[i]) for i in range(1, n)]
    angles = []
    for p, q, x in samples:
        c = float(geo.outward(surface, x) @ normal_of(p, q))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return float(np.median(angles))


def force_facets(ir: Ir, region: int, tris: np.ndarray) -> Ir:
    tris = np.asarray(tris, dtype=float)
    out = Ir.loads(ir.dumps())
    target = out.regions[region]
    index: dict[tuple[float, ...], int] = {}
    faces = []
    for t in target.triangles:
        faces.append(tuple(index.setdefault(tuple(map(float, p)), len(index)) for p in tris[t]))
    vertices = list(index)
    target.surface = Facets(vertices, faces)
    target.residual = None
    normals: dict[tuple, np.ndarray] = {}
    for f in faces:
        a, b, c = (np.asarray(vertices[i]) for i in f)
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        for i, j in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            normals[(vertices[i], vertices[j])] = n
            normals[(vertices[j], vertices[i])] = n

    def normal_of(p, q):
        return normals[(tuple(map(float, p)), tuple(map(float, q)))]

    threshold = out.tolerances.tangent_threshold_deg
    for adj in out.adjacencies:
        if region not in adj.regions:
            continue
        other = adj.regions[0] if adj.regions[1] == region else adj.regions[1]
        for bd in adj.boundaries:
            bd.dihedral_deg = _facets_dihedral(out, other, bd, normal_of)
            bd.kind = "tangent" if bd.dihedral_deg < threshold else "transversal"
    return _merge_kind_changes(out)


def _merge_kind_changes(ir: Ir) -> Ir:
    while True:
        ends: dict[int, list[tuple[Adjacency, Boundary]]] = defaultdict(list)
        for adj in ir.adjacencies:
            for bd in adj.boundaries:
                if not bd.closed:
                    ends[bd.start_vertex].append((adj, bd))
                    ends[bd.end_vertex].append((adj, bd))
        stale = next(
            (
                v.id
                for v in ir.vertices
                if v.role == "kind_change" and len({bd.kind for _, bd in ends[v.id]}) < 2
            ),
            None,
        )
        if stale is None:
            return ir
        (adj, first), (_, second) = ends[stale]
        if first.end_vertex != stale:
            first, second = second, first
        adj.boundaries.remove(first)
        if first is second:
            first.closed, first.start_vertex, first.end_vertex = True, None, None
            first.points = first.points[:-1]
            adj.boundaries.append(first)
        else:
            adj.boundaries.remove(second)
            joined = Boundary(
                first.kind,
                float(np.median([first.dihedral_deg, second.dihedral_deg])),
                first.start_vertex == second.end_vertex,
                first.start_vertex,
                second.end_vertex,
                [*first.points, *second.points[1:]],
            )
            if joined.closed:
                joined.start_vertex = joined.end_vertex = None
                joined.points = joined.points[:-1]
            adj.boundaries.append(joined)
        adj.boundaries.sort(key=lambda b: b.points[0])
        renumber = {v.id: i for i, v in enumerate(x for x in ir.vertices if x.id != stale)}
        ir.vertices = [v for v in ir.vertices if v.id != stale]
        for v in ir.vertices:
            v.id = renumber[v.id]
        for a in ir.adjacencies:
            for bd in a.boundaries:
                if not bd.closed:
                    bd.start_vertex = renumber[bd.start_vertex]
                    bd.end_vertex = renumber[bd.end_vertex]
