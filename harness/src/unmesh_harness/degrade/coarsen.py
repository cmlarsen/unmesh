from __future__ import annotations

import math

import numpy as np

from ..labels import outward_normals
from .core import register, vertex_table

MAX_PASSES = 40


def threshold_mm(severity: float, diagonal: float) -> float:
    return diagonal * 0.01 * 10.0**severity


def _find(root: list[int], i: int) -> int:
    while root[i] != i:
        root[i] = root[root[i]]
        i = root[i]
    return i


@register(
    "coarsen",
    "tessellation",
    "identity",
    "shortest edges collapsed until none is shorter than h = diagonal * 0.01 * 10**severity "
    "(diagonal / 100 at severity 0+, / 31.6 at 0.5, / 10 at 1); b collapses into a only if "
    "faces(b) is a subset of faces(a), and a node of an edge polyline only into an adjacent "
    "node of the same polyline; the surviving endpoint keeps its coordinates and no collapse "
    "may fold a triangle or create a zero-area one, so curved nodes stay on the analytic "
    "surface and cylinders become N-gon prisms",
)
def coarsen(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    n = len(uniq)
    rep = [tuple(p) for p in uniq.tolist()]
    tri = inverse.reshape(-1, 3).tolist()
    face = mesh.face_id.tolist()
    live = [True] * len(tri)
    root = list(range(n))
    cad = {tuple(p) for p in mesh.vertices} | {
        tuple(p) for adj in mesh.adjacency for p in (adj.points[:1] + adj.points[-1:])
    }
    counts = [0] * len(mesh.faces)
    for f in face:
        counts[f] += 1
    before = len(tri)
    flat = uniq
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    h = threshold_mm(severity, diagonal)
    collapses = 0
    lookup = {tuple(p): i for i, p in enumerate(uniq.tolist())}
    seqs = [[lookup[tuple(p)] for p in adj.points] for adj in mesh.adjacency]

    def resolved(t):
        return [_find(root, v) for v in t]

    def poly_allows(a: int, b: int) -> bool:
        for s in seqs:
            if b not in s:
                continue
            adj = set()
            for j, v in enumerate(s):
                if v != b:
                    continue
                if j > 0 and s[j - 1] != b:
                    adj.add(s[j - 1])
                if j < len(s) - 1 and s[j + 1] != b:
                    adj.add(s[j + 1])
            if a not in adj:
                return False
        return True

    def collapse(a: int, b: int, vtris: dict[int, set[int]], nbrs: dict[int, set[int]]) -> bool:
        if rep[a] in cad and rep[b] in cad:
            return False
        if rep[b] in cad:
            return False
        fb = {face[ti] for ti in vtris.get(b, ())}
        if not fb <= {face[ti] for ti in vtris.get(a, ())}:
            return False
        if not poly_allows(a, b):
            return False
        shared = [ti for ti in vtris.get(a, ()) if ti in vtris.get(b, set())]
        opposite = set()
        doomed = True
        for ti in shared:
            r = resolved(tri[ti])
            third = [v for v in r if v != a and v != b]
            if len(third) != 1:
                doomed = False
                break
            opposite.add(third[0])
        if not doomed:
            return False
        if (nbrs.get(a, set()) & nbrs.get(b, set())) != opposite:
            return False
        flap = [ti for ti in vtris.get(b, ()) if ti not in vtris.get(a, set())]
        if any(counts[face[ti]] <= 1 for ti in shared):
            return False
        for ti in flap:
            r = tri[ti]
            old = [np.array(rep[v]) for v in r]
            new = [np.array(rep[a] if v == b else rep[v]) for v in r]
            n_old = np.cross(old[1] - old[0], old[2] - old[0])
            n_new = np.cross(new[1] - new[0], new[2] - new[0])
            if float(np.linalg.norm(n_new)) <= 1e-14:
                return False
            try:
                on = outward_normals(mesh.faces[face[ti]], (new[0] + new[1] + new[2])[None] / 3.0)[
                    0
                ]
            except Exception:
                on = None
            if on is not None and np.all(np.isfinite(on)):
                if float(n_new @ on) <= 0.0:
                    return False
            elif float(np.linalg.norm(n_old)) > 1e-14 and float(n_old @ n_new) <= 0.0:
                return False
        root[b] = a
        for ti in list(vtris.get(b, ())):
            r = tri[ti]
            sub = [a if v == b else v for v in r]
            if len(set(sub)) < 3:
                if live[ti]:
                    live[ti] = False
                    counts[face[ti]] -= 1
                for v in set(sub):
                    if v in vtris:
                        vtris[v].discard(ti)
            else:
                tri[ti] = sub
                vtris.setdefault(a, set()).add(ti)
        vtris.pop(b, None)
        for v in list(nbrs.pop(b, set())):
            if v == a:
                continue
            nbrs[v].discard(b)
            nbrs[v].add(a)
            nbrs.setdefault(a, set()).add(v)
        nbrs.get(a, set()).discard(b)
        for s in seqs:
            if b in s:
                merged = [a if v == b else v for v in s]
                dedup = [merged[0]]
                for v in merged[1:]:
                    if v != dedup[-1]:
                        dedup.append(v)
                s[:] = dedup
        return True

    for _ in range(MAX_PASSES):
        edges: dict[tuple[int, int], list[int]] = {}
        for ti, t in enumerate(tri):
            if not live[ti]:
                continue
            r = resolved(t)
            if len(set(r)) < 3:
                live[ti] = False
                counts[face[ti]] -= 1
                continue
            for a, b in ((r[0], r[1]), (r[1], r[2]), (r[2], r[0])):
                edges.setdefault((min(a, b), max(a, b)), []).append(ti)
        ordered = sorted(
            edges.items(),
            key=lambda kv: (math.dist(rep[kv[0][0]], rep[kv[0][1]]), rep[kv[0][0]], rep[kv[0][1]]),
        )
        vtris: dict[int, set[int]] = {}
        nbrs: dict[int, set[int]] = {}
        for ti, t in enumerate(tri):
            if not live[ti]:
                continue
            r = resolved(t)
            for v in r:
                vtris.setdefault(v, set()).add(ti)
            for a, b in ((r[0], r[1]), (r[1], r[2]), (r[2], r[0])):
                nbrs.setdefault(a, set()).add(b)
                nbrs.setdefault(b, set()).add(a)
        moved = 0
        for (x, y), _ in ordered:
            ra, rb = _find(root, x), _find(root, y)
            if ra == rb or math.dist(rep[ra], rep[rb]) > h:
                continue
            a, b = ra, rb
            if rep[b] in cad and rep[a] not in cad:
                a, b = b, a
            elif rep[a] not in cad and rep[b] not in cad and b < a:
                a, b = b, a
            done = collapse(a, b, vtris, nbrs)
            if not done:
                done = collapse(b, a, vtris, nbrs)
            if done:
                collapses += 1
                moved += 1
        if not moved:
            break

    kept = [i for i, v in enumerate(live) if v]
    mesh.tris = np.array(
        [[list(rep[_find(root, v)]) for v in tri[i]] for i in kept], dtype=np.float64
    ).reshape(-1, 3, 3)
    mesh.face_id = np.array([face[i] for i in kept], dtype=mesh.face_id.dtype)

    def remap(pt):
        return list(rep[_find(root, lookup[tuple(pt)])]) if tuple(pt) in lookup else list(pt)

    if mesh.vertices:
        mesh.vertices = [remap(p) for p in mesh.vertices]
    for adj in mesh.adjacency:
        pts = [remap(p) for p in adj.points]
        dedup = [pts[0]]
        for p in pts[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        adj.points = dedup
    return {
        "threshold_mm": h,
        "bbox_diagonal_mm": diagonal,
        "collapses": collapses,
        "triangles_before": before,
        "triangles_after": len(kept),
    }
