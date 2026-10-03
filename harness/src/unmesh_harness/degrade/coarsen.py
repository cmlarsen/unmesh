from __future__ import annotations

import math

import numpy as np

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
    "(diagonal / 100 at severity 0+, / 31.6 at 0.5, / 10 at 1); the surviving endpoint keeps its "
    "coordinates, so curved nodes stay on the analytic surface and cylinders become N-gon prisms",
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

    def resolved(t):
        return [_find(root, v) for v in t]

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
            if rep[ra] in cad and rep[rb] in cad:
                continue
            if rep[rb] in cad or (rep[ra] not in cad and rb < ra):
                ra, rb = rb, ra
            shared = [ti for ti in vtris.get(ra, ()) if ti in vtris.get(rb, set())]
            opposite = set()
            doomed = True
            for ti in shared:
                r = resolved(tri[ti])
                third = [v for v in r if v != ra and v != rb]
                if len(third) != 1:
                    doomed = False
                    break
                opposite.add(third[0])
            if not doomed:
                continue
            if (nbrs.get(ra, set()) & nbrs.get(rb, set())) != opposite:
                continue
            flap = [ti for ti in vtris.get(rb, ()) if ti not in vtris.get(ra, set())]
            if any(counts[face[ti]] <= 1 for ti in shared):
                continue
            ok = True
            for ti in flap:
                r = tri[ti]
                old = [np.array(rep[v]) for v in r]
                new = [np.array(rep[ra] if v == rb else rep[v]) for v in r]
                n_old = np.cross(old[1] - old[0], old[2] - old[0])
                n_new = np.cross(new[1] - new[0], new[2] - new[0])
                if float(n_old @ n_new) <= 0.0:
                    ok = False
                    break
            if not ok:
                continue
            root[rb] = ra
            for ti in list(vtris.get(rb, ())):
                r = tri[ti]
                sub = [ra if v == rb else v for v in r]
                if len(set(sub)) < 3:
                    if live[ti]:
                        live[ti] = False
                        counts[face[ti]] -= 1
                    for v in set(sub):
                        if v in vtris:
                            vtris[v].discard(ti)
                else:
                    tri[ti] = sub
                    vtris.setdefault(ra, set()).add(ti)
            vtris.pop(rb, None)
            for v in list(nbrs.pop(rb, set())):
                if v == ra:
                    continue
                nbrs[v].discard(rb)
                nbrs[v].add(ra)
                nbrs.setdefault(ra, set()).add(v)
            nbrs.get(ra, set()).discard(rb)
            collapses += 1
            moved += 1
        if not moved:
            break

    kept = [i for i, v in enumerate(live) if v]
    mesh.tris = np.array(
        [[list(rep[_find(root, v)]) for v in tri[i]] for i in kept], dtype=np.float64
    ).reshape(-1, 3, 3)
    mesh.face_id = np.array([face[i] for i in kept], dtype=mesh.face_id.dtype)
    lookup = {tuple(p): i for i, p in enumerate(uniq.tolist())}

    def remap(pt):
        return list(rep[_find(root, lookup[tuple(pt)])])

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
