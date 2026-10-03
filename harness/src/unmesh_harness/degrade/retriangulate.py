from __future__ import annotations

import math

import numpy as np

from .core import register


def scheme_for(severity: float) -> str:
    if severity <= 1.0 / 3.0:
        return "fan"
    if severity <= 2.0 / 3.0:
        return "strip"
    return "delaunay"


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    h = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(h, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _loops(nodes: list[tuple], edges: set[tuple[tuple, tuple]]) -> list[list[tuple]]:
    nxt: dict[tuple, list[tuple]] = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    loops = []
    seen: set[tuple[tuple, tuple]] = set()
    for a, b in sorted(edges):
        if (a, b) in seen or (b, a) in seen:
            continue
        loop = [a, b]
        seen.add((a, b))
        while loop[-1] != a:
            cur, prev = loop[-1], loop[-2]
            options = sorted(p for p in nxt[cur] if p != prev)
            nxt_node = options[0] if options else prev
            if (cur, nxt_node) in seen or (nxt_node, cur) in seen:
                if nxt_node != a:
                    break
                loop.append(a)
                break
            seen.add((cur, nxt_node))
            loop.append(nxt_node)
            if len(loop) > len(edges) + 1:
                break
        if loop[-1] == a and len(loop) >= 4:
            loops.append(loop[:-1])
    return loops


def _area2(poly: list[tuple[float, float]]) -> float:
    pairs = zip(poly, poly[1:] + poly[:1], strict=True)
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs) / 2


def _proper_hit(p, q, a, b) -> bool:
    def side(x, y, z):
        return (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])

    return (side(p, q, a) * side(p, q, b) < 0.0) and (side(a, b, p) * side(a, b, q) < 0.0)


def _inside(pt, poly) -> bool:
    c = False
    for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
        if (a[1] > pt[1]) != (b[1] > pt[1]) and pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            c = not c
    return c


def _bridge(outer, holes) -> list | None:
    outer = list(outer)
    for h in holes:
        segs = [(a, b) for a, b in zip(outer, outer[1:] + outer[:1], strict=True)]
        segs += [(a, b) for loop in holes for a, b in zip(loop, loop[1:] + loop[:1], strict=True)]
        best = None
        for j, q in enumerate(h):
            for i, p in enumerate(outer):
                if any(_proper_hit(p, q, a, b) for a, b in segs if len({p, q, a, b}) > 2):
                    continue
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                if not _inside(mid, outer) or any(_inside(mid, o) for o in holes if o is not h):
                    continue
                d = math.dist(p, q)
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            return None
        _, i, j = best
        outer = outer[: i + 1] + h[j:] + h[: j + 1] + outer[i:]
    return outer


def _min_angle(a, b, c) -> float:
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
    if min(ab, bc, ca) <= 0.0:
        return 0.0
    cosines = (
        (ab * ab + bc * bc - ca * ca) / (2 * ab * bc),
        (ab * ab + ca * ca - bc * bc) / (2 * ab * ca),
        (bc * bc + ca * ca - ab * ab) / (2 * bc * ca),
    )
    return min(math.acos(max(-1.0, min(1.0, x))) for x in cosines)


def _clip(poly, scheme: str, rng) -> list[tuple] | None:
    poly = list(poly)
    if _area2(poly) < 0:
        poly = poly[::-1]
    scale = max(math.dist(a, b) for a, b in zip(poly, poly[1:] + poly[:1], strict=True))
    eps = 1e-12 * scale * scale
    focus = None
    side = 1
    if scheme == "strip":
        side = 1 if rng.random() < 0.5 else -1
    elif scheme == "fan":

        def _cross_at(k, loop):
            m = len(loop)
            a, b, c = loop[(k - 1) % m], loop[k], loop[(k + 1) % m]
            return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

        convex = [p for k, p in enumerate(poly) if _cross_at(k, poly) > eps]
        focus = convex[int(rng.integers(len(convex)))] if convex else poly[0]
    tris = []
    guard = len(poly) * len(poly) + 10
    while len(poly) > 3 and guard > 0:
        guard -= 1
        m = len(poly)
        cands = []
        for i in range(m):
            a, b, c = poly[(i - 1) % m], poly[i], poly[(i + 1) % m]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= eps:
                continue
            if all(
                min(
                    (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]),
                    (c[0] - b[0]) * (q[1] - b[1]) - (c[1] - b[1]) * (q[0] - b[0]),
                    (a[0] - c[0]) * (q[1] - c[1]) - (a[1] - c[1]) * (q[0] - c[0]),
                )
                <= eps
                for q in poly
                if q != a and q != b and q != c
            ):
                cands.append(i)
        if not cands:
            return None
        if scheme == "fan":
            ai = poly.index(focus) if focus in poly else None
            pool = [i for i in cands if ai is None or i != ai] or cands
            near = [i for i in pool if ai is None or i in ((ai - 1) % m, (ai + 1) % m)]
            pick = min(near or pool)
        elif scheme == "strip":
            if focus is not None and focus[0] in poly and focus[1] in poly:
                ai, ci = poly.index(focus[0]), poly.index(focus[1])
                want, alt = (ci, ai) if side > 0 else (ai, ci)
                if want != alt and want in cands:
                    pick = want
                    side = -side
                elif alt in cands:
                    pick = alt
                else:
                    pick = min(cands)
                    focus = None
            else:
                pick = cands[int(rng.integers(len(cands)))]
        else:
            pick = max(
                cands, key=lambda i: (_min_angle(poly[(i - 1) % m], poly[i], poly[(i + 1) % m]), -i)
            )
        a, b, c = poly[(pick - 1) % m], poly[pick], poly[(pick + 1) % m]
        tris.append((a, b, c))
        if scheme == "strip":
            focus = (a, c)
        del poly[pick]
    if len(poly) != 3 or abs(_area2(poly)) <= eps:
        return None
    tris.append((poly[0], poly[1], poly[2]))
    return tris


@register(
    "retriangulate",
    "tessellation",
    "identity",
    "planar faces re-triangulated from their boundary loops by ear clipping biased toward a fan "
    "(severity up to 1/3), a strip (up to 2/3) or a Delaunay-like choice (above); boundary "
    "vertices are untouched, curved faces and faces with interior vertices are skipped",
)
def retriangulate(mesh, severity, rng):
    scheme = scheme_for(severity)
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    by_face: dict[int, list[int]] = {}
    for ti, f in enumerate(fids):
        by_face.setdefault(f, []).append(ti)
    done = skipped = 0
    for fid in sorted(by_face):
        face = mesh.faces[fid]
        if face.surface != "plane":
            continue
        normal = np.array(face.params["normal"])
        u, v = _basis(normal)
        tis = by_face[fid]
        counts: dict[tuple[tuple, tuple], int] = {}
        used: set[tuple] = set()
        for ti in tis:
            a, b, c = tris[ti]
            used.update((a, b, c))
            for p, q in ((a, b), (b, c), (c, a)):
                key = (min(p, q), max(p, q))
                counts[key] = counts.get(key, 0) + 1
        bedges = {e for e, n in counts.items() if n == 1}
        nodes = {p for e in bedges for p in e}
        if not bedges or used != nodes:
            skipped += 1
            continue
        loops = _loops(sorted(nodes), bedges)
        if not loops or sum(len(x) for x in loops) != len(nodes):
            skipped += 1
            continue
        proj = {p: (float(np.array(p) @ u), float(np.array(p) @ v)) for p in nodes}
        loops2 = []
        for loop in loops:
            start = min(range(len(loop)), key=lambda i: loop[i])
            loops2.append([proj[loop[(start + i) % len(loop)]] for i in range(len(loop))])
        outer_k = max(range(len(loops2)), key=lambda k: (abs(_area2(loops2[k])), -len(loops2[k])))
        outer = loops2[outer_k]
        if _area2(outer) < 0:
            outer = outer[::-1]
        holes = []
        for k, x in enumerate(loops2):
            if k != outer_k:
                holes.append(x if _area2(x) < 0 else x[::-1])
        merged = _bridge(outer, holes) if holes else list(outer)
        if merged is None:
            skipped += 1
            continue
        back = {v: k for k, v in proj.items()}
        clipped = _clip(merged, scheme, rng)
        if clipped is None or len(clipped) != len(tis):
            skipped += 1
            continue
        for ti, (a, b, c) in zip(tis, clipped, strict=True):
            t = (back[a], back[b], back[c])
            n = np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0]))
            tris[ti] = (t[0], t[2], t[1]) if float(n @ normal) < 0 else t
        done += 1
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(fids, dtype=mesh.face_id.dtype)
    return {"scheme": scheme, "faces_retriangulated": done, "faces_skipped": skipped}
