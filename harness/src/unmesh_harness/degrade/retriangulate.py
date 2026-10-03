from __future__ import annotations

import math

import numpy as np

from ..labels import outward_normals
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


def _canonical_key(poly, i: int, m: int):
    a, b, c = poly[(i - 1) % m], poly[i], poly[(i + 1) % m]
    return (_min_angle(a, b, c), tuple(sorted((a, b, c))))


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
            if scheme == "canonical":
                pick = max(cands, key=lambda i: _canonical_key(poly, i, m))
            else:
                pick = max(
                    cands,
                    key=lambda i: (
                        _min_angle(poly[(i - 1) % m], poly[i], poly[(i + 1) % m]),
                        -i,
                    ),
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


def loops_from_tris(tris, tis) -> list[list[tuple]] | None:
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
        return None
    loops = _loops(sorted(nodes), bedges)
    if not loops or sum(len(x) for x in loops) != len(nodes):
        return None
    return loops


def triangulate_loops_3d(normal, loops, scheme, rng) -> list[tuple] | None:
    normal = np.array(normal, dtype=np.float64)
    u, v = _basis(normal)
    nodes = {p for loop in loops for p in loop}
    proj = {p: (float(np.array(p) @ u), float(np.array(p) @ v)) for p in nodes}
    loops2 = []
    for loop in loops:
        start = min(range(len(loop)), key=lambda i: loop[i])
        loops2.append([proj[loop[(start + i) % len(loop)]] for i in range(len(loop))])
    outer_k = max(range(len(loops2)), key=lambda k: (abs(_area2(loops2[k])), -len(loops2[k])))
    outer = loops2[outer_k]
    if _area2(outer) < 0:
        outer = outer[::-1]
    holes = [x if _area2(x) < 0 else x[::-1] for k, x in enumerate(loops2) if k != outer_k]
    merged = _bridge(outer, holes) if holes else list(outer)
    if merged is None:
        return None
    back = {v: k for k, v in proj.items()}
    clipped = _clip(merged, scheme, rng)
    if clipped is None:
        return None
    out = []
    for a, b, c in clipped:
        t = (back[a], back[b], back[c])
        n = np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0]))
        out.append((t[0], t[2], t[1]) if float(n @ normal) < 0 else t)
    return out


ROUND_DECIMALS = 9


def _r3(p) -> tuple:
    return tuple(np.round(np.array(p, dtype=np.float64), ROUND_DECIMALS).tolist())


def _newell(loop) -> np.ndarray:
    n = np.zeros(3)
    m = len(loop)
    for i in range(m):
        a = np.array(loop[i], dtype=np.float64)
        b = np.array(loop[(i + 1) % m], dtype=np.float64)
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def _rot_min(loop: list) -> list:
    k = min(range(len(loop)), key=lambda i: loop[i])
    return loop[k:] + loop[:k]


def canonical_triangulate_loops(ref_normal, loops: list[list[tuple]]) -> list[tuple] | None:
    ref = np.array(ref_normal, dtype=np.float64)
    back: dict = {}
    rloops = []
    for loop in loops:
        if len(loop) < 3:
            return None
        rl = [_r3(p) for p in loop]
        for o, r in zip(loop, rl, strict=True):
            if r in back and back[r] != tuple(o):
                return None
            back[r] = tuple(o)
        dedup = [rl[0]]
        for p in rl[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        if len(dedup) < 3:
            return None
        n = _newell(dedup)
        if float(np.linalg.norm(n)) <= 0.0:
            return None
        rloops.append(_rot_min(dedup[::-1] if float(n @ ref) < 0 else dedup))
    areas = []
    for rl in rloops:
        n = _newell(rl)
        if float(np.linalg.norm(n)) <= 0.0:
            return None
        areas.append(float(n @ ref))
    outer_k = max(range(len(rloops)), key=lambda k: abs(areas[k]))
    basis_n = _newell(rloops[outer_k])
    if float(basis_n @ ref) <= 0.0:
        return None
    u, v = _basis(basis_n)
    proj = {p: (float(np.array(p) @ u), float(np.array(p) @ v)) for rl in rloops for p in rl}
    loops2 = []
    for rl in rloops:
        loop2 = [proj[p] for p in rl]
        start = min(range(len(loop2)), key=lambda i: loop2[i])
        loops2.append(loop2[start:] + loop2[:start])
    outer = loops2[outer_k]
    if _area2(outer) < 0:
        outer = outer[::-1]
    holes = [x if _area2(x) < 0 else x[::-1] for k, x in enumerate(loops2) if k != outer_k]
    merged = _bridge(outer, holes) if holes else list(outer)
    if merged is None:
        return None
    start = min(range(len(merged)), key=lambda i: merged[i])
    merged = merged[start:] + merged[:start]
    if _area2(merged) < 0:
        merged = merged[::-1]
    back2 = {}
    for p3, p2 in proj.items():
        if p2 in back2 and back2[p2] != p3:
            return None
        back2[p2] = p3
    clipped = _clip(merged, "canonical", None)
    if clipped is None:
        return None
    out = []
    for a, b, c in clipped:
        t = (back[back2[a]], back[back2[b]], back[back2[c]])
        n = np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0]))
        out.append((t[0], t[2], t[1]) if float(n @ ref) < 0 else t)
    return out


def retriangulate_face(fid, tris, tis, faces, scheme, rng) -> bool:
    loops = loops_from_tris(tris, tis)
    if loops is None:
        return False
    if scheme == "canonical":
        clipped = canonical_triangulate_loops(np.array(faces[fid].params["normal"]), loops)
    else:
        clipped = triangulate_loops_3d(np.array(faces[fid].params["normal"]), loops, scheme, rng)
    if clipped is None or len(clipped) != len(tis):
        return False
    for ti, t in zip(tis, clipped, strict=True):
        tris[ti] = t
    return True


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
        if mesh.faces[fid].surface != "plane":
            continue
        if retriangulate_face(fid, tris, by_face[fid], mesh.faces, scheme, rng):
            done += 1
        else:
            skipped += 1
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(fids, dtype=mesh.face_id.dtype)
    return {"scheme": scheme, "faces_retriangulated": done, "faces_skipped": skipped}


def _canonical_band(face, tris, tis) -> bool:
    prm = face.params
    axis = np.array(prm["axis"], dtype=np.float64)
    if float(np.linalg.norm(axis)) <= 0.0:
        return False
    axis /= np.linalg.norm(axis)
    origin = np.array(prm["origin"], dtype=np.float64)
    corners = [tris[ti] for ti in tis]
    verts = {p for t in corners for p in t}
    stations: dict = {}
    for p in verts:
        s = round(float((np.array(p) - origin) @ axis), ROUND_DECIMALS)
        stations.setdefault(s, []).append(p)
    if len(stations) != 2:
        return False
    (_, blo), (_, bhi) = sorted(stations.items())

    def rkey(p):
        a = float((np.array(p) - origin) @ axis)
        r = np.array(p, dtype=np.float64) - origin - a * axis
        return tuple(np.round(r, ROUND_DECIMALS).tolist())

    lomap: dict = {}
    for p in blo:
        lomap.setdefault(rkey(p), []).append(p)
    himap: dict = {}
    for p in bhi:
        himap.setdefault(rkey(p), []).append(p)
    if set(lomap) != set(himap):
        return False
    if any(len(v) != 1 for v in list(lomap.values()) + list(himap.values())):
        return False
    u, v = _basis(axis)
    angles = {}
    for k in lomap:
        radial = np.array(k, dtype=np.float64)
        angles[k] = math.atan2(float(radial @ v), float(radial @ u))
    ring = sorted(lomap, key=lambda k: angles[k])
    if len(ring) < 3:
        return False
    index = {ti: t for ti, t in zip(tis, corners, strict=True)}
    repl = {}
    for j in range(len(ring)):
        k0, k1 = ring[j], ring[(j + 1) % len(ring)]
        quad = [lomap[k0][0], lomap[k1][0], himap[k1][0], himap[k0][0]]
        qset = set(quad)
        pair = [ti for ti, t in index.items() if set(t) <= qset]
        if len(pair) != 2:
            return False
        got = set()
        for ti in pair:
            got.update(index[ti])
        if got != qset:
            return False
        if len(set(index[pair[0]]) & set(index[pair[1]])) != 2:
            return False
        center = sum(np.array(p) for p in quad) / 4.0
        try:
            ref = outward_normals(face, center[None])[0]
        except Exception:
            return False
        if not np.all(np.isfinite(ref)):
            return False
        new = canonical_triangulate_loops(ref, [quad])
        if new is None or len(new) != 2:
            return False
        repl[pair[0]] = new[0]
        repl[pair[1]] = new[1]
    for ti, t in repl.items():
        tris[ti] = t
    return True


def canonical_planar(mesh) -> dict:
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    by_face: dict[int, list[int]] = {}
    for ti, f in enumerate(fids):
        by_face.setdefault(f, []).append(ti)
    done = skipped = 0
    for fid in sorted(by_face):
        face = mesh.faces[fid]
        tis = by_face[fid]
        if face.surface == "plane":
            if retriangulate_face(fid, tris, tis, mesh.faces, "canonical", None):
                done += 1
            else:
                skipped += 1
        elif face.surface == "cylinder":
            if _canonical_band(face, tris, tis):
                done += 1
            else:
                skipped += 1
        else:
            skipped += 1
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    return {"faces_retriangulated": done, "faces_skipped": skipped}
