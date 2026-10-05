from __future__ import annotations

import math

import numpy as np

from .core import register
from .retriangulate import _basis, canonical_triangulate_loops


def segments_for(severity: float) -> int:
    if severity <= 1.0 / 3.0:
        return 3
    if severity <= 2.0 / 3.0:
        return 2
    return 1


def _axial(origin, axis, p):
    return float((np.array(p) - origin) @ axis)


def _line_nodes(pts, svals, wanted):
    by_s = dict(zip(svals, pts, strict=True))
    order = sorted(range(len(pts)), key=lambda i: svals[i])
    out = []
    for s in wanted:
        if s in by_s:
            out.append(by_s[s])
            continue
        node = None
        for i, j in zip(order, order[1:], strict=False):
            if (svals[i] - s) * (svals[j] - s) <= 0.0 and svals[j] != svals[i]:
                f = (s - svals[i]) / (svals[j] - svals[i])
                a, b = np.array(pts[i]), np.array(pts[j])
                node = tuple((a + (b - a) * f).tolist())
                break
        if node is None:
            return None
        out.append(node)
    return out


def _chain(edge_polys: list[list[tuple]]) -> list[list[tuple]] | None:
    ends = {i: (p[0], p[-1]) for i, p in enumerate(edge_polys)}
    remaining = set(ends)
    loops = []
    while remaining:
        i = min(remaining)
        remaining.discard(i)
        loop = list(edge_polys[i])
        start, cur = ends[i][0], ends[i][1]
        guard = len(edge_polys) + 1
        while cur != start and guard > 0:
            guard -= 1
            nxt = None
            for j in sorted(remaining):
                s, e = ends[j]
                if s == cur:
                    nxt = (j, False)
                    break
                if e == cur:
                    nxt = (j, True)
                    break
            if nxt is None:
                return None
            j, rev = nxt
            remaining.discard(j)
            pts = edge_polys[j][::-1] if rev else edge_polys[j]
            loop += pts[1:]
            cur = pts[-1]
        if cur != start:
            return None
        loops.append(loop[:-1])
    if sum(len(x) for x in loops) != sum(len(p) - 1 for p in edge_polys):
        return None
    return loops


def _plan_face(fid, faces, adjacency, polymap, k, face_tris):
    face = faces[fid]
    if face.surface != "cylinder":
        return None
    tangent, cross = [], []
    for idx, adj in enumerate(adjacency):
        if adj.face_a != fid and adj.face_b != fid:
            continue
        (tangent if adj.tangent and adj.curve == "line" else cross).append(idx)
    if len(tangent) != 2 or len(cross) != 2:
        return None
    t1, t2 = (polymap[i] for i in sorted(tangent))
    if len(t1) < 2 or len(t2) < 2:
        return None
    t1ends, t2ends = {t1[0], t1[-1]}, {t2[0], t2[-1]}
    spans = []
    for idx in sorted(cross):
        c = polymap[idx]
        if not ((c[0] in t1ends and c[-1] in t2ends) or (c[0] in t2ends and c[-1] in t1ends)):
            return None
        spans.append(idx)
    neighbours = set()
    for idx in spans:
        adj = adjacency[idx]
        other = adj.face_b if adj.face_a == fid else adj.face_a
        if faces[other].surface != "plane":
            return None
        neighbours.add(other)
    axis = np.array(face.params["axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    origin = np.array(face.params["origin"], dtype=np.float64)
    radius = float(face.params["radius"])
    s1 = [_axial(origin, axis, p) for p in t1]
    s2 = [_axial(origin, axis, p) for p in t2]
    if sorted(set(s1)) != sorted(set(s2)):
        return None
    stations = sorted(set(s1) | set(s2))
    if len(stations) < 2:
        return None
    line0 = _line_nodes(t1, s1, stations)
    linek = _line_nodes(t2, s2, stations)
    if line0 is None or linek is None:
        return None
    foot0 = origin + ((np.array(line0[0]) - origin) @ axis) * axis
    radial = np.array(line0[0]) - foot0
    if abs(float(np.linalg.norm(radial)) - radius) > 1e-6 * (1 + radius):
        return None
    e1 = radial / np.linalg.norm(radial)
    e2 = np.cross(axis, e1)
    tol = 1e-6 * (1 + radius)
    for q in linek:
        foot = origin + ((np.array(q) - origin) @ axis) * axis
        v = np.array(q) - foot
        if abs(float(v @ axis)) > tol or abs(float(np.linalg.norm(v)) - radius) > tol:
            return None
    footk = origin + ((np.array(linek[0]) - origin) @ axis) * axis
    w = np.array(linek[0]) - footk
    theta = math.atan2(float(w @ e2), float(w @ e1))
    if abs(theta) < 1e-9 or abs(theta) > math.pi + 1e-9:
        return None
    if abs(theta) / k >= math.pi - 1e-6:
        k = 2
    lo, hi = (0.0, theta) if theta > 0 else (theta, 0.0)
    for idx in spans:
        for q in polymap[idx][1:-1]:
            foot = origin + ((np.array(q) - origin) @ axis) * axis
            v = np.array(q) - foot
            phi = math.atan2(float(v @ e2), float(v @ e1))
            if not lo - 1e-9 <= phi <= hi + 1e-9:
                return None
    grid = [line0]
    for j in range(1, k):
        th = theta * j / k
        offset = radius * (math.cos(th) * e1 + math.sin(th) * e2)
        grid.append(
            [
                tuple((origin + ((np.array(p) - origin) @ axis) * axis + offset).tolist())
                for p in line0
            ]
        )
    grid.append(linek)
    ref = np.zeros(3)
    for t in face_tris:
        ref += np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0])) / 2
    if float(np.linalg.norm(ref)) <= 0.0:
        return None
    rows = []
    for j in range(k):
        loop = grid[j] + grid[j + 1][::-1]
        band = canonical_triangulate_loops(ref, [loop])
        if band is None:
            return None
        rows += band
    probe = np.cross(np.subtract(rows[0][1], rows[0][0]), np.subtract(rows[0][2], rows[0][0]))
    if float(probe @ ref) < 0:
        rows = [(a, c, b) for a, b, c in rows]
    across = []
    for idx in spans:
        old = polymap[idx]
        first = [grid[j][0] for j in range(k + 1)]
        last = [grid[j][-1] for j in range(k + 1)]
        if {old[0], old[-1]} == {first[0], first[-1]}:
            new = first if old[0] == first[0] else first[::-1]
        elif {old[0], old[-1]} == {last[0], last[-1]}:
            new = last if old[0] == last[0] else last[::-1]
        else:
            return None
        across.append((idx, new))
    return {"grid_tris": rows, "across": across, "neighbours": sorted(neighbours), "segments": k}


@register(
    "fillet_rows",
    "tessellation",
    "identity",
    "cylinder fillet strips whose cross edges meet only planar faces re-tessellated with exactly "
    "3, 2 or 1 segments across their width (severity up to 1/3, 2/3, above), using the fillet's "
    "tangent-line endpoints as nodes (a strip spanning a half turn keeps at least 2, so no chord "
    "passes through the axis); affected planar neighbours are re-triangulated",
)
def fillet_rows(mesh, severity, rng, segments=None):
    k = segments if segments is not None else segments_for(severity)
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    polymap = {idx: [tuple(p) for p in adj.points] for idx, adj in enumerate(mesh.adjacency)}
    by_face: dict[int, list[int]] = {}
    for ti, f in enumerate(fids):
        by_face.setdefault(f, []).append(ti)
    edges_of: dict[int, list[int]] = {}
    for idx, adj in enumerate(mesh.adjacency):
        edges_of.setdefault(adj.face_a, []).append(idx)
        if adj.face_b != adj.face_a:
            edges_of.setdefault(adj.face_b, []).append(idx)
    plans = {}
    skipped = []
    current = dict(polymap)
    for fid in sorted(by_face):
        if mesh.faces[fid].surface != "cylinder":
            continue
        plan = _plan_face(
            fid, mesh.faces, mesh.adjacency, polymap, k, [tris[ti] for ti in by_face[fid]]
        )
        if plan is None or _chords_cross(mesh, plan, edges_of, current):
            skipped.append(fid)
        else:
            plans[fid] = plan
            current.update(plan["across"])
    cache: dict = {}

    def neighbour_tris(other, strips):
        key = (other, tuple(strips))
        if key not in cache:
            across = {idx: pts for f in strips for idx, pts in plans[f]["across"]}
            cache[key] = _retriangulate_neighbour(mesh, other, edges_of, polymap, across)
        return cache[key]

    def adjacent(other, strips):
        return [f for f in strips if other in plans[f]["neighbours"]]

    accepted = sorted(plans)
    while True:
        failing = [
            other
            for other in sorted({n for f in accepted for n in plans[f]["neighbours"]})
            if neighbour_tris(other, adjacent(other, accepted)) is None
        ]
        if not failing:
            break
        hopeless = {other for other in failing if neighbour_tris(other, []) is None}
        if hopeless:
            accepted = [f for f in accepted if not hopeless & set(plans[f]["neighbours"])]
            continue
        strips = adjacent(failing[0], accepted)
        kept = _greedy_keep(lambda sub, x=failing[0]: neighbour_tris(x, sub) is not None, strips)
        accepted = [f for f in accepted if f not in strips or f in kept]
    skipped = sorted(skipped + [f for f in plans if f not in accepted])
    written: dict[int, list[tuple]] = {}
    for f in accepted:
        written.pop(f, None)
        written[f] = plans[f]["grid_tris"]
        for other in plans[f]["neighbours"]:
            written.pop(other, None)
            written[other] = neighbour_tris(other, adjacent(other, accepted))
    out_tris = [t for t, f in zip(tris, fids, strict=True) if f not in written]
    out_fids = [f for f in fids if f not in written]
    for f, block in written.items():
        out_tris += block
        out_fids += [f] * len(block)
    for f in accepted:
        for idx, pts in plans[f]["across"]:
            polymap[idx] = pts
    neighbours = sorted({n for f in accepted for n in plans[f]["neighbours"]})
    mesh.tris = np.array(out_tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(out_fids, dtype=mesh.face_id.dtype)
    for idx, pts in polymap.items():
        mesh.adjacency[idx].points = [list(p) for p in pts]
    return {
        "segments": k,
        "fillet_faces": accepted,
        "neighbours_retriangulated": neighbours,
        "faces_skipped": skipped,
        "half_turn_faces": [f for f in accepted if plans[f]["segments"] != k],
        "triangles_before": len(tris),
        "triangles_after": len(out_tris),
    }


def _chords_cross(mesh, plan, edges_of, polymap) -> bool:
    replaced = {idx for idx, _ in plan["across"]}
    for idx, pts in plan["across"]:
        adj = mesh.adjacency[idx]
        for other in {adj.face_a, adj.face_b} & set(plan["neighbours"]):
            u, v = _basis(np.array(mesh.faces[other].params["normal"], dtype=np.float64))
            frame = np.stack([u, v], axis=1)
            new = np.array(pts, dtype=np.float64) @ frame
            rest = [np.array(polymap[j], dtype=np.float64) @ frame for j in edges_of[other]]
            rest = [r for j, r in zip(edges_of[other], rest, strict=True) if j not in replaced]
            if not rest:
                continue
            a = np.concatenate([r[:-1] for r in rest])
            b = np.concatenate([r[1:] for r in rest])
            if _crossing(new[:-1], new[1:], a, b):
                return True
    return False


def _crossing(p, q, a, b) -> bool:
    def side(x, y, z):
        return (y[..., 0] - x[..., 0]) * (z[..., 1] - x[..., 1]) - (y[..., 1] - x[..., 1]) * (
            z[..., 0] - x[..., 0]
        )

    p, q, a, b = p[:, None], q[:, None], a[None], b[None]
    return bool(np.any((side(p, q, a) * side(p, q, b) < 0) & (side(a, b, p) * side(a, b, q) < 0)))


def _greedy_keep(ok, strips: list, base: tuple = ()) -> tuple:
    if ok([*base, *strips]):
        return tuple(strips)
    if len(strips) == 1:
        return ()
    mid = len(strips) // 2
    left = _greedy_keep(ok, strips[:mid], base)
    return left + _greedy_keep(ok, strips[mid:], base + left)


def _retriangulate_neighbour(mesh, other, edges_of, polymap, across):
    loops = _chain([across.get(idx, polymap[idx]) for idx in edges_of.get(other, [])])
    if loops is None:
        return None
    return canonical_triangulate_loops(np.array(mesh.faces[other].params["normal"]), loops)
