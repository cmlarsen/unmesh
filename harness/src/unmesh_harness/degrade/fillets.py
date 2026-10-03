from __future__ import annotations

import math

import numpy as np

from .core import register
from .retriangulate import triangulate_loops_3d


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
    rows = []
    for si in range(len(stations) - 1):
        for j in range(k):
            a, b, c, d = grid[j][si], grid[j + 1][si], grid[j + 1][si + 1], grid[j][si + 1]
            rows += [(a, b, c), (a, c, d)]
    ref = np.zeros(3)
    for t in face_tris:
        ref += np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0])) / 2
    if float(np.linalg.norm(ref)) <= 0.0:
        return None
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
    return {"grid_tris": rows, "across": across, "neighbours": sorted(neighbours)}


@register(
    "fillet_rows",
    "tessellation",
    "identity",
    "cylinder fillet strips whose cross edges meet only planar faces re-tessellated with exactly "
    "3, 2 or 1 segments across their width (severity up to 1/3, 2/3, above), using the fillet's "
    "tangent-line endpoints as nodes; affected planar neighbours are re-triangulated",
)
def fillet_rows(mesh, severity, rng):
    k = segments_for(severity)
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    polymap = {idx: [tuple(p) for p in adj.points] for idx, adj in enumerate(mesh.adjacency)}
    by_face: dict[int, list[int]] = {}
    for ti, f in enumerate(fids):
        by_face.setdefault(f, []).append(ti)
    before = len(tris)
    done, skipped, neighbours = [], [], []
    for fid in sorted(by_face):
        if mesh.faces[fid].surface != "cylinder":
            continue
        plan = _plan_face(
            fid, mesh.faces, mesh.adjacency, polymap, k, [tris[ti] for ti in by_face[fid]]
        )
        if plan is None:
            skipped.append(fid)
            continue
        drop = set(by_face[fid])
        tris2 = [t for ti, t in enumerate(tris) if ti not in drop]
        fids2 = [f for ti, f in enumerate(fids) if ti not in drop]
        tris2 += plan["grid_tris"]
        fids2 += [fid] * len(plan["grid_tris"])
        poly2 = dict(polymap)
        for idx, pts in plan["across"]:
            poly2[idx] = pts
        ok = True
        for other in plan["neighbours"]:
            edge_polys = [
                poly2[idx]
                for idx, adj in enumerate(mesh.adjacency)
                if adj.face_a == other or adj.face_b == other
            ]
            loops = _chain(edge_polys)
            if loops is None:
                ok = False
                break
            clipped = triangulate_loops_3d(
                np.array(mesh.faces[other].params["normal"]), loops, "delaunay", rng
            )
            if clipped is None:
                ok = False
                break
            drop2 = {ti for ti, f in enumerate(fids2) if f == other}
            tris2 = [t for ti, t in enumerate(tris2) if ti not in drop2]
            fids2 = [f for ti, f in enumerate(fids2) if ti not in drop2]
            tris2 += clipped
            fids2 += [other] * len(clipped)
        if not ok:
            skipped.append(fid)
            continue
        tris, fids, polymap = tris2, fids2, poly2
        by_face = {}
        for ti, f in enumerate(fids):
            by_face.setdefault(f, []).append(ti)
        done.append(fid)
        neighbours += [n for n in plan["neighbours"] if n not in neighbours]
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(fids, dtype=mesh.face_id.dtype)
    for idx, pts in polymap.items():
        mesh.adjacency[idx].points = [list(p) for p in pts]
    return {
        "segments": k,
        "fillet_faces": done,
        "neighbours_retriangulated": sorted(neighbours),
        "faces_skipped": skipped,
        "triangles_before": before,
        "triangles_after": len(tris),
    }
