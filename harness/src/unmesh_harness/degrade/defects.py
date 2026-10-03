from __future__ import annotations

import numpy as np

from .core import register
from .refine import _expand, _key, _split_triangle, project_onto


@register(
    "slivers",
    "tessellation",
    "identity",
    "a fraction 0.3 * severity of edges is split at an off-centre point (half-length offset "
    "0.45 * severity toward a random endpoint, so 5% from the endpoint at severity 1) on both "
    "sides at once; the mesh stays conforming and watertight while sliver triangles appear",
)
def slivers(mesh, severity, rng):
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    edge_faces: dict = {}
    for t, f in zip(tris, fids, strict=True):
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault(_key(p, q), set()).add(f)
    before = len(tris)
    degenerate: set = set()
    for t in tris:
        if len(set(t)) < 3:
            degenerate.update(_key(p, q) for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    mids = {}
    for k in sorted(edge_faces):
        if k in degenerate:
            continue
        if rng.random() < 0.3 * severity:
            s = 1.0 if rng.random() < 0.5 else -1.0
            offset = 0.45 * severity
            x = None
            for _ in range(4):
                t = 0.5 + s * offset
                x = (1.0 - t) * np.array(k[0]) + t * np.array(k[1])
                if tuple(x.tolist()) not in k:
                    break
                offset /= 2
            if tuple(x.tolist()) in k:
                continue
            faces = [mesh.faces[f] for f in sorted(edge_faces[k])]
            if any(f.surface != "plane" for f in faces):
                x = project_onto(faces, x)
                if tuple(x.tolist()) in k:
                    continue
            mids[k] = tuple(x.tolist())
    edge_tris: dict = {}
    for t, _f in zip(tris, fids, strict=True):
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_tris.setdefault(_key(p, q), []).append(t)
    accepted = {}
    for k in sorted(mids):
        trial = dict(accepted)
        trial[k] = mids[k]
        ok = True
        for t in edge_tris[k]:
            tkeys = {_key(t[0], t[1]), _key(t[1], t[2]), _key(t[2], t[0])}
            local = {e: m for e, m in trial.items() if e in tkeys}
            pieces = _split_triangle(t, local, 0)
            n_old = np.cross(np.subtract(t[1], t[0]), np.subtract(t[2], t[0]))
            if float(np.linalg.norm(n_old)) <= 1e-14:
                ok = False
                break
            for p in pieces:
                n_new = np.cross(np.subtract(p[1], p[0]), np.subtract(p[2], p[0]))
                if float(np.linalg.norm(n_new)) <= 1e-14 or float(n_old @ n_new) <= 0.0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            accepted[k] = mids[k]
    mids = accepted
    new_tris, new_ids = [], []
    for t, f in zip(tris, fids, strict=True):
        pieces = _split_triangle(t, mids, f)
        new_tris += pieces
        new_ids += [f] * len(pieces)
    mesh.tris = np.array(new_tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(new_ids, dtype=mesh.face_id.dtype)
    for adj in mesh.adjacency:
        pts = [tuple(p) for p in adj.points]
        out = [pts[0]]
        for p, q in zip(pts, pts[1:], strict=False):
            out += _expand(p, q, mids)
        adj.points = [list(p) for p in out]
    return {
        "edge_fraction": 0.3 * severity,
        "offset_from_midpoint": 0.45 * severity,
        "edges_split": len(mids),
        "triangles_before": before,
        "triangles_after": len(new_tris),
    }


@register(
    "t_junctions",
    "tessellation",
    "identity",
    "up to 10% * severity of edges gain a hanging midpoint on one side only; the neighbour keeps "
    "its original edge, so the mesh is non-conforming and not watertight by design, while face "
    "ids and polylines stay aligned",
    preserves_watertight=False,
)
def t_junctions(mesh, severity, rng):
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    live = [True] * len(tris)
    edge_tris: dict = {}
    for ti, t in enumerate(tris):
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_tris.setdefault(_key(p, q), []).append(ti)
    keys = sorted(edge_tris)
    target = max(1, int(round(severity * len(keys) * 0.1)))
    order = rng.permutation(len(keys)).tolist()
    done = 0
    for oi in order:
        if done >= target:
            break
        k = keys[oi]
        a, b = k
        sides = [ti for ti in edge_tris[k] if live[ti]]
        if not sides:
            continue
        first = sides[int(rng.integers(len(sides)))]
        for ti in [first, *[s for s in sides if s != first]]:
            t = tris[ti]
            idx = next(
                (i for i in range(3) if {t[i], t[(i + 1) % 3]} == {a, b}),
                None,
            )
            if idx is None:
                continue
            x = (np.array(a) + np.array(b)) / 2
            if tuple(x.tolist()) in (a, b):
                continue
            face = mesh.faces[fids[ti]]
            if face.surface != "plane":
                x = project_onto([face], x)
                if tuple(x.tolist()) in (a, b):
                    continue
            p = tuple(x.tolist())
            r0, r1, r2 = t[idx], t[(idx + 1) % 3], t[(idx + 2) % 3]
            n_old = np.cross(np.subtract(r1, r0), np.subtract(r2, r0))
            if float(np.linalg.norm(n_old)) <= 1e-14:
                continue
            n1 = np.cross(np.subtract(p, r0), np.subtract(r2, r0))
            n2 = np.cross(np.subtract(r1, p), np.subtract(r2, p))
            if (
                float(np.linalg.norm(n1)) <= 1e-14
                or float(np.linalg.norm(n2)) <= 1e-14
                or float(n_old @ n1) <= 0.0
                or float(n_old @ n2) <= 0.0
            ):
                continue
            tris[ti] = (r0, p, r2)
            tris.append((p, r1, r2))
            fids.append(fids[ti])
            live.append(True)
            done += 1
            break
    kept = [i for i, v in enumerate(live) if v]
    mesh.tris = np.array([tris[i] for i in kept], dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array([fids[i] for i in kept], dtype=mesh.face_id.dtype)
    return {"target_splits": target, "splits": done, "triangles_after": len(kept)}
