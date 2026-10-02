from __future__ import annotations

import math

import numpy as np

from ..labels import FaceInfo
from .core import register

MAX_EDGE_AT_ZERO_MM = 10.0
MAX_PASSES = 40
_EPS = 1e-6


def target_edge_mm(severity: float) -> float:
    return MAX_EDGE_AT_ZERO_MM * 0.1**severity


def implicit(face: FaceInfo, p: np.ndarray) -> np.ndarray:
    prm = face.params
    if face.surface == "plane":
        return (p - np.array(prm["origin"])) @ np.array(prm["normal"])
    if face.surface == "sphere":
        return np.linalg.norm(p - np.array(prm["center"]), axis=1) - prm["radius"]
    origin = np.array(prm.get("origin", prm.get("apex", prm.get("center"))))
    axis = np.array(prm["axis"])
    v = p - origin
    h = v @ axis
    rho = np.linalg.norm(v - h[:, None] * axis, axis=1)
    if face.surface == "cylinder":
        return rho - prm["radius"]
    if face.surface == "torus":
        return np.hypot(rho - prm["major_radius"], h) - prm["minor_radius"]
    a = prm["half_angle"]
    return rho * math.cos(a) - np.abs(h) * math.sin(a)


def _gradient(face: FaceInfo, x: np.ndarray) -> np.ndarray:
    steps = np.eye(3) * _EPS
    plus = implicit(face, x + steps)
    minus = implicit(face, x - steps)
    return (plus - minus) / (2 * _EPS)


def project_onto(faces: list[FaceInfo], x: np.ndarray) -> np.ndarray:
    for _ in range(60):
        f = np.array([implicit(face, x[None])[0] for face in faces])
        if np.abs(f).max() < 1e-13:
            break
        jac = np.array([_gradient(face, x) for face in faces])
        x = x - np.linalg.lstsq(jac, f, rcond=None)[0]
    return x


def _key(p, q):
    return (p, q) if p <= q else (q, p)


def _split_triangle(tri, mids, tri_faces):
    a, b, c = tri
    flags = [_key(a, b) in mids, _key(b, c) in mids, _key(c, a) in mids]
    n = sum(flags)
    if n == 0:
        return [tri]
    if n == 3:
        mab, mbc, mca = mids[_key(a, b)], mids[_key(b, c)], mids[_key(c, a)]
        return [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
    for _ in range(3):
        if n == 1 and flags[0]:
            m = mids[_key(a, b)]
            return [(a, m, c), (m, b, c)]
        if n == 2 and not flags[2]:
            m1, m2 = mids[_key(a, b)], mids[_key(b, c)]
            first = [(m1, b, m2)]
            if math.dist(m1, c) <= math.dist(a, m2):
                return first + [(a, m1, c), (m1, m2, c)]
            return first + [(a, m1, m2), (a, m2, c)]
        a, b, c = b, c, a
        flags = flags[1:] + flags[:1]
    raise AssertionError("unreachable")


def _expand(p, q, mids):
    m = mids.get(_key(p, q))
    if m is None:
        return [q]
    return _expand(p, m, mids) + _expand(m, q, mids)


@register(
    "refine",
    "tessellation",
    "identity",
    "edges bisected until none exceeds 1 mm (10 mm at severity 0+, 3.2 mm at 0.5); "
    "midpoints on curved faces are projected onto the analytic surface",
    binary=False,
)
def refine(mesh, severity, rng):
    target = target_edge_mm(severity)
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    all_mids: dict = {}
    before = len(tris)
    for _ in range(MAX_PASSES):
        edge_faces: dict = {}
        long_edges = set()
        for t, f in zip(tris, fids, strict=True):
            for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                k = _key(p, q)
                edge_faces.setdefault(k, set()).add(f)
                if math.dist(p, q) > target:
                    long_edges.add(k)
        if not long_edges:
            break
        mids = {}
        for k in sorted(long_edges):
            p, q = k
            x = (np.array(p) + np.array(q)) / 2
            faces = [mesh.faces[f] for f in sorted(edge_faces[k])]
            if any(f.surface != "plane" for f in faces):
                x = project_onto(faces, x)
            mids[k] = tuple(x.tolist())
        new_tris, new_ids = [], []
        for t, f in zip(tris, fids, strict=True):
            pieces = _split_triangle(t, mids, f)
            new_tris += pieces
            new_ids += [f] * len(pieces)
        tris, fids = new_tris, new_ids
        all_mids.update(mids)
    else:
        raise RuntimeError("refine did not converge")
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    mesh.face_id = np.array(fids, dtype=mesh.face_id.dtype)
    for adj in mesh.adjacency:
        pts = [tuple(p) for p in adj.points]
        out = [pts[0]]
        for p, q in zip(pts, pts[1:], strict=False):
            out += _expand(p, q, all_mids)
        adj.points = [list(p) for p in out]
    return {
        "target_max_edge_mm": target,
        "triangles_before": before,
        "triangles_after": len(tris),
    }
