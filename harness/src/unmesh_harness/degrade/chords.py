from __future__ import annotations

import numpy as np

from ..labels import outward_normals
from .core import displace_vertices, register, vertex_table
from .refine import project_onto

CURVED = ("cylinder", "cone", "sphere", "torus")


def _incident_faces(mesh, uniq_index: int, corners: np.ndarray) -> set[int]:
    return set(mesh.face_id[np.any(corners == uniq_index, axis=1)].tolist())


def _edge_tangent(mesh, faces: set[int], point: tuple) -> tuple[np.ndarray, float] | None:
    for adj in mesh.adjacency:
        if {adj.face_a, adj.face_b} != faces:
            continue
        pts = [tuple(p) for p in adj.points]
        if point not in pts:
            continue
        k = pts.index(point)
        if k == 0 or k == len(pts) - 1:
            return None
        t = np.array(pts[k + 1]) - np.array(pts[k - 1])
        n = np.linalg.norm(t)
        if n <= 0:
            return None
        scale = min(
            float(np.linalg.norm(np.array(pts[k + 1]) - np.array(point))),
            float(np.linalg.norm(np.array(point) - np.array(pts[k - 1]))),
        )
        return t / n, scale
    return None


@register(
    "nonuniform_chords",
    "tessellation",
    "identity",
    "vertices on one or two faces moved along the surface (within a face, or along the shared "
    "edge for two-face vertices) by up to severity * 45% of the local chord length, then "
    "reprojected onto the incident analytic surfaces; chord spacing becomes non-uniform while "
    "every node stays on-surface; CAD corners and 3+ face vertices stay fixed",
)
def nonuniform_chords(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    corners = inverse.reshape(-1, 3)
    ids = mesh.face_id.astype(np.int64)
    lo = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(len(uniq), -1, dtype=np.int64)
    for corner in range(3):
        np.minimum.at(lo, corners[:, corner], ids)
        np.maximum.at(hi, corners[:, corner], ids)
    fixed = {tuple(p) for p in mesh.vertices}
    planned_disp = np.zeros_like(uniq)
    for i in range(len(uniq)):
        faces = _incident_faces(mesh, i, corners)
        if len(faces) > 2 or tuple(uniq[i].tolist()) in fixed:
            continue
        if len(faces) == 2:
            found = _edge_tangent(mesh, faces, tuple(uniq[i].tolist()))
            if found is None:
                continue
            direction, scale = found
        else:
            face = mesh.faces[int(lo[i])]
            if face.surface not in (*CURVED, "plane"):
                continue
            n = outward_normals(face, uniq[i][None])[0]
            if not np.all(np.isfinite(n)):
                continue
            direction = None
            for _ in range(8):
                r = rng.standard_normal(3)
                t = r - (r @ n) * n
                if np.linalg.norm(t) > 1e-6:
                    direction = t / np.linalg.norm(t)
                    break
            if direction is None:
                continue
            neighbours = corners[np.any(corners == i, axis=1)].ravel()
            neighbours = neighbours[neighbours != i]
            if not len(neighbours):
                continue
            scale = float(np.linalg.norm(uniq[neighbours] - uniq[i], axis=-1).min())
        planned = (rng.random() * 2.0 - 1.0) * 0.45 * scale * severity
        x = project_onto([mesh.faces[f] for f in sorted(faces)], uniq[i] + direction * planned)
        if np.all(np.isfinite(x)) and np.linalg.norm(x - uniq[i]) <= 3 * abs(planned) + 1e-12:
            planned_disp[i] = x - uniq[i]
    pos = uniq.copy()
    for i in range(len(uniq)):
        if not np.any(planned_disp[i]):
            continue
        new_p = uniq[i] + planned_disp[i]
        ok = True
        for ti in np.where(np.any(corners == i, axis=1))[0]:
            tri = pos[corners[ti]]
            k = list(corners[ti]).index(i)
            old_n = np.cross(tri[(k + 1) % 3] - tri[k], tri[(k + 2) % 3] - tri[k])
            if float(np.linalg.norm(old_n)) <= 1e-14:
                continue
            new_tri = tri.copy()
            new_tri[k] = new_p
            new_n = np.cross(new_tri[(k + 1) % 3] - new_tri[k], new_tri[(k + 2) % 3] - new_tri[k])
            if float(np.linalg.norm(new_n)) <= 1e-14 or float(old_n @ new_n) <= 0.0:
                ok = False
                break
        if ok:
            pos[i] = new_p
    flat_tris = corners.reshape(-1, 3)
    old_area = np.linalg.norm(
        np.cross(
            uniq[flat_tris[:, 1]] - uniq[flat_tris[:, 0]],
            uniq[flat_tris[:, 2]] - uniq[flat_tris[:, 0]],
        ),
        axis=-1,
    )
    while True:
        cur = pos[flat_tris]
        new_n = np.cross(cur[:, 1] - cur[:, 0], cur[:, 2] - cur[:, 0])
        new_area = np.linalg.norm(new_n, axis=-1)
        bad = (new_area <= 1e-14) & (old_area > 1e-14)
        for face in mesh.faces:
            if face.surface == "other":
                continue
            sel = np.where((ids == face.id) & ~bad & (old_area > 1e-14))[0]
            if not len(sel):
                continue
            try:
                on = outward_normals(face, cur[sel].mean(axis=1))
            except Exception:
                continue
            d = np.einsum("ij,ij->i", new_n[sel], on)
            bad[sel[np.isfinite(d) & (d <= 0.0)]] = True
        if not bad.any():
            break
        touched = set(flat_tris[bad].ravel().tolist())
        if not any(np.any(pos[v] != uniq[v]) for v in touched):
            break
        for v in touched:
            pos[v] = uniq[v]
    disp = pos - uniq
    moved = int((np.linalg.norm(disp, axis=-1) > 1e-12).sum())
    realised = displace_vertices(mesh, uniq, inverse, disp)
    return {
        "fraction_of_local_edge": 0.45 * severity,
        "distribution": "uniform_interval_along_surface",
        "max_displacement_mm": realised,
        "moved_vertices": moved,
        "moved_fraction": moved / len(uniq) if len(uniq) else 0.0,
    }
