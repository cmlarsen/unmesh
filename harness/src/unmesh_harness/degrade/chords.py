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
        scale = (
            np.linalg.norm(np.array(pts[k + 1]) - np.array(point))
            + np.linalg.norm(np.array(point) - np.array(pts[k - 1]))
        ) / 2
        return t / n, scale
    return None


@register(
    "nonuniform_chords",
    "tessellation",
    "identity",
    "vertices on one or two faces moved along the surface (within a face, or along the shared "
    "edge for two-face vertices) by up to severity * 45% of their local mean edge length, then "
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
    disp = np.zeros_like(uniq)
    moved = 0
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
            scale = np.min(
                np.linalg.norm(uniq[corners[np.any(corners == i, axis=1)]] - uniq[i], axis=1)
            )
        planned = (rng.random() * 2.0 - 1.0) * 0.45 * scale * severity
        x = project_onto([mesh.faces[f] for f in sorted(faces)], uniq[i] + direction * planned)
        if np.all(np.isfinite(x)) and np.linalg.norm(x - uniq[i]) <= 3 * abs(planned) + 1e-12:
            disp[i] = x - uniq[i]
            moved += 1
    realised = displace_vertices(mesh, uniq, inverse, disp)
    return {
        "fraction_of_local_edge": 0.45 * severity,
        "distribution": "uniform_interval_along_surface",
        "max_displacement_mm": realised,
        "moved_vertices": moved,
        "moved_fraction": moved / len(uniq) if len(uniq) else 0.0,
    }
