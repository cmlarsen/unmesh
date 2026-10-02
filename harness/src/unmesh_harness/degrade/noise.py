from __future__ import annotations

import numpy as np

from .core import displace_vertices, register, vertex_table

MAX_AMPLITUDE_MM = 0.05


def amplitude_mm(severity: float) -> float:
    return severity * MAX_AMPLITUDE_MM


def _triangle_normals(tris: np.ndarray) -> np.ndarray:
    return np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def _accumulate(inverse: np.ndarray, per_tri: np.ndarray, n_vertices: int) -> np.ndarray:
    out = np.zeros((n_vertices, 3))
    for corner in range(3):
        np.add.at(out, inverse.reshape(-1, 3)[:, corner], per_tri)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


@register(
    "noise_isotropic",
    "noise",
    "identity",
    "every distinct vertex moved by an independent Gaussian vector, 50 um per axis",
)
def noise_isotropic(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    sigma = amplitude_mm(severity)
    displace_vertices(mesh, uniq, inverse, rng.standard_normal(uniq.shape) * sigma)
    return {"sigma_mm_per_axis": sigma, "distribution": "gaussian"}


@register(
    "noise_normal",
    "noise",
    "identity",
    "every distinct vertex moved along its area-weighted vertex normal by a Gaussian, 50 um sigma",
)
def noise_normal(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    sigma = amplitude_mm(severity)
    normals = _unit(_accumulate(inverse, _triangle_normals(mesh.tris), len(uniq)))
    displace_vertices(
        mesh, uniq, inverse, normals * (rng.standard_normal(len(uniq)) * sigma)[:, None]
    )
    return {"sigma_mm_along_normal": sigma, "distribution": "gaussian"}


@register(
    "noise_off_plane",
    "noise",
    "identity",
    "vertices whose every incident triangle lies on a planar face moved off-plane along the "
    "mean plane normal by a Gaussian, 50 um sigma; vertices touching a curved face stay put",
)
def noise_off_plane(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    sigma = amplitude_mm(severity)
    planar = np.array([f.surface == "plane" for f in mesh.faces], dtype=bool)
    tri_planar = planar[mesh.face_id]
    corners = inverse.reshape(-1, 3)
    touches_curved = np.zeros(len(uniq), dtype=bool)
    for corner in range(3):
        np.logical_or.at(touches_curved, corners[~tri_planar, corner], True)
    plane_normals = np.array(
        [f.params["normal"] if f.surface == "plane" else [0, 0, 0] for f in mesh.faces],
        dtype=np.float64,
    )
    per_tri = plane_normals[mesh.face_id]
    direction = _unit(_accumulate(inverse, per_tri, len(uniq)))
    scale = np.where(touches_curved, 0.0, rng.standard_normal(len(uniq)) * sigma)
    displace_vertices(mesh, uniq, inverse, direction * scale[:, None])
    return {
        "sigma_mm_off_plane": sigma,
        "distribution": "gaussian",
        "moved_vertices": int((~touches_curved).sum()),
    }
