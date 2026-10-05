from __future__ import annotations

import numpy as np

from .core import displace_vertices, register, vertex_table

MAX_AMPLITUDE_MM = 0.05


def amplitude_mm(severity: float) -> float:
    return severity * MAX_AMPLITUDE_MM


def _accumulate(inverse: np.ndarray, per_tri: np.ndarray, n_vertices: int) -> np.ndarray:
    out = np.zeros((n_vertices, 3))
    for corner in range(3):
        np.add.at(out, inverse.reshape(-1, 3)[:, corner], per_tri)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    direction = _unit(rng.standard_normal((n, 3)))
    return direction * (radius * rng.random(n) ** (1.0 / 3.0))[:, None]


def _interval(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    return (rng.random(n) * 2.0 - 1.0) * radius


@register(
    "noise_isotropic",
    "noise",
    "identity",
    "every distinct vertex moved by an independent vector drawn uniformly from a ball of radius "
    "A = severity * 50 um (50 um at severity 1)",
)
def noise_isotropic(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    amp = amplitude_mm(severity)
    realised = displace_vertices(mesh, uniq, inverse, _ball(rng, len(uniq), amp))
    return {"amplitude_mm": amp, "distribution": "uniform_ball", "max_displacement_mm": realised}


@register(
    "noise_normal",
    "noise",
    "identity",
    "every distinct vertex moved along its area-weighted vertex normal by a scalar drawn "
    "uniformly from [-A, A], A = severity * 50 um",
)
def noise_normal(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    amp = amplitude_mm(severity)
    cross = np.cross(mesh.tris[:, 1] - mesh.tris[:, 0], mesh.tris[:, 2] - mesh.tris[:, 0])
    normals = _unit(_accumulate(inverse, cross, len(uniq)))
    disp = normals * _interval(rng, len(uniq), amp)[:, None]
    realised = displace_vertices(mesh, uniq, inverse, disp)
    return {
        "amplitude_mm": amp,
        "distribution": "uniform_interval",
        "max_displacement_mm": realised,
    }


def _interior_planar(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq, inverse = vertex_table(mesh)
    corners = inverse.reshape(-1, 3)
    lo = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(len(uniq), -1, dtype=np.int64)
    ids = mesh.face_id.astype(np.int64)
    for corner in range(3):
        np.minimum.at(lo, corners[:, corner], ids)
        np.maximum.at(hi, corners[:, corner], ids)
    planar = np.array([f.surface == "plane" for f in mesh.faces], dtype=bool)
    interior = (lo == hi) & planar[np.clip(lo, 0, len(planar) - 1)] if planar.any() else np.zeros(
        len(uniq), dtype=bool
    )
    return interior, lo, planar


@register(
    "noise_off_plane",
    "noise",
    "identity",
    "vertices interior to a single planar face (every incident triangle on that face) moved "
    "along its normal by a scalar drawn uniformly from [-A, A], A = severity * 50 um; refine "
    "first, since planar faces have no interior vertices otherwise",
    applies_to=lambda mesh: bool(_interior_planar(mesh)[0].any()),
)
def noise_off_plane(mesh, severity, rng):
    uniq, inverse = vertex_table(mesh)
    amp = amplitude_mm(severity)
    interior, lo, planar = _interior_planar(mesh)
    if not planar.any():
        return {"skipped": "no planar faces"}
    if not interior.any():
        raise ValueError(
            "noise_off_plane has no vertex interior to a planar face; apply refine first"
        )
    normals = np.array(
        [f.params["normal"] if f.surface == "plane" else [0.0, 0.0, 0.0] for f in mesh.faces],
        dtype=np.float64,
    )
    direction = normals[np.clip(lo, 0, len(planar) - 1)] * interior[:, None]
    disp = direction * _interval(rng, len(uniq), amp)[:, None]
    realised = displace_vertices(mesh, uniq, inverse, disp)
    return {
        "amplitude_mm": amp,
        "distribution": "uniform_interval",
        "max_displacement_mm": realised,
        "moved_vertices": int(interior.sum()),
        "moved_fraction": float(interior.mean()),
    }
