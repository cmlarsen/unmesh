from __future__ import annotations

import numpy as np

from .core import compose_frame, map_faces, map_points, register


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def transform(p: np.ndarray, linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return p[:, 0:1] * linear[:, 0] + p[:, 1:2] * linear[:, 1] + p[:, 2:3] * linear[:, 2] + offset


def apply_rigid(mesh, linear: np.ndarray, offset: np.ndarray) -> None:
    zero = np.zeros(3)
    map_points(mesh, lambda p: transform(p, linear, offset))
    map_faces(mesh, lambda p: transform(p, linear, offset), lambda d: transform(d, linear, zero))
    compose_frame(mesh, linear, offset)
    if np.linalg.det(linear) < 0:
        mesh.tris = mesh.tris[:, [0, 2, 1]]
        for adj in mesh.adjacency:
            adj.forward_in_a = not adj.forward_in_a


@register(
    "rotation",
    "pose",
    "identity",
    "rotation by 180 degrees about a random unit axis (angle = severity * 180 degrees)",
    changes_frame=True,
)
def rotation(mesh, severity, rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = severity * np.pi
    apply_rigid(mesh, rotation_matrix(axis, angle), np.zeros(3))
    return {"axis": axis.tolist(), "angle_deg": float(np.degrees(angle))}


@register(
    "mirror",
    "pose",
    "identity",
    "reflection through a random plane through the bounding-box centre (the severity only "
    "switches the operator on); triangles re-wound so normals stay outward",
    changes_frame=True,
)
def mirror(mesh, severity, rng):
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    flat = mesh.tris.reshape(-1, 3)
    centre = (flat.min(axis=0) + flat.max(axis=0)) / 2
    linear = np.eye(3) - 2.0 * np.outer(normal, normal)
    apply_rigid(mesh, linear, centre - linear @ centre)
    return {"plane_normal": normal.tolist(), "plane_point": centre.tolist()}
