from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..labels import LabeledMesh

Params = dict[str, Any]
OperatorFn = Callable[[LabeledMesh, float, np.random.Generator], Params]


@dataclass(frozen=True)
class Operator:
    name: str
    family: str
    fn: OperatorFn
    severity_0: str
    severity_1: str
    preserves_watertight: bool
    changes_frame: bool = False
    may_collapse: bool = False
    binary: bool = False


OPERATORS: dict[str, Operator] = {}


def register(
    name: str,
    family: str,
    severity_0: str,
    severity_1: str,
    preserves_watertight: bool = True,
    changes_frame: bool = False,
    may_collapse: bool = False,
    binary: bool = False,
):
    def wrap(fn: OperatorFn) -> OperatorFn:
        if name in OPERATORS:
            raise ValueError(f"degradation {name!r} registered twice")
        OPERATORS[name] = Operator(
            name,
            family,
            fn,
            severity_0,
            severity_1,
            preserves_watertight,
            changes_frame,
            may_collapse,
            binary,
        )
        return fn

    return wrap


def apply(name: str, mesh: LabeledMesh, severity: float, seed: int) -> LabeledMesh:
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    op = OPERATORS[name]
    out = copy.deepcopy(mesh)
    out.metadata.setdefault(
        "truth",
        {
            "vertices": copy.deepcopy(out.vertices),
            "edge_points": [copy.deepcopy(a.points) for a in out.adjacency],
        },
    )
    params: Params = {}
    if severity > 0.0:
        params = op.fn(out, float(severity), np.random.default_rng(seed)) or {}
    out.metadata.setdefault("history", []).append(
        {"op": name, "severity": float(severity), "seed": int(seed), "params": params}
    )
    return out


def chain(mesh: LabeledMesh, steps: list[tuple[str, float]], seed: int) -> LabeledMesh:
    """Apply operators in order, deriving each step's seed from the chain seed.

    Step ``i`` runs as ``apply(name, mesh, severity, seed + i)`` and appends its
    own ``(op, severity, seed, params)`` entry to ``metadata["history"]``, so a
    chain is fully described by its step list plus the single chain seed.
    Per-step seeds overlap across chain seeds: step 1 at chain seed ``s`` uses
    the same seed as step 0 at chain seed ``s + 1``.
    """
    for i, (name, severity) in enumerate(steps):
        mesh = apply(name, mesh, severity, seed + i)
    return mesh


def apply_chain(mesh: LabeledMesh, steps: list[tuple[str, float]], seed: int) -> LabeledMesh:
    return chain(mesh, steps, seed)


def to_original(mesh: LabeledMesh) -> np.ndarray:
    return np.array(mesh.metadata.get("to_original", np.eye(4).tolist()), dtype=np.float64)


def to_original_points(mesh: LabeledMesh, points: np.ndarray) -> np.ndarray:
    m = to_original(mesh)
    p = np.asarray(points, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def compose_frame(mesh: LabeledMesh, linear: np.ndarray, offset: np.ndarray) -> None:
    fwd = np.eye(4)
    fwd[:3, :3] = linear
    fwd[:3, 3] = offset
    mesh.metadata["to_original"] = (to_original(mesh) @ np.linalg.inv(fwd)).tolist()


def map_points(mesh: LabeledMesh, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    mesh.tris = fn(mesh.tris.reshape(-1, 3)).reshape(mesh.tris.shape)
    if mesh.vertices:
        mesh.vertices = fn(np.array(mesh.vertices, dtype=np.float64)).tolist()
    for adj in mesh.adjacency:
        adj.points = fn(np.array(adj.points, dtype=np.float64)).tolist()


_POINT_KEYS = ("origin", "apex", "center")
_DIRECTION_KEYS = ("normal", "axis")


def map_faces(
    mesh: LabeledMesh,
    point_fn: Callable[[np.ndarray], np.ndarray],
    direction_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    for face in mesh.faces:
        prm = face.params
        for key in _POINT_KEYS:
            if key in prm:
                prm[key] = point_fn(np.array([prm[key]], dtype=np.float64))[0].tolist()
        for key in _DIRECTION_KEYS:
            if key in prm:
                prm[key] = direction_fn(np.array([prm[key]], dtype=np.float64))[0].tolist()


def vertex_table(mesh: LabeledMesh) -> tuple[np.ndarray, np.ndarray]:
    uniq, inverse = np.unique(mesh.tris.reshape(-1, 3), axis=0, return_inverse=True)
    return uniq, inverse.reshape(-1)


def displace_vertices(
    mesh: LabeledMesh, uniq: np.ndarray, inverse: np.ndarray, disp: np.ndarray
) -> float:
    moved = uniq + disp
    mesh.tris = moved[inverse].reshape(mesh.tris.shape)
    lookup = {tuple(p): i for i, p in enumerate(uniq.tolist())}

    def remap(pts) -> list[list[float]]:
        return [moved[lookup[tuple(p)]].tolist() for p in pts]

    if mesh.vertices:
        mesh.vertices = remap(mesh.vertices)
    for adj in mesh.adjacency:
        adj.points = remap(adj.points)
    return float(np.linalg.norm(disp, axis=1).max()) if len(disp) else 0.0
