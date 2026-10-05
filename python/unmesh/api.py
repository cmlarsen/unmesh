from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from unmesh import _core
from unmesh.ir import Ir, Tolerances

MeshLike = str | os.PathLike | np.ndarray | tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ConvertOptions:
    linear_tolerance: float | None = None
    angular_snap_deg: float = Tolerances.angular_snap_deg
    tangent_threshold_deg: float = Tolerances.tangent_threshold_deg
    vertex_merge: float = Tolerances.vertex_merge


@dataclass(frozen=True)
class ConvertWarning:
    code: str
    message: str


@dataclass
class Report:
    max_deviation: float
    rms_deviation: float
    analytic_area_fraction: float
    region_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[ConvertWarning] = field(default_factory=list)


class Result(NamedTuple):
    ir: Ir
    report: Report


def _report(raw: dict) -> Report:
    return Report(
        max_deviation=raw["max_deviation"],
        rms_deviation=raw["rms_deviation"],
        analytic_area_fraction=raw["analytic_area_fraction"],
        region_counts=dict(raw["region_counts"]),
        warnings=[ConvertWarning(code, message) for code, message in raw["warnings"]],
    )


def _args(options: ConvertOptions | None) -> tuple:
    opts = options or ConvertOptions()
    return (
        opts.linear_tolerance,
        opts.angular_snap_deg,
        opts.tangent_threshold_deg,
        opts.vertex_merge,
    )


def _soup(mesh_or_path: MeshLike) -> np.ndarray:
    if isinstance(mesh_or_path, tuple):
        vertices, faces = mesh_or_path
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"expected vertices of shape (v, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"expected faces of shape (f, 3), got {faces.shape}")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face index out of range")
        return vertices[faces]
    if isinstance(mesh_or_path, str | os.PathLike):
        return _core.read_stl(mesh_or_path)
    return mesh_or_path


def convert(mesh_or_path: MeshLike, options: ConvertOptions | None = None) -> Result:
    args = _args(options)
    if isinstance(mesh_or_path, tuple):
        vertices, faces = mesh_or_path
        text, raw = _core.convert_indexed(vertices, faces, *args)
    else:
        if isinstance(mesh_or_path, str | os.PathLike):
            tris = _core.read_stl(mesh_or_path)
        else:
            tris = mesh_or_path
        text, raw = _core.convert_soup(tris, *args)
    return Result(Ir.loads(text), _report(raw))


def convert_from_labels(
    mesh_or_path: MeshLike, labels, options: ConvertOptions | None = None
) -> Result:
    text, raw = _core.convert_soup_from_labels(_soup(mesh_or_path), labels, *_args(options))
    return Result(Ir.loads(text), _report(raw))
