from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from unmesh.ir import Ir, Tolerances

MeshLike = str | os.PathLike | np.ndarray | tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ConvertOptions:
    linear_tolerance: float | None = None
    angular_snap_deg: float = Tolerances.angular_snap_deg
    tangent_threshold_deg: float = Tolerances.tangent_threshold_deg
    vertex_merge: float = Tolerances.vertex_merge


@dataclass(frozen=True)
class Warning:
    code: str
    message: str


@dataclass
class Report:
    max_deviation: float
    rms_deviation: float
    analytic_area_fraction: float
    region_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[Warning] = field(default_factory=list)


class Result(NamedTuple):
    ir: Ir
    report: Report


def convert(mesh_or_path: MeshLike, options: ConvertOptions | None = None) -> Result:
    raise NotImplementedError("unmesh.convert is not implemented yet")
