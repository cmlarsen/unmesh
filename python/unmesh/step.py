from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from unmesh.ir import Ir


@dataclass(frozen=True)
class WriteOptions:
    max_shape_tolerance: float = 1e-3


@dataclass
class FaceReport:
    region: int
    surface_type: str
    max_shape_tolerance: float


@dataclass
class WriteReport:
    valid: bool
    solids: int
    max_shape_tolerance: float
    faces: list[FaceReport] = field(default_factory=list)
    fallback: Literal["faceted"] | None = None
    fallback_reason: str | None = None


def write(ir: Ir, path: str | os.PathLike, options: WriteOptions | None = None) -> WriteReport:
    raise NotImplementedError("unmesh.step.write is not implemented yet")
