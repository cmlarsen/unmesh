from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from build123d import Shape
from OCP.BRepCheck import BRepCheck_Analyzer

Generator = Callable[[np.random.Generator], tuple[Shape, dict[str, Any], list[dict[str, Any]]]]


@dataclass
class GroundTruth:
    family: str
    seed: int
    solid: Shape
    parameters: dict[str, Any]
    features: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "parameters": self.parameters,
            "features": self.features,
            "fingerprint": fingerprint(self.solid),
        }


_REGISTRY: dict[str, Generator] = {}


def register(name: str) -> Callable[[Generator], Generator]:
    def deco(fn: Generator) -> Generator:
        if name in _REGISTRY:
            raise ValueError(f"family already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return deco


def families() -> list[str]:
    _load_builtin_families()
    return sorted(_REGISTRY)


def generate(family: str, seed: int) -> GroundTruth:
    _load_builtin_families()
    if family not in _REGISTRY:
        raise KeyError(f"unknown family: {family}")
    rng = np.random.default_rng(seed)
    solid, parameters, features = _REGISTRY[family](rng)
    return GroundTruth(family, seed, solid, parameters, features)


def _load_builtin_families() -> None:
    from . import planar  # noqa: F401


def fingerprint(shape: Shape) -> dict[str, Any]:
    bb = shape.bounding_box()
    return {
        "volume": float(shape.volume),
        "face_count": len(shape.faces()),
        "bbox_min": [bb.min.X, bb.min.Y, bb.min.Z],
        "bbox_max": [bb.max.X, bb.max.Y, bb.max.Z],
    }


def validity_problems(shape: Shape) -> list[str]:
    problems = []
    if len(shape.solids()) != 1:
        problems.append(f"expected 1 solid, found {len(shape.solids())}")
    if len(shape.shells()) != 1:
        problems.append(f"expected 1 shell, found {len(shape.shells())}")
    if not BRepCheck_Analyzer(shape.wrapped).IsValid():
        problems.append("BRepCheck_Analyzer reports invalid")
    if not shape.volume > 0:
        problems.append(f"non-positive volume {shape.volume}")
    return problems


def rect_extent(length: float, width: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return length * c + width * s, length * s + width * c
