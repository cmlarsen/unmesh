from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from build123d import Shape
from OCP.BRepCheck import BRepCheck_Analyzer

Generator = Callable[[np.random.Generator], tuple]


@dataclass
class GroundTruth:
    family: str
    seed: int
    solid: Shape
    parameters: dict[str, Any]
    features: list[dict[str, Any]] = field(default_factory=list)
    face_tags: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        meta = {
            "family": self.family,
            "seed": self.seed,
            "parameters": self.parameters,
            "features": self.features,
            "fingerprint": fingerprint(self.solid),
        }
        if self.face_tags is not None:
            meta["face_tags"] = self.face_tags
        return meta


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
    result = _REGISTRY[family](rng)
    solid, parameters, features = result[0], result[1], result[2]
    tags = result[3] if len(result) > 3 else None
    parameters = dict(parameters)
    if "pair_family" in parameters:
        parameters["pair_id"] = f"{parameters['pair_family']}-{seed:04d}"
    return GroundTruth(family, seed, solid, parameters, features, face_tags=tags)


def _load_builtin_families() -> None:
    from . import ambiguity, chamfer_fillet, complex, curved, planar  # noqa: F401


def fingerprint(shape: Shape) -> dict[str, Any]:
    bb = shape.bounding_box()
    return {
        "volume": float(shape.volume),
        "face_count": len(shape.faces()),
        "bbox_min": [bb.min.X, bb.min.Y, bb.min.Z],
        "bbox_max": [bb.max.X, bb.max.Y, bb.max.Z],
    }


def validity_problems(shape: Shape, solids: int = 1, shells: int = 1) -> list[str]:
    problems = []
    if len(shape.solids()) != solids:
        problems.append(f"expected {solids} solid(s), found {len(shape.solids())}")
    if len(shape.shells()) != shells:
        problems.append(f"expected {shells} shell(s), found {len(shape.shells())}")
    if not BRepCheck_Analyzer(shape.wrapped).IsValid():
        problems.append("BRepCheck_Analyzer reports invalid")
    if not shape.volume > 0:
        problems.append(f"non-positive volume {shape.volume}")
    return problems


def rect_extent(length: float, width: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return length * c + width * s, length * s + width * c
