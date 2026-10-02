from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unmesh import _core
from unmesh.ir import Ir


@dataclass(frozen=True)
class Stats:
    count: int
    max: float
    p99: float
    p95: float
    mean: float


@dataclass(frozen=True)
class Comparison:
    ir_to_mesh: Stats
    mesh_to_ir: Stats

    @property
    def max(self) -> float:
        return max(self.ir_to_mesh.max, self.mesh_to_ir.max)


@dataclass(frozen=True)
class JudgeResult:
    input: Comparison
    truth: Comparison | None
    region_max: np.ndarray
    calibration: float | None

    @property
    def max_deviation(self) -> float:
        return self.input.max


def _tris(mesh: Any) -> np.ndarray:
    arr = np.asarray(getattr(mesh, "tris", mesh), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"expected triangles of shape (n, 3, 3), got {arr.shape}")
    return arr


def _comparison(raw: dict[str, Any]) -> Comparison:
    return Comparison(Stats(**raw["ir_to_mesh"]), Stats(**raw["mesh_to_ir"]))


def _reported(report: Any) -> float | None:
    if report is None:
        return None
    if isinstance(report, int | float):
        return float(report)
    if isinstance(report, dict):
        return float(report["max_deviation"])
    return float(report.max_deviation)


def calibration(reported_max_deviation: float, input: Comparison) -> float:
    return reported_max_deviation - input.max


def judge(
    ir: Ir,
    input_mesh: Any,
    truth_mesh: Any = None,
    report: Any = None,
    samples_per_mm2: float = 10.0,
    seed: int = 0,
    include_vertices: bool = True,
) -> JudgeResult:
    raw = _core.judge_ir(
        ir.dumps(),
        _tris(input_mesh),
        None if truth_mesh is None else _tris(truth_mesh),
        samples_per_mm2,
        seed,
        include_vertices,
    )
    input_cmp = _comparison(raw["input"])
    reported = _reported(report)
    return JudgeResult(
        input=input_cmp,
        truth=None if raw["truth"] is None else _comparison(raw["truth"]),
        region_max=raw["region_max"],
        calibration=None if reported is None else calibration(reported, input_cmp),
    )


def sample_ir(
    ir: Ir,
    input_mesh: Any,
    samples_per_mm2: float = 10.0,
    seed: int = 0,
    include_vertices: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    return _core.sample_ir(ir.dumps(), _tris(input_mesh), samples_per_mm2, seed, include_vertices)
