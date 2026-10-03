from __future__ import annotations

from pathlib import Path
from typing import Any

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance


def _default_tolerance_bound() -> float:
    from unmesh.step import WriteOptions

    return float(WriteOptions().max_shape_tolerance)


def score_validity(
    step_path: str | Path | None,
    *,
    write: dict[str, Any] | None = None,
    expected_solids: int = 1,
    expected_shells: int = 1,
    tolerance_bound: float | None = None,
) -> dict[str, Any]:
    from ..groundtruth import validity_problems

    if tolerance_bound is None:
        tolerance_bound = _default_tolerance_bound()
    write = write or {}
    fallback = write.get("fallback") is not None
    missing: dict[str, Any] = {
        "solids": None,
        "shells": None,
        "volume": None,
        "volume_positive": None,
        "brepcheck_valid": None,
        "max_shape_tolerance": None,
        "tolerance_ok": None,
        "expected_solids": int(expected_solids),
        "expected_shells": int(expected_shells),
        "tolerance_bound": float(tolerance_bound),
        "fallback": bool(fallback),
        "problems": [],
        "valid": False,
    }
    if step_path is None or not Path(step_path).is_file() or Path(step_path).stat().st_size == 0:
        missing["problems"] = ["no STEP file written"]
        return missing
    try:
        from build123d import import_step

        shape = import_step(str(step_path))
    except Exception as e:
        missing["problems"] = [f"STEP check failed: {type(e).__name__}: {e}"]
        return missing
    problems = validity_problems(shape, expected_solids, expected_shells)
    brepcheck_valid = bool(BRepCheck_Analyzer(shape.wrapped).IsValid())
    volume = float(shape.volume)
    volume_positive = bool(volume > 0)
    max_tolerance = float(ShapeAnalysis_ShapeTolerance().Tolerance(shape.wrapped, 1))
    tolerance_ok = bool(max_tolerance <= tolerance_bound)
    if not tolerance_ok:
        problems.append(f"shape tolerance {max_tolerance:.3g} exceeds {tolerance_bound:.3g}")
    return {
        "solids": len(shape.solids()),
        "shells": len(shape.shells()),
        "volume": volume,
        "volume_positive": volume_positive,
        "brepcheck_valid": brepcheck_valid,
        "max_shape_tolerance": max_tolerance,
        "tolerance_ok": tolerance_ok,
        "expected_solids": int(expected_solids),
        "expected_shells": int(expected_shells),
        "tolerance_bound": float(tolerance_bound),
        "fallback": bool(fallback),
        "problems": problems,
        "valid": not problems,
    }
