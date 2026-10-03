from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from unmesh.ir import Ir

from ..groundtruth import validity_problems
from ..labels import LabeledMesh
from ..metrics.recovery import score_recovery

VOLUME_REL_TOL = 0.01
STEP_TESSELLATION_MM = 0.005
STEP_TESSELLATION_ANGLE = 0.1


def face_recovery(
    clean: LabeledMesh, face_id: np.ndarray, ir: Ir, to_original: np.ndarray
) -> dict[str, Any]:
    result = score_recovery(clean, face_id, ir, to_original)
    result.pop("faces_detail", None)
    edge = result.get("edge_error")
    if isinstance(edge, dict):
        edge.pop("details", None)
    return result


def load_step_shape(path):
    from build123d import import_step

    if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return None, "no STEP file written"
    try:
        return import_step(str(path)), None
    except Exception as e:
        return None, f"STEP check failed: {type(e).__name__}: {e}"


def step_problems(
    path: str | Path | None,
    truth_volume: float,
    input_tris: np.ndarray | None = None,
    bound: float | None = None,
    samples_per_mm2: float = 1.0,
    *,
    shape: Any = None,
    load_error: str | None = None,
) -> tuple[list[str], int | None]:
    if shape is None and load_error is None:
        shape, load_error = load_step_shape(path)
    if load_error is not None:
        return [load_error], None
    try:
        return _step_problems(shape, truth_volume, input_tris, bound, samples_per_mm2)
    except Exception as e:
        return [f"STEP check failed: {type(e).__name__}: {e}"], None


def _step_problems(shape, truth_volume, input_tris, bound, samples_per_mm2):
    from ..judge import judge
    from .converters import faceted_ir

    faces = len(shape.faces())
    problems = validity_problems(shape)
    if not math.isfinite(shape.volume) or abs(shape.volume - truth_volume) > (
        VOLUME_REL_TOL * truth_volume
    ):
        problems.append(f"volume {shape.volume:.4f} differs from truth {truth_volume:.4f}")
    if input_tris is not None and bound is not None and not problems:
        verts, tri_idx = shape.tessellate(STEP_TESSELLATION_MM, STEP_TESSELLATION_ANGLE)
        v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=np.float64)
        tris = v[np.array(tri_idx, dtype=np.int64)]
        result = judge(faceted_ir(tris), tris, input_tris, None, samples_per_mm2=samples_per_mm2)
        if not result.truth.max <= bound:
            problems.append(
                f"STEP deviates {result.truth.max * 1000:.1f} um from the input, "
                f"bound {bound * 1000:.1f} um"
            )
    return problems, faces
