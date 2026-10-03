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


def step_problems(
    path: str | Path | None,
    truth_volume: float,
    input_tris: np.ndarray | None = None,
    bound: float | None = None,
    samples_per_mm2: float = 1.0,
) -> tuple[list[str], int | None]:
    if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return ["no STEP file written"], None
    try:
        return _step_problems(path, truth_volume, input_tris, bound, samples_per_mm2)
    except Exception as e:
        return [f"STEP check failed: {type(e).__name__}: {e}"], None


def _step_problems(path, truth_volume, input_tris, bound, samples_per_mm2):
    from build123d import import_step

    from ..judge import judge
    from .converters import faceted_ir

    shape = import_step(str(path))
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
