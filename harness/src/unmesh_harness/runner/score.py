from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from unmesh.ir import Ir
from unmesh.step import WriteOptions

from ..groundtruth import validity_problems
from ..labels import LabeledMesh
from ..metrics.recovery import score_recovery

VOLUME_REL_TOL = 0.01
STEP_TESSELLATION_MM = 0.002
STEP_TESSELLATION_ANGLE = 0.1
STEP_TOLERANCE_MM = STEP_TESSELLATION_MM + WriteOptions().max_shape_tolerance


def face_recovery(
    clean: LabeledMesh,
    face_id: np.ndarray,
    ir: Ir,
    to_original: np.ndarray,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    result = score_recovery(clean, face_id, ir, to_original, confidence=confidence)
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


def step_triangles(shape: Any, linear: float, angular: float) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    wrapped = BRepBuilderAPI_Copy(getattr(shape, "wrapped", shape)).Shape()
    BRepMesh_IncrementalMesh(wrapped, linear, False, angular, True)
    blocks = []
    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            raise RuntimeError("STEP face has no triangulation")
        trsf = loc.Transformation()
        nodes = np.array(
            [
                (q.X(), q.Y(), q.Z())
                for q in (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))
            ],
            dtype=np.float64,
        )
        idx = np.array(
            [
                [tri.Triangle(j).Value(k) - 1 for k in (1, 2, 3)]
                for j in range(1, tri.NbTriangles() + 1)
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        if face.Orientation() == TopAbs_REVERSED:
            idx = idx[:, [0, 2, 1]]
        blocks.append(nodes[idx])
    return np.concatenate(blocks) if blocks else np.zeros((0, 3, 3))


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
        tris = step_triangles(shape, STEP_TESSELLATION_MM, STEP_TESSELLATION_ANGLE)
        result = judge(faceted_ir(tris), tris, input_tris, None, samples_per_mm2=samples_per_mm2)
        if not result.truth.max <= bound:
            problems.append(
                f"STEP deviates {result.truth.max * 1000:.1f} um from the input, "
                f"bound {bound * 1000:.1f} um"
            )
    return problems, faces
