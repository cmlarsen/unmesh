from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from unmesh.ir import Ir, Plane

from ..groundtruth import validity_problems
from ..labels import LabeledMesh

NORMAL_TOL_DEG = 0.2
OFFSET_TOL_MM = 0.01
IOU_MIN = 0.8
VOLUME_REL_TOL = 0.01


def _face_centroid(tris: np.ndarray) -> np.ndarray:
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(cross, axis=1)
    centres = tris.mean(axis=1)
    return (centres * area[:, None]).sum(axis=0) / area.sum()


def _plane_matches(face_params: dict, centroid: np.ndarray, plane: Plane, to_original: np.ndarray):
    rot, shift = to_original[:3, :3], to_original[:3, 3]
    normal = rot @ np.asarray(plane.normal, dtype=np.float64)
    origin = rot @ np.asarray(plane.origin, dtype=np.float64) + shift
    truth = np.asarray(face_params["normal"], dtype=np.float64)
    cos = abs(float(truth @ normal)) / (np.linalg.norm(truth) * np.linalg.norm(normal))
    angle = math.degrees(math.acos(min(1.0, cos)))
    offset = abs(float((centroid - origin) @ normal)) / float(np.linalg.norm(normal))
    return angle <= NORMAL_TOL_DEG and offset <= OFFSET_TOL_MM


def face_recovery(
    clean: LabeledMesh, face_id: np.ndarray, ir: Ir, to_original: np.ndarray
) -> dict[str, Any]:
    regions = {r.id: np.asarray(r.triangles, dtype=np.int64) for r in ir.regions}
    owner = np.full(len(face_id), -1, dtype=np.int64)
    for rid, tris in regions.items():
        owner[tris] = rid
    matched_regions: set[int] = set()
    matched = unsupported = 0
    for face in clean.faces:
        members = np.nonzero(face_id == face.id)[0]
        if len(members) == 0:
            continue
        counts = np.bincount(owner[members] + 1, minlength=len(ir.regions) + 1)
        counts[0] = 0
        best = int(np.argmax(counts)) - 1
        if best < 0:
            continue
        inter = int(counts[best + 1])
        iou = inter / (len(members) + len(regions[best]) - inter)
        if iou < IOU_MIN:
            continue
        surface = ir.regions[best].surface
        if face.surface != "plane":
            unsupported += 1
            continue
        if not isinstance(surface, Plane):
            continue
        centroid = _face_centroid(clean.face_tris(face.id))
        if _plane_matches(face.params, centroid, surface, to_original):
            matched += 1
            matched_regions.add(best)
    faces = len(clean.faces)
    recall = matched / faces if faces else 0.0
    precision = len(matched_regions) / len(ir.regions) if ir.regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "faces": faces,
        "regions": len(ir.regions),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_faces": unsupported,
    }


def step_problems(path: str | Path | None, truth_volume: float) -> list[str]:
    if path is None or not Path(path).is_file():
        return ["no STEP file written"]
    from build123d import import_step

    try:
        shape = import_step(str(path))
    except Exception as e:
        return [f"STEP unreadable: {type(e).__name__}: {e}"]
    problems = validity_problems(shape)
    if abs(shape.volume - truth_volume) > VOLUME_REL_TOL * truth_volume:
        problems.append(f"volume {shape.volume:.4f} differs from truth {truth_volume:.4f}")
    return problems
