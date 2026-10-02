from __future__ import annotations

import math
from typing import Any

import numpy as np

from unmesh.ir import Ir

from ..labels import LabeledMesh

IOU_MIN = 0.8
ANGLE_TOL_DEG = 0.2
OFFSET_TOL_MM = 0.01
RADIUS_REL_TOL = 0.01
RADIUS_ABS_TOL_MM = 0.01
HALF_ANGLE_TOL_DEG = 0.2
CENTER_TOL_MM = 0.01
SPLIT_SHARE = 0.95

ANALYTIC_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")


def _frame(to_original: Any) -> np.ndarray:
    if isinstance(to_original, LabeledMesh):
        from ..degrade import to_original as frame_of

        return np.array(frame_of(to_original), dtype=np.float64)
    if to_original is None:
        return np.eye(4)
    return np.array(to_original, dtype=np.float64)


def _map_points(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ frame[:3, :3].T + frame[:3, 3]


def _map_direction(direction: Any, frame: np.ndarray) -> np.ndarray:
    v = np.asarray(direction, dtype=np.float64) @ frame[:3, :3].T
    return v / np.linalg.norm(v)


def _mapped_surface(surface: Any, frame: np.ndarray) -> dict[str, Any]:
    kind = surface.type
    if kind == "plane":
        return {
            "type": kind,
            "origin": _map_points(np.array([surface.origin]), frame)[0],
            "normal": _map_direction(surface.normal, frame),
        }
    if kind == "cylinder":
        return {
            "type": kind,
            "origin": _map_points(np.array([surface.origin]), frame)[0],
            "axis": _map_direction(surface.axis, frame),
            "radius": float(surface.radius),
        }
    if kind == "cone":
        return {
            "type": kind,
            "apex": _map_points(np.array([surface.apex]), frame)[0],
            "axis": _map_direction(surface.axis, frame),
            "half_angle": float(surface.half_angle),
        }
    if kind == "sphere":
        return {
            "type": kind,
            "center": _map_points(np.array([surface.center]), frame)[0],
            "radius": float(surface.radius),
        }
    if kind == "torus":
        return {
            "type": kind,
            "center": _map_points(np.array([surface.center]), frame)[0],
            "axis": _map_direction(surface.axis, frame),
            "major_radius": float(surface.major_radius),
            "minor_radius": float(surface.minor_radius),
        }
    return {"type": kind}


def _unsigned_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return math.degrees(math.acos(min(1.0, cos)))


def _signed_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def _radius_ok(radius: float, truth: float) -> tuple[bool, float, float]:
    absolute = abs(radius - truth)
    return (
        absolute <= max(RADIUS_REL_TOL * abs(truth), RADIUS_ABS_TOL_MM),
        absolute / abs(truth),
        absolute,
    )


def surface_matches(
    face_surface: str, face_params: dict[str, Any], centroid: np.ndarray, mapped: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    if mapped["type"] != face_surface or face_surface not in ANALYTIC_TYPES:
        return False, {}
    errors: dict[str, float] = {}
    if face_surface == "plane":
        truth = np.asarray(face_params["normal"], dtype=np.float64)
        normal = np.asarray(mapped["normal"], dtype=np.float64)
        origin = np.asarray(mapped["origin"], dtype=np.float64)
        errors["normal_deg"] = _unsigned_deg(truth, normal)
        errors["offset_mm"] = abs(float((centroid - origin) @ normal)) / float(
            np.linalg.norm(normal)
        )
        return (
            errors["normal_deg"] <= ANGLE_TOL_DEG and errors["offset_mm"] <= OFFSET_TOL_MM,
            errors,
        )
    if face_surface == "cylinder":
        ok_r, rel, absolute = _radius_ok(mapped["radius"], face_params["radius"])
        errors["radius_rel"] = rel
        errors["radius_abs_mm"] = absolute
        errors["axis_deg"] = _unsigned_deg(
            np.asarray(face_params["axis"], dtype=np.float64),
            np.asarray(mapped["axis"], dtype=np.float64),
        )
        return ok_r and errors["axis_deg"] <= ANGLE_TOL_DEG, errors
    if face_surface == "cone":
        errors["axis_deg"] = _signed_deg(
            np.asarray(face_params["axis"], dtype=np.float64),
            np.asarray(mapped["axis"], dtype=np.float64),
        )
        errors["half_angle_deg"] = abs(
            math.degrees(mapped["half_angle"]) - math.degrees(face_params["half_angle"])
        )
        return (
            errors["axis_deg"] <= ANGLE_TOL_DEG and errors["half_angle_deg"] <= HALF_ANGLE_TOL_DEG
        ), errors
    if face_surface == "sphere":
        ok_r, rel, absolute = _radius_ok(mapped["radius"], face_params["radius"])
        errors["radius_rel"] = rel
        errors["radius_abs_mm"] = absolute
        errors["center_mm"] = float(
            np.linalg.norm(
                np.asarray(mapped["center"], dtype=np.float64)
                - np.asarray(face_params["center"], dtype=np.float64)
            )
        )
        return ok_r and errors["center_mm"] <= CENTER_TOL_MM, errors
    ok_major, major_rel, major_abs = _radius_ok(mapped["major_radius"], face_params["major_radius"])
    ok_minor, minor_rel, minor_abs = _radius_ok(mapped["minor_radius"], face_params["minor_radius"])
    errors["major_rel"] = major_rel
    errors["major_abs_mm"] = major_abs
    errors["minor_rel"] = minor_rel
    errors["minor_abs_mm"] = minor_abs
    errors["axis_deg"] = _unsigned_deg(
        np.asarray(face_params["axis"], dtype=np.float64),
        np.asarray(mapped["axis"], dtype=np.float64),
    )
    return ok_major and ok_minor and errors["axis_deg"] <= ANGLE_TOL_DEG, errors


def _face_centroid(tris: np.ndarray) -> np.ndarray:
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(cross, axis=1)
    centres = tris.mean(axis=1)
    return (centres * area[:, None]).sum(axis=0) / area.sum()


def _overlap(clean: LabeledMesh, face_id: np.ndarray, ir: Ir):
    regions = {r.id: np.asarray(r.triangles, dtype=np.int64) for r in ir.regions}
    owner = np.full(len(face_id), -1, dtype=np.int64)
    for rid, tris in regions.items():
        owner[tris] = rid
    return regions, owner


def match_faces(clean: LabeledMesh, face_id: np.ndarray, ir: Ir) -> tuple[list[dict], np.ndarray]:
    regions, owner = _overlap(clean, face_id, ir)
    matches = []
    for face in clean.faces:
        members = np.nonzero(face_id == face.id)[0]
        if len(members) == 0:
            matches.append({"face": face.id, "region": None, "iou": 0.0, "overlap": 0})
            continue
        counts = np.bincount(owner[members] + 1, minlength=len(ir.regions) + 1)
        counts[0] = 0
        best = int(np.argmax(counts)) - 1
        if best < 0:
            matches.append({"face": face.id, "region": None, "iou": 0.0, "overlap": 0})
            continue
        inter = int(counts[best + 1])
        iou = inter / (len(members) + len(regions[best]) - inter)
        matches.append({"face": face.id, "region": best, "iou": float(iou), "overlap": inter})
    return matches, owner


def _prf(matched: int, faces: int, regions: int) -> tuple[float, float, float]:
    recall = matched / faces if faces else 0.0
    precision = matched / regions if regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _prf_regions(
    matched: int, matched_regions: int, faces: int, regions: int
) -> tuple[float, float, float]:
    recall = matched / faces if faces else 0.0
    precision = matched_regions / regions if regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _segmentation(
    clean: LabeledMesh, face_id: np.ndarray, ir: Ir, owner: np.ndarray
) -> dict[str, Any]:
    n = len(face_id)
    n_faces = len(clean.faces)
    n_regions = len(ir.regions)
    region_size = np.zeros(n_regions, dtype=np.int64)
    for r in ir.regions:
        region_size[r.id] = len(r.triangles)
    face_size = np.bincount(np.asarray(face_id), minlength=n_faces)
    covered = owner >= 0
    inter = np.zeros((n_faces, n_regions), dtype=np.int64)
    np.add.at(inter, (np.asarray(face_id)[covered], owner[covered]), 1)
    union = face_size[:, None] + region_size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    tri_iou = np.zeros(n)
    tri_iou[covered] = iou[np.asarray(face_id)[covered], owner[covered]]
    best_iou = iou.max(axis=1) if n_regions else np.zeros(n_faces)
    over = 0
    for f in range(n_faces):
        owned = owner[np.nonzero(face_id == f)[0]]
        owned = owned[owned >= 0]
        if len(owned) == 0:
            continue
        _, counts = np.unique(owned, return_counts=True)
        if len(counts) >= 2 and counts.max() / len(owned) <= SPLIT_SHARE:
            over += 1
    under = 0
    for r in ir.regions:
        members = np.nonzero(owner == r.id)[0]
        if len(members) == 0:
            continue
        _, counts = np.unique(np.asarray(face_id)[members], return_counts=True)
        if len(counts) >= 2 and counts.max() / len(members) <= SPLIT_SHARE:
            under += 1
    return {
        "triangles": int(n),
        "covered": int(covered.sum()),
        "uncovered": int(n - covered.sum()),
        "mean_triangle_iou": float(tri_iou[covered].mean()) if covered.any() else 0.0,
        "best_iou_mean": float(best_iou.mean()) if n_faces else 0.0,
        "best_iou_min": float(best_iou.min()) if n_faces else 0.0,
        "over_segmented": int(over),
        "under_segmented": int(under),
    }


def _max_point_to_polyline(points: np.ndarray, poly: np.ndarray, closed: bool) -> float:
    if len(poly) < 2 or len(points) == 0:
        return 0.0 if len(points) == 0 else float("inf")
    start = poly if closed else poly[:-1]
    end = np.roll(poly, -1, axis=0) if closed else poly[1:]
    ab = end - start
    denom = (ab**2).sum(axis=1)
    ap = points[:, None, :] - start[None, :, :]
    t = np.zeros_like((ap * ab).sum(axis=2))
    valid = denom > 0
    t[:, valid] = (ap[:, valid, :] * ab[None, valid, :]).sum(axis=2) / denom[valid]
    t = np.clip(t, 0.0, 1.0)
    d2 = ((ap - t[:, :, None] * ab[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).max())


def _symmetric_polyline_error(
    gt_points: np.ndarray, ir_points: np.ndarray, ir_closed: bool
) -> float:
    gt_closed = (
        len(gt_points) >= 2 and bool(np.linalg.norm(gt_points[0] - gt_points[-1]) < 1e-9)
    ) or ir_closed
    forward = _max_point_to_polyline(gt_points, ir_points, ir_closed)
    backward = _max_point_to_polyline(ir_points, gt_points, gt_closed)
    return max(forward, backward)


def _edge_position_error(
    clean: LabeledMesh,
    best_region: dict[int, int | None],
    ir: Ir,
    frame: np.ndarray,
) -> dict[str, Any]:
    truth = clean.metadata.get("truth", {}).get("edge_points")
    gt_lines = (
        [np.asarray(p, dtype=np.float64) for p in truth]
        if truth is not None
        else [np.asarray(a.points, dtype=np.float64) for a in clean.adjacency]
    )
    ir_lines: dict[tuple[int, int], list[tuple[np.ndarray, bool]]] = {}
    for adj in ir.adjacencies:
        key = (min(adj.regions), max(adj.regions))
        ir_lines[key] = [
            (_map_points(np.asarray(b.points, dtype=np.float64), frame), b.closed)
            for b in adj.boundaries
        ]
    details = []
    for i, adj in enumerate(clean.adjacency):
        ra = best_region.get(adj.face_a)
        rb = best_region.get(adj.face_b)
        entry: dict[str, Any] = {
            "faces": [int(adj.face_a), int(adj.face_b)],
            "regions": [None if ra is None else int(ra), None if rb is None else int(rb)],
            "tangent": bool(adj.tangent),
            "types": [clean.faces[adj.face_a].surface, clean.faces[adj.face_b].surface],
            "error": None,
        }
        key = None
        if ra is not None and rb is not None and ra != rb:
            key = (min(ra, rb), max(ra, rb))
        if key is not None and key in ir_lines and i < len(gt_lines) and len(gt_lines[i]):
            entry["error"] = min(
                _symmetric_polyline_error(gt_lines[i], pts, closed) for pts, closed in ir_lines[key]
            )
        details.append(entry)
    evaluated = [d["error"] for d in details if d["error"] is not None]

    def stats(errors: list[float]) -> dict[str, Any]:
        return {
            "evaluated": len(errors),
            "max": float(max(errors)) if errors else 0.0,
            "mean": float(sum(errors) / len(errors)) if errors else 0.0,
        }

    sharp = [d["error"] for d in details if d["error"] is not None and not d["tangent"]]
    tangent = [d["error"] for d in details if d["error"] is not None and d["tangent"]]
    return {
        "evaluated": len(evaluated),
        "missing": len(details) - len(evaluated),
        **stats(evaluated),
        "sharp": stats(sharp),
        "tangent": stats(tangent),
        "details": details,
    }


def score_recovery(
    clean: LabeledMesh,
    face_id: np.ndarray,
    ir: Ir,
    to_original: Any = None,
) -> dict[str, Any]:
    frame = _frame(to_original)
    face_id = np.asarray(face_id)
    matches, owner = match_faces(clean, face_id, ir)
    best_region = {m["face"]: m["region"] for m in matches}
    matched_regions: set[int] = set()
    matched = unsupported = 0
    faces_detail = []
    error_values: dict[str, list[float]] = {}
    per_type: dict[str, dict[str, Any]] = {}
    for face, match in zip(clean.faces, matches, strict=True):
        bucket = per_type.setdefault(
            face.surface, {"faces": 0, "regions": 0, "matched": 0, "matched_regions": set()}
        )
        bucket["faces"] += 1
        detail: dict[str, Any] = {
            "face": face.id,
            "type": face.surface,
            "region": match["region"],
            "iou": match["iou"],
            "recovered": False,
            "errors": {},
        }
        if match["iou"] < IOU_MIN or match["region"] is None:
            faces_detail.append(detail)
            continue
        if face.surface not in ANALYTIC_TYPES:
            unsupported += 1
            faces_detail.append(detail)
            continue
        surface = ir.regions[match["region"]].surface
        centroid = _face_centroid(clean.face_tris(face.id))
        ok, errors = surface_matches(
            face.surface, face.params, centroid, _mapped_surface(surface, frame)
        )
        detail["errors"] = {k: float(v) for k, v in errors.items()}
        if ok:
            matched += 1
            bucket["matched"] += 1
            bucket["matched_regions"].add(match["region"])
            matched_regions.add(match["region"])
            detail["recovered"] = True
            for k, v in errors.items():
                error_values.setdefault(f"{face.surface}.{k}", []).append(float(v))
        faces_detail.append(detail)
    for region in ir.regions:
        if region.surface.type in per_type:
            per_type[region.surface.type]["regions"] += 1
    faces = len(clean.faces)
    precision, recall, f1 = _prf_regions(matched, len(matched_regions), faces, len(ir.regions))
    by_type = {}
    for surface, bucket in per_type.items():
        p, r, f = _prf_regions(
            bucket["matched"], len(bucket["matched_regions"]), bucket["faces"], bucket["regions"]
        )
        by_type[surface] = {
            "faces": bucket["faces"],
            "regions": bucket["regions"],
            "matched": bucket["matched"],
            "precision": p,
            "recall": r,
            "f1": f,
        }
    out: dict[str, Any] = {
        "faces": faces,
        "regions": len(ir.regions),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_faces": unsupported,
        "per_type": by_type,
        "segmentation": _segmentation(clean, face_id, ir, owner),
        "parameter_errors": {
            k: {"n": len(v), "max": max(v), "mean": sum(v) / len(v)}
            for k, v in error_values.items()
        },
        "edge_error": _edge_position_error(clean, best_region, ir, frame),
        "faces_detail": faces_detail,
    }
    return out


def aggregate(records: list[dict[str, Any]], by: str = "family") -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, int]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        if by == "strata":
            strata = r.get("strata") or {}
            key = strata.get("category", "?") if isinstance(strata, dict) else str(strata)
        else:
            key = str(r.get(by, "?"))
        bucket = groups.setdefault(key, {"cells": 0, "faces": 0, "regions": 0, "matched": 0})
        bucket["cells"] += 1
        bucket["faces"] += int(r.get("faces", 0))
        bucket["regions"] += int(r.get("regions", 0))
        bucket["matched"] += int(r.get("matched", 0))
    out = {}
    for key, bucket in groups.items():
        precision, recall, f1 = _prf(bucket["matched"], bucket["faces"], bucket["regions"])
        out[key] = {**bucket, "precision": precision, "recall": recall, "f1": f1}
    return out
