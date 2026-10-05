from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from unmesh.ir import Ir

from ..labels import FaceInfo, LabeledMesh, distance_to_surface

IOU_MIN = 0.8
ANGLE_TOL_DEG = 0.2
OFFSET_TOL_MM = 0.01
POSITION_TOL_MM = 0.01
RADIUS_REL_TOL = 0.01
RADIUS_ABS_TOL_MM = 0.01
HALF_ANGLE_TOL_DEG = 0.2
CENTER_TOL_MM = 0.01
FRAME_ORTHO_TOL = 1e-6
SPLIT_SHARE = 0.95

ANALYTIC_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")

CONFIDENCE_MASK = 0.9


@dataclass(frozen=True)
class Tolerances:
    normal_deg: float = ANGLE_TOL_DEG
    axis_deg: float = ANGLE_TOL_DEG
    offset_mm: float = OFFSET_TOL_MM
    position_mm: float = POSITION_TOL_MM
    radius_rel: float = RADIUS_REL_TOL
    radius_abs_mm: float = RADIUS_ABS_TOL_MM
    half_angle_deg: float = HALF_ANGLE_TOL_DEG
    center_mm: float = CENTER_TOL_MM


DEFAULT_TOLERANCES = Tolerances()
CURVED_ACCEPTANCE = replace(DEFAULT_TOLERANCES, axis_deg=0.1, radius_rel=0.005)


def _frame(to_original: Any) -> np.ndarray:
    if isinstance(to_original, LabeledMesh):
        from ..degrade import to_original as frame_of

        frame = np.array(frame_of(to_original), dtype=np.float64)
    elif to_original is None:
        frame = np.eye(4)
    else:
        frame = np.array(to_original, dtype=np.float64)
    if frame.shape != (4, 4) or not np.allclose(
        frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=FRAME_ORTHO_TOL
    ):
        raise ValueError("to_original frame must be rigid (orthonormal 3x3 part)")
    return frame


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


def _radius_ok(radius: float, truth: float, tol: Tolerances) -> tuple[bool, float, float]:
    absolute = abs(radius - truth)
    return (
        absolute <= max(tol.radius_rel * abs(truth), tol.radius_abs_mm),
        absolute / abs(truth),
        absolute,
    )


def _point_to_line(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.linalg.norm(rel - np.outer(rel @ unit, unit), axis=1)


def _axis_point(face_params: dict[str, Any], key: str, centroid: np.ndarray) -> np.ndarray:
    base = np.asarray(face_params[key], dtype=np.float64)
    axis = np.asarray(face_params["axis"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return base + float((np.asarray(centroid, dtype=np.float64) - base) @ axis) * axis


def _axis_line_error(face_params: dict[str, Any], mapped: dict[str, Any], centroid) -> float:
    point = _axis_point(face_params, "origin", centroid)
    return float(_point_to_line(point[None, :], mapped["origin"], mapped["axis"])[0])


def _cone_radius_at(apex, axis, half_angle: float, point: np.ndarray) -> float:
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    height = float((point - np.asarray(apex, dtype=np.float64)) @ unit)
    return height * math.tan(half_angle)


def _position_error(face: FaceInfo, gt_points: np.ndarray, mapped: dict[str, Any]) -> float:
    params = {k: v for k, v in mapped.items() if k != "type"}
    probe = FaceInfo(face.id, face.surface, params, False)
    pts = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return float("inf")
    return float(distance_to_surface(probe, pts).max())


def _orientation_mismatch(face: FaceInfo, ir_orientation: Any) -> float:
    expected = "reversed" if face.reversed else "same"
    return float(ir_orientation != expected)


def surface_matches(
    face: FaceInfo,
    gt_points: np.ndarray,
    centroid: np.ndarray,
    mapped: dict[str, Any],
    ir_orientation: Any = None,
    tol: Tolerances = DEFAULT_TOLERANCES,
) -> tuple[bool, dict[str, float]]:
    face_surface = face.surface
    face_params = face.params
    if mapped["type"] != face_surface or face_surface not in ANALYTIC_TYPES:
        return False, {}
    errors: dict[str, float] = {}
    if face_surface == "plane":
        truth = np.asarray(face_params["normal"], dtype=np.float64)
        normal = np.asarray(mapped["normal"], dtype=np.float64)
        origin = np.asarray(mapped["origin"], dtype=np.float64)
        errors["normal_deg"] = _signed_deg(truth, normal)
        errors["offset_mm"] = abs(float((centroid - origin) @ normal)) / float(
            np.linalg.norm(normal)
        )
        errors["position_mm"] = _position_error(face, gt_points, mapped)
        return (
            errors["normal_deg"] <= tol.normal_deg and errors["offset_mm"] <= tol.offset_mm,
            errors,
        )
    if face_surface == "cylinder":
        ok_r, rel, absolute = _radius_ok(mapped["radius"], face_params["radius"], tol)
        errors["radius_rel"] = rel
        errors["radius_abs_mm"] = absolute
        errors["axis_deg"] = _unsigned_deg(
            np.asarray(face_params["axis"], dtype=np.float64),
            np.asarray(mapped["axis"], dtype=np.float64),
        )
        errors["orientation_mismatch"] = _orientation_mismatch(face, ir_orientation)
        errors["axis_mm"] = _axis_line_error(face_params, mapped, centroid)
        errors["position_mm"] = _position_error(face, gt_points, mapped)
        return (
            ok_r
            and errors["axis_deg"] <= tol.axis_deg
            and not errors["orientation_mismatch"]
            and errors["axis_mm"] <= tol.position_mm,
            errors,
        )
    if face_surface == "cone":
        errors["axis_deg"] = _signed_deg(
            np.asarray(face_params["axis"], dtype=np.float64),
            np.asarray(mapped["axis"], dtype=np.float64),
        )
        errors["half_angle_deg"] = abs(
            math.degrees(mapped["half_angle"]) - math.degrees(face_params["half_angle"])
        )
        errors["orientation_mismatch"] = _orientation_mismatch(face, ir_orientation)
        errors["apex_mm"] = float(
            np.linalg.norm(
                np.asarray(mapped["apex"], dtype=np.float64)
                - np.asarray(face_params["apex"], dtype=np.float64)
            )
        )
        point = _axis_point(face_params, "apex", centroid)
        errors["axis_mm"] = float(_point_to_line(point[None, :], mapped["apex"], mapped["axis"])[0])
        ok_r, rel, absolute = _radius_ok(
            _cone_radius_at(mapped["apex"], mapped["axis"], mapped["half_angle"], point),
            _cone_radius_at(
                face_params["apex"], face_params["axis"], face_params["half_angle"], point
            ),
            tol,
        )
        errors["radius_rel"] = rel
        errors["radius_abs_mm"] = absolute
        errors["position_mm"] = _position_error(face, gt_points, mapped)
        return (
            errors["axis_deg"] <= tol.axis_deg
            and errors["half_angle_deg"] <= tol.half_angle_deg
            and not errors["orientation_mismatch"]
            and errors["axis_mm"] <= tol.position_mm
            and ok_r,
            errors,
        )
    if face_surface == "sphere":
        ok_r, rel, absolute = _radius_ok(mapped["radius"], face_params["radius"], tol)
        errors["radius_rel"] = rel
        errors["radius_abs_mm"] = absolute
        errors["center_mm"] = float(
            np.linalg.norm(
                np.asarray(mapped["center"], dtype=np.float64)
                - np.asarray(face_params["center"], dtype=np.float64)
            )
        )
        errors["orientation_mismatch"] = _orientation_mismatch(face, ir_orientation)
        errors["position_mm"] = _position_error(face, gt_points, mapped)
        return (
            ok_r and errors["center_mm"] <= tol.center_mm and not errors["orientation_mismatch"],
            errors,
        )
    ok_major, major_rel, major_abs = _radius_ok(
        mapped["major_radius"], face_params["major_radius"], tol
    )
    ok_minor, minor_rel, minor_abs = _radius_ok(
        mapped["minor_radius"], face_params["minor_radius"], tol
    )
    errors["major_rel"] = major_rel
    errors["major_abs_mm"] = major_abs
    errors["minor_rel"] = minor_rel
    errors["minor_abs_mm"] = minor_abs
    errors["axis_deg"] = _unsigned_deg(
        np.asarray(face_params["axis"], dtype=np.float64),
        np.asarray(mapped["axis"], dtype=np.float64),
    )
    errors["orientation_mismatch"] = _orientation_mismatch(face, ir_orientation)
    errors["center_mm"] = float(
        np.linalg.norm(
            np.asarray(mapped["center"], dtype=np.float64)
            - np.asarray(face_params["center"], dtype=np.float64)
        )
    )
    errors["position_mm"] = _position_error(face, gt_points, mapped)
    return (
        ok_major
        and ok_minor
        and errors["axis_deg"] <= tol.axis_deg
        and not errors["orientation_mismatch"]
        and errors["center_mm"] <= tol.center_mm,
        errors,
    )


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


def match_faces(
    clean: LabeledMesh, face_id: np.ndarray, ir: Ir, confidence: np.ndarray | None = None
) -> tuple[list[dict], np.ndarray]:
    face_id = np.asarray(face_id)
    for region in ir.regions:
        tris = np.asarray(region.triangles, dtype=np.int64)
        if len(tris) and (tris.min() < 0 or tris.max() >= len(face_id)):
            bad = int(tris[tris < 0][0]) if (tris < 0).any() else int(tris[tris >= len(face_id)][0])
            raise ValueError(
                f"region {region.id}: source triangle {bad} outside [0, {len(face_id)})"
            )
    _, owner_full = _overlap(clean, face_id, ir)
    owner = owner_full
    keep: np.ndarray | None = None
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        if len(confidence) != len(face_id):
            raise ValueError(
                f"confidence has {len(confidence)} entries for {len(face_id)} triangles"
            )
        keep = np.asarray(confidence) >= CONFIDENCE_MASK
        owner = np.where(keep, owner_full, -1)
    matches = []
    for face in clean.faces:
        members = np.nonzero(face_id == face.id)[0]
        if keep is not None:
            members = members[keep[members]]
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
        denom = len(members) + int((owner == best).sum()) - inter
        iou = inter / denom if denom else 0.0
        matches.append({"face": face.id, "region": best, "iou": float(iou), "overlap": inter})
    return matches, owner_full


def _masked_faces(clean: LabeledMesh) -> set[int]:
    tags = clean.metadata.get("face_tags") if isinstance(clean.metadata, dict) else None
    if not isinstance(tags, dict):
        return set()
    masked = set()
    for key, value in tags.items():
        if value != "corner_blend":
            continue
        try:
            fid = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= fid < len(clean.faces):
            masked.add(fid)
    return masked


def _masked_regions(face_id: np.ndarray, masked: set[int], ir: Ir) -> set[int]:
    if not masked:
        return set()
    owners = np.asarray(face_id)
    out = set()
    for region in ir.regions:
        tris = np.asarray(region.triangles, dtype=np.int64)
        tris = tris[(tris >= 0) & (tris < len(owners))]
        if len(tris) and bool(np.isin(owners[tris], list(masked)).all()):
            out.add(region.id)
    return out


def _unrecoverable(
    clean: LabeledMesh, face_id: np.ndarray, ir: Ir, confidence: np.ndarray | None
) -> tuple[set[int], set[int]]:
    if confidence is None:
        return set(), set()
    keep = np.asarray(confidence, dtype=np.float64) >= CONFIDENCE_MASK
    present = set(np.asarray(face_id)[keep].tolist())
    faces = {f.id for f in clean.faces if f.id not in present}
    regions = set()
    for region in ir.regions:
        tris = np.asarray(region.triangles, dtype=np.int64)
        if len(tris) and not keep[tris].any():
            regions.add(region.id)
    return faces, regions


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


def _poly_segments(poly: np.ndarray, closed: bool) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 2:
        return np.zeros((0, 2, 3))
    segs = np.stack([p[:-1], p[1:]], axis=1)
    if closed and np.linalg.norm(p[0] - p[-1]) > 1e-9:
        segs = np.concatenate([segs, np.stack([[p[-1]], [p[0]]], axis=1)])
    return segs


def _max_point_to_segments(points: np.ndarray, segs: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0 or len(segs) == 0:
        return float("inf")
    ab = segs[:, 1] - segs[:, 0]
    denom = (ab**2).sum(axis=1)
    ap = p[:, None, :] - segs[None, :, 0]
    t = np.zeros((len(p), len(segs)))
    valid = denom > 0
    t[:, valid] = (ap[:, valid, :] * ab[None, valid, :]).sum(axis=2) / denom[valid]
    np.clip(t, 0.0, 1.0, out=t)
    d2 = ((ap - t[:, :, None] * ab[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).max())


def _edge_position_error(
    clean: LabeledMesh,
    recovered_region: dict[int, int],
    ir: Ir,
    frame: np.ndarray,
) -> dict[str, Any]:
    truth = clean.metadata.get("truth", {}).get("edge_points")
    gt_lines = (
        [np.asarray(p, dtype=np.float64) for p in truth]
        if truth is not None
        else [np.asarray(a.points, dtype=np.float64) for a in clean.adjacency]
    )
    pairs: dict[tuple[int, int], list[int]] = {}
    for i, adj in enumerate(clean.adjacency):
        pairs.setdefault((min(adj.face_a, adj.face_b), max(adj.face_a, adj.face_b)), []).append(i)
    ir_lines: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, bool]]] = {}
    for adj in ir.adjacencies:
        key = (min(adj.regions), max(adj.regions))
        entries = []
        for b in adj.boundaries:
            pts = _map_points(np.asarray(b.points, dtype=np.float64), frame)
            entries.append((_poly_segments(pts, b.closed), pts, b.closed))
        ir_lines[key] = entries
    details = []
    for (fa, fb), idxs in sorted(pairs.items()):
        ra = recovered_region.get(fa)
        rb = recovered_region.get(fb)
        entry: dict[str, Any] = {
            "faces": [int(fa), int(fb)],
            "regions": [None if ra is None else int(ra), None if rb is None else int(rb)],
            "tangent": bool(all(clean.adjacency[i].tangent for i in idxs)),
            "types": [clean.faces[fa].surface, clean.faces[fb].surface],
            "edges": len(idxs),
            "error": None,
        }
        key = None
        if ra is not None and rb is not None and ra != rb:
            key = (min(ra, rb), max(ra, rb))
        gt_lists = []
        if key is not None and key in ir_lines:
            gt_lists = [gt_lines[i] for i in idxs if i < len(gt_lines) and len(gt_lines[i]) >= 2]
        if key is not None and key in ir_lines and gt_lists:
            bounds = ir_lines[key]
            ir_closed = any(closed for _, _, closed in bounds)
            gt_segs = np.concatenate(
                [
                    _poly_segments(g, closed=(np.linalg.norm(g[0] - g[-1]) < 1e-9 or ir_closed))
                    for g in gt_lists
                ]
            )
            forward = max(
                min(_max_point_to_segments(g, segs) for segs, _, _ in bounds) for g in gt_lists
            )
            backward = _max_point_to_segments(
                np.concatenate([pts for _, pts, _ in bounds]), gt_segs
            )
            entry["error"] = max(forward, backward)
        details.append(entry)
    evaluated = [d["error"] for d in details if d["error"] is not None]

    def stats(errors: list[float]) -> dict[str, Any]:
        if not errors:
            return {"evaluated": 0, "max": None, "mean": None}
        return {
            "evaluated": len(errors),
            "max": float(max(errors)),
            "mean": float(sum(errors) / len(errors)),
        }

    region_face: dict[int, int] = {}
    for face, region in recovered_region.items():
        region_face.setdefault(region, face)
    gt_pairs = set(pairs)
    spurious = 0
    for adj in ir.adjacencies:
        a, b = adj.regions
        fa = region_face.get(a)
        fb = region_face.get(b)
        if fa is not None and fb is not None and (min(fa, fb), max(fa, fb)) not in gt_pairs:
            spurious += 1

    sharp = [d["error"] for d in details if d["error"] is not None and not d["tangent"]]
    tangent = [d["error"] for d in details if d["error"] is not None and d["tangent"]]
    return {
        "evaluated": len(evaluated),
        "missing": len(details) - len(evaluated),
        **stats(evaluated),
        "sharp": stats(sharp),
        "tangent": stats(tangent),
        "spurious": spurious,
        "details": details,
    }


def score_recovery(
    clean: LabeledMesh,
    face_id: np.ndarray,
    ir: Ir,
    to_original: Any = None,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    frame = _frame(to_original)
    face_id = np.asarray(face_id)
    matches, owner = match_faces(clean, face_id, ir, confidence)
    masked = _masked_faces(clean)
    masked_regions = _masked_regions(face_id, masked, ir)
    lost_faces, lost_regions = _unrecoverable(clean, face_id, ir, confidence)
    matched_regions: set[int] = set()
    matched = unsupported = 0
    faces_detail = []
    error_values: dict[str, list[float]] = {}
    per_type: dict[str, dict[str, Any]] = {}
    for face, match in zip(clean.faces, matches, strict=True):
        if face.id in masked or face.id in lost_faces:
            bucket = None
        else:
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
            if face.id not in masked:
                unsupported += 1
            faces_detail.append(detail)
            continue
        surface = ir.regions[match["region"]].surface
        face_tris = clean.face_tris(face.id)
        centroid = _face_centroid(face_tris)
        ok, errors = surface_matches(
            face,
            face_tris.reshape(-1, 3),
            centroid,
            _mapped_surface(surface, frame),
            getattr(surface, "orientation", None),
            tolerances,
        )
        detail["errors"] = {k: float(v) for k, v in errors.items()}
        if ok and face.id not in masked:
            matched += 1
            assert bucket is not None
            bucket["matched"] += 1
            bucket["matched_regions"].add(match["region"])
            matched_regions.add(match["region"])
            detail["recovered"] = True
            for k, v in errors.items():
                error_values.setdefault(f"{face.surface}.{k}", []).append(float(v))
        elif ok:
            detail["recovered"] = True
        faces_detail.append(detail)
    for region in ir.regions:
        if region.id in masked_regions or region.id in lost_regions:
            continue
        if region.surface.type in per_type:
            per_type[region.surface.type]["regions"] += 1
    faces = len(clean.faces) - len(masked | lost_faces)
    regions = len(ir.regions) - len(masked_regions | lost_regions)
    precision, recall, f1 = _prf_regions(matched, len(matched_regions), faces, regions)
    if confidence is not None and faces == 0 and regions == 0:
        precision = recall = f1 = 1.0
    recovered_region = {d["face"]: d["region"] for d in faces_detail if d["recovered"]}
    by_type = {}
    for surface, bucket in per_type.items():
        if bucket["faces"] == 0:
            continue
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
    masked_faces = [d for d in faces_detail if d["face"] in masked]
    out: dict[str, Any] = {
        "faces": faces,
        "regions": regions,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_faces": unsupported,
        "masked_faces": masked_faces,
        "per_type": by_type,
        "segmentation": _segmentation(clean, face_id, ir, owner),
        "parameter_errors": {
            k: {"n": len(v), "max": max(v), "mean": sum(v) / len(v)}
            for k, v in error_values.items()
        },
        "edge_error": _edge_position_error(clean, recovered_region, ir, frame),
        "faces_detail": faces_detail,
    }
    if confidence is not None:
        out["faces_unrecoverable"] = len(lost_faces)
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
