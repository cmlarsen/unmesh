from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

import unmesh

from .labels import LabeledMesh
from .metrics.recovery import CURVED_ACCEPTANCE, Tolerances, score_recovery

DEFAULT_DEFLECTION = (0.01, 0.2)
COARSE_DEFLECTION = (0.1, 0.5)
CURVED_FAMILIES = (
    "through_bore",
    "blind_bore",
    "counterbore",
    "round_boss",
    "countersink",
    "bore_chamfer",
    "straight_fillet",
)
OPERATORS: dict[str, list[tuple[str, float]]] = {
    "identity": [],
    "float32": [("float32", 1.0)],
    "rotation": [("rotation", 0.5)],
}
CURVED_TYPES = ("cylinder", "cone")
RADIUS_REL = CURVED_ACCEPTANCE.radius_rel
RADIUS_ABS_MM = CURVED_ACCEPTANCE.radius_abs_mm


def convert_oracle(mesh: LabeledMesh, options: unmesh.ConvertOptions | None = None):
    tris = np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3)
    return unmesh.convert_from_labels(tris, np.asarray(mesh.face_id, dtype=np.int64), options)


def convert_automatic(mesh: LabeledMesh, options: unmesh.ConvertOptions | None = None):
    return unmesh.convert(np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3), options)


def _truth_radius(face_type: str, params: dict[str, Any], centroid: np.ndarray) -> float:
    if face_type == "cylinder":
        return float(params["radius"])
    axis = np.asarray(params["axis"], dtype=np.float64)
    h = float((centroid - np.asarray(params["apex"], dtype=np.float64)) @ axis)
    return h * math.tan(params["half_angle"])


def score_oracle(
    clean: LabeledMesh,
    degraded: LabeledMesh,
    tolerances: Tolerances = CURVED_ACCEPTANCE,
    automatic: bool = False,
) -> dict[str, Any]:
    from .degrade import to_original
    from .metrics.recovery import _face_centroid

    ir, report = (convert_automatic if automatic else convert_oracle)(degraded)
    rec = score_recovery(clean, degraded.face_id, ir, to_original(degraded), tolerances)
    radius_ratio = 0.0
    radius_rel = 0.0
    axis_deg = 0.0
    curved = unscored = 0
    for face, detail in zip(clean.faces, rec["faces_detail"], strict=True):
        if face.surface not in CURVED_TYPES:
            continue
        curved += 1
        errors = detail["errors"]
        if "radius_abs_mm" not in errors:
            unscored += 1
            continue
        truth = _truth_radius(face.surface, face.params, _face_centroid(clean.face_tris(face.id)))
        allowed = max(RADIUS_REL * abs(truth), RADIUS_ABS_MM)
        radius_ratio = max(radius_ratio, errors["radius_abs_mm"] / allowed)
        radius_rel = max(radius_rel, errors["radius_rel"])
        axis_deg = max(axis_deg, errors["axis_deg"])
    return {
        "faces": rec["faces"],
        "regions": rec["regions"],
        "matched": rec["matched"],
        "f1": rec["f1"],
        "curved_faces": curved,
        "curved_unscored": unscored,
        "radius_ratio_max": radius_ratio,
        "radius_rel_max": radius_rel,
        "axis_deg_max": axis_deg,
        "region_counts": dict(report.region_counts),
        "max_deviation": report.max_deviation,
    }


def oracle_cell(
    family: str,
    seed: int,
    operator: str = "identity",
    deflection: tuple[float, float] = DEFAULT_DEFLECTION,
    automatic: bool = False,
) -> dict[str, Any]:
    from .degrade import chain
    from .groundtruth import generate
    from .labels import tessellate

    gt = generate(family, seed)
    clean = tessellate(gt.solid, *deflection)
    if gt.face_tags is not None:
        clean.metadata["face_tags"] = dict(gt.face_tags)
    degraded = chain(clean, OPERATORS[operator], seed)
    record = {"family": family, "seed": seed, "operator": operator}
    record.update(score_oracle(clean, degraded, automatic=automatic))
    return record


def _cell(args: tuple) -> dict[str, Any]:
    return oracle_cell(*args)


def run_oracle(
    families: list[str],
    seeds: dict[str, list[int]],
    operators: list[str],
    deflection: tuple[float, float] = DEFAULT_DEFLECTION,
    jobs: int = 1,
    automatic: bool = False,
) -> list[dict[str, Any]]:
    tasks = [
        (family, seed, op, deflection, automatic)
        for family in families
        for seed in seeds[family]
        for op in operators
    ]
    if jobs <= 1:
        return [_cell(t) for t in tasks]
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(_cell, tasks))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["family"], r["operator"]), []).append(r)
    rows = []
    for (family, op), rs in groups.items():
        faces = sum(r["faces"] for r in rs)
        regions = sum(r["regions"] for r in rs)
        matched = sum(r["matched"] for r in rs)
        recall = matched / faces if faces else 0.0
        precision = matched / regions if regions else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "family": family,
                "operator": op,
                "parts": len(rs),
                "faces": faces,
                "f1": f1,
                "curved_faces": sum(r["curved_faces"] for r in rs),
                "curved_unscored": sum(r["curved_unscored"] for r in rs),
                "radius_ratio_max": max(r["radius_ratio_max"] for r in rs),
                "radius_rel_max": max(r["radius_rel_max"] for r in rs),
                "axis_deg_max": max(r["axis_deg_max"] for r in rs),
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    head = (
        f"{'family':<16} {'operator':<9} {'parts':>5} {'faces':>6} {'F1':>7} "
        f"{'curved':>6} {'r_err/allowed':>13} {'r_rel':>9} {'axis_deg':>9}"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['family']:<16} {r['operator']:<9} {r['parts']:>5} {r['faces']:>6} "
            f"{r['f1']:>7.4f} {r['curved_faces']:>6} {r['radius_ratio_max']:>13.3g} "
            f"{r['radius_rel_max']:>9.2e} {r['axis_deg_max']:>9.2e}"
        )
    return "\n".join(lines)


def corpus_seeds(families: list[str]) -> dict[str, list[int]]:
    from .corpus import load_manifest

    entries = load_manifest()["entries"]
    return {f: sorted(e["seed"] for e in entries if e.get("family") == f) for f in families}
