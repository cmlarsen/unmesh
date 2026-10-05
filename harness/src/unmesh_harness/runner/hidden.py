from __future__ import annotations

import os
from typing import Any

import numpy as np

ENV_VAR = "UNMESH_HIDDEN_SEEDS"

OPERATOR_FAMILY = {
    "coarsen": "tessellation",
    "refine": "tessellation",
    "retriangulate": "tessellation",
    "nonuniform_chords": "tessellation",
    "float32": "precision",
    "truncated_digits": "precision",
    "inch_round_trip": "precision",
    "rotation": "pose",
    "mirror": "pose",
    "far_translation": "pose",
    "noise_off_plane": "noise",
    "noise_isotropic": "noise",
    "noise_normal": "noise",
    "crack_seam": "defects",
    "hole_patch": "defects",
    "slivers": "defects",
    "t_junctions": "defects",
    "stray_shells": "defects",
    "duplicate_facets": "defects",
    "flipped_facets": "defects",
    "unwelded_corners": "defects",
    "unwelded_gap": "defects",
    "nonmanifold_fin": "defects",
    "fillet_rows": "defects",
    "fusion-export": "processing",
    "tinkercad-export": "processing",
    "meshmixer-edit": "processing",
    "slicer-repair": "processing",
    "inch-roundtrip": "processing",
    "identity": "identity",
}

VALUE_METRICS = (
    "f1",
    "dev_input_max",
    "dev_input_p99",
    "dev_truth_max",
    "calibration",
    "seconds",
)

RATE_METRICS = ("ok", "valid", "fallback", "under_report")


def hidden_seeds() -> list[int]:
    raw = os.environ.get(ENV_VAR, "")
    seeds = [s.strip() for s in raw.split(",") if s.strip()]
    if not seeds:
        raise ValueError(
            f"{ENV_VAR} is unset or empty: hidden mode needs a comma-separated "
            "list of integer seeds, e.g. UNMESH_HIDDEN_SEEDS=11,22,33"
        )
    try:
        return [int(s) for s in seeds]
    except ValueError:
        raise ValueError(f"{ENV_VAR} must hold comma-separated integers") from None


def family_of(operator: str) -> str:
    families = {OPERATOR_FAMILY.get(bit, "other") for bit in operator.split("+")}
    if len(families) == 1:
        return next(iter(families))
    return "mixed"


def _stats(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    import math

    return float(value) if math.isfinite(value) else None


def aggregate(records: list[dict[str, Any]], grid_hash: str) -> dict[str, Any]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (
            r["converter"],
            family_of(r["operator"]),
            r["operator"],
            float(r["severity"]),
        )
        groups.setdefault(key, []).append(r)
    rows = []
    for (converter, family, operator, severity), rs in sorted(groups.items()):
        row: dict[str, Any] = {
            "converter": converter,
            "family": family,
            "operator": operator,
            "severity": severity,
            "cells": len(rs),
        }
        for name in VALUE_METRICS:
            vals = [_finite(r.get(name)) for r in rs if r.get("status") == "ok"]
            vals = [v for v in vals if v is not None]
            if vals:
                row[name] = _stats(vals)
        ok = [r for r in rs if r.get("status") == "ok"]
        row["ok_rate"] = sum(1 for _ in ok) / len(rs)
        if ok:
            row["valid_rate"] = sum(1 for r in ok if r.get("valid") is True) / len(ok)
            row["fallback_rate"] = sum(1 for r in ok if r.get("fallback") is True) / len(ok)
            row["under_report_rate"] = sum(
                1 for r in ok if r.get("under_report") is not False
            ) / len(ok)
        rows.append(row)
    return {"grid_hash": grid_hash, "rows": rows}
