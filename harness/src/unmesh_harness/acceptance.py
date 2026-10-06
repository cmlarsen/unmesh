from __future__ import annotations

from typing import Any

from .metrics.recovery import _prf

CONVERTER = "unmesh"
FACETED = "unmesh_harness.runner.converters:convert_faceted"

F1_LINES: list[tuple[str, tuple[str, ...], float]] = [
    (
        "planes",
        (
            "boss_plate",
            "lshape_outline",
            "plate_pockets",
            "polygon_prism",
            "rotated_pockets",
            "square_slots",
            "stepped_block",
            "thin_walls",
            "through_cuts",
        ),
        0.98,
    ),
    (
        "bores, bosses and countersinks",
        ("through_bore", "blind_bore", "counterbore", "round_boss", "countersink"),
        0.95,
    ),
    ("planar chamfers", ("planar_chamfer",), 0.95),
    ("conical chamfers", ("bore_chamfer",), 0.90),
    ("straight-edge fillets", ("straight_fillet",), 0.90),
    ("circular-edge fillets", ("circular_fillet",), 0.85),
]


def _row(r: dict[str, Any]) -> tuple[str, float]:
    return r["operator"], float(r["severity"])


def _cell(r: dict[str, Any]) -> tuple:
    return r["part"], r["operator"], float(r["severity"]), int(r["seed"])


def _valid(r: dict[str, Any]) -> bool:
    return r.get("status") == "ok" and (r.get("validity") or {}).get("valid") is True


def _pooled_f1(records: list[dict[str, Any]]) -> float | None:
    faces = sum(int(r.get("faces") or 0) for r in records)
    regions = sum(int(r.get("regions") or 0) for r in records)
    matched = sum(int(r.get("matched") or 0) for r in records)
    if not records:
        return None
    return _prf(matched, faces, regions)[2]


def table(records: list[dict[str, Any]], deflection: float) -> dict[str, Any]:
    ours = [r for r in records if r.get("converter") == CONVERTER]
    faceted = {_cell(r): r for r in records if r.get("converter") == FACETED}
    rows = sorted({_row(r) for r in ours})
    lines = []
    for name, families, floor in F1_LINES:
        cells = [r for r in ours if r.get("family") in families]
        per_row = {}
        for row in rows:
            rs = [r for r in cells if _row(r) == row]
            ok = [r for r in rs if r.get("status") == "ok"]
            per_row[row] = {"f1": _pooled_f1(ok), "cells": len(rs), "failed": len(rs) - len(ok)}
        values = [v["f1"] for v in per_row.values() if v["f1"] is not None]
        failed = sum(v["failed"] for v in per_row.values())
        worst = min(values) if values else None
        lines.append(
            {
                "line": f"F1 {name} >= {floor:.2f}",
                "rows": per_row,
                "value": worst,
                "failed_cells": failed,
                "pass": worst is not None and worst >= floor and failed == 0,
            }
        )
    by_row: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        rs = [r for r in ours if _row(r) == row]
        valid = sum(1 for r in rs if _valid(r))
        measured = [r for r in rs if r.get("status") == "ok" and r.get("under_report") is not None]
        under = sum(1 for r in measured if r["under_report"] is True)
        compared = worse = 0
        missing = 0
        for r in rs:
            base = faceted.get(_cell(r))
            ours_truth = r.get("dev_truth_max") if r.get("status") == "ok" else None
            base_truth = base.get("dev_truth_max") if base and base.get("status") == "ok" else None
            if ours_truth is None or base_truth is None:
                missing += 1
                continue
            compared += 1
            if ours_truth > base_truth + deflection:
                worse += 1
        by_row[row] = {
            "cells": len(rs),
            "valid": valid,
            "calibration_measured": len(measured),
            "under_reports": under,
            "truth_compared": compared,
            "truth_worse": worse,
            "truth_missing": missing,
        }
    total = {k: sum(v[k] for v in by_row.values()) for k in next(iter(by_row.values()), {})}
    lines.append(
        {
            "line": "validity 100% (faceted fallback counts as valid)",
            "rows": {row: v["valid"] / v["cells"] for row, v in by_row.items()},
            "value": total["valid"] / total["cells"] if total else None,
            "pass": bool(total) and total["valid"] == total["cells"],
        }
    )
    lines.append(
        {
            "line": "calibration never under-reports (100% of cells)",
            "rows": {
                row: (v["calibration_measured"] - v["under_reports"]) / v["cells"]
                for row, v in by_row.items()
            },
            "value": (total["calibration_measured"] - total["under_reports"]) / total["cells"]
            if total
            else None,
            "pass": bool(total)
            and total["under_reports"] == 0
            and total["calibration_measured"] == total["cells"],
        }
    )
    lines.append(
        {
            "line": f"deviation to truth <= faceted + {deflection:g} mm",
            "rows": {
                row: (v["truth_compared"] - v["truth_worse"]) / v["cells"]
                for row, v in by_row.items()
            },
            "value": (total["truth_compared"] - total["truth_worse"]) / total["cells"]
            if total
            else None,
            "pass": bool(total) and total["truth_worse"] == 0 and total["truth_missing"] == 0,
        }
    )
    return {"rows": rows, "lines": lines, "counts": by_row, "total": total}


def failures(records: list[dict[str, Any]], deflection: float) -> list[str]:
    ours = [r for r in records if r.get("converter") == CONVERTER]
    faceted = {_cell(r): r for r in records if r.get("converter") == FACETED}
    out = []
    for r in sorted(ours, key=_cell):
        label = f"{r['part']} {r['operator']} {float(r['severity']):g} seed {r['seed']}"
        if r.get("status") != "ok":
            out.append(f"{label}: {r.get('status')}: {str(r.get('error', ''))[:120]}")
            continue
        if not _valid(r):
            problems = "; ".join((r.get("validity") or {}).get("problems") or [])[:160]
            out.append(f"{label}: invalid: {problems}")
        if r.get("under_report") is True:
            out.append(
                f"{label}: under-reports (reported {r.get('reported_deviation')}, "
                f"calibration {r.get('calibration')})"
            )
        base = faceted.get(_cell(r))
        ours_truth = r.get("dev_truth_max")
        base_truth = base.get("dev_truth_max") if base and base.get("status") == "ok" else None
        if ours_truth is not None and base_truth is not None:
            if ours_truth > base_truth + deflection:
                out.append(
                    f"{label}: deviation to truth {ours_truth * 1000:.1f} um, "
                    f"faceted {base_truth * 1000:.1f} um"
                )
    return out


def render(result: dict[str, Any]) -> str:
    rows = result["rows"]
    head = [
        "acceptance line",
        *(f"{op} {sev:g}" for op, sev in rows),
        "worst row / whole grid",
        "pass",
    ]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for line in result["lines"]:
        cells = []
        for row in rows:
            v = line["rows"].get(row)
            if isinstance(v, dict):
                f1 = v["f1"]
                shown = "-" if f1 is None else f"{f1:.3f}"
                if v["failed"]:
                    shown += f" ({v['failed']} failed)"
                cells.append(shown)
            else:
                cells.append("-" if v is None else f"{v:.1%}")
        value = line["value"]
        if value is None:
            whole = "-"
        elif line["line"].startswith("F1"):
            whole = f"{value:.3f}"
        else:
            whole = f"{value:.1%}"
        out.append(
            "| "
            + " | ".join([line["line"], *cells, whole, "PASS" if line["pass"] else "FAIL"])
            + " |"
        )
    return "\n".join(out)
