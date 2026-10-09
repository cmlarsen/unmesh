from __future__ import annotations

from typing import Any

import numpy as np

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


def family(r: dict[str, Any]) -> str:
    return str(r["part"]).rsplit("-", 1)[0]


def _pooled_f1(records: list[dict[str, Any]], truth_faces: dict[str, int]) -> float | None:
    if not records:
        return None
    faces = regions = matched = 0
    for r in records:
        if r.get("status") == "ok":
            faces += int(r.get("faces") or 0)
            regions += int(r.get("regions") or 0)
            matched += int(r.get("matched") or 0)
        else:
            faces += truth_faces.get(r["part"], truth_faces.get(family(r), 1))
    return _prf(matched, faces, regions)[2]


def table(records: list[dict[str, Any]], deflection: float) -> dict[str, Any]:
    ours = [r for r in records if r.get("converter") == CONVERTER]
    faceted = {_cell(r): r for r in records if r.get("converter") == FACETED}
    rows = sorted({_row(r) for r in ours})
    truth_faces: dict[str, int] = {}
    for r in ours:
        if r.get("status") == "ok":
            for key in (r["part"], family(r)):
                truth_faces[key] = max(truth_faces.get(key, 0), int(r.get("faces") or 0))
    lines = []
    for name, families, floor in F1_LINES:
        cells = [r for r in ours if family(r) in families]
        per_row = {}
        for row in rows:
            rs = [r for r in cells if _row(r) == row]
            ok = [r for r in rs if r.get("status") == "ok"]
            per_row[row] = {
                "f1": _pooled_f1(rs, truth_faces),
                "cells": len(rs),
                "failed": len(rs) - len(ok),
            }
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


BASELINE_MARGIN = 0.10
SHORT_NAMES = {FACETED: "faceted"}


def short_name(converter: str) -> str:
    return SHORT_NAMES.get(converter, converter)


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rs if r.get("status") == "ok"]
    dev = [r["dev_truth_max"] for r in ok if r.get("dev_truth_max") is not None]
    seconds = [r["seconds"] for r in ok if r.get("seconds") is not None]
    measured = [r for r in ok if r.get("under_report") is not None]
    return {
        "cells": len(rs),
        "failed": len(rs) - len(ok),
        "timeouts": sum(1 for r in rs if r.get("status") == "timeout"),
        "valid": sum(1 for r in rs if _valid(r)) / len(rs) if rs else None,
        "dev_truth_median": _quantile(dev, 0.5),
        "dev_truth_max": max(dev) if dev else None,
        "under_reports": sum(1 for r in measured if r["under_report"] is True),
        "unreported": sum(1 for r in ok if r.get("reported_deviation") is None),
        "seconds_median": _quantile(seconds, 0.5),
        "seconds_p90": _quantile(seconds, 0.9),
    }


def baseline_table(
    records: list[dict[str, Any]], baselines: list[str], ours: str = CONVERTER
) -> dict[str, Any]:
    converters = [ours, *baselines]
    rows = sorted({_row(r) for r in records if r.get("converter") in converters})
    truth_faces: dict[str, int] = {}
    for r in records:
        if r.get("status") == "ok":
            for key in (r["part"], family(r)):
                truth_faces[key] = max(truth_faces.get(key, 0), int(r.get("faces") or 0))
    groups = [(name, families) for name, families, _ in F1_LINES]
    groups += [(f, (f,)) for f in sorted({family(r) for r in records})]
    f1: dict[str, dict[str, dict[tuple, float | None]]] = {}
    for c in converters:
        mine = [r for r in records if r.get("converter") == c]
        f1[c] = {
            name: {
                row: _pooled_f1(
                    [r for r in mine if family(r) in fams and _row(r) == row], truth_faces
                )
                for row in rows
            }
            for name, fams in groups
        }
    stats = {
        c: {
            row: _stats([r for r in records if r.get("converter") == c and _row(r) == row])
            for row in rows
        }
        for c in converters
    }
    targets = []
    for name, _ in groups:
        for row in rows:
            best = [(f1[b][name][row], b) for b in baselines if f1[b][name][row] is not None]
            if not best:
                continue
            value, holder = max(best)
            have = f1[ours][name][row]
            target = value + BASELINE_MARGIN
            targets.append(
                {
                    "group": name,
                    "row": row,
                    "best_baseline": holder,
                    "baseline_f1": value,
                    "target": target,
                    "ours": have,
                    "pass": have is not None and have >= target,
                }
            )
    return {
        "rows": rows,
        "groups": [g for g, _ in groups],
        "f1": f1,
        "stats": stats,
        "targets": targets,
        "converters": converters,
    }


def _fmt(v: float | None, spec: str = ".3f") -> str:
    return "-" if v is None else format(v, spec)


def _um(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.1f}"


def render_baselines(result: dict[str, Any]) -> str:
    rows = result["rows"]
    heads = [f"{op} {sev:g}" for op, sev in rows]
    out = []
    for c in result["converters"]:
        out += [f"### {short_name(c)}", "", "| F1 | " + " | ".join(heads) + " |"]
        out.append("|---|" + "---|" * len(rows))
        for g in result["groups"]:
            out.append(
                f"| {g} | " + " | ".join(_fmt(result["f1"][c][g][row]) for row in rows) + " |"
            )
        out += [
            "",
            "| row | cells | failed (timeout) | valid | dev truth median / max (um) "
            "| under-reports (unreported) | seconds median / p90 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row, head in zip(rows, heads, strict=True):
            s = result["stats"][c][row]
            dev = f"{_um(s['dev_truth_median'])} / {_um(s['dev_truth_max'])}"
            out.append(
                f"| {head} | {s['cells']} | {s['failed']} ({s['timeouts']}) | "
                f"{_fmt(s['valid'], '.1%')} | {dev} | {s['under_reports']} ({s['unreported']}) | "
                f"{_fmt(s['seconds_median'], '.2f')} / {_fmt(s['seconds_p90'], '.2f')} |"
            )
        out.append("")
    out += [
        "### Targets (best baseline + 0.10)",
        "",
        "| group | row | best baseline | baseline F1 | target | unmesh | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in result["targets"]:
        op, sev = t["row"]
        out.append(
            f"| {t['group']} | {op} {sev:g} | {short_name(t['best_baseline'])} | "
            f"{_fmt(t['baseline_f1'])} | {_fmt(t['target'])} | "
            f"{_fmt(t['ours'])} | {'PASS' if t['pass'] else 'FAIL'} |"
        )
    return "\n".join(out)
