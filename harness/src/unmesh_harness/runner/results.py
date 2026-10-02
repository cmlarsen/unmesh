from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .grid import Cell

KEY_FIELDS = ("part", "operator", "severity", "seed", "converter", "git_sha", "grid_hash")


def record_key(record: dict[str, Any]) -> tuple:
    return tuple(float(record[f]) if f == "severity" else record[f] for f in KEY_FIELDS)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_result(path: Path, cell: Cell, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "part": cell.part,
        "operator": cell.operator,
        "severity": cell.severity,
        "seed": cell.seed,
        "converter": cell.converter,
        "git_sha": cell.git_sha,
        "grid_hash": cell.grid_hash,
        **result,
    }
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def completed_keys(records: list[dict[str, Any]]) -> set[tuple]:
    return {record_key(r) for r in records if r.get("status") == "ok"}


def latest(records: list[dict[str, Any]], git_sha: str | None = None) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in records:
        if git_sha is None or r["git_sha"] == git_sha:
            out[record_key(r)] = r
    return out


def _valid_rate(rs: list[dict]) -> str:
    checked = [r for r in rs if r["status"] != "ok" or r["valid"] is not None]
    if not checked:
        return "n/a"
    return f"{sum(1 for r in checked if r['status'] == 'ok' and r['valid']) / len(checked):.0%}"


def _um(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.2f}"


def summarize(records: list[dict[str, Any]]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["converter"], r["operator"], r["severity"])].append(r)
    header = (
        "converter  operator                severity  cells  F1 mean  F1 min  dev max um  "
        "calib min um  under  valid  fallback  failed  time mean s  time max s"
    )
    lines = [header, "-" * len(header)]
    for (converter, operator, severity), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        f1 = [r["f1"] for r in ok]
        calib = [r["calibration"] for r in ok if r.get("calibration") is not None]
        times = [r["seconds"] for r in ok]
        devs = [r["dev_input_max"] for r in ok if r["dev_input_max"] is not None]
        mean_f1 = float(np.mean(f1)) if f1 else float("nan")
        min_f1 = min(f1) if f1 else float("nan")
        lines.append(
            f"{converter:<10} {operator:<23} {severity:>8g}  {len(rs):>5}  "
            f"{mean_f1:>7.3f}  {min_f1:>6.3f}  "
            f"{_um(max(devs, default=None)):>10}  "
            f"{_um(min(calib) if calib else None):>12}  "
            f"{sum(1 for r in ok if r.get('under_report')):>5}  "
            f"{_valid_rate(rs):>5}  "
            f"{sum(1 for r in ok if r['fallback']) / len(rs):>8.0%}  "
            f"{len(rs) - len(ok):>6}  "
            f"{np.mean(times) if times else float('nan'):>11.3f}  "
            f"{max(times) if times else float('nan'):>10.3f}"
        )
    return "\n".join(lines)


@dataclass
class Gate:
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def _label(r: dict[str, Any]) -> str:
    return f"{r['converter']} {r['part']} {r['operator']}@{r['severity']:g} seed {r['seed']}"


def _cell_violations(r: dict[str, Any], floors: dict[str, Any]) -> list[str]:
    label = _label(r)
    out = []

    def bad(msg: str) -> None:
        out.append(f"{label}: {msg}")

    if r.get("under_report") is not False:
        calib = r.get("calibration")
        detail = "no usable reported max_deviation" if calib is None else f"{calib * 1000:.3f} um"
        bad(f"under-reports ({detail})")
    if r.get("valid") is not True:
        bad(f"invalid STEP: {'; '.join(r.get('step_problems') or ['not checked'])}")
    if "f1_cell" in floors and r["f1"] < floors["f1_cell"]:
        bad(f"F1 {r['f1']:.3f} below floor {floors['f1_cell']}")
    if floors.get("regions_equal_faces") and r["regions"] != r["faces"]:
        bad(f"{r['regions']} regions for {r['faces']} ground-truth faces")
    if "fallback_max" in floors and int(r["fallback"]) > floors["fallback_max"]:
        bad("writer fell back to faceted")
    if "dev_input_max" in floors and not r["dev_input_max"] <= floors["dev_input_max"]:
        bad(
            f"deviation to input {r['dev_input_max'] * 1000:.2f} um above "
            f"{floors['dev_input_max'] * 1000:.2f} um"
        )
    if "dev_truth_max" in floors:
        truth = r.get("dev_truth_max")
        if truth is None or not truth <= floors["dev_truth_max"]:
            shown = "not measured" if truth is None else f"{truth * 1000:.2f} um"
            bad(f"deviation to truth {shown} above {floors['dev_truth_max'] * 1000:.2f} um")
    return out


def gate(records: list[dict[str, Any]], grid, converters: list[str], sha: str | None = None):
    from .converters import BASELINES

    result = Gate()
    by_key = {record_key(r): r for r in records}
    gated = [c for c in converters if c not in BASELINES]
    if not gated:
        result.violations.append("no gated converter in this run")
    for cell, spec in grid.expand(converters, sha if sha is not None else _sha_of(records)):
        r = by_key.get(cell.key)
        if r is None:
            result.violations.append(
                f"{cell.converter} {cell.part} {cell.operator}@{cell.severity:g} "
                f"seed {cell.seed}: no result"
            )
        elif r["status"] != "ok":
            result.violations.append(f"{_label(r)}: {r['status']}: {r.get('error')}")
        elif cell.converter in gated:
            result.violations += _cell_violations(r, spec.get("floors", {}))
    for converter in gated:
        for spec in grid.cells:
            floor = spec.get("floors", {}).get("f1_row")
            if floor is None:
                continue
            rows = [
                r
                for r in records
                if r["converter"] == converter
                and r["status"] == "ok"
                and r["operator"] == spec["operator"]
                and float(r["severity"]) == float(spec["severity"])
            ]
            matched = sum(r["matched"] for r in rows)
            faces = sum(r["faces"] for r in rows)
            regions = sum(r["regions"] for r in rows)
            precision = matched / regions if regions else 0.0
            recall = matched / faces if faces else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 < floor:
                result.violations.append(
                    f"{converter} {spec['operator']}@{spec['severity']:g}: "
                    f"row micro-F1 {f1:.4f} below floor {floor}"
                )
    return result


def _sha_of(records: list[dict[str, Any]]) -> str:
    return records[-1]["git_sha"] if records else ""
