from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .grid import Cell

KEY_FIELDS = ("part", "operator", "severity", "seed", "converter", "git_sha")


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


def gate(records: list[dict[str, Any]], target: str = "unmesh", baseline: str = "faceted") -> Gate:
    result = Gate()
    by_key = {record_key(r): r for r in records}
    for r in records:
        if r["converter"] != target:
            continue
        label = f"{r['part']} {r['operator']}@{r['severity']:g} seed {r['seed']}"
        if r["status"] != "ok":
            result.violations.append(f"{label}: {r['status']}: {r.get('error')}")
            continue
        if r.get("under_report"):
            result.violations.append(
                f"{label}: under-reports (calibration {r['calibration'] * 1000:.3f} um)"
            )
        if r["valid"] is False:
            result.violations.append(f"{label}: invalid STEP: {'; '.join(r['step_problems'])}")
        base_key = record_key({**r, "converter": baseline})
        base = by_key.get(base_key)
        if base is None:
            result.violations.append(f"{label}: no {baseline} result to compare against")
        elif base["status"] == "ok" and r["f1"] <= base["f1"]:
            result.violations.append(
                f"{label}: F1 {r['f1']:.3f} does not beat {baseline} F1 {base['f1']:.3f}"
            )
    return result
