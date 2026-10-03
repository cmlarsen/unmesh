from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FLOOR_F1 = 0.01
FLOOR_DEV_MM = 0.001
FLOOR_RATE = 0.0


@dataclass(frozen=True)
class Metric:
    name: str
    higher_better: bool
    floor: float


METRICS = (
    Metric("f1", True, FLOOR_F1),
    Metric("dev_input_max", False, FLOOR_DEV_MM),
    Metric("dev_truth_max", False, FLOOR_DEV_MM),
    Metric("valid", True, FLOOR_RATE),
    Metric("fallback", False, FLOOR_RATE),
    Metric("under_report", False, FLOOR_RATE),
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def metric_samples(records: list[dict[str, Any]], name: str) -> list[float]:
    out: list[float] = []
    for r in records:
        ok = r.get("status") == "ok"
        if name == "f1":
            out.append(float(r["f1"]) if ok and _finite(r.get("f1")) is not None else 0.0)
        elif name == "dev_input_max":
            v = _finite(r.get("dev_input_max")) if ok else None
            if v is not None:
                out.append(v)
        elif name == "dev_truth_max":
            v = _finite(r.get("dev_truth_max")) if ok else None
            if v is not None:
                out.append(v)
        elif name == "valid":
            out.append(1.0 if ok and r.get("valid") is True else 0.0)
        elif name == "fallback":
            out.append(1.0 if (r.get("fallback") is True or not ok) else 0.0)
        elif name == "under_report":
            if ok:
                out.append(0.0 if r.get("under_report") is False else 1.0)
    return out


def cell_key(record: dict[str, Any]) -> tuple:
    return (record["part"], record["operator"], float(record["severity"]))


def _fmt_severity(sev: float) -> str:
    return f"{sev:g}"


@dataclass
class CellVerdict:
    key: tuple
    metric: str
    mean_a: float
    std_a: float
    n_a: int
    mean_b: float
    n_b: int
    threshold: float
    verdict: str


@dataclass
class Comparison:
    verdicts: list[CellVerdict] = field(default_factory=list)
    only_a: list[tuple] = field(default_factory=list)
    only_b: list[tuple] = field(default_factory=list)
    unchanged: int = 0
    error: str | None = None

    @property
    def regressions(self) -> list[CellVerdict]:
        return [v for v in self.verdicts if v.verdict == "REGRESSION"]

    @property
    def improvements(self) -> list[CellVerdict]:
        return [v for v in self.verdicts if v.verdict == "IMPROVEMENT"]


def compare_groups(
    groups_a: dict[tuple, list[dict[str, Any]]],
    groups_b: dict[tuple, list[dict[str, Any]]],
) -> Comparison:
    out = Comparison()
    for key in sorted(groups_a, key=str):
        if key not in groups_b:
            out.only_a.append(key)
            out.verdicts.append(
                CellVerdict(key, "missing", 1.0, 0.0, len(groups_a[key]), 0.0, 0, 0.0, "REGRESSION")
            )
            continue
        rec_a = groups_a[key]
        rec_b = groups_b[key]
        seeds_a = {r.get("seed") for r in rec_a}
        seeds_b = {r.get("seed") for r in rec_b}
        if seeds_a - seeds_b:
            out.verdicts.append(
                CellVerdict(
                    key,
                    "seeds",
                    float(len(seeds_a)),
                    0.0,
                    len(rec_a),
                    float(len(seeds_b)),
                    len(rec_b),
                    0.0,
                    "REGRESSION",
                )
            )
            continue
        for metric in METRICS:
            sa = metric_samples(groups_a[key], metric.name)
            sb = metric_samples(groups_b[key], metric.name)
            if not sa:
                continue
            if not sb:
                out.verdicts.append(
                    CellVerdict(
                        key,
                        metric.name,
                        statistics.fmean(sa),
                        statistics.pstdev(sa),
                        len(sa),
                        float("nan"),
                        0,
                        metric.floor,
                        "REGRESSION",
                    )
                )
                continue
            mean_a = statistics.fmean(sa)
            std_a = statistics.pstdev(sa)
            mean_b = statistics.fmean(sb)
            delta = mean_b - mean_a
            threshold = max(2 * std_a, metric.floor)
            if metric.higher_better:
                verdict = (
                    "REGRESSION"
                    if delta < -threshold
                    else ("IMPROVEMENT" if delta > threshold else "same")
                )
            elif delta > threshold:
                verdict = "REGRESSION"
            elif delta < -threshold:
                verdict = "IMPROVEMENT"
            else:
                verdict = "same"
            if verdict == "same":
                out.unchanged += 1
            else:
                out.verdicts.append(
                    CellVerdict(
                        key,
                        metric.name,
                        mean_a,
                        std_a,
                        len(sa),
                        mean_b,
                        len(sb),
                        threshold,
                        verdict,
                    )
                )
    for key in sorted(groups_b, key=str):
        if key not in groups_a:
            out.only_b.append(key)
    return out


def group_records(records: list[dict[str, Any]], converter: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("converter") == converter:
            groups[cell_key(r)].append(r)
    return groups


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def compare_files(path_a: Path, path_b: Path, converter: str = "unmesh") -> Comparison:
    records_a = read_jsonl(path_a)
    records_b = read_jsonl(path_b)
    comp = compare_groups(
        group_records(records_a, converter),
        group_records(records_b, converter),
    )
    if not records_b:
        comp.error = f"run B {path_b} is empty"
    elif not any(r.get("converter") == converter for r in records_b):
        comp.error = f"run B {path_b} has no records for converter {converter!r}"
    return comp


def _show(value: float, metric: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if metric.startswith("dev_"):
        return f"{value * 1000:.2f}um"
    return f"{value:.4f}"


def format_comparison(comp: Comparison, label_a: str = "A", label_b: str = "B") -> str:
    lines = []
    for v in comp.verdicts:
        part, operator, severity = v.key
        delta = v.mean_b - v.mean_a
        sign = "+" if delta >= 0 else "-"
        lines.append(
            f"{v.verdict:<11} {v.metric:<14} {part} {operator}@{_fmt_severity(severity)}  "
            f"{label_a} {_show(v.mean_a, v.metric)}±{_show(v.std_a, v.metric)} (n={v.n_a})  "
            f"{label_b} {_show(v.mean_b, v.metric)} (n={v.n_b})  "
            f"delta {sign}{_show(abs(delta), v.metric)}  thresh {_show(v.threshold, v.metric)}"
        )
    for key in comp.only_a:
        lines.append(f"ONLY-A      {'':<14} {key[0]} {key[1]}@{_fmt_severity(key[2])}  ({label_a})")
    for key in comp.only_b:
        lines.append(f"ONLY-B      {'':<14} {key[0]} {key[1]}@{_fmt_severity(key[2])}  ({label_b})")
    lines.append(
        f"{len(comp.regressions)} regression(s), {len(comp.improvements)} improvement(s), "
        f"{comp.unchanged} unchanged cell-metric(s), "
        f"{len(comp.only_a) + len(comp.only_b)} missing cell(s)"
    )
    return "\n".join(lines)
