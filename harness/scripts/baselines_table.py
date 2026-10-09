#!/usr/bin/env -S uv run --locked python
"""Per-family baseline table from grid results.

Reads one or more `<grid>.jsonl` result files (as written by
``unmesh-harness run``), deduplicates each cell to its latest record, and prints a
markdown table per converter: one row per family, with the mean F1 over the ok
cells, the rate at which the output has no analytic faces (the writer fell back
to faceted, or every written region is faceted — a ``facets`` patch, or a surface
that does not match the area-weighted majority true surface type of its input
triangles), and the failure rate.
Cells whose operator names a noise step are split into their own section so they
do not dilute the clean signal::

    uv run --locked harness/scripts/baselines_table.py RUNS/smoke.jsonl

The default converters are unmesh and the three baselines of #12; pass
``--converter`` (repeatable or comma separated) to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_CONVERTERS = ("unmesh", "faceted", "freecad-refine", "stl2step")
NOISE_MARKERS = ("noise",)


def is_noise(operator: str) -> bool:
    return any(marker in operator for marker in NOISE_MARKERS)


def no_analytic(record: dict[str, Any]) -> bool:
    if record.get("fallback"):
        return True
    analytic = record.get("analytic_regions")
    if analytic is None:
        return False
    return analytic == 0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    from unmesh_harness.runner.results import read_results

    latest: dict[tuple, dict[str, Any]] = {}
    for path in paths:
        for record in read_results(path):
            key = (
                record["part"],
                record["operator"],
                float(record["severity"]),
                record["seed"],
                record["converter"],
            )
            latest[key] = record
    return list(latest.values())


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "cells": float(len(rows)),
        "f1": _mean([float(r["f1"]) for r in ok if r.get("f1") is not None]),
        "no_analytic": _mean([1.0 if no_analytic(r) else 0.0 for r in ok]),
        "failure": (len(rows) - len(ok)) / len(rows) if rows else None,
    }


def render(records: list[dict[str, Any]], converters: list[str]) -> str:
    present = [c for c in converters if any(r["converter"] == c for r in records)]
    lines: list[str] = []
    for noise in (False, True):
        lines += [f"## {'Noise rows' if noise else 'Clean rows'}", ""]
        for converter in present:
            families = sorted(
                {
                    r.get("family", "?")
                    for r in records
                    if r["converter"] == converter and is_noise(r["operator"]) == noise
                }
            )
            if not families:
                continue
            lines += [
                f"### {converter}",
                "",
                "| family | cells | mean F1 | no-analytic | failure |",
                "|---|---|---|---|---|",
            ]
            for family in families:
                rows = [
                    r
                    for r in records
                    if r["converter"] == converter
                    and is_noise(r["operator"]) == noise
                    and r.get("family", "?") == family
                ]
                m = _metrics(rows)
                lines.append(
                    f"| {family} | {int(m['cells'])} | {_fmt(m['f1'])} | "
                    f"{_fmt(m['no_analytic'], '.0%')} | {_fmt(m['failure'], '.0%')} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--converter",
        action="append",
        default=None,
        help="repeatable or comma separated; default: unmesh, faceted, freecad-refine, stl2step",
    )
    args = parser.parse_args(argv)
    converters = [
        name
        for arg in args.converter or [",".join(DEFAULT_CONVERTERS)]
        for name in arg.split(",")
        if name
    ]
    records = load_records(args.results)
    if not records:
        print("error: no records found")
        return 2
    shas = sorted({r.get("git_sha", "") for r in records})
    if len(shas) > 1:
        print(f"warning: records carry {len(shas)} git shas {shas}")
    print(render(records, converters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
