#!/usr/bin/env -S uv run --locked python
"""Per-family baseline table from grid results.

Reads one or more `<grid>.jsonl` result files (as written by
``unmesh-harness run``), deduplicates each cell to its latest record, and prints a
markdown table per converter: one row per family, with the mean F1 over the ok
cells, the faceted share (the fraction of the cell's classified regions that are
faceted — a ``facets`` patch, or a surface that does not match the area-weighted
majority true surface type of its input triangles), the failure rate and the
timeout count.
Cells whose operator names a noise step are split into their own section so they
do not dilute the clean signal::

    uv run --locked harness/scripts/baselines_table.py RUNS/smoke.jsonl

The family is derived from the part id (``complex_void-0003`` -> ``complex_void``,
``imported-0225`` -> ``imported``); failed and skipped records carry no ``family``
field, so grouping on the record field used to hide every failure in a ``?`` row.

The faceted share is a per-cell ratio, ``faceted_regions / (faceted_regions +
analytic_regions)``, so a correctly typed base plane beside a dome written as
thousands of planes no longer reads as fully analytic; the table averages it over
the ok cells that have at least one classified region. A cell the writer fell
back on with no classified region at all is 1.0; a cell with no region of either
kind and no fallback is skipped from the share (it stays in ``cells``).

Statuses are counted separately:

- ``ok`` cells are the F1 / faceted-share sample and the numerator of the table.
- a converter ``failure`` (an error or a timeout raised by the converter, its
  external tool, or the write/score stages) and a harness ``timeout`` both count
  against the family; ``timeout`` is also shown in its own column.
- ``skipped`` cells (an inapplicable operator) are excluded from the denominator
  and reported on the ``skipped as inapplicable`` line.
- a ground-truth preparation failure or a degradation (operator) failure is a
  harness failure, not a converter failure: it is reported on its own line and
  excluded from the failure rate.

Each section also carries an ``all families`` total row.

The default converters are unmesh and the three baselines of #12; pass
``--converter`` (repeatable or comma separated) to override.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_CONVERTERS = ("unmesh", "faceted", "freecad-refine", "stl2step")
NOISE_MARKERS = ("noise",)
PREP_FAILURE = "preparation failed"


def is_noise(operator: str) -> bool:
    return any(marker in operator for marker in NOISE_MARKERS)


def family(record: dict[str, Any]) -> str:
    """The corpus family of a record, derived from its part id.

    Reuses the harness's own reverse of ``corpus.entry_id`` so failure and
    timeout records — which carry no ``family`` field — group with their family.
    """
    from unmesh_harness.acceptance import family as _family

    return _family(record)


def faceted_share(record: dict[str, Any]) -> float | None:
    """Fraction of the cell's classified regions that are faceted.

    ``faceted_regions / (faceted_regions + analytic_regions)``. A writer fallback
    with no classified region at all is 1.0; a cell with no region of either kind
    and no fallback is ``None`` (skipped from the share).
    """
    faceted = record.get("faceted_regions") or 0
    analytic = record.get("analytic_regions") or 0
    if faceted + analytic == 0:
        return 1.0 if record.get("fallback") else None
    return faceted / (faceted + analytic)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def is_harness_failure(record: dict[str, Any]) -> bool:
    """True for errors raised before the converter: prep or the degrade chain."""
    if record.get("status") != "error":
        return False
    if PREP_FAILURE in str(record.get("error", "")):
        return True
    tail = "\n".join(record.get("traceback") or [])
    return "unmesh_harness.degrade" in tail or "unmesh_harness/degrade" in tail


def is_timeout(record: dict[str, Any]) -> bool:
    if record.get("status") == "timeout":
        return True
    return str(record.get("error", "")).startswith("TimeoutError")


def classify(record: dict[str, Any]) -> str:
    status = record.get("status")
    if status == "ok":
        return "ok"
    if status == "skipped":
        return "skipped"
    if is_harness_failure(record):
        return "harness"
    if is_timeout(record):
        return "timeout"
    return "failure"


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


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(classify(r) for r in rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    considered = len(rows) - kinds["skipped"] - kinds["harness"]
    failures = kinds["timeout"] + kinds["failure"]
    harness = [r for r in rows if classify(r) == "harness"]
    shares = [s for s in (faceted_share(r) for r in ok) if s is not None]
    return {
        "cells": considered,
        "f1": _mean([float(r["f1"]) for r in ok if r.get("f1") is not None]),
        "faceted_share": _mean(shares),
        "no_regions": len(ok) - len(shares),
        "failure": failures / considered if considered else None,
        "timeout": kinds["timeout"],
        "skipped": kinds["skipped"],
        "harness": len(harness),
        "harness_prep": sum(1 for r in harness if PREP_FAILURE in str(r.get("error", ""))),
    }


def _row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {int(metrics['cells'])} | {_fmt(metrics['f1'])} | "
        f"{_fmt(metrics['faceted_share'], '.0%')} | {_fmt(metrics['failure'], '.0%')} | "
        f"{int(metrics['timeout'])} |"
    )


def _footer(metrics: dict[str, Any]) -> list[str]:
    lines = [f"skipped as inapplicable: {int(metrics['skipped'])}"]
    if metrics["no_regions"]:
        lines.append(
            f"no classified regions (excluded from the faceted share): {int(metrics['no_regions'])}"
        )
    if metrics["harness"]:
        operator = metrics["harness"] - metrics["harness_prep"]
        lines.append(
            f"harness failures (excluded): {int(metrics['harness'])} "
            f"({int(metrics['harness_prep'])} ground-truth preparation, "
            f"{int(operator)} degradation)"
        )
    return lines


def render(records: list[dict[str, Any]], converters: list[str]) -> str:
    present = [c for c in converters if any(r["converter"] == c for r in records)]
    lines: list[str] = []
    for noise in (False, True):
        lines += [f"## {'Noise rows' if noise else 'Clean rows'}", ""]
        for converter in present:
            rows = [
                r
                for r in records
                if r["converter"] == converter and is_noise(r["operator"]) == noise
            ]
            if not rows:
                continue
            lines += [
                f"### {converter}",
                "",
                "| family | cells | mean F1 | faceted share | failure | timeout |",
                "|---|---|---|---|---|---|",
            ]
            for fam in sorted({family(r) for r in rows}):
                lines.append(_row(fam, _metrics([r for r in rows if family(r) == fam])))
            total = _metrics(rows)
            lines.append(_row("**all families**", total))
            lines.append("")
            lines += _footer(total)
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
    shas = sorted({str(r.get("git_sha", "")).split("+", 1)[0] for r in records})
    if len(shas) > 1:
        print(f"warning: records carry {len(shas)} git shas {shas}")
    print(render(records, converters))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
