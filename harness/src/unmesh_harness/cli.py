from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import (
    GRID_ORDER,
    build_grid,
    default_cache_dir,
    empty_manifest,
    find_manifest,
    load_manifest,
    planar_entries,
    sync_manifest,
)


def _finish(records: list[dict], gate_enabled: bool) -> int:
    from .runner import gate, summarize

    print(summarize(records))
    verdict = gate(records)
    for v in verdict.violations:
        print(f"GATE: {v}")
    if gate_enabled and not verdict.passed:
        print(f"{len(verdict.violations)} gate violation(s)")
        return 1
    return 0


def _run(args) -> int:
    from .runner import load_grid, run_grid

    converters = [c for arg in args.converter for c in arg.split(",")]
    grid = load_grid(args.grid)
    summary = run_grid(grid, converters, args.out, args.jobs, timeout=args.timeout)
    print(f"{summary.ran} ran, {summary.skipped} skipped, {summary.seconds:.1f} s")
    print(f"results: {summary.results_path}")
    return _finish(summary.records, args.gate)


def _report(args) -> int:
    from .runner.results import latest, read_results

    records = list(latest(read_results(args.results), args.git_sha).values())
    return _finish(records, args.gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unmesh-harness")
    sub = parser.add_subparsers(dest="group", required=True)
    corpus = sub.add_parser("corpus").add_subparsers(dest="action", required=True)

    build = corpus.add_parser("build", help="write STEP + metadata JSON for a grid")
    build.add_argument("--grid", choices=GRID_ORDER, default="smoke")
    build.add_argument("--out", type=Path, default=None)
    build.add_argument("--manifest", type=Path, default=None)

    pin = corpus.add_parser(
        "pin", help="append planned entries and fingerprints missing from corpus/v0.json"
    )
    pin.add_argument("--manifest", type=Path, default=None)

    run = sub.add_parser("run", help="score converters on a grid")
    run.add_argument("--converter", action="append", required=True)
    run.add_argument("--grid", default="smoke")
    run.add_argument("--out", type=Path, default=Path("harness-results"))
    run.add_argument("--jobs", type=int, default=None)
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("--gate", action="store_true", help="exit 1 on any gate violation")

    report = sub.add_parser("report", help="summarize a results file")
    report.add_argument("results", type=Path)
    report.add_argument("--git-sha", default=None)
    report.add_argument("--gate", action="store_true")

    args = parser.parse_args(argv)
    if args.group == "run":
        return _run(args)
    if args.group == "report":
        return _report(args)
    if args.action == "build":
        manifest = load_manifest(args.manifest)
        out = (args.out or default_cache_dir()) / args.grid
        count = build_grid(manifest, args.grid, out)
        print(f"built {count} entries into {out}")
        return 0
    path = args.manifest or find_manifest()
    manifest = json.loads(path.read_text()) if path.exists() else empty_manifest()
    manifest = sync_manifest(manifest, planar_entries())
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"pinned {len(manifest['entries'])} entries in {path}")
    return 0
