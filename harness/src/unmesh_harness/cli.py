from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .corpus import (
    GRID_ORDER,
    build_grid,
    default_cache_dir,
    empty_manifest,
    find_manifest,
    load_manifest,
    planned_entries,
    rejected_reasons,
    sync_imported,
    sync_manifest,
    sync_strata,
)


def _finish(records: list[dict], gate_enabled: bool, grid, converters, sha) -> int:
    from .runner import gate, summarize

    print(summarize(records))
    verdict = gate(records, grid, converters, sha)
    for v in verdict.violations:
        print(f"GATE: {v}")
    if gate_enabled and not verdict.passed:
        print(f"{len(verdict.violations)} gate violation(s)")
        return 1
    return 0


def select_run_entries(grid, category: list[str] | None, shard: str | None):
    if category:
        wanted = set(category)
        entries = [e for e in grid.entries if e["strata"].get("category") in wanted]
        if not entries:
            raise ValueError(f"--category {sorted(wanted)} selects no parts")
        grid = dataclasses.replace(grid, entries=entries)
    if shard is not None:
        try:
            index, _, count = shard.partition("/")
            index, count = int(index), int(count)
        except ValueError:
            raise ValueError(f"--shard must look like 0/3, got {shard!r}") from None
        if not 0 <= index < count or count < 1:
            raise ValueError(f"--shard must look like 0/3, got {shard!r}")
        from .runner.shards import assign_shard, cells_per_part

        picked = assign_shard(grid.entries, index, count, cells_per_part(grid))
        grid = dataclasses.replace(grid, entries=picked)
    return grid


def _run(args) -> int:
    from .runner import load_grid, run_grid

    converters = [c for arg in args.converter for c in arg.split(",")]
    grid = load_grid(args.grid)
    try:
        grid = select_run_entries(grid, args.category, args.shard)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    try:
        summary = run_grid(
            grid,
            converters,
            args.out,
            args.jobs,
            timeout=args.timeout,
            hidden=args.hidden,
            memory_cap_mb=args.memory_cap_mb,
        )
    except ValueError as e:
        print(f"error: {e}")
        return 2
    if args.hidden:
        print(f"hidden run: {summary.ran} cells aggregated (no per-part data written)")
        print(f"results: {summary.results_path}")
        return 0
    print(f"{summary.ran} ran, {summary.skipped} skipped, {summary.seconds:.1f} s")
    print(f"results: {summary.results_path}")
    return _finish(summary.records, args.gate, grid, converters, summary.sha)


def _report(args) -> int:
    from .runner import load_grid
    from .runner.results import latest, read_results

    target = args.results
    if target.is_dir():
        if args.expect_plan is not None:
            missing = _missing_shards(target, args.expect_plan)
            if missing:
                print(f"PARTIAL: missing shard results: {', '.join(missing)}")
                return 2
        all_records = []
        for path in sorted(target.glob("*.jsonl")):
            all_records.extend(read_results(path))
        cache = target / "cache"
    else:
        all_records = read_results(target)
        cache = target.parent / "cache"
    if not all_records:
        print("error: no records found")
        return 2
    shas = sorted({r.get("git_sha", "") for r in all_records})
    if args.git_sha is not None:
        if args.git_sha not in shas:
            print(f"error: no records carry git sha {args.git_sha!r} (found: {shas})")
            return 2
        sha = args.git_sha
    elif len(shas) != 1:
        print(f"error: records carry {len(shas)} git shas {shas}; refusing to mix commits")
        return 2
    else:
        sha = shas[0]
    records = list(latest(all_records, sha).values())
    converters = sorted({r["converter"] for r in records})
    grid = load_grid(args.grid)
    if args.output is not None:
        from .runner.grid import steps_for
        from .runner.report import build_html, viewer_key, viewer_payload, worst_parts

        _ensure_viewer_cache(grid, records, cache)
        viewers = {}
        for record in worst_parts(records):
            try:
                spec = grid.row(record["operator"], float(record["severity"]))
                steps = [(name, float(sev)) for name, sev in steps_for(spec)]
            except (KeyError, ValueError, TypeError):
                continue
            try:
                payload = viewer_payload(record["part"], steps, record["seed"], cache)
            except Exception:
                continue
            if payload is not None:
                viewers[viewer_key(record)] = payload
        args.output.write_text(build_html(records, grid.name, viewers))
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
        return 0
    return _finish(records, args.gate, grid, converters, sha)


def _missing_shards(target, plan_path) -> list[str]:
    expected = json.loads(Path(plan_path).read_text())["shards"]
    have = {p.stem for p in target.glob("*.jsonl")}
    return [name for name in expected if name not in have]


def _ensure_viewer_cache(grid, records, cache) -> None:
    from .runner.execute import prep_part
    from .runner.report import worst_parts

    entries = {e["id"]: e for e in grid.entries}
    cache.mkdir(parents=True, exist_ok=True)
    for record in worst_parts(records):
        part = record["part"]
        if part not in entries or (cache / f"{part}.labeled.npz").is_file():
            continue
        try:
            prep_part(
                {
                    "entry": entries[part],
                    "cache": str(cache),
                    "need_truth": False,
                    "input_deflection": grid.input_deflection,
                    "truth_deflection": grid.truth_deflection,
                }
            )
        except Exception as e:
            print(f"note: skipping viewer for {part}: {type(e).__name__}: {e}")


def _compare(args) -> int:
    from .runner.compare import compare_files, format_comparison

    comp = compare_files(args.run_a, args.run_b, args.converter)
    print(format_comparison(comp, label_a=args.run_a.name, label_b=args.run_b.name))
    if comp.error:
        print(f"ERROR: {comp.error}")
        return 2
    if comp.regressions:
        print(f"{len(comp.regressions)} regression(s) vs {args.run_a.name}")
        return 1
    return 0


def _oracle_fit(args) -> int:
    from .oracle_fit import CURVED_FAMILIES, corpus_seeds, format_table, run_oracle, summarize

    families = [f for arg in args.family or [] for f in arg.split(",")] or list(CURVED_FAMILIES)
    seeds = corpus_seeds(families)
    if args.seeds is not None:
        seeds = {f: s[: args.seeds] for f, s in seeds.items()}
    operators = [o for arg in args.operator for o in arg.split(",")]
    records = run_oracle(
        families, seeds, operators, tuple(args.deflection), args.jobs, args.automatic, args.truth
    )
    print(format_table(summarize(records)))
    return 0


def _acceptance(args) -> int:
    from .acceptance import failures, render, table
    from .runner import load_grid
    from .runner.results import latest, read_results

    all_records = read_results(args.results)
    shas = sorted({r.get("git_sha", "") for r in all_records})
    if len(shas) != 1:
        print(f"error: records carry {len(shas)} git shas {shas}; refusing to mix commits")
        return 2
    records = list(latest(all_records, shas[0]).values())
    deflection = float(load_grid(args.grid).input_deflection[0])
    result = table(records, deflection)
    print(render(result))
    if args.failures:
        print()
        for line in failures(records, deflection):
            print(line)
    return 0 if all(line["pass"] for line in result["lines"]) else 1


def _baselines(args) -> int:
    from .acceptance import baseline_table, render_baselines
    from .runner.results import latest, read_results

    all_records = [r for path in args.results for r in read_results(path)]
    shas = sorted({r.get("git_sha", "") for r in all_records})
    if len(shas) != 1:
        print(f"error: records carry {len(shas)} git shas {shas}; refusing to mix commits")
        return 2
    records = list(latest(all_records, shas[0]).values())
    present = {r.get("converter") for r in records}
    baselines = [c for c in args.baseline if c in present]
    for c in args.baseline:
        if c not in present:
            print(f"warning: no records for baseline {c}")
    if args.converter not in present or not baselines:
        print(f"error: need records for {args.converter} and at least one baseline")
        return 2
    result = baseline_table(records, baselines, args.converter)
    print(render_baselines(result))
    return 0


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
    run.add_argument(
        "--memory-cap-mb",
        type=float,
        default=None,
        help="kill a worker whose RSS exceeds this and record its cell as a 'memory cap' "
        "error (default: the share of total RAM left after headroom, divided by --jobs)",
    )
    run.add_argument("--gate", action="store_true", help="exit 1 on any gate violation")
    run.add_argument(
        "--category",
        action="append",
        default=None,
        help="run only parts whose strata category is listed (repeatable; "
        "used to split the nightly grid into a per-category matrix)",
    )
    run.add_argument(
        "--shard",
        default=None,
        help="run every COUNT-th part starting at INDEX, e.g. 0/3 "
        "(deterministic by part id; used to split large nightly categories)",
    )
    run.add_argument(
        "--hidden",
        action="store_true",
        help="hidden-seed mode: seeds come from UNMESH_HIDDEN_SEEDS and only "
        "aggregate metrics per operator family are written (no per-part data)",
    )

    report = sub.add_parser("report", help="summarize a results file")
    report.add_argument("results", type=Path)
    report.add_argument("--git-sha", default=None)
    report.add_argument("--grid", default="smoke")
    report.add_argument("--gate", action="store_true")
    report.add_argument(
        "--expect-plan",
        type=Path,
        default=None,
        help="JSON file with a 'shards' list of expected <grid>-<category>-<i>-of-<n> "
        "stems; when reporting on a directory, fail listing any missing shard",
    )
    report.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write a self-contained HTML report instead of the text table",
    )

    compare = sub.add_parser("compare", help="compare two results files with noise bands")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--converter", default="unmesh")

    accept = sub.add_parser(
        "acceptance", help="print the #18 acceptance table from an acceptance-grid results file"
    )
    accept.add_argument("results", type=Path)
    accept.add_argument("--grid", default="acceptance")
    accept.add_argument("--failures", action="store_true", help="list every failing cell")

    base = sub.add_parser(
        "baselines", help="per-family F1 and the targets of #18 against each baseline"
    )
    base.add_argument("results", type=Path, nargs="+")
    base.add_argument("--converter", default="unmesh")
    base.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="repeatable; default: faceted (plugin), freecad-refine, stl2step",
    )

    oracle = sub.add_parser(
        "oracle-fit", help="fit surfaces on the oracle segmentation and score recovery"
    )
    oracle.add_argument("--family", action="append")
    oracle.add_argument("--operator", action="append", default=None)
    oracle.add_argument("--seeds", type=int, default=None, help="first N corpus seeds per family")
    oracle.add_argument("--deflection", type=float, nargs=2, default=[0.01, 0.2])
    oracle.add_argument("--jobs", type=int, default=1)
    oracle.add_argument(
        "--automatic",
        action="store_true",
        help="segment with unmesh.convert instead of the oracle labels",
    )
    oracle.add_argument(
        "--truth",
        action="store_true",
        help="also judge the IR against a fine tessellation (0.001, 0.1) of the ground truth",
    )

    args = parser.parse_args(argv)
    if args.group == "oracle-fit":
        args.operator = args.operator or ["identity,float32"]
        return _oracle_fit(args)
    if args.group == "compare":
        return _compare(args)
    if args.group == "run":
        return _run(args)
    if args.group == "report":
        return _report(args)
    if args.group == "acceptance":
        return _acceptance(args)
    if args.group == "baselines":
        from .acceptance import FACETED

        args.baseline = args.baseline or [FACETED, "freecad-refine", "stl2step"]
        return _baselines(args)
    if args.action == "build":
        manifest = load_manifest(args.manifest)
        out = (args.out or default_cache_dir()) / args.grid
        count = build_grid(manifest, args.grid, out)
        print(f"built {count} entries into {out}")
        return 0
    path = args.manifest or find_manifest()
    manifest = json.loads(path.read_text()) if path.exists() else empty_manifest()
    manifest = sync_manifest(manifest, planned_entries())
    manifest = sync_imported(manifest, rejected=rejected_reasons(path))
    manifest = sync_strata(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"pinned {len(manifest['entries'])} entries in {path}")
    return 0
