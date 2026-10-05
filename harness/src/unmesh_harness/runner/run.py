from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execute import run_pool
from .grid import Grid, git_sha, prep_hash, steps_for
from .results import append_result, completed_keys, latest, make_record, read_results

PREP_TIMEOUT_S = 120.0


@dataclass
class RunSummary:
    results_path: Path
    records: list[dict[str, Any]]
    skipped: int
    ran: int
    seconds: float
    sha: str = ""


def check_extension_fresh() -> None:
    import unmesh._core as core

    from .grid import repo_root

    sources = list((repo_root() / "crates").rglob("*.rs")) + list(
        (repo_root() / "crates").rglob("Cargo.toml")
    )
    built = Path(core.__file__).stat().st_mtime
    stale = [p for p in sources if p.stat().st_mtime > built]
    if stale:
        raise RuntimeError(
            f"the built extension is older than {stale[0]}; run `uv sync` (or `uv run`) to "
            "rebuild it before scoring"
        )


def results_path(out: Path, grid: Grid) -> Path:
    return out / f"{grid.name}.jsonl"


def hidden_results_path(out: Path, grid: Grid) -> Path:
    return out / f"{grid.name}.hidden.json"


def run_grid(
    grid: Grid,
    converters: list[str],
    out: Path,
    jobs: int | None = None,
    sha: str | None = None,
    timeout: float | None = None,
    log=print,
    hidden: bool = False,
) -> RunSummary:
    from .hidden import hidden_seeds

    started = time.perf_counter()
    check_extension_fresh()
    sha = sha or git_sha()
    timeout = timeout or grid.timeout_s
    jobs = jobs or os.cpu_count() or 1
    if hidden:
        grid = dataclasses.replace(grid, seeds=hidden_seeds())
    out.mkdir(parents=True, exist_ok=True)
    if hidden:
        return _run_hidden(grid, converters, out, jobs, sha, timeout, log, started)
    path = results_path(out, grid)
    done = completed_keys(read_results(path))
    cells = [(c, s) for c, s in grid.expand(converters, sha) if c.key not in done]
    skipped = len(grid.expand(converters, sha)) - len(cells)
    if not cells:
        log(f"nothing to run: {skipped} cells already complete in {path}")
        return RunSummary(
            path, list(latest(read_results(path), sha).values()), skipped, 0, 0.0, sha
        )

    cache = out / "cache"
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    stamp = cache / "STAMP"
    wanted = prep_hash(grid)
    if not stamp.is_file() or stamp.read_text() != wanted:
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir()
        stamp.write_text(wanted)

    ran = _run_cells(
        grid,
        sha,
        timeout,
        jobs,
        cache,
        work,
        cells,
        log,
        lambda cell, result: append_result(path, cell, result),
    )
    shutil.rmtree(work, ignore_errors=True)
    return RunSummary(
        path,
        list(latest(read_results(path), sha).values()),
        skipped,
        ran,
        time.perf_counter() - started,
        sha,
    )


def _run_hidden(grid, converters, out, jobs, sha, timeout, log, started) -> RunSummary:
    from .hidden import aggregate

    cells = grid.expand(converters, sha)
    if not cells:
        raise ValueError(f"hidden run of grid {grid.name!r} selects no cells")
    with tempfile.TemporaryDirectory(prefix="unmesh-hidden-") as tmp:
        cache = Path(tmp) / "cache"
        work = Path(tmp) / "work"
        cache.mkdir()
        work.mkdir()
        collected: list[dict[str, Any]] = []

        def collect(cell, result):
            collected.append(make_record(cell, result))

        _run_cells(grid, sha, timeout, jobs, cache, work, cells, log, collect)
    path = hidden_results_path(out, grid)
    payload = aggregate(collected, grid.grid_hash)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"hidden run: {len(collected)} cells aggregated into {path} ({len(payload['rows'])} rows)")
    return RunSummary(path, collected, 0, len(collected), time.perf_counter() - started, sha)


def _run_cells(grid, sha, timeout, jobs, cache, work, cells, log, emit) -> int:
    needed = sorted(
        {c.part for c, _ in cells if not (cache / f"{c.part}.labeled.npz").is_file()}
        | (
            {c.part for c, _ in cells if not (cache / f"{c.part}.truth.npy").is_file()}
            if grid.need_truth
            else set()
        )
    )
    entries = {e["id"]: e for e in grid.entries}
    prep_jobs = [
        (
            part,
            {
                "kind": "prep",
                "entry": entries[part],
                "cache": str(cache),
                "need_truth": grid.need_truth,
                "input_deflection": grid.input_deflection,
                "truth_deflection": grid.truth_deflection,
            },
        )
        for part in needed
    ]
    ready: set[str] = {c.part for c, _ in cells} - set(needed)
    for part, result in run_pool(prep_jobs, jobs, max(timeout, PREP_TIMEOUT_S)):
        if result["status"] != "ok":
            log(f"prep failed for {part}: {result.get('error')}")
        else:
            ready.add(part)

    cell_jobs = []
    failed_prep = []
    for cell, spec in cells:
        task = {
            "kind": "cell",
            "entry": entries[cell.part],
            "cache": str(cache),
            "work": str(work),
            "steps": steps_for(spec),
            "samples_per_mm2": grid.judge_samples_per_mm2,
            "judge_truth": bool(spec.get("judge_truth")),
            "step_deviation": grid.checks_step(cell),
            "dev_input_floor": spec.get("floors", {}).get("dev_input_max", float("inf")),
            "seed": cell.seed,
            "converter": cell.converter,
        }
        if cell.part in ready:
            cell_jobs.append((cell, task))
        else:
            failed_prep.append(cell)
    for cell in failed_prep:
        emit(cell, {"status": "error", "error": "ground-truth preparation failed"})

    ran = 0
    for cell, result in run_pool(cell_jobs, jobs, timeout):
        emit(cell, result)
        ran += 1
        if ran % 100 == 0 or ran == len(cell_jobs):
            log(f"{ran}/{len(cell_jobs)} cells")
    return ran + len(failed_prep)
