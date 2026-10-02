from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execute import run_pool
from .grid import Grid, git_sha, prep_hash
from .results import append_result, completed_keys, latest, read_results

PREP_TIMEOUT_S = 120.0


@dataclass
class RunSummary:
    results_path: Path
    records: list[dict[str, Any]]
    skipped: int
    ran: int
    seconds: float
    sha: str = ""


def results_path(out: Path, grid: Grid) -> Path:
    return out / f"{grid.name}.jsonl"


def run_grid(
    grid: Grid,
    converters: list[str],
    out: Path,
    jobs: int | None = None,
    sha: str | None = None,
    timeout: float | None = None,
    log=print,
) -> RunSummary:
    started = time.perf_counter()
    sha = sha or git_sha()
    timeout = timeout or grid.timeout_s
    jobs = jobs or os.cpu_count() or 1
    out.mkdir(parents=True, exist_ok=True)
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
            "steps": spec["steps"],
            "samples_per_mm2": grid.judge_samples_per_mm2,
            "judge_truth": bool(spec.get("judge_truth")),
            "seed": cell.seed,
            "converter": cell.converter,
        }
        if cell.part in ready:
            cell_jobs.append((cell, task))
        else:
            failed_prep.append(cell)
    for cell in failed_prep:
        append_result(path, cell, {"status": "error", "error": "ground-truth preparation failed"})

    ran = 0
    for cell, result in run_pool(cell_jobs, jobs, timeout):
        append_result(path, cell, result)
        ran += 1
        if ran % 100 == 0 or ran == len(cell_jobs):
            log(f"{ran}/{len(cell_jobs)} cells")
    shutil.rmtree(work, ignore_errors=True)
    return RunSummary(
        path,
        list(latest(read_results(path), sha).values()),
        skipped,
        ran + len(failed_prep),
        time.perf_counter() - started,
        sha,
    )
