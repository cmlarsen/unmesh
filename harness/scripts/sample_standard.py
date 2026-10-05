#!/usr/bin/env -S uv run --locked python
"""Seeded random sample of the standard grid with per-cell timing.

Runs N randomly sampled (part, operator, severity, seed) cells through the
normal runner machinery (prep pool + per-cell timeout pool) and prints
per-cell progress plus a classification of every timeout/error::

    uv run --locked harness/scripts/sample_standard.py --n 400 --out /tmp/sample86

Classification buckets:
  operator   - the degrade chain raised (traceback in unmesh_harness.degrade)
  prep       - ground-truth preparation failed
  converter  - convert/score raised outside degrade (converter, judge, STEP…)
  timeout    - the cell exceeded the per-cell timeout; re-runs degrade alone
               in-process to attribute the slow stage (degrade vs convert/score)

Timed-out cells are re-degraded in-process: if degrade itself exceeds the
timeout the slow stage is degrade, otherwise convert/score. Results append to
<out>/standard-sample.jsonl; the sampling seed makes reruns select the same
cells. Exit 0 unless the run itself crashes.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--converter", default="unmesh")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("harness-results/sample"))
    parser.add_argument(
        "--attribute",
        type=Path,
        default=None,
        metavar="DIR",
        help="skip sampling; attribute the timeouts in DIR/standard-sample.jsonl "
        "using DIR/cache and exit",
    )
    parser.add_argument("--shard", default=None, metavar="I/N")
    return parser.parse_args(argv)


def stage_of(record: dict) -> str:
    tail = "\n".join(record.get("traceback", [])[-6:])
    if "unmesh_harness.degrade" in tail or "unmesh_harness/degrade" in tail:
        return "operator"
    return "converter"


def main(argv=None) -> int:
    args = parse_args(argv)
    from unmesh_harness.runner.grid import git_sha, load_grid, prep_hash

    grid = load_grid("standard")
    if args.attribute is not None:
        timeout = args.timeout or grid.timeout_s
        records = [
            json.loads(line)
            for line in (args.attribute / "standard-sample.jsonl").read_text().splitlines()
            if line.strip()
        ]
        timed_out = [r for r in records if r.get("status") == "timeout"]
        print(f"attributing {len(timed_out)} timeout(s) in {args.attribute}…", flush=True)
        if args.shard is not None:
            index, _, count = args.shard.partition("/")
            timed_out = [r for i, r in enumerate(timed_out) if i % int(count) == int(index)]
            print(f"shard {args.shard}: {len(timed_out)} timeout(s)", flush=True)
        _attribute_timeouts(grid, args.attribute / "cache", timeout, timed_out)
        return 0
    from unmesh_harness.runner.results import append_result
    from unmesh_harness.runner.run import _run_cells

    sha = git_sha()
    timeout = args.timeout or grid.timeout_s
    expanded = grid.expand([args.converter], sha)
    if args.n > len(expanded):
        raise ValueError(f"sample of {args.n} exceeds {len(expanded)} expanded cells")
    picked = sorted(
        random.Random(args.seed).sample(expanded, args.n),
        key=lambda cs: (cs[0].part, cs[0].operator, cs[0].severity, cs[0].seed),
    )
    print(f"grid {grid.name} hash {grid.grid_hash} sha {sha}", flush=True)
    print(f"sampled {len(picked)} of {len(expanded)} cells (seed {args.seed})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "standard-sample.jsonl"
    cache = args.out / "cache"
    work = args.out / "work"
    cache.mkdir(exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    stamp = cache / "STAMP"
    if not stamp.is_file() or stamp.read_text() != prep_hash(grid):
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir()
        stamp.write_text(prep_hash(grid))

    counts = {"ok": 0, "operator": 0, "prep": 0, "converter": 0, "timeout": 0}
    t0 = time.perf_counter()
    done = 0

    def emit(cell, result):
        nonlocal done
        record = append_result(path, cell, result)
        done += 1
        status = result.get("status", "?")
        bucket = status
        if status == "error":
            bucket = stage_of(result)
            if "preparation failed" in str(result.get("error", "")):
                bucket = "prep"
        counts[bucket] = counts.get(bucket, 0) + 1
        wall = float(result.get("wall", 0.0))
        extra = ""
        if status == "error":
            extra = f" {result.get('error', '')[:160]}"
        elif status == "ok":
            phases = result.get("phases", {})
            extra = " " + " ".join(f"{k}={v:.1f}" for k, v in sorted(phases.items()))
        print(
            f"[{done}/{len(picked)}] {cell.part} {cell.operator}@{cell.severity:g} "
            f"seed={cell.seed} {status} {wall:.1f}s{extra}",
            flush=True,
        )
        return record

    _run_cells(grid, sha, timeout, args.jobs or 8, cache, work, picked, print, emit)
    wall_total = time.perf_counter() - t0
    shutil.rmtree(work, ignore_errors=True)

    timed_out = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("status") == "timeout"
    ]
    if timed_out:
        print(f"attributing {len(timed_out)} timeout(s) stage by stage…", flush=True)
        _attribute_timeouts(grid, cache, timeout, timed_out)

    print(f"sample wall {wall_total:.0f}s -> {path}", flush=True)
    bad = sum(v for k, v in counts.items() if k != "ok")
    print(f"counts: {counts}  timeout-or-error rate {bad / len(picked):.1%}", flush=True)
    return 0


class _StageTimeout(Exception):
    pass


def _run_judge(ir_json, tris, truth_in_input, samples_per_mm2):
    from unmesh.ir import Ir
    from unmesh_harness.judge import judge

    return judge(Ir.loads(ir_json), tris, truth_in_input, None, samples_per_mm2=samples_per_mm2)


def _attribute_timeouts(grid, cache, timeout, timed_out) -> None:
    import signal
    import tempfile

    import numpy as np

    from unmesh_harness.degrade import chain, to_original
    from unmesh_harness.labels import LabeledMesh
    from unmesh_harness.runner.converters import get_converter
    from unmesh_harness.runner.execute import _cache_paths
    from unmesh_harness.runner.grid import steps_for

    def handler(signum, frame):
        raise _StageTimeout()

    signal.signal(signal.SIGALRM, handler)

    def timed(stage_timeout, fn, *args):
        signal.alarm(int(stage_timeout))
        start = time.perf_counter()
        try:
            return True, fn(*args), time.perf_counter() - start
        except _StageTimeout:
            return False, None, stage_timeout
        except Exception as e:  # noqa: BLE001
            return True, e, time.perf_counter() - start
        finally:
            signal.alarm(0)

    for record in timed_out:
        part = record["part"]
        spec = grid.row(record["operator"], float(record["severity"]))
        steps = [(name, float(sev)) for name, sev in steps_for(spec)]
        seed = int(record["seed"])
        labeled_path, truth_path = _cache_paths(cache, part)
        clean = LabeledMesh.load(labeled_path)
        ok, degraded, dt = timed(timeout, chain, clean, steps, seed)
        if not ok:
            stage = "degrade"
        elif isinstance(degraded, Exception):
            stage = f"degrade raised {type(degraded).__name__}"
        else:
            import unmesh

            with tempfile.TemporaryDirectory(prefix="attr-") as tmp:
                stl = Path(tmp) / "input.stl"
                degraded.write_stl(stl)
                tris = unmesh.read_stl(stl)
                ok, converted, dtc = timed(timeout, get_converter("unmesh"), stl)
                if not ok:
                    stage = f"convert (degrade {dt:.1f}s)"
                else:
                    ir_json, _step_path, _report_json = converted
                    stage = f"score (degrade {dt:.1f}s, convert {dtc:.1f}s)"
                    if ir_json is not None:
                        frame = to_original(degraded)
                        truth = np.load(truth_path) if truth_path.is_file() else None
                        use_truth = bool(spec.get("judge_truth")) and truth is not None
                        truth_in_input = (
                            ((truth.reshape(-1, 3) - frame[:3, 3]) @ frame[:3, :3]).reshape(
                                truth.shape
                            )
                            if use_truth
                            else None
                        )
                        ok, _res, dtj = timed(
                            timeout,
                            _run_judge,
                            ir_json,
                            tris,
                            truth_in_input,
                            grid.judge_samples_per_mm2,
                        )
                        stage = (
                            f"score (degrade {dt:.1f}s, convert {dtc:.1f}s, judge {dtj:.1f}s)"
                            if ok
                            else f"judge (degrade {dt:.1f}s, convert {dtc:.1f}s)"
                        )
        print(
            f"timeout {part} {record['operator']}@{float(record['severity']):g} "
            f"seed={seed}: slow stage = {stage}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
