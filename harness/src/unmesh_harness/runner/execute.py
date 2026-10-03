from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Iterator
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np

_PARTS: dict[str, tuple[Any, np.ndarray]] = {}


def _cache_paths(cache: Path, part: str) -> tuple[Path, Path]:
    return cache / f"{part}.labeled.npz", cache / f"{part}.truth.npy"


def prep_part(task: dict[str, Any]) -> dict[str, Any]:
    from ..groundtruth import generate
    from ..labels import tessellate

    entry = task["entry"]
    labeled_path, truth_path = _cache_paths(Path(task["cache"]), entry["id"])
    gt = generate(entry["family"], entry["seed"])
    labeled = tessellate(gt.solid, *task["input_deflection"])
    if task["need_truth"]:
        truth = tessellate(gt.solid, *task["truth_deflection"])
        np.save(truth_path, truth.tris)
    labeled.save(labeled_path)
    return {"part": entry["id"]}


def _load_part(cache: str, part: str):
    from ..labels import LabeledMesh

    if part not in _PARTS:
        labeled_path, truth_path = _cache_paths(Path(cache), part)
        truth = np.load(truth_path) if truth_path.is_file() else None
        _PARTS[part] = (LabeledMesh.load(labeled_path), truth)
    return _PARTS[part]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _score(
    task: dict[str, Any], degraded, tris, stl_path: Path, phases: dict[str, float]
) -> dict[str, Any]:
    from unmesh.ir import Ir, IrError

    from ..degrade import to_original
    from ..judge import judge, under_reports
    from ..metrics.structure import score_structure, score_topology, score_validity
    from .converters import BASELINES, get_converter
    from .score import face_recovery, load_step_shape, step_problems

    baseline = task["converter"] in BASELINES
    clean, truth_tris = _load_part(task["cache"], task["entry"]["id"])
    started = time.perf_counter()
    ir_json, step_path, report_json = get_converter(task["converter"])(stl_path)
    seconds = time.perf_counter() - started
    record: dict[str, Any] = {
        "seconds": seconds,
        "triangles": len(tris),
        "phases": phases,
        "family": task["entry"]["family"],
        "strata": task["entry"].get("strata", {}),
    }
    phases["convert"] = seconds
    report = json.loads(report_json) if report_json else None
    if report is not None and not isinstance(report, dict):
        raise TypeError("report_json must be a JSON object")
    write = (report or {}).get("write") or {}
    record["writer_valid"] = write.get("valid")
    record["fallback"] = write.get("fallback") is not None
    reported = _finite((report or {}).get("max_deviation"))
    mark = time.perf_counter()
    if ir_json is None:
        record["status"] = "error"
        record["error"] = "converter returned no IR"
        return record
    try:
        ir = Ir.loads(ir_json)
    except (IrError, ValueError, KeyError, TypeError) as e:
        record["status"] = "invalid_ir"
        record["error"] = f"{type(e).__name__}: {e}"
        return record
    frame = to_original(degraded)
    use_truth = task["judge_truth"] and truth_tris is not None
    truth_in_input = None
    if use_truth:
        truth_in_input = ((truth_tris.reshape(-1, 3) - frame[:3, 3]) @ frame[:3, :3]).reshape(
            truth_tris.shape
        )
    if baseline:
        record.update(dict.fromkeys(JUDGE_FIELDS))
    else:
        result = judge(ir, tris, truth_in_input, reported, samples_per_mm2=task["samples_per_mm2"])
        record.update(
            {
                "dev_input_max": result.input.max,
                "dev_input_p99": max(result.input.ir_to_mesh.p99, result.input.mesh_to_ir.p99),
                "dev_truth_max": None if result.truth is None else result.truth.max,
                "reported_deviation": reported,
                "calibration": result.calibration,
                "under_report": True if reported is None else under_reports(reported, result.input),
            }
        )
    phases["judge"] = time.perf_counter() - mark
    mark = time.perf_counter()
    skipped = bool(write.get("skipped")) and baseline
    if skipped:
        record["valid"] = None
        record["step_problems"] = []
    else:
        bound = None
        if not baseline and task["step_deviation"]:
            bound = min(result.input.max, task["dev_input_floor"]) + STEP_TOLERANCE_MM
        shape, load_error = load_step_shape(step_path)
        problems, step_faces = step_problems(
            step_path,
            task["entry"]["fingerprint"]["volume"],
            tris if bound is not None else None,
            bound,
            task["samples_per_mm2"],
            shape=shape,
            load_error=load_error,
        )
        if write.get("skipped"):
            problems.append("STEP write skipped by a non-baseline converter")
        if write.get("valid") is False:
            problems.append("writer reports invalid")
        facets = [r for r in ir.regions if r.surface.type == "facets"]
        if step_faces is not None:
            if facets:
                limit = len(ir.regions) - len(facets) + sum(len(r.triangles) for r in facets)
                mismatch = step_faces > limit
            else:
                mismatch = step_faces != len(ir.regions)
            if mismatch:
                record["fallback"] = True
                problems.append(f"STEP has {step_faces} faces for {len(ir.regions)} IR regions")
        record["valid"] = not problems
        record["step_problems"] = problems
    phases["validity"] = time.perf_counter() - mark
    mark = time.perf_counter()
    record.update(face_recovery(clean, degraded.face_id, ir, frame))
    phases["f1"] = time.perf_counter() - mark
    mark = time.perf_counter()
    record["topology"] = score_topology(clean, degraded.face_id, tris, ir)
    record["structure"] = score_structure(ir, len(clean.faces), tris)
    phases["topology"] = time.perf_counter() - mark
    if skipped:
        record["validity"] = None
    else:
        outer = sum(1 for s in clean.shells if s.role == "outer")
        record["validity"] = score_validity(
            step_path,
            write=write,
            fallback=record["fallback"],
            expected_solids=outer or 1,
            expected_shells=len(clean.shells) or 1,
            shape=shape,
            load_error=load_error,
        )
    record["analytic_area_fraction"] = (report or {}).get("analytic_area_fraction")
    record["warnings"] = (report or {}).get("warnings", [])
    record["status"] = "ok"
    return record


def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    import unmesh

    from ..degrade import apply_chain

    phases: dict[str, float] = {}
    mark = time.perf_counter()
    clean, _ = _load_part(task["cache"], task["entry"]["id"])
    steps = [(name, float(sev)) for name, sev in task["steps"]]
    degraded = apply_chain(clean, steps, task["seed"])
    work = Path(tempfile.mkdtemp(prefix="cell-", dir=task["work"]))
    try:
        stl_path = work / "input.stl"
        degraded.write_stl(stl_path)
        tris = unmesh.read_stl(stl_path)
        phases["degrade"] = time.perf_counter() - mark
        return _score(task, degraded, tris, stl_path, phases)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def execute(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        record = run_prep(task) if task["kind"] == "prep" else run_cell(task)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        tail = traceback.format_exc().strip().splitlines()[-6:]
        record = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": tail}
    record["wall"] = time.perf_counter() - started
    return record


def run_prep(task: dict[str, Any]) -> dict[str, Any]:
    prep_part(task)
    return {"status": "ok"}


STEP_TOLERANCE_MM = 0.006
JUDGE_FIELDS = (
    "dev_input_max",
    "dev_input_p99",
    "dev_truth_max",
    "reported_deviation",
    "calibration",
    "under_report",
)
READY = "__ready__"
SINGLE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _serve(conn) -> None:
    from . import converters, score  # noqa: F401

    conn.send(READY)
    while True:
        task = conn.recv()
        if task is None:
            return
        conn.send(execute(task))


class _Slot:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.proc = None
        self.conn = None
        self.task: tuple[Any, dict[str, Any]] | None = None
        self.started = 0.0
        self.ready = False
        self.start()

    def start(self) -> None:
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_serve, args=(child,), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        self.ready = False

    def kill(self) -> None:
        self.proc.kill()
        self.proc.join()
        self.conn.close()


def run_pool(
    jobs: list[tuple[Any, dict[str, Any]]],
    workers: int,
    timeout: float,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    if not jobs:
        return
    for var in SINGLE_THREAD_ENV:
        os.environ.setdefault(var, "1")
    ctx = mp.get_context("spawn")
    pending = deque(jobs)
    slots = [_Slot(ctx) for _ in range(min(workers, len(jobs)))]
    try:
        while pending or any(s.task for s in slots):
            for slot in slots:
                if slot.ready and slot.task is None and pending:
                    slot.task = pending.popleft()
                    slot.started = time.monotonic()
                    slot.conn.send(slot.task[1])
            live = {s.conn: s for s in slots if s.task or not s.ready}
            for conn in wait(list(live), timeout=0.2):
                slot = live[conn]
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    slot.kill()
                    slot.start()
                    if slot.task:
                        key, _ = slot.task
                        slot.task = None
                        yield key, {"status": "error", "error": "worker process died"}
                    continue
                if message == READY and not slot.ready:
                    slot.ready = True
                    continue
                key, _ = slot.task
                slot.task = None
                yield key, message
            now = time.monotonic()
            for slot in slots:
                if slot.task and now - slot.started > timeout:
                    key, _ = slot.task
                    slot.kill()
                    slot.start()
                    slot.task = None
                    yield key, {"status": "timeout", "error": f"exceeded {timeout:g} s"}
    finally:
        for slot in slots:
            try:
                slot.conn.send(None)
            except (OSError, ValueError):
                pass
            slot.proc.join(timeout=2)
            if slot.proc.is_alive():
                slot.proc.kill()
