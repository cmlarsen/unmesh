from __future__ import annotations

import json
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
    truth = tessellate(gt.solid, *task["truth_deflection"])
    labeled.save(labeled_path)
    np.save(truth_path, truth.tris)
    return {"part": entry["id"]}


def _load_part(cache: str, part: str):
    from ..labels import LabeledMesh

    if part not in _PARTS:
        labeled_path, truth_path = _cache_paths(Path(cache), part)
        _PARTS[part] = (LabeledMesh.load(labeled_path), np.load(truth_path))
    return _PARTS[part]


def _score(
    task: dict[str, Any], degraded, tris, stl_path: Path, phases: dict[str, float]
) -> dict[str, Any]:
    from unmesh.ir import Ir

    from ..degrade import to_original
    from ..judge import judge, under_reports
    from .converters import get_converter
    from .score import face_recovery, step_problems

    clean, truth_tris = _load_part(task["cache"], task["entry"]["id"])
    started = time.perf_counter()
    ir_json, step_path, report_json = get_converter(task["converter"])(stl_path)
    seconds = time.perf_counter() - started
    record: dict[str, Any] = {"seconds": seconds, "triangles": len(tris), "phases": phases}
    report = json.loads(report_json) if report_json else None
    write = (report or {}).get("write") or {}
    record["writer_valid"] = write.get("valid")
    record["fallback"] = write.get("fallback") is not None
    phases["convert"] = seconds
    mark = time.perf_counter()
    if write.get("skipped"):
        record["valid"] = None
        record["step_problems"] = []
    else:
        problems = step_problems(step_path, task["entry"]["fingerprint"]["volume"])
        record["valid"] = not problems and write.get("valid", True) is not False
        record["step_problems"] = problems
    phases["validity"] = time.perf_counter() - mark
    mark = time.perf_counter()
    if ir_json is None:
        record["status"] = "error"
        record["error"] = "converter returned no IR"
        return record
    ir = Ir.loads(ir_json)
    frame = to_original(degraded)
    truth_in_input = (truth_tris.reshape(-1, 3) - frame[:3, 3]) @ frame[:3, :3]
    reported = None if report is None else report.get("max_deviation")
    result = judge(ir, tris, truth_in_input.reshape(truth_tris.shape), reported)
    record.update(
        {
            "dev_input_max": result.input.max,
            "dev_input_p99": max(result.input.ir_to_mesh.p99, result.input.mesh_to_ir.p99),
            "dev_truth_max": None if result.truth is None else result.truth.max,
            "reported_deviation": reported,
            "calibration": result.calibration,
            "under_report": None if reported is None else under_reports(reported, result.input),
        }
    )
    phases["judge"] = time.perf_counter() - mark
    mark = time.perf_counter()
    record.update(face_recovery(clean, degraded.face_id, ir, frame))
    phases["f1"] = time.perf_counter() - mark
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
