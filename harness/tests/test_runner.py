from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir
from unmesh_harness.runner import gate, load_grid, run_grid, summarize
from unmesh_harness.runner.converters import faceted_ir, get_converter
from unmesh_harness.runner.results import KEY_FIELDS, read_results, record_key
from unmesh_harness.runner.score import face_recovery

PLUGINS = """
import time


def boom(stl_path):
    raise RuntimeError("converter exploded")


def no_ir(stl_path):
    return None, None, None


def hang(stl_path):
    time.sleep(600)
"""


def small_grid(parts=2, cells=("identity", "float32")):
    grid = load_grid("smoke")
    return dataclasses.replace(
        grid,
        entries=grid.entries[:parts],
        cells=[c for c in grid.cells if c["operator"] in cells],
        seeds=[0],
    )


def test_smoke_grid_definition():
    grid = load_grid("smoke")
    assert len(grid.entries) == 20
    assert grid.seeds == [0, 1, 2]
    operators = {c["operator"] for c in grid.cells}
    assert {"identity", "float32", "rotation", "refine+noise_off_plane"} <= operators
    for cell in grid.cells:
        assert all(
            name in __import__("unmesh_harness.degrade").degrade.OPERATORS
            for name, _ in cell["steps"]
        )


def test_run_keys_and_resume(tmp_path):
    grid = small_grid()
    summary = run_grid(
        grid, ["unmesh", "faceted"], tmp_path, jobs=2, sha="testsha", log=lambda *_: None
    )
    records = read_results(summary.results_path)
    assert summary.ran == len(records) == 2 * 2 * 1 * 2
    assert all(r["status"] == "ok" for r in records)
    keys = [record_key(r) for r in records]
    assert len(set(keys)) == len(keys)
    assert all(set(KEY_FIELDS) <= set(r) and r["git_sha"] == "testsha" for r in records)
    assert {r["converter"] for r in records} == {"unmesh", "faceted"}
    assert gate(records, grid, ["unmesh", "faceted"], "testsha").passed

    again = run_grid(
        grid, ["unmesh", "faceted"], tmp_path, jobs=2, sha="testsha", log=lambda *_: None
    )
    assert (again.ran, again.skipped) == (0, 8)
    assert len(read_results(summary.results_path)) == 8

    lines = summary.results_path.read_text().splitlines()
    summary.results_path.write_text("\n".join(lines[:-1]) + "\n")
    resumed = run_grid(
        grid, ["unmesh", "faceted"], tmp_path, jobs=2, sha="testsha", log=lambda *_: None
    )
    assert (resumed.ran, resumed.skipped) == (1, 7)
    assert len(read_results(summary.results_path)) == 8


def test_broken_converters_are_reported_not_fatal(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "broken_plugins.py").write_text(PLUGINS)
    monkeypatch.syspath_prepend(str(plugins))
    grid = small_grid(parts=1, cells=("identity",))
    converters = [
        "broken_plugins:boom",
        "broken_plugins:no_ir",
        "broken_plugins:hang",
        "nonexistent_module:fn",
        "unmesh",
    ]
    summary = run_grid(
        grid, converters, tmp_path / "out", jobs=2, sha="s", timeout=6, log=lambda *_: None
    )
    by_converter = {r["converter"]: r for r in read_results(summary.results_path)}
    assert by_converter["unmesh"]["status"] == "ok", by_converter["unmesh"]
    assert by_converter["broken_plugins:boom"]["status"] == "error"
    assert "converter exploded" in by_converter["broken_plugins:boom"]["error"]
    assert by_converter["broken_plugins:no_ir"]["status"] == "error"
    assert by_converter["broken_plugins:hang"]["status"] == "timeout"
    assert by_converter["nonexistent_module:fn"]["status"] == "error"
    assert "broken_plugins:boom" in summarize(summary.records)


ADVERSARIES = """
import json
from pathlib import Path

import numpy as np

from unmesh.ir import Ir, Plane
from unmesh_harness.runner.converters import convert_unmesh


def _report(report, **changes):
    r = json.loads(report)
    r.update(changes)
    return json.dumps(r)


def halved(p):
    ir, step, rep = convert_unmesh(p)
    return ir, step, _report(rep, max_deviation=json.loads(rep)["max_deviation"] / 2)


def zero(p):
    ir, step, rep = convert_unmesh(p)
    return ir, step, _report(rep, max_deviation=0.0)


def nan_report(p):
    ir, step, rep = convert_unmesh(p)
    return ir, step, _report(rep, max_deviation=float("nan"))


def no_report(p):
    ir, step, rep = convert_unmesh(p)
    return ir, step, None


def skip_step(p):
    ir, step, rep = convert_unmesh(p)
    return ir, None, _report(rep, write={"skipped": True})


def empty_step(p):
    ir, step, rep = convert_unmesh(p)
    Path(step).write_text("")
    return ir, step, rep


def other_step(p):
    ir, step, rep = convert_unmesh(p)
    return ir, step, _report(rep, write={"valid": True})


def shifted(p):
    ir, step, rep = convert_unmesh(p)
    parsed = Ir.loads(ir)
    for region in parsed.regions:
        s = region.surface
        if isinstance(s, Plane):
            n = np.asarray(s.normal)
            s.origin = tuple((np.asarray(s.origin) + 0.05 * n).tolist())
    return parsed.dumps(), step, _report(rep, max_deviation=json.loads(rep)["max_deviation"] + 0.06)


def moved_step(p):
    from build123d import Pos, export_step, import_step

    ir, step, rep = convert_unmesh(p)
    export_step(Pos(0.05, 0, 0) * import_step(step), step)
    return ir, step, rep


def bad_ir(p):
    ir, step, rep = convert_unmesh(p)
    d = json.loads(ir)
    d["regions"][0]["triangles"].append(10**9)
    return json.dumps(d), step, rep
"""

ADVERSARY_NAMES = [
    "halved",
    "zero",
    "nan_report",
    "no_report",
    "skip_step",
    "empty_step",
    "shifted",
    "moved_step",
    "bad_ir",
]


def test_adversarial_plugins_fail_the_gate(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "adversaries.py").write_text(ADVERSARIES)
    monkeypatch.syspath_prepend(str(plugins))
    grid = small_grid(parts=1, cells=("identity",))
    converters = ["unmesh"] + [f"adversaries:{n}" for n in ADVERSARY_NAMES]
    summary = run_grid(
        grid, converters, tmp_path / "out", jobs=3, sha="s", timeout=60, log=lambda *_: None
    )
    verdict = gate(summary.records, grid, converters, "s")
    flagged = {v.split(" ")[0] for v in verdict.violations}
    assert "unmesh" not in flagged
    assert flagged == {f"adversaries:{n}" for n in ADVERSARY_NAMES}, verdict.violations


def row_records(grid, **changes):
    out = []
    for cell, _ in grid.expand(["unmesh"], "s"):
        r = {
            "part": cell.part,
            "operator": cell.operator,
            "severity": cell.severity,
            "seed": cell.seed,
            "converter": "unmesh",
            "git_sha": "s",
            "grid_hash": cell.grid_hash,
            "status": "ok",
            "f1": 1.0,
            "faces": 10,
            "regions": 10,
            "matched": 10,
            "valid": True,
            "under_report": False,
            "calibration": 0.0,
            "step_problems": [],
            "fallback": False,
            "dev_input_max": 0.0,
            "dev_truth_max": 0.0,
        }
        out.append(r | changes)
    return out


def test_gate_rules():
    grid = small_grid(parts=2, cells=("identity",))
    ok = row_records(grid)
    assert gate(ok, grid, ["unmesh"], "s").passed
    for changes in (
        {"under_report": True},
        {"under_report": None},
        {"valid": False, "step_problems": ["bad"]},
        {"valid": None},
        {"f1": 0.99},
        {"regions": 11},
        {"fallback": True},
        {"dev_input_max": 0.002},
        {"dev_input_max": float("nan")},
        {"status": "timeout", "error": "slow"},
        {"status": "invalid_ir", "error": "bad"},
    ):
        assert not gate(row_records(grid, **changes), grid, ["unmesh"], "s").passed, changes
    assert not gate(ok[:-1], grid, ["unmesh"], "s").passed
    assert not gate([], grid, ["faceted"], "s").passed


def test_gate_row_floors_and_truth():
    grid = load_grid("smoke")
    grid = dataclasses.replace(grid, entries=grid.entries[:2], seeds=[0])
    noisy = [c for c in grid.cells if c["operator"] == "noise_isotropic"][0]
    grid = dataclasses.replace(grid, cells=[noisy])
    assert noisy["judge_truth"] and noisy["floors"]["dev_truth_max"] > 0
    assert gate(row_records(grid), grid, ["unmesh"], "s").passed
    assert not gate(row_records(grid, dev_truth_max=None), grid, ["unmesh"], "s").passed
    assert not gate(row_records(grid, dev_truth_max=0.05), grid, ["unmesh"], "s").passed
    weak = row_records(grid, f1=0.95, matched=9, regions=10, faces=10)
    assert not gate(weak, grid, ["unmesh"], "s").passed


def test_under_reports_rejects_non_finite():
    from unmesh_harness.judge import Comparison, Stats, under_reports

    stats = Stats(1, 0.001, 0.001, 0.001, 0.001)
    cmp = Comparison(stats, stats)
    assert under_reports(float("nan"), cmp)
    assert under_reports(float("inf"), cmp)
    assert not under_reports(0.002, cmp)


@pytest.fixture(scope="module")
def part():
    entry = select(load_manifest(), "smoke")[0]
    return tessellate(generate(entry["family"], entry["seed"]).solid, 0.01, 0.2)


def test_face_recovery_oracle_and_faceted(part):
    eye = np.eye(4)
    oracle = face_recovery(part, part.face_id, build_oracle_ir(part), eye)
    assert oracle["f1"] == 1.0 and oracle["matched"] == len(part.faces)
    facets = face_recovery(part, part.face_id, faceted_ir(part.tris), eye)
    assert facets["f1"] == 0.0


def test_face_recovery_rejects_offset_plane(part):
    ir = build_oracle_ir(part)
    plane = ir.regions[0].surface
    n = np.array(plane.normal)
    ir.regions[0].surface = dataclasses.replace(
        plane, origin=tuple(np.array(plane.origin) + 0.05 * n)
    )
    result = face_recovery(part, part.face_id, ir, np.eye(4))
    assert result["matched"] == len(part.faces) - 1


def test_faceted_plugin_writes_valid_step(part, tmp_path):
    from unmesh import write_stl

    write_stl(tmp_path / "a.stl", part.tris)
    ir_json, step_path, report = get_converter("faceted")(tmp_path / "a.stl")
    assert ir_json and step_path
    assert json.loads(report)["write"]["valid"]
