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
    assert gate(records).passed

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
        grid, converters, tmp_path / "out", jobs=2, sha="s", timeout=3, log=lambda *_: None
    )
    by_converter = {r["converter"]: r for r in read_results(summary.results_path)}
    assert by_converter["unmesh"]["status"] == "ok"
    assert by_converter["broken_plugins:boom"]["status"] == "error"
    assert "converter exploded" in by_converter["broken_plugins:boom"]["error"]
    assert by_converter["broken_plugins:no_ir"]["status"] == "error"
    assert by_converter["broken_plugins:hang"]["status"] == "timeout"
    assert by_converter["nonexistent_module:fn"]["status"] == "error"
    assert "broken_plugins:boom" in summarize(summary.records)


def record(**kw):
    base = {
        "part": "p",
        "operator": "o",
        "severity": 0.5,
        "seed": 0,
        "converter": "unmesh",
        "git_sha": "s",
        "status": "ok",
        "f1": 1.0,
        "valid": True,
        "under_report": False,
        "calibration": 0.0,
        "step_problems": [],
        "fallback": False,
        "seconds": 0.1,
        "dev_input_max": 0.0,
    }
    return base | kw


def test_gate_rules():
    good = [record(), record(converter="faceted", f1=0.0)]
    assert gate(good).passed
    assert not gate([record(under_report=True, calibration=-0.01), good[1]]).passed
    assert not gate([record(valid=False, step_problems=["bad"]), good[1]]).passed
    assert not gate([record(f1=0.0), good[1]]).passed
    assert not gate([record(status="timeout", error="slow"), good[1]]).passed
    assert not gate([record()]).passed
    assert gate([record(), record(converter="faceted", f1=0.0, valid=None)]).passed


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
