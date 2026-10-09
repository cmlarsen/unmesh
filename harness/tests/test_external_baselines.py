import json
import os
import sys
import types

import numpy as np
import psutil
import pytest

from unmesh.ir import Ir
from unmesh_harness.runner import external, load_grid, run_grid
from unmesh_harness.runner.converters import get_converter, unavailable
from unmesh_harness.runner.external import (
    FREECAD_ENV,
    STL2STEP_ENV,
    ToolUnavailable,
    assign_faces,
)
from unmesh_harness.runner.results import read_results

TOOLS = {"freecad-refine": FREECAD_ENV, "stl2step": STL2STEP_ENV}


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_unset_tool_is_unavailable(name, monkeypatch, tmp_path):
    monkeypatch.delenv(TOOLS[name], raising=False)
    assert TOOLS[name] in unavailable(name)
    with pytest.raises(ToolUnavailable, match=TOOLS[name]):
        get_converter(name)(tmp_path / "input.stl")


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_missing_executable_is_unavailable(name, monkeypatch, tmp_path):
    monkeypatch.setenv(TOOLS[name], str(tmp_path / "nope"))
    assert "not an executable" in unavailable(name)


def test_builtin_converters_are_always_available():
    assert unavailable("unmesh") is None
    assert unavailable("faceted") is None


def test_run_skips_absent_tools(monkeypatch, tmp_path):
    for env in TOOLS.values():
        monkeypatch.delenv(env, raising=False)
    lines = []
    with pytest.raises(ValueError, match="no available converter"):
        run_grid(load_grid("smoke"), sorted(TOOLS), tmp_path, jobs=1, log=lines.append)
    assert len(lines) == 2
    assert all(line.startswith("skipping converter") for line in lines)


def _square(z: float, flip: bool) -> np.ndarray:
    a, b, c, d = [(0, 0, z), (4, 0, z), (4, 4, z), (0, 4, z)]
    tris = np.array([[a, b, c], [a, c, d]], dtype=np.float64)
    return tris[:, ::-1] if flip else tris


def test_assign_faces_follows_the_nearest_face():
    step = np.concatenate([_square(0.0, True), _square(0.5, False)])
    owner = np.array([0, 0, 1, 1])
    fine = []
    for z, flip in ((0.0, True), (0.5, False)):
        for i in range(4):
            for j in range(4):
                a, b, c, d = (i, j, z), (i + 1, j, z), (i + 1, j + 1, z), (i, j + 1, z)
                quad = np.array([[a, b, c], [a, c, d]], dtype=np.float64)
                fine.append(quad[:, ::-1] if flip else quad)
    tris = np.concatenate(fine)
    got = assign_faces(tris, step, owner)
    assert got.tolist() == [0] * 32 + [1] * 32


def _triangle_solid_step(tris, path):
    from build123d import Solid, export_step
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt
    from OCP.ShapeFix import ShapeFix_Shell
    from OCP.TopoDS import TopoDS

    sewing = BRepBuilderAPI_Sewing(1e-6)
    for tri in np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3):
        wire = BRepBuilderAPI_MakePolygon()
        for point in tri:
            wire.Add(gp_Pnt(float(point[0]), float(point[1]), float(point[2])))
        wire.Close()
        sewing.Add(BRepBuilderAPI_MakeFace(wire.Wire()).Face())
    sewing.Perform()
    fixer = ShapeFix_Shell(TopoDS.Shell_s(sewing.SewedShape()))
    fixer.Perform()
    solid = TopoDS.Solid_s(BRepBuilderAPI_MakeSolid(fixer.Shell()).Solid())
    export_step(Solid(solid), str(path))


def test_written_faces_are_classified_by_their_source_triangles(tmp_path):
    from build123d import Box, Polygon, export_step, extrude

    from unmesh_harness.labels import tessellate
    from unmesh_harness.runner.execute import region_kinds

    box = Box(20, 20, 10)
    tris = tessellate(box, 0.01, 0.2).tris
    mesh_step = tmp_path / "mesh.step"
    _triangle_solid_step(tris, mesh_step)
    solid_step = tmp_path / "solid.step"
    export_step(box, str(solid_step))
    prism = extrude(Polygon((0, 0), (20, 0), (0, 10), align=None), amount=10)
    prism_step = tmp_path / "prism.step"
    export_step(prism, str(prism_step))

    assert region_kinds(external.step_ir(mesh_step, tris)) == (12, 0)
    assert region_kinds(external.step_ir(solid_step, tris)) == (0, 6)
    prism_tris = tessellate(prism, 0.01, 0.2).tris
    assert region_kinds(external.step_ir(prism_step, prism_tris)) == (0, 5)


def _bored_block_stl(path):
    from build123d import Box, Cylinder

    from unmesh_harness.labels import tessellate

    tessellate(Box(20, 20, 10) - Cylinder(4, 10), 0.05, 0.3).write_stl(path)


@pytest.mark.external
@pytest.mark.parametrize("name", sorted(TOOLS))
def test_tool_smoke(name, tmp_path):
    reason = unavailable(name)
    if reason is not None:
        pytest.skip(reason)
    stl = tmp_path / "input.stl"
    _bored_block_stl(stl)
    ir_json, step_path, report_json = get_converter(name)(stl)
    ir = Ir.loads(ir_json)
    report = json.loads(report_json)
    assert step_path is not None
    assert report["tool"]["name"] in ("freecad", "stl2step")
    kinds = {r.surface.type for r in ir.regions}
    assert "plane" in kinds
    assert sum(len(r.triangles) for r in ir.regions) == ir.source.triangle_count
    if name == "stl2step":
        assert "cylinder" in kinds


def _one_part_grid():
    import dataclasses

    grid = load_grid("smoke")
    return dataclasses.replace(
        grid,
        entries=grid.entries[:1],
        cells=[c for c in grid.cells if c["operator"] == "identity"],
        seeds=[0],
        step_deviation_sample=(1, 1),
    )


def _script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return path


def test_external_tool_timeout_is_recorded_and_killed(tmp_path, monkeypatch):
    pid_file = tmp_path / "tool.pid"
    tool = _script(tmp_path, "slow-tool", f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 600\n")
    monkeypatch.setenv(FREECAD_ENV, str(tool))
    monkeypatch.delenv(external.MEMORY_ENV, raising=False)
    summary = run_grid(
        _one_part_grid(),
        ["freecad-refine"],
        tmp_path / "out",
        jobs=1,
        sha="s",
        timeout=6,
        log=lambda *_: None,
    )
    record = read_results(summary.results_path)[0]
    assert record["status"] == "error", record
    assert "exceeded" in record["error"]
    assert not psutil.pid_exists(int(pid_file.read_text().strip()))


def test_external_tool_memory_cap_kills_and_raises(tmp_path, monkeypatch):
    pid_file = tmp_path / "tool.pid"
    alloc = "import time; held = bytearray(500 * 1024 * 1024); time.sleep(600)"
    tool = _script(
        tmp_path,
        "hungry-tool",
        f'#!/bin/sh\necho $$ > {pid_file}\nexec "{sys.executable}" -c "{alloc}"\n',
    )
    monkeypatch.setenv(external.MEMORY_ENV, "256")
    monkeypatch.setenv(external.TIMEOUT_ENV, "30")
    with pytest.raises(RuntimeError, match="memory cap"):
        external._run([str(tool)])
    assert not psutil.pid_exists(int(pid_file.read_text().strip()))


KILLER = """import os
import signal


def kill(stl_path):
    os.kill(os.getpid(), signal.SIGKILL)
"""


def test_converter_that_kills_its_worker_is_recorded(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "killer.py").write_text(KILLER)
    monkeypatch.syspath_prepend(str(plugins))
    summary = run_grid(
        _one_part_grid(),
        ["killer:kill", "unmesh"],
        tmp_path / "out",
        jobs=1,
        sha="s",
        timeout=30,
        log=lambda *_: None,
    )
    by_converter = {r["converter"]: r for r in read_results(summary.results_path)}
    assert by_converter["killer:kill"]["status"] == "error"
    assert "worker process died" in by_converter["killer:kill"]["error"]
    assert by_converter["unmesh"]["status"] == "ok"


SPAWNER = """import subprocess
import sys
import time


def spawn(stl_path):
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, time; "
            "open(os.environ['UNMESH_TEST_PIDFILE'], 'w').write(str(os.getpid())); "
            "time.sleep(600)",
        ]
    )
    time.sleep(600)
"""


def test_worker_kill_reaps_a_converters_children(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "spawner.py").write_text(SPAWNER)
    monkeypatch.syspath_prepend(str(plugins))
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("UNMESH_TEST_PIDFILE", str(pid_file))
    summary = run_grid(
        _one_part_grid(),
        ["spawner:spawn"],
        tmp_path / "out",
        jobs=1,
        sha="s",
        timeout=6,
        log=lambda *_: None,
    )
    record = read_results(summary.results_path)[0]
    assert record["status"] in ("error", "timeout"), record
    assert pid_file.is_file()
    assert not psutil.pid_exists(int(pid_file.read_text().strip()))


class _DeadSlot:
    def __init__(self, ctx):
        self.ready = False
        self.task = None
        self.peak = 0.0
        self.started = 0.0
        self.conn, child = ctx.Pipe()
        child.close()
        self.proc = types.SimpleNamespace(
            pid=os.getpid(), join=lambda timeout=None: None, is_alive=lambda: False
        )

    def start(self):
        pass

    def kill(self):
        pass

    def sample(self):
        return 0.0


def test_pool_fails_fast_when_workers_die_before_ready(monkeypatch):
    from unmesh_harness.runner import execute

    monkeypatch.setattr(execute, "_Slot", _DeadSlot)
    with pytest.raises(RuntimeError, match="before they announced readiness"):
        list(execute.run_pool([("k", {})], 1, 1.0, None))
