import json

import numpy as np
import pytest

from unmesh.ir import Ir
from unmesh_harness.runner import load_grid, run_grid
from unmesh_harness.runner.converters import get_converter, unavailable
from unmesh_harness.runner.external import (
    FREECAD_ENV,
    STL2STEP_ENV,
    ToolUnavailable,
    assign_faces,
)

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
