import dataclasses
import hashlib
import subprocess
import sys

import pytest

from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS, PRESETS, apply, chain
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, tessellate
from unmesh_harness.runner import load_grid, run_grid
from unmesh_harness.runner.grid import steps_for

LIN, ANG = DEFLECTION_SETTINGS[0]
SEED = 20260101
EXPECTED = {
    "fusion-export",
    "tinkercad-export",
    "meshmixer-edit",
    "slicer-repair",
    "inch-roundtrip",
}


@pytest.fixture(scope="module")
def smoke():
    return [
        (e["id"], tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG))
        for e in select(load_manifest(), "smoke")
    ]


def test_expected_presets_are_registered():
    assert set(PRESETS) == EXPECTED
    for preset in PRESETS.values():
        assert preset.rationale
        assert len(preset.steps) >= 1
        for op, severity in preset.steps:
            assert op in OPERATORS
            assert 0.0 <= severity <= 1.0


def test_chain_applies_steps_in_order_with_derived_seeds(smoke):
    _, mesh = smoke[2]
    steps = [("rotation", 0.8), ("float32", 1.0)]
    out = chain(mesh, steps, 5)
    assert [h["op"] for h in out.metadata["history"]] == ["rotation", "float32"]
    assert [h["severity"] for h in out.metadata["history"]] == [0.8, 1.0]
    assert [h["seed"] for h in out.metadata["history"]] == [5, 6]
    expected = apply("float32", apply("rotation", mesh, 0.8, 5), 1.0, 6)
    assert (out.tris == expected.tris).all()
    assert out.metadata == expected.metadata


def test_chain_empty_is_identity(smoke):
    _, mesh = smoke[0]
    out = chain(mesh, [], 5)
    assert (out.tris == mesh.tris).all()
    assert out.metadata == mesh.metadata


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_preset_runs_on_smoke_corpus(name, smoke):
    steps = list(PRESETS[name].steps)
    for part_id, mesh in smoke:
        out = chain(mesh, steps, SEED)
        assert [h["op"] for h in out.metadata["history"]] == [s[0] for s in steps], part_id
        assert [h["seed"] for h in out.metadata["history"]] == list(
            range(SEED, SEED + len(steps))
        ), part_id
        assert len(out.face_id) == len(out.tris), part_id
        assert set(out.face_id.tolist()) <= set(mesh.face_id.tolist()) | {-1}, part_id
        assert [f.id for f in out.faces] == [f.id for f in mesh.faces], part_id
        assert [(a.face_a, a.face_b, a.edge_id) for a in out.adjacency] == [
            (a.face_a, a.face_b, a.edge_id) for a in mesh.adjacency
        ], part_id


def test_presets_deterministic_across_processes(smoke, tmp_path):
    src = tmp_path / "mesh.npz"
    smoke[3][1].save(src)
    script = (
        "import sys, hashlib\n"
        "from unmesh_harness.degrade import PRESETS, chain\n"
        "from unmesh_harness.labels import LabeledMesh\n"
        "m = LabeledMesh.load(sys.argv[1])\n"
        "h = hashlib.sha256()\n"
        "for name in sorted(PRESETS):\n"
        "    o = chain(m, list(PRESETS[name].steps), 99)\n"
        "    h.update(o.tris.tobytes())\n"
        "    h.update(repr(o.metadata).encode())\n"
        "print(h.hexdigest())\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script, str(src)], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(digests) == 1
    local = hashlib.sha256()
    loaded = LabeledMesh.load(src)
    for name in sorted(PRESETS):
        out = chain(loaded, list(PRESETS[name].steps), 99)
        local.update(out.tris.tobytes())
        local.update(repr(out.metadata).encode())
    assert digests == {local.hexdigest() + "\n"}


def test_preset_cell_expands_to_preset_steps():
    for name, preset in PRESETS.items():
        spec = {"operator": name, "severity": 1.0, "preset": name}
        assert steps_for(spec) == [[op, severity] for op, severity in preset.steps]


def test_preset_cell_rejects_steps_and_unknown_presets():
    with pytest.raises(ValueError, match="either 'preset' or 'steps'"):
        steps_for({"operator": "x", "severity": 1.0, "preset": "fusion-export", "steps": []})
    with pytest.raises(ValueError, match="unknown degradation preset"):
        steps_for({"operator": "x", "severity": 1.0, "preset": "no-such-preset"})


def test_run_grid_resolves_preset_cell(tmp_path):
    grid = load_grid("smoke")
    grid = dataclasses.replace(
        grid,
        entries=grid.entries[:1],
        cells=[{"operator": "fusion-export", "severity": 1.0, "preset": "fusion-export"}],
        seeds=[0],
        step_deviation_sample=(1, 1),
    )
    summary = run_grid(grid, ["faceted"], tmp_path, jobs=1, sha="presets", log=lambda *_: None)
    assert summary.ran == 1
    (record,) = summary.records
    assert record["status"] == "ok", record


def test_chain_matches_degrade_apply_pair(smoke):
    _, mesh = smoke[0]
    assert degrade.chain is chain
