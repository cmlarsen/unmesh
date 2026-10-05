from __future__ import annotations

import dataclasses
import json

import pytest

from unmesh_harness.runner import load_grid, run_grid
from unmesh_harness.runner.hidden import aggregate, family_of, hidden_seeds


def tiny_grid():
    grid = load_grid("smoke")
    return dataclasses.replace(
        grid,
        entries=grid.entries[:1],
        cells=[c for c in grid.cells if c["operator"] == "rotation"],
        seeds=[0],
        step_deviation_sample=(1, 1),
    )


def test_hidden_seeds_require_env(monkeypatch):
    monkeypatch.delenv("UNMESH_HIDDEN_SEEDS", raising=False)
    with pytest.raises(ValueError, match="UNMESH_HIDDEN_SEEDS"):
        hidden_seeds()
    monkeypatch.setenv("UNMESH_HIDDEN_SEEDS", "  ")
    with pytest.raises(ValueError, match="UNMESH_HIDDEN_SEEDS"):
        hidden_seeds()
    monkeypatch.setenv("UNMESH_HIDDEN_SEEDS", "11,22,33")
    assert hidden_seeds() == [11, 22, 33]
    monkeypatch.setenv("UNMESH_HIDDEN_SEEDS", "7, notanint")
    with pytest.raises(ValueError, match="UNMESH_HIDDEN_SEEDS"):
        hidden_seeds()


def test_family_of_covers_every_grid_operator():
    for name in ("smoke", "standard", "full"):
        grid = load_grid(name)
        for spec in grid.cells:
            assert family_of(spec["operator"]) != "other", spec["operator"]


def test_aggregate_has_no_per_part_data():
    records = [
        {
            "part": f"part-{i}",
            "operator": "noise_isotropic",
            "severity": 0.5,
            "seed": 100 + i,
            "converter": "unmesh",
            "status": "ok",
            "f1": 0.9 + 0.01 * i,
            "dev_input_p99": 0.001,
            "valid": True,
            "fallback": False,
            "under_report": False,
            "seconds": 1.0,
        }
        for i in range(4)
    ]
    payload = aggregate(records, "abc123")
    assert payload["grid_hash"] == "abc123"
    (row,) = payload["rows"]
    assert row["family"] == "noise" and row["cells"] == 4
    assert set(row["f1"]) == {"n", "mean", "min", "p50", "p90", "p99", "max"}
    assert row["ok_rate"] == 1.0 and row["valid_rate"] == 1.0
    text = json.dumps(payload)
    assert "part-0" not in text and "part-3" not in text and '"seed"' not in text


def test_hidden_run_writes_aggregates_only(tmp_path, monkeypatch):
    monkeypatch.setenv("UNMESH_HIDDEN_SEEDS", "101,102")
    grid = tiny_grid()
    part = grid.entries[0]["id"]
    summary = run_grid(
        grid, ["unmesh"], tmp_path, jobs=1, sha="h", timeout=120, log=lambda *_: None, hidden=True
    )
    assert summary.results_path == tmp_path / "smoke.hidden.json"
    assert summary.ran == 2
    assert list(tmp_path.glob("*.jsonl")) == []
    assert list(tmp_path.glob("cache")) == [] and list(tmp_path.glob("work")) == []
    payload = json.loads(summary.results_path.read_text())
    assert payload["grid_hash"] == grid.grid_hash
    assert payload["rows"] and all("cells" in r for r in payload["rows"])
    found_keys: set[str] = set()
    found_strings: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            found_keys.update(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found_strings.add(node)

    walk(payload)
    assert "part" not in found_keys and "seed" not in found_keys
    assert part not in found_strings


def test_hidden_run_without_env_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("UNMESH_HIDDEN_SEEDS", raising=False)
    with pytest.raises(ValueError, match="UNMESH_HIDDEN_SEEDS"):
        run_grid(
            tiny_grid(),
            ["unmesh"],
            tmp_path,
            jobs=1,
            sha="h",
            timeout=120,
            log=lambda *_: None,
            hidden=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_cli_hidden_without_env_fails_fast(tmp_path, monkeypatch, capsys):
    from unmesh_harness.cli import main

    monkeypatch.delenv("UNMESH_HIDDEN_SEEDS", raising=False)
    rc = main(
        [
            "run",
            "--converter",
            "unmesh",
            "--grid",
            "smoke",
            "--out",
            str(tmp_path),
            "--hidden",
        ]
    )
    assert rc == 2
    assert "UNMESH_HIDDEN_SEEDS" in capsys.readouterr().out


def test_hidden_mode_prints_no_part_ids(tmp_path, monkeypatch, capsys):
    import dataclasses

    monkeypatch.setenv("UNMESH_HIDDEN_SEEDS", "101,102")
    grid = tiny_grid()
    bad_id = "hidden-part-xyz-0000"
    bad_entry = {
        "id": bad_id,
        "tier": "generated",
        "family": "no_such_family",
        "seed": 0,
        "strata": {"category": "planar"},
        "grids": ["smoke"],
        "fingerprint": {"volume": 1.0, "face_count": 1},
    }
    grid = dataclasses.replace(grid, entries=[bad_entry])
    summary = run_grid(
        grid, ["unmesh"], tmp_path, jobs=1, sha="h", timeout=60, log=print, hidden=True
    )
    assert summary.ran == 2
    captured = capsys.readouterr()
    assert bad_id not in captured.out
    assert bad_id not in captured.err
    assert "1 part(s)" in captured.out
    payload = json.loads((tmp_path / "smoke.hidden.json").read_text())
    assert bad_id not in json.dumps(payload)
