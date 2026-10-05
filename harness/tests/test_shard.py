from __future__ import annotations

import pytest

from unmesh_harness.cli import main, select_run_entries
from unmesh_harness.runner import load_grid


def ids(grid):
    return [e["id"] for e in grid.entries]


def test_shard_selection_is_deterministic():
    grid = load_grid("standard")
    first = ids(select_run_entries(grid, ["complex"], "1/2"))
    second = ids(select_run_entries(grid, ["complex"], "1/2"))
    assert first == second
    assert first == sorted(first)


def test_shards_are_disjoint_and_complete():
    grid = load_grid("standard")
    full = set(ids(grid))
    assert len(full) == len(grid.entries)
    for count in (1, 2, 3, 4, 7):
        shards = [set(ids(select_run_entries(grid, None, f"{i}/{count}"))) for i in range(count)]
        assert set().union(*shards) == full
        assert sum(map(len, shards)) == len(full)


def test_shards_are_disjoint_and_complete_within_category():
    grid = load_grid("standard")
    for category in ("planar", "curved", "chamfer_fillet", "complex", "imported"):
        base = set(ids(select_run_entries(grid, [category], None)))
        assert base
        assert all(
            e["strata"].get("category") == category
            for e in select_run_entries(grid, [category], None).entries
        )
        shards = [
            set(ids(select_run_entries(grid, [category], f"{i}/3"))) for i in range(3)
        ]
        assert set().union(*shards) == base
        assert sum(map(len, shards)) == len(base)


def test_category_errors():
    grid = load_grid("standard")
    with pytest.raises(ValueError, match="selects no parts"):
        select_run_entries(grid, ["no_such_category"], None)
    with pytest.raises(ValueError, match="--shard must look like"):
        select_run_entries(grid, None, "3/3")
    with pytest.raises(ValueError, match="--shard must look like"):
        select_run_entries(grid, None, "x/2")
    with pytest.raises(ValueError, match="--shard must look like"):
        select_run_entries(grid, None, "0/0")
    with pytest.raises(ValueError, match="--shard must look like"):
        select_run_entries(grid, None, "-1/2")


def test_cli_rejects_bad_shard_and_category_without_running(tmp_path, capsys):
    for shard in ("3/3", "x/2", "0/0", "-1/2", "0/2/3"):
        rc = main(
            [
                "run",
                "--converter",
                "unmesh",
                "--grid",
                "smoke",
                "--out",
                str(tmp_path),
                f"--shard={shard}",
            ]
        )
        assert rc == 2, shard
        assert "--shard" in capsys.readouterr().out
    rc = main(
        [
            "run",
            "--converter",
            "unmesh",
            "--grid",
            "smoke",
            "--out",
            str(tmp_path),
            "--category",
            "no_such_category",
        ]
    )
    assert rc == 2
    assert "selects no parts" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def _fake_entries():
    return [
        {"id": f"part-{i:02d}", "strata": {"category": "planar" if i < 2 else "complex"}}
        for i in range(6)
    ]


def test_assign_shard_balances_cost_not_counts():
    from unmesh_harness.runner.shards import assign_shard

    table = {"planar": 100.0, "complex": 1.0}
    shards = [assign_shard(_fake_entries(), i, 2, 10, table) for i in range(2)]
    loads = [
        sum(10 * table[e["strata"]["category"]] for e in shard) for shard in shards
    ]
    assert sorted(loads) == [1020.0, 1020.0]
    got = sorted(e["id"] for shard in shards for e in shard)
    assert got == sorted(e["id"] for e in _fake_entries())


def test_assign_shard_falls_back_without_cost_table():
    from unmesh_harness.runner.shards import assign_shard

    shards = [assign_shard(_fake_entries(), i, 3, 10, {}) for i in range(3)]
    assert [len(s) for s in shards] == [2, 2, 2]
    got = sorted(e["id"] for shard in shards for e in shard)
    assert got == sorted(e["id"] for e in _fake_entries())
