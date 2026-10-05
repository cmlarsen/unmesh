from __future__ import annotations

import json
import re

from unmesh_harness.degrade import OPERATORS
from unmesh_harness.runner import load_grid
from unmesh_harness.runner.shards import (
    cells_per_part,
    projected_shard_costs,
    shard_plan,
)

SEED_INDEPENDENT = frozenset(
    {
        "coarsen",
        "fillet_rows",
        "float32",
        "inch_round_trip",
        "refine",
        "truncated_digits",
        "fusion-export",
        "tinkercad-export",
        "inch-roundtrip",
    }
)

EXTRA_SEVERITY_FAMILIES = {"noise", "precision", "tessellation"}


def _single_ops(grid):
    return sorted({c["operator"] for c in grid.cells if c.get("steps")})


def _op_family(op):
    bits = op.split("+")
    assert all(bit in OPERATORS for bit in bits), op
    return OPERATORS[bits[-1]].family


def test_standard_grid_shape():
    grid = load_grid("standard")
    assert grid.timeout_s == 60
    assert grid.seeds == [0, 1, 2]
    assert len(grid.entries) == 100
    by_cat = {}
    for e in grid.entries:
        by_cat.setdefault(e["strata"].get("category"), []).append(e["id"])
    assert sorted(by_cat) == [
        "chamfer_fillet",
        "complex",
        "curved",
        "imported",
        "planar",
    ]
    assert all(len(v) == 20 for v in by_cat.values())
    ops = _single_ops(grid)
    assert len(ops) == 24
    by_op: dict[str, list[float]] = {}
    for c in grid.cells:
        if c.get("steps"):
            by_op.setdefault(c["operator"], []).append(float(c["severity"]))
    for op in ops:
        assert 0.5 in by_op[op], op
        if _op_family(op) in EXTRA_SEVERITY_FAMILIES and op != "float32":
            assert by_op[op] == [0.5, 1.0], (op, by_op[op])
        else:
            assert by_op[op] == [0.5], (op, by_op[op])
    presets = [c["preset"] for c in grid.cells if "preset" in c]
    assert sorted(presets) == [
        "fusion-export",
        "inch-roundtrip",
        "meshmixer-edit",
        "slicer-repair",
        "tinkercad-export",
    ]
    assert cells_per_part(grid) == 99
    assert len(grid.expand(["unmesh"], "s")) == 9900


def test_seed_rule_matches_detection():
    for name in ("standard", "full"):
        grid = load_grid(name)
        for spec in grid.cells:
            op = spec["operator"] if "steps" in spec else spec["preset"]
            if op == "identity" or op in SEED_INDEPENDENT:
                assert spec.get("seeds", grid.seeds) == [0], (name, op)
            else:
                assert "seeds" not in spec, (name, op)


def test_full_grid_shape():
    grid = load_grid("full")
    assert grid.timeout_s == 120
    assert len(grid.entries) == 589
    assert cells_per_part(grid) == 190
    assert len(grid.expand(["unmesh"], "s")) == 111910
    by_op: dict[str, list[float]] = {}
    for c in grid.cells:
        if c.get("steps"):
            by_op.setdefault(c["operator"], []).append(float(c["severity"]))
    for op, sevs in by_op.items():
        assert sorted(sevs) == [0.25, 0.5, 1.0], (op, sevs)


def test_smoke_curved_grid_shape():
    grid = load_grid("smoke_curved")
    assert grid.name == "smoke_curved"
    assert grid.corpus_grid == "smoke"
    assert grid.seeds == [0, 1, 2]
    assert sorted(e["id"] for e in grid.entries) == [
        "chamfer_same_chord-0000",
        "counterbore-0000",
        "countersink-0000",
        "revolved_cone-0000",
        "revolved_torus-0000",
        "round_boss-0000",
    ]
    assert [(c["operator"], float(c["severity"])) for c in grid.cells] == [
        ("identity", 0.0),
        ("float32", 1.0),
    ]
    for spec in grid.cells:
        assert spec.get("seeds", grid.seeds) == [0]
        assert spec["floors"]["f1_cell"] == 1.0
    assert len(grid.expand(["unmesh"], "s")) == 12


def _plan(name):
    from unmesh_harness.runner.grid import repo_root

    return json.loads((repo_root() / "harness" / "grids" / f"shards_{name}.json").read_text())


def test_shard_plans_match_cost_model():
    for name, cap in (("standard", 5400), ("full", 9000)):
        grid = load_grid(name)
        plan = _plan(name)
        assert plan["grid"] == name
        assert plan["grid_hash"] == grid.grid_hash
        assert plan["workers"] == 4 and plan["max_shard_s"] == cap
        assert plan == shard_plan(grid, 4, cap)


def test_nightly_matrix_matches_standard_plan():
    from unmesh_harness.runner.grid import repo_root

    text = (repo_root() / ".github" / "workflows" / "nightly.yml").read_text()
    triples = re.findall(r"-\s+category:\s+(\S+)\s*\n\s+index:\s+(\d+)\s*\n\s+count:\s+(\d+)", text)
    assert triples, "no matrix entries parsed from nightly.yml"
    assert len(triples) == len(set(triples))
    names = [f"standard-{c}-{i}-of-{n}" for c, i, n in triples]
    assert sorted(names) == sorted(_plan("standard")["shards"])


def _hours(worker_s: float, workers: int = 4) -> float:
    return worker_s / workers / 3600


def test_projected_shard_times_within_limits():
    grid = load_grid("standard")
    plan = _plan("standard")
    costs = projected_shard_costs(grid, plan)
    assert sorted(costs) == sorted(plan["shards"])
    worst = max(costs.values())
    total = sum(costs.values())
    assert _hours(worst) <= 3, {k: round(_hours(v), 2) for k, v in costs.items()}
    assert _hours(total) <= 12, round(_hours(total), 2)

    grid = load_grid("full")
    plan = _plan("full")
    costs = projected_shard_costs(grid, plan)
    worst = max(costs.values())
    assert _hours(worst) <= 5, max(costs.items())
