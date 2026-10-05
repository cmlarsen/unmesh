from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RUNNER_WORKERS = 4


def cost_table_path() -> Path:
    from .grid import repo_root

    return repo_root() / "harness" / "grids" / "cost_table.json"


def load_cost_table() -> dict[str, float]:
    try:
        raw = json.loads(cost_table_path().read_text())
        return {k: float(v["worker_s_per_cell"]) for k, v in raw["categories"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def part_cost(entry: dict[str, Any], cells_per_part: int, table: dict[str, float]) -> float:
    return cells_per_part * table.get(entry["strata"].get("category", "?"), 1.0)


def assign_shard(
    entries: list[dict[str, Any]],
    index: int,
    count: int,
    cells_per_part: int,
    table: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda e: e["id"])
    table = table if table is not None else load_cost_table()
    if not table:
        return ordered[index::count]
    costs = {e["id"]: part_cost(e, cells_per_part, table) for e in ordered}
    ranked = sorted(ordered, key=lambda e: (-costs[e["id"]], e["id"]))
    loads = [0.0] * count
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for entry in ranked:
        target = min(range(count), key=lambda i: (loads[i], i))
        buckets[target].append(entry)
        loads[target] += costs[entry["id"]]
    return sorted(buckets[index], key=lambda e: e["id"])


def cells_per_part(grid) -> int:
    total = len(grid.expand(["unmesh"], "shards"))
    return total // len(grid.entries)


def shard_counts(
    grid, workers: int = RUNNER_WORKERS, max_shard_s: float = 3 * 3600
) -> dict[str, int]:
    table = load_cost_table()
    cpp = cells_per_part(grid)
    counts: dict[str, int] = {}
    for category in sorted({e["strata"].get("category", "?") for e in grid.entries}):
        parts = [e for e in grid.entries if e["strata"].get("category") == category]
        total = sum(part_cost(e, cpp, table) or cpp for e in parts)
        counts[category] = max(1, math.ceil(total / (max_shard_s * workers)))
    return counts


def shard_names(grid_name: str, counts: dict[str, int]) -> list[str]:
    return [
        f"{grid_name}-{category}-{i}-of-{n}"
        for category in sorted(counts)
        for i, n in [(k, counts[category]) for k in range(counts[category])]
    ]


def shard_plan(
    grid, workers: int = RUNNER_WORKERS, max_shard_s: float = 3 * 3600
) -> dict[str, Any]:
    counts = shard_counts(grid, workers, max_shard_s)
    return {
        "grid": grid.name,
        "grid_hash": grid.grid_hash,
        "workers": workers,
        "max_shard_s": max_shard_s,
        "counts": counts,
        "shards": shard_names(grid.name, counts),
    }


def projected_shard_costs(grid, plan: dict[str, Any]) -> dict[str, float]:
    table = load_cost_table()
    cpp = cells_per_part(grid)
    out: dict[str, float] = {}
    for stem in plan["shards"]:
        rest = stem[len(grid.name) + 1 :]
        bits = rest.split("-")
        assert bits[-2] == "of", stem
        category, index, count = "-".join(bits[:-3]), int(bits[-3]), int(bits[-1])
        entries = [e for e in grid.entries if e["strata"].get("category") == category]
        picked = assign_shard(entries, index, count, cpp, table)
        per_cell = table.get(category, 1.0)
        out[stem] = len(picked) * cpp * per_cell
    return out
