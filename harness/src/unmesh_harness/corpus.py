from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from build123d import export_step

from .groundtruth import families, generate

MANIFEST_VERSION = 0
GRID_SIZES = {"smoke": 20, "standard": 100}
GRID_ORDER = ["smoke", "standard"]


def entry_id(family: str, seed: int) -> str:
    return f"{family}-{seed:04d}"


def planar_entries(count: int = GRID_SIZES["standard"]) -> list[dict[str, Any]]:
    names = [f for f in families() if f in PLANAR_FAMILIES]
    entries = []
    for i in range(count):
        family, seed = names[i % len(names)], i // len(names)
        grids = [g for g in GRID_ORDER if i < GRID_SIZES[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"complexity": "planar"},
                "grids": grids,
            }
        )
    return entries


PLANAR_FAMILIES = {
    "plate_pockets",
    "rotated_pockets",
    "square_slots",
    "stepped_block",
    "boss_plate",
    "through_cuts",
    "polygon_prism",
    "lshape_outline",
    "thin_walls",
}


def default_manifest() -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "grids": {
            "smoke": "Fast PR gate, subset of standard.",
            "standard": "Nightly corpus.",
        },
        "entries": planar_entries(),
    }


def find_manifest() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "corpus" / "v0.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("corpus/v0.json not found; pass --manifest")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or find_manifest()).read_text())


def select(manifest: dict[str, Any], grid: str) -> list[dict[str, Any]]:
    if grid not in manifest["grids"]:
        raise KeyError(f"unknown grid: {grid}")
    return [e for e in manifest["entries"] if grid in e["grids"]]


def default_cache_dir() -> Path:
    override = os.environ.get("UNMESH_CACHE_DIR")
    if override:
        return Path(override) / "corpus"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "unmesh" / "corpus"


def build_entry(entry: dict[str, Any], out: Path):
    if entry["tier"] != "generated":
        raise NotImplementedError(f"tier {entry['tier']!r} is not buildable yet")
    gt = generate(entry["family"], entry["seed"])
    out.mkdir(parents=True, exist_ok=True)
    export_step(gt.solid, str(out / f"{entry['id']}.step"))
    meta = gt.metadata()
    meta["id"] = entry["id"]
    meta["tier"] = entry["tier"]
    meta["strata"] = entry.get("strata", {})
    (out / f"{entry['id']}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return gt


def build_grid(manifest: dict[str, Any], grid: str, out: Path) -> int:
    entries = select(manifest, grid)
    for entry in entries:
        build_entry(entry, out)
    return len(entries)
