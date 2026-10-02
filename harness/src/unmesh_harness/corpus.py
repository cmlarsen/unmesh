from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from build123d import export_step

from .groundtruth import fingerprint, generate

MANIFEST_VERSION = 0
GRID_SIZES = {"smoke": 20, "standard": 100}
GRID_ORDER = ["smoke", "standard"]


def entry_id(family: str, seed: int) -> str:
    return f"{family}-{seed:04d}"


PLANAR_FAMILIES = (
    "boss_plate",
    "lshape_outline",
    "plate_pockets",
    "polygon_prism",
    "rotated_pockets",
    "square_slots",
    "stepped_block",
    "thin_walls",
    "through_cuts",
)


def planar_entries(count: int = GRID_SIZES["standard"]) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        family, seed = PLANAR_FAMILIES[i % len(PLANAR_FAMILIES)], i // len(PLANAR_FAMILIES)
        grids = [g for g in GRID_ORDER if i < GRID_SIZES[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"category": "planar"},
                "grids": grids,
            }
        )
    return entries


def pinned_fingerprint(shape) -> dict[str, Any]:
    fp = fingerprint(shape)
    return {
        "volume": round(fp["volume"], 9),
        "face_count": fp["face_count"],
        "bbox_min": [round(v, 6) for v in fp["bbox_min"]],
        "bbox_max": [round(v, 6) for v in fp["bbox_max"]],
    }


def sync_manifest(manifest: dict[str, Any], planned: list[dict[str, Any]]) -> dict[str, Any]:
    known = {e["id"] for e in manifest["entries"]}
    manifest["entries"] += [e for e in planned if e["id"] not in known]
    for entry in manifest["entries"]:
        if "fingerprint" not in entry and entry["tier"] == "generated":
            gt = generate(entry["family"], entry["seed"])
            entry["fingerprint"] = pinned_fingerprint(gt.solid)
    return manifest


def empty_manifest() -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "grids": {
            "smoke": "Fast PR gate, subset of standard.",
            "standard": "Nightly corpus.",
        },
        "entries": [],
    }


def find_manifest() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "corpus").is_dir() or (parent / "Cargo.toml").is_file():
            return parent / "corpus" / "v0.json"
    raise FileNotFoundError("repo root not found; pass --manifest")


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
