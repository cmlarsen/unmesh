from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from build123d import export_step

from .groundtruth import fingerprint, generate

MANIFEST_VERSION = 0
GRID_SIZES = {"smoke": 50, "standard": 619}
GRID_ORDER = ["smoke", "standard"]
IMPORTED_TIER_CANDIDATE_CAP = 233

PLANAR_GRID_COUNTS = {"smoke": 20, "standard": 100}
CURVED_GRID_COUNTS = {"smoke": 10, "standard": 80}
CHAMFER_FILLET_GRID_COUNTS = {"smoke": 10, "standard": 120}
AMBIGUITY_GRID_COUNTS = {"smoke": 6, "standard": 30}
COMPLEX_GRID_COUNTS = {"smoke": 4, "standard": 60}


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


def planar_entries(count: int = 100) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        family, seed = PLANAR_FAMILIES[i % len(PLANAR_FAMILIES)], i // len(PLANAR_FAMILIES)
        grids = [g for g in GRID_ORDER if i < PLANAR_GRID_COUNTS[g]]
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


CURVED_FAMILIES = (
    "through_bore",
    "blind_bore",
    "round_boss",
    "counterbore",
    "countersink",
    "round_slot_through",
    "round_slot_blind",
    "revolved_cone",
    "revolved_dome",
    "revolved_torus",
)


def curved_entries(count: int = 80) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        family, seed = CURVED_FAMILIES[i % len(CURVED_FAMILIES)], i // len(CURVED_FAMILIES)
        grids = [g for g in GRID_ORDER if i < CURVED_GRID_COUNTS[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"category": "curved"},
                "grids": grids,
            }
        )
    return entries


CHAMFER_FILLET_FAMILIES = (
    "planar_chamfer",
    "straight_fillet",
    "circular_fillet",
    "bore_chamfer",
    "corner_fillet",
)


def chamfer_fillet_entries(count: int = 120) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        n = len(CHAMFER_FILLET_FAMILIES)
        family, seed = CHAMFER_FILLET_FAMILIES[i % n], i // n
        grids = [g for g in GRID_ORDER if i < CHAMFER_FILLET_GRID_COUNTS[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"category": "chamfer_fillet"},
                "grids": grids,
            }
        )
    return entries


def planned_entries() -> list[dict[str, Any]]:
    return (
        planar_entries()
        + curved_entries()
        + chamfer_fillet_entries()
        + ambiguity_entries()
        + complex_entries()
    )


AMBIGUITY_FAMILIES = (
    "ngon_prism",
    "coarse_cylinder_prism",
    "one_segment_fillet",
    "chamfer_same_chord",
    "two_segment_fillet",
    "two_planes",
)


def ambiguity_entries(count: int = 30) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        n = len(AMBIGUITY_FAMILIES)
        family, seed = AMBIGUITY_FAMILIES[i % n], i // n
        grids = [g for g in GRID_ORDER if i < AMBIGUITY_GRID_COUNTS[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"category": "ambiguity"},
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


COMPLEX_FAMILIES = (
    "complex_mixed",
    "complex_thin",
    "complex_void",
    "complex_assembly",
)


def complex_entries(count: int = 60) -> list[dict[str, Any]]:
    entries = []
    for i in range(count):
        n = len(COMPLEX_FAMILIES)
        family, seed = COMPLEX_FAMILIES[i % n], i // n
        grids = [g for g in GRID_ORDER if i < COMPLEX_GRID_COUNTS[g]]
        entries.append(
            {
                "id": entry_id(family, seed),
                "tier": "generated",
                "family": family,
                "seed": seed,
                "strata": {"category": "complex"},
                "grids": grids,
            }
        )
    return entries


def sync_manifest(manifest: dict[str, Any], planned: list[dict[str, Any]]) -> dict[str, Any]:
    known = {e["id"] for e in manifest["entries"]}
    manifest["entries"] += [e for e in planned if e["id"] not in known]
    missing = [
        entry
        for entry in manifest["entries"]
        if "fingerprint" not in entry and entry["tier"] == "generated"
    ]
    for i, entry in enumerate(missing):
        gt = generate(entry["family"], entry["seed"])
        entry["fingerprint"] = pinned_fingerprint(gt.solid)
        if (i + 1) % 25 == 0 or i + 1 == len(missing):
            print(f"fingerprinted {i + 1}/{len(missing)}")
    return manifest


def imported_candidates(cache_dir=None) -> list[dict[str, Any]]:
    from .imported import IMPORTED_DATASETS, datasets_root

    root = datasets_root(cache_dir)
    out = []
    for dataset in IMPORTED_DATASETS:
        manifest_path = root / dataset / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for item in manifest.get("entries", []):
            step = item.get("files", {}).get("step")
            if step is None:
                continue
            out.append({"dataset": dataset, "file_id": item["id"], "sha256": step["sha256"]})
    out.sort(key=lambda s: (s["dataset"], s["file_id"]))
    return out[:IMPORTED_TIER_CANDIDATE_CAP]


def rejected_reasons(manifest_path: Path | None = None) -> dict[str, str]:
    path = (manifest_path or find_manifest()).parent / "imported_rejected.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    if record.get("candidate_cap") != IMPORTED_TIER_CANDIDATE_CAP:
        raise RuntimeError(
            f"{path}: candidate_cap {record.get('candidate_cap')} != "
            f"IMPORTED_TIER_CANDIDATE_CAP {IMPORTED_TIER_CANDIDATE_CAP}"
        )
    return {item["file_id"]: item["reason"] for item in record.get("rejected", [])}


def probe_imported_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    from .groundtruth import validity_problems
    from .imported import load_imported_shape
    from .strata import compute_strata

    entry = {"id": "probe", "source": payload["source"]}
    shape = load_imported_shape(entry, payload.get("cache_dir"))
    if shape.wrapped is None:
        raise RuntimeError("STEP import produced no geometry")
    solids, shells = len(shape.solids()), len(shape.shells())
    problems = validity_problems(shape, solids=solids, shells=shells)
    if problems or not solids or not shape.faces():
        raise RuntimeError("; ".join(problems) or "no faces")
    return {
        "solids": solids,
        "shells": shells,
        "fingerprint": pinned_fingerprint(shape),
        "strata": compute_strata(shape, "imported"),
    }


def probe_main(payload_json: str) -> int:
    print(PROBE_MARKER + json.dumps(probe_imported_candidate(json.loads(payload_json))), flush=True)
    return 0


PROBE_MARKER = "UNMESH_PROBE_RESULT "
_PROBE_CODE = (
    "import json, sys; "
    "from unmesh_harness.corpus import probe_main; "
    "raise SystemExit(probe_main(sys.argv[1]))"
)


def sync_imported(
    manifest: dict[str, Any], cache_dir=None, rejected: dict[str, str] | None = None
) -> dict[str, Any]:
    import subprocess
    import sys

    from .imported import IMPORTED_FAMILY, datasets_root

    have = {e["source"]["file_id"] for e in manifest["entries"] if e.get("tier") == "imported"}
    known_bad = rejected if rejected is not None else {}
    new = [c for c in imported_candidates(cache_dir) if c["file_id"] not in have]
    if not new:
        return manifest
    start = sum(1 for e in manifest["entries"] if e.get("tier") == "imported")
    kept, skipped = 0, 0
    for i, source in enumerate(new):
        if source["file_id"] in known_bad:
            print(f"skip {source['file_id']}: previously rejected ({known_bad[source['file_id']]})")
            skipped += 1
            continue
        payload = json.dumps({"source": source, "cache_dir": str(cache_dir) if cache_dir else None})
        try:
            run = subprocess.run(
                [sys.executable, "-c", _PROBE_CODE, payload],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"skip {source['file_id']}: import timed out", flush=True)
            skipped += 1
            continue
        if run.returncode != 0:
            tail = (run.stderr.strip().splitlines() or ["import crashed"])[:1]
            print(f"skip {source['file_id']}: {run.returncode}: {tail[0][:160]}", flush=True)
            skipped += 1
            continue
        try:
            (line,) = [ln for ln in run.stdout.splitlines() if ln.startswith(PROBE_MARKER)]
            probed = json.loads(line[len(PROBE_MARKER) :])
        except ValueError:
            print(f"skip {source['file_id']}: unreadable probe output", flush=True)
            skipped += 1
            continue
        manifest["entries"].append(
            {
                "id": f"imported-{start + kept:04d}",
                "tier": "imported",
                "family": IMPORTED_FAMILY,
                "seed": start + kept,
                "source": source,
                "strata": probed["strata"],
                "grids": ["standard"],
                "fingerprint": probed["fingerprint"],
                "solids": probed["solids"],
                "shells": probed["shells"],
            }
        )
        kept += 1
        if (i + 1) % 25 == 0 or i + 1 == len(new):
            print(f"imported {i + 1}/{len(new)} ({kept} kept, {skipped} skipped)", flush=True)
    root = datasets_root(cache_dir)
    print(f"imported tier: {kept} kept, {skipped} skipped; dataset cache at {root}")
    return manifest


def sync_strata(manifest: dict[str, Any]) -> dict[str, Any]:
    from .strata import compute_strata

    want = ("face_count", "face_bucket", "min_feature_ratio", "feature_bucket")
    missing = [
        entry
        for entry in manifest["entries"]
        if entry.get("tier") == "generated" and any(k not in entry.get("strata", {}) for k in want)
    ]
    for i, entry in enumerate(missing):
        gt = generate(entry["family"], entry["seed"])
        entry["strata"] = compute_strata(gt.solid, entry["strata"].get("category", "planar"))
        if (i + 1) % 25 == 0 or i + 1 == len(missing):
            print(f"strata {i + 1}/{len(missing)}", flush=True)
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
    from .imported import imported_metadata, load_imported_shape

    out.mkdir(parents=True, exist_ok=True)
    if entry["tier"] == "imported":
        shape = load_imported_shape(entry)
        export_step(shape, str(out / f"{entry['id']}.step"))
        meta = imported_metadata(shape, entry)
    elif entry["tier"] == "generated":
        gt = generate(entry["family"], entry["seed"])
        export_step(gt.solid, str(out / f"{entry['id']}.step"))
        meta = gt.metadata()
        shape = gt.solid
    else:
        raise NotImplementedError(f"tier {entry['tier']!r} is not buildable yet")
    meta["id"] = entry["id"]
    meta["tier"] = entry["tier"]
    meta["strata"] = entry.get("strata", {})
    (out / f"{entry['id']}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return shape


def build_grid(manifest: dict[str, Any], grid: str, out: Path) -> int:
    entries = select(manifest, grid)
    for entry in entries:
        build_entry(entry, out)
    return len(entries)
