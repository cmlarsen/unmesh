from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IMPORTED_FAMILY = "imported"
IMPORTED_DATASETS = ("nist-pmi", "freecad-library")


def missing_message(entry: dict[str, Any], cache: Path) -> str:
    from .datasets import IMPORTED_TIER_COMMAND

    src = entry.get("source", {})
    return (
        f"imported part {entry.get('id')} needs {src.get('dataset')}/{src.get('file_id')} "
        f"from the dataset cache at {cache}, which is absent or incomplete. "
        f"Run `{IMPORTED_TIER_COMMAND}` from the repo root "
        "to download exactly the STEP files the corpus manifest references, then retry."
    )


def datasets_root(cache_dir: Path | None = None) -> Path:
    from .datasets import default_datasets_dir

    return Path(cache_dir) if cache_dir is not None else default_datasets_dir()


def cached_step_record(
    source: dict[str, Any], cache_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = datasets_root(cache_dir) / source["dataset"]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    for item in manifest.get("entries", []):
        if item["id"] == source["file_id"]:
            rec = item["files"]["step"]
            path = root / rec["path"]
            if not path.is_file() or rec["sha256"] != source["sha256"]:
                raise FileNotFoundError(str(path))
            return path, rec
    raise KeyError(f"{source['file_id']} not in {manifest_path}")


def load_imported_shape(entry: dict[str, Any], cache_dir: Path | None = None):
    from build123d import import_step

    from .datasets import sha256_file

    root = datasets_root(cache_dir)
    try:
        path, _ = cached_step_record(entry["source"], cache_dir)
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(missing_message(entry, root)) from e
    if sha256_file(path) != entry["source"]["sha256"]:
        raise RuntimeError(
            f"imported part {entry['id']}: {path} checksum mismatch; "
            "re-run scripts/fetch-datasets --verify to repair the dataset cache"
        )
    try:
        return import_step(path)
    except Exception as e:
        raise RuntimeError(f"imported part {entry['id']}: cannot read {path}: {e}") from e


def imported_metadata(shape, entry: dict[str, Any]) -> dict[str, Any]:
    from .corpus import pinned_fingerprint
    from .labels import face_surface_types

    return {
        "family": IMPORTED_FAMILY,
        "seed": entry["seed"],
        "parameters": {"source": dict(entry["source"])},
        "features": None,
        "faces": [
            {"id": i, "surface": surface} for i, surface in enumerate(face_surface_types(shape))
        ],
        "fingerprint": pinned_fingerprint(shape),
    }
