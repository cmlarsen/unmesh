import json

import pytest

from unmesh.ir import validate
from unmesh_harness import datasets as d
from unmesh_harness.corpus import build_entry, load_manifest, select
from unmesh_harness.groundtruth import validity_problems
from unmesh_harness.imported import (
    IMPORTED_DATASETS,
    IMPORTED_FAMILY,
    cached_step_record,
    datasets_root,
    imported_metadata,
    load_imported_shape,
)
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir
from unmesh_harness.strata import compute_strata


def _imported_entries(grid="standard"):
    return [e for e in select(load_manifest(), grid) if e["tier"] == "imported"]


def _cache_present() -> bool:
    root = datasets_root()
    return all((root / ds / "manifest.json").is_file() for ds in IMPORTED_DATASETS)


def _missing_imported_ids() -> list[str]:
    missing = []
    for entry in _imported_entries():
        try:
            path, rec = cached_step_record(entry["source"])
        except (FileNotFoundError, KeyError):
            missing.append(entry["id"])
            continue
        if not path.is_file() or rec["sha256"] != entry["source"]["sha256"]:
            missing.append(entry["id"])
    return missing


def _require_cache():
    from unmesh_harness.datasets import IMPORTED_TIER_COMMAND

    total = len(_imported_entries())
    if not _cache_present():
        pytest.skip(
            f"dataset cache absent for {total} imported entries; "
            f"run `{IMPORTED_TIER_COMMAND}` from the repo root, then retry"
        )
    missing = _missing_imported_ids()
    if missing:
        shown = ", ".join(missing[:5])
        pytest.skip(
            f"dataset cache partial: {len(missing)}/{total} imported entries do not resolve "
            f"({shown}{'...' if len(missing) > 5 else ''}); "
            f"run `{IMPORTED_TIER_COMMAND}` from the repo root, then retry"
        )


def test_imported_manifest_entries():
    entries = _imported_entries()
    assert len(entries) >= 200
    assert {e["family"] for e in entries} == {IMPORTED_FAMILY}
    for entry in entries:
        assert entry["grids"] == ["standard"]
        assert entry["strata"]["category"] == "imported"
        assert set(entry["source"]) == {"dataset", "file_id", "sha256"}
        assert entry["source"]["dataset"] in IMPORTED_DATASETS
        assert "fingerprint" in entry


def test_imported_sources_resolve_in_cache():
    entries = _imported_entries()
    _require_cache()
    for i, entry in enumerate(entries):
        path, rec = cached_step_record(entry["source"])
        assert path.is_file()
        assert rec["sha256"] == entry["source"]["sha256"]
        if (i + 1) % 50 == 0:
            print(f"resolved {i + 1}/{len(entries)}")


def test_partial_cache_skips_naming_command_and_missing_count(tmp_path, monkeypatch):
    monkeypatch.setenv("UNMESH_CACHE_DIR", str(tmp_path))
    for ds in IMPORTED_DATASETS:
        (tmp_path / "datasets" / ds).mkdir(parents=True)
        (tmp_path / "datasets" / ds / "manifest.json").write_text('{"entries": []}')
    with pytest.raises(pytest.skip.Exception) as exc:
        _require_cache()
    msg = str(exc.value.msg)
    total = len(_imported_entries())
    assert "--manifest corpus/v0.json" in msg
    assert f"{total}/{total} imported entries do not resolve" in msg


@pytest.mark.slow
def test_imported_shapes_valid_and_match_fingerprint():
    _require_cache()
    entries = _imported_entries()
    for i, entry in enumerate(entries):
        shape = load_imported_shape(entry)
        assert validity_problems(shape, solids=entry["solids"], shells=entry["shells"]) == [], (
            entry["id"]
        )
        assert len(shape.faces()) == entry["fingerprint"]["face_count"], entry["id"]
        assert shape.volume == pytest.approx(entry["fingerprint"]["volume"], rel=1e-9)
        assert entry["strata"] == compute_strata(shape, "imported"), entry["id"]
        if (i + 1) % 50 == 0:
            print(f"checked {i + 1}/{len(entries)}")


def test_imported_metadata_has_face_table_and_null_features(tmp_path):
    _require_cache()
    for entry in _imported_entries()[:2]:
        build_entry(entry, tmp_path)
        assert (tmp_path / f"{entry['id']}.step").stat().st_size > 0
        meta = json.loads((tmp_path / f"{entry['id']}.json").read_text())
        assert meta["features"] is None
        assert meta["family"] == IMPORTED_FAMILY
        assert len(meta["faces"]) == entry["fingerprint"]["face_count"]
        assert {f["surface"] for f in meta["faces"]} <= {
            "plane",
            "cylinder",
            "cone",
            "sphere",
            "torus",
            "bezier",
            "bspline",
            "revolution",
            "extrusion",
            "offset",
            "other",
        }


def test_imported_sample_tessellates_and_oracle_validates():
    _require_cache()
    smallest = sorted(_imported_entries(), key=lambda e: e["fingerprint"]["face_count"])[:3]
    assert smallest
    for entry in smallest:
        mesh = tessellate(load_imported_shape(entry), 0.01, 0.2)
        assert len(mesh.faces) == entry["fingerprint"]["face_count"], entry["id"]
        assert validate(build_oracle_ir(mesh)) == [], entry["id"]


def test_imported_metadata_marks_features_null():
    _require_cache()
    entry = _imported_entries()[0]
    assert imported_metadata(load_imported_shape(entry), entry)["features"] is None


def test_missing_cache_error_names_fetch_script(tmp_path, monkeypatch):
    monkeypatch.setenv("UNMESH_CACHE_DIR", str(tmp_path))
    entry = {
        "id": "imported-0000",
        "source": {"dataset": "nist-pmi", "file_id": "nist-pmi/x", "sha256": "0" * 64},
    }
    with pytest.raises(RuntimeError, match="scripts/fetch-datasets"):
        load_imported_shape(entry)


def test_dataset_manifests_verify(tmp_path):
    _require_cache()
    for ds in IMPORTED_DATASETS:
        assert d.verify_manifest(datasets_root() / ds) == []
