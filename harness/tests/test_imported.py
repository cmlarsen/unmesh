import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

import unmesh_harness.corpus as corpus
from unmesh.ir import validate
from unmesh_harness import datasets as d
from unmesh_harness.corpus import (
    IMPORTED_TIER_CANDIDATE_CAP,
    build_entry,
    find_manifest,
    load_manifest,
    rejected_reasons,
    select,
)
from unmesh_harness.groundtruth import validity_problems
from unmesh_harness.imported import (
    IMPORTED_DATASETS,
    IMPORTED_FAMILY,
    cached_step_record,
    datasets_root,
    imported_metadata,
    load_imported_shape,
)
from unmesh_harness.labels import face_surface_types, tessellate
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


def _rejected_record():
    return json.loads((find_manifest().parent / "imported_rejected.json").read_text())


def test_imported_candidate_cap_matches_pin():
    record = _rejected_record()
    assert record["candidate_cap"] == IMPORTED_TIER_CANDIDATE_CAP == 233
    assert len(_imported_entries()) + len(record["rejected"]) == IMPORTED_TIER_CANDIDATE_CAP


def test_rejected_files_recorded_with_reasons():
    record = _rejected_record()
    assert len(record["rejected"]) == 4
    sources = {e["source"]["file_id"] for e in _imported_entries()}
    for item in record["rejected"]:
        assert set(item) == {"dataset", "file_id", "sha256", "reason"}
        assert item["dataset"] in IMPORTED_DATASETS
        assert item["file_id"] not in sources
        assert len(item["reason"]) > 20


def test_imported_candidates_truncates_to_cap(tmp_path, monkeypatch):
    from unmesh_harness.imported import IMPORTED_DATASETS

    ids = {"nist-pmi": ["z", "y"], "freecad-library": ["b", "a", "c"]}
    for ds in IMPORTED_DATASETS:
        root = tmp_path / ds
        root.mkdir(parents=True)
        entries = [
            {"id": f"{ds}/{i}", "files": {"step": {"sha256": "0" * 64, "path": f"{i}.step"}}}
            for i in ids[ds]
        ]
        (root / "manifest.json").write_text(json.dumps({"entries": entries}))
    monkeypatch.setattr(corpus, "IMPORTED_TIER_CANDIDATE_CAP", 3)
    got = corpus.imported_candidates(tmp_path)
    assert [(c["dataset"], c["file_id"]) for c in got] == [
        ("freecad-library", "freecad-library/a"),
        ("freecad-library", "freecad-library/b"),
        ("freecad-library", "freecad-library/c"),
    ]


def test_rejected_reasons_missing_file_is_empty(tmp_path):
    assert rejected_reasons(tmp_path / "v0.json") == {}


def test_rejected_reasons_rejects_cap_mismatch(tmp_path):
    (tmp_path / "imported_rejected.json").write_text(json.dumps({"candidate_cap": 1}))
    with pytest.raises(RuntimeError, match="candidate_cap"):
        rejected_reasons(tmp_path / "v0.json")


def _fake_probe_success(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        probe = {
            "solids": 1,
            "shells": 1,
            "fingerprint": {
                "volume": 1.0,
                "face_count": 1,
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
            },
            "strata": {"category": "imported"},
        }
        return SimpleNamespace(
            returncode=0, stdout=corpus.PROBE_MARKER + json.dumps(probe) + "\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)


def _write_cache_manifest(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"entries": entries}))


def _cache_entry(fid, sha="0" * 64):
    return {"id": fid, "files": {"step": {"sha256": sha, "path": "step/x.step"}}}


def test_imported_candidates_keep_pinned_first_then_new_in_order(tmp_path, monkeypatch):
    nist = [_cache_entry("nist-pmi/b"), _cache_entry("nist-pmi/a")]
    freecad = [_cache_entry(f"freecad-library/{c}") for c in "zyxw"]
    _write_cache_manifest(tmp_path / "nist-pmi", nist)
    _write_cache_manifest(tmp_path / "freecad-library", freecad)
    monkeypatch.setattr(corpus, "IMPORTED_TIER_CANDIDATE_CAP", 4)
    got = corpus.imported_candidates(tmp_path, {"freecad-library/z", "nist-pmi/a"})
    assert [(c["dataset"], c["file_id"]) for c in got] == [
        ("freecad-library", "freecad-library/z"),
        ("nist-pmi", "nist-pmi/a"),
        ("freecad-library", "freecad-library/w"),
        ("freecad-library", "freecad-library/x"),
    ]


def test_sync_imported_repins_larger_cache_without_touching_existing(tmp_path, monkeypatch):
    existing_fc = [f"freecad-library/F{i:03d}" for i in range(198)]
    existing_nist = [f"nist-pmi/N{i:03d}" for i in range(31)]
    fresh_fc = [f"freecad-library/F{i:03d}" for i in range(198, 209)]
    _write_cache_manifest(
        tmp_path / "freecad-library", [_cache_entry(f) for f in existing_fc + fresh_fc]
    )
    _write_cache_manifest(tmp_path / "nist-pmi", [_cache_entry(f) for f in existing_nist])
    manifest = {"entries": []}
    for i, fid in enumerate(existing_fc + existing_nist):
        manifest["entries"].append(
            {
                "id": f"imported-{i:04d}",
                "tier": "imported",
                "family": IMPORTED_FAMILY,
                "seed": i,
                "source": {
                    "dataset": fid.split("/")[0],
                    "file_id": fid,
                    "sha256": "0" * 64,
                },
                "strata": {"category": "imported"},
                "grids": ["standard"],
                "fingerprint": {"volume": float(i)},
            }
        )
    before = copy.deepcopy(manifest["entries"])
    monkeypatch.setattr(corpus, "IMPORTED_TIER_CANDIDATE_CAP", 233)
    _fake_probe_success(monkeypatch)
    out = corpus.sync_imported(manifest, tmp_path, rejected={})
    assert out["entries"][:229] == before
    assert len(out["entries"]) == 233
    assert [e["source"]["file_id"] for e in out["entries"][229:]] == fresh_fc[:4]
    assert set(existing_nist) <= {e["source"]["file_id"] for e in out["entries"]}


def test_sync_imported_caps_new_entries_when_cache_misses_pinned_files(tmp_path, monkeypatch):
    existing_fc = [f"freecad-library/F{i:03d}" for i in range(198)]
    existing_nist = [f"nist-pmi/N{i:03d}" for i in range(31)]
    fresh_fc = [f"freecad-library/F{i:03d}" for i in range(198, 238)]
    _write_cache_manifest(
        tmp_path / "freecad-library", [_cache_entry(f) for f in existing_fc + fresh_fc]
    )
    manifest = {"entries": []}
    for i, fid in enumerate(existing_fc + existing_nist):
        manifest["entries"].append(
            {
                "id": f"imported-{i:04d}",
                "tier": "imported",
                "family": IMPORTED_FAMILY,
                "seed": i,
                "source": {
                    "dataset": fid.split("/")[0],
                    "file_id": fid,
                    "sha256": "0" * 64,
                },
                "strata": {"category": "imported"},
                "grids": ["standard"],
                "fingerprint": {"volume": float(i)},
            }
        )
    before = copy.deepcopy(manifest["entries"])
    monkeypatch.setattr(corpus, "IMPORTED_TIER_CANDIDATE_CAP", 233)
    _fake_probe_success(monkeypatch)
    out = corpus.sync_imported(manifest, tmp_path, rejected={})
    assert out["entries"][:229] == before
    assert len(out["entries"]) == 233
    assert [e["source"]["file_id"] for e in out["entries"][229:]] == fresh_fc[:4]


def test_sync_imported_skips_previously_rejected_without_probing(monkeypatch, capsys):
    manifest = {"entries": []}
    cands = [{"dataset": "d", "file_id": "d/x", "sha256": "s"}]
    monkeypatch.setattr(corpus, "imported_candidates", lambda cache_dir=None, have=(): cands)
    out = corpus.sync_imported(manifest, rejected={"d/x": "boom"})
    assert out["entries"] == []
    assert "previously rejected (boom)" in capsys.readouterr().out


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


ORACLE_SURFACES = {"plane", "cylinder", "cone", "sphere", "torus"}
IMPORTED_SAMPLE_BUCKETS = (("small", 0, 10), ("mid", 11, 50), ("large", 51, None))
IMPORTED_SAMPLE_PER_BUCKET = 4


def _imported_bucket_sample():
    entries = _imported_entries()
    sample = []
    for name, lo, hi in IMPORTED_SAMPLE_BUCKETS:
        bucket = sorted(
            (
                e
                for e in entries
                if e["fingerprint"]["face_count"] >= lo
                and (hi is None or e["fingerprint"]["face_count"] <= hi)
            ),
            key=lambda e: e["id"],
        )
        supported = [
            e for e in bucket if set(face_surface_types(load_imported_shape(e))) <= ORACLE_SURFACES
        ]
        assert len(supported) >= IMPORTED_SAMPLE_PER_BUCKET, name
        stride = max(1, len(supported) // IMPORTED_SAMPLE_PER_BUCKET)
        sample += supported[::stride][:IMPORTED_SAMPLE_PER_BUCKET]
    return sample


@pytest.mark.slow
def test_imported_sample_across_size_buckets_tessellates_and_oracle_validates():
    _require_cache()
    sample = _imported_bucket_sample()
    assert len(sample) == len(IMPORTED_SAMPLE_BUCKETS) * IMPORTED_SAMPLE_PER_BUCKET
    seen = set()
    for entry in sample:
        mesh = tessellate(load_imported_shape(entry), 0.01, 0.2)
        assert len(mesh.faces) == entry["fingerprint"]["face_count"], entry["id"]
        assert validate(build_oracle_ir(mesh)) == [], entry["id"]
        seen.add(entry["id"])
    print(f"checked {sorted(seen)}")


def test_imported_metadata_marks_features_null():
    _require_cache()
    entry = _imported_entries()[0]
    assert imported_metadata(load_imported_shape(entry), entry)["features"] is None


def test_imported_coincident_bodies_and_open_input_are_reported(tmp_path):
    _require_cache()
    import unmesh
    from unmesh.pipeline import convert_to_step

    by_id = {e["id"]: e for e in _imported_entries()}
    solids = tmp_path / "imported-0103.step"
    mesh = tessellate(load_imported_shape(by_id["imported-0103"]), 0.01, 0.2)
    unmesh.write_stl(str(tmp_path / "imported-0103.stl"), mesh.tris)
    conv = convert_to_step(str(tmp_path / "imported-0103.stl"), solids, measure=False)
    assert conv.fidelity["validity"]["valid"] is True, conv.fidelity["validity"]["issues"]
    assert conv.fidelity["validity"]["solids"] == 2
    assert "coincident_shells" in [w["code"] for w in conv.fidelity["input"]["warnings"]]

    open_id = tmp_path / "imported-0218.step"
    mesh = tessellate(load_imported_shape(by_id["imported-0218"]), 0.01, 0.2)
    unmesh.write_stl(str(tmp_path / "imported-0218.stl"), mesh.tris)
    conv = convert_to_step(str(tmp_path / "imported-0218.stl"), open_id, measure=False)
    boundary = conv.fidelity["validity"]["open_boundary"]
    assert boundary["edges"] == 4
    assert boundary["length"] == pytest.approx(25.4, abs=1e-3)
    assert boundary["non_manifold_edges"] > 0


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
