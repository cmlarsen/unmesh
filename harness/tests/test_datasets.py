import hashlib
import json
import os

import numpy as np
import pytest

from unmesh_harness import datasets as d


def test_classify_thingi_license():
    assert d.classify_thingi_license("Creative Commons - Attribution") == (
        d.REDISTRIBUTABLE,
        "CC-BY",
    )
    assert d.classify_thingi_license("Creative Commons - Public Domain Dedication")[1] == "CC0-1.0"
    assert d.classify_thingi_license("GNU - GPL") == (d.DOWNLOAD_ONLY, "GPL")
    assert d.classify_thingi_license("Creative Commons - Attribution - Share Alike")[0] == (
        d.DOWNLOAD_ONLY
    )
    assert d.classify_thingi_license("Creative Commons - Attribution - No Derivatives")[0] == (
        d.DOWNLOAD_ONLY
    )
    for nc in (
        "Creative Commons - Attribution - Non-Commercial",
        "Attribution - Non-Commercial - Share Alike",
        "Attribution - Non-Commercial - No Derivatives",
    ):
        assert d.classify_thingi_license(nc) == ("excluded", None)
    assert d.classify_thingi_license("unknown_license") == ("excluded", None)
    assert d.classify_thingi_license("anything new") == ("excluded", None)


def test_license_allowed():
    assert d.license_allowed(d.REDISTRIBUTABLE, d.REDISTRIBUTABLE)
    assert not d.license_allowed(d.DOWNLOAD_ONLY, d.REDISTRIBUTABLE)
    assert d.license_allowed(d.DOWNLOAD_ONLY, d.DOWNLOAD_ONLY)
    assert not d.license_allowed("excluded", d.DOWNLOAD_ONLY)


def _thingi_rows():
    summary = [
        {
            "ID": "3",
            "Thing ID": "30",
            "License": "Creative Commons - Attribution",
            "Closed": "TRUE",
            "Edge manifold": "TRUE",
        },
        {
            "ID": "1",
            "Thing ID": "10",
            "License": "Public Domain",
            "Closed": "TRUE",
            "Edge manifold": "TRUE",
        },
        {
            "ID": "2",
            "Thing ID": "20",
            "License": "GNU - GPL",
            "Closed": "TRUE",
            "Edge manifold": "TRUE",
        },
        {
            "ID": "4",
            "Thing ID": "40",
            "License": "Public Domain",
            "Closed": "FALSE",
            "Edge manifold": "TRUE",
        },
        {
            "ID": "5",
            "Thing ID": "50",
            "License": "unknown_license",
            "Closed": "TRUE",
            "Edge manifold": "TRUE",
        },
    ]
    context = [
        {"Thing ID": "10", "Category": "tools", "Name": " A ", "Author": "x"},
        {"Thing ID": "30", "Category": "art", "Name": "B", "Author": "y"},
    ]
    return summary, context


def test_select_thingi_tiers_and_order():
    summary, context = _thingi_rows()
    strict = d.select_thingi(summary, context, d.REDISTRIBUTABLE)
    assert [p["file_id"] for p in strict] == ["1", "3"]
    assert strict[0]["name"] == "A"
    loose = d.select_thingi(summary, context, d.DOWNLOAD_ONLY)
    assert [p["file_id"] for p in loose] == ["1", "2", "3"]
    only_art = d.select_thingi(summary, context, d.REDISTRIBUTABLE, {"art"})
    assert [p["file_id"] for p in only_art] == ["3"]


def test_select_freecad_pairs():
    def blob(path, size=10):
        return {"path": path, "type": "blob", "size": size, "sha": "0"}

    tree = [
        blob("Mechanical Parts/A/x.step"),
        blob("Mechanical Parts/A/x.stl"),
        blob("Mechanical Parts/A/y.stp"),
        blob("Mechanical Parts/A/z.step"),
        blob("Mechanical Parts/A/z.STL"),
        blob("Mechanical Parts/B/big.step", 10**9),
        blob("Mechanical Parts/B/big.stl"),
        blob("Electronics Parts/e.step"),
        blob("Electronics Parts/e.stl"),
        {"path": "Mechanical Parts/A", "type": "tree"},
    ]
    pairs = d.select_freecad_pairs(tree)
    assert [p["id"] for p in pairs] == ["Mechanical Parts/A/x", "Mechanical Parts/A/z"]
    assert pairs[1]["stl"]["path"].endswith("z.STL")


def test_round_robin_spreads_families():
    pairs = [{"id": f"Mechanical Parts/{f}/{i}"} for f in "ab" for i in range(5)]
    picked = d.round_robin(pairs, 4)
    assert sorted({p["id"].split("/")[1] for p in picked}) == ["a", "b"]
    assert len(picked) == 4


def test_strided_is_stable_and_bounded():
    items = [str(i) for i in range(100)]
    assert d.strided(items, 10) == items[::10]
    assert d.strided(items[:3], 10) == items[:3]
    assert d.strided(items, 0) == []


def test_git_blob_sha1_known_value():
    assert d.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_stl_bytes_roundtrip():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    f = np.array([[0, 1, 2], [0, 3, 1]])
    data = d.stl_bytes(v, f)
    assert len(data) == 84 + 50 * 2
    assert int.from_bytes(data[80:84], "little") == 2
    n0 = np.frombuffer(data[84:96], dtype="<f4")
    assert np.allclose(n0, [0, 0, 1])


def test_meta_field():
    text = "createdAt: 2014-07-24T19:51:48.213+0000\ndescription: null\nname: Sample\n"
    assert d.meta_field(text, "createdAt") == "2014-07-24T19:51:48.213+0000"
    assert d.meta_field(text, "description") is None
    assert d.meta_field(text, "missing") is None


def test_parse_sevenzip_listing():
    text = "Path = 00000000\nSize = 0\n\nPath = 00000000/a_step_000.step\nSize = 5\n"
    assert d.parse_sevenzip_listing(text) == ["00000000", "00000000/a_step_000.step"]


def test_manifest_roundtrip_and_verify(tmp_path):
    root = tmp_path / "ds"
    (root / "step").mkdir(parents=True)
    f = root / "step" / "a.step"
    f.write_bytes(b"ISO-10303-21;")
    ds = d.DATASETS["nist-pmi"]
    manifest = d._manifest_for(ds, limit=1)
    manifest.entries.append(
        {"id": "nist-pmi/a", "license": ds.license, "files": {"step": d.file_record(root, f)}}
    )
    manifest.write(root)
    loaded = d.Manifest.read(root)
    rec = loaded["entries"][0]["files"]["step"]
    assert rec["sha256"] == hashlib.sha256(b"ISO-10303-21;").hexdigest()
    assert rec["path"] == "step/a.step"
    assert loaded["license_quote"] == ds.license_quote
    assert d.verify_manifest(root) == []
    f.write_bytes(b"tampered")
    assert any("checksum mismatch" in p for p in d.verify_manifest(root))
    f.unlink()
    assert any("missing" in p for p in d.verify_manifest(root))


def test_every_dataset_has_license_evidence_and_fetcher():
    assert set(d.DATASETS) == set(d.FETCHERS)
    for ds in d.DATASETS.values():
        assert ds.license_quote and ds.license_url.startswith("https://")
        assert ds.tier in (d.REDISTRIBUTABLE, d.DOWNLOAD_ONLY)


def test_main_list_and_verify(tmp_path, capsys):
    assert d.main(["--list"]) == 0
    assert "thingi10k" in capsys.readouterr().out
    root = tmp_path / "nist-pmi"
    root.mkdir()
    (root / d.MANIFEST_NAME).write_text(json.dumps({"entries": []}))
    assert d.main(["--verify", "--dataset", "nist-pmi", "--cache-dir", str(tmp_path)]) == 0


def test_default_cache_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UNMESH_CACHE_DIR", str(tmp_path))
    assert d.default_datasets_dir() == tmp_path / "datasets"


def test_non_commercial_never_selected():
    base = {"Closed": "TRUE", "Edge manifold": "TRUE"}
    names = (
        "Creative Commons - Attribution - Non-Commercial",
        "Attribution - Non-Commercial - No Derivatives",
    )
    summary = [
        {"ID": str(i), "Thing ID": str(i), "License": n, **base} for i, n in enumerate(names)
    ]
    assert d.select_thingi(summary, [], d.DOWNLOAD_ONLY) == []


def test_no_fusion_dataset():
    assert "fusion360-segmentation" not in d.DATASETS


def test_wanted_imported_refs_groups_file_ids_by_dataset():
    corpus = {
        "entries": [
            {"tier": "imported", "source": {"dataset": "nist-pmi", "file_id": "a", "sha256": "s1"}},
            {
                "tier": "imported",
                "source": {"dataset": "nist-pmi", "file_id": "b", "sha256": "s2"},
            },
            {
                "tier": "imported",
                "source": {"dataset": "freecad-library", "file_id": "c", "sha256": "s3"},
            },
            {"tier": "generated", "family": "plate_pockets", "seed": 0},
        ]
    }
    assert d.wanted_imported_refs(corpus) == {
        "nist-pmi": {"a": "s1", "b": "s2"},
        "freecad-library": {"c": "s3"},
    }


def test_fetch_nist_refs_is_incremental_and_preserves_extras(tmp_path, monkeypatch):
    import zipfile

    root = tmp_path / "nist-pmi"
    (root / "step").mkdir(parents=True)
    bodies = {"a.stp": b"STEP-A", "b.step": b"STEP-B", "extra.stp": b"STEP-X"}
    archive = tmp_path / "arc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in bodies.items():
            z.writestr(f"inner/{name}", data)
    calls = []
    monkeypatch.setattr(d, "_nist_archive", lambda _root: calls.append(1) or archive)

    def rec(name, data):
        (root / "step" / name).write_bytes(data)
        return d.file_record(root, root / "step" / name)

    prior_a = {
        "id": "nist-pmi/a",
        "license": "x",
        "license_tier": d.REDISTRIBUTABLE,
        "source_url": "u",
        "files": {"step": rec("a.stp", bodies["a.stp"])},
    }
    prior_extra = {
        "id": "nist-pmi/extra",
        "license": "x",
        "license_tier": d.REDISTRIBUTABLE,
        "source_url": "u",
        "files": {"step": rec("extra.stp", bodies["extra.stp"])},
    }
    ds = d.DATASETS["nist-pmi"]
    first = d._manifest_for(ds, mode="manifest", refs=2)
    first.entries = [prior_a, prior_extra]
    first.write(root)
    wanted = {
        "nist-pmi/a": hashlib.sha256(bodies["a.stp"]).hexdigest(),
        "nist-pmi/b": hashlib.sha256(bodies["b.step"]).hexdigest(),
    }

    got = d.fetch_nist_refs(ds, root, wanted)
    assert calls == [1]
    assert [e["id"] for e in got.entries] == ["nist-pmi/a", "nist-pmi/b", "nist-pmi/extra"]
    assert got.entries[0]["files"]["step"]["path"] == "step/a.stp"
    assert (root / "step" / "b.step").read_bytes() == bodies["b.step"]
    assert (root / "step" / "extra.stp").read_bytes() == bodies["extra.stp"]
    got.finalize()
    got.write(root)
    assert d.verify_manifest(root) == []
    first_json = (root / d.MANIFEST_NAME).read_text()

    again = d.fetch_nist_refs(ds, root, wanted)
    again.finalize()
    again.write(root)
    assert calls == [1]
    assert (root / d.MANIFEST_NAME).read_text() == first_json
    assert [e["id"] for e in again.entries] == ["nist-pmi/a", "nist-pmi/b", "nist-pmi/extra"]


def test_fetch_nist_refs_redownloads_on_checksum_mismatch(tmp_path, monkeypatch):
    import zipfile

    root = tmp_path / "nist-pmi"
    (root / "step").mkdir(parents=True)
    archive = tmp_path / "arc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("inner/a.stp", b"STEP-A")
    calls = []
    monkeypatch.setattr(d, "_nist_archive", lambda _root: calls.append(1) or archive)
    (root / "step" / "a.stp").write_bytes(b"corrupt")
    ds = d.DATASETS["nist-pmi"]
    wanted = {"nist-pmi/a": hashlib.sha256(b"STEP-A").hexdigest()}
    got = d.fetch_nist_refs(ds, root, wanted)
    assert calls == [1]
    assert (root / "step" / "a.stp").read_bytes() == b"STEP-A"
    assert got.entries[0]["files"]["step"]["sha256"] == wanted["nist-pmi/a"]


def test_interrupted_step_write_leaves_no_partial_file(tmp_path, monkeypatch):
    import zipfile

    root = tmp_path / "nist-pmi"
    (root / "step").mkdir(parents=True)
    archive = tmp_path / "arc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("inner/a.stp", b"STEP-A")
    monkeypatch.setattr(d, "_nist_archive", lambda _root: archive)
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("power cut")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    ds = d.DATASETS["nist-pmi"]
    wanted = {"nist-pmi/a": hashlib.sha256(b"STEP-A").hexdigest()}
    with pytest.raises(RuntimeError, match="power cut"):
        d.fetch_nist_refs(ds, root, wanted)
    assert not (root / "step" / "a.stp").exists()
    assert list(root.rglob("*.unmesh-tmp")) == []
    stale = root / "step" / "orphan.stp.unmesh-tmp"
    stale.write_bytes(b"junk")
    got = d.fetch_nist_refs(ds, root, wanted)
    assert not stale.exists()
    assert (root / "step" / "a.stp").read_bytes() == b"STEP-A"
    assert got.entries[0]["files"]["step"]["sha256"] == wanted["nist-pmi/a"]


def test_truncated_prior_manifest_rebuilds_with_warning(tmp_path, monkeypatch, capsys):
    import zipfile

    root = tmp_path / "nist-pmi"
    (root / "step").mkdir(parents=True)
    (root / d.MANIFEST_NAME).write_bytes(b'{"entries": [{"id": "nist-pmi/a"')
    archive = tmp_path / "arc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("inner/a.stp", b"STEP-A")
    monkeypatch.setattr(d, "_nist_archive", lambda _root: archive)
    ds = d.DATASETS["nist-pmi"]
    wanted = {"nist-pmi/a": hashlib.sha256(b"STEP-A").hexdigest()}
    got = d.fetch_nist_refs(ds, root, wanted)
    assert [e["id"] for e in got.entries] == ["nist-pmi/a"]
    assert "warning" in capsys.readouterr().err
    assert (root / "step" / "a.stp").read_bytes() == b"STEP-A"
    got.finalize()
    got.write(root)
    assert d.verify_manifest(root) == []


def test_fetch_freecad_refs_preserves_pairs_and_skips_present(tmp_path, monkeypatch):
    root = tmp_path / "freecad-library"
    (root / "step").mkdir(parents=True)
    (root / "stl").mkdir(parents=True)
    bodies = {
        "Mechanical Parts/A/Alpha.step": b"ALPHA-STEP",
        "Mechanical Parts/A/Alpha.stl": b"ALPHA-STL",
        "Mechanical Parts/B/Beta.step": b"BETA-STEP",
        "Mechanical Parts/B/Beta.stl": b"BETA-STL",
        "Mechanical Parts/C/Gamma.step": b"GAMMA-STEP",
        "Mechanical Parts/C/Gamma.stl": b"GAMMA-STL",
    }
    tree = [
        {"path": p, "type": "blob", "size": len(b), "sha": f"blob-{i}"}
        for i, (p, b) in enumerate(bodies.items())
    ]
    tree_calls, blob_calls = [], []

    def fake_tree():
        tree_calls.append(1)
        return tree

    def fake_blob(node):
        blob_calls.append(node["path"])
        return bodies[node["path"]]

    monkeypatch.setattr(d, "_freecad_tree", fake_tree)
    monkeypatch.setattr(d, "_download_freecad_blob", fake_blob)

    def put(rel, data):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return d.file_record(root, dest)

    alpha_step = put("step/0000_Alpha.step", bodies["Mechanical Parts/A/Alpha.step"])
    alpha_stl = put("stl/0000_Alpha.stl", bodies["Mechanical Parts/A/Alpha.stl"])
    gamma_step = put("step/0001_Gamma.step", bodies["Mechanical Parts/C/Gamma.step"])
    gamma_stl = put("stl/0001_Gamma.stl", bodies["Mechanical Parts/C/Gamma.stl"])
    ds = d.DATASETS["freecad-library"]
    first = d._manifest_for(ds, mode="manifest", refs=2)
    first.entries = [
        {
            "id": "freecad-library/Mechanical Parts/A/Alpha",
            "license": "CC-BY-3.0",
            "license_tier": ds.tier,
            "files": {"step": alpha_step, "stl": alpha_stl},
        },
        {
            "id": "freecad-library/Mechanical Parts/C/Gamma",
            "license": "CC-BY-3.0",
            "license_tier": ds.tier,
            "files": {"step": gamma_step, "stl": gamma_stl},
        },
    ]
    first.write(root)
    wanted = {
        "freecad-library/Mechanical Parts/A/Alpha": hashlib.sha256(
            bodies["Mechanical Parts/A/Alpha.step"]
        ).hexdigest(),
        "freecad-library/Mechanical Parts/B/Beta": hashlib.sha256(
            bodies["Mechanical Parts/B/Beta.step"]
        ).hexdigest(),
    }

    got = d.fetch_freecad_refs(ds, root, wanted)
    assert tree_calls == [1]
    assert blob_calls == ["Mechanical Parts/B/Beta.step"]
    assert [e["id"] for e in got.entries] == [
        "freecad-library/Mechanical Parts/A/Alpha",
        "freecad-library/Mechanical Parts/B/Beta",
        "freecad-library/Mechanical Parts/C/Gamma",
    ]
    alpha, beta, gamma = got.entries
    assert alpha["files"]["step"] == alpha_step
    assert alpha["files"]["stl"] == alpha_stl
    assert set(beta["files"]) == {"step"}
    assert (root / beta["files"]["step"]["path"]).read_bytes() == bodies[
        "Mechanical Parts/B/Beta.step"
    ]
    assert gamma["files"]["step"] == gamma_step
    assert gamma["files"]["stl"] == gamma_stl
    assert (root / "stl" / "0000_Alpha.stl").read_bytes() == bodies["Mechanical Parts/A/Alpha.stl"]
    got.finalize()
    got.write(root)
    assert d.verify_manifest(root) == []
    first_json = (root / d.MANIFEST_NAME).read_text()

    again = d.fetch_freecad_refs(ds, root, wanted)
    again.finalize()
    again.write(root)
    assert tree_calls == [1]
    assert blob_calls == ["Mechanical Parts/B/Beta.step"]
    assert (root / d.MANIFEST_NAME).read_text() == first_json


def test_fetch_freecad_refs_redownloads_corrupt_step_and_keeps_stl(tmp_path, monkeypatch):
    root = tmp_path / "freecad-library"
    (root / "step").mkdir(parents=True)
    (root / "stl").mkdir(parents=True)
    step_body = b"ALPHA-STEP"
    stl_body = b"ALPHA-STL"
    tree = [
        {
            "path": "Mechanical Parts/A/Alpha.step",
            "type": "blob",
            "size": len(step_body),
            "sha": "blob-0",
        },
        {
            "path": "Mechanical Parts/A/Alpha.stl",
            "type": "blob",
            "size": len(stl_body),
            "sha": "blob-1",
        },
    ]
    blob_calls = []
    monkeypatch.setattr(d, "_freecad_tree", lambda: tree)

    def fake_blob(node):
        blob_calls.append(node["path"])
        return step_body

    monkeypatch.setattr(d, "_download_freecad_blob", fake_blob)
    (root / "step" / "0000_Alpha.step").write_bytes(b"corrupt")
    (root / "stl" / "0000_Alpha.stl").write_bytes(stl_body)
    ds = d.DATASETS["freecad-library"]
    first = d._manifest_for(ds, mode="manifest", refs=1)
    first.entries = [
        {
            "id": "freecad-library/Mechanical Parts/A/Alpha",
            "license": "CC-BY-3.0",
            "license_tier": ds.tier,
            "files": {
                "step": d.file_record(root, root / "step" / "0000_Alpha.step"),
                "stl": d.file_record(root, root / "stl" / "0000_Alpha.stl"),
            },
        }
    ]
    first.write(root)
    stl_rec = dict(first.entries[0]["files"]["stl"])
    wanted = {"freecad-library/Mechanical Parts/A/Alpha": hashlib.sha256(step_body).hexdigest()}
    got = d.fetch_freecad_refs(ds, root, wanted)
    assert blob_calls == ["Mechanical Parts/A/Alpha.step"]
    assert (root / "step" / "0000_Alpha.step").read_bytes() == step_body
    assert (root / "stl" / "0000_Alpha.stl").read_bytes() == stl_body
    assert got.entries[0]["files"]["stl"] == stl_rec
    assert (
        got.entries[0]["files"]["step"]["sha256"]
        == wanted["freecad-library/Mechanical Parts/A/Alpha"]
    )


def test_fetch_manifest_refs_rejects_unknown_dataset_without_network(tmp_path):
    corpus = {
        "entries": [
            {
                "tier": "imported",
                "source": {"dataset": "nope", "file_id": "x", "sha256": "s"},
            }
        ]
    }
    with pytest.raises(RuntimeError, match="--manifest cannot fetch"):
        d.fetch_manifest_refs(corpus, tmp_path)


def test_imported_tier_fetch_params_cover_pinned_datasets():
    assert set(d.IMPORTED_TIER_FETCH) == {"nist-pmi", "freecad-library"}
    assert "--manifest corpus/v0.json" in d.IMPORTED_TIER_COMMAND


def test_overall_tier_is_computed_from_entries():
    red = {"license_tier": d.REDISTRIBUTABLE}
    dl = {"license_tier": d.DOWNLOAD_ONLY}
    assert d.overall_tier([red, red], d.DOWNLOAD_ONLY) == d.REDISTRIBUTABLE
    assert d.overall_tier([red, dl], d.REDISTRIBUTABLE) == d.DOWNLOAD_ONLY
    assert d.overall_tier([], d.REDISTRIBUTABLE) == d.REDISTRIBUTABLE
    manifest = d._manifest_for(d.DATASETS["thingi10k"])
    manifest.entries = [red, dl]
    manifest.finalize()
    assert manifest.tier == d.DOWNLOAD_ONLY
    manifest.entries = [red]
    manifest.finalize()
    assert manifest.tier == d.REDISTRIBUTABLE


def test_abc_md5_mismatch_quarantines_archive(tmp_path, monkeypatch):
    archive = tmp_path / "_archive" / "abc_0000_meta_v00.7z"
    archive.parent.mkdir()
    archive.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        d._abc_archive(tmp_path, "meta")
    assert not archive.exists()
    assert archive.with_name(archive.name + ".bad").exists()
