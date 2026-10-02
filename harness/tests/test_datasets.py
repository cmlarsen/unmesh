import hashlib
import io
import json
import zipfile

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
    assert (
        d.classify_thingi_license("Attribution - Non-Commercial - Share Alike")[0]
        == d.DOWNLOAD_ONLY
    )
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


def test_non_commercial_dataset_requires_explicit_terms():
    ds = d.DATASETS["fusion360-segmentation"]
    with pytest.raises(SystemExit):
        d.check_terms(ds, set())
    d.check_terms(ds, {"fusion360-gallery"})
    d.check_terms(d.DATASETS["nist-pmi"], set())


def test_range_file_serves_zip_without_full_download():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.stp", b"A" * 5000)
        z.writestr("b.stp", b"B" * 5000)
    blob = buf.getvalue()
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return blob[start : end + 1]

    z = zipfile.ZipFile(io.BufferedReader(d.RangeFile(len(blob), fetch), 64))
    assert z.read("b.stp") == b"B" * 5000
    assert calls


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
