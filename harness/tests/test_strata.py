from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.runner.results import summarize
from unmesh_harness.strata import (
    STRATA_LIN,
    face_bucket,
    feature_bucket,
)


def test_every_entry_carries_complexity_strata():
    for entry in load_manifest()["entries"]:
        strata = entry["strata"]
        assert strata["category"] in (
            "planar",
            "curved",
            "chamfer_fillet",
            "ambiguity",
            "complex",
            "imported",
        )
        assert strata["face_count"] == entry["fingerprint"]["face_count"]
        assert strata["face_bucket"] == face_bucket(strata["face_count"])
        assert strata["min_feature_ratio"] > 0
        assert strata["feature_bucket"] == feature_bucket(strata["min_feature_ratio"])


def test_face_buckets():
    assert [face_bucket(n) for n in (1, 10, 11, 50, 51, 300, 301, 4000)] == [
        "0-10",
        "0-10",
        "11-50",
        "11-50",
        "51-300",
        "51-300",
        "301+",
        "301+",
    ]


def test_feature_buckets_relative_to_lin():
    assert STRATA_LIN == 0.01
    assert [feature_bucket(r) for r in (0.5, 1.0, 9.9, 10.0, 99.9, 100.0, 5000.0)] == [
        "<1",
        "1-10",
        "1-10",
        "10-100",
        "10-100",
        "100+",
        "100+",
    ]


def _record(converter, group, faces=10, regions=10, matched=10):
    return {
        "status": "ok",
        "converter": converter,
        "operator": "identity",
        "severity": 0.0,
        "family": "plate_pockets",
        "strata": {"category": "planar", **group},
        "faces": faces,
        "regions": regions,
        "matched": matched,
        "f1": 1.0,
        "dev_input_max": 0.0,
        "calibration": 0.0,
        "seconds": 0.1,
        "fallback": False,
        "valid": True,
        "topology": {},
    }


def test_summarize_groups_by_complexity_strata():
    records = [
        _record("unmesh", {"face_bucket": "0-10", "feature_bucket": "100+"}),
        _record("unmesh", {"face_bucket": "51-300", "feature_bucket": "1-10"}),
    ]
    text = summarize(records)
    assert "by face_bucket (micro over ok cells):" in text
    assert "by feature_bucket (micro over ok cells):" in text
    assert "0-10" in text and "51-300" in text and "1-10" in text


def test_smoke_covers_the_main_face_buckets():
    smoke = {e["strata"]["face_bucket"] for e in select(load_manifest(), "smoke")}
    assert {"0-10", "11-50", "51-300"} <= smoke
