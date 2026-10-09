import pytest

from unmesh_harness.acceptance import (
    CONVERTER,
    FACETED,
    baseline_table,
    failures,
    render,
    render_baselines,
    table,
)


def record(part, family, converter=CONVERTER, **kw):
    base = {
        "part": part,
        "family": family,
        "operator": "identity",
        "severity": 0.0,
        "seed": 0,
        "converter": converter,
        "status": "ok",
        "faces": 6,
        "regions": 6,
        "matched": 6,
        "validity": {"valid": True, "problems": []},
        "under_report": False,
        "dev_truth_max": 0.005,
    }
    base.update(kw)
    return base


def line(result, prefix):
    return next(x for x in result["lines"] if x["line"].startswith(prefix))


def test_all_good_passes():
    records = [
        record("stepped_block-a", "stepped_block"),
        record("stepped_block-a", "stepped_block", FACETED, dev_truth_max=0.004),
    ]
    result = table(records, 0.01)
    assert line(result, "F1 planes")["pass"]
    assert line(result, "validity")["pass"]
    assert line(result, "calibration")["pass"]
    assert line(result, "deviation to truth")["pass"]
    assert not line(result, "F1 bores")["pass"]
    assert "| F1 planes >= 0.98 | 1.000 | 1.000 | PASS |" in render(result)


def test_failures_are_counted_not_hidden():
    records = [
        record("stepped_block-a", "stepped_block", matched=5),
        record("stepped_block-b", "stepped_block", status="timeout", error="exceeded 60 s"),
        record("round_boss-c", "round_boss", validity={"valid": False, "problems": ["no STEP"]}),
        record("round_boss-d", "round_boss", under_report=True, dev_truth_max=0.03),
        record("round_boss-d", "round_boss", FACETED, dev_truth_max=0.01),
    ]
    result = table(records, 0.01)
    planes = line(result, "F1 planes")
    assert planes["rows"][("identity", 0.0)]["failed"] == 1 and not planes["pass"]
    assert line(result, "validity")["value"] == 0.5
    assert not line(result, "calibration")["pass"]
    truth = line(result, "deviation to truth")
    assert not truth["pass"]
    listed = "\n".join(failures(records, 0.01))
    assert "stepped_block-b identity 0 seed 0: timeout" in listed
    assert "round_boss-c identity 0 seed 0: invalid: no STEP" in listed
    assert "round_boss-d identity 0 seed 0: under-reports" in listed
    assert "round_boss-d identity 0 seed 0: deviation to truth 30.0 um, faceted 10.0 um" in listed


def test_timed_out_cell_without_family_fails_its_line():
    good = record("stepped_block-0000", "stepped_block")
    timed_out = {
        "part": "stepped_block-0001",
        "operator": "identity",
        "severity": 0.0,
        "seed": 0,
        "converter": CONVERTER,
        "status": "timeout",
        "error": "exceeded 60 s",
    }
    result = table([good, timed_out], 0.01)
    planes = line(result, "F1 planes")
    assert planes["rows"][("identity", 0.0)] == {
        "f1": pytest.approx(2 / 3),
        "cells": 2,
        "failed": 1,
    }
    assert not planes["pass"]
    assert line(result, "validity")["value"] == 0.5
    assert not line(result, "calibration")["pass"]
    assert not line(result, "deviation to truth")["pass"]


def test_baseline_targets_take_the_best_baseline_plus_margin():
    records = [
        record("round_boss-a", "round_boss", matched=6),
        record("round_boss-a", "round_boss", FACETED, matched=0, regions=40),
        record("round_boss-a", "round_boss", "freecad-refine", matched=3, regions=30),
        record("round_boss-a", "round_boss", "stl2step", status="timeout", error="exceeded"),
    ]
    result = baseline_table(records, [FACETED, "freecad-refine", "stl2step"])
    boss = next(t for t in result["targets"] if t["group"] == "round_boss")
    assert boss["best_baseline"] == "freecad-refine"
    assert boss["target"] == pytest.approx(2 * 3 / 30 * 3 / 6 / (3 / 30 + 3 / 6) + 0.10)
    assert boss["pass"]
    assert result["f1"]["stl2step"]["round_boss"][("identity", 0.0)] == 0.0
    assert result["stats"]["stl2step"][("identity", 0.0)]["timeouts"] == 1
    text = render_baselines(result)
    assert "### faceted" in text and "| round_boss | identity 0 | freecad-refine |" in text
