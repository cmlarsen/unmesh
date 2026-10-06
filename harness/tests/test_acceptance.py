from unmesh_harness.acceptance import CONVERTER, FACETED, failures, render, table


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
        record("a", "stepped_block"),
        record("a", "stepped_block", FACETED, dev_truth_max=0.004),
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
        record("a", "stepped_block", matched=5),
        record("b", "stepped_block", status="timeout", error="exceeded 60 s"),
        record("c", "round_boss", validity={"valid": False, "problems": ["no STEP"]}),
        record("d", "round_boss", under_report=True, dev_truth_max=0.03),
        record("d", "round_boss", FACETED, dev_truth_max=0.01),
    ]
    result = table(records, 0.01)
    planes = line(result, "F1 planes")
    assert planes["rows"][("identity", 0.0)]["failed"] == 1 and not planes["pass"]
    assert line(result, "validity")["value"] == 0.5
    assert not line(result, "calibration")["pass"]
    truth = line(result, "deviation to truth")
    assert not truth["pass"]
    listed = "\n".join(failures(records, 0.01))
    assert "b identity 0 seed 0: timeout" in listed
    assert "c identity 0 seed 0: invalid: no STEP" in listed
    assert "d identity 0 seed 0: under-reports" in listed
    assert "d identity 0 seed 0: deviation to truth 30.0 um, faceted 10.0 um" in listed
