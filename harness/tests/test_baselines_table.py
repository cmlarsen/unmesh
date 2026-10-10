from __future__ import annotations

import importlib.util


def _load():
    from unmesh_harness.runner.grid import repo_root

    path = repo_root() / "harness" / "scripts" / "baselines_table.py"
    spec = importlib.util.spec_from_file_location("baselines_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rec(part, operator="identity", severity=0.0, seed=0, converter="unmesh", **fields):
    record = {
        "part": part,
        "operator": operator,
        "severity": severity,
        "seed": seed,
        "converter": converter,
        "status": "ok",
        "family": part.rsplit("-", 1)[0],
    }
    record.update(fields)
    return record


def test_family_derived_from_part_id_without_family_field():
    mod = _load()
    assert mod.family({"part": "complex_void-0003"}) == "complex_void"
    assert mod.family({"part": "imported-0225"}) == "imported"
    assert mod.family({"part": "boss_plate-0001"}) == "boss_plate"


def test_timeout_without_family_is_attributed_and_counted():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec("alpha-0001", status="timeout", error="exceeded 60 s"),
    ]
    records[1].pop("family")
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 2 | 1.000 | 0% | 50% | 1 |" in out
    assert "| **all families** | 2 | 1.000 | 0% | 50% | 1 |" in out


def test_skipped_is_excluded_from_denominator_and_reported():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec(
            "alpha-0001",
            status="skipped",
            error="crack_seam does not apply",
            skipped_operator="crack_seam",
        ),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 1 | 1.000 | 0% | 0% | 0 |" in out
    assert "skipped as inapplicable: 1" in out


def test_ground_truth_prep_failure_is_harness_not_converter():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec("alpha-0001", status="error", error="ground-truth preparation failed"),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 1 | 1.000 | 0% | 0% | 0 |" in out
    assert "harness failures (excluded): 1 (1 ground-truth preparation, 0 degradation)" in out


def test_degradation_error_is_harness_not_converter():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec(
            "alpha-0001",
            status="error",
            error="ValueError: noise_off_plane has no vertex interior to a planar face",
            traceback=['  File ".../unmesh_harness/degrade/core.py", line 87, in apply'],
        ),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 1 | 1.000 | 0% | 0% | 0 |" in out
    assert "harness failures (excluded): 1 (0 ground-truth preparation, 1 degradation)" in out


def test_converter_error_counts_as_failure():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec("alpha-0001", status="error", error="converter returned no IR"),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 2 | 1.000 | 0% | 50% | 0 |" in out


def test_external_tool_timeout_counts_in_timeout_column():
    mod = _load()
    records = [
        _rec("alpha-0000", converter="stl2step", f1=1.0),
        _rec(
            "alpha-0001",
            converter="stl2step",
            status="error",
            error="TimeoutError: stl2step exceeded 54.9 s",
        ),
    ]
    out = mod.render(records, ["stl2step"])
    assert "| alpha | 2 | 1.000 | 0% | 50% | 1 |" in out


def test_all_families_row_aggregates_across_families():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0),
        _rec("beta-0000", f1=0.0),
        _rec("beta-0001", status="timeout", error="exceeded 60 s"),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| **all families** | 3 | 0.500 | 0% | 33% | 1 |" in out


def test_noise_rows_split_into_their_own_section():
    mod = _load()
    records = [
        _rec("alpha-0000", operator="identity", f1=1.0),
        _rec("alpha-0001", operator="noise_normal", f1=0.0),
    ]
    out = mod.render(records, ["unmesh"])
    clean, noise = out.split("## Noise rows")
    assert "| alpha | 1 | 1.000 | 0% | 0% | 0 |" in clean
    assert "| alpha | 1 | 0.000 | 0% | 0% | 0 |" in noise


def test_no_analytic_flag_and_fallback():
    mod = _load()
    records = [
        _rec("alpha-0000", f1=1.0, analytic_regions=2),
        _rec("alpha-0001", f1=1.0, analytic_regions=0),
        _rec("alpha-0002", f1=1.0, fallback=True),
    ]
    out = mod.render(records, ["unmesh"])
    assert "| alpha | 3 | 1.000 | 67% | 0% | 0 |" in out


def test_converter_missing_from_records_is_skipped():
    mod = _load()
    records = [_rec("alpha-0000", converter="unmesh", f1=1.0)]
    out = mod.render(records, ["unmesh", "faceted"])
    assert "### unmesh" in out
    assert "### faceted" not in out
