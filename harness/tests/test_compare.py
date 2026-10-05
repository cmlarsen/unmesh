from __future__ import annotations

import json
from pathlib import Path

from unmesh_harness.cli import main as cli_main
from unmesh_harness.runner.compare import (
    compare_files,
    compare_groups,
    format_comparison,
    group_records,
)


def record(part="box", operator="identity", severity=0.0, seed=0, **fields):
    base = {
        "part": part,
        "operator": operator,
        "severity": severity,
        "seed": seed,
        "converter": "unmesh",
        "git_sha": "s",
        "grid_hash": "g",
        "plugin_hash": "",
        "status": "ok",
        "f1": 1.0,
        "faces": 6,
        "regions": 6,
        "matched": 6,
        "valid": True,
        "under_report": False,
        "calibration": 0.0,
        "step_problems": [],
        "fallback": False,
        "dev_input_max": 0.0,
        "dev_truth_max": 0.0,
    }
    return base | fields


def run(records_a, records_b, converter="unmesh"):
    return compare_groups(group_records(records_a, converter), group_records(records_b, converter))


def seeds(f1s, **fields):
    return [record(seed=i, f1=f, **fields) for i, f in enumerate(f1s)]


def test_flags_regression():
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([0.9, 0.9, 0.9]))
    assert len(comp.regressions) == 1
    (v,) = comp.regressions
    assert (v.metric, v.verdict) == ("f1", "REGRESSION")
    assert "REGRESSION" in format_comparison(comp)


def test_flags_improvement_symmetrically():
    comp = run(seeds([0.9, 0.9, 0.9]), seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions
    assert len(comp.improvements) == 1
    assert comp.improvements[0].metric == "f1"


def test_within_noise_is_not_flagged():
    comp = run(seeds([0.90, 1.0, 0.95]), seeds([0.93, 1.0, 0.90]))
    assert not comp.regressions and not comp.improvements
    assert comp.unchanged > 0


def test_small_move_below_floor_is_not_flagged():
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([0.995, 0.995, 0.995]))
    assert not comp.regressions and not comp.improvements


def test_deviation_floor_is_one_micron():
    perfect = seeds([1.0, 1.0, 1.0], dev_input_max=0.0)
    comp = run(perfect, seeds([1.0, 1.0, 1.0], dev_input_max=0.0005))
    assert not comp.regressions
    comp = run(perfect, seeds([1.0, 1.0, 1.0], dev_input_max=0.005))
    assert any(v.metric == "dev_input_max" and v.verdict == "REGRESSION" for v in comp.regressions)


def test_validity_floor_is_zero():
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([1.0, 1.0, 1.0], valid=False))
    assert any(v.metric == "valid" and v.verdict == "REGRESSION" for v in comp.regressions)


def test_failed_cell_counts_as_regression():
    bad = seeds([1.0, 1.0, 1.0])
    bad[0] = record(seed=0, status="error", error="boom")
    comp = run(seeds([1.0, 1.0, 1.0]), bad)
    assert any(v.metric == "f1" and v.verdict == "REGRESSION" for v in comp.regressions)


def test_cell_missing_in_b_is_a_regression():
    comp = run(
        [record(part="a"), record(part="b")],
        [record(part="a")],
    )
    assert len(comp.only_a) == 1 and not comp.only_b
    assert [v.metric for v in comp.regressions] == ["missing"]
    assert "ONLY-A" in format_comparison(comp)


def test_cell_only_in_b_is_listed_not_flagged():
    comp = run(
        [record(part="a")],
        [record(part="a"), record(part="b")],
    )
    assert not comp.regressions
    assert not comp.only_a and len(comp.only_b) == 1
    assert "ONLY-B" in format_comparison(comp)


def noisy_cell(part="noisy-part", seed=0, **fields):
    return record(
        part=part,
        operator="noise_isotropic",
        severity=0.02,
        seed=seed,
        f1=0.5,
        faces=14,
        regions=16,
        matched=7,
        valid=False,
        fallback=True,
        under_report=False,
        dev_input_max=0.006,
        **fields,
    )


def test_new_grid_cells_bootstrap_as_only_b():
    old_a = seeds([1.0, 1.0, 1.0])
    new_cells = [noisy_cell(seed=i) for i in range(3)]
    comp = run(old_a, old_a + new_cells)
    assert not comp.regressions
    assert not comp.only_a and len(comp.only_b) == 1
    assert comp.only_b[0][0] == "noisy-part"


def test_stable_bad_compare_only_cell_is_unchanged():
    old = seeds([1.0, 1.0, 1.0]) + [noisy_cell(seed=i) for i in range(3)]
    new = seeds([1.0, 1.0, 1.0]) + [noisy_cell(seed=i) for i in range(3)]
    comp = run(old, new)
    assert not comp.regressions and not comp.improvements
    assert not comp.only_a and not comp.only_b


def test_bootstrap_still_flags_existing_cell_regressions():
    old = seeds([1.0, 1.0, 1.0])
    new_cells = [noisy_cell(seed=i) for i in range(3)]
    comp = run(old, seeds([0.9, 0.9, 0.9]) + new_cells)
    assert any(v.metric == "f1" and v.verdict == "REGRESSION" for v in comp.regressions)
    assert len(comp.only_b) == 1
    comp = run(old + [record(part="old-part")], old)
    assert [v.metric for v in comp.regressions] == ["missing"]


def test_dropped_seed_in_b_is_a_regression():
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([1.0, 1.0]))
    assert [v.metric for v in comp.regressions] == ["seeds"]
    assert not comp.improvements


def test_extra_seed_in_b_is_fine():
    comp = run(seeds([1.0, 1.0]), seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions and not comp.improvements


def test_samples_missing_in_b_are_a_regression():
    comp = run(
        seeds([1.0, 1.0, 1.0], dev_input_max=0.0),
        seeds([1.0, 1.0, 1.0], dev_input_max=None),
    )
    assert any(v.metric == "dev_input_max" and v.verdict == "REGRESSION" for v in comp.regressions)
    assert "n/a" in format_comparison(comp)


def test_samples_missing_in_a_are_skipped():
    comp = run(
        seeds([1.0, 1.0, 1.0], dev_input_max=None),
        seeds([1.0, 1.0, 1.0], dev_input_max=0.0),
    )
    assert not comp.regressions


def test_noisy_rate_transition_is_a_regression():
    a = seeds([1.0, 1.0, 1.0], under_report=False)[:2] + [record(seed=2, under_report=True)]
    comp = run(a, seeds([1.0, 1.0, 1.0], under_report=True))
    (v,) = [v for v in comp.regressions if v.metric == "under_report"]
    assert v.threshold == 0.0
    assert not comp.improvements


def test_partial_validity_loss_is_a_regression():
    a = seeds([1.0, 1.0, 1.0])[:2] + [record(seed=2, valid=False)]
    comp = run(a, seeds([1.0, 1.0, 1.0], valid=False))
    assert any(v.metric == "valid" and v.verdict == "REGRESSION" for v in comp.regressions)


def test_rate_recovery_is_an_improvement():
    a = seeds([1.0, 1.0, 1.0], valid=False)
    comp = run(a, seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions
    assert any(v.metric == "valid" and v.verdict == "IMPROVEMENT" for v in comp.improvements)


def test_steady_rate_is_unchanged():
    a = seeds([1.0, 1.0, 1.0])[:2] + [record(seed=2, valid=False)]
    b = seeds([1.0, 1.0, 1.0])[:2] + [record(seed=2, valid=False)]
    comp = run(a, b)
    assert not comp.regressions and not comp.improvements


def test_latest_duplicate_record_wins():
    b = [record(seed=0, status="error", error="boom")] + seeds([1.0, 1.0, 1.0])
    comp = run(seeds([1.0, 1.0, 1.0]), b)
    assert not comp.regressions and not comp.improvements


def test_trailing_duplicate_error_counts():
    b = seeds([1.0, 1.0, 1.0]) + [record(seed=0, status="error", error="boom")]
    comp = run(seeds([1.0, 1.0, 1.0]), b)
    assert any(v.metric == "f1" and v.verdict == "REGRESSION" for v in comp.regressions)


def test_records_from_older_runs_are_ignored():
    old = [record(seed=i, git_sha="old", grid_hash="old", f1=0.0) for i in range(3)]
    comp = run(seeds([1.0, 1.0, 1.0]), old + seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions and not comp.improvements
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([1.0, 1.0, 1.0]) + old)
    assert comp.regressions


def test_records_from_other_grids_are_ignored():
    other = [record(seed=i, grid_hash="other", f1=0.0) for i in range(3)]
    comp = run(seeds([1.0, 1.0, 1.0]), other + seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions and not comp.improvements


def test_other_converters_are_filtered():
    comp = run(
        [record(converter="faceted", f1=0.0)],
        [record(converter="faceted", f1=1.0)],
        converter="unmesh",
    )
    assert not comp.regressions and not comp.improvements
    assert not comp.only_a and not comp.only_b


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    return path


def test_cli_exit_codes(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", seeds([1.0, 1.0, 1.0]))
    b = write_jsonl(tmp_path / "b.jsonl", seeds([0.9, 0.9, 0.9]))
    assert cli_main(["compare", str(a), str(a)]) == 0
    assert cli_main(["compare", str(a), str(b)]) == 1
    assert cli_main(["compare", str(b), str(a)]) == 0


def test_empty_a_is_an_error(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [])
    b = write_jsonl(tmp_path / "b.jsonl", [record(status="error", error="boom")])
    comp = compare_files(a, b)
    assert comp.error and "empty" in comp.error
    assert "a.jsonl" in comp.error
    assert cli_main(["compare", str(a), str(b)]) == 2


def test_a_without_converter_records_is_an_error(tmp_path):
    recs = [record(converter="faceted", f1=0.5)]
    a = write_jsonl(tmp_path / "a.jsonl", recs)
    b = write_jsonl(tmp_path / "b.jsonl", seeds([0.9, 0.9, 0.9]))
    comp = compare_files(a, b)
    assert comp.error and "faceted" not in comp.error
    assert "unmesh" in comp.error
    assert cli_main(["compare", str(a), str(b)]) == 2


def test_empty_b_is_an_error(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", seeds([1.0, 1.0, 1.0]))
    b = write_jsonl(tmp_path / "b.jsonl", [])
    comp = compare_files(a, b)
    assert comp.error and "empty" in comp.error
    assert comp.regressions
    assert cli_main(["compare", str(a), str(b)]) == 2


def test_b_without_converter_records_is_an_error(tmp_path):
    recs = [record(converter="faceted", f1=0.5)]
    b = write_jsonl(tmp_path / "b.jsonl", recs)
    a = write_jsonl(tmp_path / "a.jsonl", seeds([1.0, 1.0, 1.0]))
    comp = compare_files(a, b)
    assert comp.error and "faceted" not in comp.error
    assert "unmesh" in comp.error
    assert cli_main(["compare", str(a), str(b)]) == 2


def test_unscorable_cell_is_not_scored_as_perfect_or_zero():
    from unmesh_harness.runner.compare import metric_samples
    from unmesh_harness.runner.results import _cell_violations, summarize

    blank = record(f1=None, recall=None, precision=None, faces=0, regions=0, matched=0)
    assert metric_samples([blank, record(f1=0.5)], "f1") == [0.5]
    assert not any("F1" in v for v in _cell_violations(blank, {"f1_cell": 0.9}))
    row = summarize([blank | {"seconds": 1.0, "topology": None}])
    assert "nan" in row.splitlines()[2]
    comp = run(seeds([1.0, 1.0]), [blank | {"seed": 0}, blank | {"seed": 1}])
    assert not comp.improvements


def skipped(seed=0, **fields):
    return record(
        seed=seed,
        status="skipped",
        skipped_operator="noise_off_plane",
        error="noise_off_plane does not apply",
        **fields,
    )


def test_ok_to_skipped_is_coverage_not_regression():
    comp = run(seeds([1.0, 1.0, 1.0]), [skipped(i) for i in range(3)])
    assert not comp.regressions and not comp.improvements
    assert len(comp.coverage_changed) == 1
    assert "scored in A, skipped in B" in comp.coverage_changed[0].note
    text = format_comparison(comp)
    assert "COVERAGE-CHANGED" in text and "0 regression(s)" in text


def test_skipped_to_ok_is_coverage_not_regression():
    comp = run([skipped(i) for i in range(3)], seeds([1.0, 1.0, 1.0]))
    assert not comp.regressions and not comp.improvements
    assert len(comp.coverage_changed) == 1
    assert "skipped in A, scored in B" in comp.coverage_changed[0].note


def test_skipped_to_skipped_is_silent():
    comp = run([skipped(i) for i in range(3)], [skipped(i) for i in range(3)])
    assert not comp.regressions and not comp.improvements
    assert not comp.coverage_changed and not comp.only_a and not comp.only_b


def test_partially_skipped_cell_compares_shared_scored_seeds():
    comp = run(seeds([1.0, 1.0, 1.0]), seeds([1.0, 1.0]) + [skipped(2)])
    assert not comp.regressions and not comp.improvements
    assert len(comp.coverage_changed) == 1
    assert "2" in comp.coverage_changed[0].note


def test_skipped_cell_absent_elsewhere_is_coverage_not_missing():
    comp = run([skipped()], [])
    assert not comp.regressions and not comp.only_a
    assert len(comp.coverage_changed) == 1
    comp = run([], [skipped()])
    assert not comp.regressions and not comp.only_b
    assert len(comp.coverage_changed) == 1


def test_cli_ok_to_skipped_passes(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", seeds([1.0, 1.0, 1.0]))
    b = write_jsonl(tmp_path / "b.jsonl", [skipped(i) for i in range(3)])
    assert cli_main(["compare", str(a), str(b)]) == 0
    assert cli_main(["compare", str(b), str(a)]) == 0
