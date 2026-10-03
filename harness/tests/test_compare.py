from __future__ import annotations

import json
from pathlib import Path

from unmesh_harness.cli import main as cli_main
from unmesh_harness.runner.compare import (
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


def test_missing_cells_are_listed_not_flagged():
    comp = run(
        [record(part="a"), record(part="b")],
        [record(part="a")],
    )
    assert not comp.regressions
    assert len(comp.only_a) == 1 and not comp.only_b
    assert "ONLY-A" in format_comparison(comp)


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
