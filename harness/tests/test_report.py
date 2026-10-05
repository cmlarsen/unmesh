from __future__ import annotations

import dataclasses
import json

from unmesh_harness.runner import load_grid, run_grid
from unmesh_harness.runner.report import build_html


def tiny_grid():
    grid = load_grid("smoke")
    return dataclasses.replace(
        grid,
        entries=grid.entries[:2],
        cells=[c for c in grid.cells if c["operator"] in ("identity", "rotation")],
        seeds=[0],
        step_deviation_sample=(1, 1),
    )


def run_tiny(tmp_path):
    grid = tiny_grid()
    summary = run_grid(
        grid, ["unmesh"], tmp_path, jobs=2, sha="r", timeout=120, log=lambda *_: None
    )
    assert summary.ran > 0
    return summary


def test_build_html_sections_without_records():
    page = build_html([], "smoke")
    assert "<!DOCTYPE html>" in page
    assert "Heatmaps" in page and "Breaking-point" in page and "Worst parts" in page
    assert "http" not in page


def test_report_html_on_tiny_run(tmp_path):
    from unmesh_harness.cli import main

    run_tiny(tmp_path)
    out = tmp_path / "report.html"
    rc = main(
        ["report", str(tmp_path), "--grid", "smoke", "-o", str(out)],
    )
    assert rc == 0
    text = out.read_text()
    assert "Heatmaps" in text
    assert "Breaking-point" in text
    assert "Worst parts" in text
    assert "drag to rotate" in text
    assert "canvas.mesh" in text
    assert '"viewers"' in text
    assert '"ntris"' in text
    data = json.loads(text.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert data["heat"], "heatmap data is empty"
    for metric in ("f1", "mean_triangle_iou", "dev_input_p99", "valid", "under_report"):
        assert any(metric in row for row in data["heat"]), metric
    assert "http" not in text
    assert out.stat().st_size < 15 * 1024 * 1024


def test_report_text_still_works_on_dir(tmp_path, capsys):
    from unmesh_harness.cli import main

    run_tiny(tmp_path)
    rc = main(["report", str(tmp_path), "--grid", "smoke"])
    assert rc == 0
    assert "converter" in capsys.readouterr().out


def _page_data(records, grid="t"):
    from unmesh_harness.runner.report import build_html

    text = build_html(records, grid)
    return text, json.loads(text.split("const DATA = ", 1)[1].split(";\n", 1)[0])


def test_heat_all_is_mean_over_all_matching_records():
    records = []
    specs = [
        ("alpha", "0-10", "0-10", 0.2),
        ("beta", "0-10", "0-10", 1.0),
        ("beta", "11-50", "100+", 1.0),
    ]
    for family, face, feat, f1 in specs:
        records.append(
            {
                "part": f"{family}-0",
                "converter": "unmesh",
                "family": family,
                "strata": {"face_bucket": face, "feature_bucket": feat},
                "operator": "coarsen",
                "severity": 0.5,
                "seed": 0,
                "status": "ok",
                "f1": f1,
                "mean_triangle_iou": f1,
                "dev_input_p99": 0.001,
                "valid": True,
                "under_report": False,
            }
        )
    text, data = _page_data(records)
    assert "heat_all" in data
    (marginal,) = [r for r in data["heat_all"] if r["operator"] == "coarsen"]
    assert marginal["n"] == 3
    assert marginal["f1"] == (0.2 + 1.0 + 1.0) / 3
    direct = sum(r["f1"] for r in records) / len(records)
    assert marginal["f1"] == direct
    weighted = sum(r["f1"] * r["n"] for r in data["heat"]) / sum(r["n"] for r in data["heat"])
    assert weighted == direct
    assert marginal["valid"] == 1.0
    assert marginal["under_report"] == 0.0


def test_curves_ignore_bucket_filters():
    records = [
        {
            "part": "p-0",
            "converter": "unmesh",
            "family": "planar",
            "strata": {"face_bucket": "1-10", "feature_bucket": "0-10"},
            "operator": "coarsen",
            "severity": sev,
            "seed": 0,
            "status": "ok",
            "f1": 0.9,
            "mean_triangle_iou": 0.9,
            "dev_input_p99": 0.001,
            "valid": True,
            "under_report": False,
        }
        for sev in (0.25, 0.5, 1.0)
    ]
    text, data = _page_data(records)
    assert data["curves"], "curve rows are empty"
    body = text.split("function renderCurves()", 1)[1].split("function rotMatrix", 1)[0]
    assert "f-face" not in body and "f-feat" not in body
    assert "f-conv" in body and "f-fam" in body


def test_worst_and_failed_lists():
    records = [
        {
            "part": f"p-{i}",
            "converter": "unmesh",
            "family": "planar",
            "strata": {},
            "operator": "coarsen",
            "severity": 0.5,
            "seed": 0,
            "status": "ok",
            "f1": 0.5,
            "valid": True,
        }
        for i in range(2)
    ]
    records.append(
        {
            "part": "p-9",
            "converter": "unmesh",
            "family": "planar",
            "strata": {},
            "operator": "coarsen",
            "severity": 0.5,
            "seed": 1,
            "status": "timeout",
            "error": "exceeded 60 s",
        }
    )
    records[0]["f1"] = float("nan")
    _, data = _page_data(records)
    assert all(r["f1"] is None or 0.0 <= r["f1"] <= 1.0 for r in data["worst"])
    assert data["failed_total"] == 1
    assert data["failed"][0]["part"] == "p-9"
    assert "Timeouts and errors" in _page_data(records)[0]


def _ok_record(part="p-0", f1=1.0, **fields):
    return {
        "part": part,
        "converter": "unmesh",
        "family": "planar",
        "strata": {"face_bucket": "1-10", "feature_bucket": "0-10"},
        "operator": "coarsen",
        "severity": 0.5,
        "seed": 0,
        "status": "ok",
        "f1": f1,
        "mean_triangle_iou": f1,
        "dev_input_p99": 0.001,
        "valid": True,
        "under_report": False,
        **fields,
    }


def _skipped_record(part="p-9"):
    return {
        "part": part,
        "converter": "unmesh",
        "family": "planar",
        "strata": {"face_bucket": "1-10", "feature_bucket": "0-10"},
        "operator": "crack_seam",
        "severity": 0.5,
        "seed": 0,
        "status": "skipped",
        "skipped_operator": "crack_seam",
        "error": "crack_seam does not apply",
    }


def test_failed_cells_excludes_skipped():
    from unmesh_harness.runner.report import failed_cells

    records = [_ok_record(), _skipped_record(), _ok_record("p-1", status="timeout", error="boom")]
    assert [r["part"] for r in failed_cells(records)] == ["p-1"]


def test_html_counts_skipped_and_excludes_them_from_means():
    records = [_ok_record("p-0", f1=0.5), _ok_record("p-1", f1=1.0), _skipped_record()]
    _, data = _page_data(records)
    assert data["cells"] == 3 and data["ok"] == 2 and data["skipped"] == 1
    assert data["failed_total"] == 0 and data["failed"] == []
    (marginal,) = [r for r in data["heat_all"] if r["operator"] == "coarsen"]
    assert marginal["n"] == 2 and marginal["f1"] == 0.75
