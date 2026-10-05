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
