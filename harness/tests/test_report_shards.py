from __future__ import annotations

import json


def _record(part="p-0", sha="abc123", **extra):
    base = {
        "part": part,
        "operator": "identity",
        "severity": 0.0,
        "seed": 0,
        "converter": "unmesh",
        "git_sha": sha,
        "grid_hash": "h",
        "plugin_hash": "",
        "status": "ok",
        "f1": 1.0,
        "faces": 4,
        "regions": 4,
        "matched": 4,
        "valid": True,
        "under_report": False,
        "calibration": 0.0,
        "step_problems": [],
        "fallback": False,
        "dev_input_max": 0.0,
        "dev_truth_max": 0.0,
        "seconds": 1.0,
    }
    base.update(extra)
    return base


def _write(path, records):
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")


def test_report_fails_on_missing_shards(tmp_path, capsys):
    from unmesh_harness.cli import main

    _write(tmp_path / "smoke-planar-0-of-2.jsonl", [_record()])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"shards": ["smoke-planar-0-of-2", "smoke-planar-1-of-2"]}))
    rc = main(["report", str(tmp_path), "--grid", "smoke", "--expect-plan", str(plan)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "PARTIAL" in out and "smoke-planar-1-of-2" in out


def test_report_passes_when_all_shards_present(tmp_path, capsys):
    from unmesh_harness.cli import main

    _write(tmp_path / "smoke-planar-0-of-2.jsonl", [_record(part="p-0")])
    _write(tmp_path / "smoke-planar-1-of-2.jsonl", [_record(part="p-1")])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"shards": ["smoke-planar-0-of-2", "smoke-planar-1-of-2"]}))
    rc = main(["report", str(tmp_path), "--grid", "smoke", "--expect-plan", str(plan)])
    assert rc == 0


def test_report_fails_on_mixed_git_shas(tmp_path, capsys):
    from unmesh_harness.cli import main

    _write(tmp_path / "a.jsonl", [_record(sha="aaa")])
    _write(tmp_path / "b.jsonl", [_record(sha="bbb")])
    rc = main(["report", str(tmp_path), "--grid", "smoke"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "aaa" in out and "bbb" in out


def test_report_fails_on_empty_and_unknown_sha(tmp_path, capsys):
    from unmesh_harness.cli import main

    assert main(["report", str(tmp_path), "--grid", "smoke"]) == 2
    assert "no records" in capsys.readouterr().out
    _write(tmp_path / "a.jsonl", [_record(sha="aaa")])
    assert main(["report", str(tmp_path), "--grid", "smoke", "--git-sha", "zzz"]) == 2


def test_report_html_builds_viewers_without_converter(tmp_path):
    from unmesh_harness.cli import main
    from unmesh_harness.runner import load_grid

    grid = load_grid("smoke")
    part = grid.entries[0]["id"]
    _write(tmp_path / "smoke.jsonl", [_record(part=part, f1=0.4)])
    out = tmp_path / "report.html"
    rc = main(["report", str(tmp_path), "--grid", "smoke", "-o", str(out)])
    assert rc == 0
    data = json.loads(out.read_text().split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert data["viewers"], "viewer payload is empty"
    key = next(iter(data["viewers"]))
    assert data["viewers"][key]["ntris"] > 0
