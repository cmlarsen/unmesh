import json

import pytest
from build123d import Box, Cylinder

from unmesh_harness.labels import tessellate
from unmesh_harness.metrics.structure import score_validity
from unmesh_harness.oracle import build_oracle_ir

LIN, ANG = 0.01, 0.2


def _write_box_step(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(10, 10, 10), LIN, ANG)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "box.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid and report.fallback is None
    return out, mesh


def test_validity_oracle_box_step_is_valid(tmp_path):
    out, _ = _write_box_step(tmp_path)
    result = score_validity(out)
    assert result["valid"], result["problems"]
    assert (result["solids"], result["shells"]) == (1, 1)
    assert result["volume"] == pytest.approx(1000.0)
    assert result["volume_positive"] is True
    assert result["brepcheck_valid"] is True
    assert result["max_shape_tolerance"] <= result["tolerance_bound"] == pytest.approx(1e-3)
    assert result["tolerance_ok"] is True
    assert result["fallback"] is False
    json.dumps(result)


def test_validity_cavity_expects_two_shells(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(30, 30, 30) - Box(10, 10, 10), 0.1, 0.5)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "cavity.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid, report.issues
    one_shell = score_validity(out)
    assert one_shell["valid"] is False
    assert any("shell" in p for p in one_shell["problems"])
    two_shells = score_validity(out, expected_solids=1, expected_shells=2)
    assert two_shells["valid"], two_shells["problems"]
    assert (two_shells["solids"], two_shells["shells"]) == (1, 2)
    json.dumps(two_shells)


def test_validity_missing_step_is_invalid(tmp_path):
    result = score_validity(tmp_path / "absent.step")
    assert result["valid"] is False
    assert result["problems"] == ["no STEP file written"]
    assert result["solids"] is None
    empty = tmp_path / "empty.step"
    empty.write_text("")
    assert score_validity(empty)["problems"] == ["no STEP file written"]
    json.dumps(result)


def test_validity_reports_fallback_without_failing(tmp_path):
    out, _ = _write_box_step(tmp_path)
    result = score_validity(out, write={"fallback": "faceted", "fallback_reason": "test"})
    assert result["fallback"] is True
    assert result["valid"] is True
    assert score_validity(out, write={"fallback": None})["fallback"] is False


def test_validity_through_bore_counts_one_solid(tmp_path):
    from unmesh import step

    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), LIN, ANG)
    ir = build_oracle_ir(mesh)
    out = tmp_path / "bore.step"
    report = step.write(ir, out, mesh=mesh.tris)
    assert report.valid, report.issues
    result = score_validity(out)
    assert result["valid"], result["problems"]
    assert (result["solids"], result["shells"]) == (1, 1)
    assert result["volume"] == pytest.approx(1000 - 3.14159265 * 4 * 10, rel=1e-3)
