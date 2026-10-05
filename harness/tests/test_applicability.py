from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
from build123d import Box, Cylinder

from unmesh_harness import degrade
from unmesh_harness.degrade import OPERATORS, applies
from unmesh_harness.degrade.refine import UnsupportedSurface, implicit
from unmesh_harness.degrade.retriangulate import _bridge, _clip
from unmesh_harness.labels import LabeledMesh, tessellate


@pytest.fixture(scope="module")
def box():
    return tessellate(Box(10, 10, 10), 0.1, 0.5)


@pytest.fixture(scope="module")
def cyl():
    return tessellate(Cylinder(5, 20), 0.05, 0.3)


@pytest.fixture(scope="module")
def torus():
    from unmesh_harness.groundtruth import generate

    return tessellate(generate("revolved_torus", 0).solid, 0.01, 0.2)


def test_guarded_operators_declare_predicates():
    assert OPERATORS["noise_off_plane"].applies_to is not None
    assert OPERATORS["crack_seam"].applies_to is not None


def test_noise_off_plane_needs_interior_planar_vertex(box, torus):
    assert not applies("noise_off_plane", box)
    assert not applies("noise_off_plane", torus)
    refined = degrade.apply("refine", box, 0.5, 0)
    assert applies("noise_off_plane", refined)


def test_crack_seam_needs_interior_edge(box):
    assert applies("crack_seam", box)
    doubled = copy.deepcopy(box)
    doubled.tris = np.concatenate([doubled.tris] * 4)
    doubled.face_id = np.concatenate([doubled.face_id] * 4)
    assert not applies("crack_seam", doubled)


def test_implicit_rejects_unsupported_surface(cyl):
    face = next(f for f in cyl.faces if f.surface != "plane")
    exotic = copy.copy(face)
    exotic.surface = "bspline"
    exotic.params = {}
    with pytest.raises(UnsupportedSurface):
        implicit(exotic, np.array([[0.0, 0.0, 0.0]]))


def with_exotic_face(mesh):
    out = copy.deepcopy(mesh)
    idx = next(i for i, f in enumerate(out.faces) if f.surface != "plane")
    out.faces[idx].surface = "bspline"
    out.faces[idx].params = {}
    return out


@pytest.mark.parametrize("name", ["t_junctions", "slivers", "refine", "nonuniform_chords"])
def test_projection_fallback_on_exotic_faces(name, cyl):
    mesh = with_exotic_face(cyl)
    first = degrade.apply(name, mesh, 0.5, 3)
    second = degrade.apply(name, mesh, 0.5, 3)
    assert np.array_equal(first.tris, second.tris)
    assert len(first.tris) > 0
    assert len(first.face_id) == len(first.tris)


def test_stray_stored_vertices_survive_noise_and_coarsen(box):
    stray = [1e6, -2e6, 3e6]
    for name in ("noise_isotropic", "noise_normal", "coarsen"):
        mesh = copy.deepcopy(box)
        mesh.vertices = [list(v) for v in mesh.vertices] + [list(stray)]
        out = degrade.apply(name, mesh, 0.5, 3)
        assert out.vertices[-1] == pytest.approx(stray)
        again = degrade.apply(name, mesh, 0.5, 3)
        assert np.array_equal(out.tris, again.tris)


def test_bridge_pins_hole_choice():
    outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    assert _bridge(outer, [hole]) == [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (0.0, 4.0),
    ]


def test_clip_triangulates_l_shape():
    poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    tris = _clip(poly, "fan", np.random.default_rng(0))
    assert tris is not None and len(tris) == 4
    area = sum(
        abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        for a, b, c in tris
    )
    assert area == pytest.approx(3.0)


def mini_grid():
    from unmesh_harness.runner import load_grid

    grid = load_grid("smoke")
    identity = next(c for c in grid.cells if c["operator"] == "identity")
    noise = {"operator": "noise_off_plane", "severity": 0.5, "steps": [["noise_off_plane", 0.5]]}
    return dataclasses.replace(grid, entries=grid.entries[:1], cells=[identity, noise])


def test_run_skips_inapplicable_rows(tmp_path):
    from unmesh_harness.runner import gate, run_grid, summarize
    from unmesh_harness.runner.results import read_results

    grid = mini_grid()
    summary = run_grid(grid, ["unmesh"], tmp_path, jobs=2, sha="s", log=lambda *_: None)
    assert summary.skipped_inapplicable == 3
    records = read_results(summary.results_path)
    by_status = [(r["operator"], r["status"]) for r in records]
    assert ("identity", "ok") in by_status
    assert by_status.count(("noise_off_plane", "skipped")) == 3
    assert all(r["skipped_operator"] == "noise_off_plane" for r in records if r["status"] == "skipped")
    assert gate(records, grid, ["unmesh"], "s").passed
    assert "skipped as inapplicable: 3 cells" in summarize(records)

    again = run_grid(grid, ["unmesh"], tmp_path, jobs=2, sha="s", log=lambda *_: None)
    assert (again.ran, again.skipped) == (0, 4)
    assert again.skipped_inapplicable == 3


def test_expand_skip_omits_cells():
    from unmesh_harness.runner import load_grid

    grid = load_grid("smoke")
    full = grid.expand(["unmesh"], "s")
    (cell, _) = full[0]
    skip = frozenset({(cell.part, cell.operator, cell.severity, cell.seed)})
    partial = grid.expand(["unmesh"], "s", skip=skip)
    assert len(partial) == len(full) - 1
    assert all(c.key != cell.key for c, _ in partial)


def test_labeled_mesh_roundtrip_keeps_stray_vertices(tmp_path, box):
    mesh = copy.deepcopy(box)
    mesh.vertices = [list(v) for v in mesh.vertices] + [[1e6, -2e6, 3e6]]
    mesh.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert back.vertices[-1] == pytest.approx([1e6, -2e6, 3e6])


def _fin_mesh(box):
    out = degrade.apply("nonmanifold_fin", box, 0.5, 3)
    assert (out.face_id == -1).any()
    return out


def _oracle_ir(mesh):
    from unmesh_harness.oracle import build_oracle_ir

    return build_oracle_ir(mesh)


def test_recovery_scoring_ignores_unknown_faces(box):
    from unmesh_harness.metrics.recovery import score_recovery

    degraded = _fin_mesh(box)
    result = score_recovery(box, degraded.face_id, _oracle_ir(degraded), np.eye(4))
    assert result["segmentation"]["triangles"] == len(degraded.tris)
    assert result["f1"] == pytest.approx(1.0)


def test_topology_scoring_ignores_unknown_faces(box):
    from unmesh_harness.metrics.structure import score_topology

    degraded = _fin_mesh(box)
    result = score_topology(box, degraded.face_id, degraded.tris, _oracle_ir(degraded))
    assert result["faces_match"] is True


def test_stray_shells_score_without_crash(box):
    from unmesh_harness.metrics.recovery import score_recovery
    from unmesh_harness.metrics.structure import score_topology

    degraded = degrade.apply("stray_shells", box, 0.5, 3)
    assert (degraded.face_id == -1).any()
    ir = _oracle_ir(degraded)
    assert score_recovery(box, degraded.face_id, ir, np.eye(4))["f1"] == pytest.approx(1.0)
    assert score_topology(box, degraded.face_id, degraded.tris, ir)["faces_match"] is True
