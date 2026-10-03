import subprocess
import sys

import numpy as np
import pytest
from build123d import Box, Cylinder

from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.degrade.processing import CONFIDENCE_KEY, MASK_THRESHOLD
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, tessellate

from .test_labels import closed_manifold_problems

LIN, ANG = DEFLECTION_SETTINGS[0]
SEED = 11

PROC_OPS = ("quadric_decimation", "isotropic_remesh", "laplacian_smoothing")
WATERTIGHT = {"quadric_decimation": True, "isotropic_remesh": True, "laplacian_smoothing": True}
SMOOTHING = ("laplacian_smoothing",)


def box_mesh():
    return tessellate(Box(10, 10, 10), LIN, ANG)


def cylinder_mesh():
    return tessellate(Cylinder(5, 20), 0.05, 0.3)


@pytest.fixture(scope="module")
def box():
    return box_mesh()


@pytest.fixture(scope="module")
def cyl():
    return cylinder_mesh()


@pytest.fixture(scope="module")
def smoke_pair():
    entries = select(load_manifest(), "smoke")
    out = []
    for e in (entries[0], entries[3]):
        out.append((e["id"], tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG)))
    return out


def assert_processing_labels_aligned(before: LabeledMesh, after: LabeledMesh, name: str) -> None:
    assert len(after.face_id) == len(after.tris)
    assert set(after.face_id.tolist()) <= set(before.face_id.tolist())
    conf = after.metadata.get(CONFIDENCE_KEY)
    assert conf is not None
    assert len(conf) == len(after.tris)
    assert all(0.0 <= c <= 1.0 for c in conf)
    assert [f.__dict__ for f in after.faces] == [f.__dict__ for f in before.faces]
    assert [s.__dict__ for s in after.shells] == [s.__dict__ for s in before.shells]
    if name in SMOOTHING:
        assert [(a.edge_id, a.face_a, a.face_b) for a in after.adjacency] == [
            (a.edge_id, a.face_a, a.face_b) for a in before.adjacency
        ]
        assert len(after.vertices) == len(before.vertices)
    else:
        assert [a.__dict__ for a in after.adjacency] == [a.__dict__ for a in before.adjacency]
        assert after.vertices == before.vertices
    params = after.metadata["history"][-1]["params"]
    assert params["triangles_before"] == len(before.tris)
    assert params["triangles_after"] == len(after.tris)
    assert params["fraction_below_0_9"] == pytest.approx(
        float((np.array(conf) < MASK_THRESHOLD).mean())
    )


def test_registry_covers_processing_operators():
    assert set(PROC_OPS) <= set(OPERATORS)
    for name in PROC_OPS:
        op = OPERATORS[name]
        assert op.family == "processing"
        assert op.severity_0 and op.severity_1
        assert not op.changes_frame
        assert op.preserves_watertight == WATERTIGHT[name]


@pytest.mark.parametrize("name", PROC_OPS)
def test_severity_zero_is_identity(name, box):
    out = degrade.apply(name, box, 0.0, SEED)
    assert np.array_equal(out.tris, box.tris)
    assert np.array_equal(out.face_id, box.face_id)
    assert CONFIDENCE_KEY not in out.metadata
    assert out.metadata["history"][-1]["op"] == name


@pytest.mark.parametrize("name", PROC_OPS)
def test_labels_aligned_and_tables_untouched(name, box, cyl, smoke_pair):
    for _, mesh in [(None, box), (None, cyl)] + smoke_pair:
        out = degrade.apply(name, mesh, 0.6, SEED)
        assert_processing_labels_aligned(mesh, out, name)
        assert out.metadata["history"][-1]["params"]


@pytest.mark.parametrize("name", PROC_OPS)
def test_deterministic_in_process(name, cyl):
    a = degrade.apply(name, cyl, 0.6, SEED)
    b = degrade.apply(name, cyl, 0.6, SEED)
    assert np.array_equal(a.tris, b.tris)
    assert np.array_equal(a.face_id, b.face_id)
    assert a.metadata == b.metadata


def test_deterministic_across_processes(tmp_path):
    src = tmp_path / "mesh.npz"
    cylinder_mesh().save(src)
    script = (
        "import sys, hashlib\n"
        "from unmesh_harness import degrade\n"
        "from unmesh_harness.labels import LabeledMesh\n"
        f"names = {list(PROC_OPS)!r}\n"
        "m = LabeledMesh.load(sys.argv[1])\n"
        "h = hashlib.sha256()\n"
        "for n in names:\n"
        "    o = degrade.apply(n, m, 0.6, 99)\n"
        "    h.update(o.tris.tobytes())\n"
        "    h.update(o.face_id.tobytes())\n"
        "    h.update(repr(o.metadata).encode())\n"
        "print(h.hexdigest())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(src)], capture_output=True, text=True, check=True
    ).stdout.strip()
    loaded = LabeledMesh.load(src)
    digest = __import__("hashlib").sha256()
    for n in PROC_OPS:
        o = degrade.apply(n, loaded, 0.6, 99)
        digest.update(o.tris.tobytes())
        digest.update(o.face_id.tobytes())
        digest.update(repr(o.metadata).encode())
    assert proc == digest.hexdigest()


def test_no_gpl_dependency_required(box):
    degrade.apply("quadric_decimation", box, 0.6, SEED)
    assert "pymeshlab" not in sys.modules


@pytest.mark.parametrize("name", PROC_OPS)
def test_watertight_per_flag(name, box, cyl):
    op = OPERATORS[name]
    for mesh in (box, cyl):
        assert closed_manifold_problems(mesh.tris)[0] == []
        out = degrade.apply(name, mesh, 0.6, SEED)
        if op.preserves_watertight:
            assert closed_manifold_problems(out.tris)[0] == [], name
        else:
            assert_produces_labels_only(name, out, mesh)


def assert_produces_labels_only(name, out, mesh):
    assert_processing_labels_aligned(mesh, out, name)


def test_history_params_survive_save_load(box, tmp_path):
    out = degrade.apply("quadric_decimation", box, 0.6, SEED)
    out.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert back.metadata == out.metadata
    assert np.array_equal(back.tris, out.tris)
    assert np.array_equal(back.face_id, out.face_id)


def test_decimate_reduces_triangles_with_severity(cyl, smoke_pair):
    for mesh in [cyl] + [m for _, m in smoke_pair]:
        counts = [
            len(degrade.apply("quadric_decimation", mesh, s, SEED).tris) for s in (0.3, 0.6, 1.0)
        ]
        assert counts[0] >= counts[1] >= counts[2]
        assert counts[0] < len(mesh.tris)
        assert counts[2] <= len(mesh.tris)


def test_decimate_keeps_every_face_represented(cyl):
    out = degrade.apply("quadric_decimation", cyl, 1.0, SEED)
    assert set(out.face_id.tolist()) == set(cyl.face_id.tolist())
    params = out.metadata["history"][-1]["params"]
    assert params["triangles_after"] <= params["triangles_before"]
    assert 0.0 < params["keep_ratio_achieved"] <= 1.0


def test_remesh_target_edge_scales_with_severity(cyl):
    from unmesh_harness.degrade.processing import target_edge_mm

    flat = cyl.tris.reshape(-1, 3)
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    for severity in (0.5, 1.0):
        out = degrade.apply("isotropic_remesh", cyl, severity, SEED)
        params = out.metadata["history"][-1]["params"]
        assert params["target_edge_mm"] == pytest.approx(target_edge_mm(severity, diagonal))
        assert params["bbox_diagonal_mm"] == pytest.approx(diagonal)
        assert params["passes"] == 1 + round(2 * severity)
        assert params["splits"] > 0


def test_remesh_uniformizes_edge_lengths(cyl):
    def chord_std(mesh):
        edges = set()
        for tri in mesh.tris:
            for i in range(3):
                a, b = tuple(tri[i]), tuple(tri[(i + 1) % 3])
                edges.add((min(a, b), max(a, b)))
        import math

        return float(np.std([math.dist(a, b) for a, b in edges]))

    before = chord_std(cyl)
    after = chord_std(degrade.apply("isotropic_remesh", cyl, 1.0, SEED))
    assert after < before


def test_remesh_high_confidence_on_smooth_refinement(box):
    out = degrade.apply("isotropic_remesh", box, 0.5, SEED)
    assert len(out.tris) > len(box.tris)
    assert out.metadata["history"][-1]["params"]["fraction_below_0_9"] == 0.0


def test_smoothing_keeps_connectivity_and_moves_vertices_once(cyl):
    for name in SMOOTHING:
        out = degrade.apply(name, cyl, 1.0, SEED)
        assert len(out.tris) == len(cyl.tris)
        before = cyl.tris.reshape(-1, 3)
        after = out.tris.reshape(-1, 3)
        mapping: dict[tuple, tuple] = {}
        for b, a in zip(before.tolist(), after.tolist(), strict=True):
            assert mapping.setdefault(tuple(b), tuple(a)) == tuple(a)
        assert not np.array_equal(before, after)
        params = out.metadata["history"][-1]["params"]
        measured = float(np.linalg.norm(after - before, axis=1).max())
        assert params["max_displacement_mm"] == pytest.approx(measured)
        assert measured > 0


def test_smoothing_is_graded_and_label_stable_on_fine_meshes(box):
    fine = degrade.apply("refine", box, 0.3, 0)
    assert len(fine.tris) > len(box.tris)
    disps = []
    for name in SMOOTHING:
        for severity in (0.2, 0.5, 1.0):
            out = degrade.apply(name, fine, severity, SEED)
            params = out.metadata["history"][-1]["params"]
            disps.append(params["max_displacement_mm"])
            assert np.array_equal(out.face_id, fine.face_id)
            assert params["triangles_after"] == params["triangles_before"] == len(fine.tris)
        assert disps[0] <= disps[1] <= disps[2]
        assert disps[0] > 0


def test_smoothing_at_full_severity_relabels_with_low_confidence(cyl):
    for name in SMOOTHING:
        out = degrade.apply(name, cyl, 1.0, SEED)
        assert len(out.tris) == len(cyl.tris)
        assert_processing_labels_aligned(cyl, out, name)
        assert not np.array_equal(out.tris, cyl.tris)
        assert out.metadata["history"][-1]["params"]["iterations"] > 0


def test_smoothing_iterations_scale_with_severity(cyl):
    for name in SMOOTHING:
        counts = [
            degrade.apply(name, cyl, s, SEED).metadata["history"][-1]["params"]["iterations"]
            for s in (0.25, 0.5, 1.0)
        ]
        assert counts[0] <= counts[1] <= counts[2]
        assert counts[2] > 0


def test_smoothing_polylines_move_with_mesh(cyl):
    for name in SMOOTHING:
        out = degrade.apply(name, cyl, 1.0, SEED)
        known = {tuple(p) for p in out.tris.reshape(-1, 3).tolist()}
        for adj in out.adjacency:
            assert all(tuple(p) in known for p in adj.points)
        assert all(tuple(v) in known for v in out.vertices)


@pytest.mark.slow
def test_processing_sweep_full_smoke():
    lin, ang = DEFLECTION_SETTINGS[0]
    for e in select(load_manifest(), "smoke"):
        mesh = tessellate(generate(e["family"], e["seed"]).solid, lin, ang)
        for name in PROC_OPS:
            for severity in (0.5, 1.0):
                out = degrade.apply(name, mesh, severity, SEED)
                assert_processing_labels_aligned(mesh, out, name)
                if OPERATORS[name].preserves_watertight:
                    assert closed_manifold_problems(out.tris)[0] == [], (e["id"], name, severity)
