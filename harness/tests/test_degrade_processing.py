import copy
import math
import subprocess
import sys

import numpy as np
import pytest
from build123d import Box, Cylinder, Sphere

from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.degrade.processing import CONFIDENCE_KEY, MASK_THRESHOLD, transfer_labels
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, outward_normals, tessellate

from .test_labels import closed_manifold_problems

LIN, ANG = DEFLECTION_SETTINGS[0]
SEED = 11

PROC_OPS = (
    "quadric_decimation",
    "isotropic_remesh",
    "laplacian_smoothing",
    "taubin_smoothing",
    "vertex_clustering",
)
WATERTIGHT = {
    "quadric_decimation": False,
    "isotropic_remesh": True,
    "laplacian_smoothing": True,
    "taubin_smoothing": True,
    "vertex_clustering": False,
}
SMOOTHING = ("laplacian_smoothing", "taubin_smoothing")


def box_mesh():
    return tessellate(Box(10, 10, 10), LIN, ANG)


def cylinder_mesh():
    return tessellate(Cylinder(5, 20), 0.05, 0.3)


def sphere_mesh():
    return tessellate(Sphere(10), 0.05, 0.3)


@pytest.fixture(scope="module")
def box():
    return box_mesh()


@pytest.fixture(scope="module")
def cyl():
    return cylinder_mesh()


@pytest.fixture(scope="module")
def sph():
    return sphere_mesh()


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
    for name in PROC_OPS:
        degrade.apply(name, box, 0.6, SEED)
    assert "pymeshlab" not in sys.modules
    assert "gpytoolbox" not in sys.modules
    assert "skimage" not in sys.modules


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


def test_identity_relabel_through_path_a():
    total = correct = confident = 0
    for e in select(load_manifest(), "smoke"):
        mesh = tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG)
        out = copy.deepcopy(mesh)
        transfer_labels(out, mesh.tris, mesh.face_id)
        assert np.array_equal(out.face_id, mesh.face_id), e["id"]
        conf = np.array(out.metadata[CONFIDENCE_KEY])
        assert (conf >= MASK_THRESHOLD).mean() >= 0.99, e["id"]
        total += len(mesh.tris)
        correct += int((out.face_id == mesh.face_id).sum())
        confident += int((conf >= MASK_THRESHOLD).sum())
    assert correct == total
    assert confident / total >= 0.99


def test_identity_relabel_keeps_coplanar_faces():
    entries = select(load_manifest(), "smoke")
    e = next(x for x in entries if x["id"] == "thin_walls-0001")
    mesh = tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG)
    assert 14 in set(mesh.face_id.tolist())
    out = copy.deepcopy(mesh)
    transfer_labels(out, mesh.tris, mesh.face_id)
    assert set(out.face_id.tolist()) == set(mesh.face_id.tolist())
    assert (out.face_id[mesh.face_id == 14] == 14).all()


def test_decimate_reduces_triangles_with_severity(cyl, smoke_pair):
    for mesh in [cyl] + [m for _, m in smoke_pair]:
        counts = [
            len(degrade.apply("quadric_decimation", mesh, s, SEED).tris) for s in (0.3, 0.6, 1.0)
        ]
        assert counts[0] >= counts[1] >= counts[2]
        assert counts[0] < len(mesh.tris)
        assert counts[2] <= len(mesh.tris)


def test_decimate_quadric_planar_first(cyl):
    before = {f: int((cyl.face_id == f).sum()) for f in sorted(set(cyl.face_id.tolist()))}
    assert before[0] == 84
    out = degrade.apply("quadric_decimation", cyl, 0.6, SEED)
    after = {f: int((out.face_id == f).sum()) for f in sorted(set(out.face_id.tolist()))}
    assert min(after.get(1, 0), after.get(2, 0)) <= 4
    assert after.get(0, 0) > min(after.get(1, 0), after.get(2, 0))
    assert after.get(0, 0) >= 16
    params = out.metadata["history"][-1]["params"]
    assert params["keep_ratio_achieved"] <= params["keep_ratio_target"] + 0.15


def test_decimate_extreme_has_floor(box):
    out = degrade.apply("quadric_decimation", box, 1.0, SEED)
    assert 4 <= len(out.tris) <= 6
    assert_processing_labels_aligned(box, out, "quadric_decimation")


def test_decimated_normals_agree_with_source(cyl):
    out = degrade.apply("quadric_decimation", cyl, 0.6, SEED)
    tris = np.asarray(out.tris)
    got = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    for f in cyl.faces:
        sel = np.nonzero(out.face_id == f.id)[0]
        if len(sel) == 0:
            continue
        want = outward_normals(f, tris[sel].mean(axis=1))
        dots = (got[sel] * want).sum(axis=1)
        assert (dots > 0).all(), f.id


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
        assert params["collapses"] > 0
        assert params["flips"] > 0


def _edge_cv(mesh) -> float:
    edges = set()
    for tri in np.asarray(mesh.tris):
        for i in range(3):
            a, b = tuple(tri[i]), tuple(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    lengths = np.array([math.dist(a, b) for a, b in edges])
    return float(lengths.std() / lengths.mean())


def test_remesh_halves_edge_cv(cyl):
    out = degrade.apply("isotropic_remesh", cyl, 1.0, SEED)
    assert _edge_cv(out) <= 0.5 * _edge_cv(cyl)


def test_remesh_confidence_stays_high(cyl):
    out = degrade.apply("isotropic_remesh", cyl, 0.5, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["fraction_below_0_9"] < 0.25
    assert params["mean_confidence"] > 0.9


def test_remeshed_normals_agree_with_source(cyl):
    out = degrade.apply("isotropic_remesh", cyl, 1.0, SEED)
    tris = np.asarray(out.tris)
    got = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    seen = set()
    for f in cyl.faces:
        sel = np.nonzero(out.face_id == f.id)[0]
        if len(sel) == 0:
            continue
        seen.add(f.id)
        want = outward_normals(f, tris[sel].mean(axis=1))
        assert ((got[sel] * want).sum(axis=1) > 0).all(), f.id
    assert seen == set(out.face_id.tolist())


def test_smoothing_keeps_connectivity_labels_exact(box, cyl, smoke_pair):
    for name in SMOOTHING:
        for _, mesh in [(None, box), (None, cyl)] + smoke_pair:
            out = degrade.apply(name, mesh, 0.6, SEED)
            assert len(out.tris) == len(mesh.tris)
            assert np.array_equal(out.face_id, mesh.face_id)
            assert out.metadata[CONFIDENCE_KEY] == [1.0] * len(mesh.tris)
            before = np.asarray(mesh.tris).reshape(-1, 3)
            after = np.asarray(out.tris).reshape(-1, 3)
            mapping: dict[tuple, tuple] = {}
            for b, a in zip(before.tolist(), after.tolist(), strict=True):
                assert mapping.setdefault(tuple(b), tuple(a)) == tuple(a)
            assert not np.array_equal(before, after)
            uniq_before, inv_before = np.unique(before, axis=0, return_inverse=True)
            uniq_after, _ = np.unique(after, axis=0, return_inverse=True)
            assert len(uniq_before) == len(uniq_after)
            params = out.metadata["history"][-1]["params"]
            measured = float(np.linalg.norm(after - before, axis=1).max())
            assert params["max_displacement_mm"] == pytest.approx(measured)
            assert measured > 0


def test_smoothing_displacement_budget(cyl):
    flat = np.asarray(cyl.tris).reshape(-1, 3)
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    for name in SMOOTHING:
        prev = 0.0
        for severity in (0.1, 0.5, 1.0):
            out = degrade.apply(name, cyl, severity, SEED)
            params = out.metadata["history"][-1]["params"]
            assert params["displacement_budget_mm"] == pytest.approx(severity * 0.01 * diagonal)
            assert params["max_displacement_mm"] <= params["displacement_budget_mm"] * (1 + 1e-9)
            assert params["max_displacement_mm"] >= prev
            prev = params["max_displacement_mm"]
        full = degrade.apply(name, cyl, 1.0, SEED).metadata["history"][-1]["params"]
        assert full["max_displacement_mm"] == pytest.approx(0.01 * diagonal)


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


def test_smoothing_iterations_scale_with_severity(cyl):
    for name in SMOOTHING:
        counts = [
            out_steps(degrade.apply(name, cyl, s, SEED).metadata["history"][-1]["params"])
            for s in (0.25, 0.5, 1.0)
        ]
        assert counts[0] <= counts[1] <= counts[2]
        assert counts[2] > 0


def out_steps(params):
    return params.get("iterations", params.get("pairs"))


def test_taubin_shrinks_less_than_laplacian(sph):
    def volume(t):
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)

    ref = volume(np.asarray(sph.tris))
    lap = volume(np.asarray(degrade.apply("laplacian_smoothing", sph, 1.0, SEED).tris))
    tau = volume(np.asarray(degrade.apply("taubin_smoothing", sph, 1.0, SEED).tris))
    assert abs(tau - ref) < abs(lap - ref)
    params = degrade.apply("taubin_smoothing", sph, 1.0, SEED).metadata["history"][-1]["params"]
    assert params["pairs"] == 10
    assert params["mu"] == pytest.approx(-0.53)


def test_cluster_cell_scales_with_severity(box):
    from unmesh_harness.degrade.processing import _weld, cluster_cell_mm

    V, F = _weld(np.asarray(box.tris))
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    median_edge = float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))
    flat = np.asarray(box.tris).reshape(-1, 3)
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    for severity in (0.5, 1.0):
        out = degrade.apply("vertex_clustering", box, severity, SEED)
        params = out.metadata["history"][-1]["params"]
        assert params["cell_mm"] == pytest.approx(cluster_cell_mm(severity, median_edge, diagonal))
        assert params["median_edge_mm"] == pytest.approx(median_edge)
        assert params["occupied_cells"] > 0


def test_cluster_changes_mesh_at_low_severity(cyl):
    out = degrade.apply("vertex_clustering", cyl, 0.05, SEED)
    assert len(out.tris) < len(cyl.tris)
    assert not np.array_equal(out.tris, cyl.tris)
    assert_processing_labels_aligned(cyl, out, "vertex_clustering")


def test_cluster_coarsens_and_documents_drops(cyl):
    out = degrade.apply("vertex_clustering", cyl, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["triangles_after"] <= params["triangles_before"]
    assert (
        params["triangles_before"]
        == params["triangles_after"] + params["degenerate_dropped"] + params["duplicates_dropped"]
    )


def test_smoothing_polylines_move_with_mesh(cyl):
    for name in SMOOTHING:
        out = degrade.apply(name, cyl, 1.0, SEED)
        known = {tuple(p) for p in out.tris.reshape(-1, 3).tolist()}
        for adj in out.adjacency:
            assert all(tuple(p) in known for p in adj.points)
        assert all(tuple(v) in known for v in out.vertices)


def test_largest_smoke_part_under_two_seconds():
    entries = select(load_manifest(), "smoke")
    meshes = [
        (tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG), e["id"]) for e in entries
    ]
    mesh, pid = max(meshes, key=lambda t: len(t[0].tris))
    import time

    for name in PROC_OPS:
        start = time.perf_counter()
        out = degrade.apply(name, mesh, 1.0, SEED)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, (name, pid, elapsed)
        assert len(out.tris) > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("name", PROC_OPS)
def test_50k_part_under_ten_seconds(name):
    import time

    mesh = tessellate(Sphere(10), 0.002, 0.05)
    assert len(mesh.tris) > 50000
    start = time.perf_counter()
    out = degrade.apply(name, mesh, 0.6, SEED)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, (name, elapsed)
    assert len(out.tris) > 0


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
