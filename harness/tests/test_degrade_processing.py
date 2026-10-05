import copy
import dataclasses
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Cylinder, Sphere
from scipy.spatial import cKDTree

from unmesh.ir import Region
from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.degrade.processing import (
    CONFIDENCE_KEY,
    MASK_THRESHOLD,
    _exact_votes,
    _weld,
    sample_source,
    transfer_labels,
)
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, outward_normals, tessellate
from unmesh_harness.metrics.recovery import score_recovery
from unmesh_harness.oracle import build_oracle_ir

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
    "quadric_decimation": True,
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


@pytest.fixture(scope="module")
def smoke_parts():
    return [
        (e["id"], tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG))
        for e in select(load_manifest(), "smoke")
    ]


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


def test_identity_relabel_through_path_a(smoke_parts):
    total = correct = confident = 0
    for pid, mesh in smoke_parts:
        out = copy.deepcopy(mesh)
        transfer_labels(out, mesh.tris, mesh.face_id)
        assert np.array_equal(out.face_id, mesh.face_id), pid
        conf = np.array(out.metadata[CONFIDENCE_KEY])
        assert (conf >= MASK_THRESHOLD).mean() >= 0.99, pid
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


def test_decimate_planar_regions_collapse_to_minimal_first(cyl):
    fine = degrade.apply("refine", cyl, 0.5, 0)
    rim = int((cyl.face_id == 1).sum())
    before = np.bincount(fine.face_id, minlength=3)
    assert before[1] > 10 * rim and before[2] > 10 * rim
    for severity in (0.3, 0.6, 1.0):
        out = degrade.apply("quadric_decimation", fine, severity, SEED)
        after = np.bincount(out.face_id, minlength=3)
        assert after[1] <= rim + 2 and after[2] <= rim + 2, (severity, after)
        assert after[0] / before[0] > 3 * after[1] / before[1], (severity, after)
        params = out.metadata["history"][-1]["params"]
        assert params["keep_ratio_achieved"] <= params["keep_ratio_target"] + 0.01


def test_decimate_moves_vertices_and_stays_closed(cyl, smoke_pair):
    for mesh in [cyl] + [m for _, m in smoke_pair]:
        assert closed_manifold_problems(mesh.tris)[0] == []
        source = {tuple(p) for p in mesh.tris.reshape(-1, 3).tolist()}
        for severity in (0.3, 0.6, 1.0):
            out = degrade.apply("quadric_decimation", mesh, severity, SEED)
            assert closed_manifold_problems(out.tris)[0] == [], severity
            corners = {tuple(p) for p in out.tris.reshape(-1, 3).tolist()}
            assert corners - source, severity
            assert len(out.tris) < len(mesh.tris)


def test_decimate_open_input_keeps_its_boundary(box):
    opened = copy.deepcopy(box)
    opened.tris = opened.tris[1:]
    opened.face_id = opened.face_id[1:]
    boundary = {tuple(p) for p in box.tris[0].tolist()}
    out = degrade.apply("quadric_decimation", opened, 1.0, SEED)
    assert boundary <= {tuple(p) for p in out.tris.reshape(-1, 3).tolist()}


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


@pytest.mark.benchmark
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


ORACLE_SEVERITIES = (0.1, 0.5, 1.0)


def oracle_on_degraded(clean: LabeledMesh, degraded: LabeledMesh):
    base = build_oracle_ir(clean)
    surface = {r.id: r.surface for r in base.regions}
    regions = []
    for face in clean.faces:
        ids = np.nonzero(degraded.face_id == face.id)[0]
        if len(ids):
            regions.append(Region(len(regions), surface[face.id], ids.tolist(), None))
    return dataclasses.replace(base, regions=regions, adjacencies=[])


def oracle_score(clean: LabeledMesh, degraded: LabeledMesh) -> dict:
    return score_recovery(
        clean,
        degraded.face_id,
        oracle_on_degraded(clean, degraded),
        np.eye(4),
        degraded.metadata.get(CONFIDENCE_KEY),
    )


@pytest.mark.parametrize("name", PROC_OPS)
def test_oracle_on_degraded_labels_scores_one(name, smoke_parts):
    lost = 0
    for pid, mesh in smoke_parts:
        for severity in ORACLE_SEVERITIES:
            out = degrade.apply(name, mesh, severity, SEED)
            result = oracle_score(mesh, out)
            key = (pid, severity)
            if OPERATORS[name].preserves_watertight:
                assert closed_manifold_problems(out.tris)[0] == [], key
            assert result["recall"] == 1.0, key
            assert result["precision"] == 1.0, key
            assert result["f1"] == 1.0, key
            scored = [d for d in result["faces_detail"] if d["region"] is not None]
            assert all(d["iou"] == 1.0 for d in scored), key
            assert result["faces"] + result["faces_unrecoverable"] == len(mesh.faces), key
            lost += result["faces_unrecoverable"]
    if name in SMOOTHING:
        assert lost == 0
    else:
        assert lost > 0


def _normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _nearest_source(mesh: LabeledMesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts, owner = sample_source(mesh.tris)
    return _exact_votes(points, mesh.tris, owner, cKDTree(pts), 0.0)


@pytest.mark.parametrize("pid", ["thin_walls-0000", "thin_walls-0001"])
def test_remesh_thin_walls_have_no_back_facing_triangles(pid, smoke_parts):
    mesh = dict(smoke_parts)[pid]
    diagonal = float(np.linalg.norm(np.ptp(mesh.tris.reshape(-1, 3), axis=0)))
    for severity in (0.5, 1.0):
        out = degrade.apply("isotropic_remesh", mesh, severity, SEED)
        tri, _ = _nearest_source(mesh, out.tris.mean(axis=1))
        dots = (_normals(out.tris) * _normals(mesh.tris)[tri]).sum(axis=1)
        assert (dots > 0).all(), (severity, int((dots <= 0).sum()))
        _, d2 = _nearest_source(mesh, out.tris.reshape(-1, 3))
        assert float(np.sqrt(d2.max())) <= 1e-9 * diagonal


def test_remesh_keeps_sharp_edges(box):
    out = degrade.apply("isotropic_remesh", box, 1.0, SEED)
    corners = {tuple(p) for p in box.tris.reshape(-1, 3).tolist()}
    assert corners <= {tuple(p) for p in out.tris.reshape(-1, 3).tolist()}
    tri, _ = _nearest_source(box, out.tris.mean(axis=1))
    dots = (_normals(out.tris) * _normals(box.tris)[tri]).sum(axis=1)
    assert dots.min() > 1 - 1e-9


@pytest.mark.parametrize("name", ["isotropic_remesh", "quadric_decimation", "vertex_clustering"])
def test_low_confidence_sits_on_face_boundaries(name, cyl, smoke_pair):
    low_total = 0
    for mesh in [cyl] + [m for _, m in smoke_pair]:
        out = degrade.apply(name, mesh, 0.5, SEED)
        conf = np.asarray(out.metadata[CONFIDENCE_KEY])
        low = np.nonzero(conf < MASK_THRESHOLD)[0]
        low_total += len(low)
        if len(low) == 0:
            continue
        _, F = _weld(out.tris)
        labels: dict[int, set[int]] = {}
        for row, fid in zip(F.tolist(), out.face_id.tolist(), strict=True):
            for v in row:
                labels.setdefault(v, set()).add(fid)
        mixed = np.array([any(len(labels[v]) > 1 for v in F[t]) for t in low])
        assert mixed.all(), (name, int((~mixed).sum()))
    assert low_total > 0


CHAIN_TAILS = (("refine", 0.3), ("hole_patch", 0.5), ("duplicate_facets", 0.5))


@pytest.mark.parametrize("name", PROC_OPS)
@pytest.mark.parametrize("tail", CHAIN_TAILS, ids=lambda t: t[0])
def test_processing_chains_keep_confidence_aligned(name, tail, smoke_pair):
    for _, mesh in smoke_pair:
        out = degrade.chain(mesh, [(name, 0.5), tail], SEED)
        conf = out.metadata[CONFIDENCE_KEY]
        assert len(conf) == len(out.tris) == len(out.face_id)
        assert all(0.0 <= c <= 1.0 for c in conf)
        result = oracle_score(mesh, out)
        assert result["f1"] == 1.0, (name, tail)
        first = degrade.apply(name, mesh, 0.5, SEED)
        if tail[0] == "duplicate_facets":
            kept = np.asarray(conf[: len(first.tris)])
            assert np.array_equal(kept, np.asarray(first.metadata[CONFIDENCE_KEY]))


@pytest.mark.parametrize("name", PROC_OPS)
def test_processing_after_processing_caps_confidence(name, cyl):
    first = degrade.apply("isotropic_remesh", cyl, 1.0, SEED)
    assert min(first.metadata[CONFIDENCE_KEY]) < MASK_THRESHOLD
    fresh_input = copy.deepcopy(first)
    del fresh_input.metadata[CONFIDENCE_KEY]
    chained = degrade.apply(name, first, 0.5, SEED)
    fresh = degrade.apply(name, fresh_input, 0.5, SEED)
    assert np.array_equal(chained.tris, fresh.tris)
    a = np.asarray(chained.metadata[CONFIDENCE_KEY])
    b = np.asarray(fresh.metadata[CONFIDENCE_KEY])
    assert len(a) == len(chained.tris)
    assert (a <= b).all()
    assert (a < b).any()


def test_label_votes_prefer_source_facing_the_same_way():
    plate = tessellate(Box(20, 20, 0.2), LIN, ANG)
    top = next(f.id for f in plate.faces if f.params.get("normal", [0, 0, 0])[2] > 0.5)
    bottom = next(f.id for f in plate.faces if f.params.get("normal", [0, 0, 0])[2] < -0.5)
    probe = copy.deepcopy(plate)
    down = np.array([[[0.0, 0.0, -0.09], [1.0, -1.0, -0.09], [1.0, 1.0, -0.09]]])
    probe.tris = np.concatenate([down, down[:, ::-1]])
    probe.face_id = np.zeros(2, dtype=plate.face_id.dtype)
    transfer_labels(probe, plate.tris, plate.face_id)
    assert probe.face_id.tolist() == [top, bottom]


def test_projection_stays_on_the_vertex_faces():
    from unmesh_harness.degrade.processing import _Projector

    plate = tessellate(Box(20, 20, 0.2), LIN, ANG)
    names, labels = np.unique(plate.face_id, return_inverse=True)
    top = next(f.id for f in plate.faces if f.params.get("normal", [0, 0, 0])[2] > 0.5)
    lab = int(np.searchsorted(names, top))
    proj = _Projector(plate.tris, labels)
    point = np.array([[1.0, 2.0, -0.09]])
    keys = np.array([0 * proj.n_labels + lab])
    got = proj(point, np.array([[0]]), keys, require_all=True)
    assert got[0] == pytest.approx([1.0, 2.0, 0.1])


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "processing"
PINNED = {
    ("round_boss-0000", "quadric_decimation"): "b2903b0f0ba77528",
    ("round_boss-0000", "isotropic_remesh"): "2b73dc3ce9b67db8",
    ("round_boss-0000", "laplacian_smoothing"): "10b50fff713551a9",
    ("round_boss-0000", "taubin_smoothing"): "7319d108dddc5ec3",
    ("round_boss-0000", "vertex_clustering"): "293f63b680729c73",
    ("thin_walls-0001", "quadric_decimation"): "589af057120c6e30",
    ("thin_walls-0001", "isotropic_remesh"): "121eab548dd9b365",
    ("thin_walls-0001", "laplacian_smoothing"): "91248b810140338c",
    ("thin_walls-0001", "taubin_smoothing"): "19f01541fcf984ae",
    ("thin_walls-0001", "vertex_clustering"): "90171284528ebb7a",
}


@pytest.mark.parametrize(("part", "name"), sorted(PINNED))
def test_output_is_pinned_across_platforms(part, name):
    with np.load(FIXTURES / f"{part}.npz") as z:
        mesh = LabeledMesh(z["tris"], z["face_id"], [], [], 0.0, 0.0)
    out = degrade.apply(name, mesh, 0.5, SEED)
    blob = (
        out.tris.tobytes()
        + out.face_id.astype(np.int64).tobytes()
        + np.asarray(out.metadata[CONFIDENCE_KEY]).tobytes()
    )
    assert hashlib.sha256(blob).hexdigest()[:16] == PINNED[(part, name)]


def test_qem_refuses_a_collapse_that_folds_a_triangle(box):
    from unmesh_harness.degrade.processing import _Qem

    fine = degrade.apply("refine", box, 0.3, 0)
    V, F = _weld(fine.tris)
    qem = _Qem(V, F)
    top = float(V[:, 2].max())
    inner = [
        v for v in range(len(V)) if V[v, 2] == top and all(V[n, 2] == top for n in qem.ring(v))
    ]
    a = inner[0]
    b = min(qem.ring(a))
    assert qem.collapse_ok(a, b, V[b].copy())[0]
    far = V[a] + 50.0 * (V[a] - V[b])
    assert not qem.collapse_ok(a, b, far)[0]
