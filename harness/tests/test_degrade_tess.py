import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
from build123d import Box, Cylinder

from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import (
    DEFLECTION_SETTINGS,
    LabeledMesh,
    distance_to_surface,
    outward_normals,
    tessellate,
)

from .test_labels import closed_manifold_problems

TESS_OPS = (
    "coarsen",
    "retriangulate",
    "nonuniform_chords",
    "slivers",
    "t_junctions",
    "fillet_rows",
)
SLOW_SEED = 11


def box_mesh():
    return tessellate(Box(10, 10, 10), 0.1, 0.5)


def cylinder_mesh():
    return tessellate(Cylinder(5, 20), 0.05, 0.3)


def fillet_mesh():
    gt = generate("one_segment_fillet", 0)
    lin, ang = gt.parameters["pair_deflection"]
    return tessellate(gt.solid, lin, ang)


def ngon_mesh():
    gt = generate("ngon_prism", 3)
    lin, ang = gt.parameters["pair_deflection"]
    return tessellate(gt.solid, lin, ang)


def drilled_mesh():
    return tessellate(Box(10, 10, 10) - Cylinder(2, 20), 0.1, 0.5)


@pytest.fixture(scope="module")
def box():
    return box_mesh()


@pytest.fixture(scope="module")
def cyl():
    return cylinder_mesh()


@pytest.fixture(scope="module")
def fillet():
    return fillet_mesh()


@pytest.fixture(scope="module")
def ngon():
    return ngon_mesh()


@pytest.fixture(scope="module")
def drilled():
    return drilled_mesh()


def boundary_nodes(mesh: LabeledMesh, fid: int) -> set:
    t = mesh.tris[mesh.face_id == fid]
    counts: dict = {}
    for tri in t:
        for i in range(3):
            a, b = tuple(tri[i]), tuple(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    return {v for e, n in counts.items() if n == 1 for v in e}


def valence_of(tris: np.ndarray) -> list[int]:
    import unmesh

    _, idx, *_ = unmesh.weld(tris, 0.0)
    return sorted(Counter(idx.ravel().tolist()).values())


def geometry_equal(a: LabeledMesh, b: LabeledMesh) -> bool:
    return (
        np.array_equal(a.tris, b.tris)
        and np.array_equal(a.face_id, b.face_id)
        and a.vertices == b.vertices
        and [x.points for x in a.adjacency] == [x.points for x in b.adjacency]
        and [f.params for f in a.faces] == [f.params for f in b.faces]
    )


def assert_tess_labels_aligned(before: LabeledMesh, after: LabeledMesh) -> None:
    assert set(after.face_id.tolist()) <= set(before.face_id.tolist())
    assert len(after.face_id) == len(after.tris)
    assert [f.id for f in after.faces] == [f.id for f in before.faces]
    assert [f.surface for f in after.faces] == [f.surface for f in before.faces]
    assert [(a.face_a, a.face_b, a.edge_id) for a in after.adjacency] == [
        (a.face_a, a.face_b, a.edge_id) for a in before.adjacency
    ]
    assert [s.__dict__ for s in after.shells] == [s.__dict__ for s in before.shells]
    known = {tuple(p) for p in after.tris.reshape(-1, 3).tolist()}
    for adj in after.adjacency:
        assert len(adj.points) >= 2
        assert all(tuple(p) in known for p in adj.points)
    assert all(tuple(v) in known for v in after.vertices)


def assert_on_surface(mesh: LabeledMesh, tol: float = 1e-9) -> None:
    for face in mesh.faces:
        if face.surface == "other":
            continue
        pts = mesh.tris[mesh.face_id == face.id].reshape(-1, 3)
        assert distance_to_surface(face, pts).max() < tol, (face.id, face.surface)


@pytest.mark.parametrize("name", TESS_OPS)
def test_tess_severity_zero_is_identity(name, box):
    out = degrade.apply(name, box, 0.0, 7)
    assert geometry_equal(box, out)
    assert out.metadata["history"][-1]["severity"] == 0.0


@pytest.mark.parametrize("name", TESS_OPS)
def test_tess_deterministic_in_process(name, cyl):
    a = degrade.apply(name, cyl, 0.7, SLOW_SEED)
    b = degrade.apply(name, cyl, 0.7, SLOW_SEED)
    assert geometry_equal(a, b)
    assert a.metadata == b.metadata


@pytest.mark.parametrize("name", TESS_OPS)
def test_tess_deterministic_across_processes(name, tmp_path):
    src = tmp_path / "mesh.npz"
    cylinder_mesh().save(src)
    script = (
        "import sys, hashlib\n"
        "from unmesh_harness import degrade\n"
        "from unmesh_harness.labels import LabeledMesh\n"
        "m = LabeledMesh.load(sys.argv[1])\n"
        f"o = degrade.apply({name!r}, m, 0.7, {SLOW_SEED})\n"
        "h = hashlib.sha256()\n"
        "h.update(o.tris.tobytes())\n"
        "h.update(repr(o.metadata).encode())\n"
        "print(h.hexdigest())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(src)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    local = degrade.apply(name, LabeledMesh.load(src), 0.7, SLOW_SEED)
    digest = __import__("hashlib").sha256()
    digest.update(local.tris.tobytes())
    digest.update(repr(local.metadata).encode())
    assert proc == digest.hexdigest()


def test_coarsen_preserves_watertight_and_labels(box, cyl):
    for mesh in (box, cyl):
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in (0.5, 1.0):
            out = degrade.apply("coarsen", mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], severity
            assert_tess_labels_aligned(mesh, out)
            assert out.metadata["history"][-1]["params"]


def test_coarsen_threshold_scale(box):
    flat = box.tris.reshape(-1, 3)
    diagonal = float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))
    for severity, ratio in ((0.25, 0.01 * 10.0**0.25), (0.5, 0.01 * 10.0**0.5), (1.0, 0.1)):
        out = degrade.apply("coarsen", box, severity, 0)
        params = out.metadata["history"][-1]["params"]
        assert params["threshold_mm"] == pytest.approx(diagonal * ratio)
        assert params["triangles_after"] <= params["triangles_before"]


def test_coarsen_nodes_stay_on_surface(cyl):
    for severity in (0.5, 1.0):
        out = degrade.apply("coarsen", cyl, severity, 3)
        assert len(out.tris) < len(cyl.tris)
        assert_on_surface(out)


def test_coarsen_cylinder_degrades_to_prism(cyl):
    def stations(mesh):
        face = next(f for f in mesh.faces if f.surface == "cylinder")
        pts = mesh.face_tris(face.id).reshape(-1, 3)
        axis = np.array(face.params["axis"])
        origin = np.array(face.params["origin"])
        rel = pts - origin
        radial = rel - np.outer(rel @ axis, axis)
        ang = np.arctan2(radial[:, 1], radial[:, 0])
        return len(np.unique(np.round(ang, 6)))

    assert stations(degrade.apply("coarsen", cyl, 0.5, 3)) < stations(cyl)
    out = degrade.apply("coarsen", cyl, 1.0, 3)
    assert stations(out) <= stations(degrade.apply("coarsen", cyl, 0.5, 3))
    assert closed_manifold_problems(out.tris)[0] == []


def test_coarsen_fillet_mesh_stays_valid(fillet):
    out = degrade.apply("coarsen", fillet, 1.0, 3)
    assert closed_manifold_problems(out.tris)[0] == []
    assert_tess_labels_aligned(fillet, out)
    assert_on_surface(out)
    assert OPERATORS["coarsen"].family == "tessellation"


def test_retriangulate_schemes_from_severity(ngon):
    assert (
        degrade.apply("retriangulate", ngon, 0.2, 0).metadata["history"][-1]["params"]["scheme"]
        == "fan"
    )
    assert (
        degrade.apply("retriangulate", ngon, 0.5, 0).metadata["history"][-1]["params"]["scheme"]
        == "strip"
    )
    assert (
        degrade.apply("retriangulate", ngon, 0.9, 0).metadata["history"][-1]["params"]["scheme"]
        == "delaunay"
    )


def test_retriangulate_keeps_boundary_and_watertight(ngon, drilled):
    for mesh in (ngon, drilled):
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in (0.2, 0.5, 0.9):
            out = degrade.apply("retriangulate", mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], severity
            assert_tess_labels_aligned(mesh, out)
            for fid in set(mesh.face_id.tolist()):
                assert boundary_nodes(out, fid) == boundary_nodes(mesh, fid)
            assert out.metadata["history"][-1]["params"]["faces_retriangulated"] > 0


def test_retriangulate_leaves_curved_faces_bit_identical(cyl):
    out = degrade.apply("retriangulate", cyl, 0.9, 3)
    for face in cyl.faces:
        if face.surface == "plane":
            continue
        assert np.array_equal(out.tris[out.face_id == face.id], cyl.tris[cyl.face_id == face.id])


def test_retriangulate_valence_differs_from_occt(ngon):
    cap = next(
        f.id for f in ngon.faces if f.surface == "plane" and (ngon.face_id == f.id).sum() > 10
    )
    occt = valence_of(ngon.tris[ngon.face_id == cap])
    seen = {tuple(occt)}
    for severity in (0.2, 0.5, 0.9):
        out = degrade.apply("retriangulate", ngon, severity, 3)
        assert (out.face_id == cap).sum() == (ngon.face_id == cap).sum()
        got = tuple(valence_of(out.tris[out.face_id == cap]))
        assert got not in seen, severity
        seen.add(got)
    assert len(seen) == 4


def side_circumferential_chords(mesh: LabeledMesh, fid: int) -> list[float]:
    import math

    edges = set()
    for tri in mesh.tris[mesh.face_id == fid]:
        for i in range(3):
            a, b = tuple(tri[i]), tuple(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return [math.dist(a, b) for a, b in edges if abs(a[2] - b[2]) < 1e-9]


def test_nonuniform_chords_stays_on_surface_and_watertight(cyl, drilled):
    for mesh in (cyl, drilled):
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in (0.5, 1.0):
            out = degrade.apply("nonuniform_chords", mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], severity
            assert_tess_labels_aligned(mesh, out)
            assert_on_surface(out)
            params = out.metadata["history"][-1]["params"]
            assert params["moved_vertices"] >= 0
            assert params["max_displacement_mm"] <= 0.45 * severity * 25 + 1e-9


def test_nonuniform_chords_spreads_chord_spacing(cyl):
    side = next(f.id for f in cyl.faces if f.surface == "cylinder")
    before = side_circumferential_chords(cyl, side)
    assert np.std(before) == pytest.approx(0.0, abs=1e-9)
    low = degrade.apply("nonuniform_chords", cyl, 0.5, 3)
    high = degrade.apply("nonuniform_chords", cyl, 1.0, 3)
    assert np.std(side_circumferential_chords(low, side)) > 0.05
    assert np.std(side_circumferential_chords(high, side)) > np.std(
        side_circumferential_chords(low, side)
    )


def test_nonuniform_chords_fixes_cad_corners(cyl):
    out = degrade.apply("nonuniform_chords", cyl, 1.0, 3)
    assert out.vertices == cyl.vertices
    assert [a.points[0] for a in out.adjacency] == [a.points[0] for a in cyl.adjacency]
    assert [a.points[-1] for a in out.adjacency] == [a.points[-1] for a in cyl.adjacency]


def aspect_ratios(tris: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    b = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    c = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-30))
    return np.maximum(a, np.maximum(b, c)) ** 2 / (2 * area)


def test_slivers_stay_watertight_and_labelled(cyl, drilled):
    for mesh in (cyl, drilled):
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in (0.5, 1.0):
            out = degrade.apply("slivers", mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], severity
            assert_tess_labels_aligned(mesh, out)
            params = out.metadata["history"][-1]["params"]
            assert params["edges_split"] > 0
            assert params["triangles_after"] > params["triangles_before"]


def test_slivers_raise_aspect_ratio(cyl):
    before = aspect_ratios(cyl.tris).max()
    mid = degrade.apply("slivers", cyl, 0.5, 3)
    high = degrade.apply("slivers", cyl, 1.0, 3)
    assert aspect_ratios(mid.tris).max() > before
    assert aspect_ratios(high.tris).max() > aspect_ratios(mid.tris).max()
    assert (aspect_ratios(high.tris) > 100).sum() >= 1


def test_t_junctions_break_watertight_by_design_but_keep_labels(cyl, drilled):
    for mesh in (cyl, drilled):
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in (0.5, 1.0):
            out = degrade.apply("t_junctions", mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] != [], severity
            assert_tess_labels_aligned(mesh, out)
            assert set(out.face_id.tolist()) == set(mesh.face_id.tolist())
            params = out.metadata["history"][-1]["params"]
            assert params["splits"] == params["target_splits"] >= 1
            assert len(out.tris) == len(mesh.tris) + params["splits"]
            assert OPERATORS["t_junctions"].preserves_watertight is False


def test_fillet_rows_segments_from_severity(fillet):
    assert (
        degrade.apply("fillet_rows", fillet, 0.2, 0).metadata["history"][-1]["params"]["segments"]
        == 3
    )
    assert (
        degrade.apply("fillet_rows", fillet, 0.5, 0).metadata["history"][-1]["params"]["segments"]
        == 2
    )
    assert (
        degrade.apply("fillet_rows", fillet, 0.9, 0).metadata["history"][-1]["params"]["segments"]
        == 1
    )


def test_fillet_rows_single_segment_uses_tangent_endpoints(fillet):
    out = degrade.apply("fillet_rows", fillet, 0.9, 3)
    params = out.metadata["history"][-1]["params"]
    assert params["fillet_faces"] == [2]
    assert sorted(params["neighbours_retriangulated"]) == [1, 4]
    face_tris = out.tris[out.face_id == 2]
    assert len(face_tris) == 2
    corners = {tuple(p) for p in face_tris.reshape(-1, 3).tolist()}
    tangents = set()
    for adj in fillet.adjacency:
        if adj.face_a == 2 or adj.face_b == 2:
            if adj.tangent and adj.curve == "line":
                tangents.update(map(tuple, adj.points))
    assert corners == tangents
    assert closed_manifold_problems(out.tris)[0] == []
    assert_tess_labels_aligned(fillet, out)
    assert_on_surface(out)


def test_fillet_rows_deterministic_on_fillet(fillet):
    a = degrade.apply("fillet_rows", fillet, 0.5, SLOW_SEED)
    b = degrade.apply("fillet_rows", fillet, 0.5, SLOW_SEED)
    assert geometry_equal(a, b)
    assert a.metadata == b.metadata


def test_fillet_rows_noop_without_eligible_fillets(box, cyl):
    for mesh in (box, cyl):
        out = degrade.apply("fillet_rows", mesh, 0.9, 3)
        assert geometry_equal(mesh, out)
        assert out.metadata["history"][-1]["params"]["fillet_faces"] == []


SWEEP_SEVERITIES = (0.2, 0.5, 1.0)
SWEEP_SEED = 4
SWEEP_FAST_IDS = (
    "through_bore-0000",
    "revolved_torus-0000",
    "corner_fillet-0000",
    "one_segment_fillet-0000",
)


def _sweep_meshes(ids=None):
    lin, ang = DEFLECTION_SETTINGS[0]
    out = []
    for e in select(load_manifest(), "smoke"):
        if ids is not None and e["id"] not in ids:
            continue
        out.append((e["id"], tessellate(generate(e["family"], e["seed"]).solid, lin, ang)))
    assert len(out) == (len(ids) if ids is not None else len(select(load_manifest(), "smoke")))
    return out


def _folded_count(mesh: LabeledMesh) -> int:
    bad = 0
    for face in mesh.faces:
        if face.surface == "other":
            continue
        t = mesh.tris[mesh.face_id == face.id]
        if not len(t):
            continue
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = np.linalg.norm(n, axis=1)
        try:
            on = outward_normals(face, t.mean(axis=1))
        except Exception:
            continue
        d = np.einsum("ij,ij->i", n, on)
        ok = np.isfinite(d) & (area > 1e-12)
        bad += int((d[ok] < 0).sum())
    return bad


def _degenerate_count(mesh: LabeledMesh) -> int:
    return int(
        (
            np.linalg.norm(
                np.cross(mesh.tris[:, 1] - mesh.tris[:, 0], mesh.tris[:, 2] - mesh.tris[:, 0]),
                axis=1,
            )
            <= 1e-14
        ).sum()
    )


def assert_sweep_clean(
    pid: str, op: str, sev: float, base_folds: int, base_degen: int, out: LabeledMesh
) -> None:
    for face in out.faces:
        if face.surface == "other":
            continue
        pts = out.tris[out.face_id == face.id].reshape(-1, 3)
        if len(pts):
            assert distance_to_surface(face, pts).max() <= 1e-9, (pid, op, sev, face.id)
    for adj in out.adjacency:
        poly = np.array(adj.points)
        for fid in (adj.face_a, adj.face_b):
            face = out.faces[fid]
            if face.surface == "other":
                continue
            assert distance_to_surface(face, poly).max() <= 1e-9, (pid, op, sev, fid)
    assert _folded_count(out) <= base_folds, (pid, op, sev)
    assert _degenerate_count(out) <= base_degen, (pid, op, sev)
    if op == "t_junctions":
        return
    assert closed_manifold_problems(out.tris)[0] == [], (pid, op, sev)
    ekey: dict = {}
    for t, fid in zip(out.tris.tolist(), out.face_id.tolist(), strict=True):
        t = [tuple(x) for x in t]
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ekey.setdefault(frozenset((a, b)), set()).add(fid)
    for adj in out.adjacency:
        pts = [tuple(p) for p in adj.points]
        for a, b in zip(pts, pts[1:], strict=False):
            fs = ekey.get(frozenset((a, b)))
            assert fs is not None and {adj.face_a, adj.face_b} <= fs, (pid, op, sev)


def _run_sweep(meshes) -> None:
    for pid, mesh in meshes:
        base_folds, base_degen = _folded_count(mesh), _degenerate_count(mesh)
        for op in TESS_OPS:
            for sev in SWEEP_SEVERITIES:
                out = degrade.apply(op, mesh, sev, SWEEP_SEED)
                assert_sweep_clean(pid, op, sev, base_folds, base_degen, out)
                if op == "retriangulate":
                    skipped = out.metadata["history"][-1]["params"]["skipped_faces"]
                    assert skipped["failed_clip"] == [], (pid, sev, skipped)


def test_tess_sweep_fast_subset():
    _run_sweep(_sweep_meshes(SWEEP_FAST_IDS))


@pytest.mark.slow
def test_tess_sweep_full_smoke():
    _run_sweep(_sweep_meshes())


def test_fillet_rows_half_turn_strips_keep_two_segments():
    from build123d import SlotOverall, extrude

    mesh = tessellate(extrude(SlotOverall(20, 6), 4), 0.1, 0.5)
    ends = [f.id for f in mesh.faces if f.surface == "cylinder"]
    assert len(ends) == 2
    for sev in (0.5, 1.0):
        out = degrade.apply("fillet_rows", mesh, sev, 0)
        params = out.metadata["history"][-1]["params"]
        assert params["fillet_faces"] == ends
        assert params["half_turn_faces"] == (ends if sev == 1.0 else [])
        assert not geometry_equal(mesh, out)
        assert _folded_count(out) == 0
        assert closed_manifold_problems(out.tris)[0] == []
        for fid in ends:
            face = out.faces[fid]
            assert len(out.tris[out.face_id == fid]) == 4
            axis = np.array(face.params["axis"]) / np.linalg.norm(face.params["axis"])
            rel = out.tris[out.face_id == fid].mean(axis=1) - np.array(face.params["origin"])
            off_axis = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
            assert off_axis.min() > 0.5 * float(face.params["radius"])


def test_fillet_rows_complex_assembly_does_not_fold():
    ((pid, mesh),) = _sweep_meshes(("complex_assembly-0000",))
    assert _folded_count(mesh) == 0
    for sev in SWEEP_SEVERITIES:
        out = degrade.apply("fillet_rows", mesh, sev, SWEEP_SEED)
        assert _folded_count(out) == 0, (pid, sev)
        assert_sweep_clean(pid, "fillet_rows", sev, 0, _degenerate_count(mesh), out)


def test_fillet_greedy_keep_matches_sequential_acceptance():
    from unmesh_harness.degrade.fillets import _greedy_keep

    strips = list(range(11))
    bad = {3, 8}

    def ok(sub):
        return not bad & set(sub) and not {5, 6} <= set(sub)

    kept, sequential = _greedy_keep(ok, strips), []
    for s in strips:
        if ok([*sequential, s]):
            sequential.append(s)
    assert list(kept) == sequential == [0, 1, 2, 4, 5, 7, 9, 10]
