import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Plane, Torus, fillet, mirror

from unmesh.ir import validate
from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS, to_original_points
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import (
    DEFLECTION_SETTINGS,
    LabeledMesh,
    distance_to_surface,
    tessellate,
)
from unmesh_harness.oracle import build_oracle_ir

from .test_labels import closed_manifold_problems
from .test_oracle import check_normals_and_left_hand_rule

LIN, ANG = DEFLECTION_SETTINGS[0]
POSE = [n for n, o in OPERATORS.items() if o.changes_frame]
NOISE = [n for n, o in OPERATORS.items() if o.family == "noise"]
ALL = sorted(OPERATORS)
REFINE_SEVERITY = 0.3


def prepared(name, mesh):
    if name == "noise_off_plane":
        return degrade.apply("refine", mesh, REFINE_SEVERITY, 0)
    return mesh


REPRESENTATIVE = (0, 3, 8, 13)
FULL = {"slow": False}


@pytest.fixture(autouse=True, scope="module")
def _tier(request):
    FULL["slow"] = bool(request.config.getoption("--slow"))


def levels_for(name, levels):
    if name == "refine" and not FULL["slow"]:
        return tuple(min(x, REFINE_SEVERITY) for x in levels)
    return levels


def assert_on_surface(mesh, tol=1e-9):
    for face in mesh.faces:
        if face.surface == "other":
            continue
        pts = mesh.tris[mesh.face_id == face.id].reshape(-1, 3)
        assert distance_to_surface(face, pts).max() < tol, (face.id, face.surface)


def smoke_meshes(indices=None):
    entries = select(load_manifest(), "smoke")
    if indices is not None:
        entries = [entries[i] for i in indices]
    return [
        (e["id"], tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG)) for e in entries
    ]


def curved_meshes():
    shapes = {
        "filleted_box": fillet(Box(20, 20, 20).edges(), 3),
        "mirrored_drilled": mirror(Box(10, 10, 10) - Cylinder(2, 20), Plane.YZ),
        "cone_torus": mirror(Cone(5, 2, 10) + Torus(10, 2), Plane.XZ),
    }
    return [(k, tessellate(v, 0.05, 0.3)) for k, v in shapes.items()]


@pytest.fixture(scope="module")
def smoke(request):
    if request.config.getoption("--slow"):
        return smoke_meshes()
    return smoke_meshes(REPRESENTATIVE)


@pytest.fixture(scope="module")
def curved():
    return curved_meshes()


def geometry_equal(a: LabeledMesh, b: LabeledMesh) -> bool:
    return (
        np.array_equal(a.tris, b.tris)
        and np.array_equal(a.face_id, b.face_id)
        and a.vertices == b.vertices
        and [x.points for x in a.adjacency] == [x.points for x in b.adjacency]
        and [f.params for f in a.faces] == [f.params for f in b.faces]
    )


def labels_aligned(before: LabeledMesh, after: LabeledMesh) -> None:
    if after.metadata["history"][-1]["op"] == "refine":
        assert set(after.face_id.tolist()) == set(before.face_id.tolist())
        assert len(after.tris) >= len(before.tris)
        assert len(after.face_id) == len(after.tris)
    else:
        assert after.tris.shape == before.tris.shape
        assert np.array_equal(after.face_id, before.face_id)
    assert [f.id for f in after.faces] == [f.id for f in before.faces]
    assert [f.surface for f in after.faces] == [f.surface for f in before.faces]
    assert [(a.face_a, a.face_b, a.edge_id) for a in after.adjacency] == [
        (a.face_a, a.face_b, a.edge_id) for a in before.adjacency
    ]
    assert [s.__dict__ for s in after.shells] == [s.__dict__ for s in before.shells]


def polylines_on_mesh(mesh: LabeledMesh) -> None:
    known = {tuple(p) for p in mesh.tris.reshape(-1, 3).tolist()}
    for adj in mesh.adjacency:
        assert all(tuple(p) in known for p in adj.points)
    assert all(tuple(v) in known for v in mesh.vertices)


def test_registry_documents_every_operator():
    assert {"float32", "truncated_digits", "inch_round_trip", "far_translation"} <= set(OPERATORS)
    assert {"rotation", "mirror", "noise_isotropic", "noise_normal", "noise_off_plane"} <= set(
        OPERATORS
    )
    for op in OPERATORS.values():
        assert op.severity_0 and op.severity_1


@pytest.mark.parametrize("name", ALL)
def test_severity_zero_is_identity(name, smoke):
    for _, mesh in smoke:
        out = degrade.apply(name, mesh, 0.0, 7)
        assert geometry_equal(mesh, out)
        assert out.metadata["history"][-1]["op"] == name
        assert out.metadata["history"][-1]["severity"] == 0.0
        assert "to_original" not in out.metadata


@pytest.mark.parametrize("name", ALL)
def test_inputs_are_not_mutated(name, smoke):
    _, mesh = smoke[0]
    mesh = prepared(name, mesh)
    snapshot = mesh.tris.copy()
    before = copy.deepcopy(mesh.metadata)
    degrade.apply(name, mesh, levels_for(name, (1.0,))[0], 1)
    assert np.array_equal(mesh.tris, snapshot)
    assert mesh.metadata == before


@pytest.mark.parametrize("severity", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("name", ALL)
def test_labels_aligned_and_polylines_coincide(name, severity, smoke, curved):
    for _, mesh in smoke + curved:
        mesh = prepared(name, mesh)
        out = degrade.apply(name, mesh, levels_for(name, (severity,))[0], 3)
        labels_aligned(mesh, out)
        polylines_on_mesh(out)
        assert out.metadata["history"][-1]["params"]


@pytest.mark.parametrize("name", ALL)
def test_watertight_preserved(name, smoke, curved):
    op = OPERATORS[name]
    if not op.preserves_watertight:
        pytest.skip("operator documents that it does not preserve watertightness")
    levels = levels_for(name, (0.25, 0.5) if op.may_collapse else (0.5, 1.0))
    for _, mesh in smoke + curved:
        mesh = prepared(name, mesh)
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in levels:
            out = degrade.apply(name, mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], (name, severity)


@pytest.mark.parametrize("name", ALL)
def test_deterministic_in_process(name, smoke):
    _, mesh = smoke[3]
    mesh = prepared(name, mesh)
    a = degrade.apply(name, mesh, 0.7, 11)
    b = degrade.apply(name, mesh, 0.7, 11)
    assert geometry_equal(a, b)
    assert a.metadata == b.metadata


def test_seed_changes_random_operators(smoke):
    _, mesh = smoke[3]
    for name in ("rotation", "mirror", "noise_isotropic", "noise_normal", "noise_off_plane"):
        base = prepared(name, mesh)
        a = degrade.apply(name, base, 0.7, 1)
        b = degrade.apply(name, base, 0.7, 2)
        assert not np.array_equal(a.tris, b.tris), name


def test_deterministic_across_processes(smoke, tmp_path):
    _, mesh = smoke[3]
    src = tmp_path / "mesh.npz"
    mesh.save(src)
    script = (
        "import sys, hashlib\n"
        "from unmesh_harness import degrade\n"
        "from unmesh_harness.labels import LabeledMesh\n"
        "m = LabeledMesh.load(sys.argv[1])\n"
        "h = hashlib.sha256()\n"
        "for n in sorted(degrade.OPERATORS):\n"
        "    b = degrade.apply('refine', m, 0.3, 0) if n == 'noise_off_plane' else m\n"
        "    o = degrade.apply(n, b, 0.6, 99)\n"
        "    h.update(o.tris.tobytes())\n"
        "    h.update(repr(o.metadata).encode())\n"
        "print(h.hexdigest())\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script, str(src)], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(1)
    }
    local = __import__("hashlib").sha256()
    loaded = LabeledMesh.load(src)
    for n in sorted(OPERATORS):
        o = degrade.apply(n, prepared(n, loaded), 0.6, 99)
        local.update(o.tris.tobytes())
        local.update(repr(o.metadata).encode())
    assert digests == {local.hexdigest() + "\n"}


@pytest.mark.parametrize("name", POSE)
def test_pose_round_trip_recovers_coordinates(name, smoke, curved):
    for _, mesh in smoke + curved:
        out = degrade.apply(name, mesh, 1.0, 21)
        back = to_original_points(out, out.tris.reshape(-1, 3))
        reference = mesh.tris.reshape(-1, 3)
        if name == "mirror":
            back = back.reshape(-1, 3, 3)[:, [0, 2, 1]].reshape(-1, 3)
        assert np.abs(back - reference).max() < 1e-9
        for adj_out, adj_in in zip(out.adjacency, mesh.adjacency, strict=True):
            pts = to_original_points(out, np.array(adj_out.points))
            assert np.abs(pts - np.array(adj_in.points)).max() < 1e-9
        verts = to_original_points(out, np.array(out.vertices))
        assert np.abs(verts - np.array(mesh.vertices)).max() < 1e-9


@pytest.mark.parametrize("name", POSE)
@pytest.mark.parametrize("severity", [0.3, 1.0])
def test_pose_oracle_ir_validates(name, severity, smoke, curved):
    for _, mesh in smoke + curved:
        out = degrade.apply(name, mesh, severity, 8)
        assert_on_surface(out)
        ir = build_oracle_ir(out)
        assert validate(ir) == []
        check_normals_and_left_hand_rule(out, ir)
        assert len(ir.regions) == len(mesh.faces)


def test_pose_preserves_signed_volume_sign(curved):
    def volume(t):
        return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6

    for _, mesh in curved:
        for name in POSE:
            out = degrade.apply(name, mesh, 1.0, 4)
            assert volume(out.tris) == pytest.approx(volume(mesh.tris), rel=1e-9)


def test_mirror_flips_edge_direction_flags(smoke):
    _, mesh = smoke[0]
    out = degrade.apply("mirror", mesh, 1.0, 1)
    assert all(
        a.forward_in_a != b.forward_in_a for a, b in zip(out.adjacency, mesh.adjacency, strict=True)
    )
    rot = degrade.apply("rotation", mesh, 1.0, 1)
    assert all(
        a.forward_in_a == b.forward_in_a for a, b in zip(rot.adjacency, mesh.adjacency, strict=True)
    )


def test_pose_composes_through_chain(smoke):
    _, mesh = smoke[2]
    out = degrade.apply_chain(mesh, [("rotation", 0.8), ("mirror", 1.0), ("rotation", 0.3)], 5)
    assert [h["op"] for h in out.metadata["history"]] == ["rotation", "mirror", "rotation"]
    back = to_original_points(out, out.tris.reshape(-1, 3)).reshape(-1, 3, 3)[:, [0, 2, 1]]
    assert np.abs(back.reshape(-1, 3) - mesh.tris.reshape(-1, 3)).max() < 1e-9
    assert validate(build_oracle_ir(out)) == []


def test_float32_values_are_float32(smoke):
    _, mesh = smoke[0]
    out = degrade.apply("float32", mesh, 0.5, 0)
    assert np.array_equal(out.tris, out.tris.astype(np.float32).astype(np.float64))
    assert np.array_equal(
        np.array([p for a in out.adjacency for p in a.points]),
        np.array([p for a in out.adjacency for p in a.points])
        .astype(np.float32)
        .astype(np.float64),
    )


def test_truncated_digits_severity_maps_to_digits(smoke):
    _, mesh = smoke[0]
    for severity, digits in ((0.5, 6), (1.0, 3)):
        out = degrade.apply("truncated_digits", mesh, severity, 0)
        assert out.metadata["history"][-1]["params"]["significant_digits"] == digits
        for x in out.tris.reshape(-1)[:200]:
            if x:
                assert float(f"{x:.{digits - 1}e}") == x


def test_inch_round_trip_lands_on_inch_grid(smoke):
    _, mesh = smoke[0]
    out = degrade.apply("inch_round_trip", mesh, 1.0, 0)
    inches = out.tris / 25.4
    assert np.abs(inches - np.round(inches, 3)).max() < 1e-12


def test_far_translation_error_matches_ulp(smoke):
    _, mesh = smoke[0]
    out = degrade.apply("far_translation", mesh, 1.0, 0)
    err = np.abs(out.tris - mesh.tris).max()
    assert 0 < err <= 0.5 * 0.0625 + 1e-6


@pytest.mark.parametrize("name", NOISE)
def test_noise_moves_shared_vertices_once(name, smoke, curved):
    for _, mesh in smoke + curved:
        mesh = prepared(name, mesh)
        out = degrade.apply(name, mesh, 1.0, 13)
        before = mesh.tris.reshape(-1, 3)
        after = out.tris.reshape(-1, 3)
        mapping: dict[tuple, tuple] = {}
        for b, a in zip(before.tolist(), after.tolist(), strict=True):
            assert mapping.setdefault(tuple(b), tuple(a)) == tuple(a)
        assert not np.array_equal(before, after)


@pytest.mark.parametrize("name", NOISE)
def test_noise_amplitude_is_a_hard_bound(name, smoke):
    _, mesh = smoke[1]
    mesh = prepared(name, mesh)
    for severity in (0.25, 1.0):
        out = degrade.apply(name, mesh, severity, 2)
        params = out.metadata["history"][-1]["params"]
        assert params["amplitude_mm"] == pytest.approx(0.05 * severity)
        assert params["distribution"].startswith("uniform")
        measured = np.linalg.norm(out.tris - mesh.tris, axis=2).max()
        assert measured <= params["amplitude_mm"] + 1e-12
        assert measured == pytest.approx(params["max_displacement_mm"], rel=1e-12)
        assert measured > 0.5 * params["amplitude_mm"]


@pytest.mark.parametrize("name", NOISE)
def test_noise_face_table_stays_consistent(name, smoke, curved):
    for _, mesh in smoke + curved:
        mesh = prepared(name, mesh)
        out = degrade.apply(name, mesh, 1.0, 17)
        assert [f.__dict__ for f in out.faces] == [f.__dict__ for f in mesh.faces]
        ir = build_oracle_ir(out)
        assert validate(ir) == []
        assert len(ir.regions) == len(mesh.faces)


def test_off_plane_requires_interior_vertices(curved):
    for _, mesh in curved:
        with pytest.raises(ValueError, match="refine"):
            degrade.apply("noise_off_plane", mesh, 1.0, 0)
    box = tessellate(Box(10, 10, 10), 0.1, 0.5)
    with pytest.raises(ValueError, match="refine"):
        degrade.apply("noise_off_plane", box, 1.0, 0)
    degrade.apply("noise_off_plane", box, 0.0, 0)


def test_off_plane_moves_interior_vertices_along_face_normal(smoke, curved):
    for _, mesh in smoke + curved:
        refined = degrade.apply("refine", mesh, REFINE_SEVERITY, 0)
        out = degrade.apply("noise_off_plane", refined, 1.0, 6)
        params = out.metadata["history"][-1]["params"]
        assert 0 < params["moved_fraction"] < 1
        assert params["moved_vertices"] > 0
        normals = np.array(
            [f.params["normal"] if f.surface == "plane" else [0, 0, 0] for f in refined.faces]
        )[refined.face_id]
        delta = out.tris - refined.tris
        moved = np.linalg.norm(delta, axis=2) > 0
        assert moved.any()
        for corner in range(3):
            d = delta[:, corner]
            sel = moved[:, corner]
            assert np.abs(np.cross(d[sel], normals[sel])).max() < 1e-12
        curved_ids = [f.id for f in refined.faces if f.surface != "plane"]
        for fid in curved_ids:
            assert np.array_equal(
                out.tris[refined.face_id == fid], refined.tris[refined.face_id == fid]
            )


def test_normal_noise_displaces_along_normal(curved):
    _, mesh = curved[0]
    out = degrade.apply("noise_normal", mesh, 1.0, 9)
    delta = (out.tris - mesh.tris).reshape(-1, 3)
    assert 0 < np.abs(delta).max() <= 0.05


def test_metadata_survives_save_load(smoke, tmp_path):
    _, mesh = smoke[0]
    out = degrade.apply_chain(mesh, [("rotation", 0.5), ("noise_isotropic", 0.5)], 3)
    out.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert json.dumps(back.metadata, sort_keys=True) == json.dumps(out.metadata, sort_keys=True)
    assert np.array_equal(back.tris, out.tris)


def test_severity_out_of_range_rejected(smoke):
    _, mesh = smoke[0]
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            degrade.apply("float32", mesh, bad, 0)


@pytest.mark.parametrize("name", ["rotation", "mirror", "refine"])
def test_face_table_matches_geometry(name, smoke, curved):
    for _, mesh in smoke + curved:
        out = degrade.apply(name, mesh, levels_for(name, (1.0,))[0], 31)
        assert_on_surface(out)


def test_refine_midpoints_land_on_the_surface_and_edges(curved):
    for _, mesh in curved:
        out = degrade.apply("refine", mesh, 0.3, 0)
        params = out.metadata["history"][-1]["params"]
        assert params["triangles_after"] > len(mesh.tris)
        assert_on_surface(out)
        longest = max(
            np.linalg.norm(out.tris[:, i] - out.tris[:, (i + 1) % 3], axis=1).max()
            for i in range(3)
        )
        assert longest <= params["target_mm"] + 1e-12
        by_edge = {a.edge_id: a for a in mesh.adjacency}
        for adj in out.adjacency:
            assert len(adj.points) >= len(by_edge[adj.edge_id].points)
            pts = np.array(adj.points)
            for face in (mesh.faces[adj.face_a], mesh.faces[adj.face_b]):
                assert distance_to_surface(face, pts).max() < 1e-9


def test_refine_polylines_keep_original_nodes_in_order(curved):
    _, mesh = curved[0]
    out = degrade.apply("refine", mesh, 0.3, 0)
    for before, after in zip(mesh.adjacency, out.adjacency, strict=True):
        it = iter(map(tuple, after.points))
        assert all(tuple(p) in it for p in before.points)


def test_truth_positions_survive_every_operator(smoke):
    _, mesh = smoke[2]
    steps = [
        ("refine", 0.3),
        ("rotation", 0.8),
        ("mirror", 1.0),
        ("noise_isotropic", 1.0),
        ("far_translation", 0.5),
        ("noise_off_plane", 1.0),
    ]
    out = degrade.apply_chain(mesh, steps, 4)
    truth = out.metadata["truth"]
    assert np.array_equal(np.array(truth["vertices"]), np.array(mesh.vertices))
    assert truth["edge_points"] == [a.points for a in mesh.adjacency]
    moved = to_original_points(out, np.array(out.vertices))
    assert 0 < np.abs(moved - np.array(mesh.vertices)).max() < 0.2


def test_binary_operators_flagged():
    assert {n for n, o in OPERATORS.items() if o.binary} == {"float32", "mirror"}


@pytest.mark.slow
def test_refine_severity_one_on_full_smoke():
    for _, mesh in smoke_meshes():
        out = degrade.apply("refine", mesh, 1.0, 0)
        params = out.metadata["history"][-1]["params"]
        assert params["target_mm"] == pytest.approx(params["bbox_diagonal_mm"] / 100)
        assert closed_manifold_problems(out.tris)[0] == []
        assert_on_surface(out)
        labels_aligned(mesh, out)
