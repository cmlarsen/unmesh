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
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, tessellate
from unmesh_harness.oracle import build_oracle_ir

from .test_labels import closed_manifold_problems
from .test_oracle import check_normals_and_left_hand_rule

LIN, ANG = DEFLECTION_SETTINGS[0]
POSE = [n for n, o in OPERATORS.items() if o.changes_frame]
NOISE = [n for n, o in OPERATORS.items() if o.family == "noise"]
ALL = sorted(OPERATORS)


def smoke_meshes():
    return [
        (e["id"], tessellate(generate(e["family"], e["seed"]).solid, LIN, ANG))
        for e in select(load_manifest(), "smoke")
    ]


def curved_meshes():
    shapes = {
        "filleted_box": fillet(Box(20, 20, 20).edges(), 3),
        "mirrored_drilled": mirror(Box(10, 10, 10) - Cylinder(2, 20), Plane.YZ),
        "cone_torus": mirror(Cone(5, 2, 10) + Torus(10, 2), Plane.XZ),
    }
    return [(k, tessellate(v, 0.05, 0.3)) for k, v in shapes.items()]


@pytest.fixture(scope="module")
def smoke():
    return smoke_meshes()


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
    snapshot = mesh.tris.copy()
    degrade.apply(name, mesh, 1.0, 1)
    assert np.array_equal(mesh.tris, snapshot)
    assert mesh.metadata == {}


@pytest.mark.parametrize("severity", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("name", ALL)
def test_labels_aligned_and_polylines_coincide(name, severity, smoke, curved):
    for _, mesh in smoke + curved:
        out = degrade.apply(name, mesh, severity, 3)
        labels_aligned(mesh, out)
        polylines_on_mesh(out)
        assert out.metadata["history"][-1]["params"]


@pytest.mark.parametrize("name", ALL)
def test_watertight_preserved(name, smoke, curved):
    op = OPERATORS[name]
    if not op.preserves_watertight:
        pytest.skip("operator documents that it does not preserve watertightness")
    levels = (0.25, 0.5) if op.may_collapse else (0.5, 1.0)
    for _, mesh in smoke + curved:
        assert closed_manifold_problems(mesh.tris)[0] == []
        for severity in levels:
            out = degrade.apply(name, mesh, severity, 5)
            assert closed_manifold_problems(out.tris)[0] == [], (name, severity)


@pytest.mark.parametrize("name", ALL)
def test_deterministic_in_process(name, smoke):
    _, mesh = smoke[3]
    a = degrade.apply(name, mesh, 0.7, 11)
    b = degrade.apply(name, mesh, 0.7, 11)
    assert geometry_equal(a, b)
    assert a.metadata == b.metadata


def test_seed_changes_random_operators(smoke):
    _, mesh = smoke[3]
    for name in ("rotation", "mirror", "noise_isotropic", "noise_normal", "noise_off_plane"):
        a = degrade.apply(name, mesh, 0.7, 1)
        b = degrade.apply(name, mesh, 0.7, 2)
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
        "    o = degrade.apply(n, m, 0.6, 99)\n"
        "    h.update(o.tris.tobytes())\n"
        "    h.update(repr(o.metadata).encode())\n"
        "print(h.hexdigest())\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script, str(src)], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(digests) == 1
    local = __import__("hashlib").sha256()
    loaded = LabeledMesh.load(src)
    for n in sorted(OPERATORS):
        o = degrade.apply(n, loaded, 0.6, 99)
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
        out = degrade.apply(name, mesh, 1.0, 13)
        before = mesh.tris.reshape(-1, 3)
        after = out.tris.reshape(-1, 3)
        mapping: dict[tuple, tuple] = {}
        for b, a in zip(before.tolist(), after.tolist(), strict=True):
            assert mapping.setdefault(tuple(b), tuple(a)) == tuple(a)
        assert not np.array_equal(before, after) or name == "noise_off_plane"


@pytest.mark.parametrize("name", NOISE)
def test_noise_amplitude_is_explicit_and_scales(name, smoke):
    _, mesh = smoke[8]
    sigmas = []
    rms = []
    for severity in (0.25, 1.0):
        out = degrade.apply(name, mesh, severity, 2)
        (value,) = [v for k, v in out.metadata["history"][-1]["params"].items() if "sigma" in k]
        sigmas.append(value)
        rms.append(np.sqrt(((out.tris - mesh.tris) ** 2).sum(-1).mean()))
    assert sigmas == [pytest.approx(0.0125), pytest.approx(0.05)]
    assert rms[1] > rms[0]
    assert rms[1] < 10 * sigmas[1]


@pytest.mark.parametrize("name", NOISE)
def test_noise_face_table_stays_consistent(name, smoke, curved):
    for _, mesh in smoke + curved:
        out = degrade.apply(name, mesh, 1.0, 17)
        assert [f.__dict__ for f in out.faces] == [f.__dict__ for f in mesh.faces]
        ir = build_oracle_ir(out)
        assert validate(ir) == []
        assert len(ir.regions) == len(mesh.faces)


def test_off_plane_moves_along_mean_plane_normal(smoke):
    for _, mesh in smoke:
        out = degrade.apply("noise_off_plane", mesh, 1.0, 6)
        normals = np.array([f.params["normal"] for f in mesh.faces])[mesh.face_id]
        sums: dict[tuple, np.ndarray] = {}
        for tri, n in zip(mesh.tris, normals, strict=True):
            for v in tri.tolist():
                sums[tuple(v)] = sums.get(tuple(v), 0) + n
        moved = 0
        for before, after in zip(mesh.tris.reshape(-1, 3), out.tris.reshape(-1, 3), strict=True):
            delta = after - before
            direction = sums[tuple(before.tolist())]
            if np.linalg.norm(delta) > 0:
                moved += 1
                assert np.linalg.norm(np.cross(delta, direction)) < 1e-12 * max(
                    1.0, np.linalg.norm(direction)
                )
        assert moved


def test_off_plane_leaves_curved_faces_untouched(curved):
    for _, mesh in curved:
        out = degrade.apply("noise_off_plane", mesh, 1.0, 6)
        curved_ids = [f.id for f in mesh.faces if f.surface != "plane"]
        mask = np.isin(mesh.face_id, curved_ids)
        assert np.array_equal(out.tris[mask], mesh.tris[mask])


def test_normal_noise_displaces_along_normal(curved):
    _, mesh = curved[0]
    out = degrade.apply("noise_normal", mesh, 1.0, 9)
    delta = (out.tris - mesh.tris).reshape(-1, 3)
    assert np.abs(delta).max() > 0
    assert np.abs(delta).max() < 0.5


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
