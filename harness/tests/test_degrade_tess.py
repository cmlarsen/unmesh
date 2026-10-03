import subprocess
import sys

import numpy as np
import pytest
from build123d import Box, Cylinder

from unmesh_harness import degrade
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import LabeledMesh, distance_to_surface, tessellate

from .test_labels import closed_manifold_problems

TESS_OPS = ("coarsen",)
SLOW_SEED = 11


def box_mesh():
    return tessellate(Box(10, 10, 10), 0.1, 0.5)


def cylinder_mesh():
    return tessellate(Cylinder(5, 20), 0.05, 0.3)


def fillet_mesh():
    gt = generate("one_segment_fillet", 0)
    lin, ang = gt.parameters["pair_deflection"]
    return tessellate(gt.solid, lin, ang)


@pytest.fixture(scope="module")
def box():
    return box_mesh()


@pytest.fixture(scope="module")
def cyl():
    return cylinder_mesh()


@pytest.fixture(scope="module")
def fillet():
    return fillet_mesh()


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
