import math

import numpy as np
import pytest
from build123d import Box, Cone, Cylinder, Plane, Sphere, Torus, fillet, mirror

import unmesh
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import (
    DEFLECTION_SETTINGS,
    TANGENT_THRESHOLD_DEG,
    LabeledMesh,
    distance_to_surface,
    outward_normals,
    tessellate,
)


def closed_manifold_problems(tris):
    verts, idx, *_ = unmesh.weld(tris, 0.0)
    directed: dict[tuple[int, int], int] = {}
    for a, b, c in idx.tolist():
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    problems = []
    if any(n != 1 for n in directed.values()):
        problems.append("a directed edge is used more than once")
    open_edges = [e for e in directed if (e[1], e[0]) not in directed]
    if open_edges:
        problems.append(f"{len(open_edges)} edges without an opposite half-edge")
    return problems, verts, idx


def signed_volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6)


def mesh_face_pairs(mesh):
    owners: dict[tuple[tuple, tuple], set[int]] = {}
    for tri, fid in zip(mesh.tris, mesh.face_id.tolist(), strict=True):
        keys = [tuple(map(tuple, tri[[i, (i + 1) % 3]])) for i in range(3)]
        for a, b in keys:
            owners.setdefault((min(a, b), max(a, b)), set()).add(fid)
    return {tuple(sorted(f)) for f in owners.values() if len(f) == 2}


def check_part(shape, lin, ang, exact_volume=True):
    mesh = tessellate(shape, lin, ang)
    problems, _, _ = closed_manifold_problems(mesh.tris)
    assert problems == []
    vol = signed_volume(mesh.tris)
    assert vol > 0
    assert vol == pytest.approx(shape.volume, rel=1e-9 if exact_volume else 0.02)
    assert len(mesh.faces) == len(shape.faces())
    assert set(np.unique(mesh.face_id)) == set(range(len(mesh.faces)))
    for face in mesh.faces:
        d = distance_to_surface(face, mesh.face_tris(face.id).mean(axis=1))
        allowed = lin if face.surface in ("plane", "cylinder") else 4 * lin
        assert d.max() <= allowed + 1e-9, (face.id, face.surface, d.max())
    for face in mesh.faces:
        tris = mesh.face_tris(face.id)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        area = np.linalg.norm(n, axis=1)
        keep = area > 1e-12
        assert (outward_normals(face, tris.mean(axis=1)[keep]) * n[keep]).sum(axis=1).min() > 0
    pairs = {tuple(sorted((a.face_a, a.face_b))) for a in mesh.adjacency}
    assert pairs == mesh_face_pairs(mesh)
    return mesh


@pytest.mark.parametrize(("lin", "ang"), DEFLECTION_SETTINGS)
@pytest.mark.parametrize("entry", select(load_manifest(), "smoke"), ids=lambda e: e["id"])
def test_smoke_parts_closed_manifold_and_on_surface(entry, lin, ang):
    check_part(
        generate(entry["family"], entry["seed"]).solid,
        lin,
        ang,
        exact_volume=entry["strata"].get("category", "planar") == "planar",
    )


@pytest.mark.slow
@pytest.mark.parametrize(("lin", "ang"), DEFLECTION_SETTINGS)
@pytest.mark.parametrize("entry", select(load_manifest(), "standard"), ids=lambda e: e["id"])
def test_standard_parts_closed_manifold_and_on_surface(entry, lin, ang):
    check_part(
        generate(entry["family"], entry["seed"]).solid,
        lin,
        ang,
        exact_volume=entry["strata"].get("category", "planar") == "planar",
    )


PRIMITIVES = {
    "plane": (lambda: Box(10, 20, 30), {"plane"}),
    "cylinder": (lambda: Cylinder(5, 10), {"cylinder", "plane"}),
    "cone": (lambda: Cone(5, 2, 10), {"cone", "plane"}),
    "sphere": (lambda: Sphere(5), {"sphere"}),
    "torus": (lambda: Torus(10, 2), {"torus"}),
    "filleted_all": (lambda: fillet(Box(20, 20, 20).edges(), 3), {"plane", "cylinder", "sphere"}),
    "mirrored_drilled": (
        lambda: mirror(Box(10, 10, 10) - Cylinder(2, 20), Plane.YZ),
        {"plane", "cylinder"},
    ),
    "drilled": (lambda: Box(10, 10, 10) - Cylinder(2, 20), {"plane", "cylinder"}),
}


@pytest.mark.parametrize("name", PRIMITIVES)
def test_primitive_surface_types_and_geometry(name):
    make, types = PRIMITIVES[name]
    mesh = check_part(make(), 0.01, 0.2, exact_volume=name == "plane")
    assert {f.surface for f in mesh.faces} == types


def test_surface_parameters():
    by_type = {}
    for name in ("cylinder", "cone", "sphere", "torus"):
        mesh = tessellate(PRIMITIVES[name][0](), 0.05, 0.3)
        by_type.update({f.surface: f.params for f in mesh.faces})
    cyl, cone, sph, tor = (by_type[k] for k in ("cylinder", "cone", "sphere", "torus"))
    assert cyl["radius"] == pytest.approx(5)
    assert abs(cyl["axis"][2]) == pytest.approx(1)
    assert sph["radius"] == pytest.approx(5)
    assert tor["major_radius"] == pytest.approx(10) and tor["minor_radius"] == pytest.approx(2)
    assert cone["half_angle"] == pytest.approx(math.atan2(3, 10))
    assert abs(cone["apex"][2]) > 10


def test_plane_normals_point_outward():
    mesh = tessellate(Box(10, 20, 30), 0.1, 0.5)
    for face in mesh.faces:
        tris = mesh.face_tris(face.id)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        assert np.allclose(n, face.params["normal"], atol=1e-9)
        centre = tris.mean(axis=(0, 1))
        assert centre @ np.array(face.params["normal"]) > 0


def test_adjacency_tangent_versus_transversal():
    box = Box(20, 20, 20)
    mesh = tessellate(box, 0.05, 0.3)
    assert len(mesh.adjacency) == 12
    assert all(not a.tangent and a.curve == "line" for a in mesh.adjacency)
    assert all(a.dihedral == pytest.approx(math.pi / 2) for a in mesh.adjacency)

    rounded = fillet(box.edges().filter_by(lambda e: abs(e.center().Z - 10) < 1e-6)[:1], 3)
    mesh = tessellate(rounded, 0.05, 0.3)
    tangent = [a for a in mesh.adjacency if a.tangent]
    assert len(tangent) == 2
    assert {mesh.faces[a.face_a].surface for a in tangent} | {
        mesh.faces[a.face_b].surface for a in tangent
    } == {"plane", "cylinder"}
    assert all(math.degrees(a.dihedral) < TANGENT_THRESHOLD_DEG for a in tangent)
    assert all(
        math.degrees(a.dihedral) >= TANGENT_THRESHOLD_DEG for a in mesh.adjacency if not a.tangent
    )


def test_cylinder_edges_are_circles():
    mesh = tessellate(Cylinder(5, 10), 0.01, 0.2)
    assert sorted(a.curve for a in mesh.adjacency) == ["circle", "circle"]


def test_npz_and_stl_round_trip(tmp_path):
    mesh = tessellate(Box(10, 10, 10) - Cylinder(2, 20), 0.01, 0.2)
    mesh.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert np.array_equal(back.tris, mesh.tris)
    assert np.array_equal(back.face_id, mesh.face_id)
    assert back.faces == mesh.faces and back.adjacency == mesh.adjacency
    assert (back.linear_deflection, back.angular_deflection) == (0.01, 0.2)

    mesh.write_stl(tmp_path / "m.stl")
    stl = unmesh.read_stl(tmp_path / "m.stl")
    assert stl.shape == mesh.tris.shape
    assert np.allclose(stl, mesh.tris, atol=1e-5)
    assert closed_manifold_problems(stl.astype(np.float64))[0] == []
