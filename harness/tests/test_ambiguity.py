import math

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Surface

from unmesh_harness.groundtruth import families, generate, validity_problems
from unmesh_harness.labels import tessellate

PAIR_ONE = ("ngon_prism", "coarse_cylinder_prism")


def surface_counts(solid):
    counts = {}
    for face in solid.faces():
        key = str(BRepAdaptor_Surface(face.wrapped).GetType()).split(".")[-1]
        counts[key] = counts.get(key, 0) + 1
    return counts


def vertex_set(mesh):
    return np.unique(np.round(mesh.tris.reshape(-1, 3), 6), axis=0)


def assert_vertex_sets_match(a, b, tol=1e-6):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    assert (d.min(axis=1) <= tol).all()
    assert (d.min(axis=0) <= tol).all()


def test_families_registered():
    assert set(PAIR_ONE) <= set(families())


def test_pair_one_metadata_links():
    for seed in range(8):
        for family, partner in (
            ("ngon_prism", "coarse_cylinder_prism"),
            ("coarse_cylinder_prism", "ngon_prism"),
        ):
            gt = generate(family, seed)
            p = gt.parameters
            assert p["ambiguity"] == "ngon_vs_cylinder"
            assert p["pair_family"] == partner
            assert p["pair_id"] == f"{partner}-{seed:04d}"
            assert gt.metadata()["parameters"]["pair_id"] == p["pair_id"]
            lin, ang = p["pair_deflection"]
            assert lin > 0 and ang > 0
        a, b = (generate(f, seed) for f in PAIR_ONE)
        assert a.parameters["pair_deflection"] == b.parameters["pair_deflection"]
        assert a.parameters["n"] == b.parameters["n"]
        assert (a.parameters["radius"], a.parameters["height"]) == (
            b.parameters["radius"],
            b.parameters["height"],
        )


def test_pair_one_truth_labels():
    assert generate("ngon_prism", 0).parameters["truth"] == "polygon"
    assert generate("coarse_cylinder_prism", 0).parameters["truth"] == "cylinder"


def test_pair_one_surface_types():
    for seed in range(8):
        ngon = generate("ngon_prism", seed)
        n = ngon.parameters["n"]
        assert 5 <= n <= 32
        assert surface_counts(ngon.solid) == {"GeomAbs_Plane": n + 2}
        cyl = generate("coarse_cylinder_prism", seed)
        assert surface_counts(cyl.solid) == {"GeomAbs_Plane": 2, "GeomAbs_Cylinder": 1}


def test_pair_one_valid_deterministic_and_closed_form_volume():
    from .volumes import expected_volume

    for seed in range(32):
        for family in PAIR_ONE:
            gt = generate(family, seed)
            assert validity_problems(gt.solid) == [], (family, seed)
            again = generate(family, seed)
            assert again.solid.volume == pytest.approx(gt.solid.volume, rel=1e-9)
            assert again.parameters == gt.parameters
            assert gt.solid.volume == pytest.approx(expected_volume(gt), rel=1e-9), (family, seed)


def test_pair_one_vertex_sets_match_at_pair_deflection():
    for seed in range(24):
        a = generate("ngon_prism", seed)
        b = generate("coarse_cylinder_prism", seed)
        lin, ang = a.parameters["pair_deflection"]
        va = vertex_set(tessellate(a.solid, lin, ang))
        vb = vertex_set(tessellate(b.solid, lin, ang))
        assert len(va) == len(vb) == 2 * a.parameters["n"], seed
        assert_vertex_sets_match(va, vb), seed
        assert math.isclose(ang, 4 * math.pi / a.parameters["n"], rel_tol=1e-6)
