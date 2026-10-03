import math

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Surface

from unmesh_harness.corpus import ambiguity_entries, load_manifest, select
from unmesh_harness.degrade import apply as apply_degradation
from unmesh_harness.groundtruth import families, generate, validity_problems
from unmesh_harness.labels import tessellate

PAIR_ONE = ("ngon_prism", "coarse_cylinder_prism")
PAIR_TWO = ("one_segment_fillet", "chamfer_same_chord")
PAIR_THREE = ("two_segment_fillet", "two_planes")


@pytest.fixture
def pair_seeds(request):
    return range(32) if request.config.getoption("--slow") else range(8)


def surface_counts(solid):
    counts = {}
    for face in solid.faces():
        key = str(BRepAdaptor_Surface(face.wrapped).GetType()).split(".")[-1]
        counts[key] = counts.get(key, 0) + 1
    return counts


def vertex_set(mesh):
    return np.unique(np.round(mesh.tris.reshape(-1, 3), 6), axis=0)


def triangle_set(mesh):
    return {frozenset(map(tuple, t)) for t in np.round(mesh.tris, 6).tolist()}


def assert_vertex_sets_match(a, b, tol=1e-6):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    assert (d.min(axis=1) <= tol).all()
    assert (d.min(axis=0) <= tol).all()


def preprocessed(family, seed):
    from unmesh_harness.degrade import apply_pair_preprocess

    gt = generate(family, seed)
    lin, ang = gt.parameters["pair_deflection"]
    return apply_pair_preprocess(
        tessellate(gt.solid, lin, ang), gt.parameters["pair_preprocess"], seed
    )


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


def test_pair_one_valid_deterministic_and_closed_form_volume(pair_seeds):
    from .volumes import expected_volume

    for seed in pair_seeds:
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
        assert_vertex_sets_match(va, vb)
        assert math.isclose(ang, 4 * math.pi / a.parameters["n"] * 1.02, rel_tol=1e-9)


def test_pair_one_preprocess_recorded():
    for seed in range(8):
        for family in PAIR_ONE:
            assert generate(family, seed).parameters["pair_preprocess"] == [
                ["canonical_planar", {}]
            ]


def test_pair_one_triangle_sets_match_after_preprocess(pair_seeds):
    for seed in pair_seeds:
        assert triangle_set(preprocessed("ngon_prism", seed)) == triangle_set(
            preprocessed("coarse_cylinder_prism", seed)
        ), seed


def test_pair_two_families_registered():
    assert set(PAIR_TWO) <= set(families())


def test_pair_two_metadata_links_and_labels():
    for seed in range(8):
        f = generate("one_segment_fillet", seed)
        c = generate("chamfer_same_chord", seed)
        assert f.parameters["ambiguity"] == c.parameters["ambiguity"] == "fillet1_vs_chamfer"
        assert f.parameters["truth"] == "fillet"
        assert c.parameters["truth"] == "chamfer"
        assert f.parameters["pair_id"] == f"chamfer_same_chord-{seed:04d}"
        assert c.parameters["pair_id"] == f"one_segment_fillet-{seed:04d}"
        assert f.parameters["box"] == c.parameters["box"]
        assert f.parameters["radius"] == c.parameters["chamfer"]
        assert f.parameters["pair_deflection"] == c.parameters["pair_deflection"]


def test_pair_two_surface_types():
    for seed in range(8):
        assert surface_counts(generate("one_segment_fillet", seed).solid) == {
            "GeomAbs_Plane": 6,
            "GeomAbs_Cylinder": 1,
        }
        assert surface_counts(generate("chamfer_same_chord", seed).solid) == {"GeomAbs_Plane": 7}


def test_pair_two_valid_deterministic_and_closed_form_volume(pair_seeds):
    from .volumes import expected_volume

    for seed in pair_seeds:
        for family in PAIR_TWO:
            gt = generate(family, seed)
            assert validity_problems(gt.solid) == [], (family, seed)
            again = generate(family, seed)
            assert again.solid.volume == pytest.approx(gt.solid.volume, rel=1e-9)
            assert again.parameters == gt.parameters
            assert gt.solid.volume == pytest.approx(expected_volume(gt), rel=1e-9), (family, seed)


def _face_points(mesh, surface, normal=None):
    for face in mesh.faces:
        if face.surface != surface:
            continue
        if normal is not None and not np.allclose(
            sorted(np.abs(face.params["normal"])), sorted(np.abs(normal)), atol=1e-6
        ):
            continue
        return np.unique(np.round(mesh.face_tris(face.id).reshape(-1, 3), 6), axis=0)
    raise AssertionError(f"no {surface} face found")


def test_pair_two_fillet_floor_is_three_segments():
    for seed in range(8):
        gt = generate("one_segment_fillet", seed)
        lin, ang = gt.parameters["pair_deflection"]
        pts = _face_points(tessellate(gt.solid, lin, ang), "cylinder")
        assert len({(x, y) for x, y, _ in pts.tolist()}) == 4, seed


def test_pair_two_chamfer_vertices_are_subset_of_fillet():
    for seed in range(8):
        f = generate("one_segment_fillet", seed)
        c = generate("chamfer_same_chord", seed)
        lin, ang = f.parameters["pair_deflection"]
        vf = vertex_set(tessellate(f.solid, lin, ang))
        vc = vertex_set(tessellate(c.solid, lin, ang))
        assert len(vf) == 14 and len(vc) == 10, seed
        d = np.linalg.norm(vc[:, None, :] - vf[None, :, :], axis=2)
        assert (d.min(axis=1) <= 1e-6).all(), seed


def test_pair_two_vertex_sets_match_at_pair_deflection():
    for seed in range(8):
        f = generate("one_segment_fillet", seed)
        c = generate("chamfer_same_chord", seed)
        lin, ang = f.parameters["pair_deflection"]
        degraded = apply_degradation("fillet_rows", tessellate(f.solid, lin, ang), 0.9, 0)
        assert_vertex_sets_match(
            vertex_set(degraded),
            vertex_set(tessellate(c.solid, lin, ang)),
        )


def test_pair_two_preprocess_recorded():
    for seed in range(8):
        f = generate("one_segment_fillet", seed)
        c = generate("chamfer_same_chord", seed)
        assert f.parameters["pair_preprocess"] == [
            ["fillet_rows", {"segments": 1}],
            ["canonical_planar", {}],
        ]
        assert c.parameters["pair_preprocess"] == [["canonical_planar", {}]]


def test_pair_two_triangle_sets_match_after_preprocess(pair_seeds):
    for seed in pair_seeds:
        assert triangle_set(preprocessed("one_segment_fillet", seed)) == triangle_set(
            preprocessed("chamfer_same_chord", seed)
        ), seed


def test_pair_three_families_registered():
    assert set(PAIR_THREE) <= set(families())


def test_pair_three_metadata_links_and_labels():
    for seed in range(8):
        f = generate("two_segment_fillet", seed)
        t = generate("two_planes", seed)
        assert f.parameters["ambiguity"] == t.parameters["ambiguity"] == "fillet2_vs_two_planes"
        assert f.parameters["truth"] == "fillet"
        assert t.parameters["truth"] == "two_planes"
        assert f.parameters["pair_id"] == f"two_planes-{seed:04d}"
        assert t.parameters["pair_id"] == f"two_segment_fillet-{seed:04d}"
        assert f.parameters["box"] == t.parameters["box"]
        assert f.parameters["radius"] == t.parameters["radius"]
        assert f.parameters["pair_deflection"] == t.parameters["pair_deflection"]


def test_pair_three_surface_types():
    for seed in range(8):
        assert surface_counts(generate("two_segment_fillet", seed).solid) == {
            "GeomAbs_Plane": 6,
            "GeomAbs_Cylinder": 1,
        }
        assert surface_counts(generate("two_planes", seed).solid) == {"GeomAbs_Plane": 8}


def test_pair_three_valid_deterministic_and_closed_form_volume(pair_seeds):
    from .volumes import expected_volume

    for seed in pair_seeds:
        for family in PAIR_THREE:
            gt = generate(family, seed)
            assert validity_problems(gt.solid) == [], (family, seed)
            again = generate(family, seed)
            assert again.solid.volume == pytest.approx(gt.solid.volume, rel=1e-9)
            assert again.parameters == gt.parameters
            assert gt.solid.volume == pytest.approx(expected_volume(gt), rel=1e-9), (family, seed)


def test_pair_three_cut_points_match_fillet_tangents():
    for seed in range(8):
        f = generate("two_segment_fillet", seed)
        t = generate("two_planes", seed)
        lin, ang = f.parameters["pair_deflection"]
        vf = vertex_set(tessellate(f.solid, lin, ang))
        vt = vertex_set(tessellate(t.solid, lin, ang))
        assert len(vf) == 14 and len(vt) == 12, seed
        assert (
            len(
                {
                    (x, y)
                    for x, y, _ in _face_points(tessellate(f.solid, lin, ang), "cylinder").tolist()
                }
            )
            == 4
        ), seed
        p0, _, p2 = (np.array(q) for q in t.features[0]["points"])
        for mesh in (vf, vt):
            d = np.linalg.norm(mesh[:, :2][:, None, :] - np.array([p0, p2])[None, :, :], axis=2)
            assert (d.min(axis=0) <= 1e-6).all(), seed


def test_pair_three_vertex_sets_match_at_pair_deflection():
    for seed in range(8):
        f = generate("two_segment_fillet", seed)
        t = generate("two_planes", seed)
        lin, ang = f.parameters["pair_deflection"]
        degraded = apply_degradation("fillet_rows", tessellate(f.solid, lin, ang), 0.5, 0)
        assert_vertex_sets_match(
            vertex_set(degraded),
            vertex_set(tessellate(t.solid, lin, ang)),
        )


def test_pair_three_preprocess_recorded():
    for seed in range(8):
        f = generate("two_segment_fillet", seed)
        t = generate("two_planes", seed)
        assert f.parameters["pair_preprocess"] == [
            ["fillet_rows", {"segments": 2}],
            ["canonical_planar", {}],
        ]
        assert t.parameters["pair_preprocess"] == [["canonical_planar", {}]]


def test_pair_three_triangle_sets_match_after_preprocess(pair_seeds):
    for seed in pair_seeds:
        assert triangle_set(preprocessed("two_segment_fillet", seed)) == triangle_set(
            preprocessed("two_planes", seed)
        ), seed


def test_new_smoke_entries_pinned():
    ids = {e["id"] for e in ambiguity_entries()[:6]}
    got = [(e["family"], e["seed"]) for e in select(load_manifest(), "smoke") if e["id"] in ids]
    assert got == [(family, 0) for family in PAIR_ONE + PAIR_TWO + PAIR_THREE]


def test_new_standard_entries_pinned():
    ids = {e["id"] for e in ambiguity_entries()}
    entries = [e for e in select(load_manifest(), "standard") if e["id"] in ids]
    assert len(entries) == 30
    assert all(e["strata"] == {"category": "ambiguity"} for e in entries)
