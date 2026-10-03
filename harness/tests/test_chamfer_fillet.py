import pytest
from OCP.BRepAdaptor import BRepAdaptor_Surface

from unmesh.ir import validate
from unmesh_harness.corpus import chamfer_fillet_entries, load_manifest, select
from unmesh_harness.groundtruth import families, generate, validity_problems
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir

NEW_FAMILIES = (
    "planar_chamfer",
    "straight_fillet",
    "circular_fillet",
    "bore_chamfer",
    "corner_fillet",
)

EXPECTED_SURFACES = {
    "planar_chamfer": {"GeomAbs_Plane": 10},
    "straight_fillet": {"GeomAbs_Plane": 6, "GeomAbs_Cylinder": 4},
    "circular_fillet": {"GeomAbs_Plane": 2, "GeomAbs_Cylinder": 1, "GeomAbs_Torus": 1},
    "corner_fillet": {"GeomAbs_Plane": 6, "GeomAbs_Cylinder": 12, "GeomAbs_Sphere": 8},
}


def surface_counts(solid):
    counts = {}
    for face in solid.faces():
        key = str(BRepAdaptor_Surface(face.wrapped).GetType()).split(".")[-1]
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_families_registered():
    assert set(NEW_FAMILIES) <= set(families())


def new_smoke_entries():
    ids = {e["id"] for e in chamfer_fillet_entries()[:10]}
    return [e for e in select(load_manifest(), "smoke") if e["id"] in ids]


def test_new_smoke_entries_pinned():
    assert [(e["family"], e["seed"]) for e in new_smoke_entries()] == [
        (family, seed) for seed in (0, 1) for family in NEW_FAMILIES
    ]


@pytest.mark.parametrize("family", [f for f in NEW_FAMILIES if f != "bore_chamfer"])
def test_surface_types(family):
    for seed in range(24):
        gt = generate(family, seed)
        assert validity_problems(gt.solid) == [], (family, seed)
        assert surface_counts(gt.solid) == EXPECTED_SURFACES[family], (family, seed)


def test_bore_chamfer_surface_types():
    seen = set()
    for seed in range(24):
        gt = generate("bore_chamfer", seed)
        counts = surface_counts(gt.solid)
        variant = gt.parameters["variant"]
        if variant == "bore":
            assert counts == {"GeomAbs_Plane": 6, "GeomAbs_Cylinder": 1, "GeomAbs_Cone": 1}
        else:
            assert counts == {"GeomAbs_Plane": 2, "GeomAbs_Cylinder": 1, "GeomAbs_Cone": 1}
        seen.add(variant)
    assert seen == {"bore", "disc"}


def test_feature_sizes_in_range():
    for entry in chamfer_fillet_entries():
        gt = generate(entry["family"], entry["seed"])
        params = gt.parameters
        size = params.get("chamfer", params.get("radius", params.get("fillet")))
        assert 0.2 <= size <= 10, (entry, size)


def test_corner_fillet_face_tags():
    for seed in range(24):
        gt = generate("corner_fillet", seed)
        counts = surface_counts(gt.solid)
        spheres = {
            str(i)
            for i, face in enumerate(gt.solid.faces())
            if str(BRepAdaptor_Surface(face.wrapped).GetType()).endswith("Sphere")
        }
        assert set(gt.face_tags) == spheres
        assert set(gt.face_tags.values()) == {"corner_blend"}
        assert len(spheres) == counts["GeomAbs_Sphere"] == 8
        assert gt.metadata()["face_tags"] == gt.face_tags
        again = generate("corner_fillet", seed)
        assert again.face_tags == gt.face_tags


def tangent_pairs(ir, mesh):
    out = set()
    for adj in ir.adjacencies:
        kinds = {b.kind for b in adj.boundaries}
        if kinds == {"tangent"}:
            a, b = adj.regions
            out.add(tuple(sorted((mesh.faces[a].surface, mesh.faces[b].surface))))
    return out


@pytest.mark.parametrize(
    ("family", "seed", "pairs"),
    [
        ("straight_fillet", 0, {("cylinder", "plane")}),
        ("circular_fillet", 0, {("cylinder", "torus"), ("plane", "torus")}),
        ("corner_fillet", 0, {("cylinder", "plane"), ("cylinder", "sphere")}),
    ],
)
def test_fillet_boundaries_are_tangent(family, seed, pairs):
    mesh = tessellate(generate(family, seed).solid, 0.01, 0.2)
    ir = build_oracle_ir(mesh)
    assert validate(ir) == []
    assert pairs <= tangent_pairs(ir, mesh)


@pytest.mark.slow
@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_seed_sweep_valid_and_closed_form_volume(family):
    from .volumes import expected_volume

    for seed in range(300):
        gt = generate(family, seed)
        assert validity_problems(gt.solid) == [], seed
        assert gt.solid.volume == pytest.approx(expected_volume(gt), rel=1e-9), seed


@pytest.mark.slow
def test_new_standard_parts_tessellate_and_validate():
    ids = {e["id"] for e in chamfer_fillet_entries()}
    entries = [e for e in select(load_manifest(), "standard") if e["id"] in ids]
    assert len(entries) == 120
    for entry in entries:
        solid = generate(entry["family"], entry["seed"]).solid
        for lin, ang in ((0.1, 0.5), (0.01, 0.2)):
            mesh = tessellate(solid, lin, ang)
            assert len(mesh.faces) == len(solid.faces())
            assert validate(build_oracle_ir(mesh)) == []
