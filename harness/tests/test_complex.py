import pytest

from unmesh.ir import validate
from unmesh_harness.corpus import COMPLEX_FAMILIES, load_manifest, select
from unmesh_harness.groundtruth import generate, validity_problems
from unmesh_harness.labels import tessellate
from unmesh_harness.oracle import build_oracle_ir
from unmesh_harness.strata import min_edge_length


def _complex_entries(grid="standard"):
    return [e for e in select(load_manifest(), grid) if e["family"] in COMPLEX_FAMILIES]


def test_complex_face_counts_in_range():
    for entry in _complex_entries():
        n = entry["fingerprint"]["face_count"]
        assert 50 <= n <= 300, entry["id"]


def test_complex_smoke_covers_all_complex_families():
    assert {e["family"] for e in _complex_entries("smoke")} == set(COMPLEX_FAMILIES)


def _problems(gt):
    return validity_problems(
        gt.solid, solids=gt.parameters.get("solids", 1), shells=gt.parameters.get("shells", 1)
    )


@pytest.mark.parametrize("family", COMPLEX_FAMILIES)
def test_complex_smoke_valid_with_expected_counts(family):
    gt = generate(family, 0)
    assert _problems(gt) == []
    assert gt.features


def test_complex_thin_has_thin_walls_and_small_features():
    thin = False
    small = False
    for seed in range(15):
        for f in generate("complex_thin", seed).features:
            if f["type"] == "thin_fin" and f["thickness"] <= 0.6:
                thin = True
            if f["type"] == "micro_bore" and f["radius"] <= 0.5:
                small = True
    assert thin and small


def test_complex_thin_smallest_edges_near_deflection():
    edges = min(min_edge_length(generate("complex_thin", seed).solid) for seed in range(15))
    assert edges < 1.0


def test_complex_void_has_cavity_shell():
    gt = generate("complex_void", 0)
    assert gt.parameters["shells"] == 2
    assert any(f["type"] == "internal_void" for f in gt.features)
    mesh = tessellate(gt.solid, 0.01, 0.2)
    assert [(s.role, s.closed) for s in mesh.shells] == [("outer", True), ("cavity", True)]
    assert build_oracle_ir(mesh).shells[1].role == "cavity"


def test_complex_assembly_has_three_bodies():
    gt = generate("complex_assembly", 0)
    assert (gt.parameters["solids"], gt.parameters["shells"]) == (3, 4)
    assert {f["body"] for f in gt.features} == {0, 1, 2}
    mesh = tessellate(gt.solid, 0.01, 0.2)
    assert [s.role for s in mesh.shells].count("outer") == 3


@pytest.mark.parametrize("family", COMPLEX_FAMILIES)
def test_complex_smoke_oracle_validates(family):
    mesh = tessellate(generate(family, 0).solid, 0.01, 0.2)
    assert validate(build_oracle_ir(mesh)) == []
