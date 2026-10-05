import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh.step as step  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402
from unmesh_harness.oracle import build_oracle_ir, force_facets  # noqa: E402
from unmesh_harness.oracle_fit import corpus_seeds  # noqa: E402

FAMILIES = {
    "planar": (
        "boss_plate",
        "lshape_outline",
        "plate_pockets",
        "polygon_prism",
        "rotated_pockets",
        "square_slots",
        "stepped_block",
        "thin_walls",
        "through_cuts",
    ),
    "curved": (
        "blind_bore",
        "counterbore",
        "countersink",
        "revolved_cone",
        "revolved_dome",
        "revolved_torus",
        "round_boss",
        "round_slot_blind",
        "round_slot_through",
        "through_bore",
    ),
    "chamfer_fillet": (
        "bore_chamfer",
        "circular_fillet",
        "corner_fillet",
        "planar_chamfer",
        "straight_fillet",
    ),
}
SEEDS_PER_FAMILY = 5
MAX_SHAPE_TOLERANCE_MM = 1e-3
DEFLECTION = (0.01, 0.2)


def forced_region(ir, seed: int) -> int:
    rng = np.random.default_rng(seed)
    curved = [r.id for r in ir.regions if r.surface.type not in ("plane", "facets")]
    return int(rng.choice(curved or [r.id for r in ir.regions]))


def _cases():
    families = [f for group in FAMILIES.values() for f in group]
    seeds = corpus_seeds(families)
    return [
        pytest.param(f, s, marks=[pytest.mark.slow])
        for f in families
        for s in seeds[f][:SEEDS_PER_FAMILY]
    ]


def _check(family, seed, tmp_path):
    gt = generate(family, seed)
    mesh = tessellate(gt.solid, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = build_oracle_ir(mesh)
    region = forced_region(ir, seed)
    ir = force_facets(ir, region, tris)
    report = step.write(ir, tmp_path / f"{family}-{seed}.step", mesh=tris)
    assert report.valid and report.verified and report.readback.ok, report.issues
    assert report.fallback is None, report.fallback_reason
    assert report.max_shape_tolerance <= MAX_SHAPE_TOLERANCE_MM
    assert report.faceted_regions == 1
    assert report.faceted_faces >= len(ir.regions[region].triangles)
    gap = step.WriteOptions().max_seam_gap
    assert all(s.max_gap <= gap for s in report.seams if s.surface_type != "plane")


@pytest.mark.parametrize(("family", "seed"), [("countersink", 0), ("corner_fillet", 0)])
def test_one_forced_facets_region_writes_mixed(family, seed, tmp_path):
    _check(family, seed, tmp_path)


@pytest.mark.parametrize(("family", "seed"), _cases())
def test_one_forced_facets_region_writes_mixed_all_families(family, seed, tmp_path):
    _check(family, seed, tmp_path)


def test_force_facets_keeps_a_valid_ir():
    gt = generate("corner_fillet", 0)
    mesh = tessellate(gt.solid, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = build_oracle_ir(mesh)
    for region in range(len(ir.regions)):
        forced = force_facets(ir, region, tris)
        forced.validate()
        assert forced.regions[region].surface.type == "facets"
        assert forced.regions[region].residual is None
        assert sum(r.surface.type == "facets" for r in forced.regions) == 1
