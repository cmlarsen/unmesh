"""Issue #154: the tolerance floor and the noise estimate must not follow the
part's distance from the origin.

A float64 part must convert identically (tolerances, regions and kinds)
wherever it sits. A rounded input's floor tracks the coordinate rounding it
actually carries: one f32 ulp for a binary STL, the digit step of a decimal
STL, the step of a metre- or inch-rescaled mesh. A part on any of those grids
still converts like its origin self, not like the old always-on translation
term.
"""

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
from unmesh_harness import degrade  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402

LIN, ANG = 0.01, 0.2
COARSE = (0.05, 0.5)
OFFSET = np.array([1000.0, -2000.0, 500.0])
NOISE = (("noise_normal", 0.02),)


def _part(family, seed, steps=(), lin=LIN, ang=ANG):
    mesh = tessellate(generate(family, seed).solid, lin, ang)
    tris = np.asarray(mesh.tris)
    if steps:
        tris = np.asarray(degrade.chain(mesh, list(steps), 2).tris)
    return tris.astype(np.float64)


def _as_float32(tris):
    return tris.astype(np.float32).astype(np.float64)


def _f32_ulp(x):
    return float(np.spacing(np.float32(abs(x))))


def _round_decimals(tris, digits=6):
    return np.vectorize(lambda x: float(f"{x:.{digits}e}"))(tris)


def _to_mm_from_metres(tris):
    return _as_float32(tris / 1000.0) * 1000.0


def _summary(tris):
    ir = unmesh.convert(np.ascontiguousarray(tris)).ir
    kinds: dict[str, int] = {}
    for region in ir.regions:
        kinds[region.surface.type] = kinds.get(region.surface.type, 0) + 1
    return ir.tolerances.linear, kinds, len(ir.regions)


@pytest.mark.parametrize(
    "family,seed,steps,lin,ang",
    [
        ("straight_fillet", 1, (), LIN, ANG),
        ("straight_fillet", 1, NOISE, *COARSE),
        ("corner_fillet", 2, (), LIN, ANG),
    ],
)
def test_float64_conversion_is_translation_invariant(family, seed, steps, lin, ang):
    tris = _part(family, seed, steps, lin, ang)
    tol_here, kinds_here, regions_here = _summary(tris)
    tol_there, kinds_there, regions_there = _summary(tris + OFFSET)

    assert kinds_here == kinds_there
    assert regions_here == regions_there
    # The move keeps only the reconstruction rounding between the two.
    assert tol_there == pytest.approx(tol_here, rel=1e-6)


def test_float32_moved_stays_close_to_the_origin():
    # The noisy case is the reviewer's: origin sigma 3.597e-4 / 50 regions,
    # moved 3.600e-4 / 49.
    origin = _as_float32(_part("straight_fillet", 1, NOISE, *COARSE))
    moved = _as_float32(_part("straight_fillet", 1, NOISE, *COARSE) + OFFSET)
    tol_o, _, regions_o = _summary(origin)
    tol_m, _, regions_m = _summary(moved)

    assert abs(regions_m - regions_o) <= 2
    assert tol_m == pytest.approx(tol_o, rel=0.05)
    assert tol_m >= _f32_ulp(float(np.abs(moved).max()))

    # A clean f32 part stays on the same region count and within twice the
    # origin tolerance (the quantization reads as a small extra noise).
    for family, seed in [("straight_fillet", 1), ("corner_fillet", 2), ("complex_void", 0)]:
        origin = _as_float32(_part(family, seed))
        moved = _as_float32(_part(family, seed) + OFFSET)
        tol_o, _, regions_o = _summary(origin)
        tol_m, _, regions_m = _summary(moved)
        assert abs(regions_m - regions_o) <= 2, family
        assert tol_m <= 2.0 * tol_o, family
        assert tol_m >= _f32_ulp(float(np.abs(moved).max())), family


def test_ascii_and_metre_rescaled_match_main():
    # main's always-on floor covered these grids, so its region and cylinder
    # counts are the reference. Band: +/-2 regions and +/-1 cylinder.
    cases = [
        ("straight_fillet", 1, _round_decimals, 10, 4),
        ("complex_void", 0, _round_decimals, 65, 13),
        ("straight_fillet", 1, _to_mm_from_metres, 10, 4),
        ("complex_void", 0, _to_mm_from_metres, 65, 13),
    ]
    for family, seed, transform, main_regions, main_cyls in cases:
        tris = transform(_part(family, seed) + OFFSET)
        _, kinds, regions = _summary(tris)
        assert abs(regions - main_regions) <= 2, family
        assert abs(kinds.get("cylinder", 0) - main_cyls) <= 1, family


def test_mixed_input_is_stable():
    # A single coordinate nudged off the f32 grid must not change the result.
    tris = _as_float32(_part("corner_fillet", 2) + OFFSET)
    moved = tris.copy()
    moved[0, 0, 0] += 1e-9

    tol_a, kinds_a, regions_a = _summary(tris)
    tol_b, kinds_b, regions_b = _summary(moved)
    assert kinds_a == kinds_b
    assert regions_a == regions_b
    assert tol_a == pytest.approx(tol_b, rel=1e-9)
