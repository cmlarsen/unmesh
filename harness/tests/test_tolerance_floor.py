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
BIG = np.array([3.0e4, 1.0e3, 0.0])
X1050 = np.array([1050.0, 0.0, 0.0])
X1400 = np.array([1400.0, 0.0, 0.0])
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


def _round_general(tris):
    return np.vectorize(lambda x: float(f"{x:g}"))(tris)


def _to_mm_from_metres(tris):
    return _as_float32(tris / 1000.0) * 1000.0


def _to_mm_from_inches(tris):
    return _as_float32(tris / 25.4) * 25.4


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
    # moved 3.600e-4 / 49 on Linux (47 / 40 on macOS). The region count of
    # this over-segmented noisy part varies by platform; main collapses it to
    # 18-19 when moved, which the 20% band still catches.
    origin = _as_float32(_part("straight_fillet", 1, NOISE, *COARSE))
    moved = _as_float32(_part("straight_fillet", 1, NOISE, *COARSE) + OFFSET)
    tol_o, _, regions_o = _summary(origin)
    tol_m, _, regions_m = _summary(moved)

    assert abs(regions_m - regions_o) <= max(2, 0.2 * regions_o)
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
        ("straight_fillet", 1, _to_mm_from_inches, 10, 4),
        ("complex_void", 0, _to_mm_from_inches, 65, 13),
    ]
    for family, seed, transform, main_regions, main_cyls in cases:
        tris = transform(_part(family, seed) + OFFSET)
        _, kinds, regions = _summary(tris)
        assert abs(regions - main_regions) <= 2, family
        assert abs(kinds.get("cylinder", 0) - main_cyls) <= 1, family


@pytest.mark.parametrize(
    "family,seed,offset",
    [
        ("ngon_prism", 0, OFFSET),
        ("counterbore", 1, OFFSET),
        ("through_cuts", 0, OFFSET),
        ("square_slots", 1, BIG),
        ("rotated_pockets", 0, BIG),
    ],
)
def test_float64_moved_equals_origin_on_the_rounding_defect_parts(family, seed, offset):
    # The divisor-scan detector fitted a spurious grid on these float64 parts
    # once they were moved far enough that a divisor of the geometry's own
    # vertex spacing landed inside its acceptance band. The exact-match models
    # find no grid, so the moved part keeps its origin tolerance exactly.
    tris = _part(family, seed)
    tol_o, kinds_o, regions_o = _summary(tris)
    tol_m, kinds_m, regions_m = _summary(tris + offset)

    assert kinds_m == kinds_o, family
    assert regions_m == regions_o, family
    assert tol_m == pytest.approx(tol_o, rel=1e-6), family


@pytest.mark.parametrize("offset", [X1050, X1400, BIG])
def test_seven_digit_ascii_moved_matches_main(offset):
    # A %.6e grid (7 significant digits) has step 1e-3 at max_abs in
    # [1000, 2000) and 1e-2 at 3e4. The old detector rejected the step (it
    # exceeded 5e-7 * max_abs) and fragmented through_bore-0 from 7 regions to
    # 94-156; main's region count is the reference.
    tris = _round_decimals(_part("through_bore", 0) + offset)
    _, kinds, regions = _summary(tris)
    assert abs(regions - 7) <= 2, offset
    assert kinds.get("cylinder", 0) == 1


def test_percent_g_moved_is_no_worse_than_main():
    # %g keeps 6 significant digits: step 1e-2 at the round-1 offset. main
    # fragmented through_bore-0 to 152 regions; the exact-match model recovers
    # the origin's 7.
    tris = _round_general(_part("through_bore", 0) + OFFSET)
    _, kinds, regions = _summary(tris)
    assert regions <= 152 + 2
    assert abs(regions - 7) <= 2
    assert kinds.get("cylinder", 0) == 1


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


def _plate(L, W, H, pocket, nx, ny, pitch_x, pitch_y, x0, y0, depth, holes=0, r=2.0):
    # One boolean with every pocket and hole as a tool: the same geometry as
    # cutting them one by one, but about two orders of magnitude faster.
    from build123d import Box, Cylinder, Pos

    tools = [
        Pos(x0 + i * pitch_x + pocket[0] / 2, y0 + j * pitch_y + pocket[1] / 2, H - depth / 2)
        * Box(pocket[0], pocket[1], depth)
        for i in range(nx)
        for j in range(ny)
    ]
    tools += [Pos(x0 / 2, y0 + k * pitch_y + 2.5, H / 2) * Cylinder(r, H) for k in range(holes)]
    return Pos(L / 2, W / 2, H / 2) * Box(L, W, H) - tools


def _plate_tris(solid):
    return np.asarray(tessellate(solid, LIN, ANG).tris, dtype=np.float64)


_INCH = 25.4 / 8
_FRAC = np.array([1000.37, -2000.11, 500.29])
# The reviewer's round-number plates: design geometry on a coarse grid, not
# rounded data. main's tolerance at the origin is `1e-6 * diag`, i.e. no
# rounding model at all; the offset move must not inflate it.
_PLATES = {
    "p01": (
        _plate(130.3, 90.7, 12.1, (3.3, 2.7), 20, 16, 6.1, 5.3, 4.9, 3.7, 3.3),
        np.zeros(3),
        1.5922e-4,
        1606,
    ),
    "pint": (
        _plate(130, 90, 12, (3, 2), 20, 16, 6, 5, 5, 4, 3),
        _FRAC,
        1.5857e-4,
        1606,
    ),
    "p05": (
        _plate(130.5, 90.5, 12.5, (3.5, 2.5), 20, 16, 6.0, 5.5, 4.5, 3.5, 3.5),
        _FRAC,
        1.5930e-4,
        1606,
    ),
    "pinch": (
        _plate(
            44 * _INCH,
            30 * _INCH,
            4 * _INCH,
            (1 * _INCH, 1 * _INCH),
            20,
            14,
            2 * _INCH,
            2 * _INCH,
            2 * _INCH,
            2 * _INCH,
            1 * _INCH,
        ),
        X1050,
        1.6956e-4,
        1406,
    ),
}


@pytest.mark.parametrize("name", sorted(_PLATES))
def test_round_number_design_plates_keep_their_origin_tolerance(name):
    # A 0.1 mm grid at the origin, an integer-mm and a 0.5 mm grid at
    # (1000.37, -2000.11, 500.29) and a 1/8-inch grid at x+1050 are exact
    # design geometry with nothing rounded. The coarsest accepted model used to
    # be a two-fixed-decimals (or thousandths) grid, whose last digit is
    # degenerate, raising the tolerance 6-60x. The last-digit gate rejects
    # those models, so the moved plate keeps the origin's `1e-6 * diag` floor
    # (main's origin tolerance) for every offset.
    solid, offset, origin_tol, origin_regions = _PLATES[name]
    tris = _plate_tris(solid)
    tol_o, kinds_o, regions_o = _summary(tris)
    assert tol_o == pytest.approx(origin_tol, rel=5e-3), name
    assert abs(regions_o - origin_regions) <= 2, name

    tol_m, kinds_m, regions_m = _summary(tris + offset)
    assert kinds_m == kinds_o, name
    assert regions_m == regions_o, name
    assert tol_m == pytest.approx(tol_o, rel=1e-9), name
