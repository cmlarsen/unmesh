"""Issue #154: the tolerance floor and the noise estimate must not follow the
part's distance from the origin.

Physical position dependence comes only from float32 quantization: a binary STL
stores each coordinate rounded to the float32 grid, whose step grows with |p|.
A float64 part must convert identically (tolerances, regions, kinds and snap
decisions) wherever it sits; a float32 part's floor must track the actual
quantization step and its noise estimate must not read quantization as noise.
"""

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
from unmesh_harness import degrade  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402

LIN, ANG = 0.01, 0.2
OFFSET = np.array([1000.0, -2000.0, 500.0])


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


def _diag_floor(tris):
    return 1e-6 * float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0)))


def _summary(ir):
    kinds: dict[str, int] = {}
    for region in ir.regions:
        kinds[region.surface.type] = kinds.get(region.surface.type, 0) + 1
    tangencies = sum(1 for a in ir.adjacencies for b in a.boundaries if b.kind == "tangent")
    return ir.tolerances.linear, kinds, tangencies, len(ir.regions)


@pytest.mark.parametrize(
    "family,seed,steps,lin,ang",
    [
        ("straight_fillet", 1, (), 0.01, 0.2),
        ("straight_fillet", 1, (("noise_normal", 0.02),), 0.05, 0.5),
        ("corner_fillet", 2, (), 0.01, 0.2),
    ],
)
def test_float64_conversion_is_translation_invariant(family, seed, steps, lin, ang):
    tris = _part(family, seed, steps, lin, ang)
    here = _summary(unmesh.convert(tris).ir)
    there = _summary(unmesh.convert(tris + OFFSET).ir)

    tol_here, kinds_here, tang_here, regions_here = here
    tol_there, kinds_there, tang_there, regions_there = there

    assert kinds_here == kinds_there
    assert regions_here == regions_there
    assert tang_here == tang_there
    # The move keeps only the reconstruction rounding between the two.
    assert tol_there == pytest.approx(tol_here, rel=1e-9)


def test_float32_floor_follows_the_quantization_step():
    origin = _as_float32(_part("corner_fillet", 2))
    moved = _as_float32(_part("corner_fillet", 2) + OFFSET)
    ulp = _f32_ulp(float(np.abs(moved).max()))

    # Far from the origin the quantization step dominates the diagonal floor,
    # and the clean part's tolerance is exactly that step.
    assert ulp > _diag_floor(moved)
    assert unmesh.convert(moved).ir.tolerances.linear == pytest.approx(ulp, rel=1e-12)

    # At the origin the quantization step is below the diagonal floor, so the
    # floor (and the part) is the same as for float64 input.
    assert _f32_ulp(float(np.abs(origin).max())) < _diag_floor(origin)


def test_float32_noise_estimate_subtracts_the_quantization_share():
    steps = (("noise_normal", 0.02),)
    origin = _as_float32(_part("straight_fillet", 1, steps, 0.05, 0.5))
    moved = _as_float32(_part("straight_fillet", 1, steps, 0.05, 0.5) + OFFSET)

    sigma_origin = unmesh.convert(origin).ir.tolerances.linear / 5.0
    moved_ir = unmesh.convert(moved).ir
    sigma_moved = moved_ir.tolerances.linear / 5.0

    # The far part's estimate is the origin estimate with the quantization
    # variance (ulp^2 / 12) removed in quadrature, not inflated by it.
    quant_std = _f32_ulp(float(np.abs(moved).max())) / np.sqrt(12.0)
    expected = np.sqrt(max(0.0, sigma_origin**2 - quant_std**2))
    assert sigma_moved == pytest.approx(expected, rel=0.03)
    assert sigma_moved < sigma_origin

    # Region count moves only by the few faces the quantization re-segments.
    origin_regions = len(unmesh.convert(origin).ir.regions)
    assert abs(len(moved_ir.regions) - origin_regions) <= 6
