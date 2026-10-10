"""Issue #156 curved-growth integration checks on the corpus.

The clean/noisy decision and the noise-aware growth must keep the curved faces
that the review found lost, without admitting a degenerate fit at high noise.
Each case builds one corpus part with the harness ground truth and converts it.
"""

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
from unmesh_harness import degrade  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402

LIN, ANG = 0.01, 0.2
CURVED = ("cylinder", "cone", "sphere", "torus")


def convert(family, part, steps, seed=0, flat=None):
    mesh = tessellate(generate(family, part).solid, LIN, ANG)
    tris = np.asarray(mesh.tris).copy()
    if steps:
        tris = np.asarray(degrade.chain(mesh, steps, seed).tris).copy()
    if flat is not None:
        z = tris[..., 2]
        sel = z < z.min() + flat
        z[sel] = np.median(z[sel])
    ir, report = unmesh.convert(tris)
    kinds: dict[str, int] = {}
    for r in ir.regions:
        d = r.to_dict()
        k = (d.get("surface") or {}).get("type")
        kinds[k] = kinds.get(k, 0) + 1
    return ir, report, kinds


def curved(kinds):
    return sum(v for k, v in kinds.items() if k in CURVED)


@pytest.mark.slow
def test_straight_fillet_recovers_four_cylinders():
    _, _, kinds = convert("straight_fillet", 3, [("noise_normal", 0.1)], 3)
    assert kinds.get("cylinder", 0) == 4, kinds


@pytest.mark.slow
def test_complex_void_keeps_thirteen_cylinders():
    _, _, kinds = convert("complex_void", 3, [("noise_isotropic", 0.02)], 3)
    assert kinds.get("cylinder", 0) == 13, kinds


@pytest.mark.slow
def test_flat_base_noisy_scan_still_recovers_fillets():
    _, _, kinds = convert("straight_fillet", 3, [("noise_normal", 0.1)], 0, flat=0.012)
    assert kinds.get("cylinder", 0) == 4, kinds


@pytest.mark.slow
def test_high_noise_leaves_no_degenerate_residual():
    ir, _, kinds = convert("complex_thin", 3, [("noise_normal", 0.3)], 0)
    tol = ir.tolerances.linear
    worst = max((r.to_dict().get("residual") or {}).get("max", 0.0) for r in ir.regions)
    assert worst <= 3.0 * tol, (worst, tol, kinds)


@pytest.mark.slow
def test_cap_reaching_counterbore_keeps_two_cylinders():
    _, _, kinds = convert("counterbore", 3, [("noise_normal", 0.3)], 0)
    assert kinds.get("cylinder", 0) == 2, kinds
