import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
import unmesh.step as step  # noqa: E402
from unmesh._writer import occ  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402
from unmesh_harness.metrics.edges import brep_counts, edge_hausdorff  # noqa: E402
from unmesh_harness.oracle import build_oracle_ir  # noqa: E402
from unmesh_harness.oracle_fit import corpus_seeds  # noqa: E402

FAMILIES = (
    "through_bore",
    "blind_bore",
    "counterbore",
    "round_boss",
    "countersink",
    "bore_chamfer",
    "revolved_cone",
    "revolved_torus",
)
EDGE_HAUSDORFF_MM = 1e-4
DEFLECTION = (0.01, 0.2)


def _cases(slow: bool):
    seeds = corpus_seeds(list(FAMILIES))
    out = []
    for family in FAMILIES:
        for seed in seeds[family] if slow else seeds[family][:1]:
            out.append(pytest.param(family, seed, marks=[pytest.mark.slow] if slow else []))
    return out


def _check(family, seed, tmp_path, automatic=False):
    gt = generate(family, seed)
    mesh = tessellate(gt.solid, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = unmesh.convert(tris).ir if automatic else build_oracle_ir(mesh)
    path = tmp_path / f"{family}-{seed}.step"
    report = step.write(ir, path, mesh=tris)
    assert report.valid and report.verified and report.readback.ok, report.issues
    assert report.fallback is None, report.fallback_reason
    assert not report.edge_fallbacks
    written = occ.read_step(path)
    assert brep_counts(written) == brep_counts(gt.solid)
    if not automatic:
        h = edge_hausdorff(written, gt.solid)
        assert h["edges"] == h["truth_edges"] == h["matched"]
        assert h["max"] < EDGE_HAUSDORFF_MM


@pytest.mark.parametrize(("family", "seed"), _cases(slow=False))
def test_oracle_ir_writes_exact_edges(family, seed, tmp_path):
    _check(family, seed, tmp_path)


@pytest.mark.parametrize(("family", "seed"), _cases(slow=True))
def test_oracle_ir_writes_exact_edges_all_seeds(family, seed, tmp_path):
    _check(family, seed, tmp_path)


@pytest.mark.parametrize(("family", "seed"), _cases(slow=True))
def test_automatic_ir_writes_without_fallback_all_seeds(family, seed, tmp_path):
    _check(family, seed, tmp_path, automatic=True)


@pytest.mark.parametrize("family", ["circular_fillet", "straight_fillet", "round_slot_through"])
def test_tangent_adjacencies_fall_back_with_reason(family, tmp_path):
    gt = generate(family, 0)
    mesh = tessellate(gt.solid, *DEFLECTION)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    report = step.write(build_oracle_ir(mesh), tmp_path / "t.step", mesh=tris)
    assert report.valid and report.fallback == "faceted"
    assert "tangent edges not supported yet (#33)" in report.fallback_reason
