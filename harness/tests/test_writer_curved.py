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
TANGENT_FAMILIES = (
    "straight_fillet",
    "circular_fillet",
    "corner_fillet",
    "round_slot_through",
    "round_slot_blind",
    "revolved_dome",
)
EDGE_HAUSDORFF_MM = 1e-4
MAX_SHAPE_TOLERANCE_MM = 1e-3
DEFLECTION = (0.01, 0.2)


def _cases(slow: bool, families=FAMILIES):
    seeds = corpus_seeds(list(families))
    out = []
    for family in families:
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
    assert report.max_shape_tolerance <= MAX_SHAPE_TOLERANCE_MM
    assert bool(report.tangent_edges) == (family in TANGENT_FAMILIES)
    assert {t.curve for t in report.tangent_edges} <= {"line", "circle"}
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


@pytest.mark.parametrize(("family", "seed"), _cases(False, TANGENT_FAMILIES))
def test_oracle_ir_writes_tangent_edges(family, seed, tmp_path):
    _check(family, seed, tmp_path)


@pytest.mark.parametrize(("family", "seed"), _cases(True, TANGENT_FAMILIES))
def test_oracle_ir_writes_tangent_edges_all_seeds(family, seed, tmp_path):
    _check(family, seed, tmp_path)


@pytest.mark.parametrize(("family", "seed"), _cases(True, TANGENT_FAMILIES))
def test_automatic_ir_writes_tangent_edges_without_fallback_all_seeds(family, seed, tmp_path):
    _check(family, seed, tmp_path, automatic=True)


def _operator_cases(slow: bool):
    seeds = corpus_seeds(list(TANGENT_FAMILIES))
    out = []
    for family in TANGENT_FAMILIES if slow else ("corner_fillet",):
        for seed in seeds[family][:3] if slow else seeds[family][:1]:
            for op in ("rotation", "float32"):
                for automatic in (False, True):
                    marks = [pytest.mark.slow] if slow else []
                    out.append(pytest.param(family, seed, op, automatic, marks=marks))
    return out


def _check_operator(family, seed, op, automatic, tmp_path):
    from unmesh_harness.degrade.core import apply

    gt = generate(family, seed)
    mesh = apply(op, tessellate(gt.solid, *DEFLECTION), 1.0, seed)
    tris = np.asarray(mesh.tris).reshape(-1, 3, 3)
    ir = unmesh.convert(tris).ir if automatic else build_oracle_ir(mesh)
    path = tmp_path / f"{family}-{seed}-{op}.step"
    report = step.write(ir, path, mesh=tris)
    assert report.valid and report.verified and report.readback.ok, report.issues
    assert report.fallback is None, report.fallback_reason
    assert report.max_shape_tolerance <= MAX_SHAPE_TOLERANCE_MM
    assert brep_counts(occ.read_step(path)) == brep_counts(gt.solid)


@pytest.mark.parametrize(("family", "seed", "op", "automatic"), _operator_cases(False))
def test_tangent_edges_survive_rotation_and_float32(family, seed, op, automatic, tmp_path):
    _check_operator(family, seed, op, automatic, tmp_path)


@pytest.mark.parametrize(("family", "seed", "op", "automatic"), _operator_cases(True))
def test_tangent_edges_survive_rotation_and_float32_all_families(
    family, seed, op, automatic, tmp_path
):
    _check_operator(family, seed, op, automatic, tmp_path)


NOISY_FAMILIES = (
    "through_bore",
    "blind_bore",
    "counterbore",
    "round_boss",
    "countersink",
    "revolved_cone",
    "revolved_torus",
)


def _noisy_write(family, seed, severity, tmp_path):
    from unmesh_harness import degrade
    from unmesh_harness.judge import judge, under_reports

    mesh = tessellate(generate(family, seed).solid, *DEFLECTION)
    mesh = degrade.chain(mesh, [("noise_isotropic", severity)], 0)
    tris = np.asarray(mesh.tris, dtype=np.float64)
    ir, rep = unmesh.convert(tris)
    measured = judge(ir, tris, None, rep.max_deviation, samples_per_mm2=1.0).input
    assert not under_reports(rep.max_deviation, measured)
    report = step.write(ir, tmp_path / f"{family}-{seed}.step", mesh=tris)
    assert report.valid, report.issues
    return report


@pytest.mark.slow
def test_automatic_fallback_rate_under_1um_noise(tmp_path):
    seeds = corpus_seeds(list(NOISY_FAMILIES))
    reasons = []
    total = 0
    for family in NOISY_FAMILIES:
        for seed in seeds[family]:
            total += 1
            report = _noisy_write(family, seed, 0.02, tmp_path)
            if report.fallback is not None:
                reasons.append(f"{family}-{seed}: {report.fallback_reason}")
    assert len(reasons) <= 0.05 * total, reasons


@pytest.mark.slow
@pytest.mark.parametrize(
    ("family", "seed"),
    [("through_bore", 3), ("round_boss", 5), ("blind_bore", 6), ("revolved_cone", 4)],
)
def test_noisy_junctions_write_analytic_under_5um_noise(family, seed, tmp_path):
    report = _noisy_write(family, seed, 0.1, tmp_path)
    assert report.fallback is None, report.fallback_reason


@pytest.mark.slow
def test_straight_fillet_fallback_rate_under_1um_noise(tmp_path):
    seeds = corpus_seeds(["straight_fillet"])["straight_fillet"]
    reasons = []
    for seed in seeds:
        report = _noisy_write("straight_fillet", seed, 0.02, tmp_path)
        if report.fallback is not None:
            reasons.append(f"straight_fillet-{seed}: {report.fallback_reason}")
        else:
            assert report.max_shape_tolerance <= MAX_SHAPE_TOLERANCE_MM
            assert report.tangent_edges
    assert len(reasons) < 0.05 * len(seeds), reasons
