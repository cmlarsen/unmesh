import pytest

pytest.importorskip("OCP")

from unmesh_harness.oracle_fit import (  # noqa: E402
    COARSE_DEFLECTION,
    CURVED_FAMILIES,
    corpus_seeds,
    run_oracle,
    summarize,
)

PLANAR_FAMILIES = ["boss_plate", "plate_pockets", "stepped_block", "planar_chamfer"]


def _check(rows, f1_floor=0.98):
    for row in rows:
        label = f"{row['family']} {row['operator']}"
        assert row["f1"] >= f1_floor, label
        assert row["curved_unscored"] == 0, label
        assert row["radius_ratio_max"] < 1.0, label
        assert row["axis_deg_max"] < 0.1, label


def test_curved_families_recover_on_oracle_segmentation():
    families = list(CURVED_FAMILIES)
    records = run_oracle(families, {f: [0] for f in families}, ["identity", "float32"])
    rows = summarize(records)
    assert len(rows) == 2 * len(families)
    _check(rows)


def test_coarse_tessellation_recovers_design_radius():
    families = ["through_bore", "straight_fillet", "countersink"]
    records = run_oracle(families, {f: [0] for f in families}, ["identity"], COARSE_DEFLECTION)
    rows = summarize(records)
    _check(rows)
    assert all(r["radius_rel_max"] < 1e-9 for r in rows)


def test_planar_families_stay_exact_on_oracle_segmentation():
    records = run_oracle(
        PLANAR_FAMILIES, {f: [0] for f in PLANAR_FAMILIES}, ["identity", "float32"]
    )
    assert all(r["f1"] == 1.0 for r in summarize(records))


@pytest.mark.slow
def test_curved_acceptance_over_the_corpus():
    families = list(CURVED_FAMILIES)
    records = run_oracle(families, corpus_seeds(families), ["identity", "float32"], jobs=2)
    _check(summarize(records))
