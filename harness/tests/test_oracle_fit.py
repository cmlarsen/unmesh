import pytest

pytest.importorskip("OCP")

from unmesh_harness.oracle_fit import (  # noqa: E402
    COARSE_DEFLECTION,
    CURVED_FAMILIES,
    DEFAULT_DEFLECTION,
    DOUBLY_FAMILIES,
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


def test_curved_families_recover_on_automatic_segmentation():
    families = list(CURVED_FAMILIES)
    records = run_oracle(
        families, {f: [0] for f in families}, ["identity", "float32"], automatic=True
    )
    _check(summarize(records), f1_floor=0.95)


@pytest.mark.slow
def test_automatic_acceptance_over_the_corpus():
    families = list(CURVED_FAMILIES)
    records = run_oracle(
        families, corpus_seeds(families), ["identity", "float32"], jobs=2, automatic=True
    )
    _check(summarize(records), f1_floor=0.95)


def _check_doubly(rows, f1_floor, truth=False):
    for row in rows:
        label = f"{row['family']} {row['operator']}"
        assert row["f1"] >= f1_floor, label
        assert row["curved_unscored"] == 0, label
        assert row["radius_rel_max"] < 0.01, label
        assert row["radius_ratio_max"] < 1.0, label
        if truth:
            assert row["dev_truth_max"] < 2 * DEFAULT_DEFLECTION[0], label


def test_spheres_and_tori_recover_on_oracle_segmentation():
    families = list(DOUBLY_FAMILIES)
    records = run_oracle(
        families, {f: [0] for f in families}, ["identity", "float32"], jobs=2, truth=True
    )
    rows = summarize(records)
    assert len(rows) == 2 * len(families)
    _check_doubly(rows, 1.0, truth=True)


def test_spheres_and_tori_recover_on_automatic_segmentation():
    families = list(DOUBLY_FAMILIES)
    records = run_oracle(
        families, {f: [0] for f in families}, ["identity", "float32"], jobs=2, automatic=True
    )
    _check_doubly(summarize(records), 1.0)


def test_corner_fillet_keeps_its_planes():
    records = run_oracle(
        ["corner_fillet"], {"corner_fillet": [0, 1, 2, 3]}, ["identity"], automatic=True
    )
    assert all(r["planes"] == 6 and r["planes_recovered"] == 6 for r in records)


@pytest.mark.slow
def test_doubly_acceptance_over_the_corpus():
    families = list(DOUBLY_FAMILIES)
    seeds = corpus_seeds(families)
    oracle = run_oracle(families, seeds, ["identity", "float32"], jobs=2, truth=True)
    _check_doubly(summarize(oracle), 0.95, truth=True)
    automatic = run_oracle(families, seeds, ["identity", "float32"], jobs=2, automatic=True)
    rows = summarize(automatic)
    _check_doubly(rows, 0.9)
    corner = [r for r in rows if r["family"] == "corner_fillet"]
    assert all(r["planes_recovered"] >= 0.9 * r["planes"] for r in corner)
