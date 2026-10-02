import json
import math

import pytest

from unmesh_harness.corpus import (
    GRID_SIZES,
    build_grid,
    default_manifest,
    find_manifest,
    load_manifest,
    select,
)
from unmesh_harness.groundtruth import families, fingerprint, generate, validity_problems


def _entries(grid):
    return [(e["family"], e["seed"]) for e in select(load_manifest(), grid)]


def test_manifest_is_canonical():
    assert json.loads(find_manifest().read_text()) == default_manifest()


@pytest.mark.parametrize("grid", ["smoke", "standard"])
def test_grid_sizes(grid):
    assert len(_entries(grid)) == GRID_SIZES[grid]


def test_smoke_subset_of_standard_and_all_families_covered():
    assert set(_entries("smoke")) <= set(_entries("standard"))
    assert {f for f, _ in _entries("smoke")} == set(families())


def test_unknown_family():
    with pytest.raises(KeyError):
        generate("nope", 0)


@pytest.mark.parametrize(("family", "seed"), _entries("smoke"))
def test_smoke_valid_and_deterministic(family, seed):
    a, b = generate(family, seed), generate(family, seed)
    assert validity_problems(a.solid) == []
    fa, fb = fingerprint(a.solid), fingerprint(b.solid)
    assert math.isclose(fa["volume"], fb["volume"], rel_tol=1e-9)
    assert fa["face_count"] == fb["face_count"]
    assert fa["bbox_min"] == pytest.approx(fb["bbox_min"], abs=1e-9)
    assert fa["bbox_max"] == pytest.approx(fb["bbox_max"], abs=1e-9)
    assert a.features and a.metadata()["family"] == family


@pytest.mark.slow
@pytest.mark.parametrize(("family", "seed"), _entries("standard"))
def test_standard_valid(family, seed):
    assert validity_problems(generate(family, seed).solid) == []


def test_seeds_differ():
    assert generate("plate_pockets", 0).parameters != generate("plate_pockets", 1).parameters


def test_build_writes_step_and_metadata(tmp_path):
    manifest = load_manifest()
    manifest["entries"] = manifest["entries"][:2]
    assert build_grid(manifest, "smoke", tmp_path) == 2
    for entry in manifest["entries"]:
        assert (tmp_path / f"{entry['id']}.step").stat().st_size > 0
        meta = json.loads((tmp_path / f"{entry['id']}.json").read_text())
        assert meta["family"] == entry["family"] and meta["features"]
