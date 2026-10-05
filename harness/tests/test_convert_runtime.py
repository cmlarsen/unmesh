import time

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
from unmesh_harness import degrade  # noqa: E402
from unmesh_harness.corpus import chamfer_fillet_entries, curved_entries  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402

US_PER_TRIANGLE = 20.0
REGIONS_PER_KTRI = 30.0


@pytest.mark.benchmark
def test_curved_corpus_converts_without_shattering():
    entries = [e for e in curved_entries() + chamfer_fillet_entries() if e["seed"] < 2]
    meshes = [
        np.asarray(tessellate(generate(e["family"], e["seed"]).solid, 0.01, 0.2).tris)
        for e in entries
    ]
    triangles = sum(len(m) for m in meshes)
    regions = 0
    start = time.process_time()
    for m in meshes:
        regions += len(unmesh.convert(m).ir.regions)
    seconds = time.process_time() - start
    assert seconds * 1e6 / triangles <= US_PER_TRIANGLE, (seconds, triangles)
    assert regions * 1000 / triangles <= REGIONS_PER_KTRI, (regions, triangles)


NOISY_TO_CLEAN_CPU = 8.2
NOISY_CHAINS = [
    [("noise_isotropic", 0.02)],
    [("noise_isotropic", 0.05)],
    [("noise_isotropic", 0.1)],
    [("refine", 0.15), ("noise_off_plane", 0.02)],
]


@pytest.mark.benchmark
def test_noisy_curved_corpus_costs_a_bounded_multiple_of_clean():
    entries = [e for e in curved_entries() + chamfer_fillet_entries() if e["seed"] < 2]
    clean = [tessellate(generate(e["family"], e["seed"]).solid, 0.01, 0.2) for e in entries]
    noisy = [np.asarray(degrade.chain(m, c, 0).tris) for m in clean for c in NOISY_CHAINS]

    def cpu(meshes):
        start = time.process_time()
        for m in meshes:
            unmesh.convert(m)
        return time.process_time() - start

    ratio = cpu(noisy) / cpu([np.asarray(m.tris) for m in clean])
    assert ratio <= NOISY_TO_CLEAN_CPU, ratio
