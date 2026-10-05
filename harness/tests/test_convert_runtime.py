import time

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
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
