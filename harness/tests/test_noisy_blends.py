import numpy as np
import pytest

import unmesh
from unmesh_harness import degrade
from unmesh_harness.groundtruth import generate
from unmesh_harness.judge import judge
from unmesh_harness.labels import tessellate

LIN, ANG = 0.01, 0.2
NOISE = [("noise_normal", 0.02)]
FAMILIES = ("corner_fillet", "bore_chamfer")
CASES = [(f, p, s) for f in FAMILIES for p in range(3) for s in range(3)]
FAST = [(f, 0, 0) for f in FAMILIES]


def noisy(family, part, seed):
    mesh = tessellate(generate(family, part).solid, LIN, ANG)
    return degrade.chain(mesh, NOISE, seed)


def check_regions(family, part, seed):
    mesh = noisy(family, part, seed)
    tris = np.asarray(mesh.tris)
    ir, report = unmesh.convert(tris)
    truth = len(set(np.asarray(mesh.face_id).tolist()))
    assert len(ir.regions) <= 1.5 * truth, (len(ir.regions), truth, report.region_counts)
    measured = judge(ir, tris, None, report).input.max
    assert measured <= report.max_deviation, (measured, report.max_deviation)
    return ir, tris


@pytest.mark.parametrize("family,part,seed", FAST)
def test_noisy_blends_keep_their_faces(family, part, seed):
    check_regions(family, part, seed)


@pytest.mark.slow
@pytest.mark.parametrize("family,part,seed", [c for c in CASES if c not in FAST])
def test_noisy_blends_keep_their_faces_slow(family, part, seed):
    check_regions(family, part, seed)


@pytest.mark.slow
@pytest.mark.parametrize("part,seed", [(p, s) for p in range(3) for s in range(3)])
def test_noisy_bore_chamfer_writes_analytic(tmp_path, part, seed):
    pytest.importorskip("OCP")
    import unmesh.step

    ir, tris = check_regions("bore_chamfer", part, seed)
    written = unmesh.step.write(ir, tmp_path / "part.step", mesh=tris)
    assert written.valid and written.fallback is None, written.fallback_reason


@pytest.mark.slow
def test_noisy_through_bore_seed_11_writes_analytic(tmp_path):
    pytest.importorskip("OCP")
    import unmesh.step

    ir, tris = check_regions("through_bore", 0, 11)
    written = unmesh.step.write(ir, tmp_path / "part.step", mesh=tris)
    assert written.valid and written.fallback is None, written.fallback_reason
