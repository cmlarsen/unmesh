import copy
import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
import unmesh.step as step  # noqa: E402
from unmesh.ir import Cone, Cylinder, Plane, Residual, Sphere, Torus  # noqa: E402

CASES = [
    ("sphere", 10_000),
    ("sphere", 100_000),
    ("sphere", 250_000),
    ("box", 10_000),
    ("box", 100_000),
    ("box", 250_000),
]


def uv_sphere(n_u, n_v, radius=10.0):
    assert n_v >= 3
    us = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
    vs = np.linspace(0.0, np.pi, n_v + 1)
    su, cu = np.sin(us), np.cos(us)
    sv, cv = np.sin(vs), np.cos(vs)
    north = np.array([0.0, 0.0, radius])
    south = np.array([0.0, 0.0, -radius])
    rings = np.array(
        [
            [radius * sv[j] * cu[i], radius * sv[j] * su[i], radius * cv[j]]
            for j in range(1, n_v)
            for i in range(n_u)
        ]
    )
    tris = np.empty((2 * n_u * (n_v - 1), 3, 3))
    k = 0
    for i in range(n_u):
        tris[k] = [north, rings[i], rings[(i + 1) % n_u]]
        k += 1
    for j in range(n_v - 2):
        for i in range(n_u):
            a = j * n_u + i
            b = j * n_u + (i + 1) % n_u
            c = (j + 1) * n_u + i
            d = (j + 1) * n_u + (i + 1) % n_u
            tris[k] = [rings[a], rings[c], rings[b]]
            tris[k + 1] = [rings[b], rings[c], rings[d]]
            k += 2
    base = (n_v - 2) * n_u
    for i in range(n_u):
        tris[k] = [rings[base + i], south, rings[base + (i + 1) % n_u]]
        k += 1
    assert k == len(tris)
    return tris


def assert_closed_manifold(tris):
    key = np.round(tris.reshape(-1, 3), 9)
    ids = np.unique(key, axis=0, return_inverse=True)[1].reshape(len(tris), 3)
    seen = set()
    for a, b, c in ids:
        for u, v in ((a, b), (b, c), (c, a)):
            assert (u, v) not in seen
            seen.add((u, v))
    for u, v in seen:
        assert (v, u) in seen


SPHERE_GRIDS = [(100, 51), (250, 201), (500, 251)]


@pytest.mark.parametrize("n_u,n_v", SPHERE_GRIDS)
def test_uv_sphere_is_closed_manifold(n_u, n_v):
    tris = uv_sphere(n_u, n_v)
    assert len(tris) == 2 * n_u * (n_v - 1)
    assert_closed_manifold(tris)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    areas = np.linalg.norm(np.cross(e1, e2), axis=1) / 2.0
    assert np.all(areas >= 1e-12 * areas.mean())


def subdivided_box(size=10.0, n=29):
    faces = [
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, size), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, size, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((size, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    tris = np.empty((12 * n * n, 3, 3))
    k = 0
    for origin, u, v in faces:
        o, u, v = np.array(origin), np.array(u), np.array(v)
        for j in range(n):
            for i in range(n):
                p00 = o + (i / n * size) * u + (j / n * size) * v
                p10 = o + ((i + 1) / n * size) * u + (j / n * size) * v
                p01 = o + (i / n * size) * u + ((j + 1) / n * size) * v
                p11 = o + ((i + 1) / n * size) * u + ((j + 1) / n * size) * v
                tris[k] = [p00, p10, p01]
                tris[k + 1] = [p10, p11, p01]
                k += 2
    return tris


def make_mesh(shape, target):
    if shape == "sphere":
        for n_u, n_v in SPHERE_GRIDS:
            if 2 * n_u * (n_v - 1) == target:
                return uv_sphere(n_u, n_v)
        raise AssertionError(f"no exact uv grid for {target}")
    for n in range(1, 500):
        if 12 * n * n >= target:
            return subdivided_box(n=n)
    raise AssertionError(f"no box grid for {target}")


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def break_analytic(ir):
    bad = copy.deepcopy(ir)
    for r in bad.regions:
        s = r.surface
        if isinstance(s, Plane):
            s.origin = (s.origin[0] + 1.0, s.origin[1] + 1.0, s.origin[2] + 1.0)
            return bad
        if isinstance(s, (Cylinder, Cone)):
            s.origin = (s.origin[0] + 1.0, s.origin[1] + 1.0, s.origin[2] + 1.0)
            return bad
        if isinstance(s, Torus):
            s.center = (s.center[0] + 1.0, s.center[1] + 1.0, s.center[2] + 1.0)
            return bad
        if isinstance(s, Sphere):
            s.radius = s.radius + 1.0
            return bad
    r0 = bad.regions[0]
    r0.surface = Cylinder(
        origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=1.0, orientation="same"
    )
    r0.residual = Residual(rms=0.0, max=0.0)
    return bad


@pytest.mark.benchmark
@pytest.mark.parametrize("shape,target", CASES)
def test_faceted_fallback_timings(shape, target, tmp_path):
    tris = make_mesh(shape, target)
    expected = signed_volume(tris)
    assert expected > 0
    ir, _ = unmesh.convert(tris)
    report = step.write(break_analytic(ir), tmp_path / f"{shape}.step", mesh=tris)
    write_s = report.timings["write_s"]
    readback_s = report.timings["readback_s"]
    print(
        f"[benchmark] shape={shape} triangles={len(tris)}"
        f" write_s={write_s:.3f} readback_s={readback_s:.3f}"
    )
    assert report.fallback == "faceted"
    assert report.valid
    assert report.verified and report.text_check.ok
    assert report.text_check.volume == pytest.approx(expected, rel=1e-9)
    if report.readback is not None:
        assert report.verified_by == "occt+text" and report.readback.ok
        assert report.readback.volume == pytest.approx(expected, rel=1e-9)
    else:
        assert report.verified_by == "text" and report.readback_skipped
    if target <= 100_000:
        assert write_s < 10.0


IMPORTED_VERIFY_S = 30.0
IMPORTED_PEAK_RSS_MB = 2048.0
WRITE_IN_A_FRESH_PROCESS = """
import json, resource, sys
import unmesh, unmesh.step as step
from unmesh.ir import Ir
stl, ir_path, out = sys.argv[1:4]
tris = unmesh.read_stl(stl)
ir = Ir.loads(open(ir_path).read())
r = step.write(ir, out, mesh=tris)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak /= 2**20 if sys.platform == "darwin" else 2**10
print(json.dumps({
    "valid": r.valid, "fallback": r.fallback, "verified_by": r.verified_by,
    "timings": r.timings, "peak_rss_mb": peak, "triangles": len(tris),
    "volume": r.text_check.volume if r.text_check else None,
}))
"""


@pytest.mark.benchmark
def test_large_imported_faceted_write_is_verified_in_bounded_time_and_memory(tmp_path):
    from unmesh_harness.corpus import load_manifest
    from unmesh_harness.imported import cached_step_record, load_imported_shape
    from unmesh_harness.labels import tessellate

    entry = next(e for e in load_manifest()["entries"] if e["id"] == "imported-0201")
    try:
        cached_step_record(entry["source"])
    except (FileNotFoundError, KeyError):
        pytest.skip("imported-0201 is not in the dataset cache (scripts/fetch-datasets)")
    mesh = tessellate(load_imported_shape(entry), 0.01, 0.2)
    stl = tmp_path / "part.stl"
    mesh.write_stl(stl)
    tris = unmesh.read_stl(stl)
    ir, _ = unmesh.convert(tris)
    (tmp_path / "part.json").write_text(ir.dumps())
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            WRITE_IN_A_FRESH_PROCESS,
            str(stl),
            str(tmp_path / "part.json"),
            str(tmp_path / "part.step"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(done.stdout.strip().splitlines()[-1])
    print(f"[benchmark] imported-0201 {result}")
    assert result["triangles"] > 390_000
    assert result["fallback"] == "faceted" and result["valid"]
    assert result["verified_by"] == "text"
    assert result["volume"] == pytest.approx(signed_volume(tris), rel=1e-6)
    assert result["timings"]["readback_s"] < IMPORTED_VERIFY_S
    assert result["peak_rss_mb"] < IMPORTED_PEAK_RSS_MB
