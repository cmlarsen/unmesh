import time

import numpy as np
import pytest

import unmesh
from unmesh import ConvertOptions
from unmesh.ir import Ir

pytest.importorskip("OCP")
pytest.importorskip("build123d")

import convert_eval as ce  # noqa: E402


@pytest.fixture(scope="module")
def parts():
    return ce.smoke_parts()


@pytest.fixture(scope="module")
def results(parts):
    return {case.name: ce.run_case(parts, case) for case in ce.CASES}


def case_names(*names):
    return pytest.mark.parametrize("name", names)


@case_names("clean", "float32", "rotated")
def test_exact_inputs_are_recovered_within_a_micron(results, name):
    ms = results[name]
    assert ce.micro_f1(ms) >= 0.99
    assert all(m.regions == m.faces for m in ms)
    assert max(m.dev_input for m in ms) <= 1e-3
    assert not any(m.fallback for m in ms)


@pytest.mark.parametrize("case", [c for c in ce.CASES if c.noise], ids=lambda c: c.name)
def test_noisy_inputs_keep_their_faces_and_stay_near_the_truth(results, case):
    ms = results[case.name]
    amplitude = case.noise
    assert ce.micro_f1(ms) >= 0.98
    assert max(m.dev_truth for m in ms) <= 2 * amplitude
    assert not any(m.fallback for m in ms)


def test_calibration_never_under_reports(results):
    for name, ms in results.items():
        for m in ms:
            assert m.reported >= m.dev_input, name


def test_report_describes_the_conversion(parts):
    part = parts[0]
    ir, report = unmesh.convert(part.tris)
    assert report.region_counts == {"plane": len(ir.regions)}
    assert report.analytic_area_fraction == pytest.approx(1.0)
    assert report.warnings == []
    assert 0.0 <= report.rms_deviation <= report.max_deviation


def test_every_smoke_part_validates_and_round_trips(parts):
    for part in parts:
        ir, _ = unmesh.convert(part.tris)
        ir.validate()
        assert Ir.loads(ir.dumps()).dumps() == ir.dumps()
        assert sorted(t for r in ir.regions for t in r.triangles) == list(range(len(part.tris)))


def test_conversion_is_deterministic(parts):
    part = parts[3]
    a, _ = unmesh.convert(part.tris)
    b, _ = unmesh.convert(part.tris)
    assert a.dumps() == b.dumps()


def test_indexed_and_soup_inputs_agree(parts):
    part = parts[1]
    verts, faces, _, _ = unmesh.weld(part.tris, 1e-6)
    a, _ = unmesh.convert(part.tris)
    b, _ = unmesh.convert((verts, faces))
    assert a.dumps() == b.dumps()


def test_stl_path_input(parts, tmp_path):
    part = parts[2]
    path = tmp_path / "part.stl"
    unmesh.write_stl(path, part.tris)
    ir, report = unmesh.convert(path)
    assert len(ir.regions) == int(part.labels.max()) + 1
    assert report.max_deviation < 1e-3


def test_explicit_tolerance_is_recorded(parts):
    ir, _ = unmesh.convert(parts[0].tris, ConvertOptions(linear_tolerance=0.02))
    assert ir.tolerances.linear == 0.02
    with pytest.raises(ValueError):
        unmesh.convert(parts[0].tris, ConvertOptions(linear_tolerance=-1.0))


def box(lo, hi, inward=False):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
        + [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=float,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4]]
        + [[1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    if inward:
        f = f[:, ::-1]
    return v, f


def soup_of(*boxes):
    return np.concatenate([v[f] for v, f in boxes])


def test_inverted_box_is_flipped_and_reported():
    ir, report = unmesh.convert(soup_of(box((0, 0, 0), (4, 5, 6), inward=True)))
    assert [w.code for w in report.warnings] == ["flipped_winding"]
    assert len(ir.regions) == 6
    centre = np.array([2.0, 2.5, 3.0])
    for r in ir.regions:
        assert (np.array(r.surface.origin) - centre) @ np.array(r.surface.normal) > 0


def test_cavity_and_second_body():
    ir, _ = unmesh.convert(
        soup_of(
            box((0, 0, 0), (30, 30, 30)),
            box((10, 10, 10), (20, 20, 20), inward=True),
            box((100, 0, 0), (110, 10, 10)),
        )
    )
    assert [(s.role, s.parent) for s in ir.shells] == [
        ("outer", None),
        ("cavity", 0),
        ("outer", None),
    ]
    assert len(ir.regions) == 18


def test_open_mesh_becomes_one_facets_region():
    v, f = box((0, 0, 0), (1, 1, 1))
    ir, report = unmesh.convert((v, f[:-1]))
    assert [r.surface.type for r in ir.regions] == ["facets"]
    assert not ir.shells[0].closed
    assert report.warnings[0].code == "open_edges"


def test_degenerate_and_duplicate_triangles_are_dropped():
    v, f = box((0, 0, 0), (1, 1, 1))
    soup = np.concatenate([v[f], v[f][:2], np.zeros((1, 3, 3))])
    ir, report = unmesh.convert(soup)
    assert len(ir.regions) == 6
    assert [w.code for w in report.warnings] == ["degenerate_triangles"]
    used = sorted(t for r in ir.regions for t in r.triangles)
    assert used == list(range(12))


def grid_box(div, size=(100.0, 60.0, 40.0)):
    hi = np.array(size)
    g = np.arange(div) / div
    tris = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in range(2):
            a, b = np.meshgrid(g, g, indexing="ij")
            a, b = a.ravel(), b.ravel()

            def p(da, db, a=a, b=b, axis=axis, u=u, v=v, side=side):
                q = np.zeros((len(a), 3))
                q[:, axis] = hi[axis] if side else 0
                q[:, u] = (a + da / div) * hi[u]
                q[:, v] = (b + db / div) * hi[v]
                return q

            p00, p10, p11, p01 = p(0, 0), p(1, 0), p(1, 1), p(0, 1)
            if side:
                tris += [np.stack([p00, p10, p11], 1), np.stack([p00, p11, p01], 1)]
            else:
                tris += [np.stack([p00, p11, p10], 1), np.stack([p00, p01, p11], 1)]
    return np.concatenate(tris)


def test_subdivided_planar_mesh_with_noise():
    soup = grid_box(20)
    verts, faces, _, _ = unmesh.weld(soup, 1e-6)
    noisy = verts + ce.noise_vectors(np.random.default_rng(5), len(verts), 0.01)
    ir, report = unmesh.convert((noisy, faces))
    assert len(ir.regions) == 6
    assert len(ir.vertices) == 8
    assert report.max_deviation < 0.03


@pytest.mark.benchmark
def test_million_triangle_planar_mesh_converts_fast(capsys):
    soup = grid_box(289)
    assert len(soup) > 1_000_000
    t0 = time.perf_counter()
    ir, report = unmesh.convert(soup)
    seconds = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n1M-triangle planar mesh: {len(soup)} triangles in {seconds:.2f}s")
    assert len(ir.regions) == 6
    assert report.max_deviation < 1e-6
    assert seconds < 10


@pytest.mark.benchmark
def test_largest_smoke_part_convert_time(parts, capsys):
    largest = max(parts, key=lambda p: len(p.tris))
    t0 = time.perf_counter()
    unmesh.convert(largest.tris)
    seconds = time.perf_counter() - t0
    with capsys.disabled():
        print(
            f"\nlargest smoke part {largest.name}: {len(largest.tris)} tris, {seconds * 1000:.1f}ms"
        )
    assert seconds < 1
