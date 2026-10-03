import time

import numpy as np
import pytest

import unmesh
from unmesh import ConvertOptions
from unmesh.ir import Ir

pytest.importorskip("OCP")
pytest.importorskip("build123d")

from unmesh_harness.corpus import load_manifest, select  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402
from unmesh_harness.labels import tessellate  # noqa: E402


@pytest.fixture(scope="module")
def parts():
    return [
        tessellate(generate(e["family"], e["seed"]).solid, 0.01, 0.2)
        for e in select(load_manifest(), "smoke")
        if e["strata"].get("category", "planar") == "planar"
    ]


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
        if all(f.surface == "plane" for f in part.faces):
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
    assert len(ir.regions) == len(part.faces)
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


def noise_vectors(rng, n, amplitude):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0, amplitude, size=(n, 1))


def test_subdivided_planar_mesh_with_noise():
    soup = grid_box(20)
    verts, faces, _, _ = unmesh.weld(soup, 1e-6)
    noisy = verts + noise_vectors(np.random.default_rng(5), len(verts), 0.01)
    ir, report = unmesh.convert((noisy, faces))
    assert len(ir.regions) == 6
    assert len(ir.vertices) == 8
    assert report.max_deviation < 0.03


def test_clean_plate_with_bore_gets_micron_auto_tolerance():
    from build123d import Box, Cylinder, Pos

    plate = Pos(0, 0, 2.0) * Box(20.0, 20.0, 4.0) - Pos(0, 0, 5.0) * Cylinder(3.0, 10.0)
    mesh = tessellate(plate, 0.01, 0.2)
    ir, _ = unmesh.convert(mesh.tris)
    assert ir.tolerances.linear <= 2e-3


def scaled_mesh(part, factor):
    import copy

    out = copy.deepcopy(part)
    out.tris = part.tris * factor
    for face in out.faces:
        assert face.surface == "plane"
        face.params["origin"] = (np.array(face.params["origin"]) * factor).tolist()
    for adj in out.adjacency:
        adj.points = (np.array(adj.points) * factor).tolist()
    return out


def test_scaled_planar_parts_reach_unscaled_f1(parts):
    from unmesh_harness.metrics.recovery import score_recovery

    for part in parts:
        base = score_recovery(part, part.face_id, unmesh.convert(part.tris)[0])["f1"]
        assert base == 1.0
        for factor in (1000.0, 0.001):
            scaled = scaled_mesh(part, factor)
            ir, _ = unmesh.convert(scaled.tris)
            assert score_recovery(scaled, scaled.face_id, ir)["f1"] == base


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
        print(f"\nlargest smoke part: {len(largest.tris)} tris, {seconds * 1000:.1f}ms")
    assert seconds < 1


def test_curved_smoke_parts_cover_every_nondegenerate_triangle():
    for entry in select(load_manifest(), "smoke"):
        if entry["strata"].get("category", "planar") == "planar":
            continue
        part = tessellate(generate(entry["family"], entry["seed"]).solid, 0.05, 0.5)
        ir, _ = unmesh.convert(part.tris)
        ir.validate()
        tris = part.tris
        area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        covered = {t for r in ir.regions for t in r.triangles}
        assert {int(i) for i in np.flatnonzero(area > 0)} <= covered, entry["id"]
