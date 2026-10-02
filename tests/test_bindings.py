import time

import numpy as np
import pytest

import unmesh


def grid_soup(n):
    x = np.arange(n + 1, dtype=np.float64)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    z = np.sin(gx * 0.1) * np.cos(gy * 0.1)
    p = np.stack([gx, gy, z], axis=-1)
    a, b = p[:-1, :-1], p[1:, :-1]
    c, d = p[1:, 1:], p[:-1, 1:]
    t1 = np.stack([a, b, c], axis=2)
    t2 = np.stack([a, c, d], axis=2)
    return np.concatenate([t1.reshape(-1, 3, 3), t2.reshape(-1, 3, 3)])


def test_stl_round_trip(tmp_path):
    tris = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1], [2, 0, 1], [0, 2, 1]]], dtype=np.float64
    )
    path = tmp_path / "a.stl"
    unmesh.write_stl(path, tris)
    out = unmesh.read_stl(str(path))
    assert out.dtype == np.float64
    assert out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out, tris)


def test_write_accepts_float32_and_lists(tmp_path):
    path = tmp_path / "b.stl"
    unmesh.write_stl(path, np.zeros((1, 3, 3), dtype=np.float32))
    assert unmesh.read_stl(path).shape == (1, 3, 3)
    unmesh.write_stl(path, [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    assert unmesh.read_stl(path).shape == (1, 3, 3)


def test_write_empty_round_trip(tmp_path):
    path = tmp_path / "e.stl"
    unmesh.write_stl(path, np.zeros((0, 3, 3)))
    assert unmesh.read_stl(path).shape == (0, 3, 3)


def test_write_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        unmesh.write_stl(tmp_path / "c.stl", np.zeros((4, 3)))


def test_read_truncated_is_valueerror(tmp_path):
    path = tmp_path / "t.stl"
    unmesh.write_stl(path, np.zeros((2, 3, 3)))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="truncated"):
        unmesh.read_stl(path)


def test_read_ascii(tmp_path):
    path = tmp_path / "ascii.stl"
    path.write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid t\n"
    )
    assert unmesh.read_stl(path).shape == (1, 3, 3)


def test_weld_cracked_mesh():
    tris = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1 + 1e-9, 1, 0], [0, 1, 0]],
        ],
        dtype=np.float64,
    )
    verts, faces, _, report = unmesh.weld(tris, 0.0)
    assert len(verts) == 5
    verts, faces, _, report = unmesh.weld(tris, 1e-6)
    assert verts.shape == (4, 3) and verts.dtype == np.float64
    assert faces.shape == (2, 3) and faces.dtype == np.uint32
    assert report == {"input_corners": 6, "unique_vertices": 4, "degenerate_dropped": 0}


def test_weld_drops_degenerate():
    tris = np.array([[[0, 0, 0], [1e-9, 0, 0], [0, 1, 0]]])
    verts, faces, _, report = unmesh.weld(tris, 1e-6)
    assert faces.shape == (0, 3)
    assert report["degenerate_dropped"] == 1


def test_weld_returns_source_triangle_map():
    tris = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1e-9, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        ],
        dtype=np.float64,
    )
    verts, faces, source, report = unmesh.weld(tris, 1e-6)
    assert faces.shape == (2, 3)
    assert source.dtype == np.uint32
    assert source.tolist() == [0, 2]
    assert report["degenerate_dropped"] == 1


def test_weld_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        unmesh.weld(np.zeros((1, 3, 3)), -1.0)


def test_weld_rejects_non_finite_and_handles_huge():
    for bad in (np.inf, -np.inf, np.nan):
        tris = np.zeros((1, 3, 3))
        tris[0, 0, 0] = bad
        with pytest.raises(ValueError, match="non-finite"):
            unmesh.weld(tris, 1e-6)
    huge = np.array([[[1e20, 0, 0], [-1e20, 0, 0], [0, 1e20, 0]]])
    verts, faces, _, report = unmesh.weld(huge, 1e-6)
    assert report["unique_vertices"] == 3


def test_write_rejects_non_finite(tmp_path):
    tris = np.zeros((1, 3, 3))
    tris[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        unmesh.write_stl(tmp_path / "n.stl", tris)


def test_read_rejects_non_finite(tmp_path):
    path = tmp_path / "inf.stl"
    unmesh.write_stl(path, np.zeros((1, 3, 3)))
    data = bytearray(path.read_bytes())
    data[96:100] = np.float32(np.inf).tobytes()
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="non-finite"):
        unmesh.read_stl(path)


def test_read_missing_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        unmesh.read_stl(tmp_path / "nope.stl")


def million_soup_weld(tmp_path):
    tris = grid_soup(708)
    assert len(tris) >= 1_000_000
    path = tmp_path / "big.stl"
    unmesh.write_stl(path, tris)
    soup = unmesh.read_stl(path)
    start = time.perf_counter()
    verts, faces, _, report = unmesh.weld(soup, 1e-6)
    elapsed = time.perf_counter() - start
    assert report["unique_vertices"] == 709 * 709
    assert len(faces) == len(tris)
    return elapsed


def test_weld_million_triangles_sanity(tmp_path):
    assert million_soup_weld(tmp_path) < 30.0


@pytest.mark.benchmark
def test_weld_million_triangles_benchmark(tmp_path):
    elapsed = million_soup_weld(tmp_path)
    print(f"weld 1M triangles: {elapsed:.3f}s")
    assert elapsed < 1.0
