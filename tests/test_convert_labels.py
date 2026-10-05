import math

import numpy as np
import pytest

import unmesh


def washer(ro=10.0, ri=4.0, h=5.0, n=36):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)

    def ring(r, z):
        return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], axis=1)

    o0, o1, i0, i1 = ring(ro, 0.0), ring(ro, h), ring(ri, 0.0), ring(ri, h)
    tris, labels = [], []
    for k in range(n):
        j = (k + 1) % n
        tris += [[o0[k], o0[j], o1[j]], [o0[k], o1[j], o1[k]]]
        tris += [[i0[k], i1[j], i0[j]], [i0[k], i1[k], i1[j]]]
        tris += [[i1[k], o1[k], o1[j]], [i1[k], o1[j], i1[j]]]
        tris += [[i0[k], o0[j], o0[k]], [i0[k], i0[j], o0[j]]]
        labels += [0, 0, 1, 1, 2, 2, 3, 3]
    return np.array(tris), np.array(labels)


def test_labels_give_cylinders_with_orientation():
    tris, labels = washer()
    ir, report = unmesh.convert_from_labels(tris, labels)
    assert report.region_counts == {"cylinder": 2, "plane": 2}
    cyl = {r.surface.orientation: r.surface for r in ir.regions if r.surface.type == "cylinder"}
    assert cyl["same"].radius == pytest.approx(10.0, abs=1e-9)
    assert cyl["reversed"].radius == pytest.approx(4.0, abs=1e-9)
    assert abs(cyl["same"].axis[2]) == pytest.approx(1.0)
    sagitta = 10.0 * (1 - math.cos(math.pi / 36))
    assert report.max_deviation >= sagitta
    assert max(r.residual.max for r in ir.regions if r.residual) >= sagitta


def _tilt():
    a, b = 0.37, 0.91
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return np.array([[cb, -sb * ca, sb * sa], [sb, cb * ca, -cb * sa], [0.0, sa, ca]])


def sector_prism(r, segments, step, noise, seed=0, h=5.0):
    angles = -math.pi / 2 + step * np.arange(segments + 1)
    profile = np.vstack([[0.0, 0.0], np.stack([r * np.cos(angles), r * np.sin(angles)], 1)])
    rng = np.random.default_rng(seed)
    rot, off = _tilt(), np.array([12.5, -7.25, 3.0])

    def at(q, z):
        return rot @ np.array([q[0], q[1], z]) + off + rng.uniform(-noise, noise, 3)

    pts = [(at(q, 0.0), at(q, h)) for q in profile]
    m = len(pts)
    tris, labels = [], []
    for i in range(m):
        j = (i + 1) % m
        label = 1 if i == 0 else 2 if j == 0 else 0
        tris += [[pts[i][0], pts[j][0], pts[j][1]], [pts[i][0], pts[j][1], pts[i][1]]]
        labels += [label, label]
    for i in range(1, m - 1):
        tris += [[pts[0][1], pts[i][1], pts[i + 1][1]], [pts[0][0], pts[i + 1][0], pts[i][0]]]
        labels += [4, 3]
    return np.array(tris), np.array(labels)


@pytest.mark.parametrize("segments,n", [(3, 36), (2, 24), (3, 12)])
def test_tessellation_law_recovers_short_noisy_arc(segments, n):
    tris, labels = sector_prism(10.0, segments, 2 * math.pi / n, 1e-5)
    ir, _ = unmesh.convert_from_labels(tris, labels, unmesh.ConvertOptions(linear_tolerance=1e-4))
    (cyl,) = [r.surface for r in ir.regions if r.surface.type == "cylinder"]
    assert cyl.radius == pytest.approx(10.0, abs=3e-5)


def test_twisted_cone_never_under_reports():
    from unmesh_harness.judge import judge, under_reports

    n, r0, r1, h, twist = 48, 10.0, 2.0, 4.0, math.radians(60)
    t = 2 * math.pi * np.arange(n) / n
    bottom = np.stack([r0 * np.cos(t), r0 * np.sin(t), np.zeros(n)], 1)
    top = np.stack([r1 * np.cos(t + twist), r1 * np.sin(t + twist), np.full(n, h)], 1)
    tris, labels = [], []
    for k in range(n):
        j = (k + 1) % n
        tris += [[bottom[k], bottom[j], top[j]], [bottom[k], top[j], top[k]]]
        tris += [[[0.0, 0.0, 0.0], bottom[j], bottom[k]], [[0.0, 0.0, h], top[k], top[j]]]
        labels += [0, 0, 1, 2]
    tris = np.array(tris)
    ir, report = unmesh.convert_from_labels(tris, np.array(labels))
    assert report.region_counts["cone"] == 1
    result = judge(ir, tris, report=report, samples_per_mm2=20.0)
    assert result.input.max > 0.28
    assert report.max_deviation >= result.input.max
    assert not under_reports(report.max_deviation, result.input)


def test_indexed_input_matches_soup():
    tris, labels = washer(n=12)
    verts, faces = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    a = unmesh.convert_from_labels(tris, labels)
    b = unmesh.convert_from_labels((verts, faces.reshape(-1, 3)), labels)
    assert a.ir.dumps() == b.ir.dumps()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: np.where(np.arange(len(x)) == 0, -1, x),
        lambda x: np.where(np.arange(len(x)) == 0, 2**32 - 1, x),
        lambda x: np.where(x == 3, 9, x),
        lambda x: x[:-1],
        lambda x: x.astype(np.float64),
        lambda x: x.reshape(-1, 2),
    ],
    ids=["negative", "reserved", "sparse", "short", "float", "shape"],
)
def test_bad_labels_raise(mutate):
    tris, labels = washer(n=8)
    with pytest.raises(ValueError):
        unmesh.convert_from_labels(tris, mutate(labels))


def test_writer_falls_back_on_curved_regions(tmp_path):
    pytest.importorskip("OCP")
    from unmesh import step

    tris, labels = washer(n=24)
    ir, _ = unmesh.convert_from_labels(tris, labels)
    report = step.write(ir, tmp_path / "w.step", mesh=tris)
    assert report.valid
    assert report.fallback == "faceted"
    assert "cylinder" in report.fallback_reason
    bare = step.write(ir, tmp_path / "bare.step")
    assert not bare.valid
    assert not (tmp_path / "bare.step").exists()
