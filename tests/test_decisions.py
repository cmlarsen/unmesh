import math

import numpy as np
import pytest

import unmesh


def prism(points, h):
    n = len(points)
    bottom = [(x, y, 0.0) for x, y in points]
    top = [(x, y, h) for x, y in points]
    cb = (*np.mean(points, axis=0), 0.0)
    ct = (*np.mean(points, axis=0), h)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(bottom[i], bottom[j], top[j]), (bottom[i], top[j], top[i])]
        tris += [(cb, bottom[j], bottom[i]), (ct, top[i], top[j])]
    return np.array(tris, dtype=float)


def ngon(n, r):
    return [
        (r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)
    ]


def rounded_corner(segments, r=3.0, size=20.0):
    pts = [(0.0, 0.0), (size, 0.0)]
    cx, cy = size - r, size - r
    for k in range(segments + 1):
        a = math.pi / 2 * k / segments
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((0.0, size))
    return pts


def test_regular_prism_records_prism_or_cylinder():
    ir, report = unmesh.convert(prism(ngon(12, 10.0), 5.0))
    assert report.region_counts == {"plane": 14}
    [d] = report.decisions
    assert d.kind == "prism_or_cylinder"
    assert d.chosen == "planes" and d.alternative == "cylinder"
    assert d.reason == "crease_at_least_15deg"
    assert len(d.regions) == 12 and d.segments == 12 and d.supports == ()
    assert d.radius == pytest.approx(10.0, rel=1e-9)
    assert abs(d.axis[2]) == pytest.approx(1.0)
    assert d.alternative_deviation == pytest.approx(10.0 * (1 - math.cos(math.pi / 12)))
    assert d.min_crease_deg == pytest.approx(30.0)


def test_irregular_prism_records_nothing():
    pts = ngon(8, 10.0)
    pts[0] = (12.0, 0.0)
    _, report = unmesh.convert(prism(pts, 5.0))
    assert report.decisions == []


def test_box_records_nothing():
    _, report = unmesh.convert(prism([(0, 0), (10, 0), (10, 8), (0, 8)], 5.0))
    assert report.decisions == []


@pytest.mark.parametrize(
    ("segments", "kind", "reason"),
    [
        (1, "chamfer_or_fillet", "one_facet"),
        (2, "two_planes_or_fillet", "two_facets"),
        (3, "prism_or_cylinder", "crease_at_least_15deg"),
    ],
)
def test_coarse_fillet_records_its_reading(segments, kind, reason):
    _, report = unmesh.convert(prism(rounded_corner(segments), 5.0))
    assert report.region_counts == {"plane": 6 + segments}
    [d] = report.decisions
    assert (d.kind, d.reason, d.segments) == (kind, reason, segments)
    assert len(d.supports) == 2
    assert d.radius == pytest.approx(3.0, rel=1e-9)
    assert d.axis_point[0] == pytest.approx(17.0) and d.axis_point[1] == pytest.approx(17.0)


def test_asymmetric_chamfer_is_not_a_fillet():
    pts = [(0.0, 0.0), (20.0, 0.0), (20.0, 16.0), (17.0, 20.0), (0.0, 20.0)]
    _, report = unmesh.convert(prism(pts, 5.0))
    assert report.decisions == []


def test_labels_record_no_decisions():
    tris = prism(ngon(12, 10.0), 5.0)
    labels = np.zeros(len(tris), dtype=np.int64)
    labels[2::4] = 1
    labels[3::4] = 2
    _, report = unmesh.convert_from_labels(tris, labels)
    assert report.decisions == []
