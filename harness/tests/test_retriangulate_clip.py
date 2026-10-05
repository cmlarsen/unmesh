import math

import numpy as np
import pytest

from unmesh_harness.degrade.retriangulate import _area2, _bridge, _clip

SCHEMES = ("fan", "strip", "delaunay", "canonical")


def _ring(center, radius, count, rng, phase=None):
    phase = rng.uniform(0, 2 * math.pi) if phase is None else phase
    angles = np.sort(rng.uniform(0, 2 * math.pi, count)) if count > 4 else None
    if angles is None or np.min(np.diff(np.r_[angles, angles[0] + 2 * math.pi])) < 0.05:
        angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
    return [
        (
            float(center[0] + radius * math.cos(a + phase)),
            float(center[1] + radius * math.sin(a + phase)),
        )
        for a in angles
    ]


def _rect(center, w, h, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [
        (float(center[0] + x * c - y * s), float(center[1] + x * s + y * c))
        for x, y in ((-w, -h), (w, -h), (w, h), (-w, h))
    ]


def _star_outer(rng):
    count = int(rng.integers(5, 40))
    angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
    radii = rng.uniform(6.0, 10.0, count)
    return [
        (float(r * math.cos(a)), float(r * math.sin(a))) for r, a in zip(radii, angles, strict=True)
    ]


def _random_holes(rng):
    holes, discs = [], []
    for _ in range(int(rng.integers(0, 6))):
        radius = float(rng.uniform(0.3, 1.2))
        for _ in range(50):
            center = rng.uniform(-3.8, 3.8, 2)
            if all(np.hypot(*(center - c)) > radius + r + 0.2 for c, r in discs):
                discs.append((center, radius))
                holes.append(_ring(center, radius, int(rng.integers(3, 16)), rng))
                break
    return holes


def _slot_case(rng):
    outer = _rect((0.0, 0.0), 40.0, 48.0, 0.0)
    angle = float(rng.uniform(-0.4, 0.4))
    rows = int(rng.integers(2, 6))
    holes = [_rect((0.0, -36.0 + 72.0 * k / (rows - 1)), 20.0, 3.0, angle) for k in range(rows)]
    return outer, holes


def _case(seed):
    rng = np.random.default_rng(seed)
    if seed % 3 == 0:
        return _slot_case(rng)
    return _star_outer(rng), _random_holes(rng)


def _covering(tris, pts):
    count = np.zeros(len(pts), dtype=int)
    for a, b, c in tris:
        a, b, c = np.array(a), np.array(b), np.array(c)
        den = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        u = (
            (b[0] - pts[:, 0]) * (c[1] - pts[:, 1]) - (b[1] - pts[:, 1]) * (c[0] - pts[:, 0])
        ) / den
        v = (
            (c[0] - pts[:, 0]) * (a[1] - pts[:, 1]) - (c[1] - pts[:, 1]) * (a[0] - pts[:, 0])
        ) / den
        count += (u > 1e-9) & (v > 1e-9) & (1.0 - u - v > 1e-9)
    return count


def _inside(pt, poly) -> bool:
    c = False
    for a, b in zip(poly, poly[1:] + poly[:1], strict=True):
        if (a[1] > pt[1]) != (b[1] > pt[1]) and pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            c = not c
    return c


@pytest.mark.parametrize("seed", range(60))
def test_bridge_and_clip_triangulate_random_polygons_with_holes(seed):
    outer, holes = _case(seed)
    holes = [h if _area2(h) < 0 else h[::-1] for h in holes]
    merged = _bridge(outer, holes) if holes else list(outer)
    assert merged is not None
    area = _area2(outer) + sum(_area2(h) for h in holes)
    vertices = set(outer) | {p for h in holes for p in h}
    rng = np.random.default_rng(seed)
    probe = rng.uniform(-11.0, 11.0, (1500, 2))
    if seed % 3 == 0:
        probe *= np.array([4.0, 4.5])
    inside = np.array(
        [_inside(tuple(p), outer) and not any(_inside(tuple(p), h) for h in holes) for p in probe]
    )
    for scheme in SCHEMES:
        tris = _clip(merged, scheme, np.random.default_rng(seed))
        assert tris is not None, scheme
        assert len(tris) == len(merged) - 2, scheme
        areas = [_area2(list(t)) for t in tris]
        assert min(areas) > 0.0, scheme
        assert sum(areas) == pytest.approx(area, rel=1e-9), scheme
        assert {p for t in tris for p in t} == vertices, scheme
        cover = _covering(tris, probe)
        assert cover.max() <= 1, scheme
        assert np.all(cover[~inside] == 0), scheme
        assert np.all(cover[inside] == 1), scheme
