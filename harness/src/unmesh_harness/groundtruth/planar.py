from __future__ import annotations

import math

import numpy as np
from build123d import Box, Polygon, Pos, Shape, extrude

from .core import rect_extent, register

Cell = tuple[float, float, float, float]


def _r(x: float) -> float:
    return round(float(x), 3)


def _rect_points(cx: float, cy: float, length: float, width: float, angle_deg: float):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        px, py = dx * length / 2, dy * width / 2
        pts.append((cx + px * c - py * s, cy + px * s + py * c))
    return pts


def _prism(points, height: float, z0: float) -> Shape:
    sketch = Polygon(*points, align=None)
    return Pos(0, 0, z0) * extrude(sketch, amount=height)


def _plate(length: float, width: float, thickness: float) -> Shape:
    return Pos(0, 0, thickness / 2) * Box(length, width, thickness)


def _cells(rng, length: float, width: float, max_cols: int = 3, max_rows: int = 2) -> list[Cell]:
    margin = float(rng.uniform(2.0, 4.0))
    gap = float(rng.uniform(2.0, 4.0))
    cols = int(rng.integers(1, max_cols + 1))
    rows = int(rng.integers(1, max_rows + 1))
    cw = (length - 2 * margin - (cols - 1) * gap) / cols
    ch = (width - 2 * margin - (rows - 1) * gap) / rows
    cells = []
    for i in range(cols):
        for j in range(rows):
            cx = -length / 2 + margin + cw / 2 + i * (cw + gap)
            cy = -width / 2 + margin + ch / 2 + j * (ch + gap)
            cells.append((cx, cy, cw, ch))
    return cells


def _place_rect(rng, cell: Cell, angle_deg: float, lo: float = 0.45, hi: float = 0.9):
    cx, cy, cw, ch = cell
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    length = float(rng.uniform(lo, hi)) * cw
    width = float(rng.uniform(lo, hi)) * ch
    ex, ey = length * c + width * s, length * s + width * c
    k = min(cw / ex, ch / ey, 1.0)
    length, width = length * k, width * k
    ex, ey = length * c + width * s, length * s + width * c
    ox = float(rng.uniform(-1, 1)) * (cw - ex) / 2
    oy = float(rng.uniform(-1, 1)) * (ch - ey) / 2
    return _r(cx + ox), _r(cy + oy), _r(length), _r(width)


def _plate_dims(rng):
    return (
        _r(rng.uniform(40, 120)),
        _r(rng.uniform(30, 100)),
        _r(rng.uniform(6, 20)),
    )


def _cut_rect(plate: Shape, cx, cy, length, width, angle, z0, z1) -> Shape:
    return plate - _prism(_rect_points(cx, cy, length, width, angle), z1 - z0, z0)


def _clean(shape: Shape) -> Shape:
    return shape.clean()


@register("plate_pockets")
def plate_pockets(rng):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        cx, cy, pl, pw = _place_rect(rng, cell, 0.0)
        depth = _r(rng.uniform(0.25, 0.8) * t)
        solid = _cut_rect(solid, cx, cy, pl, pw, 0.0, t - depth, t + 1)
        features.append(
            {
                "type": "rect_pocket",
                "center": [cx, cy],
                "size": [pl, pw],
                "angle_deg": 0.0,
                "depth": depth,
                "floor_z": _r(t - depth),
                "axis": [0, 0, 1],
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("rotated_pockets")
def rotated_pockets(rng):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        angle = _r(rng.uniform(8, 82))
        cx, cy, pl, pw = _place_rect(rng, cell, angle, 0.5, 0.95)
        depth = _r(rng.uniform(0.25, 0.8) * t)
        solid = _cut_rect(solid, cx, cy, pl, pw, angle, t - depth, t + 1)
        features.append(
            {
                "type": "rect_pocket",
                "center": [cx, cy],
                "size": [pl, pw],
                "angle_deg": angle,
                "depth": depth,
                "floor_z": _r(t - depth),
                "axis": [0, 0, 1],
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("square_slots")
def square_slots(rng):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    n = int(rng.integers(1, 5))
    margin = float(rng.uniform(3.0, 5.0))
    pitch = (width - 2 * margin) / n
    angle = _r(rng.choice([0.0, rng.uniform(5, 30)]))
    features = []
    for i in range(n):
        cy = -width / 2 + margin + pitch * (i + 0.5)
        slot_w = _r(rng.uniform(0.2, 0.55) * pitch)
        slot_l = float(rng.uniform(0.4, 0.9)) * (length - 2 * margin)
        _, ey = rect_extent(slot_l, slot_w, angle)
        if angle and pitch - ey < 1.0:
            slot_l = max(slot_l * 0.5, 3 * slot_w)
        slot_l = _r(slot_l)
        through = bool(rng.random() < 0.5)
        depth = _r(t if through else rng.uniform(0.3, 0.8) * t)
        solid = _cut_rect(
            solid, 0.0, cy, slot_l, slot_w, angle, t - depth - (1 if through else 0), t + 1
        )
        features.append(
            {
                "type": "square_slot",
                "center": [0.0, _r(cy)],
                "size": [slot_l, slot_w],
                "angle_deg": angle,
                "depth": depth,
                "through": through,
                "axis": [0, 0, 1],
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("stepped_block")
def stepped_block(rng):
    length = _r(rng.uniform(40, 100))
    width = _r(rng.uniform(20, 60))
    height = _r(rng.uniform(15, 40))
    solid = Pos(0, 0, height / 2) * Box(length, width, height)
    n = int(rng.integers(1, 4))
    xs = np.sort(rng.uniform(0.15, 0.85, size=n)) * length - length / 2
    xs = np.maximum.accumulate(xs + np.arange(n) * 3.0)
    hs = np.sort(rng.uniform(0.15, 0.85, size=n))[::-1] * height
    features = []
    for x, h in zip(xs, hs, strict=True):
        x, h = _r(x), _r(h)
        if x > length / 2 - 2.0:
            continue
        pts = [(x, -width), (length, -width), (length, width), (x, width)]
        solid = solid - _prism(pts, height - h + 1, h)
        features.append({"type": "step", "x_start": x, "top_z": h, "axis": [1, 0, 0]})
    return _clean(solid), {"block": [length, width, height]}, features


@register("boss_plate")
def boss_plate(rng):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        if rng.random() < 0.8:
            angle = _r(rng.choice([0.0, rng.uniform(8, 82)]))
            cx, cy, bl, bw = _place_rect(rng, cell, angle, 0.4, 0.85)
            h = _r(rng.uniform(2, 10))
            solid = solid + _prism(_rect_points(cx, cy, bl, bw, angle), h, t)
            features.append(
                {
                    "type": "rect_boss",
                    "center": [cx, cy],
                    "size": [bl, bw],
                    "angle_deg": angle,
                    "height": h,
                    "base_z": t,
                    "axis": [0, 0, 1],
                }
            )
    if not features:
        h = _r(rng.uniform(2, 10))
        solid = solid + _prism(_rect_points(0, 0, length / 3, width / 3, 0), h, t)
        features.append(
            {
                "type": "rect_boss",
                "center": [0.0, 0.0],
                "size": [_r(length / 3), _r(width / 3)],
                "angle_deg": 0.0,
                "height": h,
                "base_z": t,
                "axis": [0, 0, 1],
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("through_cuts")
def through_cuts(rng):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        angle = _r(rng.choice([0.0, rng.uniform(8, 82)]))
        cx, cy, cl, cw = _place_rect(rng, cell, angle, 0.4, 0.9)
        solid = _cut_rect(solid, cx, cy, cl, cw, angle, -1, t + 1)
        features.append(
            {
                "type": "rect_through_cut",
                "center": [cx, cy],
                "size": [cl, cw],
                "angle_deg": angle,
                "depth": t,
                "axis": [0, 0, 1],
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


def _convex_outline(rng, n: int, rx: float, ry: float):
    base = np.sort(rng.uniform(0, 2 * math.pi, size=n))
    gaps = np.diff(np.append(base, base[0] + 2 * math.pi))
    while gaps.min() < math.radians(18) or gaps.max() > math.radians(150):
        base = np.sort(rng.uniform(0, 2 * math.pi, size=n))
        gaps = np.diff(np.append(base, base[0] + 2 * math.pi))
    return [(_r(rx * math.cos(a)), _r(ry * math.sin(a))) for a in base]


@register("polygon_prism")
def polygon_prism(rng):
    n = int(rng.integers(3, 11))
    rx, ry = _r(rng.uniform(15, 50)), _r(rng.uniform(15, 40))
    h = _r(rng.uniform(5, 25))
    pts = _convex_outline(rng, n, rx, ry)
    solid = _prism(pts, h, 0.0)
    features = [{"type": "polygon_outline", "points": [list(p) for p in pts], "height": h}]
    return _clean(solid), {"n": n, "radii": [rx, ry], "height": h}, features


@register("lshape_outline")
def lshape_outline(rng):
    a = _r(rng.uniform(40, 90))
    b = _r(rng.uniform(30, 70))
    ta = _r(rng.uniform(0.3, 0.6) * b)
    tb = _r(rng.uniform(0.3, 0.6) * a)
    h = _r(rng.uniform(5, 25))
    pts = [(0, 0), (a, 0), (a, ta), (tb, ta), (tb, b), (0, b)]
    angle = _r(rng.uniform(0, 90))
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    pts = [(_r(x * ca - y * sa), _r(x * sa + y * ca)) for x, y in pts]
    solid = _prism(pts, h, 0.0)
    features = [
        {"type": "polygon_outline", "points": [list(p) for p in pts], "height": h},
        {"type": "concave_corner", "count": 1},
    ]
    return _clean(solid), {"legs": [a, b], "thickness": [ta, tb], "angle_deg": angle}, features


@register("thin_walls")
def thin_walls(rng):
    length = _r(rng.uniform(25, 80))
    width = _r(rng.uniform(20, 60))
    height = _r(rng.uniform(10, 30))
    wall = _r(rng.uniform(0.2, 0.6))
    floor = _r(rng.uniform(0.5, 2.0))
    solid = Pos(0, 0, height / 2) * Box(length, width, height)
    solid = solid - _prism(
        _rect_points(0, 0, length - 2 * wall, width - 2 * wall, 0), height, floor
    )
    features = [
        {
            "type": "rect_pocket",
            "center": [0.0, 0.0],
            "size": [_r(length - 2 * wall), _r(width - 2 * wall)],
            "angle_deg": 0.0,
            "depth": _r(height - floor),
            "floor_z": floor,
            "axis": [0, 0, 1],
        },
        {"type": "thin_wall", "thickness": wall},
    ]
    if rng.random() < 0.6:
        rib = _r(rng.uniform(0.3, 0.8))
        rib_h = _r(rng.uniform(0.4, 0.9) * (height - floor))
        solid = solid + _prism(_rect_points(0, 0, length - 2 * wall + 0.4, rib, 0), rib_h, floor)
        features.append({"type": "thin_rib", "thickness": rib, "height": rib_h, "base_z": floor})
    return _clean(solid), {"block": [length, width, height], "wall": wall}, features
