from __future__ import annotations

import math

import numpy as np
from build123d import Box, Cone, Cylinder, Polygon, Pos, Sphere, Torus, extrude

from .core import register

Cell = tuple[float, float, float, float]


def _r(x: float) -> float:
    return round(float(x), 3)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _plate_dims(rng: np.random.Generator):
    return (
        _r(rng.uniform(40, 120)),
        _r(rng.uniform(30, 100)),
        _r(rng.uniform(6, 20)),
    )


def _plate(length: float, width: float, thickness: float):
    return Pos(0, 0, thickness / 2) * Box(length, width, thickness)


def _cells(
    rng: np.random.Generator, length: float, width: float, max_cols: int = 3, max_rows: int = 2
) -> list[Cell]:
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


def _disc(rng: np.random.Generator, cell: Cell, lo: float = 0.5):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    r = _r(_loguniform(rng, lo, rmax))
    ox = float(rng.uniform(-1, 1)) * (cw / 2 - r - 0.3)
    oy = float(rng.uniform(-1, 1)) * (ch / 2 - r - 0.3)
    return _r(ccx + ox), _r(ccy + oy), r


def _shaft_tool(cx: float, cy: float, radius: float, z0: float, height: float):
    return Pos(cx, cy, z0 + height / 2) * Cylinder(radius, height)


def _clean(shape):
    return shape.clean()


@register("through_bore")
def through_bore(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    if rng.random() < 0.3:
        length, width, t = (
            _r(rng.uniform(100, 120)),
            _r(rng.uniform(100, 110)),
            _r(rng.uniform(8, 20)),
        )
        solid = _plate(length, width, t)
        rmax = min(50.0, min(length, width) / 2 - 2)
        r = _r(rng.uniform(0.6, 1.0) * rmax)
        solid = solid - _shaft_tool(0.0, 0.0, r, -1, t + 2)
        features.append(
            {
                "type": "through_bore",
                "center": [0.0, 0.0, t],
                "axis": [0, 0, 1],
                "radius": r,
                "depth": t,
            }
        )
    else:
        for cell in _cells(rng, length, width):
            cx, cy, r = _disc(rng, cell)
            solid = solid - _shaft_tool(cx, cy, r, -1, t + 2)
            features.append(
                {
                    "type": "through_bore",
                    "center": [cx, cy, t],
                    "axis": [0, 0, 1],
                    "radius": r,
                    "depth": t,
                }
            )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("blind_bore")
def blind_bore(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    if rng.random() < 0.3:
        length, width, t = (
            _r(rng.uniform(100, 120)),
            _r(rng.uniform(100, 110)),
            _r(rng.uniform(8, 20)),
        )
        solid = _plate(length, width, t)
        rmax = min(50.0, min(length, width) / 2 - 2)
        r = _r(rng.uniform(0.6, 1.0) * rmax)
        depth = _r(rng.uniform(0.25, 0.8) * t)
        solid = solid - _shaft_tool(0.0, 0.0, r, t - depth, depth)
        features.append(
            {
                "type": "blind_bore",
                "center": [0.0, 0.0, t],
                "axis": [0, 0, 1],
                "radius": r,
                "depth": depth,
                "floor_z": _r(t - depth),
            }
        )
    else:
        for cell in _cells(rng, length, width):
            cx, cy, r = _disc(rng, cell)
            depth = _r(rng.uniform(0.25, 0.8) * t)
            solid = solid - _shaft_tool(cx, cy, r, t - depth, depth)
            features.append(
                {
                    "type": "blind_bore",
                    "center": [cx, cy, t],
                    "axis": [0, 0, 1],
                    "radius": r,
                    "depth": depth,
                    "floor_z": _r(t - depth),
                }
            )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("round_boss")
def round_boss(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        if rng.random() < 0.8:
            cx, cy, r = _disc(rng, cell, lo=1.0)
            h = _r(rng.uniform(2, 10))
            solid = solid + _shaft_tool(cx, cy, r, t, h)
            features.append(
                {
                    "type": "round_boss",
                    "center": [cx, cy, t],
                    "axis": [0, 0, 1],
                    "radius": r,
                    "height": h,
                    "base_z": t,
                }
            )
    if not features:
        r = _r(min(length, width) / 6)
        h = _r(rng.uniform(2, 10))
        solid = solid + _shaft_tool(0.0, 0.0, r, t, h)
        features.append(
            {
                "type": "round_boss",
                "center": [0.0, 0.0, t],
                "axis": [0, 0, 1],
                "radius": r,
                "height": h,
                "base_z": t,
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


def _stepped_radii(rng: np.random.Generator, cell: Cell):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    rs = _r(_loguniform(rng, 1.0, max(1.2, rmax * 0.5)))
    rb = _r(min(rmax, rs + float(rng.uniform(2, 5))))
    if rb - rs < 1.0:
        rb = _r(rs + 1.0)
    ox = float(rng.uniform(-1, 1)) * (cw / 2 - rb - 0.3)
    oy = float(rng.uniform(-1, 1)) * (ch / 2 - rb - 0.3)
    return _r(ccx + ox), _r(ccy + oy), rs, rb


@register("counterbore")
def counterbore(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        cx, cy, rs, rb = _stepped_radii(rng, cell)
        db = _r(rng.uniform(0.2, 0.5) * t)
        shaft = _shaft_tool(cx, cy, rs, -1, t + 2)
        mouth = _shaft_tool(cx, cy, rb, t - db, db)
        solid = solid - (shaft + mouth)
        features.append(
            {
                "type": "counterbore",
                "center": [cx, cy, t],
                "axis": [0, 0, 1],
                "shaft_radius": rs,
                "bore_radius": rb,
                "bore_depth": db,
                "depth": t,
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("countersink")
def countersink(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cell in _cells(rng, length, width):
        cx, cy, rs, rm = _stepped_radii(rng, cell)
        dc = _r(rng.uniform(0.15, 0.4) * t)
        shaft = _shaft_tool(cx, cy, rs, -1, t + 2)
        cone = Pos(cx, cy, t - dc / 2) * Cone(rs, rm, dc)
        solid = solid - (shaft + cone)
        features.append(
            {
                "type": "countersink",
                "center": [cx, cy, t],
                "axis": [0, 0, 1],
                "shaft_radius": rs,
                "mouth_radius": rm,
                "cone_depth": dc,
                "half_angle": float(math.atan2(rm - rs, dc)),
                "depth": t,
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


def _stadium_tool(cx, cy, ux, uy, span, r, z0, height):
    px, py = -uy, ux
    pts = [
        (cx + dx * ux + dy * px, cy + dx * uy + dy * py)
        for dx, dy in ((-span / 2, -r), (span / 2, -r), (span / 2, r), (-span / 2, r))
    ]
    rect = Pos(0, 0, z0) * extrude(Polygon(*pts, align=None), amount=height)
    c1 = Pos(cx - ux * span / 2, cy - uy * span / 2, z0 + height / 2) * Cylinder(r, height)
    c2 = Pos(cx + ux * span / 2, cy + uy * span / 2, z0 + height / 2) * Cylinder(r, height)
    return rect + c1 + c2


def _slot_rows(rng: np.random.Generator, length: float, width: float):
    n = int(rng.integers(1, 5))
    margin = float(rng.uniform(3.0, 5.0))
    pitch = (width - 2 * margin) / n
    rows = []
    for i in range(n):
        cy = _r(-width / 2 + margin + pitch * (i + 0.5))
        lmax = length - 2 * margin
        slot_l = _r(float(rng.uniform(0.4, 0.9)) * lmax)
        r = _r(min(float(rng.uniform(0.25, 0.55)) * pitch / 2, slot_l / 4))
        span = _r(slot_l - 2 * r)
        rows.append((cy, r, span))
    return rows


@register("round_slot_through")
def round_slot_through(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cy, r, span in _slot_rows(rng, length, width):
        solid = solid - _stadium_tool(0.0, cy, 1.0, 0.0, span, r, -1, t + 2)
        features.append(
            {
                "type": "round_slot",
                "center": [0.0, cy, t],
                "axis": [0, 0, 1],
                "angle_deg": 0.0,
                "radius": r,
                "span": span,
                "length": _r(span + 2 * r),
                "width": _r(2 * r),
                "depth": t,
                "through": True,
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("round_slot_blind")
def round_slot_blind(rng: np.random.Generator):
    length, width, t = _plate_dims(rng)
    solid = _plate(length, width, t)
    features = []
    for cy, r, span in _slot_rows(rng, length, width):
        depth = _r(rng.uniform(0.3, 0.8) * t)
        solid = solid - _stadium_tool(0.0, cy, 1.0, 0.0, span, r, t - depth, depth + 1)
        features.append(
            {
                "type": "round_slot",
                "center": [0.0, cy, t],
                "axis": [0, 0, 1],
                "angle_deg": 0.0,
                "radius": r,
                "span": span,
                "length": _r(span + 2 * r),
                "width": _r(2 * r),
                "depth": depth,
                "floor_z": _r(t - depth),
                "through": False,
            }
        )
    return _clean(solid), {"plate": [length, width, t]}, features


@register("revolved_cone")
def revolved_cone(rng: np.random.Generator):
    r1 = _r(rng.uniform(8, 25))
    h1 = _r(rng.uniform(5, 20))
    r2 = _r(r1 * float(rng.uniform(0.35, 1.7)))
    if abs(r2 - r1) < 1.0:
        r2 = _r(r1 + 4.0)
    h2 = _r(rng.uniform(5, 20))
    base = Pos(0, 0, h1 / 2) * Cylinder(r1, h1)
    frustum = Pos(0, 0, h1 + h2 / 2) * Cone(r1, r2, h2)
    solid = base + frustum
    features = [
        {
            "type": "cylinder_section",
            "center": [0.0, 0.0, _r(h1 / 2)],
            "axis": [0, 0, 1],
            "radius": r1,
            "height": h1,
            "base_z": 0.0,
        },
        {
            "type": "cone_section",
            "center": [0.0, 0.0, _r(h1 + h2 / 2)],
            "axis": [0, 0, 1 if r2 > r1 else -1],
            "apex": [0.0, 0.0, float(h1 - r1 * h2 / (r2 - r1))],
            "radius_bottom": r1,
            "radius_top": r2,
            "height": h2,
            "half_angle": float(math.atan2(abs(r2 - r1), h2)),
            "base_z": h1,
        },
    ]
    return (
        _clean(solid),
        {"r_base": r1, "h_base": h1, "r_top": r2, "h_cone": h2, "axis": [0, 0, 1]},
        features,
    )


@register("revolved_dome")
def revolved_dome(rng: np.random.Generator):
    r = _r(rng.uniform(5, 20))
    h = _r(rng.uniform(5, 20))
    stem = Pos(0, 0, h / 2) * Cylinder(r, h)
    cap = (Pos(0, 0, h) * Sphere(r)) - Pos(0, 0, h - 15) * Box(4 * r + 10, 4 * r + 10, 30)
    solid = stem + cap
    features = [
        {
            "type": "cylinder_section",
            "center": [0.0, 0.0, _r(h / 2)],
            "axis": [0, 0, 1],
            "radius": r,
            "height": h,
            "base_z": 0.0,
        },
        {
            "type": "sphere_cap",
            "center": [0.0, 0.0, h],
            "axis": [0, 0, 1],
            "radius": r,
            "base_z": h,
        },
    ]
    return _clean(solid), {"radius": r, "h_stem": h, "axis": [0, 0, 1]}, features


@register("revolved_torus")
def revolved_torus(rng: np.random.Generator):
    major = _r(rng.uniform(10, 35))
    minor = _r(min(float(rng.uniform(1.5, 8)), major / 3))
    solid = Torus(major, minor)
    features = [
        {
            "type": "torus_ring",
            "center": [0.0, 0.0, 0.0],
            "axis": [0, 0, 1],
            "major_radius": major,
            "minor_radius": minor,
        }
    ]
    return (
        _clean(solid),
        {"major_radius": major, "minor_radius": minor, "axis": [0, 0, 1]},
        features,
    )
