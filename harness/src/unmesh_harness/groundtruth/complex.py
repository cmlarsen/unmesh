from __future__ import annotations

import math

import numpy as np
from build123d import (
    Axis,
    Box,
    Compound,
    Cone,
    Cylinder,
    Polygon,
    Pos,
    chamfer,
    extrude,
    fillet,
)

from .core import register, validity_problems


def _r(x: float) -> float:
    return round(float(x), 3)


def _plate(length: float, width: float, thickness: float):
    return Pos(0, 0, thickness / 2) * Box(length, width, thickness)


def _cells(length, width, cols, rows, margin, gap):
    cw = (length - 2 * margin - (cols - 1) * gap) / cols
    ch = (width - 2 * margin - (rows - 1) * gap) / rows
    out = []
    for i in range(cols):
        for j in range(rows):
            cx = -length / 2 + margin + cw / 2 + i * (cw + gap)
            cy = -width / 2 + margin + ch / 2 + j * (ch + gap)
            out.append((_r(cx), _r(cy), _r(cw), _r(ch)))
    return out


def _prism(points, height: float, z0: float):
    return Pos(0, 0, z0) * extrude(Polygon(*points, align=None), amount=height)


def _rect_points(cx, cy, length, width):
    return [
        (cx - length / 2, cy - width / 2),
        (cx + length / 2, cy - width / 2),
        (cx + length / 2, cy + width / 2),
        (cx - length / 2, cy + width / 2),
    ]


def _clean(shape):
    return shape.clean()


def _edge_treatment(solid, kind: str, size: float):
    if kind == "chamfer":
        return chamfer(solid.edges().group_by(Axis.Z)[-1], length=size)
    return fillet(solid.edges().filter_by(Axis.Z), radius=size)


def _apply_treatment(solid, kind: str | None, size: float, faces_if_ok: int, shells: int, loss):
    if kind is None:
        return solid, None
    size = _r(size)
    before = solid.volume
    for _ in range(8):
        try:
            candidate = _edge_treatment(solid, kind, size)
        except Exception:
            size = _r(size / 2)
            continue
        if (
            validity_problems(candidate, shells=shells) == []
            and len(candidate.faces()) == faces_if_ok
            and math.isclose(before - candidate.volume, loss(size), rel_tol=1e-9)
        ):
            return candidate, size
        size = _r(size / 2)
    return solid, None


def _chamfer_loss(length, width):
    def loss(c):
        return c * c * (length + width) - 4 * c**3 / 3

    return loss


def _fillet_loss(total_edge_length):
    def loss(r):
        return (1 - math.pi / 4) * r**2 * total_edge_length

    return loss


def _pocket_features(rng, cell, t):
    ccx, ccy, cw, ch = cell
    pl = _r(rng.uniform(0.45, 0.85) * cw)
    pw = _r(rng.uniform(0.45, 0.85) * ch)
    cx = _r(ccx + rng.uniform(-1, 1) * (cw - pl) / 2)
    cy = _r(ccy + rng.uniform(-1, 1) * (ch - pw) / 2)
    depth = _r(rng.uniform(0.25, 0.8) * t)
    tool = _prism(_rect_points(cx, cy, pl, pw), depth + 1, t - depth)
    feat = {
        "type": "rect_pocket",
        "center": [cx, cy],
        "size": [pl, pw],
        "depth": depth,
        "floor_z": _r(t - depth),
        "axis": [0, 0, 1],
    }
    return tool, feat


def _through_bore_features(rng, cell, t):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    r = _r(math.exp(rng.uniform(math.log(0.5), math.log(rmax))))
    cx = _r(ccx + rng.uniform(-1, 1) * (cw / 2 - r - 0.3))
    cy = _r(ccy + rng.uniform(-1, 1) * (ch / 2 - r - 0.3))
    tool = Pos(cx, cy, t / 2) * Cylinder(r, t + 2)
    feat = {
        "type": "through_bore",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "radius": r,
        "depth": t,
    }
    return tool, feat


def _blind_bore_features(rng, cell, t, max_depth=None):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    r = _r(math.exp(rng.uniform(math.log(0.5), math.log(rmax))))
    cx = _r(ccx + rng.uniform(-1, 1) * (cw / 2 - r - 0.3))
    cy = _r(ccy + rng.uniform(-1, 1) * (ch / 2 - r - 0.3))
    cap = t if max_depth is None else min(t, max_depth)
    depth = _r(rng.uniform(0.25, 0.8) * cap)
    tool = Pos(cx, cy, t - depth / 2 + 0.5) * Cylinder(r, depth + 1)
    feat = {
        "type": "blind_bore",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "radius": r,
        "depth": depth,
        "floor_z": _r(t - depth),
    }
    return tool, feat


def _boss_features(rng, cell, t):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    r = _r(math.exp(rng.uniform(math.log(0.7), math.log(max(0.8, rmax)))))
    cx = _r(ccx + rng.uniform(-1, 1) * (cw / 2 - r - 0.3))
    cy = _r(ccy + rng.uniform(-1, 1) * (ch / 2 - r - 0.3))
    h = _r(rng.uniform(2, 8))
    tool = Pos(cx, cy, t + h / 2) * Cylinder(r, h)
    feat = {
        "type": "round_boss",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "radius": r,
        "height": h,
        "base_z": t,
    }
    return tool, feat


def _counterbore_features(rng, cell, t):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    rs = _r(math.exp(rng.uniform(math.log(0.7), math.log(max(0.8, rmax * 0.5)))))
    rb = _r(min(rmax, rs + float(rng.uniform(2, 4))))
    if rb - rs < 1.0:
        rb = _r(rs + 1.0)
    cx = _r(ccx + rng.uniform(-1, 1) * (cw / 2 - rb - 0.3))
    cy = _r(ccy + rng.uniform(-1, 1) * (ch / 2 - rb - 0.3))
    db = _r(rng.uniform(0.2, 0.5) * t)
    shaft = Pos(cx, cy, t / 2) * Cylinder(rs, t + 2)
    mouth = Pos(cx, cy, t - db / 2 + 0.5) * Cylinder(rb, db + 1)
    feat = {
        "type": "counterbore",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "shaft_radius": rs,
        "bore_radius": rb,
        "bore_depth": db,
        "depth": t,
    }
    return shaft + mouth, feat


def _countersink_features(rng, cell, t):
    ccx, ccy, cw, ch = cell
    rmax = min(cw, ch) / 2 - 0.5
    rs = _r(math.exp(rng.uniform(math.log(0.7), math.log(max(0.8, rmax * 0.5)))))
    rm = _r(min(rmax, rs + float(rng.uniform(2, 4))))
    if rm - rs < 1.0:
        rm = _r(rs + 1.0)
    cx = _r(ccx + rng.uniform(-1, 1) * (cw / 2 - rm - 0.3))
    cy = _r(ccy + rng.uniform(-1, 1) * (ch / 2 - rm - 0.3))
    dc = _r(rng.uniform(0.15, 0.4) * t)
    shaft = Pos(cx, cy, t / 2) * Cylinder(rs, t + 2)
    cone = Pos(cx, cy, t - dc / 2) * Cone(rs, rm, dc)
    feat = {
        "type": "countersink",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "shaft_radius": rs,
        "mouth_radius": rm,
        "cone_depth": dc,
        "half_angle": float(math.atan2(rm - rs, dc)),
        "depth": t,
    }
    return shaft + cone, feat


def _stadium(cx, cy, span, r, z0, height):
    pts = [
        (cx - span / 2, cy - r),
        (cx + span / 2, cy - r),
        (cx + span / 2, cy + r),
        (cx - span / 2, cy + r),
    ]
    rect = Pos(0, 0, z0) * extrude(Polygon(*pts, align=None), amount=height)
    c1 = Pos(cx - span / 2, cy, z0 + height / 2) * Cylinder(r, height)
    c2 = Pos(cx + span / 2, cy, z0 + height / 2) * Cylinder(r, height)
    return rect + c1 + c2


def _slot_features(rng, cell, t, through: bool):
    ccx, ccy, cw, ch = cell
    slot_l = _r(rng.uniform(0.5, 0.85) * cw)
    r = _r(min(float(rng.uniform(0.2, 0.4)) * ch, slot_l / 4))
    span = _r(slot_l - 2 * r)
    cx = _r(ccx + rng.uniform(-1, 1) * (cw - slot_l) / 2)
    cy = _r(ccy + rng.uniform(-1, 1) * (ch - 2 * r) / 2)
    if through:
        tool = _stadium(cx, cy, span, r, -1, t + 2)
        depth = t
    else:
        depth = _r(rng.uniform(0.3, 0.7) * t)
        tool = _stadium(cx, cy, span, r, t - depth, depth + 1)
    feat = {
        "type": "round_slot",
        "center": [cx, cy, t],
        "axis": [0, 0, 1],
        "radius": r,
        "span": span,
        "length": _r(span + 2 * r),
        "width": _r(2 * r),
        "depth": depth,
        "through": through,
    }
    if not through:
        feat["floor_z"] = _r(t - depth)
    return tool, feat


MIXED_PATTERN = (
    "pocket",
    "counterbore",
    "slot_blind",
    "boss",
    "through_bore",
    "pocket",
    "countersink",
    "slot_through",
    "pocket",
    "counterbore",
    "blind_bore",
    "boss",
    "pocket",
    "slot_blind",
    "through_bore",
    "counterbore",
)


def _build_mixed(rng, small: bool = False):
    if small:
        length = _r(rng.uniform(60, 90))
        width = _r(rng.uniform(50, 70))
    else:
        length = _r(rng.uniform(80, 120))
        width = _r(rng.uniform(60, 100))
    cols, rows = 4, 4
    t = _r(rng.uniform(8, 16))
    kind = str(rng.choice(["chamfer", "fillet", "none"]))
    size = _r(rng.uniform(1, 3)) if kind != "none" else 0.0
    solid = _plate(length, width, t)
    if kind == "chamfer":
        solid, size = _apply_treatment(solid, kind, size, 10, 1, _chamfer_loss(length, width))
        treatment = {"kind": kind, "size": size} if size else None
    elif kind == "fillet":
        solid, size = _apply_treatment(solid, kind, size, 10, 1, _fillet_loss(4 * t))
        treatment = {"kind": kind, "size": size} if size else None
    else:
        treatment = None
    margin = _r(rng.uniform(3.0, 5.0) + (size or 0.0))
    gap = _r(rng.uniform(2.0, 4.0))
    cells = _cells(length, width, cols, rows, margin, gap)
    order = rng.permutation(len(cells))
    kinds = [MIXED_PATTERN[i % len(MIXED_PATTERN)] for i in range(len(cells))]
    kinds = [kinds[i] for i in rng.permutation(len(kinds))]
    features = []
    for pos, cell in zip(order, cells, strict=True):
        op = kinds[pos]
        if op == "pocket":
            tool, feat = _pocket_features(rng, cell, t)
            solid = solid - tool
        elif op == "through_bore":
            tool, feat = _through_bore_features(rng, cell, t)
            solid = solid - tool
        elif op == "blind_bore":
            tool, feat = _blind_bore_features(rng, cell, t)
            solid = solid - tool
        elif op == "boss":
            tool, feat = _boss_features(rng, cell, t)
            solid = solid + tool
        elif op == "counterbore":
            tool, feat = _counterbore_features(rng, cell, t)
            solid = solid - tool
        elif op == "countersink":
            tool, feat = _countersink_features(rng, cell, t)
            solid = solid - tool
        else:
            tool, feat = _slot_features(rng, cell, t, through=op == "slot_through")
            solid = solid - tool
        features.append(feat)
    solid = _clean(solid)
    params = {
        "plate": [length, width, t],
        "treatment": treatment,
        "solids": 1,
        "shells": 1,
    }
    return solid, params, features


def mixed_volume(params, features) -> float:
    length, width, t = params["plate"]
    vol = length * width * t
    tr = params.get("treatment")
    if tr is not None:
        if tr["kind"] == "chamfer":
            c = tr["size"]
            vol -= c * c * (length + width) - 4 * c**3 / 3
        else:
            vol -= 4 * (1 - math.pi / 4) * tr["size"] ** 2 * t
    for x in features:
        if x["type"] == "rect_pocket":
            vol -= x["size"][0] * x["size"][1] * x["depth"]
        elif x["type"] == "through_bore":
            vol -= math.pi * x["radius"] ** 2 * t
        elif x["type"] == "blind_bore":
            vol -= math.pi * x["radius"] ** 2 * x["depth"]
        elif x["type"] == "round_boss":
            vol += math.pi * x["radius"] ** 2 * x["height"]
        elif x["type"] == "counterbore":
            vol -= math.pi * x["bore_radius"] ** 2 * x["bore_depth"] + math.pi * x[
                "shaft_radius"
            ] ** 2 * (t - x["bore_depth"])
        elif x["type"] == "countersink":
            vol -= math.pi * x["cone_depth"] / 3 * (
                x["mouth_radius"] ** 2
                + x["mouth_radius"] * x["shaft_radius"]
                + x["shaft_radius"] ** 2
            ) + math.pi * x["shaft_radius"] ** 2 * (t - x["cone_depth"])
        elif x["type"] == "round_slot":
            vol -= (x["span"] * 2 * x["radius"] + math.pi * x["radius"] ** 2) * x["depth"]
    return vol


@register("complex_mixed")
def complex_mixed(rng):
    return _build_mixed(rng)


def _build_finned(rng, small: bool = False):
    length = _r(rng.uniform(50, 80) if small else rng.uniform(60, 100))
    width = _r(rng.uniform(36, 60) if small else rng.uniform(40, 70))
    t = _r(rng.uniform(6, 12))
    solid = _plate(length, width, t)
    n_fins = int(rng.integers(5, 7))
    inset = _r(rng.uniform(3, 5))
    ys = np.linspace(-width / 2 + 4, width / 2 - 4, n_fins)
    ys = [_r(y + rng.uniform(-0.3, 0.3)) for y in ys]
    features = []
    thin_lo = 0.05 if not small else 0.15
    for y in ys:
        th = _r(rng.uniform(thin_lo, 0.6))
        fh = _r(rng.uniform(3, 10))
        fl = _r(length - 2 * inset)
        solid = solid + _prism(_rect_points(0, y, fl, th), fh, t)
        features.append(
            {
                "type": "thin_fin",
                "center": [0.0, y],
                "length": fl,
                "thickness": th,
                "height": fh,
                "base_z": t,
                "axis": [0, 0, 1],
            }
        )
    bands = []
    bounds = [-width / 2] + sorted(ys) + [width / 2]
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 - 0.5
        inner = 0 < k < len(bounds) - 2
        if half >= 1.5 or (inner and half >= 1.2):
            bands.append((_r(mid), _r(half), inner))
    free_per_band = int(rng.integers(3, 5))
    for cy, half, inner in bands:
        if not inner:
            continue
        xs = np.linspace(-length / 2 + 4, length / 2 - 4, 2 + free_per_band)
        for j, x in enumerate(xs):
            x = _r(x + rng.uniform(-0.5, 0.5))
            if j == 0:
                s = _r(min(rng.uniform(0.3, 1.2), half - 0.3))
                depth = _r(rng.uniform(1.0, 0.4 * t))
                solid = solid - _prism(_rect_points(x, cy, s, s), depth + 1, t - depth)
                features.append(
                    {
                        "type": "micro_pocket",
                        "center": [x, cy],
                        "size": [s, s],
                        "depth": depth,
                        "floor_z": _r(t - depth),
                        "axis": [0, 0, 1],
                    }
                )
                continue
            if j == 1:
                r = _r(min(rng.uniform(0.3, 0.8), half - 0.4))
                h = _r(rng.uniform(1, 3))
                solid = solid + (Pos(x, cy, t + h / 2) * Cylinder(r, h))
                features.append(
                    {
                        "type": "micro_boss",
                        "center": [x, cy, t],
                        "axis": [0, 0, 1],
                        "radius": r,
                        "height": h,
                        "base_z": t,
                    }
                )
                continue
            pick = rng.random()
            rmax = min(half - 0.3, 1.2)
            if pick < 0.45 and rmax >= 0.1:
                r = _r(rng.uniform(0.1, max(0.11, rmax)))
                solid = solid - (Pos(x, cy, t / 2) * Cylinder(r, t + 2))
                features.append(
                    {
                        "type": "micro_bore",
                        "center": [x, cy, t],
                        "axis": [0, 0, 1],
                        "radius": r,
                        "depth": t,
                    }
                )
            elif pick < 0.7 and half >= 1.0:
                s = _r(min(rng.uniform(0.3, 1.2), half - 0.3))
                depth = _r(rng.uniform(1.0, 0.4 * t))
                solid = solid - _prism(_rect_points(x, cy, s, s), depth + 1, t - depth)
                features.append(
                    {
                        "type": "micro_pocket",
                        "center": [x, cy],
                        "size": [s, s],
                        "depth": depth,
                        "floor_z": _r(t - depth),
                        "axis": [0, 0, 1],
                    }
                )
            elif half >= 1.2:
                r = _r(min(rng.uniform(0.3, 0.8), half - 0.4))
                h = _r(rng.uniform(1, 3))
                solid = solid + (Pos(x, cy, t + h / 2) * Cylinder(r, h))
                features.append(
                    {
                        "type": "micro_boss",
                        "center": [x, cy, t],
                        "axis": [0, 0, 1],
                        "radius": r,
                        "height": h,
                        "base_z": t,
                    }
                )
    solid = _clean(solid)
    params = {"plate": [length, width, t], "solids": 1, "shells": 1}
    return solid, params, features


def finned_volume(params, features) -> float:
    length, width, t = params["plate"]
    vol = length * width * t
    for x in features:
        if x["type"] == "thin_fin":
            vol += x["length"] * x["thickness"] * x["height"]
        elif x["type"] == "micro_bore":
            vol -= math.pi * x["radius"] ** 2 * t
        elif x["type"] == "micro_pocket":
            vol -= x["size"][0] * x["size"][1] * x["depth"]
        elif x["type"] == "micro_boss":
            vol += math.pi * x["radius"] ** 2 * x["height"]
    return vol


@register("complex_thin")
def complex_thin(rng):
    return _build_finned(rng)


def _build_void(rng, small: bool = False):
    length = _r(rng.uniform(50, 70) if small else rng.uniform(60, 100))
    width = _r(rng.uniform(40, 60) if small else rng.uniform(50, 80))
    height = _r(rng.uniform(26, 40) if small else rng.uniform(30, 50))
    lc = _r(length * rng.uniform(0.45, 0.7))
    wc = _r(width * rng.uniform(0.45, 0.7))
    hc = _r(height * rng.uniform(0.35, 0.6))
    wx, wy, wz = _r((length - lc) / 2), _r((width - wc) / 2), _r((height - hc) / 2)
    solid = (Pos(0, 0, height / 2) * Box(length, width, height)) - (
        Pos(0, 0, height / 2) * Box(lc, wc, hc)
    )
    kind = str(rng.choice(["chamfer", "fillet", "none"]))
    size = _r(rng.uniform(1, 3)) if kind != "none" else 0.0
    if kind == "chamfer":
        solid, size = _apply_treatment(solid, kind, size, 16, 2, _chamfer_loss(length, width))
    elif kind == "fillet":
        solid, size = _apply_treatment(solid, kind, size, 20, 2, _fillet_loss(4 * height - 4 * hc))
    treatment = {"kind": kind, "size": size} if size else None
    features = [
        {
            "type": "internal_void",
            "center": [0.0, 0.0, _r(height / 2)],
            "size": [lc, wc, hc],
        }
    ]
    wall_top = wz
    cells = _cells(length, width, 4, 2, _r(4.0 + (size or 0.0)), _r(3.0))
    for i, cell in enumerate(cells):
        if i % 3 == 0:
            ccx, ccy, cw, ch = cell
            pl = _r(rng.uniform(0.45, 0.85) * cw)
            pw = _r(rng.uniform(0.45, 0.85) * ch)
            cx = _r(ccx + rng.uniform(-1, 1) * (cw - pl) / 2)
            cy = _r(ccy + rng.uniform(-1, 1) * (ch - pw) / 2)
            depth = _r(min(float(rng.uniform(0.25, 0.8) * height), wall_top - 3))
            tool = _prism(_rect_points(cx, cy, pl, pw), depth + 1, height - depth)
            feat = {
                "type": "rect_pocket",
                "center": [cx, cy],
                "size": [pl, pw],
                "depth": depth,
                "floor_z": _r(height - depth),
                "axis": [0, 0, 1],
            }
            solid = solid - tool
        elif i % 3 == 1:
            tool, feat = _blind_bore_features(rng, cell, height, max_depth=wall_top - 3)
            solid = solid - tool
        else:
            tool, feat = _boss_features(rng, cell, height)
            solid = solid + tool
        features.append(feat)
    for sx in (-1, 1):
        for half in (-1, 1):
            lo, hi = (3.0, width / 2 - 6) if half > 0 else (-width / 2 + 6, -3.0)
            pw = _r(rng.uniform(4, min(10.0, hi - lo - 1)))
            y = _r(rng.uniform(lo + pw / 2, hi - pw / 2))
            ph = _r(min(float(rng.uniform(4, 8)), height - wall_top - 4))
            depth = _r(rng.uniform(2, wx - 3))
            zc = _r(rng.uniform(1 + ph / 2, height - wall_top + 1 - ph / 2))
            x0 = sx * length / 2
            box = Pos(x0 - sx * (depth - 0.5) / 2, y, zc) * Box(depth + 0.5, pw, ph)
            solid = solid - box
            features.append(
                {
                    "type": "side_pocket",
                    "center": [x0, y, zc],
                    "size": [pw, ph],
                    "depth": depth,
                    "axis": [sx, 0, 0],
                }
            )
    solid = _clean(solid)
    params = {
        "box": [length, width, height],
        "cavity": [lc, wc, hc],
        "walls": [wx, wy, wz],
        "treatment": treatment,
        "solids": 1,
        "shells": 2,
    }
    return solid, params, features


def void_volume(params, features) -> float:
    length, width, height = params["box"]
    lc, wc, hc = params["cavity"]
    vol = length * width * height - lc * wc * hc
    tr = params.get("treatment")
    if tr is not None:
        if tr["kind"] == "chamfer":
            c = tr["size"]
            vol -= c * c * (length + width) - 4 * c**3 / 3
        else:
            # Fillets run along the outer vertical edges (convex: remove material)
            # and the cavity vertical edges (concave: add material).
            r = tr["size"]
            vol -= 4 * (1 - math.pi / 4) * r**2 * height
            vol += 4 * (1 - math.pi / 4) * r**2 * hc
    for x in features:
        if x["type"] in ("rect_pocket", "side_pocket"):
            vol -= x["size"][0] * x["size"][1] * x["depth"]
        elif x["type"] == "blind_bore":
            vol -= math.pi * x["radius"] ** 2 * x["depth"]
        elif x["type"] == "round_boss":
            vol += math.pi * x["radius"] ** 2 * x["height"]
    return vol


@register("complex_void")
def complex_void(rng):
    return _build_void(rng)


@register("complex_assembly")
def complex_assembly(rng):
    a, pa, fa = _build_mixed(rng, small=True)
    b, pb, fb = _build_finned(rng, small=True)
    c, pc, fc = _build_void(rng, small=True)
    bodies = [(a, pa, fa, "mixed"), (b, pb, fb, "finned"), (c, pc, fc, "void")]
    placed = []
    cursor = 0.0
    offsets = []
    for solid, _, _, _ in bodies:
        bb = solid.bounding_box()
        dx = _r(cursor - bb.min.X + 8.0)
        placed.append(Pos(dx, 0, 0) * solid)
        offsets.append(dx)
        cursor += _r(bb.max.X - bb.min.X) + 16.0
    solid = Compound(placed)
    features = []
    for i, (_, _, feats, kind) in enumerate(bodies):
        for f in feats:
            tagged = dict(f)
            tagged["body"] = i
            tagged["body_kind"] = kind
            tagged["center"] = [offsets[i] + f["center"][0], *f["center"][1:]]
            features.append(tagged)
    params = {
        "bodies": [p for _, p, _, _ in bodies],
        "body_kinds": [k for _, _, _, k in bodies],
        "offsets": offsets,
        "solids": 3,
        "shells": 4,
    }
    return solid, params, features


def assembly_volume(params, features) -> float:
    vols = []
    by_body: dict[int, list] = {}
    for f in features:
        by_body.setdefault(f["body"], []).append(f)
    pairs = zip(params["bodies"], params["body_kinds"], strict=True)
    for i, (body_params, kind) in enumerate(pairs):
        feats = by_body.get(i, [])
        if kind == "mixed":
            vols.append(mixed_volume(body_params, feats))
        elif kind == "finned":
            vols.append(finned_volume(body_params, feats))
        else:
            vols.append(void_volume(body_params, feats))
    return sum(vols)
