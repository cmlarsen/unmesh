from __future__ import annotations

import math

from build123d import Axis, Box, Cylinder, Pos, RegularPolygon, Shape, chamfer, extrude, fillet

from .core import register, validity_problems


def _r(x: float) -> float:
    return round(float(x), 3)


def _prism_params(rng):
    n = int(rng.integers(5, 33))
    radius = _r(rng.uniform(10, 40))
    height = _r(rng.uniform(5, 25))
    return n, radius, height


def _ngon_deflection(n: int, radius: float) -> tuple[float, float]:
    sagitta = radius * (1 - math.cos(math.pi / n))
    return sagitta * 2 + 1e-6, 4 * math.pi / n * (1 + 1e-9)


def _gon_points(n: int, radius: float) -> list[list[float]]:
    return [
        [_r(radius * math.cos(2 * math.pi * k / n)), _r(radius * math.sin(2 * math.pi * k / n))]
        for k in range(n)
    ]


@register("ngon_prism")
def ngon_prism(rng):
    n, radius, height = _prism_params(rng)
    solid = (Pos(0, 0, 0) * extrude(RegularPolygon(radius, n), amount=height)).clean()
    lin, ang = _ngon_deflection(n, radius)
    params = {
        "n": n,
        "radius": radius,
        "height": height,
        "ambiguity": "ngon_vs_cylinder",
        "truth": "polygon",
        "pair_family": "coarse_cylinder_prism",
        "pair_deflection": [lin, ang],
    }
    features = [
        {
            "type": "regular_polygon_outline",
            "points": _gon_points(n, radius),
            "n": n,
            "radius": radius,
            "height": height,
            "axis": [0, 0, 1],
        }
    ]
    return solid, params, features


@register("coarse_cylinder_prism")
def coarse_cylinder_prism(rng):
    n, radius, height = _prism_params(rng)
    solid: Shape = (Pos(0, 0, height / 2) * Cylinder(radius, height)).clean()
    lin, ang = _ngon_deflection(n, radius)
    params = {
        "n": n,
        "radius": radius,
        "height": height,
        "ambiguity": "ngon_vs_cylinder",
        "truth": "cylinder",
        "pair_family": "ngon_prism",
        "pair_deflection": [lin, ang],
    }
    features = [
        {
            "type": "coarse_cylinder",
            "radius": radius,
            "height": height,
            "segments": n,
            "axis": [0, 0, 1],
        }
    ]
    return solid, params, features


def _box(length: float, width: float, height: float) -> Shape:
    return Pos(0, 0, height / 2) * Box(length, width, height)


def _ok(solid: Shape, faces: int) -> bool:
    return validity_problems(solid) == [] and len(solid.faces()) == faces


def _corner_box_params(rng):
    length = _r(rng.uniform(40, 100))
    width = _r(rng.uniform(30, 80))
    height = _r(rng.uniform(10, 30))
    r = _r(rng.uniform(1, 8))
    return length, width, height, min(r, _r(0.2 * min(length, width)))


def _corner_edge(solid: Shape):
    return max(solid.edges().filter_by(Axis.Z), key=lambda e: e.center().X + e.center().Y)


def _fillet1_deflection(r: float) -> tuple[float, float]:
    return r * (1 - math.cos(math.pi / 4)) * 2 + 1e-9, math.pi * (1 + 1e-9)


@register("one_segment_fillet")
def one_segment_fillet(rng):
    length, width, height, r = _corner_box_params(rng)
    solid = _box(length, width, height)
    for _ in range(8):
        try:
            candidate = fillet([_corner_edge(solid)], radius=r)
        except Exception:
            r = _r(r / 2)
            continue
        if _ok(candidate, 7):
            solid = candidate
            break
        r = _r(r / 2)
    else:
        raise RuntimeError("one_segment_fillet: no valid build after 8 attempts")
    lin, ang = _fillet1_deflection(r)
    params = {
        "box": [length, width, height],
        "radius": r,
        "ambiguity": "fillet1_vs_chamfer",
        "truth": "fillet",
        "pair_family": "chamfer_same_chord",
        "pair_deflection": [lin, ang],
    }
    features = [{"type": "vertical_fillet", "radius": r, "edges": 1, "axis": [0, 0, 1]}]
    return solid, params, features


@register("chamfer_same_chord")
def chamfer_same_chord(rng):
    length, width, height, r = _corner_box_params(rng)
    c = r
    solid = _box(length, width, height)
    for _ in range(8):
        try:
            candidate = chamfer([_corner_edge(solid)], length=c)
        except Exception:
            c = _r(c / 2)
            r = _r(r / 2)
            continue
        if _ok(candidate, 7):
            solid = candidate
            break
        c = _r(c / 2)
        r = _r(r / 2)
    else:
        raise RuntimeError("chamfer_same_chord: no valid build after 8 attempts")
    lin, ang = _fillet1_deflection(r)
    params = {
        "box": [length, width, height],
        "chamfer": c,
        "ambiguity": "fillet1_vs_chamfer",
        "truth": "chamfer",
        "pair_family": "one_segment_fillet",
        "pair_deflection": [lin, ang],
    }
    features = [{"type": "vertical_chamfer", "length": c, "edges": 1, "axis": [0, 0, 1]}]
    return solid, params, features
