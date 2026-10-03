from __future__ import annotations

import math

from build123d import Cylinder, Pos, RegularPolygon, Shape, extrude

from .core import register


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
