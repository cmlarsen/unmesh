from __future__ import annotations

import numpy as np

from .core import map_points, register

MM_PER_INCH = 25.4
FAR_TRANSLATION_MM = 1.0e6


def significant_digits(severity: float) -> int:
    return int(np.floor(9 - 6 * severity + 0.5))


def round_significant(x: np.ndarray, digits: int) -> np.ndarray:
    out = np.zeros_like(x)
    nz = x != 0.0
    exponent = np.floor(np.log10(np.abs(x[nz])))
    scale = 10.0 ** (digits - 1 - exponent)
    out[nz] = np.round(x[nz] * scale) / scale
    return out


def inch_decimals(severity: float) -> int:
    return int(np.floor(7 - 4 * severity + 0.5))


@register(
    "float32",
    "precision",
    "identity",
    "every coordinate rounded to the nearest float32 (the severity only switches the operator on)",
    binary=True,
)
def float32(mesh, severity, rng):
    map_points(mesh, lambda p: p.astype(np.float32).astype(np.float64))
    return {"dtype": "float32"}


@register(
    "truncated_digits",
    "precision",
    "identity",
    "coordinates written with 3 significant digits (9 digits at severity 0+, 6 at 0.5)",
    may_collapse=True,
)
def truncated_digits(mesh, severity, rng):
    n = significant_digits(severity)
    map_points(mesh, lambda p: round_significant(p, n))
    return {"significant_digits": n}


@register(
    "inch_round_trip",
    "precision",
    "identity",
    "mm converted to inches, written with 3 decimals (25 um grid), converted back "
    "(7 decimals at severity 0+, 5 at 0.5)",
    may_collapse=True,
)
def inch_round_trip(mesh, severity, rng):
    d = inch_decimals(severity)
    map_points(mesh, lambda p: np.round(p / MM_PER_INCH, d) * MM_PER_INCH)
    return {"inch_decimals": d, "grid_um": 1000.0 * MM_PER_INCH * 10.0**-d}


@register(
    "far_translation",
    "precision",
    "identity",
    "part translated up to 1e6 mm (1 km) along a random direction, rounded to float32 there "
    "(ulp 62 um), translated back exactly",
)
def far_translation(mesh, severity, rng):
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    offset = direction * severity * FAR_TRANSLATION_MM
    map_points(mesh, lambda p: (p + offset).astype(np.float32).astype(np.float64) - offset)
    return {"offset_mm": offset.tolist(), "distance_mm": severity * FAR_TRANSLATION_MM}
