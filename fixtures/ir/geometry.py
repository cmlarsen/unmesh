from __future__ import annotations

import math

import numpy as np

from unmesh.ir import Cone, Cylinder, Facets, Plane, Sphere, Surface


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _sign(surface: Surface) -> float:
    return -1.0 if getattr(surface, "orientation", "same") == "reversed" else 1.0


def _radial(p, origin, axis):
    q = np.asarray(p, dtype=float) - np.asarray(origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    h = float(q @ axis)
    r = q - h * axis
    return q, h, r, float(np.linalg.norm(r))


def _closest_on_triangle(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _facets_normal(surface: Facets, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    v = np.asarray(surface.vertices, dtype=float)
    best, best_d = None, math.inf
    for f in surface.faces:
        a, b, c = v[list(f)]
        d = float(np.linalg.norm(_closest_on_triangle(p, a, b, c) - p))
        if d < best_d - 1e-12:
            best, best_d = (a, b, c), d
    a, b, c = best
    return _unit(np.cross(b - a, c - a))


def normal(surface: Surface, p) -> np.ndarray:
    if isinstance(surface, Plane):
        return _unit(surface.normal)
    if isinstance(surface, Facets):
        return _facets_normal(surface, p)
    s = _sign(surface)
    if isinstance(surface, Cylinder):
        _, _, r, n = _radial(p, surface.origin, surface.axis)
        return s * r / n
    if isinstance(surface, Cone):
        _, _, r, n = _radial(p, surface.apex, surface.axis)
        a = surface.half_angle
        return s * (math.cos(a) * r / n - math.sin(a) * np.asarray(surface.axis))
    if isinstance(surface, Sphere):
        d = np.asarray(p, dtype=float) - np.asarray(surface.center, dtype=float)
        return s * d / np.linalg.norm(d)
    q, h, r, n = _radial(p, surface.center, surface.axis)
    c = surface.major_radius * r / n
    d = q - c
    return s * d / np.linalg.norm(d)


def distance(surface: Surface, p) -> float:
    p = np.asarray(p, dtype=float)
    if isinstance(surface, Plane):
        return abs(float((p - np.asarray(surface.origin)) @ _unit(surface.normal)))
    if isinstance(surface, Cylinder):
        return abs(_radial(p, surface.origin, surface.axis)[3] - surface.radius)
    if isinstance(surface, Cone):
        _, h, _, rho = _radial(p, surface.apex, surface.axis)
        a = surface.half_angle
        return abs(rho * math.cos(a) - h * math.sin(a))
    if isinstance(surface, Sphere):
        return abs(float(np.linalg.norm(p - np.asarray(surface.center))) - surface.radius)
    _, h, _, rho = _radial(p, surface.center, surface.axis)
    return abs(math.hypot(rho - surface.major_radius, h) - surface.minor_radius)


def signed_distance(surface: Surface, p) -> float:
    p = np.asarray(p, dtype=float)
    s = _sign(surface)
    if isinstance(surface, Plane):
        return float((p - np.asarray(surface.origin)) @ _unit(surface.normal))
    if isinstance(surface, Cylinder):
        return s * (_radial(p, surface.origin, surface.axis)[3] - surface.radius)
    if isinstance(surface, Cone):
        _, h, _, rho = _radial(p, surface.apex, surface.axis)
        a = surface.half_angle
        return s * (rho * math.cos(a) - h * math.sin(a))
    if isinstance(surface, Sphere):
        return s * (float(np.linalg.norm(p - np.asarray(surface.center))) - surface.radius)
    _, h, _, rho = _radial(p, surface.center, surface.axis)
    return s * (math.hypot(rho - surface.major_radius, h) - surface.minor_radius)


def project(surface: Surface, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    for _ in range(3):
        p = p - signed_distance(surface, p) * normal(surface, p)
    return p


def dihedral_deg(a: Surface, b: Surface, points) -> float:
    angles = []
    for p in points:
        c = float(np.clip(normal(a, p) @ normal(b, p), -1.0, 1.0))
        angles.append(math.degrees(math.acos(c)))
    return float(np.median(angles))


def sample_points(points, closed: bool):
    if len(points) == 2 and not closed:
        return [tuple((np.asarray(points[0]) + np.asarray(points[1])) / 2)]
    return list(points)
