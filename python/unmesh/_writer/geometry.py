from __future__ import annotations

import math

import numpy as np

from unmesh.ir import Cone, Cylinder, Facets, Plane, Sphere, Surface, Torus

PERIODIC = (Cylinder, Cone, Sphere, Torus)
REFINE_ITERATIONS = 50
REFINE_PULL = 1e-8
REFINE_STEP = 1e-13


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def perpendicular(axis) -> np.ndarray:
    axis = unit(axis)
    for ref in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        x = np.asarray(ref) - (np.asarray(ref) @ axis) * axis
        if np.linalg.norm(x) > 0.5:
            return unit(x)
    return unit(np.cross(axis, (0.0, 0.0, 1.0)))


def axis_of(surface: Surface) -> np.ndarray | None:
    if isinstance(surface, (Cylinder, Cone, Torus)):
        return unit(surface.axis)
    return None


def origin_of(surface: Surface) -> np.ndarray:
    if isinstance(surface, Cone):
        return np.asarray(surface.apex, dtype=float)
    if isinstance(surface, (Sphere, Torus)):
        return np.asarray(surface.center, dtype=float)
    return np.asarray(surface.origin, dtype=float)


def _radial(p: np.ndarray, origin: np.ndarray, axis: np.ndarray):
    q = p - origin
    h = q @ axis
    r = q - np.outer(h, axis) if q.ndim == 2 else q - h * axis
    rho = np.linalg.norm(r, axis=-1)
    return h, r, rho


def closest(surface: Surface, p) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    if isinstance(surface, Plane):
        n = unit(surface.normal)
        o = np.asarray(surface.origin, dtype=float)
        return p - ((p - o) @ n) * n, n
    if isinstance(surface, Cylinder):
        a = unit(surface.axis)
        o = np.asarray(surface.origin, dtype=float)
        h, r, rho = _radial(p, o, a)
        radial = r / rho if rho > 0 else perpendicular(a)
        return o + h * a + surface.radius * radial, radial
    if isinstance(surface, Cone):
        a = unit(surface.axis)
        o = np.asarray(surface.apex, dtype=float)
        h, r, rho = _radial(p, o, a)
        radial = r / rho if rho > 0 else perpendicular(a)
        ca, sa = math.cos(surface.half_angle), math.sin(surface.half_angle)
        t = max(h * ca + rho * sa, 0.0)
        return o + t * ca * a + t * sa * radial, ca * radial - sa * a
    if isinstance(surface, Sphere):
        c = np.asarray(surface.center, dtype=float)
        d = unit(p - c) if np.linalg.norm(p - c) > 0 else np.array([1.0, 0.0, 0.0])
        return c + surface.radius * d, d
    if isinstance(surface, Torus):
        a = unit(surface.axis)
        c = np.asarray(surface.center, dtype=float)
        _, r, rho = _radial(p, c, a)
        radial = r / rho if rho > 0 else perpendicular(a)
        ring = c + surface.major_radius * radial
        d = unit(p - ring) if np.linalg.norm(p - ring) > 0 else radial
        return ring + surface.minor_radius * d, d
    raise TypeError(f"no closest point on {surface.type}")


def distance(surface: Surface, p) -> float:
    q, _ = closest(surface, p)
    return float(np.linalg.norm(np.asarray(p, dtype=float) - q))


def outward(surface: Surface, p) -> np.ndarray:
    _, n = closest(surface, p)
    if not isinstance(surface, Plane) and surface.orientation == "reversed":
        n = -n
    return n


def refine(surfaces: list[Surface], p0) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    x = p0.copy()
    for _ in range(REFINE_ITERATIONS):
        rows, rhs = [], []
        for s in surfaces:
            q, n = closest(s, x)
            rows.append(n)
            rhs.append(n @ q)
        n = np.array(rows)
        d = np.array(rhs)
        y = np.linalg.solve(n.T @ n + REFINE_PULL * np.eye(3), n.T @ d + REFINE_PULL * p0)
        step = float(np.linalg.norm(y - x))
        x = y
        if step <= REFINE_STEP * max(1.0, float(np.linalg.norm(x))):
            break
    return x


def is_analytic(surface: Surface) -> bool:
    return not isinstance(surface, Facets)


def is_curved(surface: Surface) -> bool:
    return isinstance(surface, PERIODIC)


def angle_about(p, origin, axis, xdir) -> float:
    q = np.asarray(p, dtype=float) - origin
    y = np.cross(axis, xdir)
    return math.atan2(float(q @ y), float(q @ xdir))


def winding(points, origin, axis, xdir) -> int:
    pts = np.asarray(points, dtype=float)
    q = pts - origin
    _, _, rho = _radial(pts, origin, axis)
    q = q[rho > 1e-9 * np.maximum(1.0, np.linalg.norm(q, axis=1))]
    y = np.cross(axis, xdir)
    ang = np.arctan2(q @ y, q @ xdir)
    d = np.diff(np.append(ang, ang[0]))
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return int(round(float(d.sum()) / (2 * math.pi)))


def tube_winding(points, center, axis, major) -> int:
    pts = np.asarray(points, dtype=float)
    h, r, rho = _radial(pts, center, axis)
    ang = np.arctan2(h, rho - major)
    d = np.diff(np.append(ang, ang[0]))
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return int(round(float(d.sum()) / (2 * math.pi)))


def plane_normal(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    return unit(vt[2])
