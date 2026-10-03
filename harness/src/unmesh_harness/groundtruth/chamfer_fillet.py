from __future__ import annotations

from build123d import Axis, Box, Cylinder, GeomType, Pos, Shape, chamfer, fillet
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Sphere

from .core import register, validity_problems


def _r(x: float) -> float:
    return round(float(x), 3)


def _box(length: float, width: float, height: float) -> Shape:
    return Pos(0, 0, height / 2) * Box(length, width, height)


def _disc(radius: float, height: float) -> Shape:
    return Pos(0, 0, height / 2) * Cylinder(radius, height)


def _ok(solid: Shape, faces: int) -> bool:
    return validity_problems(solid) == [] and len(solid.faces()) == faces


@register("planar_chamfer")
def planar_chamfer(rng):
    c = _r(rng.uniform(0.2, 10.0))
    length = _r(rng.uniform(max(40.0, 6 * c), 120.0))
    width = _r(rng.uniform(max(30.0, 6 * c), 100.0))
    height = _r(rng.uniform(max(8.0, 4 * c), max(30.0, 4 * c + 5)))
    c = min(c, _r(0.2 * min(height, length / 2, width / 2)))
    solid = _box(length, width, height)
    for _ in range(8):
        try:
            candidate = chamfer(solid.edges().group_by(Axis.Z)[-1], length=c)
        except Exception:
            c = _r(c / 2)
            continue
        if _ok(candidate, 10):
            solid = candidate
            break
        c = _r(c / 2)
    else:
        raise RuntimeError("planar_chamfer: no valid build after 8 attempts")
    features = [{"type": "top_chamfer", "length": c, "edges": 4, "axis": [0, 0, 1]}]
    return solid, {"box": [length, width, height], "chamfer": c}, features


@register("straight_fillet")
def straight_fillet(rng):
    r = _r(rng.uniform(0.2, 10.0))
    length = _r(rng.uniform(max(40.0, 5 * r), 120.0))
    width = _r(rng.uniform(max(30.0, 5 * r), 100.0))
    height = _r(rng.uniform(8.0, 30.0))
    r = min(r, _r(0.2 * min(length, width)))
    solid = _box(length, width, height)
    for _ in range(8):
        try:
            candidate = fillet(solid.edges().filter_by(Axis.Z), radius=r)
        except Exception:
            r = _r(r / 2)
            continue
        if _ok(candidate, 10):
            solid = candidate
            break
        r = _r(r / 2)
    else:
        raise RuntimeError("straight_fillet: no valid build after 8 attempts")
    features = [{"type": "vertical_fillet", "radius": r, "edges": 4, "axis": [0, 0, 1]}]
    return solid, {"box": [length, width, height], "radius": r}, features


@register("circular_fillet")
def circular_fillet(rng):
    r = _r(rng.uniform(0.2, 10.0))
    radius = _r(rng.uniform(max(15.0, 4 * r), 60.0))
    height = _r(rng.uniform(max(10.0, 3 * r), 40.0))
    r = min(r, _r(0.25 * min(radius, height)))
    solid = _disc(radius, height)
    for _ in range(8):
        try:
            candidate = fillet(solid.edges().group_by(Axis.Z)[-1], radius=r)
        except Exception:
            r = _r(r / 2)
            continue
        if _ok(candidate, 4):
            solid = candidate
            break
        r = _r(r / 2)
    else:
        raise RuntimeError("circular_fillet: no valid build after 8 attempts")
    features = [{"type": "top_edge_fillet", "radius": r, "edges": 1, "axis": [0, 0, 1]}]
    return solid, {"disc": [radius, height], "fillet": r}, features


def _bore_solid(length, width, height, bore):
    plate = _box(length, width, height)
    tool = Pos(0, 0, height / 2) * Cylinder(bore, height + 2)
    return (plate - tool).clean()


@register("bore_chamfer")
def bore_chamfer(rng):
    if rng.random() < 0.5:
        c = _r(rng.uniform(0.2, 10.0))
        bore = _r(rng.uniform(6.0, 18.0))
        length = _r(rng.uniform(max(60.0, 6 * bore), 160.0))
        width = _r(rng.uniform(max(50.0, 6 * bore), 140.0))
        height = _r(rng.uniform(12.0, 30.0))
        c = min(c, _r(0.25 * min(bore, height, min(length, width) / 2 - bore)))
        solid = _bore_solid(length, width, height, bore)
        for _ in range(8):
            try:
                edges = solid.edges().filter_by(GeomType.CIRCLE).group_by(Axis.Z)[-1]
                candidate = chamfer(edges, length=c)
            except Exception:
                c = _r(c / 2)
                continue
            if _ok(candidate, 8):
                solid = candidate
                break
            c = _r(c / 2)
        else:
            raise RuntimeError("bore_chamfer: no valid build after 8 attempts")
        features = [
            {"type": "bore_chamfer", "bore": bore, "length": c, "edges": 1, "axis": [0, 0, 1]}
        ]
        params = {
            "variant": "bore",
            "plate": [length, width, height],
            "bore": bore,
            "chamfer": c,
        }
    else:
        c = _r(rng.uniform(0.2, 10.0))
        radius = _r(rng.uniform(max(20.0, 4 * c), 60.0))
        height = _r(rng.uniform(max(12.0, 3 * c), 40.0))
        c = min(c, _r(0.25 * min(radius, height)))
        solid = _disc(radius, height)
        for _ in range(8):
            try:
                edges = solid.edges().filter_by(GeomType.CIRCLE).group_by(Axis.Z)[-1]
                candidate = chamfer(edges, length=c)
            except Exception:
                c = _r(c / 2)
                continue
            if _ok(candidate, 4):
                solid = candidate
                break
            c = _r(c / 2)
        else:
            raise RuntimeError("bore_chamfer: no valid build after 8 attempts")
        features = [{"type": "disc_edge_chamfer", "length": c, "edges": 1, "axis": [0, 0, 1]}]
        params = {"variant": "disc", "disc": [radius, height], "chamfer": c}
    return solid, params, features


@register("corner_fillet")
def corner_fillet(rng):
    r = _r(rng.uniform(0.2, 10.0))
    a = _r(rng.uniform(max(20.0, 5 * r), 80.0))
    b = _r(rng.uniform(max(20.0, 5 * r), 80.0))
    c = _r(rng.uniform(max(20.0, 5 * r), 80.0))
    r = min(r, _r(0.2 * min(a, b, c)))
    solid = _box(a, b, c)
    for _ in range(8):
        try:
            candidate = fillet(solid.edges(), radius=r)
        except Exception:
            r = _r(r / 2)
            continue
        if _ok(candidate, 26):
            solid = candidate
            break
        r = _r(r / 2)
    else:
        raise RuntimeError("corner_fillet: no valid build after 8 attempts")
    tags = {}
    for i, face in enumerate(solid.faces()):
        if BRepAdaptor_Surface(face.wrapped).GetType() == GeomAbs_Sphere:
            tags[str(i)] = "corner_blend"
    gt_params = {"box": [a, b, c], "radius": r}
    features = [{"type": "edge_fillet", "radius": r, "edges": 12, "axis": [0, 0, 1]}]
    return solid, gt_params, features, tags
