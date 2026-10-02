from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, ClassVar, Literal

IR_VERSION = 0

Point = tuple[float, float, float]
Orientation = Literal["same", "reversed"]
Kind = Literal["transversal", "tangent"]


class IrError(ValueError):
    pass


def _pt(v: Any) -> Point:
    if len(v) != 3:
        raise IrError(f"point needs 3 components: {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _keys(d: Any, keys: set[str], what: str) -> None:
    if not isinstance(d, dict) or set(d) != keys:
        got = sorted(d) if isinstance(d, dict) else d
        raise IrError(f"{what}: expected keys {sorted(keys)}, got {got!r}")


@dataclass
class Tolerances:
    linear: float = 1e-3
    angular_snap_deg: float = 0.5
    tangent_threshold_deg: float = 3.0
    vertex_merge: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "linear": self.linear,
            "angular_snap_deg": self.angular_snap_deg,
            "tangent_threshold_deg": self.tangent_threshold_deg,
            "vertex_merge": self.vertex_merge,
        }

    @staticmethod
    def from_dict(d: dict) -> Tolerances:
        keys = {"linear", "angular_snap_deg", "tangent_threshold_deg", "vertex_merge"}
        _keys(d, keys, "tolerances")
        return Tolerances(**{k: float(v) for k, v in d.items()})


@dataclass
class Plane:
    type: ClassVar[str] = "plane"
    origin: Point
    normal: Point

    def to_dict(self) -> dict:
        return {"type": "plane", "origin": list(self.origin), "normal": list(self.normal)}


@dataclass
class Cylinder:
    type: ClassVar[str] = "cylinder"
    origin: Point
    axis: Point
    radius: float
    orientation: Orientation

    def to_dict(self) -> dict:
        return {
            "type": "cylinder",
            "origin": list(self.origin),
            "axis": list(self.axis),
            "radius": self.radius,
            "orientation": self.orientation,
        }


@dataclass
class Cone:
    type: ClassVar[str] = "cone"
    apex: Point
    axis: Point
    half_angle: float
    orientation: Orientation

    def to_dict(self) -> dict:
        return {
            "type": "cone",
            "apex": list(self.apex),
            "axis": list(self.axis),
            "half_angle": self.half_angle,
            "orientation": self.orientation,
        }


@dataclass
class Sphere:
    type: ClassVar[str] = "sphere"
    center: Point
    radius: float
    orientation: Orientation

    def to_dict(self) -> dict:
        return {
            "type": "sphere",
            "center": list(self.center),
            "radius": self.radius,
            "orientation": self.orientation,
        }


@dataclass
class Torus:
    type: ClassVar[str] = "torus"
    center: Point
    axis: Point
    major_radius: float
    minor_radius: float
    orientation: Orientation

    def to_dict(self) -> dict:
        return {
            "type": "torus",
            "center": list(self.center),
            "axis": list(self.axis),
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "orientation": self.orientation,
        }


@dataclass
class Facets:
    type: ClassVar[str] = "facets"
    vertices: list[Point]
    faces: list[tuple[int, int, int]]

    def to_dict(self) -> dict:
        return {
            "type": "facets",
            "vertices": [list(v) for v in self.vertices],
            "faces": [list(f) for f in self.faces],
        }


Surface = Plane | Cylinder | Cone | Sphere | Torus | Facets


def _orientation(v: Any) -> Orientation:
    if v not in ("same", "reversed"):
        raise IrError(f"bad orientation {v!r}")
    return v


def surface_from_dict(d: dict) -> Surface:
    t = d.get("type") if isinstance(d, dict) else None
    body = {k: v for k, v in d.items() if k != "type"} if t else {}
    if t == "plane":
        _keys(body, {"origin", "normal"}, "plane")
        return Plane(_pt(body["origin"]), _pt(body["normal"]))
    if t == "cylinder":
        _keys(body, {"origin", "axis", "radius", "orientation"}, "cylinder")
        return Cylinder(
            _pt(body["origin"]),
            _pt(body["axis"]),
            float(body["radius"]),
            _orientation(body["orientation"]),
        )
    if t == "cone":
        _keys(body, {"apex", "axis", "half_angle", "orientation"}, "cone")
        return Cone(
            _pt(body["apex"]),
            _pt(body["axis"]),
            float(body["half_angle"]),
            _orientation(body["orientation"]),
        )
    if t == "sphere":
        _keys(body, {"center", "radius", "orientation"}, "sphere")
        return Sphere(_pt(body["center"]), float(body["radius"]), _orientation(body["orientation"]))
    if t == "torus":
        _keys(body, {"center", "axis", "major_radius", "minor_radius", "orientation"}, "torus")
        return Torus(
            _pt(body["center"]),
            _pt(body["axis"]),
            float(body["major_radius"]),
            float(body["minor_radius"]),
            _orientation(body["orientation"]),
        )
    if t == "facets":
        _keys(body, {"vertices", "faces"}, "facets")
        return Facets(
            [_pt(v) for v in body["vertices"]],
            [(int(f[0]), int(f[1]), int(f[2])) for f in body["faces"]],
        )
    raise IrError(f"unknown surface type {t!r}")


@dataclass
class Residual:
    rms: float
    max: float


@dataclass
class Region:
    id: int
    surface: Surface
    triangles: list[int]
    residual: Residual | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "surface": self.surface.to_dict(),
            "triangles": list(self.triangles),
            "residual": None
            if self.residual is None
            else {"rms": self.residual.rms, "max": self.residual.max},
        }

    @staticmethod
    def from_dict(d: dict) -> Region:
        _keys(d, {"id", "surface", "triangles", "residual"}, "region")
        r = d["residual"]
        if r is not None:
            _keys(r, {"rms", "max"}, "residual")
        return Region(
            int(d["id"]),
            surface_from_dict(d["surface"]),
            [int(t) for t in d["triangles"]],
            None if r is None else Residual(float(r["rms"]), float(r["max"])),
        )


@dataclass
class Boundary:
    kind: Kind
    dihedral_deg: float
    closed: bool
    start_vertex: int | None
    end_vertex: int | None
    points: list[Point]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "dihedral_deg": self.dihedral_deg,
            "closed": self.closed,
            "start_vertex": self.start_vertex,
            "end_vertex": self.end_vertex,
            "points": [list(p) for p in self.points],
        }

    @staticmethod
    def from_dict(d: dict) -> Boundary:
        keys = {"kind", "dihedral_deg", "closed", "start_vertex", "end_vertex", "points"}
        _keys(d, keys, "boundary")
        if d["kind"] not in ("transversal", "tangent"):
            raise IrError(f"bad kind {d['kind']!r}")
        if not isinstance(d["closed"], bool):
            raise IrError("closed must be a bool")
        sv, ev = d["start_vertex"], d["end_vertex"]
        return Boundary(
            d["kind"],
            float(d["dihedral_deg"]),
            d["closed"],
            None if sv is None else int(sv),
            None if ev is None else int(ev),
            [_pt(p) for p in d["points"]],
        )


@dataclass
class Adjacency:
    regions: tuple[int, int]
    boundaries: list[Boundary]

    def to_dict(self) -> dict:
        return {
            "regions": list(self.regions),
            "boundaries": [b.to_dict() for b in self.boundaries],
        }

    @staticmethod
    def from_dict(d: dict) -> Adjacency:
        _keys(d, {"regions", "boundaries"}, "adjacency")
        a, b = d["regions"]
        return Adjacency((int(a), int(b)), [Boundary.from_dict(x) for x in d["boundaries"]])


@dataclass
class Vertex:
    id: int
    position: Point
    regions: list[int]
    source_positions: list[Point]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": list(self.position),
            "regions": list(self.regions),
            "source_positions": [list(p) for p in self.source_positions],
        }

    @staticmethod
    def from_dict(d: dict) -> Vertex:
        _keys(d, {"id", "position", "regions", "source_positions"}, "vertex")
        return Vertex(
            int(d["id"]),
            _pt(d["position"]),
            [int(r) for r in d["regions"]],
            [_pt(p) for p in d["source_positions"]],
        )


@dataclass
class Shell:
    closed: bool
    regions: list[int]

    def to_dict(self) -> dict:
        return {"closed": self.closed, "regions": list(self.regions)}

    @staticmethod
    def from_dict(d: dict) -> Shell:
        _keys(d, {"closed", "regions"}, "shell")
        if not isinstance(d["closed"], bool):
            raise IrError("closed must be a bool")
        return Shell(d["closed"], [int(r) for r in d["regions"]])


@dataclass
class Ir:
    tolerances: Tolerances
    shells: list[Shell]
    regions: list[Region]
    adjacencies: list[Adjacency]
    vertices: list[Vertex]
    ir_version: int = field(default=IR_VERSION)

    def to_dict(self) -> dict:
        return {
            "ir_version": self.ir_version,
            "tolerances": self.tolerances.to_dict(),
            "shells": [s.to_dict() for s in self.shells],
            "regions": [r.to_dict() for r in self.regions],
            "adjacencies": [a.to_dict() for a in self.adjacencies],
            "vertices": [v.to_dict() for v in self.vertices],
        }

    @staticmethod
    def from_dict(d: dict) -> Ir:
        keys = {"ir_version", "tolerances", "shells", "regions", "adjacencies", "vertices"}
        _keys(d, keys, "ir")
        ir = Ir(
            Tolerances.from_dict(d["tolerances"]),
            [Shell.from_dict(s) for s in d["shells"]],
            [Region.from_dict(r) for r in d["regions"]],
            [Adjacency.from_dict(a) for a in d["adjacencies"]],
            [Vertex.from_dict(v) for v in d["vertices"]],
            int(d["ir_version"]),
        )
        ir.validate()
        return ir

    @staticmethod
    def loads(text: str) -> Ir:
        return Ir.from_dict(json.loads(text))

    def dumps(self) -> str:
        return canonical_json(self.to_dict())

    def validate(self) -> None:
        errors = validate(self)
        if errors:
            raise IrError("invalid IR: " + "; ".join(errors))


def format_float(x: float) -> str:
    if not math.isfinite(x):
        raise IrError("non-finite number")
    if x == 0.0:
        return "0.0"
    sign, digits, exp10 = Decimal(repr(x)).normalize().as_tuple()
    ds = "".join(map(str, digits))
    n = len(ds)
    e = int(exp10) + n - 1
    if 0 <= e < 16:
        body = ds + "0" * (e + 1 - n) + ".0" if n <= e + 1 else ds[: e + 1] + "." + ds[e + 1 :]
    elif -5 <= e < 0:
        body = "0." + "0" * (-e - 1) + ds
    else:
        body = f"{ds}e{e}" if n == 1 else f"{ds[0]}.{ds[1:]}e{e}"
    return ("-" if sign else "") + body


def _write_string(s: str) -> str:
    out = ['"']
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif " " <= c <= "~":
            out.append(c)
        else:
            b = c.encode("utf-16-be")
            for i in range(0, len(b), 2):
                out.append(f"\\u{int.from_bytes(b[i : i + 2], 'big'):04x}")
    out.append('"')
    return "".join(out)


def canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, str):
        return _write_string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        items = (f"{_write_string(k)}:{canonical_json(value[k])}" for k in sorted(value))
        return "{" + ",".join(items) + "}"
    raise IrError(f"cannot serialize {type(value).__name__}")


def _is_unit(v: Point) -> bool:
    return abs(math.hypot(*v) - 1.0) < 1e-9


def validate(ir: Ir) -> list[str]:
    e: list[str] = []
    if ir.ir_version != IR_VERSION:
        e.append(f"ir_version {ir.ir_version} is not {IR_VERSION}")
    n_regions = len(ir.regions)
    shell_of: dict[int, int] = {}
    for si, shell in enumerate(ir.shells):
        for r in shell.regions:
            if not 0 <= r < n_regions:
                e.append(f"shell {si} names missing region {r}")
            elif r in shell_of:
                e.append(f"region {r} is in more than one shell")
            else:
                shell_of[r] = si
        if not shell.closed:
            ok = (
                len(shell.regions) == 1
                and 0 <= shell.regions[0] < n_regions
                and isinstance(ir.regions[shell.regions[0]].surface, Facets)
            )
            if not ok:
                e.append(f"open shell {si} must be exactly one facets region")
    seen: set[int] = set()
    for i, region in enumerate(ir.regions):
        if region.id != i:
            e.append(f"region at index {i} has id {region.id}")
        if i not in shell_of:
            e.append(f"region {i} is in no shell")
        for t in region.triangles:
            if t in seen:
                e.append(f"source triangle {t} appears in more than one place")
            seen.add(t)
        s = region.surface
        if isinstance(s, Plane):
            if not _is_unit(s.normal):
                e.append(f"region {i}: plane normal is not a unit vector")
        elif isinstance(s, Cylinder):
            if not _is_unit(s.axis) or s.radius <= 0:
                e.append(f"region {i}: bad cylinder axis or radius")
        elif isinstance(s, Cone):
            if not _is_unit(s.axis) or not 0 < s.half_angle < math.pi / 2:
                e.append(f"region {i}: bad cone axis or half_angle")
        elif isinstance(s, Sphere):
            if s.radius <= 0:
                e.append(f"region {i}: bad sphere radius")
        elif isinstance(s, Torus):
            if not _is_unit(s.axis) or s.minor_radius <= 0 or s.major_radius <= s.minor_radius:
                e.append(f"region {i}: bad torus axis or radii")
        else:
            if len(s.faces) != len(region.triangles):
                e.append(f"region {i}: facets faces and triangles differ in length")
            if any(not 0 <= v < len(s.vertices) for f in s.faces for v in f):
                e.append(f"region {i}: facets face index out of range")
            if region.residual is not None:
                e.append(f"region {i}: facets region must have null residual")
        if not isinstance(s, Facets) and region.residual is None:
            e.append(f"region {i}: analytic region needs a residual")
    pairs: set[tuple[int, int]] = set()
    ends: dict[int, set[int]] = {}
    for adj in ir.adjacencies:
        a, b = adj.regions
        if not a < b < n_regions:
            e.append(f"adjacency ({a}, {b}) must satisfy a < b < region count")
            continue
        if (a, b) in pairs:
            e.append(f"adjacency ({a}, {b}) listed twice")
        pairs.add((a, b))
        if shell_of.get(a) != shell_of.get(b):
            e.append(f"adjacency ({a}, {b}) crosses shells")
        if isinstance(ir.regions[a].surface, Facets) and isinstance(ir.regions[b].surface, Facets):
            e.append(f"adjacency ({a}, {b}) joins two facets regions")
        if not adj.boundaries:
            e.append(f"adjacency ({a}, {b}) has no boundaries")
        for bd in adj.boundaries:
            if len(bd.points) < (3 if bd.closed else 2):
                e.append(f"adjacency ({a}, {b}): boundary too short")
                continue
            if not 0.0 <= bd.dihedral_deg <= 180.0:
                e.append(f"adjacency ({a}, {b}): dihedral_deg out of range")
            if (bd.dihedral_deg < ir.tolerances.tangent_threshold_deg) != (bd.kind == "tangent"):
                e.append(f"adjacency ({a}, {b}): kind disagrees with threshold")
            if bd.closed:
                if bd.start_vertex is not None or bd.end_vertex is not None:
                    e.append(f"adjacency ({a}, {b}): closed boundary names vertices")
                continue
            if bd.start_vertex is None or bd.end_vertex is None:
                e.append(f"adjacency ({a}, {b}): open boundary needs both vertices")
                continue
            for v, p in ((bd.start_vertex, bd.points[0]), (bd.end_vertex, bd.points[-1])):
                if not 0 <= v < len(ir.vertices):
                    e.append(f"adjacency ({a}, {b}): missing vertex {v}")
                    continue
                if math.dist(ir.vertices[v].position, p) > ir.tolerances.vertex_merge:
                    e.append(f"adjacency ({a}, {b}): endpoint is not at vertex {v}")
                ends.setdefault(v, set()).update((a, b))
    for i, v in enumerate(ir.vertices):
        if v.id != i:
            e.append(f"vertex at index {i} has id {v.id}")
        if (
            len(v.regions) < 3
            or any(x >= y for x, y in zip(v.regions, v.regions[1:], strict=False))
            or any(not 0 <= r < n_regions for r in v.regions)
        ):
            e.append(f"vertex {i}: regions must be sorted, distinct, >= 3")
        if not v.source_positions:
            e.append(f"vertex {i}: no source positions")
        if ends.get(v.id) != set(v.regions):
            e.append(f"vertex {i}: regions differ from those whose boundaries end here")
    return e
