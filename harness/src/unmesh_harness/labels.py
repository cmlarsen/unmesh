from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from build123d import Shape
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepClass3d import BRepClass3d
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (
    GeomAbs_BezierCurve,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Line,
    GeomAbs_OffsetSurface,
    GeomAbs_Parabola,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

TANGENT_THRESHOLD_DEG = 3.0

DEFLECTION_SETTINGS = ((0.1, 0.5), (0.01, 0.2), (0.001, 0.1))

_SURFACE_TYPES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset",
}

_CURVE_TYPES = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_Hyperbola: "hyperbola",
    GeomAbs_Parabola: "parabola",
    GeomAbs_BezierCurve: "bezier",
    GeomAbs_BSplineCurve: "bspline",
}


@dataclass
class FaceInfo:
    id: int
    surface: str
    params: dict[str, Any]
    reversed: bool


@dataclass
class EdgeAdjacency:
    edge_id: int
    face_a: int
    face_b: int
    curve: str
    tangent: bool
    dihedral: float
    dihedral_min: float
    dihedral_max: float
    points: list[list[float]]
    start_vertex: int
    end_vertex: int
    forward_in_a: bool
    dihedral_samples: list[float] = field(default_factory=list)


@dataclass
class ShellInfo:
    role: str
    parent: int | None
    closed: bool
    faces: list[int]


@dataclass
class LabeledMesh:
    tris: np.ndarray
    face_id: np.ndarray
    faces: list[FaceInfo]
    adjacency: list[EdgeAdjacency]
    linear_deflection: float
    angular_deflection: float
    vertices: list[list[float]] = field(default_factory=list)
    shells: list[ShellInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            tris=self.tris,
            face_id=self.face_id,
            faces=json.dumps([f.__dict__ for f in self.faces]),
            adjacency=json.dumps([a.__dict__ for a in self.adjacency]),
            vertices=json.dumps(self.vertices),
            shells=json.dumps([x.__dict__ for x in self.shells]),
            deflection=np.array([self.linear_deflection, self.angular_deflection]),
            metadata=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> LabeledMesh:
        with np.load(path, allow_pickle=False) as z:
            lin, ang = (float(v) for v in z["deflection"])
            return cls(
                tris=z["tris"],
                face_id=z["face_id"],
                faces=[FaceInfo(**f) for f in json.loads(str(z["faces"]))],
                adjacency=[EdgeAdjacency(**a) for a in json.loads(str(z["adjacency"]))],
                linear_deflection=lin,
                angular_deflection=ang,
                vertices=json.loads(str(z["vertices"])),
                shells=[ShellInfo(**x) for x in json.loads(str(z["shells"]))],
                metadata=json.loads(str(z["metadata"])) if "metadata" in z.files else {},
            )

    def write_stl(self, path: str | Path) -> None:
        from unmesh import write_stl

        write_stl(path, self.tris)

    def face_tris(self, face: int) -> np.ndarray:
        return self.tris[self.face_id == face]


def _vec(v) -> list[float]:
    return [v.X(), v.Y(), v.Z()]


def _surface_params(face, adaptor: BRepAdaptor_Surface) -> tuple[str, dict[str, Any], bool]:
    kind = adaptor.GetType()
    name = _SURFACE_TYPES.get(kind, "other")
    reversed_face = face.Orientation() == TopAbs_REVERSED
    if kind == GeomAbs_Plane:
        pln = adaptor.Plane()
        pos = pln.Position()
        natural = np.cross(_vec(pos.XDirection()), _vec(pos.YDirection()))
        flip = reversed_face
        return (
            name,
            {
                "origin": _vec(pln.Location()),
                "normal": (-natural if flip else natural).tolist(),
            },
            flip,
        )
    position = {
        GeomAbs_Cylinder: lambda: adaptor.Cylinder().Position(),
        GeomAbs_Cone: lambda: adaptor.Cone().Position(),
        GeomAbs_Sphere: lambda: adaptor.Sphere().Position(),
        GeomAbs_Torus: lambda: adaptor.Torus().Position(),
    }.get(kind)
    flip = reversed_face != (position is not None and not position().Direct())
    name, params = _analytic_params(kind, name, adaptor)
    return name, params, flip


def _analytic_params(kind, name, adaptor) -> tuple[str, dict[str, Any]]:
    if kind == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        return name, {
            "origin": _vec(cyl.Location()),
            "axis": _vec(cyl.Axis().Direction()),
            "radius": cyl.Radius(),
        }
    if kind == GeomAbs_Cone:
        cone = adaptor.Cone()
        axis = _vec(cone.Axis().Direction())
        return name, {
            "apex": _vec(cone.Apex()),
            "axis": [-c for c in axis] if cone.SemiAngle() < 0 else axis,
            "half_angle": abs(cone.SemiAngle()),
        }
    if kind == GeomAbs_Sphere:
        sph = adaptor.Sphere()
        return name, {"center": _vec(sph.Location()), "radius": sph.Radius()}
    if kind == GeomAbs_Torus:
        tor = adaptor.Torus()
        return name, {
            "center": _vec(tor.Location()),
            "axis": _vec(tor.Axis().Direction()),
            "major_radius": tor.MajorRadius(),
            "minor_radius": tor.MinorRadius(),
        }
    return name, {}


def _face_normal(face, edge, t: float) -> np.ndarray | None:
    curve2d = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 1.0)
    if curve2d is None:
        return None
    uv = curve2d.Value(t)
    surf = BRep_Tool.Surface_s(face)
    props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-9)
    if not props.IsNormalDefined():
        return None
    n = np.array(_vec(props.Normal()))
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _edge_dihedral(edge, face_a, face_b, samples: int = 16) -> tuple[float, float, float]:
    first, last = BRep_Tool.Range_s(edge, face_a)
    angles = []
    for k in range(samples):
        t = first + (last - first) * (k + 0.5) / samples
        na, nb = _face_normal(face_a, edge, t), _face_normal(face_b, edge, t)
        if na is None or nb is None:
            continue
        cos = float(np.clip(na @ nb, -1.0, 1.0))
        angles.append(math.acos(cos))
    if not angles:
        raise RuntimeError("no surface normals could be sampled along an edge")
    return float(np.median(angles)), min(angles), max(angles)


def _polygon_params(poly, count: int) -> list[float] | None:
    if not poly.HasParameters() or poly.NbNodes() != count:
        return None
    try:
        return [poly.Parameter(i) for i in range(1, count + 1)]
    except Exception:
        return None


def _edge_dihedral_profile(edge, face_a, face_b, params: list[float]) -> list[float] | None:
    samplers = []
    for face in (face_a, face_b):
        curve2d = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 1.0)
        if curve2d is None:
            return None
        samplers.append(
            (
                curve2d,
                BRep_Tool.Surface_s(face),
                face.Orientation() == TopAbs_REVERSED,
            )
        )
    angles = []
    try:
        for t in params:
            normals = []
            for curve2d, surf, reversed_face in samplers:
                uv = curve2d.Value(t)
                props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-9)
                if not props.IsNormalDefined():
                    return None
                n = np.array(_vec(props.Normal()))
                normals.append(-n if reversed_face else n)
            cos = float(np.clip(normals[0] @ normals[1], -1.0, 1.0))
            angles.append(math.acos(cos))
    except Exception:
        return None
    return angles


def _forward_in_face(face, edge) -> bool:
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        cur = exp.Current()
        if cur.IsSame(edge):
            return cur.Orientation() == TopAbs_FORWARD
        exp.Next()
    raise RuntimeError("edge not in face")


def _indexed(shape, kind) -> TopTools_IndexedMapOfShape:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def face_surface_types(shape: Shape | Any) -> list[str]:
    wrapped = getattr(shape, "wrapped", shape)
    faces = _indexed(wrapped, TopAbs_FACE)
    out = []
    for fi in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(fi))
        out.append(_SURFACE_TYPES.get(BRepAdaptor_Surface(face).GetType(), "other"))
    return out


def _vertex_point(vertex) -> tuple[float, float, float]:
    p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertex))
    return (p.X(), p.Y(), p.Z())


def _point(p, trsf) -> tuple[float, float, float]:
    q = p.Transformed(trsf)
    return (q.X(), q.Y(), q.Z())


def tessellate(
    shape: Shape | Any, linear_deflection: float, angular_deflection: float
) -> LabeledMesh:
    wrapped = BRepBuilderAPI_Copy(getattr(shape, "wrapped", shape)).Shape()
    BRepMesh_IncrementalMesh(wrapped, linear_deflection, False, angular_deflection, True)

    face_map = _indexed(wrapped, TopAbs_FACE)
    edge_map = _indexed(wrapped, TopAbs_EDGE)
    vertex_map = _indexed(wrapped, TopAbs_VERTEX)

    canonical: dict[tuple, tuple[float, float, float]] = {}
    polylines: dict[int, list[tuple[float, float, float]]] = {}
    polyparams: dict[int, list[float] | None] = {}
    tri_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    faces: list[FaceInfo] = []

    for fi in range(1, face_map.Extent() + 1):
        face = TopoDS.Face_s(face_map.FindKey(fi))
        adaptor = BRepAdaptor_Surface(face)
        name, params, orientation_reversed = _surface_params(face, adaptor)
        flipped = face.Orientation() == TopAbs_REVERSED
        faces.append(FaceInfo(fi - 1, name, params, orientation_reversed))

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            raise RuntimeError(f"face {fi - 1} has no triangulation")
        trsf = loc.Transformation()
        nodes = [_point(tri.Node(i), trsf) for i in range(1, tri.NbNodes() + 1)]

        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            exp.Next()
            poly = BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
            if poly is None:
                continue
            idx = list(poly.Nodes())
            eid = edge_map.FindIndex(edge)
            v_first = TopExp.FirstVertex_s(edge)
            v_last = TopExp.LastVertex_s(edge)
            last = len(idx) - 1
            line = []
            for k, node in enumerate(idx):
                if k == 0 and not v_first.IsNull():
                    key = ("v", vertex_map.FindIndex(v_first))
                elif k == last and not v_last.IsNull():
                    key = ("v", vertex_map.FindIndex(v_last))
                else:
                    key = ("e", eid, k)
                nodes[node - 1] = canonical.setdefault(key, nodes[node - 1])
                line.append(nodes[node - 1])
            if eid not in polylines:
                polylines[eid] = line
                polyparams[eid] = _polygon_params(poly, len(idx))

        pts = np.array(nodes, dtype=np.float64)
        t = np.array(
            [
                [
                    tri.Triangle(j).Value(1) - 1,
                    tri.Triangle(j).Value(2) - 1,
                    tri.Triangle(j).Value(3) - 1,
                ]
                for j in range(1, tri.NbTriangles() + 1)
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        if flipped:
            t = t[:, [0, 2, 1]]
        tri_blocks.append(pts[t])
        id_blocks.append(np.full(len(t), fi - 1, dtype=np.int32))

    ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(wrapped, TopAbs_EDGE, TopAbs_FACE, ancestors)
    adjacency: list[EdgeAdjacency] = []
    for ei in range(1, edge_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_map.FindKey(ei))
        if BRep_Tool.Degenerated_s(edge):
            continue
        owners: list[Any] = []
        for f in ancestors.FindFromKey(edge):
            if not any(f.IsSame(o) for o in owners):
                owners.append(f)
        if len(owners) != 2:
            continue
        ia, ib = sorted(face_map.FindIndex(f) for f in owners)
        if ia == ib or ei not in polylines:
            continue
        fa, fb = (TopoDS.Face_s(face_map.FindKey(i)) for i in (ia, ib))
        dihedral, dihedral_min, dihedral_max = _edge_dihedral(edge, fa, fb)
        params = polyparams[ei]
        if params is None:
            first, last = BRep_Tool.Range_s(edge)
            n = len(polylines[ei])
            params = [first if n == 1 else first + (last - first) * i / (n - 1) for i in range(n)]
        profile = _edge_dihedral_profile(edge, fa, fb, params)
        samples: list[float] = []
        if profile is not None and len(profile) == len(polylines[ei]):
            samples = profile
            dihedral_min, dihedral_max = min(profile), max(profile)
        v_first = TopExp.FirstVertex_s(edge)
        v_last = TopExp.LastVertex_s(edge)
        adjacency.append(
            EdgeAdjacency(
                edge_id=ei - 1,
                face_a=ia - 1,
                face_b=ib - 1,
                curve=_CURVE_TYPES.get(BRepAdaptor_Curve(edge).GetType(), "other"),
                tangent=math.degrees(dihedral) < TANGENT_THRESHOLD_DEG,
                dihedral=dihedral,
                dihedral_min=dihedral_min,
                dihedral_max=dihedral_max,
                points=[list(p) for p in polylines[ei]],
                start_vertex=vertex_map.FindIndex(v_first) - 1,
                end_vertex=vertex_map.FindIndex(v_last) - 1,
                forward_in_a=_forward_in_face(fa, edge),
                dihedral_samples=samples,
            )
        )

    shells: list[ShellInfo] = []
    solids = TopExp_Explorer(wrapped, TopAbs_SOLID)
    while solids.More():
        solid = solids.Current()
        solids.Next()
        outer = BRepClass3d.OuterShell_s(TopoDS.Solid_s(solid))
        parent = len(shells)
        shells.append(ShellInfo("outer", None, True, []))
        other = []
        sh = TopExp_Explorer(solid, TopAbs_SHELL)
        while sh.More():
            cur = sh.Current()
            sh.Next()
            if cur.IsSame(outer):
                other.append((0, cur))
            else:
                other.append((1, cur))
        for is_cavity, shell in sorted(other, key=lambda x: x[0]):
            faces_of = _indexed(shell, TopAbs_FACE)
            ids = sorted(
                face_map.FindIndex(faces_of.FindKey(i)) - 1 for i in range(1, faces_of.Extent() + 1)
            )
            if is_cavity:
                shells.append(ShellInfo("cavity", parent, True, ids))
            else:
                shells[parent].faces = ids

    return LabeledMesh(
        tris=np.concatenate(tri_blocks) if tri_blocks else np.zeros((0, 3, 3)),
        face_id=np.concatenate(id_blocks) if id_blocks else np.zeros(0, dtype=np.int32),
        faces=faces,
        adjacency=adjacency,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        vertices=[
            list(canonical.get(("v", i), _vertex_point(vertex_map.FindKey(i))))
            for i in range(1, vertex_map.Extent() + 1)
        ],
        shells=shells,
    )


def distance_to_surface(face: FaceInfo, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    prm = face.params
    if face.surface == "plane":
        return np.abs((p - np.array(prm["origin"])) @ np.array(prm["normal"]))
    if face.surface == "sphere":
        return np.abs(np.linalg.norm(p - np.array(prm["center"]), axis=1) - prm["radius"])
    if face.surface in ("cylinder", "cone", "torus"):
        origin = np.array(prm.get("origin", prm.get("apex", prm.get("center"))))
        axis = np.array(prm["axis"])
        v = p - origin
        h = v @ axis
        rho = np.linalg.norm(v - h[:, None] * axis, axis=1)
        if face.surface == "cylinder":
            return np.abs(rho - prm["radius"])
        if face.surface == "torus":
            return np.abs(np.hypot(rho - prm["major_radius"], h) - prm["minor_radius"])
        a = prm["half_angle"]
        c, s = math.cos(a), math.sin(a)
        best = np.full(len(p), np.inf)
        for sign in (1.0, -1.0):
            along = sign * h * c + rho * s
            perp = np.abs(rho * c - sign * h * s)
            best = np.minimum(best, np.where(along >= 0, perp, np.hypot(h, rho)))
        return best
    return np.full(len(p), np.nan)


def outward_normals(face: FaceInfo, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    prm = face.params
    if face.surface == "plane":
        return np.tile(prm["normal"], (len(p), 1))
    if face.surface == "sphere":
        n = p - np.array(prm["center"])
    else:
        origin = np.array(prm.get("origin", prm.get("apex", prm.get("center"))))
        axis = np.array(prm["axis"])
        v = p - origin
        radial = v - np.outer(v @ axis, axis)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        if face.surface == "cylinder":
            n = radial
        elif face.surface == "cone":
            a = prm["half_angle"]
            n = math.cos(a) * radial - math.sin(a) * axis
        else:
            n = p - (origin + prm["major_radius"] * radial)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return -n if face.reversed else n
