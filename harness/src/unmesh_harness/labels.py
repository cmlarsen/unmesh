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
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

TANGENT_THRESHOLD_RAD = 0.01

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


@dataclass
class LabeledMesh:
    tris: np.ndarray
    face_id: np.ndarray
    faces: list[FaceInfo]
    adjacency: list[EdgeAdjacency]
    linear_deflection: float
    angular_deflection: float
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            tris=self.tris,
            face_id=self.face_id,
            faces=json.dumps([f.__dict__ for f in self.faces]),
            adjacency=json.dumps([a.__dict__ for a in self.adjacency]),
            deflection=np.array([self.linear_deflection, self.angular_deflection]),
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
            )

    def write_stl(self, path: str | Path) -> None:
        from unmesh import write_stl

        write_stl(path, self.tris)

    def face_tris(self, face: int) -> np.ndarray:
        return self.tris[self.face_id == face]


def _vec(v) -> list[float]:
    return [v.X(), v.Y(), v.Z()]


def _surface_params(face, adaptor: BRepAdaptor_Surface) -> tuple[str, dict[str, Any]]:
    kind = adaptor.GetType()
    name = _SURFACE_TYPES.get(kind, "other")
    flip = face.Orientation() == TopAbs_REVERSED
    if kind == GeomAbs_Plane:
        pln = adaptor.Plane()
        n = _vec(pln.Axis().Direction())
        return name, {
            "origin": _vec(pln.Location()),
            "normal": [-c for c in n] if flip else n,
        }
    if kind == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        return name, {
            "origin": _vec(cyl.Location()),
            "axis": _vec(cyl.Axis().Direction()),
            "radius": cyl.Radius(),
        }
    if kind == GeomAbs_Cone:
        cone = adaptor.Cone()
        return name, {
            "apex": _vec(cone.Apex()),
            "axis": _vec(cone.Axis().Direction()),
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


def _edge_dihedral(edge, face_a, face_b, samples: int = 5) -> float:
    first, last = BRep_Tool.Range_s(edge, face_a)
    worst = 0.0
    for k in range(samples):
        t = first + (last - first) * (k + 0.5) / samples
        na, nb = _face_normal(face_a, edge, t), _face_normal(face_b, edge, t)
        if na is None or nb is None:
            continue
        cos = float(np.clip(na @ nb, -1.0, 1.0))
        worst = max(worst, math.acos(cos))
    return worst


def _indexed(shape, kind) -> TopTools_IndexedMapOfShape:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def _point(p, trsf) -> tuple[float, float, float]:
    q = p.Transformed(trsf)
    return (q.X(), q.Y(), q.Z())


def tessellate(
    shape: Shape | Any, linear_deflection: float, angular_deflection: float
) -> LabeledMesh:
    wrapped = getattr(shape, "wrapped", shape)
    BRepMesh_IncrementalMesh(wrapped, linear_deflection, False, angular_deflection, True)

    face_map = _indexed(wrapped, TopAbs_FACE)
    edge_map = _indexed(wrapped, TopAbs_EDGE)
    vertex_map = _indexed(wrapped, TopAbs_VERTEX)

    canonical: dict[tuple, tuple[float, float, float]] = {}
    tri_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    faces: list[FaceInfo] = []

    for fi in range(1, face_map.Extent() + 1):
        face = TopoDS.Face_s(face_map.FindKey(fi))
        adaptor = BRepAdaptor_Surface(face)
        name, params = _surface_params(face, adaptor)
        flipped = face.Orientation() == TopAbs_REVERSED
        faces.append(FaceInfo(fi - 1, name, params, flipped))

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
            for k, node in enumerate(idx):
                if k == 0 and not v_first.IsNull():
                    key = ("v", vertex_map.FindIndex(v_first))
                elif k == last and not v_last.IsNull():
                    key = ("v", vertex_map.FindIndex(v_last))
                else:
                    key = ("e", eid, k)
                nodes[node - 1] = canonical.setdefault(key, nodes[node - 1])

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
        fa, fb = (TopoDS.Face_s(f) for f in owners)
        dihedral = _edge_dihedral(edge, fa, fb)
        adjacency.append(
            EdgeAdjacency(
                edge_id=ei - 1,
                face_a=face_map.FindIndex(fa) - 1,
                face_b=face_map.FindIndex(fb) - 1,
                curve=_CURVE_TYPES.get(BRepAdaptor_Curve(edge).GetType(), "other"),
                tangent=dihedral < TANGENT_THRESHOLD_RAD,
                dihedral=dihedral,
            )
        )

    return LabeledMesh(
        tris=np.concatenate(tri_blocks) if tri_blocks else np.zeros((0, 3, 3)),
        face_id=np.concatenate(id_blocks) if id_blocks else np.zeros(0, dtype=np.int32),
        faces=faces,
        adjacency=adjacency,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
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
