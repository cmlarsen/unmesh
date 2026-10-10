from __future__ import annotations

import math
from typing import Any

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp
from OCP.Extrema import Extrema_ExtPC
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

SAMPLES = 64
CANDIDATES = 4
MATCH_GATE_MM = 1e-2


def brep_counts(shape) -> dict[str, int]:
    shape = getattr(shape, "wrapped", shape)
    out = {}
    for name, kind in (("faces", TopAbs_FACE), ("edges", TopAbs_EDGE), ("vertices", TopAbs_VERTEX)):
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, m)
        out[name] = m.Extent()
    return out


def minimum_edge_length(shape) -> float:
    shape = getattr(shape, "wrapped", shape)
    best = math.inf
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        edge = TopoDS.Edge_s(ex.Current())
        if not BRep_Tool.Degenerated_s(edge):
            props = GProp_GProps()
            BRepGProp.LinearProperties_s(edge, props)
            best = min(best, float(props.Mass()))
        ex.Next()
    return best


def minimum_face_area(shape) -> float:
    shape = getattr(shape, "wrapped", shape)
    best = math.inf
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(TopoDS.Face_s(ex.Current()), props)
        best = min(best, float(props.Mass()))
        ex.Next()
    return best


def boundary_edges(shape) -> list:
    shape = getattr(shape, "wrapped", shape)
    anc = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, anc)
    out = []
    for i in range(1, anc.Extent() + 1):
        edge = TopoDS.Edge_s(anc.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        faces: list = []
        for f in anc.FindFromIndex(i):
            if not any(f.IsSame(g) for g in faces):
                faces.append(f)
        if len(faces) == 2:
            out.append(edge)
    return out


def _sample(edge, n: int = SAMPLES) -> np.ndarray:
    c = BRepAdaptor_Curve(edge)
    return np.array(
        [c.Value(t).Coord() for t in np.linspace(c.FirstParameter(), c.LastParameter(), n)]
    )


def _distances(points: np.ndarray, edge) -> np.ndarray:
    c = BRepAdaptor_Curve(edge)
    first, last = c.FirstParameter(), c.LastParameter()
    ends = [np.array(c.Value(first).Coord()), np.array(c.Value(last).Coord())]
    out = np.empty(len(points))
    for k, p in enumerate(points):
        best = min(float(np.linalg.norm(p - e)) for e in ends)
        ext = Extrema_ExtPC(gp_Pnt(*map(float, p)), c, first, last)
        if ext.IsDone():
            for j in range(1, ext.NbExt() + 1):
                best = min(best, math.sqrt(ext.SquareDistance(j)))
        out[k] = best
    return out


def edge_hausdorff(written, truth) -> dict[str, Any]:
    mine = boundary_edges(written)
    theirs = boundary_edges(truth)
    truth_samples = [_sample(e) for e in theirs]
    centroids = np.array([s.mean(axis=0) for s in truth_samples]).reshape(-1, 3)
    per_edge = []
    used: set[int] = set()
    for edge in mine:
        pts = _sample(edge)
        best, best_j = math.inf, None
        order = np.argsort(np.linalg.norm(centroids - pts.mean(axis=0), axis=1))
        for j in order[:CANDIDATES]:
            forward = float(_distances(pts, theirs[j]).max())
            if forward > MATCH_GATE_MM:
                continue
            d = max(forward, float(_distances(truth_samples[j], edge).max()))
            if d < best:
                best, best_j = d, int(j)
        per_edge.append(best)
        if best_j is not None:
            used.add(best_j)
    return {
        "edges": len(mine),
        "truth_edges": len(theirs),
        "matched": len(used),
        "max": max(per_edge, default=0.0),
        "per_edge": per_edge,
    }
