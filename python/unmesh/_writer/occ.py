from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Message import Message
from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape, ShapeFix_Shell, ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shell

from .topology import BuildError, FacePlan, ShellPlan

SEW_TOLERANCE = 1e-6
KEY_SCALE = 1e7


def _key(p) -> tuple[int, int, int]:
    return (round(p[0] * KEY_SCALE), round(p[1] * KEY_SCALE), round(p[2] * KEY_SCALE))


class _Pool:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def vertex(self, p):
        k = _key(p)
        v = self.vertices.get(k)
        if v is None:
            v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex()
            self.vertices[k] = v
        return v

    def edge(self, a, b):
        ka, kb = _key(a), _key(b)
        if ka == kb:
            return None
        lo, hi = (ka, kb) if ka <= kb else (kb, ka)
        e = self.edges.get((lo, hi))
        if e is None:
            pa, pb = (a, b) if ka <= kb else (b, a)
            e = BRepBuilderAPI_MakeEdge(self.vertex(pa), self.vertex(pb)).Edge()
            self.edges[(lo, hi)] = e
        return e if ka <= kb else TopoDS.Edge_s(e.Reversed())

    def wire(self, loop):
        mk = BRepBuilderAPI_MakeWire()
        count = 0
        for i in range(len(loop)):
            e = self.edge(loop[i], loop[(i + 1) % len(loop)])
            if e is not None:
                mk.Add(e)
                count += 1
        if count < 3 or not mk.IsDone():
            raise BuildError("could not close a boundary wire")
        return mk.Wire()


def _face(pool: _Pool, plan: FacePlan):
    pln = gp_Pln(gp_Pnt(*map(float, plan.origin)), gp_Dir(*map(float, plan.normal)))
    mk = BRepBuilderAPI_MakeFace(pln, pool.wire(plan.loops[0]), True)
    for hole in plan.loops[1:]:
        mk.Add(pool.wire(hole))
    if not mk.IsDone():
        raise BuildError(f"region {plan.region}: face construction failed")
    fix = ShapeFix_Face(mk.Face())
    fix.Perform()
    return fix.Face()


def _shells_of(shape):
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    out = []
    while ex.More():
        out.append(TopoDS.Shell_s(ex.Current()))
        ex.Next()
    return out


def _faces_of(shape):
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    out = []
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current()))
        ex.Next()
    return out


def sew(faces, tolerance=SEW_TOLERANCE):
    sewing = BRepBuilderAPI_Sewing(tolerance)
    for f in faces:
        sewing.Add(f)
    sewing.Perform()
    return sewing


def tolerance_of(shape) -> float:
    return float(ShapeAnalysis_ShapeTolerance().Tolerance(shape, 1))


def volume_of(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def is_valid(shape) -> bool:
    return bool(BRepCheck_Analyzer(shape).IsValid())


def _orient_shell(shell, mapping=None):
    fix = ShapeFix_Shell(shell)
    fix.Perform()
    if mapping is not None:
        ctx = fix.Context()
        mapping[:] = [(r, TopoDS.Face_s(ctx.Apply(f))) for r, f in mapping]
    return fix.Shell()


def make_solid(outer_shell, cavities):
    mk = BRepBuilderAPI_MakeSolid(outer_shell)
    solid = mk.Solid()
    if volume_of(solid) < 0:
        fix = ShapeFix_Solid(solid)
        fix.Perform()
        solid = TopoDS.Solid_s(fix.Solid())
    if not cavities:
        return solid
    mk2 = BRepBuilderAPI_MakeSolid(solid)
    for c in cavities:
        mk2.Add(c)
    return mk2.Solid()


def make_faceted_solid(outer_shell, cavities):
    mk = BRepBuilderAPI_MakeSolid(outer_shell)
    for c in cavities:
        mk.Add(c)
    return mk.Solid()


def orient_like_mesh(shell, expect_positive: bool):
    if (volume_of(BRepBuilderAPI_MakeSolid(shell).Solid()) < 0) == expect_positive:
        return TopoDS.Shell_s(shell.Reversed())
    return shell


def build_sewn_shell(faces_with_regions, tolerance=SEW_TOLERANCE):
    if len(faces_with_regions) == 1:
        region, face = faces_with_regions[0]
        shell = TopoDS_Shell()
        builder = BRep_Builder()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        return shell, [(region, face)]
    sewing = sew([f for _, f in faces_with_regions], tolerance)
    sewn = sewing.SewedShape()
    shells = _shells_of(sewn)
    if len(shells) != 1:
        raise BuildError(f"sewing produced {len(shells)} shells, expected 1")
    mapping = []
    for region, f in faces_with_regions:
        g = sewing.Modified(f) if sewing.IsModified(f) else f
        mapping.append((region, g))
    return shells[0], mapping


def face_tolerances(shape, mapping, context=None) -> dict[int, float]:
    final = _faces_of(shape)
    out: dict[int, float] = {}
    for region, g in mapping:
        if context is not None:
            g = context.Apply(g)
        match = next((h for h in final if h.IsSame(g)), g)
        out[region] = max(out.get(region, 0.0), tolerance_of(match))
    return out


def build_plan_shell(plan: ShellPlan):
    pool = _Pool()
    faces = [(f.region, _face(pool, f)) for f in plan.faces]
    shell, mapping = build_sewn_shell(faces)
    shell = _orient_shell(shell, mapping)
    return shell, mapping


def build_triangle_shell(triangles: np.ndarray):
    pool = _Pool()
    faces = []
    for t in triangles:
        f = triangle_face(pool, t[0], t[1], t[2])
        if f is not None:
            faces.append((-1, f))
    if not faces:
        raise BuildError("no non-degenerate triangles")
    shell, _ = build_sewn_shell(faces)
    return _orient_shell(shell)


def compound_of(shapes):
    if len(shapes) == 1:
        return shapes[0]
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


def unify_exact(solid):
    usd = ShapeUpgrade_UnifySameDomain(solid, True, True, False)
    usd.SetLinearTolerance(1e-9)
    usd.SetAngularTolerance(1e-9)
    usd.Build()
    return usd.Shape()


def write_step(shape, path) -> None:
    Interface_Static.SetCVal_s("write.step.schema", "AP214CD")
    writer = STEPControl_Writer()
    messenger = Message.DefaultMessenger_s()
    printers = list(messenger.Printers())
    messenger.ChangePrinters().Clear()
    try:
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
            raise OSError("STEP transfer failed")
        if writer.Write(str(path)) != IFSelect_RetDone:
            raise OSError(f"could not write {path}")
    finally:
        for printer in printers:
            messenger.AddPrinter(printer)


def read_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise OSError(f"could not read {path}")
    reader.TransferRoots()
    return reader.OneShape()


def count_solids(shape) -> int:
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def triangle_face(pool: _Pool, a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return None
    plan = FacePlan(-1, "facets", a, n / norm, [[a, b, c]])
    return _face(pool, plan)


def fix_shape(shape):
    fix = ShapeFix_Shape(shape)
    fix.Perform()
    return fix.Shape(), fix.Context()
