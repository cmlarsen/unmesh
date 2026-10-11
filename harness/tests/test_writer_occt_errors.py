import numpy as np
import pytest

pytest.importorskip("OCP")

from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepBuilderAPI import (  # noqa: E402
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.gp import gp_Pnt  # noqa: E402
from OCP.Standard import Standard_ConstructionError, Standard_TypeMismatch  # noqa: E402
from OCP.TopAbs import TopAbs_FACE  # noqa: E402
from OCP.TopoDS import TopoDS_Compound  # noqa: E402

import unmesh  # noqa: E402
import unmesh.step as step  # noqa: E402
from unmesh._writer import occ  # noqa: E402

CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=float,
)
CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ]
)


def cube():
    tris = CUBE_VERTICES[CUBE_FACES]
    return unmesh.convert(tris).ir, tris


def square(z: float):
    points = [gp_Pnt(0, 0, z), gp_Pnt(1, 0, z), gp_Pnt(1, 1, z), gp_Pnt(0, 1, z)]
    wire = BRepBuilderAPI_MakeWire()
    for a, b in zip(points, [*points[1:], points[0]], strict=True):
        wire.Add(BRepBuilderAPI_MakeEdge(a, b).Edge())
    return BRepBuilderAPI_MakeFace(wire.Wire()).Face()


class Replace:
    def __init__(self, original, replacement):
        self.original = original
        self.replacement = replacement

    def Apply(self, shape):
        return self.replacement if shape.IsSame(self.original) else shape


def test_reshaped_faces_expands_a_compound_replacement():
    original = square(0.0)
    pieces = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(pieces)
    builder.Add(pieces, square(1.0))
    builder.Add(pieces, square(2.0))
    faces = occ._reshaped_faces(Replace(original, pieces), original)
    assert len(faces) == 2
    assert all(f.ShapeType() == TopAbs_FACE for f in faces)


def test_reshaped_faces_keeps_an_unchanged_face():
    face = square(0.0)
    faces = occ._reshaped_faces(Replace(face, face), face)
    assert len(faces) == 1
    assert faces[0].IsSame(face)


def test_a_programming_error_in_the_writer_propagates(monkeypatch, tmp_path):
    ir, tris = cube()

    def boom(*args, **kwargs):
        raise AttributeError("boom")

    monkeypatch.setattr(occ, "build_plan_shell", boom)
    with pytest.raises(AttributeError):
        step.write(ir, tmp_path / "cube.step", mesh=tris, verify=False)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (Standard_TypeMismatch("TopoDS::Face"), "occt_type_mismatch"),
        (Standard_ConstructionError("boom"), "occt_exception"),
    ],
)
def test_an_occt_failure_falls_back_and_names_its_type(error, code, monkeypatch, tmp_path):
    ir, tris = cube()

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(occ, "build_plan_shell", boom)
    report = step.write(ir, tmp_path / "cube.step", mesh=tris)
    assert report.fallback == "faceted"
    assert report.fallback_code == code
    assert type(error).__name__ in report.fallback_reason
