import copy
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("OCP")

from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCP.GeomAbs import GeomAbs_Plane  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

import unmesh  # noqa: E402
import unmesh.step as step  # noqa: E402
from unmesh._writer import occ  # noqa: E402
from unmesh.ir import Facets, Ir, IrError, Plane, Region, Shell  # noqa: E402
from unmesh_harness import labels  # noqa: E402
from unmesh_harness.degrade import core as degrade_core  # noqa: E402
from unmesh_harness.groundtruth import generate  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "ir"

VOLUMES = {
    "box": 1000.0,
    "plate_chamfer": 995.0,
    "cavity": 7784.0,
    "two_bodies": 625.0,
    "mixed_facets": 1000.0 + 10 * 5 * 2.5 / 3,
}


def load(name):
    return Ir.loads((FIXTURES / f"{name}.json").read_text())


def reimported_faces(path):
    shape = occ.read_step(path)
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    types = []
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        types.append(BRepAdaptor_Surface(face).GetType())
        ex.Next()
    return shape, types


def expected_face_count(ir):
    return sum(len(r.surface.faces) if isinstance(r.surface, Facets) else 1 for r in ir.regions)


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def mesh_from_ir(ir):
    tris = np.zeros((ir.source.triangle_count, 3, 3))
    for r in ir.regions:
        n = np.asarray(r.surface.normal)
        pts = np.array([v.position for v in ir.vertices if r.id in v.regions])
        c = pts.mean(axis=0)
        u = pts[0] - c
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        order = np.argsort([math.atan2((p - c) @ w, (p - c) @ u) for p in pts])
        quad = pts[order]
        assert len(quad) == 4 and len(r.triangles) == 2
        tris[r.triangles[0]] = quad[[0, 1, 2]]
        tris[r.triangles[1]] = quad[[0, 2, 3]]
    return tris


@pytest.mark.parametrize("name", sorted(VOLUMES))
def test_planar_fixture_roundtrip(name, tmp_path):
    ir = load(name)
    path = tmp_path / f"{name}.step"
    report = step.write(ir, path)
    assert report.valid and report.fallback is None
    assert report.max_shape_tolerance <= 1e-3
    shape, types = reimported_faces(path)
    assert len(types) == expected_face_count(ir)
    assert all(t == GeomAbs_Plane for t in types)
    assert occ.volume_of(shape) == pytest.approx(VOLUMES[name], rel=1e-9)
    assert report.solids == occ.count_solids(shape)
    assert all(f.max_shape_tolerance <= 1e-3 for f in report.faces)


def test_cavity_is_one_solid(tmp_path):
    report = step.write(load("cavity"), tmp_path / "c.step")
    assert report.solids == 1
    assert len(report.shells) == 1 and report.shells[0].kind == "solid"


def test_two_bodies_compound(tmp_path):
    report = step.write(load("two_bodies"), tmp_path / "t.step")
    assert report.solids == 2


def test_open_shell_written_as_shell(tmp_path):
    path = tmp_path / "o.step"
    report = step.write(load("open_shell"), path)
    assert report.valid
    assert report.solids == 0
    assert report.shells[0].kind == "shell" and report.shells[0].volume is None
    shape, types = reimported_faces(path)
    assert len(types) == 2
    assert occ.count_solids(shape) == 0


def test_mixed_facets_seam_gaps_recorded(tmp_path):
    report = step.write(load("mixed_facets"), tmp_path / "m.step")
    assert report.valid
    assert len(report.seams) == 4
    assert all(s.max_gap < 1e-9 for s in report.seams)


def shifted_box():
    ir = load("box")
    bad = copy.deepcopy(ir)
    for r in bad.regions:
        if isinstance(r.surface, Plane) and r.surface.normal == (0.0, 0.0, 1.0):
            r.surface.origin = (r.surface.origin[0], r.surface.origin[1], r.surface.origin[2] + 1.0)
    return ir, bad


def test_inconsistent_plane_falls_back_to_faceted(tmp_path):
    ir, bad = shifted_box()
    tris = mesh_from_ir(ir)
    path = tmp_path / "f.step"
    report = step.write(bad, path, mesh=tris)
    assert report.fallback == "faceted"
    assert report.fallback_reason
    assert report.valid
    shape, types = reimported_faces(path)
    assert occ.volume_of(shape) == pytest.approx(1000.0, rel=1e-9)
    assert report.shells[0].volume == pytest.approx(signed_volume(tris), rel=1e-9)
    assert len(types) == 6


def test_inconsistent_plane_without_mesh_does_not_raise(tmp_path):
    _, bad = shifted_box()
    path = tmp_path / "n.step"
    report = step.write(bad, path)
    assert not report.valid
    assert report.fallback is None
    assert report.issues
    assert not path.exists()


@pytest.mark.parametrize("name", ["cavity", "two_bodies"])
def test_faceted_fallback_keeps_cavities_and_bodies(name, tmp_path):
    ir = load(name)
    bad = copy.deepcopy(ir)
    bad.regions[0].surface.origin = tuple(c + 1.0 for c in bad.regions[0].surface.origin)
    tris = mesh_from_ir(ir)
    path = tmp_path / "f.step"
    report = step.write(bad, path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    shape, _ = reimported_faces(path)
    assert occ.volume_of(shape) == pytest.approx(VOLUMES[name], rel=1e-9)
    total = sum(s.volume for s in report.shells)
    assert total == pytest.approx(signed_volume(tris), rel=1e-9)
    assert report.solids == (1 if name == "cavity" else 2)


def test_flipped_mesh_winding_still_builds_faceted(tmp_path):
    ir, bad = shifted_box()
    tris = mesh_from_ir(ir)[:, ::-1]
    path = tmp_path / "w.step"
    report = step.write(bad, path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.shells[0].volume == pytest.approx(abs(signed_volume(tris)), rel=1e-9)
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(abs(signed_volume(tris)), rel=1e-9)


def test_unsupported_surface_reports_reason(tmp_path):
    report = step.write(load("plate_with_bore"), tmp_path / "p.step")
    assert not report.valid
    assert any("cylinder" in i for i in report.issues)


def test_tolerance_limit_triggers_fallback(tmp_path):
    ir = load("box")
    report = step.write(
        ir, tmp_path / "x.step", step.WriteOptions(max_shape_tolerance=1e-12), mesh=mesh_from_ir(ir)
    )
    assert report.fallback == "faceted"


def test_malformed_ir_raises(tmp_path):
    ir = load("box")
    ir.regions[0].id = 99
    with pytest.raises(IrError):
        step.write(ir, tmp_path / "bad.step")


def shift(ir, normal, d):
    out = copy.deepcopy(ir)
    for r in out.regions:
        if isinstance(r.surface, Plane) and r.surface.normal == normal:
            r.surface.origin = tuple(
                o + d * n for o, n in zip(r.surface.origin, r.surface.normal, strict=True)
            )
    return out


def test_report_measures_how_far_geometry_moved(tmp_path):
    clean = step.write(load("box"), tmp_path / "a.step")
    assert clean.max_vertex_displacement < 1e-9
    assert clean.max_boundary_deviation < 1e-9
    moved = step.write(shift(load("box"), (0.0, 0.0, 1.0), 0.002), tmp_path / "b.step")
    assert moved.valid
    assert moved.max_vertex_displacement == pytest.approx(0.002, abs=1e-9)
    assert moved.max_boundary_deviation == pytest.approx(0.002, abs=1e-9)
    assert moved.shells[0].max_vertex_displacement == pytest.approx(0.002, abs=1e-9)
    top = next(f for f in moved.faces if f.max_vertex_displacement > 1e-3)
    assert top.surface_type == "plane"
    assert top.max_boundary_deviation == pytest.approx(0.002, abs=1e-9)


def test_deviation_cap_is_enforced(tmp_path):
    bad = shift(load("box"), (0.0, 0.0, 1.0), 0.002)
    path = tmp_path / "c.step"
    report = step.write(bad, path, step.WriteOptions(max_deviation=1e-3))
    assert not report.valid
    assert not path.exists()
    assert any("limit" in i for i in report.issues)


def test_no_mesh_and_one_bad_shell_writes_nothing(tmp_path):
    ir = load("two_bodies")
    bad = copy.deepcopy(ir)
    bad.regions[0].surface.origin = tuple(c + 1.0 for c in bad.regions[0].surface.origin)
    path = tmp_path / "p.step"
    report = step.write(bad, path)
    assert not report.valid
    assert report.solids == 0
    assert not path.exists()
    assert [s.valid for s in report.shells] == [False, True]


def test_two_plane_vertices_snap_with_facets_neighbour(tmp_path):
    ir = load("mixed_facets")
    bad = shift(ir, (0.0, -1.0, 0.0), 0.002)
    path = tmp_path / "m.step"
    report = step.write(bad, path)
    assert report.valid
    assert report.max_vertex_displacement == pytest.approx(0.002, abs=1e-9)
    shape, _ = reimported_faces(path)
    assert occ.volume_of(shape) == pytest.approx(VOLUMES["mixed_facets"] - 0.002 * 20 * 5, rel=1e-3)


def test_missing_ocp_gives_install_hint():
    code = (
        "import sys\n"
        "sys.modules['OCP'] = None\n"
        "from unmesh import Ir\n"
        "import unmesh.step as step\n"
        f"ir = Ir.loads(open({str(FIXTURES / 'box.json')!r}).read())\n"
        "try:\n"
        "    step.write(ir, 'x.step')\n"
        "except ImportError as e:\n"
        "    assert 'pip install unmesh[step]' in str(e), e\n"
        "else:\n"
        "    raise SystemExit(1)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def unit_box_with_stray():
    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (1.0, 1.0, 0.0)
    d = (0.0, 1.0, 0.0)
    e = (0.0, 0.0, 1.0)
    f = (1.0, 0.0, 1.0)
    g = (1.0, 1.0, 1.0)
    h = (0.0, 1.0, 1.0)
    box = np.array(
        [
            [a, d, c],
            [a, c, b],
            [e, f, g],
            [e, g, h],
            [a, b, f],
            [a, f, e],
            [d, h, g],
            [d, g, c],
            [a, e, h],
            [a, h, d],
            [b, c, g],
            [b, g, f],
        ]
    )
    stray = np.array([[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.5, 6.0, 5.0]]])
    return box, stray


def box_plus_stray_ir():
    box, stray = unit_box_with_stray()
    ir, _ = unmesh.convert(box)
    stray_id = len(ir.regions)
    ir.regions.append(
        Region(
            stray_id,
            Facets([tuple(p) for p in stray[0].tolist()], [(0, 1, 2)]),
            [len(box)],
            None,
        )
    )
    ir.shells.append(Shell(False, "outer", None, [stray_id]))
    ir.source.triangle_count = len(box) + len(stray)
    return ir, box, stray


def test_stray_open_triangle_written_as_shell_in_compound(tmp_path):
    ir, box, stray = box_plus_stray_ir()
    mesh = np.concatenate([box, stray])
    assert signed_volume(box) == pytest.approx(1.0, rel=1e-9)
    analytic = step.write(ir, tmp_path / "a.step")
    assert analytic.valid and analytic.fallback is None
    assert [s.kind for s in analytic.shells] == ["solid", "shell"]
    assert analytic.open_shells == [1]
    bad = copy.deepcopy(ir)
    bad.regions[0].surface.origin = tuple(c + 1.0 for c in bad.regions[0].surface.origin)
    path = tmp_path / "s.step"
    report = step.write(bad, path, mesh=mesh)
    assert report.fallback == "faceted" and report.valid
    assert report.solids == 1
    assert [s.kind for s in report.shells] == ["solid", "shell"]
    assert report.open_shells == [1]
    shape, types = reimported_faces(path)
    assert occ.count_solids(shape) == 1
    assert len(types) == 7
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    shells = 0
    while ex.More():
        shells += 1
        ex.Next()
    assert shells == 2
    assert report.shells[0].volume == pytest.approx(signed_volume(box), rel=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason="OCP transfer mis-orients rough faceted shells: BRepClass3d::OuterShell's "
    "infinite-point probe returns IN for the outward noisy shell, so the single shell "
    "reads back with flipped volume; the in-memory solid is correct",
)
def test_noisy_curved_mesh_fallback_keeps_positive_volume(tmp_path):
    import unmesh_harness.degrade.defects  # noqa: F401,E402
    import unmesh_harness.degrade.noise  # noqa: F401,E402

    gt = generate("revolved_cone", 1)
    mesh = labels.tessellate(gt.solid, 0.01, 0.2)
    degraded = degrade_core.apply_chain(mesh, [("slivers", 1.0), ("noise_isotropic", 1.0)], 6001)
    tris = degraded.tris
    expected = signed_volume(tris)
    assert expected > 0
    ir, _ = unmesh.convert(tris)
    path = tmp_path / "n.step"
    report = step.write(ir, path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.shells[0].volume == pytest.approx(expected, rel=1e-9)
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(expected, rel=1e-9)


def box_tris(origin=(0.0, 0.0, 0.0), size=1.0, inward=False):
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    a = (x0, y0, z0)
    b = (x1, y0, z0)
    c = (x1, y1, z0)
    d = (x0, y1, z0)
    e = (x0, y0, z1)
    f = (x1, y0, z1)
    g = (x1, y1, z1)
    h = (x0, y1, z1)
    t = np.array(
        [
            [a, d, c],
            [a, c, b],
            [e, f, g],
            [e, g, h],
            [a, b, f],
            [a, f, e],
            [d, h, g],
            [d, g, c],
            [a, e, h],
            [a, h, d],
            [b, c, g],
            [b, g, f],
        ]
    )
    return t[:, ::-1] if inward else t


def break_first_region(ir):
    bad = copy.deepcopy(ir)
    bad.regions[0].surface.origin = tuple(c + 1.0 for c in bad.regions[0].surface.origin)
    return bad


def test_clean_box_with_cavity_fallback(tmp_path):
    tris = np.concatenate(
        [box_tris((0.0, 0.0, 0.0), 4.0), box_tris((1.0, 1.0, 1.0), 1.0, inward=True)]
    )
    assert signed_volume(tris) == pytest.approx(63.0, rel=1e-9)
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role, s.parent) for s in ir.shells] == [
        (True, "outer", None),
        (True, "cavity", 0),
    ]
    path = tmp_path / "c.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.solids == 1
    assert report.shells[0].volume == pytest.approx(signed_volume(tris), rel=1e-9)
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(signed_volume(tris), rel=1e-9)


def test_two_disjoint_boxes_fallback(tmp_path):
    tris = np.concatenate([box_tris((0.0, 0.0, 0.0), 2.0), box_tris((5.0, 0.0, 0.0), 1.0)])
    assert signed_volume(tris) == pytest.approx(9.0, rel=1e-9)
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role) for s in ir.shells] == [(True, "outer"), (True, "outer")]
    path = tmp_path / "d.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.solids == 2
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(signed_volume(tris), rel=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason="OCP transfer mis-orients rough faceted shells: BRepClass3d::OuterShell's "
    "infinite-point probe returns IN for the outward noisy shell, so "
    "TopoDSToStep_MakeBrepWithVoids maps nothing and the file reads back empty; "
    "the in-memory solid is correct",
)
def test_noisy_cone_with_cavity_fallback(tmp_path):
    import unmesh_harness.degrade.defects  # noqa: F401,E402
    import unmesh_harness.degrade.noise  # noqa: F401,E402

    gt = generate("revolved_cone", 1)
    mesh = labels.tessellate(gt.solid, 0.01, 0.2)
    degraded = degrade_core.apply_chain(mesh, [("slivers", 1.0), ("noise_isotropic", 1.0)], 6001)
    tris = np.concatenate([degraded.tris, box_tris((-1.5, -1.5, 17.5), 3.0, inward=True)])
    expected = signed_volume(tris)
    assert expected > 0
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role) for s in ir.shells] == [(True, "outer"), (True, "cavity")]
    path = tmp_path / "n.step"
    report = step.write(ir, path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.solids == 1
    assert report.shells[0].volume == pytest.approx(expected, rel=1e-9)
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(expected, rel=1e-9)


def test_faceted_mesh_winding_decision():
    outer = box_tris((0.0, 0.0, 0.0), 2.0)
    cavity = box_tris((0.5, 0.5, 0.5), 0.5, inward=True)
    expect = {0: True, 1: False}
    oriented = step._orient_closed({0: outer, 1: cavity}, expect)
    assert signed_volume(oriented[0]) == pytest.approx(8.0, rel=1e-9)
    assert signed_volume(oriented[1]) == pytest.approx(-0.125, rel=1e-9)
    flipped = step._orient_closed({0: outer[:, ::-1], 1: cavity[:, ::-1]}, expect)
    assert signed_volume(flipped[0]) == pytest.approx(8.0, rel=1e-9)
    assert signed_volume(flipped[1]) == pytest.approx(-0.125, rel=1e-9)


def test_noise_only_cone_with_cavity_fallback(tmp_path):
    import unmesh_harness.degrade.noise  # noqa: F401,E402

    gt = generate("revolved_cone", 1)
    mesh = labels.tessellate(gt.solid, 0.01, 0.2)
    degraded = degrade_core.apply_chain(mesh, [("noise_isotropic", 1.0)], 6001)
    tris = np.concatenate([degraded.tris, box_tris((-1.5, -1.5, 17.5), 3.0, inward=True)])
    expected = signed_volume(tris)
    assert expected > 0
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role) for s in ir.shells] == [(True, "outer"), (True, "cavity")]
    path = tmp_path / "m.step"
    report = step.write(ir, path, mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.solids == 1
    assert report.shells[0].volume == pytest.approx(expected, rel=1e-9)
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(expected, rel=1e-9)
