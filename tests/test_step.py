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


def test_plate_with_bore_is_written_analytic(tmp_path):
    report = step.write(load("plate_with_bore"), tmp_path / "p.step")
    assert report.valid and report.verified and report.fallback is None
    assert report.shells[0].volume == pytest.approx(30 * 20 * 6 - math.pi * 16 * 6, rel=1e-12)
    shape, types = reimported_faces(tmp_path / "p.step")
    assert types.count(GeomAbs_Plane) == 6 and len(types) == 7


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


def rough_cone():
    import unmesh_harness.degrade.defects  # noqa: F401,E402
    import unmesh_harness.degrade.noise  # noqa: F401,E402

    gt = generate("revolved_cone", 1)
    mesh = labels.tessellate(gt.solid, 0.01, 0.2)
    degraded = degrade_core.apply_chain(mesh, [("slivers", 1.0), ("noise_isotropic", 1.0)], 6001)
    return degraded.tris


def test_noisy_curved_mesh_fallback_keeps_positive_volume(tmp_path):
    tris = rough_cone()
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


def rough_cone_with_cavity():
    return np.concatenate([rough_cone(), box_tris((-1.5, -1.5, 17.5), 3.0, inward=True)])


def test_noisy_cone_with_cavity_fallback(tmp_path):
    tris = rough_cone_with_cavity()
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
    vertices = np.concatenate([outer, cavity]).reshape(-1, 3)
    ids = np.arange(len(vertices)).reshape(-1, 3)
    raw = {0: ids[:12], 1: ids[12:]}
    expect = {0: True, 1: False}
    oriented = step._orient_closed(vertices, raw, expect)
    assert signed_volume(vertices[oriented[0]]) == pytest.approx(8.0, rel=1e-9)
    assert signed_volume(vertices[oriented[1]]) == pytest.approx(-0.125, rel=1e-9)
    flipped = step._orient_closed(vertices, {k: v[:, ::-1] for k, v in raw.items()}, expect)
    assert signed_volume(vertices[flipped[0]]) == pytest.approx(8.0, rel=1e-9)
    assert signed_volume(vertices[flipped[1]]) == pytest.approx(-0.125, rel=1e-9)


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


def box_with_cavity():
    tris = np.concatenate(
        [box_tris((0.0, 0.0, 0.0), 4.0), box_tris((1.0, 1.0, 1.0), 1.0, inward=True)]
    )
    ir, _ = unmesh.convert(tris)
    return break_first_region(ir), tris


def test_readback_reports_success_with_measured_values(tmp_path):
    bad, tris = box_with_cavity()
    report = step.write(bad, tmp_path / "ok.step", mesh=tris)
    rb = report.readback
    assert report.valid and rb.ok and not rb.issues
    assert (rb.solids, rb.shells) == (rb.expected_solids, rb.expected_shells) == (1, 2)
    assert rb.expected_volume == pytest.approx(63.0, rel=1e-12)
    assert rb.volume == pytest.approx(63.0, rel=1e-9)
    assert 0 < rb.volume_tolerance < 1e-3


def test_readback_catches_a_dropped_cavity(tmp_path, monkeypatch):
    from unmesh._writer import faceted

    real = faceted.write

    def drop_voids(path, vertices, bodies, open_shells, **kw):
        real(path, vertices, [faceted.Body(b.outer) for b in bodies], open_shells, **kw)

    monkeypatch.setattr(faceted, "write", drop_voids)
    bad, tris = box_with_cavity()
    report = step.write(bad, tmp_path / "drop.step", mesh=tris)
    assert report.fallback == "faceted"
    assert not report.valid
    assert not report.readback.ok
    assert (report.readback.shells, report.readback.expected_shells) == (1, 2)
    assert report.readback.volume == pytest.approx(64.0, rel=1e-9)
    assert not report.shells[0].valid
    assert any("read-back" in i for i in report.shells[0].issues)
    assert any(i.startswith("read-back of the written file does not match") for i in report.issues)


def test_readback_catches_analytic_writer_mismatch(tmp_path, monkeypatch):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    real = occ.write_step
    unit = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    monkeypatch.setattr(occ, "write_step", lambda shape, path: real(unit, path))
    report = step.write(load("box"), tmp_path / "a.step")
    assert report.fallback is None
    assert not report.valid
    assert report.readback.volume == pytest.approx(1.0, rel=1e-9)
    assert report.readback.expected_volume == pytest.approx(1000.0, rel=1e-9)


def fold_faces(tris):
    a = tris[:, 0]
    e1 = tris[:, 1] - a
    e2 = tris[:, 2] - a
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1)[:, None]
    folds = []
    for k in range(len(tris)):
        for w in ([1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]):
            p = np.array(w) @ tris[k]
            pvec = np.cross(n[k], e2)
            inv = 1.0 / np.einsum("ij,ij->i", e1, pvec)
            tvec = p - a
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            qvec = np.cross(tvec, e1)
            v = (qvec @ n[k]) * inv
            s = np.einsum("ij,ij->i", e2, qvec) * inv
            hit = np.flatnonzero((u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9))
            last = hit[np.argmax(s[hit])]
            if n[last] @ n[k] < 0:
                folds.append(k)
                break
    return folds


def test_fold_faces_first_still_read_back_with_the_mesh_sign(tmp_path):
    tris = rough_cone()
    folds = fold_faces(tris)
    assert folds
    order = folds + [k for k in range(len(tris)) if k not in set(folds)]
    tris = tris[order]
    ir, _ = unmesh.convert(tris)
    report = step.write(ir, tmp_path / "f.step", mesh=tris)
    assert report.fallback == "faceted" and report.valid and report.readback.ok
    assert occ.volume_of(occ.read_step(tmp_path / "f.step")) == pytest.approx(
        signed_volume(tris), rel=1e-9
    )


def test_readback_catches_an_inverted_reimport(tmp_path, monkeypatch):
    real = occ.read_back

    def inverted(path, *args):
        solids, shells = real(path, *args)
        for solid in solids:
            solid.volume = -solid.volume
        return solids, shells

    monkeypatch.setattr(occ, "read_back", inverted)
    ir, bad = shifted_box()
    report = step.write(bad, tmp_path / "inv.step", mesh=mesh_from_ir(ir))
    assert report.fallback == "faceted"
    assert not report.valid and report.verified
    assert report.readback.volume == pytest.approx(-1000.0, rel=1e-9)
    assert any("volume" in i for i in report.readback.issues)
    assert not report.shells[0].valid


def step_entities(path):
    import re

    data = pathlib.Path(path).read_text().split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    ents = {}
    for m in re.finditer(r"#(\d+) = (\w+)\((.*?)\);\n", data, re.S):
        ents[int(m.group(1))] = (m.group(2), m.group(3))
    return ents


def step_signed_volume(path):
    import re

    ents = step_entities(path)

    def refs(text):
        return [int(r) for r in re.findall(r"#(\d+)", text)]

    def point(ref):
        _, args = ents[ref]
        return np.array([float(x) for x in args.split("(", 1)[1].rstrip(")").split(",")])

    def shell_volume(ref, sign):
        kind, args = ents[ref]
        if kind == "ORIENTED_CLOSED_SHELL":
            return shell_volume(refs(args)[0], sign * (1 if args.endswith(".T.") else -1))
        total = 0.0
        for face in refs(args):
            kind, fargs = ents[face]
            assert kind == "ADVANCED_FACE" and fargs.endswith(".T.")
            bound = refs(fargs)[0]
            loop = refs(ents[bound][1])[0]
            pts = []
            for oe in refs(ents[loop][1]):
                oargs = ents[oe][1]
                start, end = refs(ents[refs(oargs)[0]][1])[:2]
                vp = start if oargs.endswith(".T.") else end
                pts.append(point(refs(ents[vp][1])[0]))
            pts = np.array(pts)
            total += float(np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0) @ pts[0]) / 6.0
        return sign * total

    volume = 0.0
    for kind, args in ents.values():
        if kind == "MANIFOLD_SOLID_BREP":
            volume += shell_volume(refs(args)[0], 1)
        elif kind == "BREP_WITH_VOIDS":
            outer, *voids = refs(args)
            volume += shell_volume(outer, 1) + sum(shell_volume(v, 1) for v in voids)
    return volume


def test_faceted_step_text_matches_the_mesh_without_occt(tmp_path):
    tris = rough_cone_with_cavity()
    ir, _ = unmesh.convert(tris)
    paths = [tmp_path / "a.step", tmp_path / "b.step"]
    for path in paths:
        assert step.write(ir, path, mesh=tris).fallback == "faceted"
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert step_signed_volume(paths[0]) == pytest.approx(signed_volume(tris), rel=1e-9)
    ents = step_entities(paths[0])
    kinds = [k for k, _ in ents.values()]
    assert kinds.count("BREP_WITH_VOIDS") == 1 and kinds.count("ORIENTED_CLOSED_SHELL") == 1
    points = [a for k, a in ents.values() if k == "CARTESIAN_POINT"]
    assert len(points) == len(set(points))


def test_zero_area_triangle_is_absorbed_into_its_neighbour(tmp_path):
    tris = box_tris((0.0, 0.0, 0.0), 1.0)
    a, c, b = tris[1]
    m = (a + c) / 2
    tris = np.concatenate([tris[[0]], [[a, m, b], [m, c, b], [a, c, m]], tris[2:]])
    assert signed_volume(tris) == pytest.approx(1.0, rel=1e-12)
    ir, _ = unmesh.convert(tris)
    path = tmp_path / "z.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted" and report.valid and report.readback.ok
    shape, types = reimported_faces(path)
    assert occ.volume_of(shape) == pytest.approx(1.0, rel=1e-9)
    assert len(types) == 6


def step_topology_errors(path):
    import re

    ents = step_entities(path)

    def refs(text):
        return [int(r) for r in re.findall(r"#(\d+)", text)]

    errors = []
    for sid, (kind, args) in ents.items():
        if kind not in ("CLOSED_SHELL", "OPEN_SHELL"):
            continue
        uses: dict[tuple[int, bool], int] = {}
        for face in refs(args):
            fkind, fargs = ents[face]
            if fkind != "ADVANCED_FACE" or not fargs.endswith(".T."):
                errors.append(f"shell {sid}: face {face} is {fkind} {fargs[-4:]}")
            for bound in refs(fargs.rsplit(",", 1)[0].split(")")[0]):
                bkind, bargs = ents[bound]
                forward = bargs.endswith(".T.")
                ends = []
                for oe in refs(ents[refs(bargs)[0]][1]):
                    oargs = ents[oe][1]
                    edge = refs(oargs)[0]
                    start, end = refs(ents[edge][1])[:2]
                    sense = oargs.endswith(".T.") == forward
                    uses[(edge, sense)] = uses.get((edge, sense), 0) + 1
                    ends.append((start, end) if oargs.endswith(".T.") else (end, start))
                for (_, e), (s2, _) in zip(ends, ends[1:] + ends[:1], strict=True):
                    if e != s2:
                        errors.append(f"shell {sid}: loop of face {face} is not connected")
                        break
        for (edge, sense), n in uses.items():
            if n != 1:
                errors.append(f"shell {sid}: edge {edge} used {n} times with sense {sense}")
            if kind == "CLOSED_SHELL" and (edge, not sense) not in uses:
                errors.append(f"shell {sid}: edge {edge} used only with sense {sense}")
    return errors


def test_emitted_step_topology_is_consistent(tmp_path):
    cases = {
        "cone": rough_cone(),
        "cone_cavity": rough_cone_with_cavity(),
        "touching": touching_cavity(),
    }
    for name, tris in cases.items():
        ir, _ = unmesh.convert(tris)
        path = tmp_path / f"{name}.step"
        if name == "touching":
            ir = break_first_region(ir)
        assert step.write(ir, path, mesh=tris).fallback == "faceted"
        assert step_topology_errors(path) == [], name
    ir, box, stray = box_plus_stray_ir()
    path = tmp_path / "stray.step"
    report = step.write(break_first_region(ir), path, mesh=np.concatenate([box, stray]))
    assert report.fallback == "faceted" and report.valid
    assert step_topology_errors(path) == []
    kinds = [k for k, _ in step_entities(path).values()]
    assert kinds.count("OPEN_SHELL") == 1 and kinds.count("CLOSED_SHELL") == 1


def touching_cavity():
    o = np.array([0.0, 0.0, 0.0])
    a = np.array([1.0, 0.5, 0.5])
    b = np.array([0.5, 1.0, 0.5])
    c = np.array([0.5, 0.5, 1.0])
    inward = np.array([[o, a, b], [o, c, a], [o, b, c], [a, c, b]])
    return np.concatenate([box_tris((0.0, 0.0, 0.0), 4.0), inward])


@pytest.fixture
def deadline():
    import signal

    def expire(signum, frame):
        raise TimeoutError("step.write did not return within 60 s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def test_cavity_touching_the_outer_wall_at_a_vertex(tmp_path, deadline):
    tris = touching_cavity()
    expected = 64.0 - 1.0 / 12.0
    assert signed_volume(tris) == pytest.approx(expected, rel=1e-12)
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role) for s in ir.shells] == [(True, "outer"), (True, "cavity")]
    path = tmp_path / "t.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted" and report.valid and report.readback.ok
    assert report.solids == 1
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(expected, rel=1e-9)


def test_cavity_sharing_outer_walls_is_written_and_flagged(tmp_path, deadline):
    tris = np.concatenate([box_tris((0.0, 0.0, 0.0), 4.0), box_tris((0.0, 0.0, 0.0), 1.0, True)])
    ir, _ = unmesh.convert(tris)
    assert [(s.closed, s.role) for s in ir.shells] == [(True, "outer"), (True, "cavity")]
    path = tmp_path / "w.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted"
    assert step_signed_volume(path) == pytest.approx(63.0, rel=1e-9)
    assert step_topology_errors(path) == []
    assert not report.valid
    assert "re-imported 2 solids, wrote 1" in report.readback.issues


@pytest.mark.parametrize("deferred_first", [True, False])
def test_adjacent_zero_area_triangles_are_absorbed(tmp_path, deferred_first):
    tris = box_tris((0.0, 0.0, 0.0), 1.0)
    a, c, b = tris[1]
    m1 = a + (c - a) / 4
    m2 = a + (c - a) / 2
    d1 = [a, c, m2]
    d2 = [a, m2, m1]
    degenerate = [d2, d1] if deferred_first else [d1, d2]
    fan = [[a, m1, b], [m1, m2, b], [m2, c, b]]
    tris = np.concatenate([tris[[0]], fan, degenerate, tris[2:]])
    assert signed_volume(tris) == pytest.approx(1.0, rel=1e-12)
    ir, _ = unmesh.convert(tris)
    path = tmp_path / "z.step"
    report = step.write(break_first_region(ir), path, mesh=tris)
    assert report.fallback == "faceted" and report.valid and report.readback.ok
    assert occ.volume_of(occ.read_step(path)) == pytest.approx(1.0, rel=1e-9)
    assert step_topology_errors(path) == []


def test_verify_false_skips_readback_and_says_so(tmp_path):
    ir, bad = shifted_box()
    tris = mesh_from_ir(ir)
    unverified = step.write(bad, tmp_path / "u.step", mesh=tris, verify=False)
    assert unverified.fallback == "faceted" and unverified.valid
    assert unverified.verified is False and unverified.readback is None
    assert unverified.solids == 1
    assert set(unverified.timings) == {"write_s"}
    verified = step.write(bad, tmp_path / "v.step", mesh=tris)
    assert verified.verified is True and verified.readback.ok
    assert set(verified.timings) == {"write_s", "readback_s"}
    assert (tmp_path / "u.step").read_bytes() == (tmp_path / "v.step").read_bytes()
