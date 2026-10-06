import pathlib

import numpy as np
import pytest

pytest.importorskip("OCP")

import unmesh  # noqa: E402
import unmesh.step as step  # noqa: E402
from test_faceted_fallback_benchmark import break_analytic, signed_volume, uv_sphere  # noqa: E402
from unmesh._writer import faceted, readback, textcheck  # noqa: E402
from unmesh.ir import Ir  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ir"


@pytest.fixture(scope="module")
def sphere():
    tris = uv_sphere(100, 51)
    ir, _ = unmesh.convert(tris)
    return break_analytic(ir), tris


@pytest.fixture
def child_readback(monkeypatch):
    monkeypatch.setattr(step, "INPROCESS_READBACK_BYTES", 0)


@pytest.fixture
def text_only(child_readback):
    return step.WriteOptions(readback_memory_mb=1.0)


def test_small_faceted_write_runs_both_checks_in_process(tmp_path):
    tris = uv_sphere(30, 16)
    ir, _ = unmesh.convert(tris)
    ir = break_analytic(ir)
    report = step.write(ir, tmp_path / "s.step", mesh=tris)
    assert report.fallback == "faceted" and report.valid
    assert report.verified and report.verified_by == "occt+text"
    assert report.readback.reader == "occt" and report.readback.peak_rss_mb is None
    assert report.text_check.reader == "text" and report.text_check.ok
    assert report.readback_skipped is None
    expected = signed_volume(tris)
    assert report.readback.volume == pytest.approx(expected, rel=1e-9)
    assert report.text_check.volume == pytest.approx(expected, rel=1e-9)
    assert (report.text_check.solids, report.text_check.shells) == (1, 1)


def test_mid_size_faceted_write_reads_back_in_a_child_without_face_orientation_repair(
    sphere, tmp_path, child_readback
):
    ir, tris = sphere
    report = step.write(ir, tmp_path / "s.step", mesh=tris)
    assert report.valid and report.verified_by == "occt+text"
    assert report.readback.reader == "occt_no_face_orientation"
    assert report.readback.peak_rss_mb > 0
    assert report.readback.volume == pytest.approx(signed_volume(tris), rel=1e-9)
    assert report.solids == 1


def test_faceted_write_over_the_memory_estimate_is_verified_by_text(sphere, tmp_path, text_only):
    ir, tris = sphere
    report = step.write(ir, tmp_path / "s.step", mesh=tris, options=text_only)
    assert report.valid and report.verified
    assert report.verified_by == "text"
    assert report.readback is None
    assert "over the 1 MB cap" in report.readback_skipped
    assert report.text_check.ok
    assert report.text_check.volume == pytest.approx(signed_volume(tris), rel=1e-9)
    assert report.shells[0].volume == pytest.approx(signed_volume(tris), rel=1e-9)
    assert report.solids == 1


def test_read_back_timeout_falls_back_to_the_text_check(sphere, tmp_path, child_readback):
    ir, tris = sphere
    options = step.WriteOptions(readback_timeout_s=0.0)
    report = step.write(ir, tmp_path / "s.step", mesh=tris, options=options)
    assert report.valid and report.verified_by == "text"
    assert "timeout" in report.readback_skipped


def test_analytic_write_over_the_memory_cap_is_not_reported_valid(tmp_path, child_readback):
    ir = Ir.loads((FIXTURES / "box.json").read_text())
    options = step.WriteOptions(readback_memory_mb=1.0)
    report = step.write(ir, tmp_path / "b.step", options=options)
    assert report.fallback is None
    assert not report.valid and not report.verified and report.verified_by is None
    assert "1 MB cap" in report.readback_skipped
    assert any(i.startswith("the written file was not verified") for i in report.issues)


def test_analytic_write_past_the_timeout_is_not_reported_valid(tmp_path, child_readback):
    ir = Ir.loads((FIXTURES / "box.json").read_text())
    options = step.WriteOptions(readback_timeout_s=0.0)
    report = step.write(ir, tmp_path / "b.step", options=options)
    assert not report.valid and report.verified_by is None
    assert "timeout" in report.readback_skipped


def test_analytic_write_read_back_in_a_child(tmp_path, child_readback):
    ir = Ir.loads((FIXTURES / "box.json").read_text())
    report = step.write(ir, tmp_path / "b.step")
    assert report.valid and report.verified_by == "occt"
    assert report.readback.reader == "occt" and report.readback.peak_rss_mb > 0
    assert report.readback.volume == pytest.approx(1000.0, rel=1e-9)
    assert report.text_check is None


def test_verify_false_reports_no_verification(sphere, tmp_path):
    ir, tris = sphere
    report = step.write(ir, tmp_path / "s.step", mesh=tris, verify=False)
    assert report.valid and not report.verified
    assert report.verified_by is None and report.text_check is None and report.readback is None


def test_child_read_back_failure_is_an_issue(tmp_path):
    path = tmp_path / "junk.step"
    path.write_text("not a step file\n")
    out = readback.run(path, False, True, 2048.0, 60.0)
    assert out.limit is None
    assert out.error.startswith("could not re-import the file")


def _flip_face(monkeypatch, which, negate_normal):
    real = faceted.shell_faces

    def flipped(vertices, tris, *, outer):
        shell = real(vertices, tris, outer=outer)
        for k in which(len(shell.faces)):
            f = shell.faces[k]
            shell.faces[k] = faceted.Face(f.loop[::-1], -f.normal if negate_normal else f.normal)
        return shell

    monkeypatch.setattr(faceted, "shell_faces", flipped)


@pytest.mark.parametrize("negate_normal", [True, False])
def test_text_check_catches_one_flipped_face(
    sphere, tmp_path, monkeypatch, negate_normal, text_only
):
    _flip_face(monkeypatch, lambda n: [n // 2], negate_normal)
    ir, tris = sphere
    report = step.write(ir, tmp_path / "f.step", mesh=tris, options=text_only)
    assert report.fallback == "faceted" and report.verified_by == "text"
    assert not report.valid and not report.text_check.ok
    expected = "edges are not used once in each direction"
    if not negate_normal:
        expected = "faces wind against their plane's normal"
    assert any(expected in i for i in report.text_check.issues)
    assert any(i.startswith("text check of the written file does not match") for i in report.issues)
    assert not report.shells[0].valid


def test_text_check_catches_an_inverted_solid(sphere, tmp_path, monkeypatch, text_only):
    _flip_face(monkeypatch, range, True)
    ir, tris = sphere
    report = step.write(ir, tmp_path / "i.step", mesh=tris, options=text_only)
    assert not report.valid and report.verified_by == "text"
    assert report.text_check.volume == pytest.approx(-signed_volume(tris), rel=1e-9)
    assert any("non-positive volume" in i for i in report.text_check.issues)


def box_with_cavity():
    from test_step import box_tris, break_first_region

    tris = np.concatenate(
        [box_tris((0.0, 0.0, 0.0), 4.0), box_tris((1.0, 1.0, 1.0), 1.0, inward=True)]
    )
    ir, _ = unmesh.convert(tris)
    return break_first_region(ir), tris


def test_text_check_reads_voids(tmp_path, text_only):
    ir, tris = box_with_cavity()
    report = step.write(ir, tmp_path / "c.step", mesh=tris, options=text_only)
    assert report.valid and report.verified_by == "text"
    assert (report.text_check.solids, report.text_check.shells) == (1, 2)
    assert report.text_check.volume == pytest.approx(63.0, rel=1e-12)


def test_text_check_catches_a_dropped_cavity(tmp_path, monkeypatch, text_only):
    real = faceted.write

    def drop_voids(path, vertices, bodies, open_shells, **kw):
        real(path, vertices, [faceted.Body(b.outer) for b in bodies], open_shells, **kw)

    monkeypatch.setattr(faceted, "write", drop_voids)
    ir, tris = box_with_cavity()
    report = step.write(ir, tmp_path / "d.step", mesh=tris, options=text_only)
    assert not report.valid and report.verified_by == "text"
    assert (report.text_check.shells, report.text_check.expected_shells) == (1, 2)
    assert report.text_check.volume == pytest.approx(64.0, rel=1e-12)


def test_text_check_reads_open_shells(tmp_path, text_only):
    from test_step import box_plus_stray_ir, break_first_region

    ir, box, stray = box_plus_stray_ir()
    report = step.write(
        break_first_region(ir),
        tmp_path / "o.step",
        mesh=np.concatenate([box, stray]),
        options=text_only,
    )
    assert report.valid and report.verified_by == "text"
    assert (report.text_check.solids, report.text_check.shells) == (1, 2)


def _edit(path, old, new, count=1):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, count))


@pytest.fixture
def written(sphere, tmp_path):
    ir, tris = sphere
    path = tmp_path / "s.step"
    assert step.write(ir, path, mesh=tris, verify=False).valid
    assert textcheck.read(path).issues == []
    return path


@pytest.mark.parametrize(
    "old,new,issue",
    [
        ("DATA;\n", "DATA;\n#999999 = B_SPLINE_CURVE_WITH_KNOTS('',1);\n", "not entities"),
        ("DATA;\n", "DATA;\n#999999 = ( FOO_BAR('') BAZ() );\n", "not entities"),
        ("DATA;\n", "DATA;\nGARBAGE GOES HERE;\n", "not entities"),
        ("DATA;\n", "DATA;\n\n", "not entities"),
        ("FACE_OUTER_BOUND('',#", "FACE_BOUND('',#", "not entities"),
        ("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)", "not entities"),
        ("'mechanical'", "'electrical'", "not entities"),
        ("FILE_SCHEMA(('AUTOMOTIVE_DESIGN", "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN", "header"),
        ("END-ISO-10303-21;\n", "", "truncated"),
        (",.T.);\n", ",.F.);\n", ""),
        ("ORIENTED_EDGE('',*,*,#", "ORIENTED_EDGE('',*,*,#1", "refers to"),
        (
            "PRODUCT_DEFINITION_SHAPE('','',#",
            "PRODUCT_DEFINITION_SHAPE('','',#9",
            "definition shape",
        ),
    ],
)
def test_text_check_rejects_what_it_does_not_recognise(written, old, new, issue):
    _edit(written, old, new)
    read = textcheck.read(written)
    problems = read.issues + [i for s in read.solids for i in s.issues]
    assert problems
    assert any(issue in p for p in problems)


def test_text_check_handles_lines_longer_than_a_chunk(tmp_path, monkeypatch):
    ir, tris = box_with_cavity()
    path = tmp_path / "c.step"
    step.write(ir, path, mesh=tris, verify=False)
    whole = textcheck.read(path)
    monkeypatch.setattr(textcheck, "CHUNK", 64)
    read = textcheck.read(path)
    assert read.issues == [] and read.solids[0].valid
    assert read.solids[0].volume == whole.solids[0].volume


def test_text_check_rejects_a_truncated_file(written):
    data = written.read_bytes()
    written.write_bytes(data[: len(data) // 2])
    assert any("truncated" in i for i in textcheck.read(written).issues)


def test_text_check_rejects_a_file_without_its_shape_definition(written):
    lines = written.read_text().splitlines(keepends=True)
    written.write_text("".join(x for x in lines if "SHAPE_DEFINITION_REPRESENTATION" not in x))
    assert any("shape definition" in i for i in textcheck.read(written).issues)


def test_text_check_rejects_a_dangling_reference(written):
    text = written.read_text()
    last = max(int(x.split(" = ")[0][1:]) for x in text.splitlines() if x.startswith("#"))
    written.write_text(
        text.replace("MANIFOLD_SOLID_BREP('',#", f"MANIFOLD_SOLID_BREP('',#{last}0", 1)
    )
    assert any("not in the file" in i for i in textcheck.read(written).issues)


def test_text_check_rejects_a_loop_through_a_vertex_twice(written):
    import re

    text = written.read_text()
    m = re.search(r"EDGE_LOOP\('',\((#\d+),(#\d+),(#\d+)\)\)", text)
    a, b, c = m.groups()
    twice = f"EDGE_LOOP('',({a},{b},{c},{a},{b},{c}))"
    written.write_text(text.replace(m.group(0), twice, 1))
    read = textcheck.read(written)
    assert read.issues


def test_a_parent_side_exception_kills_the_child(written, monkeypatch):
    import subprocess

    started = []
    real = subprocess.Popen

    def record(*a, **kw):
        started.append(real(*a, **kw))
        return started[-1]

    def boom(proc, memory_mb, timeout_s):
        raise RuntimeError("parent failed while watching")

    monkeypatch.setattr(subprocess, "Popen", record)
    monkeypatch.setattr(readback, "_watch", boom)
    with pytest.raises(RuntimeError):
        readback.run(written, False, False, 2048.0, 60.0)
    assert started and started[0].returncode is not None
