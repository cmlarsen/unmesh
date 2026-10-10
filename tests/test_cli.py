import json
import math
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

import unmesh
from unmesh.cli import main

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "fidelity.schema.json").read_text())
EXIT = {"analytic": 0, "mixed": 1, "faceted": 2, "error": 3}


def quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def box(lo=(0.0, 0.0, 0.0), hi=(10.0, 8.0, 5.0)):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    p = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return np.array([[p[i] for i in t] for f in faces for t in quad(*f)], dtype=float)


def cylinder(r=10.0, h=10.0, n=96, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ang = 2 * math.pi * np.arange(n) / n
    radial = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1)
    bottom = r * radial
    top = bottom + [0.0, 0.0, h]
    if noise:
        bottom = bottom + radial * rng.uniform(-noise, noise, (n, 1))
        top = top + radial * rng.uniform(-noise, noise, (n, 1))
    cb, ct = np.zeros(3), np.array([0.0, 0.0, h])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(bottom[i], bottom[j], top[j]), (bottom[i], top[j], top[i])]
        tris += [(cb, bottom[j], bottom[i]), (ct, top[i], top[j])]
    return np.array(tris, dtype=float)


def bumped_box(n=8, size=10.0, h=5.0, amp=0.5, inner=1):
    s = size / n
    z = np.full((n + 1, n + 1), h)
    m = n - 2 * inner
    for i in range(inner, n - inner + 1):
        for j in range(inner, n - inner + 1):
            bump = math.sin(math.pi * (i - inner) / m) * math.sin(math.pi * (j - inner) / m)
            z[i, j] = h + amp * bump

    def node(i, j):
        return (i * s, j * s, z[i, j])

    tris = []
    for i in range(n):
        for j in range(n):
            tris += quad(node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1))
    edges = [
        [(i, 0) for i in range(n + 1)],
        [(n, j) for j in range(n + 1)],
        [(i, n) for i in range(n, -1, -1)],
        [(0, j) for j in range(n, -1, -1)],
    ]
    for e in edges:
        a = (e[0][0] * s, e[0][1] * s, 0.0)
        b = (e[-1][0] * s, e[-1][1] * s, 0.0)
        for k in range(n):
            tris.append((a if k < n // 2 else b, node(*e[k + 1]), node(*e[k])))
        tris.append((a, b, node(*e[n // 2])))
    tris += quad((0, 0, 0), (0, size, 0), (size, size, 0), (size, 0, 0))
    return np.array(tris, dtype=float)


def run(tmp_path, tris, *extra, name="part"):
    stl = tmp_path / f"{name}.stl"
    unmesh.write_stl(stl, tris)
    out = tmp_path / f"{name}.step"
    report = tmp_path / f"{name}.json"
    code = main(["convert", str(stl), str(out), "--report", str(report), *extra])
    data = json.loads(report.read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["exit_code"] == code == EXIT[data["outcome"]]
    return code, data, out


def test_planar_part_is_all_analytic(tmp_path):
    code, data, out = run(tmp_path, box())
    assert code == 0
    assert data["region_counts"] == {"plane": 6}
    assert out.stat().st_size > 0 and data["output"]["written"]
    assert data["faceted"] == []
    v = data["validity"]
    assert v["valid"] and v["verified"] and v["readback"]["ok"]
    assert v["volume_checked_against_input"]
    assert all(f["written_as"] == "plane" and f["writer"] is not None for f in data["faces"])
    measured = data["deviation"]["measured"]
    assert measured["max"] < 1e-9
    assert measured["step_to_input"]["samples"] > 0
    assert set(data["runtime_s"]) == {"read", "convert", "write", "readback", "measure", "total"}


def test_curved_part_reports_measured_deviation_within_the_bound(tmp_path):
    code, data, _ = run(tmp_path, cylinder())
    assert code == 0
    assert data["region_counts"] == {"cylinder": 1, "plane": 2}
    bound = data["deviation"]["converter"]["max"]
    sagitta = 10.0 * (1 - math.cos(math.pi / 96))
    assert bound >= sagitta * (1 - 1e-6)
    assert data["deviation"]["measured"]["step_to_input"]["max"] <= bound * (1 + 1e-6)
    cyl = next(f for f in data["faces"] if f["surface"] == "cylinder")
    assert cyl["converter"]["max"] >= sagitta * (1 - 1e-6)


def test_noisy_part_converts_and_reports_honestly(tmp_path):
    code, data, _ = run(tmp_path, cylinder(noise=1e-3, seed=3))
    assert code in (0, 1)
    assert data["validity"]["valid"]
    measured = data["deviation"]["measured"]
    assert measured["step_to_input"]["max"] <= data["deviation"]["converter"]["max"] * (1 + 1e-6)


def test_unfittable_patch_gives_mixed_exit_code(tmp_path):
    code, data, _ = run(tmp_path, bumped_box())
    assert code == 1
    assert data["outcome"] == "mixed"
    facets = [f for f in data["faces"] if f["surface"] == "facets"]
    assert facets and all(f["written_as"] == "triangles" for f in facets)
    assert {(f["region"], "no_surface_fit") for f in facets} == {
        (f["region"], f["reason"]) for f in data["faceted"]
    }


def test_mostly_unfittable_part_falls_back_whole(tmp_path):
    code, data, _ = run(tmp_path, bumped_box(inner=0))
    assert code == 2
    assert data["validity"]["fallback"] == "faceted"
    assert data["validity"]["valid"]
    reasons = {f["reason"] for f in data["faceted"]}
    assert "facets_share" in reasons
    assert len(data["faceted"]) == len(data["faces"])


def test_open_mesh_is_faceted(tmp_path):
    code, data, _ = run(tmp_path, box()[:-2])
    assert code == 2
    assert [f["reason"] for f in data["faceted"]] == ["open_shell"]
    assert data["validity"]["fallback"] is None
    assert data["validity"]["open_shells"] == [0]


def test_open_mesh_fidelity_reports_open_boundary(tmp_path):
    code, data, _ = run(tmp_path, box()[:-2])
    assert code == 2
    boundary = data["validity"]["open_boundary"]
    assert boundary["edges"] == 4
    assert boundary["length"] == pytest.approx(26.0, rel=1e-9)
    assert boundary["non_manifold_edges"] == 0


def test_inch_input_is_written_in_mm(tmp_path):
    from unmesh._writer import occ

    code, data, out = run(tmp_path, box(hi=(1.0, 1.0, 1.0)), "--unit", "in")
    assert code == 0
    assert data["input"]["unit"] == "in" and data["input"]["scale_to_mm"] == 25.4
    solids, _ = occ.read_back(out)
    assert solids[0].volume == pytest.approx(25.4**3, rel=1e-9)


def test_tolerance_is_in_input_units(tmp_path):
    _, data, _ = run(tmp_path, box(hi=(1.0, 1.0, 1.0)), "--unit", "in", "--tolerance", "1e-4")
    assert data["tolerances"]["linear"] == pytest.approx(25.4e-4)


def test_no_measure_leaves_measured_null(tmp_path):
    _, data, _ = run(tmp_path, box(), "--no-measure")
    assert data["deviation"]["measured"] is None
    assert data["runtime_s"]["measure"] is None


def test_obj_input(tmp_path):
    tris = box()
    v, f = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    lines = [f"v {x} {y} {z}" for x, y, z in v]
    lines += ["f " + " ".join(f"{i + 1}/1/1" for i in t) for t in f.reshape(-1, 3)]
    path = tmp_path / "box.obj"
    path.write_text("# box\n" + "\n".join(lines) + "\n")
    assert np.allclose(unmesh.read_mesh(path), tris)
    out = tmp_path / "box.step"
    assert main(["convert", str(path), str(out)]) == 0


def test_obj_quads_and_negative_indices(tmp_path):
    path = tmp_path / "q.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n")
    tris = unmesh.read_mesh(path)
    assert tris.shape == (2, 3, 3)
    assert np.allclose(tris[1], [[0, 0, 0], [1, 1, 0], [0, 1, 0]])


@pytest.mark.parametrize(
    "content",
    [b"solid nothing\nendsolid nothing\n", b"\x00" * 10],
)
def test_bad_input_is_an_error(tmp_path, content, capsys):
    stl = tmp_path / "bad.stl"
    stl.write_bytes(content)
    report = tmp_path / "r.json"
    code = main(["convert", str(stl), str(tmp_path / "o.step"), "--report", str(report)])
    assert code == 3
    data = json.loads(report.read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["outcome"] == "error" and data["error"]["message"]
    assert not (tmp_path / "o.step").exists()
    assert "error" in capsys.readouterr().err


def test_missing_input_and_bad_options_are_errors(tmp_path):
    assert main(["convert", str(tmp_path / "none.stl"), str(tmp_path / "o.step")]) == 3
    stl = tmp_path / "b.stl"
    unmesh.write_stl(stl, box())
    assert main(["convert", str(stl), str(tmp_path / "o.step"), "--tolerance", "-1"]) == 3
    with pytest.raises(SystemExit) as e:
        main(["convert", str(stl), str(tmp_path / "o.step"), "--unit", "ft"])
    assert e.value.code == 3


def test_version():
    result = subprocess.run(
        [sys.executable, "-m", "unmesh", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"unmesh {unmesh.__version__}"


def test_missing_ocp_is_an_error_with_install_hint(tmp_path):
    stl = tmp_path / "b.stl"
    unmesh.write_stl(stl, box())
    report = tmp_path / "r.json"
    code = (
        "import sys\n"
        "sys.modules['OCP'] = None\n"
        "from unmesh.cli import main\n"
        f"sys.exit(main(['convert', {str(stl)!r}, {str(tmp_path / 'o.step')!r},"
        f" '--report', {str(report)!r}]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert "pip install unmesh[step]" in result.stderr
    data = json.loads(report.read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["error"]["type"] == "ImportError"
    assert not (tmp_path / "o.step").exists()


def test_convert_to_step_api(tmp_path):
    conversion = unmesh.convert_to_step(cylinder(), tmp_path / "c.step", measure=False)
    assert conversion.outcome == "analytic" and conversion.exit_code == 0
    assert conversion.write.valid
    jsonschema.validate(conversion.fidelity, SCHEMA)
    with pytest.raises(ValueError):
        unmesh.convert_to_step(box(), tmp_path / "x.step", unit="ft")


def test_unexpected_exception_exits_3_and_leaves_no_step(tmp_path, monkeypatch):
    import unmesh.step

    real = unmesh.step.write

    def write_then_fail(ir, path, *args, **kwargs):
        real(ir, path, *args, **kwargs)
        assert Path(path).is_file()
        raise RuntimeError("injected")

    monkeypatch.setattr(unmesh.step, "write", write_then_fail)
    stl = tmp_path / "b.stl"
    unmesh.write_stl(stl, box())
    out = tmp_path / "o.step"
    report = tmp_path / "r.json"
    assert main(["convert", str(stl), str(out), "--report", str(report)]) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.stl", "r.json"]
    data = json.loads(report.read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["error"] == {"type": "RuntimeError", "message": "injected"}
    assert data["output"] == {"path": str(out), "written": False}


def test_existing_output_survives_a_failed_run(tmp_path):
    out = tmp_path / "o.step"
    out.write_text("previous")
    stl = tmp_path / "bad.stl"
    stl.write_bytes(b"\x00" * 10)
    assert main(["convert", str(stl), str(out)]) == 3
    assert out.read_text() == "previous"
