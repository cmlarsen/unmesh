import copy
import json
import math
import pathlib
import random
import shutil
import subprocess
import sys

import jsonschema
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "fixtures" / "ir"
sys.path.insert(0, str(FIXTURE_DIR))

import build  # noqa: E402
import geometry  # noqa: E402

from unmesh.ir import (  # noqa: E402
    Facets,
    Ir,
    IrError,
    Plane,
    canonical_json,
    format_float,
    validate,
)

SCHEMA = json.loads((ROOT / "docs" / "ir.schema.json").read_text())
NAMES = sorted(build.FIXTURES)


def load(name):
    return (FIXTURE_DIR / f"{name}.json").read_text()


def test_fixture_set():
    assert {
        "box",
        "plate_with_bore",
        "box_fillet",
        "plate_chamfer",
        "countersink",
        "disc_fillet",
        "mixed_facets",
        "two_bodies",
    } <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_schema(name):
    jsonschema.validate(json.loads(load(name)), SCHEMA)


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_is_byte_identical(name):
    text = load(name)
    assert Ir.loads(text.rstrip("\n")).dumps() + "\n" == text


def assert_close(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_close(x, y, f"{path}/{i}")
    elif isinstance(a, float):
        assert a == pytest.approx(b, abs=1e-9), path
    else:
        assert a == b, path


@pytest.mark.parametrize("name", NAMES)
def test_fixture_is_regenerated(name):
    assert_close(build.FIXTURES[name]().to_dict(), json.loads(load(name)))


@pytest.mark.parametrize("name", NAMES)
def test_validates(name):
    assert validate(Ir.loads(load(name))) == []


def segments(ir, a, b):
    for adj in ir.adjacencies:
        if adj.regions == (a, b):
            return adj.boundaries
    return []


@pytest.mark.parametrize("name", NAMES)
def test_boundaries_lie_on_both_surfaces(name):
    ir = Ir.loads(load(name))
    tol = ir.tolerances.linear * 0.01
    for adj in ir.adjacencies:
        for bd in adj.boundaries:
            for r in adj.regions:
                s = ir.regions[r].surface
                for p in bd.points:
                    if isinstance(s, Facets):
                        d = min(math.dist(p, v) for v in s.vertices)
                        assert d <= ir.tolerances.vertex_merge, (name, adj.regions, p)
                    else:
                        assert geometry.distance(s, p) <= tol, (name, adj.regions, p)


@pytest.mark.parametrize("name", NAMES)
def test_dihedral_and_kind_match_surfaces(name):
    ir = Ir.loads(load(name))
    for adj in ir.adjacencies:
        a, b = (ir.regions[i].surface for i in adj.regions)
        for bd in adj.boundaries:
            deg = geometry.dihedral_deg(a, b, geometry.sample_points(bd.points, bd.closed))
            assert deg == pytest.approx(bd.dihedral_deg, abs=1e-5)
            assert (bd.kind == "tangent") == (deg < ir.tolerances.tangent_threshold_deg)


@pytest.mark.parametrize("name", NAMES)
def test_vertices_match_source_positions(name):
    ir = Ir.loads(load(name))
    for v in ir.vertices:
        assert any(
            math.dist(v.position, s) <= ir.tolerances.vertex_merge for s in v.source_positions
        )
        for r in v.regions:
            s = ir.regions[r].surface
            if isinstance(s, Facets):
                assert any(math.dist(v.position, x) <= 1e-9 for x in s.vertices)
            else:
                assert geometry.distance(s, v.position) <= ir.tolerances.linear * 0.01


def region_loop_vector_area(ir, region_id):
    total = np.zeros(3)
    for adj in ir.adjacencies:
        if region_id not in adj.regions:
            continue
        for bd in adj.boundaries:
            pts = np.asarray(bd.points)
            if region_id != adj.regions[0]:
                pts = pts[::-1]
            nxt = np.roll(pts, -1, axis=0) if bd.closed else pts[1:]
            cur = pts if bd.closed else pts[:-1]
            total += 0.5 * np.cross(cur, nxt).sum(axis=0)
    return total


@pytest.mark.parametrize("name", [n for n in NAMES if n != "open_shell"])
def test_planar_loops_run_counter_clockwise_about_the_outward_normal(name):
    ir = Ir.loads(load(name))
    for r in ir.regions:
        if isinstance(r.surface, Plane):
            area = region_loop_vector_area(ir, r.id)
            assert area @ np.asarray(r.surface.normal) > 0, (name, r.id)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "open_shell"])
def test_every_polyline_is_used_twice_with_opposite_direction(name):
    ir = Ir.loads(load(name))
    total = sum((region_loop_vector_area(ir, r.id) for r in ir.regions), start=np.zeros(3))
    assert np.allclose(total, 0.0, atol=1e-9)


@pytest.mark.parametrize("name", ["box", "two_bodies"])
def test_convex_planar_edges_run_along_na_cross_nb(name):
    ir = Ir.loads(load(name))
    for adj in ir.adjacencies:
        na, nb = (np.asarray(ir.regions[r].surface.normal) for r in adj.regions)
        for bd in adj.boundaries:
            t = np.asarray(bd.points[-1]) - np.asarray(bd.points[0])
            assert np.cross(na, nb) @ t > 0


def test_compound_has_one_shell_per_body():
    ir = Ir.loads(load("two_bodies"))
    assert [len(s.regions) for s in ir.shells] == [6, 6]
    assert all(s.closed for s in ir.shells)
    for adj in ir.adjacencies:
        shells = {i for i, s in enumerate(ir.shells) for r in adj.regions if r in s.regions}
        assert len(shells) == 1


def test_open_shell_is_one_facets_region():
    ir = Ir.loads(load("open_shell"))
    assert [s.closed for s in ir.shells] == [False]
    assert isinstance(ir.regions[0].surface, Facets)


def test_mixed_part_has_exactly_one_facets_region():
    ir = Ir.loads(load("mixed_facets"))
    assert sum(isinstance(r.surface, Facets) for r in ir.regions) == 1


def test_kinds_in_curved_fixtures():
    kinds = {
        name: {bd.kind for adj in Ir.loads(load(name)).adjacencies for bd in adj.boundaries}
        for name in ("box", "box_fillet", "disc_fillet", "plate_chamfer", "countersink")
    }
    assert kinds["box"] == {"transversal"}
    assert kinds["plate_chamfer"] == {"transversal"}
    assert kinds["countersink"] == {"transversal"}
    assert kinds["box_fillet"] == {"transversal", "tangent"}
    assert kinds["disc_fillet"] == {"tangent", "transversal"}


def mutate(name, fn):
    d = json.loads(load(name))
    fn(d)
    return d


def test_validation_rejects_broken_ir():
    cases = [
        lambda d: d["regions"][0].update(id=9),
        lambda d: d["adjacencies"][0].update(regions=[3, 1]),
        lambda d: d["adjacencies"][0]["boundaries"][0].update(kind="tangent"),
        lambda d: d["vertices"][0].update(regions=[0, 1]),
        lambda d: d["regions"][0]["surface"].update(normal=[0.0, 0.0, 2.0]),
        lambda d: d["shells"][0].update(regions=[0, 1, 2, 3, 4]),
        lambda d: d["adjacencies"][0]["boundaries"][0]["points"].__setitem__(0, [9.0, 9.0, 9.0]),
        lambda d: d["regions"][1]["triangles"].append(d["regions"][0]["triangles"][0]),
    ]
    for case in cases:
        with pytest.raises(IrError):
            Ir.from_dict(mutate("box", case))


def test_unknown_keys_rejected():
    with pytest.raises(IrError):
        Ir.from_dict(mutate("box", lambda d: d.update(extra=1)))


def test_facets_facets_adjacency_rejected():
    d = json.loads(load("mixed_facets"))
    ir = Ir.from_dict(copy.deepcopy(d))
    patch = next(r for r in ir.regions if isinstance(r.surface, Facets))
    neighbour = next(a.regions[0] for a in ir.adjacencies if patch.id in a.regions)
    ir.regions[neighbour].surface = Facets(list(patch.surface.vertices), list(patch.surface.faces))
    ir.regions[neighbour].residual = None
    assert any("two facets" in e for e in validate(ir))


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (0.0, "0.0"),
        (-0.0, "0.0"),
        (1.0, "1.0"),
        (-2.5, "-2.5"),
        (100.0, "100.0"),
        (0.001, "0.001"),
        (1e-5, "0.00001"),
        (1e-6, "1e-6"),
        (1.5e-7, "1.5e-7"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e16"),
        (1.25e17, "1.25e17"),
        (0.1 + 0.2, "0.30000000000000004"),
        (123.456, "123.456"),
        (5e-324, "5e-324"),
        (686036261402521.25, "686036261402521.2"),
        (-686036261402521.75, "-686036261402521.8"),
    ],
)
def test_float_format(x, text):
    assert format_float(x) == text
    assert float(text) == x


def test_canonical_json_sorts_keys_and_is_compact():
    assert canonical_json({"b": [1, 2.0, None], "a": True}) == '{"a":true,"b":[1,2.0,null]}'
    assert canonical_json("é\n") == '"\\u00e9\\u000a"'


def test_non_finite_rejected():
    for x in (math.inf, math.nan):
        with pytest.raises(IrError):
            format_float(x)


needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not on PATH")


def rust_canon(text):
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "unmesh-core", "--example", "ir_canon"],
        input=text,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return out.stdout


@needs_cargo
@pytest.mark.parametrize("name", NAMES)
def test_rust_canonicalizes_fixtures_identically(name):
    text = load(name)
    assert rust_canon(text) + "\n" == text
    pretty = json.dumps(json.loads(text), indent=2)
    assert rust_canon(pretty) == Ir.loads(text).dumps()


@needs_cargo
def test_rust_and_python_format_random_floats_identically():
    rng = random.Random(7)
    values = [
        0.1,
        0.2,
        1 / 3,
        2 / 3,
        1e22,
        1e-7,
        123456789.123456789,
        5e-324,
        1.7976931348623157e308,
    ]
    for _ in range(3000):
        values.append(rng.choice((-1, 1)) * rng.random() * 10 ** rng.randint(-30, 30))
    for _ in range(500):
        values.append(float(np.float32(rng.uniform(-100, 100))))
    verts = [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values) - 2, 3)]
    n = len(verts)
    ir = Ir.from_dict(
        {
            "ir_version": 0,
            "tolerances": {
                "linear": 0.001,
                "angular_snap_deg": 0.5,
                "tangent_threshold_deg": 3.0,
                "vertex_merge": 1e-06,
            },
            "shells": [{"closed": False, "regions": [0]}],
            "regions": [
                {
                    "id": 0,
                    "surface": {
                        "type": "facets",
                        "vertices": [list(v) for v in verts],
                        "faces": [[0, 1, 2]] * 1,
                    },
                    "triangles": [0],
                    "residual": None,
                }
            ],
            "adjacencies": [],
            "vertices": [],
        }
    )
    assert n > 1000
    assert rust_canon(ir.dumps()) == ir.dumps()
