import copy

import numpy as np
import pytest

import unmesh
from unmesh_harness import degrade
from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.degrade import OPERATORS
from unmesh_harness.degrade.faults import DISPOSITION
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import DEFLECTION_SETTINGS, LabeledMesh, tessellate

from .test_labels import closed_manifold_problems

LIN, ANG = DEFLECTION_SETTINGS[0]
SEED = 11

EXPECTED: dict[str, bool] = {
    "unwelded_corners": False,
    "unwelded_gap": False,
    "crack_seam": False,
    "flipped_facets": False,
    "duplicate_facets": False,
    "hole_patch": False,
    "stray_shells": False,
    "nonmanifold_fin": False,
}
RNG_OPS = {
    "unwelded_corners",
    "unwelded_gap",
    "crack_seam",
    "flipped_facets",
    "duplicate_facets",
    "hole_patch",
    "stray_shells",
    "nonmanifold_fin",
}


@pytest.fixture(scope="module")
def part():
    entry = select(load_manifest(), "smoke")[0]
    return tessellate(generate(entry["family"], entry["seed"]).solid, LIN, ANG)


@pytest.fixture(scope="module")
def curved_part():
    return tessellate(generate("revolved_dome", 0).solid, LIN, ANG)


@pytest.fixture(scope="module")
def degenerate_part():
    return tessellate(generate("corner_fillet", 0).solid, LIN, ANG)


def codes(result) -> list[str]:
    return [w.code for w in result.report.warnings]


def test_registry_covers_defect_operators():
    assert set(EXPECTED) <= set(OPERATORS)
    assert set(EXPECTED) <= set(DISPOSITION)
    for name, watertight in EXPECTED.items():
        op = OPERATORS[name]
        assert op.family == "defect"
        assert op.severity_0 and op.severity_1
        assert op.preserves_watertight == watertight
        assert not op.changes_frame
        assert DISPOSITION[name].startswith(("REPAIR", "REPORT"))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_severity_zero_is_identity(name, part):
    out = degrade.apply(name, part, 0.0, SEED)
    assert np.array_equal(out.tris, part.tris)
    assert np.array_equal(out.face_id, part.face_id)
    assert out.metadata["history"][-1]["op"] == name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_deterministic_and_pure(name, part):
    snapshot = part.tris.copy()
    before = copy.deepcopy(part.metadata)
    a = degrade.apply(name, part, 0.7, SEED)
    b = degrade.apply(name, part, 0.7, SEED)
    assert np.array_equal(a.tris, b.tris)
    assert np.array_equal(a.face_id, b.face_id)
    assert a.metadata == b.metadata
    assert np.array_equal(part.tris, snapshot)
    assert part.metadata == before
    assert a.metadata["history"][-1]["params"]
    assert "truth" in a.metadata


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_tables_untouched_and_labels_in_domain(name, part):
    out = degrade.apply(name, part, 0.6, SEED)
    assert [f.__dict__ for f in out.faces] == [f.__dict__ for f in part.faces]
    assert [a.__dict__ for a in out.adjacency] == [a.__dict__ for a in part.adjacency]
    assert [s.__dict__ for s in out.shells] == [s.__dict__ for s in part.shells]
    assert out.vertices == part.vertices
    assert len(out.face_id) == len(out.tris)
    assert set(out.face_id.tolist()) <= set(part.face_id.tolist()) | {-1}


@pytest.mark.parametrize("name", sorted(RNG_OPS))
def test_seed_changes_output(name, part):
    a = degrade.apply(name, part, 0.7, 1)
    b = degrade.apply(name, part, 0.7, 2)
    assert not np.array_equal(a.tris, b.tris)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_history_params_survive_save_load(name, part, tmp_path):
    out = degrade.apply(name, part, 0.7, SEED)
    out.save(tmp_path / "m.npz")
    back = LabeledMesh.load(tmp_path / "m.npz")
    assert back.metadata == out.metadata
    assert np.array_equal(back.tris, out.tris)
    assert np.array_equal(back.face_id, out.face_id)


def test_unwelded_corners_moves_split_copies_below_weld(part):
    out = degrade.apply("unwelded_corners", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["vertices_split"] == params["distinct_vertices"] > 0
    assert params["moved_corners"] > 0
    assert params["displacement_mm"] == pytest.approx(0.5e-6)
    assert not np.array_equal(out.tris, part.tris)
    assert np.array_equal(out.face_id, part.face_id)
    peak = np.linalg.norm(out.tris - part.tris, axis=2).max()
    assert peak == pytest.approx(params["displacement_mm"])
    assert closed_manifold_problems(out.tris)[0] != []


def test_unwelded_gap_moves_split_copies_outside_weld(part):
    out = degrade.apply("unwelded_gap", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["vertices_split"] == params["distinct_vertices"] > 0
    assert params["displacement_mm"] == pytest.approx(1e-2)
    assert not np.array_equal(out.tris, part.tris)
    assert np.array_equal(out.face_id, part.face_id)
    peak = np.linalg.norm(out.tris - part.tris, axis=2).max()
    assert peak == pytest.approx(params["displacement_mm"])
    assert closed_manifold_problems(out.tris)[0] != []


@pytest.mark.parametrize("severity", [0.3, 1.0])
def test_unwelded_operators_differ_from_input(severity, part):
    for name in ("unwelded_corners", "unwelded_gap"):
        out = degrade.apply(name, part, severity, SEED)
        assert not np.array_equal(out.tris, part.tris)


def test_unwelded_gap_convert_reports_open_edges(part):
    out = degrade.apply("unwelded_gap", part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert "open_edges" in codes(got)


def test_unwelded_convert_matches_clean(part):
    clean = unmesh.convert(part.tris)
    out = degrade.apply("unwelded_corners", part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert got.ir.dumps() == clean.ir.dumps()
    assert codes(got) == codes(clean) == []


def test_crack_opens_a_seam_of_the_given_width(part):
    out = degrade.apply("crack_seam", part, 0.5, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["width_mm"] == pytest.approx(0.1)
    assert 1 <= params["seam_edges"] <= 8
    assert np.array_equal(out.face_id, part.face_id)
    peak = np.linalg.norm(out.tris - part.tris, axis=2).max()
    assert peak == pytest.approx(params["width_mm"])
    assert closed_manifold_problems(out.tris)[0] != []


def test_flipped_reverses_the_documented_rows(part):
    out = degrade.apply("flipped_facets", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["flipped"] == 8
    assert np.array_equal(out.face_id, part.face_id)
    reversed_rows = [
        i
        for i in range(len(part.tris))
        if np.array_equal(out.tris[i], part.tris[i][::-1])
        and not np.array_equal(part.tris[i], part.tris[i][::-1])
    ]
    assert len(reversed_rows) == params["flipped"]
    for i in range(len(part.tris)):
        assert np.array_equal(out.tris[i], part.tris[i]) or np.array_equal(
            out.tris[i], part.tris[i][::-1]
        )
    assert closed_manifold_problems(out.tris)[0] != []


def test_flipped_convert_repairs_winding(part):
    clean = unmesh.convert(part.tris)
    out = degrade.apply("flipped_facets", part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert "repaired_winding" in codes(got)
    assert [(s.closed, s.role) for s in got.ir.shells] == [
        (s.closed, s.role) for s in clean.ir.shells
    ]
    clean_sets = {tuple(sorted(r.triangles)) for r in clean.ir.regions}
    got_sets = {tuple(sorted(r.triangles)) for r in got.ir.regions}
    assert got_sets == clean_sets
    clean_surf = {tuple(sorted(r.triangles)): r.surface for r in clean.ir.regions}
    got_surf = {tuple(sorted(r.triangles)): r.surface for r in got.ir.regions}
    for key in clean_sets:
        a, b = clean_surf[key], got_surf[key]
        assert a.type == b.type == "plane"
        assert abs(abs(np.dot(a.normal, b.normal)) - 1.0) < 1e-9
        assert abs(np.dot(np.subtract(b.origin, a.origin), a.normal)) < 1e-9


@pytest.mark.xfail(strict=False, reason="#67; passes or fails with the platform's tessellation")
def test_flipped_convert_curved_matches_clean(curved_part):
    clean = unmesh.convert(curved_part.tris)
    out = degrade.apply("flipped_facets", curved_part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert "repaired_winding" in codes(got)
    clean_sets = {tuple(sorted(r.triangles)) for r in clean.ir.regions}
    got_sets = {tuple(sorted(r.triangles)) for r in got.ir.regions}
    assert got_sets == clean_sets


def test_crack_moves_only_one_side_of_the_chain(part):
    out = degrade.apply("crack_seam", part, 0.5, SEED)
    params = out.metadata["history"][-1]["params"]
    moved = out.tris != part.tris
    assert moved.any()
    seam = {
        tuple(part.tris[t, ci])
        for t in range(len(part.tris))
        for ci in range(3)
        if tuple(out.tris[t, ci]) != tuple(part.tris[t, ci])
    }
    for r in range(len(part.tris)):
        if all(tuple(part.tris[r, ci]) not in seam for ci in range(3)):
            assert np.array_equal(out.tris[r], part.tris[r])
    assert any(
        np.array_equal(out.tris[r], part.tris[r])
        and any(tuple(part.tris[r, ci]) in seam for ci in range(3))
        for r in range(len(part.tris))
    )
    steps = (out.tris - part.tris)[np.linalg.norm(out.tris - part.tris, axis=2) > 0]
    assert len(steps) > 0
    assert np.allclose(steps, steps[0])
    assert float(np.linalg.norm(steps[0])) == pytest.approx(params["width_mm"])


def test_crack_leaves_a_single_boundary_loop(part):
    out = degrade.apply("crack_seam", part, 0.5, SEED)
    _, idx, *_ = unmesh.weld(out.tris, 0.0)
    directed: dict[tuple[int, int], int] = {}
    for a, b, c in idx.tolist():
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    boundary = {frozenset(e) for e in directed if (e[1], e[0]) not in directed}
    assert boundary
    degrees: dict[int, int] = {}
    for e in boundary:
        for v in e:
            degrees[v] = degrees.get(v, 0) + 1
    assert set(degrees.values()) == {2}
    parent = {v: v for e in boundary for v in e}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e in boundary:
        a, b = tuple(e)
        parent[find(a)] = find(b)
    assert len({find(v) for v in parent}) == 1


def test_crack_stays_within_one_face(part):
    out = degrade.apply("crack_seam", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    moved = np.linalg.norm(out.tris - part.tris, axis=2) > 0
    rows = np.where(moved.any(axis=1))[0]
    assert len(rows) > 0
    assert (part.face_id[rows] == params["face"]).all()


@pytest.mark.parametrize("seed", range(1, 9))
@pytest.mark.parametrize("mesh", ["curved_part", "degenerate_part"])
def test_crack_tolerates_degenerate_triangles(mesh, seed, request):
    target = request.getfixturevalue(mesh)
    out = degrade.apply("crack_seam", target, 0.5, seed)
    assert len(out.tris) == len(target.tris)
    assert out.metadata["history"][-1]["params"]["seam_edges"] >= 1


def test_crack_convert_reports_an_open_shell(part):
    out = degrade.apply("crack_seam", part, 0.5, SEED)
    got = unmesh.convert(out.tris)
    assert "open_edges" in codes(got)
    assert [(s.closed, s.role) for s in got.ir.shells] == [(False, "outer")]
    assert {r.surface.type for r in got.ir.regions} == {"facets"}
    assert sum(len(r.triangles) for r in got.ir.regions) == len(out.tris)


def test_duplicate_appends_same_or_reversed_copies(part):
    out = degrade.apply("duplicate_facets", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["duplicated"] == params["same_winding"] + params["opposite_winding"] > 0
    assert len(out.tris) == len(part.tris) + params["duplicated"]
    assert set(out.face_id.tolist()) == set(part.face_id.tolist())
    for row, fid in zip(out.tris[len(part.tris) :], out.face_id[len(part.tris) :], strict=True):
        src = part.tris[part.face_id == fid]
        assert any(np.array_equal(row, s) or np.array_equal(row, s[::-1]) for s in src)


def test_duplicate_convert_repairs_to_clean_regions(part):
    clean = unmesh.convert(part.tris)
    out = degrade.apply("duplicate_facets", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    got = unmesh.convert(out.tris)
    if params["same_winding"]:
        assert "degenerate_triangles" in codes(got)
    if params["opposite_winding"]:
        assert "repaired_winding" in codes(got)
    clean_sets = {tuple(sorted(r.triangles)) for r in clean.ir.regions}
    got_sets = {tuple(sorted(r.triangles)) for r in got.ir.regions}
    assert got_sets == clean_sets


def test_hole_removes_a_connected_patch_of_one_face(part):
    out = degrade.apply("hole_patch", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert 1 <= params["removed"] <= 8
    assert len(out.tris) == len(part.tris) - params["removed"]
    assert set(out.face_id.tolist()) <= set(part.face_id.tolist())
    assert closed_manifold_problems(out.tris)[0] != []


def test_hole_convert_reports_an_open_shell(part):
    out = degrade.apply("hole_patch", part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert "open_edges" in codes(got)
    assert [(s.closed, s.role) for s in got.ir.shells] == [(False, "outer")]
    assert [r.surface.type for r in got.ir.regions] == ["facets"]
    assert got.report.region_counts == {"facets": 1}


def test_stray_labels_new_triangles_unknown(part):
    out = degrade.apply("stray_shells", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert 1 <= params["shells_added"] <= 4
    assert len(out.tris) == len(part.tris) + params["triangles_added"]
    assert np.array_equal(out.tris[: len(part.tris)], part.tris)
    assert np.array_equal(out.face_id[: len(part.tris)], part.face_id)
    assert set(out.face_id[len(part.tris) :].tolist()) == {-1}
    assert closed_manifold_problems(out.tris)[0] != []


def test_stray_shells_scatter_beyond_bbox_corners(part):
    lo = part.tris.reshape(-1, 3).min(axis=0)
    hi = part.tris.reshape(-1, 3).max(axis=0)
    mid = (lo + hi) / 2
    seen = set()
    for seed in range(1, 9):
        out = degrade.apply("stray_shells", part, 1.0, seed)
        kinds = out.metadata["history"][-1]["params"]["kinds"]
        rows = out.tris[len(part.tris) :]
        i = 0
        for kind in kinds:
            n = 1 if kind == "triangle" else 4
            center = rows[i : i + n].mean(axis=(0, 1))
            assert ((center < lo) | (center > hi)).any()
            seen.add(tuple(np.sign(center - mid).astype(int).tolist()))
            i += n
    assert len(seen) > 1


def test_stray_convert_keeps_main_shell_and_reports_strays(part):
    clean = unmesh.convert(part.tris)
    out = degrade.apply("stray_shells", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    got = unmesh.convert(out.tris)
    assert len(got.ir.shells) == 1 + params["shells_added"]
    roles = [(s.closed, s.role) for s in got.ir.shells]
    assert roles.count((True, "outer")) == 1 + params["kinds"].count("tetrahedron")
    assert roles.count((False, "outer")) == params["kinds"].count("triangle")
    if "triangle" in params["kinds"]:
        assert "open_edges" in codes(got)
    main_sets = {
        tuple(sorted(r.triangles))
        for r in got.ir.regions
        if r.triangles and max(r.triangles) < len(part.tris)
    }
    clean_sets = {tuple(sorted(r.triangles)) for r in clean.ir.regions}
    assert main_sets == clean_sets


def test_stray_convert_curved_keeps_main_shell(curved_part):
    clean = unmesh.convert(curved_part.tris)
    out = degrade.apply("stray_shells", curved_part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    main_sets = {
        tuple(sorted(r.triangles))
        for r in got.ir.regions
        if r.triangles and max(r.triangles) < len(curved_part.tris)
    }
    clean_sets = {tuple(sorted(r.triangles)) for r in clean.ir.regions}
    assert main_sets == clean_sets


def test_fin_adds_a_third_triangle_to_an_edge(part):
    out = degrade.apply("nonmanifold_fin", part, 1.0, SEED)
    params = out.metadata["history"][-1]["params"]
    assert params["fins"] == 6
    assert params["height_mm"] == 0.5
    assert len(out.tris) == len(part.tris) + params["fins"]
    assert np.array_equal(out.face_id[: len(part.tris)], part.face_id)
    assert set(out.face_id[len(part.tris) :].tolist()) == {-1}
    assert np.array_equal(out.tris[: len(part.tris)], part.tris)
    assert closed_manifold_problems(out.tris)[0] != []


def test_fin_convert_reports_a_non_manifold_shell(part):
    out = degrade.apply("nonmanifold_fin", part, 1.0, SEED)
    got = unmesh.convert(out.tris)
    assert "non_manifold_edges" in codes(got)
    assert [(s.closed, s.role) for s in got.ir.shells] == [(False, "outer")]
    assert {r.surface.type for r in got.ir.regions} == {"facets"}
    assert sum(len(r.triangles) for r in got.ir.regions) == len(out.tris)
