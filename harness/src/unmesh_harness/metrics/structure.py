from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance

WELD_TOLERANCE_MM = 1e-6


def _default_tolerance_bound() -> float:
    from unmesh.step import WriteOptions

    return float(WriteOptions().max_shape_tolerance)


def score_validity(
    step_path: str | Path | None,
    *,
    write: dict[str, Any] | None = None,
    expected_solids: int = 1,
    expected_shells: int = 1,
    tolerance_bound: float | None = None,
) -> dict[str, Any]:
    from ..groundtruth import validity_problems

    if tolerance_bound is None:
        tolerance_bound = _default_tolerance_bound()
    write = write or {}
    fallback = write.get("fallback") is not None
    missing: dict[str, Any] = {
        "solids": None,
        "shells": None,
        "volume": None,
        "volume_positive": None,
        "brepcheck_valid": None,
        "max_shape_tolerance": None,
        "tolerance_ok": None,
        "expected_solids": int(expected_solids),
        "expected_shells": int(expected_shells),
        "tolerance_bound": float(tolerance_bound),
        "fallback": bool(fallback),
        "problems": [],
        "valid": False,
    }
    if step_path is None or not Path(step_path).is_file() or Path(step_path).stat().st_size == 0:
        missing["problems"] = ["no STEP file written"]
        return missing
    try:
        from build123d import import_step

        shape = import_step(str(step_path))
    except Exception as e:
        missing["problems"] = [f"STEP check failed: {type(e).__name__}: {e}"]
        return missing
    problems = validity_problems(shape, expected_solids, expected_shells)
    brepcheck_valid = bool(BRepCheck_Analyzer(shape.wrapped).IsValid())
    volume = float(shape.volume)
    volume_positive = bool(volume > 0)
    max_tolerance = float(ShapeAnalysis_ShapeTolerance().Tolerance(shape.wrapped, 1))
    tolerance_ok = bool(max_tolerance <= tolerance_bound)
    if not tolerance_ok:
        problems.append(f"shape tolerance {max_tolerance:.3g} exceeds {tolerance_bound:.3g}")
    return {
        "solids": len(shape.solids()),
        "shells": len(shape.shells()),
        "volume": volume,
        "volume_positive": volume_positive,
        "brepcheck_valid": brepcheck_valid,
        "max_shape_tolerance": max_tolerance,
        "tolerance_ok": tolerance_ok,
        "expected_solids": int(expected_solids),
        "expected_shells": int(expected_shells),
        "tolerance_bound": float(tolerance_bound),
        "fallback": bool(fallback),
        "problems": problems,
        "valid": not problems,
    }


def _role_signatures(adjacency) -> list[tuple[int, ...]]:
    faces_at: dict[int, set[int]] = {}
    ends_at: dict[int, list[str]] = {}
    for adj in adjacency:
        kind = "tangent" if adj.tangent else "transversal"
        for v in (adj.start_vertex, adj.end_vertex):
            if v is None or v < 0:
                continue
            faces_at.setdefault(v, set()).update((adj.face_a, adj.face_b))
            ends_at.setdefault(v, []).append(kind)
    roles = []
    for v, faces in faces_at.items():
        if len(faces) >= 3:
            roles.append(tuple(sorted(faces)))
        elif len(faces) == 2 and sorted(ends_at[v]) == ["tangent", "transversal"]:
            roles.append(tuple(sorted(faces)))
    return sorted(roles)


def _pairs(ids: list[tuple[int, int]]) -> list[list[int]]:
    return sorted([min(a, b), max(a, b)] for a, b in ids)


def _role_vertices(adjacency) -> set[int]:
    faces_at: dict[int, set[int]] = {}
    ends_at: dict[int, list[str]] = {}
    for adj in adjacency:
        kind = "tangent" if adj.tangent else "transversal"
        for v in (adj.start_vertex, adj.end_vertex):
            faces_at.setdefault(v, set()).update((adj.face_a, adj.face_b))
            ends_at.setdefault(v, []).append(kind)
    return {
        v
        for v, faces in faces_at.items()
        if len(faces) >= 3
        or (len(faces) == 2 and sorted(ends_at[v]) == ["tangent", "transversal"])
    }


def _chained_pair_counts(adjacency) -> dict[tuple[int, int], int]:
    roles = _role_vertices(adjacency)
    start: list[int] = []
    end: list[int] = []
    pair_of: list[tuple[int, int]] = []
    for adj in adjacency:
        if adj.forward_in_a:
            start.append(adj.start_vertex)
            end.append(adj.end_vertex)
        else:
            start.append(adj.end_vertex)
            end.append(adj.start_vertex)
        pair_of.append((min(adj.face_a, adj.face_b), max(adj.face_a, adj.face_b)))
    by_pair: dict[tuple[int, int], list[int]] = {}
    for i, pair in enumerate(pair_of):
        by_pair.setdefault(pair, []).append(i)
    counts: dict[tuple[int, int], int] = {}
    for pair in sorted(by_pair):
        members = by_pair[pair]
        outgoing: dict[int, list[int]] = {}
        for i in members:
            outgoing.setdefault(start[i], []).append(i)
        used: set[int] = set()

        def walk(first: int) -> None:
            used.add(first)
            tip = end[first]
            while tip not in roles:
                nxt = [i for i in outgoing.get(tip, []) if i not in used]
                if not nxt:
                    break
                used.add(nxt[0])
                tip = end[nxt[0]]

        boundaries = 0
        for i in members:
            if i in used or start[i] not in roles:
                continue
            walk(i)
            boundaries += 1
        for i in members:
            if i in used:
                continue
            walk(i)
            boundaries += 1
        counts[pair] = boundaries
    return counts


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def _boundary_loops(edges: np.ndarray, incidence: np.ndarray) -> int:
    loose = edges[incidence == 1]
    if len(loose) == 0:
        return 0
    verts, packed = np.unique(loose, return_inverse=True)
    find, union = _union_find(len(verts))
    for a, b in packed.reshape(-1, 2):
        union(int(a), int(b))
    return len({find(i) for i in range(len(verts))})


def _complex_stats(faces: np.ndarray) -> tuple[int, int, int, int]:
    if len(faces) == 0:
        return 0, 0, 0, 0
    edges = np.stack(
        [
            np.minimum(faces[:, 0], faces[:, 1]),
            np.maximum(faces[:, 0], faces[:, 1]),
            np.minimum(faces[:, 1], faces[:, 2]),
            np.maximum(faces[:, 1], faces[:, 2]),
            np.minimum(faces[:, 2], faces[:, 0]),
            np.maximum(faces[:, 2], faces[:, 0]),
        ],
        axis=1,
    ).reshape(-1, 2)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    change = np.ones(len(ordered), dtype=bool)
    change[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    starts = np.nonzero(change)[0]
    counts = np.diff(np.append(starts, len(ordered)))
    verts = np.unique(faces)
    loops = _boundary_loops(ordered[starts], counts)
    return int(len(verts)), int(len(starts)), int(len(faces)), int(loops)


def _shell_genus(tris: np.ndarray, row_groups: list[np.ndarray]) -> tuple[list[float], int]:
    import unmesh

    verts, faces, kept, _ = unmesh.weld(
        np.ascontiguousarray(tris, dtype=np.float64), WELD_TOLERANCE_MM
    )
    faces = np.asarray(faces, dtype=np.int64)
    kept = np.asarray(kept)
    kept = kept[kept < len(tris)]
    n = min(len(kept), len(faces))
    row_of = np.full(len(tris), -1, dtype=np.int64)
    row_of[kept[:n]] = np.arange(n, dtype=np.int64)
    genus: list[float] = []
    for group in row_groups:
        rows = row_of[np.asarray(group, dtype=np.int64)]
        rows = rows[rows >= 0]
        v, e, f, loops = _complex_stats(faces[rows] if len(rows) else np.zeros((0, 3), np.int64))
        genus.append(0.0 if f == 0 else (2 - (v - e + f) - loops) / 2)
    return genus, int(len(kept))


def score_topology(clean, face_id: np.ndarray, tris: np.ndarray, ir) -> dict[str, Any]:
    face_id = np.asarray(face_id)
    tris = np.asarray(tris, dtype=np.float64)
    truth_faces = len(clean.faces)
    output_faces = len(ir.regions)
    if len(face_id) != len(tris):
        raise ValueError(f"face_id has {len(face_id)} entries for {len(tris)} triangles")
    if len(face_id) and (face_id.min() < 0 or face_id.max() >= truth_faces):
        raise ValueError(f"face_id outside [0, {truth_faces})")
    from .recovery import match_faces

    matches, _ = match_faces(clean, face_id, ir)
    region_to_face: dict[int, int] = {}
    for m in matches:
        if m["region"] is not None and m["region"] not in region_to_face:
            region_to_face[m["region"]] = m["face"]
    unmatched_regions = sorted(r.id for r in ir.regions if r.id not in region_to_face)
    remapped_pairs = []
    remapped_counts: dict[tuple[int, int], int] = {}
    pairs_unmatched = False
    for adj in ir.adjacencies:
        if any(r not in region_to_face for r in adj.regions):
            pairs_unmatched = True
            continue
        key = tuple(sorted(region_to_face[r] for r in adj.regions))
        remapped_pairs.append(key)
        remapped_counts[key] = remapped_counts.get(key, 0) + len(adj.boundaries)
    truth_pairs = _pairs([(a.face_a, a.face_b) for a in clean.adjacency])
    truth_counts = _chained_pair_counts(clean.adjacency)
    output_pairs = _pairs(remapped_pairs)
    pairs_match = (
        not unmatched_regions
        and not pairs_unmatched
        and set(map(tuple, output_pairs)) == set(truth_counts)
    )
    edges_match = (
        not unmatched_regions and not pairs_unmatched and remapped_counts == truth_counts
    )
    truth_roles = _role_signatures(clean.adjacency)
    remapped_roles = []
    roles_unmatched = False
    for v in ir.vertices:
        if any(r not in region_to_face for r in v.regions):
            roles_unmatched = True
            continue
        remapped_roles.append(tuple(sorted(region_to_face[r] for r in v.regions)))
    output_roles = sorted(remapped_roles)
    roles_match = (
        not unmatched_regions and not roles_unmatched and truth_roles == output_roles
    )
    truth_shells = [(s.role, len(s.faces)) for s in clean.shells]
    output_shells = [(s.role, len(s.regions)) for s in ir.shells]
    face_shell = [-1] * truth_faces
    for i, s in enumerate(clean.shells):
        for f in s.faces:
            if 0 <= f < truth_faces:
                face_shell[f] = i
    shell_match: list = []
    for s in ir.shells:
        votes = {
            face_shell[region_to_face[r]]
            for r in s.regions
            if r in region_to_face
            and 0 <= region_to_face[r] < truth_faces
            and face_shell[region_to_face[r]] >= 0
        }
        shell_match.append(next(iter(votes)) if len(votes) == 1 else None)
    shells_match = (
        len(ir.shells) == len(clean.shells)
        and not unmatched_regions
        and all(m is not None for m in shell_match)
        and sorted(shell_match) == list(range(len(clean.shells)))
    )
    if shells_match:
        for i, j in enumerate(shell_match):
            mine, gt = ir.shells[i], clean.shells[j]
            if mine.role != gt.role:
                shells_match = False
                break
            parent = (
                None
                if mine.parent is None
                else (
                    shell_match[mine.parent]
                    if 0 <= mine.parent < len(shell_match)
                    else None
                )
            )
            if (mine.parent is not None and parent is None) or parent != gt.parent:
                shells_match = False
                break

    region_shell = [-1] * output_faces
    for i, s in enumerate(ir.shells):
        for r in s.regions:
            if 0 <= r < output_faces:
                region_shell[r] = i
    owned_region: dict[int, int] = {}
    for r in ir.regions:
        for t in r.triangles:
            if 0 <= t < len(tris) and t not in owned_region:
                owned_region[t] = r.id
    gt_groups = [
        np.nonzero(np.array([face_shell[f] == i for f in face_id], dtype=bool))[0]
        for i in range(len(clean.shells))
    ]
    ir_groups = []
    for i in range(len(ir.shells)):
        members = [t for t, r in owned_region.items() if region_shell[r] == i]
        ir_groups.append(np.asarray(members, dtype=np.int64))
    truth_genus, _ = _shell_genus(tris, gt_groups)
    output_genus, _ = _shell_genus(tris, ir_groups)
    genera_match = sorted(truth_genus) == sorted(output_genus)

    def holes(shells, genus: list[float]) -> float:
        outer = [g for s, g in zip(shells, genus, strict=True) if s.role == "outer"]
        return float(sum(outer if outer else genus))

    truth_holes = holes(clean.shells, truth_genus)
    through_holes = holes(ir.shells, output_genus)
    holes_match = truth_holes == through_holes
    faces_match = truth_faces == output_faces
    topology_match = bool(
        faces_match
        and pairs_match
        and edges_match
        and roles_match
        and shells_match
        and genera_match
        and holes_match
    )
    return {
        "truth_faces": int(truth_faces),
        "output_faces": int(output_faces),
        "faces_match": bool(faces_match),
        "truth_edges": int(len(clean.adjacency)),
        "output_edges": int(sum(len(a.boundaries) for a in ir.adjacencies)),
        "truth_edge_pairs": truth_pairs,
        "output_edge_pairs": output_pairs,
        "pairs_match": bool(pairs_match),
        "edges_match": bool(edges_match),
        "truth_vertices": int(len(clean.vertices)),
        "output_vertices": int(len(ir.vertices)),
        "truth_role_vertices": int(len(truth_roles)),
        "output_role_vertices": int(len(output_roles)),
        "roles_match": bool(roles_match),
        "truth_shells": int(len(clean.shells)),
        "output_shells": int(len(ir.shells)),
        "truth_shell_roles": [list(x) for x in truth_shells],
        "output_shell_roles": [list(x) for x in output_shells],
        "shells_match": bool(shells_match),
        "truth_genus": [float(g) for g in truth_genus],
        "output_genus": [float(g) for g in output_genus],
        "genera_match": bool(genera_match),
        "truth_through_holes": float(truth_holes),
        "through_holes": float(through_holes),
        "holes_match": bool(holes_match),
        "topology_match": topology_match,
        "uncovered_triangles": int(len(tris) - len(owned_region)),
    }


def score_structure(ir, truth_faces: int, tris: np.ndarray) -> dict[str, Any]:
    tris = np.asarray(tris, dtype=np.float64)
    n = len(tris)
    areas = (
        np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
        if n
        else np.zeros(0)
    )
    total = float(areas.sum())
    analytic = sum(1 for r in ir.regions if r.surface.type != "facets")
    owned = {t for r in ir.regions if r.surface.type != "facets" for t in r.triangles}
    owned = {t for t in owned if 0 <= t < n}
    analytic_area = float(areas[sorted(owned)].sum()) if owned else 0.0
    return {
        "truth_faces": int(truth_faces),
        "output_faces": int(len(ir.regions)),
        "analytic_regions": int(analytic),
        "faceted_regions": int(len(ir.regions) - analytic),
        "total_area_mm2": total,
        "analytic_area_mm2": analytic_area,
        "analytic_area_fraction": analytic_area / total if total > 0 else 0.0,
        "face_count_ratio": len(ir.regions) / truth_faces if truth_faces > 0 else 0.0,
    }
