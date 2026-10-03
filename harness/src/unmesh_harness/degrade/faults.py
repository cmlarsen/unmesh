"""Defect degradation operators.

Each operator is a pure function ``(LabeledMesh, severity, rng) -> params``
registered under ``defect`` family in ``unmesh_harness.degrade.OPERATORS``. They
model dirty real-world input: unwelded corners, cracks, flipped and duplicate
facets, holes, stray shells and non-manifold fins.

Label policy: triangles copied from the input keep their source ``face_id``;
brand-new synthetic triangles (stray shells and fin triangles) get ``face_id`` -1,
meaning unknown face, which the oracle excludes from every region. The face table, adjacency,
edge polylines, vertices and shells are left untouched, so they stay the clean
truth the degraded mesh is judged against.

``DISPOSITION`` documents, per operator, whether the converter is expected to
REPAIR the defect (same analytic IR as the clean mesh, plus a warning) or
REPORT it (open/non-manifold facets shell, plus a warning). It is rendered
into docs/degradations.md by ``unmesh_harness.degrade.report``.
"""

from __future__ import annotations

import numpy as np

from .core import register, vertex_table

CRACK_WIDTH_AT_ONE_MM = 0.2
CRACK_MAX_EDGES = 8
UNWELDED_MAX_MM = 0.5e-6
UNWELDED_GAP_LO_MM = 1e-4
UNWELDED_GAP_HI_MM = 1e-2
FLIP_FRACTION = 0.1
DUP_FRACTION = 0.05
HOLE_MAX_REMOVED = 8
STRAY_MAX_SHELLS = 4
STRAY_SIZE_MM = 1.0
STRAY_GAP_MM = 5.0
FIN_FRACTION = 0.05
FIN_HEIGHT_MM = 0.5

DISPOSITION: dict[str, str] = {
    "unwelded_corners": (
        "REPAIR: the split copies sit within the converter weld "
        "(vertex_merge tolerance); the IR is identical to the clean mesh."
    ),
    "unwelded_gap": (
        "REPORT: the split copies sit outside the converter weld "
        "(vertex_merge tolerance); the mesh keeps open edges (warning open_edges)."
    ),
    "crack_seam": (
        "REPORT: the seam leaves open edges; the edge-connected component "
        "becomes one open facets shell (warning open_edges)."
    ),
    "flipped_facets": (
        "REPAIR (planar parts): inconsistent winding is fixed by flood fill "
        "(warning repaired_winding); the IR matches the clean mesh. On curved "
        "parts the repaired winding segments differently (converter issue #67), "
        "so region sets may differ."
    ),
    "duplicate_facets": (
        "REPAIR: same-winding copies are dropped (warning degenerate_triangles); "
        "opposite-winding twins are dropped while that leaves the component "
        "manifold (warning repaired_winding)."
    ),
    "hole_patch": (
        "REPORT: the missing patch leaves open edges; the edge-connected "
        "component becomes one open facets shell (warning open_edges)."
    ),
    "stray_shells": (
        "REPORT: each stray triangle is its own open facets shell (warning "
        "open_edges); each stray tetrahedron is an extra closed shell. "
        "The main shell still converts: on planar parts its region sets equal "
        "clean, on curved parts tolerances derive from the grown bounding box "
        "(converter issue #67), so they may differ."
    ),
    "nonmanifold_fin": (
        "REPORT: the fin edge is used by three triangles, so the whole "
        "edge-connected component becomes one open facets shell "
        "(warning non_manifold_edges)."
    ),
}


@register(
    "unwelded_corners",
    "defect",
    "identity",
    "a fraction (severity) of distinct vertices is split with the copies moved "
    "by severity * 0.5e-6 mm, below the converter weld, so the weld repairs it "
    "(all vertices split at severity 1)",
    preserves_watertight=False,
)
def unwelded_corners(mesh, severity, rng):
    return _split_vertices(mesh, severity, rng, severity * UNWELDED_MAX_MM)


@register(
    "unwelded_gap",
    "defect",
    "identity",
    "a fraction (severity) of distinct vertices is split with the copies moved "
    "by 1e-4 mm (severity 0+) up to 1e-2 mm (severity 1), outside the converter "
    "weld, so the mesh keeps open edges (all vertices split at severity 1)",
    preserves_watertight=False,
)
def unwelded_gap(mesh, severity, rng):
    gap = UNWELDED_GAP_LO_MM * (UNWELDED_GAP_HI_MM / UNWELDED_GAP_LO_MM) ** severity
    return _split_vertices(mesh, severity, rng, gap)


def _split_vertices(mesh, severity, rng, displacement):
    uniq, inverse = vertex_table(mesh)
    corners = inverse.reshape(-1, 3)
    n_split = min(len(uniq), max(1, int(round(severity * len(uniq)))))
    picked = sorted(rng.choice(len(uniq), size=n_split, replace=False).tolist())
    moved = 0
    for v in picked:
        hits = np.argwhere(corners == v)
        if len(hits) < 2:
            continue
        direction = rng.standard_normal(3)
        norm = float(np.linalg.norm(direction))
        step = direction / norm * displacement if norm > 0 else np.zeros(3)
        for ti, ci in hits[1:].tolist():
            mesh.tris[ti, ci] = mesh.tris[ti, ci] + step
            moved += 1
    return {
        "distinct_vertices": len(uniq),
        "vertices_split": n_split,
        "displacement_mm": displacement,
        "moved_corners": moved,
    }


def _edge_owners(tris: np.ndarray):
    corners = np.asarray(tris, dtype=np.float64).reshape(-1, 3)
    uniq, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    owners: dict[tuple[int, int], list[int]] = {}
    for ti, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            owners.setdefault((min(u, v), max(u, v)), []).append(ti)
    return uniq, faces, owners


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _tri_normal(mesh, t: int) -> np.ndarray:
    return np.cross(mesh.tris[t, 1] - mesh.tris[t, 0], mesh.tris[t, 2] - mesh.tris[t, 0])


def _left_owner(edge, u, v, owners, uniq, faces, mesh):
    direction = uniq[v] - uniq[u]
    left = []
    for t in owners[edge]:
        rest = [c for c in faces[t] if c != u and c != v]
        if len(rest) != 1:
            continue
        (w,) = rest
        if float(np.dot(np.cross(direction, uniq[w] - uniq[u]), _tri_normal(mesh, t))) > 0:
            left.append(t)
    return left[0] if len(left) == 1 else owners[edge][0]


def _fan_tris(v: int, faces: np.ndarray, n: int) -> list[int]:
    return [t for t in range(n) if faces[t, 0] == v or faces[t, 1] == v or faces[t, 2] == v]


def _cyclic_neighbors(v, fan, faces, uniq, normal) -> list[int]:
    axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(normal, axis)[None])[0]
    e2 = np.cross(normal, e1)
    nbrs = sorted({int(c) for t in fan for c in faces[t] if c != v})

    def angle(nb: int) -> tuple[float, int]:
        d = uniq[nb] - uniq[v]
        return (float(np.arctan2(np.dot(d, e2), np.dot(d, e1))), nb)

    return sorted(nbrs, key=angle)


def _spoke_arcs(ordered, p_nbrs: set[int], v: int, fan, faces) -> list[list[int]]:
    idx = [i for i, nb in enumerate(ordered) if nb in p_nbrs]
    runs: list[set[int]] = []
    for k in range(len(idx)):
        run: set[int] = set()
        i = (idx[k] + 1) % len(ordered)
        while i != idx[(k + 1) % len(idx)]:
            run.add(ordered[i])
            i = (i + 1) % len(ordered)
        runs.append(run)
    arcs = []
    for run in runs:
        arc = []
        for t in fan:
            others = sorted({int(c) for c in faces[t] if c != v})
            if len(others) == 2 and all(c in run for c in others):
                arc.append(t)
        arcs.append(sorted(arc))
    return arcs


def _spokes_agree(v, moving: set[int], owners, faces, n: int, p_nbrs: set[int]) -> bool:
    nbrs = sorted({int(c) for t in _fan_tris(v, faces, n) for c in faces[t] if c != v})
    for x in nbrs:
        if x in p_nbrs:
            continue
        own = owners.get((min(v, x), max(v, x)), [])
        if len(own) != 2:
            return False
        if (own[0] in moving) != (own[1] in moving):
            return False
    return True


@register(
    "crack_seam",
    "defect",
    "identity",
    "a chain of 2 (severity 0+) to 8 (severity 1) interior mesh edges sharing "
    "vertices within a single source face is split open to width severity * 0.2 mm "
    "with blunt tips, the copies on one side only shifted along the averaged "
    "in-surface perpendicular",
    preserves_watertight=False,
)
def crack_seam(mesh, severity, rng):
    uniq, faces, owners = _edge_owners(mesh.tris)
    n = len(mesh.tris)
    interior = sorted(k for k, v in owners.items() if len(v) == 2)
    want = max(2, int(round(severity * CRACK_MAX_EDGES)))
    width = severity * CRACK_WIDTH_AT_ONE_MM
    same_face = sorted(
        e for e in interior if mesh.face_id[owners[e][0]] == mesh.face_id[owners[e][1]]
    )
    if same_face:
        seed = same_face[rng.integers(len(same_face))]
        face = int(mesh.face_id[owners[seed][0]])
    else:
        seed = interior[rng.integers(len(interior))]
        face = int(mesh.face_id[owners[seed][0]])
    incident: dict[int, list[tuple[int, int]]] = {}
    for key in owners:
        for w in key:
            incident.setdefault(w, []).append(key)
    seam = [seed]
    used = {seed}
    owned = set(owners[seed])
    touched = {int(seed[0]), int(seed[1])}
    poly = [int(seed[0]), int(seed[1])]
    side = {seed: _left_owner(seed, poly[0], poly[1], owners, uniq, faces, mesh)}
    moving: dict[int, set[int]] = {v: {side[seed]} for v in poly}
    tip = poly[1]
    while len(seam) < want:
        prev = poly[-2]
        cands = sorted(
            e
            for e in incident[tip]
            if e not in used
            and len(owners[e]) == 2
            and all(mesh.face_id[t] == face for t in owners[e])
            and all(w == tip or w not in touched for w in e)
            and not (set(owners[e]) & owned)
        )
        nxt = None
        for e in [cands[i] for i in rng.permutation(len(cands)).tolist()]:
            w = e[0] if e[1] == tip else e[1]
            s_new = _left_owner(e, tip, w, owners, uniq, faces, mesh)
            if not _private(e, s_new, owners, faces, touched, mesh.tris, tip):
                continue
            joint = _joint_moving(
                tip, prev, w, side[seam[-1]], s_new, owners, uniq, faces, mesh, n, face
            )
            if joint is None:
                continue
            nxt = (e, w, s_new, joint)
            break
        if nxt is None:
            break
        e, w, s_new, joint = nxt
        seam.append(e)
        used.add(e)
        owned.update(owners[e])
        touched.add(int(w))
        poly.append(int(w))
        side[e] = s_new
        moving[tip] = joint
        moving[w] = {s_new}
        tip = int(w)
    push = np.zeros(3)
    length = 0.0
    for i, edge in enumerate(seam):
        u, v = poly[i], poly[i + 1]
        ev = uniq[v] - uniq[u]
        length += float(np.linalg.norm(ev))
        push += np.cross(_tri_normal(mesh, side[edge]), ev)
    if np.linalg.norm(push) < 1e-12 * max(length, 1e-12):
        fallback = mesh.tris[side[seam[0]], 1] - mesh.tris[side[seam[0]], 0]
        push = np.cross(np.array([1.0, 0.0, 0.0]), fallback)
    step = width * _unit(push[None])[0]
    offsets: set[tuple[int, int]] = set()
    blunt = {poly[0], poly[-1]} if len(seam) > 1 else set()
    for v, tris in moving.items():
        if v in blunt:
            continue
        for t in tris:
            for ci in range(3):
                if int(faces[t, ci]) == v:
                    offsets.add((t, ci))
    for tri, ci in offsets:
        mesh.tris[tri, ci] = mesh.tris[tri, ci] + step
    return {
        "seam_edges": len(seam),
        "width_mm": width,
        "face": face,
        "side_triangles": len(set(side.values())),
        "seam_length_mm": length,
        "moved_corners": len(offsets),
    }


def _private(edge, s_new, owners, faces, touched, tris, tip) -> bool:
    for t in owners[edge]:
        others = [int(c) for c in faces[t] if c != edge[0] and c != edge[1]]
        if any(c in touched for c in others):
            return False
    w = edge[0] if edge[1] == tip else edge[1]
    for t in _fan_tris(w, faces, len(tris)):
        if t in owners[edge]:
            continue
        if any(int(c) in touched for c in faces[t] if c != w):
            return False
    return True


def _joint_moving(tip, prev, w, s_prev, s_new, owners, uniq, faces, mesh, n, face):
    p_nbrs = {prev, w}
    fan = _fan_tris(tip, faces, n)
    normal = _tri_normal(mesh, s_prev)
    ordered = _cyclic_neighbors(tip, fan, faces, uniq, normal)
    arcs = _spoke_arcs(ordered, p_nbrs, tip, fan, faces)

    def score(arc: list[int]) -> float:
        if not arc:
            return float("-inf")
        r = np.mean([mesh.tris[t].mean(axis=0) - uniq[tip] for t in arc], axis=0)
        d_in = uniq[tip] - uniq[prev]
        d_out = uniq[w] - uniq[tip]
        return float(np.dot(np.cross(d_in, r), normal) + np.dot(np.cross(d_out, r), normal))

    for arc in sorted(arcs, key=score, reverse=True):
        joint = {s_prev, s_new} | {t for t in arc if int(mesh.face_id[t]) == face}
        if _spokes_agree(tip, joint, owners, faces, n, p_nbrs):
            return joint
    return None


@register(
    "flipped_facets",
    "defect",
    "identity",
    "a fraction (severity * 10%) of triangles is reversed in place; the "
    "converter flood-fills the winding back (up to 10% reversed at severity 1)",
    preserves_watertight=False,
)
def flipped_facets(mesh, severity, rng):
    n = len(mesh.tris)
    k = min(n, max(1, int(round(severity * FLIP_FRACTION * n))))
    idx = sorted(rng.choice(n, size=k, replace=False).tolist())
    mesh.tris[idx] = mesh.tris[idx][:, ::-1]
    return {"flipped": len(idx), "fraction": len(idx) / n}


@register(
    "duplicate_facets",
    "defect",
    "identity",
    "a fraction (severity * 5%) of triangles is appended again with the same "
    "source face_id; each copy takes same or opposite winding by rng coin "
    "(up to 5% duplicated at severity 1)",
    preserves_watertight=False,
)
def duplicate_facets(mesh, severity, rng):
    n = len(mesh.tris)
    k = min(n, max(1, int(round(severity * DUP_FRACTION * n))))
    idx = sorted(rng.choice(n, size=k, replace=False).tolist())
    coins = rng.random(k)
    rows = []
    same = 0
    for i, coin in zip(idx, coins, strict=True):
        rows.append(mesh.tris[i] if coin < 0.5 else mesh.tris[i][::-1])
        same += int(coin < 0.5)
    mesh.tris = np.concatenate([mesh.tris, np.array(rows)])
    mesh.face_id = np.concatenate([mesh.face_id, mesh.face_id[idx]])
    return {"duplicated": k, "same_winding": same, "opposite_winding": k - same}


def _tri_edges(owners: dict[tuple[int, int], list[int]]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for key, members in owners.items():
        for t in members:
            out.setdefault(t, []).append(key)
    return out


@register(
    "hole_patch",
    "defect",
    "identity",
    "a connected patch of 1 triangle (severity 0+) up to 8 (severity 1) within "
    "a single source face is removed; the converter reports the open shell",
    preserves_watertight=False,
)
def hole_patch(mesh, severity, rng):
    n = len(mesh.tris)
    want = 1 + int(round(severity * (HOLE_MAX_REMOVED - 1)))
    seed_tri = int(rng.integers(n))
    seed_face = int(mesh.face_id[seed_tri])
    _, _, owners = _edge_owners(mesh.tris)
    tri_edges = _tri_edges(owners)
    patch = {seed_tri}
    while len(patch) < want:
        fringe = set()
        for t in patch:
            for edge in tri_edges[t]:
                for u in owners[edge]:
                    if u not in patch and mesh.face_id[u] == seed_face:
                        fringe.add(u)
        if not fringe:
            break
        for u in sorted(fringe):
            if len(patch) >= want:
                break
            patch.add(u)
    drop = sorted(patch)
    if len(drop) == n:
        drop = drop[:-1]
    keep = np.ones(n, dtype=bool)
    keep[drop] = False
    mesh.tris = mesh.tris[keep]
    mesh.face_id = mesh.face_id[keep]
    return {"removed": len(drop), "requested": want, "face": seed_face}


@register(
    "stray_shells",
    "defect",
    "identity",
    "1 shell (severity 0+) up to 4 (severity 1) of floating geometry is added "
    "beyond a random bounding-box corner, each a single triangle or a tetrahedron "
    "by rng coin with 1 mm edges; new triangles get face_id -1 (unknown face)",
    preserves_watertight=False,
)
def stray_shells(mesh, severity, rng):
    m = 1 + int(round(severity * (STRAY_MAX_SHELLS - 1)))
    lo = mesh.tris.reshape(-1, 3).min(axis=0)
    hi = mesh.tris.reshape(-1, 3).max(axis=0)
    rows: list[np.ndarray] = []
    kinds: list[str] = []
    for _ in range(m):
        signs = rng.integers(0, 2, size=3) * 2 - 1
        anchor = np.where(signs > 0, hi, lo)
        tilt = np.abs(rng.standard_normal(3))
        if np.linalg.norm(tilt) < 1e-12:
            tilt = np.array([1.0, 1.0, 1.0])
        outward = signs * tilt
        outward = outward / np.linalg.norm(outward)
        center = anchor + outward * (STRAY_GAP_MM + STRAY_SIZE_MM * rng.random())
        axis = np.array([1.0, 0.0, 0.0]) if abs(outward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(outward, axis)[None])[0]
        e2 = np.cross(outward, e1)
        if rng.random() < 0.5:
            angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
            ring = STRAY_SIZE_MM / np.sqrt(3.0)
            rows.append(
                center + ring * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
            )
            kinds.append("triangle")
        else:
            unit = np.array(
                [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
            )
            pts = center + (STRAY_SIZE_MM / (2.0 * np.sqrt(2.0))) * (
                unit[:, 0, None] * e1 + unit[:, 1, None] * e2 + unit[:, 2, None] * outward
            )
            centroid = pts.mean(axis=0)
            for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                tri = np.array([pts[a], pts[b], pts[c]])
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if np.dot(normal, centroid - tri.mean(axis=0)) > 0:
                    tri = tri[::-1]
                rows.append(tri)
            kinds.append("tetrahedron")
    mesh.tris = np.concatenate([mesh.tris, np.array(rows)])
    mesh.face_id = np.concatenate([mesh.face_id, np.full(len(rows), -1, dtype=mesh.face_id.dtype)])
    return {
        "shells_added": m,
        "triangles_added": len(rows),
        "kinds": kinds,
        "size_mm": STRAY_SIZE_MM,
        "gap_mm": STRAY_GAP_MM,
    }


@register(
    "nonmanifold_fin",
    "defect",
    "identity",
    "a fraction (severity * 5%) of interior edges grows an extra triangle fin "
    "0.5 mm tall with face_id -1 (unknown face), so the shared "
    "edge is used by three triangles (up to 5% of edges finned at severity 1)",
    preserves_watertight=False,
)
def nonmanifold_fin(mesh, severity, rng):
    uniq, _, owners = _edge_owners(mesh.tris)
    interior = sorted(key for key, members in owners.items() if len(members) == 2)
    k = min(len(interior), max(1, int(round(severity * FIN_FRACTION * len(interior)))))
    picked = sorted(rng.choice(len(interior), size=k, replace=False).tolist())
    rows: list[np.ndarray] = []
    fids: list[int] = []
    for i in picked:
        (u, v) = interior[i]
        host = owners[(u, v)][0]
        a, b = uniq[u], uniq[v]
        host_tri = mesh.tris[host]
        normal = np.cross(host_tri[1] - host_tri[0], host_tri[2] - host_tri[0])
        if np.linalg.norm(normal) < 1e-12:
            normal = np.cross(b - a, np.array([1.0, 0.0, 0.0]))
        apex = (a + b) / 2.0 + _unit(normal[None])[0] * FIN_HEIGHT_MM
        rows.append(np.array([a, b, apex]))
        fids.append(-1)
    if rows:
        mesh.tris = np.concatenate([mesh.tris, np.array(rows)])
        mesh.face_id = np.concatenate([mesh.face_id, np.array(fids, dtype=mesh.face_id.dtype)])
    return {"fins": len(rows), "height_mm": FIN_HEIGHT_MM}
