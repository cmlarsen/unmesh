"""Processing-artifact degradation operators.

Each operator is a pure function ``(LabeledMesh, severity, rng) -> params``
registered under the ``processing`` family in ``unmesh_harness.degrade.OPERATORS``.
They model what mesh-processing pipelines do to CAD tessellations: quadric
decimation (via fast-simplification, MIT), isotropic remeshing (an in-house
Botsch-Kobbelt loop: split, collapse, flip, tangential relax, reproject),
Laplacian / Taubin smoothing, and vertex clustering.

Label policy: topology-changing operators relabel every output triangle from
the clean labeled source mesh. Source triangles are densely sampled
(barycentric lattice, spacing ``bbox diagonal / 400``, plus the centroid) with
each sample carrying its source triangle; a ``scipy.spatial.cKDTree`` over the
samples proposes candidate triangles for each output triangle's centroid and
3 edge midpoints (midpoints pulled 10% toward the centroid so probes never sit
exactly on a shared edge, where two faces tie at distance zero). A probe votes
the face of its exactly containing source triangle when one is within
1e-9 of the diagonal, else the majority face of its nearest samples; the
triangle takes the majority over its 4 probes with ``label_confidence`` the
agreeing fraction. Coplanar faces sharing one surface are told apart by
trimmed-triangle proximity. Smoothing keeps connectivity, so it keeps the
original labels exactly with confidence 1.0. When the input already carries
``label_confidence`` (a chain), each output triangle's confidence is capped by
the confidence of the source triangle under its centroid.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .core import CONFIDENCE_KEY, displace_vertices, register, vertex_table

MASK_THRESHOLD = 0.9
POLYLINES_COINCIDE = ("laplacian_smoothing", "taubin_smoothing")

_SAMPLE_DIVISOR = 400.0
_LABEL_KNN = 32
_LABEL_KNN_CAP = 512
_PROBE_INSET = 0.1
_EXACT_TOL_REL = 1e-9


def bbox_diagonal(tris: np.ndarray) -> float:
    flat = np.asarray(tris, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))


def _lattice(n: int) -> np.ndarray:
    return np.array(
        [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)],
        dtype=np.float64,
    )


def sample_source(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    if len(tris) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    diagonal = bbox_diagonal(tris)
    spacing = diagonal / _SAMPLE_DIVISOR if diagonal > 0 else 1.0
    edges = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=2).max(axis=1)
    order = np.clip(np.ceil(edges / spacing).astype(np.int64), 4, 128)
    parts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for n in np.unique(order):
        group = np.nonzero(order == n)[0]
        pts = _lattice(int(n)) @ tris[group]
        block = np.empty((len(group), pts.shape[1] + 1, 3))
        block[:, :-1] = pts
        block[:, -1] = tris[group].mean(axis=1)
        parts.append(block.reshape(-1, 3))
        owners.append(np.repeat(group, pts.shape[1] + 1))
    return np.concatenate(parts), np.concatenate(owners)


def _point_tri_dist2(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(axis=1)
    d2 = (ac * ap).sum(axis=1)
    bp = p - b
    d3 = (ab * bp).sum(axis=1)
    d4 = (ac * bp).sum(axis=1)
    cp = p - c
    d5 = (ab * cp).sum(axis=1)
    d6 = (ac * cp).sum(axis=1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    inside = (denom > 1e-300) & (vc >= 0.0) & (vb >= 0.0) & (va >= 0.0)
    out = np.full(len(p), np.inf)
    if inside.any():
        v = np.zeros(len(p))
        w = np.zeros(len(p))
        v[inside] = vb[inside] / denom[inside]
        w[inside] = vc[inside] / denom[inside]
        q = a + ab * v[:, None] + ac * w[:, None]
        out[inside] = ((p[inside] - q[inside]) ** 2).sum(axis=1)
    rest = ~inside
    if rest.any():
        pr, ar, br, cr = p[rest], a[rest], b[rest], c[rest]

        def seg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
            d = q1 - q0
            dd = (d**2).sum(axis=1)
            t = np.zeros(len(pr))
            ok = dd > 0
            t[ok] = ((pr[ok] - q0[ok]) * d[ok]).sum(axis=1) / dd[ok]
            t = np.clip(t, 0.0, 1.0)
            return ((pr - (q0 + t[:, None] * d)) ** 2).sum(axis=1)

        out[rest] = np.minimum.reduce([seg(ar, br), seg(br, cr), seg(cr, ar)])
    return out


def _unit_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def _exact_votes(
    probes: np.ndarray,
    src: np.ndarray,
    owner: np.ndarray,
    tree: cKDTree,
    tol2: float,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(probes, dtype=np.float64).reshape(-1, 3)
    src_n = None if normals is None else _unit_normals(src)
    k = _LABEL_KNN
    best_t = np.full(len(flat), -1, dtype=np.int64)
    best_d = np.full(len(flat), np.inf)
    for _ in range(4):
        use = min(k, len(tree.data))
        idx = tree.query(flat, k=use)[1]
        idx = np.asarray(idx).reshape(len(flat), use)
        cand = owner[idx.ravel()]
        rows = np.repeat(np.arange(len(flat)), use)
        raw = _point_tri_dist2(flat[rows], src[cand, 0], src[cand, 1], src[cand, 2])
        raw = raw.reshape(len(flat), use)
        d2 = raw
        if src_n is not None:
            facing = (src_n[cand] * normals[rows]).sum(axis=1) > 0.0
            d2 = np.where(facing.reshape(len(flat), use), raw, np.inf)
        pick = d2.argmin(axis=1)
        row_best = d2[np.arange(len(flat)), pick]
        row_tri = cand.reshape(len(flat), use)[np.arange(len(flat)), pick]
        improved = row_best < best_d
        best_d[improved] = row_best[improved]
        best_t[improved] = row_tri[improved]
        kth = raw[:, -1]
        retry = (best_d > tol2) & ((best_d > 4.0 * kth) | ~np.isfinite(best_d))
        if not retry.any() or k >= _LABEL_KNN_CAP or use >= len(tree.data):
            break
        k = min(k * 8, _LABEL_KNN_CAP)
    lost = best_t < 0
    if lost.any():
        best_t[lost], best_d[lost] = _exact_votes(flat[lost], src, owner, tree, tol2)
    return best_t, best_d


def transfer_labels(mesh, src_tris, src_ids, src_conf=None) -> dict:
    src = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 3)
    out = np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3)
    src_ids = np.asarray(src_ids)
    faces = np.unique(src_ids)
    if len(out) == 0:
        mesh.face_id = np.zeros(0, dtype=src_ids.dtype)
        mesh.metadata[CONFIDENCE_KEY] = []
        return {"mean_confidence": 1.0, "fraction_below_0_9": 0.0}
    diagonal = bbox_diagonal(src)
    tol2 = (diagonal * _EXACT_TOL_REL) ** 2
    pts, owner = sample_source(src)
    tree = cKDTree(pts)
    centroid = out.mean(axis=1)
    mids = np.stack(
        [
            out[:, 0] * 0.5 + out[:, 1] * 0.5,
            out[:, 1] * 0.5 + out[:, 2] * 0.5,
            out[:, 2] * 0.5 + out[:, 0] * 0.5,
        ],
        axis=1,
    )
    probes = np.concatenate(
        [centroid[:, None], mids * (1.0 - _PROBE_INSET) + centroid[:, None] * _PROBE_INSET],
        axis=1,
    )
    normal = np.repeat(_unit_normals(out), 4, axis=0)
    best_t, best_d = _exact_votes(probes, src, owner, tree, tol2, normal)
    exact = best_d.reshape(len(out), 4) <= tol2
    votes = np.empty((len(out), 4), dtype=src_ids.dtype)
    votes[exact] = src_ids[best_t.reshape(len(out), 4)[exact]]
    if not exact.all():
        missing = np.nonzero(~exact.all(axis=1))[0]
        src_n = _unit_normals(src)
        out_n = _unit_normals(out)
        idx = tree.query(probes[missing].reshape(-1, 3), k=_LABEL_KNN)[1]
        knn = np.asarray(idx).reshape(len(missing), 4, _LABEL_KNN)
        exm = exact[missing]
        for j in range(4):
            loc = np.nonzero(~exm[:, j])[0]
            if len(loc) == 0:
                continue
            need = missing[loc]
            near = owner[knn[loc, j]]
            sf = src_ids[near]
            facing = (src_n[near] * out_n[need][:, None, :]).sum(axis=2) > 0.0
            facing |= ~facing.any(axis=1, keepdims=True)
            counts = np.zeros((len(need), len(faces)), dtype=np.int32)
            for f, face in enumerate(faces):
                counts[:, f] = ((sf == face) & facing).sum(axis=1)
            votes[need, j] = faces[counts.argmax(axis=1)]
    counts = np.zeros((len(out), len(faces)), dtype=np.int32)
    np.add.at(counts, (np.arange(len(out))[:, None], np.searchsorted(faces, votes)), 1)
    win = counts.argmax(axis=1)
    conf = counts[np.arange(len(out)), win] / 4.0
    if src_conf is not None:
        carried = np.asarray(src_conf, dtype=np.float64)[best_t.reshape(len(out), 4)[:, 0]]
        conf = np.minimum(conf, carried)
    mesh.face_id = np.array(faces[win], dtype=src_ids.dtype)
    mesh.metadata[CONFIDENCE_KEY] = [float(v) for v in conf]
    return {
        "mean_confidence": float(conf.mean()) if len(conf) else 1.0,
        "fraction_below_0_9": float((conf < MASK_THRESHOLD).mean()) if len(conf) else 0.0,
    }


def carry_confidence(mesh, src_tris: np.ndarray, src_ids: np.ndarray, src_conf) -> None:
    conf = np.asarray(src_conf, dtype=np.float64)
    out = np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3)
    if len(out) == len(conf) and np.array_equal(mesh.face_id, src_ids):
        return
    if len(out) == 0 or len(src_tris) == 0:
        mesh.metadata[CONFIDENCE_KEY] = [1.0] * len(out)
        return
    names, labels = np.unique(np.asarray(src_ids), return_inverse=True)
    proj = _Projector(src_tris, labels)
    pos = np.clip(np.searchsorted(names, mesh.face_id), 0, len(names) - 1)
    known = names[pos] == mesh.face_id
    want = np.where(known, pos, -1)

    def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
        return labels[flat] == want[rows]

    tri = proj.search(out.mean(axis=1), accept)[1]
    carried = np.where(labels[tri] == want, conf[tri], 0.0)
    carried = np.where(known, carried, 1.0)
    mesh.metadata[CONFIDENCE_KEY] = [float(v) for v in carried]


def _weld(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inverse = np.unique(
        np.asarray(tris, dtype=np.float64).reshape(-1, 3), axis=0, return_inverse=True
    )
    return uniq, inverse.reshape(-1, 3).astype(np.int64)


def _drop_degenerate(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    scale = (V.max(axis=0) - V.min(axis=0)) ** 2
    area2 = np.linalg.norm(np.cross(V[F[:, 0]] - V[F[:, 1]], V[F[:, 2]] - V[F[:, 1]]), axis=1)
    keep = area2 > 1e-30 * (float(np.median(scale)) + 1e-300)
    return V, F[keep], int((~keep).sum())


DECIMATE_KEEP_AT_ONE = 0.1
DECIMATE_MIN_TRIANGLES = 4


@register(
    "quadric_decimation",
    "processing",
    "identity",
    "keep ratio 1 - 0.9 * severity of triangles (10% at severity 1), fast "
    "quadric-error decimation to the error-optimal position (fast-simplification, "
    "MIT) with a 4-triangle floor; no face-preservation guard, faces lost to the "
    "target show up as low-confidence or missing labels, and collapsing can "
    "crack the mesh, so watertightness is not guaranteed",
    preserves_watertight=False,
)
def quadric_decimation(mesh, severity, rng):
    import fast_simplification

    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    keep = 1.0 - 0.9 * severity
    target = max(DECIMATE_MIN_TRIANGLES, int(round(before * keep)))
    soup = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3)
    faces = np.arange(len(soup), dtype=np.int32).reshape(-1, 3)
    attempts = 0
    while True:
        P, Q = fast_simplification.simplify(soup, faces, target_count=target)
        V2, F2 = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.int64)
        V2, F2, dropped = _drop_degenerate(V2, F2)
        if len(F2) >= DECIMATE_MIN_TRIANGLES or target >= before or attempts >= 5:
            break
        target = min(before, target * 2 + 1)
        attempts += 1
    mesh.tris = V2[F2].reshape(-1, 3, 3)
    label = transfer_labels(mesh, src_tris, src_ids)
    return {
        "keep_ratio_target": keep,
        "keep_ratio_achieved": len(mesh.tris) / before,
        "target_triangles": target,
        "degenerate_dropped": dropped,
        "triangles_before": before,
        "triangles_after": len(mesh.tris),
        **label,
    }


REMESH_EDGE_AT_ZERO = 0.02
REMESH_GROWTH = 2.0


def target_edge_mm(severity: float, diagonal: float) -> float:
    return diagonal * REMESH_EDGE_AT_ZERO * REMESH_GROWTH**severity


def remesh_passes(severity: float) -> int:
    return 1 + int(round(2.0 * severity))


def _compact(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def _unique_edges(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(raw, axis=1)
    uniq, _, inverse, counts = np.unique(
        key, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    return uniq, inverse, counts


def _manifold_owners(
    n_tris: int, counts: np.ndarray, inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    tri_of = np.tile(np.arange(n_tris), 3)[order]
    t0 = np.full(len(counts), -1, dtype=np.int64)
    t1 = np.full(len(counts), -1, dtype=np.int64)
    pair = np.nonzero(counts == 2)[0]
    t0[pair] = tri_of[starts[pair]]
    t1[pair] = tri_of[starts[pair] + 1]
    return t0, t1


def _vertex_tri_map(F: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = F.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_verts)
    starts = np.cumsum(counts) - counts
    return starts, counts, np.repeat(np.arange(len(F)), 3)[order]


def _split_pass(
    V: np.ndarray, F: np.ndarray, limit: float, project
) -> tuple[np.ndarray, np.ndarray, int]:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lengths = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    long_mask = lengths > limit
    if not long_mask.any():
        return V, F, 0
    edges = np.unique(np.sort(E[long_mask], axis=1), axis=0)
    mids = project((V[edges[:, 0]] + V[edges[:, 1]]) / 2.0)
    base = len(V)
    V = np.concatenate([V, mids])
    n_all = base + len(edges)
    ekey = np.sort(
        np.minimum(edges[:, 0], edges[:, 1]) * n_all + np.maximum(edges[:, 0], edges[:, 1])
    )
    qkey = np.stack(
        [
            np.minimum(F[:, 0], F[:, 1]) * n_all + np.maximum(F[:, 0], F[:, 1]),
            np.minimum(F[:, 1], F[:, 2]) * n_all + np.maximum(F[:, 1], F[:, 2]),
            np.minimum(F[:, 2], F[:, 0]) * n_all + np.maximum(F[:, 2], F[:, 0]),
        ],
        axis=1,
    )
    pos = np.searchsorted(ekey, qkey)
    pos = np.clip(pos, 0, len(ekey) - 1)
    hit = ekey[pos] == qkey
    mid_ids = np.where(hit, base + pos, -1)
    ab, bc, ca = mid_ids[:, 0], mid_ids[:, 1], mid_ids[:, 2]
    has = (ab >= 0).astype(np.int64) + (bc >= 0).astype(np.int64) + (ca >= 0).astype(np.int64)
    blocks: list[np.ndarray] = []
    plain = np.nonzero(has == 0)[0]
    if len(plain):
        blocks.append(F[plain])
    full = np.nonzero(has == 3)[0]
    if len(full):
        a, b, c = F[full, 0], F[full, 1], F[full, 2]
        blocks.append(np.stack([a, ab[full], ca[full]], axis=1))
        blocks.append(np.stack([b, bc[full], ab[full]], axis=1))
        blocks.append(np.stack([c, ca[full], bc[full]], axis=1))
        blocks.append(np.stack([ab[full], bc[full], ca[full]], axis=1))
    two = np.nonzero(has == 2)[0]
    if len(two):
        miss = np.where(ab[two] < 0, 0, np.where(bc[two] < 0, 1, 2))
        mid = np.stack([ab[two], bc[two], ca[two]], axis=1)
        for kk in range(3):
            g = two[miss == kk]
            if len(g) == 0:
                continue
            v0 = F[g, kk]
            v1 = F[g, (kk + 1) % 3]
            v2 = F[g, (kk + 2) % 3]
            m0 = mid[miss == kk, (kk + 1) % 3]
            m1 = mid[miss == kk, (kk + 2) % 3]
            blocks.append(np.stack([v2, m1, m0], axis=1))
            blocks.append(np.stack([v0, v1, m0], axis=1))
            blocks.append(np.stack([v0, m0, m1], axis=1))
    one = np.nonzero(has == 1)[0]
    if len(one):
        hit = np.where(ab[one] >= 0, 0, np.where(bc[one] >= 0, 1, 2))
        mid = np.stack([ab[one], bc[one], ca[one]], axis=1)
        for kk in range(3):
            g = one[hit == kk]
            if len(g) == 0:
                continue
            v0 = F[g, kk]
            v1 = F[g, (kk + 1) % 3]
            v2 = F[g, (kk + 2) % 3]
            m = mid[hit == kk, kk]
            blocks.append(np.stack([v0, m, v2], axis=1))
            blocks.append(np.stack([m, v1, v2], axis=1))
    return V, np.concatenate(blocks), len(edges)


def _collapse_pass(
    V: np.ndarray, F: np.ndarray, limit: float, project
) -> tuple[np.ndarray, np.ndarray, int]:
    collapses = 0
    for _ in range(8):
        uniq, inverse, counts = _unique_edges(F)
        lengths = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        cand = np.nonzero((lengths < limit) & (counts == 2))[0]
        if len(cand) == 0:
            break
        ordered = np.lexsort(
            (
                V[uniq[cand, 1], 2],
                V[uniq[cand, 1], 1],
                V[uniq[cand, 1], 0],
                V[uniq[cand, 0], 2],
                V[uniq[cand, 0], 1],
                V[uniq[cand, 0], 0],
                lengths[cand],
            )
        )
        cand = cand[ordered]
        t0, t1 = _manifold_owners(len(F), counts, inverse)
        starts, vcounts, tri_of = _vertex_tri_map(F, len(V))
        touch = np.zeros(len(F), dtype=bool)
        remap = np.arange(len(V), dtype=np.int64)
        new_pts: list[np.ndarray] = []
        accepted = 0
        for e in cand.tolist():
            a, b = int(uniq[e, 0]), int(uniq[e, 1])
            if remap[a] != a or remap[b] != b:
                continue
            r0, r1 = int(t0[e]), int(t1[e])
            if r0 < 0 or touch[r0] or touch[r1]:
                continue
            opp = [v for v in F[r0].tolist() + F[r1].tolist() if v != a and v != b]
            if len(opp) != 2:
                continue
            ta = tri_of[starts[a] : starts[a] + vcounts[a]]
            tb = tri_of[starts[b] : starts[b] + vcounts[b]]
            if set(ta.tolist()) & set(tb.tolist()) != {r0, r1}:
                continue
            na = set(np.unique(F[ta]).tolist()) - {a}
            nb = set(np.unique(F[tb]).tolist()) - {b}
            if na & nb != set(opp):
                continue
            aff = np.unique(
                np.concatenate(
                    [
                        tri_of[starts[a] : starts[a] + vcounts[a]],
                        tri_of[starts[b] : starts[b] + vcounts[b]],
                    ]
                )
            )
            aff = aff[~((F[aff] == a).any(axis=1) & (F[aff] == b).any(axis=1))]
            if touch[aff].any():
                continue
            w = project(((V[a] + V[b]) / 2.0)[None, :])[0]
            old = V[F[aff]]
            new = old.copy()
            sel = F[aff] == a
            new[sel] = w
            sel = F[aff] == b
            new[sel] = w
            n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
            n_new = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
            if (np.linalg.norm(n_new, axis=1) <= 1e-14).any():
                continue
            if ((n_old * n_new).sum(axis=1) <= 0.0).any():
                continue
            nv = len(V) + len(new_pts)
            new_pts.append(w)
            remap[a] = nv
            remap[b] = nv
            touch[aff] = True
            accepted += 1
        if accepted == 0:
            break
        V = np.concatenate([V] + [p[None, :] for p in new_pts])
        F = remap[F]
        dead = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])
        F = F[~dead]
        collapses += accepted
    return V, F, collapses


def _flip_pass(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flips = 0
    for _ in range(3):
        uniq, inverse, counts = _unique_edges(F)
        cand = np.nonzero(counts == 2)[0]
        if len(cand) == 0:
            break
        ordered = np.lexsort(
            (
                V[uniq[cand, 1], 2],
                V[uniq[cand, 1], 1],
                V[uniq[cand, 1], 0],
                V[uniq[cand, 0], 2],
                V[uniq[cand, 0], 1],
                V[uniq[cand, 0], 0],
            )
        )
        cand = cand[ordered]
        t0, t1 = _manifold_owners(len(F), counts, inverse)
        val = np.bincount(F.ravel(), minlength=len(V))
        starts, vcounts, tri_of = _vertex_tri_map(F, len(V))
        touch = np.zeros(len(F), dtype=bool)
        swap: list[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = []
        for e in cand.tolist():
            a, b = int(uniq[e, 0]), int(uniq[e, 1])
            r0, r1 = int(t0[e]), int(t1[e])
            if r0 < 0 or touch[r0] or touch[r1]:
                continue
            opp = [v for v in F[r0].tolist() + F[r1].tolist() if v != a and v != b]
            if len(opp) != 2:
                continue
            c, d = opp
            if d in set(tri_of[starts[c] : starts[c] + vcounts[c]].tolist()):
                continue
            before = (
                abs(int(val[a]) - 6)
                + abs(int(val[b]) - 6)
                + abs(int(val[c]) - 6)
                + abs(int(val[d]) - 6)
            )
            after = (
                abs(int(val[a]) - 7)
                + abs(int(val[b]) - 7)
                + abs(int(val[c]) - 5)
                + abs(int(val[d]) - 5)
            )
            if after >= before:
                continue
            n_old0 = np.cross(V[F[r0, 1]] - V[F[r0, 0]], V[F[r0, 2]] - V[F[r0, 0]])
            n_old1 = np.cross(V[F[r1, 1]] - V[F[r1, 0]], V[F[r1, 2]] - V[F[r1, 0]])
            row0 = np.where(F[r0] == b, d, F[r0])
            row1 = np.where(F[r1] == a, c, F[r1])
            n_new0 = np.cross(V[row0[1]] - V[row0[0]], V[row0[2]] - V[row0[0]])
            n_new1 = np.cross(V[row1[1]] - V[row1[0]], V[row1[2]] - V[row1[0]])
            if min(float(np.linalg.norm(n_new0)), float(np.linalg.norm(n_new1))) <= 1e-14:
                continue
            if float(n_new0 @ n_old0) <= 0.0 or float(n_new1 @ n_old1) <= 0.0:
                continue
            swap.append((r0, tuple(int(v) for v in row0), (r1, tuple(int(v) for v in row1))))
            val[a] -= 1
            val[b] -= 1
            val[c] += 1
            val[d] += 1
            touch[r0] = True
            touch[r1] = True
            flips += 1
        if not swap:
            break
        F = F.copy()
        for r0, tri0, rest in swap:
            r1, tri1 = rest
            F[r0] = tri0
            F[r1] = tri1
    return V, F, flips


def _relax_pass(V: np.ndarray, F: np.ndarray, project) -> np.ndarray:
    E = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 0]], F[:, [1, 2]], F[:, [2, 1]], F[:, [2, 0]], F[:, [0, 2]]]
    )
    sums = np.zeros_like(V)
    np.add.at(sums, E[:, 0], V[E[:, 1]])
    cnt = np.bincount(E[:, 0], minlength=len(V))
    umbrella = sums / np.maximum(cnt, 1)[:, None] - V
    e0 = V[F[:, 1]] - V[F[:, 0]]
    e1 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e0, e1)
    area = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(area, 1e-300)
    acc = np.zeros_like(V)
    np.add.at(acc, F[:, 0], fn * area)
    np.add.at(acc, F[:, 1], fn * area)
    np.add.at(acc, F[:, 2], fn * area)
    n = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
    tangential = umbrella - (umbrella * n).sum(axis=1, keepdims=True) * n
    return project(V + 0.5 * tangential)


@register(
    "isotropic_remesh",
    "processing",
    "identity",
    "uniform target edge diagonal * 0.02 * 2**severity (diagonal/50 at severity 0+, "
    "/25 at severity 1); split edges longer than 4/3 of target, collapse shorter "
    "than 4/5, flip toward valence 6, tangential relax and reproject to the source",
    preserves_watertight=True,
)
def isotropic_remesh(mesh, severity, rng):
    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    diagonal = bbox_diagonal(src_tris)
    h = target_edge_mm(severity, diagonal)
    pts, _ = sample_source(np.asarray(src_tris, dtype=np.float64))
    tree = cKDTree(pts)

    def project(p: np.ndarray) -> np.ndarray:
        idx = tree.query(np.asarray(p, dtype=np.float64).reshape(-1, 3), k=1)[1]
        return pts[np.asarray(idx).reshape(-1)]

    V, F = _weld(src_tris)
    passes = remesh_passes(severity)
    splits = collapses = flips = 0
    for _ in range(passes):
        V, F, n = _split_pass(V, F, 4.0 / 3.0 * h, project)
        splits += n
        V, F = _compact(V, F)
        V, F, n = _collapse_pass(V, F, 4.0 / 5.0 * h, project)
        collapses += n
        V, F = _compact(V, F)
        V, F, n = _flip_pass(V, F)
        flips += n
        V = _relax_pass(V, F, project)
    V, F, dropped = _drop_degenerate(V, F)
    mesh.tris = V[F].reshape(-1, 3, 3)
    label = transfer_labels(mesh, src_tris, src_ids)
    return {
        "target_edge_mm": h,
        "bbox_diagonal_mm": diagonal,
        "passes": passes,
        "splits": splits,
        "collapses": collapses,
        "flips": flips,
        "degenerate_dropped": dropped,
        "triangles_before": before,
        "triangles_after": len(mesh.tris),
        **label,
    }


SMOOTH_MAX_ITERS = 20
LAPLACIAN_LAMBDA = 0.5


def smoothing_iterations(severity: float) -> int:
    return max(1, int(round(SMOOTH_MAX_ITERS * severity)))


def _umbrella_field(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    E = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 0]], F[:, [1, 2]], F[:, [2, 1]], F[:, [2, 0]], F[:, [0, 2]]]
    )
    sums = np.zeros_like(V)
    np.add.at(sums, E[:, 0], V[E[:, 1]])
    cnt = np.bincount(E[:, 0], minlength=len(V))
    return sums / np.maximum(cnt, 1)[:, None] - V


def _budgeted_smooth(
    mesh, severity: float, rate: float, mu: float | None, iters: int
) -> tuple[float, bool, float]:
    uniq, inverse = vertex_table(mesh)
    V0 = uniq.copy()
    diagonal = bbox_diagonal(mesh.tris)
    budget = float(severity) * 0.01 * diagonal
    pos = V0.copy()
    F = inverse.reshape(-1, 3)
    for _ in range(iters):
        pos = pos + rate * _umbrella_field(pos, F)
        if mu is not None:
            pos = pos + mu * _umbrella_field(pos, F)
    disp = pos - V0
    peak = float(np.linalg.norm(disp, axis=1).max()) if len(disp) else 0.0
    capped = peak > budget
    if capped:
        disp = disp * (budget / peak)
    peak = displace_vertices(mesh, V0, inverse, disp)
    mesh.metadata.setdefault(CONFIDENCE_KEY, [1.0] * len(mesh.tris))
    return peak, capped, budget


@register(
    "laplacian_smoothing",
    "processing",
    "identity",
    "max(1, round(20 * severity)) umbrella passes (lambda 0.5) capped to a max "
    "displacement of severity * 1% of the bounding-box diagonal (1% at severity 1); "
    "connectivity unchanged, labels kept exactly, shared vertices move once so "
    "the mesh stays watertight",
    preserves_watertight=True,
)
def laplacian_smoothing(mesh, severity, rng):
    before = len(mesh.tris)
    iters = smoothing_iterations(severity)
    peak, capped, budget = _budgeted_smooth(mesh, severity, LAPLACIAN_LAMBDA, None, iters)
    return {
        "iterations": iters,
        "lambda": LAPLACIAN_LAMBDA,
        "max_displacement_mm": peak,
        "displacement_budget_mm": budget,
        "capped_to_budget": capped,
        "triangles_before": before,
        "triangles_after": len(mesh.tris),
        "mean_confidence": 1.0,
        "fraction_below_0_9": 0.0,
    }


TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
TAUBIN_MAX_PAIRS = 10


def taubin_pairs(severity: float) -> int:
    return max(1, int(round(TAUBIN_MAX_PAIRS * severity)))


@register(
    "taubin_smoothing",
    "processing",
    "identity",
    "max(1, round(10 * severity)) Taubin lambda|mu pass pairs (lambda 0.5, mu -0.53) "
    "capped to a max displacement of severity * 1% of the bounding-box diagonal; "
    "connectivity unchanged, labels kept exactly, shared vertices move once so "
    "the mesh stays watertight",
    preserves_watertight=True,
)
def taubin_smoothing(mesh, severity, rng):
    before = len(mesh.tris)
    pairs = taubin_pairs(severity)
    peak, capped, budget = _budgeted_smooth(mesh, severity, TAUBIN_LAMBDA, TAUBIN_MU, pairs)
    return {
        "pairs": pairs,
        "lambda": TAUBIN_LAMBDA,
        "mu": TAUBIN_MU,
        "max_displacement_mm": peak,
        "displacement_budget_mm": budget,
        "capped_to_budget": capped,
        "triangles_before": before,
        "triangles_after": len(mesh.tris),
        "mean_confidence": 1.0,
        "fraction_below_0_9": 0.0,
    }


def cluster_cell_mm(severity: float, median_edge: float, diagonal: float) -> float:
    return min(median_edge * (1.0 + 1.5 * severity), diagonal / 8.0)


@register(
    "vertex_clustering",
    "processing",
    "identity",
    "vertex clustering on cells of min(median edge * (1.0 + 1.5 * severity), "
    "diagonal / 8) (about one median edge at severity 0+, capped so minimal "
    "meshes survive): corners in the same cell "
    "merge at their mean, degenerate and duplicate triangles drop; clustering "
    "can join thin walls, so watertightness is not guaranteed",
    preserves_watertight=False,
)
def vertex_clustering(mesh, severity, rng):
    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    V, F = _weld(src_tris)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    median_edge = float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))
    diagonal = bbox_diagonal(src_tris)
    size = cluster_cell_mm(severity, median_edge, diagonal)
    corners = src_tris.reshape(-1, 3)
    lo = corners.min(axis=0)
    keys = [tuple(row) for row in np.floor((corners - lo) / size).astype(np.int64).tolist()]
    order = sorted(set(keys))
    index = {k: i for i, k in enumerate(order)}
    per_corner = [index[k] for k in keys]
    sums = np.zeros((len(order), 3))
    counts = np.zeros(len(order))
    np.add.at(sums, per_corner, corners)
    np.add.at(counts, per_corner, 1)
    pos = sums / counts[:, None]
    rows = []
    degenerate = duplicates = 0
    seen: set[tuple[tuple, tuple, tuple]] = set()
    for t in pos[np.array(per_corner).reshape(-1, 3)]:
        key = (tuple(t[0]), tuple(t[1]), tuple(t[2]))
        if key[0] == key[1] or key[1] == key[2] or key[2] == key[0]:
            degenerate += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append(t)
    mesh.tris = np.array(rows, dtype=np.float64).reshape(-1, 3, 3)
    label = transfer_labels(mesh, src_tris, src_ids, mesh.metadata.get(CONFIDENCE_KEY))
    return {
        "cell_mm": size,
        "median_edge_mm": median_edge,
        "bbox_diagonal_mm": diagonal,
        "occupied_cells": len(order),
        "degenerate_dropped": degenerate,
        "duplicates_dropped": duplicates,
        "triangles_before": before,
        "triangles_after": len(rows),
        **label,
    }
