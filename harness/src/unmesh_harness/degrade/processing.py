"""Processing-artifact degradation operators.

Each operator is a pure function ``(LabeledMesh, severity, rng) -> params``
registered under the ``processing`` family in ``unmesh_harness.degrade.OPERATORS``.
They model what mesh-processing pipelines do to CAD tessellations: quadric
decimation, isotropic and voxel remeshing, and Laplacian / Taubin smoothing.

All five are implemented here in numpy with no third-party mesh dependency.
trimesh and open3d are both MIT-licensed, so either would be allowed, but a
small deterministic implementation keeps the harness dependency-free and
bit-identical across processes. pymeshlab (GPL-3.0-or-later) is not used at
all, so no operator ever skips and no test needs it.

Label policy: every output triangle keeps the face table, adjacency, edge
polylines, vertices and shells of the clean mesh as the ground truth it is
judged against, and takes the ``face_id`` of its nearest source face, measured
by triangle-centroid distance to each source face's analytic surface (centroid
distance for non-analytic faces). Each triangle also gets a
``label_confidence`` in [0, 1] (``mesh.metadata["label_confidence"]``),
``d2 / (d1 + d2)`` where ``d1`` is the distance to the nearest source face and
``d2`` to the second-nearest (1.0 when the mesh has a single face). Metrics
that score per-triangle labels should mask triangles below 0.9 confidence.
The smoothing operators move shared vertices, so their adjacency polylines move
with the mesh; the remeshing operators leave the polylines on the clean mesh.
"""

from __future__ import annotations

import math

import numpy as np

from ..labels import distance_to_surface
from .core import displace_vertices, register, vertex_table
from .refine import _split_triangle

CONFIDENCE_KEY = "label_confidence"
MASK_THRESHOLD = 0.9
POLYLINES_COINCIDE = ("laplacian_smoothing", "taubin_smoothing")

_TRANSFER_CHUNK = 512
_FOLD_EPS = 1e-14


def bbox_diagonal(tris: np.ndarray) -> float:
    flat = np.asarray(tris, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0)))


def transfer_labels(mesh, src_tris, src_ids) -> dict:
    src_ids = np.asarray(src_ids)
    src_cent = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 3).mean(axis=1)
    new_cent = np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3).mean(axis=1)
    faces = np.unique(src_ids)
    analytic = [mesh.faces[int(f)].surface != "other" for f in faces]
    fallback = [src_cent[src_ids == f] for f in faces]
    fids = np.empty(len(new_cent), dtype=src_ids.dtype)
    conf = np.ones(len(new_cent), dtype=np.float64)
    for start in range(0, len(new_cent), _TRANSFER_CHUNK):
        chunk = new_cent[start : start + _TRANSFER_CHUNK]
        best = np.empty((len(chunk), len(faces)))
        for j, f in enumerate(faces):
            if analytic[j]:
                d = distance_to_surface(mesh.faces[int(f)], chunk)
            else:
                d = np.linalg.norm(chunk[:, None, :] - fallback[j][None, :, :], axis=2).min(axis=1)
            best[:, j] = np.where(np.isfinite(d), d, 1e300)
        order = np.argsort(best, axis=1, kind="stable")
        rows = np.arange(len(chunk))
        d1 = best[rows, order[:, 0]]
        fids[start : start + _TRANSFER_CHUNK] = faces[order[:, 0]]
        if len(faces) > 1:
            d2 = best[rows, order[:, 1]]
            denom = d1 + d2
            conf[start : start + _TRANSFER_CHUNK] = np.where(
                denom > 0, d2 / np.maximum(denom, 1e-300), 1.0
            )
    mesh.face_id = np.array(fids, dtype=src_ids.dtype)
    mesh.metadata[CONFIDENCE_KEY] = [float(v) for v in conf]
    return {
        "mean_confidence": float(conf.mean()) if len(conf) else 1.0,
        "fraction_below_0_9": float((conf < MASK_THRESHOLD).mean()) if len(conf) else 0.0,
    }


def _owners(ids: list[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = {}
    for ti, (a, b, c) in enumerate(ids):
        for u, v in ((a, b), (b, c), (c, a)):
            owners.setdefault((min(u, v), max(u, v)), []).append(ti)
    return owners


def _live_neighbors(ids: list[tuple[int, int, int]], live: list[bool], n: int) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for ti, (a, b, c) in enumerate(ids):
        if not live[ti]:
            continue
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return nbrs


def _link_ok(
    a: int,
    b: int,
    ids: list[tuple[int, int, int]],
    live: list[bool],
    owners: dict[tuple[int, int], list[int]],
    nbrs: list[set[int]],
) -> bool:
    opposite = set()
    for ti in owners.get((min(a, b), max(a, b)), []):
        if not live[ti]:
            continue
        rest = [v for v in ids[ti] if v != a and v != b]
        if len(rest) != 1:
            return False
        opposite.add(rest[0])
    return nbrs[a] & nbrs[b] == opposite


def _fold_ok(
    a: int,
    b: int,
    w: np.ndarray,
    ids: list[tuple[int, int, int]],
    live: list[bool],
    coords: list[np.ndarray],
) -> bool:
    for ti, (x, y, z) in enumerate(ids):
        if not live[ti]:
            continue
        hits = (x == a or x == b, y == a or y == b, z == a or z == b)
        if sum(hits) != 1:
            continue
        old = [coords[x], coords[y], coords[z]]
        new = [w if hit else coords[v] for hit, v in zip(hits, (x, y, z), strict=True)]
        n_old = np.cross(old[1] - old[0], old[2] - old[0])
        n_new = np.cross(new[1] - new[0], new[2] - new[0])
        if float(np.linalg.norm(n_new)) <= _FOLD_EPS:
            return False
        if float(n_old @ n_new) <= 0.0:
            return False
    return True


def _count_ok(
    a: int,
    b: int,
    ids: list[tuple[int, int, int]],
    live: list[bool],
    owners: dict[tuple[int, int], list[int]],
    fids: list[int],
    counts: list[int],
) -> bool:
    dying: dict[int, int] = {}
    for ti in owners.get((min(a, b), max(a, b)), []):
        if live[ti]:
            dying[fids[ti]] = dying.get(fids[ti], 0) + 1
    return all(counts[f] > dying[f] for f in dying)


def _collapse(
    a: int,
    b: int,
    w: np.ndarray,
    ids: list[tuple[int, int, int]],
    live: list[bool],
    fids: list[int],
    counts: list[int],
    coords: list[np.ndarray],
) -> None:
    nv = len(coords)
    coords.append(np.asarray(w, dtype=np.float64))
    for ti, (x, y, z) in enumerate(ids):
        if not live[ti]:
            continue
        hits = (x == a or x == b, y == a or y == b, z == a or z == b)
        if sum(hits) > 1:
            live[ti] = False
            counts[fids[ti]] -= 1
        elif sum(hits) == 1:
            sub = tuple(nv if hit else v for hit, v in zip(hits, (x, y, z), strict=True))
            if len(set(sub)) < 3:
                live[ti] = False
                counts[fids[ti]] -= 1
            else:
                ids[ti] = sub


def _indexed(
    tris: list[tuple[tuple, tuple, tuple]],
) -> tuple[list[tuple[int, int, int]], list[np.ndarray]]:
    table = sorted({p for t in tris for p in t})
    index = {p: i for i, p in enumerate(table)}
    coords = [np.array(p, dtype=np.float64) for p in table]
    return [tuple(index[p] for p in t) for t in tris], coords


def _plane_quadrics(
    ids: list[tuple[int, int, int]], live: list[bool], coords: list[np.ndarray]
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for ti, (a, b, c) in enumerate(ids):
        if not live[ti]:
            continue
        n = np.cross(coords[b] - coords[a], coords[c] - coords[a])
        area = float(np.linalg.norm(n))
        if area <= 0.0:
            continue
        un = n / area
        plane = np.append(un, -float(un @ coords[a]))
        q = np.outer(plane, plane) * (area / 2.0)
        for v in (a, b, c):
            out[v] = out.get(v, np.zeros((4, 4))) + q
    return out


def _edge_cost(a: int, b: int, quad: dict[int, np.ndarray], coords: list[np.ndarray]) -> float:
    m = np.append((coords[a] + coords[b]) / 2.0, 1.0)
    q = quad.get(a, np.zeros((4, 4))) + quad.get(b, np.zeros((4, 4)))
    return float(m @ q @ m)


DECIMATE_KEEP_AT_ONE = 0.1
DECIMATE_MAX_PASSES = 100


@register(
    "quadric_decimation",
    "processing",
    "identity",
    "keep ratio 1 - 0.9 * severity of triangles (10% at severity 1), quadric-error "
    "ordered edge collapses to the edge midpoint with link-condition and fold guards, "
    "best effort without removing any face's last triangle",
    preserves_watertight=True,
)
def quadric_decimation(mesh, severity, rng):
    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    keep = 1.0 - 0.9 * severity
    target = max(1, int(round(before * keep)))
    corners = src_tris.reshape(-1, 3)
    uniq, inverse = np.unique(corners, axis=0, return_inverse=True)
    coords = [row for row in uniq]
    ids = [tuple(int(v) for v in t) for t in inverse.reshape(-1, 3).tolist()]
    fids = [int(v) for v in src_ids.tolist()]
    live = [True] * len(ids)
    counts = [0] * len(mesh.faces)
    for f in fids:
        counts[f] += 1
    collapses = 0
    for _ in range(DECIMATE_MAX_PASSES):
        if sum(live) <= target:
            break
        nbrs = _live_neighbors(ids, live, len(coords))
        owners = _owners(ids)
        quad = _plane_quadrics(ids, live, coords)
        edges = sorted(
            {e for e, members in owners.items() if any(live[t] for t in members)},
            key=lambda e: (
                _edge_cost(e[0], e[1], quad, coords),
                tuple(coords[e[0]].tolist()),
                tuple(coords[e[1]].tolist()),
            ),
        )
        used: set[int] = set()
        moved = False
        for a, b in edges:
            if sum(live) <= target:
                break
            if a in used or b in used:
                continue
            if not _link_ok(a, b, ids, live, owners, nbrs):
                continue
            w = (coords[a] + coords[b]) / 2.0
            if not _fold_ok(a, b, w, ids, live, coords):
                continue
            if not _count_ok(a, b, ids, live, owners, fids, counts):
                continue
            _collapse(a, b, w, ids, live, fids, counts, coords)
            nbrs = _live_neighbors(ids, live, len(coords))
            owners = _owners(ids)
            used.update((a, b))
            collapses += 1
            moved = True
        if not moved:
            break
    kept = [i for i, v in enumerate(live) if v]
    mesh.tris = np.array(
        [[coords[v].tolist() for v in ids[i]] for i in kept], dtype=np.float64
    ).reshape(-1, 3, 3)
    label = transfer_labels(mesh, src_tris, src_ids)
    return {
        "keep_ratio_target": keep,
        "keep_ratio_achieved": len(kept) / before,
        "target_triangles": target,
        "collapses": collapses,
        "triangles_before": before,
        "triangles_after": len(kept),
        **label,
    }


REMESH_EDGE_AT_ZERO = 0.02
REMESH_GROWTH = 2.0


def target_edge_mm(severity: float, diagonal: float) -> float:
    return diagonal * REMESH_EDGE_AT_ZERO * REMESH_GROWTH**severity


def remesh_passes(severity: float) -> int:
    return 1 + int(round(2.0 * severity))


def _split_long(
    tris: list[tuple[tuple, tuple, tuple]], fids: list[int], limit: float
) -> tuple[list[tuple[tuple, tuple, tuple]], list[int], int]:
    def key(p, q):
        return (p, q) if p <= q else (q, p)

    edges = set()
    for t in tris:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if math.dist(p, q) > limit:
                edges.add(key(p, q))
    if not edges:
        return tris, fids, 0
    mids = {}
    for k in sorted(edges):
        p, q = k
        mids[k] = tuple(((np.array(p) + np.array(q)) / 2.0).tolist())
    new_tris, new_ids = [], []
    for t, f in zip(tris, fids, strict=True):
        pieces = _split_triangle(t, mids, f)
        new_tris += pieces
        new_ids += [f] * len(pieces)
    return new_tris, new_ids, len(mids)


def _collapse_short(
    tris: list[tuple[tuple, tuple, tuple]], fids: list[int], limit: float, nfaces: int
) -> tuple[list[tuple[tuple, tuple, tuple]], list[int], int]:
    ids, coords = _indexed(tris)
    live = [True] * len(ids)
    counts = [0] * nfaces
    for f in fids:
        counts[f] += 1
    nbrs = _live_neighbors(ids, live, len(coords))
    owners = _owners(ids)
    edges = sorted(
        (
            e
            for e, members in owners.items()
            if any(live[t] for t in members) and math.dist(coords[e[0]], coords[e[1]]) < limit
        ),
        key=lambda e: (
            math.dist(coords[e[0]], coords[e[1]]),
            tuple(coords[e[0]].tolist()),
            tuple(coords[e[1]].tolist()),
        ),
    )
    collapses = 0
    used: set[int] = set()
    for a, b in edges:
        if a in used or b in used:
            continue
        if not _link_ok(a, b, ids, live, owners, nbrs):
            continue
        w = (coords[a] + coords[b]) / 2.0
        if not _fold_ok(a, b, w, ids, live, coords):
            continue
        if not _count_ok(a, b, ids, live, owners, fids, counts):
            continue
        _collapse(a, b, w, ids, live, fids, counts, coords)
        nbrs = _live_neighbors(ids, live, len(coords))
        owners = _owners(ids)
        used.update((a, b))
        collapses += 1
    table = [tuple(c.tolist()) for c in coords]
    return (
        [tuple(table[v] for v in ids[i]) for i, v in enumerate(live) if v],
        [f for f, v in zip(fids, live, strict=True) if v],
        collapses,
    )


@register(
    "isotropic_remesh",
    "processing",
    "identity",
    "uniform target edge diagonal * 0.02 * 2**severity (diagonal/50 at severity 0+, "
    "/25 at severity 1); 1 + round(2*severity) passes splitting edges longer than 4/3 "
    "of target conformingly at midpoints and collapsing edges shorter than 4/5 of "
    "target to midpoints under manifold guards",
    preserves_watertight=True,
)
def isotropic_remesh(mesh, severity, rng):
    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    diagonal = bbox_diagonal(src_tris)
    h = target_edge_mm(severity, diagonal)
    passes = remesh_passes(severity)
    tris = [tuple(tuple(v) for v in t) for t in mesh.tris.tolist()]
    fids = mesh.face_id.tolist()
    splits = collapses = 0
    for _ in range(passes):
        tris, fids, n_split = _split_long(tris, fids, 4.0 / 3.0 * h)
        tris, fids, n_coll = _collapse_short(tris, fids, 4.0 / 5.0 * h, len(mesh.faces))
        splits += n_split
        collapses += n_coll
        if not n_split and not n_coll:
            break
    mesh.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 3)
    label = transfer_labels(mesh, src_tris, src_ids)
    return {
        "target_edge_mm": h,
        "bbox_diagonal_mm": diagonal,
        "passes": passes,
        "splits": splits,
        "collapses": collapses,
        "triangles_before": before,
        "triangles_after": len(tris),
        **label,
    }


SMOOTH_MAX_ITERS = 20
LAPLACIAN_LAMBDA = 0.5


def smoothing_iterations(severity: float) -> int:
    return int(round(SMOOTH_MAX_ITERS * severity))


def _umbrella_neighbors(mesh) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    uniq, inverse = vertex_table(mesh)
    corners = inverse.reshape(-1, 3).tolist()
    nbrs: list[set[int]] = [set() for _ in range(len(uniq))]
    for a, b, c in corners:
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return uniq, inverse, [np.array(sorted(s), dtype=np.int64) for s in nbrs]


def _laplacian_step(pos: np.ndarray, nbrs: list[np.ndarray], rate: float) -> np.ndarray:
    out = pos.copy()
    for i, js in enumerate(nbrs):
        if len(js):
            out[i] = pos[i] + rate * (pos[js].mean(axis=0) - pos[i])
    return out


@register(
    "laplacian_smoothing",
    "processing",
    "identity",
    "round(20 * severity) umbrella passes (lambda 0.5) moving each distinct vertex "
    "toward its neighbour mean (20 at severity 1); displacement scales with the local "
    "chord length, so coarse meshes move more; connectivity unchanged, shared "
    "vertices move once so the mesh stays watertight",
    preserves_watertight=True,
)
def laplacian_smoothing(mesh, severity, rng):
    src_tris = mesh.tris.copy()
    src_ids = mesh.face_id.copy()
    uniq, inverse, nbrs = _umbrella_neighbors(mesh)
    pos = uniq.copy()
    iters = smoothing_iterations(severity)
    for _ in range(iters):
        pos = _laplacian_step(pos, nbrs, LAPLACIAN_LAMBDA)
    peak = displace_vertices(mesh, uniq, inverse, pos - uniq)
    label = transfer_labels(mesh, src_tris, src_ids)
    return {
        "iterations": iters,
        "lambda": LAPLACIAN_LAMBDA,
        "max_displacement_mm": peak,
        "triangles_before": len(src_tris),
        "triangles_after": len(mesh.tris),
        **label,
    }
