"""Processing-artifact degradation operators.

Each operator is a pure function ``(LabeledMesh, severity, rng) -> params``
registered under the ``processing`` family in ``unmesh_harness.degrade.OPERATORS``.
They model what mesh-processing pipelines do to CAD tessellations: quadric
decimation (in-house Garland-Heckbert with link-condition and fold guards),
isotropic remeshing (an in-house Botsch-Kobbelt loop: split, collapse, flip,
tangential relax, reproject), Laplacian / Taubin smoothing, and vertex
clustering. All are numpy/scipy only and deterministic across platforms up to
float rounding: work is ordered by quantized costs and lexicographic vertex keys.

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
    todo = np.arange(len(flat))
    while len(todo):
        use = min(k, len(tree.data))
        idx = _canonical_knn(tree, flat[todo], use)
        cand = owner[idx]
        rows = np.repeat(todo, use)
        raw = _point_tri_dist2(
            flat[rows], src[cand.ravel(), 0], src[cand.ravel(), 1], src[cand.ravel(), 2]
        )
        raw = raw.reshape(len(todo), use)
        d2 = raw
        if src_n is not None:
            facing = (src_n[cand.ravel()] * normals[rows]).sum(axis=1) > 0.0
            d2 = np.where(facing.reshape(len(todo), use), raw, np.inf)
        floor = d2.min(axis=1, keepdims=True)
        pick = np.where(d2 <= floor + tol2, cand, len(src)).argmin(axis=1)
        here = np.arange(len(todo))
        row_best = d2[here, pick]
        improved = row_best < best_d[todo]
        best_d[todo[improved]] = row_best[improved]
        best_t[todo[improved]] = cand[here, pick][improved]
        kth = raw[:, -1]
        bd = best_d[todo]
        retry = (bd > tol2) & ((bd > 4.0 * kth) | ~np.isfinite(bd))
        if k >= _LABEL_KNN_CAP or use >= len(tree.data):
            break
        todo = todo[retry]
        k = min(k * 8, _LABEL_KNN_CAP)
    lost = best_t < 0
    if lost.any():
        best_t[lost], best_d[lost] = _exact_votes(flat[lost], src, owner, tree, tol2)
    return best_t, best_d


def _canonical_knn(tree: cKDTree, points: np.ndarray, k: int) -> np.ndarray:
    wide = min(2 * k, len(tree.data))
    dist, idx = tree.query(points, k=wide, workers=-1 if len(points) >= 4096 else 1)
    dist = np.asarray(dist, dtype=np.float32).reshape(len(points), wide)
    key = dist.view(np.int32).astype(np.int64) << 32
    key |= np.asarray(idx, dtype=np.int64).reshape(len(points), wide)
    key.sort(axis=1)
    return key[:, :k] & 0xFFFFFFFF


def confidence_summary(conf) -> dict:
    conf = np.asarray(conf, dtype=np.float64)
    if not len(conf):
        return {"mean_confidence": 1.0, "fraction_below_0_9": 0.0}
    return {
        "mean_confidence": float(conf.mean()),
        "fraction_below_0_9": float((conf < MASK_THRESHOLD).mean()),
    }


def transfer_labels(mesh, src_tris, src_ids, src_conf=None, samples=None) -> dict:
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
    if samples is None:
        pts, owner = sample_source(src)
        samples = (pts, owner, cKDTree(pts))
    _, owner, tree = samples
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
        knn = _canonical_knn(tree, probes[missing].reshape(-1, 3), _LABEL_KNN)
        knn = knn.reshape(len(missing), 4, _LABEL_KNN)
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
    conf[welds_away(out, bbox_diagonal(src))] = 0.0
    if src_conf is not None:
        carried = np.asarray(src_conf, dtype=np.float64)[best_t.reshape(len(out), 4)[:, 0]]
        conf = np.minimum(conf, carried)
    mesh.face_id = np.array(faces[win], dtype=src_ids.dtype)
    mesh.metadata[CONFIDENCE_KEY] = [float(v) for v in conf]
    return confidence_summary(conf)


def carry_confidence(
    mesh, src_tris: np.ndarray, src_ids: np.ndarray, src_conf, source=None
) -> None:
    conf = np.asarray(src_conf, dtype=np.float64)
    out = np.asarray(mesh.tris, dtype=np.float64).reshape(-1, 3, 3)
    src = np.asarray(src_tris, dtype=np.float64).reshape(-1, 3, 3)
    src_ids = np.asarray(src_ids)
    if source is None and len(out) == len(conf) and np.array_equal(mesh.face_id, src_ids):
        return
    carried = np.full(len(out), np.nan)
    if source is not None:
        source = np.asarray(source, dtype=np.int64)
        kept = source >= 0
        carried[kept] = conf[source[kept]]
    else:
        first: dict[tuple, int] = {}
        for i in range(len(src) - 1, -1, -1):
            first[(int(src_ids[i]), frozenset(map(tuple, src[i].tolist())))] = i
        for i, (tri, fid) in enumerate(zip(out.tolist(), mesh.face_id.tolist(), strict=True)):
            hit = first.get((int(fid), frozenset(map(tuple, tri))))
            if hit is not None:
                carried[i] = conf[hit]
    fresh = np.nonzero(np.isnan(carried))[0]
    if len(fresh):
        names, labels = np.unique(src_ids, return_inverse=True)
        pos = np.clip(np.searchsorted(names, mesh.face_id[fresh]), 0, len(names) - 1)
        known = names[pos] == mesh.face_id[fresh]
        want = np.where(known, pos, -1)
        if len(src) and known.any():
            proj = _Projector(src, labels)

            def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
                return labels[flat] == want[rows]

            tri = proj.search(out[fresh].mean(axis=1), accept)[1]
            near = np.where(labels[tri] == want, conf[tri], 0.0)
        else:
            near = np.zeros(len(fresh))
        carried[fresh] = np.where(known, near, 1.0)
    carried[welds_away(out, bbox_diagonal(src) if len(src) else 0.0)] = 0.0
    mesh.metadata[CONFIDENCE_KEY] = [float(v) for v in carried]


WELD_MM = 1e-6


def welds_away(tris: np.ndarray, diagonal: float) -> np.ndarray:
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    double_area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    shortest = np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2).min(axis=1)
    return (double_area <= (diagonal * _EXACT_TOL_REL) ** 2) | (shortest <= WELD_MM)


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
_QEM_COST_QUANTUM = 1e-12
_QEM_SNAP_BITS = 36


def _plane_quadrics(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    double_area = np.linalg.norm(normal, axis=1)
    unit = normal / np.maximum(double_area, 1e-300)[:, None]
    plane = np.concatenate([unit, -(unit * V[F[:, 0]]).sum(axis=1, keepdims=True)], axis=1)
    K = plane[:, :, None] * plane[:, None, :] * (0.5 * double_area)[:, None, None]
    Q = np.zeros((len(V), 4, 4))
    for j in range(3):
        np.add.at(Q, F[:, j], K)
    return Q


def _quadric_cost(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = np.concatenate([P, np.ones((len(P), 1))], axis=1)
    return np.einsum("ni,nij,nj->n", h, Q, h)


class _Qem:
    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V = V.copy()
        self.F = F.tolist()
        self.Q = _plane_quadrics(V, F)
        self.alive = [True] * len(F)
        self.faces_of: list[set[int]] = [set() for _ in range(len(V))]
        for t, row in enumerate(self.F):
            for v in row:
                self.faces_of[v].add(t)
        self.version = [0] * len(V)
        self.dead = [False] * len(V)
        lo = V.min(axis=0)
        hi = V.max(axis=0)
        diagonal = float(np.linalg.norm(hi - lo)) or 1.0
        self.quantum = _QEM_COST_QUANTUM * diagonal**4
        self.grid = diagonal * 2.0**-_QEM_SNAP_BITS
        self.area_floor = (diagonal * _EXACT_TOL_REL) ** 2
        self.boundary = self._boundary_vertices()
        self.heap: list[tuple[float, int, int, int, int, int]] = []

    def _boundary_vertices(self) -> set[int]:
        count: dict[tuple[int, int], int] = {}
        for a, b, c in self.F:
            for p, q in ((a, b), (b, c), (c, a)):
                key = (p, q) if p < q else (q, p)
                count[key] = count.get(key, 0) + 1
        return {v for key, n in count.items() if n != 2 for v in key}

    def ring(self, v: int) -> set[int]:
        return {x for t in self.faces_of[v] for x in self.F[t]} - {v}

    def push(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs:
            return
        a = np.array([p[0] for p in pairs], dtype=np.int64)
        b = np.array([p[1] for p in pairs], dtype=np.int64)
        Qs = self.Q[a] + self.Q[b]
        pa, pb = self.V[a], self.V[b]
        mid = (pa + pb) / 2.0
        A = Qs[:, :3, :3]
        rhs = -Qs[:, :3, 3]
        det = np.linalg.det(A)
        scale = np.abs(A).max(axis=(1, 2)) ** 3
        ok = np.abs(det) > 1e-10 * np.maximum(scale, 1e-300)
        opt = mid.copy()
        if ok.any():
            opt[ok] = np.linalg.solve(A[ok], rhs[ok][:, :, None])[:, :, 0]
        opt = np.round(opt / self.grid) * self.grid
        span = np.linalg.norm(pb - pa, axis=1)
        ok &= np.linalg.norm(opt - mid, axis=1) <= span
        cands = np.stack([pa, pb, mid, opt], axis=1)
        costs = np.stack([_quadric_cost(Qs, cands[:, j]) for j in range(4)], axis=1)
        costs[~ok, 3] = np.inf
        scaled = np.maximum(costs, 0.0) / self.quantum
        q = np.where(scaled < 1.0, 0.0, scaled.astype(np.float32).astype(np.float64))
        pick = q.argmin(axis=1)
        qcost = q[np.arange(len(pairs)), pick]
        for i, (x, y) in enumerate(pairs):
            heapq.heappush(
                self.heap,
                (float(qcost[i]), int(pick[i]), x, y, self.version[x], self.version[y]),
            )

    def position(self, a: int, b: int, pick: int) -> np.ndarray:
        Qs = self.Q[a] + self.Q[b]
        if pick == 0:
            return self.V[a].copy()
        if pick == 1:
            return self.V[b].copy()
        mid = (self.V[a] + self.V[b]) / 2.0
        if pick == 2:
            return mid
        opt = np.linalg.solve(Qs[:3, :3], -Qs[:3, 3])
        return np.round(opt / self.grid) * self.grid

    def collapse_ok(self, a: int, b: int, p: np.ndarray) -> tuple[bool, set[int]]:
        shared = self.faces_of[a] & self.faces_of[b]
        if len(shared) != 2:
            return False, shared
        opposite = {x for t in shared for x in self.F[t]} - {a, b}
        if self.ring(a) & self.ring(b) != opposite:
            return False, shared
        moved = sorted((self.faces_of[a] | self.faces_of[b]) - shared)
        if not moved:
            return False, shared
        rows = np.array([self.F[t] for t in moved], dtype=np.int64)
        old = self.V[rows]
        new = old.copy()
        new[(rows == a) | (rows == b)] = p
        n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
        n_new = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
        len_new = np.linalg.norm(n_new, axis=1)
        len_old = np.linalg.norm(n_old, axis=1)
        live = len_old > self.area_floor
        if (len_new[live] <= 1e-12 * len_old[live]).any():
            return False, shared
        if ((n_old[live] * n_new[live]).sum(axis=1) <= 0.0).any():
            return False, shared
        return True, shared

    def run(self, target: int) -> int:
        live = len(self.F)
        edges: set[tuple[int, int]] = set()
        for a, b, c in self.F:
            for p, q in ((a, b), (b, c), (c, a)):
                edges.add((p, q) if p < q else (q, p))
        fixed = self.boundary
        self.push(sorted(e for e in edges if e[0] not in fixed and e[1] not in fixed))
        collapses = 0
        while live > target and self.heap:
            _, pick, a, b, va, vb = heapq.heappop(self.heap)
            if self.dead[a] or self.dead[b] or va != self.version[a] or vb != self.version[b]:
                continue
            p = self.position(a, b, pick)
            ok, shared = self.collapse_ok(a, b, p)
            if not ok:
                continue
            for t in shared:
                self.alive[t] = False
                for v in self.F[t]:
                    self.faces_of[v].discard(t)
            for t in self.faces_of[b]:
                self.F[t] = [a if v == b else v for v in self.F[t]]
                self.faces_of[a].add(t)
            self.faces_of[b] = set()
            self.V[a] = p
            self.Q[a] = self.Q[a] + self.Q[b]
            self.dead[b] = True
            self.version[a] += 1
            self.version[b] += 1
            live -= 2
            collapses += 1
            ring = [n for n in self.ring(a) if n not in self.boundary]
            self.push(sorted((min(a, n), max(a, n)) for n in ring))
        return collapses

    def result(self) -> np.ndarray:
        rows = np.array([row for row, ok in zip(self.F, self.alive, strict=True) if ok])
        return self.V[rows.reshape(-1, 3)].reshape(-1, 3, 3) + 0.0


@register(
    "quadric_decimation",
    "processing",
    "identity",
    "keep ratio 1 - 0.9 * severity of triangles (10% at severity 1, 4-triangle floor): "
    "the soup is welded into an indexed mesh and edges collapse in order of "
    "Garland-Heckbert quadric error (area-weighted plane quadrics, accumulated) to "
    "the error-optimal position; collapses that would break the link condition or "
    "fold a triangle are refused, so a closed manifold stays closed and the mesh may "
    "stop above the target; open-boundary vertices never move; faces narrower than "
    "the collapse scale vanish",
    preserves_watertight=True,
)
def quadric_decimation(mesh, severity, rng):
    src_tris = np.asarray(mesh.tris, dtype=np.float64).copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    keep = 1.0 - 0.9 * severity
    target = max(DECIMATE_MIN_TRIANGLES, int(round(before * keep)))
    V, F = _weld(src_tris)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    qem = _Qem(V, F)
    collapses = qem.run(target)
    mesh.tris = qem.result()
    label = transfer_labels(mesh, src_tris, src_ids, mesh.metadata.get(CONFIDENCE_KEY))
    return {
        "keep_ratio_target": keep,
        "keep_ratio_achieved": len(mesh.tris) / before,
        "target_triangles": target,
        "collapses": collapses,
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


def _closest_on_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c
    d1 = (ab * ap).sum(axis=1)
    d2 = (ac * ap).sum(axis=1)
    d3 = (ab * bp).sum(axis=1)
    d4 = (ac * bp).sum(axis=1)
    d5 = (ab * cp).sum(axis=1)
    d6 = (ac * cp).sum(axis=1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        safe = np.where(den != 0.0, den, 1.0)
        return np.clip(np.where(den != 0.0, num / safe, 0.0), 0.0, 1.0)[:, None]

    denom = va + vb + vc
    v = ratio(vb, denom)
    w = ratio(vc, denom)
    out = a + ab * v + ac * w
    cases = [
        (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (d1 <= 0.0) & (d2 <= 0.0),
    ]
    values = [
        b + (c - b) * ratio(d4 - d3, (d4 - d3) + (d5 - d6)),
        a + ac * ratio(d2, d2 - d6),
        c,
        a + ab * ratio(d1, d1 - d3),
        b,
        a,
    ]
    for mask, value in zip(cases, values, strict=True):
        out[mask] = value[mask]
    return out


class _Projector:
    def __init__(self, src: np.ndarray, labels: np.ndarray):
        self.src = np.asarray(src, dtype=np.float64).reshape(-1, 3, 3)
        self.labels = np.asarray(labels, dtype=np.int64)
        normal = np.cross(self.src[:, 1] - self.src[:, 0], self.src[:, 2] - self.src[:, 0])
        self.normals = normal / np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        self.n_labels = int(self.labels.max()) + 1 if len(self.labels) else 1
        spread = np.zeros(self.n_labels)
        first = np.zeros((self.n_labels, 3))
        first[self.labels[::-1]] = self.normals[::-1]
        np.maximum.at(spread, self.labels, 1.0 - (self.normals * first[self.labels]).sum(axis=1))
        self.planar = spread <= 1e-12
        self.pts, self.owner = sample_source(self.src)
        self.tree = cKDTree(self.pts)
        self.tie2 = (bbox_diagonal(self.src) * _EXACT_TOL_REL) ** 2

    def incidence(self, F: np.ndarray, S: np.ndarray) -> np.ndarray:
        return np.unique(F.ravel() * self.n_labels + np.repeat(self.labels[S], 3))

    def facing(self, tris: np.ndarray, S: np.ndarray) -> np.ndarray:
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ok = (n * self.normals[S]).sum(axis=1) > 0.0
        if not ok.all() or self.planar[self.labels[S]].all():
            return ok
        want = self.labels[S]

        def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
            return self.labels[flat] == want[rows]

        local = self.search(tris.mean(axis=1), accept)[1]
        return ok & ((n * self.normals[local]).sum(axis=1) > 0.0)

    def search(self, P: np.ndarray, accept) -> tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        out = P.copy()
        tri = np.full(len(P), -1, dtype=np.int64)
        pending = np.arange(len(P))
        k = _PROJECT_KNN
        while len(pending):
            use = min(k, len(self.pts))
            idx = _canonical_knn(self.tree, P[pending], use)
            cand = np.sort(self.owner[idx], axis=1)
            first = np.ones(cand.shape, dtype=bool)
            first[:, 1:] = cand[:, 1:] != cand[:, :-1]
            r, c = np.nonzero(first)
            rows = pending[r]
            flat = cand[r, c]
            q = _closest_on_tri(P[rows], self.src[flat, 0], self.src[flat, 1], self.src[flat, 2])
            dist = ((P[rows] - q) ** 2).sum(axis=1)
            if not (use >= len(self.pts) or k >= _PROJECT_KNN_CAP):
                dist = np.where(accept(rows, flat), dist, np.inf)
            d2 = np.full(cand.shape, np.inf)
            d2[r, c] = dist
            slot = np.zeros(cand.shape, dtype=np.int64)
            slot[r, c] = np.arange(len(r))
            floor = d2.min(axis=1, keepdims=True)
            tied = d2 <= floor * (1.0 + 1e-9) + self.tie2
            pick = np.where(tied, cand, len(self.src)).argmin(axis=1)
            here = np.arange(len(pending))
            found = np.isfinite(d2[here, pick])
            out[pending[found]] = q[slot[here, pick][found]]
            tri[pending[found]] = cand[here, pick][found]
            pending = pending[~found]
            k *= 8
        return out, tri

    def __call__(
        self, P: np.ndarray, verts: np.ndarray, keys: np.ndarray, require_all: bool
    ) -> np.ndarray:
        verts = np.asarray(verts, dtype=np.int64).reshape(len(P), -1)

        def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
            lab = self.labels[flat]
            hits = [
                _has_key(keys, verts[rows, j] * self.n_labels + lab) for j in range(verts.shape[1])
            ]
            return np.logical_and.reduce(hits) if require_all else np.logical_or.reduce(hits)

        return self.search(P, accept)[0]

    def rebase(self, V: np.ndarray, F: np.ndarray, S: np.ndarray, settled: np.ndarray):
        todo = np.nonzero(~settled)[0]
        want = self.labels[S[todo]]

        def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
            return self.labels[flat] == want[rows]

        tri = self.search(V[F[todo]].mean(axis=1), accept)[1]
        S = S.copy()
        S[todo] = np.where(self.labels[tri] == want, tri, S[todo])
        return S, np.ones(len(F), dtype=bool)


_PROJECT_KNN = 16
_PROJECT_KNN_CAP = 1024


def _has_key(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.clip(np.searchsorted(keys, query), 0, len(keys) - 1)
    return keys[pos] == query


REMESH_FEATURE_DEG = 30.0


def _features(
    F: np.ndarray, L: np.ndarray, proj: _Projector, n_verts: int, edges=None
) -> tuple[np.ndarray, np.ndarray]:
    uniq, inverse, counts = edges if edges is not None else _unique_edges(F)
    t0, t1 = _manifold_owners(len(F), counts, inverse)
    flag = counts != 2
    pair = np.nonzero(counts == 2)[0]
    s0, s1 = L[t0[pair]], L[t1[pair]]
    bend = (proj.normals[s0] * proj.normals[s1]).sum(axis=1)
    flag[pair] = (proj.labels[s0] != proj.labels[s1]) & (
        bend < np.cos(np.radians(REMESH_FEATURE_DEG))
    )
    return flag, np.bincount(uniq[flag].ravel(), minlength=n_verts)


def _split_pass(
    V: np.ndarray, F: np.ndarray, L: np.ndarray, limit: float, proj: _Projector, settled
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lengths = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    long_mask = lengths > limit
    if not long_mask.any():
        return V, F, L, settled, 0
    edges = np.unique(np.sort(E[long_mask], axis=1), axis=0)
    raw = (V[edges[:, 0]] + V[edges[:, 1]]) / 2.0
    mids = proj(raw, edges, proj.incidence(F, L), require_all=True)
    base = len(V)
    V = np.concatenate([V, mids])
    n_all = base + len(edges)
    ekey = edges[:, 0] * n_all + edges[:, 1]
    qkey = np.stack(
        [
            np.minimum(F[:, 0], F[:, 1]) * n_all + np.maximum(F[:, 0], F[:, 1]),
            np.minimum(F[:, 1], F[:, 2]) * n_all + np.maximum(F[:, 1], F[:, 2]),
            np.minimum(F[:, 2], F[:, 0]) * n_all + np.maximum(F[:, 2], F[:, 0]),
        ],
        axis=1,
    )
    pos = np.clip(np.searchsorted(ekey, qkey), 0, len(ekey) - 1)
    mid_ids = np.where(ekey[pos] == qkey, base + pos, -1)
    ab, bc, ca = mid_ids[:, 0], mid_ids[:, 1], mid_ids[:, 2]
    has = (ab >= 0).astype(np.int64) + (bc >= 0).astype(np.int64) + (ca >= 0).astype(np.int64)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    plain = np.nonzero(has == 0)[0]
    if len(plain):
        blocks.append(F[plain])
        labels.append(L[plain])
        parents.append(plain)
    full = np.nonzero(has == 3)[0]
    if len(full):
        a, b, c = F[full, 0], F[full, 1], F[full, 2]
        blocks.append(np.stack([a, ab[full], ca[full]], axis=1))
        blocks.append(np.stack([b, bc[full], ab[full]], axis=1))
        blocks.append(np.stack([c, ca[full], bc[full]], axis=1))
        blocks.append(np.stack([ab[full], bc[full], ca[full]], axis=1))
        labels += [L[full]] * 4
        parents += [full] * 4
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
            labels += [L[g]] * 3
            parents += [g] * 3
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
            labels += [L[g]] * 2
            parents += [g] * 2
    F2 = np.concatenate(blocks)
    parent = np.concatenate(parents)
    n_parent = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[parent]
    n_child = np.cross(V[F2[:, 1]] - V[F2[:, 0]], V[F2[:, 2]] - V[F2[:, 0]])
    bad = (n_child * n_parent).sum(axis=1) <= 0.0
    if bad.any():
        undo = np.unique(F2[bad])
        undo = undo[undo >= base]
        V[undo] = raw[undo - base]
    settled = np.concatenate([settled[plain], np.zeros(len(F2) - len(plain), dtype=bool)])
    return V, F2, np.concatenate(labels), settled, len(edges)


def _stars(
    verts: np.ndarray, starts: np.ndarray, vcounts: np.ndarray, tri_of: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cnt = vcounts[verts]
    row = np.repeat(np.arange(len(verts)), cnt)
    offset = np.arange(len(row)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return row, tri_of[np.repeat(starts[verts], cnt) + offset]


def _first_independent(n: int, row: np.ndarray, res: np.ndarray, n_res: int, valid) -> np.ndarray:
    chosen = np.zeros(n, dtype=bool)
    taken = np.zeros(n_res, dtype=bool)
    while len(row):
        low = np.full(n_res, n, dtype=np.int64)
        np.minimum.at(low, res, row)
        head = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        win = row[head[np.logical_and.reduceat(low[res] == row, head)]]
        good = win[valid(win)]
        chosen[good] = True
        taken[res[np.isin(row, good)]] = True
        done = np.zeros(n, dtype=bool)
        done[win] = True
        done[row[head[np.logical_or.reduceat(taken[res], head)]]] = True
        keep = ~done[row]
        row, res = row[keep], res[keep]
    return chosen


class _CollapseCheck:
    def __init__(self, V, F, L, proj, ab, r0, r1, targets, free, keys):
        self.V, self.F, self.L, self.proj = V, F, L, proj
        self.ab, self.r0, self.r1 = ab, r0, r1
        self.targets, self.free, self.keys = targets, free, keys
        self.starts, self.vcounts, self.tri_of = _vertex_tri_map(F, len(V))

    def __call__(self, idx: np.ndarray) -> np.ndarray:
        V, F, L, proj = self.V, self.F, self.L, self.proj
        n = len(idx)
        a, b = self.ab[idx, 0], self.ab[idx, 1]
        r0, r1 = self.r0[idx], self.r1[idx]
        free = np.nonzero(self.free[idx])[0]
        if len(free):
            self.targets[idx[free]] = proj(
                self.targets[idx[free]], self.ab[idx[free]], self.keys, require_all=False
            )
        targets = self.targets[idx]
        both = np.concatenate([F[r0], F[r1]], axis=1)
        other = (both != a[:, None]) & (both != b[:, None])
        ok = (r0 >= 0) & (other.sum(axis=1) == 2)
        first = np.argsort(~other, axis=1, kind="stable")[:, :2]
        cd = np.take_along_axis(both, first, axis=1)
        a_row, a_tri = _stars(a, self.starts, self.vcounts, self.tri_of)
        b_row, b_tri = _stars(b, self.starts, self.vcounts, self.tri_of)
        has_b = (F[a_tri] == b[a_row, None]).any(axis=1)
        shared = np.unique(a_row[has_b] * len(F) + a_tri[has_b]) // len(F)
        ok &= np.bincount(shared, minlength=n) == 1 + (r0 != r1)
        ring_a = np.unique(np.repeat(a_row, 3) * len(V) + F[a_tri].ravel())
        ring_a = ring_a[ring_a % len(V) != a[ring_a // len(V)]]
        ring_b = np.unique(np.repeat(b_row, 3) * len(V) + F[b_tri].ravel())
        ring_b = ring_b[ring_b % len(V) != b[ring_b // len(V)]]
        common = np.intersect1d(ring_a, ring_b, assume_unique=True) // len(V)
        ok &= np.bincount(common, minlength=n) == 1 + (cd[:, 0] != cd[:, 1])
        key = np.unique(np.concatenate([a_row, b_row]) * len(F) + np.concatenate([a_tri, b_tri]))
        row, tri = key // len(F), key % len(F)
        old_rows = F[tri]
        hit_a = old_rows == a[row, None]
        hit_b = old_rows == b[row, None]
        use = ~(hit_a.any(axis=1) & hit_b.any(axis=1))
        row, tri, old_rows = row[use], tri[use], old_rows[use]
        old = V[old_rows]
        moved = (hit_a | hit_b)[use]
        new = np.where(moved[:, :, None], targets[row][:, None, :], old)
        n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
        n_new = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
        S = L[tri]
        bad = (np.linalg.norm(n_new, axis=1) <= 1e-14) | ((n_old * n_new).sum(axis=1) <= 0.0)
        bad |= (n_new * proj.normals[S]).sum(axis=1) <= 0.0
        ok &= np.bincount(row, weights=bad, minlength=n) == 0
        curved = np.bincount(row, weights=~proj.planar[proj.labels[S]], minlength=n) > 0
        probe = np.nonzero((ok & curved)[row])[0]
        if len(probe):
            want = proj.labels[S[probe]]

            def accept(rows: np.ndarray, flat: np.ndarray) -> np.ndarray:
                return proj.labels[flat] == want[rows]

            local = proj.search(new[probe].mean(axis=1), accept)[1]
            away = (n_new[probe] * proj.normals[local]).sum(axis=1) <= 0.0
            ok &= np.bincount(row[probe], weights=away, minlength=n) == 0
        return ok


def _collapse_pass(
    V: np.ndarray, F: np.ndarray, L: np.ndarray, limit: float, proj: _Projector, settled
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
        flag, fcount = _features(F, L, proj, len(V), (uniq, inverse, counts))
        fa = fcount[uniq[cand, 0]]
        fb = fcount[uniq[cand, 1]]
        both = (fa > 0) & (fb > 0)
        allowed = ~both | (flag[cand] & ((fa == 2) | (fb == 2)))
        cand, fa, fb = cand[allowed], fa[allowed], fb[allowed]
        ab = uniq[cand]
        targets = (V[ab[:, 0]] + V[ab[:, 1]]) / 2.0
        free = (fa == 0) & (fb == 0)
        keep_a = ~free & ((fb == 0) | ((fa > 0) & (fa != 2)) | ((fa == 2) & (fb == 2)))
        keep_b = ~free & ~keep_a
        targets[keep_a] = V[ab[keep_a, 0]]
        targets[keep_b] = V[ab[keep_b, 1]]
        t0, t1 = _manifold_owners(len(F), counts, inverse)
        check = _CollapseCheck(
            V, F, L, proj, ab, t0[cand], t1[cand], targets, free, proj.incidence(F, L)
        )
        row, res = _stars(ab.ravel(), check.starts, check.vcounts, check.tri_of)
        chosen = _first_independent(len(cand), row // 2, res, len(F), check)
        if not chosen.any():
            break
        win = np.nonzero(chosen)[0]
        remap = np.arange(len(V), dtype=np.int64)
        remap[ab[win, 0]] = len(V) + np.arange(len(win))
        remap[ab[win, 1]] = len(V) + np.arange(len(win))
        V = np.concatenate([V, check.targets[win]])
        settled = settled & (remap[F] == F).all(axis=1)
        F = remap[F]
        alive = ~((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0]))
        F = F[alive]
        L = L[alive]
        settled = settled[alive]
        collapses += len(win)
    return V, F, L, settled, collapses


def _flip_pass(
    V: np.ndarray, F: np.ndarray, L: np.ndarray, proj: _Projector, settled
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        a, b = uniq[cand, 0], uniq[cand, 1]
        r0, r1 = t0[cand], t1[cand]
        both = np.concatenate([F[r0], F[r1]], axis=1)
        other = (both != a[:, None]) & (both != b[:, None])
        cd = np.take_along_axis(both, np.argsort(~other, axis=1, kind="stable")[:, :2], axis=1)
        c, d = cd[:, 0], cd[:, 1]
        ok = (r0 >= 0) & (proj.labels[L[r0]] == proj.labels[L[r1]]) & (other.sum(axis=1) == 2)
        ekey = uniq[:, 0] * len(V) + uniq[:, 1]
        ok &= (c != d) & ~_has_key(ekey, np.minimum(c, d) * len(V) + np.maximum(c, d))
        row0 = np.where(F[r0] == b[:, None], d[:, None], F[r0])
        row1 = np.where(F[r1] == a[:, None], c[:, None], F[r1])

        def normal(rows: np.ndarray) -> np.ndarray:
            return np.cross(V[rows[:, 1]] - V[rows[:, 0]], V[rows[:, 2]] - V[rows[:, 0]])

        n_old0, n_old1 = normal(F[r0]), normal(F[r1])
        n_new0, n_new1 = normal(row0), normal(row1)
        ok &= np.minimum(np.linalg.norm(n_new0, axis=1), np.linalg.norm(n_new1, axis=1)) > 1e-14
        ok &= ((n_new0 * n_old0).sum(axis=1) > 0.0) & ((n_new1 * n_old1).sum(axis=1) > 0.0)
        ok &= (n_new0 * proj.normals[L[r0]]).sum(axis=1) > 0.0
        ok &= (n_new1 * proj.normals[L[r1]]).sum(axis=1) > 0.0
        live = np.nonzero(ok)[0]
        val = np.bincount(F.ravel(), minlength=len(V)).tolist()
        touch = bytearray(len(F))
        made: set[tuple[int, int]] = set()
        swap: list[int] = []
        for i, va, vb, vc, vd, s0, s1 in zip(
            live.tolist(),
            a[live].tolist(),
            b[live].tolist(),
            c[live].tolist(),
            d[live].tolist(),
            r0[live].tolist(),
            r1[live].tolist(),
            strict=True,
        ):
            if touch[s0] or touch[s1]:
                continue
            pair = (min(vc, vd), max(vc, vd))
            if pair in made:
                continue
            before = abs(val[va] - 6) + abs(val[vb] - 6) + abs(val[vc] - 6) + abs(val[vd] - 6)
            after = abs(val[va] - 7) + abs(val[vb] - 7) + abs(val[vc] - 5) + abs(val[vd] - 5)
            if after >= before:
                continue
            swap.append(i)
            val[va] -= 1
            val[vb] -= 1
            val[vc] += 1
            val[vd] += 1
            touch[s0] = 1
            touch[s1] = 1
            made.add(pair)
        if not swap:
            break
        flips += len(swap)
        F = F.copy()
        F[r0[swap]] = row0[swap]
        F[r1[swap]] = row1[swap]
        settled = settled.copy()
        settled[r0[swap]] = False
        settled[r1[swap]] = False
    return V, F, settled, flips


def _relax_pass(V: np.ndarray, F: np.ndarray, L: np.ndarray, proj: _Projector) -> np.ndarray:
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
    moved = proj(
        V + 0.5 * tangential, np.arange(len(V))[:, None], proj.incidence(F, L), require_all=True
    )
    pinned = _features(F, L, proj, len(V))[1] > 0
    moved[pinned] = V[pinned]
    n_old = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    floor = 1e-12 * np.linalg.norm(n_old, axis=1)
    for _ in range(len(V)):
        n_new = np.cross(moved[F[:, 1]] - moved[F[:, 0]], moved[F[:, 2]] - moved[F[:, 0]])
        bad = ((n_old * n_new).sum(axis=1) <= 0.0) | (np.linalg.norm(n_new, axis=1) <= floor)
        bad |= ~proj.facing(moved[F], L)
        if not bad.any():
            break
        undo = np.unique(F[bad])
        undo = undo[(moved[undo] != V[undo]).any(axis=1)]
        if len(undo) == 0:
            break
        moved[undo] = V[undo]
    return moved


@register(
    "isotropic_remesh",
    "processing",
    "identity",
    "uniform target edge diagonal * 0.02 * 2**severity (diagonal/50 at severity 0+, "
    "/25 at severity 1); split edges longer than 4/3 of target, collapse shorter "
    "than 4/5, flip toward valence 6 (never across a face boundary), tangential "
    "relax; every new or moved vertex is projected to the closest point on the "
    "source triangles of the faces around it; edges between faces meeting at more "
    "than 30 degrees are features: feature vertices never relax, collapse only "
    "along a feature into a feature vertex, and corners never move, so sharp "
    "edges survive and narrow faces become sliver strips",
    preserves_watertight=True,
)
def isotropic_remesh(mesh, severity, rng):
    src_tris = np.asarray(mesh.tris, dtype=np.float64).copy()
    src_ids = mesh.face_id.copy()
    before = len(src_tris)
    diagonal = bbox_diagonal(src_tris)
    h = target_edge_mm(severity, diagonal)
    proj = _Projector(src_tris, np.unique(src_ids, return_inverse=True)[1])
    L = np.arange(len(src_tris), dtype=np.int64)
    V, F = _weld(src_tris)
    passes = remesh_passes(severity)
    splits = collapses = flips = 0
    settled = np.zeros(len(F), dtype=bool)
    for _ in range(passes):
        V, F, L, settled, n = _split_pass(V, F, L, 4.0 / 3.0 * h, proj, settled)
        splits += n
        V, F = _compact(V, F)
        L, settled = proj.rebase(V, F, L, settled)
        V, F, L, settled, n = _collapse_pass(V, F, L, 4.0 / 5.0 * h, proj, settled)
        collapses += n
        V, F = _compact(V, F)
        L, settled = proj.rebase(V, F, L, settled)
        V, F, settled, n = _flip_pass(V, F, L, proj, settled)
        flips += n
        L, settled = proj.rebase(V, F, L, settled)
        moved = _relax_pass(V, F, L, proj)
        settled &= ~(moved != V).any(axis=1)[F].any(axis=1)
        V = moved
        L, settled = proj.rebase(V, F, L, settled)
    V, F, dropped = _drop_degenerate(V, F)
    mesh.tris = V[F].reshape(-1, 3, 3) + 0.0
    label = transfer_labels(
        mesh,
        src_tris,
        src_ids,
        mesh.metadata.get(CONFIDENCE_KEY),
        (proj.pts, proj.owner, proj.tree),
    )
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
    if CONFIDENCE_KEY not in mesh.metadata:
        flat = welds_away(mesh.tris, diagonal)
        mesh.metadata[CONFIDENCE_KEY] = [0.0 if f else 1.0 for f in flat.tolist()]
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
        **confidence_summary(mesh.metadata[CONFIDENCE_KEY]),
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
        **confidence_summary(mesh.metadata[CONFIDENCE_KEY]),
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
    mesh.tris = np.array(rows, dtype=np.float64).reshape(-1, 3, 3) + 0.0
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
