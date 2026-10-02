from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import unmesh


def tessellate(shape, linear=0.05, angular=0.3):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_Orientation
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(shape.wrapped, linear, False, angular, True)
    tris, labels = [], []
    for fi, face in enumerate(shape.faces()):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
        assert tri is not None
        trsf = loc.Transformation()
        pts = []
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            pts.append((p.X(), p.Y(), p.Z()))
        reverse = face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            if reverse:
                b, c = c, b
            tris.append([pts[a - 1], pts[b - 1], pts[c - 1]])
            labels.append(fi)
    return np.array(tris, dtype=np.float64), np.array(labels)


def truth_planes(tris, labels):
    planes = []
    for fi in range(int(labels.max()) + 1):
        t = tris[labels == fi]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = np.linalg.norm(n, axis=1)
        k = int(np.argmax(area))
        nk = n[k] / area[k]
        d = float(np.mean(np.einsum("j,tij->ti", nk, t)))
        planes.append((nk, d, float(np.max(np.abs(np.einsum("j,tij->ti", nk, t) - d)))))
    return planes


def weld_indexed(tris):
    verts, faces, source, _ = unmesh.weld(tris, 1e-6)
    return verts, faces, source


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def noise_vectors(rng, n, amplitude):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0, amplitude, size=(n, 1))


@dataclass
class Case:
    name: str
    float32: bool = False
    rotate: bool = False
    noise: float = 0.0


CASES = [
    Case("clean"),
    Case("float32", float32=True),
    Case("rotated", rotate=True, float32=True),
    Case("noise 1um", noise=0.001),
    Case("noise 5um", noise=0.005),
    Case("noise 10um", noise=0.010, float32=True),
    Case("rotated+noise 10um", rotate=True, noise=0.010, float32=True),
]


@dataclass
class Part:
    name: str
    tris: np.ndarray
    labels: np.ndarray
    truth: list


@dataclass
class Degraded:
    soup: np.ndarray
    truth_planes: list
    truth_vertices: np.ndarray
    rotation: np.ndarray


def degrade(part: Part, case: Case, seed: int) -> Degraded:
    rng = np.random.default_rng(seed)
    clean_v, faces, _ = weld_indexed(part.tris)
    assert len(faces) == len(part.tris)
    v = clean_v.copy()
    if case.noise:
        v = v + noise_vectors(rng, len(v), case.noise)
    rot = np.eye(3)
    if case.rotate:
        rot = random_rotation(rng)
        v = v @ rot.T
    truth_v = clean_v @ rot.T
    if case.float32:
        v = v.astype(np.float32).astype(np.float64)
    planes = [(rot @ n, d, flat) for n, d, flat in part.truth]
    return Degraded(v[faces], planes, truth_v, rot)


@dataclass
class Metrics:
    faces: int
    regions: int
    matched: int
    f1: float
    dev_input: float
    dev_truth: float
    reported: float
    seconds: float
    fallback: bool = False
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def plane_of(region):
    s = region.surface
    n = np.array(s.normal)
    return n, float(n @ np.array(s.origin))


def evaluate(part: Part, deg: Degraded) -> Metrics:
    t0 = time.perf_counter()
    ir, report = unmesh.convert(deg.soup)
    seconds = time.perf_counter() - t0
    regions = ir.regions
    planes = {r.id: plane_of(r) for r in regions if r.surface.type == "plane"}
    tri_region = np.full(len(part.labels), -1)
    for r in regions:
        tri_region[np.array(r.triangles, dtype=int)] = r.id
    notes: list[str] = []
    matched = 0
    matched_regions = set()
    dev_truth = 0.0
    for fi, (tn, _, _) in enumerate(deg.truth_planes):
        mine = np.nonzero(part.labels == fi)[0]
        fverts = deg.truth_vertices[np.unique(_face_vertex_ids(part, fi))]
        centroid = fverts.mean(axis=0)
        cnt = np.bincount(tri_region[mine] + 1)
        cands = []
        for rid in np.nonzero(cnt)[0] - 1:
            if rid < 0 or rid not in planes or cnt[rid + 1] < 0.5 * len(mine):
                continue
            rn, rd = planes[rid]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(rn @ tn)))))
            off = abs(float(centroid @ rn) - rd)
            if ang <= 0.2 and off <= 0.01:
                cands.append(rid)
            else:
                notes.append(f"face {fi}: region {rid} angle {ang:.3f} offset {off * 1000:.1f}um")
        if not cands and not notes:
            notes.append(f"face {fi}: split over {int((cnt > 0).sum())} regions")
        if len(cands) == 1:
            matched += 1
            matched_regions.add(cands[0])
            rn, rd = planes[cands[0]]
            dev_truth = max(dev_truth, float(np.max(np.abs(fverts @ rn - rd))))
    n_faces, n_regions = len(deg.truth_planes), len(regions)
    recall = matched / n_faces
    precision = len(matched_regions) / n_regions if n_regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dev_input = 0.0
    for r in regions:
        if r.id in planes:
            n, d = planes[r.id]
            t = deg.soup[np.array(r.triangles, dtype=int)].reshape(-1, 3)
            dev_input = max(dev_input, float(np.max(np.abs(t @ n - d))))
    for v in ir.vertices:
        p = np.array(v.position)
        dev_input = max(
            dev_input, min(float(np.linalg.norm(p - np.array(s))) for s in v.source_positions)
        )
        dev_truth = max(dev_truth, float(np.min(np.linalg.norm(deg.truth_vertices - p, axis=1))))
    return Metrics(
        n_faces,
        n_regions,
        matched,
        f1,
        dev_input,
        dev_truth,
        report.max_deviation,
        seconds,
        any(w.code == "fallback_facets" for w in report.warnings),
        [w.code for w in report.warnings],
        notes,
    )


def _face_vertex_ids(part, fi):
    cache = part.__dict__.setdefault("_vids", {})
    if "faces" not in cache:
        clean_v, faces, _ = weld_indexed(part.tris)
        cache["faces"] = faces
    return cache["faces"][part.labels == fi].ravel()


def smoke_parts():
    from unmesh_harness.corpus import load_manifest, select
    from unmesh_harness.groundtruth import generate

    parts = []
    for entry in select(load_manifest(), "smoke"):
        if entry["tier"] == "generated":
            gt = generate(entry["family"], entry["seed"])
            tris, labels = tessellate(gt.solid)
            parts.append(Part(entry["id"], tris, labels, truth_planes(tris, labels)))
    return parts


def micro_f1(ms):
    tp = sum(m.matched for m in ms)
    faces = sum(m.faces for m in ms)
    regions = sum(m.regions for m in ms)
    precision, recall = tp / regions, tp / faces
    return 2 * precision * recall / (precision + recall)


def run_case(parts, case, seed_base=1000):
    return [evaluate(p, degrade(p, case, seed_base + i)) for i, p in enumerate(parts)]


def row(case, ms):
    return (
        f"| {case.name} | {len(ms)} | {micro_f1(ms):.4f} | {min(m.f1 for m in ms):.3f} "
        f"| {max(m.dev_input for m in ms) * 1000:.3f} | {max(m.dev_truth for m in ms) * 1000:.3f} "
        f"| {max(m.reported for m in ms) * 1000:.3f} "
        f"| {'never' if all(m.reported >= m.dev_input for m in ms) else 'UNDER'} |"
    )


TABLE_HEADER = (
    "| degradation | parts | F1 (micro) | F1 (worst part) | dev to input (um) "
    "| dev to truth (um) | reported dev (um) | under-reports |\n"
    "|---|---|---|---|---|---|---|---|"
)


def main():
    parts = smoke_parts()
    print(TABLE_HEADER)
    for case in CASES:
        print(row(case, run_case(parts, case)))


if __name__ == "__main__":
    main()
