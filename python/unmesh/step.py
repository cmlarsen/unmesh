from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from unmesh.ir import Ir

READBACK_RELATIVE = 1e-9
INPUT_VOLUME_RELATIVE = 1e-6
FACETED_SHARE = 0.5
FLUX_DEFLECTION = 1e-4
FLUX_RELATIVE = 1e-9
INPROCESS_READBACK_BYTES = 4 << 20


INSTALL_HINT = "unmesh.step needs OCP; install it with `pip install unmesh[step]`"


def require_ocp() -> None:
    try:
        from unmesh._writer import faceted, occ, topology  # noqa: F401
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e


@dataclass(frozen=True)
class WriteOptions:
    max_shape_tolerance: float = 1e-3
    max_deviation: float = 5e-3
    max_seam_gap: float = 1e-4
    readback_memory_mb: float = 2048.0
    readback_timeout_s: float = 120.0


@dataclass
class FaceReport:
    region: int
    surface_type: str
    max_shape_tolerance: float
    max_vertex_displacement: float = 0.0
    max_boundary_deviation: float = 0.0


@dataclass
class SeamReport:
    regions: tuple[int, int]
    surface_type: str
    points: int
    max_gap: float
    chord_gap: float = 0.0
    inserted: int = 0
    tolerance: float = 0.0


@dataclass
class EdgeFallback:
    regions: tuple[int, int]
    reason: str
    max_deviation: float
    kind: Literal["projected", "interpolated"] = "projected"
    intersection_distance: float | None = None


@dataclass
class TangentEdge:
    regions: tuple[int, int]
    curve: Literal["line", "circle", "bspline"]
    max_deviation: float
    boundary_distance: float


@dataclass
class ShellReport:
    shell: int
    kind: Literal["solid", "shell"]
    valid: bool
    volume: float | None
    max_shape_tolerance: float
    issues: list[str] = field(default_factory=list)
    max_vertex_displacement: float = 0.0
    max_boundary_deviation: float = 0.0


@dataclass
class ReadBack:
    ok: bool
    solids: int
    shells: int
    volume: float
    expected_solids: int
    expected_shells: int
    expected_volume: float
    volume_tolerance: float
    issues: list[str] = field(default_factory=list)
    reader: Literal["occt", "occt_no_face_orientation", "text"] = "occt"
    peak_rss_mb: float | None = None


@dataclass
class WriteReport:
    valid: bool
    solids: int
    max_shape_tolerance: float
    faces: list[FaceReport] = field(default_factory=list)
    fallback: Literal["faceted"] | None = None
    fallback_reason: str | None = None
    shells: list[ShellReport] = field(default_factory=list)
    seams: list[SeamReport] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    faceted_regions: int = 0
    faceted_faces: int = 0
    max_vertex_displacement: float = 0.0
    max_boundary_deviation: float = 0.0
    edge_fallbacks: list[EdgeFallback] = field(default_factory=list)
    tangent_edges: list[TangentEdge] = field(default_factory=list)
    open_shells: list[int] = field(default_factory=list)
    readback: ReadBack | None = None
    verified: bool = False
    verified_by: Literal["occt", "text", "occt+text"] | None = None
    text_check: ReadBack | None = None
    readback_skipped: str | None = None
    volume_checked_against_input: bool = False
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _Group:
    outer: int
    kind: str
    shape: Any = None
    faces: list[FaceReport] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    valid: bool = False
    volume: float | None = None
    tolerance: float = 0.0
    context: Any = None
    mapping: list = field(default_factory=list)
    moved: float = 0.0
    deviation: float = 0.0
    expected: float = 0.0
    area: float = 0.0
    shells: int = 1
    curved: bool = False
    edge_fallbacks: list[EdgeFallback] = field(default_factory=list)
    tangent_edges: list[TangentEdge] = field(default_factory=list)
    patches: dict = field(default_factory=dict)


def _triangles(mesh) -> np.ndarray:
    if isinstance(mesh, tuple):
        v = np.asarray(mesh[0], dtype=float)
        f = np.asarray(mesh[1], dtype=np.int64)
        tris = v[f]
    elif isinstance(mesh, (str, os.PathLike)):
        from unmesh.api import read_mesh

        tris = read_mesh(mesh)
    else:
        tris = np.asarray(mesh, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("mesh must be a path, an (n, 3, 3) array or a (vertices, faces) tuple")
    return tris


def _members(ir: Ir, outer: int) -> list[int]:
    return [outer] + [
        i for i, s in enumerate(ir.shells) if s.role == "cavity" and s.parent == outer
    ]


def _check(g: _Group, occ, max_tol: float) -> None:
    g.tolerance = occ.tolerance_of(g.shape)
    g.shells = occ.shell_count(g.shape)
    ok = occ.is_valid(g.shape)
    if not ok:
        g.issues.append("BRepCheck_Analyzer reports the shape invalid")
    if g.kind == "solid":
        g.volume = g.expected = occ.volume_of(g.shape, g.curved)
        g.area = occ.area_of(g.shape)
        if not g.volume > 0:
            ok = False
            g.issues.append(f"non-positive volume {g.volume:.6g}")
    if g.tolerance > max_tol:
        ok = False
        g.issues.append(f"shape tolerance {g.tolerance:.3g} exceeds {max_tol:.3g}")
    g.valid = ok


def _finish_group(g: _Group, occ, outer_shell, cavities, options) -> None:
    if g.kind == "solid" and g.curved:
        g.shape = occ.make_oriented_solid(outer_shell, cavities)
        if not occ.is_valid(g.shape):
            g.shape, g.context = occ.fix_shape(g.shape)
            occ.check_orientation(g.shape)
    elif g.kind == "solid":
        solid = occ.make_solid(outer_shell, cavities)
        g.shape, g.context = occ.fix_shape(solid)
    else:
        g.shape = outer_shell
    _check(g, occ, options.max_shape_tolerance)


def _analytic(ir: Ir, options: WriteOptions, occ, topology):
    groups: list[_Group] = []
    seams: list[SeamReport] = []
    limit = topology.deviation_limit(ir, options.max_deviation)
    vpos, vmoved, bad = topology.vertex_positions(ir, limit)
    refined = None
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(outer, "solid" if shell.closed else "shell")
        groups.append(g)
        try:
            built = []
            moved: dict[int, float] = {}
            dev: dict[int, float] = {}
            members = _members(ir, outer)
            if any(_is_curved(ir, idx) for idx in members):
                from unmesh._writer import curved

                g.curved = True
                if refined is None:
                    refined = curved.refine_vertices(ir, vpos, vmoved, bad, limit)
                for idx in members:
                    cs = curved.build_shell(
                        ir, idx, *refined, limit, occ.new_pool(), options.max_seam_gap
                    )
                    seams.extend(SeamReport(**vars(s)) for s in cs.seams)
                    g.patches.update(cs.patches)
                    moved.update(cs.vertex_displacement)
                    dev.update(cs.boundary_deviation)
                    g.edge_fallbacks.extend(
                        EdgeFallback(
                            p.regions, p.reason, p.max_deviation, p.kind, p.intersection_distance
                        )
                        for p in cs.projected
                    )
                    g.tangent_edges.extend(
                        TangentEdge(t.regions, t.curve, t.max_deviation, t.boundary_distance)
                        for t in cs.tangent
                    )
                    built.append((idx, cs.shell, cs.mapping))
            for idx in members if not g.curved else ():
                plan = topology.plan_shell(ir, idx, vpos, vmoved, bad, limit)
                seams.extend(SeamReport(**vars(s)) for s in plan.seams)
                g.patches.update(plan.patches)
                moved.update(plan.vertex_displacement)
                dev.update(plan.boundary_deviation)
                built.append((idx, *occ.build_plan_shell(plan)))
            g.mapping = [m for _, _, mapping in built for m in mapping]
            cavities = [sh for idx, sh, _ in built if idx != outer]
            _finish_group(g, occ, built[0][1], cavities, options)
            tols = occ.face_tolerances(g.shape, g.mapping, g.context)
            g.faces = [
                FaceReport(r, ir.regions[r].surface.type, t, moved[r], dev[r])
                for r, t in sorted(tols.items())
            ]
            g.moved = max(moved.values(), default=0.0)
            g.deviation = max(dev.values(), default=0.0)
        except topology.BuildError as e:
            g.issues.append(str(e))
        except Exception as e:
            g.issues.append(f"{type(e).__name__}: {e}")
    return groups, seams


def _check_patches(ir: Ir, g: _Group, tris: np.ndarray, seam_gap: float, limit: float) -> None:
    from unmesh._writer import geometry as geo
    from unmesh._writer import topology

    measured = {}
    for r in (r for idx in _members(ir, g.outer) for r in ir.shells[idx].regions):
        if not _is_facets(ir, r):
            continue
        patch = g.patches.get(r)
        if patch is None:
            s = ir.regions[r].surface
            v = np.asarray(s.vertices, dtype=float)
            patch = topology.PatchWrite(v[np.asarray(s.faces, dtype=np.int64).reshape(-1, 3)])
        ids = np.asarray(ir.regions[r].triangles, dtype=np.int64)
        if len(ids) != len(patch.corners) or (len(ids) and ids.max() >= len(tris)):
            g.valid = False
            g.issues.append(f"region {r}: the facets patch does not match the input mesh")
            continue
        mesh = tris[ids]
        surf = ir.regions[r].surface
        claimed = _recorded_moves(
            ir, r, np.asarray(surf.vertices, dtype=float)[np.asarray(surf.faces, dtype=np.int64)]
        )
        gaps = np.stack(
            [np.linalg.norm(patch.corners - np.roll(mesh, -k, axis=1), axis=2) for k in range(3)]
        )
        off = gaps.max(axis=2)
        turn = off.argmin(axis=0)
        best = gaps[turn, np.arange(len(ids))]
        worst = float(off.min(axis=0).max(initial=0.0))
        excess = float((best - claimed).max(initial=0.0))
        split = 0.0
        for fi, side, x, seam in patch.splits:
            corners = np.roll(mesh[fi], -turn[fi], axis=0)
            a, b = corners[side], corners[(side + 1) % 3]
            residual = ir.regions[seam].residual
            room = limit + (residual.max if residual is not None else 0.0)
            on_mesh = _segment_distance(x, a, b)
            on_surface = geo.distance(ir.regions[seam].surface, x)
            split = max(split, on_mesh)
            if on_mesh > room or on_surface > seam_gap:
                g.valid = False
                kind = ir.regions[seam].surface.type
                g.issues.append(
                    f"region {r}: a seam split point is {on_mesh:.3g} from the input mesh"
                    f" (allowed {room:.3g}) and {on_surface:.3g} from the {kind}"
                    f" (allowed {seam_gap:.3g})"
                )
                break
        measured[r] = max(worst, split)
        if excess > limit:
            g.valid = False
            g.issues.append(
                f"region {r}: a facets patch vertex is {excess:.3g} from the input mesh beyond"
                f" the IR's recorded move, over the {limit:.3g} limit"
            )
    for f in g.faces:
        if f.region in measured:
            f.max_vertex_displacement = measured[f.region]
    g.moved = max([g.moved, *measured.values()])


def _recorded_moves(ir: Ir, region: int, corners: np.ndarray) -> np.ndarray:
    from unmesh._writer import geometry as geo

    moved = {}
    for v in ir.vertices:
        p = np.asarray(v.position, dtype=float)
        moved[tuple(p)] = min(float(np.linalg.norm(p - np.asarray(q))) for q in v.source_positions)
    near = []
    for adj in ir.adjacencies:
        if region in adj.regions:
            other = ir.regions[adj.regions[0] + adj.regions[1] - region]
            if other.residual is not None:
                near.append((other.surface, other.residual.max))
    snap = ir.tolerances.vertex_merge
    out = []
    for c in corners.reshape(-1, 3):
        on = [m for s, m in near if abs(geo.distance(s, c)) <= snap]
        out.append(max([moved.get(tuple(c), 0.0), *on]))
    return np.array(out).reshape(corners.shape[:2])


def _segment_distance(x, a, b) -> float:
    d = b - a
    t = float(np.clip((x - a) @ d / max(float(d @ d), 1e-300), 0.0, 1.0))
    return float(np.linalg.norm(x - (a + t * d)))


def _is_curved(ir: Ir, idx: int) -> bool:
    return any(
        ir.regions[r].surface.type in ("cylinder", "cone", "sphere", "torus")
        for r in ir.shells[idx].regions
    )


def _check_input_volume(ir: Ir, g: _Group, tris: np.ndarray, occ) -> None:
    total = 0.0
    deviation = max(g.deviation, g.moved, g.tolerance)
    for idx in _members(ir, g.outer):
        ids = [t for r in ir.shells[idx].regions for t in ir.regions[r].triangles]
        if ids and max(ids) >= len(tris):
            g.valid = False
            g.issues.append("mesh has fewer triangles than the IR references")
            return
        v = abs(_signed_volume(tris[np.asarray(ids, dtype=np.int64)])) if ids else 0.0
        total += v if ir.shells[idx].role == "outer" else -v
        for r in ir.shells[idx].regions:
            res = ir.regions[r].residual
            if res is not None:
                deviation = max(deviation, res.max)
    tolerance = INPUT_VOLUME_RELATIVE * abs(total) + g.area * deviation
    if not abs(g.expected - total) <= tolerance:
        g.valid = False
        g.issues.append(
            f"analytic volume {g.expected:.6g} differs from mesh volume {total:.6g}"
            f" by more than {tolerance:.3g}"
        )
        return
    _check_face_fluxes(ir, g, tris, occ)


def _check_face_fluxes(ir: Ir, g: _Group, tris: np.ndarray, occ) -> None:
    members = _members(ir, g.outer)
    origins, deviation = {}, {}
    for f in g.faces:
        ids = ir.regions[f.region].triangles
        if not ids:
            continue
        pts = tris[np.asarray(ids, dtype=np.int64)].reshape(-1, 3)
        origins[f.region] = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        res = ir.regions[f.region].residual
        deviation[f.region] = max(
            res.max if res is not None else 0.0,
            f.max_boundary_deviation,
            f.max_vertex_displacement,
            f.max_shape_tolerance,
        )
    if not origins:
        return
    every = tris[
        np.asarray([t for r in origins for t in ir.regions[r].triangles], dtype=np.int64)
    ].reshape(-1, 3)
    floor = FLUX_DEFLECTION * float(np.ptp(every, axis=0).max())
    deflection = max(min(deviation.values()), floor)
    facets = frozenset(r for r in origins if _is_facets(ir, r))
    written = occ.face_fluxes(g.shape, g.mapping, g.context, origins, deflection, facets)
    for idx in members:
        shell_ids = np.asarray(
            [t for r in ir.shells[idx].regions for t in ir.regions[r].triangles], dtype=np.int64
        )
        outward = (_signed_volume(tris[shell_ids]) > 0) == (ir.shells[idx].role == "outer")
        sign = 1.0 if outward else -1.0
        for r in ir.shells[idx].regions:
            mine = written.get(r)
            if mine is None:
                continue
            t = tris[np.asarray(ir.regions[r].triangles, dtype=np.int64)] - origins[r]
            theirs = sign * _signed_volume(t)
            d = deviation[r] + deflection
            bound = (mine.area + mine.perimeter * mine.reach) * d
            bound += FLUX_RELATIVE * mine.area * mine.reach
            if not abs(mine.flux - theirs) <= bound:
                g.valid = False
                g.issues.append(
                    f"region {r}: the written face's flux about its centre is {mine.flux:.6g},"
                    f" the mesh's {theirs:.6g} (allowed {bound:.3g})"
                )


def _signed_volume(t: np.ndarray) -> float:
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def _orient_closed(
    vertices: np.ndarray, raw: dict[int, np.ndarray], expect: dict[int, bool]
) -> dict[int, np.ndarray]:
    oriented = {}
    for idx, f in raw.items():
        if idx in expect and (_signed_volume(vertices[f]) < 0) == expect[idx]:
            f = f[:, ::-1]
        oriented[idx] = f
    return oriented


def _faceted(ir: Ir, tris: np.ndarray, topology, faceted):
    from unmesh import weld

    vertices, faces, source, _ = weld(tris, ir.tolerances.vertex_merge)
    faces = faces.astype(np.int64)
    welded = np.full(len(tris), -1, dtype=np.int64)
    welded[source] = np.arange(len(faces))
    shift = np.linalg.norm(tris[source] - vertices[faces], axis=2)
    moved = float(shift.max(initial=0.0))
    groups: list[_Group] = []
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(
            outer, "solid" if shell.closed else "shell", moved=moved, tolerance=faceted.UNCERTAINTY
        )
        groups.append(g)
        try:
            raw = {}
            for idx in _members(ir, outer):
                ids = np.array(
                    sorted(t for r in ir.shells[idx].regions for t in ir.regions[r].triangles),
                    dtype=np.int64,
                )
                if len(ids) and ids[-1] >= len(tris):
                    raise topology.BuildError("mesh has fewer triangles than the IR references")
                w = welded[ids]
                raw[idx] = faces[w[w >= 0]]
                if not len(raw[idx]):
                    raise topology.BuildError(f"shell {idx} has no non-degenerate triangles")
            if g.kind == "shell":
                g.shape = faceted.open_faces(vertices, raw[outer])
                g.shells = 1
                g.valid = True
                continue
            if any(not ir.shells[idx].closed for idx in raw):
                raise topology.BuildError("a cavity shell is open")
            expect = {idx: ir.shells[idx].role == "outer" for idx in raw}
            oriented = _orient_closed(vertices, raw, expect)
            g.shape = faceted.Body(
                faceted.shell_faces(vertices, oriented[outer], outer=True),
                [
                    faceted.shell_faces(vertices, f, outer=False)
                    for idx, f in oriented.items()
                    if idx != outer
                ],
            )
            g.shells = len(oriented)
            g.expected = sum(_signed_volume(vertices[f]) for f in oriented.values())
            g.area = sum(_area(vertices[f]) for f in oriented.values())
            g.valid = True
        except topology.BuildError as e:
            g.issues.append(str(e))
        except Exception as e:
            g.issues.append(f"{type(e).__name__}: {e}")
    return groups, vertices


def _area(t: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)


@dataclass
class _Verification:
    readback: ReadBack | None = None
    text: ReadBack | None = None
    skipped: str | None = None

    @property
    def by(self) -> str | None:
        names = [n for n, r in (("occt", self.readback), ("text", self.text)) if r is not None]
        return "+".join(names) or None

    @property
    def ok(self) -> bool:
        checks = [r for r in (self.readback, self.text) if r is not None]
        return bool(checks) and all(r.ok for r in checks)

    def issues(self) -> list[str]:
        out = [
            f"{name} of the written file does not match: {'; '.join(r.issues)}"
            for name, r in (("read-back", self.readback), ("text check", self.text))
            if r is not None and not r.ok
        ]
        if not self.by:
            out.append(f"the written file was not verified: {self.skipped}")
        return out


def _verify(path, written: list[_Group], occ, faceted_path: bool, options) -> _Verification:
    result = _Verification()
    curved = any(g.curved for g in written)
    if faceted_path:
        from unmesh._writer import textcheck

        read = textcheck.read(path)
        result.text = _compare(
            "text", read.solids, read.shells, written, faceted_path, options, read.issues
        )
    size = os.path.getsize(path)
    if size <= INPROCESS_READBACK_BYTES:
        try:
            solids, shells = occ.read_back(path, curved)
        except Exception as e:
            result.readback = _expected(written, "occt")
            result.readback.issues.append(f"could not re-import the file: {type(e).__name__}: {e}")
            return result
        result.readback = _compare("occt", solids, shells, written, faceted_path, options)
        return result
    from unmesh._writer import readback

    need = readback.estimate_mb(size)
    if faceted_path and need > options.readback_memory_mb:
        result.skipped = (
            f"OCCT read-back skipped: the {size / 2**20:.0f} MB file would need about"
            f" {need:.0f} MB to re-import, over the {options.readback_memory_mb:.0f} MB cap"
        )
        return result
    reader = "occt_no_face_orientation" if faceted_path else "occt"
    out = readback.run(
        path,
        curved,
        not faceted_path,
        options.readback_memory_mb,
        options.readback_timeout_s,
    )
    if out.limit is not None:
        result.skipped = f"OCCT read-back stopped: {out.limit}"
        return result
    if out.error is not None:
        result.readback = _expected(written, reader)
        result.readback.issues.append(out.error)
    else:
        result.readback = _compare(reader, out.solids, out.shells, written, faceted_path, options)
    result.readback.peak_rss_mb = out.peak_rss_mb
    return result


def _expected(written: list[_Group], reader: str) -> ReadBack:
    solids_expected = [g for g in written if g.kind == "solid"]
    return ReadBack(
        ok=False,
        solids=0,
        shells=0,
        volume=0.0,
        expected_solids=len(solids_expected),
        expected_shells=sum(g.shells for g in written),
        expected_volume=sum(g.expected for g in solids_expected),
        volume_tolerance=sum(_volume_tolerance(g) for g in solids_expected),
        reader=reader,
    )


def _compare(
    reader: str, solids, shells: int, written: list[_Group], faceted_path: bool, options, issues=()
) -> ReadBack:
    max_tol = options.max_shape_tolerance
    label = "text check" if reader == "text" else "read-back"
    rb = _expected(written, reader)
    rb.issues.extend(issues)
    solids_expected = [g for g in written if g.kind == "solid"]
    rb.solids = len(solids)
    rb.shells = shells
    rb.volume = sum(s.volume for s in solids)
    verb = "parsed" if reader == "text" else "re-imported"
    if rb.solids != rb.expected_solids:
        rb.issues.append(f"{verb} {rb.solids} solids, wrote {rb.expected_solids}")
    if rb.shells != rb.expected_shells:
        rb.issues.append(f"{verb} {rb.shells} shells, wrote {rb.expected_shells}")
    if not abs(rb.volume - rb.expected_volume) <= rb.volume_tolerance:
        rb.issues.append(
            f"{verb} volume {rb.volume:.12g} differs from the written "
            f"{rb.expected_volume:.12g} by more than {rb.volume_tolerance:.3g}"
        )
    if rb.solids == rb.expected_solids:
        for g, s in zip(solids_expected, solids, strict=True):
            bad = list(getattr(s, "issues", ()))
            if not s.valid and not bad:
                bad.append(f"the {verb} solid is invalid")
            if s.tolerance > max_tol:
                bad.append(f"{verb} shape tolerance {s.tolerance:.3g} exceeds {max_tol:.3g}")
            if s.shells != g.shells:
                bad.append(f"the {verb} solid has {s.shells} shells, wrote {g.shells}")
            if not abs(s.volume - g.expected) <= _volume_tolerance(g):
                bad.append(
                    f"{verb} volume {s.volume:.12g}, wrote {g.expected:.12g} "
                    f"(tolerance {_volume_tolerance(g):.3g})"
                )
            if faceted_path:
                g.volume = s.volume
                if reader != "text":
                    g.tolerance = s.tolerance
            if bad:
                g.valid = False
                g.issues.extend(f"{label}: {b}" for b in bad)
                rb.issues.extend(f"shell {g.outer}: {b}" for b in bad)
    rb.ok = not rb.issues
    return rb


def _volume_tolerance(g: _Group) -> float:
    return READBACK_RELATIVE * abs(g.expected) + g.area * g.tolerance


def write(
    ir: Ir,
    path: str | os.PathLike,
    options: WriteOptions | None = None,
    *,
    mesh=None,
    verify: bool = True,
) -> WriteReport:
    started = time.perf_counter()
    ir.validate()
    options = options or WriteOptions()
    require_ocp()
    from unmesh._writer import faceted, occ, topology

    n_outer = sum(1 for s in ir.shells if s.role == "outer")
    tris = _triangles(mesh) if mesh is not None else None
    share = _faceted_share(ir)
    whole = tris is not None and FACETED_SHARE < share < 1.0
    if whole:
        groups, seams = [], []
        for outer, shell in enumerate(ir.shells):
            if shell.role == "outer":
                g = _Group(outer, "solid" if shell.closed else "shell")
                g.issues.append(
                    f"facets regions hold {share:.0%} of the triangles, over the"
                    f" {FACETED_SHARE:.0%} share above which the faceted solid is written"
                )
                groups.append(g)
    else:
        groups, seams = _analytic(ir, options, occ, topology)
    if tris is not None:
        limit = topology.deviation_limit(ir, options.max_deviation)
        for g in groups:
            if not whole:
                _check_patches(ir, g, tris, options.max_seam_gap, limit)
            if g.valid and g.kind == "solid":
                _check_input_volume(ir, g, tris, occ)
    fallback = None
    reason = None
    vertices = None
    if not all(g.valid for g in groups):
        reason = "; ".join(f"shell {g.outer}: {'; '.join(g.issues)}" for g in groups if not g.valid)
        if mesh is not None:
            groups, vertices = _faceted(ir, tris, topology, faceted)
            fallback = "faceted"
        else:
            reason += "; no mesh given, so no faceted fallback"
    complete = all(g.valid for g in groups)
    written = groups if complete else []
    issues = []
    checked = _Verification()
    timings: dict[str, float] = {}
    if written:
        if fallback == "faceted":
            faceted.write(
                path,
                vertices,
                [g.shape for g in written if g.kind == "solid"],
                [g.shape for g in written if g.kind == "shell"],
            )
        else:
            occ.write_step(occ.compound_of([g.shape for g in written]), path)
        timings["write_s"] = time.perf_counter() - started
        if verify:
            mark = time.perf_counter()
            checked = _verify(path, written, occ, fallback == "faceted", options)
            timings["readback_s"] = time.perf_counter() - mark
            issues.extend(checked.issues())
    else:
        issues.append("nothing was written")
    faces = [f for g in written for f in g.faces] if fallback is None else []
    if fallback is None and not complete and reason is not None:
        issues.append(reason)
    return WriteReport(
        valid=complete and len(written) == n_outer and (not verify or checked.ok),
        solids=_solid_count(written, occ, fallback, checked),
        max_shape_tolerance=max((g.tolerance for g in written), default=0.0),
        faces=faces,
        fallback=fallback,
        fallback_reason=reason if fallback else None,
        shells=[
            ShellReport(
                g.outer, g.kind, g.valid, g.volume, g.tolerance, g.issues, g.moved, g.deviation
            )
            for g in groups
        ],
        seams=seams,
        issues=issues,
        faceted_regions=_faceted_regions(ir, written, fallback),
        faceted_faces=sum(
            1 for g in written if fallback is None for r, _ in g.mapping if _is_facets(ir, r)
        ),
        max_vertex_displacement=max((g.moved for g in groups), default=0.0),
        max_boundary_deviation=max((g.deviation for g in groups), default=0.0),
        edge_fallbacks=[e for g in written for e in g.edge_fallbacks] if fallback is None else [],
        tangent_edges=[e for g in written for e in g.tangent_edges] if fallback is None else [],
        open_shells=[g.outer for g in written if g.kind == "shell"],
        readback=checked.readback,
        verified=checked.by is not None,
        verified_by=checked.by,
        text_check=checked.text,
        readback_skipped=checked.skipped,
        volume_checked_against_input=tris is not None,
        timings=timings,
    )


def _solid_count(written: list[_Group], occ, fallback, checked: _Verification) -> int:
    for rb in (checked.readback, checked.text):
        if rb is not None:
            return rb.solids
    if fallback == "faceted":
        return sum(1 for g in written if g.kind == "solid")
    return sum(occ.count_solids(g.shape) for g in written)


def _is_facets(ir: Ir, region: int) -> bool:
    return ir.regions[region].surface.type == "facets"


def _faceted_regions(ir: Ir, written: list[_Group], fallback) -> int:
    regions = [r for g in written for idx in _members(ir, g.outer) for r in ir.shells[idx].regions]
    if fallback == "faceted":
        return len(regions)
    return sum(1 for r in regions if _is_facets(ir, r))


def _faceted_share(ir: Ir) -> float:
    total = sum(len(r.triangles) for r in ir.regions)
    facets = sum(len(r.triangles) for r in ir.regions if _is_facets(ir, r.id))
    return facets / total if total else 0.0
