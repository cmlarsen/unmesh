from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from unmesh.ir import Ir

READBACK_RELATIVE = 1e-9
INPUT_VOLUME_RELATIVE = 1e-6
FLUX_DEFLECTION = 1e-4
FLUX_RELATIVE = 1e-9


@dataclass(frozen=True)
class WriteOptions:
    max_shape_tolerance: float = 1e-3
    max_deviation: float = 5e-3


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
    max_vertex_displacement: float = 0.0
    max_boundary_deviation: float = 0.0
    edge_fallbacks: list[EdgeFallback] = field(default_factory=list)
    tangent_edges: list[TangentEdge] = field(default_factory=list)
    open_shells: list[int] = field(default_factory=list)
    readback: ReadBack | None = None
    verified: bool = False
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


def _triangles(mesh) -> np.ndarray:
    if isinstance(mesh, tuple):
        v = np.asarray(mesh[0], dtype=float)
        f = np.asarray(mesh[1], dtype=np.int64)
        tris = v[f]
    elif isinstance(mesh, (str, os.PathLike)):
        from unmesh import read_stl

        tris = read_stl(mesh)
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
                    cs = curved.build_shell(ir, idx, *refined, limit, occ.new_pool())
                    seams.extend(
                        SeamReport(s.regions, s.surface_type, s.points, s.max_gap) for s in cs.seams
                    )
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
                seams.extend(
                    SeamReport(s.regions, s.surface_type, s.points, s.max_gap) for s in plan.seams
                )
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
    written = occ.face_fluxes(g.shape, g.mapping, g.context, origins, deflection)
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


def _verify(path, written: list[_Group], occ, faceted_path: bool, max_tol: float) -> ReadBack:
    solids_expected = [g for g in written if g.kind == "solid"]
    rb = ReadBack(
        ok=False,
        solids=0,
        shells=0,
        volume=0.0,
        expected_solids=len(solids_expected),
        expected_shells=sum(g.shells for g in written),
        expected_volume=sum(g.expected for g in solids_expected),
        volume_tolerance=sum(_volume_tolerance(g) for g in solids_expected),
    )
    try:
        solids, rb.shells = occ.read_back(path, any(g.curved for g in written))
    except Exception as e:
        rb.issues.append(f"could not re-import the file: {type(e).__name__}: {e}")
        return rb
    rb.solids = len(solids)
    rb.volume = sum(s.volume for s in solids)
    if rb.solids != rb.expected_solids:
        rb.issues.append(f"re-imported {rb.solids} solids, wrote {rb.expected_solids}")
    if rb.shells != rb.expected_shells:
        rb.issues.append(f"re-imported {rb.shells} shells, wrote {rb.expected_shells}")
    if not abs(rb.volume - rb.expected_volume) <= rb.volume_tolerance:
        rb.issues.append(
            f"re-imported volume {rb.volume:.12g} differs from the written "
            f"{rb.expected_volume:.12g} by more than {rb.volume_tolerance:.3g}"
        )
    if rb.solids == rb.expected_solids:
        for g, s in zip(solids_expected, solids, strict=True):
            bad = []
            if not s.valid:
                bad.append("the re-imported solid is invalid")
            if s.tolerance > max_tol:
                bad.append(f"re-imported shape tolerance {s.tolerance:.3g} exceeds {max_tol:.3g}")
            if s.shells != g.shells:
                bad.append(f"the re-imported solid has {s.shells} shells, wrote {g.shells}")
            if not abs(s.volume - g.expected) <= _volume_tolerance(g):
                bad.append(
                    f"re-imported volume {s.volume:.12g}, wrote {g.expected:.12g} "
                    f"(tolerance {_volume_tolerance(g):.3g})"
                )
            if faceted_path:
                g.volume = s.volume
                g.tolerance = s.tolerance
            if bad:
                g.valid = False
                g.issues.extend(f"read-back: {b}" for b in bad)
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
    try:
        from unmesh._writer import faceted, occ, topology
    except ImportError as e:
        raise ImportError(
            "unmesh.step needs OCP; install it with `pip install unmesh[step]`"
        ) from e

    n_outer = sum(1 for s in ir.shells if s.role == "outer")
    groups, seams = _analytic(ir, options, occ, topology)
    tris = _triangles(mesh) if mesh is not None else None
    if tris is not None:
        for g in groups:
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
    readback = None
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
            readback = _verify(
                path, written, occ, fallback == "faceted", options.max_shape_tolerance
            )
            timings["readback_s"] = time.perf_counter() - mark
            if not readback.ok:
                mismatch = "; ".join(readback.issues)
                issues.append(f"read-back of the written file does not match: {mismatch}")
    else:
        issues.append("nothing was written")
    faces = [f for g in written for f in g.faces] if fallback is None else []
    if fallback is None and not complete and reason is not None:
        issues.append(reason)
    return WriteReport(
        valid=complete and len(written) == n_outer and (readback is None or readback.ok),
        solids=readback.solids if readback is not None else _solid_count(written, occ, fallback),
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
        max_vertex_displacement=max((g.moved for g in groups), default=0.0),
        max_boundary_deviation=max((g.deviation for g in groups), default=0.0),
        edge_fallbacks=[e for g in written for e in g.edge_fallbacks] if fallback is None else [],
        tangent_edges=[e for g in written for e in g.tangent_edges] if fallback is None else [],
        open_shells=[g.outer for g in written if g.kind == "shell"],
        readback=readback,
        verified=readback is not None,
        volume_checked_against_input=tris is not None,
        timings=timings,
    )


def _solid_count(written: list[_Group], occ, fallback) -> int:
    if fallback == "faceted":
        return sum(1 for g in written if g.kind == "solid")
    return sum(occ.count_solids(g.shape) for g in written)
