from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from unmesh.ir import Ir


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
    open_shells: list[int] = field(default_factory=list)


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
    ok = occ.is_valid(g.shape)
    if not ok:
        g.issues.append("BRepCheck_Analyzer reports the shape invalid")
    if g.kind == "solid":
        g.volume = occ.volume_of(g.shape)
        if not g.volume > 0:
            ok = False
            g.issues.append(f"non-positive volume {g.volume:.6g}")
    if g.tolerance > max_tol:
        ok = False
        g.issues.append(f"shape tolerance {g.tolerance:.3g} exceeds {max_tol:.3g}")
    g.valid = ok


def _finish_group(g: _Group, occ, outer_shell, cavities, options) -> None:
    if g.kind == "solid":
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
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(outer, "solid" if shell.closed else "shell")
        groups.append(g)
        try:
            built = []
            moved: dict[int, float] = {}
            dev: dict[int, float] = {}
            for idx in _members(ir, outer):
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


def _signed_volume(t: np.ndarray) -> float:
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def _inside(tris: np.ndarray, point: np.ndarray) -> bool:
    direction = np.array([1.0, 0.372185571, 0.591327399])
    a = tris[:, 0]
    e1 = tris[:, 1] - a
    e2 = tris[:, 2] - a
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    valid = np.abs(det) > 1e-18
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
    tvec = point - a
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", qvec, direction) * inv
    dist = np.einsum("ij,ij->i", e2, qvec) * inv
    hits = (
        valid
        & (u >= -1e-9)
        & (u <= 1.0 + 1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (dist > 1e-9)
    )
    return bool(np.count_nonzero(hits) % 2 == 1)


def _nesting(closed: dict[int, np.ndarray]) -> dict[int, bool]:
    ids = list(closed)
    parent: dict[int, int | None] = {}
    for i in ids:
        probe = closed[i][0][0]
        vols = {
            j: abs(_signed_volume(closed[j])) for j in ids if j != i and _inside(closed[j], probe)
        }
        parent[i] = min(vols, key=vols.__getitem__, default=None)
    expect = {}
    for i in ids:
        depth = 0
        p = parent[i]
        while p is not None:
            depth += 1
            p = parent[p]
        expect[i] = depth % 2 == 0
    return expect


def _orient_closed(raw: dict[int, np.ndarray], expect: dict[int, bool]) -> dict[int, np.ndarray]:
    oriented = {}
    for idx, t in raw.items():
        if idx in expect and (_signed_volume(t) < 0) == expect[idx]:
            t = t[:, ::-1]
        oriented[idx] = t
    return oriented


def _faceted(ir: Ir, tris: np.ndarray, options: WriteOptions, occ, topology):
    groups: list[_Group] = []
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(outer, "solid" if shell.closed else "shell")
        groups.append(g)
        try:
            raw = {}
            for idx in _members(ir, outer):
                ids = sorted(t for r in ir.shells[idx].regions for t in ir.regions[r].triangles)
                if ids and ids[-1] >= len(tris):
                    raise topology.BuildError("mesh has fewer triangles than the IR references")
                raw[idx] = tris[ids]
            closed = {idx: t for idx, t in raw.items() if ir.shells[idx].closed and len(t)}
            expect = _nesting(closed)
            shells = {}
            for idx, t in _orient_closed(raw, expect).items():
                shells[idx] = occ.build_triangle_shell(np.ascontiguousarray(t))
            for idx in closed:
                shells[idx] = occ.orient_like_mesh(shells[idx], expect[idx])
            if g.kind == "solid":
                cavities = [s for idx, s in shells.items() if idx != outer]
                g.shape = occ.make_faceted_solid(shells[outer], cavities)
            else:
                g.shape = shells[outer]
            _check(g, occ, options.max_shape_tolerance)
            if g.kind == "solid" and g.valid:
                merged = occ.unify_exact(g.shape)
                trial = _Group(outer, "solid", merged)
                _check(trial, occ, options.max_shape_tolerance)
                if trial.valid and abs(trial.volume - g.volume) <= 1e-9 * abs(g.volume):
                    g.shape, g.tolerance, g.volume = merged, trial.tolerance, trial.volume
        except topology.BuildError as e:
            g.issues.append(str(e))
        except Exception as e:
            g.issues.append(f"{type(e).__name__}: {e}")
    return groups


def write(
    ir: Ir, path: str | os.PathLike, options: WriteOptions | None = None, *, mesh=None
) -> WriteReport:
    ir.validate()
    options = options or WriteOptions()
    try:
        from unmesh._writer import occ, topology
    except ImportError as e:
        raise ImportError(
            "unmesh.step needs OCP; install it with `pip install unmesh[step]`"
        ) from e

    n_outer = sum(1 for s in ir.shells if s.role == "outer")
    groups, seams = _analytic(ir, options, occ, topology)
    fallback = None
    reason = None
    if not all(g.valid for g in groups):
        reason = "; ".join(f"shell {g.outer}: {'; '.join(g.issues)}" for g in groups if not g.valid)
        if mesh is not None:
            tris = _triangles(mesh)
            groups = _faceted(ir, tris, options, occ, topology)
            fallback = "faceted"
        else:
            reason += "; no mesh given, so no faceted fallback"
    complete = all(g.valid for g in groups)
    written = groups if complete else []
    issues = []
    if written:
        occ.write_step(occ.compound_of([g.shape for g in written]), path)
    else:
        issues.append("nothing was written")
    faces = [f for g in written for f in g.faces] if fallback is None else []
    if fallback is None and not complete and reason is not None:
        issues.append(reason)
    return WriteReport(
        valid=complete and len(written) == n_outer,
        solids=sum(occ.count_solids(g.shape) for g in written),
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
        open_shells=[g.outer for g in written if g.kind == "shell"],
    )
