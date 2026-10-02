from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from unmesh.ir import Ir


@dataclass(frozen=True)
class WriteOptions:
    max_shape_tolerance: float = 1e-3


@dataclass
class FaceReport:
    region: int
    surface_type: str
    max_shape_tolerance: float


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
        g.shape = occ.fix_shape(solid)
    else:
        g.shape = outer_shell
    _check(g, occ, options.max_shape_tolerance)


def _analytic(ir: Ir, options: WriteOptions, occ, topology):
    groups: list[_Group] = []
    seams: list[SeamReport] = []
    vpos = None
    try:
        vpos = topology.vertex_positions(ir)
        vertex_error = None
    except topology.BuildError as e:
        vertex_error = str(e)
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(outer, "solid" if shell.closed else "shell")
        groups.append(g)
        try:
            if vertex_error is not None:
                raise topology.BuildError(vertex_error)
            built = []
            for idx in _members(ir, outer):
                plan = topology.plan_shell(ir, idx, vpos)
                seams.extend(
                    SeamReport(s.regions, s.surface_type, s.points, s.max_gap) for s in plan.seams
                )
                built.append((idx, *occ.build_plan_shell(plan)))
            per_region: dict[int, float] = {}
            for _, _, tols in built:
                for region, tol in tols:
                    per_region[region] = max(per_region.get(region, 0.0), tol)
            g.faces = [
                FaceReport(r, ir.regions[r].surface.type, t) for r, t in sorted(per_region.items())
            ]
            cavities = [sh for idx, sh, _ in built if idx != outer]
            _finish_group(g, occ, built[0][1], cavities, options)
        except topology.BuildError as e:
            g.issues.append(str(e))
        except Exception as e:
            g.issues.append(f"{type(e).__name__}: {e}")
    return groups, seams


def _faceted(ir: Ir, tris: np.ndarray, options: WriteOptions, occ, topology):
    groups: list[_Group] = []
    for outer, shell in enumerate(ir.shells):
        if shell.role != "outer":
            continue
        g = _Group(outer, "solid" if shell.closed else "shell")
        groups.append(g)
        try:
            shells = {}
            for idx in _members(ir, outer):
                ids = sorted(t for r in ir.shells[idx].regions for t in ir.regions[r].triangles)
                if ids and ids[-1] >= len(tris):
                    raise topology.BuildError("mesh has fewer triangles than the IR references")
                t = tris[ids]
                if ir.shells[idx].closed:
                    a, b, c = t[:, 0], t[:, 1], t[:, 2]
                    vol = float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)
                    if (vol < 0) == (ir.shells[idx].role == "outer"):
                        t = t[:, ::-1]
                shells[idx] = occ.build_triangle_shell(np.ascontiguousarray(t))
            cavities = [sh for idx, sh in shells.items() if idx != outer]
            _finish_group(g, occ, shells[outer], cavities, options)
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
    written = [g for g in groups if g.valid]
    issues = []
    if written:
        occ.write_step(occ.compound_of([g.shape for g in written]), path)
    else:
        issues.append("nothing was written")
    faces = [f for g in written for f in g.faces] if fallback is None else []
    return WriteReport(
        valid=len(written) == n_outer,
        solids=sum(occ.count_solids(g.shape) for g in written),
        max_shape_tolerance=max((g.tolerance for g in written), default=0.0),
        faces=faces,
        fallback=fallback,
        fallback_reason=reason if fallback else None,
        shells=[
            ShellReport(g.outer, g.kind, g.valid, g.volume, g.tolerance, g.issues) for g in groups
        ],
        seams=seams,
        issues=issues if reason is None or fallback else [*issues, reason],
    )
