from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unmesh import _core
from unmesh.api import ConvertOptions, MeshLike, Report, convert, read_mesh
from unmesh.ir import Ir

SCHEMA = "unmesh.fidelity"
SCHEMA_VERSION = 1
UNITS = {"mm": 1.0, "in": 25.4}
EXIT_CODES = {"analytic": 0, "mixed": 1, "faceted": 2, "error": 3}
MEASURE_RELATIVE = 2e-5
MEASURE_ANGULAR = 0.1


@dataclass
class Conversion:
    ir: Ir
    report: Report
    write: Any
    outcome: str
    fidelity: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.outcome]


def _input(mesh_or_path: MeshLike, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"path": None, "format": "array"}
    if isinstance(mesh_or_path, str | os.PathLike):
        path = os.fspath(mesh_or_path)
        info = {"path": path, "format": "obj" if path.lower().endswith(".obj") else "stl"}
        tris = read_mesh(path)
    elif isinstance(mesh_or_path, tuple):
        vertices = np.asarray(mesh_or_path[0], dtype=np.float64)
        faces = np.asarray(mesh_or_path[1], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError(f"expected vertices of shape (v, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError(f"expected faces of shape (f, 3), got {faces.shape}")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face index out of range")
        tris = vertices[faces]
    else:
        tris = np.asarray(mesh_or_path, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"expected triangles of shape (n, 3, 3), got {tris.shape}")
    return np.ascontiguousarray(tris * scale, dtype=np.float64), info


def _scaled(options: ConvertOptions | None, scale: float) -> ConvertOptions:
    options = options or ConvertOptions()
    if options.linear_tolerance is None or scale == 1.0:
        return options
    return dataclasses.replace(options, linear_tolerance=options.linear_tolerance * scale)


def convert_to_step(
    mesh_or_path: MeshLike,
    step_path: str | os.PathLike,
    options: ConvertOptions | None = None,
    *,
    unit: str = "mm",
    write_options=None,
    measure: bool = True,
) -> Conversion:
    from unmesh import step

    if unit not in UNITS:
        raise ValueError(f"unit must be one of {sorted(UNITS)}, got {unit!r}")
    step.require_ocp()
    started = time.perf_counter()
    scale = UNITS[unit]
    tris, source = _input(mesh_or_path, scale)
    read_s = time.perf_counter() - started
    mark = time.perf_counter()
    ir, report = convert(tris, _scaled(options, scale))
    convert_s = time.perf_counter() - mark
    wr = step.write(ir, step_path, write_options, mesh=tris)
    measured = None
    measure_s = None
    if measure and wr.valid and os.path.isfile(step_path):
        mark = time.perf_counter()
        measured = measure_step(step_path, tris)
        measure_s = time.perf_counter() - mark
    outcome = classify(ir, wr)
    runtime = {
        "read": read_s,
        "convert": convert_s,
        "write": wr.timings.get("write_s"),
        "readback": wr.timings.get("readback_s"),
        "measure": measure_s,
        "total": time.perf_counter() - started,
    }
    fidelity = build_report(
        ir,
        report,
        wr,
        outcome,
        source={**source, "unit": unit, "scale_to_mm": scale, "triangles": int(len(tris))},
        output=os.fspath(step_path),
        measured=measured,
        runtime=runtime,
    )
    return Conversion(ir, report, wr, outcome, fidelity)


def classify(ir: Ir, wr) -> str:
    if not wr.valid:
        return "error"
    analytic = any(r.surface.type != "facets" for r in ir.regions)
    if wr.fallback == "faceted" or not analytic:
        return "faceted"
    return "mixed" if wr.faceted_regions else "analytic"


def _stats(d: np.ndarray) -> dict[str, Any]:
    if not len(d):
        return {"max": None, "mean": None, "p99": None, "samples": 0}
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.quantile(d, 0.99)),
        "samples": int(len(d)),
    }


def measure_step(step_path: str | os.PathLike, tris: np.ndarray) -> dict[str, Any]:
    from unmesh._writer import occ

    corners = tris.reshape(-1, 3)
    diagonal = float(np.linalg.norm(np.ptp(corners, axis=0)))
    deflection = MEASURE_RELATIVE * diagonal
    shape = occ.read_step(step_path)
    nodes, written = occ.tessellate(shape, deflection, MEASURE_ANGULAR)
    to_input = _core.mesh_distances(tris, nodes) if len(nodes) else np.zeros(0)
    vertices = np.unique(corners, axis=0)
    to_step = _core.mesh_distances(written, vertices) if len(written) else np.zeros(0)
    both = np.concatenate([to_input, to_step])
    return {
        "max": float(both.max()) if len(both) else None,
        "tessellation_deflection": deflection,
        "step_to_input": _stats(to_input),
        "input_to_step": _stats(to_step),
    }


def _faceted_reason(ir: Ir, region: int, shell_of: dict[int, int], codes: set[str]) -> str:
    if not ir.shells[shell_of[region]].closed:
        return "non_manifold" if "non_manifold_edges" in codes else "open_shell"
    if "fallback_facets" in codes:
        return "assembly_failed"
    return "no_surface_fit"


def _writer_reason(wr) -> str:
    reason = wr.fallback_reason or ""
    return "facets_share" if "facets regions hold" in reason else "shell_failed"


def build_report(
    ir: Ir,
    report: Report,
    wr,
    outcome: str,
    *,
    source: dict[str, Any],
    output: str | None,
    measured: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    from unmesh import __version__

    shell_of = {r: i for i, s in enumerate(ir.shells) for r in s.regions}
    codes = {w.code for w in report.warnings}
    written = {f.region: f for f in wr.faces}
    fallback = wr.fallback == "faceted"
    faces, faceted = [], []
    for region in ir.regions:
        kind = region.surface.type
        face = written.get(region.id)
        faces.append(
            {
                "region": region.id,
                "surface": kind,
                "shell": shell_of[region.id],
                "triangles": len(region.triangles),
                "written_as": "triangles" if fallback or kind == "facets" else kind,
                "converter": None
                if region.residual is None
                else {"max": region.residual.max, "rms": region.residual.rms},
                "writer": None
                if face is None
                else {
                    "max_shape_tolerance": face.max_shape_tolerance,
                    "max_vertex_displacement": face.max_vertex_displacement,
                    "max_boundary_deviation": face.max_boundary_deviation,
                },
            }
        )
        if kind == "facets":
            faceted.append(
                {"region": region.id, "reason": _faceted_reason(ir, region.id, shell_of, codes)}
            )
        elif fallback:
            faceted.append({"region": region.id, "reason": _writer_reason(wr)})
    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "unmesh_version": __version__,
        "outcome": outcome,
        "exit_code": EXIT_CODES[outcome],
        "units": "mm",
        "input": {**source, "warnings": [dataclasses.asdict(w) for w in report.warnings]},
        "output": {"path": output, "written": bool(output and os.path.isfile(output))},
        "tolerances": dataclasses.asdict(ir.tolerances),
        "region_counts": dict(report.region_counts),
        "analytic_area_fraction": report.analytic_area_fraction,
        "deviation": {
            "converter": {
                "max": report.max_deviation,
                "rms": report.rms_deviation,
                "kind": "bound",
            },
            "writer": {
                "max_vertex_displacement": wr.max_vertex_displacement,
                "max_boundary_deviation": wr.max_boundary_deviation,
                "max_shape_tolerance": wr.max_shape_tolerance,
            },
            "measured": measured,
        },
        "faces": faces,
        "faceted": faceted,
        "decisions": [_decision(d) for d in report.decisions],
        "validity": {
            "valid": wr.valid,
            "verified": wr.verified,
            "readback": None
            if wr.readback is None
            else {"ok": wr.readback.ok, "issues": list(wr.readback.issues)},
            "volume_checked_against_input": wr.volume_checked_against_input,
            "fallback": wr.fallback,
            "fallback_reason": wr.fallback_reason,
            "solids": wr.solids,
            "open_shells": list(wr.open_shells),
            "issues": list(wr.issues),
            "edge_fallbacks": [
                {"regions": list(e.regions), "kind": e.kind, "max_deviation": e.max_deviation}
                for e in wr.edge_fallbacks
            ],
        },
        "runtime_s": runtime,
        "error": None
        if outcome != "error"
        else {"type": "WriteError", "message": "; ".join(wr.issues) or "the STEP is not valid"},
    }


def _decision(d) -> dict[str, Any]:
    out = dataclasses.asdict(d)
    for key in ("regions", "supports", "axis", "axis_point"):
        out[key] = list(out[key])
    return out


def error_report(
    error: BaseException,
    *,
    source: dict[str, Any] | None = None,
    output: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from unmesh import __version__

    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "unmesh_version": __version__,
        "outcome": "error",
        "exit_code": EXIT_CODES["error"],
        "units": "mm",
        "input": source,
        "output": {"path": output, "written": False},
        "runtime_s": runtime or {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(_finite(report), indent=2, allow_nan=False) + "\n"


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _finite(value.item())
    return value
