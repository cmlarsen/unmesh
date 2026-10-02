from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

import unmesh
from unmesh.ir import Facets, Ir, Region, Shell, Source, Tolerances

ConverterResult = tuple[str | None, str | None, str | None]
Converter = Callable[[Path], ConverterResult]

FACETED_WELD = 1e-6
FACETED_WRITE_MAX_TRIANGLES = 1000


def _write_report(wr) -> dict:
    return {
        "valid": wr.valid,
        "fallback": wr.fallback,
        "fallback_reason": wr.fallback_reason,
        "solids": wr.solids,
        "issues": list(wr.issues),
    }


def convert_unmesh(stl_path: Path) -> ConverterResult:
    from unmesh import step

    tris = unmesh.read_stl(stl_path)
    ir, report = unmesh.convert(tris)
    step_path = Path(stl_path).with_suffix(".step")
    wr = step.write(ir, step_path, mesh=tris)
    payload = {
        "max_deviation": report.max_deviation,
        "rms_deviation": report.rms_deviation,
        "analytic_area_fraction": report.analytic_area_fraction,
        "warnings": [w.code for w in report.warnings],
        "write": _write_report(wr),
    }
    return ir.dumps(), str(step_path) if step_path.exists() else None, json.dumps(payload)


def _is_closed(faces: np.ndarray) -> bool:
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    forward = {(int(a), int(b)) for a, b in directed}
    return len(forward) == len(directed) and all((b, a) in forward for a, b in forward)


def faceted_ir(tris: np.ndarray) -> Ir:
    verts, faces, kept, _ = unmesh.weld(tris, FACETED_WELD)
    surface = Facets([tuple(v) for v in verts.tolist()], [tuple(f) for f in faces.tolist()])
    region = Region(0, surface, kept.tolist(), None)
    shell = Shell(_is_closed(faces), "outer", None, [0])
    return Ir(
        Tolerances(linear=FACETED_WELD),
        Source(len(tris), len(verts)),
        [shell],
        [region],
        [],
        [],
    )


def convert_faceted(stl_path: Path) -> ConverterResult:
    from unmesh import step

    tris = unmesh.read_stl(stl_path)
    ir = faceted_ir(tris)
    payload: dict = {"max_deviation": FACETED_WELD, "warnings": []}
    if len(tris) > FACETED_WRITE_MAX_TRIANGLES:
        payload["write"] = {"skipped": True}
        return ir.dumps(), None, json.dumps(payload)
    step_path = Path(stl_path).with_suffix(".step")
    wr = step.write(ir, step_path, mesh=tris)
    payload["write"] = _write_report(wr)
    return ir.dumps(), str(step_path) if step_path.exists() else None, json.dumps(payload)


BASELINES = frozenset({"faceted"})

CONVERTERS: dict[str, Converter] = {"unmesh": convert_unmesh, "faceted": convert_faceted}


def get_converter(name: str) -> Converter:
    if name in CONVERTERS:
        return CONVERTERS[name]
    if ":" in name:
        module, attr = name.split(":", 1)
        return getattr(importlib.import_module(module), attr)
    raise KeyError(f"unknown converter {name!r}; known: {sorted(CONVERTERS)} or module:function")


def plugin_hash(name: str) -> str:
    import hashlib
    import importlib.util

    if name in CONVERTERS or ":" not in name:
        return ""
    spec = importlib.util.find_spec(name.split(":", 1)[0])
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        return "unresolved"
    return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()[:12]
