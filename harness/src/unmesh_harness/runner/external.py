from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from unmesh.ir import Facets, Ir, Region, Residual, Shell, Source, Tolerances

FREECAD_ENV = "UNMESH_FREECAD"
STL2STEP_ENV = "UNMESH_STL2STEP"
TIMEOUT_ENV = "UNMESH_BASELINE_TIMEOUT"
MEMORY_ENV = "UNMESH_BASELINE_MEMORY_MB"
PGID_ENV = "UNMESH_BASELINE_PGID_FILE"
DEFAULT_TIMEOUT_S = 55.0
VERSION_TIMEOUT_S = 15.0
POLL_S = 0.05
FREECAD_MESH_TOLERANCE = 0.05
LABEL_DEFLECTION = (0.005, 0.1)
CANDIDATES = 16
NORMAL_WEIGHT = 1e-3
SUBDIVIDE_LEVELS = 8

FREECAD_SCRIPT = Path(__file__).with_name("freecad_refine.py")


class ToolUnavailable(RuntimeError):
    pass


class MemoryCapExceeded(RuntimeError):
    def __init__(self, name: str, cap_mb: float, peak_mb: float) -> None:
        self.cap_mb = cap_mb
        self.peak_mb = peak_mb
        super().__init__(
            f"external tool {name} exceeded memory cap of {cap_mb:g} MB "
            f"(observed peak {peak_mb:.0f} MB)"
        )


def _executable(env: str, what: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise ToolUnavailable(f"{what} not configured: set {env} to its executable")
    path = shutil.which(value) or (value if Path(value).is_file() else None)
    if path is None or not os.access(path, os.X_OK):
        raise ToolUnavailable(f"{what} not found: {env}={value} is not an executable file")
    return path


def freecad_command() -> str:
    return _executable(FREECAD_ENV, "FreeCAD (freecadcmd)")


def stl2step_command() -> str:
    return _executable(STL2STEP_ENV, "stl2step")


def _timeout() -> float:
    return float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))


def _memory_cap_mb() -> float | None:
    value = os.environ.get(MEMORY_ENV)
    return float(value) if value else None


def _tree_rss_mb(pid: int) -> float:
    import psutil

    try:
        procs = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0.0
    for proc in procs:
        try:
            total += proc.memory_info().rss / 2**20
        except psutil.Error:
            pass
    return total


def _kill_tree(proc: subprocess.Popen) -> None:
    import psutil

    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        children = []
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _record_tool_pgid(pid: int) -> str | None:
    path = os.environ.get(PGID_ENV)
    if not path:
        return None
    try:
        Path(path).write_text(str(os.getpgid(pid)))
    except (OSError, PermissionError):
        return None
    return path


def _clear_tool_pgid(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _run(
    cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = None
) -> tuple[int, str, str]:
    timeout = _timeout() if timeout is None else timeout
    cap = _memory_cap_mb()
    name = Path(cmd[0]).name
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    pgid_file = _record_tool_pgid(proc.pid)
    deadline = time.monotonic() + timeout
    peak = 0.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{name} exceeded {timeout:g} s")
            try:
                out, err = proc.communicate(timeout=min(POLL_S, remaining))
                break
            except subprocess.TimeoutExpired:
                if cap is None:
                    continue
                peak = max(peak, _tree_rss_mb(proc.pid))
                if peak > cap:
                    raise MemoryCapExceeded(name, cap, peak) from None
    except BaseException:
        _kill_tree(proc)
        raise
    finally:
        _clear_tool_pgid(pgid_file)
    return proc.returncode, out, err


def _result_line(out: str, prefix: str = "RESULT ") -> dict[str, Any] | None:
    for line in reversed(out.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    return None


def _subdivide(tris: np.ndarray, owner: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    done_t, done_o = [], []
    for _ in range(SUBDIVIDE_LEVELS):
        edges = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=2).max(axis=1)
        big = edges > h
        done_t.append(tris[~big])
        done_o.append(owner[~big])
        if not big.any():
            tris = tris[:0]
            break
        t, o = tris[big], owner[big]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tris = np.concatenate(
            [
                np.stack([a, ab, ca], 1),
                np.stack([ab, b, bc], 1),
                np.stack([ca, bc, c], 1),
                np.stack([ab, bc, ca], 1),
            ]
        )
        owner = np.tile(o, 4)
    done_t.append(tris)
    done_o.append(owner)
    return np.concatenate(done_t), np.concatenate(done_o)


def _unit_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def assign_faces(
    input_tris: np.ndarray, step_tris: np.ndarray, step_face: np.ndarray
) -> np.ndarray:
    from scipy.spatial import cKDTree

    from ..degrade.processing import _point_tri_dist2

    tris = np.asarray(input_tris, dtype=np.float64).reshape(-1, 3, 3)
    edge = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=2)
    h = max(float(np.median(edge)), 1e-6)
    ref, owner = _subdivide(np.asarray(step_tris, dtype=np.float64), step_face, h)
    k = min(CANDIDATES, len(ref))
    _, idx = cKDTree(ref.mean(axis=1)).query(tris.mean(axis=1), k=k)
    idx = np.asarray(idx).reshape(len(tris), k)
    centroid = np.repeat(tris.mean(axis=1), k, axis=0)
    cand = ref[idx.reshape(-1)]
    dist = np.sqrt(_point_tri_dist2(centroid, cand[:, 0], cand[:, 1], cand[:, 2]))
    align = (np.repeat(_unit_normals(tris), k, axis=0) * _unit_normals(cand)).sum(axis=1)
    score = (dist + NORMAL_WEIGHT * (1.0 - align)).reshape(len(tris), k)
    return owner[idx[np.arange(len(tris)), score.argmin(axis=1)]]


def step_faces(shape: Any) -> tuple[list[Any], np.ndarray, np.ndarray]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    from ..labels import FaceInfo, _surface_params

    wrapped = BRepBuilderAPI_Copy(getattr(shape, "wrapped", shape)).Shape()
    linear, angular = LABEL_DEFLECTION
    BRepMesh_IncrementalMesh(wrapped, linear, False, angular, True)
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_FACE, face_map)
    faces, blocks, owners = [], [], []
    for fi in range(face_map.Extent()):
        face = TopoDS.Face_s(face_map.FindKey(fi + 1))
        name, params, reversed_ = _surface_params(face, BRepAdaptor_Surface(face))
        faces.append(FaceInfo(fi, name, params, reversed_))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.array(
            [
                (q.X(), q.Y(), q.Z())
                for q in (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes() + 1))
            ],
            dtype=np.float64,
        )
        idx = np.array(
            [
                [tri.Triangle(j).Value(k) - 1 for k in (1, 2, 3)]
                for j in range(1, tri.NbTriangles() + 1)
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        blocks.append(nodes[idx])
        owners.append(np.full(len(idx), fi, dtype=np.int64))
    if not blocks:
        raise RuntimeError("STEP shape has no triangulated faces")
    return faces, np.concatenate(blocks), np.concatenate(owners)


def step_ir(step_path: str | Path, input_tris: np.ndarray) -> Ir:
    from ..oracle import _residual, _surface
    from .score import load_step_shape

    shape, error = load_step_shape(step_path)
    if error is not None:
        raise RuntimeError(error)
    faces, step_tris, step_face = step_faces(shape)
    tris = np.asarray(input_tris, dtype=np.float64).reshape(-1, 3, 3)
    face_of = assign_faces(tris, step_tris, step_face)
    regions = []
    for i, face in enumerate(faces):
        members = np.flatnonzero(face_of == face.id)
        try:
            surface = _surface(face)
            residual = _residual(face, tris[members]) if len(members) else Residual(0.0, 0.0)
        except NotImplementedError:
            points = tris[members].reshape(-1, 3)
            surface = Facets(
                [tuple(p) for p in points.tolist()],
                [(3 * j, 3 * j + 1, 3 * j + 2) for j in range(len(members))],
            )
            residual = None
        regions.append(Region(i, surface, members.tolist(), residual))
    vertices = len(np.unique(tris.reshape(-1, 3), axis=0))
    return Ir(
        Tolerances(linear=1e-6),
        Source(len(tris), vertices),
        [Shell(True, "outer", None, list(range(len(regions))))],
        regions,
        [],
        [],
    )


def _finish(
    stl_path: Path, step_path: Path, payload: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    import unmesh

    if not step_path.is_file() or step_path.stat().st_size == 0:
        raise RuntimeError(f"no STEP written: {payload.get('tool_result')}")
    ir = step_ir(step_path, unmesh.read_stl(stl_path))
    payload["write"] = {"valid": None, "fallback": None}
    return ir.dumps(), str(step_path), json.dumps(payload)


def convert_freecad(stl_path: Path) -> tuple[str | None, str | None, str | None]:
    exe = freecad_command()
    stl_path = Path(stl_path)
    step_path = stl_path.with_name(stl_path.stem + ".freecad.step")
    env = {
        **os.environ,
        "UNMESH_FC_IN": str(stl_path),
        "UNMESH_FC_OUT": str(step_path),
        "UNMESH_FC_TOLERANCE": str(FREECAD_MESH_TOLERANCE),
    }
    code, out, err = _run([exe, str(FREECAD_SCRIPT)], env)
    result = _result_line(out)
    if code != 0 or result is None or not result.get("ok"):
        raise RuntimeError(f"freecad-refine failed (exit {code}): {(result or err[-400:])}")
    payload = {
        "max_deviation": None,
        "warnings": [],
        "tool": {"name": "freecad", "version": result.get("version")},
        "tool_result": result,
    }
    return _finish(stl_path, step_path, payload)


def convert_stl2step(stl_path: Path) -> tuple[str | None, str | None, str | None]:
    exe = stl2step_command()
    stl_path = Path(stl_path)
    step_path = stl_path.with_name(stl_path.stem + ".stl2step.step")
    _, version, _ = _run([exe, "--version"], timeout=min(VERSION_TIMEOUT_S, _timeout()))
    cmd = [exe, str(stl_path), "-o", str(step_path), "--engine", "trueform", "--quiet"]
    code, out, err = _run([*cmd, "--threads", "1"])
    result = _result_line(out)
    if code not in (0, 2) or result is None or not result.get("ok"):
        raise RuntimeError(f"stl2step failed (exit {code}): {result or err[-400:]}")
    payload = {
        "max_deviation": result.get("smoothMaxDevMM"),
        "warnings": list(result.get("warnings") or []),
        "tool": {"name": "stl2step", "version": version.strip()},
        "tool_result": result,
    }
    return _finish(stl_path, step_path, payload)


convert_freecad.require = freecad_command
convert_stl2step.require = stl2step_command
