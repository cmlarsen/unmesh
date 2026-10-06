from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

BASE_MB = 300.0
MB_PER_FILE_MB = 26.0
POLL_S = 0.05


@dataclass
class Solid:
    volume: float
    valid: bool
    tolerance: float
    shells: int


@dataclass
class Outcome:
    solids: list[Solid] = field(default_factory=list)
    shells: int = 0
    peak_rss_mb: float = 0.0
    error: str | None = None
    limit: str | None = None


def estimate_mb(size_bytes: int) -> float:
    return BASE_MB + MB_PER_FILE_MB * size_bytes / 2**20


def run(path, precise: bool, face_orientation: bool, memory_mb: float, timeout_s: float) -> Outcome:
    if not sys.executable:
        return Outcome(error="no Python executable to run the OCCT read-back in")
    with tempfile.TemporaryDirectory(prefix="unmesh-readback-") as tmp:
        out = os.path.join(tmp, "result.json")
        err = os.path.join(tmp, "stderr.txt")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        args = [
            sys.executable,
            "-c",
            "from unmesh._writer.readback import main; main()",
            os.fspath(path),
            "1" if precise else "0",
            "1" if face_orientation else "0",
            out,
        ]
        with open(err, "wb") as stderr:
            proc = subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr, env=env
            )
            outcome = _watch(proc, memory_mb, timeout_s)
        if outcome.limit is not None:
            return outcome
        if proc.returncode != 0 or not os.path.exists(out):
            with open(err, "rb") as fh:
                tail = fh.read()[-400:].decode("utf-8", "replace").strip().splitlines()
            detail = f": {tail[-1]}" if tail else ""
            outcome.error = f"the read-back process exited with code {proc.returncode}{detail}"
            return outcome
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
    if "error" in data:
        outcome.error = data["error"]
        return outcome
    outcome.solids = [Solid(**s) for s in data["solids"]]
    outcome.shells = data["shells"]
    return outcome


def _watch(proc, memory_mb: float, timeout_s: float) -> Outcome:
    outcome = Outcome()
    start = time.monotonic()
    while proc.poll() is None:
        rss = _rss_mb(proc.pid)
        if rss is not None:
            outcome.peak_rss_mb = max(outcome.peak_rss_mb, rss)
            if rss > memory_mb:
                outcome.limit = f"its memory reached {rss:.0f} MB, over the {memory_mb:.0f} MB cap"
        if outcome.limit is None and time.monotonic() - start > timeout_s:
            outcome.limit = f"it ran longer than the {timeout_s:g} s timeout"
        if outcome.limit is not None:
            proc.kill()
            proc.wait()
            return outcome
        time.sleep(POLL_S)
    return outcome


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / 2**20
    except ImportError:
        pass
    except Exception:
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.split()
        return int(out[0]) / 1024 if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> None:
    path, precise, face_orientation, out = sys.argv[1:5]
    try:
        from OCP.Interface import Interface_Static
        from OCP.STEPControl import STEPControl_Reader

        from . import occ

        STEPControl_Reader()
        if face_orientation == "0":
            Interface_Static.SetCVal_s("read.step.resource.name", "")
            Interface_Static.SetIVal_s("FromSTEP.FixShape.FixFaceOrientationMode", 0)
        solids, shells = occ.read_back(path, precise == "1")
        data = {"solids": [vars(s) for s in solids], "shells": shells}
    except Exception as e:
        data = {"error": f"could not re-import the file: {type(e).__name__}: {e}"}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
