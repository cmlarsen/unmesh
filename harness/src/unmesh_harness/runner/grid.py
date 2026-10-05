from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..corpus import find_manifest, load_manifest, select
from ..degrade.presets import PRESETS
from .converters import plugin_hash

Key = tuple[str, str, float, int, str, str, str, str]

FACE_LABEL_SENSITIVE_OPS = frozenset(
    {
        "coarsen",
        "crack_seam",
        "fillet_rows",
        "hole_patch",
        "noise_off_plane",
        "nonuniform_chords",
        "refine",
        "retriangulate",
        "slivers",
        "t_junctions",
    }
)


def ambiguity_rejected_ops(cells: list[dict[str, Any]]) -> list[str]:
    rejected = []
    for spec in cells:
        for name, _ in steps_for(spec):
            if name in FACE_LABEL_SENSITIVE_OPS and name not in rejected:
                rejected.append(name)
    return rejected


@dataclass(frozen=True)
class Cell:
    part: str
    operator: str
    severity: float
    seed: int
    converter: str
    git_sha: str
    grid_hash: str
    plugin_hash: str = ""

    @property
    def key(self) -> Key:
        return (
            self.part,
            self.operator,
            self.severity,
            self.seed,
            self.converter,
            self.git_sha,
            self.grid_hash,
            self.plugin_hash,
        )


@dataclass
class Grid:
    name: str
    corpus_grid: str
    seeds: list[int]
    timeout_s: float
    input_deflection: tuple[float, float]
    truth_deflection: tuple[float, float]
    cells: list[dict[str, Any]]
    judge_samples_per_mm2: float
    grid_hash: str
    step_deviation_sample: tuple[int, int]
    entries: list[dict[str, Any]]

    @property
    def need_truth(self) -> bool:
        return any(c.get("judge_truth") for c in self.cells)

    def row(self, operator: str, severity: float) -> dict[str, Any]:
        for spec in self.cells:
            if spec["operator"] == operator and float(spec["severity"]) == float(severity):
                return spec
        raise KeyError((operator, severity))

    def checks_step(self, cell: Cell) -> bool:
        take, of = self.step_deviation_sample
        label = f"{cell.part}|{cell.operator}|{cell.severity}|{cell.seed}|{cell.git_sha}"
        return int(hashlib.sha256(label.encode()).hexdigest(), 16) % of < take

    def expand(self, converters: list[str], git_sha: str) -> list[tuple[Cell, dict[str, Any]]]:
        out = []
        for converter in converters:
            for entry in self.entries:
                for spec in self.cells:
                    for seed in spec.get("seeds", self.seeds):
                        cell = Cell(
                            entry["id"],
                            spec["operator"],
                            float(spec["severity"]),
                            seed,
                            converter,
                            git_sha,
                            self.grid_hash,
                            plugin_hash(converter),
                        )
                        out.append((cell, spec))
        return out


def steps_for(spec: dict[str, Any]) -> list[list[Any]]:
    if "preset" not in spec:
        return spec["steps"]
    if "steps" in spec:
        raise ValueError("a grid cell takes either 'preset' or 'steps', not both")
    name = spec["preset"]
    try:
        preset = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown degradation preset {name!r}") from None
    return [[op, severity] for op, severity in preset.steps]


def repo_root() -> Path:
    return find_manifest().parent.parent


def find_grid(name: str) -> Path:
    path = repo_root() / "harness" / "grids" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no grid definition at {path}")
    return path


def load_grid(name: str, manifest_path: Path | None = None) -> Grid:
    raw = json.loads(find_grid(name).read_text())
    for spec in raw["cells"]:
        steps_for(spec)
        if "preset" in spec:
            if "operator" not in spec:
                spec["operator"] = spec["preset"]
            elif spec["operator"] != spec["preset"]:
                raise ValueError(
                    f"grid cell preset {spec['preset']!r} disagrees with operator "
                    f"{spec['operator']!r}: preset cells must be labeled with their "
                    "preset name"
                )
    if "ambiguity" in raw.get("categories", []):
        rejected = ambiguity_rejected_ops(raw["cells"])
        if rejected:
            raise ValueError(
                f"grid {name} selects ambiguity parts but rows {rejected} read face "
                "labels: pair members share triangles but carry different truth faces, "
                "so those rows diverge by design"
            )
    entries = select(load_manifest(manifest_path), raw["corpus_grid"])
    if "categories" in raw:
        wanted = set(raw["categories"])
        entries = [e for e in entries if e["strata"].get("category", "planar") in wanted]
        if not entries:
            raise KeyError(f"grid {name} selects no parts for categories {sorted(wanted)}")
    if raw.get("indistinguishable_only"):
        from ..groundtruth import generate

        entries = [
            e
            for e in entries
            if generate(e["family"], e["seed"]).parameters.get("indistinguishable") is True
        ]
        if not entries:
            raise KeyError(f"grid {name} selects no indistinguishable parts")
    return Grid(
        raw["name"],
        raw["corpus_grid"],
        [int(s) for s in raw["seeds"]],
        float(raw["timeout_s"]),
        tuple(raw["input_deflection"]),
        tuple(raw["truth_deflection"]),
        raw["cells"],
        float(raw.get("judge_samples_per_mm2", 10.0)),
        hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12],
        tuple(raw.get("step_deviation_sample", (1, 1))),
        entries,
    )


def git_sha() -> str:
    root = repo_root()

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    try:
        sha = git("rev-parse", "--short=12", "HEAD").decode().strip()
        diff = git("diff", "HEAD", "--binary")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    digest = hashlib.sha256(diff)
    for name in sorted(n for n in untracked if n):
        digest.update(name)
        try:
            digest.update((root / name.decode()).read_bytes())
        except OSError:
            pass
    if not diff and not any(untracked):
        return sha
    return f"{sha}+{digest.hexdigest()[:12]}"


def prep_hash(grid: Grid, manifest_path: Path | None = None) -> str:
    base = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256((manifest_path or find_manifest()).read_bytes())
    digest.update(
        json.dumps([grid.input_deflection, grid.truth_deflection, grid.need_truth]).encode()
    )
    files = (
        sorted((base / "groundtruth").glob("*.py"))
        + [base / "labels.py", base / "corpus.py", base / "imported.py", base / "strata.py"]
        + [base / "runner" / "execute.py"]
        + [base / "degrade" / name for name in ("__init__.py", "fillets.py", "retriangulate.py")]
    )
    for f in files:
        digest.update(f.read_bytes())
    return digest.hexdigest()[:16]
