from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..corpus import find_manifest, load_manifest, select

Key = tuple[str, str, float, int, str, str]


@dataclass(frozen=True)
class Cell:
    part: str
    operator: str
    severity: float
    seed: int
    converter: str
    git_sha: str

    @property
    def key(self) -> Key:
        return (self.part, self.operator, self.severity, self.seed, self.converter, self.git_sha)


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
    judge_truth: bool
    entries: list[dict[str, Any]]

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
                        )
                        out.append((cell, spec))
        return out


def repo_root() -> Path:
    return find_manifest().parent.parent


def find_grid(name: str) -> Path:
    path = repo_root() / "harness" / "grids" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no grid definition at {path}")
    return path


def load_grid(name: str, manifest_path: Path | None = None) -> Grid:
    raw = json.loads(find_grid(name).read_text())
    entries = select(load_manifest(manifest_path), raw["corpus_grid"])
    return Grid(
        raw["name"],
        raw["corpus_grid"],
        [int(s) for s in raw["seeds"]],
        float(raw["timeout_s"]),
        tuple(raw["input_deflection"]),
        tuple(raw["truth_deflection"]),
        raw["cells"],
        float(raw.get("judge_samples_per_mm2", 10.0)),
        bool(raw.get("judge_truth", True)),
        entries,
    )


def git_sha() -> str:
    root = repo_root()
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}+dirty" if dirty else sha
