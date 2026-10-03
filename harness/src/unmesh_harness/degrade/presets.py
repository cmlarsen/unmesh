from __future__ import annotations

from dataclasses import dataclass

from .core import OPERATORS

Step = tuple[str, float]


@dataclass(frozen=True)
class Preset:
    name: str
    steps: tuple[Step, ...]
    rationale: str


PRESETS: dict[str, Preset] = {}


def _register(name: str, steps: list[Step], rationale: str) -> None:
    for op, severity in steps:
        if op not in OPERATORS:
            raise KeyError(f"preset {name!r} uses unknown operator {op!r}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"preset {name!r} step {op!r} severity {severity} not in [0, 1]")
    if name in PRESETS:
        raise ValueError(f"preset {name!r} registered twice")
    PRESETS[name] = Preset(name, tuple(steps), rationale)


_register(
    "fusion-export",
    [("retriangulate", 1.0), ("float32", 1.0)],
    "Imitates a clean high-quality CAD export saved through STL. STL stores "
    "coordinates as single-precision floats, hence the float32 rounding. "
    "retriangulate keeps every face boundary bit-exact while replacing the "
    "interior triangulation, standing in for a different tessellator's planar "
    "triangulation of the same faces.",
)

_register(
    "tinkercad-export",
    [("coarsen", 0.5), ("truncated_digits", 1.0)],
    "Imitates a coarse browser-based export. coarsen stands in for the visibly "
    "low curve resolution of such meshes (cylinders degrade toward N-gon "
    "prisms) while keeping the mesh a closed manifold. truncated_digits to 3 "
    "significant digits stands in for ASCII STL text with truncated decimals.",
)

_register(
    "meshmixer-edit",
    [("retriangulate", 0.5), ("noise_normal", 0.2)],
    "Imitates a remeshed-then-smoothed sculpting edit. retriangulate rebuilds "
    "the triangulation from the same face boundaries, as a remesher does. "
    "Light normal noise (10 um bound) stands in for the small surface "
    "displacement smoothing leaves behind.",
)

_register(
    "slicer-repair",
    [
        ("float32", 1.0),
        ("unwelded_corners", 0.5),
        ("flipped_facets", 0.2),
        ("duplicate_facets", 0.2),
    ],
    "Imitates input headed for slicer-style repair, exercising the converter "
    "REPAIR path. Single-precision coordinates plus defects the converter is "
    "documented to repair: split vertices within the weld, inconsistent "
    "winding fixed by flood fill, and duplicate facets dropped.",
)

_register(
    "inch-roundtrip",
    [("inch_round_trip", 1.0), ("float32", 1.0)],
    "Imitates a file passed through inch-unit software: coordinates quantized "
    "to a 0.001-inch (25.4 um) grid and converted back, then written as "
    "single-precision STL.",
)
