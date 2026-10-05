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
    "Approximation of a clean high-quality CAD export saved through STL. STL "
    "stores coordinates as single-precision floats, binary STL included, "
    "hence the float32 rounding. retriangulate keeps every face boundary "
    "bit-exact while replacing the interior triangulation, standing in for a "
    "different tessellator's planar triangulation of the same faces. "
    "Curvature-dependent mesh density is not modelled.",
)

_register(
    "tinkercad-export",
    [("coarsen", 0.5), ("truncated_digits", 0.5)],
    "Approximation of a coarse, low-precision export: coarsen lowers curve "
    "resolution while keeping the mesh a closed manifold, and truncated_digits "
    "to 6 significant digits stands in for low-precision decimal text. "
    "Format (ASCII vs binary) unverified until #29.",
)

_register(
    "meshmixer-edit",
    [("retriangulate", 0.5), ("noise_normal", 0.2)],
    "A loose approximation: retriangulated interior plus small normal "
    "displacement; does not model edge rounding or isotropic remeshing. "
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
    "Approximate model of a slicer repair pass over single-precision input: "
    "float32 rounding first, then split vertices, flipped winding, and "
    "duplicate facets to exercise the converter REPAIR path. Approximate "
    "only: flipped_facets REPAIR is planar-only, duplicate facets depress "
    "the area score even when dropped, and unwelded_corners is unrecoverable "
    "on its own — it only repairs after float32 snaps split vertices onto a "
    "shared grid, so float32 must run first.",
)

_register(
    "inch-roundtrip",
    [("inch_round_trip", 1.0), ("float32", 1.0)],
    "Imitates a file passed through inch-unit software: coordinates quantized "
    "to a 0.001-inch (25.4 um) grid and converted back, then written as "
    "single-precision STL.",
)
