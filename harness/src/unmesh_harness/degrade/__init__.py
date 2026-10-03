from . import (
    chords,
    coarsen,
    defects,
    faults,
    fillets,
    noise,
    pose,
    precision,
    presets,
    refine,
    retriangulate,
)
from .core import (
    OPERATORS,
    Operator,
    apply,
    apply_chain,
    chain,
    register,
    to_original,
    to_original_points,
)
from .presets import PRESETS, Preset

__all__ = [
    "OPERATORS",
    "Operator",
    "PRESETS",
    "Preset",
    "apply",
    "apply_chain",
    "chain",
    "chords",
    "coarsen",
    "defects",
    "faults",
    "fillets",
    "noise",
    "pose",
    "precision",
    "presets",
    "refine",
    "register",
    "retriangulate",
    "to_original",
    "to_original_points",
]


def apply_pair_preprocess(mesh, steps, seed=0):
    import copy

    import numpy as np

    out = copy.deepcopy(mesh)
    rng = np.random.default_rng(seed)
    for op, args in steps:
        args = dict(args)
        if op == "fillet_rows":
            fillets.fillet_rows(out, 1.0, rng, segments=args.get("segments"))
        elif op == "canonical_planar":
            retriangulate.canonical_planar(out)
        else:
            raise ValueError(f"unknown pair preprocessing step {op!r}")
    return out
