from . import (
    chords,
    coarsen,
    defects,
    faults,
    fillets,
    noise,
    pose,
    precision,
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

__all__ = [
    "OPERATORS",
    "Operator",
    "apply",
    "apply_chain",
    "chain",
    "apply_pair_preprocess",
    "chords",
    "coarsen",
    "defects",
    "faults",
    "fillets",
    "noise",
    "pose",
    "precision",
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
    for i, (op, args) in enumerate(steps):
        args = dict(args)
        step_seed = int(seed) + i
        if op == "fillet_rows":
            params = fillets.fillet_rows(
                out, 1.0, np.random.default_rng(step_seed), segments=args.get("segments")
            )
        elif op == "canonical_planar":
            params = retriangulate.canonical_planar(out)
            step_seed = None
        else:
            raise ValueError(f"unknown pair preprocessing step {op!r}")
        out.metadata.setdefault("history", []).append(
            {"op": op, "severity": 1.0, "seed": step_seed, "params": params}
        )
    return out
