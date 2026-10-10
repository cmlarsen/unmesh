"""Measure the converter's noise estimate against the degradation's true noise.

For each requested family part, tessellate its ground truth, degrade it with a
noise operator, convert, and compare ``ir.tolerances.linear / 5`` (the noise the
converter applied) with the standard deviation of the displacement along the
surface normal:

* ``noise_normal`` moves every vertex along its area-weighted normal by a scalar
  uniform on ``[-A, A]``, so the component along the normal has sd ``A / sqrt(3)``.
* ``noise_isotropic`` moves every vertex by a vector uniform in a ball of radius
  ``A``; its component along any fixed direction has sd ``A / sqrt(5)``.

``A = severity * 0.05 mm``.  Because ``linear = min(initial, max(5 sigma, floor))``,
``linear / 5`` equals the applied sigma except at the floor, where it is an upper
bound.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import unmesh
from unmesh_harness import degrade

MAX_AMPLITUDE_MM = 0.05
DEFAULT_FAMILIES = (
    "straight_fillet",
    "circular_fillet",
    "corner_fillet",
    "bore_chamfer",
    "through_bore",
    "round_boss",
)
KINDS = ("noise_normal", "noise_isotropic")
SEVERITIES = (0.02, 0.1)
SEEDS = (0, 1, 2)


def true_sigma(kind: str, severity: float) -> float:
    amp = severity * MAX_AMPLITUDE_MM
    return amp / np.sqrt(3.0 if kind == "noise_normal" else 5.0)


def part_seeds(family: str, count: int | None) -> list[int]:
    import json as _json
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[2] / "corpus" / "v0.json"
    if not manifest.exists():
        manifest = Path("corpus/v0.json")
    entries = _json.loads(manifest.read_text())["entries"]
    seeds = sorted(e["seed"] for e in entries if e["family"] == family)
    return seeds if count is None else seeds[:count]


def measure_part(job: tuple) -> list[dict]:
    family, part, kinds, severities, seeds, lin, ang = job
    from unmesh_harness.groundtruth import generate
    from unmesh_harness.labels import tessellate

    mesh = tessellate(generate(family, part).solid, lin, ang)
    rows = []
    for kind in kinds:
        for sev in severities:
            truth = true_sigma(kind, sev)
            for seed in seeds:
                deg = degrade.chain(mesh, [(kind, sev)], seed)
                ir, _ = unmesh.convert(np.asarray(deg.tris))
                est = ir.tolerances.linear / 5.0
                rows.append(
                    {
                        "family": family,
                        "part": part,
                        "kind": kind,
                        "severity": sev,
                        "seed": seed,
                        "sigma_est": est,
                        "sigma_true": truth,
                        "ratio": est / truth,
                        "regions": len(ir.regions),
                    }
                )
    return rows


def run(args):
    jobs = [
        (family, part, args.kinds, args.severities, args.seeds, *args.deflection)
        for family in args.families
        for part in part_seeds(family, args.parts)
    ]
    rows: list[dict] = []
    if args.jobs > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            for part_rows in pool.map(measure_part, jobs):
                rows.extend(part_rows)
    else:
        for job in jobs:
            rows.extend(measure_part(job))
    rows.sort(key=lambda r: (r["family"], r["part"], r["kind"], r["severity"], r["seed"]))
    for r in rows:
        print(
            f"{r['family']}-{r['part']:04d} {r['kind']:16s} sev={r['severity']:<4} "
            f"seed={r['seed']} sigma_est={r['sigma_est']:.3e} sigma_true={r['sigma_true']:.3e} "
            f"ratio={r['ratio']:.3f} regions={r['regions']}",
            flush=True,
        )
    return rows


def summarize(rows) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["kind"], r["severity"])].append(r["ratio"])
    print("\nsummary (est/true)")
    print(f"{'family':16s} {'kind':16s} {'sev':>4s} {'n':>3s} {'min':>6s} {'med':>6s} {'max':>6s}")
    worst = 1.0
    for (family, kind, sev), ratios in sorted(groups.items()):
        arr = np.array(ratios)
        print(
            f"{family:16s} {kind:16s} {sev:>4} {len(arr):>3d} "
            f"{arr.min():>6.2f} {np.median(arr):>6.2f} {arr.max():>6.2f}"
        )
        for rat in arr:
            worst = min(worst, min(rat, 1.0 / rat) if rat > 0 else 0.0)
    print(f"\nworst (est/true within 2x, min folded ratio): {worst:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--severities", default=",".join(str(s) for s in SEVERITIES))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--parts", type=int, default=None, help="keep only the first N seeds per family")
    p.add_argument(
        "--deflection", default="0.01,0.2", help="linear,angular tessellation deflection"
    )
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--out", default=None, help="write rows as JSONL")
    args = p.parse_args(argv)
    args.families = tuple(args.families.split(","))
    args.kinds = tuple(args.kinds.split(","))
    args.severities = tuple(float(s) for s in args.severities.split(","))
    args.seeds = tuple(int(s) for s in args.seeds.split(","))
    args.deflection = tuple(float(s) for s in args.deflection.split(","))

    rows = run(args)
    summarize(rows)
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0 if all(min(r["ratio"], 1.0 / r["ratio"]) >= 0.5 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
