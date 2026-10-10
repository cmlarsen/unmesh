"""Measure the faceted-fallback rate of the analytic writer under degradation.

Converts each requested family part and degradation cell to STEP and reports the
share of cells whose write fell back to a faceted solid (``WriteReport.fallback``).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from unmesh_harness import degrade
from unmesh_harness.groundtruth import generate
from unmesh_harness.labels import tessellate


def part_seeds(family: str, count: int | None) -> list[int]:
    manifest = Path(__file__).resolve().parents[2] / "corpus" / "v0.json"
    entries = json.loads(manifest.read_text())["entries"]
    seeds = sorted(e["seed"] for e in entries if e["family"] == family)
    return seeds if count is None else seeds[:count]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", default="straight_fillet")
    p.add_argument("--kind", default="noise_normal")
    p.add_argument("--severity", type=float, default=0.02)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--parts", type=int, default=None)
    p.add_argument("--deflection", default="0.01,0.2")
    args = p.parse_args(argv)

    from unmesh.pipeline import convert_to_step

    families = tuple(args.families.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    lin, ang = (float(x) for x in args.deflection.split(","))
    total = fallback = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "part.step"
        for family in families:
            for part in part_seeds(family, args.parts):
                mesh = tessellate(generate(family, part).solid, lin, ang)
                for seed in seeds:
                    cell = degrade.chain(mesh, [(args.kind, args.severity)], seed)
                    try:
                        conversion = convert_to_step(cell.tris, out, measure=False)
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        print(f"{family}-{part:04d} seed={seed} FAILED {exc}", flush=True)
                        continue
                    total += 1
                    fb = conversion.write.fallback == "faceted"
                    fallback += int(fb)
                    print(
                        f"{family}-{part:04d} seed={seed} "
                        f"fallback={conversion.write.fallback} "
                        f"valid={conversion.write.valid} outcome={conversion.outcome}",
                        flush=True,
                    )
    rate = fallback / total if total else float("nan")
    print(f"\nfaceted fallback: {fallback}/{total} = {rate:.1%} ({failed} failed)")
    return 0 if rate < 0.10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
