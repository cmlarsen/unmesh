"""Snapshot canonical IR JSON for every smoke+standard corpus part.

Tessellates each corpus part with the harness at the smoke grid's input
deflection (0.01, 0.2 -- the middle DEFLECTION_SETTINGS entry), converts
with default ConvertOptions, and writes canonical IR JSON per part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--grid", default="all", choices=["all", "smoke", "standard"])
    ap.add_argument("--lin", type=float, default=0.01)
    ap.add_argument("--ang", type=float, default=0.2)
    args = ap.parse_args()

    import unmesh
    from unmesh_harness.corpus import load_manifest
    from unmesh_harness.groundtruth import generate
    from unmesh_harness.labels import tessellate

    manifest = load_manifest()
    if args.grid == "all":
        entries = [
            e for e in manifest["entries"] if "smoke" in e["grids"] or "standard" in e["grids"]
        ]
    else:
        entries = [e for e in manifest["entries"] if args.grid in e["grids"]]
    entries.sort(key=lambda e: e["id"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"lin": args.lin, "ang": args.ang, "parts": []}
    for e in entries:
        gt = generate(e["family"], e["seed"])
        mesh = tessellate(gt.solid, args.lin, args.ang)
        result = unmesh.convert(mesh.tris)
        text = result.ir.dumps()
        (out / f"{e['id']}.json").write_text(text + "\n")
        summary["parts"].append(e["id"])
        print(f"{e['id']}: {len(mesh.tris)} tris -> {len(result.ir.regions)} regions", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(entries)} parts to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
