from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import (
    GRID_ORDER,
    build_grid,
    default_cache_dir,
    default_manifest,
    find_manifest,
    load_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unmesh-harness")
    sub = parser.add_subparsers(dest="group", required=True)
    corpus = sub.add_parser("corpus").add_subparsers(dest="action", required=True)

    build = corpus.add_parser("build", help="write STEP + metadata JSON for a grid")
    build.add_argument("--grid", choices=GRID_ORDER, default="smoke")
    build.add_argument("--out", type=Path, default=None)
    build.add_argument("--manifest", type=Path, default=None)

    gen = corpus.add_parser("manifest", help="print the canonical generated manifest")
    gen.add_argument("--write", action="store_true", help="overwrite corpus/v0.json")

    args = parser.parse_args(argv)
    if args.action == "build":
        manifest = load_manifest(args.manifest)
        out = (args.out or default_cache_dir()) / args.grid
        count = build_grid(manifest, args.grid, out)
        print(f"built {count} entries into {out}")
        return 0
    text = json.dumps(default_manifest(), indent=2) + "\n"
    if args.write:
        find_manifest().write_text(text)
    else:
        print(text, end="")
    return 0
