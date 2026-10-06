from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time

USAGE_ERROR = 3


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(USAGE_ERROR, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    from unmesh import __version__

    parser = _Parser(
        prog="unmesh",
        description="Convert a triangle mesh exported from CAD into a STEP solid.",
    )
    parser.add_argument("--version", action="version", version=f"unmesh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    convert = sub.add_parser(
        "convert",
        help="convert an STL or OBJ mesh to STEP",
        description=(
            "Exit codes: 0 every face analytic; 1 written with facets regions (mixed); "
            "2 written as a whole-part faceted solid; 3 error (nothing usable written)."
        ),
    )
    convert.add_argument("input", help="input mesh (.stl or .obj)")
    convert.add_argument("output", help="output STEP file")
    convert.add_argument(
        "--unit",
        choices=("mm", "in"),
        default="mm",
        help="unit of the input coordinates; the STEP is written in mm (default: mm)",
    )
    convert.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="linear tolerance in input units (default: derived from the mesh's noise)",
    )
    convert.add_argument("--report", help="write the JSON fidelity report to this path")
    convert.add_argument(
        "--no-measure",
        action="store_true",
        help="skip sampling the written STEP against the input mesh",
    )
    return parser


def _write_report(path: str | None, report: dict) -> None:
    if path is None:
        return
    from unmesh.pipeline import dumps

    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(report))


def _summary(conversion) -> str:
    f = conversion.fidelity
    dev = f["deviation"]
    measured = dev["measured"]
    lines = [
        f"{conversion.outcome}: {f['output']['path']}",
        "  regions: " + ", ".join(f"{k} {v}" for k, v in sorted(f["region_counts"].items())),
        f"  converter deviation bound: {dev['converter']['max']:.3g} mm",
    ]
    if measured is not None and measured["max"] is not None:
        lines.append(f"  measured STEP-to-input deviation: {measured['max']:.3g} mm")
    if f["faceted"]:
        lines.append(f"  faceted regions: {len(f['faceted'])}")
    if conversion.write.fallback_reason:
        lines.append(f"  fallback: {conversion.write.fallback_reason}")
    if f["error"]:
        lines.append(f"  error: {f['error']['message']}")
    return "\n".join(lines)


def _partial(output: str) -> str:
    folder = os.path.dirname(os.path.abspath(output))
    fd, path = tempfile.mkstemp(prefix=".unmesh-", suffix=".step", dir=folder)
    os.close(fd)
    os.unlink(path)
    return path


def _remove(path: str | None) -> None:
    if path is not None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _convert(args) -> int:
    from unmesh import ConvertOptions
    from unmesh.pipeline import EXIT_CODES, UNITS, convert_to_step, error_report

    started = time.perf_counter()
    source = {"path": args.input, "unit": args.unit, "scale_to_mm": UNITS[args.unit]}
    partial = None
    try:
        if args.tolerance is not None and not args.tolerance > 0:
            raise ValueError(f"--tolerance must be positive, got {args.tolerance}")
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"no such input file: {args.input}")
        partial = _partial(args.output)
        conversion = convert_to_step(
            args.input,
            partial,
            ConvertOptions(linear_tolerance=args.tolerance),
            unit=args.unit,
            measure=not args.no_measure,
        )
        output = conversion.fidelity["output"]
        output["path"] = args.output
        if conversion.outcome == "error":
            _remove(partial)
            output["written"] = False
        _write_report(args.report, conversion.fidelity)
        if conversion.outcome != "error":
            os.replace(partial, args.output)
    except Exception as e:
        _remove(partial)
        print(f"unmesh: error: {type(e).__name__}: {e}", file=sys.stderr)
        runtime = {"total": time.perf_counter() - started}
        report = error_report(e, source=source, output=args.output, runtime=runtime)
        try:
            _write_report(args.report, report)
        except Exception as w:
            print(f"unmesh: error: could not write the report: {w}", file=sys.stderr)
        return EXIT_CODES["error"]
    stream = sys.stderr if conversion.outcome == "error" else sys.stdout
    print(_summary(conversion), file=stream)
    return conversion.exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "convert":
        return _convert(args)
    return USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
