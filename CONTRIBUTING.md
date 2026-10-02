# Contributing

## Run the checks

```sh
uv sync
scripts/check.sh
```

This needs `uv` and a Rust toolchain (`rustup` picks up `rust-toolchain.toml`). It runs the Rust and
Python gates listed in [AGENTS.md](AGENTS.md). Every PR runs the same script in CI on macOS and
Linux.

## The harness

The evaluation harness is the separate `unmesh-harness` package in `harness/`, installed by
`uv sync`. It generates parts, tessellates and degrades them, converts them with unmesh, and scores
the result. Its commands arrive with the harness issues; until then it holds a smoke test.

## Reading the grid

The harness reports a grid: rows are corpus families and degradations, columns are metrics, and each
cell is the mean over 3 seeds. A cell regresses when it moves by more than the larger of 2 sigma
across seeds and a per-metric floor. The grids you can run locally (`smoke`, `standard`, `full`) use
the frozen public manifest, so you can reproduce any number a PR quotes.

## Hold-out and hidden seeds

Two sets are not available to contributors:

- The **hidden seed set** runs nightly on a maintainer-owned runner and reports aggregates only.
- The **hold-out set** of real STLs runs nightly with results hidden from implementation work.

You do not need either to contribute. If your change passes `scripts/check.sh` and does not regress
the public `smoke` grid, a maintainer will see the nightly aggregates and tell you if something
regressed there. Do not tune against them, and do not waive a regression in your own PR; only a
maintainer applies the `waiver:approved` label.

## Licensing

unmesh is Apache-2.0. GPL tools such as pymeshlab may only be optional dependencies of
`unmesh-harness`, never of `unmesh`.
