# AGENTS.md

Conventions for anyone, human or agent, working in this repo. Architecture and rationale live in
[docs/plan.md](docs/plan.md).

## Setup

```sh
uv sync
scripts/check.sh
```

`uv sync` builds the `unmesh` wheel with maturin (needs a Rust toolchain) and installs the
`unmesh-harness` workspace member. `.cargo/config.toml` points PyO3 at `.venv/bin/python`, so run
`uv sync` before any `cargo` command, including `cargo test -p unmesh-core` (the workspace builds `unmesh-py` too). Setting `UV_PROJECT_ENVIRONMENT` to relocate the venv breaks that path.

## Gates

`scripts/check.sh` runs every gate and exits non-zero on the first failure:

- Rust: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
- Python: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`.

`uv run` rebuilds the extension itself when Rust sources change (via `cache-keys`).

## Layout

- `crates/unmesh-core`: Rust geometry core, no kernel dependency.
- `crates/unmesh-py`: PyO3 bindings, built as `unmesh._core`.
- `python/unmesh`: the public Python package (depends on numpy only).
- `harness/`: the separate `unmesh-harness` package. GPL tools may appear only here, as optional
  dependencies.
- `scripts/`: `check.sh` and other tooling.

## Conventions

- One issue is one worktree is one PR. Claim an issue before starting, by commenting and adding the
  `in-progress` label.
- A reconstruction PR posts the `smoke` grid diff against `main` and links the latest nightly
  `standard` diff.
- A PR never waives its own regression, and implementation agents never see hold-out or hidden-seed
  results.
- Acceptance criteria are numeric. An issue that lacks a number gets one from a human before work
  starts.
- Do not add code comments unless a non-obvious invariant needs one. Never leave commented-out code.
- `unmesh[step]` (OCP) stays optional: nothing in the core import path may import `OCP`.
