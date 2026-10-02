#!/usr/bin/env bash
set -euo pipefail

export UV_LOCKED=1

cd "$(dirname "$0")/.."

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
uv run ruff check
uv run ruff format --check
uv run pytest -m "not benchmark"
