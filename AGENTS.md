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
- `corpus/v0.json`: the corpus manifest. `uv run unmesh-harness corpus build [--grid smoke|standard] [--out DIR]`
  writes STEP + metadata JSON to `$UNMESH_CACHE_DIR/corpus` or `~/.cache/unmesh/corpus` (never committed).
  The full `standard` validity test runs with `uv run pytest harness/tests --slow`.
- `harness/src/unmesh_harness/labels.py`: `tessellate(shape, lin, ang)` gives a `LabeledMesh` (per-triangle face id, analytic face table, edge adjacency); deflection settings in `DEFLECTION_SETTINGS`.
- `harness/src/unmesh_harness/oracle.py`: `build_oracle_ir(mesh)` builds the IR straight from a `LabeledMesh` (oracle input for fitting and edge-building).
- `harness/src/unmesh_harness/judge.py`: `judge(ir, input_mesh, truth_mesh, report, samples_per_mm2, seed)`, the independent fidelity judge (Rust: `unmesh_core::judge`, parry3d BVH). Samples each region on its own source triangles projected onto the analytic surface, and measures both directions (IR to mesh, mesh to IR footprint) against the input mesh and the fine ground truth. `calibration` is the converter's reported `max_deviation` minus the measured two-sided max; negative means it under-reports. It imports nothing from the converter path.
- `harness/src/unmesh_harness/runner/`: `uv run unmesh-harness run --converter unmesh --grid smoke [--out DIR] [--jobs N] [--gate]` scores converters on a grid defined in `harness/grids/<name>.json` (parts x operator chain x severity x seeds). A grid's optional `categories` list keeps only corpus entries whose `strata.category` is listed; `smoke` uses `["planar"]` until the curved fitters land. Parts carrying `parameters.pair_deflection` are tessellated at that deflection (not the grid's `input_deflection`), run through their `pair_preprocess` steps at a fixed seed, then canonicalised (1e-9 snap, min-first triangle rotation, sorted order) so indistinguishable pair members cache bit-identical triangles. A grid's optional `indistinguishable_only` flag keeps only parts whose ground-truth parameters carry `indistinguishable: true`; `harness/grids/ambiguity.json` sets it with `categories: ["ambiguity"]` and identity + float32 rows only. Grids whose categories include `"ambiguity"` reject face-label-sensitive rows (coarsen, nonuniform_chords, refine, retriangulate, slivers and the other face-table readers): pair members share triangles but carry different truth faces, so those rows diverge by design. Cells run in a spawn pool with a per-cell timeout and append to `DIR/<grid>.jsonl`, keyed by (part, operator, severity, seed, converter, git sha); a rerun skips cells already `ok`. `unmesh-harness report FILE [--gate]` re-prints the table. A converter is `convert(stl_path) -> (ir_json | None, step_path | None, report_json | None)`, named `unmesh`, `faceted` or `module:function`. `--gate` exits 1 on any missing, failed or invalid cell of a non-baseline converter (`faceted` is a baseline, F1 only) and on any breach of the per-row `floors` declared in the grid file (F1, region count, deviation to input and truth, fallback, under-report, STEP validity and deviation). Resume keys include the git sha, a hash of the dirty tree and a hash of the grid file. CI runs it on every PR.
- `harness/src/unmesh_harness/datasets.py` and `scripts/fetch-datasets`: `uv run scripts/fetch-datasets [--dataset NAME|all] [--limit K] [--verify]` downloads pinned subsets of the datasets in `docs/datasets.md` to `$UNMESH_CACHE_DIR/datasets` or `~/.cache/unmesh/datasets` with a per-dataset `manifest.json` (sha256 and license per file). Never committed.
- Regression compare: `uv run unmesh-harness compare <main.jsonl> <pr.jsonl> [--converter unmesh]`
  deduplicates each file to its latest run (`results.latest()` semantics: keep the last
  record per key, filtered to the file's own git sha and grid hash), then aggregates each
  (part, operator, severity) cell over seeds. A cell present in A but missing in B is a
  REGRESSION; an empty B, or a B with no records for the converter, is an error (exit 2).
  B's seed set must cover A's per cell. f1 and the deviation metrics flag a
  REGRESSION/IMPROVEMENT when run B moves by more than max(2σ of run A, floor) (floors:
  F1 0.01, deviation 1 µm); a metric sampled in A but not in B is a REGRESSION.
  `valid`/`fallback`/`under_report` have no noise band: any per-seed good→bad move is a
  REGRESSION. It exits 1 on any regression. A regression is waived only by
  the `waiver:approved` label: `uv run scripts/check-waiver --repo o/r --pr N --run-a A --run-b B`
  accepts it only if the labeling actor's login is in `.github/waiver-approvers` as read from
  main, the labeling event is newer than the PR head commit's committer date, the label is
  still present, and a comment by the same actor, also newer than the head commit, names every
  regressed cell (`waive: <part> <operator> <severity>` lines); only named cells are waived.
  Events and comments are read through the paginated GitHub API. While agents share the
  owner's credentials this check is NOT agent-proof (residual risk): anything acting with an
  approver's credentials can waive. The planned identity split gives agents their own
  credentials so the allowlist and timing rules bind them.
  CI uploads each main push's smoke JSONL as the `smoke-baseline-<os>` artifact. PR runs pin
  `harness/grids/smoke.json` to main's copy, then compare the PR results against the baseline
  from the run for the PR base sha (falling back to the latest successful main run with a
  warning). Compare and check-waiver always execute from a worktree of origin/main, never
  from PR code. No main run at all skips compare with a warning; a main run whose artifact
  cannot be downloaded fails the job.
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
