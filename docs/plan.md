# unmesh plan

**Status:** revision 2, 2026-10-01, folding in the first plan review. Tracked by epic #1; every work
item is a sub-issue.

## Goal

Convert a triangle mesh exported from CAD (STL; OBJ, 3MF and PLY later) into a valid STEP solid whose
faces are real analytic surfaces (planes, cylinders, cones, spheres, tori), with a measured fidelity
report.

- **Never fails a whole part.** Regions that cannot be fitted stay faceted. If the analytic solid
  fails validation, unmesh emits the faceted solid and reports it as a fallback.
- **Never claims a fidelity it did not measure.**
- **STL to STEP only.** unmesh knows nothing about CAM. OttoCAM is one consumer and keeps its own
  manufacturing-parity gate privately.

### In scope for v0

- **Meshes exported from CAD:** Fusion, Onshape, SolidWorks, FreeCAD, OpenSCAD, Tinkercad.
- **Degradation:** clean files, and files degraded by low-poly curves, float32 rounding, vertex noise
  and light editing.
- **Surfaces:** plane, cylinder, cone, sphere, torus.
- **Chamfers and fillets are table stakes, not stretch goals.** These must ship in v0, each with its
  own acceptance gate:
  - planar chamfers
  - conical chamfers on bores
  - straight-edge fillets (cylinders)
  - circular-edge fillets (tori)
- **Multi-body STLs:** each closed shell converts independently into its own solid in a compound.

### Out of scope for v0

- Scans and sculpted models.
- Freeform NURBS fitting.
- Recovering a feature history.
- Variable-radius fillets, and setback vertex blends (stay faceted; M4 measures how they fail).

## Architecture

```
mesh ─▶ unmesh-core (Rust) ─▶ IR ─▶ writer (Python + OCP, extra `unmesh[step]`) ─▶ STEP + report
        weld, repair,          surfaces,          edges: OCCT intersection where
        segment, fit,          region adjacency,  faces cross, the boundary polyline
        snap                   boundary polylines where they're tangent; build, heal,
                               per region,        validate, faceted fallback
                               provenance
```

**Rust fits surfaces, and OCCT computes edges.** Edges are not computed in Rust, for two reasons:
- Intersecting a fillet with its tangent neighbour is ill-conditioned, so a pure intersection-based
  solve cannot produce fillet edges.
- OCCT already has robust surface-surface intersection (`GeomAPI_IntSS`), and rebuilding it in Rust is
  the biggest available yak-shave.

So the IR carries:
- fitted surfaces
- the region adjacency graph
- for each pair of adjacent regions, the mesh boundary polyline between them, tagged *transversal*
  (faces meet at an angle) or *tangent*

The writer then builds edges:
- **Transversal pairs:** the OCCT intersection, trimmed near the polyline.
- **Tangent pairs:** the polyline projected onto both surfaces, then fitted.

| layer | language | notes |
|---|---|---|
| `unmesh-core` | Rust | Weld, repair, segmentation, fitting, snapping. Deviation queries use `parry3d`. No kernel dependency. |
| `unmesh` (Python) | PyO3/maturin abi3 wheel | The public API: `unmesh.convert(mesh_or_path, options) -> Result(ir, report)`. Depends only on numpy. |
| `unmesh[step]` | Python on OCP | `unmesh.step.write(ir, path) -> WriteReport`. An optional extra, so an OCP pin can't collide with a host app's own OCP, and the core stays light. OCCT is LGPL-2.1 with an exception. |
| `unmesh-harness` | Python, a separate package | The judge. GPL tools such as pymeshlab are allowed **only** here, as optional dependencies. |
| CLI | `unmesh convert in.stl out.step` | Needs `unmesh[step]`. |

No C++ is written in this project.

## The harness is the spec

1. **Ground truth.**
   - Procedurally generated build123d parts, seeded and license-clean, in families: planar,
     curved, chamfer and fillet, and complex.
   - An imported-STEP tier from public datasets, fetched by a download script and never
     redistributed.
   - Every metric is reported by **complexity**: ground-truth face count, and the smallest feature
     size relative to the tessellation deflection.
2. **Labeled tessellation.** Every triangle carries its source face id, the surface type and its
   parameters, plus face adjacency and edge types.
3. **Degradation operators.**
   - Every operator takes a severity and a seed, and carries the labels through.
   - Families: tessellation (including fan/strip variants and non-uniform chord spacing, so tuning
     doesn't overfit OCCT's tessellator), precision, pose, noise, defects, and processing artifacts.
   - Chains and presets compose them.
   - **A fixed set of meshes exported by hand from Fusion, Onshape and SolidWorks** is a visible grid
     row from M2, because simulated exporters can't stand in for real ones.
4. **Ambiguity sets.** These cases are labeled, not guessed:
   - an intended n-gon prism against a coarse cylinder
   - a small fillet tessellated with a single segment against a planar chamfer
   - a 2-segment fillet against two planes

   The library documents its policy for each.
5. **The judge is independent of the library.**
   - It scores **the IR directly**, by sampling the analytic surfaces inside their trims with its own
     code.
   - The STEP file is checked for validity only.
   - The ground truth, the tessellation and the writer all use OCCT, so the hold-out set adds a
     non-OCCT check: a human opens files in Fusion or Onshape.
6. **Metrics.**

   | metric | measured against | gates? |
   |---|---|---|
   | Face recovery F1: the right type, with parameters within tolerance, per ground-truth face | truth | **primary gate** |
   | Topology match: face, edge and vertex counts, genus, number of through-holes | truth | gate |
   | Two-sided deviation (max, p99, p95, mean) | the **input mesh** and the **truth**, reported separately | guard |
   | Sharp-edge position error (chamfer and fillet boundaries) | truth | gate per family |
   | Segmentation IoU per triangle, parameter error | truth | stage gates |
   | Validity, fallback rate | — | gate |
   | Analytic-area fraction, face-count ratio | truth | report only; both are gameable |
   | **Calibration:** reported deviation compared with the harness-measured deviation | **input mesh** | gate: never under-reports |
   | Runtime, peak memory | — | report |

   Calibration is measured against the input mesh because that's all the library can see. Deviation
   from the truth after smoothing or decimation is a fidelity metric, not a calibration failure.
   Genus is computed from each shell's triangle coverage in the welded input mesh, not from IR
   adjacency, so a writer that drops a face is caught by the volume and validity checks rather
   than by genus.
7. **Stage isolation through oracle inputs.** The labels let every stage run on perfect inputs from
   the stage before it:
   - segmentation is scored per triangle
   - fitting consumes the ground-truth segmentation
   - edge building consumes the ground-truth surfaces

   So the stages develop **in parallel**, and an integration issue measures how errors compound.
8. **Anti-overfitting.**
   - The frozen manifest is visible to everyone.
   - A **hidden seed set** (a CI secret) runs nightly and reports aggregates only.
   - The **hold-out set** of real STLs runs nightly with results hidden from implementation agents.
   - Waiving a regressed cell needs the `waiver:approved` label, which only a human applies.
9. **Regressions.**
   - Every cell runs 3 seeds.
   - A cell regresses if it moves by more than max(2σ across the seeds, a per-metric absolute floor).

## Milestones

- **M0 Foundation:** repo tooling and CI, the public API and IR v0, the Rust/Python bindings.
- **M1 Walking skeleton:** the thinnest end-to-end slice. It covers:
  - planar ground truth
  - labeled tessellation
  - three degradations (off-plane noise, float32, rotation)
  - IR-sampled deviation and calibration
  - a smoke runner
  - an all-planar converter (normal clustering, plane fit and snap, adjacency, boundary polylines)
  - writer v0 with the faceted fallback

  The IR, the writer, the harness wiring and the calibration definition all prove out here, before
  anything else is built on them.
- **M2 Harness complete:**
  - the curved, chamfer/fillet and complex corpus families
  - the remaining degradations and the real-exporter set
  - the full metric set
  - runner v1 (HTML report, compare with noise bands, human-only waivers, hidden seeds)
  - baselines
  - the hold-out set, whose curation starts on day one
- **M3 Reconstruction v0:**
  - in parallel on oracle inputs: segmentation; cylinder and cone fitting; sphere and torus
    fitting; writer edge building for transversal conics, tangent edges and seams where faceted
    patches meet analytic faces
  - an integration issue with gates per family (planes, bores, bosses, countersinks, planar
    chamfers, conical chamfers, straight-edge fillets, circular-edge fillets), plus the production
    CLI and fidelity report
- **M4 Reconstruction v1:** blend-boundary segmentation for corner fillets, intent snapping, and
  robustness to processing artifacts.

## CI

- **Every PR:** the `smoke` grid (under 2 min) plus the Rust and Python gates, on GitHub-hosted
  runners.
- **Nightly on GitHub-hosted runners:** the slow test tier, then the `standard` and `full` grids. The slow
  tier (`-m "slow and not benchmark"`) is split into four jobs by pytest-split, balanced on the measured
  per-test durations in `harness/tests/.test_durations`; wall-clock `benchmark` tests run in their own
  single-worker job so parallel load cannot skew them; each grid shard caps worker memory so a
  runaway cell is recorded as a `memory cap` error instead of losing the runner. The
  repo is public, so hosted runners are free. No self-hosted runner is used: on a public repo a fork
  pull request could run its own code on one. Hidden seeds and the hold-out set run from a private
  companion repo, so their files and results never appear here.
- **Hidden-seed runs from the private companion repo:** that repo checks out this repo at a pinned
  sha and runs `uv run unmesh-harness run --converter unmesh --grid standard --hidden --out results`
  with the `UNMESH_HIDDEN_SEEDS` environment variable set from a CI secret (comma-separated integer
  seeds; the run errors if it is unset).   Hidden mode writes only `results/<grid>.hidden.json`: per-metric means and percentiles grouped by operator family, with no part ids, no per-cell rows,
  no seed values and no meshes. The companion publishes that aggregate file only; per-part data
  never leaves its runners and is never visible to implementation agents.

## Data and licensing

- **Generated corpus:** license-clean, redistributable.
- **Imported-STEP tier:** fetched by `scripts/fetch-datasets`, never committed. Dataset terms and the
  redistributable / download-only / excluded decision per source are in [datasets.md](datasets.md).
- **Hold-out STLs:** CC0 or CC-BY, or with the owner's consent. Stored outside this repo.

## Conventions for agents

- One issue is one worktree is one PR. Claim an issue before starting, by commenting and adding the
  `in-progress` label.
- Gates:
  - Rust: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
  - Python: `ruff check`, `ruff format --check`, `pytest`.
  - `scripts/check.sh` runs them all.
- A reconstruction PR posts the `smoke` grid diff against `main` and links the latest nightly
  `standard` diff.
- A PR never waives its own regression, and implementation agents never see hold-out or hidden-seed
  results.
- Acceptance criteria are numeric. An issue that lacks a number gets one from a human before work
  starts.

## Open decisions

- None.

## Decided

- Non-commercial datasets (Fusion 360 Gallery, Thingi10K NC files) are excluded, because OttoCAM is sold and consumes unmesh. ND, SA, GPL and ABC data stay download-only and unpublished. The owner can revisit; see [datasets.md](datasets.md).
