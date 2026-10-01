# unmesh plan

**Status:** draft, 2026-10-01. Tracked by the epic issue; every work item below is a sub-issue.

## Goal

Convert a triangle mesh (STL, later OBJ/3MF/PLY) exported from CAD into a valid STEP solid whose
faces are real analytic surfaces (planes, cylinders, cones, spheres, tori), with a measured
fidelity report. Regions that cannot be fitted stay faceted and are sewn in; the converter never
fails a whole part for one bad region, and never claims a fidelity it did not measure.

unmesh does STL to STEP and nothing else. It knows nothing about CAM. OttoCAM is one consumer and
keeps its own manufacturing-parity acceptance gate privately.

### In scope for v1

- Meshes exported from CAD tools (Fusion, Onshape, SolidWorks, FreeCAD, OpenSCAD, Tinkercad), clean
  or moderately degraded: low-poly curves, float32 rounding, vertex noise, light editing.
- Surface types: plane, cylinder, cone, sphere, torus (constant-radius fillets). Everything else
  stays faceted.

### Out of scope for v1

Scans, sculpted or organic models, freeform NURBS fitting, recovering a feature history.

## Architecture

```
mesh file ─▶ unmesh-core (Rust) ─▶ analytic B-rep IR ─▶ writer (Python + OCP/OCCT) ─▶ STEP + fidelity report
             weld, repair,         surfaces, edges,      build solid, heal,
             segment, fit,         loops, faces, plus     validate, write
             snap, solve           per-face triangle ids
             topology              and residuals
```

| layer | language | why |
|---|---|---|
| `unmesh-core` | Rust | The hot loops (region growing, fitting, snapping) run over 10^5–10^6 triangles. Strict compiler feedback suits agent-driven development. Pure maths, no kernel dependency. |
| `unmesh._core` | Rust → Python via PyO3/maturin, abi3 wheels | One core, callable from Python with numpy arrays. |
| IR | versioned JSON schema, with serde types (Rust) and dataclasses (Python) | The only contract between the core and any writer. Can be scored directly, before any B-rep exists. Keeps the kernel swappable: a truck or opencascade.js writer can come later. |
| writer | Python on OCP (`cadquery-ocp`) | OCP is the most complete OCCT binding. Nobody writes C++. |
| harness | Python, its own package (`unmesh-harness`) | The independent judge. It measures deviation with its own code and does not trust OCCT's checks alone. |
| CLI | Python entry point, `unmesh in.stl out.step` | Ships with the Python package. |

No C++ is written in this project. OCCT arrives as a prebuilt wheel.

## The harness is the spec

The harness lands before any reconstruction code. Every reconstruction task is then phrased as
"move these cells of the grid without regressing any other cell."

1. **Ground truth.** Procedurally generated build123d parts (seeded and license-clean), plus
   imported public STEP sets where the license allows.
2. **Labeled tessellation.** Every triangle carries the source face id, the surface type and its
   parameters. Face adjacency and edge types come with them. The labels let the harness score
   segmentation, fitting and topology separately, so a failure points at a stage.
3. **Degradation operators.** Each takes a severity knob and a seed, carries the labels through,
   and composes into chains that mimic real toolchains.
   - Tessellation: chord and angle tolerance, fans, slivers, T-junctions.
   - Precision: float32 rounding, truncated ASCII digits, unit scaling, a part far from the origin.
   - Noise: isotropic, along the normal, off-plane on flats.
   - Pose: rotation, mirroring.
   - Defects: unwelded corners, cracks, holes, flipped and duplicate facets, non-manifold edges.
   - Processing: decimation, remesh, smoothing, voxel remesh.
   - Real exporters: the same models exported for real, not simulated.
4. **Ambiguity is labeled, not guessed.** An intended 12-gon and a coarse cylinder can tessellate
   identically. The corpus contains both, each labeled with its true answer, and the library
   documents its policy and exposes an override.
5. **Metrics.**
   - Two-sided deviation, with points sampled independently of OCCT.
   - Validity, including whether the file opens in a second reader.
   - Fraction of area fitted analytically, and the face-count ratio.
   - Per-triangle segmentation accuracy, surface-type accuracy, parameter error.
   - Runtime.
   - **Calibration:** the deviation the library reports must match what the harness measures.
6. **Reports.** An operator × severity grid of curves showing where each approach breaks, and a
   comparison against a stored baseline. CI blocks regressions.
7. **Hold-out set.** Real-world STLs, curated by a human and never used for tuning. Run only at
   review time, because agents overfit whatever score they are given.

## Milestones

- **M0 Foundation:** repo, CI, the Rust core plus bindings, the IR spec, and the writer spike.
- **M1 Harness:** ground truth, labels, degradations, metrics, the runner, baselines, the hold-out
  set.
- **M2 Reconstruction v0:** segmentation, planes with snapping, cylinders and cones, the topology
  solve, the production writer and CLI. The bar is to beat every baseline on the clean and low-noise
  rows of the grid.
- **M3 Reconstruction v1:** spheres and tori (fillets), intent snapping (equal radii, round
  values), and processing-artifact robustness.
- **Later:** learned segmentation at blend boundaries, a WASM build, a public release and
  benchmark.

## Conventions for agents

- One issue is one worktree is one PR. Claim an issue before starting, by commenting and adding the
  `in-progress` label.
- Gates:
  - Rust: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
  - Python: `ruff check`, `ruff format --check`, `pytest`.
- Every reconstruction PR posts the harness grid diff against `main`. A cell that regresses needs
  an explicit justification in the PR.
- The hold-out set is never run by implementation agents.

## Open decisions

- **CI runner.** GitHub-hosted runners, or the self-hosted `shop16` runner OttoCAM uses.
- **External datasets.** ABC and Fusion 360 Gallery licenses, before anything is redistributed.
- **Default tolerances.** The fit tolerance, and the snapping tolerance for angles and values.
- **Ambiguity policy default.** Whether a regular n-gon with n ≥ k becomes a cylinder.
