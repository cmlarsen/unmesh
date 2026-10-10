# Baseline converters on the grid

`harness/grids/*.json` define the scoring grid; `unmesh-harness run` scores one
or more converters over it and appends a JSONL record per (part, operator,
severity, seed, converter). This document records what each baseline is, how to
run it, and the score per family. `harness/scripts/baselines_table.py` turns the
result files into the tables below.

## Converters

### `faceted` (in-harness baseline)

`converters.py::convert_faceted` welds the input triangles and emits a single
`Facets` region over them, writing a faceted STEP. It is the reference "no
analytic recovery" point: F1 only, always available, no external tool.

- Version: the harness commit (`git rev-parse --short=12 HEAD`).
- License: Apache-2.0 (this repository).

### `freecad-refine`

Runs FreeCAD headless (`freecadcmd`) through
`harness/src/unmesh_harness/runner/freecad_refine.py`:
`Mesh.Mesh` → `Part.Shape.makeShapeFromMesh` (0.05 mm) →
`Part.Solid(Part.Shell(...)).removeSplitter()` → `exportStep`. The written STEP
is mapped back to the input triangles by `external.step_ir`.

- Version: FreeCAD 1.1.3 (`freecadcmd -c 'import FreeCAD; print(".".join(FreeCAD.Version()[:3]))'`).
- License: LGPL-2.1-or-later (FreeCAD). It is never imported by the harness —
  only executed as an external subprocess from `harness/`.

### `stl2step`

Runs the external [stl2step](https://github.com/LostProphet1234/stl2step)
binary in TrueForm mode:

```
stl2step INPUT.stl -o OUTPUT.step --engine trueform --quiet --threads 1
```

Exit code 2 (ok with warnings) is accepted. The STEP is mapped back to the input
triangles exactly as for `freecad-refine`.

- Version: stl2step 1.4.4 (`stl2step --version`).
- License: MIT (the stl2step project); the shipped macOS build bundles
  OpenCASCADE 7.9 dylibs (LGPL-2.1). Used only as an external subprocess.

### Mapping the STEP back to the mesh

`external.step_ir` re-tessellates the written STEP, assigns every input triangle
to its nearest STEP face (distance plus a small normal-alignment term), and
builds an IR with one region per STEP face — analytic (`plane`, `cylinder`, …)
where the surface is supported, `facets` otherwise. The F1 scorer and the
independent judge consume that IR, so baselines and `unmesh` are scored the same
way.

## Running a baseline

Both external tools are located by environment variables, read when a run
starts; a converter whose tool is not configured is skipped with a log line.

```sh
# unmesh's own converter (the thing #18 must pass)
uv run --locked unmesh-harness run --grid standard --converter unmesh \
    --out RUNS/standard/unmesh --jobs N

# the in-harness faceted baseline
uv run --locked unmesh-harness run --grid standard --converter faceted \
    --out RUNS/standard/faceted --jobs N

# FreeCAD refine
UNMESH_FREECAD=/path/to/freecadcmd \
uv run --locked unmesh-harness run --grid standard --converter freecad-refine \
    --out RUNS/standard/freecad --jobs N

# stl2step (TrueForm)
UNMESH_STL2STEP=/path/to/stl2step \
uv run --locked unmesh-harness run --grid standard --converter stl2step \
    --out RUNS/standard/stl2step --jobs N

# the table for all of the above
uv run --locked harness/scripts/baselines_table.py RUNS/standard/*/standard.jsonl
```

`N` is the job count; the harness caps each worker's RSS and kills it if a cell
overruns its timeout. The external tools get the same limits: the cell's
timeout (minus a small reserve so the tool is killed before its worker) and the
cell's memory cap are passed down, and the tool's whole process group is killed
if it exceeds either. A tool that times out, is killed, or otherwise fails is
recorded as that cell's failure; the run continues.

Use `--jobs 2` (or fewer) when several external tools would otherwise contend
for cores. `freecadcmd` and `stl2step` each read `--threads`/single-thread flags
internally, and the pool already pins BLAS/OpenMP to one thread.

## Results

### Run conditions

The tables below are the four `standard`-grid runs (one per converter) recorded
on 2026-10-09/10. Each run scored 9900 cells: the grid's 100 explicit `parts`
(planar, curved, chamfer/fillet, complex and imported), its 43 operator rows and
its seeds, i.e. 99 cells per part. Harness commit `80f3fad` (`git_sha`
`80f3fad4fab0` plus the run's dirty-tree hash), `--jobs 6` on a shared 12-core
Linux box, 60 s per-cell timeout, FreeCAD 1.1.3 and stl2step 1.4.4. The runs were
not repeated; the result files are not committed.

```sh
uv run --locked harness/scripts/baselines_table.py RUNS/standard/*/standard.jsonl
```

### How a face counts as analytic

The faceted share is classified the same way for every converter, against the
input's ground-truth faces rather than from the written solid's own shape. Each
input triangle carries the true face it came from, and that face's surface type.
A written region is **faceted** when the writer emitted it as a `facets` patch.
Any other region is **analytic** if and only if its surface type equals the
area-weighted majority true surface type of the input triangles mapped to it;
otherwise it is faceted, an approximation. So a plane sitting on a true cylinder,
cone, sphere or torus counts as faceted, and a designed plane counts as analytic
however few input triangles it carries. Regions no input triangle maps to are
ignored. A per-triangle plane soup on a true plane therefore counts as analytic
by design: the metric measures surface type, and over-segmentation shows in F1.

Regions roll up into a cell as the **faceted share**,
`faceted_regions / (faceted_regions + analytic_regions)`. One correctly typed
base plane therefore no longer hides a dome written as thousands of planes: a
cell with one analytic and 7588 faceted regions has a share of 100%, where the
old per-cell flag read 0%. A cell the writer fell back on with no classified
region at all is 100%; a cell with no region of either kind is skipped from the
share (it stays in `cells`). The table then averages the per-cell share over the
ok cells only, so a family's share is the mean of its scored cells' shares, not a
pooled region ratio.

The share is a region-count ratio, so a soup of many tiny approximate planes can
pull it toward 100% (or a single correctly typed region can dominate a count)
without regard to the areas involved; an area-weighted share would be more
faithful, and that is a later change.

The faceted share therefore measures whether curved truth became non-curved
surfaces: a converter that turns a true cylinder, cone, sphere or torus into
planes reads as almost entirely faceted on those cells even when the planes are
exact. F1 separately penalises over-segmentation — its precision divides by the
number of written regions — so the two metrics cannot be traded against each other.

### Counting failures, timeouts and skips

`harness/scripts/baselines_table.py` derives every record's family from its part
id (`complex_void-0003` → `complex_void`, `imported-0225` → `imported`), so
failures and timeouts — which carry no `family` field — are attributed to their
family instead of a `?` row.

- `cells` is the failure-rate denominator: ok cells plus timeouts and other
  converter failures, excluding skips and harness failures.
- `failure` = (timeouts + other converter failures) / `cells`.
- `timeout` is the number of timeouts in that family; for the external baselines
  it includes the tool's own `TimeoutError`, which the runner records as an
  error rather than a `timeout` status.
- `skipped as inapplicable` counts inapplicable-operator cells; they are
  excluded from every rate.
- `harness failures (excluded)` counts cells that failed before the converter ran
  — ground-truth preparation, or a degradation (operator) step — and are excluded
  from the converter's failure rate.
- `mean F1` and `faceted share` are means over the ok cells only; a family with no
  ok cell shows `-`.

Per converter the grid has 3 skipped cells and 105 harness failures (99
ground-truth preparation, 6 `noise_off_plane` degradation) in every run.

Because `mean F1` is over ok cells only, a converter that fails or times out more
often is scored on an easier remainder — stl2step's clean `through_bore` F1 0.089
comes with 57% failures and its `revolved_dome` F1 0.001 with 44%, so those means
flatter it; the clean planar families, where its failure rate is ~1%, are not
affected.

## Clean rows

### unmesh

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 243 | 0.596 | 40% | 0% | 1 |
| bore_chamfer | 324 | 0.599 | 39% | 1% | 2 |
| boss_plate | 162 | 0.718 | 24% | 0% | 0 |
| circular_fillet | 324 | 0.575 | 40% | 5% | 15 |
| complex_assembly | 405 | 0.643 | 32% | 1% | 5 |
| complex_mixed | 324 | 0.573 | 43% | 2% | 8 |
| complex_thin | 567 | 0.599 | 39% | 0% | 0 |
| complex_void | 324 | 0.604 | 30% | 1% | 4 |
| corner_fillet | 324 | 0.566 | 40% | 2% | 6 |
| counterbore | 324 | 0.576 | 42% | 0% | 1 |
| countersink | 324 | 0.573 | 43% | 1% | 2 |
| imported | 1536 | 0.491 | 45% | 43% | 658 |
| lshape_outline | 162 | 0.722 | 25% | 0% | 0 |
| planar_chamfer | 324 | 0.705 | 24% | 0% | 0 |
| plate_pockets | 162 | 0.712 | 25% | 0% | 0 |
| polygon_prism | 324 | 0.734 | 24% | 0% | 0 |
| revolved_cone | 81 | 0.667 | 31% | 1% | 1 |
| revolved_dome | 81 | 0.591 | 35% | 1% | 1 |
| revolved_torus | 81 | 0.862 | 14% | 25% | 20 |
| rotated_pockets | 162 | 0.724 | 24% | 0% | 0 |
| round_boss | 162 | 0.580 | 41% | 1% | 1 |
| round_slot_blind | 81 | 0.599 | 40% | 0% | 0 |
| round_slot_through | 162 | 0.594 | 41% | 1% | 1 |
| square_slots | 162 | 0.703 | 26% | 0% | 0 |
| stepped_block | 162 | 0.739 | 23% | 0% | 0 |
| straight_fillet | 324 | 0.548 | 43% | 0% | 1 |
| thin_walls | 162 | 0.733 | 23% | 0% | 0 |
| through_bore | 81 | 0.576 | 40% | 1% | 1 |
| through_cuts | 162 | 0.715 | 25% | 0% | 0 |
| **all families** | 8016 | 0.615 | 36% | 9% | 728 |

skipped as inapplicable: 3
harness failures (excluded): 81 (81 ground-truth preparation, 0 degradation)

### faceted

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 243 | 0.000 | 100% | 0% | 0 |
| bore_chamfer | 324 | 0.000 | 100% | 0% | 0 |
| boss_plate | 162 | 0.000 | 100% | 0% | 0 |
| circular_fillet | 324 | 0.000 | 100% | 0% | 0 |
| complex_assembly | 405 | 0.000 | 100% | 0% | 0 |
| complex_mixed | 324 | 0.000 | 100% | 0% | 0 |
| complex_thin | 567 | 0.000 | 100% | 0% | 0 |
| complex_void | 324 | 0.000 | 100% | 0% | 0 |
| corner_fillet | 324 | 0.000 | 100% | 0% | 0 |
| counterbore | 324 | 0.000 | 100% | 0% | 0 |
| countersink | 324 | 0.000 | 100% | 0% | 0 |
| imported | 1536 | 0.000 | 100% | 6% | 86 |
| lshape_outline | 162 | 0.000 | 100% | 0% | 0 |
| planar_chamfer | 324 | 0.000 | 100% | 0% | 0 |
| plate_pockets | 162 | 0.000 | 100% | 0% | 0 |
| polygon_prism | 324 | 0.000 | 100% | 0% | 0 |
| revolved_cone | 81 | 0.000 | 100% | 0% | 0 |
| revolved_dome | 81 | 0.000 | 100% | 0% | 0 |
| revolved_torus | 81 | 0.000 | 100% | 0% | 0 |
| rotated_pockets | 162 | 0.000 | 100% | 0% | 0 |
| round_boss | 162 | 0.000 | 100% | 0% | 0 |
| round_slot_blind | 81 | 0.000 | 100% | 0% | 0 |
| round_slot_through | 162 | 0.000 | 100% | 0% | 0 |
| square_slots | 162 | 0.000 | 100% | 0% | 0 |
| stepped_block | 162 | 0.000 | 100% | 0% | 0 |
| straight_fillet | 324 | 0.000 | 100% | 0% | 0 |
| thin_walls | 162 | 0.000 | 100% | 0% | 0 |
| through_bore | 81 | 0.000 | 100% | 0% | 0 |
| through_cuts | 162 | 0.000 | 100% | 0% | 0 |
| **all families** | 8016 | 0.000 | 100% | 1% | 86 |

skipped as inapplicable: 3
harness failures (excluded): 81 (81 ground-truth preparation, 0 degradation)

### freecad-refine

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 243 | 0.068 | 85% | 11% | 4 |
| bore_chamfer | 324 | 0.014 | 95% | 13% | 8 |
| boss_plate | 162 | 0.779 | 0% | 3% | 2 |
| circular_fillet | 324 | 0.003 | 99% | 47% | 121 |
| complex_assembly | 405 | 0.127 | 71% | 76% | 284 |
| complex_mixed | 324 | 0.050 | 85% | 13% | 8 |
| complex_thin | 567 | 0.120 | 76% | 16% | 16 |
| complex_void | 324 | 0.164 | 78% | 3% | 8 |
| corner_fillet | 324 | 0.050 | 99% | 43% | 42 |
| counterbore | 324 | 0.050 | 85% | 10% | 8 |
| countersink | 324 | 0.019 | 95% | 13% | 8 |
| imported | 1536 | 0.089 | 80% | 62% | 824 |
| lshape_outline | 162 | 0.731 | 0% | 2% | 2 |
| planar_chamfer | 324 | 0.751 | 0% | 1% | 4 |
| plate_pockets | 162 | 0.799 | 0% | 6% | 2 |
| polygon_prism | 324 | 0.742 | 0% | 3% | 4 |
| revolved_cone | 81 | 0.004 | 97% | 12% | 2 |
| revolved_dome | 81 | 0.001 | 100% | 21% | 9 |
| revolved_torus | 81 | 0.000 | 100% | 96% | 78 |
| rotated_pockets | 162 | 0.758 | 0% | 6% | 2 |
| round_boss | 162 | 0.074 | 85% | 9% | 2 |
| round_slot_blind | 81 | 0.136 | 78% | 9% | 1 |
| round_slot_through | 162 | 0.103 | 81% | 10% | 3 |
| square_slots | 162 | 0.773 | 0% | 5% | 2 |
| stepped_block | 162 | 0.798 | 0% | 3% | 2 |
| straight_fillet | 324 | 0.115 | 78% | 6% | 4 |
| thin_walls | 162 | 0.795 | 0% | 6% | 2 |
| through_bore | 81 | 0.047 | 86% | 10% | 2 |
| through_cuts | 162 | 0.764 | 0% | 3% | 2 |
| **all families** | 8016 | 0.294 | 58% | 26% | 1456 |

skipped as inapplicable: 3
harness failures (excluded): 81 (81 ground-truth preparation, 0 degradation)

### stl2step

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 243 | 0.237 | 72% | 28% | 69 |
| bore_chamfer | 324 | 0.023 | 96% | 53% | 171 |
| boss_plate | 162 | 0.884 | 0% | 1% | 2 |
| circular_fillet | 324 | 0.003 | 100% | 18% | 59 |
| complex_assembly | 405 | 0.138 | 78% | 68% | 275 |
| complex_mixed | 324 | 0.076 | 85% | 55% | 179 |
| complex_thin | 567 | 0.157 | 76% | 58% | 328 |
| complex_void | 324 | 0.403 | 50% | 27% | 89 |
| corner_fillet | 324 | 0.060 | 99% | 28% | 18 |
| counterbore | 324 | 0.312 | 60% | 16% | 52 |
| countersink | 324 | 0.040 | 95% | 39% | 125 |
| imported | 1536 | 0.224 | 73% | 61% | 884 |
| lshape_outline | 162 | 0.883 | 0% | 1% | 1 |
| planar_chamfer | 324 | 0.865 | 0% | 1% | 4 |
| plate_pockets | 162 | 0.880 | 0% | 1% | 2 |
| polygon_prism | 324 | 0.875 | 1% | 1% | 3 |
| revolved_cone | 81 | 0.009 | 98% | 37% | 30 |
| revolved_dome | 81 | 0.001 | 100% | 44% | 36 |
| revolved_torus | 81 | 0.000 | 100% | 88% | 71 |
| rotated_pockets | 162 | 0.893 | 0% | 1% | 2 |
| round_boss | 162 | 0.345 | 61% | 15% | 24 |
| round_slot_blind | 81 | 0.160 | 83% | 1% | 1 |
| round_slot_through | 162 | 0.163 | 80% | 15% | 24 |
| square_slots | 162 | 0.878 | 0% | 1% | 2 |
| stepped_block | 162 | 0.898 | 0% | 1% | 2 |
| straight_fillet | 324 | 0.265 | 68% | 1% | 4 |
| thin_walls | 162 | 0.866 | 0% | 1% | 2 |
| through_bore | 81 | 0.089 | 84% | 57% | 46 |
| through_cuts | 162 | 0.889 | 0% | 1% | 2 |
| **all families** | 8016 | 0.429 | 50% | 33% | 2507 |

skipped as inapplicable: 3
harness failures (excluded): 81 (81 ground-truth preparation, 0 degradation)

## Noise rows

### unmesh

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 54 | 0.390 | 34% | 20% | 11 |
| bore_chamfer | 72 | 0.739 | 7% | 19% | 14 |
| boss_plate | 36 | 0.824 | 1% | 0% | 0 |
| circular_fillet | 72 | 0.164 | 67% | 33% | 24 |
| complex_assembly | 90 | 0.228 | 49% | 27% | 24 |
| complex_mixed | 72 | 0.167 | 75% | 33% | 24 |
| complex_thin | 126 | 0.192 | 37% | 33% | 42 |
| complex_void | 72 | 0.333 | 42% | 28% | 20 |
| corner_fillet | 72 | 0.020 | 91% | 36% | 26 |
| counterbore | 72 | 0.507 | 40% | 29% | 21 |
| countersink | 72 | 0.415 | 51% | 29% | 21 |
| imported | 336 | 0.091 | 56% | 53% | 177 |
| lshape_outline | 36 | 0.771 | 1% | 0% | 0 |
| planar_chamfer | 72 | 0.646 | 2% | 1% | 1 |
| plate_pockets | 36 | 0.823 | 1% | 0% | 0 |
| polygon_prism | 72 | 0.686 | 4% | 0% | 0 |
| revolved_cone | 18 | 0.213 | 35% | 33% | 6 |
| revolved_dome | 18 | 0.003 | 75% | 17% | 3 |
| revolved_torus | 18 | 1.000 | 0% | 56% | 10 |
| rotated_pockets | 36 | 0.808 | 1% | 0% | 0 |
| round_boss | 36 | 0.292 | 42% | 6% | 2 |
| round_slot_blind | 18 | 0.557 | 8% | 0% | 0 |
| round_slot_through | 36 | 0.443 | 36% | 17% | 6 |
| square_slots | 36 | 0.790 | 1% | 0% | 0 |
| stepped_block | 36 | 0.695 | 2% | 3% | 1 |
| straight_fillet | 72 | 0.149 | 51% | 1% | 1 |
| thin_walls | 36 | 0.636 | 2% | 8% | 3 |
| through_bore | 18 | 0.665 | 10% | 17% | 3 |
| through_cuts | 36 | 0.712 | 3% | 0% | 0 |
| **all families** | 1776 | 0.420 | 33% | 25% | 440 |

skipped as inapplicable: 0
harness failures (excluded): 24 (18 ground-truth preparation, 6 degradation)

### faceted

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 54 | 0.000 | 100% | 0% | 0 |
| bore_chamfer | 72 | 0.000 | 100% | 0% | 0 |
| boss_plate | 36 | 0.000 | 100% | 0% | 0 |
| circular_fillet | 72 | 0.000 | 100% | 0% | 0 |
| complex_assembly | 90 | 0.000 | 100% | 0% | 0 |
| complex_mixed | 72 | 0.000 | 100% | 0% | 0 |
| complex_thin | 126 | 0.000 | 100% | 0% | 0 |
| complex_void | 72 | 0.000 | 100% | 0% | 0 |
| corner_fillet | 72 | 0.000 | 100% | 0% | 0 |
| counterbore | 72 | 0.000 | 100% | 0% | 0 |
| countersink | 72 | 0.000 | 100% | 0% | 0 |
| imported | 336 | 0.000 | 100% | 2% | 6 |
| lshape_outline | 36 | 0.000 | 100% | 0% | 0 |
| planar_chamfer | 72 | 0.000 | 100% | 0% | 0 |
| plate_pockets | 36 | 0.000 | 100% | 0% | 0 |
| polygon_prism | 72 | 0.000 | 100% | 0% | 0 |
| revolved_cone | 18 | 0.000 | 100% | 0% | 0 |
| revolved_dome | 18 | 0.000 | 100% | 0% | 0 |
| revolved_torus | 18 | 0.000 | 100% | 0% | 0 |
| rotated_pockets | 36 | 0.000 | 100% | 0% | 0 |
| round_boss | 36 | 0.000 | 100% | 0% | 0 |
| round_slot_blind | 18 | 0.000 | 100% | 0% | 0 |
| round_slot_through | 36 | 0.000 | 100% | 0% | 0 |
| square_slots | 36 | 0.000 | 100% | 0% | 0 |
| stepped_block | 36 | 0.000 | 100% | 0% | 0 |
| straight_fillet | 72 | 0.000 | 100% | 0% | 0 |
| thin_walls | 36 | 0.000 | 100% | 0% | 0 |
| through_bore | 18 | 0.000 | 100% | 0% | 0 |
| through_cuts | 36 | 0.000 | 100% | 0% | 0 |
| **all families** | 1776 | 0.000 | 100% | 0% | 6 |

skipped as inapplicable: 0
harness failures (excluded): 24 (18 ground-truth preparation, 6 degradation)

### freecad-refine

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 54 | 0.000 | 49% | 33% | 18 |
| bore_chamfer | 72 | 0.000 | 79% | 33% | 24 |
| boss_plate | 36 | 0.000 | 0% | 17% | 6 |
| circular_fillet | 72 | 0.000 | 96% | 50% | 36 |
| complex_assembly | 90 | 0.000 | 50% | 38% | 34 |
| complex_mixed | 72 | 0.000 | 54% | 33% | 24 |
| complex_thin | 126 | 0.000 | 47% | 33% | 42 |
| complex_void | 72 | 0.000 | 46% | 33% | 24 |
| corner_fillet | 72 | 0.000 | 100% | 49% | 24 |
| counterbore | 72 | 0.000 | 49% | 33% | 24 |
| countersink | 72 | 0.000 | 81% | 33% | 24 |
| imported | 336 | 0.000 | 50% | 72% | 241 |
| lshape_outline | 36 | 0.000 | 0% | 0% | 0 |
| planar_chamfer | 72 | 0.000 | 0% | 21% | 15 |
| plate_pockets | 36 | 0.000 | 0% | 17% | 6 |
| polygon_prism | 72 | 0.000 | 0% | 0% | 0 |
| revolved_cone | 18 | 0.000 | 87% | 33% | 6 |
| revolved_dome | 18 | 0.000 | 98% | 33% | 6 |
| revolved_torus | 18 | - | - | 100% | 18 |
| rotated_pockets | 36 | 0.000 | 0% | 11% | 4 |
| round_boss | 36 | 0.000 | 49% | 33% | 12 |
| round_slot_blind | 18 | 0.000 | 47% | 33% | 6 |
| round_slot_through | 36 | 0.000 | 47% | 33% | 12 |
| square_slots | 36 | 0.000 | 0% | 17% | 6 |
| stepped_block | 36 | 0.000 | 0% | 17% | 6 |
| straight_fillet | 72 | 0.000 | 44% | 25% | 18 |
| thin_walls | 36 | 0.000 | 0% | 31% | 11 |
| through_bore | 18 | 0.000 | 49% | 33% | 6 |
| through_cuts | 36 | 0.000 | 0% | 33% | 12 |
| **all families** | 1776 | 0.000 | 39% | 38% | 665 |

skipped as inapplicable: 0
harness failures (excluded): 24 (18 ground-truth preparation, 6 degradation)

### stl2step

| family | cells | mean F1 | faceted share | failure | timeout |
|---|---|---|---|---|---|
| blind_bore | 54 | 0.011 | 47% | 39% | 21 |
| bore_chamfer | 72 | 0.001 | 73% | 60% | 43 |
| boss_plate | 36 | 0.197 | 0% | 17% | 6 |
| circular_fillet | 72 | 0.000 | 96% | 51% | 37 |
| complex_assembly | 90 | 0.005 | 50% | 63% | 57 |
| complex_mixed | 72 | 0.006 | 54% | 56% | 40 |
| complex_thin | 126 | 0.003 | 47% | 33% | 42 |
| complex_void | 72 | 0.020 | 48% | 33% | 24 |
| corner_fillet | 72 | 0.002 | 100% | 65% | 37 |
| counterbore | 72 | 0.020 | 56% | 33% | 24 |
| countersink | 72 | 0.003 | 80% | 64% | 46 |
| imported | 336 | 0.005 | 50% | 75% | 244 |
| lshape_outline | 36 | 0.164 | 0% | 14% | 5 |
| planar_chamfer | 72 | 0.097 | 1% | 0% | 0 |
| plate_pockets | 36 | 0.186 | 0% | 17% | 6 |
| polygon_prism | 72 | 0.154 | 0% | 8% | 6 |
| revolved_cone | 18 | 0.000 | 84% | 83% | 15 |
| revolved_dome | 18 | 0.000 | 98% | 83% | 15 |
| revolved_torus | 18 | - | - | 100% | 18 |
| rotated_pockets | 36 | 0.197 | 0% | 17% | 6 |
| round_boss | 36 | 0.023 | 46% | 33% | 12 |
| round_slot_blind | 18 | 0.014 | 36% | 17% | 3 |
| round_slot_through | 36 | 0.009 | 46% | 31% | 11 |
| square_slots | 36 | 0.141 | 0% | 17% | 6 |
| stepped_block | 36 | 0.268 | 0% | 11% | 4 |
| straight_fillet | 72 | 0.002 | 35% | 0% | 0 |
| thin_walls | 36 | 0.029 | 0% | 19% | 7 |
| through_bore | 18 | 0.082 | 63% | 61% | 11 |
| through_cuts | 36 | 0.125 | 0% | 19% | 7 |
| **all families** | 1776 | 0.062 | 34% | 43% | 753 |

skipped as inapplicable: 0
harness failures (excluded): 24 (18 ground-truth preparation, 6 degradation)

## Findings

On the **clean rows** (8016 scored cells per converter) unmesh's mean F1 is
0.615, against faceted 0.000, freecad-refine 0.294 and stl2step 0.429. Its
faceted share is 36%, against faceted's 100%, freecad-refine's 58% and
stl2step's 50%. On the **noise rows** (1776 cells) unmesh is 0.420 and 33%,
against faceted 0.000 and 100%, freecad-refine 0.000 and 39%, stl2step 0.062 and
34%. unmesh's clean failure rate is 9% (728 timeouts), below freecad-refine's 26%
(1456 timeouts) and stl2step's 33% (2507); on noise rows it is 25% (440) against
38% (665) and 43% (753).

The external tools' low faceted share on planar cells is not an advantage: there
they emit correctly typed planes, so their share is near zero (freecad-refine
`boss_plate` 0%, `polygon_prism` 0%; stl2step `polygon_prism` 1%), but on curved
truth they emit planes, which count as faceted — freecad-refine `revolved_dome`
100%, `circular_fillet` 99%, `through_bore` 86%; stl2step `revolved_dome` 100%,
`circular_fillet` 100%, `through_bore` 84% — so a near-zero F1 on curved truth
comes with a near-100% share. unmesh sits between, recovering curved surfaces on
most cells and falling back to facets on hard ones (`revolved_torus` 14%,
`polygon_prism` 24%, `revolved_dome` 35%). A single correctly typed base plane no
longer hides a dome written as thousands of planes: the freecad-refine
`revolved_dome` identity cell with one analytic and 7588 faceted regions reads as
100%, where the old per-cell flag read 0%.

By **family group** (cell-pooled by the corpus `strata.category`, where the
bores, `counterbore` and `countersink` fall under `curved` and `planar_chamfer`
under `chamfer/fillet`), on clean planar families unmesh (0.723) trails stl2step
(0.882) and freecad-refine (0.768); on clean curved families unmesh (0.598) leads
stl2step (0.193) and freecad-refine (0.054); on clean chamfer/fillet families
unmesh (0.599) leads stl2step (0.294) and freecad-refine (0.229). The same split
holds on noise rows: planar unmesh 0.744 vs stl2step 0.163 and freecad-refine
0.000; curved unmesh 0.425 vs stl2step 0.016 and freecad-refine 0.000;
chamfer/fillet unmesh 0.368 vs stl2step 0.031 and freecad-refine 0.000. So unmesh
is behind the best baseline on the planar families and ahead of both on the
curved and chamfer/fillet families, on clean and noisy input alike.

By **timeouts**, unmesh's 1168 (728 clean, 440 noise) cluster on
`refine+noise_off_plane` (332), `t_junctions` (75), `refine` (63), `slivers`
(57), `noise_isotropic` and `noise_normal` (54 each), `nonuniform_chords` (48)
and `retriangulate` (47); by family `imported` accounts for 835 (72%), then
complex (127), curved (112), chamfer/fillet (90) and planar (4), and severity 0.0
accounts for 8, 0.5 for 665 and 1.0 for 495. freecad-refine's 2121 timeouts
(including the tool's own `TimeoutError`) lead with `refine+noise_off_plane`
(504), `refine` (162), `slivers` (155) and `t_junctions` (123), with 1065
`imported`. stl2step's 3260 lead with `refine+noise_off_plane` (463),
`nonuniform_chords` (305), `slivers` (289) and `far_translation` (286), with 1128
`imported`. faceted times out only on `imported` (92).

The timeout share depends on machine load: the runs used `--jobs 6` on a shared
12-core box, so each cell's 60 s wall-clock budget is contended by the other
workers and the counts are not a fixed property of a converter alone.
