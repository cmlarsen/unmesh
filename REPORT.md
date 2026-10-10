# Issue #156 — curved growth vs an accurate σ

This lane is stacked on PR #141 (`i133-noisesigma`), which makes the noise σ estimate
accurate. On its own, #141 sets the auto tolerance to `max(floor, 5 σ)`; when σ is
accurate that tolerance grows to the true noise, and the curved-region growth in
`crates/unmesh-core/src/convert/grow.rs` was tuned only for the old (underestimated)
tolerance. When σ is accurate the pipeline over-merges and under-fits: it loses true
curved faces and worsens the deviation from the ground truth.

`crates/unmesh-core/src/convert/noise.rs` is not touched.

## Root causes

Three tests used the tolerance as if it were the noise scale. #141 raises `tol` to
`5 σ`, so each was 5× too permissive.

1. **Initial planar segmentation — `segment.rs::accept` (fixed here).** The angular
   allowance for growing a plane was `BASE_ANGLE + atan(2·tol / min_alt)`. A triangle's
   normal is uncertain by about `σ / min_alt`, so that term must scale with the *noise*,
   not the tolerance. With `tol = 5 σ` the allowance for a small triangle reaches ~70°,
   and a plane grows straight across a fillet. On
   `straight_fillet-0003 noise_normal@0.1 s3` this turned all four fillets into two
   planes: 4 cylinders → 1 (main: 0) and `dev_truth` 6.0e-3 → **1.85e-2**.
   Fix: pass the noise scale into `segment::run` and use `atan(2·noise / min_alt)`. The
   point-distance bound stays `tol`. Same for `resplit_loose`.
   `merge_coplanar` deliberately keeps `tol`: using σ there too over-splits and then a
   seed mis-fits a large region (a 0.4–0.7 mm outlier appears on
   `noise_isotropic@0.02` cells only when both stages use σ — see the experiment log).
2. **Growth merge tests — `grow.rs` (snapshot `db976ca`/`807d3f8`).** `smooth_pair`
   now thresholds a crease at `2 σ`; `sag_limit` bounds a noisy non-sphere/torus
   member's chord sagitta by `3·max(median, σ)`; `max_turn` rises with the chord
   deflection but never past a real fold.
3. **Clean-mesh gate — `mod.rs::finish` (snapshot `807d3f8`).** σ is not zero on a clean
   mesh (the chord error), so the `noisy` branch engaged on clean input and perturbed
   clean IR. `finish` now computes the share of exactly-coplanar interior face pairs and
   treats a mesh with `≥ CLEAN_COPLANAR_FRACTION` (1%) as clean, keeping the pre-#141
   growth behaviour.

The three original round-3 regressions are cured, with main's deviation kept or beaten:

| cell | main | #141 alone | branch |
| --- | --- | --- | --- |
| complex_assembly-0004 nn@0.1 s3 | 441 r / 28 cyc / 14.9 µm | 562 / 19 / 16.3 | **223 / 41 / 14.9** |
| complex_assembly-0003 nn@0.1 s3 | 348 / 38 / 13.5 | 451 / 30 / 18.2 | **208 / 46 / 11.4** |
| complex_void-0003 iso@0.02 s3 | 65 / 13 / 3.7 | 109 / 6 / 13.4 | **65 / 13 / 3.7** |
| through_bore-0003 nn@0.1 s4 | 8 / 2 / 5.9 | 26 / 1 / 15.3 | **8 / 2 / 5.8** |
| complex_thin-0003 nn@0.02 s3 | 120 / 20 / 1.9 | 141 / 19 / 2.5 | **120 / 20 / 1.9** |
| straight_fillet-0003 nn@0.1 s3 | 74 / 0 / 6.0 | 13 / 0 / 18.5 | **10 / 4 / 4.7** |

## Clean-mesh contamination and its gate

σ is **not zero on a clean mesh**: a clean curved surface has a nonzero plane-fit
residual (the chord error), so the `noisy` branch engaged on clean input too. The signal
that separates clean from noisy is **exact coplanarity**: a CAD tessellator keeps a flat
quad split as two exactly coplanar triangles (and keeps planar patches planar), while any
vertex noise destroys that. Over the whole mesh, the share of interior face pairs whose
normals are parallel to within `1e-12` is

| mesh | coplanar share |
| --- | --- |
| clean `identity` / `float32` / `refine` / `retriangulate` / `rotation` | 0.18 – 0.33 |
| noisy, held set (max over 384 cells) | 0.033 |
| noisy, cells the fix improves (max) | 0.0013 |
| clean `nonuniform_chords`, most families | 0.01 – 0.33 |

`finish` sets `noisy = !clean && 5σ > floor && 5σ < tol` with
`CLEAN_COPLANAR_FRACTION = 0.01`. Every held cell the fix improves has a share below
0.0013, so none is misclassified. This lane's own changes (the `segment` angle and
everything in `grow.rs`) are gated on `noisy`, so a clean mesh takes the identical path
to main.

## Acceptance

### A. Held-out cells (384), branch vs main — `accA.py`

```
cells=384 lose_curved=4 reg_hi=0 dev_worse=4 improve=36 worse=8
```

(#141 alone, measured the same way: `lose_curved=19 reg_hi=15 dev_worse=16 improve=15
worse=24`.)

* **Region count:** no cell is more than 1 above main (`reg_hi = 0`).
* **No true curved face is lost.** The four `lose_curved` cells are all
  `straight_fillet-0004`, where main over-segments the four ground-truth fillets into
  **13** cylinders (44 regions) and both #141 and this branch recover exactly **4**
  (10 regions) — the true count. Checking every held cell against the fine
  ground-truth analytic faces (`.scratch/156/trueloss.py`, counts in
  `.scratch/156/oracle_curved.json`) finds no cell where the branch recovers fewer true
  curved faces than main. The only cells where the branch misses a true curved face are
  `one_segment_fillet-0003` and `two_segment_fillet-0003` (`noise_normal@0.02`), where
  **main also recovers 0** of the 1 true curved face; that is pre-existing, not a
  regression.
* **`dev_worse = 4`** (branch truth deviation worse than main by > 1 µm), all at
  `noise_normal@0.1`:

  | cell | main | branch | Δ |
  | --- | --- | --- | --- |
  | straight_fillet-0003 s4 | 5.497e-3 | 1.246e-2 | +6.96 µm |
  | through_bore-0004 s4 | 6.632e-3 | 8.990e-3 | +2.36 µm |
  | through_bore-0004 s3 | 6.421e-3 | 7.545e-3 | +1.12 µm |
  | round_boss-0004 s4 | 5.421e-3 | 6.647e-3 | +1.23 µm |

  These are the residual cost of the larger tolerance. Main keeps `tol ≈ 1.5e-3`
  (its σ estimate folds to the floor); the branch keeps `tol = 5 σ ≈ 1e-2`, so a
  surface fit to within tolerance can sit up to ~tol from the truth where a plane or
  cylinder has a long lever arm. `straight_fillet-0003 s4` is a coverage effect at the
  fillet end (the branch's cylinder region leaves a ~1.2e-2 gap at z=0 that the fine
  truth sees; infinite-surface distances are only ~3e-3), not a fit-tolerance bug.
  Bounding a *plane* region by `2 σ` instead of `tol` in `finalize` cures
  through_bore and round_boss but destabilises `complex_thin-0004`/`lshape_outline`
  (a +18 µm cell appears), so that was rejected.
* **Improvements:** 36 cells improve (curved faces recovered, region count down, or
  truth deviation down); #141 alone showed 15 improve / 24 worse.

### B. #141 gains kept

* `uv run python harness/scripts/noise_sigma.py --parts 3 --seeds 0,1,2 --jobs 2`:
  `worst (est/true within 2x, min folded ratio): 0.519` → every cell within 2× of truth.
* `uv run python harness/scripts/noise_fallback.py --families straight_fillet --kind
  noise_normal --severity 0.02 --seeds 0,1,2 --parts 30`: `faceted fallback: 1/72 =
  1.4%` (< 10%).
* The straight_fillet held-out improvement is kept (the issue table above and the
  36-cell improvement set).

### C. Clean IR

* `fine.jsonl` (136 cells, deflection 0.001/0.1): **0 diffs vs main**
  (`.scratch/156/fine-verify.jsonl`: `done 136 ok=136 diff=0`).
* Round-1 clean set (2340 cells): the branch vs main drops from #141's 116 to **8**,
  all `revolved_torus nonuniform_chords`; imported clean set (454 cells) drops from 231
  to **18**. `nonuniform_chords` removes the quad-split coplanarity the gate keys on, so
  those tori are indistinguishable from a noisy torus by any local statistic. Note #141
  itself already disagreed with main on 262 of the 2340 coarse-clean cells; the gate
  removes the changes *this* lane introduced, which is all a stacked branch can do.

### D. CI smoke gates (`.github/workflows/ci.yml`, `--gate --jobs 2`)

* `smoke` (planar): all 340 cells F1 1.000, exit 0.
* `smoke_curved`: exit 0. Branch and `u156-main` produce **identical** summaries —
  `through_bore 6 cells P 0.027`, plane precision 0.124, curved F1 0.223. Main is not
  1.0 on through_bore, so the branch did not break it.

### Item 5 — env knobs

No `UNMESH_CAP_F`, `UNMESH_M2`, `UNMESH_SAG_K`, `UNMESH_SAG_RATIO` or `UNMESH_DBG_AT`
remains anywhere in `crates/`, `python/` or `harness/` (the only `std::env` read in
`convert/` is the pre-existing `UNMESH_TIMINGS`). The tuned values are named constants:
`SAG_RATIO = 3`, `SAG_RATIO_CLEAN = 16`, `MAX_CHORD_TURN_DEG = 80`,
`CLEAN_COPLANAR_FRACTION = 0.01`.
