# The judge

`unmesh_core::judge` (Rust, parry3d-f64 BVH queries) and `unmesh_harness.judge` (Python) score an IR
against a mesh without touching the converter or OCCT's tessellation of the output.

```python
result = judge(ir, input_mesh, truth_mesh=None, report=None, samples_per_mm2=10.0, seed=0)
```

`input_mesh` and `truth_mesh` are `(n, 3, 3)` triangle arrays or anything with a `.tris` (a
`LabeledMesh`). `report` is the converter's report, a dict, or a float; only `max_deviation` is read.

## What is sampled

- **IR samples.** Each region's source triangles (indices into the input mesh, or the facets region's
  own triangles) are sampled uniformly by area, plus every distinct triangle corner and edge midpoint
  (zero-area triangles included) when `include_vertices` is on. Analytic samples are moved onto the region's surface by closed-form
  projection (plane, cylinder, cone on its apex-to-widening nappe, sphere, torus). A facets region is
  sampled in place.
- **Reference samples.** The input mesh (and the truth mesh, if given) is sampled the same way.

The draw for a triangle depends only on `(seed, stream, triangle index)`, so a run is reproducible and
independent of region order.

## What is measured

For each reference mesh, two directions, each with `max`, `p99`, `p95`, `mean`:

- `ir_to_mesh`: distance from every IR sample to the nearest reference triangle.
- `mesh_to_ir`: distance from every reference sample to the nearest point of the IR's
  footprint-bounded surfaces, taken over all regions. Near a shared edge it therefore reads the
  smaller of the two neighbouring regions' distances. That point is found on the footprint: each analytic region's source
  triangles with their corners projected onto the surface (a facets region's own triangles). The
  nearest point of that flat footprint is then moved onto its region's analytic surface, which removes
  the chord error of the flat triangle inside a face. Against the input mesh, each sample is also
  measured to its own projection onto the surface of the analytic region that owns its triangle, and
  the smaller of the two is kept (see below).

`JudgeResult.input.max` is the larger of the two directions against the input mesh. `region_max` is the
largest `ir_to_mesh` distance per region (IR to mesh only; there is no per-region reverse distance).

## Sampled max with refinement

`max` is a sampled maximum, hence a lower bound of the true maximum. To tighten it, the 256 worst
samples in each direction are refined by a pattern search in barycentric coordinates over their
triangle (step 0.25 halving to 1e-6), and `max` is the larger of the sampled and refined values.
`p99`, `p95` and `mean` are plain sample statistics.

Measured residual: on noisy planar parts (box with a hole, 4 um vertex noise) and on a coarse sphere,
the default density (10/mm2), 1 /mm2 and 1000/mm2 give the same max to five digits, so the residual is
far below the 0.5% the test enforces. The refinement relies on corner and midpoint samples: with
`include_vertices` off and under about 1 sample/mm2 the max can read 15-50% low.

Calibration should therefore allow a small tolerance: a converter under-reports only if
`reported < measured - UNDER_REPORT_ABS_TOL - UNDER_REPORT_REL_TOL * measured` (1e-9 mm and 0.5%),
which is `under_reports()` in Python and `unmesh_core::judge::under_reports` in Rust.

## Footprint approximation

The footprint is the region's source triangles, not the trimmed face the writer will build. The judged
patch is those triangles projected onto the surface. Area that a writer extends a wrong surface over,
out to its trims, is not judged. Near a footprint boundary on a curved region the nearest flat point can belong to a
neighbouring region. Its projection then sits on that region's surface, up to a chord gap away from
the flat point, while the region that owns the sample can be closer. The nearest flat triangle is
the wrong argmin there: on noisy fillet parts it read the reverse distance up to 25% above both the
converter's honest per-region deviation and a dense brute-force check, and flagged honest reports as
under-reports (issue #94). So a sample of the input mesh is also measured to the projection of itself
onto its owning region's surface. That point lies in the owner's judged patch (the projection of the
owner's source triangles), so the minimum is still an upper bound on the true distance to the IR, and
it never exceeds the owner's own residual at that point. The truth mesh has no owners and keeps the
flat-footprint value, which near a curved boundary is an upper bound, not a measurement of the fit.

## Calibration

`calibration(reported_max_deviation, result.input)` is the converter's reported value minus the
measured two-sided max against the input. Negative means the converter under-reports.

## Independence

`unmesh_harness.judge` imports only `unmesh._core` and `unmesh.ir.Ir`. The Rust module uses `ir` types
and parry3d only.
