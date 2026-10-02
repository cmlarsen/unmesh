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
  own triangles) are sampled uniformly by area, plus every distinct triangle corner when
  `include_vertices` is on. Analytic samples are moved onto the region's surface by closed-form
  projection (plane, cylinder, cone on its apex-to-widening nappe, sphere, torus). A facets region is
  sampled in place.
- **Reference samples.** The input mesh (and the truth mesh, if given) is sampled the same way.

The draw for a triangle depends only on `(seed, stream, triangle index)`, so a run is reproducible and
independent of region order.

## What is measured

For each reference mesh, two directions, each with `max`, `p99`, `p95`, `mean`:

- `ir_to_mesh`: distance from every IR sample to the nearest reference triangle.
- `mesh_to_ir`: distance from every reference sample to the nearest point of the IR's
  footprint-bounded surfaces. That point is found on the footprint: each analytic region's source
  triangles with their corners projected onto the surface (a facets region's own triangles). The
  nearest point of that flat footprint is then moved onto its region's analytic surface, which removes
  the chord error of the flat triangle inside a face.

`JudgeResult.input.max` is the larger of the two directions against the input mesh. `region_max` is the
largest `ir_to_mesh` distance per region.

## Footprint approximation

The footprint is the region's source triangles, not the trimmed face the writer will build. The
analytic surface is therefore sampled only where the source triangles are, shifted by at most the
region's residual. Near a footprint boundary on a curved region the nearest flat point can belong to a
neighbouring region, and the reverse distance then reads up to the chord deflection of the input
tessellation instead of zero. Against the input mesh itself this is exactly the deviation being
measured; against the truth mesh it is an upper bound, not a measurement of the fit.

## Calibration

`calibration(reported_max_deviation, result.input)` is the converter's reported value minus the
measured two-sided max against the input. Negative means the converter under-reports.

## Independence

`unmesh_harness.judge` imports only `unmesh._core` and `unmesh.ir.Ir`. The Rust module uses `ir` types
and parry3d only.
