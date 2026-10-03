# API

Three public surfaces: the Python package `unmesh`, the optional `unmesh.step` writer, and the Rust
crate `unmesh-core`. They share one contract, the [IR](ir.md).

Everything here is v0 and may change freely before the first release. `convert` handles all-planar
parts today (what it cannot fit stays `facets`). `step.write` is implemented for planes and `facets`
(#5).

## Python

```python
import unmesh

result = unmesh.convert("part.stl")  # -> Result(ir, report)
result.ir  # unmesh.ir.Ir
result.report.max_deviation

import unmesh.step  # needs `pip install unmesh[step]`

write_report = unmesh.step.write(result.ir, "part.step")
```

### `unmesh.convert(mesh_or_path, options=None) -> Result`

`mesh_or_path` is one of:

- a path (`str` or `os.PathLike`) to an STL file;
- a NumPy array of shape `(n, 3, 3)`: `n` triangles of three corners (a triangle soup);
- a tuple `(vertices, faces)` of arrays shaped `(v, 3)` float and `(f, 3)` integer.

`options` is a `ConvertOptions` (`None` means all defaults):

| field | default | meaning |
|---|---|---|
| `linear_tolerance` | `None` | Linear tolerance. `None` derives it from the mesh: the converter segments once at 5e-4 of the bounding-box diagonal, estimates the vertex noise of its best-supported planar regions, and refits at five times that noise, clamped between a floor of 2e-5 of the diagonal and the first guess. Becomes `ir.tolerances.linear`. |
| `angular_snap_deg` | 0.5 | See [IR § Tolerances](ir.md#tolerances). |
| `tangent_threshold_deg` | 3.0 | See [IR § Tangent versus transversal](ir.md#tangent-versus-transversal). |
| `vertex_merge` | 1e-6 | See IR tolerances. |

It returns `Result`, a named tuple `(ir, report)`, so `ir, report = unmesh.convert(...)` works.

`Report`:

| field | meaning |
|---|---|
| `max_deviation` | Upper bound on the distance between the input mesh and the IR's surfaces, in both directions. **Never under-reports**: the harness checks it against the deviation it measures itself. |
| `rms_deviation` | RMS of the same. |
| `analytic_area_fraction` | Share of mesh area in analytic (non-`facets`) regions. A report only, not a quality measure. |
| `region_counts` | Count of regions per surface type, e.g. `{"plane": 6, "cylinder": 1}`. |
| `warnings` | List of `ConvertWarning(code, message)`. Codes: `degenerate_triangles`, `flipped_winding`, `repaired_winding`, `open_edges`, `non_manifold_edges`. See [IR § Non-manifold and open input](ir.md#non-manifold-and-open-input). |

A clean mesh therefore gets the floor (2e-5 of the bounding-box diagonal, so the
tolerance scales with the part's units) and a noisy one a tolerance that follows its noise, so snapping
(see [IR § Tolerances](ir.md#tolerances)) never moves a surface by more than the data justifies.

`convert` raises `ValueError` for input it cannot read (no triangles, wrong array shape, non-finite
coordinates) and `OSError` for an unreadable file. It never raises for a hard-to-fit part: those regions
come back as `facets`.

### `unmesh.step.write(ir, path, options=None, *, mesh=None) -> WriteReport`

An optional extra: `pip install unmesh[step]` installs OCP. `import unmesh` never imports OCP;
`unmesh.step` imports it only when `write` runs, and raises `ImportError` with the install hint when
it is missing. `unmesh.step` is also reachable as an attribute of `unmesh`.

`options` is a `WriteOptions`:

- `max_shape_tolerance` (default `1e-3`): the largest OCCT shape tolerance a written shell may need before the writer falls back.
- `max_deviation` (default `5e-3`, absolute, in mesh units): the cap on how far the writer may move geometry. The limit used is `min(5 * ir.tolerances.linear, max_deviation)`. A vertex further than that from its IR position, or a boundary point further than that from the built edge, fails the shell.

`mesh` is the source mesh the IR was converted from, in any form `convert` accepts (path, `(n, 3, 3)`
array, `(vertices, faces)` tuple). The IR does not embed the mesh, and analytic regions carry no
triangles, so the faceted fallback can only be built when the caller passes it. Region triangle ids index
this mesh. **Callers that went through `convert` must pass the mesh** (the converter, #15, will wire
this); `mesh=None` is for callers that only have an IR. Without `mesh`, if any shell fails, `write`
writes nothing (no partial part, no file), returns `valid=False`, and says why in `issues` and
`shells[*].issues`; it still does not raise.

Building (v0, planes and `facets` only): a vertex with at least two incident plane regions is moved
onto those planes (least-squares, pulled toward the IR position along any free direction, so two planes
give the point on their line nearest the IR position and one plane plus facets is not snapped). Patch
vertices of a `facets` region that coincide with a moved IR vertex move with it. Plane-plane edges are
straight lines between vertices, and every boundary point (closed plane-plane boundaries included, measured
against both planes) must lie within the deviation limit of the built edge. Loops are chained
from the boundaries in each region's direction, faces are built on the analytic planes, sewn, and
passed through ShapeFix. Boundaries touching a `facets` region keep their polyline. An outer shell and
the cavity shells that name it become one solid; an open shell becomes a sewn shell, never a solid;
several outer shells become a compound. A region with a non-plane surface (cylinder, cone, sphere,
torus) is not supported yet and triggers the fallback.

Validation, per solid: `BRepCheck_Analyzer`, positive volume, and the largest shape tolerance within
`max_shape_tolerance`. If any shell fails, and `mesh` is given, the whole part is written as a faceted
solid: one planar face per source triangle, winding corrected per shell from its signed volume, merging
only exactly coplanar faces, sewn at 1e-6.

`write` never produces an invalid solid and does not raise for a hard IR. It raises only for I/O errors
and for an `ir` that fails `Ir.validate()`.

`WriteReport`:

| field | meaning |
|---|---|
| `valid` | Whether every outer shell was written and passed validation. |
| `solids` | Number of solids in the written compound. |
| `max_shape_tolerance` | Largest shape tolerance in the written shape. |
| `faces` | `FaceReport(region, surface_type, max_shape_tolerance, max_vertex_displacement, max_boundary_deviation)` for each written region (empty after a fallback). Tolerances are measured on the final healed shape. |
| `max_vertex_displacement` | Largest distance the writer moved a vertex from its IR position, over all shells. `0.0` for the faceted fallback, which uses source triangles as they are. |
| `max_boundary_deviation` | Largest distance from an IR boundary point to the edge the writer built. |
| `fallback` | `None`, or `"faceted"` when the whole part fell back. |
| `fallback_reason` | Why, when `fallback` is set. |
| `shells` | `ShellReport(shell, kind, valid, volume, max_shape_tolerance, issues, max_vertex_displacement, max_boundary_deviation)` per outer shell; `kind` is `"solid"` or `"shell"`. |
| `seams` | `SeamReport(regions, surface_type, points, max_gap)` for each facets/analytic boundary: the largest distance from the boundary points to the analytic surface. |
| `issues` | Reasons something was not written, including the failure when there was no mesh to fall back on. |

### `unmesh.ir`

The IR as dataclasses: `Ir`, `Region`, `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `Facets`,
`Adjacency`, `Boundary`, `Vertex`, `Shell`, `Tolerances`. `Ir.loads(text)` and `ir.dumps()` read and
write [canonical JSON](ir.md#canonical-json); `ir.validate()` raises `IrError` listing every
structural problem; `unmesh.ir.validate(ir)` returns the list instead. `Ir.loads` validates.

### `unmesh.convert` internals

The planar converter welds the input, groups triangles into regions by growing them while every
vertex stays within tolerance of the region's plane, merges adjacent coplanar regions, fits each
plane robustly (Huber IRLS), snaps normals to the world axes and to exact parallel and perpendicular
relations when the snapped plane still fits within the estimated noise, and moves every vertex onto
the planes it belongs to (one plane: projection; two: their line; three or more: their
least-squares point, with a conditioning guard). `report.max_deviation` is the largest distance any
vertex moved plus its remaining distance to its planes, so it bounds the distance between the input
mesh and the IR in both directions. Shells follow [IR § Non-manifold and open
input](ir.md#non-manifold-and-open-input).

### `unmesh.weld(tris, tolerance)`

Returns `(vertices, faces, source_triangles, report)`: the welded arrays, a `uint32` array giving the input
triangle index of each welded face, and the report dict (`input_corners`, `unique_vertices`,
`degenerate_dropped`).

## Rust: `unmesh-core`

No kernel dependency, no Python.

```rust
use unmesh_core::{convert, ConvertOptions, IndexedMesh};
use unmesh_core::ir::Ir;

let out = convert(&mesh, &ConvertOptions::default())?;   // ConvertOutput { ir, report }
let text = out.ir.to_canonical_json()?;
let ir = Ir::from_json(&text)?;                          // validates
```

Public surface:

- `mesh`: `Point`, `TriangleSoup`, `IndexedMesh`.
- `stl`: `read_stl`, `write_stl_binary`, `StlError`.
- `weld`: `weld(&TriangleSoup, tolerance) -> (IndexedMesh, Vec<u32>, WeldReport)`, `WeldReport`, `WeldError`. The
  `Vec<u32>` maps each welded face to its index in the input soup; degenerate triangles are dropped, so
  welded face `i` is input triangle `map[i]`. IR source triangle ids are input indices.
- `ir`: the IR types (`Ir`, `Tolerances`, `Shell`, `Region`, `Residual`, `Surface`, `Orientation`, `Source`, `ShellRole`, `VertexRole`,
  `Adjacency`, `Boundary`, `Kind`, `Vertex`), `IR_VERSION`, `validate`, `canonicalize`, `format_f64`,
  `IrError`. All types are `serde` `Serialize`/`Deserialize`; JSON field names equal the Rust names.
- `api` (re-exported at the crate root): `convert(&IndexedMesh, &ConvertOptions) -> Result<ConvertOutput,
  ConvertError>`, `convert_soup(&TriangleSoup, &ConvertOptions)` (the same for a triangle soup, which
  is what STL files and `(n, 3, 3)` arrays are), `ConvertOptions`, `Report`, `ConvertWarning`,
  `ConvertOutput`, `ConvertError` (`EmptyMesh`, `InvalidInput`). Source triangle ids in the IR are
  indices into the soup, or into `mesh.faces`.

The `ir_canon` example (`cargo run -p unmesh-core --example ir_canon < ir.json`) prints the canonical
form of an IR read from stdin; the cross-language tests use it.
