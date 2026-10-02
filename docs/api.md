# API

Three public surfaces: the Python package `unmesh`, the optional `unmesh.step` writer, and the Rust
crate `unmesh-core`. They share one contract, the [IR](ir.md).

Everything here is v0 and may change freely before the first release. `convert` and `step.write` are
specified now and implemented by later issues (#15, #5); until then they raise `NotImplementedError`
(Python) or return `ConvertError::NotImplemented` (Rust).

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
| `linear_tolerance` | `None` | Linear tolerance. `None` derives it from the mesh. Becomes `ir.tolerances.linear`. |
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
| `warnings` | List of `Warning(code, message)`. Codes: `degenerate_triangles`, `flipped_winding`, `repaired_winding`, `open_edges`, `non_manifold_edges`. See [IR § Non-manifold and open input](ir.md#non-manifold-and-open-input). |

`convert` raises `ValueError` for input it cannot read (no triangles, wrong array shape, non-finite
coordinates) and `OSError` for an unreadable file. It never raises for a hard-to-fit part: those regions
come back as `facets`.

### `unmesh.step.write(ir, path, options=None) -> WriteReport`

An optional extra: `pip install unmesh[step]` installs OCP. `import unmesh` never imports OCP;
`unmesh.step` imports it only when `write` runs, and raises `ImportError` with the install hint when
it is missing. `unmesh.step` is also reachable as an attribute of `unmesh`.

`options` is a `WriteOptions`: `max_shape_tolerance` (default `1e-3`), the largest OCCT shape
tolerance a written face may need before the writer falls back.

`write` never produces an invalid solid and does not raise for a hard IR. If building or validating the
analytic solid fails, it writes the faceted solid (one face per triangle, merging only exactly
coplanar faces) and says so. It raises only for I/O errors and for an `ir` that fails
`Ir.validate()`.

`WriteReport`:

| field | meaning |
|---|---|
| `valid` | Whether the written file passed validation (`BRepCheck`, positive volume, tolerance limit). |
| `solids` | Number of solids in the written compound. |
| `max_shape_tolerance` | Largest shape tolerance in the written shape. |
| `faces` | `FaceReport(region, surface_type, max_shape_tolerance)` for each written analytic face. |
| `fallback` | `None`, or `"faceted"` when the whole part fell back. |
| `fallback_reason` | Why, when `fallback` is set. |

### `unmesh.ir`

The IR as dataclasses: `Ir`, `Region`, `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `Facets`,
`Adjacency`, `Boundary`, `Vertex`, `Shell`, `Tolerances`. `Ir.loads(text)` and `ir.dumps()` read and
write [canonical JSON](ir.md#canonical-json); `ir.validate()` raises `IrError` listing every
structural problem; `unmesh.ir.validate(ir)` returns the list instead. `Ir.loads` validates.

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
- `weld`: `weld`, `WeldReport`.
- `ir`: the IR types (`Ir`, `Tolerances`, `Shell`, `Region`, `Residual`, `Surface`, `Orientation`,
  `Adjacency`, `Boundary`, `Kind`, `Vertex`), `IR_VERSION`, `validate`, `canonicalize`, `format_f64`,
  `IrError`. All types are `serde` `Serialize`/`Deserialize`; JSON field names equal the Rust names.
- `api` (re-exported at the crate root): `convert(&IndexedMesh, &ConvertOptions) -> Result<ConvertOutput,
  ConvertError>`, `ConvertOptions`, `Report`, `Warning`, `ConvertOutput`, `ConvertError`.

The `ir_canon` example (`cargo run -p unmesh-core --example ir_canon < ir.json`) prints the canonical
form of an IR read from stdin; the cross-language tests use it.
