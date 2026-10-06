# API

Four public surfaces: the Python package `unmesh`, the optional `unmesh.step` writer, the `unmesh`
command line, and the Rust crate `unmesh-core`. They share one contract, the [IR](ir.md).
`unmesh.convert_to_step` and the CLI run the whole pipeline and return the
[fidelity report](#fidelity-report).

Everything here is v0 and may change freely before the first release. `convert` fits planes,
cylinders, cones, spheres and tori (what it cannot fit stays `facets`);
`convert_from_labels` fits the same surfaces on a given segmentation. `step.write` is implemented for planes and `facets`
(#5).

## Python

```python
import unmesh

result = unmesh.convert("part.stl")  # -> Result(ir, report)
result.ir  # unmesh.ir.Ir
result.report.max_deviation

import unmesh.step  # needs `pip install unmesh[step]`

write_report = unmesh.step.write(result.ir, "part.step", mesh="part.stl")

conversion = unmesh.convert_to_step("part.stl", "part.step")  # the whole pipeline
conversion.outcome, conversion.exit_code, conversion.fidelity  # see Fidelity report
```

### `unmesh.convert(mesh_or_path, options=None) -> Result`

`mesh_or_path` is one of:

- a path (`str` or `os.PathLike`) to an STL file (binary or ASCII) or, by its `.obj` suffix, a
  Wavefront OBJ file (`v` and `f` lines only; polygons are fanned from their first corner, and
  `f` indices may be negative or carry `/vt/vn` parts, which are ignored). `unmesh.read_mesh(path)`
  returns either as an `(n, 3, 3)` array;
- a NumPy array of shape `(n, 3, 3)`: `n` triangles of three corners (a triangle soup);
- a tuple `(vertices, faces)` of arrays shaped `(v, 3)` float and `(f, 3)` integer.

`options` is a `ConvertOptions` (`None` means all defaults):

| field | default | meaning |
|---|---|---|
| `linear_tolerance` | `None` | Linear tolerance. `None` derives it from the mesh before segmenting: five times the estimated vertex noise σ, floored at the larger of 1e-6 of the bounding-box diagonal and 5e-7 of the largest absolute coordinate, and capped at 5e-4 of the diagonal. σ is curvature-independent: around every welded vertex the converter takes each smooth sector of its fan (triangles joined across edges under 15°), gathers the sector's 2-ring without crossing an edge of 15° or more or tilting past 60° from the sector normal, keeps only the points within three times the distance to the sixth-nearest of them (a long facet of a neighbouring surface reaches far past the region a quadric can model; when that leaves too few points for a quadric with three degrees of freedom the sector gives no sample), and fits a height-field quadric in the frame of the sector normal. On a neighbourhood of fewer than 12 points it keeps a plane instead when the quadric does not reduce the residual significantly (F < 3) or when there are too few points for three degrees of freedom; on 12 or more points it trims the worst quarter twice (least trimmed squares) and rescales by the Gaussian truncated variance and a measured selection factor, so σ matches the standard deviation of Gaussian noise (uniform noise reads about 5% high). Each sector contributes its residual variance and degrees of freedom; for the pooled aggregate below it contributes the plane fit's instead whenever the plane is adequate by the same F-test, since a flat patch then keeps three more degrees of freedom. σ combines two aggregates of those samples: the area-weighted median of the per-sample RMS (each median-unbiased for its degrees of freedom by Wilson-Hilferty), which is robust, and a pooled variance over the samples whose variance is plausible for that σ (inside the 1% and 99.999% chi-square quantiles, iterated from the median and capped at four times it), weighted by degrees of freedom per point, which is efficient. The pooled estimate dominates when the mesh carries little evidence (total degrees of freedom D well under 200) and the median when it carries much: σ² = λ·pooled² + (1 − λ)·median² with λ = 200² / (200² + D²). A quadric absorbs plane, cylinder, cone and sphere curvature, so a clean tessellation gives σ near zero and a noisy mesh gives σ near the noise's standard deviation along the normal. Becomes `ir.tolerances.linear`. |
| `angular_snap_deg` | 0.5 | See [IR § Tolerances](ir.md#tolerances). The cap widens under noise by `atan(3σ/width)` per region, with σ the quadric noise estimate above, clamped to a fifth of the final tolerance (with an explicit `linear_tolerance`, σ is a fifth of it). |
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

A clean mesh therefore gets the floor (1e-6 of the bounding-box diagonal, so the tolerance scales
with the part's units), curved or not, and a noisy one a tolerance that follows its noise, so snapping
(see [IR § Tolerances](ir.md#tolerances)) never moves a surface by more than the data justifies.
Two limits follow from the median on well-sampled meshes. Noise confined to less than half of the area (for example, only on
the planar faces) is not seen, and the tolerance stays near the floor. And where a surface's facets
are long and thin around the normal (the coarse ring direction of a torus), a quadric cannot follow
even the near points, so curvature reads as noise and the tolerance rises toward the 5e-4 cap. Tangent
seams between different surfaces no longer do: on a box with every edge and corner filleted,
tessellated coarsely, nearly every vertex sits on such a seam, and only the near-point limit keeps
the neighbouring surface's far facet ends out of the fit.

`convert` raises `ValueError` for input it cannot read (no triangles, wrong array shape, non-finite
coordinates) and `OSError` for an unreadable file. It never raises for a hard-to-fit part: those regions
come back as `facets`.

### `unmesh.convert_to_step(mesh_or_path, step_path, options=None, *, unit="mm", write_options=None, measure=True) -> Conversion`

The whole pipeline in one call: read the mesh (any form `convert` accepts), scale it to millimetres
(`unit` is `"mm"` or `"in"`; an explicit `options.linear_tolerance` is in the input's unit and is
scaled with it), `convert`, then `unmesh.step.write(ir, step_path, write_options, mesh=...)` with the
scaled mesh, so the faceted fallback and the input-volume checks are always available. The STEP is
always in millimetres. With `measure`, a valid written file is then read back and sampled against the
input mesh (see `deviation.measured` below). It needs `unmesh[step]`, and checks for OCP before
reading the mesh: a missing OCP raises `ImportError` with the install hint. Bad input raises
`ValueError` or `OSError` as `convert` does; a hard part never raises.

`Conversion` has `ir`, `report` (the converter's `Report`), `write` (the `WriteReport`), `outcome`,
`exit_code` and `fidelity` (the [fidelity report](#fidelity-report) as a dict;
`unmesh.pipeline.dumps(fidelity)` writes it as JSON with non-finite numbers as `null`). `outcome`:

| `outcome` | exit code | meaning |
|---|---|---|
| `analytic` | 0 | Written and verified, every region an analytic face. |
| `mixed` | 1 | Written and verified, with one or more `facets` regions as triangle faces next to analytic faces. |
| `faceted` | 2 | Written and verified as a whole-part faceted solid: the writer fell back, or the converter found no analytic region at all (an open or non-manifold mesh, for example). |
| `error` | 3 | Nothing usable was written: bad input, a missing OCP, or a `WriteReport` that is not `valid` (a part that failed and had no fallback, or a written file whose read-back does not match; such a file is left on disk). |

### `unmesh.step.write(ir, path, options=None, *, mesh=None, verify=True) -> WriteReport`

An optional extra: `pip install unmesh[step]` installs OCP. `import unmesh` never imports OCP;
`unmesh.step` imports it only when `write` runs, and raises `ImportError` with the install hint when
it is missing. `unmesh.step` is also reachable as an attribute of `unmesh`.

`options` is a `WriteOptions`:

- `max_shape_tolerance` (default `1e-3`): the largest OCCT shape tolerance a written shell may need before the writer falls back.
- `max_deviation` (default `5e-3`, absolute, in mesh units): the cap on how far the writer may move geometry. The limit used is `min(5 * ir.tolerances.linear, max_deviation)`. A vertex further than that from its IR position, or a boundary point further than that from the built edge, fails the shell.
- `max_seam_gap` (default `1e-4`, absolute): the largest distance a straight seam edge between a `facets` patch and a curved face may keep from the curved surface. Patch boundary chords are subdivided until every piece is within it.
- `readback_memory_mb` (default `2048`) and `readback_timeout_s` (default `120`): the memory cap and timeout of the OCCT read-back process (see below).

`mesh` is the source mesh the IR was converted from, in any form `convert` accepts (path, `(n, 3, 3)`
array, `(vertices, faces)` tuple). The IR does not embed the mesh, and analytic regions carry no
triangles, so the faceted fallback can only be built when the caller passes it. Region triangle ids index
this mesh. **Callers that went through `convert` must pass the mesh** (the converter, #15, will wire
this); `mesh=None` is for callers that only have an IR. Without `mesh`, if any shell fails, `write`
writes nothing (no partial part, no file), returns `valid=False`, and says why in `issues` and
`shells[*].issues`; it still does not raise.

Building, planar shells (planes and `facets` only): a vertex with at least two incident plane regions is moved
onto those planes (least-squares, pulled toward the IR position along any free direction, so two planes
give the point on their line nearest the IR position and one plane plus facets is not snapped). Patch
vertices of a `facets` region that coincide with a moved IR vertex move with it. Plane-plane edges are
straight lines between vertices, and every boundary point (closed plane-plane boundaries included, measured
against both planes) must lie within the deviation limit of the built edge. Loops are chained
from the boundaries in each region's direction, faces are built on the analytic planes, sewn, and
passed through ShapeFix. Boundaries touching a `facets` region keep their polyline. An outer shell and
the cavity shells that name it become one solid; an open shell becomes a sewn shell, never a solid;
several outer shells become a compound.

Building, curved shells (any cylinder, cone, sphere or torus region in the outer shell or its cavities):
the shell is assembled from shared edges, without sewing or ShapeFix.

- **Vertices** touching a curved region are moved onto every analytic surface that meets there
  (iterated least squares on the tangent planes, pulled toward the IR position). A direction in
  which the surfaces' normals do not constrain the point (they are within about 1.6 degrees of
  each other, as at a tangent junction) keeps the IR position instead of drifting along it. A
  junction on a `tangent` boundary is then solved for the point on every surface where each tangent
  pair's outward normals agree, and a vertex on planes is put back on them exactly. Once the edge
  curves are built, a vertex where two or more of them meet settles on their common point (when
  that is closer to all of them and within the deviation limit).
- **Transversal edges** are the OCCT intersection of the two surfaces (`GeomAPI_IntSS`). The branch
  nearest the boundary polyline is kept and trimmed at the IR vertices in the polyline's direction;
  every boundary point must lie within the deviation limit of the trimmed edge. When OCCT returns
  the curve as several branches (cylinder-cylinder, for example) whose union covers the polyline,
  or as a closed curve that is not periodic, the edge is the interpolation of the polyline points
  (and their midpoints) moved onto both surfaces, recorded in `edge_fallbacks` with kind
  `interpolated` and its largest distance to OCCT's branches, sampled at no fewer than 400 points
  along the built curve. When the intersection fails, strays further than
  the limit from the polyline, or cannot be trimmed along it, the same projected polyline is used
  and recorded in `edge_fallbacks` with kind `projected`. An interpolated curve is split at polyline
  corners (a turn over 30 degrees and over three times either neighbour's), where intersection
  branches meet at a kink, and the pieces are joined with C0 continuity.
- **Tangent edges** (`tangent` boundaries next to a curved region) are not intersected: the two
  surfaces touch, so their intersection is ill-conditioned. Each polyline point is moved onto both
  surfaces and slid across the boundary to where their outward normals agree, which fixes the
  tangency's position to the accuracy of the surfaces rather than of the mesh boundary (a boundary
  within `tolerances.linear` of both surfaces can sit `sqrt(2 R linear)` to one side). A line (open
  boundaries only) or circle is fitted to those points, and kept when it lies within the deviation
  limit of both surfaces and of the moved points (a line wins a near-tie). Otherwise the moved points
  and their midpoints are interpolated as a B-spline. A tangent run that ends at a `kind_change`
  vertex is the grazing part of a transversal intersection (an equal-radius tee), so its points are
  moved onto the intersection instead and interpolated. Both faces share the one edge; its
  tolerance covers the largest distance of the curve from either surface, and a vertex's tolerance
  twice its distance from each curve end that meets it. Each is reported in
  `tangent_edges`.
- **A sphere region with no closed boundary** (a corner blend) takes its axis through a vertex
  where two great-circle edges meet, so that vertex is the pole, those edges are meridians, and the
  face closes there with a degenerate edge, as OCCT's own fillets do. Tangent circles on a sphere
  are moved onto it (onto a great circle when within the deviation limit of one), the pole is the
  exact intersection of the two meridians, and tangent edges ending there are refitted to it.
- **`facets` regions next to a curved region** share straight seam edges with it. The interior
  boundary points are moved onto the curved surface (the patch vertices with them; a vertex where
  only one analytic surface meets is moved onto it), and every chord of the boundary is split at
  the surface point nearest its midpoint, recursively, until each piece is within `max_seam_gap` of
  the surface (sampled at seven interior points). A closed boundary that winds around the surface
  also gets a point where the surface's seam crosses it. The patch triangle on a split chord is
  fanned from its opposite corner (from its centroid when more than one of its sides is split), and
  a fan triangle that folds over fails the shell. The curved face and the triangles use the same
  edges, so there is no gap and no T-junction; each edge's tolerance covers its distance from the
  curved surface (1.01 times the sampled distance, at least `1e-7`). Each seam is reported in `seams`.
  Two alternatives were measured and rejected (#34), on the oracle IRs of five seeds of every
  curved and chamfer_fillet family with one seeded non-planar region forced to `facets` (75 parts,
  deflection (0.01, 0.2)): keeping the mesh chords as edges with their tolerance loosened to cover
  the gap, and trimming the curved face by the chord polyline projected onto it (the triangles then
  share the projected curves). Both need tolerances up to 5e-3, so 29 of the 75 parts exceed the
  `1e-3` cap and fall back whole; subdividing needs at most 8.8e-5, writes all 75 mixed, and adds
  13% more triangle faces.
- **Faces** lie on the IR surface, oriented by the IR (`reversed` faces are built on the natural
  surface and reversed), with explicit pcurves. A pcurve is OCCT's projection of the edge, re-anchored
  on exact surface parameters; where that projection is not exact, the pcurve interpolates the exact
  surface parameters of the curve sampled twice per knot span at its own parameters, whichever is
  closer to the edge. (Denser sampling gave pcurves of ~4000 intervals, on which OCCT's area and
  volume integration silently lost accuracy: 1.3e-5 relative on a boss meeting a pipe.)
  On a periodic surface the seam is an isoparametric
  line: through the vertex of a wrapping loop made of open edges when there is one, otherwise placed
  away from the face's other boundaries; closed loops get their vertex where the seam crosses them,
  and coaxial neighbours share the seam's half-plane. A face with one wrapping loop closes at a
  sphere pole or cone apex with a degenerate edge; a sphere or torus region with no boundary is the
  whole surface.
- **Orientation check:** each face is probed just inside every boundary, on the side the IR's loop
  direction puts it, by classifying the point's exact surface parameters against the face: at least
  one probe must be inside and none decisively outside. A boundary whose two IR orientations meet at
  an angle more than 10 degrees from its recorded `dihedral_deg` and nearer its supplement (one
  orientation flipped) fails the shell. The outer shell must enclose positive
  volume and each cavity negative. Nothing flips a face to fix it: a mismatch fails the shell. ShapeFix
  runs only on a solid that `BRepCheck_Analyzer` rejects, and the orientation is checked again after it.

**Facets patches.** A part whose `facets` regions hold more than half of the IR's triangles (and
that also has analytic regions) is written as the whole-part faceted solid when `mesh` is given,
with `fallback_reason` naming the share: one `ADVANCED_FACE` per triangle on its own plane costs
about 2.4 KB, so the mixed solid is several times larger and slower than the faceted one and no
more exact (circular_fillet seed 6 with its torus forced to `facets`: 28.3 MB in 13.3 s mixed, 5.9
MB in 1.2 s faceted). Every boundary point between a patch and its neighbour must be one of the
patch's vertices (else the shell fails, naming the patch). In a closed shell, every edge a patch
triangle shares must be used in opposite directions by its two faces, checked on the built faces
before any ShapeFix runs, on planar and curved shells alike; a flipped patch fails the shell
instead of being re-oriented by healing. When `mesh` is given, each patch triangle is matched to
its source triangle (`triangles[i]` to `faces[i]`, up to a rotation of its corners) and every
written corner must lie within the deviation limit of the source corner; each seam split point
must lie within `max_seam_gap` of its curved surface and within the deviation limit plus that
region's `residual.max` (the chord sagitta) of the source triangle's side. A patch beyond either
bound fails the shell, named, and the measured largest distance is the patch's
`max_vertex_displacement` in `faces`. Without `mesh`, a patch is written as given and unchecked.

Validation, per solid: `BRepCheck_Analyzer`, positive volume, and the largest shape tolerance within
`max_shape_tolerance`. When `mesh` is given, each analytic solid's volume is also compared with the
mesh's: the outer shell's triangles (IR shell grouping) enclose `|V|`, each cavity's subtract theirs, and
the two must agree within `1e-6 * |V| + area * d`, where `d` is the largest of the regions' residual
max, the boundary deviation, the vertex displacement and the shape tolerance. A mismatch (an IR
orientation that turns a pocket into a boss, say) fails the shell, named as `analytic volume X differs
from mesh volume Y`, and the part falls back to faceted. Then each written face is checked on its
own, so a small flipped feature cannot hide in a large part's volume budget: its divergence-theorem
flux `∫ (x - o)·n dA / 3` about the centre `o` of its region's mesh triangles (the written face
triangulated at the smallest region deviation, floored at 1e-4 of the part's extent) must match the
same sum over the region's mesh triangles (wound outward) within `(A + P L) d + 1e-9 A L`, where `A`
is the face's area, `P` its perimeter, `L` its farthest point from `o`, and `d` the region's residual
max, boundary deviation, vertex displacement and shape tolerance plus the triangulation deflection. A
face built on the wrong side changes the flux's sign; the region is named in the reason. Without `mesh` nothing ties the written volume
to the input: the IR alone cannot tell a flipped pocket from a boss, and `volume_checked_against_input`
is `False`. If any shell fails, and `mesh` is given, the whole part is written as a faceted
solid. The mesh is welded at `ir.tolerances.vertex_merge`. The IR's shell roles decide orientation: each
`outer` shell's winding is flipped to positive signed volume and each `cavity` shell's to negative. No
OCCT healing or classification touches it. The fallback STEP is
emitted as text, without an OCCT transfer: `MANIFOLD_SOLID_BREP` (or `BREP_WITH_VOIDS` with
`ORIENTED_CLOSED_SHELL(.F.)` voids) of `CLOSED_SHELL`s, `ADVANCED_FACE`s on `PLANE`s bounded by
`EDGE_LOOP`s of straight `EDGE_CURVE`s, in an `ADVANCED_BREP_SHAPE_REPRESENTATION` (mm, uncertainty
1e-7); open shells go in a `SHELL_BASED_SURFACE_MODEL` related to it. Points are shared, output is
deterministic. One face per triangle, except that edge-connected exactly coplanar triangles whose union has
one simple boundary loop become one polygon face, and zero-area triangles are absorbed into their neighbour
across the long edge (repeated until none is left, so adjacent zero-area triangles chain). Faces on the convex hull are listed first in each outer shell: OCCT's reader re-orients
every solid from the first face whose probe ray gives a decisive answer, and on a rough, locally folded
mesh a probe from an arbitrary face can land on a fold and invert the solid; a hull face's probe cannot.

After writing, `write` verifies the file it wrote, in up to two independent ways, and says which ran
in `verified_by`.

**OCCT read-back** re-imports the file with OCCT's STEP reader, with healing on, as a consumer would
read it, and compares the solid count, the shell count, and each solid's volume and validity with
what it wrote. The volume tolerance is `1e-9 * |V| + area * shape tolerance`. Curved solids are
measured with OCCT's adaptive volume integration at precision 1e-9 (the default fixed-order rule was
off by 0.6% on faces bounded by kinked B-spline edges); planar and faceted solids use the default.
A file of up to 4 MB is re-imported in the calling process with the default healing (`reader`
`"occt"`). A larger file is re-imported in a short-lived child process (the same Python executable),
so the memory OCCT keeps is returned when it exits; the child is killed when its resident memory
passes `readback_memory_mb` or it runs past `readback_timeout_s`, and `readback.peak_rss_mb` records
the largest resident memory sampled. In the child, a faceted file is read with every healing step
except `ShapeFix_Shell`'s face-orientation repair (`reader` `"occt_no_face_orientation"`): that
repair is quadratic in the face count (478 s for the 108k-face imported-0207 with `noise_normal`,
more than 15 minutes for the 387k-face imported-0201), and the text check below proves that every
closed shell's faces are already consistently oriented, so on a file that passes it the repair has
nothing to do. The solid-orientation step that #81 found can invert a rough shell still runs.

**Text check** (faceted fallback only) parses the written STEP text with unmesh's own parser,
independently of OCCT and of the writer's data, as a strict grammar for exactly what the faceted
writer emits, so that a file that passes it is one an OCCT reader can load with the structure that
was written. The file must start with the writer's Part 21 header (`FILE_DESCRIPTION`, `FILE_NAME`,
the AP214 `FILE_SCHEMA`) and end with `ENDSEC;` and `END-ISO-10303-21;` (a truncated file fails);
every line of the DATA section must be one `#n = ...;` instance of an allowed entity in the
writer's exact form (any other entity, complex instance, blank or stray line, a `FACE_BOUND`, a
reversed bound, a changed unit or context string fails it); every reference must resolve to an
entity of the right type; and the product chain must be present once and connected:
`APPLICATION_CONTEXT`, `PRODUCT_CONTEXT` → `PRODUCT` → `PRODUCT_DEFINITION_FORMATION` →
`PRODUCT_DEFINITION` → `PRODUCT_DEFINITION_SHAPE` → `SHAPE_DEFINITION_REPRESENTATION` → the brep
representation (or the surface representation, with a `SHAPE_REPRESENTATION_RELATIONSHIP` linking
it when both exist), whose context carries the length uncertainty and the millimetre, radian and
steradian units. Every solid must be an item of the brep representation and every surface model of
the surface representation, once. It then checks: every edge loop is connected and passes through
no vertex twice, every face's loop lies within `max_shape_tolerance` of its plane and winds
counter-clockwise about the plane's normal, every edge of a closed shell is used exactly once in
each direction and by no other shell, every edge of an open shell at most once in each direction,
every face is in exactly one shell and every closed shell in exactly one solid, the outer shell of
each solid encloses a positive volume and each void a negative one, and each solid's signed volume
(the divergence sum over its polygon faces) matches the volume written from the mesh within the
read-back tolerance. Its result is `text_check`, a `ReadBack` with `reader` `"text"`; its
`max_shape_tolerance` analogue is the largest distance of a vertex from its face's plane or its
edge's line.

A faceted file whose OCCT read-back would need more than `readback_memory_mb` (estimated as
300 MB plus 26 times the file size, measured at 25 times on 55 MB and 104 MB faceted files) is
verified by the text check alone: `verified_by` is `"text"` and `readback_skipped` says why. A faceted
read-back stopped by the memory cap or the timeout is likewise reported in `readback_skipped`, with
the text check as the verification. The residual risk of text-only verification is the one the
read-back exists for: an OCCT reader whose healing inverts or rejects the file (#81) goes unnoticed.
Faces on the convex hull are written first so that the solid-orientation probe cannot land on a fold,
which is what inverted the #81 files. An analytic or mixed file has no text check, so a read-back
stopped by the cap or the timeout leaves it unverified: `verified_by` is `None`, `valid` is `False`
and `issues` says why. A large analytic or mixed file still gets the full healing, quadratic
face-orientation repair included, so it can hit the default 120 s timeout and be reported invalid
for that reason, not because the writer built it wrong. The memory cap is sampled with `psutil`,
which `unmesh[step]` requires; the parent kills the child on any exception of its own.

Measured on an M-series Mac under load, before and after (#106): imported-0201 (393,588 triangles,
a 347 MB faceted file) went from more than 15 minutes of read-back to 15 s of text check, with the
whole write peaking at 1.1 GB instead of 2.6 GB before any read-back; imported-0207 with
`noise_normal` (108,668 triangles) from 478 s and 2.4 GB to 3 s and 0.7 GB.

Any mismatch makes the report invalid and is named in `readback.issues` or `text_check.issues`,
`issues` and the shell's `issues`; the file is left on disk. For the faceted fallback,
`shells[*].volume` is the re-imported volume (the parsed one when OCCT did not run) and
`max_shape_tolerance` the re-imported tolerance. The verification is on by default because
OCCT-based consumers (OttoCAM, FreeCAD) read through the same healing that can invert a file.
`verify=False` skips it for callers that accept unverified output: the report then has
`verified=False`, `verified_by=None`, `readback=None`, `text_check=None`, and `valid` covers
construction only.

`write` does not raise for a hard IR. It raises only for I/O errors and for an `ir` that fails
`Ir.validate()`.

`WriteReport`:

| field | meaning |
|---|---|
| `valid` | Whether every outer shell was written, passed validation, and read back as written. |
| `solids` | Number of solids in the re-imported file (the written count when `verify=False`). |
| `max_shape_tolerance` | Largest shape tolerance in the written shape. |
| `faces` | `FaceReport(region, surface_type, max_shape_tolerance, max_vertex_displacement, max_boundary_deviation)` for each written region (empty after a fallback). Tolerances are measured on the final healed shape. |
| `max_vertex_displacement` | Largest distance the writer moved a vertex from its IR position, over all shells. For the faceted fallback, the largest distance the weld moved a source vertex. |
| `max_boundary_deviation` | Largest distance from an IR boundary point to the edge the writer built; for a tangent edge, from the point moved onto the tangency (the IR point's own distance is the edge's `boundary_distance`). |
| `edge_fallbacks` | `EdgeFallback(regions, reason, max_deviation, kind, intersection_distance)` for each transversal edge built from the projected boundary polyline instead of a trimmed surface intersection (empty after a faceted fallback). `max_deviation` is the larger of the curve's distance to both surfaces and the boundary points' distance to it. `kind` is `projected` (the intersection failed, strayed or could not be trimmed) or `interpolated` (the intersection came in several branches); for `interpolated`, `intersection_distance` is the largest distance from the built curve to the nearest branch, otherwise `None`. |
| `tangent_edges` | `TangentEdge(regions, curve, max_deviation, boundary_distance)` for each tangent edge (empty after a faceted fallback). `curve` is `line`, `circle` or `bspline`; `max_deviation` is the larger of the curve's distance to both surfaces and the moved boundary points' distance to it; `boundary_distance` is the IR boundary points' distance to it, which across a tangency measures the mesh boundary's sideways uncertainty, not an error of the edge. |
| `fallback` | `None`, or `"faceted"` when the whole part fell back. |
| `fallback_reason` | Why, when `fallback` is set. |
| `shells` | `ShellReport(shell, kind, valid, volume, max_shape_tolerance, issues, max_vertex_displacement, max_boundary_deviation)` per outer shell; `kind` is `"solid"` or `"shell"`. |
| `seams` | `SeamReport(regions, surface_type, points, max_gap, chord_gap, inserted, tolerance)` for each facets/analytic boundary. Next to a plane, `max_gap` is the largest distance from the boundary points to the plane and the rest are 0. Next to a curved surface, `max_gap` is the largest distance from the written seam edges to the surface, `chord_gap` the same for the unsplit boundary chords (the gap the subdivision closed), `inserted` the number of points added, and `tolerance` the largest tolerance of the seam's edges. |
| `faceted_regions` | Number of regions written as triangle faces: the `facets` regions of a mixed solid, or every region after a faceted fallback. |
| `faceted_faces` | Number of triangle faces written for `facets` regions, after seam splits (0 after a faceted fallback). |
| `issues` | Reasons something was not written, including the failure when there was no mesh to fall back on, any read-back or text-check mismatch, and a write that could not be verified. |
| `readback` | `ReadBack(ok, solids, shells, volume, expected_solids, expected_shells, expected_volume, volume_tolerance, issues, reader, peak_rss_mb)` from re-importing the written file with OCCT, or `None` when nothing was written, `verify=False`, or the read-back was skipped or stopped. `reader` is `"occt"` or `"occt_no_face_orientation"`; `peak_rss_mb` is the child process's largest sampled resident memory (`None` in-process). |
| `text_check` | The same `ReadBack` from the text check (`reader` `"text"`, faceted fallback only), or `None`. |
| `verified` | Whether any verification of the written file ran. |
| `verified_by` | `"occt"`, `"text"`, `"occt+text"`, or `None` when nothing was verified. |
| `readback_skipped` | Why the OCCT read-back did not run or did not finish (its estimated memory over the cap, the cap reached, the timeout), else `None`. |
| `volume_checked_against_input` | Whether `mesh` was given, so each written solid's volume was compared with the input mesh's (or is the mesh, after a faceted fallback). |
| `timings` | Seconds: `write_s` (everything up to the written file) and `readback_s` (the whole verification: text check and read-back), when they ran. |

### `unmesh.ir`

The IR as dataclasses: `Ir`, `Region`, `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `Facets`,
`Adjacency`, `Boundary`, `Vertex`, `Shell`, `Tolerances`. `Ir.loads(text)` and `ir.dumps()` read and
write [canonical JSON](ir.md#canonical-json); `ir.validate()` raises `IrError` listing every
structural problem; `unmesh.ir.validate(ir)` returns the list instead. `Ir.loads` validates.

### `unmesh.convert_from_labels(mesh_or_path, labels, options=None) -> Result`

Converts with a given segmentation instead of segmenting: `labels` is one integer per input triangle
(the oracle face ids of a labelled tessellation, for example). Labels must be contiguous from 0, every
id in `0..max` used, and below `2**32 - 1`; anything else raises `ValueError`, as does a label array
that is not one-dimensional, not integer, or not one per triangle. Labelled regions are never merged
with each other, and a label whose triangles fall apart into several edge-connected pieces becomes one
region per piece. Each region gets the most parsimonious surface that fits its vertices within
`tolerances.linear`: a plane, else a cylinder, else a cone, else a sphere, else a torus, else it is kept
as `facets`. A torus patch that a cylinder, cone or sphere also fits within the tolerance is therefore
the simpler surface, and so is a band of two vertex rings, which every one of them fits. The automatic
`convert` finds its own curved regions (see [internals](#unmeshconvert-internals)) and fits them the
same way.

- **Cylinder**: the axis is the smallest eigenvector of the area-weighted scatter of the triangle normals
  (they lie on a great circle of the Gaussian sphere), the centre and radius a circle fit of the vertices
  projected along it, refined by Levenberg-Marquardt on the vertex distances with Huber weights.
- **Cone**: the normals lie on a small circle, so the axis is the smallest eigenvector of their centred
  scatter; the apex is the least-squares intersection of the triangle planes (every chord facet of a
  cone passes through its apex), the half-angle the mean angle of the vertices about the axis; then the
  same refinement.
- **Sphere**: `|p|² + D·p + E = 0` is linear in `D` and `E`, so a least-squares solve over the
  vertices gives the centre and radius exactly on exact vertices; then the same refinement (the axis
  direction plays no part, so only the centre and radius move).
- **Torus**: the refinement fits the centre, axis, major and minor radius together, from the best of
  three starts by largest vertex residual. (1) The spine circle traced through the centres of
  curvature: for a trial tube radius `s` (either sign, for either orientation), each triangle's
  centroid moved by `s` against its normal lies on the spine when `s` is the minor radius; a log grid
  then golden-section search on `s` minimises the misfit of a 3D circle (plane, then circle in it)
  through those points, which gives the centre, axis and major radius, and the mean vertex distance to
  that circle gives the minor radius. (2) The axis of revolution: every normal line of a surface of
  revolution meets its axis, a condition linear in the axis' Plücker coordinates, after which the
  vertices' (distance, height) about the axis lie on the tube's circle. (3) For a patch too small to
  fix either circle, the local shape: a quadric through the vertices gives the point, normal and
  principal curvatures at the patch's middle; the larger is the tube's, and the tube angle (hence the
  axis and the major radius) is the grid value with the smallest largest vertex residual. A torus
  needs `major_radius > minor_radius`, and its vertices must need at least four rings (bands of the
  tube angle `2 tolerances.linear` wide) to hold them all: any three coaxial circles lie on some torus.
- **Tessellation law**: when the rulings of a cylinder are cut into equal chords `c` with equal
  turning angles, including a partial arc, the turning angle gives `N` and the radius is
  `c / (2 sin(π/N))`. Chords and turning angles are measured between rulings, so the estimate does not
  depend on the fitted centre. It replaces the least-squares radius only when (a) chords and turning
  angles are equal within what `tolerances.linear` allows, (b) the mean turning angle is `2π/N` within
  its own measurement uncertainty and the chord count agrees with `N` (exactly `N` for a closed ring,
  fewer for an arc), and (c) with the radius fixed at the law's value and the axis refitted, the
  vertices fit within the tolerance and their RMS is no worse than the least-squares RMS beyond the
  allowance for one fewer free parameter (`rms² ≤ rms_lsq² (1 + 4/(n - 5))`). Exact vertices therefore
  keep their exact least-squares radius (an arc of 90.05° in 9 chords is refused), and the law matters
  on short noisy arcs, where the least-squares radius is poorly conditioned. A law radius stays fixed
  through the coaxial and world-axis refits below.
- **Coaxial snapping**: cylinders, cones and tori whose axes are parallel within `angular_snap_deg` and
  collinear within three times `tolerances.linear` are refitted with one shared axis, and cylinders among
  them with radii within the tolerance share one radius, so a bore split across several regions is one
  cylinder. A group's axis within `angular_snap_deg` of a world axis is snapped to it. Either is kept
  only when every member still fits within the tolerance, its RMS within half of it, and every cone's
  half-angle stays in (0, π/2).

Vertices of a tessellated curved face lie on the surface, but the triangles between them do not:
`residual.max` and `report.max_deviation` include each triangle's chord sagitta, so a coarse cylinder
reports a deviation near `r (1 - cos(π/N))` while its fitted radius is exact. The sagitta is computed
exactly, never sampled: for a cylinder as the 2D distance from the axis to the projected triangle, for
a cone as the minimum of the convex signed distance `ρ cos α - h sin α` over the triangle (corners,
the stationary points along each edge, and the point where the axis pierces the triangle), for a
sphere from the point-triangle distance to the centre and the farthest corner. For a torus it is a
provable upper bound, not the exact value: with `g` the distance to the spine circle, the distance to
the torus is `|g - minor|`. The largest `g` is bounded by a convex function of the point (the tube
radius `(ρ - R)` split into its outward and inward parts, the inward one with `ρ` replaced by its
support `d·q` along a direction `d` across the axis) whose maximum is at a corner; its slack grows with
the square of the triangle's span about the axis, so the triangle is split exactly at its edge midpoints
until the bound is within 2% of a value the true maximum reaches. The smallest `g` is the distance from
the triangle to the arc of the spine over the triangle's azimuths, bounded below by covering the arc
with pieces, each inside the triangle of its chord and end tangents, and taking the exact
triangle-triangle distance. On random tori and triangles the bound is never below a dense numeric
maximum; it is within 4% of it for minor/major radius ratios of 0.02 to 0.92, and up to about 1.5×
on thin tori (smaller ratios); a triangle spanning more than 90° about the axis falls back to its corners
plus its longest edge. `residual.rms`
is the RMS over the region's vertices only, without the sagitta.

**Ambiguity sets (n-gon prism versus coarse cylinder, chamfer versus one-segment fillet):** on this
path the labels decide. A regular n-gon prism whose sides carry one label each stays n planes; the same
sides under one label become one cylinder with the exact circumradius and an honest deviation equal to
the polygon's sagitta. The automatic `convert` has to choose without labels, and decides by the crease
between neighbouring facets: planes meeting at 15° or more stay planes, so a regular polygon with up to
24 sides stays a prism, and three or more facets meeting at smaller creases become one cylinder (or
cone) when they fit one. A coarse cylinder exported with fewer than 25 segments around therefore comes
back as a prism. A one-segment fillet stays one plane (the chamfer reading) and a two-segment fillet
stays two planes: any two planes meeting at a crease fit a cylinder, so curved recovery needs at least
three facets.

### `unmesh.convert` internals

**Curved regions.** After the planar stages below, plane regions that meet across smooth creases (under
15°, and more than the angle their own noise allows) are grown into spheres, tori, cylinders and cones.
Regions on a surface curved in two directions (a sphere or a torus: normals of the smooth neighbours,
weighted by shared boundary length and centred, differ along two directions and bend both ways along
the second, and the region is not much larger than those neighbours) seed first: the flagged regions
nearest the seed, breadth first, up to 16, 32 or 64 vertices, unless a cylinder or cone already fits
them, fitted as a sphere else a torus (above). The group grows as below; when it stalls it is refined on
its members, else fitted afresh on their union (a small seed's torus is right only locally), and
retries what it rejected. Then cylinders and cones grow from the remaining regions. A seed is a
region and its two most opposed smooth neighbours, extended along the same turning direction to 12,
24 or 48 vertices (a short arc of a staggered cone band does not determine its axis); it is fitted from
the normal-scatter and circle estimates, plus, for cones, an apex and half-angle solved linearly from the
vertices about an axis taken from the normals or from the rows of vertices (each row a circle across the
axis). Under noise a short arc of a narrow cone band also fits a tilted cylinder, so a seed whose largest
residual exceeds a tenth of `tolerances.linear` keeps widening to the larger sizes while one surface still
fits it, and on such a widened chain (or a chain of 48 vertices or more whose plane regions are themselves
noisy) a cone is also refined from its normal-based estimate when the cheap screen rejects it. The group
then absorbs neighbouring regions whose vertices are within `tolerances.linear` of the
surface, whose plane normal lies within the spread of the surface normals over its vertices, and whose
chord sagitta is at most 16 times the group's median: tessellated curves cut every chord at a similar
deflection, while a flat face is never absorbed into a large cylinder through its four edges (the side
of a rounded rectangle and the strips beside it always lie on one). A stalled group is refitted and
retries what it rejected (three times, or for as long as the retries keep adding regions); adjacent groups
are joined when the surface of either, or a fresh fit of their union (a cylinder or cone, else a cone
refined from its normals), holds both; the result is refitted
(tessellation law included) and joins the coaxial snapping above. On a sphere or a torus the normal
test allows twice the spread plus the chord sagitta across the facet's width (a facet whose corners sit
on different rings tilts beyond the normals at its corners, and a sliver's normal is set by how far its
middle bows), and the sagitta compared is the distance at the corners, edge midpoints and centroid.
Where two surfaces meet tangentially the planar stage can join a strip of one with a sliver of the next,
or keep a few nearly coplanar triangles of a sphere or torus together, so that region fits neither:
afterwards each triangle of an ungrouped region of at most 64 triangles moves into an adjacent curved
group it fits as members do, and groups that then touch are joined when one refitted surface holds
both. Under noise (a seed fit whose rms exceeds a tenth of `tolerances.linear`) such a strip never joins a
cylinder or cone group while it holds a triangle that fits the adjacent sphere or torus group; instead growth runs again (up to three rounds) once those triangles have moved: the
groups of earlier rounds come in fixed, grow into the plane regions beside them and join new groups, and
lone triangles flagged `facets` are planes again until a round leaves them ungrouped. A flagged region left ungrouped becomes `facets`, as does a lone triangle whose three neighbours
all meet it at smooth creases; a cylinder or cone group with two other cylinder or cone groups each
sharing 15% of its boundary is a slice of a sphere or torus that was not recovered (rings of a torus
are cones) and becomes `facets` too. Last, a region of at most two triangles that is not curved
joins an adjacent curved region when that surface passes within `tolerances.linear` of each of its
vertices and the region's own largest chord sagitta (or the tolerance, if larger) bounds the remnant's:
the long slivers a tessellator leaves where a curved face meets another (a bore through a cone) have
every corner on the surface and a sagitta like their neighbours', which growth refused. A flat cut into
the surface (a D-flat on a boss or in a bore) also has its corners on it, but its sagitta is the flat's
depth, so it stays a plane. The surface is not refitted; its `residual.max` takes their vertex
distances.

**Vertices and boundary kinds.** A mesh vertex where three or more regions meet (a junction) is moved
to the least-squares point of its incident analytic surfaces: Gauss-Newton to convergence, with the
directions the surfaces do not determine (eigenvalues of the normals' matrix below 1e-6) pulled to the
mesh vertex, so on an intersection curve it takes the point nearest the vertex. The writer's own
intersection solve then starts at its fixed point. The move is kept when it is at most ten times
`tolerances.linear` and the point's distance to the input mesh (the two-ring of the vertex) plus its
largest distance to those surfaces is at most five times `tolerances.linear`; that sum is the vertex's
contribution to `report.max_deviation`. Next to a `facets` region the vertex is a corner of the
patch, so its whole displacement counts instead of its distance to the mesh, and a junction of a
`facets` region with a curved one is not moved at all (the patch would then stand in for the curved
surface across a moved chord). Otherwise the vertex keeps its projection below. A boundary's kind follows [IR § Tangent versus
transversal](ir.md#tangent-versus-transversal): the dihedral of the fitted surfaces is sampled at each
polyline node (for a `facets` side, against the facet carrying the adjacent boundary edge) and runs
are split with the hysteresis band there. A `kind_change` vertex is then moved, within the polyline
segments next to its node, onto the point of both surfaces where their dihedral crosses
`tangent_threshold_deg` (onto both surfaces at the node when it does not cross there), with the same
acceptance and reporting as a junction. `Vertex.source_positions` keeps the mesh position, so every
vertex's movement can be read from the IR.

The planar converter welds the input, groups triangles into regions by growing them while every
vertex stays within tolerance of the region's plane, merges adjacent coplanar regions, fits each
plane robustly (Huber IRLS), re-grows regions that still exceed the tolerance at strict tolerance
over the union of creased regions and re-merges, snaps normals to the world axes and to exact
parallel and perpendicular relations when the snapped plane still fits within the estimated noise
(allowing `angular_snap_deg + atan(3σ/width)` per region), and moves every vertex onto
the planes it belongs to (one plane: projection; two: their line; three or more: their
least-squares point, with a conditioning guard). `report.max_deviation` is the largest distance any
vertex moved plus its remaining distance to its planes, so it bounds the distance between the input
mesh and the IR in both directions. Shells follow [IR § Non-manifold and open
input](ir.md#non-manifold-and-open-input).

### `unmesh.weld(tris, tolerance)`

Returns `(vertices, faces, source_triangles, report)`: the welded arrays, a `uint32` array giving the input
triangle index of each welded face, and the report dict (`input_corners`, `unique_vertices`,
`degenerate_dropped`).

## Command line

`pip install unmesh[step]` installs the `unmesh` command (also `python -m unmesh`):

```sh
unmesh convert in.stl out.step [--unit mm|in] [--tolerance T] [--report report.json] [--no-measure]
unmesh --version
```

`convert` runs `unmesh.convert_to_step`. The input is an STL file, or an OBJ file by its suffix.
`--unit` is the unit of the input coordinates (default `mm`); the STEP is always written in mm.
`--tolerance` sets `ConvertOptions.linear_tolerance` in the input's unit (default: derived from the
mesh's noise). `--report` writes the [fidelity report](#fidelity-report) as JSON, also on error.
`--no-measure` skips sampling the written STEP. A one-line outcome and the key numbers go to stdout
(stderr on error). The STEP is written to a temporary file next to the output and renamed into place
only when the outcome is not `error`, after the report is written: on exit code 3 nothing is left at the
output path (an existing file there is untouched). Any exception, expected or not, exits 3, and with
`--report` still writes a report whose `error` names the exception's type and message.

Exit codes:

| code | outcome |
|---|---|
| 0 | every face analytic |
| 1 | converted with `facets` regions (mixed) |
| 2 | whole-part faceted (the writer's fallback, or no analytic region) |
| 3 | error: bad arguments or input, OCP missing (the message names `pip install unmesh[step]`), or nothing valid written |

## Fidelity report

`unmesh convert --report` and `Conversion.fidelity` produce the same JSON object, described by
[`fidelity.schema.json`](fidelity.schema.json). It is versioned: `schema` is `"unmesh.fidelity"` and
`version` is `1`. A new field may be added within a version, so readers must ignore fields they do
not know (the schema allows extra properties); removing or renaming a field, or changing a field's
meaning or unit, bumps the version. All lengths are in millimetres (`units`), after
the `--unit` scaling. Every number is either measured or a bound computed from measured
quantities, and says which; a quantity that was not measured is `null`, never a guess.

| field | meaning |
|---|---|
| `schema`, `version`, `unmesh_version` | Format identity and the library version. |
| `outcome`, `exit_code` | As in the table above. |
| `units` | `"mm"`. |
| `input` | `path`, `format` (`stl`, `obj` or `array`), `unit`, `scale_to_mm`, `triangles`, and the converter's `warnings` (`code`, `message`). |
| `output` | `path`, and `written`: whether the file exists. |
| `tolerances` | The IR's tolerances. |
| `region_counts`, `analytic_area_fraction` | From the converter's `Report`. |
| `deviation.converter` | `max` and `rms` from `Report`: a bound (`kind: "bound"`) on the distance between the input mesh and the IR's surfaces, both directions, computed from how far each vertex moved and the exact chord sagitta; the harness checks it never under-reports. |
| `deviation.writer` | `max_vertex_displacement`, `max_boundary_deviation` and `max_shape_tolerance` from the `WriteReport`: how far the writer moved geometry from the IR. |
| `deviation.measured` | `null` with `--no-measure` or when nothing valid was written. Otherwise the written STEP sampled against the input: it is re-imported and tessellated at `tessellation_deflection` (2e-5 of the input's bounding-box diagonal, 0.1 rad). `step_to_input` is the distance from each tessellation node (a point exactly on a written face) to the input triangles: a sampled lower bound on the true maximum. `input_to_step` is the distance from each input vertex to the tessellated faces, exact to within the deflection. Each has `max`, `mean`, `p99` and `samples`; `max` is the larger of the two maxima. |
| `faces` | One entry per IR region: `region`, `surface`, `shell`, `triangles`, `written_as` (the surface type, or `triangles` for a `facets` region or after a fallback), `converter` (`max`, `rms` of the region's residual; `null` for `facets`) and `writer` (the region's `FaceReport`: `max_shape_tolerance`, `max_vertex_displacement`, `max_boundary_deviation`; `null` after a fallback). |
| `faceted` | One entry per region written as triangles, with a `reason`: `no_surface_fit` (the converter fitted no plane, cylinder, cone, sphere or torus within tolerance), `open_shell` (an open component is kept as one `facets` region), `non_manifold` (the same, on a mesh with non-manifold edges), `assembly_failed` (the converter's analytic assembly failed validation and every shell was kept as facets), `facets_share` (the writer wrote the whole part faceted because `facets` regions held most of the triangles), `shell_failed` (the analytic solid failed construction or verification and the writer fell back; `validity.fallback_reason` says why). |
| `validity` | `valid`, `verified`, `verified_by`, `readback` and `text_check` (each `ok`, `issues`, or `null`), `readback_skipped`, `volume_checked_against_input`, `fallback`, `fallback_reason`, `solids`, `open_shells`, `issues`, and `edge_fallbacks` (`regions`, `kind`, `max_deviation`), from the `WriteReport`. |
| `runtime_s` | Seconds per stage: `read`, `convert`, `write`, `readback`, `measure` (`null` when skipped) and `total`. |
| `error` | `null`, or `type` and `message` when `outcome` is `error`. An error report raised before conversion has only the identity fields, `input`, `output`, `runtime_s` and `error`. |

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
  is what STL files and `(n, 3, 3)` arrays are), `convert_from_labels(&IndexedMesh, &[u32],
  &ConvertOptions)` and `convert_soup_from_labels(&TriangleSoup, &[u32], &ConvertOptions)` (fit a given
  segmentation, see `unmesh.convert_from_labels`), `ConvertOptions`, `Report`, `ConvertWarning`,
  `ConvertOutput`, `ConvertError` (`EmptyMesh`, `InvalidInput`). Source triangle ids in the IR are
  indices into the soup, or into `mesh.faces`.

The `ir_canon` example (`cargo run -p unmesh-core --example ir_canon < ir.json`) prints the canonical
form of an IR read from stdin; the cross-language tests use it.
