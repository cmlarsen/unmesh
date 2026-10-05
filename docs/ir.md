# IR v0

The IR is the one contract between the Rust core (which fits surfaces) and every writer (which builds
edges and solids). It is a JSON document with `"ir_version": 0`. The machine-readable schema is
[`ir.schema.json`](ir.schema.json); the Rust types are `unmesh_core::ir`, the Python types are
`unmesh.ir`. Every fixture in [`fixtures/ir/`](../fixtures/ir) validates against the schema and
round-trips byte-identically through both languages.

Pre-release there is no compatibility promise: any change to this document bumps `ir_version`, and a
reader rejects a version it does not know.

**Edges are not in the IR.** The IR carries fitted surfaces, which regions touch, and the mesh
boundary polyline for each touching pair. The writer computes edges (OCCT intersection for
`transversal` pairs, projected polylines for `tangent` pairs). See [plan.md](plan.md) § Architecture.

Units are whatever the input mesh used (millimetres for CAD exports). Angles in parameters are
radians; angles named `*_deg` are degrees.

## Answers to the five review questions

1. **How is a region's orientation defined?** [§ Orientation](#orientation).
2. **How are boundary polylines ordered and paired?** [§ Adjacency](#adjacency-and-boundary-polylines).
3. **How is the tangency threshold chosen?** [§ Tangent versus transversal](#tangent-versus-transversal).
4. **How does a `facets` region border an analytic one?** [§ Facets regions](#facets-regions).
5. **What happens with non-manifold input?** [§ Non-manifold and open input](#non-manifold-and-open-input).

## Document layout

```json
{
  "ir_version": 0,
  "tolerances": { "linear": 0.001, "angular_snap_deg": 0.5, "tangent_threshold_deg": 3.0, "vertex_merge": 1e-06 },
  "source":      { "triangle_count": 12, "vertex_count": 8 },
  "shells":      [ { "closed": true, "role": "outer", "parent": null, "regions": [0, 1, 2] } ],
  "regions":     [ { "id": 0, "surface": {}, "triangles": [], "residual": {} } ],
  "adjacencies": [ { "regions": [0, 1], "boundaries": [] } ],
  "vertices":    [ { "id": 0, "role": "junction", "position": [], "regions": [], "source_positions": [] } ]
}
```

Regions, vertices and shells are flat arrays. `region.id` and `vertex.id` equal the array index
(validated), so references are plain integers. All keys are required; unknown keys are rejected.
Optional values are `null`, never absent.

### Determinism

Converters should emit arrays in a deterministic order so the same input gives the same text:
regions by lowest source triangle id, adjacencies by `(regions[0], regions[1])`, vertices by
lexicographic `position`, and the boundaries within one adjacency by lexicographic first point.
Readers must not depend on the order beyond the id and index rules above.

### Canonical JSON

Canonical text is what the round-trip guarantee compares. It is deterministic, so two implementations
produce the same bytes from the same data:

- UTF-8, no whitespace, a single trailing newline only in files, not in the text itself.
- Object keys sorted by code point.
- Strings escape `"` and `\`, and write every character outside printable ASCII as `\uXXXX`
  (lowercase hex, UTF-16 code units).
- Integers (indices, ids, `ir_version`) have no decimal point or exponent.
- Floats are always written as floats: shortest decimal digits that round-trip the `f64`, ties between
  two equally short candidates resolved to the even last digit, then laid out by decimal exponent `e`
  (the exponent of the first digit): plain notation `123.456`, `100.0`, `0.00001` for `-5 <= e < 16`,
  otherwise `d[.ddd]e<exp>` with a bare signed exponent (`1e-6`, `1.5e21`). Zero, including negative
  zero, is `0.0`. NaN and infinity are errors.
- A field typed as a float is a float even when it holds a whole number (`90.0`), so the type
  survives a round trip.

Rust: `Ir::to_canonical_json`, `Ir::from_json`. Python: `Ir.dumps`, `Ir.loads`.

## Source

`source.triangle_count` is the number of input triangles as the user supplied them (STL order, or the
face list of a `(vertices, faces)` input). It bounds every source triangle id in the IR, which a
validator checks. `source.vertex_count` is the number of unique vertices of the input after the
converter's weld at `tolerances.vertex_merge`; it is informational, kept as a cheap fingerprint of the
input, and no id is validated against it. The IR does not embed the mesh.

## Tolerances

| field | default | meaning |
|---|---|---|
| `linear` | 1e-3 | Linear tolerance the IR was built with. Boundary points lie within it of both adjacent surfaces; a region whose max residual exceeds it becomes `facets`. |
| `angular_snap_deg` | 0.5 | Angle within which near-parallel, near-perpendicular and near-axis-aligned surfaces were snapped to exact relations. |
| `tangent_threshold_deg` | 3.0 | Dihedral angle below which a boundary is `tangent`. See below. |
| `vertex_merge` | 1e-6 | Distance within which mesh vertices were merged into one IR vertex. Also the tolerance for "this point is that vertex". |

A writer reads these from the IR rather than assuming defaults.

## Regions

A region is a connected set of source triangles explained by one surface.

```json
{ "id": 2, "surface": { "type": "cylinder", ... }, "triangles": [14, 15, 16], "residual": { "rms": 2e-05, "max": 6e-05 } }
```

- `triangles`: ids of the source triangles, as indices into the **input** triangles (STL order, or the
  face list of a `(vertices, faces)` input), each below `source.triangle_count`. They are not indices
  into the welded mesh: weld drops degenerate triangles, so welded indices drift from input indices. The
  converter maps welded faces back through the source-triangle map that `unmesh_core::weld` returns
  (`unmesh.weld` returns it as an extra array). Triangles dropped as degenerate belong to no region. Every
  triangle id appears in at most one region.
- `residual`: RMS and max distance from the region's source triangles to its surface, in the units of
  the mesh. `null` for `facets` regions, which make no claim about a surface.

### Surfaces

| `type` | parameters | natural normal |
|---|---|---|
| `plane` | `origin`, `normal` (unit) | `normal` |
| `cylinder` | `origin` (on the axis), `axis` (unit), `radius` | away from the axis |
| `cone` | `apex`, `axis` (unit, from the apex toward the widening side), `half_angle` in (0, pi/2) | away from the axis, tilted back toward the apex: `cos(a)*radial - sin(a)*axis` |
| `sphere` | `center`, `radius` | away from the center |
| `torus` | `center`, `axis` (unit), `major_radius`, `minor_radius` with `major_radius > minor_radius` | away from the tube's center circle |
| `facets` | `vertices`, `faces` | per triangle, see [§ Facets regions](#facets-regions) |

Surfaces are the full unbounded analytic surfaces; the region's extent comes from its boundary
polylines. A cylinder's angular seam is the writer's business.

### Orientation

Every region has an outward normal: the direction pointing out of the solid's material, into empty
space. The IR stores this unambiguously:

- A `plane` stores the outward normal itself.
- `cylinder`, `cone`, `sphere` and `torus` store `"orientation"`: `"same"` when the outward normal
  equals the surface's natural normal in the table above (a boss, a convex fillet, a disc's rim), and
  `"reversed"` when it is the opposite (a bore, a countersink, a concave fillet, a spherical cavity).
- `facets` triangles are wound counter-clockwise seen from outside, so the right-hand-rule normal of
  each triangle is the outward normal.

There is no separate sign on the parameters: radii and half-angles are always positive.

The orientation of shells follows from this. An `outer` shell's outward normals point away
from its enclosed volume. A converter that finds an outer shell with negative signed volume flips its
triangle winding before fitting and reports `flipped_winding`. A `cavity` shell is the opposite on
purpose: its normals point into the void (away from the material), so its signed volume is negative, and
it is exempt from the flip rule. The IR never contains an inside-out outer shell.

On periodic surfaces (cylinder, cone, sphere, torus) the IR gives no seam and no angular reference. The
writer chooses them, and it must pick the seam so it does not coincide with a boundary the face is
trimmed by where that can be avoided.

## Adjacency and boundary polylines

Two regions are adjacent when source triangles from each share a mesh edge. There is one `adjacencies`
entry per unordered pair `[a, b]` with `a < b`, and it holds one or more `boundaries`. A pair gets
several boundaries when it touches along disjoint runs (a through-bore meets the top face along one
loop and the bottom face along another, but those are different pairs; a single pair with two runs
is, for example, a slot wall meeting the floor in two separate arcs).

A boundary is the chain of mesh edges between the two regions, as 3D points:

```json
{ "kind": "transversal", "dihedral_deg": 90.0, "closed": false,
  "start_vertex": 0, "end_vertex": 1, "points": [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]] }
```

**Ordering.** Points are listed in travel order such that region `a` (the lower id) is on the left and
region `b` on the right, looking at the solid from outside, that is, looking against the outward
normal. Equivalently the vector `n_a x t` points from the boundary into region `a`, where `n_a` is
`a`'s outward normal and `t` the direction of travel. It follows that traversing region `a`'s own
boundary loops counter-clockwise about its outward normal uses each boundary as written when the
region is `a`, and reversed when it is `b`. For two planar regions this also gives the convexity: the
edge is convex when `t` is parallel to `n_a x n_b`, and concave when anti-parallel.

**Splitting.** A boundary is split wherever an IR vertex lies on it, and wherever `kind` changes; a change of
kind is marked with a `kind_change` vertex (see [Vertices](#vertices)). So
a boundary is either `closed` (a loop with no vertex on it: the first point is not repeated at the
end, there are at least 3 points, and `start_vertex` and `end_vertex` are `null`), or open (at least 2
points, whose first and last points are the positions of IR vertices `start_vertex` and
`end_vertex`). A fillet's tangent edge along a straight run therefore appears as a separate boundary
from its transversal ends.

**Which side is the face.** The loop direction is what tells the writer which side of a boundary the
face lies on: with `a` on the left, the face of region `a` is the part of its surface to the left of the
boundary, and the face of `b` is the part to the right. On a periodic surface a closed boundary alone
does not say which of the two sides it separates is the face; the direction does.

**Head to tail.** For every region, take its boundaries in that region's own direction (as written when
the region is `a`, reversed when it is `b`). At every vertex, as many of them must arrive as leave. A
validator checks this; it catches a single reversed open boundary. Closed boundaries have no vertices
and are covered by the geometric tests instead.

**Points.** Boundary points are mesh vertex positions, possibly moved by snapping. Each point lies
within `tolerances.linear` of both adjacent surfaces. Intermediate points are the original mesh
vertices along the boundary, so their spacing is the tessellation's.

### Tangent versus transversal

`kind` is `tangent` when the two regions meet smoothly along the boundary, and `transversal` when
they meet at an angle.

`dihedral_deg` is the angle in [0, 180] between the two regions' outward normals along the boundary: 0
is perfectly smooth and 180 is a knife edge. It is the **median** over the boundary's sample points
(every point; for a two-point boundary, its midpoint) of the angle between the two outward normals
evaluated on the fitted analytic surfaces at that point, never between mesh triangle normals. For a
`facets` side, the normal is that of the facets triangle the boundary edge belongs to.

`kind` is `tangent` exactly when `dihedral_deg < tolerances.tangent_threshold_deg`. The threshold is
stored in the IR and `kind` is authoritative; a validator checks the two agree.

Where `kind` changes partway along one edge, the edge is split with hysteresis around the threshold:
a crossing counts only if the per-node dihedral samples fall below `threshold - 0.5°` on one side and
above `threshold + 0.5°` on the other, and every resulting run spans at least two polyline segments.
The split lands on the existing polyline node nearest the threshold, so every `kind_change` vertex
and boundary point stays a mesh vertex position. An edge that only touches the threshold, or hovers
inside the band, keeps a single boundary classified by its median dihedral.

How the default of 3.0 degrees was chosen:

- It uses fitted surfaces, so the facet chord angle of a coarse tessellation (a fillet drawn with 10
  degree segments has 10 degree creases between its triangles) does not enter. Only the error of the
  surface fit does, typically well under 1 degree for regions with enough triangles.
- It is more than three times that fit error, so a true fillet boundary is not misread as a crease.
- It is below the smallest crease the library promises to keep as a real edge. The ambiguity sets in
  [plan.md](plan.md) (a one-segment fillet versus a planar chamfer) are decided by the segmentation
  stage, not by this threshold: a boundary the segmentation kept as an edge between two surfaces with
  a dihedral under the threshold is still reported `tangent`.

The value is a default, not a proof. The harness's fillet and chamfer families (M2) measure
misclassification in both directions, and the threshold is retuned there; a retune changes the default
in `Tolerances`, not the format.

## Vertices

A vertex is a point where boundaries end. It exists so a writer need not rediscover topology from
polyline endpoints.

```json
{ "id": 0, "role": "junction", "position": [0.0, 0.0, 0.0], "regions": [0, 2, 4], "source_positions": [[0.0, 0.0, 0.0]] }
```

- `role` is `junction` for a point where three or more regions meet, or `kind_change` for a point on a
  single pair's boundary where `kind` changes. A boundary must end at a vertex, and a `kind` change
  mid-chain would otherwise have nowhere to end. Dihedral angles that vary along a boundary (a slanted
  plane cutting a cylinder) cross the tangent threshold at such points.
- `position` is the IR's position after snapping (the least-squares point of the meeting surfaces).
- `regions` are the sorted ids of the regions that meet there: at least three for a `junction`, exactly
  two for a `kind_change`.
- `source_positions` are the mesh vertex positions that were merged into this vertex.
- Every open boundary starts and ends at a vertex, and the regions of a vertex are exactly the regions
  of the boundaries that end there. A `kind_change` vertex is the end of exactly two boundaries of the
  same pair, one `tangent` and one `transversal`. A writer treats it as a vertex of degree two on the
  edge: the edge continues, and only the method that builds it changes.

A closed boundary has no vertex: the circular edge of a bore or of a disc fillet is a loop with
nothing on it, and the writer treats it as a closed edge with a seam.

## Facets regions

A `facets` region keeps a patch of triangles as they are, for a part of the mesh that no analytic
surface explains within tolerance.

```json
{ "type": "facets", "vertices": [[...]], "faces": [[0, 1, 4]] }
```

`faces` index into `vertices`, wind counter-clockwise from outside, and there is one face per entry in
the region's `triangles` (same order). An open-shell facets region keeps the input winding as it is.

**Bordering an analytic region.**

- The facets patch and the analytic region share mesh vertices, so the boundary polyline between them
  is a chain of boundary edges of the patch. Every point of such a polyline is, bit for bit, one of the
  patch's `vertices`, in chain order.
- Snapping moves a shared vertex onto the analytic surface, and then the patch's `vertices` carry the
  moved position, so the two sides still agree exactly at the points. Between points the analytic
  face's true edge curves away from the patch's straight chord by the chord's sagitta. That gap is
  expected, small, and the writer's to close (issue #34); the IR does not hide it.
- The dihedral and `kind` come from the analytic normal against the normal of the patch triangle
  carrying that boundary edge.
- Two facets regions are never adjacent: touching patches are one patch. A validator rejects an
  adjacency between two `facets` regions.
- A vertex where a patch meets two analytic regions is an ordinary vertex.

## Shells

A shell is one connected component of the input mesh; `shells[i].regions` lists its region ids. Each
region is in exactly one shell, and an adjacency never crosses shells.

```json
{ "closed": true, "role": "outer", "parent": null, "regions": [0, 1, 2, 3, 4, 5] }
```

- `role: "outer"` shells have `parent: null`. Their normals point out of the material.
- `role: "cavity"` shells are closed shells inside an outer shell: a void in the material. `parent` is the
  index of the enclosing outer shell, which must itself be closed. Cavity normals point into the void (still out of the material), so
  their signed volume is negative, and they are not flipped. Cavities do not nest: a body floating
  inside a void is a separate `outer` shell.
- The writer builds **one solid** from each outer shell together with all the cavities whose `parent` is
  that shell. Outer shells without cavities are plain solids. The result is a compound with one solid per
  outer shell.
- An open shell (`closed: false`) is always `outer`.

A multi-body input is several outer shells, which the writer emits as a compound of solids. Bodies that
only touch at a vertex are separate shells; no adjacency joins them. Bodies that touch along an edge
share that edge between four triangles, so the edge is non-manifold and the bodies form one
edge-connected component, kept as a single open facets shell as the table below requires.

## Non-manifold and open input

Components are formed by edge connectivity after welding. A component is a **closed shell** only if
every one of its mesh edges is shared by exactly two triangles, with consistent winding.

| input | result |
|---|---|
| Degenerate (zero-area) triangles | Dropped before fitting. Report warning `degenerate_triangles`. |
| Duplicate triangles (same three vertices after welding), same winding | Dropped before fitting. Report warning `degenerate_triangles`. |
| Coincident triangles with opposite winding | The reverse twin is dropped when removing it leaves its edge-connected component manifold (every edge used by exactly two triangles), reported as `repaired_winding`. Otherwise both copies are kept. |
| Face-touching bodies (a coincident face pair with opposite winding shared by two bodies) | Both copies kept: the shared edges are used by more than two triangles, so the edge-connected component is a single open facets shell holding every triangle. Warning `non_manifold_edges`. |
| Closed manifold, consistent winding, positive volume | Normal shell. |
| Closed manifold, negative volume, not enclosed by another shell | Winding flipped. Warning `flipped_winding`. |
| Closed manifold, negative volume, inside another outer shell | A `cavity` shell. Not flipped. |
| Closed manifold, mixed winding | Winding repaired by flood fill when that makes it consistent; otherwise treated as non-manifold. Warning `repaired_winding`. |
| Open edges (shared by one triangle) | The component is an **open shell**: `"closed": false`, exactly one `facets` region holding all its triangles, no adjacencies, no vertices. Warning `open_edges`. |
| Non-manifold edges (shared by three or more triangles) | The whole edge-connected component is an open shell as above. Warning `non_manifold_edges`. The converter never guesses which triangles belong together. |

The writer emits an open shell as a sewn shell of triangle faces, never as a solid, and reports it. A
part with some open shells and some closed ones still converts: the closed shells are analytic
solids, the open ones are faceted shells, all in one compound. Nothing fails a whole file.

## Consumers

- The writer (#5, #17, #33, #34) builds faces from regions, trims them with boundary loops, and
  derives edges from `kind`.
- The oracle builder (#7) emits this IR from labelled tessellation, so each field is something a
  B-rep can give: surface parameters, face adjacency and tangency, edge discretisations.
- The harness sampler (#10) samples surfaces inside the footprint given by each region's
  `triangles`, and `facets` regions on their triangles.
