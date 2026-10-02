mod sample;
mod surface;

use std::fmt;

use parry3d_f64::math::Vector;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::TriMesh;

use crate::ir::{Ir, Point, Surface};

pub use sample::{Triangle, sample_triangles, total_area, triangle_area};
pub use surface::project_onto_surface;

#[derive(Debug)]
pub enum JudgeError {
    EmptyReference(&'static str),
    TriangleIdOutOfRange { region: u32, id: u32, input: usize },
    FacetIndexOutOfRange { region: u32 },
    SourceMismatch { ir: u32, input: usize },
}

impl fmt::Display for JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JudgeError::EmptyReference(what) => write!(f, "{what} has no triangles"),
            JudgeError::TriangleIdOutOfRange { region, id, input } => write!(
                f,
                "region {region} references triangle {id} but the input has {input}"
            ),
            JudgeError::FacetIndexOutOfRange { region } => {
                write!(f, "facets region {region} has an out-of-range vertex index")
            }
            JudgeError::SourceMismatch { ir, input } => write!(
                f,
                "IR was built from {ir} triangles but the input mesh has {input}"
            ),
        }
    }
}

impl std::error::Error for JudgeError {}

#[derive(Debug, Clone, Copy)]
pub struct JudgeOptions {
    pub samples_per_mm2: f64,
    pub seed: u64,
    pub include_vertices: bool,
}

impl Default for JudgeOptions {
    fn default() -> Self {
        Self {
            samples_per_mm2: 10.0,
            seed: 0,
            include_vertices: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub max: f64,
    pub p99: f64,
    pub p95: f64,
    pub mean: f64,
}

impl Stats {
    pub fn of(mut d: Vec<f64>) -> Stats {
        if d.is_empty() {
            return Stats {
                count: 0,
                max: 0.0,
                p99: 0.0,
                p95: 0.0,
                mean: 0.0,
            };
        }
        d.sort_unstable_by(f64::total_cmp);
        let q = |p: f64| d[(((d.len() - 1) as f64) * p).ceil() as usize];
        Stats {
            count: d.len(),
            max: d[d.len() - 1],
            p99: q(0.99),
            p95: q(0.95),
            mean: d.iter().sum::<f64>() / d.len() as f64,
        }
    }
}

/// Both directions against one reference mesh. `ir_to_mesh` is the distance from IR samples to the
/// nearest reference triangle; `mesh_to_ir` is the distance from reference samples to the nearest
/// point of the IR's footprint-bounded surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    pub ir_to_mesh: Stats,
    pub mesh_to_ir: Stats,
}

impl Comparison {
    pub fn max(&self) -> f64 {
        self.ir_to_mesh.max.max(self.mesh_to_ir.max)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JudgeResult {
    pub input: Comparison,
    pub truth: Option<Comparison>,
    pub region_max: Vec<f64>,
}

/// Converter-reported max deviation minus the harness-measured two-sided max to the input mesh.
/// Negative means the converter under-reports.
pub fn calibration(reported_max_deviation: f64, input: &Comparison) -> f64 {
    reported_max_deviation - input.max()
}

fn vec3(p: &Point) -> Vector {
    Vector::new(p[0], p[1], p[2])
}

fn build_trimesh(tris: &[Triangle], what: &'static str) -> Result<TriMesh, JudgeError> {
    if tris.is_empty() {
        return Err(JudgeError::EmptyReference(what));
    }
    let vertices: Vec<Vector> = tris.iter().flatten().map(vec3).collect();
    let indices: Vec<[u32; 3]> = (0..tris.len() as u32)
        .map(|i| [3 * i, 3 * i + 1, 3 * i + 2])
        .collect();
    TriMesh::new(vertices, indices).map_err(|_| JudgeError::EmptyReference(what))
}

fn dist(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1024);
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("distance worker panicked"))
            .collect()
    })
}

fn distances_to_mesh(mesh: &TriMesh, pts: &[Point]) -> Vec<f64> {
    par_map(pts, |p| {
        let q = mesh.project_local_point(vec3(p), false).point;
        dist(*p, [q.x, q.y, q.z])
    })
}

pub fn mesh_distances(tris: &[Triangle], pts: &[Point]) -> Result<Vec<f64>, JudgeError> {
    Ok(distances_to_mesh(&build_trimesh(tris, "mesh")?, pts))
}

/// The IR's surfaces bounded by their footprints. Each analytic region's footprint is its source
/// triangles with every corner projected onto the region's surface; a facets region's footprint is
/// its own triangles. The nearest point of the IR to a query is found by the nearest point on the
/// flat footprint triangles, then moved onto the analytic surface by the closed-form projection.
/// The flat triangle only locates the footprint: it is within the chord sagitta of the curved
/// surface, and the final projection removes that error except where the nearest point sits within
/// one chord of the footprint boundary.
pub struct Footprint<'a> {
    mesh: TriMesh,
    tri_region: Vec<u32>,
    surfaces: Vec<&'a Surface>,
}

impl Footprint<'_> {
    fn distance(&self, p: &Point) -> f64 {
        let (proj, _) = self.mesh.project_local_point_and_get_feature(vec3(p));
        let q = [proj.point.x, proj.point.y, proj.point.z];
        let surface = self.surfaces[self.tri_region[proj.subshape as usize] as usize];
        dist(*p, project_onto_surface(surface, q))
    }

    fn distances(&self, pts: &[Point]) -> Vec<f64> {
        par_map(pts, |p| self.distance(p))
    }
}

fn region_triangles(
    region: &crate::ir::Region,
    input: &[Triangle],
) -> Result<Vec<Triangle>, JudgeError> {
    if let Surface::Facets { vertices, faces } = &region.surface {
        return faces
            .iter()
            .map(|f| {
                let mut t = [[0.0; 3]; 3];
                for (k, &i) in f.iter().enumerate() {
                    t[k] = *vertices
                        .get(i as usize)
                        .ok_or(JudgeError::FacetIndexOutOfRange { region: region.id })?;
                }
                Ok(t)
            })
            .collect();
    }
    region
        .triangles
        .iter()
        .map(|&id| {
            input
                .get(id as usize)
                .copied()
                .ok_or(JudgeError::TriangleIdOutOfRange {
                    region: region.id,
                    id,
                    input: input.len(),
                })
        })
        .collect()
}

pub fn build_footprint<'a>(ir: &'a Ir, input: &[Triangle]) -> Result<Footprint<'a>, JudgeError> {
    let mut tris = Vec::new();
    let mut tri_region = Vec::new();
    for (ri, region) in ir.regions.iter().enumerate() {
        for mut t in region_triangles(region, input)? {
            if !region.surface.is_facets() {
                for c in &mut t {
                    *c = project_onto_surface(&region.surface, *c);
                }
            }
            tris.push(t);
            tri_region.push(ri as u32);
        }
    }
    Ok(Footprint {
        mesh: build_trimesh(&tris, "IR footprint")?,
        tri_region,
        surfaces: ir.regions.iter().map(|r| &r.surface).collect(),
    })
}

/// Region index and point for every sample. An analytic region is sampled uniformly by area on its
/// source triangles and each sample is moved onto the analytic surface by closed-form projection;
/// the region's analytic surface is therefore only sampled inside the footprint of its source
/// triangles, shifted by at most the region's own residual.
pub fn sample_ir(
    ir: &Ir,
    input: &[Triangle],
    opts: &JudgeOptions,
) -> Result<(Vec<Point>, Vec<u32>), JudgeError> {
    if ir.source.triangle_count as usize != input.len() {
        return Err(JudgeError::SourceMismatch {
            ir: ir.source.triangle_count,
            input: input.len(),
        });
    }
    let mut points = Vec::new();
    let mut owner = Vec::new();
    for (ri, region) in ir.regions.iter().enumerate() {
        let tris = region_triangles(region, input)?;
        let salt = 0x1000 + ri as u64;
        for (p, _) in sample_triangles(
            &tris,
            opts.samples_per_mm2,
            opts.seed,
            salt,
            opts.include_vertices,
        ) {
            points.push(if region.surface.is_facets() {
                p
            } else {
                project_onto_surface(&region.surface, p)
            });
            owner.push(ri as u32);
        }
    }
    Ok((points, owner))
}

fn compare(
    ir_points: &[Point],
    footprint: &Footprint<'_>,
    reference: &[Triangle],
    what: &'static str,
    opts: &JudgeOptions,
    salt: u64,
) -> Result<(Comparison, Vec<f64>), JudgeError> {
    let mesh = build_trimesh(reference, what)?;
    let forward = distances_to_mesh(&mesh, ir_points);
    let ref_samples: Vec<Point> = sample_triangles(
        reference,
        opts.samples_per_mm2,
        opts.seed,
        salt,
        opts.include_vertices,
    )
    .into_iter()
    .map(|(p, _)| p)
    .collect();
    let reverse = footprint.distances(&ref_samples);
    let comparison = Comparison {
        ir_to_mesh: Stats::of(forward.clone()),
        mesh_to_ir: Stats::of(reverse),
    };
    Ok((comparison, forward))
}

pub fn judge(
    ir: &Ir,
    input: &[Triangle],
    truth: Option<&[Triangle]>,
    opts: &JudgeOptions,
) -> Result<JudgeResult, JudgeError> {
    let (points, owner) = sample_ir(ir, input, opts)?;
    let footprint = build_footprint(ir, input)?;
    let (input_cmp, forward) = compare(&points, &footprint, input, "input mesh", opts, 1)?;
    let mut region_max = vec![0.0_f64; ir.regions.len()];
    for (d, r) in forward.iter().zip(&owner) {
        let slot = &mut region_max[*r as usize];
        *slot = slot.max(*d);
    }
    let truth_cmp = truth
        .map(|t| compare(&points, &footprint, t, "truth mesh", opts, 2).map(|c| c.0))
        .transpose()?;
    Ok(JudgeResult {
        input: input_cmp,
        truth: truth_cmp,
        region_max,
    })
}

#[cfg(test)]
mod tests;
