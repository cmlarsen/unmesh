mod sample;
mod surface;

use std::fmt;

use parry3d_f64::math::Vector;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::TriMesh;

use crate::ir::{Ir, Point, Surface};

pub use sample::{Sample, Triangle, point_at, sample_triangles, total_area, triangle_area};
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

pub const UNDER_REPORT_ABS_TOL: f64 = 1e-9;
pub const UNDER_REPORT_REL_TOL: f64 = 0.005;

/// True when the converter reports less than the measured max by more than the judge's own sampling
/// uncertainty (`UNDER_REPORT_ABS_TOL` plus `UNDER_REPORT_REL_TOL` of the measured value).
pub fn under_reports(reported_max_deviation: f64, input: &Comparison) -> bool {
    let measured = input.max();
    reported_max_deviation < measured - UNDER_REPORT_ABS_TOL - UNDER_REPORT_REL_TOL * measured
}

pub const REFINE_TOP_K: usize = 256;

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

struct IrTriangles {
    tris: Vec<Triangle>,
    region: Vec<u32>,
}

fn ir_triangles(ir: &Ir, input: &[Triangle]) -> Result<IrTriangles, JudgeError> {
    if ir.source.triangle_count as usize != input.len() {
        return Err(JudgeError::SourceMismatch {
            ir: ir.source.triangle_count,
            input: input.len(),
        });
    }
    let mut tris = Vec::new();
    let mut region = Vec::new();
    for (ri, r) in ir.regions.iter().enumerate() {
        for t in region_triangles(r, input)? {
            tris.push(t);
            region.push(ri as u32);
        }
    }
    Ok(IrTriangles { tris, region })
}

fn on_surface(surface: &Surface, p: Point) -> Point {
    if surface.is_facets() {
        p
    } else {
        project_onto_surface(surface, p)
    }
}

/// The IR's surfaces bounded by their footprints. Each analytic region's footprint is its source
/// triangles with every corner projected onto the surface (a facets region's own triangles). The
/// nearest point of the IR to a query is found on the flat footprint triangles, over all regions,
/// then moved onto its region's analytic surface by the closed-form projection. That removes the
/// chord error of the flat triangle inside a face, except within one chord of a footprint boundary:
/// there the nearest flat triangle can belong to a neighbouring curved region, whose projected point
/// sits a sagitta away. A sample of the input mesh is therefore also measured to the projection of
/// itself onto the surface of the region that owns its triangle. That point lies in the owner's patch
/// by construction, so the smaller of the two is still an upper bound on the distance to the IR.
pub struct Footprint<'a> {
    mesh: TriMesh,
    tri_region: Vec<u32>,
    surfaces: Vec<&'a Surface>,
    owner: Vec<Option<u32>>,
}

impl Footprint<'_> {
    fn distance(&self, p: &Point) -> f64 {
        let (proj, _) = self.mesh.project_local_point_and_get_feature(vec3(p));
        let q = [proj.point.x, proj.point.y, proj.point.z];
        let surface = self.surfaces[self.tri_region[proj.subshape as usize] as usize];
        dist(*p, on_surface(surface, q))
    }

    fn owned_distance(&self, p: &Point, input_tri: u32) -> f64 {
        let d = self.distance(p);
        match self.owner.get(input_tri as usize) {
            Some(&Some(r)) => d.min(dist(*p, on_surface(self.surfaces[r as usize], *p))),
            _ => d,
        }
    }
}

fn build_footprint<'a>(ir: &'a Ir, it: &IrTriangles) -> Result<Footprint<'a>, JudgeError> {
    let surfaces: Vec<&Surface> = ir.regions.iter().map(|r| &r.surface).collect();
    let tris: Vec<Triangle> = it
        .tris
        .iter()
        .zip(&it.region)
        .map(|(t, &r)| t.map(|c| on_surface(surfaces[r as usize], c)))
        .collect();
    let mut owner = vec![None; ir.source.triangle_count as usize];
    for (ri, r) in ir.regions.iter().enumerate() {
        if r.surface.is_facets() {
            continue;
        }
        for &id in &r.triangles {
            owner[id as usize] = Some(ri as u32);
        }
    }
    Ok(Footprint {
        mesh: build_trimesh(&tris, "IR footprint")?,
        tri_region: it.region.clone(),
        surfaces,
        owner,
    })
}

fn ir_point(ir: &Ir, it: &IrTriangles, s: &Sample) -> Point {
    let surface = &ir.regions[it.region[s.tri as usize] as usize].surface;
    on_surface(surface, point_at(&it.tris[s.tri as usize], s.b, s.c))
}

/// Region index and point for every sample. An analytic region is sampled uniformly by area on its
/// source triangles (plus corners and edge midpoints) and each sample is moved onto the analytic
/// surface by closed-form projection. The judged patch is therefore the source triangles projected
/// onto the surface; area a writer extends a wrong surface over beyond them is not judged.
pub fn sample_ir(
    ir: &Ir,
    input: &[Triangle],
    opts: &JudgeOptions,
) -> Result<(Vec<Point>, Vec<u32>), JudgeError> {
    let it = ir_triangles(ir, input)?;
    let samples = ir_samples(&it, opts);
    Ok((
        samples.iter().map(|s| ir_point(ir, &it, s)).collect(),
        samples.iter().map(|s| it.region[s.tri as usize]).collect(),
    ))
}

fn ir_samples(it: &IrTriangles, opts: &JudgeOptions) -> Vec<Sample> {
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < it.tris.len() {
        let r = it.region[start];
        let end = start + it.region[start..].iter().take_while(|&&x| x == r).count();
        let salt = 0x1000 + u64::from(r);
        for mut s in sample_triangles(
            &it.tris[start..end],
            opts.samples_per_mm2,
            opts.seed,
            salt,
            opts.include_vertices,
        ) {
            s.tri += start as u32;
            out.push(s);
        }
        start = end;
    }
    out
}

fn maximize(start: &Sample, f: &(impl Fn(&Sample) -> f64 + Sync)) -> (u32, f64) {
    let clamp = |b: f64, c: f64| {
        let (b, c) = (b.clamp(0.0, 1.0), c.clamp(0.0, 1.0));
        let sum = b + c;
        if sum > 1.0 {
            (b / sum, c / sum)
        } else {
            (b, c)
        }
    };
    let mut cur = *start;
    let mut best = f(&cur);
    let mut step = 0.25;
    for _ in 0..200 {
        if step < 1e-6 {
            break;
        }
        let mut moved = None;
        for (db, dc) in [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
        ] {
            let (b, c) = clamp(cur.b + db * step, cur.c + dc * step);
            let cand = Sample { tri: cur.tri, b, c };
            let v = f(&cand);
            if v > best {
                best = v;
                moved = Some(cand);
            }
        }
        match moved {
            Some(c) => cur = c,
            None => step /= 2.0,
        }
    }
    (cur.tri, best)
}

fn refine_top(
    samples: &[Sample],
    d: &[f64],
    f: &(impl Fn(&Sample) -> f64 + Sync),
) -> Vec<(u32, f64)> {
    let mut order: Vec<usize> = (0..d.len()).collect();
    let k = REFINE_TOP_K.min(order.len());
    if k == 0 {
        return Vec::new();
    }
    order.select_nth_unstable_by(k - 1, |&a, &b| d[b].total_cmp(&d[a]));
    let starts: Vec<Sample> = order[..k].iter().map(|&i| samples[i]).collect();
    par_map(&starts, |s| maximize(s, f))
}

fn with_refined_max(mut stats: Stats, refined: &[(u32, f64)]) -> Stats {
    for (_, v) in refined {
        stats.max = stats.max.max(*v);
    }
    stats.p99 = stats.p99.min(stats.max);
    stats
}

type Compared = (Comparison, Vec<f64>, Vec<(u32, f64)>);

struct Setup<'a> {
    ir: &'a Ir,
    it: IrTriangles,
    samples: Vec<Sample>,
    footprint: Footprint<'a>,
}

fn compare(
    setup: &Setup<'_>,
    reference: &[Triangle],
    what: &'static str,
    owned: bool,
    opts: &JudgeOptions,
    salt: u64,
) -> Result<Compared, JudgeError> {
    let Setup {
        ir,
        it,
        samples,
        footprint,
    } = setup;
    let mesh = build_trimesh(reference, what)?;
    let ir_points: Vec<Point> = samples.iter().map(|s| ir_point(ir, it, s)).collect();
    let forward = distances_to_mesh(&mesh, &ir_points);
    let forward_refined = refine_top(samples, &forward, &|s| {
        let p = ir_point(ir, it, s);
        let q = mesh.project_local_point(vec3(&p), false).point;
        dist(p, [q.x, q.y, q.z])
    });

    let ref_samples = sample_triangles(
        reference,
        opts.samples_per_mm2,
        opts.seed,
        salt,
        opts.include_vertices,
    );
    let ref_points: Vec<Point> = ref_samples
        .iter()
        .map(|s| point_at(&reference[s.tri as usize], s.b, s.c))
        .collect();
    let reverse_at = |p: &Point, tri: u32| {
        if owned {
            footprint.owned_distance(p, tri)
        } else {
            footprint.distance(p)
        }
    };
    let reverse = par_map(
        &ref_samples
            .iter()
            .zip(&ref_points)
            .map(|(s, p)| (s.tri, *p))
            .collect::<Vec<_>>(),
        |(tri, p)| reverse_at(p, *tri),
    );
    let reverse_refined = refine_top(&ref_samples, &reverse, &|s| {
        reverse_at(&point_at(&reference[s.tri as usize], s.b, s.c), s.tri)
    });

    let comparison = Comparison {
        ir_to_mesh: with_refined_max(Stats::of(forward.clone()), &forward_refined),
        mesh_to_ir: with_refined_max(Stats::of(reverse), &reverse_refined),
    };
    Ok((comparison, forward, forward_refined))
}

pub fn judge(
    ir: &Ir,
    input: &[Triangle],
    truth: Option<&[Triangle]>,
    opts: &JudgeOptions,
) -> Result<JudgeResult, JudgeError> {
    let it = ir_triangles(ir, input)?;
    let samples = ir_samples(&it, opts);
    let footprint = build_footprint(ir, &it)?;
    let setup = Setup {
        ir,
        it,
        samples,
        footprint,
    };
    let (input_cmp, forward, refined) = compare(&setup, input, "input mesh", true, opts, 1)?;
    let mut region_max = vec![0.0_f64; ir.regions.len()];
    for (s, d) in setup.samples.iter().zip(&forward) {
        let slot = &mut region_max[setup.it.region[s.tri as usize] as usize];
        *slot = slot.max(*d);
    }
    for (tri, v) in refined {
        let slot = &mut region_max[setup.it.region[tri as usize] as usize];
        *slot = slot.max(v);
    }
    let truth_cmp = truth
        .map(|t| compare(&setup, t, "truth mesh", false, opts, 2).map(|c| c.0))
        .transpose()?;
    Ok(JudgeResult {
        input: input_cmp,
        truth: truth_cmp,
        region_max,
    })
}

#[cfg(test)]
mod tests;
