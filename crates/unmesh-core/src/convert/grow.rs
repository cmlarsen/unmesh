use std::cell::OnceCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};

use super::curved::{self, Axis, Kind, Member, Single, Tri};
use super::fit::{Region, collect_verts, region_pairs};
use super::linalg::{V3, add, dot, scale, sub};
use super::segment::TriInfo;
use super::surface::Surface;
use super::topology::NONE;

pub const MAX_TURN_DEG: f64 = 15.0;
const MIN_MEMBERS: usize = 3;
const SEED_TRIES: usize = 3;
const INIT_GATE: f64 = 5.0;
const ESCALATE: f64 = 500.0;
const SAG_RATIO: f64 = 16.0;
const SEED_POINTS: [usize; 3] = [12, 24, 48];
const DOUBLY_POINTS: [usize; 3] = [16, 32, 64];
const PEEL_MAX: usize = 64;
const THIN_POINTS: usize = 2048;
const RETRY_ROUNDS: usize = 3;
const RETRY_ROUNDS_DOUBLY: usize = 12;
const SLICE_SHARE: f64 = 0.15;
const DOUBLY_AREA: f64 = 4.0;
const DOUBLY_RATIO: f64 = 0.01;

pub struct Grown {
    pub label: Vec<u32>,
    pub regions: Vec<Region>,
    pub prefit: Vec<(usize, Single)>,
}

struct Graph {
    adj: Vec<Vec<u32>>,
}

fn smooth_pair(a: &Region, b: &Region, tol: f64, cos_max: f64) -> bool {
    let (Some((na, _)), Some((nb, _))) = (a.surface.as_plane(), b.surface.as_plane()) else {
        return false;
    };
    if a.area <= 0.0 || b.area <= 0.0 {
        return false;
    }
    if dot(na, nb) < cos_max {
        return false;
    }
    let turn = super::linalg::norm(super::linalg::cross(na, nb)).atan2(dot(na, nb));
    if turn >= MAX_TURN_DEG.to_radians() - 1e-9 {
        return false;
    }
    let noise = (2.0 * tol).atan2(a.width.max(1e-300)) + (2.0 * tol).atan2(b.width.max(1e-300));
    turn > noise
}

fn graph(regions: &[Region], pairs: &[(u32, u32)], tol: f64) -> Option<Graph> {
    let cos_max = MAX_TURN_DEG.to_radians().cos();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); regions.len()];
    let mut any = false;
    for &(a, b) in pairs {
        if smooth_pair(&regions[a as usize], &regions[b as usize], tol, cos_max) {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
            any = true;
        }
    }
    any.then_some(Graph { adj })
}

struct Pool<'a> {
    vc: &'a [V3],
    faces: &'a [[u32; 3]],
    info: &'a [TriInfo],
    regions: &'a [Region],
    mark: Vec<u32>,
    pos: Vec<u32>,
    token: u32,
}

impl Pool<'_> {
    fn centroid(&self, r: u32) -> V3 {
        let reg = &self.regions[r as usize];
        let mut c = [0.0; 3];
        let mut w = 0.0;
        for &f in &reg.faces {
            let t = self.faces[f as usize];
            let a = self.info[f as usize].area;
            let m = add(
                add(self.vc[t[0] as usize], self.vc[t[1] as usize]),
                self.vc[t[2] as usize],
            );
            c = add(c, scale(m, a / 3.0));
            w += a;
        }
        scale(c, 1.0 / w.max(f64::MIN_POSITIVE))
    }

    /// Every vertex of region `r` within `tol` of the surface, and its plane
    /// normal within the spread of the surface normals over its vertices (a
    /// facet inscribed in the surface is parallel to it somewhere inside),
    /// measured from the normal at its centroid (see `tilt`). Returns the
    /// region's largest chord sagitta.
    fn fits(&self, r: u32, kind: Kind, axis: &Axis, shape: &[f64], tol: f64) -> Option<f64> {
        let reg = &self.regions[r as usize];
        let (n, _) = reg.surface.as_plane()?;
        if curved::max_residual(kind, axis, shape, &self.points(r)) > tol {
            return None;
        }
        let surf = curved::surface_of(
            axis,
            shape,
            &Member {
                pts: Vec::new(),
                kind,
                slot: 0,
            },
        );
        let c = self.centroid(r);
        let normal = surf.normal_at(c)?;
        let spread = reg
            .verts
            .iter()
            .filter_map(|&(v, _)| surf.normal_at(self.vc[v as usize]))
            .map(|m| dot(m, normal).min(1.0).acos())
            .fold(0.0, f64::max);
        let sag = reg
            .faces
            .iter()
            .map(|&f| {
                let t = self.faces[f as usize];
                chord_sag(&surf, t.map(|v| self.vc[v as usize]))
            })
            .fold(0.0, f64::max);
        (dot(n, normal).abs().min(1.0).acos() <= tilt(kind, tol, sag, reg.width, spread))
            .then_some(sag)
    }

    /// `fits` for the single triangle `f`, its chord sagitta at most `limit`.
    fn face_fits(&self, f: u32, fit: &Single, surf: &Surface, limit: f64, tol: f64) -> bool {
        let t = self.faces[f as usize].map(|v| self.vc[v as usize]);
        let pts = t.map(|p| (p, 1.0));
        if curved::max_residual(fit.0, &fit.1, &fit.2, &pts) > tol {
            return false;
        }
        let ti = &self.info[f as usize];
        if ti.area > 0.0 {
            let c = scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
            let Some(normal) = surf.normal_at(c) else {
                return false;
            };
            let spread = t
                .iter()
                .filter_map(|&p| surf.normal_at(p))
                .map(|m| dot(m, normal).min(1.0).acos())
                .fold(0.0, f64::max);
            let longest = (0..3)
                .map(|i| super::linalg::norm(sub(t[(i + 1) % 3], t[i])))
                .fold(0.0, f64::max);
            let width = 2.0 * ti.area / longest.max(f64::MIN_POSITIVE);
            let sag = chord_sag(surf, t);
            if dot(ti.normal, normal).abs().min(1.0).acos() > tilt(fit.0, tol, sag, width, spread) {
                return false;
            }
        }
        chord_sag(surf, t) <= limit
    }

    /// `fits` for every member, and no member's chord sagitta more than
    /// `SAG_RATIO` times the median member's (or `tol`): a tessellated curve
    /// cuts its chords at a similar deflection, while a deliberate flat face
    /// absorbed into a large cylinder carries a sagitta far beyond its
    /// neighbours'.
    fn all_fit(&self, members: &[u32], f: &Single, tol: f64) -> Option<Vec<f64>> {
        let sags: Vec<f64> = members
            .iter()
            .map(|&m| self.fits(m, f.0, &f.1, &f.2, tol))
            .collect::<Option<_>>()?;
        let limit = SAG_RATIO * median(&sags).max(tol);
        sags.iter().all(|&s| s <= limit).then_some(sags)
    }

    fn points(&self, r: u32) -> Vec<(V3, f64)> {
        self.regions[r as usize]
            .verts
            .iter()
            .map(|&(v, w)| (self.vc[v as usize], w))
            .collect()
    }

    fn union(&mut self, members: &[u32]) -> (Vec<(V3, f64)>, Vec<Tri>) {
        let pts = self.union_points(members);
        let list: Vec<u32> = members
            .iter()
            .flat_map(|&r| self.regions[r as usize].faces.iter().copied())
            .collect();
        let tris = curved::face_tris(self.vc, self.faces, self.info, &list);
        (pts, tris)
    }

    /// The vertices of the union of `members` (each once, its weight
    /// summed), in first-seen order.
    fn union_points(&mut self, members: &[u32]) -> Vec<(V3, f64)> {
        self.token += 1;
        let token = self.token;
        let mut pts: Vec<(V3, f64)> = Vec::new();
        for &r in members {
            let reg = &self.regions[r as usize];
            for &(v, w) in &reg.verts {
                let vi = v as usize;
                if self.mark[vi] == token {
                    pts[self.pos[vi] as usize].1 += w;
                } else {
                    self.mark[vi] = token;
                    self.pos[vi] = pts.len() as u32;
                    pts.push((self.vc[vi], w));
                }
            }
        }
        pts
    }

    /// The number of distinct vertices of the union of `members`.
    fn union_len(&mut self, members: &[u32]) -> usize {
        self.token += 1;
        let token = self.token;
        let mut count = 0;
        for &r in members {
            for &(v, _) in &self.regions[r as usize].verts {
                if self.mark[v as usize] != token {
                    self.mark[v as usize] = token;
                    count += 1;
                }
            }
        }
        count
    }
}

/// The running `median` of a growing list (two heaps of order keys).
struct Median {
    lo: BinaryHeap<u64>,
    hi: BinaryHeap<Reverse<u64>>,
}

fn order_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

fn from_key(k: u64) -> f64 {
    f64::from_bits(if k >> 63 == 1 { k & !(1 << 63) } else { !k })
}

impl Median {
    fn from(v: Vec<f64>) -> Self {
        let mut m = Median {
            lo: BinaryHeap::new(),
            hi: BinaryHeap::new(),
        };
        for x in v {
            m.push(x);
        }
        m
    }

    fn push(&mut self, x: f64) {
        let k = order_key(x);
        match self.hi.peek() {
            Some(&Reverse(h)) if k < h => self.lo.push(k),
            _ => self.hi.push(Reverse(k)),
        }
        let want = (self.lo.len() + self.hi.len()).div_ceil(2);
        if self.hi.len() > want {
            let Reverse(h) = self.hi.pop().unwrap();
            self.lo.push(h);
        } else if self.hi.len() < want {
            let l = self.lo.pop().unwrap();
            self.hi.push(Reverse(l));
        }
    }

    fn get(&self) -> f64 {
        self.hi.peek().map_or(0.0, |&Reverse(k)| from_key(k))
    }
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn opposite(regions: &[Region], s: u32, a: u32, b: u32) -> f64 {
    let n = |r: u32| {
        regions[r as usize]
            .surface
            .as_plane()
            .map_or([0.0; 3], |p| p.0)
    };
    let (ns, na, nb) = (n(s), n(a), n(b));
    dot(sub(na, ns), sub(nb, ns))
}

fn seed_triples(
    regions: &[Region],
    g: &Graph,
    s: u32,
    free: &dyn Fn(u32) -> bool,
) -> Vec<[u32; 3]> {
    let near: Vec<u32> = g.adj[s as usize]
        .iter()
        .copied()
        .filter(|&r| free(r))
        .collect();
    let mut pairs: Vec<(f64, [u32; 3])> = Vec::new();
    for (i, &a) in near.iter().enumerate() {
        for &b in &near[i + 1..] {
            pairs.push((opposite(regions, s, a, b), [a, s, b]));
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<[u32; 3]> = pairs.into_iter().map(|p| p.1).collect();
    for &a in &near {
        for &x in &g.adj[a as usize] {
            if x != s && free(x) && !near.contains(&x) {
                out.push([s, a, x]);
            }
        }
    }
    out.truncate(SEED_TRIES);
    out
}

/// Extends a seed chain at both ends, each step to the free neighbour that
/// best keeps turning the same way, until the chain has `min_points`
/// vertices: three strips of a cone (eight vertices) or three lone triangles
/// (five) leave a cylinder or cone fit with almost no redundancy, and a short
/// arc of a staggered cone band needs more before its axis is determined.
fn extend_chain(
    pool: &mut Pool<'_>,
    g: &Graph,
    chain: [u32; 3],
    free: &dyn Fn(u32) -> bool,
    min_points: usize,
) -> Option<Vec<u32>> {
    let n = |r: u32| {
        pool.regions[r as usize]
            .surface
            .as_plane()
            .map_or([0.0; 3], |p| p.0)
    };
    let mut chain: VecDeque<u32> = chain.into_iter().collect();
    let mut stuck = [false; 2];
    loop {
        let members: Vec<u32> = chain.iter().copied().collect();
        if pool.union_len(&members) >= min_points {
            return Some(members);
        }
        if stuck[0] && stuck[1] {
            return None;
        }
        for side in 0..2 {
            if stuck[side] {
                continue;
            }
            let len = chain.len();
            let (e, p) = if side == 0 {
                (chain[0], chain[1])
            } else {
                (chain[len - 1], chain[len - 2])
            };
            let step = sub(n(e), n(p));
            let next = g.adj[e as usize]
                .iter()
                .copied()
                .filter(|&x| free(x) && !chain.contains(&x))
                .map(|x| (dot(sub(n(x), n(e)), step), x))
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            match next {
                Some((_, x)) if side == 0 => chain.push_front(x),
                Some((_, x)) => chain.push_back(x),
                None => stuck[side] = true,
            }
        }
    }
}

/// `seed_fit`, remembering failures: seeds from neighbouring regions often
/// extend to the same chain, and the outcome depends only on the chain.
fn seed_fit_cached(
    pool: &mut Pool<'_>,
    failed: &mut FxHashMap<Vec<u32>, f64>,
    chain: &[u32],
    tol: f64,
) -> Result<Single, f64> {
    if let Some(&init) = failed.get(chain) {
        return Err(init);
    }
    let fit = seed_fit(pool, chain, tol);
    if let Err(init) = fit {
        failed.insert(chain.to_vec(), init);
    }
    fit
}

fn seed_fit(pool: &mut Pool<'_>, chain: &[u32], tol: f64) -> Result<Single, f64> {
    let (pts, tris) = pool.union(chain);
    let fit = curved::fit_quick(&pts, &tris, tol, INIT_GATE * tol)?;
    if pool.all_fit(chain, &fit, tol).is_some() {
        Ok(fit)
    } else {
        Err(0.0)
    }
}

/// The surfaces of the sphere and torus groups, with the chord sagitta
/// limit `peel` would apply to them, by group.
/// Where a fillet meets a sphere or torus blend tangentially, the planar
/// stage can join a fillet strip with a sliver of the blend into one region,
/// and such a member can spoil an otherwise exact cylinder or cone seed.
/// Which strips pick up slivers depends on how the tessellator split each
/// quad. A seed chain that fails is retried without those members (the
/// regions with a triangle that fits an adjacent blend group); `peel` later
/// moves the strips' own triangles into the group.
struct Junction<'a> {
    nbr: &'a [[u32; 3]],
    label: &'a [u32],
    blends: Vec<Option<Blend>>,
}

struct Blend {
    members: Vec<u32>,
    fit: Single,
    surface: OnceCell<Option<(Surface, f64)>>,
}

impl<'a> Junction<'a> {
    fn new(nbr: &'a [[u32; 3]], label: &'a [u32], groups: &[Group]) -> Self {
        let blends = groups
            .iter()
            .map(|grp| {
                doubly(grp.fit.0).then(|| Blend {
                    members: grp.members.clone(),
                    fit: grp.fit.clone(),
                    surface: OnceCell::new(),
                })
            })
            .collect();
        Self { nbr, label, blends }
    }

    fn any(&self) -> bool {
        self.blends.iter().any(Option::is_some)
    }

    /// Whether a triangle of region `r` borders a sphere or torus group and
    /// fits that group's surface as `peel` would accept it.
    fn sliver(&self, pool: &Pool<'_>, owner: &[u32], r: u32, tol: f64) -> bool {
        pool.regions[r as usize].faces.iter().any(|&f| {
            self.nbr[f as usize].iter().any(|&h| {
                if h == NONE {
                    return false;
                }
                let x = self.label[h as usize];
                if x == r || x == NONE || owner[x as usize] == NONE {
                    return false;
                }
                let Some(Some(blend)) = self.blends.get(owner[x as usize] as usize) else {
                    return false;
                };
                let surface = blend.surface.get_or_init(|| {
                    let sags = pool.all_fit(&blend.members, &blend.fit, tol)?;
                    let (kind, axis, shape) = (blend.fit.0, &blend.fit.1, &blend.fit.2);
                    let m = Member {
                        pts: Vec::new(),
                        kind,
                        slot: 0,
                    };
                    let surf = curved::surface_of(axis, shape, &m);
                    Some((surf, SAG_RATIO * median(&sags).max(tol)))
                });
                surface
                    .as_ref()
                    .is_some_and(|(surf, limit)| pool.face_fits(f, &blend.fit, surf, *limit, tol))
            })
        })
    }
}

/// A fresh fit of the union (which can take the tessellation-law radius),
/// else a refit started from `start`, each kept only if every member fits.
fn final_fit(pool: &mut Pool<'_>, members: &[u32], start: &Single, tol: f64) -> Option<Single> {
    let (pts, tris) = pool.union(members);
    if doubly(start.0) {
        return curved::refit(&thin(&pts), start.0, start.1, &start.2, tol)
            .filter(|f| pool.all_fit(members, f, tol).is_some());
    }
    curved::fit_single_gated(&pts, &tris, tol, INIT_GATE * tol)
        .filter(|f| pool.all_fit(members, f, tol).is_some())
        .or_else(|| {
            curved::refit(&pts, start.0, start.1, &start.2, tol)
                .filter(|f| pool.all_fit(members, f, tol).is_some())
        })
}

/// At most `THIN_POINTS` of the items, evenly strided, for refining a fit
/// whose acceptance is then checked on every vertex.
fn thin<T: Copy>(v: &[T]) -> Vec<T> {
    let step = v.len().div_ceil(THIN_POINTS).max(1);
    v.iter().step_by(step).copied().collect()
}

/// A stalled sphere or torus group's next fit: the current one refined on
/// the members so far, else a fresh fit of their union (a small seed's
/// torus is right only locally, and its refinement can stall in that
/// minimum), whichever fits every member.
fn refit_doubly(
    pool: &mut Pool<'_>,
    members: &[u32],
    kind: Kind,
    axis: Axis,
    shape: &[f64],
    tol: f64,
) -> Option<Single> {
    let (pts, tris) = pool.union(members);
    let (pts, tris) = (thin(&pts), thin(&tris));
    curved::refit(&pts, kind, axis, shape, tol)
        .filter(|f| pool.all_fit(members, f, tol).is_some())
        .or_else(|| {
            super::doubly::fit_doubly_with(&pts, &tris, tol, f64::INFINITY, false)
                .ok()
                .filter(|f| f.0 == kind)
        })
}

struct Group {
    members: Vec<u32>,
    fit: Single,
}

fn grow_group(
    pool: &mut Pool<'_>,
    g: &Graph,
    seed: Vec<u32>,
    fit: Single,
    owner: &mut [u32],
    gid: u32,
    tol: f64,
) -> Group {
    let mut members = seed.clone();
    let mut sags = Median::from(pool.all_fit(&members, &fit, tol).unwrap_or_default());
    for &r in &seed {
        owner[r as usize] = gid;
    }
    let (kind, mut axis, mut shape) = (fit.0, fit.1, fit.2.clone());
    let mut tried: FxHashSet<u32> = FxHashSet::default();
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &r in &seed {
        queue.extend(g.adj[r as usize].iter().copied());
    }
    let mut rounds = 0;
    loop {
        while let Some(r) = queue.pop_front() {
            if owner[r as usize] != NONE || tried.contains(&r) {
                continue;
            }
            let limit = SAG_RATIO * sags.get().max(tol);
            match pool.fits(r, kind, &axis, &shape, tol) {
                Some(s) if s <= limit => sags.push(s),
                _ => {
                    tried.insert(r);
                    continue;
                }
            }
            owner[r as usize] = gid;
            members.push(r);
            queue.extend(g.adj[r as usize].iter().copied());
        }
        rounds += 1;
        let cap = if doubly(kind) {
            RETRY_ROUNDS_DOUBLY
        } else {
            RETRY_ROUNDS
        };
        if tried.is_empty() || rounds >= cap {
            break;
        }
        let next = if doubly(kind) {
            refit_doubly(pool, &members, kind, axis, &shape, tol)
        } else {
            curved::refit_quick(&pool.union_points(&members), kind, axis, &shape, tol)
        };
        match next {
            Some(f)
                if (f.1.a != axis.a || f.1.c != axis.c || f.2 != shape)
                    && let Some(s) = pool.all_fit(&members, &f, tol) =>
            {
                axis = f.1;
                shape = f.2;
                sags = Median::from(s);
            }
            _ => break,
        }
        queue.extend(tried.drain());
    }
    let pts = pool.union_points(&members);
    let (rms, max) = curved::member_residual(&axis, &shape, &Member { pts, kind, slot: 0 });
    let fit = (kind, axis, shape, rms, max, false);
    Group { members, fit }
}

/// Joins adjacent groups whose union still fits one surface, starting from
/// the larger group's: separate seeds can claim pieces of one face before
/// either reaches the other. Adjacency is any shared edge, since slivers
/// near a cone's apex can turn by less than their own noise allows.
fn merge_groups(
    pool: &mut Pool<'_>,
    near: &[Vec<u32>],
    mut groups: Vec<Group>,
    owner: &[u32],
    tol: f64,
) -> Vec<Group> {
    let mut alive = vec![true; groups.len()];
    let mut root: Vec<u32> = (0..groups.len() as u32).collect();
    let mut version = vec![0u32; groups.len()];
    let mut failed: FxHashSet<(u32, u32, u32, u32)> = FxHashSet::default();
    loop {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (gi, grp) in groups.iter().enumerate() {
            if !alive[gi] {
                continue;
            }
            for &r in &grp.members {
                for &x in &near[r as usize] {
                    let o = owner[x as usize];
                    if o == NONE {
                        continue;
                    }
                    let h = find(&mut root, o);
                    if h != gi as u32 {
                        pairs.push((gi.min(h as usize) as u32, gi.max(h as usize) as u32));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut merged = false;
        for (a, b) in pairs {
            let (a, b) = (find(&mut root, a) as usize, find(&mut root, b) as usize);
            if a == b || !alive[a] || !alive[b] {
                continue;
            }
            let key = (a as u32, b as u32, version[a], version[b]);
            if failed.contains(&key) {
                continue;
            }
            let (big, small) = if groups[a].members.len() >= groups[b].members.len() {
                (a, b)
            } else {
                (b, a)
            };
            let mut members = groups[big].members.clone();
            members.extend_from_slice(&groups[small].members);
            let pts = pool.union_points(&members);
            let Some(lead) = [&groups[big].fit, &groups[small].fit]
                .into_iter()
                .find(|f| {
                    curved::max_residual(f.0, &f.1, &f.2, &pts) <= tol
                        && pool.all_fit(&members, f, tol).is_some()
                })
                .cloned()
            else {
                failed.insert(key);
                continue;
            };
            let pts = if doubly(lead.0) { thin(&pts) } else { pts };
            let fit = curved::refit_quick(&pts, lead.0, lead.1, &lead.2, tol)
                .filter(|f| pool.all_fit(&members, f, tol).is_some())
                .or(Some(lead));
            if let Some(fit) = fit {
                groups[big] = Group { members, fit };
                version[big] += 1;
                groups[small].members.clear();
                alive[small] = false;
                root[small] = big as u32;
                merged = true;
            }
        }
        if !merged {
            break;
        }
    }
    groups
        .into_iter()
        .zip(alive)
        .filter_map(|(grp, a)| a.then_some(grp))
        .collect()
}

/// How far a facet's normal may tilt from the surface normal at its
/// centroid: the noise allowance across its width, plus the spread of the
/// surface normals over its corners. On a sphere or a torus a facet whose
/// corners sit on different rings tilts beyond the normals at its corners,
/// and a sliver's normal is set by how far its middle bows off the line of
/// the other two corners, so there the allowance is twice the spread and
/// the noise term includes the chord sagitta.
fn tilt(kind: Kind, tol: f64, sag: f64, width: f64, spread: f64) -> f64 {
    let (reach, bow) = if doubly(kind) { (2.0, sag) } else { (1.0, 0.0) };
    (2.0 * tol + bow).atan2(width.max(1e-300)) + reach * spread + 1e-9
}

/// The chord sagitta the growth heuristics compare: the exact bound, except
/// on a torus, where the distance at the corners, edge midpoints and centroid
/// stands in for it (the reported deviation still uses the bound).
fn chord_sag(surf: &Surface, t: [V3; 3]) -> f64 {
    if !matches!(surf, Surface::Torus { .. }) {
        return curved::sagitta(surf, t);
    }
    let mid = |a: V3, b: V3| scale(add(a, b), 0.5);
    [
        t[0],
        t[1],
        t[2],
        mid(t[0], t[1]),
        mid(t[1], t[2]),
        mid(t[2], t[0]),
        scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0),
    ]
    .iter()
    .map(|&p| surf.distance(p).abs())
    .fold(0.0, f64::max)
}

fn doubly(kind: Kind) -> bool {
    matches!(kind, Kind::Sphere | Kind::Torus)
}

/// The free regions nearest `s` across smooth creases (breadth first, whole
/// regions) until they hold `min_points` vertices.
fn patch(
    pool: &mut Pool<'_>,
    g: &Graph,
    s: u32,
    free: &dyn Fn(u32) -> bool,
    min_points: usize,
) -> Option<Vec<u32>> {
    let mut members = vec![s];
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    seen.insert(s);
    let mut verts: FxHashSet<u32> = FxHashSet::default();
    let mut head = 0;
    loop {
        for &(v, _) in &pool.regions[members[head] as usize].verts {
            verts.insert(v);
        }
        if verts.len() >= min_points {
            return Some(members);
        }
        for &x in &g.adj[members[head] as usize] {
            if free(x) && seen.insert(x) {
                members.push(x);
            }
        }
        head += 1;
        if head == members.len() {
            return None;
        }
    }
}

fn find(root: &mut [u32], mut x: u32) -> u32 {
    while root[x as usize] != x {
        root[x as usize] = root[root[x as usize] as usize];
        x = root[x as usize];
    }
    x
}

/// Merges adjacent plane regions that meet at smooth creases (under
/// `MAX_TURN_DEG`, and more than their own noise allows) into one cylinder or
/// cone when every vertex of the union fits it within `tol`. A group needs at
/// least three regions: any two planes meeting at a crease fit a cylinder.
/// A lone triangle whose three neighbours all meet it at smooth creases
/// carries no evidence of a plane and becomes facets.
pub fn run(
    vc: &[V3],
    faces: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    label: &[u32],
    regions: Vec<Region>,
    tol: f64,
) -> Grown {
    let pairs = region_pairs(nbr, label);
    let Some(g) = graph(&regions, &pairs, tol) else {
        return Grown {
            label: label.to_vec(),
            regions,
            prefit: Vec::new(),
        };
    };
    let n = regions.len();
    let mut owner = vec![NONE; n];
    let mut groups: Vec<Group> = Vec::new();
    let mut pool = Pool {
        vc,
        faces,
        info,
        regions: &regions,
        mark: vec![0; vc.len()],
        pos: vec![0; vc.len()],
        token: 0,
    };
    let doubly_region = doubly_curved(vc, faces, nbr, label, &regions, &g);
    let spent = doubly_region.clone();
    let mut seeded = vec![false; n];
    for s in 0..n as u32 {
        if owner[s as usize] != NONE || !doubly_region[s as usize] || seeded[s as usize] {
            continue;
        }
        let free = |r: u32| owner[r as usize] == NONE && doubly_region[r as usize];
        let mut last: Vec<u32> = Vec::new();
        for k in DOUBLY_POINTS {
            let Some(seed) = patch(&mut pool, &g, s, &free, k) else {
                break;
            };
            if seed.len() == last.len() {
                break;
            }
            let (pts, tris) = pool.union(&seed);
            if curved::fit_quick(&pts, &tris, tol, INIT_GATE * tol).is_ok() {
                break;
            }
            match super::doubly::fit_doubly_with(&pts, &tris, tol, f64::INFINITY, false) {
                Ok(fit) => {
                    if pool.all_fit(&seed, &fit, tol).is_some() {
                        let gid = groups.len() as u32;
                        let grp = grow_group(&mut pool, &g, seed, fit, &mut owner, gid, tol);
                        groups.push(grp);
                        last.clear();
                        break;
                    }
                }
                Err(init) if init > ESCALATE * tol => {
                    last = seed;
                    break;
                }
                Err(_) => {}
            }
            last = seed;
        }
        for r in last {
            seeded[r as usize] = true;
        }
    }
    super::timing::lap("grow.doubly");
    let junction = Junction::new(nbr, label, &groups);
    let mut failed: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
    for s in 0..n as u32 {
        if owner[s as usize] != NONE || spent[s as usize] || g.adj[s as usize].is_empty() {
            continue;
        }
        let free = |r: u32| owner[r as usize] == NONE;
        let triples = seed_triples(&regions, &g, s, &free);
        'seed: for t in triples {
            let mut last = 0;
            for k in SEED_POINTS {
                let Some(chain) = extend_chain(&mut pool, &g, t, &free, k) else {
                    break;
                };
                if chain.len() == last {
                    break;
                }
                let init = match seed_fit_cached(&mut pool, &mut failed, &chain, tol) {
                    Ok(fit) => {
                        let gid = groups.len() as u32;
                        let grp = grow_group(&mut pool, &g, chain, fit, &mut owner, gid, tol);
                        groups.push(grp);
                        break 'seed;
                    }
                    Err(init) => init,
                };
                if init > ESCALATE * tol {
                    break;
                }
                if junction.any() {
                    let pure: Vec<u32> = chain
                        .iter()
                        .copied()
                        .filter(|&r| !junction.sliver(&pool, &owner, r, tol))
                        .collect();
                    if pure.len() < chain.len()
                        && pure.len() >= MIN_MEMBERS
                        && let Ok(fit) = seed_fit_cached(&mut pool, &mut failed, &pure, tol)
                    {
                        let gid = groups.len() as u32;
                        let grp = grow_group(&mut pool, &g, pure, fit, &mut owner, gid, tol);
                        groups.push(grp);
                        break 'seed;
                    }
                }
                last = chain.len();
            }
        }
    }
    super::timing::lap("grow.seed");
    let mut near: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in &pairs {
        near[a as usize].push(b);
        near[b as usize].push(a);
    }
    let groups = merge_groups(&mut pool, &near, groups, &owner, tol);
    super::timing::lap("grow.merge");
    let groups: Vec<Group> = groups
        .into_iter()
        .filter(|grp| grp.members.len() >= MIN_MEMBERS && grp.fit.4 <= tol)
        .collect();
    let mut group_of = vec![NONE; n];
    for (gi, grp) in groups.iter().enumerate() {
        for &r in &grp.members {
            group_of[r as usize] = gi as u32;
        }
    }
    let lone: Vec<bool> = (0..n)
        .map(|r| group_of[r] == NONE && (lone_smooth(nbr, info, &regions[r]) || doubly_region[r]))
        .collect();
    let sliced = slices(&g, &groups, &group_of, &pool, nbr, label);
    let groups: Vec<Group> = groups
        .into_iter()
        .zip(&sliced)
        .map(|(grp, &cut)| {
            if cut {
                return grp;
            }
            let fit = final_fit(&mut pool, &grp.members, &grp.fit, tol).unwrap_or(grp.fit);
            Group {
                members: grp.members,
                fit,
            }
        })
        .collect();

    super::timing::lap("grow.final");
    let mut face_group = vec![NONE; faces.len()];
    for (gi, grp) in groups.iter().enumerate() {
        if sliced[gi] {
            continue;
        }
        for &r in &grp.members {
            for &f in &regions[r as usize].faces {
                face_group[f as usize] = gi as u32;
            }
        }
    }
    let candidates: Vec<u32> = (0..n as u32)
        .filter(|&r| group_of[r as usize] == NONE && regions[r as usize].faces.len() <= PEEL_MAX)
        .collect();
    let (mut extra, mut groups) = peel(
        &mut pool,
        nbr,
        &candidates,
        &mut face_group,
        groups,
        &sliced,
        tol,
    );
    super::timing::lap("grow.peel");
    join_touching(
        &mut pool,
        nbr,
        &mut face_group,
        &mut group_of,
        &mut groups,
        &mut extra,
        &sliced,
        tol,
    );

    super::timing::lap("grow.join");
    let mut out: Vec<Region> = Vec::new();
    let mut out_label = vec![NONE; faces.len()];
    let mut prefit = Vec::new();
    let mut scratch = super::fit::Scratch::new(vc.len());
    let mut emitted = vec![false; groups.len()];
    let mut groups: Vec<Option<Group>> = groups.into_iter().map(Some).collect();
    for r in 0..n {
        let gi = group_of[r];
        if gi != NONE {
            if emitted[gi as usize] {
                continue;
            }
            emitted[gi as usize] = true;
            let grp = groups[gi as usize].take().unwrap();
            let mut fl: Vec<u32> = std::mem::take(&mut extra[gi as usize]);
            for &m in &grp.members {
                fl.extend_from_slice(&regions[m as usize].faces);
            }
            fl.sort_unstable();
            let area = fl.iter().map(|&f| info[f as usize].area).sum();
            let verts = collect_verts(faces, info, &fl, &mut scratch);
            let id = out.len();
            for &f in &fl {
                out_label[f as usize] = id as u32;
            }
            let cut = sliced[gi as usize];
            let fit = if cut {
                grp.fit
            } else {
                refreshed(vc, &verts, grp.fit)
            };
            out.push(Region {
                faces: fl,
                surface: Surface::Facets,
                area,
                width: 0.0,
                verts,
                rms: fit.3,
                max: if cut { f64::INFINITY } else { fit.4 },
                sag: 0.0,
            });
            if !cut {
                prefit.push((id, fit));
            }
            continue;
        }
        let reg = &regions[r];
        let left: Vec<u32> = reg
            .faces
            .iter()
            .copied()
            .filter(|&f| face_group[f as usize] == NONE)
            .collect();
        let pieces = if left.len() == reg.faces.len() {
            vec![left]
        } else {
            components(nbr, &left)
        };
        let whole = pieces.len() == 1 && pieces[0].len() == reg.faces.len();
        for piece in pieces {
            let mut part = if whole {
                Region {
                    faces: reg.faces.clone(),
                    surface: reg.surface,
                    area: reg.area,
                    width: reg.width,
                    verts: reg.verts.clone(),
                    rms: reg.rms,
                    max: reg.max,
                    sag: reg.sag,
                }
            } else {
                super::fit::build_region(vc, faces, info, piece, tol, &mut scratch)
            };
            if lone_smooth(nbr, info, &part) || (whole && lone[r]) {
                part.surface = Surface::Facets;
                part.max = f64::INFINITY;
            }
            let id = out.len() as u32;
            for &f in &part.faces {
                out_label[f as usize] = id;
            }
            out.push(part);
        }
    }
    Grown {
        label: out_label,
        regions: out,
        prefit,
    }
}

/// Moves single triangles of small ungrouped regions into an adjacent
/// curved group when the triangle fits it as the group's members do (corners
/// within `tol`, facet normal, chord sagitta). The planar stage can join a
/// strip of one curved face with a sliver of the next where they meet
/// tangentially, or keep a few nearly coplanar triangles of a sphere or a
/// torus together, and such a region fits neither surface as a whole.
fn peel(
    pool: &mut Pool<'_>,
    nbr: &[[u32; 3]],
    candidates: &[u32],
    face_group: &mut [u32],
    groups: Vec<Group>,
    sliced: &[bool],
    tol: f64,
) -> (Vec<Vec<u32>>, Vec<Group>) {
    let mut extra: Vec<Vec<u32>> = vec![Vec::new(); groups.len()];
    let limits: Vec<f64> = groups
        .iter()
        .zip(sliced)
        .map(|(grp, &cut)| {
            if cut {
                return 0.0;
            }
            let sags = pool
                .all_fit(&grp.members, &grp.fit, tol)
                .unwrap_or_default();
            SAG_RATIO * median(&sags).max(tol)
        })
        .collect();
    let surfaces: Vec<Surface> = groups
        .iter()
        .map(|grp| {
            curved::surface_of(
                &grp.fit.1,
                &grp.fit.2,
                &Member {
                    pts: Vec::new(),
                    kind: grp.fit.0,
                    slot: 0,
                },
            )
        })
        .collect();
    let mut open: FxHashSet<u32> = FxHashSet::default();
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &r in candidates {
        for &f in &pool.regions[r as usize].faces {
            open.insert(f);
            queue.push_back(f);
        }
    }
    while let Some(f) = queue.pop_front() {
        if !open.contains(&f) {
            continue;
        }
        let mut near: Vec<u32> = nbr[f as usize]
            .iter()
            .filter(|&&h| h != NONE)
            .map(|&h| face_group[h as usize])
            .filter(|&gi| gi != NONE && !sliced[gi as usize])
            .collect();
        near.sort_unstable();
        near.dedup();
        for gi in near {
            let grp = &groups[gi as usize];
            if pool.face_fits(
                f,
                &grp.fit,
                &surfaces[gi as usize],
                limits[gi as usize],
                tol,
            ) {
                face_group[f as usize] = gi;
                extra[gi as usize].push(f);
                open.remove(&f);
                for &h in &nbr[f as usize] {
                    if h != NONE && open.contains(&h) {
                        queue.push_back(h);
                    }
                }
                break;
            }
        }
    }
    (extra, groups)
}

/// Joins groups that touch once peeled triangles fill the gap between them
/// (pieces of one face that separate seeds grew), when one surface refitted
/// to their union still holds every vertex within `tol`.
#[allow(clippy::too_many_arguments)]
fn join_touching(
    pool: &mut Pool<'_>,
    nbr: &[[u32; 3]],
    face_group: &mut [u32],
    group_of: &mut [u32],
    groups: &mut [Group],
    extra: &mut [Vec<u32>],
    sliced: &[bool],
    tol: f64,
) {
    let mut version = vec![0u32; groups.len()];
    let mut failed: FxHashSet<(u32, u32, u32, u32)> = FxHashSet::default();
    loop {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (f, nb) in nbr.iter().enumerate() {
            let a = face_group[f];
            if a == NONE {
                continue;
            }
            for &h in nb {
                if h == NONE {
                    continue;
                }
                let b = face_group[h as usize];
                if b != NONE && a < b {
                    pairs.push((a, b));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut merged = false;
        for (a, b) in pairs {
            let (a, b) = (a as usize, b as usize);
            if sliced[a]
                || sliced[b]
                || groups[a].members.is_empty()
                || groups[b].members.is_empty()
            {
                continue;
            }
            let key = (a as u32, b as u32, version[a], version[b]);
            if failed.contains(&key) {
                continue;
            }
            let (big, small) = if groups[a].members.len() >= groups[b].members.len() {
                (a, b)
            } else {
                (b, a)
            };
            let mut members = groups[big].members.clone();
            members.extend_from_slice(&groups[small].members);
            let mut pts = pool.union_points(&members);
            let mut seen: FxHashSet<u32> = members
                .iter()
                .flat_map(|&r| pool.regions[r as usize].verts.iter().map(|x| x.0))
                .collect();
            for &f in extra[big].iter().chain(&extra[small]) {
                for v in pool.faces[f as usize] {
                    if seen.insert(v) {
                        pts.push((pool.vc[v as usize], 0.0));
                    }
                }
            }
            let lead = &groups[big].fit;
            let Some(fit) = curved::refit_quick(&thin(&pts), lead.0, lead.1, &lead.2, tol)
                .filter(|f| curved::max_residual(f.0, &f.1, &f.2, &pts) <= tol)
            else {
                failed.insert(key);
                continue;
            };
            let small_members = std::mem::take(&mut groups[small].members);
            for &r in &small_members {
                group_of[r as usize] = big as u32;
            }
            groups[big].members.extend(small_members);
            groups[big].fit = fit;
            version[big] += 1;
            let moved = std::mem::take(&mut extra[small]);
            extra[big].extend(moved);
            for g in face_group.iter_mut() {
                if *g == small as u32 {
                    *g = big as u32;
                }
            }
            merged = true;
        }
        if !merged {
            break;
        }
    }
}

fn components(nbr: &[[u32; 3]], list: &[u32]) -> Vec<Vec<u32>> {
    let set: FxHashSet<u32> = list.iter().copied().collect();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut out = Vec::new();
    for &f in list {
        if !seen.insert(f) {
            continue;
        }
        let mut comp = vec![f];
        let mut head = 0;
        while head < comp.len() {
            let g = comp[head];
            head += 1;
            for &h in &nbr[g as usize] {
                if h != NONE && set.contains(&h) && seen.insert(h) {
                    comp.push(h);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out
}

fn refreshed(vc: &[V3], verts: &[(u32, f64)], fit: Single) -> Single {
    let pts: Vec<(V3, f64)> = verts.iter().map(|&(v, w)| (vc[v as usize], w)).collect();
    let (rms, max) = curved::member_residual(
        &fit.1,
        &fit.2,
        &Member {
            pts,
            kind: fit.0,
            slot: 0,
        },
    );
    (fit.0, fit.1, fit.2, rms, max, fit.5)
}

/// Groups sandwiched between curved neighbours: at least two other groups
/// that each share `SLICE_SHARE` of the group's boundary across smooth
/// creases. A torus or a sphere has no cylinder or cone through it, but it
/// cuts into strips that each fit one within tolerance (rings of a torus are
/// cones, its meridian strips nearly cylinders); such slices stay facets. A
/// cylinder wall that only touches the ends of many slices keeps its fit.
fn slices(
    g: &Graph,
    groups: &[Group],
    group_of: &[u32],
    pool: &Pool<'_>,
    nbr: &[[u32; 3]],
    label: &[u32],
) -> Vec<bool> {
    let face_group = |f: u32| {
        let l = label[f as usize];
        if l == NONE {
            NONE
        } else {
            group_of[l as usize]
        }
    };
    groups
        .iter()
        .enumerate()
        .map(|(gi, grp)| {
            let smooth_to: Vec<u32> = grp
                .members
                .iter()
                .flat_map(|&r| g.adj[r as usize].iter().copied())
                .filter(|&x| {
                    let o = group_of[x as usize];
                    o != NONE && o != gi as u32 && !doubly(groups[o as usize].fit.0)
                })
                .collect();
            if doubly(grp.fit.0) || smooth_to.len() < 2 {
                return false;
            }
            let mut perimeter = 0.0;
            let mut shared: Vec<(u32, f64)> = Vec::new();
            for &r in &grp.members {
                for &f in &pool.regions[r as usize].faces {
                    let t = pool.faces[f as usize];
                    for k in 0..3 {
                        let h = nbr[f as usize][k];
                        let hg = if h == NONE { NONE } else { face_group(h) };
                        if hg == gi as u32 {
                            continue;
                        }
                        let (p, q) = (pool.vc[t[k] as usize], pool.vc[t[(k + 1) % 3] as usize]);
                        let len = super::linalg::norm(sub(q, p));
                        perimeter += len;
                        if hg == NONE || !smooth_to.contains(&label[h as usize]) {
                            continue;
                        }
                        match shared.iter_mut().find(|e| e.0 == hg) {
                            Some(e) => e.1 += len,
                            None => shared.push((hg, len)),
                        }
                    }
                }
            }
            shared
                .iter()
                .filter(|e| e.1 >= SLICE_SHARE * perimeter)
                .count()
                >= 2
        })
        .collect()
}

/// Regions on a surface curved in two directions (a sphere or a torus),
/// judged by how their smooth neighbours' normals differ from their own:
/// across a cylinder or cone the differences line up along one direction
/// (around the axis), on a doubly curved surface they span two. Each
/// difference is weighted by the length of the shared boundary, so the short
/// ends of a strip count little, and the differences are centred first, so
/// the constant tilt between the two kinds of triangle in a staggered cone
/// band (two corners up or two down) does not count as a second direction.
/// The second direction must also bend both ways: a strip beside a crease
/// into another face has its one odd neighbour on one side only. A region
/// much larger than its neighbours is a face of its own (the flat top of a
/// box with filleted edges turns into its fillets on every side) and is
/// never flagged.
fn doubly_curved(
    vc: &[V3],
    faces: &[[u32; 3]],
    nbr: &[[u32; 3]],
    label: &[u32],
    regions: &[Region],
    g: &Graph,
) -> Vec<bool> {
    let n = regions.len();
    let mut out = vec![false; n];
    let mut shared: Vec<(u32, f64)> = Vec::new();
    let cos_max = MAX_TURN_DEG.to_radians().cos();
    for r in 0..n {
        if g.adj[r].is_empty() {
            continue;
        }
        let Some((nr, _)) = regions[r].surface.as_plane() else {
            continue;
        };
        shared.clear();
        for &f in &regions[r].faces {
            let t = faces[f as usize];
            for k in 0..3 {
                let h = nbr[f as usize][k];
                if h == NONE {
                    continue;
                }
                let x = label[h as usize];
                if x == r as u32 || x == NONE {
                    continue;
                }
                match regions[x as usize].surface.as_plane() {
                    Some((nx, _)) if dot(nx, nr) >= cos_max => {}
                    _ => continue,
                }
                let len = super::linalg::norm(sub(vc[t[(k + 1) % 3] as usize], vc[t[k] as usize]));
                match shared.iter_mut().find(|e| e.0 == x) {
                    Some(e) => e.1 += len,
                    None => shared.push((x, len)),
                }
            }
        }
        let diff = |x: u32| {
            regions[x as usize]
                .surface
                .as_plane()
                .map_or([0.0; 3], |p| sub(p.0, nr))
        };
        let total: f64 = shared.iter().map(|e| e.1).sum();
        if shared.len() < 3 || total <= 0.0 {
            continue;
        }
        let mean = scale(
            shared
                .iter()
                .fold([0.0; 3], |acc, &(x, w)| add(acc, scale(diff(x), w))),
            1.0 / total,
        );
        let mut m = [[0.0; 3]; 3];
        for &(x, w) in &shared {
            let d = sub(diff(x), mean);
            super::linalg::outer_add(&mut m, d, d, w);
        }
        let v = sym_eigenvalues(m);
        let areas: Vec<f64> = shared.iter().map(|e| regions[e.0 as usize].area).collect();
        if !(v[0] > 0.0
            && v[1] >= DOUBLY_RATIO * v[0]
            && regions[r].area <= DOUBLY_AREA * median(&areas))
        {
            continue;
        }
        let (_, vecs) = super::linalg::sym_eigen(m);
        let e2 = vecs[1];
        let tau = 0.3 * (v[1] / total).sqrt();
        let side = |sign: f64| shared.iter().any(|&(x, _)| sign * dot(diff(x), e2) > tau);
        out[r] = side(1.0) && side(-1.0);
    }
    out
}

/// Eigenvalues of a symmetric 3x3 matrix, largest first (closed form).
fn sym_eigenvalues(m: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    let p2 = (m[0][0] - q).powi(2) + (m[1][1] - q).powi(2) + (m[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p <= 0.0 {
        return [q; 3];
    }
    let b = |i: usize, j: usize| (m[i][j] - if i == j { q } else { 0.0 }) / p;
    let det = b(0, 0) * (b(1, 1) * b(2, 2) - b(1, 2) * b(2, 1))
        - b(0, 1) * (b(1, 0) * b(2, 2) - b(1, 2) * b(2, 0))
        + b(0, 2) * (b(1, 0) * b(2, 1) - b(1, 1) * b(2, 0));
    let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}

fn lone_smooth(nbr: &[[u32; 3]], info: &[TriInfo], r: &Region) -> bool {
    if r.faces.len() != 1 || r.area <= 0.0 {
        return false;
    }
    let cos_max = MAX_TURN_DEG.to_radians().cos();
    let f = r.faces[0] as usize;
    nbr[f].iter().all(|&g| {
        g != NONE
            && info[g as usize].area > 0.0
            && dot(info[f].normal, info[g as usize].normal) >= cos_max
    })
}

#[cfg(test)]
mod tests {
    use super::super::linalg::V3;
    use super::super::tests::{Frame, washer};
    use crate::api::{ConvertOptions, ConvertOutput};
    use crate::convert::convert_soup;
    use crate::ir::Surface;
    use crate::mesh::TriangleSoup;

    fn auto(soup: &TriangleSoup) -> ConvertOutput {
        let out = convert_soup(soup, &ConvertOptions::default()).unwrap();
        out.ir.validate().unwrap();
        out
    }

    fn count(out: &ConvertOutput, kind: &str) -> u32 {
        out.report.region_counts.get(kind).copied().unwrap_or(0)
    }

    /// Closed solid of revolution: rings `(radius, z)` from bottom to top,
    /// `n` segments each, fan caps; triangles between rings are staggered
    /// when `stagger` is set.
    fn revolved(profile: &[(f64, f64)], n: usize, stagger: bool, frame: &Frame) -> TriangleSoup {
        let at = |k: usize, i: usize| {
            let half = if stagger && k % 2 == 1 { 0.5 } else { 0.0 };
            let t = std::f64::consts::TAU * (i as f64 + half) / n as f64;
            let (r, z) = profile[k];
            frame.map([r * t.cos(), r * t.sin(), z])
        };
        let mut tris = Vec::new();
        for k in 0..profile.len() - 1 {
            for i in 0..n {
                let (a, b) = (at(k, i), at(k, i + 1));
                let (c, d) = (at(k + 1, i), at(k + 1, i + 1));
                if stagger && k % 2 == 1 {
                    tris.push([a, b, d]);
                    tris.push([a, d, c]);
                } else {
                    tris.push([a, b, c]);
                    tris.push([b, d, c]);
                }
            }
        }
        let (bottom, top) = (profile[0].1, profile[profile.len() - 1].1);
        let last = profile.len() - 1;
        for i in 0..n {
            tris.push([frame.map([0.0, 0.0, bottom]), at(0, i + 1), at(0, i)]);
            tris.push([frame.map([0.0, 0.0, top]), at(last, i), at(last, i + 1)]);
        }
        TriangleSoup { triangles: tris }
    }

    #[test]
    fn tessellation_strips_become_one_cylinder_each() {
        for (frame, n) in [(Frame::identity(), 64), (Frame::tilted(), 40)] {
            let (soup, _) = washer(10.0, 4.0, 5.0, n, &frame, false);
            let out = auto(&soup);
            assert_eq!(
                out.ir.regions.len(),
                4,
                "n={n}: {:?}",
                out.report.region_counts
            );
            assert_eq!(count(&out, "cylinder"), 2);
            let mut radii: Vec<f64> = out
                .ir
                .regions
                .iter()
                .filter_map(|r| match r.surface {
                    Surface::Cylinder { radius, .. } => Some(radius),
                    _ => None,
                })
                .collect();
            radii.sort_by(f64::total_cmp);
            assert!((radii[0] - 4.0).abs() < 1e-9 && (radii[1] - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn staggered_cone_band_becomes_one_cone() {
        let soup = revolved(
            &[(30.0, 0.0), (30.0, 6.0), (28.0, 8.0), (26.0, 10.0)],
            180,
            true,
            &Frame::tilted(),
        );
        let out = auto(&soup);
        assert_eq!(count(&out, "cone"), 1, "{:?}", out.report.region_counts);
        assert_eq!(count(&out, "cylinder"), 1);
        assert_eq!(out.ir.regions.len(), 4);
    }

    #[test]
    fn deliberate_prism_stays_planes() {
        for n in [6, 12, 24] {
            let (soup, _) = washer(10.0, 4.0, 5.0, n, &Frame::tilted(), false);
            let out = auto(&soup);
            assert_eq!(count(&out, "plane") as usize, 2 * n + 2, "n={n}");
            assert_eq!(out.ir.regions.len(), 2 * n + 2);
        }
    }

    #[test]
    fn small_crease_separates_cylinder_and_cone() {
        let t = 10f64.to_radians().tan();
        let soup = revolved(
            &[(10.0, 0.0), (10.0, 5.0), (10.0 - 4.0 * t, 9.0)],
            90,
            false,
            &Frame::identity(),
        );
        let out = auto(&soup);
        assert_eq!(count(&out, "cylinder"), 1, "{:?}", out.report.region_counts);
        assert_eq!(count(&out, "cone"), 1);
        assert_eq!(out.ir.regions.len(), 4);
    }

    fn rounded_rectangle(r: f64, n: usize) -> TriangleSoup {
        let (hx, hy, h) = (20.0, 12.0, 6.0);
        let mut outline: Vec<[f64; 2]> = Vec::new();
        for (cx, cy, start) in [
            (hx, hy, 0.0),
            (-hx, hy, 90.0),
            (-hx, -hy, 180.0),
            (hx, -hy, 270.0),
        ] {
            for k in 0..=n {
                let a = (start + 90.0 * k as f64 / n as f64).to_radians();
                outline.push([cx + r * a.cos(), cy + r * a.sin()]);
            }
        }
        let m = outline.len();
        let p = |i: usize, z: f64| [outline[i % m][0], outline[i % m][1], z];
        let mut tris = Vec::new();
        for i in 0..m {
            tris.push([p(i, 0.0), p(i + 1, 0.0), p(i + 1, h)]);
            tris.push([p(i, 0.0), p(i + 1, h), p(i, h)]);
            tris.push([[0.0, 0.0, 0.0], p(i + 1, 0.0), p(i, 0.0)]);
            tris.push([[0.0, 0.0, h], p(i, h), p(i + 1, h)]);
        }
        TriangleSoup { triangles: tris }
    }

    #[test]
    fn flat_sides_between_fillets_stay_planes() {
        let out = auto(&rounded_rectangle(3.0, 16));
        assert_eq!(count(&out, "cylinder"), 4, "{:?}", out.report.region_counts);
        assert_eq!(count(&out, "plane"), 6);
        for r in &out.ir.regions {
            if let Surface::Cylinder { radius, .. } = r.surface {
                assert!((radius - 3.0).abs() < 1e-9);
            }
        }
    }

    fn torus_soup(big: f64, small: f64, nu: usize, nv: usize, frame: &Frame) -> TriangleSoup {
        let at = |i: usize, j: usize| -> V3 {
            let (u, v) = (
                std::f64::consts::TAU * (i % nu) as f64 / nu as f64,
                std::f64::consts::TAU * (j % nv) as f64 / nv as f64,
            );
            let rho = big + small * v.cos();
            frame.map([rho * u.cos(), rho * u.sin(), small * v.sin()])
        };
        let mut tris = Vec::new();
        for i in 0..nu {
            for j in 0..nv {
                tris.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                tris.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        TriangleSoup { triangles: tris }
    }

    struct Planar {
        vc: Vec<V3>,
        faces: Vec<[u32; 3]>,
        info: Vec<super::TriInfo>,
        regions: Vec<super::Region>,
        tol: f64,
    }

    fn planar(soup: &TriangleSoup) -> Planar {
        let (w, shells, info, tol, _, _) =
            super::super::prepare(soup, &ConvertOptions::default()).unwrap();
        let (label, n) = super::super::segment::run(
            &w.vc,
            &w.faces,
            &shells.topo.nbr,
            &info,
            &shells.eligible,
            tol,
            false,
        );
        let mut scratch = super::super::fit::Scratch::new(w.vc.len());
        let regions =
            super::super::fit::build_regions(&w.vc, &w.faces, &info, &label, n, tol, &mut scratch);
        Planar {
            vc: w.vc,
            faces: w.faces,
            info,
            regions,
            tol,
        }
    }

    fn pool(p: &Planar) -> super::Pool<'_> {
        super::Pool {
            vc: &p.vc,
            faces: &p.faces,
            info: &p.info,
            regions: &p.regions,
            mark: vec![0; p.vc.len()],
            pos: vec![0; p.vc.len()],
            token: 0,
        }
    }

    #[test]
    fn union_points_and_count_match_the_full_union() {
        let p = planar(&torus_soup(10.0, 3.0, 24, 12, &Frame::tilted()));
        assert!(p.regions.len() > 20);
        let mut pool = pool(&p);
        let n = p.regions.len() as u32;
        for members in [
            vec![0],
            vec![0, 1, 2],
            (0..n).step_by(3).collect(),
            (0..n).collect(),
        ] {
            let (pts, tris) = pool.union(&members);
            assert_eq!(pool.union_points(&members), pts);
            assert_eq!(pool.union_len(&members), pts.len());
            let faces: usize = members
                .iter()
                .map(|&r| p.regions[r as usize].faces.len())
                .sum();
            assert_eq!(tris.len(), faces);
            let mut ids: Vec<V3> = pts.iter().map(|x| x.0).collect();
            ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ids.dedup();
            assert_eq!(ids.len(), pts.len());
        }
    }

    #[test]
    fn cached_seed_fit_matches_and_remembers_failures() {
        let p = planar(&torus_soup(10.0, 3.0, 24, 12, &Frame::tilted()));
        let mut pool = pool(&p);
        let mut failed = rustc_hash::FxHashMap::default();
        let n = p.regions.len() as u32;
        let chains: Vec<Vec<u32>> = (0..n.saturating_sub(3))
            .map(|r| vec![r, r + 1, r + 2])
            .collect();
        let mut fails = 0;
        for chain in chains.iter().chain(&chains) {
            let plain = super::seed_fit(&mut pool, chain, p.tol);
            let cached = super::seed_fit_cached(&mut pool, &mut failed, chain, p.tol);
            match (&plain, &cached) {
                (Ok(a), Ok(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                (Err(a), Err(b)) => {
                    assert_eq!(a.to_bits(), b.to_bits());
                    assert_eq!(
                        failed.get(chain).map(|x: &f64| x.to_bits()),
                        Some(a.to_bits())
                    );
                    fails += 1;
                }
                _ => panic!("cached seed fit disagrees for {chain:?}"),
            }
        }
        assert!(fails > 0);
        assert_eq!(failed.len() * 2, fails);
    }

    #[test]
    fn running_median_matches_the_sorted_median() {
        let mut m = super::Median::from(Vec::new());
        let mut v: Vec<f64> = Vec::new();
        let mut x = 0.37f64;
        for i in 0..200 {
            x = (x * 7919.0 + 0.123).fract();
            let y = if i % 7 == 0 { -x } else { x * 1e-3 };
            m.push(y);
            v.push(y);
            assert_eq!(m.get(), super::median(&v));
        }
    }

    #[test]
    fn torus_becomes_one_torus() {
        for frame in [Frame::identity(), Frame::tilted()] {
            let out = auto(&torus_soup(20.0, 4.0, 96, 32, &frame));
            assert_eq!(out.ir.regions.len(), 1, "{:?}", out.report.region_counts);
            let Surface::Torus {
                major_radius,
                minor_radius,
                ..
            } = out.ir.regions[0].surface
            else {
                panic!("{:?}", out.report.region_counts)
            };
            assert!((major_radius - 20.0).abs() < 1e-9 && (minor_radius - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dome_on_a_stem_becomes_a_sphere_and_a_cylinder() {
        let r = 8.0;
        let mut profile = vec![(r, 0.0)];
        for k in 0..12 {
            let phi = std::f64::consts::FRAC_PI_2 * k as f64 / 12.0;
            profile.push((r * phi.cos(), 10.0 + r * phi.sin()));
        }
        profile.push((r * 0.13, 10.0 + r * (1.0 - 0.13f64 * 0.13).sqrt()));
        let soup = revolved(&profile, 64, false, &Frame::tilted());
        let out = auto(&soup);
        assert_eq!(count(&out, "sphere"), 1, "{:?}", out.report.region_counts);
        assert_eq!(count(&out, "cylinder"), 1);
        for reg in &out.ir.regions {
            if let Surface::Sphere { radius, .. } = reg.surface {
                assert!((radius - r).abs() < 1e-9);
            }
        }
    }
}
