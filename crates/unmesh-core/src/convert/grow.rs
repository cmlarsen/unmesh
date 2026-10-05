use std::collections::VecDeque;

use rustc_hash::FxHashSet;

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
const RETRY_ROUNDS: usize = 3;
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
    /// measured from the normal at its centroid. Returns the region's largest
    /// chord sagitta.
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
        let allow = (2.0 * tol).atan2(reg.width.max(1e-300)) + spread + 1e-9;
        if dot(n, normal).abs().min(1.0).acos() > allow {
            return None;
        }
        Some(
            reg.faces
                .iter()
                .map(|&f| {
                    let t = self.faces[f as usize];
                    curved::sagitta(&surf, t.map(|v| self.vc[v as usize]))
                })
                .fold(0.0, f64::max),
        )
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
        self.token += 1;
        let token = self.token;
        let mut pts: Vec<(V3, f64)> = Vec::new();
        let mut list: Vec<u32> = Vec::new();
        for &r in members {
            let reg = &self.regions[r as usize];
            list.extend_from_slice(&reg.faces);
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
        let tris = curved::face_tris(self.vc, self.faces, self.info, &list);
        (pts, tris)
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
        if pool.union(&members).0.len() >= min_points {
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

/// A fresh fit of the union (which can take the tessellation-law radius),
/// else a refit started from `start`, each kept only if every member fits.
fn final_fit(pool: &mut Pool<'_>, members: &[u32], start: &Single, tol: f64) -> Option<Single> {
    let (pts, tris) = pool.union(members);
    curved::fit_single_gated(&pts, &tris, tol, INIT_GATE * tol)
        .filter(|f| pool.all_fit(members, f, tol).is_some())
        .or_else(|| {
            curved::refit(&pts, start.0, start.1, &start.2, tol)
                .filter(|f| pool.all_fit(members, f, tol).is_some())
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
    let mut sags = pool.all_fit(&members, &fit, tol).unwrap_or_default();
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
            let limit = SAG_RATIO * median(&sags).max(tol);
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
        if tried.is_empty() || rounds >= RETRY_ROUNDS {
            break;
        }
        let (pts, _) = pool.union(&members);
        match curved::refit_quick(&pts, kind, axis, &shape, tol) {
            Some(f)
                if (f.1.a != axis.a || f.1.c != axis.c || f.2 != shape)
                    && let Some(s) = pool.all_fit(&members, &f, tol) =>
            {
                axis = f.1;
                shape = f.2;
                sags = s;
            }
            _ => break,
        }
        queue.extend(tried.drain());
    }
    let (pts, _) = pool.union(&members);
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
            let (big, small) = if groups[a].members.len() >= groups[b].members.len() {
                (a, b)
            } else {
                (b, a)
            };
            let mut members = groups[big].members.clone();
            members.extend_from_slice(&groups[small].members);
            let lead = groups[big].fit.clone();
            if pool.all_fit(&members, &lead, tol).is_none() {
                continue;
            }
            let fit = curved::refit_quick(&pool.union(&members).0, lead.0, lead.1, &lead.2, tol)
                .filter(|f| pool.all_fit(&members, f, tol).is_some())
                .or(Some(lead));
            if let Some(fit) = fit {
                groups[big] = Group { members, fit };
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
                let (pts, tris) = pool.union(&chain);
                match curved::fit_quick(&pts, &tris, tol, INIT_GATE * tol) {
                    Ok(fit) if pool.all_fit(&chain, &fit, tol).is_some() => {
                        let gid = groups.len() as u32;
                        let grp = grow_group(&mut pool, &g, chain, fit, &mut owner, gid, tol);
                        groups.push(grp);
                        break 'seed;
                    }
                    Err(init) if init > ESCALATE * tol => break,
                    _ => {}
                }
                last = chain.len();
            }
        }
    }
    let mut near: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in &pairs {
        near[a as usize].push(b);
        near[b as usize].push(a);
    }
    let groups = merge_groups(&mut pool, &near, groups, &owner, tol);
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

    let mut new_id = vec![NONE; n];
    let mut group_id = vec![NONE; groups.len()];
    let mut next = 0u32;
    for r in 0..n {
        let gi = group_of[r];
        if gi == NONE {
            new_id[r] = next;
            next += 1;
        } else {
            if group_id[gi as usize] == NONE {
                group_id[gi as usize] = next;
                next += 1;
            }
            new_id[r] = group_id[gi as usize];
        }
    }
    let new_label: Vec<u32> = label
        .iter()
        .map(|&l| if l == NONE { NONE } else { new_id[l as usize] })
        .collect();

    let mut slots: Vec<Option<Region>> = (0..next).map(|_| None).collect();
    let mut prefit = Vec::new();
    let mut scratch = super::fit::Scratch::new(vc.len());
    for (gi, grp) in groups.into_iter().enumerate() {
        let id = group_id[gi] as usize;
        let mut fl: Vec<u32> = Vec::new();
        let mut area = 0.0;
        for &r in &grp.members {
            fl.extend_from_slice(&regions[r as usize].faces);
            area += regions[r as usize].area;
        }
        fl.sort_unstable();
        let verts = collect_verts(faces, info, &fl, &mut scratch);
        let max = if sliced[gi] { f64::INFINITY } else { grp.fit.4 };
        slots[id] = Some(Region {
            faces: fl,
            surface: Surface::Facets,
            area,
            width: 0.0,
            verts,
            rms: grp.fit.3,
            max,
            sag: 0.0,
        });
        if !sliced[gi] {
            prefit.push((id, grp.fit));
        }
    }
    for (r, (reg, is_lone)) in regions.into_iter().zip(lone).enumerate() {
        if group_of[r] != NONE {
            continue;
        }
        let mut reg = reg;
        if is_lone {
            reg.surface = Surface::Facets;
            reg.max = f64::INFINITY;
        }
        slots[new_id[r] as usize] = Some(reg);
    }
    Grown {
        label: new_label,
        regions: slots.into_iter().map(|s| s.unwrap()).collect(),
        prefit,
    }
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
                .filter(|&x| group_of[x as usize] != NONE && group_of[x as usize] != gi as u32)
                .collect();
            if smooth_to.len() < 2 {
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
    use super::super::linalg::{V3, add, dot, scale};
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

    #[test]
    fn torus_strips_stay_facets() {
        let (big, small, nu, nv) = (20.0, 4.0, 96, 32);
        let at = |i: usize, j: usize| -> V3 {
            let (u, v) = (
                std::f64::consts::TAU * (i % nu) as f64 / nu as f64,
                std::f64::consts::TAU * (j % nv) as f64 / nv as f64,
            );
            let rho = big + small * v.cos();
            [rho * u.cos(), rho * u.sin(), small * v.sin()]
        };
        let mut tris = Vec::new();
        for i in 0..nu {
            for j in 0..nv {
                tris.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                tris.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let out = auto(&TriangleSoup { triangles: tris });
        let analytic = out
            .ir
            .regions
            .iter()
            .filter(|r| !r.surface.is_facets())
            .count();
        assert!(
            analytic * 20 < out.ir.regions.len().max(1) * 20 && analytic <= 8,
            "{:?}",
            out.report.region_counts
        );
        let _ = (add, dot, scale);
    }
}
