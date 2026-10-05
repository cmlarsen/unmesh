use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const IR_VERSION: u32 = 0;

pub use crate::mesh::Point;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ir {
    pub ir_version: u32,
    pub tolerances: Tolerances,
    pub source: Source,
    pub shells: Vec<Shell>,
    pub regions: Vec<Region>,
    pub adjacencies: Vec<Adjacency>,
    pub vertices: Vec<Vertex>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerances {
    pub linear: f64,
    pub angular_snap_deg: f64,
    pub tangent_threshold_deg: f64,
    pub vertex_merge: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            linear: 1e-3,
            angular_snap_deg: 0.5,
            tangent_threshold_deg: 3.0,
            vertex_merge: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub triangle_count: u32,
    pub vertex_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellRole {
    Outer,
    Cavity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shell {
    pub closed: bool,
    pub role: ShellRole,
    pub parent: Option<u32>,
    pub regions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub id: u32,
    pub surface: Surface,
    pub triangles: Vec<u32>,
    pub residual: Option<Residual>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residual {
    pub rms: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    Same,
    Reversed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Surface {
    Plane {
        origin: Point,
        normal: Point,
    },
    Cylinder {
        origin: Point,
        axis: Point,
        radius: f64,
        orientation: Orientation,
    },
    Cone {
        apex: Point,
        axis: Point,
        half_angle: f64,
        orientation: Orientation,
    },
    Sphere {
        center: Point,
        radius: f64,
        orientation: Orientation,
    },
    Torus {
        center: Point,
        axis: Point,
        major_radius: f64,
        minor_radius: f64,
        orientation: Orientation,
    },
    Facets {
        vertices: Vec<Point>,
        faces: Vec<[u32; 3]>,
    },
}

impl Surface {
    pub fn is_facets(&self) -> bool {
        matches!(self, Surface::Facets { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adjacency {
    pub regions: [u32; 2],
    pub boundaries: Vec<Boundary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Transversal,
    Tangent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub kind: Kind,
    pub dihedral_deg: f64,
    pub closed: bool,
    pub start_vertex: Option<u32>,
    pub end_vertex: Option<u32>,
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VertexRole {
    Junction,
    KindChange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vertex {
    pub id: u32,
    pub role: VertexRole,
    pub position: Point,
    pub regions: Vec<u32>,
    pub source_positions: Vec<Point>,
}

#[derive(Debug)]
pub enum IrError {
    Json(serde_json::Error),
    NonFinite,
    Invalid(Vec<String>),
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrError::Json(e) => write!(f, "invalid IR JSON: {e}"),
            IrError::NonFinite => write!(f, "IR contains a non-finite number"),
            IrError::Invalid(v) => write!(f, "invalid IR: {}", v.join("; ")),
        }
    }
}

impl std::error::Error for IrError {}

impl From<serde_json::Error> for IrError {
    fn from(e: serde_json::Error) -> Self {
        IrError::Json(e)
    }
}

impl Ir {
    pub fn from_json(text: &str) -> Result<Ir, IrError> {
        let ir: Ir = serde_json::from_str(text)?;
        ir.validate()?;
        Ok(ir)
    }

    pub fn to_canonical_json(&self) -> Result<String, IrError> {
        if !self.floats().iter().all(|x| x.is_finite()) {
            return Err(IrError::NonFinite);
        }
        let value = serde_json::to_value(self)?;
        let mut out = String::new();
        write_canonical(&value, &mut out)?;
        Ok(out)
    }

    fn floats(&self) -> Vec<f64> {
        let t = &self.tolerances;
        let mut out = vec![
            t.linear,
            t.angular_snap_deg,
            t.tangent_threshold_deg,
            t.vertex_merge,
        ];
        for r in &self.regions {
            if let Some(res) = &r.residual {
                out.extend([res.rms, res.max]);
            }
            match &r.surface {
                Surface::Plane { origin, normal } => out.extend(origin.iter().chain(normal)),
                Surface::Cylinder {
                    origin,
                    axis,
                    radius,
                    ..
                } => {
                    out.extend(origin.iter().chain(axis));
                    out.push(*radius);
                }
                Surface::Cone {
                    apex,
                    axis,
                    half_angle,
                    ..
                } => {
                    out.extend(apex.iter().chain(axis));
                    out.push(*half_angle);
                }
                Surface::Sphere { center, radius, .. } => {
                    out.extend(center);
                    out.push(*radius);
                }
                Surface::Torus {
                    center,
                    axis,
                    major_radius,
                    minor_radius,
                    ..
                } => {
                    out.extend(center.iter().chain(axis));
                    out.extend([*major_radius, *minor_radius]);
                }
                Surface::Facets { vertices, .. } => out.extend(vertices.iter().flatten()),
            }
        }
        for a in &self.adjacencies {
            for b in &a.boundaries {
                out.push(b.dihedral_deg);
                out.extend(b.points.iter().flatten());
            }
        }
        for v in &self.vertices {
            out.extend(v.position.iter().chain(v.source_positions.iter().flatten()));
        }
        out
    }

    pub fn validate(&self) -> Result<(), IrError> {
        let errors = validate(self);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(IrError::Invalid(errors))
        }
    }
}

pub fn canonicalize(text: &str) -> Result<String, IrError> {
    Ir::from_json(text)?.to_canonical_json()
}

pub fn format_f64(x: f64) -> String {
    let x = if x == 0.0 { 0.0 } else { x };
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits = break_tie(
        x.abs(),
        mantissa.chars().filter(|c| *c != '.').collect(),
        exp,
    );
    let n = digits.len() as i32;
    let body = if (0..16).contains(&exp) {
        if n <= exp + 1 {
            format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..(exp + 1) as usize],
                &digits[(exp + 1) as usize..]
            )
        }
    } else if (-5..0).contains(&exp) {
        format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
    } else if n == 1 {
        format!("{digits}e{exp}")
    } else {
        format!("{}.{}e{exp}", &digits[..1], &digits[1..])
    };
    format!("{sign}{body}")
}

/// A tie needs the exact expansion to continue with a 5 and then only
/// zeros, so its digit `n + 1` (correctly rounded) must be a 5; checking that
/// first skips the slow 40-digit expansion for nearly every value.
fn break_tie(x: f64, digits: String, exp: i32) -> String {
    let n = digits.len();
    let next = format!("{x:.n$e}");
    if next.split_once('e').and_then(|(m, _)| m.bytes().last()) != Some(b'5') {
        return digits;
    }
    break_tie_exact(x, digits, exp)
}

fn break_tie_exact(x: f64, digits: String, exp: i32) -> String {
    let n = digits.len();
    let exact = format!("{x:.40e}");
    let (mantissa, e) = exact
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exact_digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let tail_is_half = exact_digits[n..]
        .strip_prefix('5')
        .is_some_and(|t| t.bytes().all(|b| b == b'0'));
    if e.parse::<i32>() != Ok(exp) || !tail_is_half {
        return digits;
    }
    let lower = exact_digits[..n].to_string();
    let last = lower.as_bytes()[n - 1] - b'0';
    if last == 9 {
        return digits;
    }
    let even = if last.is_multiple_of(2) {
        lower
    } else {
        format!("{}{}", &lower[..n - 1], last + 1)
    };
    let candidate = if n == 1 {
        format!("{even}e{exp}")
    } else {
        format!("{}.{}e{exp}", &even[..1], &even[1..])
    };
    if candidate.parse::<f64>() == Ok(x) {
        even
    } else {
        digits
    }
}

fn write_canonical(value: &Value, out: &mut String) -> Result<(), IrError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                write!(out, "{u}").unwrap();
            } else if let Some(i) = n.as_i64() {
                write!(out, "{i}").unwrap();
            } else {
                let f = n.as_f64().ok_or(IrError::NonFinite)?;
                if !f.is_finite() {
                    return Err(IrError::NonFinite);
                }
                out.push_str(&format_f64(f));
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push('{');
            for (i, (k, v)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_canonical(v, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
        }
    }
    out.push('"');
}

fn dist(a: &Point, b: &Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn is_unit(v: &Point) -> bool {
    (v.iter().map(|c| c * c).sum::<f64>().sqrt() - 1.0).abs() < 1e-9
}

pub fn validate(ir: &Ir) -> Vec<String> {
    let mut e = Vec::new();
    if ir.ir_version != IR_VERSION {
        e.push(format!("ir_version {} is not {IR_VERSION}", ir.ir_version));
    }
    let n_regions = ir.regions.len() as u32;
    let mut seen_triangles = BTreeSet::new();
    let mut shell_of: BTreeMap<u32, usize> = BTreeMap::new();
    for (si, shell) in ir.shells.iter().enumerate() {
        for r in &shell.regions {
            if *r >= n_regions {
                e.push(format!("shell {si} names missing region {r}"));
            } else if shell_of.insert(*r, si).is_some() {
                e.push(format!("region {r} is in more than one shell"));
            }
        }
        match shell.role {
            ShellRole::Outer => {
                if shell.parent.is_some() {
                    e.push(format!("outer shell {si} must have a null parent"));
                }
            }
            ShellRole::Cavity => {
                let parent_ok = shell
                    .parent
                    .and_then(|p| ir.shells.get(p as usize))
                    .is_some_and(|p| p.role == ShellRole::Outer && p.closed);
                if !parent_ok {
                    e.push(format!(
                        "cavity shell {si} must have a closed outer shell as parent"
                    ));
                }
                if !shell.closed {
                    e.push(format!("cavity shell {si} must be closed"));
                }
            }
        }
        if !shell.closed {
            let single_facets = shell.regions.len() == 1
                && ir
                    .regions
                    .get(shell.regions[0] as usize)
                    .is_some_and(|r| r.surface.is_facets());
            if !single_facets {
                e.push(format!("open shell {si} must be exactly one facets region"));
            }
        }
    }
    for (i, region) in ir.regions.iter().enumerate() {
        if region.id as usize != i {
            e.push(format!("region at index {i} has id {}", region.id));
        }
        if !shell_of.contains_key(&region.id) {
            e.push(format!("region {i} is in no shell"));
        }
        for t in &region.triangles {
            if *t >= ir.source.triangle_count {
                e.push(format!(
                    "region {i}: source triangle {t} is outside the input mesh"
                ));
            }
            if !seen_triangles.insert(*t) {
                e.push(format!(
                    "source triangle {t} appears in more than one place"
                ));
            }
        }
        match &region.surface {
            Surface::Plane { normal, .. } => {
                if !is_unit(normal) {
                    e.push(format!("region {i}: plane normal is not a unit vector"));
                }
            }
            Surface::Cylinder { axis, radius, .. } => {
                if !is_unit(axis) || *radius <= 0.0 {
                    e.push(format!("region {i}: bad cylinder axis or radius"));
                }
            }
            Surface::Cone {
                axis, half_angle, ..
            } => {
                if !is_unit(axis)
                    || !(*half_angle > 0.0 && *half_angle < std::f64::consts::FRAC_PI_2)
                {
                    e.push(format!("region {i}: bad cone axis or half_angle"));
                }
            }
            Surface::Sphere { radius, .. } => {
                if *radius <= 0.0 {
                    e.push(format!("region {i}: bad sphere radius"));
                }
            }
            Surface::Torus {
                axis,
                major_radius,
                minor_radius,
                ..
            } => {
                if !is_unit(axis) || *minor_radius <= 0.0 || *major_radius <= *minor_radius {
                    e.push(format!("region {i}: bad torus axis or radii"));
                }
            }
            Surface::Facets { vertices, faces } => {
                if faces.len() != region.triangles.len() {
                    e.push(format!(
                        "region {i}: facets faces and triangles differ in length"
                    ));
                }
                if faces
                    .iter()
                    .flatten()
                    .any(|v| *v as usize >= vertices.len())
                {
                    e.push(format!("region {i}: facets face index out of range"));
                }
                if region.residual.is_some() {
                    e.push(format!("region {i}: facets region must have null residual"));
                }
            }
        }
        if !region.surface.is_facets() && region.residual.is_none() {
            e.push(format!("region {i}: analytic region needs a residual"));
        }
    }
    let mut pairs = BTreeSet::new();
    let mut ends: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    let mut end_kinds: BTreeMap<u32, Vec<Kind>> = BTreeMap::new();
    let mut balance: BTreeMap<(u32, u32), i32> = BTreeMap::new();
    let n_vertices = ir.vertices.len() as u32;
    for adj in &ir.adjacencies {
        let [a, b] = adj.regions;
        if a >= b || b >= n_regions {
            e.push(format!(
                "adjacency ({a}, {b}) must satisfy a < b < region count"
            ));
            continue;
        }
        if !pairs.insert((a, b)) {
            e.push(format!("adjacency ({a}, {b}) listed twice"));
        }
        if shell_of.get(&a) != shell_of.get(&b) {
            e.push(format!("adjacency ({a}, {b}) crosses shells"));
        }
        if ir.regions[a as usize].surface.is_facets() && ir.regions[b as usize].surface.is_facets()
        {
            e.push(format!("adjacency ({a}, {b}) joins two facets regions"));
        }
        if adj.boundaries.is_empty() {
            e.push(format!("adjacency ({a}, {b}) has no boundaries"));
        }
        for bd in &adj.boundaries {
            let min_points = if bd.closed { 3 } else { 2 };
            if bd.points.len() < min_points {
                e.push(format!("adjacency ({a}, {b}): boundary too short"));
                continue;
            }
            if !(0.0..=180.0).contains(&bd.dihedral_deg) {
                e.push(format!("adjacency ({a}, {b}): dihedral_deg out of range"));
            }
            let tangent = bd.dihedral_deg < ir.tolerances.tangent_threshold_deg;
            if tangent != (bd.kind == Kind::Tangent) {
                e.push(format!(
                    "adjacency ({a}, {b}): kind disagrees with threshold"
                ));
            }
            if bd.closed {
                if bd.start_vertex.is_some() || bd.end_vertex.is_some() {
                    e.push(format!(
                        "adjacency ({a}, {b}): closed boundary names vertices"
                    ));
                }
                continue;
            }
            let (Some(s), Some(t)) = (bd.start_vertex, bd.end_vertex) else {
                e.push(format!(
                    "adjacency ({a}, {b}): open boundary needs both vertices"
                ));
                continue;
            };
            for (v, p) in [(s, bd.points[0]), (t, bd.points[bd.points.len() - 1])] {
                match ir.vertices.get(v as usize) {
                    None => e.push(format!("adjacency ({a}, {b}): missing vertex {v}")),
                    Some(vx) => {
                        if dist(&vx.position, &p) > ir.tolerances.vertex_merge {
                            e.push(format!(
                                "adjacency ({a}, {b}): endpoint is not at vertex {v}"
                            ));
                        }
                        ends.entry(v).or_default().extend([a, b]);
                        end_kinds.entry(v).or_default().push(bd.kind);
                    }
                }
            }
            for (r, tail, head) in [(a, s, t), (b, t, s)] {
                *balance.entry((r, tail)).or_default() += 1;
                *balance.entry((r, head)).or_default() -= 1;
            }
        }
    }
    for (i, v) in ir.vertices.iter().enumerate() {
        if v.id as usize != i {
            e.push(format!("vertex at index {i} has id {}", v.id));
        }
        let sorted = v.regions.windows(2).all(|w| w[0] < w[1]);
        let min_regions = if v.role == VertexRole::KindChange {
            2
        } else {
            3
        };
        if v.role == VertexRole::KindChange {
            let kinds = end_kinds.get(&v.id).map(Vec::as_slice).unwrap_or(&[]);
            if v.regions.len() != 2 || kinds.len() != 2 || kinds[0] == kinds[1] {
                e.push(format!(
                    "vertex {i}: a kind_change vertex joins one tangent and one transversal boundary"
                ));
            }
        }
        if v.regions.len() < min_regions || !sorted || v.regions.iter().any(|r| *r >= n_regions) {
            e.push(format!(
                "vertex {i}: regions must be sorted, distinct, and at least 3 for a junction"
            ));
        }
        if v.source_positions.is_empty() {
            e.push(format!("vertex {i}: no source positions"));
        }
        let expect: BTreeSet<u32> = v.regions.iter().copied().collect();
        if ends.get(&v.id) != Some(&expect) {
            e.push(format!(
                "vertex {i}: regions differ from those whose boundaries end here"
            ));
        }
    }
    for ((r, v), n) in &balance {
        if *n != 0 {
            e.push(format!(
                "region {r}: boundaries do not run head to tail at vertex {v}"
            ));
        }
    }
    if ends.keys().any(|v| *v >= n_vertices) {
        e.push("boundary names a missing vertex".to_string());
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_precheck_agrees_with_the_exact_expansion() {
        let mut state = 0x853c49e6748fea9bu64;
        let mut values: Vec<f64> = vec![0.5, 2.5, 0.125, 1.5e-3, 9.5, 1e23, 5e-324, 2.5e15];
        for _ in 0..20000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(f64::from_bits(state >> 2));
            let k = (state % 1000) as f64;
            values.push(k / 8.0);
            values.push(k * 0.05);
        }
        for x in values.into_iter().filter(|x| x.is_finite() && *x != 0.0) {
            let sci = format!("{x:e}");
            let (m, e) = sci.split_once('e').unwrap();
            let digits: String = m
                .trim_start_matches('-')
                .chars()
                .filter(|c| *c != '.')
                .collect();
            let exp: i32 = e.parse().unwrap();
            assert_eq!(
                break_tie(x.abs(), digits.clone(), exp),
                break_tie_exact(x.abs(), digits, exp),
                "{x:e}"
            );
        }
    }

    #[test]
    fn float_format() {
        for (x, s) in [
            (0.0, "0.0"),
            (-0.0, "0.0"),
            (1.0, "1.0"),
            (-2.5, "-2.5"),
            (100.0, "100.0"),
            (0.001, "0.001"),
            (1e-5, "0.00001"),
            (1e-6, "1e-6"),
            (1.5e-7, "1.5e-7"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e16"),
            (1.25e17, "1.25e17"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123.456, "123.456"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(format_f64(x), s, "{x:e}");
            assert_eq!(
                format_f64(s.parse::<f64>().unwrap()),
                if x == 0.0 { "0.0" } else { s }
            );
        }
    }

    #[test]
    fn float_format_breaks_exact_ties_to_even() {
        for (text, want) in [
            ("686036261402521.25", "686036261402521.2"),
            ("-686036261402521.75", "-686036261402521.8"),
        ] {
            assert_eq!(format_f64(text.parse().unwrap()), want);
        }
    }

    #[test]
    fn rejects_non_finite() {
        let mut ir = tiny();
        ir.tolerances.linear = f64::INFINITY;
        assert!(matches!(
            ir.to_canonical_json(),
            Err(IrError::NonFinite) | Err(IrError::Json(_))
        ));
    }

    fn tiny() -> Ir {
        Ir {
            ir_version: IR_VERSION,
            tolerances: Tolerances::default(),
            source: Source {
                triangle_count: 1,
                vertex_count: 3,
            },
            shells: vec![Shell {
                closed: false,
                role: ShellRole::Outer,
                parent: None,
                regions: vec![0],
            }],
            regions: vec![Region {
                id: 0,
                surface: Surface::Facets {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    faces: vec![[0, 1, 2]],
                },
                triangles: vec![0],
                residual: None,
            }],
            adjacencies: vec![],
            vertices: vec![],
        }
    }

    #[test]
    fn open_shell_roundtrip() {
        let ir = tiny();
        ir.validate().unwrap();
        let text = ir.to_canonical_json().unwrap();
        assert!(text.starts_with("{\"adjacencies\":[],"));
        let back = Ir::from_json(&text).unwrap();
        assert_eq!(back, ir);
        assert_eq!(back.to_canonical_json().unwrap(), text);
    }

    #[test]
    fn validation_catches_errors() {
        let mut ir = tiny();
        ir.shells[0].closed = true;
        assert!(ir.validate().is_ok());
        ir.regions[0].id = 3;
        assert!(ir.validate().is_err());
        let mut ir = tiny();
        ir.regions[0].residual = Some(Residual { rms: 0.0, max: 0.0 });
        assert!(ir.validate().is_err());
    }

    fn fixture(name: &str) -> Ir {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/ir")
            .join(format!("{name}.json"));
        Ir::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn reversed_open_boundary_breaks_head_to_tail() {
        let mut ir = fixture("box");
        let bd = &mut ir.adjacencies[0].boundaries[0];
        bd.points.reverse();
        std::mem::swap(&mut bd.start_vertex, &mut bd.end_vertex);
        assert!(ir.validate().is_err());
    }

    #[test]
    fn kind_change_and_cavity_rules() {
        let ir = fixture("kind_change");
        assert_eq!(
            ir.vertices
                .iter()
                .filter(|v| v.role == VertexRole::KindChange)
                .count(),
            2
        );
        let mut bad = ir.clone();
        bad.vertices[0].role = VertexRole::Junction;
        assert!(bad.validate().is_err());
        let mut cav = fixture("cavity");
        assert_eq!(cav.shells[1].role, ShellRole::Cavity);
        let mut open_parent = cav.clone();
        open_parent.shells[0].closed = false;
        assert!(open_parent.validate().is_err());
        cav.shells[1].parent = None;
        assert!(cav.validate().is_err());
        let mut far = fixture("box");
        far.source.triangle_count = 3;
        assert!(far.validate().is_err());
    }

    #[test]
    fn fixtures_are_canonical() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ir");
        let mut count = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|x| x == "json") {
                let text = std::fs::read_to_string(&path).unwrap();
                let canon = canonicalize(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"));
                assert_eq!(canon, text.trim_end_matches('\n'), "{path:?}");
                count += 1;
            }
        }
        assert!(count >= 8);
    }
}
