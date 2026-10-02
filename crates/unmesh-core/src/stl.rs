use std::fmt;

use crate::mesh::{Point, TriangleSoup};

#[derive(Debug, Clone, PartialEq)]
pub enum StlError {
    Truncated { expected: usize, actual: usize },
    Ascii { line: usize, message: String },
    NonFinite,
}

impl fmt::Display for StlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StlError::Truncated { expected, actual } => {
                write!(
                    f,
                    "binary STL truncated: expected {expected} bytes, got {actual}"
                )
            }
            StlError::Ascii { line, message } => write!(f, "ASCII STL line {line}: {message}"),
            StlError::NonFinite => write!(f, "STL contains a non-finite coordinate"),
        }
    }
}

impl std::error::Error for StlError {}

const HEADER_LEN: usize = 80;
const RECORD_LEN: usize = 50;

pub fn read_stl(data: &[u8]) -> Result<TriangleSoup, StlError> {
    let soup = if is_binary(data) {
        read_binary(data)?
    } else {
        read_ascii(data)?
    };
    if soup
        .triangles
        .iter()
        .flatten()
        .flatten()
        .any(|c| !c.is_finite())
    {
        return Err(StlError::NonFinite);
    }
    Ok(soup)
}

fn is_binary(data: &[u8]) -> bool {
    if data.len() < HEADER_LEN + 4 {
        return false;
    }
    let count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    if data.len() == HEADER_LEN + 4 + count * RECORD_LEN {
        return true;
    }
    !data.trim_ascii_start().starts_with(b"solid")
}

fn read_binary(data: &[u8]) -> Result<TriangleSoup, StlError> {
    let count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    let expected = HEADER_LEN + 4 + count * RECORD_LEN;
    if data.len() < expected {
        return Err(StlError::Truncated {
            expected,
            actual: data.len(),
        });
    }
    let triangles = data[84..expected]
        .as_chunks::<RECORD_LEN>()
        .0
        .iter()
        .map(|rec| {
            let coord = |i: usize| {
                let at = 12 + i * 4;
                f32::from_le_bytes(rec[at..at + 4].try_into().unwrap()) as f64
            };
            [0, 1, 2].map(|v| [coord(v * 3), coord(v * 3 + 1), coord(v * 3 + 2)])
        })
        .collect();
    Ok(TriangleSoup { triangles })
}

fn read_ascii(data: &[u8]) -> Result<TriangleSoup, StlError> {
    let text = String::from_utf8_lossy(data);
    let mut triangles = Vec::new();
    let mut pending: Vec<Point> = Vec::with_capacity(3);
    for (index, line) in text.lines().enumerate() {
        let mut words = line.split_ascii_whitespace();
        match words.next() {
            Some("vertex") => {
                let mut point = [0.0; 3];
                for slot in &mut point {
                    let word = words.next().ok_or_else(|| StlError::Ascii {
                        line: index + 1,
                        message: "vertex needs three coordinates".into(),
                    })?;
                    *slot = word.parse().map_err(|_| StlError::Ascii {
                        line: index + 1,
                        message: format!("bad coordinate {word:?}"),
                    })?;
                }
                pending.push(point);
            }
            Some("endfacet") => {
                if pending.len() != 3 {
                    return Err(StlError::Ascii {
                        line: index + 1,
                        message: format!("facet has {} vertices", pending.len()),
                    });
                }
                triangles.push([pending[0], pending[1], pending[2]]);
                pending.clear();
            }
            _ => {}
        }
    }
    Ok(TriangleSoup { triangles })
}

pub fn write_stl_binary(soup: &TriangleSoup) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 4 + soup.len() * RECORD_LEN);
    out.extend_from_slice(&[0u8; HEADER_LEN]);
    out.extend_from_slice(&(soup.len() as u32).to_le_bytes());
    for tri in &soup.triangles {
        let normal = unit_normal(tri);
        for c in normal.iter().chain(tri.iter().flatten()) {
            out.extend_from_slice(&(*c as f32).to_le_bytes());
        }
        out.extend_from_slice(&[0, 0]);
    }
    out
}

fn unit_normal(tri: &[Point; 3]) -> Point {
    let u = sub(tri[1], tri[0]);
    let v = sub(tri[2], tri[0]);
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len == 0.0 {
        [0.0; 3]
    } else {
        n.map(|c| c / len)
    }
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra() -> TriangleSoup {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        TriangleSoup {
            triangles: vec![
                [p[0], p[2], p[1]],
                [p[0], p[1], p[3]],
                [p[1], p[2], p[3]],
                [p[2], p[0], p[3]],
            ],
        }
    }

    #[test]
    fn binary_round_trip() {
        let soup = tetra();
        assert_eq!(read_stl(&write_stl_binary(&soup)).unwrap(), soup);
    }

    #[test]
    fn binary_header_starting_with_solid_is_still_binary() {
        let mut bytes = write_stl_binary(&tetra());
        bytes[..5].copy_from_slice(b"solid");
        assert_eq!(read_stl(&bytes).unwrap(), tetra());
    }

    #[test]
    fn ascii_parses() {
        let text = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let soup = read_stl(text.as_bytes()).unwrap();
        assert_eq!(
            soup.triangles,
            vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        );
    }

    #[test]
    fn truncated_binary_errors() {
        let bytes = write_stl_binary(&tetra());
        let cut = &bytes[..bytes.len() - 10];
        assert!(matches!(read_binary(cut), Err(StlError::Truncated { .. })));
    }
}
