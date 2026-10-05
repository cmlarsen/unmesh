pub mod api;
mod convert;
pub mod ir;
pub mod judge;
pub mod mesh;
pub mod stl;
pub mod weld;

pub use api::{
    ConvertError, ConvertOptions, ConvertOutput, ConvertWarning, Report, convert,
    convert_from_labels, convert_soup, convert_soup_from_labels,
};
pub use ir::{Ir, IrError};
pub use mesh::{IndexedMesh, TriangleSoup};
pub use stl::{StlError, read_stl, write_stl_binary};
pub use weld::{WeldError, WeldReport, weld};
