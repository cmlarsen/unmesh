use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use unmesh_core::{StlError, TriangleSoup};

#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn stl_err(e: StlError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn soup_to_flat(soup: &TriangleSoup) -> Vec<f64> {
    let mut flat = Vec::with_capacity(soup.triangles.len() * 9);
    for tri in &soup.triangles {
        for p in tri {
            flat.extend_from_slice(p);
        }
    }
    flat
}

fn soup_from_array(py: Python<'_>, tris: &Bound<'_, PyAny>) -> PyResult<TriangleSoup> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let arr = np.call_method("ascontiguousarray", (tris,), Some(&kwargs))?;
    let arr = arr
        .cast_into::<numpy::PyArrayDyn<f64>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let shape = arr.shape();
    let valid = matches!(shape, [_, 3, 3] | [0]);
    if !valid {
        return Err(PyValueError::new_err(format!(
            "expected an array of shape (n, 3, 3), got {shape:?}"
        )));
    }
    let readonly = arr.readonly();
    let data = readonly.as_slice()?;
    if data.iter().any(|c| !c.is_finite()) {
        return Err(PyValueError::new_err(
            "triangle array has a non-finite coordinate",
        ));
    }
    Ok(TriangleSoup {
        triangles: data
            .as_chunks::<9>()
            .0
            .iter()
            .map(|c| [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]])
            .collect(),
    })
}

#[pyfunction]
fn read_stl<'py>(py: Python<'py>, path: std::path::PathBuf) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let flat = py.detach(|| -> PyResult<Vec<f64>> {
        let data = std::fs::read(&path).map_err(PyErr::from)?;
        let soup = unmesh_core::read_stl(&data).map_err(stl_err)?;
        Ok(soup_to_flat(&soup))
    })?;
    let n = flat.len() / 9;
    flat.into_pyarray(py).reshape([n, 3, 3])
}

#[pyfunction]
fn write_stl(py: Python<'_>, path: std::path::PathBuf, tris: &Bound<'_, PyAny>) -> PyResult<()> {
    let soup = soup_from_array(py, tris)?;
    py.detach(|| {
        let bytes = unmesh_core::write_stl_binary(&soup);
        std::fs::write(&path, bytes).map_err(PyErr::from)
    })
}

type WeldOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyDict>,
);

#[pyfunction]
fn weld<'py>(
    py: Python<'py>,
    tris: &Bound<'py, PyAny>,
    tolerance: f64,
) -> PyResult<WeldOutput<'py>> {
    let soup = soup_from_array(py, tris)?;
    let (verts, faces, source, report) = py.detach(|| {
        let (mesh, source, report) = unmesh_core::weld(&soup, tolerance)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let verts: Vec<f64> = mesh.vertices.iter().flatten().copied().collect();
        let faces: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
        Ok::<_, PyErr>((verts, faces, source, report))
    })?;
    let nv = verts.len() / 3;
    let nf = faces.len() / 3;
    let dict = PyDict::new(py);
    dict.set_item("input_corners", report.input_corners)?;
    dict.set_item("unique_vertices", report.unique_vertices)?;
    dict.set_item("degenerate_dropped", report.degenerate_dropped)?;
    Ok((
        verts.into_pyarray(py).reshape([nv, 3])?,
        faces.into_pyarray(py).reshape([nf, 3])?,
        source.into_pyarray(py),
        dict,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(read_stl, m)?)?;
    m.add_function(wrap_pyfunction!(write_stl, m)?)?;
    m.add_function(wrap_pyfunction!(weld, m)?)?;
    Ok(())
}
