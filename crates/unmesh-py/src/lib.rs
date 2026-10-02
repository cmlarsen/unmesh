use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use unmesh_core::{
    ConvertError, ConvertOptions, ConvertOutput, IndexedMesh, StlError, TriangleSoup,
};

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

fn convert_err(e: ConvertError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn options_from(
    linear_tolerance: Option<f64>,
    angular_snap_deg: f64,
    tangent_threshold_deg: f64,
    vertex_merge: f64,
) -> ConvertOptions {
    ConvertOptions {
        linear_tolerance,
        angular_snap_deg,
        tangent_threshold_deg,
        vertex_merge,
    }
}

fn output_to_py<'py>(
    py: Python<'py>,
    out: ConvertOutput,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let text = out
        .ir
        .to_canonical_json()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let report = PyDict::new(py);
    report.set_item("max_deviation", out.report.max_deviation)?;
    report.set_item("rms_deviation", out.report.rms_deviation)?;
    report.set_item("analytic_area_fraction", out.report.analytic_area_fraction)?;
    let counts = PyDict::new(py);
    for (k, v) in &out.report.region_counts {
        counts.set_item(k, v)?;
    }
    report.set_item("region_counts", counts)?;
    let warnings = PyList::empty(py);
    for w in &out.report.warnings {
        warnings.append((&w.code, &w.message))?;
    }
    report.set_item("warnings", warnings)?;
    Ok((text, report))
}

#[pyfunction]
#[pyo3(signature = (tris, linear_tolerance, angular_snap_deg, tangent_threshold_deg, vertex_merge))]
fn convert_soup<'py>(
    py: Python<'py>,
    tris: &Bound<'py, PyAny>,
    linear_tolerance: Option<f64>,
    angular_snap_deg: f64,
    tangent_threshold_deg: f64,
    vertex_merge: f64,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let soup = soup_from_array(py, tris)?;
    let options = options_from(
        linear_tolerance,
        angular_snap_deg,
        tangent_threshold_deg,
        vertex_merge,
    );
    let out = py
        .detach(|| unmesh_core::convert_soup(&soup, &options))
        .map_err(convert_err)?;
    output_to_py(py, out)
}

#[pyfunction]
#[pyo3(signature = (vertices, faces, linear_tolerance, angular_snap_deg, tangent_threshold_deg, vertex_merge))]
fn convert_indexed<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyAny>,
    faces: &Bound<'py, PyAny>,
    linear_tolerance: Option<f64>,
    angular_snap_deg: f64,
    tangent_threshold_deg: f64,
    vertex_merge: f64,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let varr = np.call_method("ascontiguousarray", (vertices,), Some(&kwargs))?;
    let varr = varr
        .cast_into::<numpy::PyArrayDyn<f64>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !matches!(varr.shape(), [_, 3]) {
        return Err(PyValueError::new_err(format!(
            "expected vertices of shape (v, 3), got {:?}",
            varr.shape()
        )));
    }
    let vdata = varr.readonly();
    let vdata = vdata.as_slice()?;
    if vdata.iter().any(|c| !c.is_finite()) {
        return Err(PyValueError::new_err(
            "vertices have a non-finite coordinate",
        ));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("int64")?)?;
    let farr = np.call_method("ascontiguousarray", (faces,), Some(&kwargs))?;
    let farr = farr
        .cast_into::<numpy::PyArrayDyn<i64>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !matches!(farr.shape(), [_, 3] | [0]) {
        return Err(PyValueError::new_err(format!(
            "expected faces of shape (f, 3), got {:?}",
            farr.shape()
        )));
    }
    let fdata = farr.readonly();
    let fdata = fdata.as_slice()?;
    let nv = vdata.len() / 3;
    if fdata.iter().any(|&i| i < 0 || i as usize >= nv) {
        return Err(PyValueError::new_err("face index out of range"));
    }
    let mesh = IndexedMesh {
        vertices: vdata
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        faces: fdata
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
            .collect(),
    };
    let options = options_from(
        linear_tolerance,
        angular_snap_deg,
        tangent_threshold_deg,
        vertex_merge,
    );
    let out = py
        .detach(|| unmesh_core::convert(&mesh, &options))
        .map_err(convert_err)?;
    output_to_py(py, out)
type SampleOutput<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<u32>>);

fn stats_dict<'py>(py: Python<'py>, s: &unmesh_core::judge::Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("count", s.count)?;
    d.set_item("max", s.max)?;
    d.set_item("p99", s.p99)?;
    d.set_item("p95", s.p95)?;
    d.set_item("mean", s.mean)?;
    Ok(d)
}

fn comparison_dict<'py>(
    py: Python<'py>,
    c: &unmesh_core::judge::Comparison,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ir_to_mesh", stats_dict(py, &c.ir_to_mesh)?)?;
    d.set_item("mesh_to_ir", stats_dict(py, &c.mesh_to_ir)?)?;
    Ok(d)
}

fn judge_options(
    samples_per_mm2: f64,
    seed: u64,
    include_vertices: bool,
) -> unmesh_core::judge::JudgeOptions {
    unmesh_core::judge::JudgeOptions {
        samples_per_mm2,
        seed,
        include_vertices,
    }
}

fn judge_err(e: unmesh_core::judge::JudgeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (ir_json, input_tris, truth_tris=None, samples_per_mm2=10.0, seed=0, include_vertices=true))]
fn judge_ir<'py>(
    py: Python<'py>,
    ir_json: &str,
    input_tris: &Bound<'py, PyAny>,
    truth_tris: Option<&Bound<'py, PyAny>>,
    samples_per_mm2: f64,
    seed: u64,
    include_vertices: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let ir =
        unmesh_core::Ir::from_json(ir_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let input = soup_from_array(py, input_tris)?;
    let truth = truth_tris.map(|t| soup_from_array(py, t)).transpose()?;
    let opts = judge_options(samples_per_mm2, seed, include_vertices);
    let result = py
        .detach(|| {
            unmesh_core::judge::judge(
                &ir,
                &input.triangles,
                truth.as_ref().map(|t| t.triangles.as_slice()),
                &opts,
            )
        })
        .map_err(judge_err)?;
    let out = PyDict::new(py);
    out.set_item("input", comparison_dict(py, &result.input)?)?;
    match &result.truth {
        Some(c) => out.set_item("truth", comparison_dict(py, c)?)?,
        None => out.set_item("truth", py.None())?,
    }
    out.set_item("region_max", result.region_max.into_pyarray(py))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (ir_json, input_tris, samples_per_mm2=10.0, seed=0, include_vertices=true))]
fn sample_ir<'py>(
    py: Python<'py>,
    ir_json: &str,
    input_tris: &Bound<'py, PyAny>,
    samples_per_mm2: f64,
    seed: u64,
    include_vertices: bool,
) -> PyResult<SampleOutput<'py>> {
    let ir =
        unmesh_core::Ir::from_json(ir_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let input = soup_from_array(py, input_tris)?;
    let opts = judge_options(samples_per_mm2, seed, include_vertices);
    let (points, owner) = py
        .detach(|| unmesh_core::judge::sample_ir(&ir, &input.triangles, &opts))
        .map_err(judge_err)?;
    let n = points.len();
    let flat: Vec<f64> = points.into_iter().flatten().collect();
    Ok((
        flat.into_pyarray(py).reshape([n, 3])?,
        owner.into_pyarray(py),
    ))
}

#[pyfunction]
fn mesh_distances<'py>(
    py: Python<'py>,
    tris: &Bound<'py, PyAny>,
    points: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let soup = soup_from_array(py, tris)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (points, np.getattr("float64")?))?;
    let arr = arr
        .cast_into::<numpy::PyArrayDyn<f64>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if arr.shape().len() != 2 || arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("expected points of shape (n, 3)"));
    }
    let readonly = arr.readonly();
    let pts: Vec<[f64; 3]> = readonly.as_slice()?.as_chunks::<3>().0.to_vec();
    let d = py
        .detach(|| unmesh_core::judge::mesh_distances(&soup.triangles, &pts))
        .map_err(judge_err)?;
    Ok(d.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(read_stl, m)?)?;
    m.add_function(wrap_pyfunction!(write_stl, m)?)?;
    m.add_function(wrap_pyfunction!(weld, m)?)?;
    m.add_function(wrap_pyfunction!(convert_soup, m)?)?;
    m.add_function(wrap_pyfunction!(convert_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(judge_ir, m)?)?;
    m.add_function(wrap_pyfunction!(sample_ir, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_distances, m)?)?;
    Ok(())
}
