pub mod moran;

#[cfg(feature = "python")]
mod python {
    use crate::moran::{self, Rule};
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::{PyNotImplementedError, PyValueError};
    use pyo3::prelude::*;

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn simulate<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i32>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        n_mutants: Vec<u32>,
        fitness: Vec<f64>,
        update_rule: &str,
        skip_null_steps: bool,
        n_repeats: usize,
        max_steps: u64,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
        let rule = match update_rule {
            "Bd" => Rule::Bd,
            // Implemented and tested in moran.rs; blocked until the dB definition
            // (raw vs row-normalized weights) is settled.
            "dB" => return Err(PyNotImplementedError::new_err("dB is not available yet")),
            _ => return Err(PyValueError::new_err("update_rule must be 'Bd' or 'dB'")),
        };
        let (indptr, indices, weights) = (indptr.as_slice()?, indices.as_slice()?, weights.as_slice()?);
        let runs = py.detach(|| {
            moran::simulate(
                indptr, indices, weights, &n_mutants, &fitness, rule, skip_null_steps, n_repeats, max_steps,
                seed,
            )
        });
        Ok((runs.winner.into_pyarray(py), runs.steps.into_pyarray(py), runs.n_changes.into_pyarray(py)))
    }

    #[pymodule]
    fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(simulate, m)?)
    }
}
