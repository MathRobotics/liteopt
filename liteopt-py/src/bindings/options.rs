use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) struct PyOptions<'py> {
    dict: Option<Bound<'py, PyDict>>,
    solver: &'static str,
    name: &'static str,
}

impl<'py> PyOptions<'py> {
    pub(crate) fn from_python(
        py: Python<'py>,
        solver: &'static str,
        name: &'static str,
        options: Option<Py<PyAny>>,
        allowed: &[&str],
    ) -> PyResult<Self> {
        let Some(options) = options else {
            return Ok(Self {
                dict: None,
                solver,
                name,
            });
        };

        let options = options.bind(py);
        let Ok(dict) = options.cast::<PyDict>() else {
            return Err(PyValueError::new_err(format!(
                "{solver}: {name} must be a dict"
            )));
        };

        for (key, _) in dict.iter() {
            let key = key.extract::<String>()?;
            if !allowed.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "{solver}: unknown {name} key '{key}'"
                )));
            }
        }

        Ok(Self {
            dict: Some(dict.clone()),
            solver,
            name,
        })
    }

    pub(crate) fn f64(&self, key: &str) -> PyResult<Option<f64>> {
        self.extract(key)
    }

    pub(crate) fn usize(&self, key: &str) -> PyResult<Option<usize>> {
        self.extract(key)
    }

    pub(crate) fn bool(&self, key: &str) -> PyResult<Option<bool>> {
        self.extract(key)
    }

    pub(crate) fn string(&self, key: &str) -> PyResult<Option<String>> {
        self.extract(key)
    }

    pub(crate) fn py(&self, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let Some(dict) = &self.dict else {
            return Ok(None);
        };
        Ok(dict.get_item(key)?.map(|value| value.unbind()))
    }

    fn extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        for<'a> T: FromPyObject<'a, 'py>,
    {
        let Some(dict) = &self.dict else {
            return Ok(None);
        };
        let Some(value) = dict.get_item(key)? else {
            return Ok(None);
        };
        value.extract::<T>().map(Some).map_err(|_| {
            PyValueError::new_err(format!("{}: invalid {}.{key}", self.solver, self.name))
        })
    }
}
