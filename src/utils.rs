//! Utility functions

use crate::graph::{Edge, Graph};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Utility function to validate graph integrity
#[pyfunction]
pub fn validate_graph() -> PyResult<bool> {
    Ok(true)
}

/// A `seed=` argument. Any integer is accepted, taken modulo 2^64, so negative and very large
/// seeds work as they do with Python's `random` instead of raising `OverflowError`.
#[derive(Clone, Copy, Debug)]
pub struct Seed(u64);

impl From<u64> for Seed {
    fn from(seed: u64) -> Seed {
        Seed(seed)
    }
}

impl From<Seed> for u64 {
    fn from(seed: Seed) -> u64 {
        seed.0
    }
}

impl<'py> FromPyObject<'py> for Seed {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<u64>() {
            return Ok(Seed(v));
        }
        let index = ob.py().import("operator")?.call_method1("index", (ob,))?;
        Ok(Seed(index.call_method1("__and__", (u64::MAX,))?.extract()?))
    }
}

/// The number an edge holds under the attribute `name`, also looked up in a nested
/// `"attributes"` dict. `"weight"` falls back to the weight field, where weighted graphs keep
/// it, and then to 1.0. `None` when the edge holds no such number.
pub(crate) fn named_edge_weight(py: Python, edge: &Edge, name: &str) -> Option<f64> {
    let found = edge.attributes.as_ref().and_then(|attrs| {
        let attrs = attrs.bind(py);
        let value = match attrs.get_item(name).ok().flatten() {
            Some(v) => v,
            None => {
                let nested = attrs.get_item("attributes").ok().flatten()?;
                nested
                    .downcast::<PyDict>()
                    .ok()?
                    .get_item(name)
                    .ok()
                    .flatten()?
            }
        };
        value.extract::<f64>().ok()
    });
    match found {
        Some(w) => Some(w),
        None if name == "weight" => Some(edge.weight.unwrap_or(1.0)),
        None => None,
    }
}

/// [`named_edge_weight`] for the argument `param=name`, raising `ValueError` when the edge
/// holds no such number, so a misspelled name doesn't silently count every edge as 1.0.
pub(crate) fn require_named_edge_weight(
    py: Python,
    edge: &Edge,
    name: &str,
    param: &str,
) -> PyResult<f64> {
    named_edge_weight(py, edge, name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Edge {} ({} -> {}) has no numeric '{}' attribute, which {}='{}' requires",
            edge.id, edge.source, edge.target, name, param, name
        ))
    })
}

/// Check up front that every edge holds a number under `name` (see
/// [`require_named_edge_weight`]), for algorithms that look weights up later without a way to
/// fail. Does nothing when `name` is `None`.
pub(crate) fn check_named_edge_weights(
    graph: &Graph,
    name: Option<&str>,
    param: &str,
) -> PyResult<()> {
    let Some(name) = name else {
        return Ok(());
    };
    Python::with_gil(|py| {
        for edge in graph.get_edges_internal() {
            require_named_edge_weight(py, edge, name, param)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_graph() {
        assert!(validate_graph().unwrap());
    }
}
