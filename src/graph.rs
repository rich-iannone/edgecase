//! Graph data structure and core operations

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustc_hash::FxHashMap;
use std::collections::HashSet;

/// A node in the graph
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub attributes: Option<Py<PyDict>>,
}

#[pymethods]
impl Node {
    #[new]
    fn new(id: String, attributes: Option<Py<PyDict>>) -> Self {
        Node { id, attributes }
    }

    fn __repr__(&self) -> String {
        format!("Node(id='{}')", self.id)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// An edge connecting two nodes
#[pyclass]
#[derive(Clone, Debug)]
pub struct Edge {
    #[pyo3(get, set)]
    pub source: String,
    #[pyo3(get, set)]
    pub target: String,
    #[pyo3(get, set)]
    pub weight: Option<f64>,
    #[pyo3(get, set)]
    pub attributes: Option<Py<PyDict>>,
}

#[pymethods]
impl Edge {
    #[new]
    fn new(
        source: String,
        target: String,
        weight: Option<f64>,
        attributes: Option<Py<PyDict>>,
    ) -> Self {
        Edge {
            source,
            target,
            weight,
            attributes,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Edge(source='{}', target='{}', weight={:?})",
            self.source, self.target, self.weight
        )
    }
}

