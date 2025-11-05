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

/// Main graph structure
#[pyclass]
pub struct Graph {
    nodes: FxHashMap<String, Node>,
    edges: Vec<Edge>,
    adjacency_list: FxHashMap<String, Vec<String>>,
    directed: bool,
    weighted: bool,
}

#[pymethods]
impl Graph {
    #[new]
    #[pyo3(signature = (directed=false, weighted=false))]
    fn new(directed: bool, weighted: bool) -> Self {
        Graph {
            nodes: FxHashMap::default(),
            edges: Vec::new(),
            adjacency_list: FxHashMap::default(),
            directed,
            weighted,
        }
    }

    /// Add a node to the graph
    fn add_node(&mut self, node: Node) -> PyResult<()> {
        let node_id = node.id.clone();
        self.nodes.insert(node_id.clone(), node);
        self.adjacency_list
            .entry(node_id)
            .or_insert_with(Vec::new);
        Ok(())
    }

    /// Add an edge to the graph
    fn add_edge(&mut self, edge: Edge) -> PyResult<()> {
        // Ensure nodes exist
        if !self.nodes.contains_key(&edge.source) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Source node '{}' does not exist",
                edge.source
            )));
        }
        if !self.nodes.contains_key(&edge.target) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target node '{}' does not exist",
                edge.target
            )));
        }

        // Add to adjacency list
        self.adjacency_list
            .entry(edge.source.clone())
            .or_insert_with(Vec::new)
            .push(edge.target.clone());

        if !self.directed {
            self.adjacency_list
                .entry(edge.target.clone())
                .or_insert_with(Vec::new)
                .push(edge.source.clone());
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Get the number of nodes
    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    fn num_edges(&self) -> usize {
        self.edges.len()
    }

