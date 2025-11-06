//! Graph data structure and core operations

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

/// A node in the graph
#[pyclass]
#[derive(Debug)]
pub struct Node {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub attributes: Option<Py<PyDict>>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Node {
            id: self.id.clone(),
            attributes: self.attributes.as_ref().map(|d| d.clone_ref(py)),
        })
    }
}

#[pymethods]
impl Node {
    #[new]
    #[pyo3(signature = (id, attributes=None))]
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
#[derive(Debug)]
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

impl Clone for Edge {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Edge {
            source: self.source.clone(),
            target: self.target.clone(),
            weight: self.weight,
            attributes: self.attributes.as_ref().map(|d| d.clone_ref(py)),
        })
    }
}

#[pymethods]
impl Edge {
    #[new]
    #[pyo3(signature = (source, target, weight=None, attributes=None))]
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
            .or_default();
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

        // Check for duplicate edges
        let duplicate_exists = self.edges.iter().any(|e| {
            e.source == edge.source && e.target == edge.target
        });
        
        if duplicate_exists {
            return Ok(()); // Silently ignore duplicate
        }

        // Add to adjacency list
        self.adjacency_list
            .entry(edge.source.clone())
            .or_default()
            .push(edge.target.clone());

        if !self.directed {
            self.adjacency_list
                .entry(edge.target.clone())
                .or_default()
                .push(edge.source.clone());
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        self.adjacency_list
            .get(node_id)
            .cloned()
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "Node '{}' does not exist",
                    node_id
                ))
            })
    }

    /// Check if the graph is directed
    #[getter]
    fn is_directed(&self) -> bool {
        self.directed
    }

    /// Check if the graph is weighted
    #[getter]
    fn is_weighted(&self) -> bool {
        self.weighted
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get all edges as (source, target) tuples
    pub fn edges(&self) -> Vec<(String, String)> {
        self.edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect()
    }

    /// Check if a node exists (O(1) hash lookup)
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={}, directed={})",
            self.num_nodes(),
            self.num_edges(),
            self.directed
        )
    }
}

impl Graph {
    /// Get the adjacency list (internal method)
    pub fn get_adjacency_list(&self) -> &FxHashMap<String, Vec<String>> {
        &self.adjacency_list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        let graph = Graph::new(false, false);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
        assert!(!graph.is_directed());
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = Graph::new(false, false);
        let node1 = Node::new("A".to_string(), None);
        let node2 = Node::new("B".to_string(), None);
        
        assert!(graph.add_node(node1).is_ok());
        assert!(graph.add_node(node2).is_ok());
        assert_eq!(graph.num_nodes(), 2);
    }

    #[test]
    fn test_add_edges() {
        let mut graph = Graph::new(false, false);
        
        // Add nodes first
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        
        // Add edge
        let edge = Edge::new("A".to_string(), "B".to_string(), Some(1.0), None);
        assert!(graph.add_edge(edge).is_ok());
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new(false, false);
        
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("A".to_string(), "C".to_string(), None, None))
            .unwrap();
        
        let neighbors = graph.neighbors("A").unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"B".to_string()));
        assert!(neighbors.contains(&"C".to_string()));
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::new(true, false);
        
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        
        // In directed graph, B should not have A as neighbor
        let b_neighbors = graph.neighbors("B").unwrap();
        assert_eq!(b_neighbors.len(), 0);
        
        let a_neighbors = graph.neighbors("A").unwrap();
        assert_eq!(a_neighbors.len(), 1);
    }
}
