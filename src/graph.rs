//! Graph data structure and core operations

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};

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
    pub id: u64,
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
            id: self.id,
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
    #[pyo3(signature = (id, source, target, weight=None, attributes=None))]
    fn new(
        id: u64,
        source: String,
        target: String,
        weight: Option<f64>,
        attributes: Option<Py<PyDict>>,
    ) -> Self {
        Edge {
            id,
            source,
            target,
            weight,
            attributes,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Edge(id={}, source='{}', target='{}', weight={:?})",
            self.id, self.source, self.target, self.weight
        )
    }
}

/// Main graph structure
#[pyclass]
pub struct Graph {
    nodes: FxHashMap<String, Node>,
    edges: Vec<Edge>,
    adjacency_list: FxHashMap<String, Vec<String>>,
    next_edge_id: u64,
    directed: bool,
    weighted: bool,
    // Selection state
    selected_nodes: FxHashSet<String>,
    selected_edges: FxHashSet<u64>,
    // Creation order tracking
    node_creation_order: Vec<String>,
    edge_creation_order: Vec<u64>,
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
            next_edge_id: 1,
            directed,
            weighted,
            selected_nodes: FxHashSet::default(),
            selected_edges: FxHashSet::default(),
            node_creation_order: Vec::new(),
            edge_creation_order: Vec::new(),
        }
    }

    /// Add a node to the graph
    fn add_node(&mut self, node: Node) -> PyResult<()> {
        let node_id = node.id.clone();
        
        // Track creation order only if this is a new node
        if !self.nodes.contains_key(&node_id) {
            self.node_creation_order.push(node_id.clone());
        }
        
        self.nodes.insert(node_id.clone(), node);
        self.adjacency_list
            .entry(node_id)
            .or_default();
        Ok(())
    }

    /// Add an edge to the graph
    fn add_edge(&mut self, mut edge: Edge) -> PyResult<()> {
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

        // Auto-assign edge ID if it's 0 (default/unset)
        if edge.id == 0 {
            edge.id = self.next_edge_id;
            self.next_edge_id += 1;
        } else {
            // If user provided an ID, make sure next_edge_id is beyond it
            if edge.id >= self.next_edge_id {
                self.next_edge_id = edge.id + 1;
            }
        }

        // Add to adjacency list (multi-edges allowed)
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

        // Track creation order
        self.edge_creation_order.push(edge.id);
        
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

    /// Get all edges as (id, source, target) tuples
    pub fn edges(&self) -> Vec<(u64, String, String)> {
        self.edges
            .iter()
            .map(|e| (e.id, e.source.clone(), e.target.clone()))
            .collect()
    }

    /// Get all edge IDs
    pub fn edge_ids(&self) -> Vec<u64> {
        self.edges.iter().map(|e| e.id).collect()
    }

    /// Get edge by ID
    pub fn get_edge_by_id(&self, edge_id: u64) -> Option<(String, String)> {
        self.edges
            .iter()
            .find(|e| e.id == edge_id)
            .map(|e| (e.source.clone(), e.target.clone()))
    }

    /// Check if an edge exists by ID
    pub fn has_edge_id(&self, edge_id: u64) -> bool {
        self.edges.iter().any(|e| e.id == edge_id)
    }

    /// Delete edge by ID
    pub fn delete_edge_by_id(&mut self, edge_id: u64) -> PyResult<()> {
        let edge_idx = self.edges.iter().position(|e| e.id == edge_id);
        
        if let Some(idx) = edge_idx {
            let edge = &self.edges[idx];
            
            // Remove from adjacency list
            if let Some(neighbors) = self.adjacency_list.get_mut(&edge.source) {
                if let Some(pos) = neighbors.iter().position(|n| n == &edge.target) {
                    neighbors.remove(pos);
                }
            }
            
            if !self.directed {
                if let Some(neighbors) = self.adjacency_list.get_mut(&edge.target) {
                    if let Some(pos) = neighbors.iter().position(|n| n == &edge.source) {
                        neighbors.remove(pos);
                    }
                }
            }
            
            self.edges.remove(idx);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Edge with ID {} does not exist",
                edge_id
            )))
        }
    }

    /// Check if a node exists (O(1) hash lookup)
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    // ========== Selection Methods ==========

    /// Select nodes by their IDs
    pub fn select_nodes_by_id(&mut self, node_ids: Vec<String>) -> PyResult<()> {
        for node_id in node_ids {
            if !self.nodes.contains_key(&node_id) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Node '{}' does not exist",
                    node_id
                )));
            }
            self.selected_nodes.insert(node_id);
        }
        Ok(())
    }

    /// Select edges by their IDs
    pub fn select_edges_by_id(&mut self, edge_ids: Vec<u64>) -> PyResult<()> {
        for edge_id in edge_ids {
            if !self.has_edge_id(edge_id) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Edge with ID {} does not exist",
                    edge_id
                )));
            }
            self.selected_edges.insert(edge_id);
        }
        Ok(())
    }

    /// Select edges connected to specified nodes
    pub fn select_edges_by_node_id(&mut self, node_ids: Vec<String>) -> PyResult<()> {
        let node_set: FxHashSet<String> = node_ids.into_iter().collect();
        
        for edge in &self.edges {
            if node_set.contains(&edge.source) || node_set.contains(&edge.target) {
                self.selected_edges.insert(edge.id);
            }
        }
        Ok(())
    }

    /// Select nodes by degree criteria
    #[pyo3(signature = (min_degree=None, max_degree=None, degree_type="total"))]
    pub fn select_nodes_by_degree(
        &mut self,
        min_degree: Option<usize>,
        max_degree: Option<usize>,
        degree_type: &str,
    ) -> PyResult<()> {
        for node_id in self.nodes.keys() {
            let degree = match degree_type {
                "in" => self.in_degree(node_id),
                "out" => self.out_degree(node_id),
                "total" => self.degree(node_id),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid degree_type '{}'. Must be 'in', 'out', or 'total'",
                        degree_type
                    )))
                }
            };

            let matches = match (min_degree, max_degree) {
                (Some(min), Some(max)) => degree >= min && degree <= max,
                (Some(min), None) => degree >= min,
                (None, Some(max)) => degree <= max,
                (None, None) => true,
            };

            if matches {
                self.selected_nodes.insert(node_id.clone());
            }
        }
        Ok(())
    }

    /// Get the degree of a node (total neighbors)
    fn degree(&self, node_id: &str) -> usize {
        self.adjacency_list
            .get(node_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get in-degree of a node (for directed graphs)
    fn in_degree(&self, node_id: &str) -> usize {
        if !self.directed {
            return self.degree(node_id);
        }
        
        self.edges
            .iter()
            .filter(|e| &e.target == node_id)
            .count()
    }

    /// Get out-degree of a node (for directed graphs)
    fn out_degree(&self, node_id: &str) -> usize {
        if !self.directed {
            return self.degree(node_id);
        }
        
        self.edges
            .iter()
            .filter(|e| &e.source == node_id)
            .count()
    }

    /// Deselect specific nodes
    pub fn deselect_nodes(&mut self, node_ids: Vec<String>) -> PyResult<()> {
        for node_id in node_ids {
            self.selected_nodes.remove(&node_id);
        }
        Ok(())
    }

    /// Deselect specific edges
    pub fn deselect_edges(&mut self, edge_ids: Vec<u64>) -> PyResult<()> {
        for edge_id in edge_ids {
            self.selected_edges.remove(&edge_id);
        }
        Ok(())
    }

    /// Clear all selections
    pub fn clear_selection(&mut self) -> PyResult<()> {
        self.selected_nodes.clear();
        self.selected_edges.clear();
        Ok(())
    }

    /// Invert node selection
    pub fn invert_node_selection(&mut self) -> PyResult<()> {
        let all_nodes: FxHashSet<String> = self.nodes.keys().cloned().collect();
        self.selected_nodes = all_nodes
            .difference(&self.selected_nodes)
            .cloned()
            .collect();
        Ok(())
    }

    /// Invert edge selection
    pub fn invert_edge_selection(&mut self) -> PyResult<()> {
        let all_edges: FxHashSet<u64> = self.edges.iter().map(|e| e.id).collect();
        self.selected_edges = all_edges
            .difference(&self.selected_edges)
            .cloned()
            .collect();
        Ok(())
    }

    /// Get currently selected node IDs
    pub fn get_selected_nodes(&self) -> Vec<String> {
        self.selected_nodes.iter().cloned().collect()
    }

    /// Get currently selected edge IDs
    pub fn get_selected_edges(&self) -> Vec<u64> {
        self.selected_edges.iter().cloned().collect()
    }

    /// Get count of selected nodes
    pub fn count_selected_nodes(&self) -> usize {
        self.selected_nodes.len()
    }

    /// Get count of selected edges
    pub fn count_selected_edges(&self) -> usize {
        self.selected_edges.len()
    }

    /// Select nodes using a Python callable/predicate function
    /// The predicate receives a Node object and should return True/False
    pub fn select_nodes(&mut self, py: Python, predicate: PyObject) -> PyResult<()> {
        for (node_id, node) in &self.nodes {
            // Call the predicate with the node
            let result = predicate.call1(py, (node.clone(),))?;
            let matches: bool = result.extract(py)?;
            
            if matches {
                self.selected_nodes.insert(node_id.clone());
            }
        }
        Ok(())
    }

    /// Select edges using a Python callable/predicate function
    /// The predicate receives an Edge object and should return True/False
    pub fn select_edges(&mut self, py: Python, predicate: PyObject) -> PyResult<()> {
        for edge in &self.edges {
            // Call the predicate with the edge
            let result = predicate.call1(py, (edge.clone(),))?;
            let matches: bool = result.extract(py)?;
            
            if matches {
                self.selected_edges.insert(edge.id);
            }
        }
        Ok(())
    }

    /// Get a node object by ID (for use with selection predicates)
    pub fn get_node(&self, node_id: &str) -> Option<Node> {
        self.nodes.get(node_id).cloned()
    }

    /// Get an edge object by ID (for use with selection predicates)
    pub fn get_edge(&self, edge_id: u64) -> Option<Edge> {
        self.edges.iter().find(|e| e.id == edge_id).cloned()
    }

    /// Select the last N nodes that were created
    #[pyo3(signature = (n))]
    pub fn select_last_nodes_created(&mut self, n: usize) -> PyResult<()> {
        let total = self.node_creation_order.len();
        let start_idx = if n >= total { 0 } else { total - n };
        
        for node_id in &self.node_creation_order[start_idx..] {
            self.selected_nodes.insert(node_id.clone());
        }
        Ok(())
    }

    /// Select the last N edges that were created
    #[pyo3(signature = (n))]
    pub fn select_last_edges_created(&mut self, n: usize) -> PyResult<()> {
        let total = self.edge_creation_order.len();
        let start_idx = if n >= total { 0 } else { total - n };
        
        for edge_id in &self.edge_creation_order[start_idx..] {
            self.selected_edges.insert(*edge_id);
        }
        Ok(())
    }

    /// Get the creation order of all nodes
    pub fn get_node_creation_order(&self) -> Vec<String> {
        self.node_creation_order.clone()
    }

    /// Get the creation order of all edges
    pub fn get_edge_creation_order(&self) -> Vec<u64> {
        self.edge_creation_order.clone()
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
        
        // Add edge (ID will be auto-assigned)
        let edge = Edge::new(0, "A".to_string(), "B".to_string(), Some(1.0), None);
        assert!(graph.add_edge(edge).is_ok());
        assert_eq!(graph.num_edges(), 1);
        
        // Check edge ID was auto-assigned
        let edge_ids = graph.edge_ids();
        assert_eq!(edge_ids.len(), 1);
        assert_eq!(edge_ids[0], 1);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new(false, false);
        
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new(0, "A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new(0, "A".to_string(), "C".to_string(), None, None))
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
            .add_edge(Edge::new(0, "A".to_string(), "B".to_string(), None, None))
            .unwrap();
        
        // In directed graph, B should not have A as neighbor
        let b_neighbors = graph.neighbors("B").unwrap();
        assert_eq!(b_neighbors.len(), 0);
        
        let a_neighbors = graph.neighbors("A").unwrap();
        assert_eq!(a_neighbors.len(), 1);
    }
}
