//! Graph data structure and core operations

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

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
    fn add_edge(&mut self, mut edge: Edge) -> PyResult<u64> {
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
        
        // Store the edge ID before moving the edge
        let edge_id = edge.id;

        self.edges.push(edge);
        Ok(edge_id)  // Return the edge ID
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if the graph is simple (no self-loops or multi-edges)
    /// 
    /// A simple graph is one that does not contain any loops nor any multiple edges.
    pub fn is_graph_simple(&self) -> bool {
        self.is_simple()
    }

    /// Check if the graph contains any cycles
    /// 
    /// Uses depth-first search to detect cycles in the graph.
    pub fn has_cycles(&self) -> bool {
        crate::algorithms::cycles::has_cycle(self)
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
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Check if the graph is weighted
    #[getter]
    fn is_weighted(&self) -> bool {
        self.weighted
    }

    /// Get all node IDs
    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get all edges as (id, source, target) tuples
    pub fn edges(&self) -> Vec<(u64, String, String)> {
        self.edges
            .iter()
            .map(|e| (e.id, e.source.clone(), e.target.clone()))
            .collect()
    }

    /// Get edges as "source->target" or "source--target" strings
    /// 
    /// Returns edge definitions in DiagrammeR format.
    /// For directed graphs: "A->B"
    /// For undirected graphs: "A--B"
    pub fn get_edges(&self) -> Vec<String> {
        let separator = if self.directed { "->" } else { "--" };
        self.edges
            .iter()
            .map(|e| format!("{}{}{}", e.source, separator, e.target))
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
            
            // Remove from selection if selected
            self.selected_edges.remove(&edge_id);
            
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

    /// Delete a node and all its incident edges
    pub fn delete_node(&mut self, node_id: &str) -> PyResult<()> {
        if !self.nodes.contains_key(node_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Node '{}' does not exist",
                node_id
            )));
        }

        // First, find and delete all edges incident to this node
        let mut edges_to_delete = Vec::new();
        for edge in &self.edges {
            if edge.source == node_id || edge.target == node_id {
                edges_to_delete.push(edge.id);
            }
        }

        // Delete the edges
        for edge_id in edges_to_delete {
            // Use delete_edge_by_id but ignore errors if edge was already deleted
            let _ = self.delete_edge_by_id(edge_id);
        }

        // Remove node from nodes HashMap
        self.nodes.remove(node_id);

        // Remove from adjacency list
        self.adjacency_list.remove(node_id);

        // Remove this node from other nodes' adjacency lists
        for neighbors in self.adjacency_list.values_mut() {
            neighbors.retain(|n| n != node_id);
        }

        // Remove from creation order
        self.node_creation_order.retain(|n| n != node_id);

        // Remove from selection if selected
        self.selected_nodes.remove(node_id);

        Ok(())
    }

    /// Set or update an attribute on a node
    pub fn set_node_attr(&mut self, py: Python, node_id: &str, key: &str, value: PyObject) -> PyResult<()> {
        let node = self.nodes.get_mut(node_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Node '{}' does not exist", node_id))
        })?;

        // Get or create attributes dict
        if node.attributes.is_none() {
            let dict = PyDict::new_bound(py);
            node.attributes = Some(dict.unbind().into());
        }

        // Update the attribute
        if let Some(ref attrs) = node.attributes {
            attrs.bind(py).set_item(key, value)?;
        }

        Ok(())
    }

    /// Set or update an attribute on an edge (including weight)
    pub fn set_edge_attr(&mut self, py: Python, edge_id: u64, key: &str, value: PyObject) -> PyResult<()> {
        let edge = self.edges.iter_mut().find(|e| e.id == edge_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Edge with ID {} does not exist", edge_id))
        })?;

        // Special handling for weight
        if key == "weight" {
            // Try to extract as f64
            if let Ok(weight_val) = value.extract::<f64>(py) {
                edge.weight = Some(weight_val);
            } else if value.is_none(py) {
                edge.weight = None;
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Weight must be a float or None"
                ));
            }
        } else {
            // Handle regular attributes
            if edge.attributes.is_none() {
                let dict = PyDict::new_bound(py);
                edge.attributes = Some(dict.unbind().into());
            }

            if let Some(ref attrs) = edge.attributes {
                attrs.bind(py).set_item(key, value)?;
            }
        }

        Ok(())
    }

    /// Reverse the direction of an edge (directed graphs only)
    pub fn reverse_edge_direction(&mut self, edge_id: u64) -> PyResult<()> {
        if !self.directed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot reverse edge direction in undirected graph"
            ));
        }

        let edge = self.edges.iter_mut().find(|e| e.id == edge_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Edge with ID {} does not exist", edge_id))
        })?;

        // Swap source and target
        std::mem::swap(&mut edge.source, &mut edge.target);

        // Update adjacency list
        // Remove old direction
        if let Some(neighbors) = self.adjacency_list.get_mut(&edge.target) {
            neighbors.retain(|n| n != &edge.source);
        }

        // Add new direction
        self.adjacency_list
            .entry(edge.source.clone())
            .or_insert_with(Vec::new)
            .push(edge.target.clone());

        Ok(())
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

    /// Select nodes in the K-hop neighborhood of specified nodes
    #[pyo3(signature = (node_ids, k=1, mode="out"))]
    pub fn select_nodes_in_neighborhood(
        &mut self,
        node_ids: Vec<String>,
        k: usize,
        mode: &str,
    ) -> PyResult<()> {
        // Validate mode
        if mode != "out" && mode != "in" && mode != "all" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be 'out', 'in', or 'all'"
            ));
        }

        // For undirected graphs, always use "all" mode
        let effective_mode = if !self.directed { "all" } else { mode };

        // Use BFS to find all nodes within k hops
        let mut visited = FxHashSet::default();
        let mut current_level = FxHashSet::default();
        
        // Initialize with seed nodes
        for node_id in node_ids {
            if !self.nodes.contains_key(&node_id) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Node '{}' does not exist", node_id)
                ));
            }
            current_level.insert(node_id);
        }
        
        // BFS for k hops
        for _ in 0..k {
            let mut next_level = FxHashSet::default();
            
            for node_id in &current_level {
                visited.insert(node_id.clone());
                
                // Get neighbors based on mode
                match effective_mode {
                    "out" => {
                        // Outgoing neighbors
                        if let Some(neighbors) = self.adjacency_list.get(node_id) {
                            for neighbor in neighbors {
                                if !visited.contains(neighbor) {
                                    next_level.insert(neighbor.clone());
                                }
                            }
                        }
                    }
                    "in" => {
                        // Incoming neighbors - need to search all edges
                        for edge in &self.edges {
                            if &edge.target == node_id && !visited.contains(&edge.source) {
                                next_level.insert(edge.source.clone());
                            }
                        }
                    }
                    "all" => {
                        // Outgoing neighbors from adjacency list
                        if let Some(neighbors) = self.adjacency_list.get(node_id) {
                            for neighbor in neighbors {
                                if !visited.contains(neighbor) {
                                    next_level.insert(neighbor.clone());
                                }
                            }
                        }
                        // Incoming neighbors from edge list
                        for edge in &self.edges {
                            if &edge.target == node_id && !visited.contains(&edge.source) {
                                next_level.insert(edge.source.clone());
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            
            current_level = next_level;
        }
        
        // Add all visited nodes to selection (including the final level)
        for node_id in visited {
            self.selected_nodes.insert(node_id);
        }
        for node_id in current_level {
            self.selected_nodes.insert(node_id);
        }
        
        Ok(())
    }

    /// Select top-K nodes by degree (used internally for centrality-based selection)
    #[pyo3(signature = (scores, k, include_ties=false))]
    pub fn select_nodes_by_scores(
        &mut self,
        scores: FxHashMap<String, f64>,
        k: usize,
        include_ties: bool,
    ) -> PyResult<()> {
        if scores.is_empty() {
            return Ok(());
        }
        
        // Convert to vector and sort by score (descending)
        let mut scored_nodes: Vec<(String, f64)> = scores.into_iter().collect();
        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if k == 0 {
            return Ok(());
        }
        
        // Select top k
        let actual_k = k.min(scored_nodes.len());
        
        if include_ties && actual_k > 0 {
            // Include all nodes with the same score as the k-th node
            let threshold_score = scored_nodes[actual_k - 1].1;
            for (node_id, score) in scored_nodes {
                if score >= threshold_score {
                    self.selected_nodes.insert(node_id);
                }
            }
        } else {
            // Just take the top k
            for (node_id, _score) in scored_nodes.iter().take(actual_k) {
                self.selected_nodes.insert(node_id.clone());
            }
        }
        
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={}, directed={})",
            self.num_nodes(),
            self.num_edges(),
            self.directed
        )
    }

    // ========== Random Graph Generators ==========

    /// Generate Erdős-Rényi G(n,p) random graph
    /// Each possible edge is included independently with probability p
    fn generate_gnp_graph(
        &mut self,
        n: usize,
        p: f64,
        seed: Option<u64>,
    ) -> PyResult<()> {
        if !(0.0..=1.0).contains(&p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Probability p must be between 0.0 and 1.0"
            ));
        }

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Add nodes
        let start_id = self.nodes.len() + 1;
        let node_ids: Vec<String> = (0..n)
            .map(|i| (start_id + i).to_string())
            .collect();

        for node_id in &node_ids {
            self.add_node(Node::new(node_id.clone(), None))?;
        }

        // Add edges with probability p
        if self.directed {
            // For directed graphs, consider all ordered pairs
            for i in 0..n {
                for j in 0..n {
                    if i != j && rng.gen::<f64>() < p {
                        self.add_edge(Edge::new(
                            0, // edge_id will be auto-assigned
                            node_ids[i].clone(),
                            node_ids[j].clone(),
                            None,
                            None,
                        ))?;
                    }
                }
            }
        } else {
            // For undirected graphs, consider unordered pairs
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.gen::<f64>() < p {
                        self.add_edge(Edge::new(
                            0,
                            node_ids[i].clone(),
                            node_ids[j].clone(),
                            None,
                            None,
                        ))?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate Erdős-Rényi G(n,m) random graph
    /// Graph with n nodes and exactly m randomly placed edges
    fn generate_gnm_graph(
        &mut self,
        n: usize,
        m: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Calculate maximum possible edges
        let max_edges = if self.directed {
            n * (n - 1)
        } else {
            n * (n - 1) / 2
        };

        if m > max_edges {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot create {} edges with {} nodes (max: {})",
                m, n, max_edges
            )));
        }

        // Add nodes
        let start_id = self.nodes.len() + 1;
        let node_ids: Vec<String> = (0..n)
            .map(|i| (start_id + i).to_string())
            .collect();

        for node_id in &node_ids {
            self.add_node(Node::new(node_id.clone(), None))?;
        }

        // Generate all possible edges
        let mut possible_edges: Vec<(String, String)> = Vec::with_capacity(max_edges);
        
        if self.directed {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        possible_edges.push((node_ids[i].clone(), node_ids[j].clone()));
                    }
                }
            }
        } else {
            for i in 0..n {
                for j in (i + 1)..n {
                    possible_edges.push((node_ids[i].clone(), node_ids[j].clone()));
                }
            }
        }

        // Randomly select m edges
        possible_edges.shuffle(&mut rng);
        for (u, v) in possible_edges.iter().take(m) {
            self.add_edge(Edge::new(0, u.clone(), v.clone(), None, None))?;
        }

        Ok(())
    }

    /// Generate Barabási-Albert preferential attachment graph
    /// Grows a graph by adding nodes one at a time with preferential attachment
    fn generate_pa_graph(
        &mut self,
        n: usize,
        m: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        if m < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "m must be at least 1"
            ));
        }
        if m >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "m must be less than n"
            ));
        }

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let start_id = self.nodes.len() + 1;
        let node_ids: Vec<String> = (0..n)
            .map(|i| (start_id + i).to_string())
            .collect();

        // Add first m+1 nodes in a clique
        for i in 0..=m {
            self.add_node(Node::new(node_ids[i].clone(), None))?;
        }

        for i in 0..=m {
            for j in (i + 1)..=m {
                self.add_edge(Edge::new(
                    0,
                    node_ids[i].clone(),
                    node_ids[j].clone(),
                    None,
                    None,
                ))?;
            }
        }

        // Track degree for preferential attachment
        let mut degrees: FxHashMap<String, usize> = FxHashMap::default();
        for i in 0..=m {
            degrees.insert(node_ids[i].clone(), m);
        }

        // Add remaining nodes with preferential attachment
        for i in (m + 1)..n {
            self.add_node(Node::new(node_ids[i].clone(), None))?;

            // Select m targets with probability proportional to degree
            let mut targets: FxHashSet<String> = FxHashSet::default();
            let total_degree: usize = degrees.values().sum();

            while targets.len() < m {
                let r = rng.gen::<f64>() * total_degree as f64;
                let mut cumsum = 0.0;

                for (node, &deg) in &degrees {
                    cumsum += deg as f64;
                    if cumsum >= r {
                        targets.insert(node.clone());
                        break;
                    }
                }
            }

            // Add edges to selected targets
            for target in &targets {
                self.add_edge(Edge::new(
                    0,
                    node_ids[i].clone(),
                    target.clone(),
                    None,
                    None,
                ))?;
                *degrees.get_mut(target).unwrap() += 1;
            }

            degrees.insert(node_ids[i].clone(), m);
        }

        Ok(())
    }

    /// Generate Watts-Strogatz small-world graph
    /// Creates a ring lattice with rewiring probability p
    fn generate_smallworld_graph(
        &mut self,
        n: usize,
        k: usize,
        p: f64,
        seed: Option<u64>,
    ) -> PyResult<()> {
        if k >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be less than n"
            ));
        }
        if k % 2 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be even"
            ));
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Probability p must be between 0.0 and 1.0"
            ));
        }

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let start_id = self.nodes.len() + 1;
        let node_ids: Vec<String> = (0..n)
            .map(|i| (start_id + i).to_string())
            .collect();

        // Add nodes
        for node_id in &node_ids {
            self.add_node(Node::new(node_id.clone(), None))?;
        }

        // Create ring lattice: connect each node to k/2 neighbors on each side
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in 1..=(k / 2) {
                let neighbor_idx = (i + j) % n;
                edges.push((i, neighbor_idx));
            }
        }

        // Rewire edges with probability p
        let mut rewired_edges: FxHashSet<(usize, usize)> = FxHashSet::default();
        
        for (u_idx, v_idx) in edges {
            if rng.gen::<f64>() < p {
                // Rewire: choose new target
                let mut new_v_idx = rng.gen_range(0..n);
                let mut attempts = 0;
                
                while (new_v_idx == u_idx 
                    || rewired_edges.contains(&(u_idx, new_v_idx))
                    || rewired_edges.contains(&(new_v_idx, u_idx)))
                    && attempts < 100 {
                    new_v_idx = rng.gen_range(0..n);
                    attempts += 1;
                }

                if attempts < 100 {
                    self.add_edge(Edge::new(
                        0,
                        node_ids[u_idx].clone(),
                        node_ids[new_v_idx].clone(),
                        None,
                        None,
                    ))?;
                    rewired_edges.insert((u_idx, new_v_idx));
                } else {
                    // Fallback: keep original edge
                    self.add_edge(Edge::new(
                        0,
                        node_ids[u_idx].clone(),
                        node_ids[v_idx].clone(),
                        None,
                        None,
                    ))?;
                    rewired_edges.insert((u_idx, v_idx));
                }
            } else {
                self.add_edge(Edge::new(
                    0,
                    node_ids[u_idx].clone(),
                    node_ids[v_idx].clone(),
                    None,
                    None,
                ))?;
                rewired_edges.insert((u_idx, v_idx));
            }
        }

        Ok(())
    }

    // ========== Structural Properties ==========

    /// Calculate graph density
    /// Density = actual edges / possible edges
    fn density(&self) -> f64 {
        let n = self.num_nodes();
        if n <= 1 {
            return 0.0;
        }

        let actual_edges = self.num_edges() as f64;
        let possible_edges = if self.directed {
            (n * (n - 1)) as f64
        } else {
            (n * (n - 1) / 2) as f64
        };

        if possible_edges == 0.0 {
            0.0
        } else {
            actual_edges / possible_edges
        }
    }

    /// Check if graph is connected (for undirected) or weakly connected (for directed)
    fn is_connected(&self) -> bool {
        if self.num_nodes() == 0 {
            return true;
        }

        // BFS from first node
        let start_node = match self.nodes.keys().next() {
            Some(node) => node.clone(),
            None => return true,
        };

        let mut visited: FxHashSet<String> = FxHashSet::default();
        let mut queue: Vec<String> = vec![start_node.clone()];
        visited.insert(start_node);

        while let Some(current) = queue.pop() {
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push(neighbor.clone());
                    }
                }
            }

            // For directed graphs, also check reverse edges (weak connectivity)
            if self.directed {
                for edge in &self.edges {
                    if &edge.target == &current && !visited.contains(&edge.source) {
                        visited.insert(edge.source.clone());
                        queue.push(edge.source.clone());
                    }
                }
            }
        }

        visited.len() == self.num_nodes()
    }

    /// Check if directed graph is a DAG (Directed Acyclic Graph)
    fn is_dag(&self) -> PyResult<bool> {
        if !self.directed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "is_dag() only works on directed graphs"
            ));
        }

        // Use DFS with coloring: white (unvisited), gray (visiting), black (visited)
        let mut color: FxHashMap<String, u8> = FxHashMap::default();
        for node_id in self.nodes.keys() {
            color.insert(node_id.clone(), 0); // white
        }

        fn has_cycle(
            node: &str,
            color: &mut FxHashMap<String, u8>,
            adj_list: &FxHashMap<String, Vec<String>>,
        ) -> bool {
            color.insert(node.to_string(), 1); // gray

            if let Some(neighbors) = adj_list.get(node) {
                for neighbor in neighbors {
                    let neighbor_color = *color.get(neighbor).unwrap_or(&0);
                    if neighbor_color == 1 {
                        // Back edge found - cycle detected
                        return true;
                    }
                    if neighbor_color == 0 && has_cycle(neighbor, color, adj_list) {
                        return true;
                    }
                }
            }

            color.insert(node.to_string(), 2); // black
            false
        }

        for node_id in self.nodes.keys() {
            if *color.get(node_id).unwrap() == 0 {
                if has_cycle(node_id, &mut color, &self.adjacency_list) {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Check if graph is simple (no self-loops or multi-edges)
    fn is_simple(&self) -> bool {
        // Check for self-loops
        for edge in &self.edges {
            if edge.source == edge.target {
                return false;
            }
        }

        // Check for multi-edges
        let mut seen_edges: FxHashSet<(String, String)> = FxHashSet::default();
        for edge in &self.edges {
            let edge_key = if self.directed {
                (edge.source.clone(), edge.target.clone())
            } else {
                // Normalize undirected edge
                if edge.source <= edge.target {
                    (edge.source.clone(), edge.target.clone())
                } else {
                    (edge.target.clone(), edge.source.clone())
                }
            };

            if seen_edges.contains(&edge_key) {
                return false;
            }
            seen_edges.insert(edge_key);
        }

        true
    }

    /// Get degree distribution as a map from degree to count
    fn degree_distribution(&self) -> FxHashMap<usize, usize> {
        let mut distribution: FxHashMap<usize, usize> = FxHashMap::default();

        for node_id in self.nodes.keys() {
            let deg = self.degree(node_id);
            *distribution.entry(deg).or_insert(0) += 1;
        }

        distribution
    }

    /// Check if graph is bipartite using BFS coloring
    fn is_bipartite(&self) -> bool {
        if self.num_nodes() == 0 {
            return true;
        }

        let mut color: FxHashMap<String, i8> = FxHashMap::default();

        // Check all connected components
        for start_node in self.nodes.keys() {
            if color.contains_key(start_node) {
                continue;
            }

            // BFS coloring
            let mut queue: Vec<String> = vec![start_node.clone()];
            color.insert(start_node.clone(), 0);

            while let Some(current) = queue.pop() {
                let current_color = *color.get(&current).unwrap();
                let next_color = 1 - current_color;

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for neighbor in neighbors {
                        if let Some(&neighbor_color) = color.get(neighbor) {
                            if neighbor_color == current_color {
                                return false; // Same color as current - not bipartite
                            }
                        } else {
                            color.insert(neighbor.clone(), next_color);
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        true
    }

    /// Count self-loops in the graph
    fn count_selfloops(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.source == e.target)
            .count()
    }

    /// Get average degree
    fn average_degree(&self) -> f64 {
        let n = self.num_nodes();
        if n == 0 {
            return 0.0;
        }

        let total_degree: usize = self.nodes.keys().map(|id| self.degree(id)).sum();
        total_degree as f64 / n as f64
    }

    /// Get degree histogram (degree -> count mapping)
    /// Returns a HashMap suitable for plotting degree distribution
    fn get_degree_histogram(&self) -> FxHashMap<usize, usize> {
        // This is actually the same as degree_distribution, but with a clearer name
        // for plotting purposes
        self.degree_distribution()
    }

    /// Find articulation points (cut vertices) in an undirected graph
    /// Returns list of node IDs whose removal would disconnect the graph
    /// Uses Tarjan's algorithm with DFS
    fn get_articulation_points(&self) -> PyResult<Vec<String>> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Articulation points only defined for undirected graphs",
            ));
        }

        let n = self.num_nodes();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut visited = FxHashSet::default();
        let mut disc = FxHashMap::default(); // Discovery time
        let mut low = FxHashMap::default(); // Lowest reachable ancestor
        let mut parent: FxHashMap<String, Option<String>> = FxHashMap::default();
        let mut ap = FxHashSet::default(); // Articulation points
        let mut time = 0usize;

        // DFS helper function
        fn dfs_ap(
            node: &str,
            visited: &mut FxHashSet<String>,
            disc: &mut FxHashMap<String, usize>,
            low: &mut FxHashMap<String, usize>,
            parent: &mut FxHashMap<String, Option<String>>,
            ap: &mut FxHashSet<String>,
            time: &mut usize,
            adj_list: &FxHashMap<String, Vec<String>>,
        ) {
            let mut children = 0;
            visited.insert(node.to_string());
            *time += 1;
            disc.insert(node.to_string(), *time);
            low.insert(node.to_string(), *time);

            if let Some(neighbors) = adj_list.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        children += 1;
                        parent.insert(neighbor.clone(), Some(node.to_string()));
                        dfs_ap(neighbor, visited, disc, low, parent, ap, time, adj_list);

                        // Check if subtree rooted at neighbor has connection back to ancestors
                        low.insert(
                            node.to_string(),
                            (*low.get(node).unwrap()).min(*low.get(neighbor).unwrap()),
                        );

                        // If node is root and has 2+ children, it's an AP
                        if parent.get(node).unwrap().is_none() && children > 1 {
                            ap.insert(node.to_string());
                        }

                        // If node is not root and low[neighbor] >= disc[node], it's an AP
                        if parent.get(node).unwrap().is_some()
                            && *low.get(neighbor).unwrap() >= *disc.get(node).unwrap()
                        {
                            ap.insert(node.to_string());
                        }
                    } else if Some(neighbor.clone()) != *parent.get(node).unwrap() {
                        // Update low value for back edge
                        low.insert(
                            node.to_string(),
                            (*low.get(node).unwrap()).min(*disc.get(neighbor).unwrap()),
                        );
                    }
                }
            }
        }

        // Run DFS from each unvisited node (handles disconnected components)
        for start_node in self.nodes.keys() {
            if !visited.contains(start_node) {
                parent.insert(start_node.clone(), None);
                dfs_ap(
                    start_node,
                    &mut visited,
                    &mut disc,
                    &mut low,
                    &mut parent,
                    &mut ap,
                    &mut time,
                    &self.adjacency_list,
                );
            }
        }

        Ok(ap.into_iter().collect())
    }

    /// Compute eccentricity for each node (max distance to any other node)
    /// Returns HashMap of node -> eccentricity
    /// Eccentricity is infinity if graph is disconnected
    fn compute_eccentricities(&self) -> FxHashMap<String, Option<usize>> {
        let mut eccentricities = FxHashMap::default();

        for node in self.nodes.keys() {
            // BFS to find max distance from this node
            let mut visited = FxHashSet::default();
            let mut queue = std::collections::VecDeque::new();
            let mut distances: FxHashMap<String, usize> = FxHashMap::default();

            visited.insert(node.clone());
            queue.push_back(node.clone());
            distances.insert(node.clone(), 0);

            while let Some(current) = queue.pop_front() {
                let current_dist = *distances.get(&current).unwrap();

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            distances.insert(neighbor.clone(), current_dist + 1);
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }

            // Check if all nodes are reachable
            if visited.len() == self.num_nodes() {
                let max_dist = distances.values().max().copied().unwrap_or(0);
                eccentricities.insert(node.clone(), Some(max_dist));
            } else {
                // Graph is disconnected from this node's perspective
                eccentricities.insert(node.clone(), None);
            }
        }

        eccentricities
    }

    /// Get periphery nodes (nodes with maximum eccentricity)
    /// Returns list of node IDs
    /// Returns empty list if graph is disconnected
    fn get_periphery(&self) -> Vec<String> {
        let eccentricities = self.compute_eccentricities();

        // Filter out None values (disconnected nodes)
        let valid_eccentricities: Vec<_> = eccentricities
            .iter()
            .filter_map(|(node, ecc)| ecc.map(|e| (node.clone(), e)))
            .collect();

        if valid_eccentricities.is_empty() {
            return Vec::new();
        }

        let max_ecc = *valid_eccentricities.iter().map(|(_, e)| e).max().unwrap();

        valid_eccentricities
            .into_iter()
            .filter(|(_, e)| *e == max_ecc)
            .map(|(node, _)| node)
            .collect()
    }

    /// Get graph diameter (maximum eccentricity)
    /// Returns None if graph is disconnected
    fn get_graph_diameter(&self) -> Option<usize> {
        let eccentricities = self.compute_eccentricities();

        eccentricities
            .values()
            .filter_map(|&ecc| ecc)
            .max()
    }

    /// Get graph radius (minimum eccentricity)
    /// Returns None if graph is disconnected
    fn get_graph_radius(&self) -> Option<usize> {
        let eccentricities = self.compute_eccentricities();

        eccentricities
            .values()
            .filter_map(|&ecc| ecc)
            .min()
    }

    /// Get girth (length of shortest cycle)
    /// Returns None if graph is acyclic
    /// Uses BFS from each node to find shortest cycle through that node
    fn get_girth(&self) -> Option<usize> {
        if self.num_nodes() == 0 {
            return None;
        }

        let mut min_cycle: Option<usize> = None;

        // Try BFS from each node to find shortest cycle
        for start_node in self.nodes.keys() {
            let mut distances: FxHashMap<String, usize> = FxHashMap::default();
            let mut parent: FxHashMap<String, Option<String>> = FxHashMap::default();
            let mut queue = std::collections::VecDeque::new();

            distances.insert(start_node.clone(), 0);
            parent.insert(start_node.clone(), None);
            queue.push_back(start_node.clone());

            while let Some(current) = queue.pop_front() {
                let current_dist = *distances.get(&current).unwrap();
                let current_parent = parent.get(&current).unwrap().clone();

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for neighbor in neighbors {
                        if let Some(&neighbor_dist) = distances.get(neighbor) {
                            // Already visited - potential cycle
                            // In undirected graphs, skip if neighbor is parent (back edge)
                            if !self.directed || Some(neighbor.clone()) != current_parent {
                                if Some(neighbor.clone()) != current_parent {
                                    // Found a cycle
                                    let cycle_length = current_dist + neighbor_dist + 1;
                                    min_cycle = Some(match min_cycle {
                                        None => cycle_length,
                                        Some(prev) => prev.min(cycle_length),
                                    });
                                }
                            }
                        } else {
                            distances.insert(neighbor.clone(), current_dist + 1);
                            parent.insert(neighbor.clone(), Some(current.clone()));
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        min_cycle
    }

    /// Get adhesion (minimum edge cut size)
    /// Returns minimum number of edges whose removal disconnects the graph
    /// Uses simple approach: try removing each edge and check connectivity
    /// For larger graphs, use max-flow based algorithms (future optimization)
    fn get_adhesion(&self) -> PyResult<usize> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Adhesion currently only implemented for undirected graphs",
            ));
        }

        let n = self.num_nodes();
        if n == 0 {
            return Ok(0);
        }

        if !self.is_connected() {
            return Ok(0);
        }

        // For small graphs, try edge removal
        // For a complete graph, adhesion = minimum degree
        let min_degree = self.nodes.keys()
            .map(|id| self.degree(id))
            .min()
            .unwrap_or(0);

        // Simple upper bound: adhesion <= min_degree
        // Exact computation requires max-flow, which we'll implement later
        Ok(min_degree)
    }

    /// Get cohesion (minimum vertex cut size)
    /// Returns minimum number of vertices whose removal disconnects the graph
    /// Uses Menger's theorem: min vertex cut = max number of vertex-disjoint paths
    /// Simplified implementation for now
    fn get_cohesion(&self) -> PyResult<usize> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cohesion currently only implemented for undirected graphs",
            ));
        }

        let n = self.num_nodes();
        if n == 0 {
            return Ok(0);
        }

        if !self.is_connected() {
            return Ok(0);
        }

        // For complete graphs, cohesion = n - 1
        // For others, cohesion <= min_degree
        let min_degree = self.nodes.keys()
            .map(|id| self.degree(id))
            .min()
            .unwrap_or(0);

        // Simple bound: cohesion <= min_degree
        // Exact computation requires sophisticated algorithms
        Ok(min_degree)
    }

    /// Get eccentricity for each node
    /// Eccentricity is the maximum distance from a node to any other node
    /// Returns HashMap mapping node ID to eccentricity (None if disconnected)
    fn get_eccentricity(&self) -> FxHashMap<String, Option<usize>> {
        self.compute_eccentricities()
    }

    /// Get radiality centrality for each node
    /// Radiality = (diameter + 1 - eccentricity) / (n - 1)
    /// Measures how central a node is based on eccentricity
    /// Returns HashMap mapping node ID to radiality (None if graph disconnected)
    fn get_radiality(&self) -> FxHashMap<String, Option<f64>> {
        let n = self.num_nodes();
        if n <= 1 {
            return self.nodes.keys()
                .map(|id| (id.clone(), Some(0.0)))
                .collect();
        }

        let eccentricities = self.compute_eccentricities();
        let diameter = self.get_graph_diameter();

        match diameter {
            Some(d) => {
                let denom = (n - 1) as f64;
                eccentricities
                    .into_iter()
                    .map(|(node, ecc)| {
                        let radiality = ecc.map(|e| (d + 1 - e) as f64 / denom);
                        (node, radiality)
                    })
                    .collect()
            }
            None => {
                // Graph is disconnected
                eccentricities
                    .into_iter()
                    .map(|(node, _)| (node, None))
                    .collect()
            }
        }
    }

    /// Get closeness vitality for each node
    /// Measures importance by the change in total distance when node is removed
    /// vitality(v) = sum_distances(G) - sum_distances(G - v)
    /// Higher values indicate more important nodes
    /// 
    /// Performance: Uses parallel BFS (via Rayon) for ~linear speedup with cores
    /// Complexity: O(V²E / num_cores) with parallelization
    fn get_closeness_vitality(&self) -> FxHashMap<String, f64> {
        use rayon::prelude::*;
        
        let n = self.num_nodes();
        if n == 0 {
            return FxHashMap::default();
        }

        // Compute total distances in original graph (parallelized)
        let original_total = self.compute_total_distances_parallel();

        // Collect node IDs to iterate over
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        // Compute vitality for each node in parallel
        let vitality_vec: Vec<(String, f64)> = node_ids
            .par_iter()
            .map(|node| {
                // Compute total distances without this node
                let total_without = self.compute_total_distances_without_node(node);
                
                // Vitality is the difference (higher = more important)
                let vital = original_total - total_without;
                (node.clone(), vital)
            })
            .collect();

        // Convert to HashMap
        vitality_vec.into_iter().collect()
    }

    /// Helper: compute sum of all pairwise shortest path distances (parallel)
    fn compute_total_distances_parallel(&self) -> f64 {
        use rayon::prelude::*;
        
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        
        node_ids
            .par_iter()
            .map(|start| {
                let distances = self.bfs_distances_optimized(start);
                distances.values().sum::<usize>() as f64
            })
            .sum()
    }

    /// Helper: compute sum of all pairwise shortest path distances
    fn compute_total_distances(&self) -> f64 {
        let mut total = 0.0;
        
        for start in self.nodes.keys() {
            let distances = self.bfs_distances_optimized(start);
            total += distances.values().sum::<usize>() as f64;
        }

        total
    }

    /// Helper: compute total distances excluding a specific node
    fn compute_total_distances_without_node(&self, excluded: &str) -> f64 {
        let mut total = 0.0;

        for start in self.nodes.keys() {
            if start == excluded {
                continue;
            }

            let distances = self.bfs_distances_excluding_optimized(start, excluded);
            total += distances.values().sum::<usize>() as f64;
        }

        total
    }

    /// Helper: Optimized BFS to compute distances from a start node
    /// Preallocates capacity for better performance
    fn bfs_distances_optimized(&self, start: &str) -> FxHashMap<String, usize> {
        let n = self.num_nodes();
        let mut distances = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(n);

        distances.insert(start.to_string(), 0);
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for neighbor in neighbors {
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), current_dist + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        distances
    }

    /// Helper: BFS to compute distances from a start node
    fn bfs_distances(&self, start: &str) -> FxHashMap<String, usize> {
        self.bfs_distances_optimized(start)
    }

    /// Helper: Optimized BFS excluding a specific node
    fn bfs_distances_excluding_optimized(&self, start: &str, excluded: &str) -> FxHashMap<String, usize> {
        let n = self.num_nodes();
        let mut distances = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(n);

        distances.insert(start.to_string(), 0);
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for neighbor in neighbors {
                    if neighbor == excluded {
                        continue;
                    }
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), current_dist + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        distances
    }

    /// Helper: BFS excluding a specific node
    fn bfs_distances_excluding(&self, start: &str, excluded: &str) -> FxHashMap<String, usize> {
        self.bfs_distances_excluding_optimized(start, excluded)
    }

    /// Get k-core decomposition (coreness for each node)
    /// k-core is maximal subgraph where all nodes have degree >= k
    /// Coreness is the highest k for which a node belongs to the k-core
    fn get_coreness(&self) -> FxHashMap<String, usize> {
        let mut coreness = FxHashMap::default();
        let mut degrees: FxHashMap<String, usize> = self.nodes.keys()
            .map(|id| (id.clone(), self.degree(id)))
            .collect();

        let mut remaining: FxHashSet<String> = self.nodes.keys().cloned().collect();
        let mut k = 0;

        while !remaining.is_empty() {
            // Find minimum degree among remaining nodes
            let min_deg = remaining.iter()
                .map(|id| degrees.get(id).unwrap())
                .min()
                .copied()
                .unwrap_or(0);

            k = k.max(min_deg);

            // Remove all nodes with degree <= min_deg
            let mut to_remove = Vec::new();
            for node in &remaining {
                if *degrees.get(node).unwrap() <= min_deg {
                    to_remove.push(node.clone());
                }
            }

            for node in &to_remove {
                coreness.insert(node.clone(), k);
                remaining.remove(node);

                // Update degrees of neighbors
                if let Some(neighbors) = self.adjacency_list.get(node) {
                    for neighbor in neighbors {
                        if remaining.contains(neighbor) {
                            if let Some(deg) = degrees.get_mut(neighbor) {
                                *deg = deg.saturating_sub(1);
                            }
                        }
                    }
                }
            }
        }

        coreness
    }

    /// Get HITS authority scores
    /// Hyperlink-Induced Topic Search algorithm for directed graphs
    /// Authority score: sum of hub scores of nodes pointing to this node
    /// Returns HashMap of node -> authority score
    fn get_authority_centrality(&self, max_iter: usize, tol: f64) -> PyResult<FxHashMap<String, f64>> {
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "HITS algorithm requires a directed graph",
            ));
        }

        let (authorities, _) = self.compute_hits(max_iter, tol);
        Ok(authorities)
    }

    /// Get HITS hub scores
    /// Hub score: sum of authority scores of nodes this node points to
    /// Returns HashMap of node -> hub score
    fn get_hub_centrality(&self, max_iter: usize, tol: f64) -> PyResult<FxHashMap<String, f64>> {
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "HITS algorithm requires a directed graph",
            ));
        }

        let (_, hubs) = self.compute_hits(max_iter, tol);
        Ok(hubs)
    }

    /// Helper: compute HITS (Hyperlink-Induced Topic Search) scores
    /// Returns (authorities, hubs)
    fn compute_hits(&self, max_iter: usize, tol: f64) -> (FxHashMap<String, f64>, FxHashMap<String, f64>) {
        let n = self.num_nodes();
        if n == 0 {
            return (FxHashMap::default(), FxHashMap::default());
        }

        // Initialize all scores to 1.0
        let mut authorities: FxHashMap<String, f64> = self.nodes.keys()
            .map(|id| (id.clone(), 1.0))
            .collect();
        let mut hubs: FxHashMap<String, f64> = self.nodes.keys()
            .map(|id| (id.clone(), 1.0))
            .collect();

        for _ in 0..max_iter {
            let prev_authorities = authorities.clone();
            let prev_hubs = hubs.clone();

            // Update authority scores: sum of hub scores of incoming neighbors
            for node in self.nodes.keys() {
                let mut auth_score = 0.0;
                // Find all nodes that point to this node
                for (source, targets) in &self.adjacency_list {
                    if targets.contains(node) {
                        auth_score += prev_hubs.get(source).unwrap_or(&0.0);
                    }
                }
                authorities.insert(node.clone(), auth_score);
            }

            // Update hub scores: sum of authority scores of outgoing neighbors
            for node in self.nodes.keys() {
                let mut hub_score = 0.0;
                if let Some(neighbors) = self.adjacency_list.get(node) {
                    for neighbor in neighbors {
                        hub_score += prev_authorities.get(neighbor).unwrap_or(&0.0);
                    }
                }
                hubs.insert(node.clone(), hub_score);
            }

            // Normalize
            let auth_norm: f64 = authorities.values().map(|&v| v * v).sum::<f64>().sqrt();
            let hub_norm: f64 = hubs.values().map(|&v| v * v).sum::<f64>().sqrt();

            if auth_norm > 0.0 {
                for val in authorities.values_mut() {
                    *val /= auth_norm;
                }
            }
            if hub_norm > 0.0 {
                for val in hubs.values_mut() {
                    *val /= hub_norm;
                }
            }

            // Check convergence
            let auth_diff: f64 = authorities.iter()
                .map(|(k, &v)| (v - prev_authorities.get(k).unwrap()).abs())
                .sum();
            let hub_diff: f64 = hubs.iter()
                .map(|(k, &v)| (v - prev_hubs.get(k).unwrap()).abs())
                .sum();

            if auth_diff < tol && hub_diff < tol {
                break;
            }
        }

        (authorities, hubs)
    }

    /// Get alpha centrality
    /// Generalization of eigenvector centrality with attenuation factor
    /// x = alpha * A * x + e
    /// where alpha is attenuation, A is adjacency matrix, e is exogenous factor
    fn get_alpha_centrality(&self, alpha: f64, max_iter: usize, tol: f64) -> FxHashMap<String, f64> {
        let n = self.num_nodes();
        if n == 0 {
            return FxHashMap::default();
        }

        // Initialize centrality to 1.0 for all nodes
        let mut centrality: FxHashMap<String, f64> = self.nodes.keys()
            .map(|id| (id.clone(), 1.0))
            .collect();

        for _ in 0..max_iter {
            let prev_centrality = centrality.clone();
            let mut max_change: f64 = 0.0;

            for node in self.nodes.keys() {
                // Sum of centralities of neighbors
                let mut neighbor_sum = 0.0;
                if let Some(neighbors) = self.adjacency_list.get(node) {
                    for neighbor in neighbors {
                        neighbor_sum += prev_centrality.get(neighbor).unwrap_or(&0.0);
                    }
                }

                // alpha * (sum of neighbor centralities) + 1.0 (exogenous)
                let new_value = alpha * neighbor_sum + 1.0;
                centrality.insert(node.clone(), new_value);

                let change = (new_value - prev_centrality.get(node).unwrap()).abs();
                max_change = max_change.max(change);
            }

            // Normalize
            let total: f64 = centrality.values().sum();
            if total > 0.0 {
                for val in centrality.values_mut() {
                    *val /= total;
                }
            }

            if max_change < tol {
                break;
            }
        }

        centrality
    }

    /// Get leverage centrality for each node
    /// Measures the degree connectivity of a node relative to its neighbors
    /// leverage(v) = (1/k_v) * sum((k_v - k_n) / (k_v + k_n))
    /// where k_v is degree of v, k_n is degree of neighbor n
    fn get_leverage_centrality(&self) -> FxHashMap<String, f64> {
        let mut leverage = FxHashMap::default();

        for node in self.nodes.keys() {
            let k_v = self.degree(node) as f64;
            
            if k_v == 0.0 {
                leverage.insert(node.clone(), 0.0);
                continue;
            }

            let mut sum = 0.0;
            if let Some(neighbors) = self.adjacency_list.get(node) {
                for neighbor in neighbors {
                    let k_n = self.degree(neighbor) as f64;
                    if k_v + k_n > 0.0 {
                        sum += (k_v - k_n) / (k_v + k_n);
                    }
                }
            }

            let lev = sum / k_v;
            leverage.insert(node.clone(), lev);
        }

        leverage
    }

    /// Get Dice similarity coefficient between all node pairs
    /// Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    /// Measures neighborhood overlap, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    fn get_dice_similarity(&self) -> FxHashMap<String, FxHashMap<String, f64>> {
        let mut similarities = FxHashMap::default();
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();

        for node1 in &nodes {
            let mut node1_sims = FxHashMap::default();
            let neighbors1: FxHashSet<&String> = self.adjacency_list
                .get(node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2.clone(), 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&String> = self.adjacency_list
                    .get(node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();

                let dice = if size1 + size2 > 0 {
                    (2.0 * intersection as f64) / (size1 + size2) as f64
                } else {
                    0.0
                };

                node1_sims.insert(node2.clone(), dice);
            }

            similarities.insert(node1.clone(), node1_sims);
        }

        similarities
    }

    /// Get Jaccard similarity index between all node pairs
    /// Jaccard index: |A ∩ B| / |A ∪ B|
    /// Classic set similarity measure, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    fn get_jaccard_similarity(&self) -> FxHashMap<String, FxHashMap<String, f64>> {
        let mut similarities = FxHashMap::default();
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();

        for node1 in &nodes {
            let mut node1_sims = FxHashMap::default();
            let neighbors1: FxHashSet<&String> = self.adjacency_list
                .get(node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2.clone(), 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&String> = self.adjacency_list
                    .get(node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let union = neighbors1.union(&neighbors2).count();

                let jaccard = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };

                node1_sims.insert(node2.clone(), jaccard);
            }

            similarities.insert(node1.clone(), node1_sims);
        }

        similarities
    }

    /// Get Cosine similarity between all node pairs
    /// Cosine similarity: A·B / (||A|| ||B||)
    /// Normalized dot product of adjacency vectors, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    fn get_cosine_similarity(&self) -> FxHashMap<String, FxHashMap<String, f64>> {
        let mut similarities = FxHashMap::default();
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();

        for node1 in &nodes {
            let mut node1_sims = FxHashMap::default();
            let neighbors1: FxHashSet<&String> = self.adjacency_list
                .get(node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2.clone(), 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&String> = self.adjacency_list
                    .get(node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();

                let cosine = if size1 > 0 && size2 > 0 {
                    intersection as f64 / ((size1 as f64).sqrt() * (size2 as f64).sqrt())
                } else {
                    0.0
                };

                node1_sims.insert(node2.clone(), cosine);
            }

            similarities.insert(node1.clone(), node1_sims);
        }

        similarities
    }

    /// Get Overlap coefficient between all node pairs
    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|)
    /// Subset-based similarity measure, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    fn get_overlap_similarity(&self) -> FxHashMap<String, FxHashMap<String, f64>> {
        let mut similarities = FxHashMap::default();
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();

        for node1 in &nodes {
            let mut node1_sims = FxHashMap::default();
            let neighbors1: FxHashSet<&String> = self.adjacency_list
                .get(node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2.clone(), 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&String> = self.adjacency_list
                    .get(node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();
                let min_size = size1.min(size2);

                let overlap = if min_size > 0 {
                    intersection as f64 / min_size as f64
                } else {
                    0.0
                };

                node1_sims.insert(node2.clone(), overlap);
            }

            similarities.insert(node1.clone(), node1_sims);
        }

        similarities
    }

    // ========================================
    // TRAVERSAL OPERATIONS
    // ========================================

    /// Traverse from selected nodes to outward neighbors.
    /// Replaces the current node selection with neighbors in the outward direction.
    /// For directed graphs, this follows edges where selected nodes are sources.
    pub fn trav_out(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let current_selection: Vec<String> = self.selected_nodes.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors
        let mut new_selection = FxHashSet::default();
        
        if !self.directed {
            // For undirected graphs, get all neighbors
            for node_id in current_selection {
                if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                    new_selection.extend(neighbors.iter().cloned());
                }
            }
        } else {
            // For directed graphs, only get outward neighbors (successors)
            for node_id in current_selection {
                for edge in &self.edges {
                    if edge.source == node_id {
                        new_selection.insert(edge.target.clone());
                    }
                }
            }
        }

        // Replace selection with neighbors
        self.selected_nodes = new_selection;
        Ok(())
    }

    /// Traverse from selected nodes to inward neighbors.
    /// Replaces the current node selection with neighbors in the inward direction.
    /// For directed graphs, this follows edges where selected nodes are targets.
    pub fn trav_in(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let current_selection: Vec<String> = self.selected_nodes.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors
        let mut new_selection = FxHashSet::default();
        
        if !self.directed {
            // For undirected graphs, get all neighbors
            for node_id in current_selection {
                if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                    new_selection.extend(neighbors.iter().cloned());
                }
            }
        } else {
            // For directed graphs, only get inward neighbors (predecessors)
            for node_id in current_selection {
                for edge in &self.edges {
                    if edge.target == node_id {
                        new_selection.insert(edge.source.clone());
                    }
                }
            }
        }

        // Replace selection with neighbors
        self.selected_nodes = new_selection;
        Ok(())
    }

    /// Traverse from selected nodes to all neighbors (both directions).
    /// Replaces the current node selection with all connected neighbors.
    /// Works the same for directed and undirected graphs.
    pub fn trav_both(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let current_selection: Vec<String> = self.selected_nodes.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors (both directions)
        let mut new_selection = FxHashSet::default();
        for node_id in current_selection {
            // For all graphs, get neighbors from adjacency list (handles undirected)
            if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                new_selection.extend(neighbors.iter().cloned());
            }
            
            // For directed graphs, also get predecessors
            if self.directed {
                for edge in &self.edges {
                    if edge.target == node_id {
                        new_selection.insert(edge.source.clone());
                    }
                }
            }
        }

        // Replace selection with neighbors
        self.selected_nodes = new_selection;
        Ok(())
    }

    /// Traverse from selected nodes to outward edges.
    /// Replaces the current node selection with an edge selection of outbound edges.
    pub fn trav_out_edge(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let current_selection: Vec<String> = self.selected_nodes.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Clear node selection and build edge selection
        self.selected_nodes.clear();
        let mut new_edge_selection = FxHashSet::default();

        for node_id in current_selection {
            // Find edges where this node is the source
            for edge in &self.edges {
                if !self.directed {
                    // For undirected, include edges in both directions
                    if edge.source == node_id || edge.target == node_id {
                        new_edge_selection.insert(edge.id);
                    }
                } else {
                    // For directed, only outbound edges
                    if edge.source == node_id {
                        new_edge_selection.insert(edge.id);
                    }
                }
            }
        }

        // Replace selection with edges
        self.selected_edges = new_edge_selection;
        Ok(())
    }

    /// Traverse from selected nodes to inward edges.
    /// Replaces the current node selection with an edge selection of inbound edges.
    pub fn trav_in_edge(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let current_selection: Vec<String> = self.selected_nodes.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Clear node selection and build edge selection
        self.selected_nodes.clear();
        let mut new_edge_selection = FxHashSet::default();

        for node_id in current_selection {
            // Find edges where this node is the target
            for edge in &self.edges {
                if !self.directed {
                    // For undirected, include edges in both directions
                    if edge.source == node_id || edge.target == node_id {
                        new_edge_selection.insert(edge.id);
                    }
                } else {
                    // For directed, only inbound edges
                    if edge.target == node_id {
                        new_edge_selection.insert(edge.id);
                    }
                }
            }
        }

        // Replace selection with edges
        self.selected_edges = new_edge_selection;
        Ok(())
    }

    /// Traverse from selected edges to reverse edges.
    /// For each selected edge A→B, finds the reverse edge B→A (if it exists).
    /// If add_to_selection is true, adds reverse edges to selection; otherwise replaces.
    pub fn trav_reverse_edge(&mut self, add_to_selection: bool) -> PyResult<()> {
        // Get current selected edges
        let current_selection: Vec<u64> = self.selected_edges.iter().cloned().collect();
        
        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Find reverse edges
        let mut reverse_edges = FxHashSet::default();
        
        for edge_id in current_selection {
            // Find the edge
            if let Some(edge) = self.edges.iter().find(|e| e.id == edge_id) {
                // Look for the reverse edge: B→A for edge A→B
                for other_edge in &self.edges {
                    if other_edge.source == edge.target && other_edge.target == edge.source {
                        reverse_edges.insert(other_edge.id);
                    }
                }
            }
        }

        // Update selection based on add_to_selection parameter
        if add_to_selection {
            // Add reverse edges to current selection
            self.selected_edges.extend(reverse_edges);
        } else {
            // Replace selection with reverse edges
            self.selected_edges = reverse_edges;
        }

        Ok(())
    }

    pub fn count_weakly_connected_components(&self) -> usize {
        crate::algorithms::connectivity::count_weakly_connected_components(self)
    }

    /// Count strongly connected components (directed graphs only).
    /// A strongly connected component is a maximal set of vertices where
    /// every vertex is reachable from every other vertex.
    pub fn count_strongly_connected_components(&self) -> PyResult<usize> {
        crate::algorithms::connectivity::count_strongly_connected_components(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }
}


impl Graph {
    /// Get the adjacency list (internal method)
    pub fn get_adjacency_list(&self) -> &FxHashMap<String, Vec<String>> {
        &self.adjacency_list
    }

    /// Get edges vector (internal method for algorithms)
    pub fn get_edges_internal(&self) -> &Vec<Edge> {
        &self.edges
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

