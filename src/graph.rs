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
