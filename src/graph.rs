//! Graph data structure and core operations

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use nalgebra as na;
use crate::algorithms::sparse_eigen::CsrMatrix;
use std::collections::VecDeque;

/// (blocks, cohesion of each block, parent index of each block) from cohesive blocking.
type CohesiveBlocks = (Vec<Vec<u64>>, Vec<usize>, Vec<Option<usize>>);

/// A node in the graph
#[pyclass]
#[derive(Debug)]
pub struct Node {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get, set)]
    pub label: Option<String>,
    #[pyo3(get, set)]
    pub attributes: Option<Py<PyDict>>,
}

/// Shallow copy of an attributes dict, so a cloned node or edge (a derived graph, or the object
/// `get_node()`/`get_edge()` returns) never shares its dict with the original
pub(crate) fn copy_attrs(py: Python, attrs: &Option<Py<PyDict>>) -> Option<Py<PyDict>> {
    attrs.as_ref().map(|d| {
        d.bind(py).copy().map(|c| c.unbind()).unwrap_or_else(|_| d.clone_ref(py))
    })
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Node {
            id: self.id,
            label: self.label.clone(),
            attributes: copy_attrs(py, &self.attributes),
        })
    }
}

#[pymethods]
impl Node {
    #[new]
    #[pyo3(signature = (id, label=None, attributes=None))]
    fn new(id: u64, label: Option<String>, attributes: Option<Py<PyDict>>) -> Self {
        Node { id, label, attributes }
    }

    fn __repr__(&self) -> String {
        if let Some(ref label) = self.label {
            format!("Node(id={}, label='{}')", self.id, label)
        } else {
            format!("Node(id={})", self.id)
        }
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
    #[pyo3(get)]
    pub source: u64,
    #[pyo3(get)]
    pub target: u64,
    #[pyo3(get, set)]
    pub weight: Option<f64>,
    #[pyo3(get, set)]
    pub attributes: Option<Py<PyDict>>,
}

impl Clone for Edge {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Edge {
            id: self.id,
            source: self.source,
            target: self.target,
            weight: self.weight,
            attributes: copy_attrs(py, &self.attributes),
        })
    }
}

#[pymethods]
impl Edge {
    #[new]
    #[pyo3(signature = (id, source, target, weight=None, attributes=None))]
    fn new(
        id: u64,
        source: u64,
        target: u64,
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
            "Edge(id={}, {} -> {})",
            self.id, self.source, self.target
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Union-Find data structure with path-halving and union-by-rank
/// Used for efficient cycle detection in Kruskal's MST algorithm
struct UnionFind {
    parent: FxHashMap<u64, u64>,
    rank: FxHashMap<u64, usize>,
}

impl UnionFind {
    #[allow(dead_code)]
    #[inline]
    fn new() -> Self {
        UnionFind {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        UnionFind {
            parent: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            rank: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Find the root of element x with path-halving
    #[inline]
    fn find(&mut self, mut x: u64) -> u64 {
        // Initialize if not seen
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(x) {
            e.insert(x);
            self.rank.insert(x, 0);
            return x;
        }

        // Path-halving: make every node point to its grandparent
        while self.parent[&x] != x {
            let parent = self.parent[&x];
            let grandparent = self.parent[&parent];
            self.parent.insert(x, grandparent);
            x = parent;
        }
        x
    }

    /// Union two sets. Returns true if they were merged (not already connected)
    #[inline]
    fn union(&mut self, a: u64, b: u64) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);

        if ra == rb {
            return false; // Already in same set - would create cycle
        }

        // Union by rank
        let rank_a = self.rank[&ra];
        let rank_b = self.rank[&rb];

        if rank_a < rank_b {
            self.parent.insert(ra, rb);
        } else if rank_a > rank_b {
            self.parent.insert(rb, ra);
        } else {
            self.parent.insert(rb, ra);
            self.rank.insert(ra, rank_a + 1);
        }

        true
    }
}

/// Main graph structure
#[pyclass]
pub struct Graph {
    // Node system
    next_node_id: u64,
    nodes: FxHashMap<u64, Node>,
    node_labels: FxHashMap<String, u64>,
    deleted_node_ids: FxHashSet<u64>,
    // Edge system
    edges: Vec<Edge>,
    next_edge_id: u64,
    deleted_edge_ids: FxHashSet<u64>,
    // Adjacency
    adjacency_list: FxHashMap<u64, Vec<u64>>,
    // Graph properties
    directed: bool,
    weighted: bool,
    // Selection state
    selected_nodes: FxHashSet<u64>,
    selected_edges: FxHashSet<u64>,
    // Creation order tracking
    node_creation_order: Vec<u64>,
    edge_creation_order: Vec<u64>,
    // Lazily built lookup tables over `edges` (see `EdgeIndex`); reset whenever edges are
    // removed or edited, and extended in place when edges are added
    edge_index: std::sync::OnceLock<EdgeIndex>,
}

/// O(1) lookups over `Graph::edges`, built on first use.
///
/// `get_edge_weight()` and `get_edge()` used to scan the whole edge list, and Dijkstra calls
/// `get_edge_weight()` on every relaxation, so all-pairs shortest paths cost O(V·E²).
struct EdgeIndex {
    /// (from, to) -> smallest weight among the edges joining them, i.e. the cost of stepping
    /// from `from` to `to`; undirected edges count in both orientations. Missing weights are
    /// 1.0. (This used to be the *first* edge's weight, so a cheaper parallel edge added later
    /// was ignored by every weighted path search.)
    weight_by_pair: FxHashMap<(u64, u64), f64>,
    /// (from, to) -> sum of the weights of all edges joining them (parallel edges add up), for
    /// tie-strength style measures; undirected edges count in both orientations
    total_weight_by_pair: FxHashMap<(u64, u64), f64>,
    /// edge ID -> position in `edges`
    position_by_id: FxHashMap<u64, usize>,
    /// (source, target) -> positions of every edge stored in exactly that orientation (used by
    /// AdaStar, which resolves each edge's weight, with an attribute fallback, before taking
    /// the minimum)
    positions_by_pair: FxHashMap<(u64, u64), Vec<usize>>,
    /// node -> distinct nodes with an edge into it (both endpoints for undirected edges)
    predecessors: FxHashMap<u64, Vec<u64>>,
    /// node -> number of edges stored with it as target (parallel edges and self-loops each
    /// count once), so a directed in-degree is O(1) instead of a scan of every edge
    in_degree: FxHashMap<u64, usize>,
}

impl EdgeIndex {
    fn build(edges: &[Edge], directed: bool) -> Self {
        let mut index = EdgeIndex {
            weight_by_pair: FxHashMap::with_capacity_and_hasher(edges.len() * 2, Default::default()),
            position_by_id: FxHashMap::with_capacity_and_hasher(edges.len(), Default::default()),
            total_weight_by_pair: FxHashMap::with_capacity_and_hasher(
                edges.len() * 2,
                Default::default(),
            ),
            positions_by_pair: FxHashMap::with_capacity_and_hasher(
                edges.len(),
                Default::default(),
            ),
            predecessors: FxHashMap::default(),
            in_degree: FxHashMap::default(),
        };
        for (position, edge) in edges.iter().enumerate() {
            index.add(edge, position, directed);
        }
        index
    }

    /// Record an edge stored at `position`
    fn add(&mut self, edge: &Edge, position: usize, directed: bool) {
        let weight = edge.weight.unwrap_or(1.0);
        let mut orientations = vec![(edge.source, edge.target)];
        if !directed && edge.source != edge.target {
            orientations.push((edge.target, edge.source));
        }
        for pair in orientations {
            self.weight_by_pair
                .entry(pair)
                .and_modify(|w| *w = w.min(weight))
                .or_insert(weight);
            *self.total_weight_by_pair.entry(pair).or_insert(0.0) += weight;
        }
        self.position_by_id.insert(edge.id, position);
        self.positions_by_pair
            .entry((edge.source, edge.target))
            .or_default()
            .push(position);
        *self.in_degree.entry(edge.target).or_insert(0) += 1;
        Self::add_predecessor(&mut self.predecessors, edge.target, edge.source);
        if !directed {
            Self::add_predecessor(&mut self.predecessors, edge.source, edge.target);
        }
    }

    fn add_predecessor(predecessors: &mut FxHashMap<u64, Vec<u64>>, node: u64, pred: u64) {
        let preds = predecessors.entry(node).or_default();
        if !preds.contains(&pred) {
            preds.push(pred);
        }
    }
}

// Edge-index helpers (outside #[pymethods], so they aren't exposed to Python)
impl Graph {
    /// The edge lookup index, built from `edges` on first use
    fn edge_index(&self) -> &EdgeIndex {
        self.edge_index.get_or_init(|| EdgeIndex::build(&self.edges, self.directed))
    }

    /// Drop the edge lookup index after edges are removed or edited
    fn invalidate_edge_index(&mut self) {
        self.edge_index.take();
    }

    /// Append an edge, keeping an already-built index up to date
    fn push_edge(&mut self, edge: Edge) {
        let position = self.edges.len();
        let directed = self.directed;
        if let Some(index) = self.edge_index.get_mut() {
            index.add(&edge, position, directed);
        }
        self.edges.push(edge);
    }
}

// SAFETY: Graph uses Send + Sync types (FxHashMap, Vec, primitives, OnceLock)
// and has no interior mutability beyond the thread-safe OnceLock cache
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

#[pymethods]
impl Graph {
    #[new]
    #[pyo3(signature = (directed=false, weighted=false))]
    fn new(directed: bool, weighted: bool) -> Self {
        Graph {
            next_node_id: 1,
            nodes: FxHashMap::default(),
            node_labels: FxHashMap::default(),
            deleted_node_ids: FxHashSet::default(),
            edges: Vec::new(),
            next_edge_id: 1,
            deleted_edge_ids: FxHashSet::default(),
            adjacency_list: FxHashMap::default(),
            directed,
            weighted,
            selected_nodes: FxHashSet::default(),
            selected_edges: FxHashSet::default(),
            node_creation_order: Vec::new(),
            edge_creation_order: Vec::new(),
            edge_index: std::sync::OnceLock::new(),
        }
    }

    /// Add a node to the graph
    fn add_node(&mut self, mut node: Node) -> PyResult<u64> {
        // IDs and labels are unique: check both before changing anything (an existing ID used to
        // overwrite that node, and a repeated label left the earlier node unreachable by label)
        if node.id != 0 && self.nodes.contains_key(&node.id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node with ID {} already exists",
                node.id
            )));
        }
        if let Some(ref label) = node.label {
            if self.node_labels.contains_key(label) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node with label '{}' already exists",
                    label
                )));
            }
        }

        // Auto-assign node ID if it's 0 (default/unset)
        let node_id = if node.id == 0 {
            let new_id = self.next_node_id;
            self.next_node_id += 1;
            node.id = new_id;
            new_id
        } else {
            // If user provided an ID, make sure next_node_id is beyond it
            if node.id >= self.next_node_id {
                self.next_node_id = node.id + 1;
            }
            node.id
        };

        // Add to label index if label exists
        if let Some(ref label) = node.label {
            self.node_labels.insert(label.clone(), node_id);
        }

        self.node_creation_order.push(node_id);
        self.nodes.insert(node_id, node);
        self.adjacency_list
            .entry(node_id)
            .or_default();

        Ok(node_id)
    }

    /// Add an edge to the graph
    fn add_edge(&mut self, mut edge: Edge) -> PyResult<u64> {
        // Ensure nodes exist
        if !self.nodes.contains_key(&edge.source) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Source node {} does not exist",
                edge.source
            )));
        }
        if !self.nodes.contains_key(&edge.target) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target node {} does not exist",
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
            .entry(edge.source)
            .or_default()
            .push(edge.target);

        if !self.directed {
            self.adjacency_list
                .entry(edge.target)
                .or_default()
                .push(edge.source);
        }

        // Track creation order
        self.edge_creation_order.push(edge.id);

        // Store the edge ID before moving the edge
        let edge_id = edge.id;

        self.push_edge(edge);
        Ok(edge_id)  // Return the edge ID
    }

    /// Get the number of nodes
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get node ID from label
    ///
    /// # Arguments
    /// * `label` - The label to look up
    ///
    /// # Returns
    /// The node ID associated with the label, or an error if not found
    pub fn id_from_label(&self, label: &str) -> PyResult<u64> {
        self.node_labels
            .get(label)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("No node with label '{}'", label)
            ))
    }

    /// Get node label from ID
    ///
    /// # Arguments
    /// * `node_id` - The node ID to look up
    ///
    /// # Returns
    /// The label associated with the node ID, or None if the node has no label
    pub fn label_from_id(&self, node_id: u64) -> PyResult<Option<String>> {
        Ok(self.nodes.get(&node_id).and_then(|n| n.label.clone()))
    }

    /// The ID the next auto-assigned node will get (IDs of deleted nodes are never reused)
    #[getter]
    pub fn next_node_id(&self) -> u64 {
        self.next_node_id
    }

    /// A mark for `_discard_additions()`: the lengths of the node and edge creation logs and
    /// both ID counters
    #[pyo3(name = "_addition_mark")]
    pub fn addition_mark(&self) -> (usize, usize, u64, u64) {
        (
            self.node_creation_order.len(),
            self.edge_creation_order.len(),
            self.next_node_id,
            self.next_edge_id,
        )
    }

    /// Remove every node and edge added since `_addition_mark()` returned `mark` (with the edges
    /// touching those nodes) and restore the ID counters, in one pass. Deletions made since the
    /// mark are not undone.
    #[pyo3(name = "_discard_additions")]
    pub fn discard_additions(&mut self, mark: (usize, usize, u64, u64)) {
        let (node_len, edge_len, next_node, next_edge) = mark;
        let node_len = node_len.min(self.node_creation_order.len());
        let edge_len = edge_len.min(self.edge_creation_order.len());
        let new_nodes: FxHashSet<u64> = self.node_creation_order[node_len..].iter().copied().collect();
        let new_edges: FxHashSet<u64> = self.edge_creation_order[edge_len..].iter().copied().collect();

        let mut removed: Vec<Edge> = Vec::new();
        let mut kept: Vec<Edge> = Vec::with_capacity(self.edges.len());
        for edge in self.edges.drain(..) {
            if new_edges.contains(&edge.id)
                || new_nodes.contains(&edge.source)
                || new_nodes.contains(&edge.target)
            {
                removed.push(edge);
            } else {
                kept.push(edge);
            }
        }
        self.edges = kept;

        // Each added edge appended its endpoints to the adjacency lists, so taking out the last
        // occurrence, newest edge first, restores the lists of the surviving nodes exactly
        fn drop_last(list: Option<&mut Vec<u64>>, value: u64) {
            if let Some(list) = list {
                if let Some(pos) = list.iter().rposition(|&n| n == value) {
                    list.remove(pos);
                }
            }
        }
        for edge in removed.iter().rev() {
            if !new_nodes.contains(&edge.source) {
                drop_last(self.adjacency_list.get_mut(&edge.source), edge.target);
            }
            if !self.directed && !new_nodes.contains(&edge.target) {
                drop_last(self.adjacency_list.get_mut(&edge.target), edge.source);
            }
            self.selected_edges.remove(&edge.id);
        }

        for id in &new_nodes {
            if let Some(node) = self.nodes.remove(id) {
                if let Some(label) = node.label {
                    if self.node_labels.get(&label) == Some(id) {
                        self.node_labels.remove(&label);
                    }
                }
            }
            self.adjacency_list.remove(id);
            self.selected_nodes.remove(id);
        }

        self.node_creation_order.truncate(node_len);
        self.edge_creation_order.truncate(edge_len);
        // Every ID handed out since the mark is gone again, so the counters can rewind
        if !self.nodes.keys().any(|&id| id >= next_node) {
            self.next_node_id = next_node;
            self.deleted_node_ids.retain(|&id| id < next_node);
        }
        if !self.edges.iter().any(|e| e.id >= next_edge) {
            self.next_edge_id = next_edge;
            self.deleted_edge_ids.retain(|&id| id < next_edge);
        }
        self.invalidate_edge_index();
    }

    /// Rename nodes atomically.
    ///
    /// `mapping` pairs node IDs with their new labels. Every ID must exist, and after renaming
    /// all labels must still be unique; otherwise nothing changes and `ValueError` is raised.
    /// Node IDs, attributes and edges are untouched.
    pub fn relabel_nodes(&mut self, mapping: Vec<(u64, String)>) -> PyResult<()> {
        let mut renamed: FxHashMap<u64, String> = FxHashMap::default();
        for (id, label) in mapping {
            if !self.nodes.contains_key(&id) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node {} does not exist",
                    id
                )));
            }
            renamed.insert(id, label);
        }
        // New labels must not collide with each other or with labels that are kept
        let mut seen: FxHashMap<&str, u64> = FxHashMap::default();
        for (id, label) in &renamed {
            if let Some(other) = seen.insert(label.as_str(), *id) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Nodes {} and {} would both be labeled '{}'",
                    other, id, label
                )));
            }
            if let Some(&owner) = self.node_labels.get(label) {
                if !renamed.contains_key(&owner) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Label '{}' is already used by node {}",
                        label, owner
                    )));
                }
            }
        }
        for (id, _) in &renamed {
            if let Some(old) = self.nodes.get(id).and_then(|n| n.label.clone()) {
                self.node_labels.remove(&old);
            }
        }
        for (id, label) in renamed {
            self.node_labels.insert(label.clone(), id);
            if let Some(node) = self.nodes.get_mut(&id) {
                node.label = Some(label);
            }
        }
        Ok(())
    }

    /// Check if a node with the given ID exists
    ///
    /// # Arguments
    /// * `node_id` - The node ID to check
    ///
    /// # Returns
    /// True if the node exists, false otherwise
    #[inline]
    pub fn has_node_id(&self, node_id: u64) -> bool {
        self.nodes.contains_key(&node_id)
    }

    /// Check if a node with the given label exists
    ///
    /// # Arguments
    /// * `label` - The label to check
    ///
    /// # Returns
    /// True if a node with this label exists, false otherwise
    pub fn has_node_label(&self, label: &str) -> bool {
        self.node_labels.contains_key(label)
    }

    /// Get all node labels as a dictionary mapping ID -> label
    ///
    /// # Returns
    /// A Python dict mapping node IDs to their labels (only for nodes that have labels)
    pub fn get_all_labels(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (node_id, node) in &self.nodes {
            if let Some(ref label) = node.label {
                dict.set_item(node_id, label)?;
            }
        }
        Ok(dict.into())
    }

    /// Check if the graph contains any cycles
    ///
    /// Uses depth-first search to detect cycles in the graph.
    pub fn has_cycles(&self) -> bool {
        crate::algorithms::cycles::has_cycle(self)
    }

    /// Find all elementary cycles in the graph using Johnson's algorithm
    ///
    /// Returns all elementary (simple) cycles as vectors of node IDs.
    ///
    /// # Arguments
    /// * `min_length` - Minimum cycle length to report (default 2.0)
    ///
    /// # Returns
    /// Vector of cycles, where each cycle is a vector of node IDs
    ///
    /// # Errors
    /// Returns error if graph is undirected (cycle enumeration requires directed graphs)
    pub fn get_cycles(&self, min_length: f64) -> PyResult<Vec<Vec<u64>>> {
        crate::algorithms::cycles::find_cycles(self, Some(min_length))
    }

    /// Get neighbors of a node
    #[inline]
    pub fn neighbors(&self, node_id: u64) -> PyResult<Vec<u64>> {
        self.adjacency_list
            .get(&node_id)
            .cloned()
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "Node {} does not exist",
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
    pub fn is_weighted(&self) -> bool {
        self.weighted
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<u64> {
        self.nodes.keys().copied().collect()
    }

    /// Get all edges as (id, source, target) tuples
    pub fn edges(&self) -> Vec<(u64, u64, u64)> {
        self.edges
            .iter()
            .map(|e| (e.id, e.source, e.target))
            .collect()
    }

    /// Get edges as "source->target" or "source--target" strings
    ///
    /// Returns edge definitions.
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
    pub fn get_edge_by_id(&self, edge_id: u64) -> Option<(u64, u64)> {
        self.edges
            .iter()
            .find(|e| e.id == edge_id)
            .map(|e| (e.source, e.target))
    }

    /// Check if an edge exists by ID
    #[inline]
    pub fn has_edge_id(&self, edge_id: u64) -> bool {
        self.edges.iter().any(|e| e.id == edge_id)
    }

    /// Find edge IDs by from/to node IDs
    /// Returns all matching edges (supports multi-edges)
    pub fn find_edge_ids(&self, from: u64, to: u64) -> Vec<u64> {
        // Positions of edges stored as from -> to (and, for undirected graphs, to -> from),
        // from the edge index; sorted so IDs come back in edge-list order as before
        let index = self.edge_index();
        let mut positions: Vec<usize> = index
            .positions_by_pair
            .get(&(from, to))
            .cloned()
            .unwrap_or_default();
        if !self.directed && from != to {
            if let Some(reverse) = index.positions_by_pair.get(&(to, from)) {
                positions.extend_from_slice(reverse);
            }
        }
        positions.sort_unstable();
        positions.into_iter().map(|p| self.edges[p].id).collect()
    }

    /// Get weight of edge between two nodes (for internal use in algorithms)
    #[inline]
    pub(crate) fn get_edge_weight(&self, from: u64, to: u64) -> f64 {
        // Cost of stepping from `from` to `to`: the cheapest of the edges joining them (an
        // undirected edge stored the other way round counts too). A missing edge or weight
        // counts as 1.0.
        self.edge_index()
            .weight_by_pair
            .get(&(from, to))
            .copied()
            .unwrap_or(1.0)
    }

    /// Combined strength of the tie from `from` to `to`: the sum of the weights of all edges
    /// joining them (parallel edges add up; undirected edges count in either orientation).
    /// 0.0 if there is no such edge.
    pub(crate) fn get_total_edge_weight(&self, from: u64, to: u64) -> f64 {
        self.edge_index()
            .total_weight_by_pair
            .get(&(from, to))
            .copied()
            .unwrap_or(0.0)
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
            self.invalidate_edge_index();

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

    /// Delete a node and all its incident edges
    pub fn delete_node(&mut self, node_id: u64) -> PyResult<()> {
        let node = self.nodes.remove(&node_id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!(
                "Node {} does not exist",
                node_id
            )))?;

        // Remove from label index if it has a label
        if let Some(ref label) = node.label {
            self.node_labels.remove(label);
        }

        // Add to tombstones
        self.deleted_node_ids.insert(node_id);

        // Remove from adjacency list
        self.adjacency_list.remove(&node_id);

        // Remove the deleted node from all other nodes' adjacency lists
        for (_, neighbors) in self.adjacency_list.iter_mut() {
            neighbors.retain(|&n| n != node_id);
        }

        // Remove all incident edges, dropping them from the edge selection too
        let selected_edges = &mut self.selected_edges;
        self.edges.retain(|e| {
            let incident = e.source == node_id || e.target == node_id;
            if incident {
                selected_edges.remove(&e.id);
            }
            !incident
        });
        self.invalidate_edge_index();

        // Clean up selection state
        self.selected_nodes.remove(&node_id);

        Ok(())
    }

    /// Set or update an attribute on a node
    pub fn set_node_attr(&mut self, py: Python, node_id: u64, key: &str, value: PyObject) -> PyResult<()> {
        let node = self.nodes.get_mut(&node_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Node '{}' does not exist", node_id))
        })?;

        // Get or create attributes dict
        if node.attributes.is_none() {
            let dict = PyDict::new(py);
            node.attributes = Some(dict.unbind());
        }

        // Update the attribute
        if let Some(ref attrs) = node.attributes {
            attrs.bind(py).set_item(key, value)?;
        }

        Ok(())
    }

    /// Set or update an attribute on an edge (including weight)
    pub fn set_edge_attr(&mut self, py: Python, edge_id: u64, key: &str, value: PyObject) -> PyResult<()> {
        // The weight may change; the index is rebuilt on next use
        self.invalidate_edge_index();
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
                let dict = PyDict::new(py);
                edge.attributes = Some(dict.unbind());
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

        // Endpoints change; the index is rebuilt on next use
        self.invalidate_edge_index();
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
            .entry(edge.source)
            .or_default()
            .push(edge.target);

        Ok(())
    }

    // ========== Selection Methods ==========

    /// Select nodes by their IDs
    #[inline]
    pub fn select_nodes_by_id(&mut self, node_ids: Vec<u64>) -> PyResult<()> {
        // Check every ID before selecting any, so a bad ID leaves the selection unchanged
        if let Some(missing) = node_ids.iter().find(|id| !self.nodes.contains_key(id)) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Node {} does not exist",
                missing
            )));
        }
        self.selected_nodes.extend(node_ids);
        Ok(())
    }

    /// Select edges by their IDs
    #[inline]
    pub fn select_edges_by_id(&mut self, edge_ids: Vec<u64>) -> PyResult<()> {
        // Check every ID before selecting any, so a bad ID leaves the selection unchanged
        let existing: FxHashSet<u64> = self.edges.iter().map(|e| e.id).collect();
        if let Some(missing) = edge_ids.iter().find(|id| !existing.contains(id)) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Edge with ID {} does not exist",
                missing
            )));
        }
        self.selected_edges.extend(edge_ids);
        Ok(())
    }

    /// Select edges connected to specified nodes
    #[inline]
    pub fn select_edges_by_node_id(&mut self, node_ids: Vec<u64>) -> PyResult<()> {
        let mut node_set = FxHashSet::with_capacity_and_hasher(node_ids.len(), Default::default());
        node_set.extend(node_ids);

        for edge in &self.edges {
            if node_set.contains(&edge.source) || node_set.contains(&edge.target) {
                self.selected_edges.insert(edge.id);
            }
        }
        Ok(())
    }

    /// Select nodes by degree criteria
    #[inline]
    #[pyo3(signature = (min_degree=None, max_degree=None, degree_type="total"))]
    pub fn select_nodes_by_degree(
        &mut self,
        min_degree: Option<usize>,
        max_degree: Option<usize>,
        degree_type: &str,
    ) -> PyResult<()> {
        if !matches!(degree_type, "in" | "out" | "total") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid degree_type '{}'. Must be 'in', 'out', or 'total'",
                degree_type
            )));
        }
        let degrees = self.degrees_by_type(degree_type);
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        for node_id in node_ids {
            let degree = degrees.get(&node_id).copied().unwrap_or(0);

            let matches = match (min_degree, max_degree) {
                (Some(min), Some(max)) => degree >= min && degree <= max,
                (Some(min), None) => degree >= min,
                (None, Some(max)) => degree <= max,
                (None, None) => true,
            };

            if matches {
                self.selected_nodes.insert(node_id);
            }
        }
        Ok(())
    }

    /// Degree of every node in one pass: `"in"`, `"out"`, or `"total"` (in + out on directed
    /// graphs, as in `get_degree()`). On undirected graphs all three are the adjacency length
    fn degrees_by_type(&self, degree_type: &str) -> FxHashMap<u64, usize> {
        let mut degrees: FxHashMap<u64, usize> =
            FxHashMap::with_capacity_and_hasher(self.nodes.len(), Default::default());
        let count_out = !self.directed || degree_type != "in";
        let count_in = self.directed && degree_type != "out";
        for &node_id in self.nodes.keys() {
            degrees.insert(node_id, if count_out { self.degree(node_id) } else { 0 });
        }
        if count_in {
            for (&node_id, &d) in &self.edge_index().in_degree {
                *degrees.entry(node_id).or_insert(0) += d;
            }
        }
        degrees
    }

    /// Get the degree of a node (total neighbors)
    #[inline]
    fn degree(&self, node_id: u64) -> usize {
        self.adjacency_list
            .get(&node_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get in-degree of a node (for directed graphs)
    #[inline]
    fn in_degree(&self, node_id: u64) -> usize {
        self.get_in_degree(node_id)
    }

    /// Get out-degree of a node (for directed graphs)
    #[inline]
    fn out_degree(&self, node_id: u64) -> usize {
        self.get_out_degree(node_id)
    }

    /// Deselect specific nodes
    #[inline]
    pub fn deselect_nodes(&mut self, node_ids: Vec<u64>) -> PyResult<()> {
        for node_id in node_ids {
            self.selected_nodes.remove(&node_id);
        }
        Ok(())
    }

    /// Deselect specific edges
    #[inline]
    pub fn deselect_edges(&mut self, edge_ids: Vec<u64>) -> PyResult<()> {
        for edge_id in edge_ids {
            self.selected_edges.remove(&edge_id);
        }
        Ok(())
    }

    /// Clear all selections
    #[inline]
    pub fn clear_selection(&mut self) -> PyResult<()> {
        self.selected_nodes.clear();
        self.selected_edges.clear();
        Ok(())
    }

    /// Invert node selection
    #[inline]
    pub fn invert_node_selection(&mut self) -> PyResult<()> {
        let n = self.nodes.len();
        let mut all_nodes = FxHashSet::with_capacity_and_hasher(n, Default::default());
        all_nodes.extend(self.nodes.keys().copied());
        self.selected_nodes = all_nodes
            .difference(&self.selected_nodes)
            .copied()
            .collect();
        Ok(())
    }

    /// Invert edge selection
    #[inline]
    pub fn invert_edge_selection(&mut self) -> PyResult<()> {
        let n = self.edges.len();
        let mut all_edges = FxHashSet::with_capacity_and_hasher(n, Default::default());
        all_edges.extend(self.edges.iter().map(|e| e.id));
        self.selected_edges = all_edges
            .difference(&self.selected_edges)
            .cloned()
            .collect();
        Ok(())
    }

    /// Get currently selected node IDs
    #[inline]
    pub fn get_selected_node_ids(&self) -> Vec<u64> {
        self.selected_nodes.iter().copied().collect()
    }

    /// Get currently selected edge IDs
    #[inline]
    pub fn get_selected_edge_ids(&self) -> Vec<u64> {
        self.selected_edges.iter().cloned().collect()
    }

    /// Get count of selected nodes
    #[inline]
    pub fn count_selected_nodes(&self) -> usize {
        self.selected_nodes.len()
    }

    /// Get count of selected edges
    #[inline]
    pub fn count_selected_edges(&self) -> usize {
        self.selected_edges.len()
    }

    /// Select nodes using a Python callable/predicate function
    /// The predicate receives a Node object and should return True/False
    #[inline]
    pub fn select_nodes(&mut self, py: Python, predicate: PyObject) -> PyResult<()> {
        for (&node_id, node) in &self.nodes {
            // Call the predicate with the node
            let result = predicate.call1(py, (node.clone(),))?;
            let matches: bool = result.extract(py)?;

            if matches {
                self.selected_nodes.insert(node_id);
            }
        }
        Ok(())
    }

    /// Select edges using a Python callable/predicate function
    /// The predicate receives an Edge object and should return True/False
    #[inline]
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
    #[inline]
    pub fn get_node(&self, node_id: u64) -> Option<Node> {
        self.nodes.get(&node_id).cloned()
    }

    /// Get an edge object by ID (for use with selection predicates)
    pub fn get_edge(&self, edge_id: u64) -> Option<Edge> {
        let position = *self.edge_index().position_by_id.get(&edge_id)?;
        self.edges.get(position).cloned()
    }

    /// Select the last N nodes that were created
    #[pyo3(signature = (n))]
    pub fn select_last_nodes_created(&mut self, n: usize) -> PyResult<()> {
        let order = self.live_node_creation_order();
        let start_idx = order.len().saturating_sub(n);

        for &node_id in &order[start_idx..] {
            self.selected_nodes.insert(node_id);
        }
        Ok(())
    }

    /// Select the last N edges that were created
    #[pyo3(signature = (n))]
    pub fn select_last_edges_created(&mut self, n: usize) -> PyResult<()> {
        let order = self.live_edge_creation_order();
        let start_idx = order.len().saturating_sub(n);

        for &edge_id in &order[start_idx..] {
            self.selected_edges.insert(edge_id);
        }
        Ok(())
    }

    /// Get the creation order of all nodes
    pub fn get_node_creation_order(&self) -> Vec<u64> {
        self.live_node_creation_order()
    }

    /// Get the creation order of all edges
    pub fn get_edge_creation_order(&self) -> Vec<u64> {
        self.live_edge_creation_order()
    }

    /// Select nodes in the K-hop neighborhood of specified nodes
    #[pyo3(signature = (node_ids, k=1, mode="out"))]
    #[inline]
    pub fn select_nodes_in_neighborhood(
        &mut self,
        node_ids: Vec<u64>,
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
        let n = self.num_nodes();
        let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
        let mut current_level = FxHashSet::with_capacity_and_hasher(node_ids.len(), Default::default());

        // Initialize with seed nodes
        for node_id in node_ids {
            if !self.nodes.contains_key(&node_id) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Node {} does not exist", node_id)
                ));
            }
            current_level.insert(node_id);
        }

        // BFS for k hops
        for _ in 0..k {
            let mut next_level = FxHashSet::with_capacity_and_hasher(n, Default::default());

            for &node_id in &current_level {
                visited.insert(node_id);

                // Get neighbors based on mode
                match effective_mode {
                    "out" => {
                        // Outgoing neighbors
                        if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                            for &neighbor in neighbors {
                                if !visited.contains(&neighbor) {
                                    next_level.insert(neighbor);
                                }
                            }
                        }
                    }
                    "in" => {
                        // Incoming neighbors - need to search all edges
                        for edge in &self.edges {
                            if edge.target == node_id && !visited.contains(&edge.source) {
                                next_level.insert(edge.source);
                            }
                        }
                    }
                    "all" => {
                        // Outgoing neighbors from adjacency list
                        if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                            for &neighbor in neighbors {
                                if !visited.contains(&neighbor) {
                                    next_level.insert(neighbor);
                                }
                            }
                        }
                        // Incoming neighbors from edge list
                        for edge in &self.edges {
                            if edge.target == node_id && !visited.contains(&edge.source) {
                                next_level.insert(edge.source);
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
    #[inline]
    pub fn select_nodes_by_scores(
        &mut self,
        scores: FxHashMap<u64, f64>,
        k: usize,
        include_ties: bool,
    ) -> PyResult<()> {
        if scores.is_empty() {
            return Ok(());
        }

        // Convert to vector and sort by score (descending)
        let mut scored_nodes = Vec::with_capacity(scores.len());
        scored_nodes.extend(scores);
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
                self.selected_nodes.insert(*node_id);
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
    #[inline]
    #[pyo3(signature = (n, p, seed=None))]
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

        // Add nodes with integer IDs
        let mut node_ids: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(id);
        }

        for &node_id in &node_ids {
            self.add_node(Node::new(node_id, None, None))?;
        }

        // Add edges with probability p
        if self.directed {
            // For directed graphs, consider all ordered pairs
            for i in 0..n {
                for j in 0..n {
                    if i != j && rng.gen::<f64>() < p {
                        self.add_edge(Edge::new(
                            0, // edge_id will be auto-assigned
                            node_ids[i],
                            node_ids[j],
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
                            node_ids[i],
                            node_ids[j],
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
    #[inline]
    #[pyo3(signature = (n, m, seed=None))]
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

        // Add nodes using next_node_id counter
        let mut node_ids: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(id);
        }

        for &node_id in &node_ids {
            self.add_node(Node::new(node_id, None, None))?;
        }

        // Generate all possible edges
        let mut possible_edges: Vec<(u64, u64)> = Vec::with_capacity(max_edges);

        if self.directed {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        possible_edges.push((node_ids[i], node_ids[j]));
                    }
                }
            }
        } else {
            for i in 0..n {
                for j in (i + 1)..n {
                    possible_edges.push((node_ids[i], node_ids[j]));
                }
            }
        }

        // Randomly select m edges
        possible_edges.shuffle(&mut rng);
        for &(u, v) in possible_edges.iter().take(m) {
            self.add_edge(Edge::new(0, u, v, None, None))?;
        }

        Ok(())
    }

    /// Generate Barabási-Albert preferential attachment graph
    /// Grows a graph by adding nodes one at a time with preferential attachment
    #[inline]
    #[pyo3(signature = (n, m, seed=None))]
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

        // Generate node IDs using next_node_id counter
        let mut node_ids: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(id);
        }

        // Add first m+1 nodes in a clique
        for &id in &node_ids[..=m] {
            self.add_node(Node::new(id, None, None))?;
        }

        for i in 0..=m {
            for j in (i + 1)..=m {
                self.add_edge(Edge::new(
                    0,
                    node_ids[i],
                    node_ids[j],
                    None,
                    None,
                ))?;
            }
        }

        // Track degree for preferential attachment
        let mut degrees: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &id in &node_ids[..=m] {
            degrees.insert(id, m);
        }

        // Add remaining nodes with preferential attachment
        for &new_id in &node_ids[m + 1..n] {
            self.add_node(Node::new(new_id, None, None))?;

            // Select m targets with probability proportional to degree
            let mut targets: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(m, Default::default());
            let total_degree: usize = degrees.values().sum();

            while targets.len() < m {
                let r = rng.gen::<f64>() * total_degree as f64;
                let mut cumsum = 0.0;

                for (&node, &deg) in &degrees {
                    cumsum += deg as f64;
                    if cumsum >= r {
                        targets.insert(node);
                        break;
                    }
                }
            }

            // Add edges to selected targets
            for &target in &targets {
                self.add_edge(Edge::new(
                    0,
                    new_id,
                    target,
                    None,
                    None,
                ))?;
                *degrees.get_mut(&target).unwrap() += 1;
            }

            degrees.insert(new_id, m);
        }

        Ok(())
    }

    /// Generate Watts-Strogatz small-world graph
    /// Creates a ring lattice with rewiring probability p
    #[inline]
    #[pyo3(signature = (n, k, p, seed=None))]
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

        // Generate node IDs using next_node_id counter
        let mut node_ids: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(id);
        }

        // Add nodes
        for &node_id in &node_ids {
            self.add_node(Node::new(node_id, None, None))?;
        }

        // Ring lattice with k/2 neighbors on each side, then rewiring; shared with
        // `add_connected_watts_strogatz_graph()` so both draw the same graphs
        let edges = crate::algorithms::random_models::watts_strogatz(&mut rng, n, k, p);

        for (u_idx, v_idx) in edges {
            self.add_edge(Edge::new(0, node_ids[u_idx], node_ids[v_idx], None, None))?;
        }

        Ok(())
    }

    /// Generate Johnson graph J(n, k)
    ///
    /// Creates a graph where vertices are k-subsets of {0, 1, ..., n-1},
    /// and two vertices are adjacent if their subsets intersect in exactly k-1 elements
    /// (i.e., they differ by exactly one element).
    ///
    /// # Properties
    /// - Regular graph: each node has degree k(n-k)
    /// - Distance-regular
    /// C(n,k) vertices where C is binomial coefficient
    /// - Undirected, no self-loops
    ///
    /// # Arguments
    /// * `n` - Size of the base set (must be >= 0)
    /// * `k` - Size of each subset (must satisfy 0 <= k <= n)
    ///
    /// # Returns
    /// PyResult<()> - Ok if successful, Err if parameters invalid
    ///
    /// # Algorithm
    /// 1. Generate all k-combinations of {0, 1, ..., n-1}
    /// 2. For each pair of combinations, check if intersection size = k-1
    /// 3. Add edge if condition satisfied
    ///
    /// Time complexity: O(C(n,k)^2 * k) where C(n,k) is binomial coefficient
    #[pyo3(signature = (n, k))]
    #[inline]
    fn generate_johnson_graph(&mut self, n: usize, k: usize) -> PyResult<()> {
        if k > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be between 0 and n"
            ));
        }

        // Generate all k-combinations of {0, 1, ..., n-1}
        let elements: Vec<usize> = (0..n).collect();
        let combinations = Self::generate_combinations(&elements, k);

        // Labels are comma-separated sorted subsets; check them all before adding any node
        let labels: Vec<String> = combinations
            .iter()
            .map(|combo| combo.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
            .collect();
        self.check_new_labels(&labels)?;

        // Add nodes for each combination; node IDs are assigned sequentially
        let mut node_ids = Vec::with_capacity(combinations.len());
        for label in labels {
            let node_id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(node_id);
            self.add_node(Node::new(node_id, Some(label), None))?;
        }

        // Add edges between combinations that differ by exactly one element
        // Two k-subsets are adjacent if they intersect in k-1 elements
        for i in 0..combinations.len() {
            for j in (i + 1)..combinations.len() {
                let mut set_i = FxHashSet::with_capacity_and_hasher(k, Default::default());
                set_i.extend(combinations[i].iter().copied());
                let mut set_j = FxHashSet::with_capacity_and_hasher(k, Default::default());
                set_j.extend(combinations[j].iter().copied());

                // Intersection size
                let intersection_size = set_i.intersection(&set_j).count();

                if intersection_size == k.saturating_sub(1) {
                    let edge = Edge::new(
                        0,
                        node_ids[i],
                        node_ids[j],
                        None,
                        None,
                    );
                    self.add_edge(edge)?;
                }
            }
        }

        Ok(())
    }

    /// Generate a random geometric graph.
    ///
    /// Places n nodes uniformly at random in the unit cube (or unit hypercube for
    /// higher dimensions), and connects nodes whose Euclidean distance is at most radius.
    ///
    /// # Arguments
    /// * `n` - Number of nodes to create
    /// * `radius` - Connection threshold distance
    /// * `dim` - Number of dimensions (default 2)
    /// * `p_norm` - Minkowski distance parameter (2.0 = Euclidean, default 2.0)
    /// * `seed` - Optional seed for reproducibility
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if parameters are invalid
    #[pyo3(signature = (n, radius, dim=2, p_norm=2.0, seed=None))]
    fn generate_random_geometric_graph(
        &mut self,
        n: usize,
        radius: f64,
        dim: usize,
        p_norm: f64,
        seed: Option<u64>,
    ) -> PyResult<()> {
        if radius < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "radius must be non-negative",
            ));
        }
        if dim < 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dimension must be at least 1",
            ));
        }
        if p_norm <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "p_norm must be positive",
            ));
        }

        if n == 0 {
            return Ok(());
        }

        // Create RNG with optional seed
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Generate random positions in [0, 1]^dim for each node
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f64>()).collect())
            .collect();

        // Store each position as JSON in its label; check them all before adding any node
        let labels: Vec<String> = positions
            .iter()
            .map(|pos| {
                format!(
                    "[{}]",
                    pos.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>().join(", ")
                )
            })
            .collect();
        self.check_new_labels(&labels)?;

        let mut node_ids: Vec<u64> = Vec::with_capacity(n);
        for label in labels {
            let node_id = self.next_node_id;
            self.next_node_id += 1;
            node_ids.push(node_id);
            self.add_node(Node::new(node_id, Some(label), None))?;
        }

        // Pre-compute radius^p for comparison
        let radius_p = if p_norm == 2.0 {
            radius * radius  // Compare squared distances for efficiency
        } else {
            radius.powf(p_norm)
        };

        // Connect nodes within radius using Minkowski distance
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_p = if p_norm == 2.0 {
                    // Squared Euclidean distance (common case, optimized)
                    positions[i].iter()
                        .zip(positions[j].iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum::<f64>()
                } else {
                    // General Minkowski distance
                    positions[i].iter()
                        .zip(positions[j].iter())
                        .map(|(a, b)| (a - b).abs().powf(p_norm))
                        .sum::<f64>()
                };

                if dist_p <= radius_p {
                    let edge = Edge::new(
                        0,
                        node_ids[i],
                        node_ids[j],
                        None,
                        None,
                    );
                    self.add_edge(edge)?;
                }
            }
        }

        Ok(())
    }

    // ========== Community Detection ==========

    // ========== Structural Properties ==========

    /// Calculate graph density
    /// Density = actual edges / possible edges
    #[inline]
    fn density(&self) -> f64 {
        let n = self.num_nodes();
        if n <= 1 {
            return 0.0;
        }

        let actual_edges = self.num_edges() as f64;
        let possible_edges = if self.directed {
            (n * (n - 1)) as f64
        } else {
            (n * (n - 1)) as f64 * 0.5
        };

        if possible_edges == 0.0 {
            0.0
        } else {
            actual_edges * (1.0 / possible_edges)
        }
    }

    /// Check if graph is connected (for undirected) or weakly connected (for directed)
    #[inline]
    fn is_connected(&self) -> bool {
        let n = self.num_nodes();
        if n == 0 {
            return true;
        }

        // BFS from first node
        let start_node = match self.nodes.keys().next() {
            Some(&node) => node,
            None => return true,
        };

        let mut visited: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
        let mut queue: Vec<u64> = Vec::with_capacity(n);
        queue.push(start_node);
        visited.insert(start_node);

        while let Some(current) = queue.pop() {
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }

            // For directed graphs, also check reverse edges (weak connectivity)
            if self.directed {
                for edge in &self.edges {
                    if edge.target == current && !visited.contains(&edge.source) {
                        visited.insert(edge.source);
                        queue.push(edge.source);
                    }
                }
            }
        }

        visited.len() == self.num_nodes()
    }

    /// Check if directed graph is a DAG (Directed Acyclic Graph)
    #[inline]
    fn is_dag(&self) -> PyResult<bool> {
        if !self.directed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "is_dag() only works on directed graphs"
            ));
        }

        let n = self.num_nodes();
        // Use DFS with coloring: white (unvisited), gray (visiting), black (visited)
        let mut color: FxHashMap<u64, u8> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &node_id in self.nodes.keys() {
            color.insert(node_id, 0); // white
        }

        fn has_cycle(
            node: u64,
            color: &mut FxHashMap<u64, u8>,
            adj_list: &FxHashMap<u64, Vec<u64>>,
        ) -> bool {
            color.insert(node, 1); // gray

            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    let neighbor_color = *color.get(&neighbor).unwrap_or(&0);
                    if neighbor_color == 1 {
                        // Back edge found - cycle detected
                        return true;
                    }
                    if neighbor_color == 0 && has_cycle(neighbor, color, adj_list) {
                        return true;
                    }
                }
            }

            color.insert(node, 2); // black
            false
        }

        for &node_id in self.nodes.keys() {
            if *color.get(&node_id).unwrap() == 0
                && has_cycle(node_id, &mut color, &self.adjacency_list) {
                    return Ok(false);
                }
        }

        Ok(true)
    }

    /// Check if graph is simple (no self-loops or multi-edges)
    #[inline]
    fn is_simple(&self) -> bool {
        // Check for self-loops
        for edge in &self.edges {
            if edge.source == edge.target {
                return false;
            }
        }

        // Check for multi-edges
        let mut seen_edges: FxHashSet<(u64, u64)> = FxHashSet::with_capacity_and_hasher(self.num_edges(), Default::default());
        for edge in &self.edges {
            let edge_key = if self.directed {
                (edge.source, edge.target)
            } else {
                // Normalize undirected edge
                if edge.source <= edge.target {
                    (edge.source, edge.target)
                } else {
                    (edge.target, edge.source)
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
    #[inline]
    fn degree_distribution(&self) -> FxHashMap<usize, usize> {
        let n = self.num_nodes();
        let mut distribution: FxHashMap<usize, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());

        // Total degree (in + out on directed graphs), matching get_degree()
        for (_, deg) in self.degrees_by_type("total") {
            *distribution.entry(deg).or_insert(0) += 1;
        }

        distribution
    }

    /// Check if graph is bipartite using BFS coloring
    #[inline]
    fn is_bipartite(&self) -> bool {
        // Same 2-coloring as bipartite_sets(), which ignores edge direction, so the two agree
        crate::algorithms::bipartite::bipartite_sets(self).is_ok()
    }

    /// Count self-loops in the graph
    #[inline]
    fn count_all_loops(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.source == e.target)
            .count()
    }

    /// Count multi-edges (multiple edges between same node pair)
    /// Only counts parallel edges beyond the first one
    #[inline]
    fn count_multi_edges(&self) -> usize {
        let mut edge_pairs: FxHashMap<(u64, u64), usize> = FxHashMap::with_capacity_and_hasher(self.num_edges(), Default::default());

        for edge in &self.edges {
            // Undirected A-B and B-A are the same pair, as in is_simple()
            let key = if !self.directed && edge.source > edge.target {
                (edge.target, edge.source)
            } else {
                (edge.source, edge.target)
            };
            *edge_pairs.entry(key).or_insert(0) += 1;
        }

        // Count edges beyond the first for each pair
        edge_pairs.values().filter(|&&count| count > 1).map(|&count| count - 1).sum()
    }

    /// Count mutual node pairs (bidirectional edges in directed graphs)
    /// Returns number of node pairs (a,b) where both (a->b) and (b->a) edges exist
    #[inline]
    fn count_mutual_node_pairs(&self) -> PyResult<usize> {
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Mutual node pairs only defined for directed graphs",
            ));
        }

        // Distinct (source, target) pairs, so parallel edges count once
        let mut edge_pairs: FxHashSet<(u64, u64)> = FxHashSet::with_capacity_and_hasher(self.num_edges(), Default::default());
        for edge in &self.edges {
            // Skip self-loops
            if edge.source != edge.target {
                edge_pairs.insert((edge.source, edge.target));
            }
        }

        // Count each mutual pair once, from its lower-ID end
        let mutual_count = edge_pairs
            .iter()
            .filter(|&&(src, tgt)| src < tgt && edge_pairs.contains(&(tgt, src)))
            .count();

        Ok(mutual_count)
    }

    /// Count asymmetric node pairs (one-way edges in directed graphs)
    /// Returns number of node pairs (a,b) where edge (a->b) exists but not (b->a)
    #[inline]
    fn count_asymmetric_node_pairs(&self) -> PyResult<usize> {
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Asymmetric node pairs only defined for directed graphs",
            ));
        }

        let mut edge_pairs: FxHashSet<(u64, u64)> = FxHashSet::with_capacity_and_hasher(self.num_edges(), Default::default());

        // Collect all edges
        for edge in &self.edges {
            // Skip self-loops
            if edge.source != edge.target {
                edge_pairs.insert((edge.source, edge.target));
            }
        }

        // Count edges without reverse
        let mut asymmetric_count = 0;
        for &(src, tgt) in &edge_pairs {
            if !edge_pairs.contains(&(tgt, src)) {
                asymmetric_count += 1;
            }
        }

        Ok(asymmetric_count)
    }

    /// Get average degree
    #[inline]
    fn average_degree(&self) -> f64 {
        let n = self.num_nodes();
        if n == 0 {
            return 0.0;
        }

        // For directed graphs `degree()` is the out-degree only; the documented
        // average is of total degree (in + out), and every arc adds one to each.
        let total_degree: usize = if self.directed {
            2 * self.edges.len()
        } else {
            self.nodes.keys().map(|&id| self.degree(id)).sum()
        };
        total_degree as f64 * (1.0 / n as f64)
    }

    /// Find articulation points (cut vertices) in an undirected graph
    /// Returns list of node IDs whose removal would disconnect the graph
    /// Uses Tarjan's algorithm with DFS
    fn get_articulation_points(&self) -> PyResult<Vec<u64>> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Articulation points only defined for undirected graphs",
            ));
        }

        let n = self.num_nodes();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
        let mut disc = FxHashMap::with_capacity_and_hasher(n, Default::default()); // Discovery time
        let mut low = FxHashMap::with_capacity_and_hasher(n, Default::default()); // Lowest reachable ancestor
        let mut parent: FxHashMap<u64, Option<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut ap = FxHashSet::with_capacity_and_hasher(n, Default::default()); // Articulation points
        let mut time = 0usize;

        // DFS helper function
        #[allow(clippy::too_many_arguments)] // recursive Tarjan DFS threading its state
        fn dfs_ap(
            node: u64,
            visited: &mut FxHashSet<u64>,
            disc: &mut FxHashMap<u64, usize>,
            low: &mut FxHashMap<u64, usize>,
            parent: &mut FxHashMap<u64, Option<u64>>,
            ap: &mut FxHashSet<u64>,
            time: &mut usize,
            adj_list: &FxHashMap<u64, Vec<u64>>,
        ) {
            let mut children = 0;
            visited.insert(node);
            *time += 1;
            disc.insert(node, *time);
            low.insert(node, *time);

            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        children += 1;
                        parent.insert(neighbor, Some(node));
                        dfs_ap(neighbor, visited, disc, low, parent, ap, time, adj_list);

                        // Check if subtree rooted at neighbor has connection back to ancestors
                        low.insert(
                            node,
                            (*low.get(&node).unwrap()).min(*low.get(&neighbor).unwrap()),
                        );

                        // If node is root and has 2+ children, it's an AP
                        if parent.get(&node).unwrap().is_none() && children > 1 {
                            ap.insert(node);
                        }

                        // If node is not root and low[neighbor] >= disc[node], it's an AP
                        if parent.get(&node).unwrap().is_some()
                            && *low.get(&neighbor).unwrap() >= *disc.get(&node).unwrap()
                        {
                            ap.insert(node);
                        }
                    } else if Some(neighbor) != *parent.get(&node).unwrap() {
                        // Update low value for back edge
                        low.insert(
                            node,
                            (*low.get(&node).unwrap()).min(*disc.get(&neighbor).unwrap()),
                        );
                    }
                }
            }
        }

        // Run DFS from each unvisited node (handles disconnected components)
        for &start_node in self.nodes.keys() {
            if !visited.contains(&start_node) {
                parent.insert(start_node, None);
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
    fn compute_eccentricities(&self) -> FxHashMap<u64, Option<usize>> {
        let n = self.num_nodes();
        let mut eccentricities = FxHashMap::with_capacity_and_hasher(n, Default::default());

        for &node in self.nodes.keys() {
            // BFS to find max distance from this node
            let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
            let mut queue = std::collections::VecDeque::with_capacity(n);
            let mut distances: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());

            visited.insert(node);
            queue.push_back(node);
            distances.insert(node, 0);

            while let Some(current) = queue.pop_front() {
                let current_dist = *distances.get(&current).unwrap();

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            distances.insert(neighbor, current_dist + 1);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            // For unconnected graphs:
            // - If all nodes reachable: eccentricity = max distance
            // - If only some reachable: eccentricity = max distance within component (for unconnected=true)
            //   or None (for unconnected=false, handled by caller)
            let max_dist = distances.values().max().copied().unwrap_or(0);
            eccentricities.insert(node, Some(max_dist));
        }

        eccentricities
    }

    /// Get periphery nodes (nodes with maximum eccentricity)
    /// Returns list of node IDs
    /// Returns empty list if graph is disconnected
    #[inline]
    fn get_periphery(&self) -> Vec<u64> {
        let eccentricities = self.compute_eccentricities();

        // Filter out None values (disconnected nodes)
        let valid_eccentricities: Vec<_> = eccentricities
            .iter()
            .filter_map(|(&node, ecc)| ecc.map(|e| (node, e)))
            .collect();

        if valid_eccentricities.is_empty() {
            return Vec::new();
        }

        let max_ecc = *valid_eccentricities.iter().map(|(_, e)| e).max().unwrap();

        let mut result = Vec::with_capacity(valid_eccentricities.len());
        for (node, e) in valid_eccentricities {
            if e == max_ecc {
                result.push(node);
            }
        }
        result
    }

    /// Get graph diameter (maximum eccentricity)
    /// Returns None if graph is disconnected
    #[inline]
    fn get_diameter(&self) -> Option<usize> {
        self.get_diameter_with_unconnected(true)
    }

    /// Get graph diameter with optional handling of disconnected graphs
    ///
    /// # Arguments
    /// * `unconnected` - If true (default), compute max diameter across all components.
    ///                   If false, return None for disconnected graphs.
    /// True when every node can reach every other: strongly connected if directed, connected
    /// otherwise. False for an empty graph.
    fn all_pairs_reachable(&self) -> bool {
        if self.directed {
            crate::algorithms::connectivity::count_strongly_connected_components(self)
                .map_or(false, |c| c == 1)
        } else {
            self.count_weakly_connected_components() == 1
        }
    }

    fn get_diameter_with_unconnected(&self, unconnected: bool) -> Option<usize> {
        // unconnected=false needs every node to reach every other: strong connectivity on
        // directed graphs (as get_mean_distance requires), plain connectivity otherwise
        let is_connected = self.all_pairs_reachable();

        // If unconnected=false and graph is disconnected, return None
        if !unconnected && !is_connected {
            return None;
        }

        // Compute eccentricities and return max
        let eccentricities = self.compute_eccentricities();
        eccentricities
            .values()
            .filter_map(|&ecc| ecc)
            .max()
    }

    /// Get graph radius (minimum eccentricity)
    /// Returns None if graph is disconnected (by default handles disconnected graphs)
    #[inline]
    fn get_radius(&self) -> Option<usize> {
        self.get_radius_with_unconnected(true)
    }

    /// Get graph radius with optional handling of disconnected graphs
    ///
    /// # Arguments
    /// * `unconnected` - If true (default), compute min radius across all components.
    ///                   If false, return None for disconnected graphs.
    fn get_radius_with_unconnected(&self, unconnected: bool) -> Option<usize> {
        // unconnected=false needs every node to reach every other: strong connectivity on
        // directed graphs (as get_mean_distance requires), plain connectivity otherwise
        let is_connected = self.all_pairs_reachable();

        // If unconnected=false and graph is disconnected, return None
        if !unconnected && !is_connected {
            return None;
        }

        self.center_eccentricities().into_values().min()
    }

    /// Eccentricities of the nodes eligible to be central (used by the radius and the center).
    ///
    /// A node's eccentricity only covers what it can reach, so a node reaching nothing (an
    /// isolated node, a sink) would have eccentricity 0 and pull the radius to 0. Only nodes that
    /// reach every node of their weakly connected component count, which on undirected graphs
    /// is every node. Single-node components are skipped unless every component is a single
    /// node (a graph without edges between distinct nodes), where each node is its own center
    /// with eccentricity 0. Empty when no node qualifies, e.g. a directed in-star.
    fn center_eccentricities(&self) -> FxHashMap<u64, usize> {
        let components = self.weakly_connected_components();
        let all_singletons = components.iter().all(|c| c.len() == 1);
        let mut result: FxHashMap<u64, usize> = FxHashMap::default();
        for component in &components {
            if component.len() == 1 {
                if all_singletons {
                    result.insert(component[0], 0);
                }
                continue;
            }
            for &node in component {
                let distances = self.bfs_distances(node);
                if distances.len() == component.len() {
                    result.insert(node, distances.values().max().copied().unwrap_or(0));
                }
            }
        }
        result
    }

    /// Get mean distance (average shortest path length)
    /// Returns None if graph is disconnected (by default handles disconnected graphs)
    /// Computes the average of all pairwise shortest path lengths
    #[inline]
    fn get_mean_distance(&self) -> Option<f64> {
        self.get_mean_distance_with_unconnected(true)
    }

    /// Get mean distance with optional handling of disconnected graphs
    ///
    /// # Arguments
    /// * `unconnected` - If true (default), compute weighted average across all components.
    ///                   If false, return None for disconnected graphs.
    fn get_mean_distance_with_unconnected(&self, unconnected: bool) -> Option<f64> {
        if self.num_nodes() == 0 {
            return None;
        }

        let mut total_distance: usize = 0;
        let mut path_count: usize = 0;

        // Compute shortest paths from each node using BFS
        for &start_node in self.nodes.keys() {
            let distances = self.bfs_distances(start_node);

            // If unconnected=false and graph is disconnected, return None
            if !unconnected && distances.len() != self.num_nodes() {
                return None;
            }

            // Sum all distances (excluding the distance to self, which is 0)
            for (&target, &dist) in distances.iter() {
                if target != start_node {
                    total_distance += dist;
                    path_count += 1;
                }
            }
        }

        if path_count == 0 {
            None
        } else {
            Some(total_distance as f64 * (1.0 / path_count as f64))
        }
    }

    /// Get girth (length of shortest cycle)
    /// Returns None if graph is acyclic
    /// Uses BFS from each node to find shortest cycle, with early termination
    #[inline]
    fn get_girth(&self) -> Option<usize> {
        let n = self.num_nodes();
        if n == 0 {
            return None;
        }

        // Fast path 1: Check for self-loops (girth = 1)
        // This is O(E) but typically very fast due to early termination
        for edge in &self.edges {
            if !self.deleted_edge_ids.contains(&edge.id) && edge.source == edge.target {
                return Some(1);
            }
        }

        // Fast path 2: Check for multi-edges (girth = 2)
        // In undirected graphs, parallel edges create a 2-cycle
        if !self.directed {
            let mut seen_pairs: FxHashSet<(u64, u64)> = FxHashSet::default();
            for edge in &self.edges {
                if self.deleted_edge_ids.contains(&edge.id) {
                    continue;
                }
                // Normalize pair (smaller, larger) for undirected comparison
                let pair = if edge.source < edge.target {
                    (edge.source, edge.target)
                } else {
                    (edge.target, edge.source)
                };
                if !seen_pairs.insert(pair) {
                    return Some(2); // Found duplicate edge = 2-cycle
                }
            }
        }

        let mut min_cycle: Option<usize> = None;
        let mut visited_global = FxHashSet::default();

        // Try BFS from each node to find shortest cycle
        for &start_node in self.nodes.keys() {
            // Early termination: if we found a cycle of length k,
            // we only need to search up to distance k/2 from unvisited nodes
            if let Some(current_min) = min_cycle {
                if visited_global.contains(&start_node) {
                    continue; // Already part of a discovered component
                }
                // For very small cycles, we can stop early
                if current_min == 3 {
                    break; // Can't get smaller than a triangle
                }
            }

            let mut distances: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let mut parent: FxHashMap<u64, Option<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let mut queue = std::collections::VecDeque::with_capacity(n);
            let mut found_cycle_from_start = false;

            distances.insert(start_node, 0);
            parent.insert(start_node, None);
            queue.push_back(start_node);

            while let Some(current) = queue.pop_front() {
                let current_dist = *distances.get(&current).unwrap();
                let current_parent = *parent.get(&current).unwrap();

                visited_global.insert(current);

                // Early termination: if we already found a smaller cycle, stop this BFS
                if let Some(current_min) = min_cycle {
                    if current_dist * 2 >= current_min {
                        break;
                    }
                }

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for &neighbor in neighbors {
                        if let Some(&neighbor_dist) = distances.get(&neighbor) {
                            // Already visited - potential cycle
                            // In undirected graphs, skip if neighbor is parent (back edge)
                            if (!self.directed || Some(neighbor) != current_parent)
                                && Some(neighbor) != current_parent {
                                    // Found a cycle
                                    let cycle_length = current_dist + neighbor_dist + 1;
                                    min_cycle = Some(match min_cycle {
                                        None => cycle_length,
                                        Some(prev) => prev.min(cycle_length),
                                    });
                                    found_cycle_from_start = true;
                                    // If we found a triangle, we can stop immediately
                                    if cycle_length == 3 {
                                        return Some(3);
                                    }
                                }
                        } else {
                            // Early termination within BFS
                            if let Some(current_min) = min_cycle {
                                // Don't explore beyond distance that could improve result
                                if (current_dist + 1) * 2 >= current_min {
                                    continue;
                                }
                            }
                            distances.insert(neighbor, current_dist + 1);
                            parent.insert(neighbor, Some(current));
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            // If we didn't find any cycle from this node and it's a new component,
            // we might be in a tree - mark all reachable nodes as visited
            if !found_cycle_from_start {
                for node in distances.keys() {
                    visited_global.insert(*node);
                }
            }
        }

        min_cycle
    }

    /// Check if the graph is a tree
    ///
    /// A tree is a connected acyclic graph. For undirected graphs, this means:
    /// - Connected (single component)
    /// - No cycles
    /// - Has exactly n-1 edges where n is number of nodes
    ///
    /// For directed graphs, use is_arborescence() instead.
    ///
    /// # Returns
    /// true if graph is a tree, false otherwise
    #[inline]
    fn is_tree(&self) -> bool {
        let n = self.num_nodes();

        // Directed graphs should use is_arborescence(), whatever their size
        if self.directed {
            return false;
        }

        // Empty graph or single node is considered a tree, unless the node has a self-loop
        if n == 0 || n == 1 {
            return self.num_edges() == 0;
        }

        // A tree with n nodes has exactly n-1 edges
        if self.num_edges() != n - 1 {
            return false;
        }

        // Must be connected
        if self.count_weakly_connected_components() != 1 {
            return false;
        }

        // Must be acyclic (if connected and has n-1 edges, it's automatically acyclic)
        // But we can verify by checking for cycles
        true
    }

    /// Check if the graph is a forest
    ///
    /// A forest is a disjoint union of trees (acyclic graph).
    /// For a graph with n nodes, k components, and e edges:
    /// - Forest condition: e = n - k
    ///
    /// For directed graphs, checks if it's a collection of arborescences.
    ///
    /// # Returns
    /// true if graph is a forest, false otherwise
    #[inline]
    fn is_forest(&self) -> bool {
        let n = self.num_nodes();

        // Empty graph is a forest
        if n == 0 {
            return true;
        }

        // For directed graphs, not implemented (would need to check each component is arborescence)
        if self.directed {
            return false;
        }

        let k = self.count_weakly_connected_components();
        let e = self.num_edges();

        // Forest condition: e = n - k
        // Each tree with n_i nodes has n_i - 1 edges
        // Sum over all trees: sum(n_i - 1) = sum(n_i) - k = n - k
        e == n - k
    }

    /// Check if the directed graph is an arborescence
    ///
    /// An arborescence (directed tree) is a directed graph where:
    /// - Has exactly one node with in-degree 0 (the root)
    /// - All other nodes have in-degree 1
    /// - The graph is connected (when viewed as undirected)
    /// - Has exactly n-1 edges
    ///
    /// For undirected graphs, use is_tree() instead.
    ///
    /// # Returns
    /// true if graph is an arborescence, false otherwise
    #[inline]
    fn is_arborescence(&self) -> bool {
        let n = self.num_nodes();

        // Empty graph or single node is an arborescence
        if n == 0 || n == 1 {
            return true;
        }

        // Only for directed graphs
        if !self.directed {
            return false;
        }

        // Must have exactly n-1 edges
        if self.num_edges() != n - 1 {
            return false;
        }

        // Check in-degrees
        let mut root_count = 0;
        for &node_id in self.nodes.keys() {
            let in_degree = self.in_degree(node_id);

            if in_degree == 0 {
                root_count += 1;
                if root_count > 1 {
                    return false; // Multiple roots
                }
            } else if in_degree != 1 {
                return false; // Node has in-degree > 1
            }
        }

        // Must have exactly one root
        if root_count != 1 {
            return false;
        }

        // Must be weakly connected
        if self.count_weakly_connected_components() != 1 {
            return false;
        }

        true
    }

    /// Check if the graph is planar using Euler's formula.
    /// A planar graph can be drawn on a plane without edge crossings.
    ///
    /// Uses necessary conditions:
    /// - For simple planar graphs: e ≤ 3n - 6 (n ≥ 3)
    /// - For simple planar graphs without triangles: e ≤ 2n - 4
    ///
    /// Note: This is a necessary but not sufficient test. Some non-planar
    /// graphs may pass this test. For a complete test, use specialized
    /// planarity testing algorithms (Kuratowski's theorem).
    ///
    /// # Returns
    /// true if graph satisfies planarity conditions, false otherwise
    #[inline]
    fn is_planar(&self) -> bool {
        let n = self.num_nodes();
        let e = self.num_edges();

        // Empty graph, single node, or very small graphs are planar
        if n <= 2 {
            return true;
        }

        // Must be simple (no self-loops or multi-edges)
        if !self.is_simple() {
            return false;
        }

        // Euler's formula for planar graphs: e ≤ 3n - 6
        if e > 3 * n - 6 {
            return false;
        }

        // If no triangles exist, stricter bound: e ≤ 2n - 4
        // Check for triangles by looking for 3-cycles
        let has_triangles = self.has_triangles();

        if !has_triangles && e > 2 * n - 4 {
            return false;
        }

        // Passed necessary conditions (but not sufficient)
        true
    }

    /// Helper method to check if graph has any triangles (3-cycles)
    fn has_triangles(&self) -> bool {
        for (&node, neighbors) in &self.adjacency_list {
            for &neighbor in neighbors {
                if let Some(neighbor_neighbors) = self.adjacency_list.get(&neighbor) {
                    for &third in neighbor_neighbors {
                        if third != node && self.adjacency_list.get(&third)
                            .is_some_and(|thirds| thirds.contains(&node))
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Check if this graph is isomorphic to another graph.
    /// Two graphs are isomorphic if there exists a bijection between their
    /// nodes that preserves adjacency.
    ///
    /// This implementation uses a simple necessary conditions check:
    /// - Same number of nodes
    /// - Same number of edges
    /// - Same degree sequence
    /// - Same directedness
    ///
    /// Note: These are necessary but not sufficient conditions. This is a
    /// heuristic check that will correctly identify *non-isomorphic* graphs
    /// but may give false positives. A complete isomorphism test requires
    /// algorithms like VF2.
    ///
    /// # Arguments
    /// * `other` - Another graph to compare with
    ///
    /// # Returns
    /// true if graphs satisfy isomorphism conditions, false otherwise
    fn is_isomorphic(&self, other: &Graph) -> bool {
        // Must have same directedness
        if self.directed != other.directed {
            return false;
        }

        // Must have same number of nodes
        if self.num_nodes() != other.num_nodes() {
            return false;
        }

        // Must have same number of edges
        if self.num_edges() != other.num_edges() {
            return false;
        }

        // Must have same degree sequence
        let mut degrees1: Vec<usize> = self.nodes.keys()
            .map(|&node_id| self.degree(node_id))
            .collect();
        degrees1.sort_unstable();

        let mut degrees2: Vec<usize> = other.nodes.keys()
            .map(|&node_id| other.degree(node_id))
            .collect();
        degrees2.sort_unstable();

        if degrees1 != degrees2 {
            return false;
        }

        // Passed necessary conditions
        true
    }

    /// Find all maximal cliques in the graph using the Bron-Kerbosch algorithm.
    /// A clique is a subset of nodes where every pair is connected.
    /// A maximal clique cannot be extended by adding another node.
    ///
    /// Only works on undirected graphs.
    ///
    /// # Returns
    /// Vector of cliques, where each clique is a vector of node IDs
    fn find_cliques(&self) -> PyResult<Vec<Vec<u64>>> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "find_cliques() only works on undirected graphs"
            ));
        }

        let mut cliques = Vec::new();
        let mut r = FxHashSet::default(); // Current clique being built
        let mut p: FxHashSet<u64> = self.nodes.keys().copied().collect(); // Candidate nodes
        let mut x = FxHashSet::default(); // Already processed nodes

        self.bron_kerbosch_internal(&mut r, &mut p, &mut x, &mut cliques);

        Ok(cliques)
    }

    /// Find a maximum clique (largest clique) in the graph.
    /// Returns one of potentially multiple maximum cliques.
    ///
    /// Only works on undirected graphs.
    ///
    /// # Returns
    /// Vector of node IDs forming a maximum clique, or empty vector if graph is empty
    fn max_clique(&self) -> PyResult<Vec<u64>> {
        let cliques = self.find_cliques()?;

        Ok(cliques
            .into_iter()
            .max_by_key(|clique| clique.len())
            .unwrap_or_default())
    }

    /// Get the clique number (size of the maximum clique).
    ///
    /// Only works on undirected graphs.
    ///
    /// # Returns
    /// Size of the maximum clique (0 if graph is empty)
    fn clique_number(&self) -> PyResult<usize> {
        let max_clique = self.max_clique()?;
        Ok(max_clique.len())
    }

    /// Breadth-first search (BFS) traversal from a start node
    /// Returns list of node IDs in BFS order
    ///
    /// # Arguments
    /// * `start` - Starting node ID
    ///
    /// # Returns
    /// Vector of node IDs visited in BFS order
    #[inline]
    fn bfs(&self, start: u64) -> PyResult<Vec<u64>> {
        if !self.nodes.contains_key(&start) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node {} not found in graph", start)
            ));
        }

        let mut visited = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();

        visited.insert(start);
        queue.push_back(start);
        result.push(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                        result.push(neighbor);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Depth-first search (DFS) traversal from a start node
    /// Returns list of node IDs in DFS order
    ///
    /// # Arguments
    /// * `start` - Starting node ID
    ///
    /// # Returns
    /// Vector of node IDs visited in DFS order
    #[inline]
    fn dfs(&self, start: u64) -> PyResult<Vec<u64>> {
        if !self.nodes.contains_key(&start) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Node {} not found in graph", start)
            ));
        }

        let mut visited = FxHashSet::default();
        let mut stack = vec![start];
        let mut result = Vec::new();

        while let Some(current) = stack.pop() {
            if visited.insert(current) {
                result.push(current);

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    // Push neighbors in reverse order to maintain left-to-right traversal
                    for &neighbor in neighbors.iter().rev() {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get adhesion: the edge connectivity of an undirected graph (the minimum
    /// number of edges whose removal disconnects it).
    #[inline]
    fn get_adhesion(&self) -> PyResult<usize> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Adhesion currently only implemented for undirected graphs",
            ));
        }
        self.edge_connectivity()
    }

    /// Get cohesion: the node connectivity of an undirected graph (the minimum
    /// number of nodes whose removal disconnects it).
    #[inline]
    fn get_cohesion(&self) -> PyResult<usize> {
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cohesion currently only implemented for undirected graphs",
            ));
        }
        self.node_connectivity()
    }

    /// Compute node connectivity between two nodes
    ///
    /// Node connectivity is the minimum number of nodes that must be removed
    /// to disconnect source from target. Uses Menger's theorem: this equals
    /// the maximum number of node-disjoint paths between the nodes.
    ///
    /// Implementation: Transform graph by splitting each node v into v_in and v_out
    /// with edge (v_in, v_out) of capacity 1. This converts node connectivity to
    /// edge connectivity, which we compute using max-flow.
    pub fn node_connectivity_between(&self, source: u64, target: u64) -> PyResult<usize> {
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} does not exist", source),
            ));
        }
        if !self.nodes.contains_key(&target) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node {} does not exist", target),
            ));
        }
        if source == target {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Source and target must be different nodes",
            ));
        }

        // Special case: if not connected, connectivity is 0
        if !self.is_connected() {
            // Check if source and target are in same component
            let components = self.find_connected_components();
            let source_comp = components.get(&source);
            let target_comp = components.get(&target);

            if source_comp != target_comp {
                return Ok(0);
            }
        }

        // Build transformed graph for node connectivity.
        // Every node v becomes v_in (v*2) and v_out (v*2+1). Internal nodes get a
        // capacity-1 edge v_in -> v_out (removing v costs 1); source and target get
        // none, and the flow runs from source_out to target_in. Splitting *every*
        // node keeps all transformed ids in one space: leaving source/target as raw
        // ids let them collide with another node's split id (e.g. node 6 vs. 3*2).
        let n = self.num_nodes();
        let estimated_capacity = n + self.edges.len() * 2;
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::with_capacity_and_hasher(estimated_capacity, Default::default());

        for &node in self.nodes.keys() {
            if node != source && node != target {
                capacity.insert((node * 2, node * 2 + 1), 1);
            }
        }

        // Add edges from original graph: u_out -> v_in (and v_out -> u_in if undirected)
        for edge in &self.edges {
            let (s, t) = (edge.source, edge.target);
            *capacity.entry((s * 2 + 1, t * 2)).or_insert(0) += 1;
            if !self.directed {
                *capacity.entry((t * 2 + 1, s * 2)).or_insert(0) += 1;
            }
        }
        let (flow_source, flow_sink) = (source * 2 + 1, target * 2);

        // Run max-flow from source to target on transformed graph
        let mut residual = capacity.clone();
        let mut max_flow = 0;

        loop {
            // Find augmenting path using BFS
            let path = self.bfs_augmenting_path_custom(flow_source, flow_sink, &residual);

            if path.is_empty() {
                break;
            }

            // Find minimum capacity along the path
            let mut path_flow = usize::MAX;
            for i in 0..path.len() - 1 {
                let edge_key = (path[i], path[i + 1]);
                let cap = residual.get(&edge_key).copied().unwrap_or(0);
                path_flow = path_flow.min(cap);
            }

            // Update residual capacities
            for i in 0..path.len() - 1 {
                let forward_key = (path[i], path[i + 1]);
                let backward_key = (path[i + 1], path[i]);

                *residual.entry(forward_key).or_insert(0) -= path_flow;
                *residual.entry(backward_key).or_insert(0) += path_flow;
            }

            max_flow += path_flow;
        }

        Ok(max_flow)
    }

    /// Compute global node connectivity
    ///
    /// Returns the minimum number of nodes whose removal disconnects the graph.
    /// This is computed as the minimum pairwise node connectivity over all pairs
    /// of non-adjacent nodes.
    pub fn node_connectivity(&self) -> PyResult<usize> {
        let n = self.num_nodes();

        if n == 0 {
            return Ok(0);
        }

        if n == 1 {
            return Ok(0);
        }

        // If disconnected, connectivity is 0
        if !self.is_connected() {
            return Ok(0);
        }

        // If complete graph, connectivity is n - 1
        let num_edges = self.num_edges();
        let expected_complete = n * (n - 1) / 2;
        if !self.directed && num_edges == expected_complete {
            return Ok(n - 1);
        }

        // Optimization: connectivity <= min_degree
        let min_degree = self.nodes.keys()
            .map(|&id| self.degree(id))
            .min()
            .unwrap_or(0);

        if min_degree == 0 {
            return Ok(0);
        }

        if min_degree == 1 {
            return Ok(1);
        }

        // Compute minimum over all non-adjacent pairs
        let mut min_connectivity = usize::MAX;
        let mut nodes: Vec<u64> = Vec::with_capacity(n);
        nodes.extend(self.nodes.keys().copied());
        nodes.sort(); // Ensure consistent ordering

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let u = nodes[i];
                let v = nodes[j];

                // Only check non-adjacent pairs (adjacent pairs have higher connectivity)
                let are_adjacent = self.edges.iter().any(|e| {
                    (e.source == u && e.target == v) || (e.source == v && e.target == u)
                });

                if !are_adjacent {
                    let conn = self.node_connectivity_between(u, v)?;
                    min_connectivity = min_connectivity.min(conn);

                    // Early termination if we found connectivity 1
                    if min_connectivity == 1 {
                        return Ok(1);
                    }
                }
            }
        }

        // If all pairs are adjacent (complete graph), connectivity is n-1
        if min_connectivity == usize::MAX {
            return Ok(n - 1);
        }

        Ok(min_connectivity)
    }

    /// Find cohesive blocks in the graph
    ///
    /// A k-cohesive block is a maximal subgraph where at least k nodes must be removed
    /// to disconnect it. Returns hierarchical structure where child blocks have higher
    /// cohesion than their parents.
    ///
    /// Based on Moody & White (2003) "Structural Cohesion and Embeddedness"
    ///
    /// Returns:
    ///   - blocks: Vec of node sets (each block as list of labels)
    ///   - cohesion: Vec of k-values (cohesion of each block)
    ///   - parent: Vec of parent block indices (None for root blocks)
    pub fn cohesive_blocks(&self, py: Python) -> PyResult<PyObject> {
        let n = self.num_nodes();

        // Handle empty or trivial graphs
        if n == 0 {
            return Ok((vec![] as Vec<Vec<u64>>, vec![] as Vec<usize>, vec![] as Vec<Option<usize>>).into_pyobject(py)?.into_any().unbind());
        }

        if n == 1 {
            let node_id = *self.nodes.keys().next().unwrap();
            return Ok((vec![vec![node_id]], vec![0_usize], vec![None::<usize>]).into_pyobject(py)?.into_any().unbind());
        }

        // If graph is disconnected, process each component separately
        if !self.is_connected() {
            let component_map = self.find_connected_components();
            let mut all_blocks = Vec::new();
            let mut all_cohesions = Vec::new();
            let mut all_parents = Vec::new();

            // Group nodes by component ID
            let mut comp_groups: FxHashMap<usize, Vec<u64>> = FxHashMap::with_capacity_and_hasher(component_map.len(), Default::default());
            for (&node, &comp_id) in &component_map {
                comp_groups.entry(comp_id).or_default().push(node);
            }

            for comp_nodes in comp_groups.values() {
                if comp_nodes.len() > 1 {
                    // Process each component recursively
                    // Use a separate blocks vector for each component
                    let mut comp_all_blocks = Vec::new();
                    let (comp_blocks, comp_cohesions, comp_parents) =
                        self.cohesive_blocks_for_subgraph(comp_nodes, None, &mut comp_all_blocks)?;

                    // Add the returned blocks (not comp_all_blocks, as it's already populated)
                    all_blocks.extend(comp_blocks);
                    all_cohesions.extend(comp_cohesions);
                    all_parents.extend(comp_parents);
                } else if comp_nodes.len() == 1 {
                    // Single node component
                    all_blocks.push(comp_nodes.clone());
                    all_cohesions.push(0);
                    all_parents.push(None);
                }
            }

            // Return node IDs directly (no label conversion)
            return Ok((all_blocks, all_cohesions, all_parents).into_pyobject(py)?.into_any().unbind());
        }

        // Process connected graph
        let all_nodes: Vec<u64> = self.nodes.keys().copied().collect();
        let mut all_blocks = Vec::new();
        let (blocks, cohesions, parents) =
            self.cohesive_blocks_for_subgraph(&all_nodes, None, &mut all_blocks)?;

        // Return node IDs directly (no label conversion)
        Ok((blocks, cohesions, parents).into_pyobject(py)?.into_any().unbind())
    }

    /// Get eccentricity for each node
    /// Eccentricity is the maximum distance from a node to any other node
    /// Returns HashMap mapping node ID to eccentricity (None if disconnected)
    #[inline]
    fn get_eccentricity(&self) -> FxHashMap<u64, Option<usize>> {
        self.compute_eccentricities()
    }

    /// Get radiality centrality for each node
    /// Radiality = (diameter + 1 - eccentricity) / (n - 1)
    /// Measures how central a node is based on eccentricity
    /// Returns HashMap mapping node ID to radiality (None if graph disconnected)
    #[inline]
    fn get_radiality(&self) -> FxHashMap<u64, Option<f64>> {
        let n = self.num_nodes();
        if n <= 1 {
            return self.nodes.keys()
                .map(|&id| (id, Some(0.0)))
                .collect();
        }

        // Eccentricities are measured within each component, so radiality (which compares
        // them against one graph-wide diameter) is only meaningful for connected graphs
        if self.count_weakly_connected_components() != 1 {
            return self.nodes.keys().map(|&id| (id, None)).collect();
        }

        let eccentricities = self.compute_eccentricities();
        let diameter = self.get_diameter();

        match diameter {
            Some(d) => {
                let inv_denom = 1.0 / (n - 1) as f64;
                eccentricities
                    .into_iter()
                    .map(|(node, ecc)| {
                        let radiality = ecc.map(|e| (d + 1 - e) as f64 * inv_denom);
                        (node, radiality)
                    })
                    .collect()
            }
            None => {
                // Graph is disconnected
                eccentricities.into_keys().map(|node| (node, None))
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
    #[inline]
    fn get_closeness_vitality(&self) -> FxHashMap<u64, f64> {
        use rayon::prelude::*;

        let n = self.num_nodes();
        if n == 0 {
            return FxHashMap::default();
        }

        // Compute total distances in original graph (parallelized)
        let original_total = self.compute_total_distances_parallel();

        // Collect node IDs to iterate over
        let mut node_ids = Vec::with_capacity(n);
        node_ids.extend(self.nodes.keys().copied());

        // Compute vitality for each node in parallel
        let vitality_vec: Vec<(u64, f64)> = node_ids
            .par_iter()
            .map(|&node| {
                // Compute total distances without this node
                let total_without = self.compute_total_distances_without_node(node);

                // Vitality is the difference (higher = more important)
                let vital = original_total - total_without;
                (node, vital)
            })
            .collect();

        // Convert to HashMap with preallocated capacity
        let mut result = FxHashMap::with_capacity_and_hasher(n, Default::default());
        result.extend(vitality_vec);
        result
    }

    /// Get reciprocity (fraction of reciprocal edges in a directed graph)
    /// Reciprocity is the fraction of edges that have a reverse edge
    /// Only defined for directed graphs
    /// Returns a value between 0.0 (no reciprocal edges) and 1.0 (all edges reciprocal)
    #[inline]
    fn get_reciprocity(&self) -> PyResult<f64> {
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Reciprocity only defined for directed graphs",
            ));
        }

        if self.edges.is_empty() {
            return Ok(0.0);
        }

        // Count reciprocal edges
        let mut reciprocal_count = 0;

        // Build a set of edges for fast lookup
        let m = self.num_edges();
        let mut edge_set = FxHashSet::with_capacity_and_hasher(m, Default::default());
        edge_set.extend(self.edges.iter().map(|e| (e.source, e.target)));

        // Check each edge for a reverse edge
        for edge in &self.edges {
            // Skip self-loops
            if edge.source == edge.target {
                continue;
            }

            // Check if reverse edge exists
            if edge_set.contains(&(edge.target, edge.source)) {
                reciprocal_count += 1;
            }
        }

        // Each reciprocal pair is counted twice (once from each direction)
        // So we divide by 2 to get the number of reciprocal pairs
        // Then divide by total edges to get the fraction
        let inv_edges = 1.0 / self.edges.len() as f64;
        let reciprocity = (reciprocal_count as f64) * inv_edges;

        Ok(reciprocity)
    }

    /// Count triangles in the graph
    /// Returns a Python dictionary mapping node labels to triangle counts
    ///
    /// For undirected graphs, counts the number of triangles that include each node.
    /// For directed graphs, counts directed triangles (closed triplets).
    ///
    /// # Time Complexity
    /// O(m * d_max) where m is the number of edges and d_max is the maximum degree
    ///
    /// # Returns
    /// Dictionary mapping node labels (or IDs as strings) to triangle counts
    ///
    /// # Example
    /// ```python
    /// import edgecase as ec
    /// g = ec.Graph()
    /// g.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
    /// counts = g.count_triangles()
    /// # Returns: {node_id_a: 1, node_id_b: 1, node_id_c: 1}
    /// ```
    pub fn count_triangles(&self, py: Python) -> PyResult<Py<PyDict>> {
        let triangle_counts = crate::algorithms::structural::count_triangles(self);

        // Convert to Python dict with node IDs
        let result = PyDict::new(py);
        for (node_id, count) in triangle_counts {
            result.set_item(node_id, count)?;
        }

        Ok(result.into())
    }

    /// List all triangles in the graph
    /// Returns a list of triangles, where each triangle is a tuple of three node labels
    ///
    /// For undirected graphs, returns unique triangles.
    /// For directed graphs, returns directed triangles (closed triplets).
    ///
    /// # Time Complexity
    /// O(m * d_max) where m is the number of edges and d_max is the maximum degree
    ///
    /// # Returns
    /// List of tuples (node_id1, node_id2, node_id3) representing triangles
    ///
    /// # Example
    /// ```python
    /// import edgecase as ec
    /// g = ec.Graph()
    /// g.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
    /// triangles = g.list_triangles()
    /// # Returns: [(a_id, b_id, c_id)] where a_id, b_id, c_id are node IDs
    /// ```
    pub fn list_triangles(&self, py: Python) -> PyResult<Py<PyList>> {
        let triangles = crate::algorithms::structural::list_triangles(self);

        // Convert to Python list with node IDs
        let result = PyList::empty(py);
        for (n1, n2, n3) in triangles {
            let tuple = PyTuple::new(py, [n1, n2, n3])?;
            result.append(tuple)?;
        }

        Ok(result.into())
    }

    /// Compute global transitivity (clustering coefficient)
    /// Transitivity = 3 * (number of triangles) / (number of connected triples)
    ///
    /// For undirected graphs, a connected triple is a set of three nodes where
    /// at least two edges connect them (a path of length 2).
    ///
    /// Returns None if there are no connected triples in the graph.
    ///
    /// # Time Complexity
    /// O(m * d_max) where m is the number of edges and d_max is the maximum degree
    ///
    /// # Returns
    /// Global clustering coefficient as a float, or None if undefined
    ///
    /// # Example
    /// ```python
    /// import edgecase as ec
    /// g = ec.Graph()
    /// g.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
    /// trans = g.transitivity()
    /// # Returns: 1.0 (perfect triangle)
    /// ```
    pub fn transitivity(&self) -> Option<f64> {
        crate::algorithms::structural::transitivity(self)
    }

    /// Compute local clustering coefficient for each node
    /// For undirected graphs: C(v) = 2 * triangles(v) / (degree(v) * (degree(v) - 1))
    /// For directed graphs: C(v) = triangles(v) / (out_degree(v) * (out_degree(v) - 1))
    ///
    /// Returns None for nodes with degree < 2 (no triangles possible)
    ///
    /// # Time Complexity
    /// O(m * d_max) where m is the number of edges and d_max is the maximum degree
    ///
    /// # Returns
    /// Dictionary mapping node labels to clustering coefficients (or None)
    ///
    /// # Example
    /// ```python
    /// import edgecase as ec
    /// g = ec.Graph()
    /// g.add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")])
    /// clustering = g.local_clustering_coefficient()
    /// # Returns clustering coefficient for each node (by node ID)
    /// ```
    pub fn local_clustering_coefficient(&self, py: Python) -> PyResult<Py<PyDict>> {
        let clustering = crate::algorithms::structural::local_clustering_coefficient(self);

        // Convert to Python dict with node IDs
        let result = PyDict::new(py);
        for (node_id, coeff_opt) in clustering {
            match coeff_opt {
                Some(coeff) => result.set_item(node_id, coeff)?,
                None => result.set_item(node_id, py.None())?,
            }
        }

        Ok(result.into())
    }

    /// Get minimum cut between two nodes using Ford-Fulkerson with BFS (Edmonds-Karp)
    /// Returns the size of the minimum cut that separates source from sink
    /// This is equivalent to the maximum flow from source to sink
    fn get_min_cut_between(&self, source: u64, sink: u64) -> PyResult<usize> {
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} does not exist", source),
            ));
        }
        if !self.nodes.contains_key(&sink) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Sink node {} does not exist", sink),
            ));
        }
        if source == sink {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Source and sink must be different nodes",
            ));
        }

        // Build capacity matrix (using adjacency list representation for efficiency)
        // For unweighted graphs, each edge has capacity 1
        let estimated_size = if self.directed { self.edges.len() } else { self.edges.len() * 2 };
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::with_capacity_and_hasher(estimated_size, Default::default());

        for edge in &self.edges {
            let key = (edge.source, edge.target);
            *capacity.entry(key).or_insert(0) += 1;

            // For undirected graphs, add reverse edge
            if !self.directed {
                let rev_key = (edge.target, edge.source);
                *capacity.entry(rev_key).or_insert(0) += 1;
            }
        }

        // Initialize residual capacity = capacity
        let mut residual = capacity.clone();

        // Ford-Fulkerson with BFS (Edmonds-Karp algorithm)
        let mut max_flow = 0;

        loop {
            // Find augmenting path using BFS
            let path = self.bfs_augmenting_path(source, sink, &residual);

            if path.is_empty() {
                break; // No more augmenting paths
            }

            // Find minimum capacity along the path
            let mut path_flow = usize::MAX;
            for i in 0..path.len() - 1 {
                let edge_key = (path[i], path[i + 1]);
                let cap = residual.get(&edge_key).copied().unwrap_or(0);
                path_flow = path_flow.min(cap);
            }

            // Update residual capacities
            for i in 0..path.len() - 1 {
                let forward_key = (path[i], path[i + 1]);
                let backward_key = (path[i + 1], path[i]);

                // Decrease forward capacity
                *residual.entry(forward_key).or_insert(0) -= path_flow;

                // Increase backward capacity (for flow reversal)
                *residual.entry(backward_key).or_insert(0) += path_flow;
            }

            max_flow += path_flow;
        }

        Ok(max_flow)
    }

    /// Get minimum edge cut between two nodes
    /// Returns the set of edges (as tuples) that form the minimum cut separating source from sink
    /// Uses Ford-Fulkerson algorithm and then identifies the cut edges
    pub fn minimum_edge_cut<'py>(&self, py: Python<'py>, source: u64, sink: u64) -> PyResult<Bound<'py, PySet>> {
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} does not exist", source),
            ));
        }
        if !self.nodes.contains_key(&sink) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Sink node {} does not exist", sink),
            ));
        }
        if source == sink {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Source and sink must be different nodes",
            ));
        }

        // Build capacity matrix
        let estimated_size = if self.directed { self.edges.len() } else { self.edges.len() * 2 };
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::with_capacity_and_hasher(estimated_size, Default::default());

        for edge in &self.edges {
            let key = (edge.source, edge.target);
            *capacity.entry(key).or_insert(0) += 1;

            if !self.directed {
                let rev_key = (edge.target, edge.source);
                *capacity.entry(rev_key).or_insert(0) += 1;
            }
        }

        // Run max flow algorithm
        let mut residual = capacity.clone();

        loop {
            let path = self.bfs_augmenting_path(source, sink, &residual);
            if path.is_empty() {
                break;
            }

            let mut path_flow = usize::MAX;
            for i in 0..path.len() - 1 {
                let edge_key = (path[i], path[i + 1]);
                let cap = residual.get(&edge_key).copied().unwrap_or(0);
                path_flow = path_flow.min(cap);
            }

            for i in 0..path.len() - 1 {
                let forward_key = (path[i], path[i + 1]);
                let backward_key = (path[i + 1], path[i]);
                *residual.entry(forward_key).or_insert(0) -= path_flow;
                *residual.entry(backward_key).or_insert(0) += path_flow;
            }
        }

        // Find reachable nodes from source in residual graph
        let mut reachable = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);
        reachable.insert(source);

        while let Some(current) = queue.pop_front() {
            for &other_node in self.nodes.keys() {
                let edge_key = (current, other_node);
                let cap = residual.get(&edge_key).copied().unwrap_or(0);
                if cap > 0 && !reachable.contains(&other_node) {
                    reachable.insert(other_node);
                    queue.push_back(other_node);
                }
            }
        }

        // Find cut edges: edges from reachable to non-reachable nodes
        let result = PySet::empty(py)?;
        for edge in &self.edges {
            let in_source_side = reachable.contains(&edge.source);
            let in_sink_side = reachable.contains(&edge.target);

            // For undirected graphs, we need to check both directions
            if !self.directed {
                if (in_source_side && !in_sink_side) || (!in_source_side && in_sink_side) {
                    let edge_tuple = (edge.source, edge.target);
                    result.add(edge_tuple)?;
                }
            } else {
                // For directed graphs, only edges going from source side to sink side
                if in_source_side && !in_sink_side {
                    let edge_tuple = (edge.source, edge.target);
                    result.add(edge_tuple)?;
                }
            }
        }

        Ok(result)
    }

    /// Get minimum node cut between two nodes
    /// Returns the set of nodes whose removal disconnects source from sink
    /// Uses transformation to edge connectivity problem
    pub fn minimum_node_cut<'py>(&self, py: Python<'py>, source: u64, sink: u64) -> PyResult<Bound<'py, PySet>> {
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} does not exist", source),
            ));
        }
        if !self.nodes.contains_key(&sink) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Sink node {} does not exist", sink),
            ));
        }
        if source == sink {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Source and sink must be different nodes",
            ));
        }
        // A direct source -> sink edge (either orientation when undirected) can't be cut by
        // removing other nodes. It would also carry unbounded flow and corrupt the residual
        if !self.find_edge_ids(source, sink).is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Source {} and sink {} are adjacent, so no node cut separates them",
                    source, sink
                ),
            ));
        }

        // Node connectivity is computed by transforming to edge connectivity:
        // Split each node v into v_in and v_out with an edge of capacity 1
        // This ensures removing a node requires removing its internal edge

        // Create node mapping: original_node -> (in_node, out_node)
        // Use high IDs to avoid collisions: in = id * 2, out = id * 2 + 1
        let max_id = self.nodes.keys().max().copied().unwrap_or(0);
        let offset = max_id + 1;

        // Build capacity matrix for transformed graph
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::default();

        // Add internal edges (node capacity = 1)
        // Exception: source and sink have infinite capacity (don't count them in the cut)
        for &node_id in self.nodes.keys() {
            let in_node = node_id + offset;
            let out_node = node_id + offset * 2;

            if node_id != source && node_id != sink {
                capacity.insert((in_node, out_node), 1);
            } else {
                // Source and sink should not be part of the cut
                capacity.insert((in_node, out_node), usize::MAX / 2);
            }
        }

        // Add external edges (infinite capacity, as we only cut nodes, not edges)
        for edge in &self.edges {
            let from_out = edge.source + offset * 2;
            let to_in = edge.target + offset;

            capacity.insert((from_out, to_in), usize::MAX / 2);

            if !self.directed {
                let to_out = edge.target + offset * 2;
                let from_in = edge.source + offset;
                capacity.insert((to_out, from_in), usize::MAX / 2);
            }
        }

        // Run max flow from source_out to sink_in
        let source_transformed = source + offset * 2;
        let sink_transformed = sink + offset;

        let mut residual = capacity.clone();

        loop {
            let path = self.bfs_augmenting_path_custom(source_transformed, sink_transformed, &residual);
            if path.is_empty() {
                break;
            }

            let mut path_flow = usize::MAX;
            for i in 0..path.len() - 1 {
                let edge_key = (path[i], path[i + 1]);
                let cap = residual.get(&edge_key).copied().unwrap_or(0);
                path_flow = path_flow.min(cap);
            }

            for i in 0..path.len() - 1 {
                let forward_key = (path[i], path[i + 1]);
                let backward_key = (path[i + 1], path[i]);
                *residual.entry(forward_key).or_insert(0) -= path_flow;
                *residual.entry(backward_key).or_insert(0) += path_flow;
            }
        }

        // Find reachable nodes from source in residual graph
        let mut reachable = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source_transformed);
        reachable.insert(source_transformed);

        while let Some(current) = queue.pop_front() {
            for (&_from, &to, &_cap) in residual.iter().filter_map(|(k, v)| {
                if k.0 == current && *v > 0 {
                    Some((&k.0, &k.1, v))
                } else {
                    None
                }
            }) {
                if !reachable.contains(&to) {
                    reachable.insert(to);
                    queue.push_back(to);
                }
            }
        }

        // Find nodes whose internal edge was saturated (in reachable but out not reachable)
        let result = PySet::empty(py)?;
        for &node_id in self.nodes.keys() {
            if node_id == source || node_id == sink {
                continue; // Don't include source or sink in the cut
            }

            let in_node = node_id + offset;
            let out_node = node_id + offset * 2;

            // If in_node is reachable but out_node is not, this node is in the cut
            if reachable.contains(&in_node) && !reachable.contains(&out_node) {
                result.add(node_id)?;
            }
        }

        Ok(result)
    }

    /// Compute edge connectivity (minimum edge cut)
    ///
    /// Returns the minimum number of edges whose removal disconnects the graph.
    /// For undirected graphs, this is computed as the minimum over all pairs of nodes.
    /// For directed graphs, uses a similar approach but respects edge directions.
    pub fn edge_connectivity(&self) -> PyResult<usize> {
        let n = self.num_nodes();

        if n == 0 {
            return Ok(0);
        }

        if n == 1 {
            return Ok(0);
        }

        // If disconnected, edge connectivity is 0
        if !self.is_connected() {
            return Ok(0);
        }

        // Optimization: edge connectivity <= min_degree
        let min_degree = self.nodes.keys()
            .map(|&id| self.degree(id))
            .min()
            .unwrap_or(0);

        if min_degree == 0 {
            return Ok(0);
        }

        if min_degree == 1 {
            return Ok(1);
        }

        // For small graphs, check all pairs
        // For larger graphs, use heuristic: check min_degree + 1 random pairs
        let mut min_cut = usize::MAX;

        // Strategy: compute min-cut from node with minimum degree to several other nodes
        // This is much faster than checking all pairs and usually finds the minimum
        let nodes: Vec<u64> = Vec::from_iter(self.nodes.keys().copied());

        if let Some(&source) = nodes.iter().min_by_key(|&&id| self.degree(id)) {
            // Check cuts to several other nodes
            let check_count = (nodes.len()).min(10); // Check up to 10 other nodes

            for &target in nodes.iter().take(check_count) {
                if target != source {
                    let cut = self.get_min_cut_between(source, target)?;
                    min_cut = min_cut.min(cut);

                    // Early termination
                    if min_cut == 1 {
                        return Ok(1);
                    }
                }
            }
        }

        if min_cut == usize::MAX {
            return Ok(0);
        }

        Ok(min_cut)
    }

    /// Perform a random walk on the graph
    ///
    /// Performs a random walk starting from a given vertex and taking a specified
    /// number of steps. Returns the sequence of vertices visited.
    ///
    /// # Arguments
    /// * `start` - ID of the starting vertex
    /// * `steps` - Number of steps to take
    /// * `mode` - Direction to follow: "out" (default), "in", "all"
    /// * `stuck` - What to do if stuck: "return" (default) or "error"
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// Vector of vertex IDs of length steps + 1 (including start vertex), shorter if the walk
    /// gets stuck with `stuck="return"`
    #[pyo3(signature = (start, steps, mode="out", stuck="return", seed=None))]
    pub fn random_walk(
        &self,
        start: u64,
        steps: usize,
        mode: &str,
        stuck: &str,
        seed: Option<u64>,
    ) -> PyResult<Vec<u64>> {
        if !self.nodes.contains_key(&start) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Start node {} does not exist", start)
            ));
        }
        let start_id = start;

        let mut rng: ChaCha8Rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut path = vec![start_id];
        let mut current = start_id;

        for _ in 0..steps {
            // Get neighbors based on mode
            let neighbors = match mode {
                "out" => {
                    // Follow outgoing edges
                    if let Some(neighbors) = self.adjacency_list.get(&current) {
                        neighbors.clone()
                    } else {
                        Vec::new()
                    }
                }
                "in" => {
                    // Follow incoming edges (reverse direction)
                    let mut in_neighbors = Vec::new();
                    for (&node, neighbors) in &self.adjacency_list {
                        if neighbors.contains(&current) {
                            in_neighbors.push(node);
                        }
                    }
                    in_neighbors
                }
                "all" => {
                    // Follow edges in both directions (ignore direction)
                    let mut all_neighbors = FxHashSet::default();

                    // Outgoing
                    if let Some(neighbors) = self.adjacency_list.get(&current) {
                        all_neighbors.extend(neighbors);
                    }

                    // Incoming
                    for (&node, neighbors) in &self.adjacency_list {
                        if neighbors.contains(&current) {
                            all_neighbors.insert(node);
                        }
                    }

                    all_neighbors.into_iter().collect()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Invalid mode: '{}'. Must be 'out', 'in', or 'all'", mode)
                    ));
                }
            };

            if neighbors.is_empty() {
                // Stuck! Handle according to stuck parameter
                match stuck {
                    "return" => break, // Return partial walk
                    "error" => {
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "Random walk got stuck at vertex with no neighbors".to_string()
                        ));
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Invalid stuck: '{}'. Must be 'return' or 'error'", stuck)
                        ));
                    }
                }
            }

            // Choose next vertex uniformly at random
            let next = *neighbors.choose(&mut rng).unwrap();
            path.push(next);
            current = next;
        }

        Ok(path)
    }

    /// Perform a random walk on the graph and return edges
    ///
    /// Like random_walk() but returns the sequence of edges traversed instead
    /// of vertices visited.
    ///
    /// # Arguments
    /// * `start` - ID of the starting vertex
    /// * `steps` - Number of steps to take
    /// * `mode` - Direction to follow: "out" (default), "in", "all"
    /// * `stuck` - What to do if stuck: "return" (default) or "error"
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// Vector of tuples (edge_id, source_label, target_label) of length steps
    #[pyo3(signature = (start, steps, mode="out", stuck="return", seed=None))]
    pub fn random_edge_walk(
        &self,
        start: u64,
        steps: usize,
        mode: &str,
        stuck: &str,
        seed: Option<u64>,
    ) -> PyResult<Vec<(u64, u64, u64)>> {
        // The start is a node ID (Python resolves labels), so digit-only labels aren't mistaken
        // for IDs
        if !self.nodes.contains_key(&start) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Start node {} does not exist",
                start
            )));
        }
        let start_id = start;

        let mut rng: ChaCha8Rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut edges_traversed = Vec::new();
        let mut current = start_id;

        for _ in 0..steps {
            // Get available edges based on mode. An undirected edge is stored once
            // with an arbitrary orientation, so every mode follows any incident edge
            // (otherwise "out" could only walk edges in their insertion direction).
            let available_edges: Vec<&Edge> = match mode {
                "out" | "in" | "all" if !self.directed => {
                    self.edges.iter()
                        .filter(|e| e.source == current || e.target == current)
                        .collect()
                }
                "out" => {
                    // Outgoing edges from current
                    self.edges.iter()
                        .filter(|e| e.source == current)
                        .collect()
                }
                "in" => {
                    // Incoming edges to current (traverse backwards)
                    self.edges.iter()
                        .filter(|e| e.target == current)
                        .collect()
                }
                "all" => {
                    // All edges connected to current
                    self.edges.iter()
                        .filter(|e| e.source == current || e.target == current)
                        .collect()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Invalid mode: '{}'. Must be 'out', 'in', or 'all'", mode)
                    ));
                }
            };

            if available_edges.is_empty() {
                // Stuck! Handle according to stuck parameter
                match stuck {
                    "return" => break, // Return partial walk
                    "error" => {
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "Random walk got stuck at vertex with no outgoing edges".to_string()
                        ));
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Invalid stuck: '{}'. Must be 'return' or 'error'", stuck)
                        ));
                    }
                }
            }

            // Choose next edge uniformly at random
            let edge = *available_edges.choose(&mut rng).unwrap();

            // Determine next vertex based on mode
            let next = match mode {
                _ if !self.directed => {
                    if edge.source == current { edge.target } else { edge.source }
                }
                "out" => edge.target,
                "in" => edge.source,
                "all" => {
                    if edge.source == current {
                        edge.target
                    } else {
                        edge.source
                    }
                }
                _ => unreachable!(),
            };

            // Store edge as (edge_id, source_id, target_id). For undirected graphs the
            // pair is oriented in the direction actually walked (current -> next).
            if self.directed {
                edges_traversed.push((edge.id, edge.source, edge.target));
            } else {
                edges_traversed.push((edge.id, current, next));
            }
            current = next;
        }

        Ok(edges_traversed)
    }

    /// Topological sort using Kahn's algorithm (BFS-based)
    ///
    /// Returns nodes in topological order such that for every directed edge u -> v,
    /// u appears before v in the ordering. Only works on DAGs (directed acyclic graphs).
    ///
    /// Time complexity: O(V + E)
    /// Space complexity: O(V)
    pub fn get_topological_sort_order(&self) -> PyResult<Vec<u64>> {
        // Check if graph is directed
        if !self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Topological sort only works on directed graphs"
            ));
        }

        // Handle empty graph
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.nodes.len();
        let mut node_ids = Vec::with_capacity(n);
        node_ids.extend(self.nodes.keys().copied());

        // Calculate in-degree for each node - preallocate with capacity n
        let mut in_degree = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &id in &node_ids {
            in_degree.insert(id, 0);
        }

        // Count incoming edges for each node
        for edge in &self.edges {
            if !self.deleted_edge_ids.contains(&edge.id) {
                *in_degree.entry(edge.target).or_insert(0) += 1;
            }
        }

        // Queue of nodes with in-degree 0 - preallocate with reasonable capacity
        let mut queue = VecDeque::with_capacity(n);
        for &id in &node_ids {
            if in_degree[&id] == 0 {
                queue.push_back(id);
            }
        }

        let mut result = Vec::with_capacity(n);

        // Process nodes in topological order
        while let Some(node) = queue.pop_front() {
            result.push(node);

            // Get outgoing neighbors from this node
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &target in neighbors {
                    // Decrease in-degree
                    if let Some(degree) = in_degree.get_mut(&target) {
                        *degree -= 1;

                        // If in-degree becomes 0, add to queue
                        if *degree == 0 {
                            queue.push_back(target);
                        }
                    }
                }
            }
        }

        // Check if all nodes were processed (cycle detection)
        if result.len() != node_ids.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Graph contains a cycle - not a DAG. Processed {} of {} nodes",
                    result.len(),
                    node_ids.len()
                )
            ));
        }

        // Return node IDs directly
        Ok(result)
    }

    /// Calculate Burt's constraint for each vertex
    ///
    /// Burt's constraint measures the extent to which a vertex's neighbors are
    /// connected to each other, indicating lack of structural holes. Higher values
    /// indicate more redundant contacts and less brokerage opportunity.
    ///
    /// Formula: C_i = Σ_j (p_ij + Σ_q p_iq * p_qj)²
    /// where p_ij = (a_ij + a_ji) / Σ_k (a_ik + a_ki) are proportional tie strengths
    ///
    /// Returns None for isolated vertices (no connections)
    #[inline]
    #[pyo3(signature = (weights=None))]
    pub fn constraint(&self, weights: Option<&str>) -> PyResult<FxHashMap<u64, Option<f64>>> {
        // Edge weight: the weight field by default (or for "weight"), else the named numeric
        // attribute; anything missing counts as 1.0
        let weight_of = |edge: &Edge| -> f64 {
            match weights {
                None | Some("weight") => edge.weight.unwrap_or(1.0),
                Some(name) => edge
                    .attributes
                    .as_ref()
                    .and_then(|attrs| {
                        Python::with_gil(|py| {
                            attrs
                                .bind(py)
                                .get_item(name)
                                .ok()
                                .flatten()
                                .and_then(|v| v.extract::<f64>().ok())
                        })
                    })
                    .unwrap_or(1.0),
            }
        };

        // a[(u, v)]: total weight of the edges u -> v (parallel edges add up; an undirected
        // edge counts in both orientations)
        let mut a: FxHashMap<(u64, u64), f64> = FxHashMap::default();
        // Neighbors of each node, ignoring direction (in- and out-neighbors), as in NetworkX's
        // `all_neighbors()`; a self-loop makes a node its own neighbor
        let mut neighbors: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::default();
        for edge in &self.edges {
            let w = weight_of(edge);
            *a.entry((edge.source, edge.target)).or_insert(0.0) += w;
            if !self.directed && edge.source != edge.target {
                *a.entry((edge.target, edge.source)).or_insert(0.0) += w;
            }
            neighbors.entry(edge.source).or_default().insert(edge.target);
            neighbors.entry(edge.target).or_default().insert(edge.source);
        }

        // Mutual tie strength m(u, v) = a_uv + a_vu
        let mutual = |u: u64, v: u64| -> f64 {
            a.get(&(u, v)).copied().unwrap_or(0.0) + a.get(&(v, u)).copied().unwrap_or(0.0)
        };

        // Each node's total tie strength, the denominator of p_uv
        let empty = FxHashSet::default();
        let scale: FxHashMap<u64, f64> = self
            .nodes
            .keys()
            .map(|&u| {
                let total = neighbors.get(&u).unwrap_or(&empty).iter().map(|&w| mutual(u, w)).sum();
                (u, total)
            })
            .collect();

        // Proportional tie strength p_uv = m(u, v) / Σ_w m(u, w)
        let p = |u: u64, v: u64| -> f64 {
            let s = scale.get(&u).copied().unwrap_or(0.0);
            if s == 0.0 { 0.0 } else { mutual(u, v) / s }
        };

        // C_u = Σ_v (p_uv + Σ_w p_uw · p_wv)², over u's neighbors v and w (as in Burt 1992 and
        // NetworkX). Undefined for isolated nodes (a self-loop alone doesn't count as a tie,
        // matching NetworkX) and for nodes whose ties all weigh 0.
        let mut result = FxHashMap::with_capacity_and_hasher(self.nodes.len(), Default::default());
        for &u in self.nodes.keys() {
            let n_u = neighbors.get(&u).unwrap_or(&empty);
            let has_other_tie = n_u.iter().any(|&v| v != u);
            if !has_other_tie || scale.get(&u).copied().unwrap_or(0.0) == 0.0 {
                result.insert(u, None);
                continue;
            }
            let c: f64 = n_u
                .iter()
                .map(|&v| {
                    let indirect: f64 = n_u.iter().map(|&w| p(u, w) * p(w, v)).sum();
                    let local = p(u, v) + indirect;
                    local * local
                })
                .sum();
            result.insert(u, Some(c));
        }

        Ok(result)
    }

    /// Count the number of walks of a specific length between all pairs of nodes.
    ///
    /// A walk is a sequence of nodes where each adjacent pair is connected by an edge.
    /// Walks can repeat edges and nodes. The number of walks of length k from node i
    /// to node j equals the (i,j) entry of A^k where A is the adjacency matrix.
    ///
    /// # Arguments
    /// * `walk_length` - Non-negative integer representing the number of edges in the walk
    ///
    /// # Returns
    /// Nested HashMap: outer key = source node label, inner key = target node label,
    /// inner value = number of walks of length `walk_length` from source to target
    ///
    /// # Time Complexity
    /// O(V³ × log k) using fast matrix exponentiation where V is number of vertices
    /// and k is walk_length
    #[inline]
    pub fn count_walks(&self, walk_length: usize) -> PyResult<FxHashMap<u64, FxHashMap<u64, u64>>> {
        // Get sorted list of node IDs for consistent matrix indexing
        let n = self.num_nodes();
        let mut node_ids = Vec::with_capacity(n);
        node_ids.extend(self.nodes.keys().copied());
        node_ids.sort_unstable();

        // Create node_id -> matrix index mapping
        let mut id_to_idx = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (idx, &node_id) in node_ids.iter().enumerate() {
            id_to_idx.insert(node_id, idx);
        }

        // Build adjacency matrix
        let mut adj_matrix = vec![vec![0u64; n]; n];

        for edge in &self.edges {
            if self.deleted_edge_ids.contains(&edge.id) {
                continue;
            }

            let source_idx = id_to_idx[&edge.source];
            let target_idx = id_to_idx[&edge.target];

            // Add edge (in directed graphs, only source -> target)
            adj_matrix[source_idx][target_idx] += 1;

            // In undirected graphs, also add reverse edge
            if !self.directed {
                adj_matrix[target_idx][source_idx] += 1;
            }
        }

        // Compute A^walk_length using fast matrix exponentiation
        let result_matrix = if walk_length == 0 {
            // A^0 = Identity matrix
            let mut identity = vec![vec![0u64; n]; n];
            for (i, row) in identity.iter_mut().enumerate() {
                row[i] = 1;
            }
            identity
        } else {
            matrix_power(&adj_matrix, walk_length)
        };

        // Convert result matrix to nested HashMap with node IDs
        let mut result = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (i, &source_id) in node_ids.iter().enumerate() {
            let mut target_map = FxHashMap::with_capacity_and_hasher(n, Default::default());

            for (j, &target_id) in node_ids.iter().enumerate() {
                target_map.insert(target_id, result_matrix[i][j]);
            }

            result.insert(source_id, target_map);
        }

        Ok(result)
    }

    /// Compute a spanner of the graph with the given stretch
    ///
    /// A spanner is a sparse subgraph that approximates distances in the original graph.
    /// For any pair of nodes u, v: distance_H(u,v) <= stretch * distance_G(u,v)
    ///
    /// Convert directed graph to undirected by making all edges bidirectional
    ///
    /// Creates a new graph with the same nodes but converts directed edges to undirected.
    /// If the graph is already undirected, returns a copy.
    ///
    /// # Returns
    /// A new undirected Graph
    pub fn to_undirected(&self, py: Python) -> PyResult<Graph> {
        let mut result = Graph::new(false, self.weighted);

        // Copy all nodes
        for (node_id, node) in &self.nodes {
            let label = node.label.clone();
            let attrs = copy_attrs(py, &node.attributes);
            let node_obj = Node::new(*node_id, label, attrs);
            result.add_node(node_obj)?;
        }

        // Copy all edges (removing directedness)
        // For undirected graphs, this is already correct
        // For directed graphs, we just change the graph type
        let mut seen_pairs: FxHashSet<(u64, u64)> = FxHashSet::default();

        for edge in &self.edges {
            let pair = if edge.source < edge.target {
                (edge.source, edge.target)
            } else {
                (edge.target, edge.source)
            };

            // Skip if we've already added this undirected edge
            if seen_pairs.contains(&pair) {
                continue;
            }
            seen_pairs.insert(pair);

            let weight = edge.weight;
            let attrs = copy_attrs(py, &edge.attributes);
            // The kept edge keeps its ID
            let edge_obj = Edge::new(edge.id, edge.source, edge.target, weight, attrs);
            result.add_edge(edge_obj)?;
        }

        Ok(result)
    }

    /// Convert undirected graph to directed by creating both directions for each edge
    ///
    /// Creates a new graph with the same nodes but converts undirected edges to directed
    /// by creating both directions (u->v and v->u) for each edge.
    /// If the graph is already directed, returns a copy.
    ///
    /// # Returns
    /// A new directed Graph
    pub fn to_directed(&self, py: Python) -> PyResult<Graph> {
        let mut result = Graph::new(true, self.weighted);

        // Copy all nodes
        for (node_id, node) in &self.nodes {
            let label = node.label.clone();
            let attrs = copy_attrs(py, &node.attributes);
            let node_obj = Node::new(*node_id, label, attrs);
            result.add_node(node_obj)?;
        }

        // Copy all edges in their stored direction, keeping their IDs
        for edge in &self.edges {
            let attrs = copy_attrs(py, &edge.attributes);
            result.add_edge(Edge::new(edge.id, edge.source, edge.target, edge.weight, attrs))?;
        }

        // If originally undirected, add the reverse direction too. These are new edges, added
        // after all the originals so their fresh IDs can't collide with a kept one
        if !self.directed {
            for edge in &self.edges {
                let attrs = copy_attrs(py, &edge.attributes);
                result.add_edge(Edge::new(0, edge.target, edge.source, edge.weight, attrs))?;
            }
        }

        Ok(result)
    }

    /// Reverse the direction of all edges in a directed graph
    ///
    /// Creates a new graph with all edge directions reversed.
    /// For undirected graphs, returns a copy (no change in structure).
    ///
    /// # Returns
    /// A new Graph with reversed edges
    pub fn reverse(&self, py: Python) -> PyResult<Graph> {
        let mut result = Graph::new(self.directed, self.weighted);

        // Copy all nodes
        for (node_id, node) in &self.nodes {
            let label = node.label.clone();
            let attrs = copy_attrs(py, &node.attributes);
            let node_obj = Node::new(*node_id, label, attrs);
            result.add_node(node_obj)?;
        }

        // Copy edges with reversed direction (if directed)
        for edge in &self.edges {
            let weight = edge.weight;
            let attrs = copy_attrs(py, &edge.attributes);

            if self.directed {
                // Swap source and target, keeping the edge's ID
                let edge_obj = Edge::new(edge.id, edge.target, edge.source, weight, attrs);
                result.add_edge(edge_obj)?;
            } else {
                // For undirected, keep as is
                let edge_obj = Edge::new(edge.id, edge.source, edge.target, weight, attrs);
                result.add_edge(edge_obj)?;
            }
        }

        Ok(result)
    }

    /// Return the complement of the graph
    ///
    /// Creates a new graph with the same nodes but complementary edges:
    /// - Edges NOT in the original graph become edges in the complement
    /// - Edges in the original graph are NOT in the complement
    /// - Self-loops are never included in the complement
    ///
    /// For directed graphs, complements all possible directed edges.
    /// For undirected graphs, complements all possible undirected edges.
    ///
    /// # Returns
    /// A new Graph with complementary edges (no attributes on new edges)
    pub fn complement(&self, py: Python) -> PyResult<Graph> {
        let mut result = Graph::new(self.directed, self.weighted);

        // Copy all nodes (with attributes)
        for (node_id, node) in &self.nodes {
            let label = node.label.clone();
            let attrs = copy_attrs(py, &node.attributes);
            let node_obj = Node::new(*node_id, label, attrs);
            result.add_node(node_obj)?;
        }

        // Build set of existing edges for fast lookup
        let mut existing_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
        for edge in &self.edges {
            if self.directed {
                existing_edges.insert((edge.source, edge.target));
            } else {
                // For undirected, normalize the pair
                let pair = if edge.source < edge.target {
                    (edge.source, edge.target)
                } else {
                    (edge.target, edge.source)
                };
                existing_edges.insert(pair);
            }
        }

        // Add all edges that don't exist in the original graph
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        for &source in &node_ids {
            for &target in &node_ids {
                // Skip self-loops
                if source == target {
                    continue;
                }

                if self.directed {
                    // For directed graphs, check each direction separately
                    if !existing_edges.contains(&(source, target)) {
                        let weight = if self.weighted { Some(1.0) } else { None };
                        let edge_obj = Edge::new(0, source, target, weight, None);
                        result.add_edge(edge_obj)?;
                    }
                } else {
                    // For undirected graphs, only check one direction to avoid duplicates
                    if source < target {
                        let pair = (source, target);
                        if !existing_edges.contains(&pair) {
                            let weight = if self.weighted { Some(1.0) } else { None };
                            let edge_obj = Edge::new(0, source, target, weight, None);
                            result.add_edge(edge_obj)?;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// # Arguments
    /// * `stretch` - Maximum distance approximation factor (must be >= 1.0)
    /// * `weight` - Optional edge attribute name to use as weight
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// A new Graph with the same nodes but a subset of edges
    ///
    /// # Algorithm
    /// Baswana-Sen randomized algorithm:
    /// - Phase 1: Clustering nodes and selecting edges between clusters
    /// - Phase 2: Connecting vertices to nearby clusters
    /// - Expected time: O(km) where k = (stretch+1)//2, m = number of edges
    /// - Sparsity: O(k × n^(1+1/k)) edges
    #[pyo3(signature = (stretch, weight=None, seed=None))]
    pub fn spanner(&self, py: Python, stretch: f64, weight: Option<String>, seed: Option<u64>) -> PyResult<Graph> {
        // Validate input
        if stretch < 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "stretch must be at least 1.0"
            ));
        }

        // Only works on undirected simple graphs
        if self.directed {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "spanner only works on undirected graphs"
            ));
        }

        // For now, implement a simple greedy spanner algorithm
        // This is simpler than Baswana-Sen but still gives good results

        // Initialize RNG
        let _rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        // Initialize spanner with all nodes, no edges
        let mut spanner = Graph::new(false, false);

        // Copy all nodes
        for (node_id, node) in &self.nodes {
            let label = node.label.clone();
            let attrs = copy_attrs(py, &node.attributes);
            let node_obj = Node::new(*node_id, label, attrs);
            spanner.add_node(node_obj)?;
        }

        // Build distance matrix in spanner (initially infinite)
        // Preallocate with capacity n*n for all pairs
        let n = self.nodes.len();
        let mut spanner_distances = FxHashMap::with_capacity_and_hasher(n * n, Default::default());
        for &u in self.nodes.keys() {
            for &v in self.nodes.keys() {
                if u == v {
                    spanner_distances.insert((u, v), 0.0);
                } else {
                    spanner_distances.insert((u, v), f64::INFINITY);
                }
            }
        }

        // Collect and sort edges by weight
        let mut sorted_edges = Vec::with_capacity(self.edges.len());
        sorted_edges.extend(self.edges.iter());
        sorted_edges.sort_by(|a, b| {
            let w_a = if let Some(ref w) = weight {
                a.attributes.as_ref().and_then(|attrs| {
                    let dict = attrs.bind(py).downcast::<PyDict>().ok()?;
                    dict.get_item(w).ok().flatten()?.extract::<f64>().ok()
                }).unwrap_or(1.0)
            } else {
                1.0
            };

            let w_b = if let Some(ref w) = weight {
                b.attributes.as_ref().and_then(|attrs| {
                    let dict = attrs.bind(py).downcast::<PyDict>().ok()?;
                    dict.get_item(w).ok().flatten()?.extract::<f64>().ok()
                }).unwrap_or(1.0)
            } else {
                1.0
            };

            w_a.partial_cmp(&w_b).unwrap()
        });

        // Greedy spanner: add edge if it improves stretch
        for edge in sorted_edges {
            let u = edge.source;
            let v = edge.target;

            let edge_weight = if let Some(ref w) = weight {
                edge.attributes.as_ref().and_then(|attrs| {
                    let dict = attrs.bind(py).downcast::<PyDict>().ok()?;
                    dict.get_item(w).ok().flatten()?.extract::<f64>().ok()
                }).unwrap_or(1.0)
            } else {
                1.0
            };

            // Check if this edge is needed (distance in spanner > stretch * edge_weight)
            let current_dist = spanner_distances[&(u, v)];
            if current_dist > stretch * edge_weight {
                // Add edge to spanner
                let attrs = copy_attrs(py, &edge.attributes);
                let edge_obj = Edge::new(0, u, v, Some(edge_weight), attrs);
                spanner.add_edge(edge_obj)?;

                // Update distances using Floyd-Warshall-like update
                // For each pair of nodes, check if going through this new edge improves distance
                for &x in self.nodes.keys() {
                    for &y in self.nodes.keys() {
                        let dist_through_edge = spanner_distances[&(x, u)]
                            .min(spanner_distances[&(x, v)])
                            + edge_weight
                            + spanner_distances[&(u, y)].min(spanner_distances[&(v, y)]);

                        let current = spanner_distances[&(x, y)];
                        if dist_through_edge < current {
                            spanner_distances.insert((x, y), dist_through_edge);
                        }
                    }
                }
            }
        }

        Ok(spanner)
    }

    /// Helper: compute sum of all pairwise shortest path distances (parallel)
    #[inline]
    fn compute_total_distances_parallel(&self) -> f64 {
        use rayon::prelude::*;

        let n = self.num_nodes();
        let mut node_ids = Vec::with_capacity(n);
        node_ids.extend(self.nodes.keys().copied());

        node_ids
            .par_iter()
            .map(|&start| {
                let distances = self.bfs_distances_optimized(start);
                distances.values().sum::<usize>() as f64
            })
            .sum()
    }

    /// Helper: compute sum of all pairwise shortest path distances
    fn compute_total_distances(&self) -> f64 {
        let mut total = 0.0;

        for &start in self.nodes.keys() {
            let distances = self.bfs_distances_optimized(start);
            total += distances.values().sum::<usize>() as f64;
        }

        total
    }

    /// Helper: compute total distances excluding a specific node
    #[inline]
    fn compute_total_distances_without_node(&self, excluded: u64) -> f64 {
        let mut total = 0.0;

        for &start in self.nodes.keys() {
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
    #[inline]
    fn bfs_distances_optimized(&self, start: u64) -> FxHashMap<u64, usize> {
        let n = self.num_nodes();
        let mut distances = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(n);

        distances.insert(start, 0);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                        e.insert(current_dist + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    /// Helper: BFS to compute distances from a start node
    #[inline]
    fn bfs_distances(&self, start: u64) -> FxHashMap<u64, usize> {
        self.bfs_distances_optimized(start)
    }

    /// Helper: Optimized BFS excluding a specific node
    #[inline]
    fn bfs_distances_excluding_optimized(&self, start: u64, excluded: u64) -> FxHashMap<u64, usize> {
        let n = self.num_nodes();
        let mut distances = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(n);

        distances.insert(start, 0);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let current_dist = *distances.get(&current).unwrap();

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if neighbor == excluded {
                        continue;
                    }
                    if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                        e.insert(current_dist + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    /// Helper: BFS excluding a specific node
    #[inline]
    fn bfs_distances_excluding(&self, start: u64, excluded: u64) -> FxHashMap<u64, usize> {
        self.bfs_distances_excluding_optimized(start, excluded)
    }

    /// Get k-core decomposition (coreness for each node)
    /// k-core is maximal subgraph where all nodes have degree >= k
    /// Coreness is the highest k for which a node belongs to the k-core
    #[inline]
    pub fn get_coreness(&self) -> FxHashMap<u64, usize> {
        let n = self.nodes.len();
        let mut coreness: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut degrees: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &id in self.nodes.keys() {
            degrees.insert(id, self.degree(id));
        }

        let mut remaining: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
        remaining.extend(self.nodes.keys().copied());
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
            let mut to_remove = Vec::with_capacity(remaining.len());
            for &node in &remaining {
                if *degrees.get(&node).unwrap() <= min_deg {
                    to_remove.push(node);
                }
            }

            for &node in &to_remove {
                coreness.insert(node, k);
                remaining.remove(&node);

                // Update degrees of neighbors
                if let Some(neighbors) = self.adjacency_list.get(&node) {
                    for &neighbor in neighbors {
                        if remaining.contains(&neighbor) {
                            if let Some(deg) = degrees.get_mut(&neighbor) {
                                *deg = deg.saturating_sub(1);
                            }
                        }
                    }
                }
            }
        }

        coreness
    }

    /// Get local clustering coefficient for each node
    /// Clustering coefficient measures the degree to which nodes cluster together
    /// For a node: C(v) = (number of triangles through v) / (number of possible triangles)
    /// = (number of edges between neighbors) / (k * (k-1) / 2)
    /// where k is the degree of v
    /// Returns 0.0 for nodes with degree < 2
    ///
    /// OPTIMIZED: Uses pre-computed triangle counts instead of recounting edges
    #[inline]
    fn get_clustering_coefficient(&self) -> FxHashMap<u64, f64> {
        // Use the optimized version from algorithms::structural
        let clustering_opt = crate::algorithms::structural::local_clustering_coefficient(self);

        // Preallocate result HashMap with exact capacity needed
        let n = clustering_opt.len();
        let mut result = FxHashMap::with_capacity_and_hasher(n, Default::default());

        // Convert Option<f64> to f64, mapping None to 0.0
        for (node_id, opt_coeff) in clustering_opt {
            result.insert(node_id, opt_coeff.unwrap_or(0.0));
        }

        result
    }

    /// Get average clustering coefficient for the graph
    /// Returns the mean of all node clustering coefficients
    #[inline]
    fn get_avg_clustering_coefficient(&self) -> f64 {
        let clustering = self.get_clustering_coefficient();

        if clustering.is_empty() {
            return 0.0;
        }

        let n = clustering.len() as f64;
        let inv_n = 1.0 / n;
        let sum: f64 = clustering.values().sum();
        sum * inv_n
    }

    /// Get edge clustering coefficient for specified edges
    ///
    /// Edge clustering coefficient C^(k)_ij measures how clustered an edge's
    /// neighborhood is based on k-cycles it participates in.
    ///
    /// For k=3 (triangles):
    ///   - z^(3)_ij = number of triangles containing edge (i,j)
    ///   - s^(3)_ij = min(d_i - 1, d_j - 1) (maximum possible triangles)
    ///   - C^(3)_ij = (z^(3)_ij + offset) / s^(3)_ij (if normalize=true)
    ///
    /// For k=4 (4-cycles):
    ///   - z^(4)_ij = number of 4-cycles containing edge (i,j)
    ///   - s^(4)_ij = (d_i - 1)(d_j - 1) (maximum possible 4-cycles)
    ///   - C^(4)_ij = (z^(4)_ij + offset) / s^(4)_ij (if normalize=true)
    ///
    /// # Arguments
    /// * edge_ids - Vector of edge IDs to compute coefficient for (empty = all edges)
    /// * k - Cycle size (3 or 4)
    /// * normalize - Whether to normalize by maximum possible cycles s^(k)
    /// * offset - Whether to add 1 to cycle count (original definition uses offset=true)
    ///
    /// # Returns
    /// HashMap mapping edge_id -> clustering coefficient
    ///
    /// # Reference
    /// Radicchi et al. "Defining and identifying communities in networks" PNAS 101, 2658 (2004)
    fn get_edge_clustering_coefficient(
        &self,
        edge_ids: Vec<u64>,
        k: usize,
        normalize: bool,
        offset: bool,
    ) -> PyResult<FxHashMap<u64, f64>> {
        // Validate k parameter
        if k != 3 && k != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "k must be 3 or 4"
            ));
        }

        // Build neighbor sets for efficient lookup
        let neighbor_sets: FxHashMap<u64, FxHashSet<u64>> = self.adjacency_list
            .iter()
            .map(|(&node, neighbors)| (node, neighbors.iter().copied().collect()))
            .collect();

        // Determine which edges to process
        let edges_to_process: Vec<&Edge> = if edge_ids.is_empty() {
            self.edges.iter().collect()
        } else {
            let edge_id_set: FxHashSet<u64> = edge_ids.iter().copied().collect();
            self.edges.iter()
                .filter(|e| edge_id_set.contains(&e.id))
                .collect()
        };

        let mut result = FxHashMap::default();

        for edge in edges_to_process {
            let i = edge.source;
            let j = edge.target;

            // Get degrees
            let d_i = self.degree(i);
            let d_j = self.degree(j);

            // Get neighbors (excluding the edge endpoints from each other's neighbor list)
            let neighbors_i = neighbor_sets.get(&i).cloned().unwrap_or_default();
            let neighbors_j = neighbor_sets.get(&j).cloned().unwrap_or_default();

            let coeff = if k == 3 {
                // Count triangles: common neighbors of i and j
                let common_neighbors: FxHashSet<_> = neighbors_i
                    .intersection(&neighbors_j)
                    .copied()
                    .collect();

                let mut z = common_neighbors.len() as f64;
                if offset {
                    z += 1.0;
                }

                if normalize {
                    // s^(3) = min(d_i - 1, d_j - 1)
                    let s = (d_i.saturating_sub(1)).min(d_j.saturating_sub(1)) as f64;
                    if s > 0.0 {
                        z / s
                    } else {
                        0.0
                    }
                } else {
                    z
                }
            } else {
                // k == 4: Count 4-cycles
                // A 4-cycle through edge (i,j) has form: i -> a -> j -> b -> i
                // where a ∈ neighbors(i), b ∈ neighbors(j), and a != j, b != i
                // We need a and b to be connected

                let mut cycle_count = 0;

                // For each neighbor a of i (excluding j)
                for &a in &neighbors_i {
                    if a == j {
                        continue;
                    }

                    // For each neighbor b of j (excluding i)
                    for &b in &neighbors_j {
                        if b == i || b == a {
                            continue;
                        }

                        // Check if a and b are connected
                        if let Some(a_neighbors) = neighbor_sets.get(&a) {
                            if a_neighbors.contains(&b) {
                                cycle_count += 1;
                            }
                        }
                    }
                }

                // Each 4-cycle is counted once in the above loop
                let mut z = cycle_count as f64;
                if offset {
                    z += 1.0;
                }

                if normalize {
                    // s^(4) = (d_i - 1)(d_j - 1)
                    let s = (d_i.saturating_sub(1) * d_j.saturating_sub(1)) as f64;
                    if s > 0.0 {
                        z / s
                    } else {
                        0.0
                    }
                } else {
                    z
                }
            };

            result.insert(edge.id, coeff);
        }

        Ok(result)
    }

    /// Get HITS authority scores
    /// Hyperlink-Induced Topic Search algorithm for directed graphs
    /// Authority score: sum of hub scores of nodes pointing to this node
    /// Returns HashMap of node -> authority score
    #[inline]
    fn get_authority_centrality(&self, max_iter: usize, tol: f64) -> PyResult<FxHashMap<u64, f64>> {
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
    #[inline]
    fn get_hub_centrality(&self, max_iter: usize, tol: f64) -> PyResult<FxHashMap<u64, f64>> {
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
    #[inline]
    fn compute_hits(&self, max_iter: usize, tol: f64) -> (FxHashMap<u64, f64>, FxHashMap<u64, f64>) {
        let n = self.num_nodes();
        if n == 0 {
            return (FxHashMap::default(), FxHashMap::default());
        }

        // Preallocate HashMaps with capacity for better performance
        let mut authorities: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut hubs: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut new_authorities: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut new_hubs: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());

        // Initialize all scores to 1.0
        for &id in self.nodes.keys() {
            authorities.insert(id, 1.0);
            hubs.insert(id, 1.0);
        }

        for _ in 0..max_iter {
            // Clear new maps for this iteration
            new_authorities.clear();
            new_hubs.clear();

            // Update authority scores: sum of hub scores of incoming neighbors
            for &node in self.nodes.keys() {
                let mut auth_score = 0.0;
                // Find all nodes that point to this node
                for (&source, targets) in &self.adjacency_list {
                    if targets.contains(&node) {
                        auth_score += hubs.get(&source).unwrap_or(&0.0);
                    }
                }
                new_authorities.insert(node, auth_score);
            }

            // Update hub scores: sum of authority scores of outgoing neighbors
            for &node in self.nodes.keys() {
                let mut hub_score = 0.0;
                if let Some(neighbors) = self.adjacency_list.get(&node) {
                    for &neighbor in neighbors {
                        hub_score += authorities.get(&neighbor).unwrap_or(&0.0);
                    }
                }
                new_hubs.insert(node, hub_score);
            }

            // Normalize using multiplication instead of division
            let auth_norm: f64 = new_authorities.values().map(|&v| v * v).sum::<f64>().sqrt();
            let hub_norm: f64 = new_hubs.values().map(|&v| v * v).sum::<f64>().sqrt();

            if auth_norm > 0.0 {
                let inv_auth_norm = 1.0 / auth_norm;
                for val in new_authorities.values_mut() {
                    *val *= inv_auth_norm;
                }
            }
            if hub_norm > 0.0 {
                let inv_hub_norm = 1.0 / hub_norm;
                for val in new_hubs.values_mut() {
                    *val *= inv_hub_norm;
                }
            }

            // Check convergence
            let auth_diff: f64 = new_authorities.iter()
                .map(|(&k, &v)| (v - authorities.get(&k).unwrap()).abs())
                .sum();
            let hub_diff: f64 = new_hubs.iter()
                .map(|(&k, &v)| (v - hubs.get(&k).unwrap()).abs())
                .sum();

            if auth_diff < tol && hub_diff < tol {
                // Return the new values on convergence
                return (new_authorities, new_hubs);
            }

            // Swap for next iteration - zero-cost move
            std::mem::swap(&mut authorities, &mut new_authorities);
            std::mem::swap(&mut hubs, &mut new_hubs);
        }

        (authorities, hubs)
    }

    /// Get eigenvector centrality
    /// Standard eigenvector centrality: principal eigenvector of adjacency matrix
    /// x = A * x (power iteration method)
    /// Measures importance based on connections to other important nodes
    ///
    /// Optimized implementation using dense arrays instead of HashMaps for better cache locality
    ///
    /// Returns the scores and whether the iteration converged within `max_iter` rounds.
    #[inline]
    fn get_eigen_centrality(&self, max_iter: usize, tol: f64) -> (FxHashMap<u64, f64>, bool) {
        let n = self.num_nodes();
        if n == 0 {
            return (FxHashMap::default(), true);
        }

        // Special case: single node
        if n == 1 {
            return (self.nodes.keys().map(|&id| (id, 1.0)).collect(), true);
        }

        // Create node_id -> index mapping for dense array access
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        let mut node_to_idx: FxHashMap<u64, usize> = FxHashMap::default();
        node_to_idx.reserve(n);
        for (idx, &id) in node_ids.iter().enumerate() {
            node_to_idx.insert(id, idx);
        }

        // Build neighbor list using indices for cache-friendly access
        // Preallocate capacity for better performance
        let mut neighbor_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
        for _ in 0..n {
            neighbor_indices.push(Vec::with_capacity(8)); // Typical degree estimate
        }
        for (&node_id, neighbors) in &self.adjacency_list {
            if let Some(&node_idx) = node_to_idx.get(&node_id) {
                for &neighbor_id in neighbors {
                    if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                        neighbor_indices[node_idx].push(neighbor_idx);
                    }
                }
            }
        }

        // Power iteration on (A + I), as NetworkX does. A + I has the same eigenvectors as A,
        // but the shift breaks the tie between eigenvalues λ and -λ on bipartite graphs (trees,
        // stars, even cycles): the unwanted component shrinks by (λ - 1)/(λ + 1) per round. A
        // tiny shift (it used to be 0.01) leaves that ratio near 1, so the scores kept
        // oscillating and 100 rounds stopped far from the answer (star hub 0.645 vs 0.707)
        let shift = 1.0;

        let mut centrality = vec![1.0; n];
        let mut new_centrality = vec![0.0; n];
        let mut converged = false;

        for _iter in 0..max_iter {
            // Matrix-vector multiply: new = (A + εI) * centrality
            // Reset array using fast fill operation
            new_centrality.fill(0.0);

            for node_idx in 0..n {
                let mut sum = shift * centrality[node_idx]; // Diagonal shift term
                for &neighbor_idx in &neighbor_indices[node_idx] {
                    sum += centrality[neighbor_idx];
                }
                new_centrality[node_idx] = sum;
            }

            // Normalize using L2 norm
            let norm: f64 = new_centrality.iter()
                .map(|&v| v * v)
                .sum::<f64>()
                .sqrt();

            if norm < 1e-10 {
                // Converged to zero (disconnected or DAG), use uniform
                let uniform_value = 1.0 / (n as f64).sqrt();
                centrality.fill(uniform_value);
                converged = true;
                break;
            }

            // Normalize and check convergence using L1 norm
            // Use multiplication instead of division (3-4× faster)
            let inv_norm = 1.0 / norm;
            let delta: f64 = new_centrality.iter_mut()
                .zip(centrality.iter())
                .map(|(new, &old)| {
                    *new *= inv_norm;
                    (*new - old).abs()
                })
                .sum();

            // Swap arrays (no allocation)
            std::mem::swap(&mut centrality, &mut new_centrality);

            if delta < tol {
                converged = true;
                break;
            }
        }

        // Convert back to HashMap with node IDs
        let scores = node_ids.iter()
            .enumerate()
            .map(|(idx, &node_id)| (node_id, centrality[idx]))
            .collect();
        (scores, converged)
    }

    /// Get Bonacich power centrality
    /// Computes centrality based on: c(α, β) = α(I - βA)^(-1) A 1
    /// β > 0: cooperative relations (powerful neighbors increase power)
    /// β < 0: competitive relations (weak neighbors increase power)
    /// β = 1/λ_max: equivalent to eigenvector centrality
    /// Returns empty HashMap if matrix is singular
    #[inline]
    fn get_bonacich_power_centrality(&self, beta: f64, normalized: bool) -> FxHashMap<u64, f64> {
        let n = self.num_nodes();
        if n == 0 {
            return FxHashMap::default();
        }

        // Create node ID to index mapping
        let mut node_ids = Vec::with_capacity(n);
        node_ids.extend(self.nodes.keys().copied());

        let mut _node_to_idx = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (idx, &id) in node_ids.iter().enumerate() {
            _node_to_idx.insert(id, idx);
        }

        // Build adjacency matrix A using nalgebra
        let mut a_matrix = na::DMatrix::<f64>::zeros(n, n);
        for (&node, neighbors) in &self.adjacency_list {
            let i = _node_to_idx[&node];
            for &neighbor in neighbors {
                let j = _node_to_idx[&neighbor];
                a_matrix[(i, j)] = 1.0;
            }
        }

        // Compute (I - β*A)
        let identity = na::DMatrix::<f64>::identity(n, n);
        let i_minus_beta_a = &identity - beta * &a_matrix;

        // Solve (I - β*A) * x = A * 1
        // This is equivalent to x = (I - β*A)^(-1) * A * 1
        let ones = na::DVector::<f64>::from_element(n, 1.0);
        let a_ones = &a_matrix * &ones;

        // Use LU decomposition to solve the system
        let lu = i_minus_beta_a.lu();
        if !lu.is_invertible() {
            // Matrix is singular, return empty map
            return FxHashMap::default();
        }

        let solution = match lu.solve(&a_ones) {
            Some(sol) => sol,
            None => return FxHashMap::default(),
        };

        // Extract centrality scores
        let mut centrality = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (idx, &id) in node_ids.iter().enumerate() {
            centrality.insert(id, solution[idx]);
        }

        // Normalize
        if normalized {
            // Sum normalization: scores sum to 1
            let sum: f64 = centrality.values().sum();
            if sum.abs() > 1e-10 {
                let inv_sum = 1.0 / sum;
                for val in centrality.values_mut() {
                    *val *= inv_sum;
                }
            }
        } else {
            // L2 normalization: sqrt(sum of squares) = sqrt(n)
            // So that middle centrality = 1
            let sum_squares: f64 = centrality.values().map(|&v| v * v).sum();
            if sum_squares > 1e-10 {
                let scale = (n as f64 / sum_squares).sqrt();
                for val in centrality.values_mut() {
                    *val *= scale;
                }
            }
        }

        centrality
    }

    /// Get Katz centrality
    /// Measures node influence based on weighted walks from all other nodes
    /// x = alpha * A^T * x + beta
    /// where alpha is attenuation (must be < 1/lambda_max), beta is bias vector
    /// For directed graphs, uses transpose to count incoming paths
    ///
    /// OPTIMIZED: Pre-computes transpose graph to avoid O(n²) iteration
    ///
    /// `beta` maps every node to its baseline (exogenous) score; nodes missing from it get 0.
    /// With `normalized`, the scores are rescaled to sum to 1 (which cancels a constant beta);
    /// otherwise the raw fixed point is returned. Also reports whether the iteration converged
    /// within `max_iter` rounds (it can't when alpha >= 1 / lambda_max).
    #[inline]
    fn get_katz_centrality(
        &self,
        alpha: f64,
        beta: FxHashMap<u64, f64>,
        max_iter: usize,
        tol: f64,
        normalized: bool,
    ) -> (FxHashMap<u64, f64>, bool) {
        let n = self.num_nodes();
        if n == 0 {
            return (FxHashMap::default(), true);
        }

        // Pre-compute transpose adjacency list (incoming edges)
        // This is the key optimization: O(m) instead of O(n²)
        let mut transpose: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        transpose.reserve(n);
        for &node in self.nodes.keys() {
            transpose.insert(node, Vec::with_capacity(8)); // Preallocate typical size
        }

        for (&from_node, neighbors) in &self.adjacency_list {
            for &to_node in neighbors {
                if let Some(incoming) = transpose.get_mut(&to_node) {
                    incoming.push(from_node);
                }
            }
        }

        // Use dense arrays for better cache locality (like PageRank optimization)
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        let mut node_to_idx: FxHashMap<u64, usize> = FxHashMap::default();
        node_to_idx.reserve(n);
        for (idx, &id) in node_ids.iter().enumerate() {
            node_to_idx.insert(id, idx);
        }

        // Per-node baseline, aligned with node_ids
        let beta: Vec<f64> = node_ids.iter().map(|id| beta.get(id).copied().unwrap_or(0.0)).collect();

        // Initialize centrality arrays
        let mut centrality = beta.clone();
        let mut new_centrality = beta.clone();
        let mut converged = false;

        for _ in 0..max_iter {
            // Compute new centrality for each node
            for (idx, &node) in node_ids.iter().enumerate() {
                // Sum centrality from incoming neighbors
                let incoming_sum: f64 = transpose
                    .get(&node)
                    .map(|predecessors| {
                        predecessors.iter()
                            .filter_map(|&pred| node_to_idx.get(&pred))
                            .map(|&pred_idx| centrality[pred_idx])
                            .sum()
                    })
                    .unwrap_or(0.0);

                // x_new = alpha * (incoming sum) + beta
                new_centrality[idx] = alpha * incoming_sum + beta[idx];
            }

            // Check convergence using vectorized L1 norm
            let max_change: f64 = new_centrality.iter()
                .zip(centrality.iter())
                .map(|(new, old)| (new - old).abs())
                .fold(0.0, f64::max);

            // Swap arrays (zero-cost with move semantics)
            std::mem::swap(&mut centrality, &mut new_centrality);

            if max_change < tol {
                converged = true;
                break;
            }
            if centrality.iter().any(|v| !v.is_finite()) {
                break; // Diverged (alpha too large); no point iterating further
            }
        }

        // Normalize by the sum using multiplication (faster than division)
        if normalized {
            let total: f64 = centrality.iter().sum();
            if total.is_finite() && total != 0.0 {
                let inv_total = 1.0 / total;
                for val in centrality.iter_mut() {
                    *val *= inv_total;
                }
            }
        }

        // Convert back to HashMap with node IDs
        let scores = node_ids
            .iter()
            .enumerate()
            .map(|(idx, &node_id)| (node_id, centrality[idx]))
            .collect();
        (scores, converged)
    }

    /// Get alpha centrality
    /// Generalization of eigenvector centrality with attenuation factor
    /// x = alpha * A * x + e
    /// where alpha is attenuation, A is adjacency matrix, e is exogenous factor
    ///
    /// Also reports whether the iteration converged within `max_iter` rounds (it can't when
    /// alpha >= 1 / lambda_max).
    #[inline]
    fn get_alpha_centrality(&self, alpha: f64, max_iter: usize, tol: f64) -> (FxHashMap<u64, f64>, bool) {
        let n = self.num_nodes();
        if n == 0 {
            return (FxHashMap::default(), true);
        }

        // Initialize centrality to 1.0 for all nodes
        let mut centrality: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &id in self.nodes.keys() {
            centrality.insert(id, 1.0);
        }
        let mut prev_centrality: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut converged = false;

        for _ in 0..max_iter {
            // Swap instead of clone
            std::mem::swap(&mut centrality, &mut prev_centrality);
            let mut max_change: f64 = 0.0;

            for &node in self.nodes.keys() {
                // Sum of centralities of neighbors
                let neighbor_sum: f64 = self.adjacency_list
                    .get(&node)
                    .map(|neighbors| {
                        neighbors.iter()
                            .filter_map(|&neighbor| prev_centrality.get(&neighbor))
                            .sum()
                    })
                    .unwrap_or(0.0);

                // x = alpha * A x + e, with e = 1 for every node. Iterate the equation
                // itself: renormalizing inside the loop and then adding e (as this used
                // to) converges to something else, with nearly uniform scores.
                let new_value = alpha * neighbor_sum + 1.0;
                let prev_value = *prev_centrality.get(&node).unwrap();
                centrality.insert(node, new_value);

                let change = (new_value - prev_value).abs() / new_value.abs().max(1.0);
                max_change = max_change.max(change);
            }

            if max_change < tol {
                converged = true;
                break;
            }
            if centrality.values().any(|v| !v.is_finite()) {
                break;
            }
        }

        // Report the fixed point scaled to sum to 1.
        let total: f64 = centrality.values().sum();
        if total.is_finite() && total > 0.0 {
            let inv_total = 1.0 / total;
            for val in centrality.values_mut() {
                *val *= inv_total;
            }
        }

        (centrality, converged)
    }

    /// Get leverage centrality for each node
    /// Measures the degree connectivity of a node relative to its neighbors
    /// leverage(v) = (1/k_v) * sum((k_v - k_n) / (k_v + k_n))
    /// where k_v is degree of v, k_n is degree of neighbor n
    #[inline]
    fn get_leverage_centrality(&self) -> FxHashMap<u64, f64> {
        let n = self.num_nodes();
        let mut leverage = FxHashMap::with_capacity_and_hasher(n, Default::default());

        for &node in self.nodes.keys() {
            let k_v = self.degree(node) as f64;

            if k_v == 0.0 {
                leverage.insert(node, 0.0);
                continue;
            }

            let mut sum = 0.0;
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    let k_n = self.degree(neighbor) as f64;
                    let k_sum = k_v + k_n;
                    if k_sum > 0.0 {
                        let inv_k_sum = 1.0 / k_sum;
                        sum += (k_v - k_n) * inv_k_sum;
                    }
                }
            }

            let inv_k_v = 1.0 / k_v;
            let lev = sum * inv_k_v;
            leverage.insert(node, lev);
        }

        leverage
    }

    /// Get Dice similarity coefficient between all node pairs
    /// Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    /// Measures neighborhood overlap, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    #[inline]
    fn get_dice_similarity(&self) -> FxHashMap<u64, FxHashMap<u64, f64>> {
        let n = self.num_nodes();
        let mut similarities = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut nodes = Vec::with_capacity(n);
        nodes.extend(self.nodes.keys().copied());

        for &node1 in &nodes {
            let mut node1_sims = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let neighbors1: FxHashSet<&u64> = self.adjacency_list
                .get(&node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for &node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2, 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&u64> = self.adjacency_list
                    .get(&node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();
                let size_sum = size1 + size2;

                let dice = if size_sum > 0 {
                    (2.0 * intersection as f64) * (1.0 / size_sum as f64)
                } else {
                    0.0
                };

                node1_sims.insert(node2, dice);
            }

            similarities.insert(node1, node1_sims);
        }

        similarities
    }

    /// Get Jaccard similarity index between all node pairs
    /// Jaccard index: |A ∩ B| / |A ∪ B|
    /// Classic set similarity measure, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    #[inline]
    fn get_jaccard_similarity(&self) -> FxHashMap<u64, FxHashMap<u64, f64>> {
        let n = self.num_nodes();
        let mut similarities = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut nodes = Vec::with_capacity(n);
        nodes.extend(self.nodes.keys().copied());

        for &node1 in &nodes {
            let mut node1_sims = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let neighbors1: FxHashSet<&u64> = self.adjacency_list
                .get(&node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for &node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2, 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&u64> = self.adjacency_list
                    .get(&node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let union = neighbors1.union(&neighbors2).count();

                let jaccard = if union > 0 {
                    (intersection as f64) * (1.0 / union as f64)
                } else {
                    0.0
                };

                node1_sims.insert(node2, jaccard);
            }

            similarities.insert(node1, node1_sims);
        }

        similarities
    }

    /// Get Cosine similarity between all node pairs
    /// Cosine similarity: A·B / (||A|| ||B||)
    /// Normalized dot product of adjacency vectors, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    #[inline]
    fn get_cosine_similarity(&self) -> FxHashMap<u64, FxHashMap<u64, f64>> {
        let n = self.num_nodes();
        let mut similarities = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut nodes = Vec::with_capacity(n);
        nodes.extend(self.nodes.keys().copied());

        for &node1 in &nodes {
            let mut node1_sims = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let neighbors1: FxHashSet<&u64> = self.adjacency_list
                .get(&node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for &node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2, 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&u64> = self.adjacency_list
                    .get(&node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();

                let cosine = if size1 > 0 && size2 > 0 {
                    let inv_denominator = 1.0 / ((size1 as f64).sqrt() * (size2 as f64).sqrt());
                    let result = (intersection as f64) * inv_denominator;
                    // Clamp to [0, 1] to handle floating-point precision issues
                    result.clamp(0.0, 1.0)
                } else {
                    0.0
                };

                node1_sims.insert(node2, cosine);
            }

            similarities.insert(node1, node1_sims);
        }

        similarities
    }

    /// Get Overlap coefficient between all node pairs
    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|)
    /// Subset-based similarity measure, range [0, 1]
    /// Returns nested HashMap: node1 -> node2 -> similarity
    #[inline]
    fn get_overlap_similarity(&self) -> FxHashMap<u64, FxHashMap<u64, f64>> {
        let n = self.num_nodes();
        let mut similarities = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut nodes = Vec::with_capacity(n);
        nodes.extend(self.nodes.keys().copied());

        for &node1 in &nodes {
            let mut node1_sims = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let neighbors1: FxHashSet<&u64> = self.adjacency_list
                .get(&node1)
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for &node2 in &nodes {
                if node1 == node2 {
                    node1_sims.insert(node2, 1.0);
                    continue;
                }

                let neighbors2: FxHashSet<&u64> = self.adjacency_list
                    .get(&node2)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                let intersection = neighbors1.intersection(&neighbors2).count();
                let size1 = neighbors1.len();
                let size2 = neighbors2.len();
                let min_size = size1.min(size2);

                let overlap = if min_size > 0 {
                    (intersection as f64) * (1.0 / min_size as f64)
                } else {
                    0.0
                };

                node1_sims.insert(node2, overlap);
            }

            similarities.insert(node1, node1_sims);
        }

        similarities
    }

    // ========================================
    // TRAVERSAL OPERATIONS
    // ========================================

    /// Traverse from selected nodes to outward neighbors.
    /// Replaces the current node selection with neighbors in the outward direction.
    /// For directed graphs, this follows edges where selected nodes are sources.
    #[inline]
    pub fn trav_out(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let n = self.selected_nodes.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_nodes.iter().copied());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors
        let mut new_selection = FxHashSet::with_capacity_and_hasher(n, Default::default());

        if !self.directed {
            // For undirected graphs, get all neighbors
            for node_id in current_selection {
                if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                    new_selection.extend(neighbors.iter().copied());
                }
            }
        } else {
            // For directed graphs, only get outward neighbors (successors)
            for node_id in current_selection {
                for edge in &self.edges {
                    if edge.source == node_id {
                        new_selection.insert(edge.target);
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
    #[inline]
    pub fn trav_in(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let n = self.selected_nodes.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_nodes.iter().copied());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors
        let mut new_selection = FxHashSet::with_capacity_and_hasher(n, Default::default());

        if !self.directed {
            // For undirected graphs, get all neighbors
            for node_id in current_selection {
                if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                    new_selection.extend(neighbors.iter().copied());
                }
            }
        } else {
            // For directed graphs, only get inward neighbors (predecessors)
            for node_id in current_selection {
                for edge in &self.edges {
                    if edge.target == node_id {
                        new_selection.insert(edge.source);
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
    #[inline]
    pub fn trav_both(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let n = self.selected_nodes.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_nodes.iter().copied());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Collect all neighbors (both directions)
        let mut new_selection = FxHashSet::with_capacity_and_hasher(n, Default::default());
        for node_id in current_selection {
            // For all graphs, get neighbors from adjacency list (handles undirected)
            if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                new_selection.extend(neighbors.iter().copied());
            }

            // For directed graphs, also get predecessors
            if self.directed {
                for edge in &self.edges {
                    if edge.target == node_id {
                        new_selection.insert(edge.source);
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
    #[inline]
    pub fn trav_out_edge(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let n = self.selected_nodes.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_nodes.iter().copied());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Clear node selection and build edge selection
        self.selected_nodes.clear();
        let mut new_edge_selection = FxHashSet::with_capacity_and_hasher(n, Default::default());

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
    #[inline]
    pub fn trav_in_edge(&mut self) -> PyResult<()> {
        // Get current selected nodes
        let n = self.selected_nodes.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_nodes.iter().copied());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Clear node selection and build edge selection
        self.selected_nodes.clear();
        let mut new_edge_selection = FxHashSet::with_capacity_and_hasher(n, Default::default());

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
    #[inline]
    pub fn trav_reverse_edge(&mut self, add_to_selection: bool) -> PyResult<()> {
        // Get current selected edges
        let n = self.selected_edges.len();
        let mut current_selection = Vec::with_capacity(n);
        current_selection.extend(self.selected_edges.iter().cloned());

        if current_selection.is_empty() {
            return Ok(()); // No selection, nothing to do
        }

        // Find reverse edges
        let mut reverse_edges = FxHashSet::with_capacity_and_hasher(n, Default::default());

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

    /// Count strongly connected components.
    /// For directed graphs: maximal sets where every node is reachable from every other.
    /// For undirected graphs: equivalent to connected components (same as weakly connected).
    pub fn count_strongly_connected_components(&self) -> PyResult<usize> {
        crate::algorithms::connectivity::count_strongly_connected_components(self)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Get all weakly connected components as sets of node IDs.
    /// For undirected graphs, this is the same as regular connected components.
    /// For directed graphs, ignores edge direction (treats as undirected).
    pub fn weakly_connected_components(&self) -> Vec<Vec<u64>> {
        crate::algorithms::connectivity::weakly_connected_components(self)
    }

    /// Get all strongly connected components as sets of node IDs.
    /// For directed graphs: maximal sets where every node is reachable from every other.
    /// For undirected graphs: equivalent to connected components (same as weakly connected).
    pub fn strongly_connected_components(&self) -> PyResult<Vec<Vec<u64>>> {
        crate::algorithms::connectivity::strongly_connected_components(self)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Check if a directed graph is strongly connected.
    /// A directed graph is strongly connected if there is a path from every vertex
    /// to every other vertex following directed edges.
    pub fn is_strongly_connected(&self) -> PyResult<bool> {
        crate::algorithms::connectivity::is_strongly_connected(self)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Check if a directed graph is weakly connected.
    /// A directed graph is weakly connected if the underlying undirected graph
    /// (ignoring edge directions) is connected.
    pub fn is_weakly_connected(&self) -> bool {
        crate::algorithms::connectivity::is_weakly_connected(self)
    }

    // ========== Trail Enumeration ==========

    /// Count all trails (walks with distinct edges) from source to target
    ///
    /// A trail is a walk where all edges are distinct, but nodes may repeat.
    /// This is different from:
    /// - Path: no repeated nodes or edges
    /// - Walk: nodes and edges may repeat freely
    ///
    /// Uses iterative DFS with edge tracking to avoid recursion limits.
    ///
    /// # Arguments
    /// * source - Starting node ID
    /// * target - Ending node ID
    /// * max_trails - Optional limit on number of trails to count (prevents explosion)
    ///
    /// # Returns
    /// * Number of distinct trails from source to target
    /// * Returns None if max_trails exceeded
    ///
    /// # Complexity
    /// * Can grow exponentially with graph size
    /// * Time: O(trails × avg_trail_length)
    /// * Space: O(edges) per trail in stack
    ///
    /// # Note
    /// * For undirected graphs, edges are considered bidirectional
    /// * Self-loops are traversed at most once per trail
    #[pyo3(signature = (source, target, max_trails=None))]
    fn count_trails(
        &self,
        source: u64,
        target: u64,
        max_trails: Option<usize>,
    ) -> PyResult<Option<usize>> {
        // Validate nodes exist
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} not found", source)
            ));
        }
        if !self.nodes.contains_key(&target) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node {} not found", target)
            ));
        }

        // If source == target, there's 1 trivial trail (empty path)
        if source == target {
            return Ok(Some(1));
        }

        let mut trail_count = 0usize;
        let limit = max_trails.unwrap_or(usize::MAX);

        // Stack stores: (current_node, visited_edges)
        // For undirected graphs, we store edges as (min, max) to handle both directions
        let mut stack: Vec<(u64, FxHashSet<(u64, u64)>)> = Vec::new();
        stack.push((source, FxHashSet::default()));

        while let Some((current, visited_edges)) = stack.pop() {
            // Get neighbors
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    // Create edge representation
                    let edge = if self.directed {
                        (current, neighbor)
                    } else {
                        // For undirected, normalize edge as (min, max)
                        if current < neighbor {
                            (current, neighbor)
                        } else {
                            (neighbor, current)
                        }
                    };

                    // Check if edge not visited
                    if !visited_edges.contains(&edge) {
                        // If we reached target, count this trail
                        if neighbor == target {
                            trail_count += 1;
                            if trail_count >= limit {
                                return Ok(None); // Exceeded limit
                            }
                        } else {
                            // Continue exploration
                            let mut new_visited = visited_edges.clone();
                            new_visited.insert(edge);
                            stack.push((neighbor, new_visited));
                        }
                    }
                }
            }
        }

        Ok(Some(trail_count))
    }

    /// Enumerate all trails (walks with distinct edges) from source to target
    ///
    /// Returns a list of trails, where each trail is a sequence of node IDs.
    ///
    /// # Arguments
    /// * source - Starting node ID
    /// * target - Ending node ID
    /// * max_trails - Optional limit on number of trails to return
    ///
    /// # Returns
    /// * Vec of trails, where each trail is Vec<u64> of node IDs
    /// * If max_trails exceeded, returns partial list
    ///
    /// # Warning
    /// * Can generate massive output for dense graphs
    /// * Use max_trails to limit memory usage
    #[pyo3(signature = (source, target, max_trails=None))]
    fn enumerate_trails(
        &self,
        source: u64,
        target: u64,
        max_trails: Option<usize>,
    ) -> PyResult<Vec<Vec<u64>>> {
        // Validate nodes exist
        if !self.nodes.contains_key(&source) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node {} not found", source)
            ));
        }
        if !self.nodes.contains_key(&target) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node {} not found", target)
            ));
        }

        let mut trails: Vec<Vec<u64>> = Vec::new();
        let limit = max_trails.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(trails);
        }

        // If source == target, return single empty trail
        if source == target {
            return Ok(vec![vec![source]]);
        }

        // Stack stores: (current_node, path, visited_edges)
        type TrailFrame = (u64, Vec<u64>, FxHashSet<(u64, u64)>);
        let mut stack: Vec<TrailFrame> = Vec::new();
        stack.push((source, vec![source], FxHashSet::default()));

        while let Some((current, path, visited_edges)) = stack.pop() {
            // Get neighbors
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    // Create edge representation
                    let edge = if self.directed {
                        (current, neighbor)
                    } else {
                        // For undirected, normalize edge as (min, max)
                        if current < neighbor {
                            (current, neighbor)
                        } else {
                            (neighbor, current)
                        }
                    };

                    // Check if edge not visited
                    if !visited_edges.contains(&edge) {
                        let mut new_path = path.clone();
                        new_path.push(neighbor);

                        // If we reached target, save this trail
                        if neighbor == target {
                            trails.push(new_path);
                            if trails.len() >= limit {
                                return Ok(trails); // Reached limit
                            }
                        } else {
                            // Continue exploration
                            let mut new_visited = visited_edges.clone();
                            new_visited.insert(edge);
                            stack.push((neighbor, new_path, new_visited));
                        }
                    }
                }
            }
        }

        Ok(trails)
    }

    // ========== Reciprocity (Mutual Edges) ==========

    /// Find pairs of mutual (reciprocal) edges in directed graph
    ///
    /// Returns edge IDs of edges that have a reciprocal counterpart.
    /// For an edge A→B to be mutual, there must exist an edge B→A.
    ///
    /// # Returns
    /// * Vec<(u64, u64)> - Vector of (edge_id1, edge_id2) pairs where each pair
    ///   represents mutual edges. edge_id1 < edge_id2 for consistency.
    ///
    /// # Errors
    /// * Returns error if graph is undirected (only meaningful for directed graphs)
    ///
    /// # Examples
    /// ```ignore
    /// let mut g = Graph::new(true, false);
    /// g.add_edge("A", "B");  // edge_id 1
    /// g.add_edge("B", "A");  // edge_id 2 - mutual with 1
    /// g.add_edge("A", "C");  // edge_id 3 - no reciprocal
    ///
    /// let mutual = g.which_mutual();
    /// // Returns [(1, 2)]
    /// ```
    #[inline]
    pub fn which_mutual(&self) -> PyResult<Vec<(u64, u64)>> {
        if !self.directed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "which_mutual() requires a directed graph"
            ));
        }

        let m = self.num_edges();
        let estimated_pairs = m / 10; // Estimate 10% mutual
        let mut mutual_pairs = Vec::with_capacity(estimated_pairs);
        let mut seen_pairs = FxHashSet::with_capacity_and_hasher(estimated_pairs, Default::default());

        // For each edge, check if reverse edge exists
        for edge in &self.edges {
            let source = edge.source;
            let target = edge.target;

            // Skip self-loops
            if source == target {
                continue;
            }

            // Create canonical pair representation (smaller node first)
            let canonical = if source < target {
                (source, target)
            } else {
                (target, source)
            };

            // Skip if we've already processed this pair
            if seen_pairs.contains(&canonical) {
                continue;
            }

            // Find all edges from source to target
            let forward_edges: Vec<u64> = self.edges
                .iter()
                .filter(|e| e.source == source && e.target == target)
                .map(|e| e.id)
                .collect();

            // Find all edges from target to source (reverse)
            let reverse_edges: Vec<u64> = self.edges
                .iter()
                .filter(|e| e.source == target && e.target == source)
                .map(|e| e.id)
                .collect();

            // If both directions exist, they are mutual
            if !forward_edges.is_empty() && !reverse_edges.is_empty() {
                // Mark this node pair as seen
                seen_pairs.insert(canonical);

                // Create pairs of mutual edges
                for &fwd_id in &forward_edges {
                    for &rev_id in &reverse_edges {
                        let pair = if fwd_id < rev_id {
                            (fwd_id, rev_id)
                        } else {
                            (rev_id, fwd_id)
                        };
                        mutual_pairs.push(pair);
                    }
                }
            }
        }

        Ok(mutual_pairs)
    }

    /// Calculate reciprocity (fraction of edges with reciprocal counterpart)
    ///
    /// Reciprocity measures the proportion of edges in a directed graph that are
    /// bidirectional (mutual). Returns a value between 0.0 and 1.0.
    ///
    /// The calculation depends on the mode:
    /// - "default": ratio of bidirectional connections to total connections
    /// - "ratio": (# mutual edges) / (total # edges)
    ///
    /// # Arguments
    /// * `mode` - Calculation mode: "default" or "ratio"
    ///
    /// # Returns
    /// * f64 - Reciprocity value between 0.0 (no mutual edges) and 1.0 (all mutual)
    ///
    /// # Errors
    /// * Returns error if graph is undirected
    /// * Returns error if mode is invalid
    ///
    /// # Examples
    /// ```ignore
    /// let mut g = Graph::new(true, false);
    /// g.add_edge("A", "B");
    /// g.add_edge("B", "A");  // Mutual with above
    /// g.add_edge("A", "C");  // No reciprocal
    ///
    /// let recip = g.reciprocity("ratio");
    /// // 2 mutual edges out of 3 total = 0.666...
    /// ```
    #[pyo3(signature = (mode="default"))]
    #[inline]
    pub fn reciprocity(&self, mode: &str) -> PyResult<f64> {
        if !self.directed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "reciprocity() requires a directed graph"
            ));
        }

        let num_edges = self.num_edges();
        if num_edges == 0 {
            return Ok(0.0);
        }

        match mode {
            "default" | "ratio" => {
                // Count edges that are mutual
                let mut mutual_count = 0;
                let estimated_pairs = num_edges / 2;
                let mut seen_pairs = FxHashSet::with_capacity_and_hasher(estimated_pairs, Default::default());

                for edge in &self.edges {
                    let source = edge.source;
                    let target = edge.target;

                    // Skip self-loops
                    if source == target {
                        continue;
                    }

                    // Create canonical pair
                    let canonical = if source < target {
                        (source, target)
                    } else {
                        (target, source)
                    };

                    // Skip if already counted
                    if seen_pairs.contains(&canonical) {
                        continue;
                    }

                    // Check if reverse edge exists
                    let has_forward = self.edges.iter().any(|e|
                        e.source == source && e.target == target
                    );
                    let has_reverse = self.edges.iter().any(|e|
                        e.source == target && e.target == source
                    );

                    if has_forward && has_reverse {
                        seen_pairs.insert(canonical);
                        // Count both directions as mutual
                        mutual_count += 2;
                    }
                }

                let inv_num_edges = 1.0 / num_edges as f64;
                Ok((mutual_count as f64) * inv_num_edges)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid mode '{}'. Use 'default' or 'ratio'", mode)
            )),
        }
    }

    /// Find indices of multiple edges (edges between the same node pairs)
    ///
    /// Returns a list of edge IDs that are duplicates - i.e., edges that share
    /// the same source and target nodes as another edge. For undirected graphs,
    /// edges (u,v) and (v,u) are considered the same pair.
    ///
    /// # Returns
    ///
    /// Vector of edge IDs representing multiple/duplicate edges
    ///
    /// # Examples
    ///
    /// ```python
    /// # Create graph with multiple edges between same nodes
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B", label="edge1")
    /// g.add_edge("A", "B", label="edge2")  # Duplicate
    /// g.add_edge("B", "C", label="edge3")
    /// g.add_edge("B", "C", label="edge4")  # Duplicate
    ///
    /// multiple = g.which_multiple()
    /// # Returns edge IDs of duplicates (e.g., [2, 4])
    /// ```
    #[inline]
    pub fn which_multiple(&self) -> PyResult<Vec<u64>> {
        let m = self.num_edges();
        let mut seen_pairs = FxHashMap::with_capacity_and_hasher(m, Default::default());
        let estimated_multiples = m / 10; // Estimate 10% are multiples
        let mut multiple_edges = Vec::with_capacity(estimated_multiples);

        // Iterate through all edges
        for edge in &self.edges {
            let source = edge.source;
            let target = edge.target;
            let edge_id = edge.id;

            // For undirected graphs, normalize the pair (smaller node ID first)
            let pair = if self.directed || source <= target {
                (source, target)
            } else {
                (target, source)
            };

            // Check if we've seen this pair before
            if let Some(&first_edge_id) = seen_pairs.get(&pair) {
                // This is a multiple edge - add to results if not already there
                if !multiple_edges.contains(&first_edge_id) {
                    multiple_edges.push(first_edge_id);
                }
                multiple_edges.push(edge_id);
            } else {
                // First time seeing this pair
                seen_pairs.insert(pair, edge_id);
            }
        }

        Ok(multiple_edges)
    }

    /// Count multiple edges grouped by node pair
    ///
    /// Returns a dictionary mapping node pairs to the count of edges between them.
    /// Only includes pairs with 2 or more edges. For undirected graphs, returns
    /// unordered pairs (smaller node ID first).
    ///
    /// # Returns
    ///
    /// Dictionary mapping (source_id, target_id) tuples to edge counts
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B")
    /// g.add_edge("A", "B")  # Second edge
    /// g.add_edge("A", "B")  # Third edge
    /// g.add_edge("B", "C")  # Single edge
    ///
    /// counts = g.count_multiple()
    /// # Returns node IDs: {(node_id_a, node_id_b): 3}
    /// # Single edges like (node_id_b, node_id_c) are not included
    /// ```
    #[inline]
    pub fn count_multiple(&self) -> PyResult<std::collections::HashMap<(u64, u64), usize>> {
        let m = self.num_edges();
        let mut pair_counts = FxHashMap::with_capacity_and_hasher(m, Default::default());

        // Count edges between each node pair
        for edge in &self.edges {
            let source = edge.source;
            let target = edge.target;

            // For undirected graphs, normalize the pair
            let pair = if self.directed || source <= target {
                (source, target)
            } else {
                (target, source)
            };

            *pair_counts.entry(pair).or_insert(0) += 1;
        }

        // Filter for multiples only (count >= 2) and return node IDs
        let estimated_result_size = pair_counts.len() / 2;
        let mut result = std::collections::HashMap::with_capacity(estimated_result_size);
        for ((source_id, target_id), count) in pair_counts.iter() {
            if *count >= 2 {
                result.insert((*source_id, *target_id), *count);
            }
        }

        Ok(result)
    }

    /// Find indices of self-loop edges (edges from a node to itself)
    ///
    /// Returns a list of edge IDs for all edges where source equals target.
    ///
    /// # Returns
    ///
    /// Vector of edge IDs representing self-loops
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B")
    /// g.add_edge("A", "A")  # Self-loop
    /// g.add_edge("B", "C")
    /// g.add_edge("C", "C")  # Self-loop
    ///
    /// loops = g.which_loops()
    /// # Returns edge IDs of self-loops (e.g., [2, 4])
    /// ```
    #[inline]
    pub fn which_loops(&self) -> PyResult<Vec<u64>> {
        let m = self.num_edges();
        let estimated_loops = m / 20; // Estimate 5% are loops
        let mut loop_edges = Vec::with_capacity(estimated_loops);

        for edge in &self.edges {
            if edge.source == edge.target {
                loop_edges.push(edge.id);
            }
        }

        Ok(loop_edges)
    }

    /// Count self-loops grouped by node
    ///
    /// Returns a dictionary mapping node labels to the count of self-loops for that node.
    /// Only includes nodes with at least one self-loop.
    ///
    /// # Returns
    ///
    /// Dictionary mapping node IDs to self-loop counts
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B")
    /// g.add_edge("A", "A")  # Self-loop
    /// g.add_edge("A", "A")  # Another self-loop
    /// g.add_edge("B", "B")  # Self-loop
    ///
    /// loops = g.count_loops()
    /// # Returns: {a_id: 2, b_id: 1} where a_id, b_id are node IDs
    /// ```
    #[inline]
    pub fn count_loops(&self) -> PyResult<std::collections::HashMap<u64, usize>> {
        let n = self.num_nodes();
        let mut loop_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::with_capacity(n);

        // Count self-loops per node
        for edge in &self.edges {
            if edge.source == edge.target {
                *loop_counts.entry(edge.source).or_insert(0) += 1;
            }
        }

        Ok(loop_counts)
    }

    /// Remove all self-loops from the graph
    ///
    /// A self-loop is an edge where source == target. This method removes all such edges.
    ///
    /// # Returns
    /// * Number of self-loops removed
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B")
    /// g.add_edge("A", "A")  # Self-loop
    /// g.add_edge("B", "B")  # Self-loop
    ///
    /// removed = g.remove_selfloops()
    /// # removed == 2
    /// ```
    pub fn remove_selfloops(&mut self) -> PyResult<usize> {
        // Find all self-loop edge IDs
        let loop_ids: Vec<u64> = self.edges
            .iter()
            .filter(|e| e.source == e.target)
            .map(|e| e.id)
            .collect();

        let count = loop_ids.len();

        // Remove from adjacency list and edges
        for edge_id in &loop_ids {
            if let Some(idx) = self.edges.iter().position(|e| e.id == *edge_id) {
                let edge = &self.edges[idx];
                let node = edge.source;

                // Remove from adjacency list (self-loop appears once for undirected, once for directed)
                if let Some(neighbors) = self.adjacency_list.get_mut(&node) {
                    if let Some(pos) = neighbors.iter().position(|n| *n == node) {
                        neighbors.remove(pos);
                    }
                }

                // Remove from selection if selected
                self.selected_edges.remove(edge_id);

                // Remove edge
                self.edges.remove(idx);
                self.invalidate_edge_index();
            }
        }

        Ok(count)
    }

    /// Remove duplicate edges, keeping only one edge per node pair
    ///
    /// For each pair of nodes with multiple edges, keeps the first edge (lowest ID)
    /// and removes all others.
    ///
    /// # Arguments
    /// * `keep_distinct_attrs` - If true, edges with different attributes are kept
    ///
    /// # Returns
    /// * Number of duplicate edges removed
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B", weight=1.0)
    /// g.add_edge("A", "B", weight=2.0)  # Duplicate
    /// g.add_edge("A", "B", weight=3.0)  # Duplicate
    ///
    /// removed = g.remove_duplicate_edges(False)
    /// # removed == 2
    /// ```
    pub fn remove_duplicate_edges(&mut self, keep_distinct_attrs: bool) -> PyResult<usize> {
        use std::collections::BTreeMap;

        let m = self.num_edges();
        let mut edges_to_remove: Vec<u64> = Vec::with_capacity(m / 10);

        if keep_distinct_attrs {
            // Group by (node_pair, attribute_signature)
            // Use BTreeMap for deterministic ordering
            let mut seen: FxHashMap<(u64, u64), BTreeMap<String, u64>> = FxHashMap::default();

            for edge in &self.edges {
                let pair = if self.directed {
                    (edge.source, edge.target)
                } else {
                    (edge.source.min(edge.target), edge.source.max(edge.target))
                };

                // Create attribute signature
                let mut sig_parts: Vec<String> = Vec::new();

                // Include weight in signature
                if let Some(w) = edge.weight {
                    sig_parts.push(format!("w:{}", w));
                }

                // Include sorted attributes (need GIL to access PyDict)
                if let Some(ref attrs) = edge.attributes {
                    Python::with_gil(|py| {
                        let dict = attrs.bind(py);
                        let mut attr_pairs: Vec<(String, String)> = dict.iter()
                            .map(|(k, v)| (k.to_string(), format!("{:?}", v)))
                            .collect();
                        attr_pairs.sort_by(|a, b| a.0.cmp(&b.0));
                        for (k, v) in attr_pairs {
                            sig_parts.push(format!("{}:{}", k, v));
                        }
                    });
                }

                let signature = sig_parts.join("|");

                let pair_map = seen.entry(pair).or_default();
                if let Some(&first_id) = pair_map.get(&signature) {
                    // Duplicate with same attributes - mark for removal
                    if edge.id != first_id {
                        edges_to_remove.push(edge.id);
                    }
                } else {
                    pair_map.insert(signature, edge.id);
                }
            }
        } else {
            // Simple mode: just look at node pairs
            let mut seen: FxHashMap<(u64, u64), u64> = FxHashMap::default();

            for edge in &self.edges {
                let pair = if self.directed {
                    (edge.source, edge.target)
                } else {
                    (edge.source.min(edge.target), edge.source.max(edge.target))
                };

                if let Some(&first_id) = seen.get(&pair) {
                    // Duplicate - mark for removal (keep first)
                    if edge.id != first_id {
                        edges_to_remove.push(edge.id);
                    }
                } else {
                    seen.insert(pair, edge.id);
                }
            }
        }

        let count = edges_to_remove.len();

        // Remove edges (in reverse order to maintain indices, though we use IDs)
        for edge_id in &edges_to_remove {
            // Find and remove edge
            if let Some(idx) = self.edges.iter().position(|e| e.id == *edge_id) {
                let edge = &self.edges[idx];

                // Remove from adjacency list
                if let Some(neighbors) = self.adjacency_list.get_mut(&edge.source) {
                    if let Some(pos) = neighbors.iter().position(|n| *n == edge.target) {
                        neighbors.remove(pos);
                    }
                }

                if !self.directed {
                    if let Some(neighbors) = self.adjacency_list.get_mut(&edge.target) {
                        if let Some(pos) = neighbors.iter().position(|n| *n == edge.source) {
                            neighbors.remove(pos);
                        }
                    }
                }

                // Remove from selection if selected
                self.selected_edges.remove(edge_id);

                // Remove edge
                self.edges.remove(idx);
                self.invalidate_edge_index();
            }
        }

        Ok(count)
    }

    /// Simplify the graph by removing self-loops and duplicate edges
    ///
    /// Equivalent to calling remove_selfloops() followed by remove_duplicate_edges().
    ///
    /// # Returns
    /// * Tuple of (selfloops_removed, duplicates_removed)
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph(directed=True)
    /// g.add_edge("A", "B")
    /// g.add_edge("A", "B")  # Duplicate
    /// g.add_edge("A", "A")  # Self-loop
    ///
    /// loops, dupes = g.simplify()
    /// # loops == 1, dupes == 1
    /// ```
    pub fn simplify(&mut self) -> PyResult<(usize, usize)> {
        let loops_removed = self.remove_selfloops()?;
        let dupes_removed = self.remove_duplicate_edges(false)?;
        Ok((loops_removed, dupes_removed))
    }

    // ========== Assortativity ==========

    /// Calculate degree assortativity coefficient
    ///
    /// Measures the tendency of nodes to connect to other nodes with similar degrees.
    /// Returns Pearson correlation coefficient of node degrees across edges.
    ///
    /// Computes node degrees and uses them as numeric attributes in the
    /// assortativity() function.
    ///
    /// # Returns
    /// * f64 - Assortativity coefficient in range [-1, 1]
    ///   - r > 0: assortative (high-degree nodes connect to high-degree nodes)
    ///   - r < 0: disassortative (high-degree nodes connect to low-degree nodes)
    ///   - r ≈ 0: no degree correlation
    ///
    /// # Errors
    /// * Returns error if graph has no edges or insufficient variance
    ///
    /// # Examples
    ///
    /// ```python
    /// # Assortative network (similar degrees connect)
    /// g = Graph(directed=False)
    /// g.add_edge("Hub1", "Hub2")  # High-degree hubs connect
    /// g.add_edge("Hub1", "Hub3")
    /// g.add_edge("Hub2", "Hub3")
    /// g.add_edge("Leaf1", "Leaf2")  # Low-degree leaves connect
    ///
    /// r = g.assortativity_degree()
    /// # Returns positive value (assortative)
    /// ```
    ///
    /// # References
    /// Newman, M.E.J. "Assortative mixing in networks" (2002)
    #[inline]
    pub fn assortativity_degree(&self) -> PyResult<f64> {
        use rayon::prelude::*;

        let m = self.edges.len();
        if m == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot calculate assortativity for graph with no edges"
            ));
        }

        // Calculate degrees for all nodes
        // For undirected graphs, this is total degree
        // For directed graphs, we use out-degree for source and in-degree for target

        let (degree_source, degree_target): (Vec<f64>, Vec<f64>) = if self.directed {
            // Directed: out-degree for source, in-degree for target
            let n = self.nodes.len();
            let mut out_degrees: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
            let mut in_degrees: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());

            // Build degree maps
            for edge in &self.edges {
                *out_degrees.entry(edge.source).or_insert(0) += 1;
                *in_degrees.entry(edge.target).or_insert(0) += 1;
            }

            // Build degree vectors with pre-allocated capacity
            let mut sources: Vec<f64> = Vec::with_capacity(m);
            let mut targets: Vec<f64> = Vec::with_capacity(m);

            for edge in &self.edges {
                // Use unchecked get since we know keys exist
                sources.push(*out_degrees.get(&edge.source).unwrap() as f64);
                targets.push(*in_degrees.get(&edge.target).unwrap() as f64);
            }

            (sources, targets)
        } else {
            // Undirected: total degree for both ends
            let n = self.nodes.len();
            let mut degrees: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());

            // Build degree map
            for edge in &self.edges {
                *degrees.entry(edge.source).or_insert(0) += 1;
                *degrees.entry(edge.target).or_insert(0) += 1;
            }

            // Build degree vectors with pre-allocated capacity
            let mut sources: Vec<f64> = Vec::with_capacity(m);
            let mut targets: Vec<f64> = Vec::with_capacity(m);

            for edge in &self.edges {
                // Use unchecked get since we know keys exist
                sources.push(*degrees.get(&edge.source).unwrap() as f64);
                targets.push(*degrees.get(&edge.target).unwrap() as f64);
            }

            (sources, targets)
        };

        // Compute sums in parallel for large graphs
        let m_f64 = m as f64;

        let (num1, sum_x, sum_y, sum_x2, sum_y2) = if m >= 10000 {
            // Parallel computation for large graphs
            degree_source.par_iter()
                .zip(degree_target.par_iter())
                .map(|(x, y)| {
                    (*x * *y, *x, *y, *x * *x, *y * *y)
                })
                .reduce(
                    || (0.0, 0.0, 0.0, 0.0, 0.0),
                    |acc, item| {
                        (acc.0 + item.0, acc.1 + item.1, acc.2 + item.2, acc.3 + item.3, acc.4 + item.4)
                    }
                )
        } else {
            // Sequential for small graphs
            let mut num1: f64 = 0.0;
            let mut sum_x: f64 = 0.0;
            let mut sum_y: f64 = 0.0;
            let mut sum_x2: f64 = 0.0;
            let mut sum_y2: f64 = 0.0;

            for (x, y) in degree_source.iter().zip(degree_target.iter()) {
                num1 += x * y;
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }

            (num1, sum_x, sum_y, sum_x2, sum_y2)
        };

        // Apply undirected/directed normalization (same as assortativity())
        let (numerator, denominator) = if !self.directed {
            // Undirected: special formula
            let num2 = (sum_x + sum_y) / (m_f64 * 2.0);
            let den1 = (sum_x2 + sum_y2) / (m_f64 * 2.0);
            let num1_normalized = num1 / m_f64;

            let num = num1_normalized - num2 * num2;
            let den = den1 - num2 * num2;
            (num, den)
        } else {
            // Directed: standard Pearson correlation
            let mean_x = sum_x / m_f64;
            let mean_y = sum_y / m_f64;

            let num = num1 / m_f64 - mean_x * mean_y;
            let var_x = sum_x2 / m_f64 - mean_x * mean_x;
            let var_y = sum_y2 / m_f64 - mean_y * mean_y;
            let den = (var_x * var_y).sqrt();

            (num, den)
        };

        // Handle zero variance case (all nodes have same degree) - return NaN
        if denominator.abs() < 1e-10 {
            return Ok(f64::NAN);
        }

        Ok(numerator / denominator)
    }

    // ========== Node Similarity ==========

    /// Calculate pairwise similarity between nodes based on their neighborhoods
    ///
    /// Computes structural similarity using various metrics that compare the neighborhoods
    /// (adjacent nodes) of node pairs. Higher values indicate more similar neighborhood
    /// structure, useful for link prediction and community detection.
    ///
    /// # Arguments
    /// * `nodes` - List of node labels to compute similarities for. If empty, computes for all nodes.
    /// * `method` - Similarity metric: "jaccard", "dice", "invlogweighted", "cosine"
    ///
    /// # Returns
    /// * Vec<Vec<f64>> - Similarity matrix where entry (i,j) is similarity between nodes[i] and nodes[j]
    ///
    /// # Similarity Metrics
    ///
    /// ## Jaccard Similarity
    /// ```text
    /// sim(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    /// ```
    /// - Range: [0, 1]
    /// - 1 = identical neighborhoods, 0 = no common neighbors
    ///
    /// ## Dice Similarity
    /// ```text
    /// sim(u,v) = 2|N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
    /// ```
    /// - Range: [0, 1]
    /// - Gives more weight to common neighbors than Jaccard
    ///
    /// ## Inverse Log-Weighted Similarity
    /// ```text
    /// sim(u,v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(|N(w)|)
    /// ```
    /// - Higher weight for rare neighbors (low degree)
    /// - Used in link prediction (Adamic-Adar index)
    ///
    /// ## Cosine Similarity
    /// ```text
    /// sim(u,v) = |N(u) ∩ N(v)| / sqrt(|N(u)| × |N(v)|)
    /// ```
    /// - Range: [0, 1]
    /// - Normalized dot product of adjacency vectors
    ///
    /// # Examples
    ///
    /// ```python
    /// # Link prediction: find similar nodes
    /// g = Graph(directed=False)
    /// g.add_edges([("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")])
    ///
    /// sim = g.get_pairwise_similarity(["A", "B"], method="jaccard")
    /// # A and B have identical neighborhoods → high similarity
    /// ```
    ///
    /// # References
    /// - Liben-Nowell & Kleinberg (2007). "The link-prediction problem for social networks"
    /// - Adamic & Adar (2003). "Friends and neighbors on the Web"
    pub fn get_pairwise_similarity(
        &self,
        py: Python,
        nodes: Vec<PyObject>,
        method: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        // Resolve nodes (can be IDs or labels) to IDs
        let node_ids: Vec<u64> = if nodes.is_empty() {
            // Use all nodes
            self.nodes.keys().copied().collect()
        } else {
            nodes
                .iter()
                .map(|node| {
                    // Try as integer (node ID)
                    if let Ok(id) = node.extract::<u64>(py) {
                        if self.nodes.contains_key(&id) {
                            Ok(id)
                        } else {
                            Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Node with ID {} not found",
                                id
                            )))
                        }
                    }
                    // Try as string (node label)
                    else if let Ok(label) = node.extract::<String>(py) {
                        self.node_labels
                            .get(&label)
                            .copied()
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Node with label '{}' not found",
                                    label
                                ))
                            })
                    } else {
                        Err(pyo3::exceptions::PyValueError::new_err(
                            "Node must be an integer (ID) or string (label)",
                        ))
                    }
                })
                .collect::<PyResult<Vec<u64>>>()?
        };

        let n = node_ids.len();
        let mut sim_matrix = vec![vec![0.0; n]; n];

        // Build neighborhood sets for each node
        let mut neighborhoods: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::default();

        for &node_id in &node_ids {
            let mut neighbors = FxHashSet::default();

            for edge in &self.edges {
                if edge.source == node_id {
                    neighbors.insert(edge.target);
                }
                if !self.directed && edge.target == node_id {
                    neighbors.insert(edge.source);
                }
            }

            neighborhoods.insert(node_id, neighbors);
        }

        // Compute pairwise similarities
        match method {
            "jaccard" => {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            sim_matrix[i][j] = 1.0;
                        } else {
                            let n_i = &neighborhoods[&node_ids[i]];
                            let n_j = &neighborhoods[&node_ids[j]];

                            let intersection: usize = n_i.intersection(n_j).count();
                            let union: usize = n_i.union(n_j).count();

                            sim_matrix[i][j] = if union > 0 {
                                intersection as f64 / union as f64
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            "dice" => {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            sim_matrix[i][j] = 1.0;
                        } else {
                            let n_i = &neighborhoods[&node_ids[i]];
                            let n_j = &neighborhoods[&node_ids[j]];

                            let intersection: usize = n_i.intersection(n_j).count();
                            let sum_sizes = n_i.len() + n_j.len();

                            sim_matrix[i][j] = if sum_sizes > 0 {
                                (2.0 * intersection as f64) / sum_sizes as f64
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            "invlogweighted" => {
                // Compute degree for all nodes (needed for weighting)
                let mut degrees: FxHashMap<u64, usize> = FxHashMap::default();
                for edge in &self.edges {
                    *degrees.entry(edge.source).or_insert(0) += 1;
                    if !self.directed {
                        *degrees.entry(edge.target).or_insert(0) += 1;
                    }
                }

                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            sim_matrix[i][j] = 1.0;
                        } else {
                            let n_i = &neighborhoods[&node_ids[i]];
                            let n_j = &neighborhoods[&node_ids[j]];

                            let mut weighted_sum = 0.0;
                            for common_neighbor in n_i.intersection(n_j) {
                                let degree = *degrees.get(common_neighbor).unwrap_or(&1);
                                if degree > 1 {
                                    weighted_sum += 1.0 / (degree as f64).ln();
                                }
                            }

                            sim_matrix[i][j] = weighted_sum;
                        }
                    }
                }
            }
            "cosine" => {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            sim_matrix[i][j] = 1.0;
                        } else {
                            let n_i = &neighborhoods[&node_ids[i]];
                            let n_j = &neighborhoods[&node_ids[j]];

                            let intersection: usize = n_i.intersection(n_j).count();
                            let denominator = ((n_i.len() * n_j.len()) as f64).sqrt();

                            sim_matrix[i][j] = if denominator > 0.0 {
                                intersection as f64 / denominator
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown similarity method '{}'. Choose from: jaccard, dice, invlogweighted, cosine",
                    method
                )));
            }
        }

        Ok(sim_matrix)
    }

    /// Calculate assortativity coefficient for nominal (categorical) node attributes
    ///
    /// Measures the tendency of nodes with the same attribute value to connect.
    /// Uses Newman's formula for categorical attributes.
    ///
    /// Formula:
    /// r = (Σᵢ eᵢᵢ - Σᵢ aᵢbᵢ) / (1 - Σᵢ aᵢbᵢ)
    ///
    /// where:
    /// - eᵢⱼ = fraction of edges connecting attribute i to attribute j
    /// - aᵢ = fraction of edges with source node having attribute i
    /// - bᵢ = fraction of edges with target node having attribute i
    ///
    /// # Arguments
    /// * `attribute` - Name of the categorical node attribute to analyze
    ///
    /// # Returns
    /// * f64 - Assortativity coefficient in range [-1, 1]
    ///   - r > 0: assortative (same types connect)
    ///   - r < 0: disassortative (different types connect)
    ///   - r = 1: perfect assortativity (only same-type connections)
    ///
    /// # Errors
    /// * Returns error if attribute doesn't exist on nodes or graph has no edges
    ///
    /// # Examples
    ///
    /// ```python
    /// # Social network with user types
    /// g = Graph(directed=False)
    /// g.add_node("Alice", type="student")
    /// g.add_node("Bob", type="student")
    /// g.add_node("Prof1", type="professor")
    ///
    /// g.add_edge("Alice", "Bob")  # student-student
    /// g.add_edge("Alice", "Prof1")  # student-professor
    ///
    /// r = g.get_assortativity_nominal("type")
    /// # Positive if students mostly connect to students
    /// ```
    ///
    /// # References
    /// Newman, M.E.J. "Mixing patterns in networks" (2003)
    #[inline]
    pub fn get_assortativity_nominal(&self, attribute: &str) -> PyResult<f64> {
        let m = self.edges.len();
        if m == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot calculate assortativity for graph with no edges"
            ));
        }

        Python::with_gil(|py| {
            // Extract attribute values for all nodes
            let n = self.nodes.len();
            let mut node_attrs = FxHashMap::with_capacity_and_hasher(n, Default::default());

            for (node_id, node) in &self.nodes {
                if let Some(ref attrs) = node.attributes {
                    let dict = attrs.bind(py);
                    if let Ok(Some(value)) = dict.get_item(attribute) {
                        let attr_str = value.to_string();
                        node_attrs.insert(*node_id, attr_str);
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("Attribute '{}' not found on node {}", attribute, node_id)
                        ));
                    }
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Node {} has no attributes", node_id)
                    ));
                }
            }

            // Build mixing matrix: count edges between attribute types
            let mut attribute_types = Vec::with_capacity(n);
            attribute_types.extend(node_attrs.values().cloned());
            attribute_types.sort();
            attribute_types.dedup();

            let n_types = attribute_types.len();
            if n_types < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Need at least 2 different attribute values to calculate assortativity"
                ));
            }

            // Map attribute values to indices
            let mut attr_to_idx = FxHashMap::with_capacity_and_hasher(n_types, Default::default());
            for (idx, attr) in attribute_types.iter().enumerate() {
                attr_to_idx.insert(attr.clone(), idx);
            }

            // Count edges for mixing matrix
            let mut e_matrix = vec![vec![0usize; n_types]; n_types];

            for edge in &self.edges {
                let src_attr = node_attrs.get(&edge.source)
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                        format!("Missing attribute for node {}", edge.source)
                    ))?;
                let tgt_attr = node_attrs.get(&edge.target)
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                        format!("Missing attribute for node {}", edge.target)
                    ))?;

                let i = attr_to_idx[src_attr];
                let j = attr_to_idx[tgt_attr];

                // For undirected graphs, each edge contributes to both (i,j) and (j,i)
                // This makes the mixing matrix symmetric
                e_matrix[i][j] += 1;
                if !self.directed {
                    e_matrix[j][i] += 1;
                }
            }

            // Calculate assortativity using Newman's formula
            let m_f64 = m as f64;

            // Calculate eᵢⱼ (edge fractions)
            // For undirected: each edge counted twice in e_matrix, so divide by 2*m
            // For directed: edges counted once, so divide by m
            let inv_normalization = if self.directed { 1.0 / m_f64 } else { 1.0 / (2.0 * m_f64) };
            let e_ij: Vec<Vec<f64>> = e_matrix.iter()
                .map(|row| row.iter().map(|&count| count as f64 * inv_normalization).collect())
                .collect();

            // Calculate aᵢ and bᵢ (marginal sums)
            let a_i: Vec<f64> = (0..n_types)
                .map(|i| (0..n_types).map(|j| e_ij[i][j]).sum())
                .collect();

            let b_i: Vec<f64> = (0..n_types)
                .map(|j| (0..n_types).map(|i| e_ij[i][j]).sum())
                .collect();

            // Σᵢ eᵢᵢ (trace - same-type connections)
            let trace: f64 = (0..n_types).map(|i| e_ij[i][i]).sum();

            // Σᵢ aᵢbᵢ (expected same-type connections)
            let sum_ab: f64 = (0..n_types).map(|i| a_i[i] * b_i[i]).sum();

            let numerator = trace - sum_ab;
            let denominator = 1.0 - sum_ab;

            if denominator.abs() < 1e-10 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Insufficient mixing to calculate assortativity"
                ));
            }

            Ok(numerator / denominator)
        })
    }

    /// Calculate assortativity coefficient for numeric node attributes
    ///
    /// Measures the tendency of nodes with similar numeric attribute values to connect.
    /// Returns Pearson correlation coefficient of attribute values across edges.
    ///
    /// Formula:
    /// r = (Σᵢ xᵢyᵢ - Σᵢ xᵢ Σᵢ yᵢ / M) / √[(Σᵢ xᵢ² - (Σᵢ xᵢ)² / M)(Σᵢ yᵢ² - (Σᵢ yᵢ)² / M)]
    ///
    /// where:
    /// - xᵢ, yᵢ = attribute values of nodes at ends of edge i
    /// - M = number of edges
    ///
    /// # Arguments
    /// * `attribute` - Name of the numeric node attribute to analyze
    ///
    /// # Returns
    /// * f64 - Assortativity coefficient in range [-1, 1]
    ///   - r > 0: assortative (similar values connect)
    ///   - r < 0: disassortative (dissimilar values connect)
    ///   - r ≈ 0: no correlation
    ///
    /// # Errors
    /// * Returns error if attribute doesn't exist, isn't numeric, or has no variance
    ///
    /// # Examples
    ///
    /// ```python
    /// # Network with age-based connections
    /// g = Graph(directed=False)
    /// g.add_node("Alice", age=25)
    /// g.add_node("Bob", age=27)
    /// g.add_node("Carol", age=55)
    ///
    /// g.add_edge("Alice", "Bob")  # Similar ages
    /// g.add_edge("Carol", "Carol")  # Self-loop
    ///
    /// r = g.assortativity("age")
    /// # Positive if similar ages tend to connect
    /// ```
    ///
    /// # References
    /// Newman, M.E.J. "Assortative mixing in networks" (2002)
    #[inline]
    pub fn assortativity(&self, attribute: &str) -> PyResult<f64> {
        let m = self.edges.len();
        if m == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot calculate assortativity for graph with no edges"
            ));
        }

        Python::with_gil(|py| {
            // Extract numeric attribute values for all nodes
            let n = self.nodes.len();
            let mut node_attrs = FxHashMap::with_capacity_and_hasher(n, Default::default());

            for (node_id, node) in &self.nodes {
                if let Some(ref attrs) = node.attributes {
                    let dict = attrs.bind(py);
                    if let Ok(Some(value)) = dict.get_item(attribute) {
                        // Try to extract as float
                        let attr_val: f64 = value.extract()
                            .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                                format!("Attribute '{}' on node {} is not numeric", attribute, node_id)
                            ))?;
                        node_attrs.insert(*node_id, attr_val);
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("Attribute '{}' not found on node {}", attribute, node_id)
                        ));
                    }
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Node {} has no attributes", node_id)
                    ));
                }
            }

            // Extract attribute values for source and target of each edge
            let mut x_vals: Vec<f64> = Vec::with_capacity(m);
            let mut y_vals: Vec<f64> = Vec::with_capacity(m);

            for edge in &self.edges {
                let x = *node_attrs.get(&edge.source)
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                        format!("Missing attribute for node {}", edge.source)
                    ))?;
                let y = *node_attrs.get(&edge.target)
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                        format!("Missing attribute for node {}", edge.target)
                    ))?;

                x_vals.push(x);
                y_vals.push(y);
            }

            // Calculate Pearson correlation coefficient
            let m_f64 = m as f64;

            if !self.directed {
                // Undirected graph - special handling
                let inv_m = 1.0 / m_f64;
                let inv_2m = 1.0 / (m_f64 * 2.0);
                let mut num1 = 0.0;
                let mut num2 = 0.0;
                let mut den1 = 0.0;

                for (x, y) in x_vals.iter().zip(y_vals.iter()) {
                    num1 += x * y;
                    num2 += x + y;
                    den1 += x * x + y * y;
                }

                num1 *= inv_m;
                den1 *= inv_2m;
                num2 *= inv_2m;
                let num2_sq = num2 * num2;

                let numerator = num1 - num2_sq;
                let denominator = den1 - num2_sq;

                if denominator.abs() < 1e-10 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Insufficient variance in attribute values to calculate assortativity"
                    ));
                }

                Ok(numerator / denominator)
            } else {
                // Directed graph
                let inv_m = 1.0 / m_f64;
                let sum_x: f64 = x_vals.iter().sum();
                let sum_y: f64 = y_vals.iter().sum();
                let sum_xy: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * y).sum();
                let sum_x2: f64 = x_vals.iter().map(|x| x * x).sum();
                let sum_y2: f64 = y_vals.iter().map(|y| y * y).sum();

                let numerator = sum_xy - (sum_x * sum_y) * inv_m;
                let denominator_x = sum_x2 - (sum_x * sum_x) * inv_m;
                let denominator_y = sum_y2 - (sum_y * sum_y) * inv_m;
                let denominator = (denominator_x * denominator_y).sqrt();

                if denominator.abs() < 1e-10 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Insufficient variance in attribute values to calculate assortativity"
                    ));
                }

                Ok(numerator / denominator)
            }
        })
    }

    // ========== Graph Spectrum ==========

    /// Compute the spectrum of the adjacency matrix
    ///
    /// Matrix rows/columns follow node creation order, returned as "nodes" so that
    /// `vectors[i][j]` is the entry of eigenvector `i` for node `nodes[j]`. Parallel edges
    /// sum their weights and an undirected self-loop contributes its weight once to the
    /// diagonal (the `networkx.to_scipy_sparse_array` convention).
    ///
    /// Undirected graphs use a symmetric eigensolver and return real eigenvalues with
    /// eigenvectors. A directed graph's adjacency matrix is not symmetric, so its
    /// eigenvalues come from a general (Francis QR) solver and are returned as complex numbers;
    /// "vectors" and "spectral_gap" are None in that case.
    ///
    /// # Arguments
    /// * `which` - Which eigenvalues to compute:
    ///   - "all": All eigenvalues (dense O(n³))
    ///   - "largest": Largest k eigenvalues (by real part)
    ///   - "smallest": Smallest k eigenvalues (by real part)
    /// * `k` - Number of eigenvalues for "largest"/"smallest" (default min(6, n))
    ///
    /// # Returns
    /// Dictionary with:
    /// - "values": Eigenvalues sorted descending (by real part for directed graphs)
    /// - "vectors": Unit eigenvectors aligned with "values" (None for directed graphs)
    /// - "nodes": Node IDs giving the order of eigenvector entries
    /// - "spectral_gap": λ₁ - λ₂ over the full spectrum (None if n < 2 or directed)
    ///
    /// # Errors
    /// - Empty graph
    /// - Invalid `which`
    /// - Eigenvalue computation failed to converge (directed graphs)
    #[pyo3(signature = (which="largest", k=None))]
    pub fn spectrum(&self, py: Python, which: &str, k: Option<usize>) -> PyResult<PyObject> {
        let count = spectral_count(which, k, self.nodes.len())?;
        let (node_ids, adj) = self.spectral_adjacency("spectrum")?;

        let result = PyDict::new(py);
        result.set_item("nodes", &node_ids)?;

        if self.directed {
            let mut values = spectral_complex_eigenvalues(adj.to_dense())?;
            values.sort_by(|a, b| b.re.total_cmp(&a.re).then(b.im.total_cmp(&a.im)));
            let selected = spectral_select(values, which, count, true);
            result.set_item("values", spectral_complex_list(py, &selected)?)?;
            result.set_item("vectors", py.None())?;
            result.set_item("spectral_gap", py.None())?;
        } else {
            let (pairs, top) = spectral_symmetric(&adj, which, count, true)?;
            let gap = (top.len() >= 2).then(|| top[0] - top[1]);
            let (values, vectors): (Vec<f64>, Vec<Vec<f64>>) = pairs.into_iter().unzip();
            result.set_item("values", values)?;
            result.set_item("vectors", vectors)?;
            result.set_item("spectral_gap", gap)?;
        }

        Ok(result.into())
    }

    /// Compute the spectrum of the graph Laplacian
    ///
    /// Uses `L = D - A`, where `A` is built as in `spectrum()` and `D` holds the row sums of
    /// `A` (out-degrees for directed graphs). Self-loops therefore cancel out of `L`. With
    /// `normalized=true`, returns the spectrum of `D^(-1/2) L D^(-1/2)`, with isolated nodes
    /// contributing a zero row and column. Both conventions match
    /// `networkx.laplacian_matrix` / `networkx.normalized_laplacian_matrix`.
    ///
    /// Directed graphs give a non-symmetric Laplacian whose eigenvalues are returned as
    /// complex numbers; "vectors" and "algebraic_connectivity" are None in that case.
    ///
    /// # Arguments
    /// * `which` - Which eigenvalues to compute:
    ///   - "all": All eigenvalues (dense O(n³))
    ///   - "largest": Largest k eigenvalues (by real part)
    ///   - "smallest": Smallest k eigenvalues (by real part)
    /// * `k` - Number of eigenvalues for "largest"/"smallest" (default min(6, n))
    /// * `normalized` - Use the symmetric normalized Laplacian
    ///
    /// # Returns
    /// Dictionary with:
    /// - "values": Eigenvalues sorted ascending (by real part for directed graphs)
    /// - "vectors": Unit eigenvectors aligned with "values" (None for directed graphs)
    /// - "nodes": Node IDs giving the order of eigenvector entries
    /// - "algebraic_connectivity": λ₂ over the full spectrum (None if n < 2 or directed)
    ///
    /// # Errors
    /// - Empty graph
    /// - Invalid `which`
    /// - Eigenvalue computation failed to converge (directed graphs)
    #[pyo3(signature = (which="smallest", k=None, normalized=false))]
    pub fn get_laplacian_spectrum(
        &self,
        py: Python,
        which: &str,
        k: Option<usize>,
        normalized: bool,
    ) -> PyResult<PyObject> {
        let count = spectral_count(which, k, self.nodes.len())?;
        let (node_ids, adj) = self.spectral_adjacency("Laplacian spectrum")?;
        let laplacian = spectral_laplacian(&adj, normalized);

        let result = PyDict::new(py);
        result.set_item("nodes", &node_ids)?;

        if self.directed {
            let mut values = spectral_complex_eigenvalues(laplacian.to_dense())?;
            values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            let selected = spectral_select(values, which, count, false);
            result.set_item("values", spectral_complex_list(py, &selected)?)?;
            result.set_item("vectors", py.None())?;
            result.set_item("algebraic_connectivity", py.None())?;
        } else {
            let (pairs, bottom) = spectral_symmetric(&laplacian, which, count, false)?;
            let connectivity = (bottom.len() >= 2).then(|| bottom[1]);
            let (values, vectors): (Vec<f64>, Vec<Vec<f64>>) = pairs.into_iter().unzip();
            result.set_item("values", values)?;
            result.set_item("vectors", vectors)?;
            result.set_item("algebraic_connectivity", connectivity)?;
        }

        Ok(result.into())
    }

    // ========== Community Detection ==========

    /// Get community membership using Edge Betweenness (Girvan-Newman) algorithm
    ///
    /// Iteratively removes edges with highest betweenness centrality to reveal communities.
    /// Uses Brandes' algorithm for efficient betweenness calculation.
    ///
    /// Algorithm:
    /// 1. Compute edge betweenness for all edges using Brandes' algorithm
    /// 2. Remove edge with highest betweenness
    /// 3. Recompute betweenness and check if modularity improved
    /// 4. Repeat until modularity stops improving or graph fully disconnected
    /// 5. Return partition with maximum modularity
    ///
    /// Complexity: O(m²n) where m = edges, n = nodes
    ///
    /// # Arguments
    /// * None - automatically finds best partition
    ///
    /// # Returns
    /// * HashMap mapping node_id -> community_id (0-indexed)
    ///
    /// # Errors
    /// * Returns error if graph is directed (requires undirected graph)
    ///
    /// # References
    /// Girvan & Newman. "Community structure in social and biological networks" (2002)
    /// Brandes. "A faster algorithm for betweenness centrality" (2001)
    #[inline]
    fn get_cmty_edge_btwns(&self) -> PyResult<FxHashMap<u64, usize>> {
        // Require undirected graph
        if self.is_directed() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Edge Betweenness requires an undirected graph"
            ));
        }

        // Same algorithm as cmty_girvan_newman(), so share its implementation: two
        // copies with different (hash-order) tie-breaking used to disagree on about
        // a fifth of small graphs. The first dendrogram entry is the best-modularity
        // partition; communities are numbered by their smallest node ID.
        let dendrogram = crate::algorithms::community::girvan_newman(self, None)?;
        let mut result: FxHashMap<u64, usize> = FxHashMap::default();
        if let Some(best) = dendrogram.first() {
            for (comm_id, members) in best.iter().enumerate() {
                for &node_id in members {
                    result.insert(node_id, comm_id);
                }
            }
        }
        Ok(result)
    }

    /// Get community membership using Leading Eigenvector (Newman) algorithm
    ///
    /// Spectral optimization using recursive bisection based on the leading eigenvector
    /// of the modularity matrix. Splits communities when modularity improves.
    ///
    /// Algorithm:
    /// 1. Start with all nodes in one community
    /// 2. Compute modularity matrix B_ij = A_ij - (k_i * k_j) / 2m
    /// 3. Find leading eigenvector using power iteration
    /// 4. Split by sign: positive → community 1, negative → community 2
    /// 5. Accept split if modularity improves
    /// 6. Recursively split each community
    ///
    /// Complexity: O(n²) per split, recursive
    ///
    /// # Arguments
    /// * None - automatically finds best hierarchical partition
    ///
    /// # Returns
    /// * HashMap mapping node_id -> community_id (0-indexed)
    ///
    /// # Errors
    /// * Returns error if graph is directed (requires undirected graph)
    ///
    /// # References
    /// Newman, M. E. J. "Finding community structure in networks using the
    /// eigenvectors of matrices" (2006)
    #[inline]
    fn get_cmty_l_eigenvec(&self) -> PyResult<FxHashMap<u64, usize>> {
        // Require undirected graph
        if self.is_directed() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Leading Eigenvector requires an undirected graph"
            ));
        }

        let n = self.num_nodes();
        if n == 0 {
            return Ok(FxHashMap::default());
        }

        // Special case: single node
        if n == 1 {
            let mut result = FxHashMap::default();
            if let Some(&node_id) = self.nodes.keys().next() {
                result.insert(node_id, 0);
            }
            return Ok(result);
        }

        // No edges: each node is its own community
        if self.num_edges() == 0 {
            let result: FxHashMap<u64, usize> = self.nodes.keys()
                .enumerate()
                .map(|(idx, &id)| (id, idx))
                .collect();
            return Ok(result);
        }

        // Start with all nodes in one community
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        let mut communities: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &id in &node_ids {
            communities.insert(id, 0);
        }

        // Recursive splitting
        let mut next_community_id = 1;
        let mut to_split: Vec<Vec<u64>> = Vec::with_capacity(n / 10);
        to_split.push(node_ids);

        while let Some(group) = to_split.pop() {
            if group.len() <= 1 {
                continue; // Can't split single node
            }

            // Try to split this group
            if let Some((group1, group2)) = self.try_split_eigenvector(&group) {
                // Assign new community IDs
                let comm1_id = *communities.get(&group1[0]).unwrap();
                let comm2_id = next_community_id;
                next_community_id += 1;

                // Update communities
                for &node_id in &group1 {
                    communities.insert(node_id, comm1_id);
                }
                for &node_id in &group2 {
                    communities.insert(node_id, comm2_id);
                }

                // Add both groups for potential further splitting
                if group1.len() > 1 {
                    to_split.push(group1);
                }
                if group2.len() > 1 {
                    to_split.push(group2);
                }
            }
        }

        // Renumber communities to be contiguous
        let mut community_map: FxHashMap<usize, usize> = FxHashMap::with_capacity_and_hasher(n / 10, Default::default());
        let mut next_id = 0;

        for comm_id in communities.values() {
            if !community_map.contains_key(comm_id) {
                community_map.insert(*comm_id, next_id);
                next_id += 1;
            }
        }

        let result: FxHashMap<u64, usize> = communities.iter()
            .map(|(&node_id, old_comm)| (node_id, *community_map.get(old_comm).unwrap()))
            .collect();

        Ok(result)
    }

    /// Detect communities using the Louvain method (PyO3 wrapper).
    ///
    /// The Louvain method is a hierarchical multi-level optimization algorithm.
    /// It maximizes modularity through iterative local optimization and aggregation.
    ///
    /// # Arguments
    /// * `resolution` - Resolution parameter (default 1.0). Higher values find smaller communities.
    ///
    /// # Returns
    /// HashMap mapping node IDs to community IDs (0-indexed, contiguous).
    ///
    /// # Errors
    /// * Returns error if graph is directed (requires undirected graph)
    #[inline]
    fn get_cmty_louvain(&self, resolution: f64) -> PyResult<FxHashMap<u64, usize>> {
        if self.is_directed() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Louvain algorithm requires an undirected graph"
            ));
        }

        // Handle edge cases
        let n = self.num_nodes();
        if n == 0 {
            return Ok(FxHashMap::default());
        }

        if n == 1 {
            let mut result = FxHashMap::default();
            if let Some(&node_id) = self.nodes.keys().next() {
                result.insert(node_id, 0);
            }
            return Ok(result);
        }

        let m = self.num_edges() as f64;
        if m == 0.0 {
            // No edges - each node is its own community, numbered in node ID order
            let mut ids: Vec<u64> = self.nodes.keys().copied().collect();
            ids.sort_unstable();
            return Ok(ids.into_iter().enumerate().map(|(idx, id)| (id, idx)).collect());
        }

        // Multi-level Louvain (local moving + aggregation); the fixed seed keeps it deterministic
        let (ids, membership) =
            crate::algorithms::community_detection::louvain_partition(self, resolution, 42);
        Ok(ids.into_iter().zip(membership).collect())
    }

    /// Detect communities using the Walktrap algorithm.
    ///
    /// This method uses random walks to compute distances between nodes and applies
    /// hierarchical clustering to find communities.
    ///
    /// # Arguments
    /// * `steps` - Number of random walk steps (default: 4)
    ///
    /// # Returns
    /// A mapping from node ID (String) to community ID (usize)
    #[inline]
    fn get_cmty_walktrap(&self, steps: usize) -> PyResult<FxHashMap<u64, usize>> {
        if self.is_directed() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Walktrap algorithm only works on undirected graphs"
            ));
        }

        // Handle edge cases
        if self.nodes.is_empty() {
            return Ok(FxHashMap::default());
        }

        if self.nodes.len() == 1 {
            let mut result = FxHashMap::default();
            result.insert(*self.nodes.keys().next().unwrap(), 0);
            return Ok(result);
        }

        // Create node index mapping
        let node_ids: Vec<u64> = self.nodes.keys().copied().collect();
        let n = node_ids.len();
        let mut node_to_idx = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (idx, &node_id) in node_ids.iter().enumerate() {
            node_to_idx.insert(node_id, idx);
        }

        // Check for disconnected components first
        let components = self.find_connected_components();
        let num_components = components.values().copied().max().unwrap_or(0) + 1;

        if num_components > 1 {
            // Return components as communities
            return Ok(components);
        }

        // Compute transition probability matrix P
        #[allow(non_snake_case)]
        let P = self.compute_transition_matrix(&node_ids, &node_to_idx);

        // Compute t-step transition probabilities P^t
        #[allow(non_snake_case)]
        let P_t = self.matrix_power(&P, steps);

        // Compute distance matrix between nodes
        let distances = self.compute_random_walk_distances(&P_t, &node_ids, &node_to_idx);

        // Hierarchical clustering using Ward's method
        let dendrogram = self.ward_clustering(&distances, n);

        // Find best partition by maximizing modularity
        let best_partition = self.find_best_partition(&dendrogram, &node_ids, &node_to_idx);

        // Convert to result format
        let mut result = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (idx, &node_id) in node_ids.iter().enumerate() {
            result.insert(node_id, best_partition[idx]);
        }

        Ok(result)
    }

    // ========== Spanning Tree Algorithms ==========

    /// Find minimum or maximum spanning tree using Kruskal's algorithm
    ///
    /// Returns a vector of edge IDs that form the spanning tree.
    /// For a graph with n nodes, the MST will have exactly n-1 edges (if connected).
    ///
    /// # Arguments
    /// * `minimum` - If true, find minimum spanning tree; if false, find maximum
    /// * `weight_attr` - Optional attribute name for edge weights (default: uses edge.weight)
    ///
    /// # Returns
    /// Vector of edge IDs in the spanning tree, or empty vector if graph has < 2 nodes
    ///
    /// # Algorithm
    /// Time complexity: O(E log E) for sorting edges
    /// Space complexity: O(V) for union-find structure
    #[pyo3(signature = (minimum=true, weight_attr=None))]
    fn kruskal_mst_edges(
        &self,
        py: Python,
        minimum: bool,
        weight_attr: Option<&str>,
    ) -> PyResult<Vec<u64>> {
        let num_nodes = self.nodes.len();

        if num_nodes < 2 {
            return Ok(Vec::new());
        }

        let edges_needed = num_nodes - 1;

        // Collect all edges with their weights
        let mut weighted_edges: Vec<(f64, u64, u64, u64)> = Vec::new(); // (weight, edge_id, source, target)

        for edge in &self.edges {
            let weight = if let Some(attr) = weight_attr {
                // Get weight from custom attribute
                if let Some(ref attrs) = edge.attributes {
                    let dict = attrs.bind(py).downcast::<PyDict>()?;
                    if let Some(val) = dict.get_item(attr)? {
                        val.extract::<f64>().unwrap_or(1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            } else {
                // Use edge.weight field, default to 1.0
                edge.weight.unwrap_or(1.0)
            };

            weighted_edges.push((weight, edge.id, edge.source, edge.target));
        }

        // Sort edges by weight
        weighted_edges.sort_by(|a, b| {
            if minimum {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Run Kruskal's algorithm with union-find
        let n = self.num_nodes();
        let mut uf = UnionFind::with_capacity(n);
        let mut mst_edges = Vec::with_capacity(edges_needed);

        for (_, edge_id, source, target) in weighted_edges {
            if uf.union(source, target) {
                mst_edges.push(edge_id);
                if mst_edges.len() == edges_needed {
                    break;
                }
            }
        }

        Ok(mst_edges)
    }

    /// Enumerate all simple paths between source and target nodes.
    ///
    /// A simple path is a path with no repeated nodes. Uses depth-first search
    /// with backtracking to explore all possibilities.
    ///
    /// # Arguments
    /// * `source` - Source node label
    /// * `target` - Target node label
    /// * `cutoff` - Optional maximum path length (number of edges). If None, no limit.
    ///
    /// # Returns
    /// Vec of paths, where each path is a Vec of node labels (String)
    ///
    /// # Algorithm
    /// - Uses DFS with backtracking
    /// - Tracks visited nodes to ensure "simple" property
    /// - Time complexity: O(n!) in worst case (all permutations)
    /// - Space complexity: O(n) for recursion stack
    ///
    /// # Example
    /// ```python
    /// # Find all simple paths from 'A' to 'D' with max length 4
    /// paths = graph.all_simple_paths(a_id, d_id, cutoff=4)
    /// ```
    #[pyo3(signature = (source, target, cutoff=None))]
    fn all_simple_paths(
        &self,
        source: u64,
        target: u64,
        cutoff: Option<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        // Endpoints are node IDs, so unlabelled nodes work too
        for (role, id) in [("Source", source), ("Target", target)] {
            if !self.nodes.contains_key(&id) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("{} node {} does not exist", role, id)
                ));
            }
        }
        let (source_id, target_id) = (&source, &target);

        // Initialize visited set and result container
        let mut visited = FxHashSet::default();
        let mut paths = Vec::new();
        let mut current_path = Vec::new();

        // Start DFS
        self.dfs_simple_paths(
            *source_id,
            *target_id,
            &mut visited,
            &mut current_path,
            &mut paths,
            cutoff,
        );

        Ok(paths)
    }

    /// Check if a sequence of nodes forms a simple path in the graph.
    ///
    /// A simple path is a sequence of nodes where:
    /// 1. All nodes exist in the graph
    /// 2. Consecutive nodes are connected by edges
    /// 3. No node appears more than once (no cycles)
    ///
    /// # Arguments
    /// * `nodes` - Sequence of node IDs to validate
    ///
    /// # Returns
    /// true if the sequence forms a valid simple path, false otherwise
    ///
    /// # Algorithm
    /// - O(n) time complexity where n = number of nodes in path
    /// - O(n) space complexity for tracking visited nodes
    ///
    /// # Example
    /// ```python
    /// # Check if ['A', 'B', 'C'] forms a simple path
    /// is_valid = graph.is_simple_path([a_id, b_id, c_id])
    /// ```
    #[inline]
    fn is_simple_path(&self, nodes: Vec<u64>) -> PyResult<bool> {
        // Nodes are IDs (Python resolves labels), so a digit-only label is never taken for an
        // ID or vice versa. Every node must exist and appear once, and each consecutive pair must
        // be joined by an edge (in that direction on directed graphs). Empty and one-node
        // sequences of existing nodes are trivially simple.
        let mut visited = FxHashSet::with_capacity_and_hasher(nodes.len(), Default::default());
        for &id in &nodes {
            if !self.nodes.contains_key(&id) || !visited.insert(id) {
                return Ok(false);
            }
        }
        Ok(nodes.windows(2).all(|pair| self.has_edge(pair[0], pair[1])))
    }
}


impl Graph {
    // ========== Creation-order Helpers ==========

    /// The node creation log, minus nodes that have since been deleted.
    ///
    /// The log itself is append-only (deletions don't touch it), so every reader
    /// filters against the live graph instead of reporting or selecting stale IDs.
    pub(crate) fn live_node_creation_order(&self) -> Vec<u64> {
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        self.node_creation_order
            .iter()
            .copied()
            .filter(|id| self.nodes.contains_key(id) && seen.insert(*id))
            .collect()
    }

    /// The edge creation log, minus edges that have since been deleted.
    pub(crate) fn live_edge_creation_order(&self) -> Vec<u64> {
        let live: FxHashSet<u64> = self.edges.iter().map(|e| e.id).collect();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        self.edge_creation_order
            .iter()
            .copied()
            .filter(|id| live.contains(id) && seen.insert(*id))
            .collect()
    }

    // ========== Spectral Helpers ==========

    /// Node IDs in creation order (the row/column order of spectral matrices)
    ///
    /// Any node missing from the creation log is appended in ascending ID order so the
    /// ordering is always complete and deterministic.
    fn spectral_node_order(&self) -> Vec<u64> {
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        let mut order: Vec<u64> = self
            .node_creation_order
            .iter()
            .copied()
            .filter(|id| self.nodes.contains_key(id) && seen.insert(*id))
            .collect();

        if order.len() < self.nodes.len() {
            let mut rest: Vec<u64> =
                self.nodes.keys().copied().filter(|id| !seen.contains(id)).collect();
            rest.sort_unstable();
            order.extend(rest);
        }
        order
    }

    /// Weighted adjacency matrix for spectral methods
    ///
    /// Parallel edges sum their weights; an undirected self-loop adds its weight once
    /// to the diagonal.
    fn spectral_adjacency(&self, what: &str) -> PyResult<(Vec<u64>, CsrMatrix)> {
        let node_ids = self.spectral_node_order();
        let n = node_ids.len();

        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot compute {} of empty graph",
                what
            )));
        }

        let node_to_idx: FxHashMap<u64, usize> =
            node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut triplets = Vec::with_capacity(2 * self.edges.len());

        for edge in &self.edges {
            if let (Some(&i), Some(&j)) =
                (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target))
            {
                let weight = edge.weight.unwrap_or(1.0);
                triplets.push((i, j, weight));
                if !self.directed && i != j {
                    triplets.push((j, i, weight));
                }
            }
        }

        Ok((node_ids, CsrMatrix::from_triplets(n, triplets)))
    }

    /// Bron-Kerbosch algorithm for finding maximal cliques (internal helper)
    ///
    /// # Arguments
    /// * `r` - Current clique being built
    /// * `p` - Candidate nodes that could extend the clique
    /// * `x` - Already processed nodes
    /// * `cliques` - Vector to store found maximal cliques
    fn bron_kerbosch_internal(
        &self,
        r: &mut FxHashSet<u64>,
        p: &mut FxHashSet<u64>,
        x: &mut FxHashSet<u64>,
        cliques: &mut Vec<Vec<u64>>,
    ) {
        if p.is_empty() && x.is_empty() {
            // Found a maximal clique
            let mut clique: Vec<u64> = r.iter().copied().collect();
            clique.sort_unstable();
            cliques.push(clique);
            return;
        }

        // Choose a pivot to minimize recursion
        let pivot = p.union(x).next().copied();

        let mut candidates: Vec<u64> = if let Some(pivot_node) = pivot {
            let pivot_neighbors: FxHashSet<u64> = self.adjacency_list
                .get(&pivot_node)
                .map(|neighbors| neighbors.iter().copied().collect())
                .unwrap_or_default();
            p.difference(&pivot_neighbors).copied().collect()
        } else {
            p.iter().copied().collect()
        };

        for v in candidates.drain(..) {
            let neighbors: FxHashSet<u64> = self.adjacency_list
                .get(&v)
                .map(|n| n.iter().copied().collect())
                .unwrap_or_default();

            r.insert(v);

            let mut new_p: FxHashSet<u64> = p.intersection(&neighbors).copied().collect();
            let mut new_x: FxHashSet<u64> = x.intersection(&neighbors).copied().collect();

            self.bron_kerbosch_internal(r, &mut new_p, &mut new_x, cliques);

            r.remove(&v);
            p.remove(&v);
            x.insert(v);
        }
    }

    /// Get the adjacency list (internal method)
    pub fn get_adjacency_list(&self) -> &FxHashMap<u64, Vec<u64>> {
        &self.adjacency_list
    }

    /// Raise `ValueError` if any label is already used in the graph or repeats within `labels`
    /// (so a generator can check every label before adding a node)
    pub(crate) fn check_new_labels(&self, labels: &[String]) -> PyResult<()> {
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        for label in labels {
            if self.node_labels.contains_key(label) || !seen.insert(label.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node with label '{}' already exists",
                    label
                )));
            }
        }
        Ok(())
    }

    /// Get edges vector (internal method for algorithms)
    pub fn get_edges_internal(&self) -> &Vec<Edge> {
        &self.edges
    }

    /// The ID the next auto-assigned edge will get (above every edge ID ever used, including
    /// deleted ones)
    pub fn next_edge_id_internal(&self) -> u64 {
        self.next_edge_id
    }

    /// Create a new graph (public constructor for Rust code)
    pub fn new_rust(directed: bool, weighted: bool) -> Self {
        Graph {
            next_node_id: 1,
            nodes: FxHashMap::default(),
            node_labels: FxHashMap::default(),
            deleted_node_ids: FxHashSet::default(),
            edges: Vec::new(),
            next_edge_id: 1,
            deleted_edge_ids: FxHashSet::default(),
            adjacency_list: FxHashMap::default(),
            directed,
            weighted,
            selected_nodes: FxHashSet::default(),
            selected_edges: FxHashSet::default(),
            node_creation_order: Vec::new(),
            edge_creation_order: Vec::new(),
            edge_index: std::sync::OnceLock::new(),
        }
    }

    /// Add a node (public method for Rust code)
    pub fn add_node_rust(&mut self, mut node: Node) -> PyResult<u64> {
        // Auto-assign node ID if it's 0
        let node_id = if node.id == 0 {
            let new_id = self.next_node_id;
            self.next_node_id += 1;
            node.id = new_id;
            new_id
        } else {
            if node.id >= self.next_node_id {
                self.next_node_id = node.id + 1;
            }
            node.id
        };

        if self.nodes.contains_key(&node_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node with ID {} already exists",
                node_id
            )));
        }

        if let Some(ref label) = node.label {
            if self.node_labels.contains_key(label) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node with label '{}' already exists",
                    label
                )));
            }
            self.node_labels.insert(label.clone(), node_id);
        }

        self.adjacency_list.insert(node_id, Vec::new());
        self.nodes.insert(node_id, node);
        self.node_creation_order.push(node_id);

        Ok(node_id)
    }

    /// Add an edge (public method for Rust code)
    pub fn add_edge_rust(&mut self, mut edge: Edge) -> PyResult<u64> {
        let edge_id = if edge.id == 0 {
            let new_id = self.next_edge_id;
            self.next_edge_id += 1;
            edge.id = new_id;
            new_id
        } else {
            if edge.id >= self.next_edge_id {
                self.next_edge_id = edge.id + 1;
            }
            edge.id
        };

        if !self.nodes.contains_key(&edge.source) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Source node {} does not exist",
                edge.source
            )));
        }
        if !self.nodes.contains_key(&edge.target) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target node {} does not exist",
                edge.target
            )));
        }

        self.adjacency_list
            .entry(edge.source)
            .or_default()
            .push(edge.target);

        if !self.directed {
            self.adjacency_list
                .entry(edge.target)
                .or_default()
                .push(edge.source);
        }

        self.push_edge(edge);
        self.edge_creation_order.push(edge_id);

        Ok(edge_id)
    }

    /// Remove an edge by ID (public method for Rust code)
    pub fn remove_edge_rust(&mut self, edge_id: u64) -> PyResult<()> {
        let edge_idx = self.edges.iter().position(|e| e.id == edge_id);

        if let Some(idx) = edge_idx {
            let edge = &self.edges[idx];
            let source = edge.source;
            let target = edge.target;

            // Remove from adjacency list
            if let Some(neighbors) = self.adjacency_list.get_mut(&source) {
                neighbors.retain(|&n| n != target);
            }

            if !self.directed {
                if let Some(neighbors) = self.adjacency_list.get_mut(&target) {
                    neighbors.retain(|&n| n != source);
                }
            }

            // Remove edge
            self.edges.remove(idx);
            self.invalidate_edge_index();
            self.deleted_edge_ids.insert(edge_id);

            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Edge {} not found",
                edge_id
            )))
        }
    }

    /// Remove a node by ID (public method for Rust code)
    pub fn remove_node_rust(&mut self, node_id: u64) -> PyResult<()> {
        if !self.nodes.contains_key(&node_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} not found",
                node_id
            )));
        }

        // Remove all edges connected to this node
        let mut edges_to_remove = Vec::new();
        for edge in &self.edges {
            if edge.source == node_id || edge.target == node_id {
                edges_to_remove.push(edge.id);
            }
        }

        for edge_id in edges_to_remove {
            self.remove_edge_rust(edge_id)?;
        }

        // Remove from adjacency list
        self.adjacency_list.remove(&node_id);

        // Remove node from other nodes' adjacency lists
        for neighbors in self.adjacency_list.values_mut() {
            neighbors.retain(|&n| n != node_id);
        }

        // Remove node
        if let Some(node) = self.nodes.remove(&node_id) {
            if let Some(label) = &node.label {
                self.node_labels.remove(label);
            }
        }

        self.deleted_node_ids.insert(node_id);

        Ok(())
    }

    /// Get degree of a node (public method for Rust code)
    pub fn degree_rust(&self, node_id: u64) -> usize {
        self.adjacency_list
            .get(&node_id)
            .map(|neighbors| neighbors.len())
            .unwrap_or(0)
    }

    /// Get in-degree of a node (public method for Rust code)
    pub fn get_in_degree(&self, node_id: u64) -> usize {
        if !self.directed {
            return self.degree_rust(node_id);
        }
        // Counted once in the edge index rather than scanning every edge
        self.edge_index().in_degree.get(&node_id).copied().unwrap_or(0)
    }

    /// Get out-degree of a node (public method for Rust code)
    pub fn get_out_degree(&self, node_id: u64) -> usize {
        // A directed adjacency list has one entry per outgoing edge (parallel edges and
        // self-loops included), so its length is the out-degree
        self.degree_rust(node_id)
    }

    /// Get nodes hashmap (public method for Rust code)
    pub fn get_nodes(&self) -> &FxHashMap<u64, Node> {
        &self.nodes
    }

    /// Check if there's an edge between two nodes
    #[inline]
    pub(crate) fn has_edge(&self, source: u64, target: u64) -> bool {
        if let Some(neighbors) = self.adjacency_list.get(&source) {
            neighbors.contains(&target)
        } else {
            false
        }
    }


    /// Helper for all_simple_paths: DFS with backtracking
    fn dfs_simple_paths(
        &self,
        current: u64,
        target: u64,
        visited: &mut FxHashSet<u64>,
        current_path: &mut Vec<String>,
        paths: &mut Vec<Vec<String>>,
        cutoff: Option<usize>,
    ) {
        // Get label for current node
        let current_label = self.nodes.get(&current)
            .and_then(|n| n.label.clone())
            .unwrap_or_else(|| current.to_string());

        // Add current node to path and mark as visited
        visited.insert(current);
        current_path.push(current_label);

        // Check if we've reached the target
        if current == target {
            paths.push(current_path.clone());
        } else {
            // Check cutoff (cutoff is number of edges, path length is nodes)
            let can_continue = cutoff.map_or(true, |max_edges| current_path.len() <= max_edges);

            if can_continue {
                // Explore neighbors
                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            self.dfs_simple_paths(
                                neighbor,
                                target,
                                visited,
                                current_path,
                                paths,
                                cutoff,
                            );
                        }
                    }
                }
            }
        }

        // Backtrack: remove current node from path and visited set
        current_path.pop();
        visited.remove(&current);
    }

    /// Generate all k-combinations of elements
    /// Helper function for Johnson graph generation
    #[inline]
    fn generate_combinations<T: Clone>(elements: &[T], k: usize) -> Vec<Vec<T>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k > elements.len() {
            return vec![];
        }
        if k == elements.len() {
            return vec![elements.to_vec()];
        }

        // Estimate capacity: binomial coefficient C(n,k) ≈ n^k/k! for small k
        let n = elements.len();
        let estimated_size = if k <= 3 {
            match k {
                1 => n,
                2 => n * (n - 1) / 2,
                3 => n * (n - 1) * (n - 2) / 6,
                _ => 100,
            }
        } else {
            100 // Reasonable default for larger k
        };
        let mut result = Vec::with_capacity(estimated_size);

        // Generate combinations recursively
        fn combine<T: Clone>(
            elements: &[T],
            k: usize,
            start: usize,
            current: &mut Vec<T>,
            result: &mut Vec<Vec<T>>,
        ) {
            if current.len() == k {
                result.push(current.clone());
                return;
            }

            for i in start..elements.len() {
                current.push(elements[i].clone());
                combine(elements, k, i + 1, current, result);
                current.pop();
            }
        }

        let mut current = Vec::with_capacity(k);
        combine(elements, k, 0, &mut current, &mut result);
        result
    }

    /// Try to split a group of nodes using eigenvector method
    /// Returns Some((group1, group2)) if split improves modularity, None otherwise
    #[inline]
    fn try_split_eigenvector(&self, nodes: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        if nodes.len() <= 1 {
            return None;
        }

        let m = self.num_edges() as f64;
        if m == 0.0 {
            return None;
        }

        // Create node index mapping
        let _n = nodes.len();
        let node_to_idx: FxHashMap<u64, usize> = FxHashMap::from_iter(
            nodes.iter().enumerate().map(|(idx, &id)| (id, idx))
        );

        let n = nodes.len();

        // Compute leading eigenvector using power iteration
        let eigenvector = self.compute_leading_eigenvector(nodes, &node_to_idx, m)?;

        // Split by sign of eigenvector
        let mut group1 = Vec::with_capacity(n / 2);
        let mut group2 = Vec::with_capacity(n / 2);

        for (idx, &node_id) in nodes.iter().enumerate() {
            if eigenvector[idx] >= 0.0 {
                group1.push(node_id);
            } else {
                group2.push(node_id);
            }
        }

        // Check if split is valid (both groups non-empty)
        if group1.is_empty() || group2.is_empty() {
            return None;
        }

        // Check if split improves modularity:
        // Create split vector: +1 for group1, -1 for group2
        let n = nodes.len();
        let mut s: Vec<f64> = vec![0.0; n];
        for (idx, &node_id) in nodes.iter().enumerate() {
            if group1.contains(&node_id) {
                s[idx] = 1.0;
            } else {
                s[idx] = -1.0;
            }
        }

        // Precompute degrees
        let degrees: Vec<f64> = nodes.iter()
            .map(|&id| self.degree(id) as f64)
            .collect();

        // Compute B^(g) matrix (same as used in eigenvector computation)
        let mut b_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        let inv_two_m = 1.0 / (2.0 * m);

        for (i, &node_i) in nodes.iter().enumerate() {
            // First compute row sum: Σ_k B_ik
            let mut row_sum = 0.0;
            for (k, &node_k) in nodes.iter().enumerate() {
                let has_edge = self.adjacency_list.get(&node_i)
                    .map(|neighbors| neighbors.contains(&node_k))
                    .unwrap_or(false);
                let a_ik = if has_edge { 1.0 } else { 0.0 };
                let b_ik = a_ik - (degrees[i] * degrees[k] * inv_two_m);
                row_sum += b_ik;
            }

            // Now compute B^(g)_ij = B_ij - δ_ij * row_sum
            for (j, &node_j) in nodes.iter().enumerate() {
                let has_edge = self.adjacency_list.get(&node_i)
                    .map(|neighbors| neighbors.contains(&node_j))
                    .unwrap_or(false);
                let a_ij = if has_edge { 1.0 } else { 0.0 };
                let b_ij = a_ij - (degrees[i] * degrees[j] * inv_two_m);

                if i == j {
                    b_matrix[i][j] = b_ij - row_sum;
                } else {
                    b_matrix[i][j] = b_ij;
                }
            }
        }

        // Compute B^(g) * s
        let mut bs: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                bs[i] += b_matrix[i][j] * s[j];
            }
        }

        // Compute s^T * B^(g) * s (the modularity contribution)
        let mut mod_contribution = 0.0;
        for i in 0..n {
            mod_contribution += s[i] * bs[i];
        }

        // Accept split if modularity contribution is positive
        if mod_contribution > 1e-8 {
            Some((group1, group2))
        } else {
            None
        }
    }

    /// Compute leading eigenvector of modularity matrix using nalgebra eigensolver
    /// For a subgraph, we use the generalized modularity matrix:
    /// B^(g)_ij = B_ij - δ_ij * Σ_k B_ik
    /// This modification is essential for correct bisection (Newman 2006, Eq. 5)
    ///
    /// Uses nalgebra's symmetric_eigen() for robust eigenvalue computation,
    /// specifically targeting the largest real eigenvalue (most positive).
    fn compute_leading_eigenvector(
        &self,
        nodes: &[u64],
        _node_to_idx: &FxHashMap<u64, usize>,
        m: f64,
    ) -> Option<Vec<f64>> {
        let n = nodes.len();

        // Precompute degrees for efficiency
        let degrees: Vec<f64> = nodes.iter()
            .map(|&id| self.degree(id) as f64)
            .collect();

        // Build B^(g) matrix for the subgraph using nalgebra DMatrix
        let mut b_matrix = na::DMatrix::<f64>::zeros(n, n);

        for (i, &node_i) in nodes.iter().enumerate() {
            // First compute row sum: Σ_k B_ik
            let mut row_sum = 0.0;
            for (k, &node_k) in nodes.iter().enumerate() {
                let has_edge = self.adjacency_list.get(&node_i)
                    .map(|neighbors| neighbors.contains(&node_k))
                    .unwrap_or(false);
                let a_ik = if has_edge { 1.0 } else { 0.0 };
                let b_ik = a_ik - (degrees[i] * degrees[k]) / (2.0 * m);
                row_sum += b_ik;
            }

            // Now compute B^(g)_ij = B_ij - δ_ij * row_sum
            for (j, &node_j) in nodes.iter().enumerate() {
                let has_edge = self.adjacency_list.get(&node_i)
                    .map(|neighbors| neighbors.contains(&node_j))
                    .unwrap_or(false);
                let a_ij = if has_edge { 1.0 } else { 0.0 };
                let b_ij = a_ij - (degrees[i] * degrees[j]) / (2.0 * m);

                if i == j {
                    b_matrix[(i, j)] = b_ij - row_sum;
                } else {
                    b_matrix[(i, j)] = b_ij;
                }
            }
        }

        // Compute eigenvalues and eigenvectors using nalgebra
        // symmetric_eigen() is optimized for real symmetric matrices
        let eigen = b_matrix.symmetric_eigen();

        // Find the eigenvalue with the largest value
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Find index of maximum eigenvalue
        let mut max_idx = 0;
        let mut max_eigenvalue = eigenvalues[0];

        for (i, &val) in eigenvalues.iter().enumerate() {
            if val > max_eigenvalue {
                max_eigenvalue = val;
                max_idx = i;
            }
        }

        // Check if largest eigenvalue is positive
        if max_eigenvalue <= 1e-10 {
            return None;
        }

        // Extract the corresponding eigenvector
        let eigenvector = eigenvectors.column(max_idx);

        // Convert to Vec<f64>
        let v: Vec<f64> = eigenvector.iter().copied().collect();

        Some(v)
    }

    /// Helper: BFS to find augmenting path in residual graph (for min cut)
    fn bfs_augmenting_path(
        &self,
        source: u64,
        sink: u64,
        residual: &FxHashMap<(u64, u64), usize>,
    ) -> Vec<u64> {
        let n = self.num_nodes();
        let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
        let mut parent: FxHashMap<u64, u64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(n);

        visited.insert(source);
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            if current == sink {
                // Reconstruct path
                let mut path = vec![current];
                let mut node = current;

                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }

                path.reverse();
                return path;
            }

            // Explore neighbors with residual capacity > 0
            for &other_node in self.nodes.keys() {
                let edge_key = (current, other_node);
                let capacity = residual.get(&edge_key).copied().unwrap_or(0);

                if capacity > 0 && !visited.contains(&other_node) {
                    visited.insert(other_node);
                    parent.insert(other_node, current);
                    queue.push_back(other_node);
                }
            }
        }

        Vec::new() // No path found
    }

    /// BFS for finding augmenting paths in transformed graph (for node connectivity)
    /// Similar to bfs_augmenting_path but works with transformed node IDs
    fn bfs_augmenting_path_custom(
        &self,
        source: u64,
        sink: u64,
        residual: &FxHashMap<(u64, u64), usize>,
    ) -> Vec<u64> {
        let n = self.num_nodes();
        let capacity = n * 2; // Transformed graph has split nodes
        let mut visited = FxHashSet::with_capacity_and_hasher(capacity, Default::default());
        let mut parent: FxHashMap<u64, u64> = FxHashMap::with_capacity_and_hasher(capacity, Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(capacity);

        visited.insert(source);
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            if current == sink {
                // Reconstruct path
                let mut path = vec![current];
                let mut node = current;

                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }

                path.reverse();
                return path;
            }

            // Explore all possible outgoing edges with residual capacity > 0
            // Check all possible transformed nodes
            for (&from, &to, &_cap) in residual.iter().filter_map(|(k, v)| {
                if k.0 == current && *v > 0 {
                    Some((&k.0, &k.1, v))
                } else {
                    None
                }
            }) {
                if !visited.contains(&to) {
                    visited.insert(to);
                    parent.insert(to, from);
                    queue.push_back(to);
                }
            }
        }

        Vec::new() // No path found
    }

    /// Helper for cohesive_blocks: recursively process a subgraph to find cohesive structure
    ///
    /// Args:
    ///   nodes: Node set defining the subgraph
    ///   parent_idx: Index of parent block (None for root)
    ///   all_blocks: Accumulated list of blocks (passed for indexing)
    ///
    /// Returns: (blocks, cohesions, parent_indices) for this subgraph and its children
    fn cohesive_blocks_for_subgraph(
        &self,
        nodes: &[u64],
        parent_idx: Option<usize>,
        all_blocks: &mut Vec<Vec<u64>>,
    ) -> PyResult<CohesiveBlocks> {
        let mut blocks = Vec::new();
        let mut cohesions = Vec::new();
        let mut parents = Vec::new();

        // If subgraph is too small, return as single block
        if nodes.len() <= 2 {
            blocks.push(nodes.to_vec());
            cohesions.push(if nodes.len() == 2 { 1 } else { 0 });
            parents.push(parent_idx);
            return Ok((blocks, cohesions, parents));
        }

        // Compute connectivity of this subgraph
        let k = self.induced_subgraph_connectivity(nodes)?;

        // Add this block
        let current_idx = all_blocks.len();
        blocks.push(nodes.to_vec());
        cohesions.push(k);
        parents.push(parent_idx);
        all_blocks.push(nodes.to_vec());

        // If k is 0 (disconnected) or very small, we're done with this branch
        if k == 0 {
            return Ok((blocks, cohesions, parents));
        }

        // Find minimal k-separators
        let separators = self.find_minimal_separators(nodes, k)?;

        if separators.is_empty() {
            // No separators found - this is a leaf block
            return Ok((blocks, cohesions, parents));
        }

        // Try each separator to find higher-cohesion subblocks
        let mut processed_components = FxHashSet::default();

        for separator in &separators {
            // Remove separator nodes and find connected components
            let remaining_nodes: Vec<u64> = nodes
                .iter()
                .filter(|&&n| !separator.contains(&n))
                .copied()
                .collect();

            if remaining_nodes.is_empty() {
                continue;
            }

            // Find components in remaining graph
            let components = self.find_components_in_subgraph(&remaining_nodes);

            for comp_nodes in components {
                if comp_nodes.len() <= 1 {
                    continue;
                }

                // Create a signature for this component to avoid duplicates
                let mut comp_sorted = comp_nodes.clone();
                comp_sorted.sort_unstable();
                let comp_signature = format!("{:?}", comp_sorted);

                if processed_components.contains(&comp_signature) {
                    continue;
                }
                processed_components.insert(comp_signature);

                // Check component + separator for higher connectivity
                // A k-cohesive block can contain a (k+1)-cohesive sub-block
                // formed by one component plus the separator
                let mut candidate_nodes = comp_nodes.clone();
                candidate_nodes.extend(separator.iter().copied());

                let candidate_connectivity = self.induced_subgraph_connectivity(&candidate_nodes)?;

                if candidate_connectivity > k {
                    // Recursively process this higher-cohesion component
                    let (sub_blocks, sub_cohesions, sub_parents) =
                        self.cohesive_blocks_for_subgraph(&candidate_nodes, Some(current_idx), all_blocks)?;

                    blocks.extend(sub_blocks);
                    cohesions.extend(sub_cohesions);
                    parents.extend(sub_parents);
                }
            }
        }

        Ok((blocks, cohesions, parents))
    }

    /// Find minimal k-separators in a subgraph
    ///
    /// A k-separator is a set of k nodes whose removal disconnects the subgraph.
    /// A minimal separator has no proper subset that is also a separator.
    pub(crate) fn find_minimal_separators(
        &self,
        nodes: &[u64],
        k: usize,
    ) -> PyResult<Vec<Vec<u64>>> {
        let n = nodes.len();
        let mut separators = Vec::new();

        if k == 0 || k >= n {
            return Ok(separators);
        }

        // For small k, we can enumerate candidate separators
        // For k=1: find articulation points
        if k == 1 {
            let articulation_points = self.find_articulation_points_in_subgraph(nodes);
            for node in articulation_points {
                separators.push(vec![node]);
            }
            return Ok(separators);
        }

        // For k > 1: use connectivity checks to find separators
        // Strategy: For each pair of non-adjacent nodes, find min-cut
        // If min-cut size == k, those cut nodes form a k-separator
        let node_set: FxHashSet<u64> = nodes.iter().copied().collect();
        let mut candidate_separators = FxHashSet::default();

        for i in 0..n {
            for j in (i + 1)..n {
                let u = nodes[i];
                let v = nodes[j];

                // Skip if adjacent
                if self.has_edge(u, v) {
                    continue;
                }

                // Find connectivity between u and v in this subgraph
                let conn = self.node_connectivity_between_in_subgraph(u, v, &node_set)?;

                if conn == k {
                    // Find the actual separator (min-cut) between u and v
                    let separator = self.find_min_node_cut_in_subgraph(u, v, &node_set)?;

                    if separator.len() == k {
                        let mut sep_sorted = separator.clone();
                        sep_sorted.sort_unstable();
                        candidate_separators.insert(sep_sorted);
                    }
                }
            }
        }

        // Filter to only minimal separators
        let all_seps: Vec<Vec<u64>> = candidate_separators.into_iter().collect();
        for sep in &all_seps {
            let is_minimal = !all_seps.iter().any(|other_sep| {
                other_sep != sep && other_sep.iter().all(|node| sep.contains(node))
            });

            if is_minimal {
                separators.push(sep.clone());
            }
        }

        Ok(separators)
    }

    /// Compute node connectivity of an induced subgraph
    pub(crate) fn induced_subgraph_connectivity(&self, nodes: &[u64]) -> PyResult<usize> {
        let n = nodes.len();

        if n <= 1 {
            return Ok(0);
        }

        if n == 2 {
            return Ok(if self.has_edge(nodes[0], nodes[1]) { 1 } else { 0 });
        }

        // If this is the whole graph, use the efficient global method
        if n == self.num_nodes() {
            return self.node_connectivity();
        }

        // Check if subgraph is connected
        let mut node_set = FxHashSet::with_capacity_and_hasher(n, Default::default());
        node_set.extend(nodes.iter().copied());
        if !self.is_subgraph_connected(&node_set) {
            return Ok(0);
        }

        // Check if complete
        let mut edge_count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.has_edge(nodes[i], nodes[j]) {
                    edge_count += 1;
                }
            }
        }
        let max_edges = n * (n - 1) / 2;
        if edge_count == max_edges {
            return Ok(n - 1);
        }

        // Find minimum degree in subgraph
        let mut min_degree = usize::MAX;
        for &node in nodes {
            let degree = self.adjacency_list
                .get(&node)
                .map(|neighbors| {
                    neighbors.iter().filter(|&&n| node_set.contains(&n)).count()
                })
                .unwrap_or(0);
            min_degree = min_degree.min(degree);
        }

        if min_degree == 1 {
            return Ok(1);
        }

        // Compute connectivity between non-adjacent pairs
        let mut min_connectivity = usize::MAX;
        for i in 0..n {
            for j in (i + 1)..n {
                let u = nodes[i];
                let v = nodes[j];

                if !self.has_edge(u, v) {
                    let conn = self.node_connectivity_between_in_subgraph(u, v, &node_set)?;
                    min_connectivity = min_connectivity.min(conn);

                    if min_connectivity == 1 {
                        return Ok(1);
                    }
                }
            }
        }

        if min_connectivity == usize::MAX {
            Ok(n - 1)
        } else {
            Ok(min_connectivity)
        }
    }

    /// Check if a subgraph defined by node set is connected
    pub(crate) fn is_subgraph_connected(&self, node_set: &FxHashSet<u64>) -> bool {
        if node_set.is_empty() {
            return true;
        }

        let start = *node_set.iter().next().unwrap();
        let mut visited = FxHashSet::with_capacity_and_hasher(node_set.len(), Default::default());
        let mut queue = std::collections::VecDeque::with_capacity(node_set.len());

        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == node_set.len()
    }

    /// Find connected components within a subgraph
    pub(crate) fn find_components_in_subgraph(&self, nodes: &[u64]) -> Vec<Vec<u64>> {
        let n = nodes.len();
        let mut node_set = FxHashSet::with_capacity_and_hasher(n, Default::default());
        node_set.extend(nodes.iter().copied());
        let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
        let mut components = Vec::with_capacity(n.min(4));

        for &node in nodes {
            if visited.contains(&node) {
                continue;
            }

            let mut component = Vec::with_capacity(n);
            let mut queue = std::collections::VecDeque::with_capacity(n);

            visited.insert(node);
            queue.push_back(node);

            while let Some(current) = queue.pop_front() {
                component.push(current);

                if let Some(neighbors) = self.adjacency_list.get(&current) {
                    for &neighbor in neighbors {
                        if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Find articulation points (cut vertices) in a subgraph
    fn find_articulation_points_in_subgraph(&self, nodes: &[u64]) -> Vec<u64> {
        let n = nodes.len();
        let mut node_set = FxHashSet::with_capacity_and_hasher(n, Default::default());
        node_set.extend(nodes.iter().copied());
        let mut articulation_points = Vec::with_capacity(n);

        if n <= 2 {
            return articulation_points;
        }

        for &candidate in nodes {
            // Remove candidate and check if subgraph becomes disconnected
            let mut remaining = Vec::with_capacity(n - 1);
            remaining.extend(nodes.iter().filter(|&&n| n != candidate).copied());

            let mut remaining_set = FxHashSet::with_capacity_and_hasher(n - 1, Default::default());
            remaining_set.extend(remaining.iter().copied());

            if !self.is_subgraph_connected(&remaining_set) {
                articulation_points.push(candidate);
            }
        }

        articulation_points
    }

    /// Compute node connectivity between two nodes in a subgraph
    pub(crate) fn node_connectivity_between_in_subgraph(
        &self,
        source: u64,
        target: u64,
        node_set: &FxHashSet<u64>,
    ) -> PyResult<usize> {
        if !node_set.contains(&source) || !node_set.contains(&target) {
            return Ok(0);
        }

        if source == target {
            return Ok(0);
        }

        // Check direct connection
        if self.has_edge(source, target) {
            // Still need to check if there's a path without the direct edge
            // by computing node connectivity properly
        }

        // Transform graph for node connectivity: split each node except source/target
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::default();

        // Use large offset to avoid ID collisions
        let offset = 1_000_000_000u64;

        // Split internal nodes (not source or target)
        for &node in node_set.iter() {
            if node != source && node != target {
                let v_in = offset + node * 2;
                let v_out = offset + node * 2 + 1;
                capacity.insert((v_in, v_out), 1);
            }
        }

        // Map edges in the subgraph to the transformed graph
        for &from in node_set.iter() {
            if let Some(neighbors) = self.adjacency_list.get(&from) {
                for &to in neighbors {
                    if !node_set.contains(&to) {
                        continue;
                    }

                    let from_node = if from == source || from == target {
                        from
                    } else {
                        offset + from * 2 + 1 // out-node
                    };

                    let to_node = if to == source || to == target {
                        to
                    } else {
                        offset + to * 2 // in-node
                    };

                    *capacity.entry((from_node, to_node)).or_insert(0) += 1;
                }
            }
        }

        // Run Ford-Fulkerson to find max flow
        let max_flow = self.ford_fulkerson_on_capacity_graph(source, target, &capacity);

        Ok(max_flow)
    }

    /// Find minimum node cut between two nodes in a subgraph
    /// Returns the set of nodes whose removal disconnects source from target
    fn find_min_node_cut_in_subgraph(
        &self,
        source: u64,
        target: u64,
        node_set: &FxHashSet<u64>,
    ) -> PyResult<Vec<u64>> {
        // Compute max flow first
        let max_flow = self.node_connectivity_between_in_subgraph(source, target, node_set)?;

        if max_flow == 0 {
            return Ok(vec![]);
        }

        // Transform graph and compute residual graph after max flow
        let mut capacity: FxHashMap<(u64, u64), usize> = FxHashMap::default();

        // Use large offset to avoid ID collisions
        let offset = 1_000_000_000u64;

        for &node in node_set.iter() {
            if node != source && node != target {
                let v_in = offset + node * 2;
                let v_out = offset + node * 2 + 1;
                capacity.insert((v_in, v_out), 1);
            }
        }

        for &from in node_set.iter() {
            if let Some(neighbors) = self.adjacency_list.get(&from) {
                for &to in neighbors {
                    if !node_set.contains(&to) {
                        continue;
                    }

                    let from_node = if from == source || from == target {
                        from
                    } else {
                        offset + from * 2 + 1
                    };

                    let to_node = if to == source || to == target {
                        to
                    } else {
                        offset + to * 2
                    };

                    capacity.insert((from_node, to_node), usize::MAX);
                }
            }
        }

        // Run max flow to get residual graph
        let residual = self.compute_residual_after_max_flow(source, target, &capacity);

        // Find min-cut: nodes reachable from source in residual graph
        let reachable_from_source = self.find_reachable_in_residual(source, &residual);

        // Min-cut nodes are those where v_in is reachable but v_out is not
        let mut cut_nodes = Vec::new();
        for &node in node_set.iter() {
            if node == source || node == target {
                continue;
            }

            let v_in = offset + node * 2;
            let v_out = offset + node * 2 + 1;

            if reachable_from_source.contains(&v_in) && !reachable_from_source.contains(&v_out) {
                cut_nodes.push(node);
            }
        }

        Ok(cut_nodes)
    }

    /// Ford-Fulkerson max flow on a capacity graph
    fn ford_fulkerson_on_capacity_graph(
        &self,
        source: u64,
        sink: u64,
        capacity: &FxHashMap<(u64, u64), usize>,
    ) -> usize {
        let mut residual = capacity.clone();
        let mut max_flow = 0;

        loop {
            let path = self.bfs_in_capacity_graph(source, sink, &residual);
            if path.is_empty() {
                break;
            }

            // Find bottleneck capacity
            let mut flow = usize::MAX;
            for i in 0..(path.len() - 1) {
                let edge = (path[i], path[i + 1]);
                flow = flow.min(*residual.get(&edge).unwrap_or(&0));
            }

            // Update residual graph
            for i in 0..(path.len() - 1) {
                let forward = (path[i], path[i + 1]);
                let backward = (path[i + 1], path[i]);

                *residual.entry(forward).or_insert(0) -= flow;
                *residual.entry(backward).or_insert(0) += flow;
            }

            max_flow += flow;
        }

        max_flow
    }

    /// BFS to find augmenting path in capacity graph
    fn bfs_in_capacity_graph(
        &self,
        source: u64,
        sink: u64,
        residual: &FxHashMap<(u64, u64), usize>,
    ) -> Vec<u64> {
        let mut visited = FxHashSet::default();
        let mut parent: FxHashMap<u64, u64> = FxHashMap::default();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(source);
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            if current == sink {
                let mut path = vec![sink];
                let mut node = sink;

                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }

                path.reverse();
                return path;
            }

            // Check all possible edges in residual graph
            for (&(from, to), &cap) in residual.iter() {
                if from == current && cap > 0 && !visited.contains(&to) {
                    visited.insert(to);
                    parent.insert(to, from);
                    queue.push_back(to);
                }
            }
        }

        Vec::new()
    }

    /// Compute residual graph after running max flow
    fn compute_residual_after_max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: &FxHashMap<(u64, u64), usize>,
    ) -> FxHashMap<(u64, u64), usize> {
        let mut residual = capacity.clone();

        loop {
            let path = self.bfs_in_capacity_graph(source, sink, &residual);
            if path.is_empty() {
                break;
            }

            let mut flow = usize::MAX;
            for i in 0..(path.len() - 1) {
                let edge = (path[i], path[i + 1]);
                flow = flow.min(*residual.get(&edge).unwrap_or(&0));
            }

            for i in 0..(path.len() - 1) {
                let forward = (path[i], path[i + 1]);
                let backward = (path[i + 1], path[i]);

                *residual.entry(forward).or_insert(0) -= flow;
                *residual.entry(backward).or_insert(0) += flow;
            }
        }

        residual
    }

    /// Find all nodes reachable from source in residual graph
    fn find_reachable_in_residual(
        &self,
        source: u64,
        residual: &FxHashMap<(u64, u64), usize>,
    ) -> FxHashSet<u64> {
        let mut reachable = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();

        reachable.insert(source);
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            for (&(from, to), &cap) in residual.iter() {
                if from == current && cap > 0 && !reachable.contains(&to) {
                    reachable.insert(to);
                    queue.push_back(to);
                }
            }
        }

        reachable
    }

    // ===== Walktrap Helper Methods =====

    /// Find connected components using BFS
    fn find_connected_components(&self) -> FxHashMap<u64, usize> {
        let mut components = FxHashMap::default();
        let mut visited = FxHashSet::default();
        let mut component_id = 0;

        for &start_node in self.nodes.keys() {
            if visited.contains(&start_node) {
                continue;
            }

            // BFS from this node
            let mut queue = vec![start_node];
            visited.insert(start_node);
            components.insert(start_node, component_id);

            while let Some(node) = queue.pop() {
                if let Some(neighbors) = self.adjacency_list.get(&node) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            components.insert(neighbor, component_id);
                            queue.push(neighbor);
                        }
                    }
                }
            }

            component_id += 1;
        }

        components
    }

    /// Compute transition probability matrix P
    /// P[i][j] = 1/deg(i) if edge (i,j) exists, 0 otherwise
    #[allow(non_snake_case)]
    fn compute_transition_matrix(
        &self,
        node_ids: &[u64],
        node_to_idx: &FxHashMap<u64, usize>,
    ) -> Vec<Vec<f64>> {
        let n = node_ids.len();
        let mut P = vec![vec![0.0; n]; n];

        for (i, &node_i) in node_ids.iter().enumerate() {
            let degree = self.degree(node_i) as f64;
            if degree == 0.0 {
                continue;
            }

            if let Some(neighbors) = self.adjacency_list.get(&node_i) {
                for &neighbor in neighbors {
                    if let Some(&j) = node_to_idx.get(&neighbor) {
                        P[i][j] = 1.0 / degree;
                    }
                }
            }
        }

        P
    }

    /// Compute P^t (matrix to the power of t)
    #[inline]
    #[allow(non_snake_case)]
    fn matrix_power(&self, P: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
        let n = P.len();
        let mut result = vec![vec![0.0; n]; n];

        // Initialize as identity matrix
        for (i, row) in result.iter_mut().enumerate() {
            row[i] = 1.0;
        }

        // Repeated matrix multiplication
        for _ in 0..t {
            result = self.matrix_multiply(&result, P);
        }

        result
    }

    /// Matrix multiplication
    #[inline]
    #[allow(non_snake_case)]
    fn matrix_multiply(&self, A: &[Vec<f64>], B: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = A.len();
        let mut C = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        C
    }

    /// Compute random walk distances between all pairs of nodes
    /// d(i,j) = sqrt(Σ_k [(P^t_ik - P^t_jk)^2 / deg(k)])
    #[allow(non_snake_case)]
    fn compute_random_walk_distances(
        &self,
        P_t: &[Vec<f64>],
        node_ids: &[u64],
        _node_to_idx: &FxHashMap<u64, usize>,
    ) -> Vec<Vec<f64>> {
        let n = node_ids.len();
        let mut distances = vec![vec![0.0; n]; n];

        // Precompute degrees
        let degrees: Vec<f64> = node_ids.iter()
            .map(|&node_id| self.degree(node_id) as f64)
            .collect();

        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..n {
                    if degrees[k] > 0.0 {
                        let diff = P_t[i][k] - P_t[j][k];
                        sum += diff * diff / degrees[k];
                    }
                }
                let dist = sum.sqrt();
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Ward's hierarchical clustering
    /// Returns dendrogram as Vec<(cluster1, cluster2, distance)>
    fn ward_clustering(&self, distances: &[Vec<f64>], n: usize) -> Vec<(usize, usize, f64)> {
        let mut dendrogram = Vec::new();
        let mut active_clusters: FxHashSet<usize> = (0..n).collect();
        let mut cluster_sizes = vec![1; n];
        let mut next_cluster_id = n;

        // Distance matrix (we'll update it as we merge)
        let mut dist_matrix = distances.to_vec();

        for _ in 0..(n - 1) {
            // Find closest pair of clusters
            let mut min_dist = f64::INFINITY;
            let mut best_i = 0;
            let mut best_j = 0;

            let active_vec: Vec<usize> = active_clusters.iter().copied().collect();
            for (idx_i, &i) in active_vec.iter().enumerate() {
                for &j in active_vec.iter().skip(idx_i + 1) {
                    let d = if i < dist_matrix.len() && j < dist_matrix[i].len() {
                        dist_matrix[i][j]
                    } else {
                        f64::INFINITY
                    };

                    if d < min_dist {
                        min_dist = d;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Merge clusters
            dendrogram.push((best_i, best_j, min_dist));
            active_clusters.remove(&best_i);
            active_clusters.remove(&best_j);

            // Create new cluster
            let new_size = cluster_sizes.get(best_i).copied().unwrap_or(1)
                         + cluster_sizes.get(best_j).copied().unwrap_or(1);
            if cluster_sizes.len() <= next_cluster_id {
                cluster_sizes.resize(next_cluster_id + 1, 0);
            }
            cluster_sizes[next_cluster_id] = new_size;

            // Update distances to other clusters (Ward's method)
            // Extend dist_matrix if needed
            if dist_matrix.len() <= next_cluster_id {
                dist_matrix.resize(next_cluster_id + 1, vec![]);
                for row in dist_matrix.iter_mut() {
                    if row.len() <= next_cluster_id {
                        row.resize(next_cluster_id + 1, 0.0);
                    }
                }
            }

            for &k in &active_clusters {
                if k == next_cluster_id {
                    continue;
                }

                let size_i = cluster_sizes.get(best_i).copied().unwrap_or(1) as f64;
                let size_j = cluster_sizes.get(best_j).copied().unwrap_or(1) as f64;
                let size_k = cluster_sizes.get(k).copied().unwrap_or(1) as f64;
                let size_new = size_i + size_j;

                let d_ik = if best_i < dist_matrix.len() && k < dist_matrix[best_i].len() {
                    dist_matrix[best_i][k]
                } else {
                    0.0
                };

                let d_jk = if best_j < dist_matrix.len() && k < dist_matrix[best_j].len() {
                    dist_matrix[best_j][k]
                } else {
                    0.0
                };

                let d_ij = min_dist;

                // Ward's formula
                let new_dist = ((size_i + size_k) * d_ik + (size_j + size_k) * d_jk
                              - size_k * d_ij) / (size_new + size_k);

                dist_matrix[next_cluster_id][k] = new_dist;
                dist_matrix[k][next_cluster_id] = new_dist;
            }

            active_clusters.insert(next_cluster_id);
            next_cluster_id += 1;
        }

        dendrogram
    }

    /// Find best partition from dendrogram by maximizing modularity
    fn find_best_partition(
        &self,
        dendrogram: &[(usize, usize, f64)],
        node_ids: &[u64],
        _node_to_idx: &FxHashMap<u64, usize>,
    ) -> Vec<usize> {
        let n = node_ids.len();

        if dendrogram.is_empty() {
            // No merges, each node is own community
            return (0..n).collect();
        }

        // Try all possible cuts of the dendrogram
        let mut best_partition = (0..n).collect::<Vec<_>>();
        let mut best_modularity = self.compute_modularity_from_partition(node_ids, &best_partition);

        // Try each level of merging
        for num_merges in 1..=dendrogram.len() {
            // Build partition by applying merges using union-find
            let mut parent: Vec<usize> = (0..n).collect();

            // Apply first num_merges from dendrogram
            for i in 0..num_merges {
                let (c1, c2, _) = dendrogram[i];

                // Get original nodes for these clusters
                let nodes_c1 = Self::get_cluster_nodes(c1, n, &dendrogram[..i]);
                let nodes_c2 = Self::get_cluster_nodes(c2, n, &dendrogram[..i]);

                // Union all nodes from c2 into c1's community
                if let (Some(rep1), Some(_rep2)) = (nodes_c1.iter().next(), nodes_c2.iter().next()) {
                    let root1 = Self::find_root(&mut parent, *rep1);
                    for &node in &nodes_c2 {
                        parent[node] = root1;
                    }
                }
            }

            // Compress paths and assign contiguous IDs
            let partition: Vec<usize> = (0..n).map(|i| Self::find_root(&mut parent, i)).collect();

            // Renumber to be contiguous
            let mut unique_comms: Vec<usize> = partition.to_vec();
            unique_comms.sort_unstable();
            unique_comms.dedup();

            let mut comm_remap = FxHashMap::default();
            for (new_id, &old_id) in unique_comms.iter().enumerate() {
                comm_remap.insert(old_id, new_id);
            }

            let renumbered: Vec<usize> = partition.iter()
                .map(|&c| comm_remap.get(&c).copied().unwrap_or(0))
                .collect();

            // Compute modularity for this partition
            #[allow(non_snake_case)]
            let Q = self.compute_modularity_from_partition(node_ids, &renumbered);

            if Q > best_modularity {
                best_modularity = Q;
                best_partition = renumbered;
            }
        }

        best_partition
    }

    /// Get all original nodes that belong to a cluster
    fn get_cluster_nodes(cluster_id: usize, n: usize, merges: &[(usize, usize, f64)]) -> FxHashSet<usize> {
        let mut nodes = FxHashSet::default();

        // If cluster_id < n, it's an original node
        if cluster_id < n {
            nodes.insert(cluster_id);
            return nodes;
        }

        // Otherwise, find which merge created this cluster
        let merge_idx = cluster_id - n;
        if merge_idx >= merges.len() {
            return nodes;
        }

        let (c1, c2, _) = merges[merge_idx];

        // Recursively get nodes from both children
        nodes.extend(Self::get_cluster_nodes(c1, n, merges));
        nodes.extend(Self::get_cluster_nodes(c2, n, merges));

        nodes
    }

    /// Union-Find helper: find root with path compression
    fn find_root(parent: &mut [usize], node: usize) -> usize {
        if parent[node] != node {
            parent[node] = Self::find_root(parent, parent[node]);
        }
        parent[node]
    }

    /// Compute modularity for a given partition
    #[allow(non_snake_case)]
    fn compute_modularity_from_partition(&self, node_ids: &[u64], partition: &[usize]) -> f64 {
        let m = self.num_edges() as f64;
        if m == 0.0 {
            return 0.0;
        }

        let mut Q = 0.0;

        for (i, &node_i) in node_ids.iter().enumerate() {
            for (j, &node_j) in node_ids.iter().enumerate() {
                if partition[i] == partition[j] {
                    let has_edge = self.adjacency_list.get(&node_i)
                        .map(|neighbors| neighbors.contains(&node_j))
                        .unwrap_or(false);

                    let A_ij = if has_edge { 1.0 } else { 0.0 };
                    let k_i = self.degree(node_i) as f64;
                    let k_j = self.degree(node_j) as f64;

                    Q += A_ij - (k_i * k_j) / (2.0 * m);
                }
            }
        }

        Q / (2.0 * m)
    }
}

// ========== Spectral Utilities ==========

/// Validate `which` and resolve how many eigenvalues to return
fn spectral_count(which: &str, k: Option<usize>, n: usize) -> PyResult<usize> {
    match which {
        "all" => Ok(n),
        "largest" | "smallest" => Ok(k.unwrap_or(6).min(n)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid 'which' parameter: '{}'. Use 'all', 'largest', or 'smallest'",
            which
        ))),
    }
}

/// Pick the requested eigenvalues from a fully sorted list, keeping its order
///
/// `descending` describes the sort order of `items`, which determines whether the
/// largest values sit at the front or the back.
fn spectral_select<T>(mut items: Vec<T>, which: &str, count: usize, descending: bool) -> Vec<T> {
    let take_front = match which {
        "all" => return items,
        "largest" => descending,
        _ => !descending,
    };
    let count = count.min(items.len());
    if take_front {
        items.truncate(count);
        items
    } else {
        let start = items.len() - count;
        items.split_off(start)
    }
}

/// Laplacian `L = D - A` (or `D^(-1/2) L D^(-1/2)`), with `D` the row sums of `A`
///
/// Isolated nodes get a zero row and column in the normalized variant.
pub(crate) fn spectral_laplacian(adj: &CsrMatrix, normalized: bool) -> CsrMatrix {
    let n = adj.n();
    let degrees = adj.row_sums();
    let scale: Vec<f64> = if normalized {
        degrees.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect()
    } else {
        vec![1.0; n]
    };

    let mut triplets = Vec::new();
    for i in 0..n {
        for (j, v) in adj.row(i) {
            if j != i {
                triplets.push((i, j, -v * scale[i] * scale[j]));
            }
        }
        triplets.push((i, i, (degrees[i] - adj.diagonal(i)) * scale[i] * scale[i]));
    }
    CsrMatrix::from_triplets(n, triplets)
}

/// Smallest graph size for which `"largest"`/`"smallest"` use the sparse solver
const SPARSE_SPECTRUM_MIN_NODES: usize = 400;

/// Largest graph size for which a failed sparse solve falls back to the dense solver
const DENSE_SPECTRUM_FALLBACK_MAX_NODES: usize = 3000;

/// Flip an eigenvector so its first non-negligible entry is positive
pub(crate) fn normalize_eigenvector_sign(vector: &mut [f64]) {
    if let Some(&first) = vector.iter().find(|v| v.abs() > 1e-12) {
        if first < 0.0 {
            vector.iter_mut().for_each(|v| *v = -*v);
        }
    }
}

/// (eigenpairs as (value, vector), first two eigenvalues of the full spectrum)
pub(crate) type SpectralPairs = (Vec<(f64, Vec<f64>)>, Vec<f64>);

/// Selected eigenpairs of a symmetric matrix plus its two leading eigenvalues
///
/// Pairs are in output order (`descending` or ascending by eigenvalue). The second
/// return value holds the first two eigenvalues of the *full* sorted spectrum (the
/// two largest if `descending`, else the two smallest), for spectral gap and
/// algebraic connectivity. Large sparse problems asking for a few extremal
/// eigenvalues use Chebyshev-filtered subspace iteration; everything else, and any
/// sparse solve that fails to converge on a moderately sized matrix, uses the dense
/// symmetric solver.
pub(crate) fn spectral_symmetric(
    matrix: &CsrMatrix,
    which: &str,
    count: usize,
    descending: bool,
) -> PyResult<SpectralPairs> {
    use crate::algorithms::sparse_eigen::{
        block_size, iteration_budget, symmetric_extremal_eigenpairs, Extremal, SubspaceOptions,
    };

    let n = matrix.n();
    let wanted = match which {
        "largest" => Some(Extremal::Largest),
        "smallest" => Some(Extremal::Smallest),
        _ => None,
    };
    let mut options = SubspaceOptions::default();
    let sparse_ok = |k: usize| n >= SPARSE_SPECTRUM_MIN_NODES && block_size(k, &options) * 8 <= n;

    if let Some(end) = wanted.filter(|_| sparse_ok(count.max(2))) {
        // With a dense fallback available, don't let a slow sparse solve cost more
        if n <= DENSE_SPECTRUM_FALLBACK_MAX_NODES {
            options.max_iter = iteration_budget(matrix, count.max(2), &options);
        }
        let leading_end = if descending { Extremal::Largest } else { Extremal::Smallest };
        let solve = |k: usize, end: Extremal| {
            symmetric_extremal_eigenpairs(matrix, k.min(n), end, &options)
        };

        let main = solve(if end == leading_end { count.max(2) } else { count }, end);
        let leading: Option<Vec<f64>> = match &main {
            Some(pairs) if end == leading_end => {
                Some(pairs.iter().take(2).map(|p| p.0).collect())
            }
            Some(_) => solve(2, leading_end).map(|p| p.into_iter().map(|p| p.0).collect()),
            None => None,
        };

        if let (Some(mut pairs), Some(leading)) = (main, leading) {
            pairs.truncate(count);
            // Solver output runs from the wanted end inwards
            if (end == Extremal::Largest) != descending {
                pairs.reverse();
            }
            pairs.iter_mut().for_each(|(_, v)| normalize_eigenvector_sign(v));
            return Ok((pairs, leading));
        }

        if n > DENSE_SPECTRUM_FALLBACK_MAX_NODES {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sparse eigenvalue computation failed to converge",
            ));
        }
    }

    let eigen = na::SymmetricEigen::new(matrix.to_dense());
    let mut pairs: Vec<(f64, Vec<f64>)> = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let mut vector: Vec<f64> = eigen.eigenvectors.column(idx).iter().copied().collect();
            normalize_eigenvector_sign(&mut vector);
            (value, vector)
        })
        .collect();

    if descending {
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    } else {
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let leading = pairs.iter().take(2).map(|p| p.0).collect();
    Ok((spectral_select(pairs, which, count, descending), leading))
}

/// Eigenvalues of a general (non-symmetric) real matrix
fn spectral_complex_eigenvalues(matrix: na::DMatrix<f64>) -> PyResult<Vec<na::Complex<f64>>> {
    crate::algorithms::eigen::general_eigenvalues(&matrix).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Eigenvalue computation failed to converge for directed graph",
        )
    })
}

/// Convert complex eigenvalues into a list of Python `complex` objects
fn spectral_complex_list<'py>(
    py: Python<'py>,
    values: &[na::Complex<f64>],
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        py,
        values
            .iter()
            .map(|c| pyo3::types::PyComplex::from_doubles(py, c.re, c.im)),
    )
}

// ============================================================================
// Test-support helpers (compiled only for `cargo test`)
// ============================================================================

/// Small, panicking convenience wrappers around `add_node_rust` / `add_edge_rust`
/// for use in Rust unit tests across the crate.
#[cfg(test)]
impl Graph {
    /// Add a node with the given label (id auto-assigned); returns the new node id.
    pub(crate) fn add_test_node(&mut self, label: impl Into<String>) -> u64 {
        self.add_node_rust(Node {
            id: 0,
            label: Some(label.into()),
            attributes: None,
        })
        .expect("add_test_node failed")
    }

    /// Add an unlabeled node with an explicit id (must be >= 1; id 0 means auto-assign).
    pub(crate) fn add_test_node_with_id(&mut self, id: u64) -> u64 {
        assert!(id != 0, "node id 0 is reserved for auto-assignment");
        self.add_node_rust(Node {
            id,
            label: None,
            attributes: None,
        })
        .expect("add_test_node_with_id failed")
    }

    /// Add an edge between two node ids; returns the new edge id.
    pub(crate) fn add_test_edge(&mut self, source: u64, target: u64, weight: Option<f64>) -> u64 {
        self.add_edge_rust(Edge {
            id: 0,
            source,
            target,
            weight,
            attributes: None,
        })
        .expect("add_test_edge failed")
    }

    /// Add an edge between two nodes identified by label; returns the new edge id.
    pub(crate) fn add_test_edge_by_label(
        &mut self,
        source: &str,
        target: &str,
        weight: Option<f64>,
    ) -> u64 {
        let s = self.id_from_label(source).expect("unknown source label");
        let t = self.id_from_label(target).expect("unknown target label");
        self.add_test_edge(s, t, weight)
    }

    /// Add a path of `n` unlabeled nodes; returns the node ids in path order.
    pub(crate) fn add_test_path(&mut self, n: usize) -> Vec<u64> {
        let ids: Vec<u64> = (0..n).map(|_| self.add_test_unlabeled_node()).collect();
        for w in ids.windows(2) {
            self.add_test_edge(w[0], w[1], None);
        }
        ids
    }

    /// Add a cycle of `n` unlabeled nodes; returns the node ids in cycle order.
    pub(crate) fn add_test_cycle(&mut self, n: usize) -> Vec<u64> {
        let ids = self.add_test_path(n);
        if n > 2 {
            self.add_test_edge(ids[n - 1], ids[0], None);
        }
        ids
    }

    /// Add a star of `n` unlabeled nodes in total (hub plus `n - 1` leaves);
    /// returns the node ids with the hub first.
    pub(crate) fn add_test_star(&mut self, n: usize) -> Vec<u64> {
        let ids: Vec<u64> = (0..n).map(|_| self.add_test_unlabeled_node()).collect();
        for &leaf in ids.iter().skip(1) {
            self.add_test_edge(ids[0], leaf, None);
        }
        ids
    }

    fn add_test_unlabeled_node(&mut self) -> u64 {
        self.add_node_rust(Node {
            id: 0,
            label: None,
            attributes: None,
        })
        .expect("add_test_unlabeled_node failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        let graph = Graph::new_rust(false, false);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
        assert!(!graph.is_directed());
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = Graph::new_rust(false, false);
        // id 0 = auto-assign
        let node1 = Node::new(0, Some("A".to_string()), None);
        let node2 = Node::new(0, Some("B".to_string()), None);

        assert!(graph.add_node(node1).is_ok());
        assert!(graph.add_node(node2).is_ok());
        assert_eq!(graph.num_nodes(), 2);
    }

    #[test]
    fn test_add_edges() {
        let mut graph = Graph::new_rust(false, false);

        // Add nodes first
        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");

        // Add edge (ID will be auto-assigned)
        let edge = Edge::new(0, a, b, Some(1.0), None);
        assert!(graph.add_edge(edge).is_ok());
        assert_eq!(graph.num_edges(), 1);

        // Check edge ID was auto-assigned
        let edge_ids = graph.edge_ids();
        assert_eq!(edge_ids.len(), 1);
        assert_eq!(edge_ids[0], 1);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = Graph::new_rust(false, false);

        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");
        let c = graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("A", "C", None);

        let neighbors = graph.neighbors(a).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b));
        assert!(neighbors.contains(&c));
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::new_rust(true, false);

        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");

        graph.add_test_edge_by_label("A", "B", None);

        // In directed graph, B should not have A as neighbor
        let b_neighbors = graph.neighbors(b).unwrap();
        assert_eq!(b_neighbors.len(), 0);

        let a_neighbors = graph.neighbors(a).unwrap();
        assert_eq!(a_neighbors.len(), 1);
    }
}

/// A (key, node) pair in AdaStar's OPEN heap, ordered by key (k1, then k2) and then by node
/// ID so ties resolve deterministically. Keys are compared with `total_cmp`.
#[derive(Clone, Copy, PartialEq)]
struct OpenEntry {
    key: (f64, f64),
    node: u64,
}

impl Eq for OpenEntry {}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.0.total_cmp(&other.key.0)
            .then(self.key.1.total_cmp(&other.key.1))
            .then(self.node.cmp(&other.node))
    }
}

/// Anytime Dynamic A* (ADA*) path planner
///
/// A stateful search algorithm that can:
/// - Return suboptimal paths quickly (anytime property)
/// - Iteratively improve paths with lower epsilon (anytime refinement)
/// - Handle dynamic graph changes efficiently (reuses computation)
///
/// Based on: Likhachev et al. "Anytime dynamic A*" ICAPS 2005
#[pyclass]
pub struct AdaStar {
    graph_id: Option<usize>,  // Track which graph this search is for
    source: u64,
    target: u64,
    weight_attr: String,
    epsilon: f64,

    // g-values: estimate of cost from state to goal
    state_to_goal_est: FxHashMap<u64, f64>,

    // rhs-values: one-step lookahead cost
    one_step_lookahead: FxHashMap<u64, f64>,

    // Priority queue sets
    open: FxHashMap<u64, (f64, f64)>,  // node -> (k1, k2) key
    // Min-heap mirror of `open` for O(log n) minimum lookup. Entries are never removed
    // eagerly: an entry is stale once its node left `open` or got a different key, and stale
    // entries are skipped when they reach the top (see `peek_min_open`)
    open_heap: std::collections::BinaryHeap<std::cmp::Reverse<OpenEntry>>,
    closed: FxHashSet<u64>,
    incons: FxHashSet<u64>,

    visited: FxHashSet<u64>,
    initialize: bool,

    // Store heuristic as Python callback
    heuristic: Option<PyObject>,
}

#[pymethods]
impl AdaStar {
    #[new]
    #[pyo3(signature = (source, target, weight="weight", initial_epsilon=1000.0, heuristic=None))]
    fn new(
        source: u64,
        target: u64,
        weight: &str,
        initial_epsilon: f64,
        heuristic: Option<PyObject>,
    ) -> Self {
        AdaStar {
            graph_id: None,
            source,
            target,
            weight_attr: weight.to_string(),
            epsilon: initial_epsilon,
            state_to_goal_est: FxHashMap::default(),
            one_step_lookahead: FxHashMap::default(),
            open: FxHashMap::default(),
            open_heap: std::collections::BinaryHeap::new(),
            closed: FxHashSet::default(),
            incons: FxHashSet::default(),
            visited: FxHashSet::default(),
            initialize: true,
            heuristic,
        }
    }

    /// Initialize the search with a graph
    fn initialize_search(&mut self, graph: &Graph) {
        self.graph_id = Some(graph as *const _ as usize);

        // Preallocate HashMaps with graph size
        let n = graph.nodes.len();
        self.state_to_goal_est = FxHashMap::with_capacity_and_hasher(n, Default::default());
        self.one_step_lookahead = FxHashMap::with_capacity_and_hasher(n, Default::default());
        self.open = FxHashMap::with_capacity_and_hasher(n, Default::default());
        self.open_heap.clear();
        self.closed = FxHashSet::with_capacity_and_hasher(n, Default::default());
        self.incons = FxHashSet::with_capacity_and_hasher(n, Default::default());
        self.visited = FxHashSet::with_capacity_and_hasher(n, Default::default());

        // Initialize g and rhs values
        for &node_id in graph.nodes.keys() {
            self.state_to_goal_est.insert(node_id, f64::INFINITY);
            self.one_step_lookahead.insert(node_id, f64::INFINITY);
        }

        // Target has rhs = 0
        self.one_step_lookahead.insert(self.target, 0.0);

        // Add target to OPEN with its key
        let key = self.compute_key(self.target, graph);
        self.open_insert(self.target, key);
    }

    /// Compute or improve the path with given epsilon
    fn compute_or_improve_path(&mut self, py: Python, graph: &Graph, epsilon: f64) -> PyResult<()> {
        // Initialize on first call
        if self.initialize {
            self.initialize_search(graph);
            self.epsilon = epsilon;
            self.initialize = false;
        } else {
            // Move INCONS to OPEN
            self.epsilon = epsilon;
            let incons: Vec<u64> = self.incons.drain().collect();
            for state in incons {
                let key = self.compute_key(state, graph);
                self.open_insert(state, key);
            }

            // Update all OPEN keys
            let n = self.open.len();
            let mut open_nodes = Vec::with_capacity(n);
            open_nodes.extend(self.open.keys().copied());
            for node in open_nodes {
                let key = self.compute_key(node, graph);
                self.open_insert(node, key);
            }

            self.closed.clear();
        }

        // Main search loop
        loop {
            // Get minimum key from OPEN
            let (min_node, min_key) = match self.peek_min_open() {
                Some((node, key)) => (Some(node), key),
                None => (None, (f64::INFINITY, f64::INFINITY)),
            };

            // Termination condition
            let source_key = self.compute_key(self.source, graph);
            let source_g = self.state_to_goal_est.get(&self.source).copied().unwrap_or(f64::INFINITY);
            let source_rhs = self.one_step_lookahead.get(&self.source).copied().unwrap_or(f64::INFINITY);

            if min_key >= source_key && source_g == source_rhs {
                break;
            }

            if let Some(u) = min_node {
                self.open.remove(&u);
                self.visited.insert(u);

                let g_u = self.state_to_goal_est.get(&u).copied().unwrap_or(f64::INFINITY);
                let rhs_u = self.one_step_lookahead.get(&u).copied().unwrap_or(f64::INFINITY);

                if g_u > rhs_u {
                    // Overconsistent: make consistent
                    self.state_to_goal_est.insert(u, rhs_u);
                    self.closed.insert(u);

                    // Update predecessors
                    for pred in Self::predecessors_of(u, graph) {
                        self.update_state(py, pred, graph)?;
                    }
                } else {
                    // Underconsistent: set g to infinity and update
                    self.state_to_goal_est.insert(u, f64::INFINITY);

                    // Update u and its predecessors
                    self.update_state(py, u, graph)?;
                    for pred in Self::predecessors_of(u, graph) {
                        self.update_state(py, pred, graph)?;
                    }
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Update graph with edge weight changes
    fn update_graph(&mut self, py: Python, graph: &Graph, changes: Vec<(u64, u64, f64)>) -> PyResult<()> {
        // Apply changes to graph (caller must do this)
        // Just update affected states

        for (u, v, _new_weight) in changes {
            self.update_state(py, u, graph)?;
            self.update_state(py, v, graph)?;
        }

        // Move INCONS to OPEN
        let incons: Vec<u64> = self.incons.drain().collect();
        for state in incons {
            let key = self.compute_key(state, graph);
            self.open_insert(state, key);
        }

        // Update all OPEN keys
        let n = self.open.len();
        let mut open_nodes = Vec::with_capacity(n);
        open_nodes.extend(self.open.keys().copied());
        for node in open_nodes {
            let key = self.compute_key(node, graph);
            self.open_insert(node, key);
        }

        self.closed.clear();

        Ok(())
    }

    /// Extract current path from source to target
    fn extract_path(&self, graph: &Graph) -> PyResult<Vec<u64>> {
        // Check if source is reachable from target (backward search direction)
        let source_g = self.state_to_goal_est.get(&self.source).copied().unwrap_or(f64::INFINITY);
        if source_g.is_infinite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("No path exists from {} to {}", self.source, self.target)
            ));
        }

        // Preallocate path with reasonable capacity (won't exceed n)
        let mut path = Vec::with_capacity(graph.nodes.len());
        path.push(self.source);
        let mut current = self.source;

        // Follow greedy policy: pick successor that minimizes cost(current,succ) + g(succ)
        // This reconstructs the optimal path found by the backward search
        while current != self.target {
            let neighbors = graph.adjacency_list.get(&current).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("No path exists: node {} has no neighbors", current)
                )
            })?;

            // Find neighbor that minimizes edge_cost + g_value
            let next = neighbors
                .iter()
                .min_by(|&&a, &&b| {
                    let g_a = self
                        .state_to_goal_est
                        .get(&a)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let g_b = self
                        .state_to_goal_est
                        .get(&b)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let cost_a = self.get_edge_weight(current, a, graph) + g_a;
                    let cost_b = self.get_edge_weight(current, b, graph) + g_b;
                    cost_a
                        .partial_cmp(&cost_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("No path exists from {} to {}", self.source, self.target)
                    )
                })?;

            if path.len() > graph.nodes.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Path contains cycle or is too long"
                ));
            }

            path.push(next);
            current = next;
        }

        Ok(path)
    }
}

impl AdaStar {
    /// Update the state (rhs value) of a node
    fn update_state(&mut self, _py: Python, n: u64, graph: &Graph) -> PyResult<()> {
        if n != self.target {
            // Compute rhs(n) = min over successors of (g(s') + c(n, s'))
            let mut min_rhs = f64::INFINITY;

            if let Some(neighbors) = graph.adjacency_list.get(&n) {
                for &succ in neighbors {
                    let g_succ = self.state_to_goal_est.get(&succ).copied().unwrap_or(f64::INFINITY);
                    let edge_cost = self.get_edge_weight(n, succ, graph);
                    let rhs_candidate = g_succ + edge_cost;

                    if rhs_candidate < min_rhs {
                        min_rhs = rhs_candidate;
                    }
                }
            }

            self.one_step_lookahead.insert(n, min_rhs);
        }

        // Remove from OPEN if present
        self.open.remove(&n);

        // Update OPEN/INCONS based on consistency
        let g_n = self.state_to_goal_est.get(&n).copied().unwrap_or(f64::INFINITY);
        let rhs_n = self.one_step_lookahead.get(&n).copied().unwrap_or(f64::INFINITY);

        if g_n != rhs_n {
            if !self.closed.contains(&n) {
                let key = self.compute_key(n, graph);
                self.open_insert(n, key);
            } else {
                self.incons.insert(n);
            }
        }

        Ok(())
    }

    /// Nodes with an edge into `u`, from the graph's edge index. This used to scan every
    /// node's adjacency list on each expansion, which made the search O(V²) on its own.
    fn predecessors_of(u: u64, graph: &Graph) -> Vec<u64> {
        graph.edge_index().predecessors.get(&u).cloned().unwrap_or_default()
    }

    /// Insert or re-key a node in OPEN, mirroring it into the heap
    fn open_insert(&mut self, node: u64, key: (f64, f64)) {
        self.open.insert(node, key);
        self.open_heap.push(std::cmp::Reverse(OpenEntry { key, node }));
        // Bulk re-keying (new epsilon, graph updates) leaves stale entries behind; rebuild
        // from `open` before they outnumber the live ones too much
        if self.open_heap.len() > 4 * self.open.len() + 64 {
            self.open_heap = self
                .open
                .iter()
                .map(|(&node, &key)| std::cmp::Reverse(OpenEntry { key, node }))
                .collect();
        }
    }

    /// The OPEN node with the smallest key (ties broken by smallest node ID), without
    /// removing it. This used to scan all of OPEN on every expansion.
    fn peek_min_open(&mut self) -> Option<(u64, (f64, f64))> {
        while let Some(std::cmp::Reverse(top)) = self.open_heap.peek() {
            match self.open.get(&top.node) {
                Some(&key) if key == top.key => return Some((top.node, key)),
                _ => {
                    self.open_heap.pop(); // stale: removed from OPEN or re-keyed
                }
            }
        }
        None
    }

    /// Compute priority key for a node
    #[inline]
    fn compute_key(&self, n: u64, _graph: &Graph) -> (f64, f64) {
        let g_n = self.state_to_goal_est.get(&n).copied().unwrap_or(f64::INFINITY);
        let rhs_n = self.one_step_lookahead.get(&n).copied().unwrap_or(f64::INFINITY);

        // Heuristic estimates distance from n to source (backward search direction)
        let h = self.get_heuristic(n, self.source);

        // ARA*/ADA* key: (min(g,rhs) + epsilon*h, min(g,rhs))
        let min_val = if g_n < rhs_n { g_n } else { rhs_n };
        (min_val + self.epsilon * h, min_val)
    }

    /// Get heuristic value
    #[inline]
    fn get_heuristic(&self, from: u64, to: u64) -> f64 {
        if let Some(ref heuristic_fn) = self.heuristic {
            Python::with_gil(|py| {
                match heuristic_fn.call1(py, (from, to)) {
                    Ok(result) => result.extract::<f64>(py).unwrap_or(0.0),
                    Err(_) => 0.0,
                }
            })
        } else {
            0.0  // Default: Dijkstra-like (no heuristic)
        }
    }

    /// Get edge weight from graph
    #[inline]
    fn get_edge_weight(&self, from: u64, to: u64, graph: &Graph) -> f64 {
        // Cost of stepping from `from` to `to`: the cheapest edge stored as `from -> to` or, on
        // an undirected graph, as `to -> from`. Each edge's weight comes from its weight field,
        // else the `weight_attr` attribute, else 1.0. Positions come from the graph's edge
        // index rather than a scan of every edge.
        let index = graph.edge_index();
        let forward = index.positions_by_pair.get(&(from, to));
        let reverse = if graph.directed || from == to {
            None
        } else {
            index.positions_by_pair.get(&(to, from))
        };

        let mut best = f64::INFINITY; // stays infinite if no edge exists
        for &position in forward.into_iter().chain(reverse).flatten() {
            best = best.min(self.resolve_edge_weight(&graph.edges[position]));
        }
        best
    }

    /// An edge's weight for AdaStar: the weight field, else the `weight_attr` attribute, else 1.0
    fn resolve_edge_weight(&self, edge: &Edge) -> f64 {
        if let Some(w) = edge.weight {
            return w;
        }
        if let Some(ref attrs) = edge.attributes {
            return Python::with_gil(|py| {
                let dict = attrs.bind(py);
                if let Ok(Some(weight)) = dict.get_item(&self.weight_attr) {
                    if let Ok(w) = weight.extract::<f64>() {
                        return w;
                    }
                }
                1.0 // Default weight
            });
        }
        1.0
    }
}

// ============================================================================
// Matrix Operations for Walk Counting
// ============================================================================

/// Multiply two square matrices
#[inline]
fn multiply_matrices(a: &[Vec<u64>], b: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let n = a.len();
    let mut result = vec![vec![0u64; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] = result[i][j].saturating_add(a[i][k].saturating_mul(b[k][j]));
            }
        }
    }

    result
}

/// Compute matrix power using fast exponentiation
/// Time complexity: O(n³ × log k) where n is matrix size and k is the power
#[inline]
fn matrix_power(matrix: &[Vec<u64>], power: usize) -> Vec<Vec<u64>> {
    let n = matrix.len();

    if power == 0 {
        // Return identity matrix
        let mut identity = vec![vec![0u64; n]; n];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1;
        }
        return identity;
    }

    if power == 1 {
        return matrix.to_vec();
    }

    // Fast exponentiation: A^k = (A^(k/2))^2 if k is even
    //                      A^k = A × A^(k-1) if k is odd
    if power % 2 == 0 {
        let half_power = matrix_power(matrix, power / 2);
        multiply_matrices(&half_power, &half_power)
    } else {
        let sub_power = matrix_power(matrix, power - 1);
        multiply_matrices(matrix, &sub_power)
    }
}
