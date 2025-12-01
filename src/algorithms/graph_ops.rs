//! Graph algebra operations
//!
//! This module provides set-theoretic operations on graphs:
//! - union: combines two graphs (all nodes and edges from both)
//! - intersection: edges present in both graphs
//! - difference: edges in G1 but not in G2
//! - symmetric_difference: edges in either graph but not both
//! - compose: like union but also handles node label/attribute merging

use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{Edge, Graph, Node};

/// Helper to create a normalized edge key for comparison.
/// For undirected graphs, ensures (min, max) ordering.
#[inline]
fn edge_key(source: u64, target: u64, directed: bool) -> (u64, u64) {
    if directed || source <= target {
        (source, target)
    } else {
        (target, source)
    }
}

/// Compute the union of two graphs.
///
/// Returns a new graph containing all nodes and edges from both graphs.
/// Nodes with the same ID are merged (G1's attributes take precedence).
/// Duplicate edges are not added.
///
/// # Arguments
/// * `g1` - First graph
/// * `g2` - Second graph
///
/// # Returns
/// A new graph that is the union of G1 and G2.
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_union(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute union of directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted() || g2.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Track which edges we've added to avoid duplicates
    let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

    // Add all nodes from G1
    for &node_id in g1.get_node_ids().iter() {
        let label = g1.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add nodes from G2 that aren't in G1
    for &node_id in g2.get_node_ids().iter() {
        if !result.has_node_id(node_id) {
            let label = g2.label_from_id(node_id).ok().flatten();
            let node = Node {
                id: node_id,
                label,
                attributes: None,
            };
            result.add_node_rust(node)?;
        }
    }

    // Add all edges from G1
    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if !added_edges.contains(&key) {
            let new_edge = Edge {
                id: 0, // Auto-assign
                source: edge.source,
                target: edge.target,
                weight: edge.weight,
                attributes: None,
            };
            result.add_edge_rust(new_edge)?;
            added_edges.insert(key);
        }
    }

    // Add edges from G2 that aren't already present
    for edge in g2.get_edges_internal() {
        // Only add if both nodes exist in result
        if result.has_node_id(edge.source) && result.has_node_id(edge.target) {
            let key = edge_key(edge.source, edge.target, directed);
            if !added_edges.contains(&key) {
                let new_edge = Edge {
                    id: 0,
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight,
                    attributes: None,
                };
                result.add_edge_rust(new_edge)?;
                added_edges.insert(key);
            }
        }
    }

    Ok(result)
}

/// Compute the intersection of two graphs.
///
/// Returns a new graph containing only edges that exist in both graphs.
/// Nodes are included only if they are endpoints of common edges.
///
/// # Arguments
/// * `g1` - First graph
/// * `g2` - Second graph
///
/// # Returns
/// A new graph containing edges present in both G1 and G2.
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_intersection(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute intersection of directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted() || g2.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Build set of edges in G2 for fast lookup
    let mut g2_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    for edge in g2.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        g2_edges.insert(key);
    }

    // Track nodes we need to add
    let mut nodes_to_add: FxHashSet<u64> = FxHashSet::default();
    let mut edges_to_add: Vec<(u64, u64, Option<f64>)> = Vec::new();

    // Find edges present in both graphs
    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if g2_edges.contains(&key) {
            nodes_to_add.insert(edge.source);
            nodes_to_add.insert(edge.target);
            edges_to_add.push((edge.source, edge.target, edge.weight));
        }
    }

    // Add necessary nodes (use G1's labels)
    for &node_id in &nodes_to_add {
        let label = g1.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add intersection edges
    for (source, target, weight) in edges_to_add {
        let edge = Edge {
            id: 0,
            source,
            target,
            weight,
            attributes: None,
        };
        result.add_edge_rust(edge)?;
    }

    Ok(result)
}

/// Compute the difference of two graphs.
///
/// Returns a new graph containing edges that are in G1 but not in G2.
/// Nodes from G1 are preserved if they are endpoints of remaining edges.
///
/// # Arguments
/// * `g1` - First graph (base)
/// * `g2` - Second graph (to subtract)
///
/// # Returns
/// A new graph containing edges in G1 but not in G2.
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_difference(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute difference of directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Build set of edges in G2 for fast lookup
    let mut g2_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    for edge in g2.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        g2_edges.insert(key);
    }

    // Track nodes we need to add
    let mut nodes_to_add: FxHashSet<u64> = FxHashSet::default();
    let mut edges_to_add: Vec<(u64, u64, Option<f64>)> = Vec::new();

    // Find edges in G1 that are NOT in G2
    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if !g2_edges.contains(&key) {
            nodes_to_add.insert(edge.source);
            nodes_to_add.insert(edge.target);
            edges_to_add.push((edge.source, edge.target, edge.weight));
        }
    }

    // Add necessary nodes
    for &node_id in &nodes_to_add {
        let label = g1.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add difference edges
    for (source, target, weight) in edges_to_add {
        let edge = Edge {
            id: 0,
            source,
            target,
            weight,
            attributes: None,
        };
        result.add_edge_rust(edge)?;
    }

    Ok(result)
}

/// Compute the symmetric difference of two graphs.
///
/// Returns a new graph containing edges that are in exactly one of the two graphs.
/// Equivalent to (G1 - G2) ∪ (G2 - G1).
///
/// # Arguments
/// * `g1` - First graph
/// * `g2` - Second graph
///
/// # Returns
/// A new graph containing edges in G1 XOR G2.
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_symmetric_difference(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute symmetric difference of directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted() || g2.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Build sets of edges for both graphs
    let mut g1_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    let mut g2_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        g1_edges.insert(key);
    }

    for edge in g2.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        g2_edges.insert(key);
    }

    // Track nodes and edges to add
    let mut nodes_to_add: FxHashSet<u64> = FxHashSet::default();
    let mut edges_to_add: Vec<(u64, u64, Option<f64>)> = Vec::new();

    // Edges in G1 but not G2
    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if !g2_edges.contains(&key) {
            nodes_to_add.insert(edge.source);
            nodes_to_add.insert(edge.target);
            edges_to_add.push((edge.source, edge.target, edge.weight));
        }
    }

    // Edges in G2 but not G1
    for edge in g2.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if !g1_edges.contains(&key) {
            nodes_to_add.insert(edge.source);
            nodes_to_add.insert(edge.target);
            edges_to_add.push((edge.source, edge.target, edge.weight));
        }
    }

    // Add necessary nodes (prefer G1 labels, fall back to G2)
    for &node_id in &nodes_to_add {
        let label = g1
            .label_from_id(node_id)
            .ok()
            .flatten()
            .or_else(|| g2.label_from_id(node_id).ok().flatten());
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add symmetric difference edges
    for (source, target, weight) in edges_to_add {
        let edge = Edge {
            id: 0,
            source,
            target,
            weight,
            attributes: None,
        };
        result.add_edge_rust(edge)?;
    }

    Ok(result)
}

/// Compose two graphs (union with node merging).
///
/// Similar to union, but G2's node attributes/labels can override G1's.
/// This is useful when you want to merge updates from G2 into G1.
///
/// # Arguments
/// * `g1` - First graph (base)
/// * `g2` - Second graph (updates take precedence)
///
/// # Returns
/// A new graph that is the composition of G1 and G2.
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_compose(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compose directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted() || g2.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Track which edges we've added to avoid duplicates
    let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

    // Collect all node IDs from both graphs
    let mut all_node_ids: FxHashSet<u64> = FxHashSet::default();
    for &node_id in g1.get_node_ids().iter() {
        all_node_ids.insert(node_id);
    }
    for &node_id in g2.get_node_ids().iter() {
        all_node_ids.insert(node_id);
    }

    // Add all nodes (G2 labels take precedence over G1)
    for node_id in all_node_ids {
        // Prefer G2 label if node exists in G2, otherwise use G1
        let label = if g2.has_node_id(node_id) {
            g2.label_from_id(node_id).ok().flatten()
        } else {
            g1.label_from_id(node_id).ok().flatten()
        };

        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add all edges from G1
    for edge in g1.get_edges_internal() {
        let key = edge_key(edge.source, edge.target, directed);
        if !added_edges.contains(&key) {
            let new_edge = Edge {
                id: 0,
                source: edge.source,
                target: edge.target,
                weight: edge.weight,
                attributes: None,
            };
            result.add_edge_rust(new_edge)?;
            added_edges.insert(key);
        }
    }

    // Add edges from G2 that aren't already present
    for edge in g2.get_edges_internal() {
        if result.has_node_id(edge.source) && result.has_node_id(edge.target) {
            let key = edge_key(edge.source, edge.target, directed);
            if !added_edges.contains(&key) {
                let new_edge = Edge {
                    id: 0,
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight,
                    attributes: None,
                };
                result.add_edge_rust(new_edge)?;
                added_edges.insert(key);
            }
        }
    }

    Ok(result)
}

/// Disjoint union of two graphs.
///
/// Combines graphs by renumbering G2's nodes to avoid conflicts with G1.
/// The resulting graph has all nodes and edges from both, with no overlap.
///
/// # Arguments
/// * `g1` - First graph
/// * `g2` - Second graph
///
/// # Returns
/// A new graph that is the disjoint union of G1 and G2.
/// G2's nodes are renumbered starting after G1's maximum node ID.
/// If G2 nodes have labels that conflict with G1, G2's labels are suffixed with "_2".
///
/// # Errors
/// Returns an error if the graphs have different directedness.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn graph_disjoint_union(py: Python<'_>, g1: &Graph, g2: &Graph) -> PyResult<Graph> {
    // Validate compatible graph types
    if g1.is_directed() != g2.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute disjoint union of directed and undirected graphs",
        ));
    }

    let directed = g1.is_directed();
    let weighted = g1.is_weighted() || g2.is_weighted();

    let mut result = Graph::new_rust(directed, weighted);

    // Collect all labels from G1 to detect conflicts (track labels in result)
    let mut used_labels: FxHashSet<String> = FxHashSet::default();
    for &node_id in g1.get_node_ids().iter() {
        if let Some(label) = g1.label_from_id(node_id).ok().flatten() {
            used_labels.insert(label);
        }
    }

    // Add all nodes from G1 as-is
    for &node_id in g1.get_node_ids().iter() {
        let label = g1.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Find the maximum node ID in G1 to offset G2
    let max_g1_id = g1.get_node_ids().iter().max().copied().unwrap_or(0);
    let offset = max_g1_id;

    // Build mapping from old G2 IDs to new IDs
    let mut g2_id_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for &old_id in g2.get_node_ids().iter() {
        let new_id = old_id + offset;
        g2_id_map.insert(old_id, new_id);

        // Get label from G2, and handle conflicts with used labels
        let original_label = g2.label_from_id(old_id).ok().flatten();
        let label = match original_label {
            Some(lbl) => {
                if used_labels.contains(&lbl) {
                    // Label conflicts, find a unique suffix
                    let mut suffix = 2;
                    let mut new_label = format!("{}_{}", lbl, suffix);
                    while used_labels.contains(&new_label) {
                        suffix += 1;
                        new_label = format!("{}_{}", lbl, suffix);
                    }
                    used_labels.insert(new_label.clone());
                    Some(new_label)
                } else {
                    used_labels.insert(lbl.clone());
                    Some(lbl)
                }
            }
            None => None,
        };
        let node = Node {
            id: new_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    // Add all edges from G1
    for edge in g1.get_edges_internal() {
        let new_edge = Edge {
            id: 0,
            source: edge.source,
            target: edge.target,
            weight: edge.weight,
            attributes: None,
        };
        result.add_edge_rust(new_edge)?;
    }

    // Add all edges from G2 with remapped node IDs
    for edge in g2.get_edges_internal() {
        let new_source = g2_id_map[&edge.source];
        let new_target = g2_id_map[&edge.target];
        let new_edge = Edge {
            id: 0,
            source: new_source,
            target: new_target,
            weight: edge.weight,
            attributes: None,
        };
        result.add_edge_rust(new_edge)?;
    }

    Ok(result)
}

/// Compute the transitive closure of a directed graph.
///
/// The transitive closure is a graph where there is an edge (u, v) if and only if
/// there is a path from u to v in the original graph. Uses a BFS-based approach
/// from each node to find all reachable nodes.
///
/// # Arguments
/// * `graph` - The input directed graph
///
/// # Returns
/// A new directed graph representing the transitive closure.
///
/// # Errors
/// Returns an error if the graph is not directed.
///
/// # Time Complexity
/// O(V * (V + E)) where V is the number of vertices and E is the number of edges.
#[pyfunction]
pub fn transitive_closure(graph: &Graph) -> PyResult<Graph> {
    // Validate directed graph
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "transitive_closure() only works on directed graphs",
        ));
    }

    let mut result = Graph::new_rust(true, false);

    // Map old node IDs to new node IDs
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add all nodes
    for node in graph.get_nodes().values() {
        let new_node = Node {
            id: 0, // Auto-assign
            label: node.label.clone(),
            attributes: None,
        };
        let new_id = result.add_node_rust(new_node)?;
        node_map.insert(node.id, new_id);
    }

    let adj_list = graph.get_adjacency_list();
    let node_ids: Vec<u64> = graph.get_node_ids();

    // For each node, find all reachable nodes using BFS
    for &source in &node_ids {
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();

        // Start BFS from source's neighbors (not source itself for self-loops handling)
        if let Some(neighbors) = adj_list.get(&source) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        // BFS to find all reachable nodes
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Add edges from source to all reachable nodes
        let new_source = node_map[&source];
        for &target in &visited {
            let new_target = node_map[&target];
            let edge = Edge {
                id: 0,
                source: new_source,
                target: new_target,
                weight: None,
                attributes: None,
            };
            result.add_edge_rust(edge)?;
        }
    }

    Ok(result)
}

/// Compute the transitive reduction of a directed acyclic graph (DAG).
///
/// The transitive reduction is the minimal graph that has the same transitive closure
/// as the original graph. In other words, it removes redundant edges while preserving
/// all reachability relationships.
///
/// # Arguments
/// * `graph` - The input directed acyclic graph (DAG)
///
/// # Returns
/// A new directed graph representing the transitive reduction.
///
/// # Errors
/// Returns an error if the graph is not directed or contains cycles.
///
/// # Algorithm
/// Uses topological sort followed by edge analysis. For each edge (u, v), we check if
/// there's an alternate path from u to v that doesn't use the edge directly. If so,
/// the edge is redundant and removed.
///
/// # Time Complexity
/// O(V * E) where V is the number of vertices and E is the number of edges.
#[pyfunction]
pub fn transitive_reduction(graph: &Graph) -> PyResult<Graph> {
    // Validate directed graph
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "transitive_reduction() only works on directed graphs",
        ));
    }

    // Verify it's a DAG by attempting topological sort
    let topo_order = graph.get_topological_sort_order()?;

    let mut result = Graph::new_rust(true, false);

    // Map old node IDs to new node IDs
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add all nodes
    for node in graph.get_nodes().values() {
        let new_node = Node {
            id: 0, // Auto-assign
            label: node.label.clone(),
            attributes: None,
        };
        let new_id = result.add_node_rust(new_node)?;
        node_map.insert(node.id, new_id);
    }

    // Build position map from topological order (for efficient path checking)
    let mut topo_pos: FxHashMap<u64, usize> = FxHashMap::default();
    for (pos, &node_id) in topo_order.iter().enumerate() {
        topo_pos.insert(node_id, pos);
    }

    let adj_list = graph.get_adjacency_list();

    // For each edge (u, v), check if there's an alternate path from u to v
    // An edge is redundant if v is reachable from u through other edges
    for edge in graph.get_edges_internal() {
        let u = edge.source;
        let v = edge.target;

        // Check if v is reachable from u without using the direct edge
        // We do a BFS from u's other successors to see if we can reach v
        let mut reachable_without_direct = false;

        if let Some(u_neighbors) = adj_list.get(&u) {
            // Get neighbors other than v
            let other_neighbors: Vec<u64> = u_neighbors.iter()
                .filter(|&&n| n != v)
                .copied()
                .collect();

            if !other_neighbors.is_empty() {
                // BFS to see if v is reachable
                let mut visited: FxHashSet<u64> = FxHashSet::default();
                let mut queue = std::collections::VecDeque::new();

                for &neighbor in &other_neighbors {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }

                while let Some(current) = queue.pop_front() {
                    if current == v {
                        reachable_without_direct = true;
                        break;
                    }
                    if let Some(neighbors) = adj_list.get(&current) {
                        for &neighbor in neighbors {
                            if !visited.contains(&neighbor) {
                                visited.insert(neighbor);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }
        }

        // Keep the edge only if v is NOT reachable without the direct edge
        if !reachable_without_direct {
            let new_edge = Edge {
                id: 0,
                source: node_map[&u],
                target: node_map[&v],
                weight: edge.weight,
                attributes: None,
            };
            result.add_edge_rust(new_edge)?;
        }
    }

    Ok(result)
}

/// Extract an induced subgraph containing the specified nodes.
///
/// Returns a new graph containing only the specified nodes and all edges
/// between them (the induced subgraph). Node labels and edge weights are preserved.
///
/// # Arguments
/// * `graph` - The input graph
/// * `nodes` - List of node IDs to include in the subgraph
///
/// # Returns
/// A new graph containing only the specified nodes and edges between them.
///
/// # Errors
/// Returns an error if any node ID is not found in the graph.
#[pyfunction]
#[pyo3(signature = (graph, nodes))]
pub fn subgraph(graph: &Graph, nodes: Vec<u64>) -> PyResult<Graph> {
    // Create a set for efficient lookup
    let node_set: FxHashSet<u64> = nodes.iter().copied().collect();

    // Validate all nodes exist
    for &node_id in &nodes {
        if !graph.get_nodes().contains_key(&node_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node with ID {} not found in graph",
                node_id
            )));
        }
    }

    // Create new graph with same directedness and weightedness
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Map old node IDs to new node IDs
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add nodes to the new graph
    for &node_id in &nodes {
        if let Some(node) = graph.get_nodes().get(&node_id) {
            let new_node = Node {
                id: 0, // Will be auto-assigned
                label: node.label.clone(),
                attributes: None, // Attributes cannot be cloned across PyO3 boundary
            };
            let new_id = result.add_node_rust(new_node)?;
            node_map.insert(node_id, new_id);
        }
    }

    // Add edges where both endpoints are in the node set
    for edge in graph.get_edges_internal() {
        if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
            if let (Some(&new_source), Some(&new_target)) =
                (node_map.get(&edge.source), node_map.get(&edge.target))
            {
                let new_edge = Edge {
                    id: 0,
                    source: new_source,
                    target: new_target,
                    weight: edge.weight,
                    attributes: None, // Attributes cannot be cloned across PyO3 boundary
                };
                result.add_edge_rust(new_edge)?;
            }
        }
    }

    Ok(result)
}

/// Extract a subgraph containing the specified edges and their endpoint nodes.
///
/// Returns a new graph containing only the specified edges and the nodes
/// that are endpoints of those edges. Node labels and edge weights are preserved.
///
/// # Arguments
/// * `graph` - The input graph
/// * `edges` - List of edge IDs to include in the subgraph
///
/// # Returns
/// A new graph containing only the specified edges and their endpoint nodes.
///
/// # Errors
/// Returns an error if any edge ID is not found in the graph.
#[pyfunction]
#[pyo3(signature = (graph, edges))]
pub fn edge_subgraph(graph: &Graph, edges: Vec<u64>) -> PyResult<Graph> {
    // Create a set for efficient lookup
    let edge_set: FxHashSet<u64> = edges.iter().copied().collect();

    // Collect all nodes that are endpoints of the selected edges
    let mut node_ids: FxHashSet<u64> = FxHashSet::default();

    // Validate edges and collect endpoint nodes
    for &edge_id in &edges {
        let mut found = false;
        for edge in graph.get_edges_internal() {
            if edge.id == edge_id {
                node_ids.insert(edge.source);
                node_ids.insert(edge.target);
                found = true;
                break;
            }
        }
        if !found {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Edge with ID {} not found in graph",
                edge_id
            )));
        }
    }

    // Create new graph with same directedness and weightedness
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Map old node IDs to new node IDs
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add nodes to the new graph
    for &node_id in &node_ids {
        if let Some(node) = graph.get_nodes().get(&node_id) {
            let new_node = Node {
                id: 0,
                label: node.label.clone(),
                attributes: None, // Attributes cannot be cloned across PyO3 boundary
            };
            let new_id = result.add_node_rust(new_node)?;
            node_map.insert(node_id, new_id);
        }
    }

    // Add only the selected edges
    for edge in graph.get_edges_internal() {
        if edge_set.contains(&edge.id) {
            if let (Some(&new_source), Some(&new_target)) =
                (node_map.get(&edge.source), node_map.get(&edge.target))
            {
                let new_edge = Edge {
                    id: 0,
                    source: new_source,
                    target: new_target,
                    weight: edge.weight,
                    attributes: None, // Attributes cannot be cloned across PyO3 boundary
                };
                result.add_edge_rust(new_edge)?;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triangle() -> Graph {
        let mut g = Graph::new_rust(false, false);
        g.add_node_rust(Node { id: 1, label: Some("A".to_string()), attributes: None }).unwrap();
        g.add_node_rust(Node { id: 2, label: Some("B".to_string()), attributes: None }).unwrap();
        g.add_node_rust(Node { id: 3, label: Some("C".to_string()), attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 1, target: 2, weight: None, attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 2, target: 3, weight: None, attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 3, target: 1, weight: None, attributes: None }).unwrap();
        g
    }

    fn create_square() -> Graph {
        let mut g = Graph::new_rust(false, false);
        g.add_node_rust(Node { id: 1, label: Some("A".to_string()), attributes: None }).unwrap();
        g.add_node_rust(Node { id: 2, label: Some("B".to_string()), attributes: None }).unwrap();
        g.add_node_rust(Node { id: 3, label: Some("C".to_string()), attributes: None }).unwrap();
        g.add_node_rust(Node { id: 4, label: Some("D".to_string()), attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 1, target: 2, weight: None, attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 2, target: 3, weight: None, attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 3, target: 4, weight: None, attributes: None }).unwrap();
        g.add_edge_rust(Edge { id: 0, source: 4, target: 1, weight: None, attributes: None }).unwrap();
        g
    }

    #[test]
    fn test_union_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = create_triangle();
            let g2 = create_square();

            let result = graph_union(py, &g1, &g2).unwrap();

            // Should have 4 nodes (1,2,3 from both, 4 from square)
            assert_eq!(result.num_nodes(), 4);
            // Triangle has 3 edges, square has 4, but edges 1-2 and 2-3 are shared
            // So total = 3 + 4 - 2 = 5 edges
            assert_eq!(result.num_edges(), 5);
        });
    }

    #[test]
    fn test_intersection_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = create_triangle();
            let g2 = create_square();

            let result = graph_intersection(py, &g1, &g2).unwrap();

            // Shared edges are 1-2 and 2-3
            assert_eq!(result.num_edges(), 2);
            // Nodes are 1, 2, 3 (endpoints of shared edges)
            assert_eq!(result.num_nodes(), 3);
        });
    }

    #[test]
    fn test_difference_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = create_triangle();
            let g2 = create_square();

            let result = graph_difference(py, &g1, &g2).unwrap();

            // Triangle has edges: 1-2, 2-3, 3-1
            // Square has edges: 1-2, 2-3, 3-4, 4-1
            // Difference: 3-1 only (1 edge)
            assert_eq!(result.num_edges(), 1);
        });
    }

    #[test]
    fn test_symmetric_difference_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = create_triangle();
            let g2 = create_square();

            let result = graph_symmetric_difference(py, &g1, &g2).unwrap();

            // Triangle: 1-2, 2-3, 3-1
            // Square: 1-2, 2-3, 3-4, 4-1
            // XOR: 3-1, 3-4, 4-1 (3 edges)
            assert_eq!(result.num_edges(), 3);
        });
    }

    #[test]
    fn test_disjoint_union_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = create_triangle();
            let g2 = create_triangle();

            let result = graph_disjoint_union(py, &g1, &g2).unwrap();

            // Should have 6 nodes (3 + 3, no overlap)
            assert_eq!(result.num_nodes(), 6);
            // Should have 6 edges (3 + 3)
            assert_eq!(result.num_edges(), 6);
        });
    }

    #[test]
    fn test_directed_graphs() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut g1 = Graph::new_rust(true, false);
            g1.add_node_rust(Node { id: 1, label: None, attributes: None }).unwrap();
            g1.add_node_rust(Node { id: 2, label: None, attributes: None }).unwrap();
            g1.add_edge_rust(Edge { id: 0, source: 1, target: 2, weight: None, attributes: None }).unwrap();

            let mut g2 = Graph::new_rust(true, false);
            g2.add_node_rust(Node { id: 1, label: None, attributes: None }).unwrap();
            g2.add_node_rust(Node { id: 2, label: None, attributes: None }).unwrap();
            g2.add_edge_rust(Edge { id: 0, source: 2, target: 1, weight: None, attributes: None }).unwrap();

            // Union should have both directed edges
            let union = graph_union(py, &g1, &g2).unwrap();
            assert_eq!(union.num_edges(), 2);

            // Intersection: directed 1->2 and 2->1 are different, so no intersection
            let intersection = graph_intersection(py, &g1, &g2).unwrap();
            assert_eq!(intersection.num_edges(), 0);
        });
    }

    #[test]
    fn test_incompatible_directedness() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let g1 = Graph::new_rust(true, false);
            let g2 = Graph::new_rust(false, false);

            assert!(graph_union(py, &g1, &g2).is_err());
            assert!(graph_intersection(py, &g1, &g2).is_err());
            assert!(graph_difference(py, &g1, &g2).is_err());
            assert!(graph_symmetric_difference(py, &g1, &g2).is_err());
        });
    }
}
