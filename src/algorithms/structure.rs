//! Graph structure and decomposition algorithms

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Compute the k-core of the graph.
///
/// The k-core is the maximal subgraph where all nodes have degree >= k.
///
/// # Arguments
/// * `graph` - The input graph
/// * `k` - The minimum degree
///
/// # Returns
/// A new graph containing only the k-core nodes and edges
#[pyfunction]
pub fn k_core(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Start with a copy of all nodes and edges that might be in the k-core
    let mut core_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Map old node IDs to new node IDs
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add all nodes first
    for node in graph.get_nodes().values() {
        let new_node = Node {
            id: 0, // Will be auto-assigned
            label: node.label.clone(),
            attributes: None, // Simplified - no attribute cloning
        };
        let new_id = core_graph.add_node_rust(new_node)?;
        node_map.insert(node.id, new_id);
    }

    // Add all edges
    for edge in graph.get_edges_internal() {
        if let (Some(&new_source), Some(&new_target)) =
            (node_map.get(&edge.source), node_map.get(&edge.target)) {
            let new_edge = Edge {
                id: 0, // Will be auto-assigned
                source: new_source,
                target: new_target,
                weight: edge.weight,
                attributes: None, // Simplified - no attribute cloning
            };
            core_graph.add_edge_rust(new_edge)?;
        }
    }

    // Iteratively remove nodes with degree < k
    let mut changed = true;
    while changed {
        changed = false;
        let mut nodes_to_remove = Vec::new();

        for &node_id in core_graph.get_nodes().keys() {
            if core_graph.degree_rust(node_id) < k {
                nodes_to_remove.push(node_id);
                changed = true;
            }
        }

        for node_id in nodes_to_remove {
            core_graph.remove_node_rust(node_id)?;
        }
    }

    Ok(core_graph)
}

/// Compute the k-shell of the graph.
///
/// The k-shell is the set of nodes with coreness exactly equal to k.
///
/// # Arguments
/// * `graph` - The input graph
/// * `k` - The coreness value
///
/// # Returns
/// A new graph containing only the k-shell nodes and their connecting edges
#[pyfunction]
pub fn k_shell(graph: &Graph, k: usize) -> PyResult<Graph> {
    let coreness = graph.get_coreness();

    let mut shell_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add nodes with coreness == k
    for (node_id, &core_value) in coreness.iter() {
        if core_value == k {
            if let Some(node) = graph.get_node(*node_id) {
                let new_node = Node {
                    id: 0,
                    label: node.label.clone(),
                    attributes: None,
                };
                let new_id = shell_graph.add_node_rust(new_node)?;
                node_map.insert(*node_id, new_id);
            }
        }
    }

    // Add edges between k-shell nodes
    for edge in graph.get_edges_internal() {
        if let (Some(&new_source), Some(&new_target)) =
            (node_map.get(&edge.source), node_map.get(&edge.target)) {
            let new_edge = Edge {
                id: 0,
                source: new_source,
                target: new_target,
                weight: edge.weight,
                attributes: None,
            };
            shell_graph.add_edge_rust(new_edge)?;
        }
    }

    Ok(shell_graph)
}

/// Compute the k-crust of the graph.
///
/// The k-crust is the set of nodes with coreness <= k.
///
/// # Arguments
/// * `graph` - The input graph
/// * `k` - The maximum coreness value
///
/// # Returns
/// A new graph containing only the k-crust nodes and their connecting edges
#[pyfunction]
pub fn k_crust(graph: &Graph, k: usize) -> PyResult<Graph> {
    let coreness = graph.get_coreness();

    let mut crust_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add nodes with coreness <= k
    for (node_id, &core_value) in coreness.iter() {
        if core_value <= k {
            if let Some(node) = graph.get_node(*node_id) {
                let new_node = Node {
                    id: 0,
                    label: node.label.clone(),
                    attributes: None,
                };
                let new_id = crust_graph.add_node_rust(new_node)?;
                node_map.insert(*node_id, new_id);
            }
        }
    }

    // Add edges between k-crust nodes
    for edge in graph.get_edges_internal() {
        if let (Some(&new_source), Some(&new_target)) =
            (node_map.get(&edge.source), node_map.get(&edge.target)) {
            let new_edge = Edge {
                id: 0,
                source: new_source,
                target: new_target,
                weight: edge.weight,
                attributes: None,
            };
            crust_graph.add_edge_rust(new_edge)?;
        }
    }

    Ok(crust_graph)
}

/// Count triangles containing a specific edge.
///
/// # Arguments
/// * `graph` - The input graph
/// * `source` - Source node of the edge
/// * `target` - Target node of the edge
///
/// # Returns
/// Number of triangles containing the edge (source, target)
fn count_triangles_containing_edge(graph: &Graph, source: u64, target: u64) -> usize {
    let source_neighbors: FxHashSet<u64> = graph
        .get_adjacency_list()
        .get(&source)
        .map(|v| v.iter().copied().collect())
        .unwrap_or_default();

    let target_neighbors: FxHashSet<u64> = graph
        .get_adjacency_list()
        .get(&target)
        .map(|v| v.iter().copied().collect())
        .unwrap_or_default();

    source_neighbors.intersection(&target_neighbors).count()
}

/// Compute the k-truss of the graph.
///
/// The k-truss is the maximal subgraph where every edge is part of at least (k-2) triangles.
///
/// # Arguments
/// * `graph` - The input graph
/// * `k` - The minimum number of triangles + 2
///
/// # Returns
/// A new graph containing only the k-truss edges and their endpoints
#[pyfunction]
pub fn k_truss(graph: &Graph, k: usize) -> PyResult<Graph> {
    if k < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be at least 3 for k-truss"
        ));
    }

    let min_triangles = k - 2;

    // Start with a copy of the graph
    let mut truss_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let mut node_map: FxHashMap<u64, u64> = FxHashMap::default();

    // Add all nodes
    for node in graph.get_nodes().values() {
        let new_node = Node {
            id: 0,
            label: node.label.clone(),
            attributes: None,
        };
        let new_id = truss_graph.add_node_rust(new_node)?;
        node_map.insert(node.id, new_id);
    }

    // Add all edges
    for edge in graph.get_edges_internal() {
        if let (Some(&new_source), Some(&new_target)) =
            (node_map.get(&edge.source), node_map.get(&edge.target)) {
            let new_edge = Edge {
                id: 0,
                source: new_source,
                target: new_target,
                weight: edge.weight,
                attributes: None,
            };
            truss_graph.add_edge_rust(new_edge)?;
        }
    }

    // Iteratively remove edges with insufficient triangle support
    let mut changed = true;
    while changed {
        changed = false;
        let mut edges_to_remove = Vec::new();

        for edge in truss_graph.get_edges_internal() {
            let triangle_count = count_triangles_containing_edge(
                &truss_graph,
                edge.source,
                edge.target
            );

            if triangle_count < min_triangles {
                edges_to_remove.push(edge.id);
                changed = true;
            }
        }

        for edge_id in edges_to_remove {
            truss_graph.remove_edge_rust(edge_id)?;
        }
    }

    // Remove isolated nodes (nodes with no edges)
    let mut nodes_to_remove = Vec::new();
    for &node_id in truss_graph.get_nodes().keys() {
        if truss_graph.degree_rust(node_id) == 0 {
            nodes_to_remove.push(node_id);
        }
    }

    for node_id in nodes_to_remove {
        truss_graph.remove_node_rust(node_id)?;
    }

    Ok(truss_graph)
}
