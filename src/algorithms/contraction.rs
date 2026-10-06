//! Node contraction algorithms
//!
//! This module provides functions for contracting (merging) nodes in a graph.
//! Node contraction is useful for graph simplification, hierarchical analysis,
//! and creating quotient graphs.

use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::graph::{Edge, Graph, Node};

/// Contract two nodes into one, merging v into u.
///
/// All edges incident to v are redirected to u, and v is removed from the graph.
/// Self-loops may be created if there was an edge between u and v.
///
/// # Arguments
/// * `graph` - The graph to modify
/// * `u` - The node to keep (target of contraction)
/// * `v` - The node to merge into u (will be removed)
/// * `self_loops` - Whether to preserve self-loops created by contraction
///
/// # Returns
/// A new graph with v contracted into u.
#[pyfunction]
#[pyo3(signature = (graph, u, v, self_loops = true))]
pub fn contract_nodes(
    py: Python<'_>,
    graph: &Graph,
    u: u64,
    v: u64,
    self_loops: bool,
) -> PyResult<Graph> {
    // Validate nodes exist
    if !graph.has_node_id(u) {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "Node {} does not exist",
            u
        )));
    }
    if !graph.has_node_id(v) {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "Node {} does not exist",
            v
        )));
    }
    if u == v {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot contract a node with itself",
        ));
    }

    // Create a new graph with same properties
    let mut new_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Copy all nodes except v
    for &node_id in graph.get_node_ids().iter() {
        if node_id != v {
            let label = graph.label_from_id(node_id).ok().flatten();
            let node = Node {
                id: node_id,
                label,
                // u keeps its own attributes; v's are dropped along with v
                attributes: crate::graph::copy_attrs(py, &graph.get_nodes()[&node_id].attributes),
            };
            new_graph.add_node_rust(node)?;
        }
    }

    // Process edges
    let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    let edges_internal = graph.get_edges_internal();

    for edge in edges_internal {
        let mut source = edge.source;
        let mut target = edge.target;

        // Redirect edges from v to u
        if source == v {
            source = u;
        }
        if target == v {
            target = u;
        }

        // Skip self-loops if not wanted
        if source == target && !self_loops {
            continue;
        }

        // For undirected graphs, normalize edge direction for deduplication
        let edge_key = if !graph.is_directed() && source > target {
            (target, source)
        } else {
            (source, target)
        };

        // Avoid duplicate edges (simple graph behavior)
        if added_edges.contains(&edge_key) {
            continue;
        }
        added_edges.insert(edge_key);

        // Create new edge, keeping the ID of the edge it came from
        let new_edge = Edge {
            id: edge.id,
            source,
            target,
            weight: edge.weight,
            attributes: crate::graph::copy_attrs(py, &edge.attributes),
        };

        new_graph.add_edge_rust(new_edge)?;
    }

    Ok(new_graph)
}

/// Contract an edge by merging its endpoints.
///
/// Given an edge (u, v), merges v into u. This is equivalent to
/// `contract_nodes(u, v)` but takes an edge specification instead.
///
/// # Arguments
/// * `graph` - The graph to modify
/// * `u` - Source node of the edge to contract
/// * `v` - Target node of the edge to contract
/// * `self_loops` - Whether to preserve self-loops created by contraction
///
/// # Returns
/// A new graph with the edge contracted.
#[pyfunction]
#[pyo3(signature = (graph, u, v, self_loops = true))]
pub fn contracted_edge(
    py: Python<'_>,
    graph: &Graph,
    u: u64,
    v: u64,
    self_loops: bool,
) -> PyResult<Graph> {
    // Verify the edge exists
    let edges_internal = graph.get_edges_internal();
    let edge_exists = edges_internal.iter().any(|e| {
        (e.source == u && e.target == v)
            || (!graph.is_directed() && e.source == v && e.target == u)
    });

    if !edge_exists {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Edge ({}, {}) does not exist in the graph",
            u, v
        )));
    }

    // Delegate to contract_nodes
    contract_nodes(py, graph, u, v, self_loops)
}

/// Contract multiple nodes into a single node.
///
/// All nodes in the `nodes` list are merged into the first node in the list.
/// Edges between contracted nodes become self-loops (if self_loops=True).
///
/// # Arguments
/// * `graph` - The graph to modify
/// * `nodes` - List of node IDs to contract (first node is the target)
/// * `self_loops` - Whether to preserve self-loops created by contraction
///
/// # Returns
/// A new graph with all specified nodes contracted into one.
#[pyfunction]
#[pyo3(signature = (graph, nodes, self_loops = true))]
pub fn contract_nodes_multi(
    py: Python<'_>,
    graph: &Graph,
    nodes: Vec<u64>,
    self_loops: bool,
) -> PyResult<Graph> {
    if nodes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must specify at least one node to contract",
        ));
    }
    if nodes.len() == 1 {
        // Nothing to contract, return a copy
        return copy_graph(py, graph);
    }

    // Validate all nodes exist
    for &node_id in &nodes {
        if !graph.has_node_id(node_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Node {} does not exist",
                node_id
            )));
        }
    }

    // Contract iteratively: merge all nodes into the first one
    let target = nodes[0];
    let mut result = copy_graph(py, graph)?;

    for &node in nodes.iter().skip(1) {
        result = contract_nodes(py, &result, target, node, self_loops)?;
    }

    Ok(result)
}

/// Create a quotient graph from a partition of nodes.
///
/// Each block (set of nodes) in the partition becomes a single node in the
/// quotient graph. Edges between blocks in the original graph become edges
/// between the corresponding nodes in the quotient graph.
///
/// # Arguments
/// * `graph` - The graph to create quotient from
/// * `partition` - List of node ID lists, where each inner list is a block
///
/// # Returns
/// A new graph where each partition block is a single node.
#[pyfunction]
#[pyo3(signature = (graph, partition))]
pub fn quotient_graph(
    py: Python<'_>,
    graph: &Graph,
    partition: Vec<Vec<u64>>,
) -> PyResult<Graph> {
    // Validate partition covers all nodes exactly once
    let mut seen_nodes: FxHashSet<u64> = FxHashSet::default();
    for block in &partition {
        for &node in block {
            if !graph.has_node_id(node) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Node {} does not exist",
                    node
                )));
            }
            if seen_nodes.contains(&node) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node {} appears in multiple partition blocks",
                    node
                )));
            }
            seen_nodes.insert(node);
        }
    }

    // Check all nodes are covered
    for &node_id in graph.get_node_ids().iter() {
        if !seen_nodes.contains(&node_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} is not in any partition block",
                node_id
            )));
        }
    }

    // Create mapping from original nodes to block index
    let mut node_block_map: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for (block_idx, block) in partition.iter().enumerate() {
        for &node in block {
            node_block_map.insert(node, block_idx);
        }
    }

    // Create quotient graph
    let mut quotient = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Add one node per block (using block index + 1 as node ID for 1-indexing). Blocks are new
    // nodes, so instead of inheriting attributes each records the original node IDs it contains
    for (block_idx, block) in partition.iter().enumerate() {
        let attrs = pyo3::types::PyDict::new(py);
        let mut members = block.clone();
        members.sort_unstable();
        attrs.set_item("members", members)?;
        let node = Node {
            id: block_idx as u64 + 1,
            label: Some(format!("block_{}", block_idx)),
            attributes: Some(attrs.unbind()),
        };
        quotient.add_node_rust(node)?;
    }

    // Add edges between blocks
    let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    let edges_internal = graph.get_edges_internal();

    for edge in edges_internal {
        let source_block = *node_block_map.get(&edge.source).unwrap();
        let target_block = *node_block_map.get(&edge.target).unwrap();

        // Skip intra-block edges (they become nothing in quotient)
        if source_block == target_block {
            continue;
        }

        let source_node = source_block as u64 + 1;
        let target_node = target_block as u64 + 1;

        // Normalize for undirected graphs
        let edge_key = if !graph.is_directed() && source_node > target_node {
            (target_node, source_node)
        } else {
            (source_node, target_node)
        };

        // Avoid duplicate edges
        if added_edges.contains(&edge_key) {
            continue;
        }
        added_edges.insert(edge_key);

        // Create edge
        let new_edge = Edge {
            id: 0, // Auto-assigned
            source: source_node,
            target: target_node,
            weight: edge.weight,
            attributes: crate::graph::copy_attrs(py, &edge.attributes),
        };

        quotient.add_edge_rust(new_edge)?;
    }

    Ok(quotient)
}

/// Helper function to create a copy of a graph
fn copy_graph(py: Python<'_>, graph: &Graph) -> PyResult<Graph> {
    let mut new_graph = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Copy nodes
    for &node_id in graph.get_node_ids().iter() {
        let label = graph.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: crate::graph::copy_attrs(py, &graph.get_nodes()[&node_id].attributes),
        };
        new_graph.add_node_rust(node)?;
    }

    // Copy edges, keeping their IDs
    for edge in graph.get_edges_internal() {
        let new_edge = Edge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            weight: edge.weight,
            attributes: crate::graph::copy_attrs(py, &edge.attributes),
        };
        new_graph.add_edge_rust(new_edge)?;
    }

    Ok(new_graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new_rust(false, false);
        // Create a simple path: 1 - 2 - 3 - 4
        for i in 1u64..=4 {
            let node = Node {
                id: i,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }

        // Add edges
        let edges = [(1u64, 2u64), (2, 3), (3, 4)];
        for (s, t) in edges {
            let edge = Edge {
                id: 0,
                source: s,
                target: t,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }

        graph
    }

    #[test]
    fn test_contract_nodes_basic() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Contract nodes 2 and 3
            let result = contract_nodes(py, &graph, 2, 3, true).unwrap();

            // Should have 3 nodes: 1, 2, 4
            assert_eq!(result.num_nodes(), 3);
            assert!(result.has_node_id(1));
            assert!(result.has_node_id(2));
            assert!(!result.has_node_id(3)); // 3 is merged into 2
            assert!(result.has_node_id(4));

            // Should have edges 1-2 and 2-4, plus the self-loop 2-2 from the
            // contracted 2-3 edge (self_loops=true, the documented behavior)
            assert_eq!(result.num_edges(), 3);
            let self_loops = result
                .get_edges_internal()
                .iter()
                .filter(|e| e.source == e.target)
                .count();
            assert_eq!(self_loops, 1);
        });
    }

    #[test]
    fn test_contract_nodes_creates_self_loop() {
        Python::with_gil(|py| {
            let mut graph = Graph::new_rust(false, false);
            // Triangle: 1-2-3-1
            for i in 1u64..=3 {
                let node = Node {
                    id: i,
                    label: None,
                    attributes: None,
                };
                graph.add_node_rust(node).unwrap();
            }
            let edges = [(1u64, 2u64), (2, 3), (3, 1)];
            for (s, t) in edges {
                let edge = Edge {
                    id: 0,
                    source: s,
                    target: t,
                    weight: None,
                    attributes: None,
                };
                graph.add_edge_rust(edge).unwrap();
            }

            // Contract 1 and 2 (with self_loops=true)
            let result = contract_nodes(py, &graph, 1, 2, true).unwrap();

            // Should have 2 nodes: 1, 3
            assert_eq!(result.num_nodes(), 2);

            // Check for self-loop
            let edges = result.get_edges_internal();
            let self_loop_count = edges.iter().filter(|e| e.source == e.target).count();
            assert_eq!(self_loop_count, 1);
        });
    }

    #[test]
    fn test_contract_nodes_no_self_loops() {
        Python::with_gil(|py| {
            let mut graph = Graph::new_rust(false, false);
            // Triangle: 1-2-3-1
            for i in 1u64..=3 {
                let node = Node {
                    id: i,
                    label: None,
                    attributes: None,
                };
                graph.add_node_rust(node).unwrap();
            }
            let edges = [(1u64, 2u64), (2, 3), (3, 1)];
            for (s, t) in edges {
                let edge = Edge {
                    id: 0,
                    source: s,
                    target: t,
                    weight: None,
                    attributes: None,
                };
                graph.add_edge_rust(edge).unwrap();
            }

            // Contract 1 and 2 (with self_loops=false)
            let result = contract_nodes(py, &graph, 1, 2, false).unwrap();

            // No self-loops
            let edges = result.get_edges_internal();
            let self_loop_count = edges.iter().filter(|e| e.source == e.target).count();
            assert_eq!(self_loop_count, 0);
        });
    }

    #[test]
    fn test_contracted_edge() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Contract edge (2, 3)
            let result = contracted_edge(py, &graph, 2, 3, true).unwrap();

            // Same as contract_nodes(2, 3)
            assert_eq!(result.num_nodes(), 3);
            assert!(!result.has_node_id(3));
        });
    }

    #[test]
    fn test_contracted_edge_invalid() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Try to contract non-existent edge
            let result = contracted_edge(py, &graph, 1, 4, true);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_contract_nodes_multi() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Contract nodes 2, 3, 4 into 2
            let result = contract_nodes_multi(py, &graph, vec![2, 3, 4], true).unwrap();

            // Should have 2 nodes: 1 and 2
            assert_eq!(result.num_nodes(), 2);
            assert!(result.has_node_id(1));
            assert!(result.has_node_id(2));
        });
    }

    #[test]
    fn test_quotient_graph() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Partition: {1, 2} and {3, 4}
            let partition = vec![vec![1, 2], vec![3, 4]];
            let quotient = quotient_graph(py, &graph, partition).unwrap();

            // Should have 2 nodes (one per block)
            assert_eq!(quotient.num_nodes(), 2);

            // Should have 1 edge (between the two blocks, from edge 2-3)
            assert_eq!(quotient.num_edges(), 1);
        });
    }

    #[test]
    fn test_quotient_graph_invalid_partition() {
        Python::with_gil(|py| {
            let graph = create_test_graph();

            // Node appears twice
            let partition = vec![vec![1, 2], vec![2, 3, 4]];
            let result = quotient_graph(py, &graph, partition);
            assert!(result.is_err());

            // Missing node
            let partition = vec![vec![1, 2], vec![3]]; // missing 4
            let result = quotient_graph(py, &graph, partition);
            assert!(result.is_err());
        });
    }
}
