//! Line graph construction
//!
//! The line graph L(G) of a graph G transforms edges into nodes.
//! Each edge in the original graph becomes a node in the line graph,
//! and two nodes in the line graph are connected if their corresponding
//! edges in the original graph share a common endpoint.
//!
//! This module provides:
//!
//! - `line_graph`: Construct the line graph of a given graph

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};

/// Construct the line graph L(G) of a graph G.
///
/// In the line graph:
/// - Each edge in G becomes a node in L(G)
/// - Two nodes in L(G) are connected if their corresponding edges in G share an endpoint
///
/// # Arguments
/// * `graph` - The input graph
/// * `create_using_directed` - If Some(true), create directed line graph; if Some(false), undirected;
///                             if None, match the directedness of input graph
///
/// # Returns
/// A new Graph representing the line graph
///
/// # Notes
/// - For undirected graphs: two edges are adjacent if they share a vertex
/// - For directed graphs: edge (u,v) connects to edge (v,w) if v is shared
/// - Node attributes in L(G) contain the original edge endpoints as "source" and "target"
/// - Self-loops in G create nodes in L(G) that connect to all edges incident to that vertex
#[pyfunction]
#[pyo3(signature = (graph, create_using_directed=None))]
pub fn line_graph(graph: &Graph, create_using_directed: Option<bool>) -> PyResult<Graph> {
    // Determine directedness of result graph
    let directed = create_using_directed.unwrap_or_else(|| graph.is_directed());

    // Create new graph for the line graph
    let mut lg = Graph::new_rust(directed, graph.is_weighted());

    // Get all edges from the original graph
    let edges_internal = graph.get_edges_internal();

    if edges_internal.is_empty() {
        return Ok(lg);
    }

    // Map from original edge IDs to line graph node IDs
    let mut edge_to_node: FxHashMap<u64, u64> = FxHashMap::default();

    // Create a node in L(G) for each edge in G
    // First, detect if there are multi-edges (multiple edges between same endpoints)
    // If so, we need to include edge ID in label to make them unique
    let mut endpoint_counts: FxHashMap<(u64, u64), u32> = FxHashMap::default();
    for edge in edges_internal {
        let key = if edge.source <= edge.target {
            (edge.source, edge.target)
        } else {
            (edge.target, edge.source)
        };
        *endpoint_counts.entry(key).or_insert(0) += 1;
    }
    let has_multi_edges = endpoint_counts.values().any(|&count| count > 1);

    Python::with_gil(|py| -> PyResult<()> {
        for edge in edges_internal {
            // Create attributes dict with original edge endpoints
            let attrs = PyDict::new(py);
            attrs.set_item("original_edge_id", edge.id)?;
            attrs.set_item("original_source", edge.source)?;
            attrs.set_item("original_target", edge.target)?;

            // Get original edge weight if available
            if let Some(weight) = edge.weight {
                attrs.set_item("original_weight", weight)?;
            }

            // Create label from original edge
            // Include edge ID if there are multi-edges to ensure uniqueness
            let label = if has_multi_edges {
                format!("{}-{}-e{}", edge.source, edge.target, edge.id)
            } else {
                format!("{}-{}", edge.source, edge.target)
            };

            // Create node struct directly (all fields are public)
            let node = Node {
                id: 0, // Will be auto-assigned
                label: Some(label),
                attributes: Some(attrs.into()),
            };

            let node_id = lg.add_node_rust(node)?;
            edge_to_node.insert(edge.id, node_id);
        }
        Ok(())
    })?;

    // Build adjacency structure: for each vertex, track which edges are incident to it
    // For undirected: an edge (u,v) is incident to both u and v
    // For directed: an edge (u,v) has u as source and v as target
    let mut vertex_to_edges: FxHashMap<u64, Vec<u64>> = FxHashMap::default();

    if graph.is_directed() {
        // For directed graphs, we need separate tracking for outgoing/incoming
        // Edge (u,v) -> (v,w): edges leaving v connect to edge (u,v)'s target
        let mut vertex_outgoing: FxHashMap<u64, Vec<u64>> = FxHashMap::default();

        for edge in edges_internal {
            vertex_outgoing
                .entry(edge.source)
                .or_default()
                .push(edge.id);
        }

        // For directed line graph: connect edge1 to edge2 if edge1.target == edge2.source
        // i.e., if edge1 = (u,v) and edge2 = (v,w), then edge1 -> edge2 in L(G)
        let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

        for edge in edges_internal {
            let lg_node1 = edge_to_node[&edge.id];

            // Find all edges that start from where this edge ends
            if let Some(outgoing_from_target) = vertex_outgoing.get(&edge.target) {
                for other_edge_id in outgoing_from_target {
                    if edge.id == *other_edge_id {
                        continue; // Skip self (though this shouldn't happen for simple graphs)
                    }

                    let lg_node2 = edge_to_node[other_edge_id];

                    // Add directed edge in line graph
                    let edge_pair = (lg_node1, lg_node2);
                    if !added_edges.contains(&edge_pair) {
                        // Create edge struct directly
                        let new_edge = Edge {
                            id: 0,
                            source: lg_node1,
                            target: lg_node2,
                            weight: None,
                            attributes: None,
                        };
                        lg.add_edge_rust(new_edge)?;
                        added_edges.insert(edge_pair);
                    }
                }
            }
        }
    } else {
        // For undirected graphs: edges are adjacent if they share any endpoint
        for edge in edges_internal {
            vertex_to_edges
                .entry(edge.source)
                .or_default()
                .push(edge.id);
            if edge.source != edge.target {
                // Don't add twice for self-loops
                vertex_to_edges
                    .entry(edge.target)
                    .or_default()
                    .push(edge.id);
            }
        }

        // Connect nodes in L(G) if their corresponding edges share a vertex
        let mut added_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

        for edges_at_vertex in vertex_to_edges.values() {
            // All edges incident to this vertex are pairwise adjacent in L(G)
            let n = edges_at_vertex.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let edge1 = edges_at_vertex[i];
                    let edge2 = edges_at_vertex[j];

                    let lg_node1 = edge_to_node[&edge1];
                    let lg_node2 = edge_to_node[&edge2];

                    // Ensure consistent ordering to avoid duplicates
                    let (n1, n2) = if lg_node1 < lg_node2 {
                        (lg_node1, lg_node2)
                    } else {
                        (lg_node2, lg_node1)
                    };

                    if !added_edges.contains(&(n1, n2)) {
                        // Create edge struct directly
                        let new_edge = Edge {
                            id: 0,
                            source: n1,
                            target: n2,
                            weight: None,
                            attributes: None,
                        };
                        lg.add_edge_rust(new_edge)?;
                        added_edges.insert((n1, n2));
                    }
                }
            }
        }
    }

    Ok(lg)
}

/// Get the inverse line graph mapping: returns a dict mapping line graph node IDs
/// to their original edge information (source, target, edge_id)
#[pyfunction]
pub fn line_graph_node_mapping(line_graph: &Graph) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let result = PyDict::new(py);

        for node_id in line_graph.get_node_ids() {
            if let Some(node) = line_graph.get_node(node_id) {
                if let Some(ref attrs) = node.attributes {
                    let attrs_dict = attrs.bind(py);

                    let info = PyDict::new(py);

                    if let Ok(Some(orig_edge_id)) = attrs_dict.get_item("original_edge_id") {
                        info.set_item("edge_id", orig_edge_id)?;
                    }
                    if let Ok(Some(orig_source)) = attrs_dict.get_item("original_source") {
                        info.set_item("source", orig_source)?;
                    }
                    if let Ok(Some(orig_target)) = attrs_dict.get_item("original_target") {
                        info.set_item("target", orig_target)?;
                    }
                    if let Ok(Some(orig_weight)) = attrs_dict.get_item("original_weight") {
                        info.set_item("weight", orig_weight)?;
                    }

                    result.set_item(node_id, info)?;
                }
            }
        }

        Ok(result.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_graph_empty() {
        pyo3::prepare_freethreaded_python();

        let g = Graph::new_rust(false, false);
        let lg = line_graph(&g, None).unwrap();
        assert_eq!(lg.num_nodes(), 0);
        assert_eq!(lg.num_edges(), 0);
    }

    #[test]
    fn test_line_graph_single_edge() {
        pyo3::prepare_freethreaded_python();

        let mut g = Graph::new_rust(false, false);

        // Create nodes directly using struct initialization
        let node1 = Node {
            id: 0,
            label: Some("A".to_string()),
            attributes: None,
        };
        let node2 = Node {
            id: 0,
            label: Some("B".to_string()),
            attributes: None,
        };
        let n1_id = g.add_node_rust(node1).unwrap();
        let n2_id = g.add_node_rust(node2).unwrap();

        // Create edge directly using struct initialization
        let edge = Edge {
            id: 0,
            source: n1_id,
            target: n2_id,
            weight: None,
            attributes: None,
        };
        g.add_edge_rust(edge).unwrap();

        let lg = line_graph(&g, None).unwrap();

        // Line graph of single edge has 1 node and 0 edges
        assert_eq!(lg.num_nodes(), 1);
        assert_eq!(lg.num_edges(), 0);
    }

    #[test]
    fn test_line_graph_path() {
        pyo3::prepare_freethreaded_python();

        // Path: A -- B -- C (2 edges)
        let mut g = Graph::new_rust(false, false);

        let n1 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("A".to_string()),
                attributes: None,
            })
            .unwrap();
        let n2 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("B".to_string()),
                attributes: None,
            })
            .unwrap();
        let n3 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("C".to_string()),
                attributes: None,
            })
            .unwrap();

        g.add_edge_rust(Edge {
            id: 0,
            source: n1,
            target: n2,
            weight: None,
            attributes: None,
        })
        .unwrap(); // A-B
        g.add_edge_rust(Edge {
            id: 0,
            source: n2,
            target: n3,
            weight: None,
            attributes: None,
        })
        .unwrap(); // B-C

        let lg = line_graph(&g, None).unwrap();

        // Line graph of path with 2 edges:
        // - 2 nodes (one for each edge)
        // - 1 edge (connecting them because they share vertex B)
        assert_eq!(lg.num_nodes(), 2);
        assert_eq!(lg.num_edges(), 1);
    }

    #[test]
    fn test_line_graph_triangle() {
        pyo3::prepare_freethreaded_python();

        // Triangle: A -- B -- C -- A (3 edges)
        let mut g = Graph::new_rust(false, false);

        let n1 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("A".to_string()),
                attributes: None,
            })
            .unwrap();
        let n2 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("B".to_string()),
                attributes: None,
            })
            .unwrap();
        let n3 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("C".to_string()),
                attributes: None,
            })
            .unwrap();

        g.add_edge_rust(Edge {
            id: 0,
            source: n1,
            target: n2,
            weight: None,
            attributes: None,
        })
        .unwrap(); // A-B
        g.add_edge_rust(Edge {
            id: 0,
            source: n2,
            target: n3,
            weight: None,
            attributes: None,
        })
        .unwrap(); // B-C
        g.add_edge_rust(Edge {
            id: 0,
            source: n3,
            target: n1,
            weight: None,
            attributes: None,
        })
        .unwrap(); // C-A

        let lg = line_graph(&g, None).unwrap();

        // Line graph of triangle:
        // - 3 nodes (one for each edge)
        // - 3 edges (each pair of edges shares a vertex)
        // The line graph of K3 is also K3!
        assert_eq!(lg.num_nodes(), 3);
        assert_eq!(lg.num_edges(), 3);
    }

    #[test]
    fn test_line_graph_star() {
        pyo3::prepare_freethreaded_python();

        // Star: center connected to 4 leaves (4 edges)
        let mut g = Graph::new_rust(false, false);

        let center = g
            .add_node_rust(Node {
                id: 0,
                label: Some("center".to_string()),
                attributes: None,
            })
            .unwrap();

        for i in 1..=4 {
            let leaf = g
                .add_node_rust(Node {
                    id: 0,
                    label: Some(format!("leaf{}", i)),
                    attributes: None,
                })
                .unwrap();
            g.add_edge_rust(Edge {
                id: 0,
                source: center,
                target: leaf,
                weight: None,
                attributes: None,
            })
            .unwrap();
        }

        let lg = line_graph(&g, None).unwrap();

        // Line graph of star with 4 edges:
        // - 4 nodes (one for each edge)
        // - 6 edges (K4: each pair of edges shares the center vertex)
        // The line graph of a star K_{1,n} is K_n
        assert_eq!(lg.num_nodes(), 4);
        assert_eq!(lg.num_edges(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_line_graph_directed() {
        pyo3::prepare_freethreaded_python();

        // Directed chain: A -> B -> C (2 edges)
        let mut g = Graph::new_rust(true, false);

        let n1 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("A".to_string()),
                attributes: None,
            })
            .unwrap();
        let n2 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("B".to_string()),
                attributes: None,
            })
            .unwrap();
        let n3 = g
            .add_node_rust(Node {
                id: 0,
                label: Some("C".to_string()),
                attributes: None,
            })
            .unwrap();

        g.add_edge_rust(Edge {
            id: 0,
            source: n1,
            target: n2,
            weight: None,
            attributes: None,
        })
        .unwrap(); // A->B
        g.add_edge_rust(Edge {
            id: 0,
            source: n2,
            target: n3,
            weight: None,
            attributes: None,
        })
        .unwrap(); // B->C

        let lg = line_graph(&g, None).unwrap();

        // Line graph of directed chain:
        // - 2 nodes (one for each edge)
        // - 1 edge: (A->B) connects to (B->C) because B is shared
        assert_eq!(lg.num_nodes(), 2);
        assert_eq!(lg.num_edges(), 1);
        assert!(lg.is_directed());
    }
}
