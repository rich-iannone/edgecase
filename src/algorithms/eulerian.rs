//! Eulerian path and circuit algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Check if the graph has an Eulerian circuit.
///
/// An Eulerian circuit is a path that visits every edge exactly once and returns
/// to the starting vertex. A graph has an Eulerian circuit if and only if:
/// - The graph is connected (or weakly connected for directed graphs)
/// - Every vertex has even degree (undirected) or equal in-degree and out-degree (directed)
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// True if the graph has an Eulerian circuit, False otherwise
#[pyfunction]
pub fn is_eulerian(graph: &Graph) -> PyResult<bool> {
    let node_count = graph.get_nodes().len();

    // Empty graph or single node is considered Eulerian
    if node_count == 0 || node_count == 1 {
        return Ok(true);
    }

    // Check if graph is connected (ignore isolated nodes)
    if !is_connected_for_euler(graph) {
        return Ok(false);
    }

    if graph.is_directed() {
        // For directed graphs: in-degree must equal out-degree for all vertices
        for node_id in graph.get_nodes().keys() {
            let in_deg = graph.get_in_degree(*node_id);
            let out_deg = graph.get_out_degree(*node_id);

            if in_deg != out_deg {
                return Ok(false);
            }
        }
    } else {
        // For undirected graphs: all vertices must have even degree
        for node_id in graph.get_nodes().keys() {
            let degree = graph.degree_rust(*node_id);

            if degree % 2 != 0 {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Check if the graph has an Eulerian path but not an Eulerian circuit.
///
/// An Eulerian path visits every edge exactly once. A graph is semi-Eulerian if:
/// - It has an Eulerian path but not an Eulerian circuit
/// - For undirected graphs: exactly 2 vertices have odd degree
/// - For directed graphs: exactly one vertex has out-degree - in-degree = 1,
///   and exactly one vertex has in-degree - out-degree = 1
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// True if the graph is semi-Eulerian, False otherwise
#[pyfunction]
pub fn is_semieulerian(graph: &Graph) -> PyResult<bool> {
    let node_count = graph.get_nodes().len();

    // Empty graph or single node is not semi-Eulerian
    if node_count == 0 || node_count == 1 {
        return Ok(false);
    }

    // Check if graph is connected
    if !is_connected_for_euler(graph) {
        return Ok(false);
    }

    if graph.is_directed() {
        // For directed graphs: exactly one vertex with out-degree - in-degree = 1
        // and exactly one vertex with in-degree - out-degree = 1
        let mut start_vertices = 0;
        let mut end_vertices = 0;

        for node_id in graph.get_nodes().keys() {
            let in_deg = graph.get_in_degree(*node_id) as i32;
            let out_deg = graph.get_out_degree(*node_id) as i32;
            let diff = out_deg - in_deg;

            if diff == 1 {
                start_vertices += 1;
            } else if diff == -1 {
                end_vertices += 1;
            } else if diff != 0 {
                return Ok(false);
            }
        }

        Ok(start_vertices == 1 && end_vertices == 1)
    } else {
        // For undirected graphs: exactly 2 vertices with odd degree
        let mut odd_degree_count = 0;

        for node_id in graph.get_nodes().keys() {
            let degree = graph.degree_rust(*node_id);

            if degree % 2 != 0 {
                odd_degree_count += 1;
            }
        }

        Ok(odd_degree_count == 2)
    }
}

/// Find an Eulerian circuit in the graph.
///
/// Uses Hierholzer's algorithm to find an Eulerian circuit if one exists.
/// Returns the circuit as a list of node IDs in order.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// List of node IDs forming an Eulerian circuit, or None if no circuit exists
#[pyfunction]
pub fn eulerian_circuit(graph: &Graph) -> PyResult<Option<Vec<u64>>> {
    // Check if Eulerian circuit exists
    if !is_eulerian(graph)? {
        return Ok(None);
    }

    let node_count = graph.get_nodes().len();
    if node_count == 0 {
        return Ok(Some(Vec::new()));
    }

    if node_count == 1 {
        let node_id = *graph.get_nodes().keys().next().unwrap();
        return Ok(Some(vec![node_id]));
    }

    // Use Hierholzer's algorithm
    let result = hierholzer_circuit(graph)?;
    Ok(Some(result))
}

/// Hierholzer's algorithm for finding Eulerian circuit
fn hierholzer_circuit(graph: &Graph) -> PyResult<Vec<u64>> {
    // Build adjacency list with edge tracking
    let mut adj: FxHashMap<u64, Vec<(u64, u64)>> = FxHashMap::default();
    let mut used_edges: FxHashSet<u64> = FxHashSet::default();

    // Build adjacency list
    for edge in graph.get_edges_internal() {
        adj.entry(edge.source)
            .or_insert_with(Vec::new)
            .push((edge.target, edge.id));

        if !graph.is_directed() {
            // For undirected graphs, add reverse edge
            adj.entry(edge.target)
                .or_insert_with(Vec::new)
                .push((edge.source, edge.id));
        }
    }

    // Start from any node with edges
    let start_node = *graph.get_nodes().keys().next().unwrap();

    let mut stack = vec![start_node];
    let mut circuit = Vec::new();

    while let Some(&current) = stack.last() {
        if let Some(neighbors) = adj.get_mut(&current) {
            // Find an unused edge from current
            let mut found_edge = None;
            while let Some((next, edge_id)) = neighbors.pop() {
                if !used_edges.contains(&edge_id) {
                    used_edges.insert(edge_id);
                    found_edge = Some(next);
                    break;
                }
            }

            if let Some(next) = found_edge {
                stack.push(next);
            } else {
                // No more edges from current, add to circuit
                circuit.push(stack.pop().unwrap());
            }
        } else {
            // No edges from this node, add to circuit
            circuit.push(stack.pop().unwrap());
        }
    }

    // Reverse to get correct order
    circuit.reverse();

    Ok(circuit)
}

/// Check if graph is connected for Eulerian purposes
/// (ignores isolated vertices)
fn is_connected_for_euler(graph: &Graph) -> bool {
    // Find a node with at least one edge
    let mut start_node = None;
    for node_id in graph.get_nodes().keys() {
        let degree = if graph.is_directed() {
            let in_deg = graph.get_in_degree(*node_id);
            let out_deg = graph.get_out_degree(*node_id);
            in_deg + out_deg
        } else {
            graph.degree_rust(*node_id)
        };

        if degree > 0 {
            start_node = Some(*node_id);
            break;
        }
    }

    let start = match start_node {
        Some(n) => n,
        None => return true, // No edges means connected for Euler purposes
    };

    // BFS to check connectivity
    let mut visited = FxHashSet::default();
    let mut queue = vec![start];
    visited.insert(start);

    while let Some(current) = queue.pop() {
        // Get outgoing neighbors
        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        // For directed graphs, also consider incoming edges (weakly connected)
        if graph.is_directed() {
            // Find nodes that point to current
            for node_id in graph.get_nodes().keys() {
                if visited.contains(node_id) {
                    continue;
                }

                if let Ok(neighbors) = graph.neighbors(*node_id) {
                    if neighbors.contains(&current) {
                        visited.insert(*node_id);
                        queue.push(*node_id);
                    }
                }
            }
        }
    }

    // Check if all non-isolated nodes are visited
    for node_id in graph.get_nodes().keys() {
        let degree = if graph.is_directed() {
            let in_deg = graph.get_in_degree(*node_id);
            let out_deg = graph.get_out_degree(*node_id);
            in_deg + out_deg
        } else {
            graph.degree_rust(*node_id)
        };

        if degree > 0 && !visited.contains(node_id) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_eulerian_cycle() {
        let mut graph = Graph::new_rust(false, false);

        // Create a cycle: 1-2-3-4-1 (all nodes have degree 2)
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");
        let n4 = graph.add_test_node("4");

        graph.add_test_edge(n1, n2, Some(1.0));
        graph.add_test_edge(n2, n3, Some(1.0));
        graph.add_test_edge(n3, n4, Some(1.0));
        graph.add_test_edge(n4, n1, Some(1.0));

        assert!(is_eulerian(&graph).unwrap());
        assert!(!is_semieulerian(&graph).unwrap());
    }

    #[test]
    fn test_is_semieulerian_path() {
        let mut graph = Graph::new_rust(false, false);

        // Create a path: 1-2-3 (endpoints have degree 1, middle has degree 2)
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");

        graph.add_test_edge(n1, n2, Some(1.0));
        graph.add_test_edge(n2, n3, Some(1.0));

        assert!(!is_eulerian(&graph).unwrap());
        assert!(is_semieulerian(&graph).unwrap());
    }

    #[test]
    fn test_not_eulerian() {
        let mut graph = Graph::new_rust(false, false);

        // Create a star: center with 3 leaves (center has degree 3, odd)
        let center = graph.add_test_node("center");
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");

        graph.add_test_edge(center, n1, Some(1.0));
        graph.add_test_edge(center, n2, Some(1.0));
        graph.add_test_edge(center, n3, Some(1.0));

        assert!(!is_eulerian(&graph).unwrap());
        assert!(!is_semieulerian(&graph).unwrap());
    }

    #[test]
    fn test_eulerian_circuit_simple() {
        let mut graph = Graph::new_rust(false, false);

        // Simple cycle
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");

        graph.add_test_edge(n1, n2, Some(1.0));
        graph.add_test_edge(n2, n3, Some(1.0));
        graph.add_test_edge(n3, n1, Some(1.0));

        let circuit = eulerian_circuit(&graph).unwrap();
        assert!(circuit.is_some());
        let path = circuit.unwrap();
        assert_eq!(path.len(), 4); // Should visit all edges and return to start
    }
}
