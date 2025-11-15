//! Structural analysis algorithms
//! Includes triangle counting, clustering coefficients, and transitivity

use crate::graph::Graph;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;

/// Count triangles in the graph
/// Returns a HashMap mapping node ID to the number of triangles that include that node
///
/// Algorithm: For each edge (u,v), count common neighbors.
/// Each common neighbor w forms a triangle (u,v,w).
/// For undirected graphs, each triangle is counted 3 times (once per node).
/// For directed graphs, counts closed triplets.
///
/// Time complexity: O(m * d_max) where d_max is maximum degree
/// Space complexity: O(n) for the result map
pub fn count_triangles(graph: &Graph) -> FxHashMap<u64, usize> {
    let mut triangle_count: FxHashMap<u64, usize> = FxHashMap::default();

    // Initialize all nodes to 0
    for &node_id in graph.get_node_ids().iter() {
        triangle_count.insert(node_id, 0);
    }

    if graph.is_directed() {
        // For directed graphs: count directed triangles
        count_triangles_directed(graph, &mut triangle_count);
    } else {
        // For undirected graphs: use node iterator algorithm
        count_triangles_undirected(graph, &mut triangle_count);
    }

    triangle_count
}

/// Count triangles in undirected graphs
/// Uses the NodeIterator algorithm: for each edge (u,v), count common neighbors
fn count_triangles_undirected(graph: &Graph, triangle_count: &mut FxHashMap<u64, usize>) {
    // Convert adjacency list to HashSet for fast lookup
    let adj_sets: FxHashMap<u64, HashSet<u64>> = graph.get_node_ids()
        .iter()
        .filter_map(|&node_id| {
            graph.neighbors(node_id).ok().map(|neighbors| {
                (node_id, neighbors.into_iter().collect::<HashSet<u64>>())
            })
        })
        .collect();

    // Track which triangles we've already counted
    let mut seen_triangles: HashSet<(u64, u64, u64)> = HashSet::new();

    // For each edge (u, v)
    for (_, u, v) in graph.edges() {
        // Find common neighbors of u and v (excluding u and v themselves)
        if let (Some(u_neighbors), Some(v_neighbors)) = (adj_sets.get(&u), adj_sets.get(&v)) {
            // Intersection of neighbors, excluding the edge endpoints
            let common: Vec<_> = u_neighbors
                .intersection(v_neighbors)
                .filter(|&&w| w != u && w != v)
                .collect();

            // Each common neighbor w forms a triangle (u, v, w)
            for &&w in &common {
                // Create canonical representation (sorted triple)
                let mut tri = vec![u, v, w];
                tri.sort_unstable();
                let triangle = (tri[0], tri[1], tri[2]);

                // Only count if we haven't seen this triangle before
                if seen_triangles.insert(triangle) {
                    *triangle_count.entry(u).or_insert(0) += 1;
                    *triangle_count.entry(v).or_insert(0) += 1;
                    *triangle_count.entry(w).or_insert(0) += 1;
                }
            }
        }
    }
}

/// Count triangles in directed graphs
/// Counts closed triplets where edges form a cycle
fn count_triangles_directed(graph: &Graph, triangle_count: &mut FxHashMap<u64, usize>) {
    let nodes = graph.get_node_ids();

    // For each node u
    for &u in nodes.iter() {
        if let Ok(u_neighbors) = graph.neighbors(u) {
            // For each neighbor v of u (u -> v)
            for &v in u_neighbors.iter() {
                if let Ok(v_neighbors) = graph.neighbors(v) {
                    // For each neighbor w of v (v -> w)
                    for &w in v_neighbors.iter() {
                        // Check if w connects back to u (w -> u), forming a cycle u -> v -> w -> u
                        if w != u {
                            if let Ok(w_neighbors) = graph.neighbors(w) {
                                if w_neighbors.contains(&u) {
                                    *triangle_count.entry(u).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// List all triangles in the graph
/// Returns a vector of triangles, where each triangle is a sorted triple of node IDs
///
/// Time complexity: O(m * d_max)
/// Space complexity: O(t) where t is the number of triangles
pub fn list_triangles(graph: &Graph) -> Vec<(u64, u64, u64)> {
    let mut triangles = Vec::new();

    if graph.is_directed() {
        let mut seen_triangles: HashSet<(u64, u64, u64)> = HashSet::new();
        list_triangles_directed(graph, &mut triangles, &mut seen_triangles);
    } else {
        list_triangles_undirected(graph, &mut triangles);
    }

    triangles
}

/// List triangles in undirected graphs
/// Optimized node-iterator with adjacency sets
/// Ensures u < v < w ordering to avoid duplicates
fn list_triangles_undirected(
    graph: &Graph,
    triangles: &mut Vec<(u64, u64, u64)>,
) {
    // Pre-build adjacency sets for O(1) lookups
    let adj_sets: FxHashMap<u64, FxHashSet<u64>> = graph.get_node_ids()
        .iter()
        .filter_map(|&node_id| {
            graph.neighbors(node_id).ok().map(|neighbors| {
                (node_id, neighbors.into_iter().collect())
            })
        })
        .collect();

    let nodes: Vec<u64> = graph.get_node_ids().iter().copied().collect();

    // For each node u, iterate over pairs of its neighbors
    for i in 0..nodes.len() {
        let u = nodes[i];

        if let Some(u_neighbors) = adj_sets.get(&u) {
            let neighbors: Vec<u64> = u_neighbors.iter().copied()
                .filter(|&v| v > u)  // Only consider neighbors with ID > u
                .collect();

            // Check all pairs of neighbors
            for j in 0..neighbors.len() {
                let v = neighbors[j];

                if let Some(v_neighbors) = adj_sets.get(&v) {
                    for k in (j + 1)..neighbors.len() {
                        let w = neighbors[k];

                        // Check if edge (v, w) exists - if so, triangle found
                        if v_neighbors.contains(&w) {
                            triangles.push((u, v, w));
                        }
                    }
                }
            }
        }
    }
}

/// List triangles in directed graphs
fn list_triangles_directed(
    graph: &Graph,
    triangles: &mut Vec<(u64, u64, u64)>,
    seen_triangles: &mut HashSet<(u64, u64, u64)>,
) {
    let nodes = graph.get_node_ids();

    for &u in nodes.iter() {
        if let Ok(u_neighbors) = graph.neighbors(u) {
            for &v in u_neighbors.iter() {
                if let Ok(v_neighbors) = graph.neighbors(v) {
                    for &w in v_neighbors.iter() {
                        if w != u {
                            // Check if w -> u exists (forming cycle u -> v -> w -> u)
                            if let Ok(w_neighbors) = graph.neighbors(w) {
                                if w_neighbors.contains(&u) {
                                    // Found triangle u -> v -> w -> u
                                    let mut tri = vec![u, v, w];
                                    tri.sort_unstable();
                                    let triangle = (tri[0], tri[1], tri[2]);

                                    if seen_triangles.insert(triangle) {
                                        triangles.push(triangle);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute global transitivity (clustering coefficient)
/// Transitivity = 3 * (number of triangles) / (number of connected triples)
///
/// For undirected graphs, a connected triple is a set of three nodes where
/// at least two edges connect them (a path of length 2).
///
/// Returns None if there are no connected triples
pub fn transitivity(graph: &Graph) -> Option<f64> {
    if graph.is_directed() {
        transitivity_directed(graph)
    } else {
        transitivity_undirected(graph)
    }
}

/// Compute transitivity for undirected graphs
fn transitivity_undirected(graph: &Graph) -> Option<f64> {
    let triangle_counts = count_triangles(graph);
    let total_triangles: usize = triangle_counts.values().sum::<usize>() / 3;

    // Count connected triples (paths of length 2)
    let mut connected_triples = 0;
    for &node_id in graph.get_node_ids().iter() {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            let degree = neighbors.len();
            if degree >= 2 {
                // Number of triples centered at this node is C(degree, 2)
                connected_triples += degree * (degree - 1) / 2;
            }
        }
    }

    if connected_triples == 0 {
        return None;
    }

    Some(3.0 * total_triangles as f64 / connected_triples as f64)
}

/// Compute transitivity for directed graphs
fn transitivity_directed(graph: &Graph) -> Option<f64> {
    let triangle_counts = count_triangles(graph);
    let total_triangles: usize = triangle_counts.values().sum();

    // For directed graphs, count directed paths of length 2
    let mut connected_triples = 0;
    for &node_id in graph.get_node_ids().iter() {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for &neighbor in neighbors.iter() {
                if let Ok(second_neighbors) = graph.neighbors(neighbor) {
                    connected_triples += second_neighbors.len();
                }
            }
        }
    }

    if connected_triples == 0 {
        return None;
    }

    Some(total_triangles as f64 / connected_triples as f64)
}

/// Compute local clustering coefficient for each node
/// For undirected graphs: C(v) = 2 * triangles(v) / (degree(v) * (degree(v) - 1))
/// For directed graphs: C(v) = triangles(v) / (out_degree(v) * (out_degree(v) - 1))
///
/// Returns None for nodes with degree < 2 (no triangles possible)
pub fn local_clustering_coefficient(graph: &Graph) -> FxHashMap<u64, Option<f64>> {
    let triangle_counts = count_triangles(graph);
    let mut clustering: FxHashMap<u64, Option<f64>> = FxHashMap::default();

    for &node_id in graph.get_node_ids().iter() {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            let degree = neighbors.len();

            if degree < 2 {
                clustering.insert(node_id, None);
                continue;
            }

            let triangles = triangle_counts.get(&node_id).copied().unwrap_or(0);

            if graph.is_directed() {
                // Directed: triangles / (degree * (degree - 1))
                let denominator = degree * (degree - 1);
                clustering.insert(node_id, Some(triangles as f64 / denominator as f64));
            } else {
                // Undirected: triangles / C(degree, 2)
                let denominator = degree * (degree - 1) / 2;
                clustering.insert(node_id, Some(triangles as f64 / denominator as f64));
            }
        } else {
            clustering.insert(node_id, None);
        }
    }

    clustering
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_count_triangles_empty() {
        let graph = Graph::new(false, false);
        let counts = count_triangles(&graph);
        assert_eq!(counts.len(), 0);
    }

    #[test]
    fn test_count_triangles_no_triangles() {
        let mut graph = Graph::new(false, false);
        graph.add_node(1, None, None).unwrap();
        graph.add_node(2, None, None).unwrap();
        graph.add_node(3, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();

        let counts = count_triangles(&graph);
        assert_eq!(counts.get(&1), Some(&0));
        assert_eq!(counts.get(&2), Some(&0));
        assert_eq!(counts.get(&3), Some(&0));
    }

    #[test]
    fn test_count_triangles_one_triangle() {
        let mut graph = Graph::new(false, false);
        graph.add_node(1, None, None).unwrap();
        graph.add_node(2, None, None).unwrap();
        graph.add_node(3, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 1, None, None).unwrap();

        let counts = count_triangles(&graph);
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&1));
        assert_eq!(counts.get(&3), Some(&1));
    }

    #[test]
    fn test_list_triangles() {
        let mut graph = Graph::new(false, false);
        graph.add_node(1, None, None).unwrap();
        graph.add_node(2, None, None).unwrap();
        graph.add_node(3, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 1, None, None).unwrap();

        let triangles = list_triangles(&graph);
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0], (1, 2, 3));
    }

    #[test]
    fn test_transitivity() {
        let mut graph = Graph::new(false, false);
        graph.add_node(1, None, None).unwrap();
        graph.add_node(2, None, None).unwrap();
        graph.add_node(3, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 1, None, None).unwrap();

        let trans = transitivity(&graph);
        assert!(trans.is_some());
        assert!((trans.unwrap() - 1.0).abs() < 1e-10); // Perfect triangle
    }

    #[test]
    fn test_local_clustering() {
        let mut graph = Graph::new(false, false);
        graph.add_node(1, None, None).unwrap();
        graph.add_node(2, None, None).unwrap();
        graph.add_node(3, None, None).unwrap();
        graph.add_edge(1, 2, None, None).unwrap();
        graph.add_edge(2, 3, None, None).unwrap();
        graph.add_edge(3, 1, None, None).unwrap();

        let clustering = local_clustering_coefficient(&graph);
        assert!(clustering.get(&1).unwrap().is_some());
        assert!((clustering.get(&1).unwrap().unwrap() - 1.0).abs() < 1e-10);
    }
}
