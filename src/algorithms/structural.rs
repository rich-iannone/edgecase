//! Structural analysis algorithms
//! Includes triangle counting, clustering coefficients, and transitivity

use crate::graph::Graph;
use rustc_hash::FxHashMap;
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
#[inline]
pub fn count_triangles(graph: &Graph) -> FxHashMap<u64, usize> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut triangle_count: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());

    // Initialize all nodes to 0
    for &node_id in node_ids.iter() {
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
/// OPTIMIZED: Uses FxHashSet, node ordering, and efficient memory access patterns
/// PARALLELIZED: Uses Rayon for parallel edge processing on very large graphs
#[inline]
fn count_triangles_undirected(graph: &Graph, triangle_count: &mut FxHashMap<u64, usize>) {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Convert adjacency lists to sorted vectors for binary search
    // Sequential construction is faster for most graphs (avoids parallel overhead)
    let mut adj_vecs: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for &node_id in node_ids.iter() {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            let mut sorted_neighbors = neighbors;
            sorted_neighbors.sort_unstable();
            adj_vecs.insert(node_id, sorted_neighbors);
        }
    }

    // Get edges and filter to u < v
    let edges: Vec<(u64, u64)> = graph.edges()
        .into_iter()
        .filter_map(|(_, u, v)| if u < v { Some((u, v)) } else { None })
        .collect();

    // Parallelize triangle counting only for very large graphs
    // Threshold chosen to avoid Rayon overhead on medium graphs
    let local_counts: FxHashMap<u64, usize> = if edges.len() >= 50000 {
        edges.par_iter()
            .fold(
                || FxHashMap::with_capacity_and_hasher(n / 10, Default::default()),
                |mut local_map, &(u, v)| {
                    // Find common neighbors using merge-like algorithm
                    if let (Some(u_neighbors), Some(v_neighbors)) = (adj_vecs.get(&u), adj_vecs.get(&v)) {
                        let mut i = 0;
                        let mut j = 0;

                        while i < u_neighbors.len() && j < v_neighbors.len() {
                            let un = u_neighbors[i];
                            let vn = v_neighbors[j];

                            if un == vn && un > v {
                                // Found a common neighbor w > v, forming triangle (u, v, w)
                                *local_map.entry(u).or_insert(0) += 1;
                                *local_map.entry(v).or_insert(0) += 1;
                                *local_map.entry(un).or_insert(0) += 1;
                                i += 1;
                                j += 1;
                            } else if un < vn {
                                i += 1;
                            } else {
                                j += 1;
                            }
                        }
                    }
                    local_map
                },
            )
            .reduce(
                || FxHashMap::with_capacity_and_hasher(n / 10, Default::default()),
                |mut acc, local_map| {
                    for (node, count) in local_map {
                        *acc.entry(node).or_insert(0) += count;
                    }
                    acc
                },
            )
    } else {
        // Sequential for small graphs
        let mut local_map = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (u, v) in edges {
            if let (Some(u_neighbors), Some(v_neighbors)) = (adj_vecs.get(&u), adj_vecs.get(&v)) {
                let mut i = 0;
                let mut j = 0;

                while i < u_neighbors.len() && j < v_neighbors.len() {
                    let un = u_neighbors[i];
                    let vn = v_neighbors[j];

                    if un == vn && un > v {
                        *local_map.entry(u).or_insert(0) += 1;
                        *local_map.entry(v).or_insert(0) += 1;
                        *local_map.entry(un).or_insert(0) += 1;
                        i += 1;
                        j += 1;
                    } else if un < vn {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
            }
        }
        local_map
    };

    // Merge local counts into the result
    for (node, count) in local_counts {
        *triangle_count.entry(node).or_insert(0) += count;
    }
}

/// Count triangles in directed graphs
/// Counts closed triplets where edges form a cycle
#[inline]
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
/// Compact forward algorithm with sorted adjacency lists
/// PARALLELIZED: Uses Rayon for parallel node processing on large graphs
fn list_triangles_undirected(
    graph: &Graph,
    triangles: &mut Vec<(u64, u64, u64)>,
) {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Build sorted adjacency lists - can be done in parallel
    let adj_lists: FxHashMap<u64, Vec<u64>> = node_ids.par_iter()
        .filter_map(|&node_id| {
            graph.neighbors(node_id).ok().and_then(|neighbors| {
                let mut sorted_neighbors: Vec<u64> = neighbors.into_iter()
                    .filter(|&n| n > node_id)  // Only keep neighbors with higher IDs
                    .collect();
                sorted_neighbors.sort_unstable();
                if !sorted_neighbors.is_empty() {
                    Some((node_id, sorted_neighbors))
                } else {
                    None
                }
            })
        })
        .collect();

    // For very large graphs, parallelize triangle enumeration per node
    // High threshold to avoid Rayon overhead
    let all_triangles: Vec<(u64, u64, u64)> = if n >= 10000 {
        node_ids.par_iter()
            .filter_map(|&u| adj_lists.get(&u).map(|u_neighbors| (u, u_neighbors)))
            .flat_map(|(u, u_neighbors)| {
                let mut local_triangles = Vec::new();
                // For each pair of neighbors (v, w) where v < w
                for (i, &v) in u_neighbors.iter().enumerate() {
                    if let Some(v_neighbors) = adj_lists.get(&v) {
                        // Only check w's that are after v in u's neighbor list
                        for &w in &u_neighbors[i+1..] {
                            // Binary search for w in v's sorted neighbor list
                            if v_neighbors.binary_search(&w).is_ok() {
                                local_triangles.push((u, v, w));
                            }
                        }
                    }
                }
                local_triangles
            })
            .collect()
    } else {
        // Sequential for small graphs
        let mut local_triangles = Vec::new();
        for &u in node_ids.iter() {
            if let Some(u_neighbors) = adj_lists.get(&u) {
                // For each pair of neighbors (v, w) where v < w
                for (i, &v) in u_neighbors.iter().enumerate() {
                    if let Some(v_neighbors) = adj_lists.get(&v) {
                        // Only check w's that are after v in u's neighbor list
                        for &w in &u_neighbors[i+1..] {
                            // Binary search for w in v's sorted neighbor list
                            if v_neighbors.binary_search(&w).is_ok() {
                                local_triangles.push((u, v, w));
                            }
                        }
                    }
                }
            }
        }
        local_triangles
    };

    triangles.extend(all_triangles);
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
                                    let mut tri = [u, v, w];
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
/// Optimized: uses direct adjacency list access and SIMD-accelerated intersection
fn transitivity_undirected(graph: &Graph) -> Option<f64> {
    use crate::algorithms::simd_intersection::count_intersection_sorted_above;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Get direct reference to adjacency list (zero-copy)
    let adj_list = graph.get_adjacency_list();

    // Build sorted adjacency lists + count triples in a single pass
    let mut sorted_adj: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut connected_triples: usize = 0;

    for &node_id in node_ids.iter() {
        if let Some(neighbors) = adj_list.get(&node_id) {
            let degree = neighbors.len();

            // Count connected triples for this node
            if degree >= 2 {
                connected_triples += degree * (degree - 1) / 2;
            }

            // Clone and sort neighbors only once
            let mut sorted_neighbors = neighbors.clone();
            sorted_neighbors.sort_unstable();
            sorted_adj.insert(node_id, sorted_neighbors);
        }
    }

    if connected_triples == 0 {
        return None;
    }

    // Count triangles using SIMD-accelerated sorted intersection
    let mut total_triangles: usize = 0;

    // Iterate over edges
    for (_, u, v) in graph.edges() {
        // Normalize edge direction for undirected graphs (always process as min < max)
        let (u, v) = if u < v { (u, v) } else { (v, u) };

        // Use SIMD-accelerated intersection for efficient counting
        if let (Some(u_neighbors), Some(v_neighbors)) = (sorted_adj.get(&u), sorted_adj.get(&v)) {
            // Count common neighbors > v
            total_triangles += count_intersection_sorted_above(u_neighbors, v_neighbors, v);
        }
    }

    Some(3.0 * total_triangles as f64 / connected_triples as f64)
}

/// Compute transitivity for directed graphs
fn transitivity_directed(graph: &Graph) -> Option<f64> {
    use rayon::prelude::*;

    let triangle_counts = count_triangles(graph);
    let total_triangles: usize = triangle_counts.values().sum();

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // For directed graphs, count directed paths of length 2 in parallel
    let connected_triples: usize = if n >= 1000 {
        // Parallel for large graphs
        node_ids.par_iter()
            .map(|&node_id| {
                if let Ok(neighbors) = graph.neighbors(node_id) {
                    neighbors.iter()
                        .filter_map(|&neighbor| graph.neighbors(neighbor).ok())
                        .map(|second_neighbors| second_neighbors.len())
                        .sum()
                } else {
                    0
                }
            })
            .sum()
    } else {
        // Sequential for small graphs
        node_ids.iter()
            .map(|&node_id| {
                if let Ok(neighbors) = graph.neighbors(node_id) {
                    neighbors.iter()
                        .filter_map(|&neighbor| graph.neighbors(neighbor).ok())
                        .map(|second_neighbors| second_neighbors.len())
                        .sum()
                } else {
                    0
                }
            })
            .sum()
    };

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
/// PARALLELIZED: Uses Rayon for parallel coefficient computation on large graphs
#[inline]
pub fn local_clustering_coefficient(graph: &Graph) -> FxHashMap<u64, Option<f64>> {
    use rayon::prelude::*;

    let triangle_counts = count_triangles(graph);
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let is_directed = graph.is_directed();

    // Parallelize for very large graphs
    // High threshold to avoid Rayon overhead on medium graphs
    if n >= 10000 {
        node_ids.par_iter()
            .map(|&node_id| {
                let coeff = if let Ok(neighbors) = graph.neighbors(node_id) {
                    let degree = neighbors.len();

                    if degree < 2 {
                        None
                    } else {
                        let triangles = triangle_counts.get(&node_id).copied().unwrap_or(0);
                        let triangles_f64 = triangles as f64;

                        if is_directed {
                            // Directed: triangles / (degree * (degree - 1))
                            let denominator = (degree * (degree - 1)) as f64;
                            let inv_denominator = 1.0 / denominator;
                            Some(triangles_f64 * inv_denominator)
                        } else {
                            // Undirected: triangles / C(degree, 2) = 2 * triangles / (degree * (degree - 1))
                            let denominator = (degree * (degree - 1)) as f64;
                            let inv_denominator = 2.0 / denominator;
                            Some(triangles_f64 * inv_denominator)
                        }
                    }
                } else {
                    None
                };

                (node_id, coeff)
            })
            .collect()
    } else {
        // Sequential for small graphs
        let mut clustering: FxHashMap<u64, Option<f64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for &node_id in node_ids.iter() {
            if let Ok(neighbors) = graph.neighbors(node_id) {
                let degree = neighbors.len();

                if degree < 2 {
                    clustering.insert(node_id, None);
                    continue;
                }

                let triangles = triangle_counts.get(&node_id).copied().unwrap_or(0);
                let triangles_f64 = triangles as f64;

                if is_directed {
                    // Directed: triangles / (degree * (degree - 1))
                    let denominator = (degree * (degree - 1)) as f64;
                    let inv_denominator = 1.0 / denominator;
                    clustering.insert(node_id, Some(triangles_f64 * inv_denominator));
                } else {
                    // Undirected: triangles / C(degree, 2) = 2 * triangles / (degree * (degree - 1))
                    let denominator = (degree * (degree - 1)) as f64;
                    let inv_denominator = 2.0 / denominator;
                    clustering.insert(node_id, Some(triangles_f64 * inv_denominator));
                }
            } else {
                clustering.insert(node_id, None);
            }
        }
        clustering
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_count_triangles_empty() {
        let graph = Graph::new_rust(false, false);
        let counts = count_triangles(&graph);
        assert_eq!(counts.len(), 0);
    }

    #[test]
    fn test_count_triangles_no_triangles() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node_with_id(1);
        graph.add_test_node_with_id(2);
        graph.add_test_node_with_id(3);
        graph.add_test_edge(1, 2, None);

        let counts = count_triangles(&graph);
        assert_eq!(counts.get(&1), Some(&0));
        assert_eq!(counts.get(&2), Some(&0));
        assert_eq!(counts.get(&3), Some(&0));
    }

    #[test]
    fn test_count_triangles_one_triangle() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node_with_id(1);
        graph.add_test_node_with_id(2);
        graph.add_test_node_with_id(3);
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 1, None);

        let counts = count_triangles(&graph);
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&1));
        assert_eq!(counts.get(&3), Some(&1));
    }

    #[test]
    fn test_list_triangles() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node_with_id(1);
        graph.add_test_node_with_id(2);
        graph.add_test_node_with_id(3);
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 1, None);

        let triangles = list_triangles(&graph);
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0], (1, 2, 3));
    }

    #[test]
    fn test_transitivity() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node_with_id(1);
        graph.add_test_node_with_id(2);
        graph.add_test_node_with_id(3);
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 1, None);

        let trans = transitivity(&graph);
        assert!(trans.is_some());
        assert!((trans.unwrap() - 1.0).abs() < 1e-10); // Perfect triangle
    }

    #[test]
    fn test_local_clustering() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node_with_id(1);
        graph.add_test_node_with_id(2);
        graph.add_test_node_with_id(3);
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 1, None);

        let clustering = local_clustering_coefficient(&graph);
        assert!(clustering.get(&1).unwrap().is_some());
        assert!((clustering.get(&1).unwrap().unwrap() - 1.0).abs() < 1e-10);
    }
}
