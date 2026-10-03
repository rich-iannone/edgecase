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

/// Build sorted, deduplicated neighbor lists with self-loops removed
///
/// Triangles, transitivity, and clustering are defined on the simple graph, so parallel
/// edges (including, for undirected graphs, an A-B edge stored alongside B-A) must count
/// once and a self-loop must not make a node its own neighbor. Every node gets an entry,
/// so isolated nodes map to an empty list.
fn simple_adjacency(graph: &Graph) -> FxHashMap<u64, Vec<u64>> {
    let node_ids = graph.get_node_ids();
    let adj_list = graph.get_adjacency_list();
    let mut simple: FxHashMap<u64, Vec<u64>> =
        FxHashMap::with_capacity_and_hasher(node_ids.len(), Default::default());

    for &node_id in node_ids.iter() {
        let mut neighbors: Vec<u64> = adj_list
            .get(&node_id)
            .map(|nbrs| nbrs.iter().copied().filter(|&nb| nb != node_id).collect())
            .unwrap_or_default();
        neighbors.sort_unstable();
        neighbors.dedup();
        simple.insert(node_id, neighbors);
    }

    simple
}

/// Distinct undirected edges as (u, v) pairs with u < v, taken from a simple adjacency
fn simple_undirected_edges(adj: &FxHashMap<u64, Vec<u64>>) -> Vec<(u64, u64)> {
    adj.iter()
        .flat_map(|(&u, nbrs)| nbrs.iter().filter(move |&&v| v > u).map(move |&v| (u, v)))
        .collect()
}

/// Count triangles in undirected graphs
/// Uses the NodeIterator algorithm: for each edge (u,v), count common neighbors
/// OPTIMIZED: Uses sorted adjacency vectors and a merge-style intersection
/// PARALLELIZED: Uses Rayon for parallel edge processing on very large graphs
#[inline]
fn count_triangles_undirected(graph: &Graph, triangle_count: &mut FxHashMap<u64, usize>) {
    use rayon::prelude::*;

    let n = graph.get_node_ids().len();
    let adj_vecs = simple_adjacency(graph);

    // Each distinct edge once, normalized to u < v regardless of its stored direction
    let edges = simple_undirected_edges(&adj_vecs);

    // Count the common neighbors w > v of edge (u, v); each triangle is found exactly once
    let count_edge = |local_map: &mut FxHashMap<u64, usize>, u: u64, v: u64| {
        if let (Some(u_neighbors), Some(v_neighbors)) = (adj_vecs.get(&u), adj_vecs.get(&v)) {
            let mut i = 0;
            let mut j = 0;

            while i < u_neighbors.len() && j < v_neighbors.len() {
                let un = u_neighbors[i];
                let vn = v_neighbors[j];

                if un == vn {
                    if un > v {
                        *local_map.entry(u).or_insert(0) += 1;
                        *local_map.entry(v).or_insert(0) += 1;
                        *local_map.entry(un).or_insert(0) += 1;
                    }
                    i += 1;
                    j += 1;
                } else if un < vn {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
    };

    // Parallelize triangle counting only for very large graphs
    // Threshold chosen to avoid Rayon overhead on medium graphs
    let local_counts: FxHashMap<u64, usize> = if edges.len() >= 50000 {
        edges.par_iter()
            .fold(
                || FxHashMap::with_capacity_and_hasher(n / 10, Default::default()),
                |mut local_map, &(u, v)| {
                    count_edge(&mut local_map, u, v);
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
        for &(u, v) in edges.iter() {
            count_edge(&mut local_map, u, v);
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
    let adj = simple_adjacency(graph);

    // For each node u
    for (&u, u_neighbors) in adj.iter() {
        // For each neighbor v of u (u -> v)
        for v in u_neighbors.iter() {
            if let Some(v_neighbors) = adj.get(v) {
                // For each neighbor w of v (v -> w)
                for w in v_neighbors.iter() {
                    // Check if w connects back to u (w -> u), forming a cycle u -> v -> w -> u
                    if *w != u {
                        if let Some(w_neighbors) = adj.get(w) {
                            if w_neighbors.binary_search(&u).is_ok() {
                                *triangle_count.entry(u).or_insert(0) += 1;
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

    // Keep only neighbors with higher IDs (lists are already sorted and deduplicated)
    let adj_lists: FxHashMap<u64, Vec<u64>> = simple_adjacency(graph)
        .into_iter()
        .filter_map(|(node_id, neighbors)| {
            let higher: Vec<u64> = neighbors.into_iter().filter(|&nb| nb > node_id).collect();
            if higher.is_empty() {
                None
            } else {
                Some((node_id, higher))
            }
        })
        .collect();

    let triangles_from = |u: u64, u_neighbors: &Vec<u64>| {
        let mut local_triangles = Vec::new();
        // For each pair of neighbors (v, w) where v < w
        for (i, &v) in u_neighbors.iter().enumerate() {
            if let Some(v_neighbors) = adj_lists.get(&v) {
                // Only check w's that are after v in u's neighbor list
                for &w in &u_neighbors[i + 1..] {
                    // Binary search for w in v's sorted neighbor list
                    if v_neighbors.binary_search(&w).is_ok() {
                        local_triangles.push((u, v, w));
                    }
                }
            }
        }
        local_triangles
    };

    // For very large graphs, parallelize triangle enumeration per node
    // High threshold to avoid Rayon overhead
    let all_triangles: Vec<(u64, u64, u64)> = if n >= 10000 {
        node_ids.par_iter()
            .filter_map(|&u| adj_lists.get(&u).map(|u_neighbors| (u, u_neighbors)))
            .flat_map(|(u, u_neighbors)| triangles_from(u, u_neighbors))
            .collect()
    } else {
        // Sequential for small graphs
        node_ids.iter()
            .filter_map(|&u| adj_lists.get(&u).map(|u_neighbors| (u, u_neighbors)))
            .flat_map(|(u, u_neighbors)| triangles_from(u, u_neighbors))
            .collect()
    };

    triangles.extend(all_triangles);
}

/// List triangles in directed graphs
fn list_triangles_directed(
    graph: &Graph,
    triangles: &mut Vec<(u64, u64, u64)>,
    seen_triangles: &mut HashSet<(u64, u64, u64)>,
) {
    let adj = simple_adjacency(graph);
    let nodes = graph.get_node_ids();

    for &u in nodes.iter() {
        let Some(u_neighbors) = adj.get(&u) else { continue };
        for v in u_neighbors.iter() {
            let Some(v_neighbors) = adj.get(v) else { continue };
            for &w in v_neighbors.iter() {
                if w != u {
                    // Check if w -> u exists (forming cycle u -> v -> w -> u)
                    if let Some(w_neighbors) = adj.get(&w) {
                        if w_neighbors.binary_search(&u).is_ok() {
                            // Found triangle u -> v -> w -> u
                            let mut tri = [u, *v, w];
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
/// Optimized: uses SIMD-accelerated intersection of sorted adjacency lists
fn transitivity_undirected(graph: &Graph) -> Option<f64> {
    use crate::algorithms::simd_intersection::count_intersection_sorted_above;

    // Sorted, deduplicated neighbors without self-loops, so degrees are simple-graph degrees
    let sorted_adj = simple_adjacency(graph);

    // Count connected triples: C(degree, 2) per node
    let connected_triples: usize = sorted_adj
        .values()
        .map(|neighbors| {
            let degree = neighbors.len();
            if degree >= 2 { degree * (degree - 1) / 2 } else { 0 }
        })
        .sum();

    if connected_triples == 0 {
        return None;
    }

    // Count triangles using SIMD-accelerated sorted intersection, once per distinct edge
    let mut total_triangles: usize = 0;

    for (u, v) in simple_undirected_edges(&sorted_adj) {
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

    let adj = simple_adjacency(graph);
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Directed paths of length 2 (u -> v -> w) starting at a node, over distinct neighbors.
    // A path that returns to its start (u -> v -> u) can never close a triangle, so it isn't
    // a connected triple; counting it would lower the score whenever edges are reciprocated
    let paths_from = |node_id: &u64| -> usize {
        adj.get(node_id)
            .map(|neighbors| {
                neighbors.iter()
                    .filter_map(|neighbor| adj.get(neighbor))
                    .map(|second| second.len() - usize::from(second.binary_search(node_id).is_ok()))
                    .sum()
            })
            .unwrap_or(0)
    };

    // For directed graphs, count directed paths of length 2 in parallel
    let connected_triples: usize = if n >= 1000 {
        // Parallel for large graphs
        node_ids.par_iter().map(paths_from).sum()
    } else {
        // Sequential for small graphs
        node_ids.iter().map(paths_from).sum()
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
/// Degrees count distinct neighbors other than the node itself, so parallel edges and
/// self-loops don't lower the coefficient. Returns None for nodes with degree < 2.
/// PARALLELIZED: Uses Rayon for parallel coefficient computation on large graphs
#[inline]
pub fn local_clustering_coefficient(graph: &Graph) -> FxHashMap<u64, Option<f64>> {
    use rayon::prelude::*;

    let triangle_counts = count_triangles(graph);
    let adj = simple_adjacency(graph);
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let is_directed = graph.is_directed();

    let coefficient = |node_id: &u64| -> (u64, Option<f64>) {
        let degree = adj.get(node_id).map_or(0, |neighbors| neighbors.len());
        if degree < 2 {
            return (*node_id, None);
        }

        let triangles = triangle_counts.get(node_id).copied().unwrap_or(0) as f64;
        let pairs = (degree * (degree - 1)) as f64;

        // Directed: triangles / (d * (d - 1)); undirected: triangles / C(d, 2)
        let coeff = if is_directed { triangles / pairs } else { 2.0 * triangles / pairs };
        (*node_id, Some(coeff))
    };

    // Parallelize for very large graphs
    // High threshold to avoid Rayon overhead on medium graphs
    if n >= 10000 {
        node_ids.par_iter().map(coefficient).collect()
    } else {
        node_ids.iter().map(coefficient).collect()
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
