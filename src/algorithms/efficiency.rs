use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Compute shortest path distances using BFS (for unweighted graphs)
fn bfs_distances(graph: &Graph, start: u64) -> Result<FxHashMap<u64, u64>, PyErr> {
    let mut distances = FxHashMap::default();
    let mut queue = VecDeque::new();

    queue.push_back(start);
    distances.insert(start, 0);

    while let Some(current) = queue.pop_front() {
        let current_dist = distances[&current];

        for neighbor in graph.neighbors(current)? {
            if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                e.insert(current_dist + 1);
                queue.push_back(neighbor);
            }
        }
    }

    Ok(distances)
}

/// Compute global efficiency of a graph.
///
/// The efficiency of a pair of nodes is the multiplicative inverse of the
/// shortest path distance between them (0 if the second node is unreachable).
/// Global efficiency is the average efficiency over all n(n-1) ordered pairs of
/// distinct nodes (Latora & Marchiori, 2001), so disconnected graphs score lower.
///
/// # Arguments
///
/// * `graph` - The input graph
///
/// # Returns
///
/// The global efficiency (float between 0 and 1)
#[pyfunction]
pub fn global_efficiency(graph: &Graph) -> PyResult<f64> {
    let nodes = graph.get_node_ids();
    let n = nodes.len();

    if n <= 1 {
        return Ok(0.0);
    }

    let mut total_efficiency = 0.0;

    // Compute shortest paths from each node; unreachable pairs contribute 0
    for &u in &nodes {
        let distances = bfs_distances(graph, u)?;

        for (&v, &dist) in &distances {
            if u != v && dist > 0 {
                total_efficiency += 1.0 / (dist as f64);
            }
        }
    }

    // Average over all ordered pairs of distinct nodes, not just the reachable ones
    Ok(total_efficiency / ((n * (n - 1)) as f64))
}

/// Compute local efficiency of a graph.
///
/// The local efficiency of a node is the global efficiency of the subgraph
/// induced by the node's distinct neighbors (the node itself excluded). The
/// average local efficiency is the average of local efficiencies across all nodes.
///
/// # Arguments
///
/// * `graph` - The input graph
///
/// # Returns
///
/// The average local efficiency (float between 0 and 1)
#[pyfunction]
pub fn local_efficiency(graph: &Graph) -> PyResult<f64> {
    let nodes = graph.get_node_ids();
    let n = nodes.len();

    if n == 0 {
        return Ok(0.0);
    }

    let mut total_local_eff = 0.0;

    for &node in &nodes {
        // Distinct neighbors of this node, excluding the node itself: parallel edges must not
        // list a neighbor twice, and a self-loop must not put the node into its own
        // neighborhood (the measure is about routing *around* the node)
        let mut neighbors: Vec<u64> = graph.neighbors(node)?;
        neighbors.retain(|&nb| nb != node);
        neighbors.sort_unstable();
        neighbors.dedup();
        let neighbor_count = neighbors.len();

        if neighbor_count <= 1 {
            // Need at least 2 neighbors to have efficiency
            continue;
        }

        // Compute efficiency within the neighborhood subgraph
        let neighbor_set: FxHashSet<u64> = neighbors.iter().copied().collect();
        let mut subgraph_efficiency = 0.0;

        for &u in &neighbors {
            // BFS limited to neighborhood
            let mut distances: FxHashMap<u64, u64> = FxHashMap::default();
            let mut queue = VecDeque::new();
            queue.push_back(u);
            distances.insert(u, 0);

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];

                for neighbor in graph.neighbors(current)? {
                    // Only consider edges within the neighborhood
                    if neighbor_set.contains(&neighbor) && !distances.contains_key(&neighbor) {
                        distances.insert(neighbor, current_dist + 1);
                        queue.push_back(neighbor);
                    }
                }
            }

            // Efficiency from u to every other neighbor (unreachable ones contribute 0).
            // Ordered pairs: on directed graphs u -> v and v -> u can differ, and using only
            // one of them would make the result depend on the neighbors' storage order
            for (&v, &d) in &distances {
                if v != u && d > 0 {
                    subgraph_efficiency += 1.0 / (d as f64);
                }
            }
        }

        let ordered_pairs = (neighbor_count * (neighbor_count - 1)) as f64;
        total_local_eff += subgraph_efficiency / ordered_pairs;
    }

    Ok(total_local_eff / (n as f64))
}
