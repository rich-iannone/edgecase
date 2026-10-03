use pyo3::prelude::*;
use crate::graph::Graph;
use rustc_hash::FxHashMap;
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
/// shortest path distance between them. Global efficiency is the average
/// efficiency over all pairs of nodes.
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
    let mut pair_count = 0;

    // Compute shortest paths from each node
    for &u in &nodes {
        let distances = bfs_distances(graph, u)?;

        for (&v, &dist) in &distances {
            if u != v && dist > 0 {
                total_efficiency += 1.0 / (dist as f64);
                pair_count += 1;
            }
        }
    }

    if pair_count > 0 {
        Ok(total_efficiency / (pair_count as f64))
    } else {
        Ok(0.0)
    }
}

/// Compute local efficiency of a graph.
///
/// The local efficiency of a node is the global efficiency of the subgraph
/// induced by the neighbors of the node. The average local efficiency is
/// the average of local efficiencies across all nodes.
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
        // Get neighbors of this node
        let neighbors: Vec<u64> = graph.neighbors(node)?;
        let neighbor_count = neighbors.len();

        if neighbor_count <= 1 {
            // Need at least 2 neighbors to have efficiency
            continue;
        }

        // Compute efficiency within the neighborhood subgraph
        let neighbor_set: FxHashMap<u64, ()> = neighbors.iter().map(|&n| (n, ())).collect();
        let mut subgraph_efficiency = 0.0;
        let mut pair_count = 0;

        for i in 0..neighbor_count {
            let u = neighbors[i];

            // BFS limited to neighborhood
            let mut distances: FxHashMap<u64, u64> = FxHashMap::default();
            let mut queue = VecDeque::new();
            queue.push_back(u);
            distances.insert(u, 0);

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];

                for neighbor in graph.neighbors(current)? {
                    // Only consider edges within the neighborhood
                    if neighbor_set.contains_key(&neighbor) && !distances.contains_key(&neighbor) {
                        distances.insert(neighbor, current_dist + 1);
                        queue.push_back(neighbor);
                    }
                }
            }

            // Calculate efficiency from u to other neighbors
            for &v in &neighbors[i + 1..neighbor_count] {
                if let Some(&d) = distances.get(&v) {
                    if d > 0 {
                        subgraph_efficiency += 1.0 / (d as f64);
                    }
                }
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_local_eff += subgraph_efficiency / (pair_count as f64);
        }
    }

    Ok(total_local_eff / (n as f64))
}
