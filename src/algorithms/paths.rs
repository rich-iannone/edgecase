//! Shortest path algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Priority queue item for Dijkstra's algorithm
#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the ordering
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Find shortest path between two nodes
#[pyfunction]
#[inline]
pub fn shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    // Check if nodes exist
    if !graph.has_node_id(source) || !graph.has_node_id(target) {
        return Ok(None);
    }

    // Check if graph is weighted - use Dijkstra if so, BFS otherwise
    if graph.is_weighted() {
        dijkstra_shortest_path(graph, source, target)
    } else {
        bfs_shortest_path(graph, source, target)
    }
}

/// Dijkstra's algorithm for weighted graphs
#[inline]
fn dijkstra_shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut node_to_idx: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut idx_to_node: Vec<u64> = Vec::with_capacity(n);

    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }

    let source_idx = *node_to_idx.get(&source).unwrap();
    let target_idx = *node_to_idx.get(&target).unwrap();

    // Initialize distances and predecessors
    let n = node_ids.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[source_idx] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: source_idx,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we've found a better path already
        if cost > dist[node] {
            continue;
        }

        // Found target
        if node == target_idx {
            break;
        }

        // Explore neighbors
        let node_id = idx_to_node[node];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                    // Get edge weight
                    let weight = graph.get_edge_weight(node_id, neighbor_id);
                    let new_cost = cost + weight;

                    // Relaxation step
                    if new_cost < dist[neighbor_idx] {
                        dist[neighbor_idx] = new_cost;
                        prev[neighbor_idx] = Some(node);
                        heap.push(State {
                            cost: new_cost,
                            node: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    // Reconstruct path
    if dist[target_idx].is_infinite() {
        return Ok(None);
    }

    let mut path = Vec::with_capacity(n / 10);
    let mut current = target_idx;

    while let Some(p) = prev[current] {
        path.push(idx_to_node[current]);
        current = p;
    }
    path.push(idx_to_node[current]);
    path.reverse();

    Ok(Some(path))
}

/// BFS for unweighted graphs
#[inline]
fn bfs_shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    // Simple BFS for unweighted graphs
    let n = graph.num_nodes();
    let mut queue = VecDeque::with_capacity(n);
    let mut visited = FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut parent: FxHashMap<u64, u64> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    queue.push_back(source);
    visited.insert(source, true);

    while let Some(current) = queue.pop_front() {
        if current == target {
            // Reconstruct path
            let mut path = Vec::with_capacity(n / 10);
            path.push(current);
            let mut node = current;

            while let Some(&p) = parent.get(&node) {
                path.push(p);
                node = p;
            }

            path.reverse();
            return Ok(Some(path));
        }

        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(neighbor) {
                    e.insert(true);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // No path found
    Ok(None)
}

/// Bellman-Ford algorithm for graphs with negative weights
///
/// Returns (path, distance) tuple, or None if target is unreachable
/// Returns error if a negative cycle is detected
#[pyfunction]
#[inline]
pub fn bellman_ford_shortest_path(
    graph: &Graph,
    source: u64,
    target: u64,
) -> PyResult<Option<(Vec<u64>, f64)>> {
    // Check if nodes exist
    if !graph.has_node_id(source) || !graph.has_node_id(target) {
        return Ok(None);
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut node_to_idx: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut idx_to_node: Vec<u64> = Vec::with_capacity(n);

    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }

    let source_idx = *node_to_idx.get(&source).unwrap();
    let target_idx = *node_to_idx.get(&target).unwrap();

    let n = node_ids.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];

    dist[source_idx] = 0.0;

    // Relax edges |V| - 1 times
    for _ in 0..n - 1 {
        let mut updated = false;

        for &node_id in &node_ids {
            if let Some(&node_idx) = node_to_idx.get(&node_id) {
                if dist[node_idx].is_infinite() {
                    continue;
                }

                if let Ok(neighbors) = graph.neighbors(node_id) {
                    for neighbor_id in neighbors {
                        if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                            let weight = graph.get_edge_weight(node_id, neighbor_id);
                            let new_dist = dist[node_idx] + weight;

                            if new_dist < dist[neighbor_idx] {
                                dist[neighbor_idx] = new_dist;
                                prev[neighbor_idx] = Some(node_idx);
                                updated = true;
                            }
                        }
                    }
                }
            }
        }

        // Early termination if no updates in this iteration
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for &node_id in &node_ids {
        if let Some(&node_idx) = node_to_idx.get(&node_id) {
            if dist[node_idx].is_infinite() {
                continue;
            }

            if let Ok(neighbors) = graph.neighbors(node_id) {
                for neighbor_id in neighbors {
                    if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                        let weight = graph.get_edge_weight(node_id, neighbor_id);
                        let new_dist = dist[node_idx] + weight;

                        if new_dist < dist[neighbor_idx] {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Graph contains a negative weight cycle",
                            ));
                        }
                    }
                }
            }
        }
    }

    // Check if target is reachable
    if dist[target_idx].is_infinite() {
        return Ok(None);
    }

    // Reconstruct path
    let mut path = Vec::with_capacity(n / 10);
    let mut current = target_idx;

    while let Some(p) = prev[current] {
        path.push(idx_to_node[current]);
        current = p;
    }
    path.push(idx_to_node[current]);
    path.reverse();

    Ok(Some((path, dist[target_idx])))
}

/// Compute shortest path lengths for all pairs of nodes
///
/// Returns a nested dictionary where result[source][target] = distance
/// Uses Dijkstra's algorithm from each source node
#[pyfunction]
#[inline]
pub fn all_pairs_shortest_path_length(
    graph: &Graph,
) -> PyResult<FxHashMap<u64, FxHashMap<u64, f64>>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Result map: source -> (target -> distance)
    let mut result: FxHashMap<u64, FxHashMap<u64, f64>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    // Run Dijkstra from each node
    for &source in &node_ids {
        let distances = single_source_dijkstra_distances(graph, source)?;
        result.insert(source, distances);
    }

    Ok(result)
}

/// Compute shortest paths for all pairs of nodes
///
/// Returns a nested dictionary where result[source][target] = path (list of node IDs)
/// Uses Dijkstra's algorithm from each source node
#[pyfunction]
#[inline]
pub fn all_pairs_shortest_path(
    graph: &Graph,
) -> PyResult<FxHashMap<u64, FxHashMap<u64, Vec<u64>>>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Result map: source -> (target -> path)
    let mut result: FxHashMap<u64, FxHashMap<u64, Vec<u64>>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    // Run Dijkstra from each node
    for &source in &node_ids {
        let paths = single_source_dijkstra_paths(graph, source)?;
        result.insert(source, paths);
    }

    Ok(result)
}

/// Reject negative edge weights before running Dijkstra.
///
/// Dijkstra assumes no path gets cheaper by adding an edge. With a negative weight it can
/// return wrong distances, and on an undirected graph a negative edge is a negative cycle
/// (walk it back and forth) that keeps relaxing forever, so the call never returned.
fn ensure_non_negative_weights(graph: &Graph) -> PyResult<()> {
    if let Some(edge) = graph
        .get_edges_internal()
        .iter()
        .find(|e| e.weight.is_some_and(|w| w < 0.0))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dijkstra-based shortest paths require non-negative edge weights, but edge {} \
             ({} -> {}) has weight {}. Use bellman_ford_shortest_path() for graphs with \
             negative weights (on an undirected graph a negative edge is a negative cycle, \
             so shortest paths are undefined).",
            edge.id,
            edge.source,
            edge.target,
            edge.weight.unwrap_or_default()
        )));
    }
    Ok(())
}

/// Helper function: Compute distances from a single source using Dijkstra
#[inline]
pub(crate) fn single_source_dijkstra_distances(
    graph: &Graph,
    source: u64,
) -> PyResult<FxHashMap<u64, f64>> {
    ensure_non_negative_weights(graph)?;
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut node_to_idx: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut idx_to_node: Vec<u64> = Vec::with_capacity(n);

    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }

    let source_idx = *node_to_idx.get(&source).unwrap();

    // Initialize distances
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    dist[source_idx] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: source_idx,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we've found a better path already
        if cost > dist[node] {
            continue;
        }

        // Explore neighbors
        let node_id = idx_to_node[node];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                    // Get edge weight
                    let weight = graph.get_edge_weight(node_id, neighbor_id);
                    let new_cost = cost + weight;

                    // Relaxation step
                    if new_cost < dist[neighbor_idx] {
                        dist[neighbor_idx] = new_cost;
                        heap.push(State {
                            cost: new_cost,
                            node: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    // Convert to result map (only include reachable nodes)
    let mut result: FxHashMap<u64, f64> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &distance) in dist.iter().enumerate() {
        if distance.is_finite() {
            result.insert(idx_to_node[idx], distance);
        }
    }

    Ok(result)
}

/// Helper function: Compute paths from a single source using Dijkstra
#[inline]
fn single_source_dijkstra_paths(graph: &Graph, source: u64) -> PyResult<FxHashMap<u64, Vec<u64>>> {
    ensure_non_negative_weights(graph)?;
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut node_to_idx: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut idx_to_node: Vec<u64> = Vec::with_capacity(n);

    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }

    let source_idx = *node_to_idx.get(&source).unwrap();

    // Initialize distances and predecessors
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[source_idx] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: source_idx,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we've found a better path already
        if cost > dist[node] {
            continue;
        }

        // Explore neighbors
        let node_id = idx_to_node[node];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                    // Get edge weight
                    let weight = graph.get_edge_weight(node_id, neighbor_id);
                    let new_cost = cost + weight;

                    // Relaxation step
                    if new_cost < dist[neighbor_idx] {
                        dist[neighbor_idx] = new_cost;
                        prev[neighbor_idx] = Some(node);
                        heap.push(State {
                            cost: new_cost,
                            node: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    // Reconstruct paths for all reachable nodes
    let mut result: FxHashMap<u64, Vec<u64>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    for target_idx in 0..n {
        if dist[target_idx].is_finite() {
            let mut path = Vec::new();
            let mut current = target_idx;

            while let Some(p) = prev[current] {
                path.push(idx_to_node[current]);
                current = p;
            }
            path.push(idx_to_node[current]);
            path.reverse();

            result.insert(idx_to_node[target_idx], path);
        }
    }

    Ok(result)
}

/// Find k shortest paths between source and target using Yen's algorithm
///
/// Returns a list of tuples (path, distance) sorted by distance
#[pyfunction]
#[inline]
pub fn k_shortest_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    k: usize,
) -> PyResult<Vec<(Vec<u64>, f64)>> {
    // Check if nodes exist
    if !graph.has_node_id(source) || !graph.has_node_id(target) {
        return Ok(Vec::new());
    }

    if k == 0 {
        return Ok(Vec::new());
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut node_to_idx: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut idx_to_node: Vec<u64> = Vec::with_capacity(n);

    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }

    let source_idx = *node_to_idx.get(&source).unwrap();
    let target_idx = *node_to_idx.get(&target).unwrap();

    // Result list
    let mut k_paths: Vec<(Vec<u64>, f64)> = Vec::with_capacity(k);

    // Find first shortest path
    if let Some((path, dist)) = dijkstra_path_with_distance(
        graph,
        &node_to_idx,
        &idx_to_node,
        source_idx,
        target_idx,
        &[],
        &[],
    )? {
        k_paths.push((path, dist));
    } else {
        // No path exists
        return Ok(k_paths);
    }

    // Candidate paths (min-heap by distance)
    let mut candidates: BinaryHeap<CandidatePath> = BinaryHeap::new();

    // Find k-1 more paths
    for k_idx in 1..k {
        let (prev_path, _) = &k_paths[k_idx - 1];

        // Iterate over each node in the previous k-shortest path except the target
        for i in 0..prev_path.len() - 1 {
            let spur_node = prev_path[i];
            let spur_idx = *node_to_idx.get(&spur_node).unwrap();

            // Root path: portion from source to spur node
            let root_path = &prev_path[0..=i];

            // Collect edges and nodes to remove
            let mut removed_edges: Vec<(u64, u64)> = Vec::new();
            let mut removed_nodes: Vec<u64> = Vec::new();

            // Remove edges that are part of previous paths with the same root path
            for (p_path, _) in &k_paths {
                if p_path.len() > i && &p_path[0..=i] == root_path && i + 1 < p_path.len() {
                    removed_edges.push((p_path[i], p_path[i + 1]));
                }
            }

            // Remove nodes in the root path except the spur node
            for &node in &prev_path[0..i] {
                removed_nodes.push(node);
            }

            // Find spur path from spur node to target
            if let Some((spur_path, spur_dist)) = dijkstra_path_with_distance(
                graph,
                &node_to_idx,
                &idx_to_node,
                spur_idx,
                target_idx,
                &removed_edges,
                &removed_nodes,
            )? {
                // Combine root path and spur path
                let mut total_path: Vec<u64> = root_path[0..i].to_vec();
                total_path.extend_from_slice(&spur_path);

                // Calculate total distance
                let root_dist: f64 = (0..i)
                    .map(|j| graph.get_edge_weight(prev_path[j], prev_path[j + 1]))
                    .sum();
                let total_dist = root_dist + spur_dist;

                // Add to candidates if it's a new path
                let candidate = CandidatePath {
                    path: total_path.clone(),
                    distance: total_dist,
                };

                // Check if this path is already in candidates or k_paths
                let is_duplicate = k_paths.iter().any(|(p, _)| p == &total_path)
                    || candidates.iter().any(|c| c.path == total_path);

                if !is_duplicate {
                    candidates.push(candidate);
                }
            }
        }

        // No more candidates
        if candidates.is_empty() {
            break;
        }

        // Add best candidate to k_paths
        if let Some(best) = candidates.pop() {
            k_paths.push((best.path, best.distance));
        }
    }

    Ok(k_paths)
}

/// Priority queue item for k-shortest paths
#[derive(Clone, PartialEq)]
struct CandidatePath {
    path: Vec<u64>,
    distance: f64,
}

impl Eq for CandidatePath {}

impl Ord for CandidatePath {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the ordering
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CandidatePath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra with edge/node removal for k-shortest paths
#[inline]
fn dijkstra_path_with_distance(
    graph: &Graph,
    node_to_idx: &FxHashMap<u64, usize>,
    idx_to_node: &[u64],
    source_idx: usize,
    target_idx: usize,
    removed_edges: &[(u64, u64)],
    removed_nodes: &[u64],
) -> PyResult<Option<(Vec<u64>, f64)>> {
    let n = idx_to_node.len();

    // Convert removed items to sets for faster lookup
    let removed_edge_set: FxHashMap<(u64, u64), bool> =
        removed_edges.iter().map(|&e| (e, true)).collect();
    let removed_node_set: FxHashMap<u64, bool> = removed_nodes.iter().map(|&n| (n, true)).collect();

    // Initialize distances and predecessors
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[source_idx] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: source_idx,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we've found a better path already
        if cost > dist[node] {
            continue;
        }

        // Found target
        if node == target_idx {
            break;
        }

        // Explore neighbors
        let node_id = idx_to_node[node];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                // Skip removed nodes
                if removed_node_set.contains_key(&neighbor_id) {
                    continue;
                }

                // Skip removed edges
                if removed_edge_set.contains_key(&(node_id, neighbor_id)) {
                    continue;
                }

                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                    // Get edge weight
                    let weight = graph.get_edge_weight(node_id, neighbor_id);
                    let new_cost = cost + weight;

                    // Relaxation step
                    if new_cost < dist[neighbor_idx] {
                        dist[neighbor_idx] = new_cost;
                        prev[neighbor_idx] = Some(node);
                        heap.push(State {
                            cost: new_cost,
                            node: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    // Reconstruct path
    if dist[target_idx].is_infinite() {
        return Ok(None);
    }

    let mut path = Vec::with_capacity(n / 10);
    let mut current = target_idx;

    while let Some(p) = prev[current] {
        path.push(idx_to_node[current]);
        current = p;
    }
    path.push(idx_to_node[current]);
    path.reverse();

    Ok(Some((path, dist[target_idx])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_path_direct_connection() {
        let mut graph = Graph::new_rust(false, false);
        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");

        graph.add_test_edge_by_label("A", "B", None);

        let path = shortest_path(&graph, a, b).unwrap();
        assert_eq!(path, Some(vec![a, b]));
    }

    #[test]
    fn test_shortest_path_multiple_hops() {
        let mut graph = Graph::new_rust(false, false);
        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");
        let c = graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);

        let path = shortest_path(&graph, a, c).unwrap();
        assert_eq!(path, Some(vec![a, b, c]));
    }

    #[test]
    fn test_shortest_path_no_connection() {
        let mut graph = Graph::new_rust(false, false);
        let a = graph.add_test_node("A");
        let b = graph.add_test_node("B");

        // Unreachable target is reported as Ok(None) (older API returned Err)
        let result = shortest_path(&graph, a, b).unwrap();
        assert_eq!(result, None);
    }
}
