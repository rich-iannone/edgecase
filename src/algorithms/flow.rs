//! Network flow algorithms
//!
//! This module implements network flow algorithms including:
//! - Maximum flow (Edmonds-Karp algorithm)
//! - Minimum cost flow (Successive Shortest Path algorithm)
//!
//! The maximum flow problem finds the maximum amount of flow that can be sent
//! from a source to a sink through a network with edge capacities.
//!
//! The minimum cost flow problem finds the cheapest way to send a specified
//! amount of flow from a source to a sink, or the maximum flow at minimum cost.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute the maximum flow from source to sink using Edmonds-Karp algorithm.
///
/// Returns a dictionary mapping edge_id -> flow amount for edges with positive flow.
///
/// # Arguments
/// * `graph` - The input graph (directed or undirected)
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `capacity` - Optional capacity attribute name on edges. If None, uses edge weight
///                for weighted graphs, or 1.0 for unweighted graphs.
///
/// # Returns
/// Dictionary mapping edge_id -> flow amount
///
/// # Notes
/// - For undirected graphs, each edge is treated as two directed edges
/// - Capacities must be non-negative
/// - Self-loops are ignored
#[pyfunction]
#[pyo3(signature = (graph, source, sink, capacity=None))]
pub fn max_flow(
    graph: &Graph,
    source: u64,
    sink: u64,
    capacity: Option<&str>,
) -> PyResult<FxHashMap<u64, f64>> {
    // Validate nodes exist
    if !graph.get_node_ids().contains(&source) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Source node {} does not exist",
            source
        )));
    }
    if !graph.get_node_ids().contains(&sink) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Sink node {} does not exist",
            sink
        )));
    }
    if source == sink {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Source and sink must be different nodes",
        ));
    }

    let edges = graph.get_edges_internal();
    let is_directed = graph.is_directed();

    // Build capacity map: (source, target) -> total capacity
    // For multigraphs, sum capacities of parallel edges
    let mut cap: FxHashMap<(u64, u64), f64> = FxHashMap::default();

    for edge in edges {
        // Skip self-loops
        if edge.source == edge.target {
            continue;
        }

        // Determine capacity for this edge
        let edge_cap = get_edge_capacity(&edge, capacity, graph.is_weighted())?;

        // Add to capacity map
        let key = (edge.source, edge.target);
        *cap.entry(key).or_insert(0.0) += edge_cap;

        // For undirected graphs, add reverse capacity
        if !is_directed {
            let rev_key = (edge.target, edge.source);
            *cap.entry(rev_key).or_insert(0.0) += edge_cap;
        }
    }

    // Initialize residual graph = capacity
    let mut residual = cap.clone();

    // Track flow on each directed edge pair
    let mut flow: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    for &key in cap.keys() {
        flow.insert(key, 0.0);
    }

    // Edmonds-Karp: BFS to find augmenting paths
    loop {
        // BFS to find augmenting path
        let path = bfs_augmenting_path(source, sink, &residual);

        if path.is_empty() {
            break; // No more augmenting paths
        }

        // Find minimum residual capacity along the path (bottleneck)
        let mut path_flow = f64::INFINITY;
        for i in 0..path.len() - 1 {
            let edge_key = (path[i], path[i + 1]);
            let res_cap = residual.get(&edge_key).copied().unwrap_or(0.0);
            path_flow = path_flow.min(res_cap);
        }

        // Update residual capacities and flow
        for i in 0..path.len() - 1 {
            let forward_key = (path[i], path[i + 1]);
            let backward_key = (path[i + 1], path[i]);

            // Decrease forward residual capacity
            if let Some(res) = residual.get_mut(&forward_key) {
                *res -= path_flow;
            }

            // Increase backward residual capacity (for flow cancellation)
            *residual.entry(backward_key).or_insert(0.0) += path_flow;

            // Update flow
            *flow.entry(forward_key).or_insert(0.0) += path_flow;
            // Decrease reverse flow if it exists
            if let Some(rev_flow) = flow.get_mut(&backward_key) {
                if *rev_flow > 0.0 {
                    let cancel = path_flow.min(*rev_flow);
                    *rev_flow -= cancel;
                }
            }
        }
    }

    // Build flow dict mapping edge_id -> flow
    let mut flow_dict: FxHashMap<u64, f64> = FxHashMap::default();

    for edge in edges {
        if edge.source == edge.target {
            continue; // Skip self-loops
        }

        let forward_flow = flow.get(&(edge.source, edge.target)).copied().unwrap_or(0.0);

        if is_directed {
            // Directed: flow is in edge direction
            if forward_flow > 1e-10 {
                flow_dict.insert(edge.id, forward_flow);
            }
        } else {
            // Undirected: flow could be in either direction
            let backward_flow = flow.get(&(edge.target, edge.source)).copied().unwrap_or(0.0);
            // Net flow on this edge
            let net_flow = (forward_flow - backward_flow).abs();
            if net_flow > 1e-10 {
                flow_dict.insert(edge.id, net_flow);
            }
        }
    }

    Ok(flow_dict)
}

/// Compute the maximum flow value only (without flow assignment).
/// This is faster if you only need the flow value.
#[pyfunction]
#[pyo3(signature = (graph, source, sink, capacity=None))]
pub fn max_flow_value(
    graph: &Graph,
    source: u64,
    sink: u64,
    capacity: Option<&str>,
) -> PyResult<f64> {
    // Validate nodes exist
    if !graph.get_node_ids().contains(&source) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Source node {} does not exist",
            source
        )));
    }
    if !graph.get_node_ids().contains(&sink) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Sink node {} does not exist",
            sink
        )));
    }
    if source == sink {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Source and sink must be different nodes",
        ));
    }

    let edges = graph.get_edges_internal();
    let is_directed = graph.is_directed();

    // Build capacity map
    let mut cap: FxHashMap<(u64, u64), f64> = FxHashMap::default();

    for edge in edges {
        if edge.source == edge.target {
            continue;
        }

        let edge_cap = get_edge_capacity(&edge, capacity, graph.is_weighted())?;

        let key = (edge.source, edge.target);
        *cap.entry(key).or_insert(0.0) += edge_cap;

        if !is_directed {
            let rev_key = (edge.target, edge.source);
            *cap.entry(rev_key).or_insert(0.0) += edge_cap;
        }
    }

    let mut residual = cap;
    let mut total_flow = 0.0;

    loop {
        let path = bfs_augmenting_path(source, sink, &residual);

        if path.is_empty() {
            break;
        }

        let mut path_flow = f64::INFINITY;
        for i in 0..path.len() - 1 {
            let edge_key = (path[i], path[i + 1]);
            let res_cap = residual.get(&edge_key).copied().unwrap_or(0.0);
            path_flow = path_flow.min(res_cap);
        }

        for i in 0..path.len() - 1 {
            let forward_key = (path[i], path[i + 1]);
            let backward_key = (path[i + 1], path[i]);

            if let Some(res) = residual.get_mut(&forward_key) {
                *res -= path_flow;
            }
            *residual.entry(backward_key).or_insert(0.0) += path_flow;
        }

        total_flow += path_flow;
    }

    Ok(total_flow)
}

/// Compute minimum cost flow from source to sink.
///
/// Finds the flow with minimum total cost. Can either send a specified amount of
/// flow (demand) or find the maximum flow at minimum cost.
///
/// Uses the Successive Shortest Path algorithm (Bellman-Ford based).
///
/// # Arguments
/// * `graph` - The input graph (must be directed)
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `demand` - Required flow amount. If None, finds maximum flow at minimum cost.
/// * `capacity` - Optional capacity attribute name on edges. If None, uses edge weight
///                for weighted graphs, or 1.0 for unweighted graphs.
/// * `cost` - Optional cost attribute name on edges. If None, uses 1.0 per unit flow.
///
/// # Returns
/// A tuple of (total_flow, total_cost, flow_dict) where flow_dict maps edge_id -> flow amount
///
/// # Notes
/// - Graph must be directed
/// - Self-loops are ignored
/// - Negative costs are allowed (handled by Bellman-Ford)
/// - Returns error if demand cannot be satisfied
#[pyfunction]
#[pyo3(signature = (graph, source, sink, demand=None, capacity=None, cost=None))]
pub fn min_cost_flow(
    graph: &Graph,
    source: u64,
    sink: u64,
    demand: Option<f64>,
    capacity: Option<&str>,
    cost: Option<&str>,
) -> PyResult<(f64, f64, FxHashMap<u64, f64>)> {
    // Validate graph is directed
    if !graph.is_directed() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_cost_flow requires a directed graph",
        ));
    }

    // Validate nodes exist
    if !graph.get_node_ids().contains(&source) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Source node {} does not exist",
            source
        )));
    }
    if !graph.get_node_ids().contains(&sink) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Sink node {} does not exist",
            sink
        )));
    }
    if source == sink {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Source and sink must be different nodes",
        ));
    }

    // Validate demand
    if let Some(d) = demand {
        if d < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Demand must be non-negative",
            ));
        }
    }

    let edges = graph.get_edges_internal();

    // Build capacity and cost maps: (source, target) -> (total_capacity, cost_per_unit)
    // For parallel edges, we sum capacities but keep the minimum cost
    let mut cap: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    let mut edge_cost: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    // Map from (source, target) to original edge id (for output)
    let mut edge_id_map: FxHashMap<(u64, u64), u64> = FxHashMap::default();

    for edge in edges {
        // Skip self-loops
        if edge.source == edge.target {
            continue;
        }

        let edge_cap = get_edge_capacity(edge, capacity, graph.is_weighted())?;
        let unit_cost = get_edge_cost(edge, cost)?;

        let key = (edge.source, edge.target);
        *cap.entry(key).or_insert(0.0) += edge_cap;

        // For parallel edges, use minimum cost
        edge_cost
            .entry(key)
            .and_modify(|c| *c = c.min(unit_cost))
            .or_insert(unit_cost);

        // Store edge ID (first one wins for parallel edges)
        edge_id_map.entry(key).or_insert(edge.id);
    }

    // Initialize residual graph = capacity
    let mut residual = cap.clone();

    // For the residual graph, we need reverse edges with negative cost
    // Initialize reverse edges with 0 capacity (for flow cancellation)
    // Collect edges first to avoid borrow issues
    let forward_edges: Vec<((u64, u64), f64)> = edge_cost.iter().map(|(&k, &v)| (k, v)).collect();
    for ((u, v), c) in forward_edges {
        residual.entry((v, u)).or_insert(0.0);
        // Reverse edge has negative cost (canceling flow refunds cost)
        edge_cost.entry((v, u)).or_insert(-c);
    }

    // Track flow on each directed edge pair
    let mut flow: FxHashMap<(u64, u64), f64> = FxHashMap::default();

    let mut total_flow = 0.0;
    let mut total_cost = 0.0;

    // Determine target flow
    let target_flow = demand.unwrap_or(f64::INFINITY);

    // Successive Shortest Path algorithm
    loop {
        if total_flow >= target_flow - 1e-10 {
            break; // Satisfied demand
        }

        // Find shortest path (by cost) using Bellman-Ford
        let path_result = bellman_ford_shortest_path(source, sink, &residual, &edge_cost);

        let (path, path_cost) = match path_result {
            Some(p) => p,
            None => break, // No more augmenting paths
        };

        if path.is_empty() {
            break;
        }

        // Find minimum residual capacity along the path (bottleneck)
        let mut path_flow = f64::INFINITY;
        for i in 0..path.len() - 1 {
            let edge_key = (path[i], path[i + 1]);
            let res_cap = residual.get(&edge_key).copied().unwrap_or(0.0);
            path_flow = path_flow.min(res_cap);
        }

        // Limit flow to remaining demand
        let remaining = target_flow - total_flow;
        path_flow = path_flow.min(remaining);

        if path_flow < 1e-10 {
            break;
        }

        // Update residual capacities and flow
        for i in 0..path.len() - 1 {
            let forward_key = (path[i], path[i + 1]);
            let backward_key = (path[i + 1], path[i]);

            // Decrease forward residual capacity
            if let Some(res) = residual.get_mut(&forward_key) {
                *res -= path_flow;
            }

            // Increase backward residual capacity (for flow cancellation)
            *residual.entry(backward_key).or_insert(0.0) += path_flow;

            // Update flow
            *flow.entry(forward_key).or_insert(0.0) += path_flow;
            // Decrease reverse flow if it exists
            if let Some(rev_flow) = flow.get_mut(&backward_key) {
                if *rev_flow > 0.0 {
                    let cancel = path_flow.min(*rev_flow);
                    *rev_flow -= cancel;
                }
            }
        }

        total_flow += path_flow;
        total_cost += path_flow * path_cost;
    }

    // Check if demand was satisfied
    if let Some(d) = demand {
        if total_flow < d - 1e-10 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot satisfy demand of {}. Maximum achievable flow is {}",
                d, total_flow
            )));
        }
    }

    // Build flow dict mapping edge_id -> flow
    let mut flow_dict: FxHashMap<u64, f64> = FxHashMap::default();

    for edge in edges {
        if edge.source == edge.target {
            continue; // Skip self-loops
        }

        let forward_flow = flow.get(&(edge.source, edge.target)).copied().unwrap_or(0.0);

        if forward_flow > 1e-10 {
            flow_dict.insert(edge.id, forward_flow);
        }
    }

    Ok((total_flow, total_cost, flow_dict))
}

/// Helper function to get edge cost
fn get_edge_cost(edge: &crate::graph::Edge, cost: Option<&str>) -> PyResult<f64> {
    // Helper to extract float from edge attributes
    let get_attr = |attr_name: &str| -> Option<f64> {
        if let Some(attrs) = &edge.attributes {
            Python::with_gil(|py| {
                if let Ok(val) = attrs.bind(py).get_item(attr_name) {
                    if let Some(v) = val {
                        v.extract::<f64>().ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        } else {
            None
        }
    };

    let unit_cost = if let Some(cost_attr) = cost {
        // Use specified cost attribute
        get_attr(cost_attr).unwrap_or(1.0)
    } else {
        // Default: try "cost" attribute, otherwise 1.0
        get_attr("cost").unwrap_or(1.0)
    };

    Ok(unit_cost)
}

/// Bellman-Ford shortest path algorithm for finding minimum cost path.
/// Returns (path, total_cost) or None if no path exists.
/// Handles negative edge costs.
fn bellman_ford_shortest_path(
    source: u64,
    sink: u64,
    residual: &FxHashMap<(u64, u64), f64>,
    edge_cost: &FxHashMap<(u64, u64), f64>,
) -> Option<(Vec<u64>, f64)> {
    // Collect all nodes that appear in residual graph
    let mut nodes: Vec<u64> = Vec::new();
    for &(u, v) in residual.keys() {
        if !nodes.contains(&u) {
            nodes.push(u);
        }
        if !nodes.contains(&v) {
            nodes.push(v);
        }
    }

    if nodes.is_empty() {
        return None;
    }

    // Distance and parent maps
    let mut dist: FxHashMap<u64, f64> = FxHashMap::default();
    let mut parent: FxHashMap<u64, u64> = FxHashMap::default();

    for &node in &nodes {
        dist.insert(node, f64::INFINITY);
    }
    dist.insert(source, 0.0);

    // Build edge list for Bellman-Ford
    let mut edge_list: Vec<(u64, u64, f64)> = Vec::new();
    for (&(u, v), &cap) in residual.iter() {
        if cap > 1e-10 {
            let cost = edge_cost.get(&(u, v)).copied().unwrap_or(1.0);
            edge_list.push((u, v, cost));
        }
    }

    // Bellman-Ford relaxation
    let n = nodes.len();
    for _ in 0..n - 1 {
        let mut updated = false;
        for &(u, v, cost) in &edge_list {
            let du = dist.get(&u).copied().unwrap_or(f64::INFINITY);
            let dv = dist.get(&v).copied().unwrap_or(f64::INFINITY);
            if du < f64::INFINITY && du + cost < dv - 1e-10 {
                dist.insert(v, du + cost);
                parent.insert(v, u);
                updated = true;
            }
        }
        if !updated {
            break; // Early termination if no updates
        }
    }

    // Check if sink is reachable
    let sink_dist = dist.get(&sink).copied().unwrap_or(f64::INFINITY);
    if sink_dist == f64::INFINITY {
        return None;
    }

    // Reconstruct path
    let mut path = Vec::new();
    let mut current = sink;
    path.push(current);

    while current != source {
        if let Some(&p) = parent.get(&current) {
            current = p;
            path.push(current);
        } else {
            return None; // Path broken
        }
    }

    path.reverse();
    Some((path, sink_dist))
}

/// Helper function to get edge capacity
fn get_edge_capacity(
    edge: &crate::graph::Edge,
    capacity: Option<&str>,
    is_weighted: bool,
) -> PyResult<f64> {
    // Helper to extract float from edge attributes
    let get_attr = |attr_name: &str| -> Option<f64> {
        if let Some(attrs) = &edge.attributes {
            Python::with_gil(|py| {
                if let Ok(val) = attrs.bind(py).get_item(attr_name) {
                    if let Some(v) = val {
                        v.extract::<f64>().ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        } else {
            None
        }
    };

    let edge_cap = if let Some(cap_attr) = capacity {
        // Use specified capacity attribute
        get_attr(cap_attr).unwrap_or(1.0)
    } else {
        // Default behavior:
        // 1. Try "capacity" attribute first
        // 2. Fall back to edge weight for weighted graphs
        // 3. Default to 1.0
        if let Some(cap) = get_attr("capacity") {
            cap
        } else if is_weighted {
            edge.weight.unwrap_or(1.0)
        } else {
            1.0
        }
    };

    if edge_cap < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Negative capacity {} on edge from {} to {}",
            edge_cap, edge.source, edge.target
        )));
    }

    Ok(edge_cap)
}

/// BFS to find an augmenting path from source to sink in the residual graph.
/// Returns the path as a vector of node IDs, or empty vector if no path exists.
fn bfs_augmenting_path(
    source: u64,
    sink: u64,
    residual: &FxHashMap<(u64, u64), f64>,
) -> Vec<u64> {
    let mut visited: FxHashMap<u64, u64> = FxHashMap::default(); // node -> parent
    let mut queue: VecDeque<u64> = VecDeque::new();

    queue.push_back(source);
    visited.insert(source, source); // Source is its own parent

    // Build adjacency list from residual graph for faster lookup
    let mut adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for (&(u, v), &cap) in residual.iter() {
        if cap > 1e-10 {
            // Only edges with positive residual capacity
            adj.entry(u).or_default().push(v);
        }
    }

    while let Some(current) = queue.pop_front() {
        if current == sink {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = sink;
            while node != source {
                path.push(node);
                node = *visited.get(&node).unwrap();
            }
            path.push(source);
            path.reverse();
            return path;
        }

        if let Some(neighbors) = adj.get(&current) {
            for &neighbor in neighbors {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    Vec::new() // No path found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_augmenting_path() {
        let mut residual: FxHashMap<(u64, u64), f64> = FxHashMap::default();
        residual.insert((1, 2), 10.0);
        residual.insert((2, 3), 5.0);
        residual.insert((1, 3), 3.0);

        let path = bfs_augmenting_path(1, 3, &residual);
        assert!(!path.is_empty());
        assert_eq!(path[0], 1);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_no_path() {
        let mut residual: FxHashMap<(u64, u64), f64> = FxHashMap::default();
        residual.insert((1, 2), 10.0);
        // No path from 1 to 3

        let path = bfs_augmenting_path(1, 3, &residual);
        assert!(path.is_empty());
    }
}
