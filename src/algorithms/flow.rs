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

    let (_, edge_flows, _) = edmonds_karp(graph, source, sink, capacity)?;

    // Only report edges that carry flow
    Ok(edge_flows.into_iter().filter(|&(_, flow)| flow > 1e-10).collect())
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

    let (total_flow, _, _) = edmonds_karp(graph, source, sink, capacity)?;
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

    // Residual network with one forward arc per edge, each paired with its own reverse
    // (cancellation) arc. Keeping arcs per edge, rather than per (source, target) pair,
    // means parallel edges keep their own costs and capacities, and a real v -> u edge
    // never shares capacity or cost with the reverse arc of u -> v.
    let mut node_index: FxHashMap<u64, usize> = FxHashMap::default();
    for node_id in graph.get_node_ids() {
        let next = node_index.len();
        node_index.insert(node_id, next);
    }

    let mut arcs: Vec<ResidualArc> = Vec::with_capacity(edges.len() * 2);
    let mut out_arcs: Vec<Vec<usize>> = vec![Vec::new(); node_index.len()];
    // (edge id, index of its forward arc, original capacity)
    let mut edge_arcs: Vec<(u64, usize, f64)> = Vec::with_capacity(edges.len());

    for edge in edges {
        // Skip self-loops
        if edge.source == edge.target {
            continue;
        }

        let edge_cap = get_edge_capacity(edge, capacity, graph.is_weighted())?;
        let unit_cost = get_edge_cost(edge, cost)?;
        let (u, v) = (node_index[&edge.source], node_index[&edge.target]);

        let forward = arcs.len();
        arcs.push(ResidualArc { to: v, residual: edge_cap, cost: unit_cost, rev: forward + 1 });
        arcs.push(ResidualArc { to: u, residual: 0.0, cost: -unit_cost, rev: forward });
        out_arcs[u].push(forward);
        out_arcs[v].push(forward + 1);
        edge_arcs.push((edge.id, forward, edge_cap));
    }

    let (source_idx, sink_idx) = (node_index[&source], node_index[&sink]);
    let mut total_flow = 0.0;
    let mut total_cost = 0.0;

    // Determine target flow
    let target_flow = demand.unwrap_or(f64::INFINITY);

    // Successive Shortest Path algorithm
    loop {
        if total_flow >= target_flow - 1e-10 {
            break; // Satisfied demand
        }

        // Find the cheapest augmenting path (as arc indices) using Bellman-Ford
        let (path, path_cost) = match cheapest_residual_path(source_idx, sink_idx, &arcs, &out_arcs)? {
            Some(p) => p,
            None => break, // No more augmenting paths
        };

        // Find minimum residual capacity along the path (bottleneck)
        let mut path_flow = path
            .iter()
            .map(|&a| arcs[a].residual)
            .fold(f64::INFINITY, f64::min);

        // Limit flow to remaining demand
        path_flow = path_flow.min(target_flow - total_flow);

        if path_flow < 1e-10 {
            break;
        }

        // Push flow: shrink each arc's residual and grow its paired reverse arc
        for &a in &path {
            arcs[a].residual -= path_flow;
            let rev = arcs[a].rev;
            arcs[rev].residual += path_flow;
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

    // Build flow dict mapping edge_id -> flow: each edge's flow is how much of its own
    // capacity has been used
    let mut flow_dict: FxHashMap<u64, f64> = FxHashMap::default();

    for &(edge_id, forward, edge_cap) in &edge_arcs {
        let edge_flow = edge_cap - arcs[forward].residual;
        if edge_flow > 1e-10 {
            flow_dict.insert(edge_id, edge_flow);
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
                if let Ok(Some(v)) = attrs.bind(py).get_item(attr_name) {
                    v.extract::<f64>().ok()
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

/// An arc of the min-cost-flow residual network
#[derive(Clone, Copy)]
struct ResidualArc {
    /// Index of the node the arc points to
    to: usize,
    /// Remaining capacity on this arc
    residual: f64,
    /// Cost per unit of flow (negated on reverse arcs, so cancelling flow refunds its cost)
    cost: f64,
    /// Index of the paired arc in the opposite direction
    rev: usize,
}

/// Bellman-Ford search for the cheapest source -> sink path over arcs with residual capacity.
/// Returns (arc indices along the path, total cost per unit), or None if the sink is
/// unreachable. Handles negative arc costs; errors if a negative-cost cycle is reachable,
/// since the cheapest path is then undefined.
fn cheapest_residual_path(
    source: usize,
    sink: usize,
    arcs: &[ResidualArc],
    out_arcs: &[Vec<usize>],
) -> PyResult<Option<(Vec<usize>, f64)>> {
    let n = out_arcs.len();
    let mut dist = vec![f64::INFINITY; n];
    // Arc used to reach each node on the current cheapest path
    let mut parent_arc: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;

    // Bellman-Ford relaxation; an n-th round that still improves something means a
    // negative-cost cycle is reachable from the source
    for round in 0..n {
        let mut updated = false;
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            for &a in &out_arcs[u] {
                let arc = &arcs[a];
                if arc.residual > 1e-10 && dist[u] + arc.cost < dist[arc.to] - 1e-10 {
                    dist[arc.to] = dist[u] + arc.cost;
                    parent_arc[arc.to] = Some(a);
                    updated = true;
                }
            }
        }
        if !updated {
            break; // Early termination if no updates
        }
        if round == n - 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Graph contains a negative-cost cycle, so the minimum cost flow is undefined",
            ));
        }
    }

    // Check if sink is reachable
    if dist[sink] == f64::INFINITY {
        return Ok(None);
    }

    // Reconstruct path by walking parent arcs back from the sink
    let mut path = Vec::new();
    let mut current = sink;
    while current != source {
        let a = parent_arc[current].expect("reachable node has a parent arc");
        path.push(a);
        current = arcs[arcs[a].rev].to;
    }

    path.reverse();
    Ok(Some((path, dist[sink])))
}

/// Helper function to get edge capacity
pub(crate) fn get_edge_capacity(
    edge: &crate::graph::Edge,
    capacity: Option<&str>,
    is_weighted: bool,
) -> PyResult<f64> {
    // Helper to extract float from edge attributes
    let get_attr = |attr_name: &str| -> Option<f64> {
        if let Some(attrs) = &edge.attributes {
            Python::with_gil(|py| {
                if let Ok(Some(v)) = attrs.bind(py).get_item(attr_name) {
                    v.extract::<f64>().ok()
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

/// Run Edmonds-Karp from `source` to `sink`.
///
/// Returns the maximum flow value and the flow on every non-self-loop edge, as
/// (edge_id, flow) pairs. The residual network has one forward arc per edge, paired with
/// its own reverse arc, so parallel edges keep separate capacities and flows and a real
/// v -> u edge never shares capacity with the cancellation arc of u -> v. On directed
/// graphs the reverse arc starts empty (it only cancels flow); on undirected graphs it
/// starts with the full capacity, so the edge can carry flow either way, and the edge's
/// reported flow is the magnitude of its net flow.
///
/// Also returns the source side of a minimum cut: the nodes still reachable from `source`
/// in the final residual network.
pub(crate) fn edmonds_karp(
    graph: &Graph,
    source: u64,
    sink: u64,
    capacity: Option<&str>,
) -> PyResult<(f64, Vec<(u64, f64)>, Vec<u64>)> {
    let edges = graph.get_edges_internal();
    let is_directed = graph.is_directed();

    let mut node_index: FxHashMap<u64, usize> = FxHashMap::default();
    for node_id in graph.get_node_ids() {
        let next = node_index.len();
        node_index.insert(node_id, next);
    }

    let mut arcs: Vec<ResidualArc> = Vec::with_capacity(edges.len() * 2);
    let mut out_arcs: Vec<Vec<usize>> = vec![Vec::new(); node_index.len()];
    // (edge id, index of its forward arc, original capacity)
    let mut edge_arcs: Vec<(u64, usize, f64)> = Vec::with_capacity(edges.len());

    for edge in edges {
        // Skip self-loops
        if edge.source == edge.target {
            continue;
        }

        let edge_cap = get_edge_capacity(edge, capacity, graph.is_weighted())?;
        let (u, v) = (node_index[&edge.source], node_index[&edge.target]);
        let reverse_cap = if is_directed { 0.0 } else { edge_cap };

        let forward = arcs.len();
        arcs.push(ResidualArc { to: v, residual: edge_cap, cost: 0.0, rev: forward + 1 });
        arcs.push(ResidualArc { to: u, residual: reverse_cap, cost: 0.0, rev: forward });
        out_arcs[u].push(forward);
        out_arcs[v].push(forward + 1);
        edge_arcs.push((edge.id, forward, edge_cap));
    }

    let (source_idx, sink_idx) = (node_index[&source], node_index[&sink]);
    let mut total_flow = 0.0;

    // Augment along shortest (fewest-arc) paths until the sink is unreachable
    while let Some(path) = bfs_augmenting_path(source_idx, sink_idx, &arcs, &out_arcs) {
        // Find minimum residual capacity along the path (bottleneck)
        let path_flow = path
            .iter()
            .map(|&a| arcs[a].residual)
            .fold(f64::INFINITY, f64::min);

        // Push flow: shrink each arc's residual and grow its paired reverse arc
        for &a in &path {
            arcs[a].residual -= path_flow;
            let rev = arcs[a].rev;
            arcs[rev].residual += path_flow;
        }

        total_flow += path_flow;
    }

    let edge_flows = edge_arcs
        .into_iter()
        .map(|(edge_id, forward, edge_cap)| (edge_id, (edge_cap - arcs[forward].residual).abs()))
        .collect();

    // Source side of the minimum cut: everything reachable through residual capacity
    let mut reach = vec![false; out_arcs.len()];
    let mut queue: VecDeque<usize> = VecDeque::from([source_idx]);
    reach[source_idx] = true;
    while let Some(v) = queue.pop_front() {
        for &a in &out_arcs[v] {
            let arc = &arcs[a];
            if arc.residual > 1e-10 && !reach[arc.to] {
                reach[arc.to] = true;
                queue.push_back(arc.to);
            }
        }
    }
    let mut source_side: Vec<u64> = node_index
        .iter()
        .filter(|(_, &i)| reach[i])
        .map(|(&id, _)| id)
        .collect();
    source_side.sort_unstable();

    Ok((total_flow, edge_flows, source_side))
}

/// BFS for a shortest augmenting path over arcs with positive residual capacity.
/// Returns the arc indices from source to sink, or None if the sink is unreachable.
fn bfs_augmenting_path(
    source: usize,
    sink: usize,
    arcs: &[ResidualArc],
    out_arcs: &[Vec<usize>],
) -> Option<Vec<usize>> {
    // Arc used to reach each node (source is marked visited with no arc)
    let mut parent_arc: Vec<Option<usize>> = vec![None; out_arcs.len()];
    let mut visited = vec![false; out_arcs.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();

    visited[source] = true;
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        if current == sink {
            // Reconstruct path by walking parent arcs back from the sink
            let mut path = Vec::new();
            let mut node = sink;
            while node != source {
                let a = parent_arc[node].expect("visited node has a parent arc");
                path.push(a);
                node = arcs[arcs[a].rev].to;
            }
            path.reverse();
            return Some(path);
        }

        for &a in &out_arcs[current] {
            let arc = &arcs[a];
            if arc.residual > 1e-10 && !visited[arc.to] {
                visited[arc.to] = true;
                parent_arc[arc.to] = Some(a);
                queue.push_back(arc.to);
            }
        }
    }

    None // No path found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build arcs/out_arcs for a directed network given (u, v, capacity) triples
    fn network(n: usize, edges: &[(usize, usize, f64)]) -> (Vec<ResidualArc>, Vec<Vec<usize>>) {
        let mut arcs = Vec::new();
        let mut out_arcs = vec![Vec::new(); n];
        for &(u, v, cap) in edges {
            let forward = arcs.len();
            arcs.push(ResidualArc { to: v, residual: cap, cost: 0.0, rev: forward + 1 });
            arcs.push(ResidualArc { to: u, residual: 0.0, cost: 0.0, rev: forward });
            out_arcs[u].push(forward);
            out_arcs[v].push(forward + 1);
        }
        (arcs, out_arcs)
    }

    #[test]
    fn test_bfs_augmenting_path() {
        let (arcs, out_arcs) = network(3, &[(0, 1, 10.0), (1, 2, 5.0), (0, 2, 3.0)]);

        // The direct arc 0 -> 2 is the shortest path
        let path = bfs_augmenting_path(0, 2, &arcs, &out_arcs).unwrap();
        assert_eq!(path, vec![4]);
    }

    #[test]
    fn test_no_path() {
        let (arcs, out_arcs) = network(3, &[(0, 1, 10.0)]);
        assert!(bfs_augmenting_path(0, 2, &arcs, &out_arcs).is_none());
    }

    #[test]
    fn test_saturated_arc_is_skipped() {
        let (arcs, out_arcs) = network(3, &[(0, 2, 0.0), (0, 1, 1.0), (1, 2, 1.0)]);

        let path = bfs_augmenting_path(0, 2, &arcs, &out_arcs).unwrap();
        assert_eq!(path, vec![2, 4]);
    }
}
