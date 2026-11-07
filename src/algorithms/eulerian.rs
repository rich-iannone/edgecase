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
            .or_default()
            .push((edge.target, edge.id));

        if !graph.is_directed() {
            // For undirected graphs, add reverse edge
            adj.entry(edge.target)
                .or_default()
                .push((edge.source, edge.id));
        }
    }

    // Start from the first node (in storage order) that has an edge to leave by. Taking the
    // first node regardless of degree returned just [node] whenever it was isolated; isolated
    // nodes don't affect the existence check, so they can be present here. In a graph that
    // passes that check every node with edges has an outgoing one (in = out when directed).
    // With no edges at all, any node gives the one-node circuit.
    let start_node = graph
        .get_nodes()
        .keys()
        .copied()
        .find(|node| adj.get(node).is_some_and(|out| !out.is_empty()))
        .unwrap_or_else(|| *graph.get_nodes().keys().next().unwrap());

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

// ---------------------------------------------------------------------------------------------
// Eulerian paths and eulerization
// ---------------------------------------------------------------------------------------------

/// Edge-labelled adjacency (sorted node IDs): `adj[i] = [(j, edge_id)]`, both directions for
/// undirected graphs, plus out/in degrees (a self-loop adds one to each, i.e. 2 to the degree).
struct EulerAdj {
    ids: Vec<u64>,
    idx: FxHashMap<u64, usize>,
    adj: Vec<Vec<(usize, u64)>>,
    outd: Vec<i64>,
    ind: Vec<i64>,
    m: usize,
}

fn euler_adjacency(graph: &Graph) -> EulerAdj {
    let mut ids: Vec<u64> = graph.get_nodes().keys().copied().collect();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let n = ids.len();
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n];
    let mut outd = vec![0i64; n];
    let mut ind = vec![0i64; n];
    let directed = graph.is_directed();
    let edges = graph.get_edges_internal();
    for e in edges {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        adj[s].push((t, e.id));
        outd[s] += 1;
        ind[t] += 1;
        // an undirected self-loop is listed once (it is traversed once)
        if !directed && s != t {
            adj[t].push((s, e.id));
        }
    }
    EulerAdj {
        ids,
        idx,
        adj,
        outd,
        ind,
        m: edges.len(),
    }
}

/// Hierholzer's algorithm from `start`, returning (nodes, edge IDs) of a trail that uses every
/// edge reachable from `start` exactly once.
fn hierholzer_trail(ea: &EulerAdj, start: usize) -> (Vec<u64>, Vec<u64>) {
    let mut ptr = vec![0usize; ea.ids.len()];
    let mut used: FxHashSet<u64> = FxHashSet::default();
    // stack of (node, edge used to arrive)
    let mut stack: Vec<(usize, Option<u64>)> = vec![(start, None)];
    let mut nodes_rev: Vec<u64> = Vec::new();
    let mut edges_rev: Vec<u64> = Vec::new();
    while let Some(&(v, arrived)) = stack.last() {
        let mut advanced = false;
        while ptr[v] < ea.adj[v].len() {
            let (w, eid) = ea.adj[v][ptr[v]];
            ptr[v] += 1;
            if used.insert(eid) {
                stack.push((w, Some(eid)));
                advanced = true;
                break;
            }
        }
        if !advanced {
            stack.pop();
            nodes_rev.push(ea.ids[v]);
            if let Some(e) = arrived {
                edges_rev.push(e);
            }
        }
    }
    nodes_rev.reverse();
    edges_rev.reverse();
    (nodes_rev, edges_rev)
}

/// Where an Eulerian path must start, or None if the graph has none.
///
/// Isolated nodes are ignored, as in `is_eulerian()`. If `source` is given it must be a valid
/// start: for a circuit any node with edges, otherwise the odd-degree node (undirected) or the
/// node with one more out-edge than in-edges (directed).
fn euler_start(graph: &Graph, ea: &EulerAdj, source: Option<u64>) -> PyResult<Option<usize>> {
    if !is_connected_for_euler(graph) {
        return Ok(None);
    }
    let n = ea.ids.len();
    let directed = graph.is_directed();
    let src = match source {
        Some(s) => Some(*ea.idx.get(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Node {} not found in graph", s))
        })?),
        None => None,
    };
    let mut starts: Vec<usize> = Vec::new();
    if directed {
        let mut ends = 0;
        for v in 0..n {
            let d = ea.outd[v] - ea.ind[v];
            if d == 1 {
                starts.push(v);
            } else if d == -1 {
                ends += 1;
            } else if d != 0 {
                return Ok(None);
            }
        }
        if starts.len() > 1 || ends != starts.len() {
            return Ok(None);
        }
    } else {
        for v in 0..n {
            if (ea.outd[v] + ea.ind[v]) % 2 == 1 {
                starts.push(v);
            }
        }
        if starts.len() != 0 && starts.len() != 2 {
            return Ok(None);
        }
    }
    let has_edges = |v: usize| ea.outd[v] + ea.ind[v] > 0;
    if starts.is_empty() {
        // Eulerian circuit: start anywhere with edges
        return Ok(match src {
            Some(s) if has_edges(s) || ea.m == 0 => Some(s),
            Some(_) => None,
            None => (0..n)
                .find(|&v| has_edges(v))
                .or(if n > 0 { Some(0) } else { None }),
        });
    }
    Ok(match src {
        Some(s) if starts.contains(&s) => Some(s),
        Some(_) => None,
        None => Some(starts[0]),
    })
}

/// An Eulerian path (a trail using every edge exactly once), as `(node_ids, edge_ids)`, or None
/// if the graph has none. On an Eulerian graph this is a circuit. Isolated nodes are ignored.
#[pyfunction]
#[pyo3(signature = (graph, source=None))]
pub fn eulerian_path(graph: &Graph, source: Option<u64>) -> PyResult<Option<(Vec<u64>, Vec<u64>)>> {
    let ea = euler_adjacency(graph);
    let Some(start) = euler_start(graph, &ea, source)? else {
        return Ok(None);
    };
    if ea.ids.is_empty() {
        return Ok(Some((Vec::new(), Vec::new())));
    }
    let (nodes, edges) = hierholzer_trail(&ea, start);
    if edges.len() != ea.m {
        return Ok(None);
    }
    Ok(Some((nodes, edges)))
}

/// Whether the graph has an Eulerian path (starting at `source`, if given).
#[pyfunction]
#[pyo3(signature = (graph, source=None))]
pub fn has_eulerian_path(graph: &Graph, source: Option<u64>) -> PyResult<bool> {
    let ea = euler_adjacency(graph);
    Ok(euler_start(graph, &ea, source)?.is_some())
}

/// Minimum-cost perfect matching of `k` items (k even) given a cost matrix.
///
/// Exact bitmask DP for k <= 20; beyond that, greedy pairing improved by 2-opt swaps (an
/// approximation). Returns pairs of indices.
fn min_cost_pairing(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let k = cost.len();
    if k == 0 {
        return Vec::new();
    }
    if k <= 20 {
        let full = (1usize << k) - 1;
        let mut dp = vec![f64::INFINITY; 1 << k];
        let mut choice = vec![(0usize, 0usize); 1 << k];
        dp[0] = 0.0;
        for mask in 0..=full {
            if dp[mask].is_infinite() {
                continue;
            }
            // lowest unpaired item is paired next
            let Some(i) = (0..k).find(|&i| mask & (1 << i) == 0) else {
                continue;
            };
            for j in (i + 1)..k {
                if mask & (1 << j) == 0 {
                    let nm = mask | (1 << i) | (1 << j);
                    let c = dp[mask] + cost[i][j];
                    if c < dp[nm] {
                        dp[nm] = c;
                        choice[nm] = (i, j);
                    }
                }
            }
        }
        let mut pairs = Vec::new();
        let mut mask = full;
        while mask != 0 {
            let (i, j) = choice[mask];
            pairs.push((i, j));
            mask &= !((1 << i) | (1 << j));
        }
        pairs.sort_unstable();
        return pairs;
    }
    // Greedy: cheapest available pair first
    let mut cand: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            cand.push((cost[i][j], i, j));
        }
    }
    cand.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    let mut taken = vec![false; k];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (_, i, j) in cand {
        if !taken[i] && !taken[j] {
            taken[i] = true;
            taken[j] = true;
            pairs.push((i, j));
        }
    }
    // 2-opt: re-pair two pairs if that lowers the total
    let mut improved = true;
    while improved {
        improved = false;
        for a in 0..pairs.len() {
            for b in (a + 1)..pairs.len() {
                let (p, q) = pairs[a];
                let (r, s) = pairs[b];
                let now = cost[p][q] + cost[r][s];
                let alt1 = cost[p][r] + cost[q][s];
                let alt2 = cost[p][s] + cost[q][r];
                if alt1 + 1e-12 < now && alt1 <= alt2 {
                    pairs[a] = (p, r);
                    pairs[b] = (q, s);
                    improved = true;
                } else if alt2 + 1e-12 < now {
                    pairs[a] = (p, s);
                    pairs[b] = (q, r);
                    improved = true;
                }
            }
        }
    }
    pairs
}

/// Eulerize an undirected graph: duplicate the fewest (or lightest) edges so that every node has
/// even degree, pairing odd-degree nodes by a minimum-cost matching of their shortest-path
/// distances and duplicating each pair's shortest path (the Chinese postman construction).
///
/// Returns a new graph with the same nodes (IDs kept) and edges, plus the duplicates, which get
/// the original edge's weight and a copy of its attributes with `eulerize_added = True`. Uses
/// hop counts, or weights when `weighted`. Raises unless the graph is undirected and connected
/// (ignoring isolated nodes).
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn eulerize(py: Python, graph: &Graph, weighted: bool) -> PyResult<(Graph, Vec<u64>)> {
    use crate::graph::{Edge, Node};
    use std::collections::BinaryHeap;
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eulerize() requires an undirected graph",
        ));
    }
    if !is_connected_for_euler(graph) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eulerize() requires a connected graph (isolated nodes are ignored)",
        ));
    }
    let ea = euler_adjacency(graph);
    let n = ea.ids.len();
    let weight_of: FxHashMap<u64, f64> = graph
        .get_edges_internal()
        .iter()
        .map(|e| {
            (
                e.id,
                if weighted {
                    e.weight.unwrap_or(1.0)
                } else {
                    1.0
                },
            )
        })
        .collect();
    if weight_of.values().any(|&w| w < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eulerize() requires non-negative edge weights",
        ));
    }
    // An infinite path cost leaves the matching (and its traceback) undefined
    if weight_of.values().any(|&w| w.is_infinite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eulerize() requires finite edge weights",
        ));
    }
    let odd: Vec<usize> = (0..n)
        .filter(|&v| (ea.outd[v] + ea.ind[v]) % 2 == 1)
        .collect();

    // Shortest paths from each odd node (Dijkstra; ties keep the first predecessor found)
    let mut dist_rows: Vec<Vec<f64>> = Vec::with_capacity(odd.len());
    let mut pred_rows: Vec<Vec<Option<(usize, u64)>>> = Vec::with_capacity(odd.len());
    for &s in &odd {
        let mut dist = vec![f64::INFINITY; n];
        let mut pred: Vec<Option<(usize, u64)>> = vec![None; n];
        dist[s] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push((std::cmp::Reverse(OrdF(0.0)), s));
        while let Some((std::cmp::Reverse(OrdF(d)), v)) = heap.pop() {
            if d > dist[v] {
                continue;
            }
            for &(w, eid) in &ea.adj[v] {
                let nd = d + weight_of[&eid];
                if nd < dist[w] {
                    dist[w] = nd;
                    pred[w] = Some((v, eid));
                    heap.push((std::cmp::Reverse(OrdF(nd)), w));
                }
            }
        }
        dist_rows.push(dist);
        pred_rows.push(pred);
    }
    let cost: Vec<Vec<f64>> = (0..odd.len())
        .map(|a| (0..odd.len()).map(|b| dist_rows[a][odd[b]]).collect())
        .collect();
    let mut duplicates: Vec<u64> = Vec::new();
    for (a, b) in min_cost_pairing(&cost) {
        let mut v = odd[b];
        while let Some((p, eid)) = pred_rows[a][v] {
            duplicates.push(eid);
            v = p;
        }
    }

    // Build the result: same nodes (IDs kept) and edges, then the duplicates
    let mut result = Graph::new_rust(false, graph.is_weighted());
    let copy_attrs =
        |attrs: &Option<Py<pyo3::types::PyDict>>| -> PyResult<Option<Py<pyo3::types::PyDict>>> {
            Ok(match attrs {
                Some(d) => Some(d.bind(py).copy()?.unbind()),
                None => None,
            })
        };
    let nodes = graph.get_nodes();
    for &id in &ea.ids {
        let node = &nodes[&id];
        result.add_node_rust(Node {
            id,
            label: node.label.clone(),
            attributes: copy_attrs(&node.attributes)?,
        })?;
    }
    let by_id: FxHashMap<u64, &Edge> = graph
        .get_edges_internal()
        .iter()
        .map(|e| (e.id, e))
        .collect();
    for e in graph.get_edges_internal() {
        // keep original edge IDs; duplicates get fresh IDs after them. Originals are marked
        // `eulerize_added = False` (not left null) so filters on either value work.
        let attrs = match copy_attrs(&e.attributes)? {
            Some(d) => d,
            None => pyo3::types::PyDict::new(py).unbind(),
        };
        attrs.bind(py).set_item("eulerize_added", false)?;
        result.add_edge_rust(Edge {
            id: e.id,
            source: e.source,
            target: e.target,
            weight: e.weight,
            attributes: Some(attrs),
        })?;
    }
    let mut added_ids = Vec::with_capacity(duplicates.len());
    for eid in &duplicates {
        let e = by_id[eid];
        let attrs = match copy_attrs(&e.attributes)? {
            Some(d) => d,
            None => pyo3::types::PyDict::new(py).unbind(),
        };
        attrs.bind(py).set_item("eulerize_added", true)?;
        let new_id = result.add_edge_rust(Edge {
            id: 0,
            source: e.source,
            target: e.target,
            weight: e.weight,
            attributes: Some(attrs),
        })?;
        added_ids.push(new_id);
    }
    Ok((result, added_ids))
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct OrdF(f64);
impl Eq for OrdF {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
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
