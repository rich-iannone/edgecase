//! Node vitality computation - measuring the impact of removing a node
//!
//! Node vitality measures the importance of a node by quantifying how much
//! the graph's total distance increases when the node is removed.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute vitality for a single node
///
/// Node vitality is defined as:
/// vitality(v) = sum_distances(G) - sum_distances(G - v)
///
/// This measures how much removing the node increases total path lengths.
/// Higher values indicate more important nodes whose removal would significantly
/// impact graph connectivity.
///
/// # Arguments
///
/// * `graph` - The graph to analyze
/// * `node` - The node ID to compute vitality for
///
/// # Returns
///
/// The vitality score for the node (change in sum of distances)
///
/// # Complexity
///
/// O(V * E) where V is number of vertices and E is number of edges
#[pyfunction]
#[pyo3(signature = (graph, node, weighted=false))]
pub fn node_vitality(graph: &Graph, node: u64, weighted: bool) -> PyResult<f64> {
    // Verify node exists
    let node_ids = graph.get_node_ids();
    if !node_ids.contains(&node) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} does not exist in graph",
            node
        )));
    }

    // If graph has only one node, vitality is 0
    if node_ids.len() == 1 {
        return Ok(0.0);
    }

    if weighted {
        let adj = weighted_adjacency(graph)?;
        return Ok(weighted_sum_of_distances(&adj, &node_ids, None)
            - weighted_sum_of_distances(&adj, &node_ids, Some(node)));
    }

    // Hop adjacency built once (it used to be rebuilt from the whole edge list at every BFS
    // step); sum of distances with and without the node
    let adj = hop_adjacency(graph);
    Ok(hop_sum_of_distances(&adj, &node_ids, None)
        - hop_sum_of_distances(&adj, &node_ids, Some(node)))
}

/// Out-neighbors (or neighbors, if undirected) of every node, one entry per edge; self-loops
/// skipped since they never shorten a path
fn hop_adjacency(graph: &Graph) -> FxHashMap<u64, Vec<u64>> {
    let mut adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for edge in graph.get_edges_internal() {
        if edge.source == edge.target {
            continue;
        }
        adj.entry(edge.source).or_default().push(edge.target);
        if !graph.is_directed() {
            adj.entry(edge.target).or_default().push(edge.source);
        }
    }
    adj
}

/// Sum of all finite hop distances over ordered pairs, with `exclude` (if any) removed
fn hop_sum_of_distances(
    adj: &FxHashMap<u64, Vec<u64>>,
    node_ids: &[u64],
    exclude: Option<u64>,
) -> f64 {
    let mut total = 0.0;
    for &source in node_ids {
        if Some(source) == exclude {
            continue;
        }
        let mut dist: FxHashMap<u64, usize> = FxHashMap::default();
        dist.insert(source, 0);
        let mut queue = VecDeque::from([source]);
        while let Some(v) = queue.pop_front() {
            let d = dist[&v];
            for &w in adj.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
                if Some(w) != exclude && !dist.contains_key(&w) {
                    dist.insert(w, d + 1);
                    queue.push_back(w);
                }
            }
        }
        total += dist.values().sum::<usize>() as f64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_vitality_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node("A");
        let node_id = graph.id_from_label("A").unwrap();

        let vitality = node_vitality(&graph, node_id, false).unwrap();
        assert_eq!(vitality, 0.0);
    }

    #[test]
    fn test_node_vitality_path() {
        let mut graph = Graph::new_rust(false, false);
        let nodes = graph.add_test_path(3);

        // Middle node is most vital (connects the path)
        let middle_vitality = node_vitality(&graph, nodes[1], false).unwrap();
        let end_vitality = node_vitality(&graph, nodes[0], false).unwrap();

        // Removing middle node disconnects graph (infinite distances)
        // Removing end node just removes 2 paths
        assert!(middle_vitality.is_infinite() || middle_vitality > end_vitality);
    }

    #[test]
    fn test_node_vitality_star() {
        let mut graph = Graph::new_rust(false, false);
        let nodes = graph.add_test_star(4);

        // First node is hub in star graph
        let hub_vitality = node_vitality(&graph, nodes[0], false).unwrap();
        let leaf_vitality = node_vitality(&graph, nodes[1], false).unwrap();

        // Hub is most vital (removal disconnects all leaves)
        assert!(hub_vitality.is_infinite() || hub_vitality > leaf_vitality);
    }
}

/// Closeness vitality of every node with distances summed over edge weights.
///
/// Same definition as the hop-count version, `sum_distances(G) - sum_distances(G - v)` over
/// ordered pairs with a finite distance, but each distance is a Dijkstra path cost.
#[pyfunction]
pub fn closeness_vitality_weighted(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let adj = weighted_adjacency(graph)?;
    let total = weighted_sum_of_distances(&adj, &node_ids, None);
    Ok(node_ids
        .par_iter()
        .map(|&v| {
            (
                v,
                total - weighted_sum_of_distances(&adj, &node_ids, Some(v)),
            )
        })
        .collect())
}

/// `(neighbor, weight)` lists from the edge weight field (`1.0` when unset), following edge
/// direction on directed graphs. Self-loops are skipped; parallel edges are all kept, and
/// Dijkstra uses the cheapest. Negative weights raise `ValueError`.
fn weighted_adjacency(graph: &Graph) -> PyResult<FxHashMap<u64, Vec<(u64, f64)>>> {
    let mut adj: FxHashMap<u64, Vec<(u64, f64)>> = FxHashMap::default();
    for edge in graph.get_edges_internal() {
        let w = edge.weight.unwrap_or(1.0);
        if w < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Weighted vitality requires non-negative edge weights, but edge {} ({} -> {}) \
                 has weight {}",
                edge.id, edge.source, edge.target, w
            )));
        }
        if edge.source == edge.target {
            continue;
        }
        adj.entry(edge.source).or_default().push((edge.target, w));
        if !graph.is_directed() {
            adj.entry(edge.target).or_default().push((edge.source, w));
        }
    }
    Ok(adj)
}

/// Sum of all finite weighted shortest-path distances over ordered pairs, with `exclude` (if
/// any) removed from the graph.
fn weighted_sum_of_distances(
    adj: &FxHashMap<u64, Vec<(u64, f64)>>,
    node_ids: &[u64],
    exclude: Option<u64>,
) -> f64 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct Cost(f64);
    impl Eq for Cost {}
    impl Ord for Cost {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
    impl PartialOrd for Cost {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut total = 0.0;
    for &source in node_ids {
        if Some(source) == exclude {
            continue;
        }
        let mut dist: FxHashMap<u64, f64> = FxHashMap::default();
        let mut heap = BinaryHeap::new();
        dist.insert(source, 0.0);
        heap.push(Reverse((Cost(0.0), source)));
        while let Some(Reverse((Cost(d), v))) = heap.pop() {
            if d > dist[&v] {
                continue;
            }
            for &(w, weight) in adj.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
                if Some(w) == exclude {
                    continue;
                }
                let nd = d + weight;
                if dist.get(&w).is_none_or(|&old| nd < old) {
                    dist.insert(w, nd);
                    heap.push(Reverse((Cost(nd), w)));
                }
            }
        }
        total += dist
            .iter()
            .filter(|(&t, _)| t != source)
            .map(|(_, &d)| d)
            .sum::<f64>();
    }
    total
}
