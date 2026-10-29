//! Rich Club Coefficient algorithm
//!
//! The rich-club coefficient measures the connectivity among high-degree nodes.
//! For each degree k, it computes the ratio of actual to potential edges
//! among nodes with degree > k. Optionally, each value is divided by the same
//! coefficient averaged over degree-preserving randomizations of the graph.

use crate::graph::Graph;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};

/// Compute the rich-club coefficient for all degree values.
///
/// The rich-club coefficient φ(k) is defined as:
/// φ(k) = 2 * E_k / (N_k * (N_k - 1))
///
/// where:
/// - N_k is the number of nodes with degree > k
/// - E_k is the number of edges among those nodes
///
/// With `normalized`, φ(k) is divided by the average φ(k) of `n_samples`
/// random graphs, each made from a copy of the graph by `q * m` successful
/// double-edge swaps (at most `10 * q * m` attempts), which keep every degree
/// (in- and out-degree for directed graphs) and never create self-loops or
/// parallel edges. This follows NetworkX's `rich_club_coefficient()`.
///
/// # Arguments
/// * `graph` - The input graph (should be undirected without self-loops)
/// * `normalized` - Divide by the coefficient of degree-preserving random graphs
/// * `q` - Swaps per edge used to randomize each reference graph
/// * `seed` - Optional random seed for the swaps
/// * `n_samples` - Number of random graphs to average
///
/// # Returns
/// A HashMap mapping degree k to rich-club coefficient φ(k)
///
/// # Notes
/// - This algorithm ignores edge weights
/// - For directed graphs, uses total degree
/// - Self-loops are ignored in edge counting (and rejected when normalizing)
#[pyfunction]
#[pyo3(signature = (graph, normalized=false, q=100, seed=None, n_samples=1))]
pub fn rich_club_coefficient(
    graph: &Graph,
    normalized: bool,
    q: usize,
    seed: Option<u64>,
    n_samples: usize,
) -> PyResult<FxHashMap<usize, f64>> {
    let rc = raw_rich_club(graph);
    if !normalized {
        return Ok(rc);
    }
    if n_samples < 1 {
        return Err(PyValueError::new_err(
            "n_samples must be >= 1 if normalized=True",
        ));
    }

    // Dense indices, degrees and the edge list of the graph
    let index: FxHashMap<u64, usize> = graph
        .get_nodes()
        .keys()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let directed = graph.is_directed();
    let key = |a: usize, b: usize| if directed || a < b { (a, b) } else { (b, a) };
    let mut degree = vec![0usize; index.len()];
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(graph.get_edges_internal().len());
    let mut present: FxHashSet<(usize, usize)> = FxHashSet::default();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            return Err(PyValueError::new_err(
                "The normalized rich-club coefficient is not defined for graphs with self-loops",
            ));
        }
        let (a, b) = (index[&e.source], index[&e.target]);
        if !present.insert(key(a, b)) {
            return Err(PyValueError::new_err(
                "The normalized rich-club coefficient is not defined for graphs with parallel edges",
            ));
        }
        degree[a] += 1;
        degree[b] += 1;
        edges.push((a, b));
    }
    if index.len() < 4 || edges.len() < 2 {
        return Err(PyValueError::new_err(
            "The normalized rich-club coefficient needs at least four nodes and two edges",
        ));
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let m = edges.len();
    let n_swaps = q.saturating_mul(m);
    let max_tries = n_swaps.saturating_mul(10);

    let mut averages: FxHashMap<usize, f64> = rc.keys().map(|&k| (k, 0.0)).collect();
    for _ in 0..n_samples {
        let mut r_edges = edges.clone();
        let mut r_present = present.clone();
        let mut done = 0;
        let mut tries = 0;
        while done < n_swaps && tries < max_tries {
            tries += 1;
            let i = rng.gen_range(0..m);
            let j = rng.gen_range(0..m);
            if i == j {
                continue;
            }
            let (a, b) = r_edges[i];
            // an undirected edge can be read either way round
            let (c, d) = if !directed && rng.gen::<bool>() {
                (r_edges[j].1, r_edges[j].0)
            } else {
                r_edges[j]
            };
            // a-b, c-d becomes a-d, c-b
            if a == d || c == b {
                continue;
            }
            let (k1, k2) = (key(a, d), key(c, b));
            if k1 == k2 || r_present.contains(&k1) || r_present.contains(&k2) {
                continue;
            }
            r_present.remove(&key(a, b));
            r_present.remove(&key(c, d));
            r_present.insert(k1);
            r_present.insert(k2);
            r_edges[i] = (a, d);
            r_edges[j] = (c, b);
            done += 1;
        }
        for (k, coef) in edge_list_rich_club(&degree, &r_edges, directed) {
            if let Some(avg) = averages.get_mut(&k) {
                *avg += coef / n_samples as f64;
            }
        }
    }

    let mut keys: Vec<usize> = rc.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        if averages[&k] == 0.0 {
            return Err(PyValueError::new_err(format!(
                "Rich-club coefficient normalization failed at degree {k}. The average \
                 rich-club coefficient of {n_samples} random graphs is zero. Try increasing \
                 n_samples."
            )));
        }
    }
    Ok(rc
        .into_iter()
        .map(|(k, coef)| (k, coef / averages[&k]))
        .collect())
}

/// Rich-club coefficients of a simple graph given as degrees and an edge list
/// over dense indices. An edge lies inside the club of degree > k exactly when
/// both its ends have degree > k, that is for every k below the smaller of the
/// two degrees, so one pass over the edges gives every E_k.
fn edge_list_rich_club(
    degree: &[usize],
    edges: &[(usize, usize)],
    directed: bool,
) -> FxHashMap<usize, f64> {
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    // nodes_above[k] = N_k, edges_ending[d] = edges whose smaller end degree is d
    let mut nodes_above = vec![0usize; max_degree + 1];
    for &d in degree {
        for slot in nodes_above.iter_mut().take(d) {
            *slot += 1;
        }
    }
    let mut edges_ending = vec![0usize; max_degree + 1];
    for &(a, b) in edges {
        edges_ending[degree[a].min(degree[b])] += 1;
    }
    let mut result = FxHashMap::default();
    let mut e_k: usize = edges_ending.iter().sum();
    for k in 0..max_degree {
        e_k -= edges_ending[k];
        let n_k = nodes_above[k];
        if n_k < 2 {
            continue;
        }
        let potential = (n_k * (n_k - 1)) as f64;
        let factor = if directed { 1.0 } else { 2.0 };
        result.insert(k, factor * e_k as f64 / potential);
    }
    result
}

/// The unnormalized coefficient, as computed before normalization existed:
/// total degree for directed graphs, self-loops skipped when counting edges.
fn raw_rich_club(graph: &Graph) -> FxHashMap<usize, f64> {
    let mut result: FxHashMap<usize, f64> = FxHashMap::default();

    // Compute degree for each node
    // For directed graphs, use total degree (in + out)
    let nodes: Vec<u64> = graph.get_nodes().keys().copied().collect();
    let mut degrees: FxHashMap<u64, usize> = FxHashMap::default();

    for &node_id in &nodes {
        let degree = if graph.is_directed() {
            // Total degree = in-degree + out-degree
            graph.get_in_degree(node_id) + graph.get_out_degree(node_id)
        } else {
            graph.degree_rust(node_id)
        };
        degrees.insert(node_id, degree);
    }

    if nodes.is_empty() {
        return result;
    }

    // Find the maximum degree
    let max_degree = *degrees.values().max().unwrap_or(&0);

    if max_degree == 0 {
        return result;
    }

    // For each k from 0 to max_degree - 1, compute the rich club coefficient
    for k in 0..max_degree {
        // Find nodes with degree > k
        let rich_nodes: FxHashSet<u64> = degrees
            .iter()
            .filter(|(_, &deg)| deg > k)
            .map(|(&node_id, _)| node_id)
            .collect();

        let n_k = rich_nodes.len();

        // Need at least 2 nodes to have edges between them
        if n_k < 2 {
            continue;
        }

        // Count edges among rich nodes
        let mut e_k: usize = 0;
        for edge in graph.get_edges_internal() {
            // Skip self-loops
            if edge.source == edge.target {
                continue;
            }

            if rich_nodes.contains(&edge.source) && rich_nodes.contains(&edge.target) {
                e_k += 1;
            }
        }

        // For undirected graphs, each edge is counted once in get_edges_internal
        // For directed graphs, we count each directed edge
        // The formula uses 2*E_k in the numerator for undirected graphs

        // Potential edges = N_k * (N_k - 1) for directed, N_k * (N_k - 1) / 2 for undirected
        // So φ(k) = E_k / (N_k * (N_k - 1) / 2) = 2 * E_k / (N_k * (N_k - 1)) for undirected
        // For directed: φ(k) = E_k / (N_k * (N_k - 1))

        let potential = n_k * (n_k - 1);

        let phi_k = if graph.is_directed() {
            e_k as f64 / potential as f64
        } else {
            // For undirected, each edge appears once, so multiply by 2
            (2.0 * e_k as f64) / potential as f64
        };

        result.insert(k, phi_k);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn create_test_graph() -> Graph {
        // Create a simple undirected graph:
        // 1 -- 2 -- 3
        // |    |
        // 4 -- 5
        // Node degrees: 1->2, 2->3, 3->1, 4->2, 5->2
        let mut graph = Graph::new_rust(false, false);

        for i in 1..=5 {
            graph
                .add_node_rust(Node {
                    id: i,
                    label: Some(i.to_string()),
                    attributes: None,
                })
                .unwrap();
        }

        // Edges
        let edges = vec![(1, 2), (2, 3), (1, 4), (4, 5), (2, 5)];
        for (s, t) in edges {
            graph
                .add_edge_rust(Edge {
                    id: 0,
                    source: s,
                    target: t,
                    weight: Some(1.0),
                    attributes: None,
                })
                .unwrap();
        }

        graph
    }

    #[test]
    fn test_rich_club_basic() {
        let graph = create_test_graph();
        let rc = rich_club_coefficient(&graph, false, 100, None, 1).unwrap();

        // With degrees [2, 3, 1, 2, 2]:
        // k=0: all 5 nodes (degree > 0), potential = 5*4 = 20, edges among them = 5
        //      φ(0) = 2*5/20 = 0.5
        // k=1: nodes with degree > 1 are [1, 2, 4, 5] (4 nodes), edges among them = 4 (1-2, 1-4, 4-5, 2-5)
        //      potential = 4*3 = 12, φ(1) = 2*4/12 = 0.667
        // k=2: nodes with degree > 2 are [2] (1 node), can't compute

        assert!(rc.contains_key(&0));
        assert!(rc.contains_key(&1));
        assert!(!rc.contains_key(&2)); // Only 1 node with degree > 2
    }

    #[test]
    fn test_rich_club_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let rc = rich_club_coefficient(&graph, false, 100, None, 1).unwrap();
        assert!(rc.is_empty());
    }

    #[test]
    fn test_rich_club_complete_graph() {
        // K4: 4 nodes, all connected
        // Each node has degree 3
        let mut graph = Graph::new_rust(false, false);

        for i in 1..=4 {
            graph
                .add_node_rust(Node {
                    id: i,
                    label: Some(i.to_string()),
                    attributes: None,
                })
                .unwrap();
        }

        // All pairs
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                graph
                    .add_edge_rust(Edge {
                        id: 0,
                        source: i,
                        target: j,
                        weight: Some(1.0),
                        attributes: None,
                    })
                    .unwrap();
            }
        }

        let rc = rich_club_coefficient(&graph, false, 100, None, 1).unwrap();

        // All nodes have degree 3
        // k=0, 1, 2: all 4 nodes, 6 edges, potential = 12, φ = 1.0
        for k in 0..3 {
            assert!((rc.get(&k).unwrap() - 1.0).abs() < 1e-10);
        }
    }
}
