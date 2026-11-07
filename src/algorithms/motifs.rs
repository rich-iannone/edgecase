use crate::graph::Graph;
use crate::utils::Seed;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

fn adjacency_set(graph: &Graph) -> FxHashMap<u64, FxHashSet<u64>> {
    let directed = graph.is_directed();
    let mut adj: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::default();
    for &nid in graph.get_node_ids().iter() {
        adj.insert(nid, FxHashSet::default());
    }
    for &(_, s, t) in graph.edges().iter() {
        adj.entry(s).or_default().insert(t);
        if !directed {
            adj.entry(t).or_default().insert(s);
        }
    }
    adj
}

fn canonical_key(nodes: &[u64], adj: &FxHashMap<u64, FxHashSet<u64>>, directed: bool) -> u64 {
    let k = nodes.len();
    // Build local adjacency matrix (k x k bitmask)
    let mut mat = vec![vec![false; k]; k];
    for i in 0..k {
        for j in 0..k {
            if i != j && adj.get(&nodes[i]).is_some_and(|s| s.contains(&nodes[j])) {
                mat[i][j] = true;
            }
        }
    }

    // Try all permutations and pick the smallest bitmask as the canonical form
    let perms = permutations(k);
    let mut min_key = u64::MAX;
    for perm in &perms {
        let mut key: u64 = 0;
        let mut bit = 0;
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    if mat[perm[i]][perm[j]] {
                        key |= 1 << bit;
                    }
                    bit += 1;
                }
            }
        }
        if !directed {
            // For undirected, also consider the transpose as equivalent
            let mut key_t: u64 = 0;
            bit = 0;
            for i in 0..k {
                for j in 0..k {
                    if i != j {
                        if mat[perm[j]][perm[i]] {
                            key_t |= 1 << bit;
                        }
                        bit += 1;
                    }
                }
            }
            key = key.min(key_t);
        }
        min_key = min_key.min(key);
    }
    min_key
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let mut items: Vec<usize> = (0..n).collect();
    generate_perms(&mut items, n, &mut result);
    result
}

fn generate_perms(items: &mut Vec<usize>, k: usize, result: &mut Vec<Vec<usize>>) {
    if k == 1 {
        result.push(items.clone());
        return;
    }
    for i in 0..k {
        generate_perms(items, k - 1, result);
        if k % 2 == 0 {
            items.swap(i, k - 1);
        } else {
            items.swap(0, k - 1);
        }
    }
}

fn is_connected_subgraph(
    nodes: &[u64],
    adj: &FxHashMap<u64, FxHashSet<u64>>,
    directed: bool,
) -> bool {
    if nodes.len() <= 1 {
        return true;
    }
    let node_set: FxHashSet<u64> = nodes.iter().copied().collect();
    let mut visited = FxHashSet::default();
    let mut stack = vec![nodes[0]];
    visited.insert(nodes[0]);

    while let Some(v) = stack.pop() {
        if let Some(nbrs) = adj.get(&v) {
            for &w in nbrs {
                if node_set.contains(&w) && visited.insert(w) {
                    stack.push(w);
                }
            }
        }
        if directed {
            // For connectivity check on directed graphs, also follow reverse edges
            for &u in nodes {
                if adj.get(&u).is_some_and(|s| s.contains(&v))
                    && node_set.contains(&u)
                    && visited.insert(u)
                {
                    stack.push(u);
                }
            }
        }
    }
    visited.len() == nodes.len()
}

fn enumerate_connected_subgraphs(
    size: usize,
    node_ids: &[u64],
    adj: &FxHashMap<u64, FxHashSet<u64>>,
    directed: bool,
) -> Vec<Vec<u64>> {
    let mut result = Vec::new();
    let n = node_ids.len();
    if size > n {
        return result;
    }

    // Enumerate all size-k combinations in sorted index order
    // and filter for connectivity.
    let mut combo = vec![0usize; size];
    enumerate_combos(node_ids, adj, directed, size, 0, 0, &mut combo, &mut result);
    result
}

#[allow(clippy::too_many_arguments)] // recursive helper threading its search state
fn enumerate_combos(
    node_ids: &[u64],
    adj: &FxHashMap<u64, FxHashSet<u64>>,
    directed: bool,
    size: usize,
    depth: usize,
    start: usize,
    combo: &mut Vec<usize>,
    result: &mut Vec<Vec<u64>>,
) {
    if depth == size {
        let nodes: Vec<u64> = combo.iter().map(|&i| node_ids[i]).collect();
        if is_connected_subgraph(&nodes, adj, directed) {
            result.push(nodes);
        }
        return;
    }
    let remaining = size - depth;
    let n = node_ids.len();
    for i in start..=(n - remaining) {
        combo[depth] = i;
        enumerate_combos(
            node_ids,
            adj,
            directed,
            size,
            depth + 1,
            i + 1,
            combo,
            result,
        );
    }
}

fn motif_class_name(key: u64, size: usize, directed: bool) -> String {
    if !directed && size == 3 {
        // Undirected 3-node motifs: triangle (3 edges) or path/wedge (2 edges)
        let edge_count = key.count_ones();
        // In undirected encoding, each edge appears in both directions in the bitmask
        // k=3, k*(k-1)=6 bits. Each undirected edge sets 2 bits.
        let undirected_edges = edge_count / 2;
        return match undirected_edges {
            3 => "triangle".to_string(),
            2 => "path".to_string(),
            _ => format!("motif_3_{}", key),
        };
    }
    if !directed && size == 4 {
        let edge_count = key.count_ones() / 2; // undirected edges
        return match edge_count {
            6 => "clique_4".to_string(),
            5 => "near_clique_4".to_string(),
            4 => {
                // Could be cycle, star, or other 4-edge patterns
                format!("motif_4e{}_k{}", edge_count, key)
            }
            3 => format!("motif_4e{}_k{}", edge_count, key),
            _ => format!("motif_4e{}_k{}", edge_count, key),
        };
    }
    // For directed or fallback, use generic naming
    format!("motif_{}_{}", size, key)
}

/// Count occurrences of all non-isomorphic connected subgraphs of a given size, keyed by
/// class name in sorted order so the result doesn't depend on hashing.
#[pyfunction]
#[pyo3(signature = (graph, size))]
pub fn motif_census(graph: &Graph, size: usize) -> PyResult<BTreeMap<String, u64>> {
    if !(3..=4).contains(&size) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Motif size must be 3 or 4",
        ));
    }

    let node_ids = graph.get_node_ids();
    if node_ids.len() < size {
        return Ok(BTreeMap::new());
    }

    let directed = graph.is_directed();
    let adj = adjacency_set(graph);

    let mut sorted_ids = node_ids.clone();
    sorted_ids.sort_unstable();

    let subgraphs = enumerate_connected_subgraphs(size, &sorted_ids, &adj, directed);

    let mut counts: FxHashMap<u64, u64> = FxHashMap::default();
    for sg in &subgraphs {
        if is_connected_subgraph(sg, &adj, directed) {
            let key = canonical_key(sg, &adj, directed);
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut result = BTreeMap::new();
    for (key, count) in &counts {
        let name = motif_class_name(*key, size, directed);
        *result.entry(name).or_insert(0) += count;
    }

    Ok(result)
}

/// Compare motif counts against degree-preserving random graphs to find significant motifs.
/// Classes come in sorted order, each with its statistics in a fixed order.
#[pyfunction]
#[pyo3(signature = (graph, size, n_random=100, seed=None))]
pub fn motif_significance<'py>(
    py: Python<'py>,
    graph: &Graph,
    size: usize,
    n_random: usize,
    seed: Option<Seed>,
) -> PyResult<BTreeMap<String, Bound<'py, PyDict>>> {
    let seed = seed.map(u64::from);
    use crate::algorithms::null_models::degree_preserving_rewire;

    if !(3..=4).contains(&size) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Motif size must be 3 or 4",
        ));
    }

    if n_random == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_random must be at least 1",
        ));
    }

    // Get observed counts
    let observed = motif_census(graph, size)?;

    // Collect all motif classes found in the observed graph
    let mut all_classes: FxHashSet<String> = observed.keys().cloned().collect();

    // Generate random graphs and count motifs
    let mut random_counts: Vec<BTreeMap<String, u64>> = Vec::with_capacity(n_random);
    for i in 0..n_random {
        let rand_seed = seed.map(|s| s.wrapping_add(i as u64));
        let random_graph = degree_preserving_rewire(graph, None, rand_seed.map(Seed::from))?;
        let rc = motif_census(&random_graph, size)?;
        for k in rc.keys() {
            all_classes.insert(k.clone());
        }
        random_counts.push(rc);
    }

    // Compute z-scores for each motif class
    let mut result = BTreeMap::new();
    let n_f = n_random as f64;

    for class in &all_classes {
        let obs = *observed.get(class).unwrap_or(&0) as f64;

        // Mean and std of random counts
        let rand_vals: Vec<f64> = random_counts
            .iter()
            .map(|rc| *rc.get(class).unwrap_or(&0) as f64)
            .collect();

        let mean: f64 = rand_vals.iter().sum::<f64>() / n_f;
        let variance: f64 = rand_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_f;
        let std_dev = variance.sqrt();

        let z_score = if std_dev > 1e-10 {
            (obs - mean) / std_dev
        } else if (obs - mean).abs() < 1e-10 {
            0.0
        } else if obs > mean {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        // p-value: fraction of random graphs with count >= observed
        let p_value = rand_vals.iter().filter(|&&v| v >= obs).count() as f64 / n_f;

        let entry = [
            ("count", obs),
            ("mean_random", mean),
            ("std_random", std_dev),
            ("z_score", z_score),
            ("p_value", p_value),
        ]
        .into_py_dict(py)?;
        result.insert(class.clone(), entry);
    }

    Ok(result)
}
