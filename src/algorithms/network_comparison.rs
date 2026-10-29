//! Network comparison and differential analysis.
//!
//! - `edge_overlap`: Jaccard similarity of edge sets
//! - `differential_edges`: edges present in one graph but not the other
//! - `graph_edit_distance`: minimum edit operations to transform one graph into another

use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::graph::Graph;

fn collect_label_edges(graph: &Graph) -> PyResult<FxHashSet<(String, String)>> {
    let directed = graph.is_directed();
    let raw_edges = graph.edges();
    let mut edge_set = FxHashSet::default();
    for &(_, s, t) in &raw_edges {
        let s_label = graph.label_from_id(s)?.unwrap_or_else(|| format!("{}", s));
        let t_label = graph.label_from_id(t)?.unwrap_or_else(|| format!("{}", t));
        let key = if directed || s_label <= t_label {
            (s_label, t_label)
        } else {
            (t_label, s_label)
        };
        edge_set.insert(key);
    }
    Ok(edge_set)
}

fn collect_node_labels(graph: &Graph) -> PyResult<FxHashSet<String>> {
    let mut labels = FxHashSet::default();
    for &nid in graph.get_node_ids().iter() {
        let label = graph
            .label_from_id(nid)?
            .unwrap_or_else(|| format!("{}", nid));
        labels.insert(label);
    }
    Ok(labels)
}

/// Compute edge overlap (Jaccard similarity) between two graphs.
///
/// Returns a dict with keys: `jaccard` (|intersection|/|union|),
/// `overlap` (|intersection|), `union_size`, `g1_only`, `g2_only`.
/// Edges are compared by node labels.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn edge_overlap(g1: &Graph, g2: &Graph) -> PyResult<std::collections::HashMap<String, f64>> {
    let edges1 = collect_label_edges(g1)?;
    let edges2 = collect_label_edges(g2)?;

    let intersection_size = edges1.intersection(&edges2).count();
    let union_size = edges1.union(&edges2).count();
    let g1_only = edges1.difference(&edges2).count();
    let g2_only = edges2.difference(&edges1).count();

    let jaccard = if union_size == 0 {
        1.0
    } else {
        intersection_size as f64 / union_size as f64
    };

    let mut result = std::collections::HashMap::new();
    result.insert("jaccard".to_string(), jaccard);
    result.insert("overlap".to_string(), intersection_size as f64);
    result.insert("union_size".to_string(), union_size as f64);
    result.insert("g1_only".to_string(), g1_only as f64);
    result.insert("g2_only".to_string(), g2_only as f64);

    Ok(result)
}

/// Find edges present in one graph but not the other.
///
/// Returns a dict with `added` (in g2 only) and `removed` (in g1 only),
/// each a list of `(source_label, target_label)` tuples.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn differential_edges(
    g1: &Graph,
    g2: &Graph,
) -> PyResult<std::collections::HashMap<String, Vec<(String, String)>>> {
    let edges1 = collect_label_edges(g1)?;
    let edges2 = collect_label_edges(g2)?;

    let removed: Vec<(String, String)> = edges1.difference(&edges2).cloned().collect();
    let added: Vec<(String, String)> = edges2.difference(&edges1).cloned().collect();

    let mut result = std::collections::HashMap::new();
    result.insert("added".to_string(), added);
    result.insert("removed".to_string(), removed);

    Ok(result)
}

/// Approximate graph edit distance between two graphs.
///
/// Counts the minimum number of node insertions, node deletions,
/// edge insertions, and edge deletions needed to transform g1 into g2.
/// Uses label-based matching (exact label match = same node).
///
/// Returns a dict with `distance`, `node_insertions`, `node_deletions`,
/// `edge_insertions`, `edge_deletions`.
#[pyfunction]
#[pyo3(signature = (g1, g2, node_cost=1.0, edge_cost=1.0))]
pub fn graph_edit_distance(
    g1: &Graph,
    g2: &Graph,
    node_cost: f64,
    edge_cost: f64,
) -> PyResult<std::collections::HashMap<String, f64>> {
    let nodes1 = collect_node_labels(g1)?;
    let nodes2 = collect_node_labels(g2)?;
    let edges1 = collect_label_edges(g1)?;
    let edges2 = collect_label_edges(g2)?;

    let node_ins = nodes2.difference(&nodes1).count();
    let node_del = nodes1.difference(&nodes2).count();
    let edge_ins = edges2.difference(&edges1).count();
    let edge_del = edges1.difference(&edges2).count();

    let distance =
        (node_ins + node_del) as f64 * node_cost + (edge_ins + edge_del) as f64 * edge_cost;

    let mut result = std::collections::HashMap::new();
    result.insert("distance".to_string(), distance);
    result.insert("node_insertions".to_string(), node_ins as f64);
    result.insert("node_deletions".to_string(), node_del as f64);
    result.insert("edge_insertions".to_string(), edge_ins as f64);
    result.insert("edge_deletions".to_string(), edge_del as f64);

    Ok(result)
}
