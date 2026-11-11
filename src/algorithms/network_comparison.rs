//! Network comparison and differential analysis.
//!
//! - `edge_overlap`: Jaccard similarity of edge sets
//! - `differential_edges`: edges present in one graph but not the other
//! - `graph_edit_distance`: minimum edit operations to transform one graph into another

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
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
/// `overlap` (|intersection|), `union_size`, `g1_only`, `g2_only`, in that order.
/// Edges are compared by node labels.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn edge_overlap<'py>(py: Python<'py>, g1: &Graph, g2: &Graph) -> PyResult<Bound<'py, PyDict>> {
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

    [
        ("jaccard", jaccard),
        ("overlap", intersection_size as f64),
        ("union_size", union_size as f64),
        ("g1_only", g1_only as f64),
        ("g2_only", g2_only as f64),
    ]
    .into_py_dict(py)
}

/// Find edges present in one graph but not the other.
///
/// Returns a dict with `added` (in g2 only) and `removed` (in g1 only),
/// each a sorted list of `(source_label, target_label)` tuples.
#[pyfunction]
#[pyo3(signature = (g1, g2))]
pub fn differential_edges<'py>(
    py: Python<'py>,
    g1: &Graph,
    g2: &Graph,
) -> PyResult<Bound<'py, PyDict>> {
    let edges1 = collect_label_edges(g1)?;
    let edges2 = collect_label_edges(g2)?;

    let mut removed: Vec<(String, String)> = edges1.difference(&edges2).cloned().collect();
    let mut added: Vec<(String, String)> = edges2.difference(&edges1).cloned().collect();
    removed.sort_unstable();
    added.sort_unstable();

    [("added", added), ("removed", removed)].into_py_dict(py)
}

/// Approximate graph edit distance between two graphs.
///
/// Counts the minimum number of node insertions, node deletions,
/// edge insertions, and edge deletions needed to transform g1 into g2.
/// Uses label-based matching (exact label match = same node).
///
/// Returns a dict with `distance`, `node_insertions`, `node_deletions`,
/// `edge_insertions`, `edge_deletions`, in that order.
#[pyfunction]
#[pyo3(signature = (g1, g2, node_cost=1.0, edge_cost=1.0))]
pub fn graph_edit_distance<'py>(
    py: Python<'py>,
    g1: &Graph,
    g2: &Graph,
    node_cost: f64,
    edge_cost: f64,
) -> PyResult<Bound<'py, PyDict>> {
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

    [
        ("distance", distance),
        ("node_insertions", node_ins as f64),
        ("node_deletions", node_del as f64),
        ("edge_insertions", edge_ins as f64),
        ("edge_deletions", edge_del as f64),
    ]
    .into_py_dict(py)
}
