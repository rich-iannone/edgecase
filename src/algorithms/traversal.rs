//! Traversal outputs beyond the visit order: BFS/DFS tree edges, BFS layers, DFS pre- and
//! post-order.
//!
//! Neighbors are visited in the graph's edge insertion order, the same order `Graph.bfs()` and
//! `Graph.dfs()` use, so these functions agree with them (and with NetworkX when edges are added
//! in the same order). `mode` selects the direction on directed graphs: "out" follows edges,
//! "in" follows them backward, "all" ignores direction (out-neighbors first, then in-neighbors).

use crate::algorithms::path_queries::IndexedGraph;
use crate::graph::Graph;
use pyo3::prelude::*;

fn parse_mode(mode: &str) -> PyResult<(bool, bool)> {
    match mode {
        "out" => Ok((true, false)),
        "in" => Ok((false, true)),
        "all" | "total" => Ok((true, true)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid mode '{}'. Must be 'out', 'in', or 'all'",
            mode
        ))),
    }
}

/// Neighbor lists for a traversal in the requested direction (undirected graphs: both ways).
fn neighbor_lists(graph: &Graph, mode: &str) -> PyResult<(IndexedGraph, Vec<Vec<usize>>)> {
    let (use_out, use_in) = parse_mode(mode)?;
    let g = IndexedGraph::new(graph, false);
    let directed = graph.is_directed();
    let lists = (0..g.len())
        .map(|v| {
            let mut out: Vec<usize> = Vec::new();
            if use_out || !directed {
                out.extend(g.out[v].iter().map(|&(w, _)| w));
            }
            if directed && use_in {
                out.extend(g.inn[v].iter().map(|&(w, _)| w));
            }
            out
        })
        .collect();
    Ok((g, lists))
}

/// Breadth-first search from one or more sources.
///
/// Returns `(tree_edges, layers)`: tree edges `(parent, child)` in the order children are
/// discovered, and the layers (layer 0 = the sources, layer k = nodes first reached at k hops).
/// With `depth_limit`, nodes farther than that many hops are not visited.
#[pyfunction]
#[pyo3(signature = (graph, sources, depth_limit=None, mode="out"))]
pub fn bfs_search(
    graph: &Graph,
    sources: Vec<u64>,
    depth_limit: Option<usize>,
    mode: &str,
) -> PyResult<(Vec<(u64, u64)>, Vec<Vec<u64>>)> {
    let (g, nbrs) = neighbor_lists(graph, mode)?;
    let mut seen = vec![false; g.len()];
    let mut layer: Vec<usize> = Vec::new();
    for &s in &sources {
        let i = g.index_of(s)?;
        if !seen[i] {
            seen[i] = true;
            layer.push(i);
        }
    }
    let mut edges = Vec::new();
    let mut layers: Vec<Vec<u64>> = Vec::new();
    let mut depth = 0usize;
    while !layer.is_empty() {
        layers.push(layer.iter().map(|&i| g.ids[i]).collect());
        if depth_limit.is_some_and(|d| depth >= d) {
            break;
        }
        let mut next = Vec::new();
        for &v in &layer {
            for &w in &nbrs[v] {
                if !seen[w] {
                    seen[w] = true;
                    edges.push((g.ids[v], g.ids[w]));
                    next.push(w);
                }
            }
        }
        layer = next;
        depth += 1;
    }
    Ok((edges, layers))
}

/// Depth-first search from `source`, or from every unvisited node in ID order if `source` is
/// None (covering the whole graph).
///
/// Returns `(preorder, postorder, tree_edges)`. Tree edges are `(parent, child)` in discovery
/// order. With `depth_limit`, nodes more than that many hops from their root are not visited.
#[pyfunction]
#[pyo3(signature = (graph, source=None, depth_limit=None, mode="out"))]
pub fn dfs_search(
    graph: &Graph,
    source: Option<u64>,
    depth_limit: Option<usize>,
    mode: &str,
) -> PyResult<(Vec<u64>, Vec<u64>, Vec<(u64, u64)>)> {
    let (g, nbrs) = neighbor_lists(graph, mode)?;
    let n = g.len();
    let roots: Vec<usize> = match source {
        Some(s) => vec![g.index_of(s)?],
        None => (0..n).collect(),
    };
    let mut seen = vec![false; n];
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut edges = Vec::new();
    for root in roots {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        pre.push(g.ids[root]);
        // (node, next neighbor index, depth)
        let mut stack: Vec<(usize, usize, usize)> = vec![(root, 0, 0)];
        while let Some(&mut (v, ref mut i, d)) = stack.last_mut() {
            let can_descend = depth_limit.is_none_or(|lim| d < lim);
            if can_descend && *i < nbrs[v].len() {
                let w = nbrs[v][*i];
                *i += 1;
                if !seen[w] {
                    seen[w] = true;
                    pre.push(g.ids[w]);
                    edges.push((g.ids[v], g.ids[w]));
                    stack.push((w, 0, d + 1));
                }
            } else {
                post.push(g.ids[v]);
                stack.pop();
            }
        }
    }
    Ok((pre, post, edges))
}

/// Build a directed tree (or forest) graph from traversal output.
///
/// Nodes keep their IDs, labels and a copy of their attributes; each tree edge `(parent, child)`
/// copies the weight and attributes of an original edge joining the two (in either direction,
/// since undirected and `mode="in"` traversals cross edges backward).
#[pyfunction]
pub fn tree_graph(
    py: Python,
    graph: &Graph,
    nodes: Vec<u64>,
    edges: Vec<(u64, u64)>,
) -> PyResult<Graph> {
    use crate::graph::{Edge, Node};
    use rustc_hash::FxHashMap;
    let copy = |a: &Option<Py<pyo3::types::PyDict>>| -> PyResult<Option<Py<pyo3::types::PyDict>>> {
        Ok(match a {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        })
    };
    let mut result = Graph::new_rust(true, graph.is_weighted());
    let all_nodes = graph.get_nodes();
    for id in nodes {
        let node = all_nodes.get(&id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Node {} not found in graph", id))
        })?;
        result.add_node_rust(Node { id, label: node.label.clone(), attributes: copy(&node.attributes)? })?;
    }
    let mut by_pair: FxHashMap<(u64, u64), &Edge> = FxHashMap::default();
    for e in graph.get_edges_internal() {
        by_pair.entry((e.source, e.target)).or_insert(e);
    }
    for (p, c) in edges {
        let orig = by_pair.get(&(p, c)).or_else(|| by_pair.get(&(c, p)));
        let (weight, attributes) = match orig {
            Some(e) => (e.weight, copy(&e.attributes)?),
            None => (None, None),
        };
        result.add_edge_rust(Edge { id: 0, source: p, target: c, weight, attributes })?;
    }
    Ok(result)
}
