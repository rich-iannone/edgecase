//! Traversal outputs beyond the visit order: BFS/DFS tree edges, BFS layers, DFS pre- and
//! post-order, beam search, and the labeled edges of a DFS.
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

/// `neighbor_lists` with repeats removed, keeping each neighbor's first position, and with
/// self-loops kept (a node is its own neighbor once): what NetworkX's `G.neighbors()` gives
/// on a graph built from the same edges.
fn distinct_neighbor_lists(graph: &Graph, mode: &str) -> PyResult<(IndexedGraph, Vec<Vec<usize>>)> {
    let (use_out, use_in) = parse_mode(mode)?;
    let g = IndexedGraph::new(graph, false);
    let directed = graph.is_directed();
    let n = g.len();
    let (mut out, mut inn) = (vec![Vec::new(); n], vec![Vec::new(); n]);
    for e in graph.get_edges_internal() {
        let (s, t) = (g.idx[&e.source], g.idx[&e.target]);
        out[s].push(t);
        inn[t].push(s);
        if !directed {
            out[t].push(s);
        }
    }
    let mut lists: Vec<Vec<usize>> = (0..n)
        .map(|v| {
            let mut list = Vec::new();
            if use_out || !directed {
                list.append(&mut out[v]);
            }
            if directed && use_in {
                list.append(&mut inn[v]);
            }
            list
        })
        .collect();
    let mut mark = vec![usize::MAX; n];
    for (v, list) in lists.iter_mut().enumerate() {
        list.retain(|&w| std::mem::replace(&mut mark[w], v) != v);
    }
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

/// Breadth-first search from `source` that, at each node, only follows the `width` neighbors
/// with the highest `value(node_id)` (NetworkX's `bfs_beam_edges`).
///
/// As in NetworkX, the best neighbors are chosen among all of a node's distinct neighbors,
/// visited or not, and ties keep the neighbor order. Each node's value is computed once.
/// Returns the tree edges `(parent, child)` in discovery order.
#[pyfunction]
#[pyo3(signature = (graph, source, value, width=None, mode="out"))]
pub fn bfs_beam_search(
    graph: &Graph,
    source: u64,
    value: &Bound<'_, PyAny>,
    width: Option<usize>,
    mode: &str,
) -> PyResult<Vec<(u64, u64)>> {
    let (g, nbrs) = distinct_neighbor_lists(graph, mode)?;
    let width = width.unwrap_or(g.len());
    let mut values: Vec<Option<f64>> = vec![None; g.len()];
    let mut score = |i: usize| -> PyResult<f64> {
        if let Some(x) = values[i] {
            return Ok(x);
        }
        let x: f64 = value.call1((g.ids[i],))?.extract().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "value= must give a number for every node, not for node {}",
                g.ids[i]
            ))
        })?;
        if x.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "value= gave NaN for node {}",
                g.ids[i]
            )));
        }
        values[i] = Some(x);
        Ok(x)
    };
    let s = g.index_of(source)?;
    let mut seen = vec![false; g.len()];
    seen[s] = true;
    let mut edges = Vec::new();
    let mut layer = vec![s];
    while !layer.is_empty() {
        let mut next = Vec::new();
        for &v in &layer {
            let mut best: Vec<(usize, f64)> =
                nbrs[v].iter().map(|&w| Ok((w, score(w)?))).collect::<PyResult<_>>()?;
            // stable, so equal values keep the neighbor order (like Python's sorted)
            best.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(w, _) in best.iter().take(width) {
                if !seen[w] {
                    seen[w] = true;
                    edges.push((g.ids[v], g.ids[w]));
                    next.push(w);
                }
            }
        }
        layer = next;
    }
    Ok(edges)
}

/// Depth-first search with every edge it looks at labeled (NetworkX's `dfs_labeled_edges`).
///
/// Yields `(u, v, label)`: `"forward"` when the search first reaches `v` from `u`, `"nontree"`
/// for an edge to an already visited node, `"reverse"` when it backs up from `v` to `u`, and
/// `"reverse-depth_limit"` when `v` is reached but not explored because it is `depth_limit` hops
/// from the root. Each tree starts with `(root, root, "forward")` and ends with
/// `(root, root, "reverse")`. Repeated neighbors (parallel edges) are looked at once. With no
/// `source`, every unvisited node starts a new tree, in ID order.
#[pyfunction]
#[pyo3(signature = (graph, source=None, depth_limit=None, mode="out"))]
pub fn dfs_labeled_search(
    graph: &Graph,
    source: Option<u64>,
    depth_limit: Option<usize>,
    mode: &str,
) -> PyResult<Vec<(u64, u64, &'static str)>> {
    let (g, nbrs) = distinct_neighbor_lists(graph, mode)?;
    let n = g.len();
    let roots: Vec<usize> = match source {
        Some(s) => vec![g.index_of(s)?],
        None => (0..n).collect(),
    };
    let id = |i: usize| g.ids[i];
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for root in roots {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        out.push((id(root), id(root), "forward"));
        if depth_limit != Some(0) {
            // (node, next neighbor index); the stack height is the depth of the next child
            let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
            while let Some(&mut (v, ref mut i)) = stack.last_mut() {
                if *i < nbrs[v].len() {
                    let w = nbrs[v][*i];
                    *i += 1;
                    if seen[w] {
                        out.push((id(v), id(w), "nontree"));
                    } else {
                        seen[w] = true;
                        out.push((id(v), id(w), "forward"));
                        if depth_limit.map_or(true, |lim| stack.len() < lim) {
                            stack.push((w, 0));
                        } else {
                            out.push((id(v), id(w), "reverse-depth_limit"));
                        }
                    }
                } else {
                    stack.pop();
                    if let Some(&(p, _)) = stack.last() {
                        out.push((id(p), id(v), "reverse"));
                    }
                }
            }
        }
        out.push((id(root), id(root), "reverse"));
    }
    Ok(out)
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
