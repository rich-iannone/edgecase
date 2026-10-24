//! Clustering and degree-mixing extras (square clustering, butterflies, generalized degree,
//! average degree connectivity, degree pairs for mixing matrices) and the cut profile of a node
//! set (cut size, volumes, boundaries) behind conductance, expansion and related measures.
//!
//! Semantics follow NetworkX. Where weights matter, `weighted` selects each edge's `weight`
//! (1.0 if missing) instead of counting edges.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

fn err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn weight_of(e: &crate::graph::Edge, weighted: bool) -> f64 {
    if weighted {
        e.weight.unwrap_or(1.0)
    } else {
        1.0
    }
}

/// Neighbor sets without self-loops: successors on directed graphs (as NetworkX's adjacency),
/// both directions on undirected graphs.
fn neighbor_sets(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<FxHashSet<usize>> {
    let mut nb: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); idx.len()];
    let undirected = !graph.is_directed();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (s, t) = (idx[&e.source], idx[&e.target]);
        nb[s].insert(t);
        if undirected {
            nb[t].insert(s);
        }
    }
    nb
}

/// Square clustering (Lind et al. 2005): the share of possible squares through each node that
/// exist. Matches NetworkX (neighbors are successors on directed graphs; self-loops ignored).
#[pyfunction]
pub fn square_clustering(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    let (ids, idx) = sorted_ids(graph);
    let nb = neighbor_sets(graph, &idx);
    let mut out = FxHashMap::default();
    for v in 0..ids.len() {
        let vn = &nb[v];
        let dm1 = vn.len() as i64 - 1;
        if dm1 <= 0 {
            out.insert(ids[v], 0.0);
            continue;
        }
        let mut uw_degrees = 0i64;
        let uw_count = vn.len() as i64 * dm1;
        let mut triangles = 0i64;
        let mut squares = 0i64;
        for &u in vn {
            uw_degrees += nb[u].len() as i64 * dm1;
            let p2 = nb[u].intersection(vn).count() as i64;
            triangles += p2;
            squares += p2 * (p2 - 1);
        }
        let mut two_hop: FxHashSet<usize> = FxHashSet::default();
        for &u in vn {
            two_hop.extend(nb[u].iter().copied());
        }
        for x in two_hop {
            if x == v || vn.contains(&x) {
                continue;
            }
            let p2 = vn.intersection(&nb[x]).count() as i64;
            squares += p2 * (p2 - 1);
        }
        squares /= 2;
        let potential = uw_degrees - uw_count - triangles - squares;
        out.insert(ids[v], if potential > 0 { squares as f64 / potential as f64 } else { 0.0 });
    }
    Ok(out)
}

/// Butterflies (4-cycles, `K_{2,2}` in a bipartite graph) through each node, counted with
/// the vertex-priority wedge enumeration of BFC-VP (Wang et al. 2019): each 4-cycle is found
/// once, from its highest-priority node `u` and the opposite node `w`, as a pair of wedges
/// `u - v - w`. Matches NetworkX `bipartite.butterflies`. Undirected graphs only; self-loops
/// and parallel edges are ignored.
#[pyfunction]
pub fn butterflies(graph: &Graph) -> PyResult<FxHashMap<u64, u64>> {
    if graph.is_directed() {
        return Err(err("butterflies() requires an undirected graph"));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let nb = neighbor_sets(graph, &idx);
    // Priority by (degree, index); neighbor lists sorted by ascending priority
    let prio = |v: usize| (nb[v].len(), v);
    let sorted_nb: Vec<Vec<usize>> = nb
        .iter()
        .map(|s| {
            let mut v: Vec<usize> = s.iter().copied().collect();
            v.sort_unstable_by_key(|&x| prio(x));
            v
        })
        .collect();
    let mut count = vec![0u64; n];
    let mut wedges = vec![0u64; n];
    let mut touched: Vec<usize> = Vec::new();
    for u in 0..n {
        let pu = prio(u);
        let lower = |v: &&usize| prio(**v) < pu;
        for v in sorted_nb[u].iter().take_while(lower) {
            for w in sorted_nb[*v].iter().take_while(lower) {
                if wedges[*w] == 0 {
                    touched.push(*w);
                }
                wedges[*w] += 1;
            }
        }
        for &w in &touched {
            let k = wedges[w];
            if k >= 2 {
                let bf = k * (k - 1) / 2;
                count[u] += bf;
                count[w] += bf;
            }
        }
        // Each middle node of a wedge pair (u, w) shares a butterfly with the other k - 1
        for v in sorted_nb[u].iter().take_while(lower) {
            for w in sorted_nb[*v].iter().take_while(lower) {
                if wedges[*w] >= 2 {
                    count[*v] += wedges[*w] - 1;
                }
            }
        }
        for &w in &touched {
            wedges[w] = 0;
        }
        touched.clear();
    }
    Ok(ids.into_iter().zip(count).collect())
}

/// Generalized degree (Hoang et al.): for each node, how many of its edges lie in exactly k
/// triangles, as a map k -> count. Undirected graphs only; self-loops ignored.
#[pyfunction]
pub fn generalized_degree(graph: &Graph) -> PyResult<FxHashMap<u64, FxHashMap<usize, usize>>> {
    if graph.is_directed() {
        return Err(err("generalized_degree() requires an undirected graph"));
    }
    let (ids, idx) = sorted_ids(graph);
    let nb = neighbor_sets(graph, &idx);
    Ok((0..ids.len())
        .map(|v| {
            let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
            for &w in &nb[v] {
                *counts.entry(nb[v].intersection(&nb[w]).count()).or_insert(0) += 1;
            }
            (ids[v], counts)
        })
        .collect())
}

/// Per-node degrees in the requested direction ("in", "out", or "in+out"), optionally
/// weighted. Undirected degrees count both endpoints (a self-loop adds 2).
fn degrees(graph: &Graph, idx: &FxHashMap<u64, usize>, which: &str, weighted: bool) -> Vec<f64> {
    let mut d = vec![0.0; idx.len()];
    let directed = graph.is_directed();
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        if !directed || which != "in" {
            d[s] += w;
        }
        if !directed || which != "out" {
            d[t] += w;
        }
    }
    d
}

/// Average degree connectivity (average nearest-neighbor degree as a function of degree), as
/// NetworkX `average_degree_connectivity`: a map from (unweighted) source degree to the mean
/// target degree of those nodes' neighbors, weighted by edge weight when `weighted`.
#[pyfunction]
#[pyo3(signature = (graph, source="in+out", target="in+out", weighted=false))]
pub fn average_degree_connectivity(
    graph: &Graph,
    source: &str,
    target: &str,
    weighted: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    let directed = graph.is_directed();
    for (name, v) in [("source", source), ("target", target)] {
        if !["in", "out", "in+out"].contains(&v) {
            return Err(err(format!("{} must be one of 'in', 'out', or 'in+out'", name)));
        }
        if !directed && v != "in+out" {
            return Err(err("source and target arguments are only supported for directed graphs"));
        }
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let src_deg = degrees(graph, &idx, source, false);
    let src_wdeg = degrees(graph, &idx, source, weighted);
    let tgt_deg = degrees(graph, &idx, target, false);
    // (neighbor, weight) once per incident edge in the source direction, so the numerator
    // matches the (weighted) degree it is divided by
    let mut nbr: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        let mut add = |a: usize, b: usize| {
            nbr[a].push((b, w));
        };
        if !directed {
            add(s, t);
            if s != t {
                add(t, s);
            }
        } else {
            match source {
                "out" => add(s, t),
                "in" => add(t, s),
                _ => {
                    add(s, t);
                    add(t, s);
                }
            }
        }
    }
    let mut dsum: FxHashMap<u64, f64> = FxHashMap::default();
    let mut dnorm: FxHashMap<u64, f64> = FxHashMap::default();
    for v in 0..n {
        let k = src_deg[v].round() as u64;
        let s: f64 = nbr[v].iter().map(|&(u, w)| w * tgt_deg[u]).sum();
        *dsum.entry(k).or_insert(0.0) += s;
        *dnorm.entry(k).or_insert(0.0) += src_wdeg[v];
    }
    Ok(dsum
        .into_iter()
        .map(|(k, s)| {
            let d = dnorm[&k];
            (k, if d == 0.0 { s } else { s / d })
        })
        .collect())
}

/// Degree pairs `(x_degree(u), y_degree(v))` over every edge (u, v) followed from u, as
/// NetworkX `node_degree_xy`: out-edges on directed graphs, both directions on undirected ones
/// (a self-loop once). Returned as aggregated `(x, y, count)` triples.
#[pyfunction]
#[pyo3(signature = (graph, x="out", y="in", weighted=false))]
pub fn degree_pairs(
    graph: &Graph,
    x: &str,
    y: &str,
    weighted: bool,
) -> PyResult<Vec<(f64, f64, u64)>> {
    let directed = graph.is_directed();
    for v in [x, y] {
        if directed && v != "in" && v != "out" {
            return Err(err("x and y must be 'in' or 'out' on directed graphs"));
        }
    }
    let (_, idx) = sorted_ids(graph);
    let dx = degrees(graph, &idx, if directed { x } else { "in+out" }, weighted);
    let dy = degrees(graph, &idx, if directed { y } else { "in+out" }, weighted);
    let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> = FxHashMap::default();
    let mut push = |a: usize, b: usize| {
        let key = (dx[a].to_bits(), dy[b].to_bits());
        counts.entry(key).or_insert((dx[a], dy[b], 0)).2 += 1;
    };
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        push(s, t);
        if !directed && s != t {
            push(t, s);
        }
    }
    let mut out: Vec<(f64, f64, u64)> = counts.into_values().collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(out)
}

/// Everything needed for cut and boundary measures of a node set S against T (default: the
/// complement of S):
/// `(cut_size, volume_S, volume_T, size_S, size_T, edge_count, edge_boundary, node_boundary,
/// neighborhood_size)`.
///
/// - cut size: total weight of edges from S to T, plus T to S on directed graphs
/// - volume: total (out-)degree, weighted when `weighted`
/// - edge boundary: IDs of edges leaving S for T (out-edges on directed graphs)
/// - node boundary: nodes outside S adjacent from S (and in T, if given)
/// - neighborhood size: all nodes adjacent from S, members of S included (NetworkX's
///   `node_expansion` numerator)
#[pyfunction]
#[pyo3(signature = (graph, s, t=None, weighted=false))]
#[allow(clippy::type_complexity)]
pub fn cut_profile(
    graph: &Graph,
    s: Vec<u64>,
    t: Option<Vec<u64>>,
    weighted: bool,
) -> PyResult<(f64, f64, f64, usize, usize, usize, Vec<u64>, Vec<u64>, usize)> {
    let (ids, _) = sorted_ids(graph);
    let known: FxHashSet<u64> = ids.iter().copied().collect();
    let sset: FxHashSet<u64> = s.into_iter().collect();
    for v in &sset {
        if !known.contains(v) {
            return Err(err(format!("Node {} does not exist", v)));
        }
    }
    let explicit_t = t.is_some();
    let tset: FxHashSet<u64> = match t {
        Some(list) => {
            let ts: FxHashSet<u64> = list.into_iter().collect();
            for v in &ts {
                if !known.contains(v) {
                    return Err(err(format!("Node {} does not exist", v)));
                }
            }
            ts
        }
        None => ids.iter().copied().filter(|v| !sset.contains(v)).collect(),
    };
    let directed = graph.is_directed();
    let mut cut = 0.0;
    let mut vol_s = 0.0;
    let mut vol_t = 0.0;
    let mut eb: Vec<u64> = Vec::new();
    let mut boundary: FxHashSet<u64> = FxHashSet::default();
    let mut neighborhood: FxHashSet<u64> = FxHashSet::default();
    for e in graph.get_edges_internal() {
        let w = weight_of(e, weighted);
        let (a, b) = (e.source, e.target);
        // volumes: out-degree on directed graphs, degree (both endpoints) otherwise
        for (node, counts) in [(a, true), (b, !directed)] {
            if counts {
                if sset.contains(&node) {
                    vol_s += w;
                }
                if tset.contains(&node) {
                    vol_t += w;
                }
            }
        }
        let crosses = |p: u64, q: u64| sset.contains(&p) && tset.contains(&q);
        if crosses(a, b) || (!directed && crosses(b, a)) {
            eb.push(e.id);
        }
        if crosses(a, b) || crosses(b, a) {
            cut += w;
        }
        // neighbors reached from S (successors on directed graphs)
        let mut reach = |p: u64, q: u64| {
            if sset.contains(&p) {
                if p != q {
                    neighborhood.insert(q);
                }
                if !sset.contains(&q) && (!explicit_t || tset.contains(&q)) {
                    boundary.insert(q);
                }
            }
        };
        reach(a, b);
        if !directed {
            reach(b, a);
        }
    }
    eb.sort_unstable();
    let mut nb: Vec<u64> = boundary.into_iter().collect();
    nb.sort_unstable();
    Ok((
        cut,
        vol_s,
        vol_t,
        sset.len(),
        tset.len(),
        graph.get_edges_internal().len(),
        eb,
        nb,
        neighborhood.len(),
    ))
}
