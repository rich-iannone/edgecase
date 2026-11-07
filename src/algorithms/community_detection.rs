//! Partition quality and community detection: modularity, Leiden, asynchronous label
//! propagation, and k-clique percolation.
//!
//! Edge weights are used on weighted graphs (`Graph(weighted=True)`; a missing weight counts
//! as 1.0) and every edge counts 1.0 otherwise. Parallel edges add up.

use crate::graph::Graph;
use crate::utils::Seed;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

fn rng_from(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

fn edge_weight(graph: &Graph, w: Option<f64>) -> f64 {
    if graph.is_weighted() {
        w.unwrap_or(1.0)
    } else {
        1.0
    }
}

/// Undirected weighted graph on node indices `0..n` (nodes sorted by ID).
///
/// `adj` holds merged neighbor weights without self-loops; `self_w[i]` is the total weight of
/// self-loops (in aggregate graphs: the weight inside the aggregated community, each edge once);
/// `deg[i]` is the strength, with self-loops counted twice; `m2 = Σ deg = 2m`.
///
/// `size` and `div` define the null model the optimizers use: moving node `v` into community
/// `c` gains `w(v, c) - γ·size[v]·size(c)/div`. For modularity `size = deg` and `div = m2`; for
/// the Constant Potts Model (`cpm`) every node has size 1 and `div = 1`.
#[derive(Clone)]
struct WGraph {
    adj: Vec<Vec<(usize, f64)>>,
    self_w: Vec<f64>,
    deg: Vec<f64>,
    m2: f64,
    size: Vec<f64>,
    div: f64,
    cpm: bool,
}

impl WGraph {
    fn n(&self) -> usize {
        self.adj.len()
    }

    fn from_graph(graph: &Graph) -> (Self, Vec<u64>) {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let mut maps: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
        let mut self_w = vec![0.0; n];
        let mut deg = vec![0.0; n];
        for e in graph.get_edges_internal() {
            let (Some(&s), Some(&t)) = (idx.get(&e.source), idx.get(&e.target)) else {
                continue;
            };
            let w = edge_weight(graph, e.weight);
            deg[s] += w;
            deg[t] += w;
            if s == t {
                self_w[s] += w;
            } else {
                *maps[s].entry(t).or_insert(0.0) += w;
                *maps[t].entry(s).or_insert(0.0) += w;
            }
        }
        let adj = maps
            .into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f64)> = m.into_iter().collect();
                v.sort_unstable_by_key(|&(j, _)| j);
                v
            })
            .collect();
        let m2 = deg.iter().sum();
        let size = deg.clone();
        (
            WGraph {
                adj,
                self_w,
                deg,
                m2,
                size,
                div: m2,
                cpm: false,
            },
            ids,
        )
    }

    /// Switch the null model to the Constant Potts Model (unit node sizes).
    fn into_cpm(mut self) -> Self {
        self.size = vec![1.0; self.n()];
        self.div = 1.0;
        self.cpm = true;
        self
    }

    /// Collapse each group of `assign` (values `0..k`) into one node.
    fn aggregate(&self, assign: &[usize], k: usize) -> WGraph {
        let mut maps: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); k];
        let mut self_w = vec![0.0; k];
        let mut deg = vec![0.0; k];
        let mut size = vec![0.0; k];
        for v in 0..self.n() {
            let a = assign[v];
            deg[a] += self.deg[v];
            size[a] += self.size[v];
            self_w[a] += self.self_w[v];
            for &(u, w) in &self.adj[v] {
                let b = assign[u];
                if a == b {
                    self_w[a] += w / 2.0; // each internal edge is seen from both ends
                } else {
                    *maps[a].entry(b).or_insert(0.0) += w;
                }
            }
        }
        let adj = maps
            .into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f64)> = m.into_iter().collect();
                v.sort_unstable_by_key(|&(j, _)| j);
                v
            })
            .collect();
        WGraph {
            adj,
            self_w,
            deg,
            m2: self.m2,
            size,
            div: self.div,
            cpm: self.cpm,
        }
    }

    /// The quality being optimized: modularity, or the Constant Potts Model
    /// `Σ_c E_c - γ·n_c²/2` (each internal edge counted once).
    fn quality(&self, part: &[usize], gamma: f64) -> f64 {
        if !self.cpm {
            return self.modularity(part, gamma);
        }
        let k = part.iter().copied().max().map_or(0, |x| x + 1);
        let mut inside = vec![0.0; k];
        let mut tot = vec![0.0; k];
        for v in 0..self.n() {
            let c = part[v];
            tot[c] += self.size[v];
            inside[c] += self.self_w[v];
            for &(u, w) in &self.adj[v] {
                if part[u] == c {
                    inside[c] += w / 2.0;
                }
            }
        }
        (0..k)
            .map(|c| inside[c] - gamma * tot[c] * tot[c] / 2.0)
            .sum()
    }

    /// Modularity of a partition of this graph's nodes.
    fn modularity(&self, part: &[usize], gamma: f64) -> f64 {
        if self.m2 == 0.0 {
            return 0.0;
        }
        let k = part.iter().copied().max().map_or(0, |x| x + 1);
        let mut inside = vec![0.0; k];
        let mut tot = vec![0.0; k];
        for v in 0..self.n() {
            let c = part[v];
            tot[c] += self.deg[v];
            inside[c] += self.self_w[v];
            for &(u, w) in &self.adj[v] {
                if part[u] == c {
                    inside[c] += w / 2.0;
                }
            }
        }
        let m = self.m2 / 2.0;
        (0..k)
            .map(|c| inside[c] / m - gamma * (tot[c] / self.m2).powi(2))
            .sum()
    }
}

/// Renumber group labels to `0..k` in order of first appearance; returns k.
fn renumber(part: &mut [usize]) -> usize {
    let mut map: FxHashMap<usize, usize> = FxHashMap::default();
    for p in part.iter_mut() {
        let next = map.len();
        *p = *map.entry(*p).or_insert(next);
    }
    map.len()
}

fn membership_map(ids: &[u64], part: &[usize]) -> FxHashMap<u64, usize> {
    ids.iter().copied().zip(part.iter().copied()).collect()
}

/// Modularity of a node → community partition (Newman-Girvan; Leicht-Newman for directed
/// graphs), with resolution `gamma`. Matches NetworkX `community.modularity`.
#[pyfunction]
#[pyo3(signature = (graph, membership, resolution=1.0))]
pub fn modularity(
    graph: &Graph,
    membership: FxHashMap<u64, usize>,
    resolution: f64,
) -> PyResult<f64> {
    let mut inside: FxHashMap<usize, f64> = FxHashMap::default();
    let mut out_tot: FxHashMap<usize, f64> = FxHashMap::default();
    let mut in_tot: FxHashMap<usize, f64> = FxHashMap::default();
    let mut total = 0.0;
    let comm = |id: u64| -> PyResult<usize> {
        membership.get(&id).copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} is not assigned to a community",
                id
            ))
        })
    };
    for e in graph.get_edges_internal() {
        let w = edge_weight(graph, e.weight);
        let (a, b) = (comm(e.source)?, comm(e.target)?);
        total += w;
        *out_tot.entry(a).or_insert(0.0) += w;
        *in_tot.entry(b).or_insert(0.0) += w;
        if a == b {
            *inside.entry(a).or_insert(0.0) += w;
        }
    }
    if total == 0.0 {
        return Ok(0.0);
    }
    let q = if graph.is_directed() {
        let norm = 1.0 / (total * total);
        out_tot
            .keys()
            .chain(in_tot.keys())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .map(|c| {
                inside.get(c).unwrap_or(&0.0) / total
                    - resolution
                        * out_tot.get(c).unwrap_or(&0.0)
                        * in_tot.get(c).unwrap_or(&0.0)
                        * norm
            })
            .sum()
    } else {
        // Undirected: a community's degree total is its out + in endpoint weight
        let m2 = 2.0 * total;
        out_tot
            .keys()
            .chain(in_tot.keys())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .map(|c| {
                let d = out_tot.get(c).unwrap_or(&0.0) + in_tot.get(c).unwrap_or(&0.0);
                inside.get(c).unwrap_or(&0.0) / total - resolution * (d / m2).powi(2)
            })
            .sum()
    };
    Ok(q)
}

/// Constant Potts Model quality of a node → community partition (Traag, Van Dooren & Nesterov
/// 2011): `Σ_c E_c - γ·n_c²/2` on undirected graphs and `Σ_c E_c - γ·n_c²` on directed ones,
/// where `E_c` is the edge weight inside community `c` (each edge and self-loop once) and `n_c`
/// its total node weight (`node_weights`, missing nodes count 1). Matches NetworkX
/// `community.constant_potts_model`.
#[pyfunction]
#[pyo3(signature = (graph, membership, resolution=1.0, node_weights=None))]
pub fn constant_potts_model(
    graph: &Graph,
    membership: FxHashMap<u64, usize>,
    resolution: f64,
    node_weights: Option<FxHashMap<u64, f64>>,
) -> PyResult<f64> {
    let comm = |id: u64| -> PyResult<usize> {
        membership.get(&id).copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} is not assigned to a community",
                id
            ))
        })
    };
    let mut inside: FxHashMap<usize, f64> = FxHashMap::default();
    let mut size: FxHashMap<usize, f64> = FxHashMap::default();
    for id in graph.get_node_ids() {
        let w = node_weights
            .as_ref()
            .and_then(|m| m.get(&id).copied())
            .unwrap_or(1.0);
        *size.entry(comm(id)?).or_insert(0.0) += w;
    }
    for e in graph.get_edges_internal() {
        let (a, b) = (comm(e.source)?, comm(e.target)?);
        if a == b {
            *inside.entry(a).or_insert(0.0) += edge_weight(graph, e.weight);
        }
    }
    let factor = if graph.is_directed() { 1.0 } else { 0.5 };
    Ok(size
        .iter()
        .map(|(c, &n_c)| inside.get(c).unwrap_or(&0.0) - resolution * factor * n_c * n_c)
        .sum())
}

/// Shen et al.'s extended modularity of an overlapping cover of an undirected graph: each node
/// `v` in `O_v` communities contributes `1/O_v` of its degree to each, and each internal edge
/// `(u, v)` contributes `w/(O_u·O_v)`. Equals modularity when the cover is a partition. Matches
/// NetworkX `community.overlapping_modularity`; every node must be in at least one community.
#[pyfunction]
#[pyo3(signature = (graph, cover, resolution=1.0))]
pub fn overlapping_modularity(
    graph: &Graph,
    cover: Vec<Vec<u64>>,
    resolution: f64,
) -> PyResult<f64> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlapping modularity requires an undirected graph",
        ));
    }
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let n = ids.len();
    let groups: Vec<Vec<usize>> = cover
        .iter()
        .map(|c| {
            let set: FxHashSet<usize> = c.iter().filter_map(|id| idx.get(id).copied()).collect();
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            v
        })
        .collect();
    let mut count = vec![0usize; n];
    for g in &groups {
        for &v in g {
            count[v] += 1;
        }
    }
    if let Some(v) = count.iter().position(|&c| c == 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} is not in any community",
            ids[v]
        )));
    }
    // inc[v]: (other endpoint, weight) of each incident edge; a self-loop is listed once
    let mut inc: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut deg = vec![0.0; n];
    for e in graph.get_edges_internal() {
        let (Some(&s), Some(&t)) = (idx.get(&e.source), idx.get(&e.target)) else {
            continue;
        };
        let w = edge_weight(graph, e.weight);
        deg[s] += w;
        deg[t] += w;
        inc[s].push((t, w));
        if s != t {
            inc[t].push((s, w));
        }
    }
    let deg_sum: f64 = deg.iter().sum();
    if deg_sum == 0.0 {
        return Ok(0.0);
    }
    let mut in_group = vec![false; n];
    let mut q = 0.0;
    for g in &groups {
        for &v in g {
            in_group[v] = true;
        }
        let mut l = 0.0;
        let mut k = 0.0;
        for &u in g {
            k += deg[u] / count[u] as f64;
            for &(v, w) in &inc[u] {
                if in_group[v] && v >= u {
                    l += w / (count[u] * count[v]) as f64;
                }
            }
        }
        q += 2.0 * l / deg_sum - resolution * (k / deg_sum).powi(2);
        for &v in g {
            in_group[v] = false;
        }
    }
    Ok(q)
}

/// Barber's bipartite modularity of a node → community partition of an undirected bipartite
/// graph: `Σ_c [L_c/m - γ·k_c·d_c/m²]`, where `L_c` is the edge weight inside `c`, `k_c` and
/// `d_c` are the degree totals of its top-side and bottom-side nodes, and `m` the total edge
/// weight. The sides come from `top_nodes` or, if omitted, from a 2-coloring of each
/// component. Matches NetworkX `bipartite.modularity`.
#[pyfunction]
#[pyo3(signature = (graph, membership, top_nodes=None, resolution=1.0))]
pub fn bipartite_modularity(
    graph: &Graph,
    membership: FxHashMap<u64, usize>,
    top_nodes: Option<Vec<u64>>,
    resolution: f64,
) -> PyResult<f64> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bipartite modularity requires an undirected graph",
        ));
    }
    let ids = graph.get_node_ids();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let n = ids.len();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        let (Some(&s), Some(&t)) = (idx.get(&e.source), idx.get(&e.target)) else {
            continue;
        };
        if s == t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the graph is not bipartite (it has a self-loop)",
            ));
        }
        adj[s].push((t, 0));
        adj[t].push((s, 0));
    }
    let side = crate::algorithms::matching_cover::bipartition(n, &adj, &idx, top_nodes)?;
    let comm = |id: u64| -> PyResult<usize> {
        membership.get(&id).copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} is not assigned to a community",
                id
            ))
        })
    };
    let mut inside: FxHashMap<usize, f64> = FxHashMap::default();
    let mut top_deg: FxHashMap<usize, f64> = FxHashMap::default();
    let mut bottom_deg: FxHashMap<usize, f64> = FxHashMap::default();
    let mut m = 0.0;
    for &id in &ids {
        comm(id)?;
    }
    for e in graph.get_edges_internal() {
        let w = edge_weight(graph, e.weight);
        let (a, b) = (comm(e.source)?, comm(e.target)?);
        let (top, bottom) = if side[idx[&e.source]] == 0 {
            (a, b)
        } else {
            (b, a)
        };
        m += w;
        *top_deg.entry(top).or_insert(0.0) += w;
        *bottom_deg.entry(bottom).or_insert(0.0) += w;
        if a == b {
            *inside.entry(a).or_insert(0.0) += w;
        }
    }
    if m == 0.0 {
        return Ok(0.0);
    }
    Ok(top_deg
        .keys()
        .chain(bottom_deg.keys())
        .collect::<FxHashSet<_>>()
        .into_iter()
        .map(|c| {
            inside.get(c).unwrap_or(&0.0) / m
                - resolution * top_deg.get(c).unwrap_or(&0.0) * bottom_deg.get(c).unwrap_or(&0.0)
                    / (m * m)
        })
        .sum())
}

// ---------------------------------------------------------------------------------------------
// Leiden
// ---------------------------------------------------------------------------------------------

/// Default randomness for the refinement step (Traag et al. use θ = 0.01).
const THETA: f64 = 0.01;

/// Local moving with a queue: visit nodes in random order, move each to the neighboring
/// community (or an empty one) with the largest modularity gain, and re-queue neighbors that
/// now sit outside the node's new community. Returns whether anything moved.
fn move_nodes_fast(g: &WGraph, part: &mut [usize], gamma: f64, rng: &mut ChaCha8Rng) -> bool {
    let n = g.n();
    let mut tot = vec![0.0; n];
    let mut size = vec![0usize; n];
    for v in 0..n {
        tot[part[v]] += g.size[v];
        size[part[v]] += 1;
    }
    let mut empty: Vec<usize> = (0..n).filter(|&c| size[c] == 0).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<usize> = order.into_iter().collect();
    let mut queued = vec![true; n];
    let mut w_to = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut moved = false;

    while let Some(v) = queue.pop_front() {
        queued[v] = false;
        let cur = part[v];
        let kv = g.size[v];
        for &(u, w) in &g.adj[v] {
            let c = part[u];
            if w_to[c] == 0.0 {
                touched.push(c);
            }
            w_to[c] += w;
        }
        tot[cur] -= kv;
        size[cur] -= 1;
        let gain = |c: usize, w: f64| w - gamma * kv * tot[c] / g.div;
        let mut best = cur;
        let mut best_gain = gain(cur, w_to[cur]);
        for &c in &touched {
            let gc = gain(c, w_to[c]);
            if gc > best_gain {
                best = c;
                best_gain = gc;
            }
        }
        if best_gain < 0.0 {
            // An empty community has gain 0; reuse v's own if it just emptied
            best = if size[cur] == 0 {
                cur
            } else {
                empty.pop().unwrap_or(cur)
            };
        }
        if size[cur] == 0 && best != cur {
            empty.push(cur);
        }
        tot[best] += kv;
        size[best] += 1;
        part[v] = best;
        for &c in &touched {
            w_to[c] = 0.0;
        }
        touched.clear();
        if best != cur {
            moved = true;
            for &(u, _) in &g.adj[v] {
                if part[u] != best && !queued[u] {
                    queued[u] = true;
                    queue.push_back(u);
                }
            }
        }
    }
    moved
}

/// Refinement: within each community, start from singletons and merge well-connected nodes
/// into well-connected subsets, chosen at random with probability ∝ exp(gain / θ). Every
/// refined subset is connected, which is what gives Leiden its connectivity guarantee.
fn refine(g: &WGraph, part: &[usize], gamma: f64, theta: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let n = g.n();
    let mut refined: Vec<usize> = (0..n).collect();
    let mut tot_r: Vec<f64> = g.size.clone();
    let mut size_r = vec![1usize; n];
    // ext[s]: weight between subset s and the rest of its community
    let mut ext = vec![0.0; n];

    let k = part.iter().copied().max().map_or(0, |x| x + 1);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for v in 0..n {
        members[part[v]].push(v);
    }
    let mut w_to = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::new();

    for comm in members.iter() {
        if comm.len() < 2 {
            continue;
        }
        let c = part[comm[0]];
        let tot_c: f64 = comm.iter().map(|&v| g.size[v]).sum();
        for &v in comm {
            ext[v] = g.adj[v]
                .iter()
                .filter(|&&(u, _)| part[u] == c)
                .map(|&(_, w)| w)
                .sum();
        }
        let mut cand: Vec<usize> = comm
            .iter()
            .copied()
            .filter(|&v| ext[v] >= gamma * g.size[v] * (tot_c - g.size[v]) / g.div)
            .collect();
        cand.shuffle(rng);
        for v in cand {
            if size_r[refined[v]] != 1 {
                continue; // v already absorbed others
            }
            let kv = g.size[v];
            let own = refined[v];
            for &(u, w) in &g.adj[v] {
                if part[u] != c {
                    continue;
                }
                let s = refined[u];
                if w_to[s] == 0.0 {
                    touched.push(s);
                }
                w_to[s] += w;
            }
            tot_r[own] -= kv;
            // Candidates: v's own singleton (gain 0) and well-connected subsets with gain >= 0
            let mut options: Vec<(usize, f64)> = vec![(own, 0.0)];
            for &s in &touched {
                if s == own {
                    continue;
                }
                let well = ext[s] >= gamma * tot_r[s] * (tot_c - tot_r[s]) / g.div;
                let gain = w_to[s] - gamma * kv * tot_r[s] / g.div;
                if well && gain >= 0.0 {
                    options.push((s, gain));
                }
            }
            let max_gain = options.iter().map(|o| o.1).fold(f64::MIN, f64::max);
            let weights: Vec<f64> = options
                .iter()
                .map(|o| ((o.1 - max_gain) / theta).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = options[options.len() - 1].0;
            for (o, w) in options.iter().zip(&weights) {
                if r < *w {
                    chosen = o.0;
                    break;
                }
                r -= w;
            }
            if chosen != own {
                let w_vs = w_to[chosen];
                ext[chosen] = ext[chosen] + ext[own] - 2.0 * w_vs;
                refined[v] = chosen;
                size_r[own] = 0;
                size_r[chosen] += 1;
            }
            tot_r[chosen] += kv;
            for &s in &touched {
                w_to[s] = 0.0;
            }
            touched.clear();
        }
    }
    refined
}

/// One full Leiden run starting from `init` (a partition of the base graph's nodes).
fn leiden_run(
    base: &WGraph,
    init: &[usize],
    gamma: f64,
    theta: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let mut g = base.clone();
    let mut part = init.to_vec();
    renumber(&mut part);
    let mut node_of: Vec<usize> = (0..base.n()).collect(); // base node -> aggregate node
    loop {
        move_nodes_fast(&g, &mut part, gamma, rng);
        let mut flat = part.clone();
        let k = renumber(&mut flat);
        if k == g.n() {
            break;
        }
        let mut refined = refine(&g, &part, gamma, theta, rng);
        let mut kr = renumber(&mut refined);
        if kr == g.n() {
            // Refinement found nothing to merge: aggregate the moved partition instead so the
            // graph still shrinks (Louvain step)
            refined = flat.clone();
            kr = k;
        }
        let agg = g.aggregate(&refined, kr);
        let mut next = vec![0usize; kr];
        for v in 0..g.n() {
            next[refined[v]] = flat[v];
        }
        for b in node_of.iter_mut() {
            *b = refined[*b];
        }
        g = agg;
        part = next;
    }
    let mut out: Vec<usize> = node_of.iter().map(|&a| part[a]).collect();
    renumber(&mut out);
    out
}

/// Leiden community detection (Traag, Waltman & van Eck 2019) maximizing modularity
/// (`objective="modularity"`) or the Constant Potts Model (`objective="cpm"`).
///
/// Repeats full Leiden runs, each starting from the previous result, until the quality stops
/// improving or `max_iter` runs have been made. `theta` is the refinement randomness.
/// Undirected graphs only.
#[pyfunction]
#[pyo3(signature = (graph, resolution=1.0, seed=None, max_iter=10, objective="modularity", theta=0.01))]
pub fn leiden(
    graph: &Graph,
    resolution: f64,
    seed: Option<Seed>,
    max_iter: usize,
    objective: &str,
    theta: f64,
) -> PyResult<FxHashMap<u64, usize>> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Leiden algorithm requires an undirected graph",
        ));
    }
    let cpm = match objective {
        "modularity" => false,
        "cpm" => true,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "objective must be 'modularity' or 'cpm', got '{}'",
                objective
            )))
        }
    };
    if !(theta > 0.0 && theta.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must be a positive number",
        ));
    }
    let (g, ids) = WGraph::from_graph(graph);
    let g = if cpm { g.into_cpm() } else { g };
    let n = g.n();
    if n == 0 {
        return Ok(FxHashMap::default());
    }
    let mut rng = rng_from(seed);
    let mut part: Vec<usize> = (0..n).collect();
    let mut q = g.quality(&part, resolution);
    for _ in 0..max_iter.max(1) {
        let next = leiden_run(&g, &part, resolution, theta, &mut rng);
        let nq = g.quality(&next, resolution);
        let improved = nq > q + 1e-12;
        if nq >= q {
            part = next;
            q = nq;
        }
        if !improved {
            break;
        }
    }
    renumber(&mut part);
    Ok(membership_map(&ids, &part))
}

// ---------------------------------------------------------------------------------------------
// Louvain
// ---------------------------------------------------------------------------------------------

/// Multi-level Louvain (Blondel et al. 2008): local moving, then collapse each community into a
/// single node and repeat on the smaller graph until no node moves. Returns the node IDs (sorted)
/// and their community labels, numbered in order of the lowest node ID in each.
///
/// Used by `Graph.cmty_louvain()`. The fixed `seed` keeps results deterministic.
pub(crate) fn louvain_partition(
    graph: &Graph,
    resolution: f64,
    seed: u64,
) -> (Vec<u64>, Vec<usize>) {
    let (base, ids) = WGraph::from_graph(graph);
    let n = base.n();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut g = base;
    let mut part: Vec<usize> = (0..n).collect();
    let mut node_of: Vec<usize> = (0..n).collect(); // base node -> aggregate node
    loop {
        let moved = move_nodes_fast(&g, &mut part, resolution, &mut rng);
        let k = renumber(&mut part);
        if !moved || k == g.n() {
            break;
        }
        let agg = g.aggregate(&part, k);
        for b in node_of.iter_mut() {
            *b = part[*b];
        }
        g = agg;
        part = (0..k).collect();
    }
    let mut out: Vec<usize> = node_of.iter().map(|&a| part[a]).collect();
    renumber(&mut out);
    (ids, out)
}

/// The partition of the base nodes after each aggregation level of a Louvain run (`leiden=false`)
/// or a single Leiden run from singletons (`leiden=true`), coarsest last. Each level is returned
/// as labels in node ID order, numbered in order of the lowest node ID in each community.
#[pyfunction]
#[pyo3(signature = (graph, leiden=false, resolution=1.0, seed=None))]
pub fn community_levels(
    graph: &Graph,
    leiden: bool,
    resolution: f64,
    seed: Option<Seed>,
) -> PyResult<(Vec<u64>, Vec<Vec<usize>>)> {
    let seed = seed.map(u64::from);
    if leiden && graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Leiden algorithm requires an undirected graph",
        ));
    }
    let (base, ids) = WGraph::from_graph(graph);
    let n = base.n();
    let mut rng = rng_from(seed);
    let mut g = base;
    let mut part: Vec<usize> = (0..n).collect();
    let mut node_of: Vec<usize> = (0..n).collect();
    let mut levels: Vec<Vec<usize>> = Vec::new();
    if n == 0 {
        return Ok((ids, levels));
    }
    loop {
        let moved = move_nodes_fast(&g, &mut part, resolution, &mut rng);
        let mut flat = part.clone();
        let k = renumber(&mut flat);
        let mut level: Vec<usize> = node_of.iter().map(|&a| flat[a]).collect();
        renumber(&mut level);
        if levels.last() != Some(&level) && (k < g.n() || levels.is_empty()) {
            levels.push(level);
        }
        if k == g.n() || (!leiden && !moved) {
            break;
        }
        let (aggregate_by, kr) = if leiden {
            let mut refined = refine(&g, &part, resolution, THETA, &mut rng);
            let kr = renumber(&mut refined);
            if kr == g.n() {
                (flat.clone(), k)
            } else {
                (refined, kr)
            }
        } else {
            (flat.clone(), k)
        };
        let agg = g.aggregate(&aggregate_by, kr);
        let mut next = vec![0usize; kr];
        for v in 0..g.n() {
            next[aggregate_by[v]] = flat[v];
        }
        for b in node_of.iter_mut() {
            *b = aggregate_by[*b];
        }
        g = agg;
        part = if leiden { next } else { (0..kr).collect() };
    }
    Ok((ids, levels))
}

// ---------------------------------------------------------------------------------------------
// Asynchronous label propagation
// ---------------------------------------------------------------------------------------------

/// Asynchronous label propagation (Raghavan, Albert & Kumara 2007).
///
/// Every node starts with its own label; nodes are visited in random order and adopt the label
/// carrying the most edge weight among their neighbors (ties broken at random, keeping the
/// current label if it is among the best). Stops when a full pass changes nothing, or after
/// `max_iter` passes. Edge direction is ignored.
#[pyfunction]
#[pyo3(signature = (graph, seed=None, max_iter=100))]
pub fn label_propagation_communities(
    graph: &Graph,
    seed: Option<Seed>,
    max_iter: usize,
) -> PyResult<FxHashMap<u64, usize>> {
    let seed = seed.map(u64::from);
    let (g, ids) = WGraph::from_graph(graph);
    let n = g.n();
    let mut rng = rng_from(seed);
    let mut label: Vec<usize> = (0..n).collect();
    let mut order: Vec<usize> = (0..n).collect();
    let mut freq = vec![0.0; n];
    let mut touched: Vec<usize> = Vec::new();
    for _ in 0..max_iter.max(1) {
        order.shuffle(&mut rng);
        let mut changed = false;
        for &v in &order {
            if g.adj[v].is_empty() {
                continue;
            }
            for &(u, w) in &g.adj[v] {
                let l = label[u];
                if freq[l] == 0.0 {
                    touched.push(l);
                }
                freq[l] += w;
            }
            let top = touched.iter().map(|&l| freq[l]).fold(f64::MIN, f64::max);
            let mut best: Vec<usize> = touched
                .iter()
                .copied()
                // `==` first: with an infinite weight, inf - inf is NaN
                .filter(|&l| freq[l] == top || (freq[l] - top).abs() <= 1e-12 * top.max(1.0))
                .collect();
            if !best.contains(&label[v]) {
                best.sort_unstable();
                label[v] = *best.choose(&mut rng).unwrap();
                changed = true;
            }
            for &l in &touched {
                freq[l] = 0.0;
            }
            touched.clear();
        }
        if !changed {
            break;
        }
    }
    renumber(&mut label);
    Ok(membership_map(&ids, &label))
}

// ---------------------------------------------------------------------------------------------
// k-clique percolation
// ---------------------------------------------------------------------------------------------

/// Maximal cliques with at least `min_size` nodes (Bron–Kerbosch with pivoting).
fn maximal_cliques(nbrs: &[FxHashSet<usize>], min_size: usize) -> Vec<Vec<usize>> {
    let n = nbrs.len();
    let mut out = Vec::new();
    // Explicit stack of (R, P, X)
    let mut stack: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> =
        vec![(Vec::new(), (0..n).collect(), Vec::new())];
    while let Some((r, p, mut x)) = stack.pop() {
        if p.is_empty() {
            if x.is_empty() && r.len() >= min_size {
                let mut c = r.clone();
                c.sort_unstable();
                out.push(c);
            }
            continue;
        }
        if r.len() + p.len() < min_size {
            continue;
        }
        let pivot = *p
            .iter()
            .chain(x.iter())
            .max_by_key(|&&u| p.iter().filter(|&&v| nbrs[u].contains(&v)).count())
            .unwrap();
        let mut p_rest = p.clone();
        for v in p.iter().copied().filter(|v| !nbrs[pivot].contains(v)) {
            let mut r2 = r.clone();
            r2.push(v);
            let p2: Vec<usize> = p_rest
                .iter()
                .copied()
                .filter(|u| nbrs[v].contains(u))
                .collect();
            let x2: Vec<usize> = x.iter().copied().filter(|u| nbrs[v].contains(u)).collect();
            stack.push((r2, p2, x2));
            p_rest.retain(|&u| u != v);
            x.push(v);
        }
    }
    out
}

/// k-clique communities (Palla et al. 2005): unions of k-cliques that can be reached from one
/// another through k-cliques sharing k − 1 nodes. Communities may overlap, and nodes in no
/// k-clique belong to none. Returned as sorted node-ID lists, largest first. Undirected only.
#[pyfunction]
pub fn k_clique_communities(graph: &Graph, k: usize) -> PyResult<Vec<Vec<u64>>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k-clique communities require an undirected graph",
        ));
    }
    if k < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k must be at least 2, got {}",
            k
        )));
    }
    let (g, ids) = WGraph::from_graph(graph);
    let nbrs: Vec<FxHashSet<usize>> = g
        .adj
        .iter()
        .map(|a| a.iter().map(|&(u, _)| u).collect())
        .collect();
    let cliques = maximal_cliques(&nbrs, k);
    let c = cliques.len();

    // Union-find over cliques; only cliques sharing a node can overlap by k - 1
    let mut parent: Vec<usize> = (0..c).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut by_node: Vec<Vec<usize>> = vec![Vec::new(); g.n()];
    for (i, cl) in cliques.iter().enumerate() {
        for &v in cl {
            by_node[v].push(i);
        }
    }
    let mut checked: FxHashSet<(usize, usize)> = FxHashSet::default();
    for list in &by_node {
        for (a_pos, &a) in list.iter().enumerate() {
            for &b in &list[a_pos + 1..] {
                if !checked.insert((a, b)) {
                    continue;
                }
                let shared = cliques[a]
                    .iter()
                    .filter(|v| cliques[b].binary_search(v).is_ok())
                    .count();
                if shared >= k - 1 {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
    }
    let mut groups: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
    for (i, cl) in cliques.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().extend(cl.iter().copied());
    }
    let mut out: Vec<Vec<u64>> = groups
        .into_values()
        .map(|s| {
            let mut v: Vec<u64> = s.into_iter().map(|i| ids[i]).collect();
            v.sort_unstable();
            v
        })
        .collect();
    out.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Partition quality
// ---------------------------------------------------------------------------------------------

/// Coverage and performance of a partition (Fortunato 2010), matching NetworkX
/// `community.partition_quality` on simple graphs.
///
/// Coverage: the fraction of edges (parallel edges and self-loops included) that fall inside a
/// community. Performance: the fraction of node pairs classified correctly, i.e. pairs in the
/// same community that are adjacent plus pairs in different communities that are not, over all
/// pairs (ordered pairs on directed graphs). Performance counts each adjacent pair once and
/// ignores self-loops. Edgeless graphs have coverage 0; graphs with fewer than two nodes have
/// performance 0.
#[pyfunction]
pub fn partition_quality(graph: &Graph, membership: FxHashMap<u64, usize>) -> PyResult<(f64, f64)> {
    let comm = |id: u64| -> PyResult<usize> {
        membership.get(&id).copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} is not assigned to a community",
                id
            ))
        })
    };
    let directed = graph.is_directed();
    let edges = graph.get_edges_internal();
    let mut intra_edges = 0usize;
    let mut pairs: FxHashSet<(u64, u64)> = FxHashSet::default();
    for e in edges {
        let same = comm(e.source)? == comm(e.target)?;
        if same {
            intra_edges += 1;
        }
        if e.source != e.target {
            let key = if directed || e.source < e.target {
                (e.source, e.target)
            } else {
                (e.target, e.source)
            };
            pairs.insert(key);
        }
    }
    let coverage = if edges.is_empty() {
        0.0
    } else {
        intra_edges as f64 / edges.len() as f64
    };

    let n = graph.num_nodes();
    if n < 2 {
        return Ok((coverage, 0.0));
    }
    let mut sizes: FxHashMap<usize, usize> = FxHashMap::default();
    for id in graph.get_node_ids() {
        *sizes.entry(comm(id)?).or_insert(0) += 1;
    }
    let mut intra_pairs = 0usize;
    let mut inter_pairs = 0usize;
    for &(a, b) in &pairs {
        if comm(a)? == comm(b)? {
            intra_pairs += 1;
        } else {
            inter_pairs += 1;
        }
    }
    let n_f = n as f64;
    let same_pairs: f64 = sizes.values().map(|&s| (s * (s - 1)) as f64 / 2.0).sum();
    let mut total = n_f * (n_f - 1.0) / 2.0;
    let mut possible_inter = total - same_pairs;
    if directed {
        total *= 2.0;
        possible_inter *= 2.0;
    }
    let performance = (intra_pairs as f64 + possible_inter - inter_pairs as f64) / total;
    Ok((coverage, performance))
}

// ---------------------------------------------------------------------------------------------
// Greedy modularity (Clauset-Newman-Moore)
// ---------------------------------------------------------------------------------------------

#[derive(PartialEq)]
struct RowTop {
    dq: f64,
    row: usize,
    stamp: u32,
}

impl Eq for RowTop {}

impl Ord for RowTop {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // Largest gain first; ties go to the lowest community, for determinism
        self.dq
            .partial_cmp(&o.dq)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| o.row.cmp(&self.row))
    }
}

impl PartialOrd for RowTop {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// Greedy agglomerative modularity maximization (Clauset, Newman & Moore 2004).
///
/// Starts from singletons and repeatedly merges the pair of adjacent communities with the
/// largest modularity gain `ΔQ = w_ij / m − 2γ a_i a_j` (`a` = degree share), stopping when no
/// merge increases modularity. Undirected only.
///
/// Each community keeps its best partner and an upper bound on its best gain, and a global heap
/// holds one entry per community. Merging only ever *lowers* the gains of pairs that do not
/// touch the absorbed community, so stale bounds stay valid and are re-checked lazily when they
/// reach the top of the heap. Only neighbors of the absorbed (smaller) community need eager
/// updates, which keeps each merge cheap even when one community grows very large.
#[pyfunction]
#[pyo3(signature = (graph, resolution=1.0))]
pub fn greedy_modularity(graph: &Graph, resolution: f64) -> PyResult<FxHashMap<u64, usize>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Fast Greedy requires an undirected graph",
        ));
    }
    let (g, ids) = WGraph::from_graph(graph);
    let n = g.n();
    if n == 0 {
        return Ok(FxHashMap::default());
    }
    if g.m2 == 0.0 {
        return Ok(membership_map(&ids, &(0..n).collect::<Vec<_>>()));
    }
    let m = g.m2 / 2.0;
    let mut nbr: Vec<FxHashMap<usize, f64>> =
        g.adj.iter().map(|a| a.iter().copied().collect()).collect();
    let mut share: Vec<f64> = g.deg.iter().map(|d| d / g.m2).collect();
    let mut alive = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    // Per row: best partner, an upper bound on the row's best gain, and whether it is exact
    let mut best_k = vec![usize::MAX; n];
    let mut best_dq = vec![f64::NEG_INFINITY; n];
    let mut exact = vec![true; n];
    let mut stamp = vec![0u32; n];
    let gain = |w: f64, a: f64, b: f64| w / m - 2.0 * resolution * a * b;
    let mut heap = std::collections::BinaryHeap::new();

    let recompute = |i: usize,
                     nbr: &[FxHashMap<usize, f64>],
                     share: &[f64],
                     best_k: &mut [usize],
                     best_dq: &mut [f64],
                     exact: &mut [bool]| {
        let mut bk = usize::MAX;
        let mut bd = f64::NEG_INFINITY;
        for (&k, &w) in &nbr[i] {
            let d = gain(w, share[i], share[k]);
            if d > bd || (d == bd && k < bk) {
                bd = d;
                bk = k;
            }
        }
        best_k[i] = bk;
        best_dq[i] = bd;
        exact[i] = true;
    };

    for i in 0..n {
        recompute(i, &nbr, &share, &mut best_k, &mut best_dq, &mut exact);
        if best_k[i] != usize::MAX {
            heap.push(RowTop {
                dq: best_dq[i],
                row: i,
                stamp: 0,
            });
        }
    }

    while let Some(RowTop {
        dq,
        row: i,
        stamp: st,
    }) = heap.pop()
    {
        if !alive[i] || st != stamp[i] {
            continue;
        }
        let p = best_k[i];
        let valid = exact[i]
            && p != usize::MAX
            && alive[p]
            && nbr[i]
                .get(&p)
                .is_some_and(|&w| gain(w, share[i], share[p]) == best_dq[i]);
        if !valid {
            // The bound was stale: find the row's true best and requeue it
            recompute(i, &nbr, &share, &mut best_k, &mut best_dq, &mut exact);
            stamp[i] += 1;
            if best_k[i] != usize::MAX {
                heap.push(RowTop {
                    dq: best_dq[i],
                    row: i,
                    stamp: stamp[i],
                });
            }
            continue;
        }
        if dq <= 0.0 {
            break; // every other row's bound is at most this, so no merge helps
        }

        // Fold the community with fewer neighbors into the other
        let (keep, gone) = if nbr[i].len() >= nbr[p].len() {
            (i, p)
        } else {
            (p, i)
        };
        let moved: Vec<(usize, f64)> = std::mem::take(&mut nbr[gone]).into_iter().collect();
        for &(k, w) in &moved {
            if k == keep {
                continue;
            }
            nbr[k].remove(&gone);
            *nbr[k].entry(keep).or_insert(0.0) += w;
            *nbr[keep].entry(k).or_insert(0.0) += w;
        }
        nbr[keep].remove(&gone);
        share[keep] += share[gone];
        alive[gone] = false;
        parent[gone] = keep;

        // Neighbors of `gone` may now gain more by joining `keep`: raise their bound if so
        for &(k, _) in &moved {
            if k == keep {
                continue;
            }
            let d = gain(nbr[k][&keep], share[k], share[keep]);
            if d >= best_dq[k] {
                best_k[k] = keep;
                best_dq[k] = d;
                exact[k] = true;
                stamp[k] += 1;
                heap.push(RowTop {
                    dq: d,
                    row: k,
                    stamp: stamp[k],
                });
            }
        }
        // `keep`'s other gains only fell, so its old bound still covers them; add the pairs
        // that came over from `gone` and requeue as a bound to be checked when popped
        let mut ub = best_dq[keep];
        let mut uk = best_k[keep];
        for &(k, _) in &moved {
            if k == keep {
                continue;
            }
            let d = gain(nbr[keep][&k], share[keep], share[k]);
            if d > ub {
                ub = d;
                uk = k;
            }
        }
        best_dq[keep] = ub;
        best_k[keep] = uk;
        exact[keep] = false;
        stamp[keep] += 1;
        if !nbr[keep].is_empty() {
            heap.push(RowTop {
                dq: ub,
                row: keep,
                stamp: stamp[keep],
            });
        }
    }
    let find = |mut x: usize| {
        while parent[x] != x {
            x = parent[x];
        }
        x
    };
    let mut part: Vec<usize> = (0..n).map(find).collect();
    renumber(&mut part);
    Ok(membership_map(&ids, &part))
}
