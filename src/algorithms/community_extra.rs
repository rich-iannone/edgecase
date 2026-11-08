//! More community detection and node classification: asynchronous fluid communities,
//! Kernighan-Lin bisection, fast and semi-synchronous label propagation, Lukes' optimal tree
//! partitioning, and the harmonic-function and local-and-global-consistency label spreading
//! classifiers.

use crate::graph::Graph;
use crate::utils::Seed;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn rng_from(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Undirected weighted adjacency on node indices (parallel edges summed, direction ignored).
struct Adj {
    ids: Vec<u64>,
    idx: FxHashMap<u64, usize>,
    nbrs: Vec<Vec<(usize, f64)>>,
}

impl Adj {
    fn new(graph: &Graph, weighted: bool, keep_loops: bool) -> Self {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let mut acc: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
        let mut edges: Vec<_> = graph.get_edges_internal().iter().collect();
        edges.sort_unstable_by_key(|e| e.id);
        let mut order: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in edges {
            let (a, b) = (idx[&e.source], idx[&e.target]);
            if a == b && !keep_loops {
                continue;
            }
            let w = if weighted {
                e.weight.unwrap_or(1.0)
            } else {
                1.0
            };
            for (x, y) in [(a, b), (b, a)] {
                if a == b && x != a {
                    continue;
                }
                if !acc[x].contains_key(&y) {
                    order[x].push(y);
                }
                *acc[x].entry(y).or_insert(0.0) += w;
                if a == b {
                    break;
                }
            }
        }
        let nbrs = (0..n)
            .map(|u| order[u].iter().map(|&v| (v, acc[u][&v])).collect())
            .collect();
        Adj { ids, idx, nbrs }
    }
    fn n(&self) -> usize {
        self.ids.len()
    }
}

fn connected(adj: &Adj) -> bool {
    let n = adj.n();
    if n == 0 {
        return false;
    }
    let mut seen = vec![false; n];
    seen[0] = true;
    let mut q = VecDeque::from([0usize]);
    let mut count = 1;
    while let Some(u) = q.pop_front() {
        for &(v, _) in &adj.nbrs[u] {
            if !seen[v] {
                seen[v] = true;
                count += 1;
                q.push_back(v);
            }
        }
    }
    count == n
}

/// Asynchronous fluid communities (Parés et al., 2017): `k` communities on a connected graph.
/// Returns a community label (0..k) per node, in node ID order.
#[pyfunction]
#[pyo3(signature = (graph, k, max_iter=100, seed=None))]
pub fn fluid_communities(
    graph: &Graph,
    k: usize,
    max_iter: usize,
    seed: Option<Seed>,
) -> PyResult<(Vec<u64>, Vec<usize>)> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return value_err("fluid communities require an undirected graph");
    }
    let adj = Adj::new(graph, false, false);
    let n = adj.n();
    if k == 0 {
        return value_err("k must be at least 1");
    }
    if !connected(&adj) {
        return value_err("fluid communities require a connected graph");
    }
    if k > n {
        return value_err("k cannot exceed the number of nodes");
    }
    let mut rng = rng_from(seed);
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    let mut com: Vec<Option<usize>> = vec![None; n];
    let mut size = vec![1usize; k];
    let mut density = vec![1.0f64; k];
    for (c, &v) in order[..k].iter().enumerate() {
        com[v] = Some(c);
    }
    let mut iterations = 0;
    loop {
        let mut changed = false;
        iterations += 1;
        order.shuffle(&mut rng);
        for &v in &order {
            let mut score: FxHashMap<usize, f64> = FxHashMap::default();
            if let Some(c) = com[v] {
                *score.entry(c).or_insert(0.0) += density[c];
            }
            for &(u, _) in &adj.nbrs[v] {
                if let Some(c) = com[u] {
                    *score.entry(c).or_insert(0.0) += density[c];
                }
            }
            if score.is_empty() {
                continue;
            }
            let top = score.values().cloned().fold(f64::MIN, f64::max);
            let mut best: Vec<usize> = score
                .iter()
                .filter(|(_, &s)| top - s < 1e-4)
                .map(|(&c, _)| c)
                .collect();
            best.sort_unstable();
            if let Some(c) = com[v] {
                if best.contains(&c) {
                    continue;
                }
            }
            changed = true;
            let new = best[rng.gen_range(0..best.len())];
            if let Some(old) = com[v] {
                size[old] -= 1;
                density[old] = if size[old] > 0 {
                    1.0 / size[old] as f64
                } else {
                    1.0
                };
            }
            com[v] = Some(new);
            size[new] += 1;
            density[new] = 1.0 / size[new] as f64;
        }
        if !changed || iterations > max_iter {
            break;
        }
    }
    // relabel in order of the lowest node ID
    let mut map: FxHashMap<usize, usize> = FxHashMap::default();
    let labels = com
        .into_iter()
        .map(|c| {
            let c = c.expect("every node of a connected graph is reached");
            let next = map.len();
            *map.entry(c).or_insert(next)
        })
        .collect();
    Ok((adj.ids, labels))
}

/// Kernighan-Lin bisection: two node sets of (nearly) equal size with a small cut. `initial`
/// optionally gives the first side; otherwise a random half. Returns `(side A, side B)`.
#[pyfunction]
#[pyo3(signature = (graph, initial=None, max_iter=10, weighted=false, seed=None))]
pub fn kernighan_lin(
    graph: &Graph,
    initial: Option<Vec<u64>>,
    max_iter: usize,
    weighted: bool,
    seed: Option<Seed>,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return value_err("Kernighan-Lin bisection requires an undirected graph");
    }
    let adj = Adj::new(graph, weighted, false);
    let n = adj.n();
    let mut rng = rng_from(seed);
    let mut side = vec![false; n];
    match initial {
        Some(list) => {
            for id in list {
                match adj.idx.get(&id) {
                    Some(&i) => side[i] = true,
                    None => return value_err(format!("Node {id} does not exist")),
                }
            }
            // as in NetworkX, the given set starts on the second side, so each swap pass
            // takes its first node from the other set
        }
        None => {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);
            for &v in &order[n / 2..] {
                side[v] = true;
            }
        }
    }
    for _ in 0..max_iter {
        // d[u]: external minus internal weight, i.e. the cut reduction of moving u alone
        let mut d: Vec<f64> = (0..n)
            .map(|u| {
                adj.nbrs[u]
                    .iter()
                    .map(|&(v, w)| if side[v] != side[u] { w } else { -w })
                    .sum()
            })
            .collect();
        let mut locked = vec![false; n];
        let mut swaps: Vec<(usize, usize)> = Vec::new();
        let mut total = 0.0;
        let mut best = (0.0f64, 0usize);
        let take = |s: bool, d: &[f64], locked: &[bool], side: &[bool]| -> Option<usize> {
            (0..n)
                .filter(|&u| side[u] == s && !locked[u])
                .max_by(|&x, &y| d[x].total_cmp(&d[y]).then(y.cmp(&x)))
        };
        loop {
            let Some(a) = take(false, &d, &locked, &side) else {
                break;
            };
            if take(true, &d, &locked, &side).is_none() {
                break;
            }
            // tentatively move a and update its neighbors' gains
            let mut gain = d[a];
            locked[a] = true;
            for &(v, w) in &adj.nbrs[a] {
                if !locked[v] {
                    d[v] += if side[v] == side[a] {
                        2.0 * w
                    } else {
                        -2.0 * w
                    };
                }
            }
            let b = take(true, &d, &locked, &side).unwrap();
            gain += d[b];
            locked[b] = true;
            for &(v, w) in &adj.nbrs[b] {
                if !locked[v] {
                    d[v] += if side[v] == side[b] {
                        2.0 * w
                    } else {
                        -2.0 * w
                    };
                }
            }
            total += gain;
            swaps.push((a, b));
            if total > best.0 + 1e-12 {
                best = (total, swaps.len());
            }
        }
        for &(a, b) in &swaps[..best.1] {
            side[a] = true;
            side[b] = false;
        }
        if best.1 == 0 {
            break;
        }
    }
    // return the side that grew from `initial` (the second side) first
    let a: Vec<u64> = (0..n).filter(|&u| side[u]).map(|u| adj.ids[u]).collect();
    let b: Vec<u64> = (0..n).filter(|&u| !side[u]).map(|u| adj.ids[u]).collect();
    Ok((a, b))
}

/// Greedy node-swap modularity bipartition (Newman 2011, as NetworkX's
/// `greedy_node_swap_bipartition`). Each pass moves every node once, always making the move
/// that raises modularity most (or lowers it least; ties by lowest ID), then keeps the best
/// split seen along the way. Passes stop when one brings no improvement, or after `max_iter`.
/// `initial` gives the first side (otherwise a random half of the nodes). Returns
/// `(first side, second side, modularity)`.
#[pyfunction]
#[pyo3(signature = (graph, initial=None, max_iter=10, weighted=false, seed=None))]
pub fn greedy_node_swap_bipartition(
    graph: &Graph,
    initial: Option<Vec<u64>>,
    max_iter: usize,
    weighted: bool,
    seed: Option<Seed>,
) -> PyResult<(Vec<u64>, Vec<u64>, f64)> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return value_err("greedy_node_swap_bipartition() requires an undirected graph");
    }
    let adj = Adj::new(graph, weighted, true);
    let n = adj.n();
    let mut side = vec![false; n]; // true: the first side
    match initial {
        Some(list) => {
            for id in list {
                match adj.idx.get(&id) {
                    Some(&i) => side[i] = true,
                    None => return value_err(format!("Node {id} does not exist")),
                }
            }
        }
        None => {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng_from(seed));
            for &v in &order[..n / 2] {
                side[v] = true;
            }
        }
    }
    // deg counts a self-loop twice; link[u] holds u's weight to its neighbors, loops left out
    let deg: Vec<f64> = (0..n)
        .map(|u| {
            adj.nbrs[u]
                .iter()
                .map(|&(v, w)| if v == u { 2.0 * w } else { w })
                .sum()
        })
        .collect();
    let m: f64 = deg.iter().sum::<f64>() / 2.0;
    let modularity = |side: &[bool]| -> f64 {
        if m == 0.0 {
            return 0.0;
        }
        let (mut inner, mut d) = (0.0, [0.0f64; 2]);
        for u in 0..n {
            d[side[u] as usize] += deg[u];
            for &(v, w) in &adj.nbrs[u] {
                if side[v] == side[u] {
                    inner += if v == u { 2.0 * w } else { w };
                }
            }
        }
        inner / (2.0 * m) - (d[0] * d[0] + d[1] * d[1]) / (4.0 * m * m)
    };
    if m == 0.0 {
        let first: Vec<u64> = (0..n).filter(|&u| side[u]).map(|u| adj.ids[u]).collect();
        let second: Vec<u64> = (0..n).filter(|&u| !side[u]).map(|u| adj.ids[u]).collect();
        return Ok((first, second, 0.0));
    }
    for _ in 0..max_iter {
        let mut cur = side.clone();
        // link[u][s]: u's weight to side s; dsum[s]: total degree of side s
        let mut link: Vec<[f64; 2]> = vec![[0.0; 2]; n];
        let mut dsum = [0.0f64; 2];
        for u in 0..n {
            dsum[cur[u] as usize] += deg[u];
            for &(v, w) in &adj.nbrs[u] {
                if v != u {
                    link[u][cur[v] as usize] += w;
                }
            }
        }
        let mut moved = vec![false; n];
        let mut moves: Vec<usize> = Vec::with_capacity(n);
        let (mut total, mut best) = (0.0f64, (0.0f64, 0usize));
        for _ in 0..n {
            let mut pick: Option<(usize, f64)> = None;
            for u in (0..n).filter(|&u| !moved[u]) {
                let (s, o) = (cur[u] as usize, 1 - cur[u] as usize);
                let delta = (link[u][o] - link[u][s]) / m
                    + deg[u] * (dsum[s] - dsum[o] - deg[u]) / (2.0 * m * m);
                if pick.is_none_or(|(_, b)| delta > b) {
                    pick = Some((u, delta));
                }
            }
            let (u, delta) = pick.unwrap();
            let (s, o) = (cur[u] as usize, 1 - cur[u] as usize);
            moved[u] = true;
            cur[u] = !cur[u];
            dsum[s] -= deg[u];
            dsum[o] += deg[u];
            for &(v, w) in &adj.nbrs[u] {
                if v != u {
                    link[v][s] -= w;
                    link[v][o] += w;
                }
            }
            moves.push(u);
            total += delta;
            if total > best.0 + 1e-12 {
                best = (total, moves.len());
            }
        }
        if best.1 == 0 {
            break;
        }
        for &u in &moves[..best.1] {
            side[u] = !side[u];
        }
    }
    let first: Vec<u64> = (0..n).filter(|&u| side[u]).map(|u| adj.ids[u]).collect();
    let second: Vec<u64> = (0..n).filter(|&u| !side[u]).map(|u| adj.ids[u]).collect();
    Ok((first, second, modularity(&side)))
}

/// Fast label propagation (Traag and Šubelj, 2023): a queue of nodes whose neighborhood
/// changed; each takes the label with the most (weighted) neighbors, ties at random.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, seed=None))]
pub fn fast_label_propagation(
    graph: &Graph,
    weighted: bool,
    seed: Option<Seed>,
) -> PyResult<(Vec<u64>, Vec<usize>)> {
    let seed = seed.map(u64::from);
    let adj = Adj::new(graph, weighted, false);
    let n = adj.n();
    let mut rng = rng_from(seed);
    let mut label: Vec<usize> = (0..n).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    let mut queue: VecDeque<usize> = order.into_iter().collect();
    let mut queued = vec![true; n];
    while let Some(v) = queue.pop_front() {
        queued[v] = false;
        if adj.nbrs[v].is_empty() {
            continue;
        }
        let mut freq: FxHashMap<usize, f64> = FxHashMap::default();
        for &(u, w) in &adj.nbrs[v] {
            *freq.entry(label[u]).or_insert(0.0) += w;
        }
        let top = freq.values().cloned().fold(f64::MIN, f64::max);
        let mut best: Vec<usize> = freq
            .iter()
            .filter(|(_, &f)| f == top)
            .map(|(&l, _)| l)
            .collect();
        best.sort_unstable();
        let chosen = best[rng.gen_range(0..best.len())];
        if label[v] != chosen {
            label[v] = chosen;
            for &(u, _) in &adj.nbrs[v] {
                if label[u] != chosen && !queued[u] {
                    queued[u] = true;
                    queue.push_back(u);
                }
            }
        }
    }
    let mut map: FxHashMap<usize, usize> = FxHashMap::default();
    let labels = label
        .into_iter()
        .map(|l| {
            let next = map.len();
            *map.entry(l).or_insert(next)
        })
        .collect();
    Ok((adj.ids, labels))
}

/// Semi-synchronous label propagation (Cordasco and Gargano, 2010), NetworkX's
/// `label_propagation_communities`.
///
/// The nodes are colored greedily, highest degree first (ties by ID; a self-loop counts twice),
/// so no two neighbors share a color. Every node starts with its own label; then, color class
/// by color class, each node takes the label most common among its distinct neighbors. On a
/// tie it keeps its label if that is among the best and otherwise takes the highest
/// (Prec-Max). Nodes of one color are never neighbors, so updating a class at once is the
/// same as updating its nodes one by one. Stops when every node with neighbors holds one of
/// its most common neighbor labels. Edge weights and direction are ignored; deterministic.
#[pyfunction]
pub fn semi_synchronous_label_propagation(graph: &Graph) -> PyResult<(Vec<u64>, Vec<usize>)> {
    let adj = Adj::new(graph, false, true);
    let n = adj.n();
    let nbrs: Vec<Vec<usize>> = adj
        .nbrs
        .iter()
        .map(|l| l.iter().map(|&(u, _)| u).collect())
        .collect();

    // greedy coloring, largest degree first (NetworkX's greedy_color default)
    let degree = |v: usize| nbrs[v].len() + usize::from(nbrs[v].contains(&v));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| std::cmp::Reverse(degree(v)));
    let mut color = vec![usize::MAX; n];
    let mut classes: Vec<Vec<usize>> = Vec::new();
    let mut taken = vec![usize::MAX; n + 1];
    for &v in &order {
        for &u in &nbrs[v] {
            if color[u] != usize::MAX {
                taken[color[u]] = v;
            }
        }
        let c = (0..).find(|&c| taken[c] != v).unwrap();
        color[v] = c;
        if c == classes.len() {
            classes.push(Vec::new());
        }
        classes[c].push(v);
    }

    let mut label: Vec<usize> = (0..n).collect();
    let mut freq = vec![0usize; n];
    // the labels with the highest count among v's neighbors (v's own label if it has none)
    let best = |v: usize, label: &[usize], freq: &mut [usize]| -> Vec<usize> {
        if nbrs[v].is_empty() {
            return vec![label[v]];
        }
        for &u in &nbrs[v] {
            freq[label[u]] += 1;
        }
        let top = nbrs[v].iter().map(|&u| freq[label[u]]).max().unwrap_or(0);
        let mut out: Vec<usize> = nbrs[v]
            .iter()
            .map(|&u| label[u])
            .filter(|&l| freq[l] == top)
            .collect();
        for &u in &nbrs[v] {
            freq[label[u]] = 0;
        }
        out.sort_unstable();
        out.dedup();
        out
    };
    loop {
        let done =
            (0..n).all(|v| nbrs[v].is_empty() || best(v, &label, &mut freq).contains(&label[v]));
        if done {
            break;
        }
        for class in &classes {
            for &v in class {
                let high = best(v, &label, &mut freq);
                if high.len() == 1 {
                    label[v] = high[0];
                } else if !high.contains(&label[v]) {
                    label[v] = *high.last().unwrap();
                }
            }
        }
    }
    let mut map: FxHashMap<usize, usize> = FxHashMap::default();
    let labels = label
        .into_iter()
        .map(|l| {
            let next = map.len();
            *map.entry(l).or_insert(next)
        })
        .collect();
    Ok((adj.ids, labels))
}

fn attr_f64(py: Python, attrs: &Option<Py<PyDict>>, name: &str) -> PyResult<Option<f64>> {
    match attrs {
        Some(d) => match d.bind(py).downcast::<PyDict>()?.get_item(name)? {
            Some(v) => Ok(Some(v.extract::<f64>()?)),
            None => Ok(None),
        },
        None => Ok(None),
    }
}

/// Lukes' algorithm: the partition of a tree into clusters of total node weight at most
/// `max_size` that maximizes the total weight of the edges kept inside clusters. Node weights
/// must be non-negative integers (1 by default); edge weights default to the weight field (1
/// where an edge has none).
/// Returns `(clusters, kept edge weight)`.
#[pyfunction]
#[pyo3(signature = (graph, max_size, node_weight=None, edge_weight=None))]
pub fn lukes_partitioning(
    py: Python,
    graph: &Graph,
    max_size: usize,
    node_weight: Option<&str>,
    edge_weight: Option<&str>,
) -> PyResult<(Vec<Vec<u64>>, f64)> {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let n = ids.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let edges = graph.get_edges_internal();
    if edges.len() != n - 1 {
        return value_err("Lukes partitioning requires a tree");
    }
    let nodes = graph.get_nodes();
    let mut nw = vec![1usize; n];
    if let Some(attr) = node_weight {
        for (i, id) in ids.iter().enumerate() {
            match attr_f64(py, &nodes[id].attributes, attr)? {
                Some(x) if x >= 0.0 && x.fract() == 0.0 => nw[i] = x as usize,
                Some(_) => return value_err("node weights must be non-negative integers"),
                None => return value_err(format!("node {id} has no attribute {attr:?}")),
            }
        }
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for e in edges {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a == b {
            return value_err("Lukes partitioning requires a tree");
        }
        let w = match edge_weight {
            // "weight" also falls back to the weight field, where weighted graphs keep it
            Some(attr) => crate::utils::require_named_edge_weight(py, e, attr, "weight_attr")?,
            None => e.weight.unwrap_or(1.0),
        };
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    // root at node 0; children order from BFS, which also checks connectivity
    let mut parent = vec![usize::MAX; n];
    let mut order = vec![0usize];
    let mut seen = vec![false; n];
    seen[0] = true;
    let mut head = 0;
    while head < order.len() {
        let u = order[head];
        head += 1;
        for &(v, _) in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                parent[v] = u;
                order.push(v);
            }
        }
    }
    if order.len() != n {
        return value_err("Lukes partitioning requires a tree");
    }
    if nw.iter().any(|&w| w > max_size) {
        return value_err("a node weighs more than max_size");
    }
    let k = max_size;
    const NEG: f64 = f64::NEG_INFINITY;
    // dp[v][w]: best kept weight in v's subtree when v's cluster weighs exactly w
    let mut dp: Vec<Vec<f64>> = vec![Vec::new(); n];
    // choice[v] records, per child in processing order, the table before the merge, for rebuild
    let mut history: Vec<Vec<(usize, f64, Vec<f64>)>> = vec![Vec::new(); n];
    for &v in order.iter().rev() {
        let mut table = vec![NEG; k + 1];
        table[nw[v]] = 0.0;
        for &(c, w) in &adj[v] {
            if c == parent[v] {
                continue;
            }
            let child = &dp[c];
            let best_child = child.iter().cloned().fold(NEG, f64::max);
            let mut next = vec![NEG; k + 1];
            for a in 0..=k {
                if table[a] == NEG {
                    continue;
                }
                // cut the edge to the child
                if table[a] + best_child > next[a] {
                    next[a] = table[a] + best_child;
                }
                // merge the child's cluster into v's
                for b in 0..=k - a {
                    if child[b] == NEG {
                        continue;
                    }
                    let val = table[a] + child[b] + w;
                    if val > next[a + b] {
                        next[a + b] = val;
                    }
                }
            }
            history[v].push((c, w, table));
            table = next;
        }
        dp[v] = table;
    }
    // reconstruct
    let mut cluster_of = vec![usize::MAX; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let root_best = (0..=k)
        .max_by(|&a, &b| dp[0][a].total_cmp(&dp[0][b]))
        .unwrap();
    let total = dp[0][root_best];
    let mut stack: Vec<(usize, usize, usize)> = vec![(0, root_best, usize::MAX)]; // node, weight, cluster
    while let Some((v, target, cl)) = stack.pop() {
        let cluster = if cl == usize::MAX {
            clusters.push(Vec::new());
            clusters.len() - 1
        } else {
            cl
        };
        cluster_of[v] = cluster;
        clusters[cluster].push(v);
        let mut want = target;
        let mut value = dp[v][want];
        for (c, w, before) in history[v].iter().rev() {
            let child = &dp[*c];
            let best_child = child.iter().cloned().fold(NEG, f64::max);
            // was it a cut?
            if before[want] != NEG && (before[want] + best_child - value).abs() < 1e-9 {
                let cb = (0..=k).find(|&b| child[b] == best_child).unwrap();
                stack.push((*c, cb, usize::MAX));
                value = before[want];
                continue;
            }
            let mut done = false;
            for b in 0..=want {
                let a = want - b;
                if before[a] != NEG
                    && child[b] != NEG
                    && (before[a] + child[b] + w - value).abs() < 1e-9
                {
                    stack.push((*c, b, cluster));
                    value = before[a];
                    done = true;
                    want = a;
                    break;
                }
            }
            debug_assert!(done);
        }
    }
    let _ = cluster_of;
    let mut out: Vec<Vec<u64>> = clusters
        .into_iter()
        .map(|c| {
            let mut ids_c: Vec<u64> = c.into_iter().map(|v| ids[v]).collect();
            ids_c.sort_unstable();
            ids_c
        })
        .collect();
    out.sort();
    Ok((out, total))
}

/// Semi-supervised node classification by label spreading: `"harmonic"` (Zhu, Ghahramani and
/// Lafferty, 2003) or `"consistency"` (Zhou et al., 2004), iterated `max_iter` times exactly as
/// in NetworkX. `labels[i]` is the class index of node i (in ID order) or -1 if unlabeled.
/// Returns the predicted class index per node.
#[pyfunction]
#[pyo3(signature = (graph, labels, n_classes, kind, alpha=0.99, max_iter=30, weighted=false))]
pub fn spread_labels(
    graph: &Graph,
    labels: Vec<i64>,
    n_classes: usize,
    kind: &str,
    alpha: f64,
    max_iter: usize,
    weighted: bool,
) -> PyResult<Vec<usize>> {
    if graph.is_directed() {
        return value_err("node classification requires an undirected graph");
    }
    let adj = Adj::new(graph, weighted, true);
    let n = adj.n();
    if labels.len() != n {
        return value_err("one label entry per node is required");
    }
    let degree: Vec<f64> = (0..n)
        .map(|u| {
            let d: f64 = adj.nbrs[u].iter().map(|&(_, w)| w).sum();
            if d == 0.0 {
                1.0
            } else {
                d
            }
        })
        .collect();
    let harmonic = match kind {
        "harmonic" => true,
        "consistency" => false,
        _ => return value_err(format!("unknown method {kind:?}")),
    };
    let mut f = vec![vec![0.0f64; n_classes]; n];
    let mut b = vec![vec![0.0f64; n_classes]; n];
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            b[i][l as usize] = if harmonic { 1.0 } else { 1.0 - alpha };
        }
    }
    for _ in 0..max_iter {
        let mut next = b.clone();
        for u in 0..n {
            if harmonic && labels[u] >= 0 {
                continue; // labeled rows of P are zero
            }
            for &(v, w) in &adj.nbrs[u] {
                let coef = if harmonic {
                    w / degree[u]
                } else {
                    alpha * w / (degree[u].sqrt() * degree[v].sqrt())
                };
                for c in 0..n_classes {
                    next[u][c] += coef * f[v][c];
                }
            }
        }
        f = next;
    }
    Ok(f.iter()
        .map(|row| {
            // near-ties (equal up to rounding) go to the class seen first
            let top = row.iter().cloned().fold(f64::MIN, f64::max);
            let tol = 1e-9 * top.abs().max(1e-300);
            (0..n_classes).find(|&c| row[c] >= top - tol).unwrap_or(0)
        })
        .collect())
}
