//! Random graph models: growth processes, degree sequences, directed growth, and
//! geometric graphs.
//!
//! Like `generators`, these return plain index structures over nodes `0..n` (edge
//! pairs and, for the geometric models, node positions). The Python wrappers add the
//! nodes and wire the edges by ID. Symmetric models return each edge as
//! `(lower, higher)`, so that in a directed graph it points from the earlier-created
//! node to the later one, except that the growth models with a newcomer
//! (`powerlaw_cluster_edges`, `dual_barabasi_albert_edges`) point from the new node to
//! the node it attaches to, like `Graph.add_pa_graph()`. Inherently directed models
//! return `(source, target)`.
//!
//! - random: `random_regular_edges`, `powerlaw_cluster_edges`,
//!   `newman_watts_strogatz_edges`, `connected_watts_strogatz_edges`,
//!   `extended_barabasi_albert_edges`, `dual_barabasi_albert_edges`,
//!   `random_lobster_edges`, `random_labeled_tree_edges`, `random_powerlaw_tree_edges`
//! - degree sequence: `havel_hakimi_edges`, `directed_configuration_edges`,
//!   `directed_havel_hakimi_edges`, `random_degree_sequence_edges`, `joint_degree_edges`
//! - directed growth: `gn_edges`, `gnr_edges`, `gnc_edges`, `scale_free_edges`,
//!   `random_k_out_edges`
//! - geometric: `waxman_edges`, `soft_random_geometric_edges`,
//!   `geographical_threshold_edges`, `navigable_small_world_edges`

use crate::utils::Seed;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use super::generators::{check_probability, key, make_rng, EdgeList};

type Positions = Vec<Vec<f64>>;

/// Consecutive rejections after which a rejection-sampling loop gives up.
const STALL_LIMIT: usize = 1_000_000;

fn value_error<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

/// Sorted list of the edges in an adjacency structure, each as `(lower, higher)`.
fn edges_from_adj(adj: &[FxHashSet<usize>]) -> EdgeList {
    let mut edges: EdgeList = adj
        .iter()
        .enumerate()
        .flat_map(|(u, nbrs)| nbrs.iter().filter(move |&&v| u < v).map(move |&v| (u, v)))
        .collect();
    edges.sort_unstable();
    edges
}

/// `m` distinct elements of `seq` drawn uniformly with replacement (so an element
/// repeated in `seq` is proportionally more likely), in the order they were drawn.
fn random_subset(rng: &mut ChaCha8Rng, seq: &[usize], m: usize) -> Vec<usize> {
    let mut seen = FxHashSet::default();
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let x = seq[rng.gen_range(0..seq.len())];
        if seen.insert(x) {
            out.push(x);
        }
    }
    out
}

/// True if the undirected graph on `0..n` with these edges is connected.
fn is_connected(n: usize, edges: &[(usize, usize)]) -> bool {
    if n <= 1 {
        return true;
    }
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut components = n;
    for &(a, b) in edges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
            components -= 1;
        }
    }
    components == 1
}

/// Fenwick tree over non-negative weights, for drawing an index with probability
/// proportional to its weight while the weights change.
struct WeightTree {
    tree: Vec<f64>,
    weight: Vec<f64>,
}

impl WeightTree {
    fn new(n: usize) -> Self {
        WeightTree {
            tree: vec![0.0; n + 1],
            weight: vec![0.0; n],
        }
    }

    fn set(&mut self, i: usize, w: f64) {
        let delta = w - self.weight[i];
        self.weight[i] = w;
        let mut j = i + 1;
        while j < self.tree.len() {
            self.tree[j] += delta;
            j += j & j.wrapping_neg();
        }
    }

    fn total(&self) -> f64 {
        let mut j = self.weight.len();
        let mut sum = 0.0;
        while j > 0 {
            sum += self.tree[j];
            j &= j - 1;
        }
        sum
    }

    /// An index drawn with probability proportional to its weight, or `None` if every
    /// weight is zero.
    fn sample(&self, rng: &mut ChaCha8Rng) -> Option<usize> {
        let n = self.weight.len();
        let total = self.total();
        if total.is_nan() || total <= 0.0 {
            return None;
        }
        let mut target = rng.gen::<f64>() * total;
        let mut pos = 0;
        let mut step = n.next_power_of_two();
        while step > 0 {
            let next = pos + step;
            if next <= n && self.tree[next] <= target {
                pos = next;
                target -= self.tree[next];
            }
            step >>= 1;
        }
        // rounding can land on a zero-weight index at a boundary; step to the nearest
        // index that has weight
        if pos < n && self.weight[pos] > 0.0 {
            return Some(pos);
        }
        (0..pos.min(n))
            .rev()
            .chain(pos.min(n)..n)
            .find(|&i| self.weight[i] > 0.0)
    }
}

/// Minkowski distance of order `p` (`p = inf` is the Chebyshev distance).
fn minkowski(a: &[f64], b: &[f64], p: f64) -> f64 {
    let diffs = a.iter().zip(b).map(|(x, y)| (x - y).abs());
    if p.is_infinite() {
        diffs.fold(0.0, f64::max)
    } else if p == 2.0 {
        diffs.map(|d| d * d).sum::<f64>().sqrt()
    } else if p == 1.0 {
        diffs.sum()
    } else {
        diffs.map(|d| d.powf(p)).sum::<f64>().powf(1.0 / p)
    }
}

fn uniform_positions(rng: &mut ChaCha8Rng, n: usize, dim: usize) -> Positions {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f64>()).collect())
        .collect()
}

fn check_geometry(dim: usize, p_norm: f64) -> PyResult<()> {
    if dim < 1 {
        return value_error("dim must be at least 1");
    }
    if p_norm.is_nan() || p_norm <= 0.0 {
        return value_error(format!("p_norm must be positive, got {}", p_norm));
    }
    Ok(())
}

/// Erdős-Gallai test: can `degrees` be the degree sequence of a simple graph?
fn is_graphical(degrees: &[usize]) -> bool {
    let n = degrees.len();
    if degrees.iter().sum::<usize>() % 2 == 1 || degrees.iter().any(|&d| d >= n) {
        return false;
    }
    let mut s = degrees.to_vec();
    s.sort_unstable_by(|a, b| b.cmp(a));
    let mut suffix = vec![0usize; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] + s[i];
    }
    let mut at_least_k = n; // count of entries >= k
    let mut left = 0;
    for k in 1..=n {
        left += s[k - 1];
        while at_least_k > 0 && s[at_least_k - 1] < k {
            at_least_k -= 1;
        }
        let right = if at_least_k > k {
            k * (at_least_k - k) + suffix[at_least_k]
        } else {
            suffix[k]
        };
        if left > k * (k - 1) + right {
            return false;
        }
    }
    true
}

/// Validate a degree sequence given as Python integers.
pub(crate) fn degree_list(name: &str, seq: &[i64]) -> PyResult<Vec<usize>> {
    seq.iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                PyValueError::new_err(format!(
                    "{} must hold non-negative integers, got {}",
                    name, d
                ))
            })
        })
        .collect()
}

// --- random graphs -------------------------------------------------------------

/// Random `d`-regular graph on `n` nodes (Steger and Wormald 1999).
///
/// Shuffles `d` stubs per node and pairs them, keeping pairs that are neither loops
/// nor repeats; the leftover stubs are re-paired the same way. An attempt fails when
/// the leftover stubs can't form any new edge, and the whole construction restarts.
#[pyfunction]
#[pyo3(signature = (n, d, seed=None))]
pub fn random_regular_edges(n: usize, d: usize, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if (n * d) % 2 != 0 {
        return value_error(format!("n * d must be even, got n={} and d={}", n, d));
    }
    if d > 0 && d >= n {
        return value_error(format!("d must be less than n, got n={} and d={}", n, d));
    }
    if d == 0 {
        return Ok(Vec::new());
    }
    let mut rng = make_rng(seed);
    for _attempt in 0..10_000 {
        if let Some(edges) = try_regular(&mut rng, n, d) {
            let mut edges: EdgeList = edges.into_iter().collect();
            edges.sort_unstable();
            return Ok(edges);
        }
    }
    value_error(format!(
        "could not build a {}-regular graph on {} nodes in 10000 attempts",
        d, n
    ))
}

fn try_regular(rng: &mut ChaCha8Rng, n: usize, d: usize) -> Option<FxHashSet<(usize, usize)>> {
    let mut edges: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut stubs: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat(v).take(d)).collect();
    while !stubs.is_empty() {
        stubs.shuffle(rng);
        let mut leftover: BTreeMap<usize, usize> = BTreeMap::new();
        for pair in stubs.chunks_exact(2) {
            let (a, b) = key(pair[0], pair[1]);
            if a == b || !edges.insert((a, b)) {
                *leftover.entry(a).or_insert(0) += 1;
                *leftover.entry(b).or_insert(0) += 1;
            }
        }
        let nodes: Vec<usize> = leftover.keys().copied().collect();
        let suitable = nodes.is_empty()
            || nodes
                .iter()
                .enumerate()
                .any(|(i, &a)| nodes[..i].iter().any(|&b| !edges.contains(&key(a, b))));
        if !suitable {
            return None;
        }
        stubs = leftover
            .iter()
            .flat_map(|(&v, &c)| std::iter::repeat(v).take(c))
            .collect();
    }
    Some(edges)
}

/// Holme-Kim powerlaw cluster graph: preferential attachment with triad formation.
///
/// Starts from `m` isolated nodes. Each new node picks `m` distinct targets by
/// preferential attachment and links to the first; for each further link, with
/// probability `p` it instead closes a triangle by linking to a random neighbor of the
/// last target (one it isn't already linked to), otherwise it links to the next target.
/// Edges are returned as `(new node, target)`, like `Graph.add_pa_graph()`.
#[pyfunction]
#[pyo3(signature = (n, m, p, seed=None))]
pub fn powerlaw_cluster_edges(
    n: usize,
    m: usize,
    p: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if m < 1 || n < m {
        return value_error(format!(
            "m must be at least 1 and at most n, got n={} and m={}",
            n, m
        ));
    }
    check_probability("p", p)?;
    let mut rng = make_rng(seed);
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let mut edges: EdgeList = Vec::new();
    let mut link = |adj: &mut Vec<FxHashSet<usize>>, source: usize, target: usize| {
        // a planned target may already have been reached by closing a triangle
        if adj[source].insert(target) {
            adj[target].insert(source);
            edges.push((source, target));
        }
    };
    let mut repeated: Vec<usize> = (0..m).collect();
    for source in m..n {
        let mut targets = random_subset(&mut rng, &repeated, m);
        let mut target = targets.pop().unwrap();
        link(&mut adj, source, target);
        repeated.push(target);
        let mut count = 1;
        while count < m {
            if rng.gen::<f64>() < p {
                let mut hood: Vec<usize> = adj[target]
                    .iter()
                    .copied()
                    .filter(|&x| x != source && !adj[source].contains(&x))
                    .collect();
                hood.sort_unstable();
                if let Some(&nbr) = hood.choose(&mut rng) {
                    link(&mut adj, source, nbr);
                    repeated.push(nbr);
                    count += 1;
                    continue;
                }
            }
            target = targets.pop().unwrap();
            link(&mut adj, source, target);
            repeated.push(target);
            count += 1;
        }
        repeated.extend(std::iter::repeat(source).take(m));
    }
    Ok(edges)
}

/// Newman-Watts-Strogatz graph: a ring lattice plus random shortcuts.
///
/// Each node is joined to its `k / 2` nearest neighbors on each side; then, for each
/// lattice edge `(u, v)`, with probability `p` a new edge `(u, w)` is added to a node
/// `w` chosen uniformly among those not already adjacent to `u`. `k == n` gives the
/// complete graph.
#[pyfunction]
#[pyo3(signature = (n, k, p, seed=None))]
pub fn newman_watts_strogatz_edges(
    n: usize,
    k: usize,
    p: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if k > n {
        return value_error(format!("k must be at most n, got n={} and k={}", n, k));
    }
    check_probability("p", p)?;
    if k == n {
        return Ok((0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect());
    }
    let mut rng = make_rng(seed);
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let mut lattice: EdgeList = Vec::new();
    for j in 1..=(k / 2) {
        for i in 0..n {
            let v = (i + j) % n;
            if v != i && adj[i].insert(v) {
                adj[v].insert(i);
                lattice.push((i, v));
            }
        }
    }
    for (u, _) in lattice {
        if rng.gen::<f64>() >= p || adj[u].len() + 1 >= n {
            continue;
        }
        loop {
            let w = rng.gen_range(0..n);
            if w != u && !adj[u].contains(&w) {
                adj[u].insert(w);
                adj[w].insert(u);
                break;
            }
        }
    }
    Ok(edges_from_adj(&adj))
}

/// The Watts-Strogatz process behind `Graph.add_smallworld_graph()`.
///
/// Ring edges `(i, i + j mod n)` for `j` in `1..=k/2`, visited in order; with
/// probability `p` the far endpoint is replaced by a uniformly random node that is
/// neither `u` nor adjacent to it (100 draws, after which the edge is kept). Callers
/// validate `k < n`, `k` even, and `p` in `[0, 1]`.
pub(crate) fn watts_strogatz(rng: &mut ChaCha8Rng, n: usize, k: usize, p: f64) -> EdgeList {
    let mut edges: EdgeList = Vec::with_capacity(n * (k / 2));
    for i in 0..n {
        for j in 1..=(k / 2) {
            edges.push((i, (i + j) % n));
        }
    }
    // `present` holds every edge of the graph as it currently stands, including
    // lattice edges not yet visited, so a rewired edge never duplicates a later one.
    let mut present: FxHashSet<(usize, usize)> = edges.iter().map(|&(u, v)| key(u, v)).collect();
    for edge in edges.iter_mut() {
        let (u, v) = *edge;
        if rng.gen::<f64>() < p {
            let mut w = rng.gen_range(0..n);
            let mut attempts = 0;
            while (w == u || present.contains(&key(u, w))) && attempts < 100 {
                w = rng.gen_range(0..n);
                attempts += 1;
            }
            // otherwise (no free target found) the original edge is kept
            if attempts < 100 {
                present.remove(&key(u, v));
                present.insert(key(u, w));
                *edge = (u, w);
            }
        }
    }
    edges
}

/// Watts-Strogatz graph redrawn (continuing the same random stream) until connected.
#[pyfunction]
#[pyo3(signature = (n, k, p, tries=100, seed=None))]
pub fn connected_watts_strogatz_edges(
    n: usize,
    k: usize,
    p: f64,
    tries: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if k >= n {
        return value_error(format!("k must be less than n, got n={} and k={}", n, k));
    }
    if k % 2 != 0 {
        return value_error(format!("k must be even, got {}", k));
    }
    check_probability("p", p)?;
    if tries < 1 {
        return value_error("tries must be at least 1");
    }
    let mut rng = make_rng(seed);
    for _ in 0..tries {
        let edges = watts_strogatz(&mut rng, n, k, p);
        if is_connected(n, &edges) {
            return Ok(edges);
        }
    }
    value_error(format!(
        "no connected Watts-Strogatz graph found in {} tries; raise k or tries, or lower p",
        tries
    ))
}

/// Extended Barabási-Albert graph (Albert and Barabási 2000).
///
/// Starting from `m` isolated nodes, each step is one of: with probability `p`, add
/// `m` new edges between existing nodes; with probability `q`, rewire `m` existing
/// edges; otherwise add a new node with `m` edges. Endpoints are chosen by
/// preferential attachment (weight `degree + 1`).
#[pyfunction]
#[pyo3(signature = (n, m, p, q, seed=None))]
pub fn extended_barabasi_albert_edges(
    n: usize,
    m: usize,
    p: f64,
    q: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if m < 1 || m >= n {
        return value_error(format!(
            "m must be at least 1 and less than n, got n={} and m={}",
            n, m
        ));
    }
    check_probability("p", p)?;
    check_probability("q", q)?;
    if p + q >= 1.0 {
        return value_error(format!(
            "p + q must be less than 1, got p={} and q={}",
            p, q
        ));
    }
    let mut rng = make_rng(seed);
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let mut size = 0usize; // edge count
                           // each node appears once plus once per incident edge
    let mut preference: Vec<usize> = (0..m).collect();
    let mut new_node = m;
    while new_node < n {
        let a: f64 = rng.gen();
        let clique_degree = new_node - 1;
        let clique_size = new_node * clique_degree / 2;
        if a < p && size + m <= clique_size {
            // add m edges between existing nodes
            let mut eligible: Vec<usize> = (0..new_node)
                .filter(|&v| adj[v].len() < clique_degree)
                .collect();
            for _ in 0..m {
                let Some(&src) = eligible.choose(&mut rng) else {
                    break;
                };
                let candidates: Vec<usize> = preference
                    .iter()
                    .copied()
                    .filter(|&v| v != src && !adj[src].contains(&v))
                    .collect();
                let Some(&dest) = candidates.choose(&mut rng) else {
                    break;
                };
                adj[src].insert(dest);
                adj[dest].insert(src);
                size += 1;
                preference.push(src);
                preference.push(dest);
                if adj[src].len() == clique_degree {
                    eligible.retain(|&v| v != src);
                }
                if adj[dest].len() == clique_degree {
                    eligible.retain(|&v| v != dest);
                }
            }
        } else if p <= a && a < p + q && m <= size && size < clique_size {
            // rewire m edges
            let mut eligible: Vec<usize> = (0..new_node)
                .filter(|&v| !adj[v].is_empty() && adj[v].len() < clique_degree)
                .collect();
            for _ in 0..m {
                let Some(&node) = eligible.choose(&mut rng) else {
                    break;
                };
                let mut nbrs: Vec<usize> = adj[node].iter().copied().collect();
                nbrs.sort_unstable();
                let src = *nbrs.choose(&mut rng).unwrap();
                let candidates: Vec<usize> = preference
                    .iter()
                    .copied()
                    .filter(|&v| v != node && !adj[node].contains(&v))
                    .collect();
                let Some(&dest) = candidates.choose(&mut rng) else {
                    break;
                };
                adj[node].remove(&src);
                adj[src].remove(&node);
                adj[node].insert(dest);
                adj[dest].insert(node);
                if let Some(i) = preference.iter().position(|&v| v == src) {
                    preference.remove(i);
                }
                preference.push(dest);
                if adj[src].is_empty() {
                    eligible.retain(|&v| v != src);
                }
                if eligible.contains(&dest) {
                    if adj[dest].len() == clique_degree {
                        eligible.retain(|&v| v != dest);
                    }
                } else if adj[dest].len() == 1 {
                    eligible.push(dest);
                }
            }
        } else {
            // add a node with m edges
            let targets = random_subset(&mut rng, &preference, m);
            for &t in &targets {
                adj[new_node].insert(t);
                adj[t].insert(new_node);
            }
            size += m;
            preference.extend(&targets);
            preference.extend(std::iter::repeat(new_node).take(m + 1));
            new_node += 1;
        }
    }
    Ok(edges_from_adj(&adj))
}

/// Dual Barabási-Albert graph (Moshiri 2018).
///
/// Grows from a star on `max(m1, m2) + 1` nodes; each new node attaches with `m1`
/// edges with probability `p` and with `m2` edges otherwise, choosing distinct targets
/// with probability proportional to degree. With `p` of `1` (or `0`) it is the
/// Barabási-Albert model with `m1` (or `m2`), started from a star on `m1 + 1` (or
/// `m2 + 1`) nodes. Star edges are returned as `(center, leaf)` and the rest as
/// `(new node, target)`, like `Graph.add_pa_graph()`.
#[pyfunction]
#[pyo3(signature = (n, m1, m2, p, seed=None))]
pub fn dual_barabasi_albert_edges(
    n: usize,
    m1: usize,
    m2: usize,
    p: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    for (name, m) in [("m1", m1), ("m2", m2)] {
        if m < 1 || m >= n {
            return value_error(format!(
                "{} must be at least 1 and less than n, got n={} and {}={}",
                name, n, name, m
            ));
        }
    }
    check_probability("p", p)?;
    let m0 = if p == 1.0 {
        m1
    } else if p == 0.0 {
        m2
    } else {
        m1.max(m2)
    };
    let mut rng = make_rng(seed);
    let mut edges: EdgeList = (1..=m0).map(|leaf| (0, leaf)).collect();
    let mut repeated: Vec<usize> = std::iter::repeat(0).take(m0).chain(1..=m0).collect();
    for source in (m0 + 1)..n {
        let m = if rng.gen::<f64>() < p { m1 } else { m2 };
        let targets = random_subset(&mut rng, &repeated, m);
        edges.extend(targets.iter().map(|&t| (source, t)));
        repeated.extend(&targets);
        repeated.extend(std::iter::repeat(source).take(m));
    }
    Ok(edges)
}

/// Random lobster: a path with random legs, each of which may carry random leaves.
///
/// The backbone has `int(2 * U * n + 0.5)` nodes for a uniform `U`, so its length is
/// `n` on average. Each backbone node grows a leg while a draw falls below `p1`, and
/// each leg grows a leaf while a draw falls below `p2`. Returns the node count and
/// the edges.
#[pyfunction]
#[pyo3(signature = (n, p1, p2, seed=None))]
pub fn random_lobster_edges(
    n: usize,
    p1: f64,
    p2: f64,
    seed: Option<Seed>,
) -> PyResult<(usize, EdgeList)> {
    let seed = seed.map(u64::from);
    for (name, p) in [("p1", p1), ("p2", p2)] {
        if !(0.0..1.0).contains(&p) {
            return value_error(format!(
                "{} must be at least 0 and less than 1, got {}",
                name, p
            ));
        }
    }
    let mut rng = make_rng(seed);
    let backbone = (2.0 * rng.gen::<f64>() * n as f64 + 0.5) as usize;
    if backbone == 0 {
        return Ok((0, Vec::new()));
    }
    let mut edges: EdgeList = (1..backbone).map(|i| (i - 1, i)).collect();
    let mut last = backbone - 1;
    for b in 0..backbone {
        while rng.gen::<f64>() < p1 {
            last += 1;
            edges.push((b, last));
            let leg = last;
            while rng.gen::<f64>() < p2 {
                last += 1;
                edges.push((leg, last));
            }
        }
    }
    Ok((last + 1, edges))
}

/// Tree decoded from a Prüfer sequence over `0..n` (linear-time decoding).
fn prufer_tree(n: usize, seq: &[usize]) -> EdgeList {
    let mut degree = vec![1usize; n];
    for &v in seq {
        degree[v] += 1;
    }
    let mut edges = Vec::with_capacity(n - 1);
    let mut index = (0..n).find(|&k| degree[k] == 1).unwrap();
    let mut leaf = index;
    for &v in seq {
        edges.push(key(leaf, v));
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            leaf = v;
        } else {
            index = ((index + 1)..n).find(|&k| degree[k] == 1).unwrap();
            leaf = index;
        }
    }
    edges.push(key(leaf, n - 1));
    edges.sort_unstable();
    edges
}

/// Uniformly random labeled tree on `n` nodes, from a uniformly random Prüfer sequence.
#[pyfunction]
#[pyo3(signature = (n, seed=None))]
pub fn random_labeled_tree_edges(n: usize, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if n == 0 {
        return value_error("a tree needs at least 1 node, got n=0");
    }
    if n == 1 {
        return Ok(Vec::new());
    }
    let mut rng = make_rng(seed);
    let seq: Vec<usize> = (0..n - 2).map(|_| rng.gen_range(0..n)).collect();
    Ok(prufer_tree(n, &seq))
}

/// Tree with a power-law degree sequence.
///
/// Draws `n` degrees from a Pareto distribution with exponent `gamma` (rounded and
/// capped at `n`), then, until the degrees sum to `2n - 2`, replaces a random entry
/// with a fresh draw, up to `tries` times. The tree is built from the sequence by
/// joining the high-degree nodes in a path and hanging leaves off it.
#[pyfunction]
#[pyo3(signature = (n, gamma=3.0, tries=100, seed=None))]
pub fn random_powerlaw_tree_edges(
    n: usize,
    gamma: f64,
    tries: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if n == 0 {
        return value_error("a tree needs at least 1 node, got n=0");
    }
    if !(gamma.is_finite() && gamma > 1.0) {
        return value_error(format!("gamma must be greater than 1, got {}", gamma));
    }
    if n == 1 {
        return Ok(Vec::new());
    }
    let mut rng = make_rng(seed);
    let alpha = gamma - 1.0;
    let draw = |rng: &mut ChaCha8Rng| {
        let u = 1.0 - rng.gen::<f64>();
        (u.powf(-1.0 / alpha).round() as usize).min(n)
    };
    let mut seq: Vec<usize> = (0..n).map(|_| draw(&mut rng)).collect();
    for _ in 0..tries {
        if seq.iter().sum::<usize>() == 2 * n - 2 {
            return Ok(degree_sequence_tree(&seq));
        }
        let i = rng.gen_range(0..n);
        seq[i] = draw(&mut rng);
    }
    value_error(format!(
        "no valid tree degree sequence found in {} tries; raise tries or gamma",
        tries
    ))
}

/// Tree realizing a degree sequence of positive entries summing to `2(len - 1)`:
/// the internal nodes form a path, and each takes its remaining degree in leaves.
fn degree_sequence_tree(seq: &[usize]) -> EdgeList {
    let mut internal: Vec<usize> = seq.iter().copied().filter(|&d| d > 1).collect();
    internal.sort_unstable_by(|a, b| b.cmp(a));
    let path_len = internal.len() + 2;
    let mut edges: EdgeList = (1..path_len).map(|i| (i - 1, i)).collect();
    let mut last = path_len;
    for source in 1..path_len - 1 {
        let extra = internal.pop().unwrap() - 2;
        edges.extend((last..last + extra).map(|t| (source, t)));
        last += extra;
    }
    edges
}

// --- degree sequences ----------------------------------------------------------

/// Havel-Hakimi graph: deterministic simple graph with the given degree sequence.
///
/// Repeatedly takes a node of largest remaining degree `d` and joins it to the `d`
/// nodes of next-largest remaining degree, following NetworkX's
/// `havel_hakimi_graph()`. Node `i` gets degree `degree_sequence[i]`.
#[pyfunction]
pub fn havel_hakimi_edges(degree_sequence: Vec<i64>) -> PyResult<EdgeList> {
    let seq = degree_list("degree_sequence", &degree_sequence)?;
    if !is_graphical(&seq) {
        return value_error("degree_sequence is not graphical: no simple graph has these degrees");
    }
    let p = seq.len();
    // nonzero-degree nodes are numbered by rank, as in NetworkX; map back to positions
    let original: Vec<usize> = (0..p).filter(|&i| seq[i] > 0).collect();
    let mut num_degs: Vec<Vec<usize>> = vec![Vec::new(); p];
    let mut dmax = 0;
    let mut n = 0;
    for &d in &seq {
        if d > 0 {
            num_degs[d].push(n);
            dmax = dmax.max(d);
            n += 1;
        }
    }
    let mut edges = Vec::new();
    let mut modstubs: Vec<(usize, usize)> = vec![(0, 0); dmax + 1];
    while n > 0 {
        while num_degs[dmax].is_empty() {
            dmax -= 1;
        }
        let source = num_degs[dmax].pop().unwrap();
        n -= 1;
        let mut mslen = 0;
        let mut k = dmax;
        for _ in 0..dmax {
            while num_degs[k].is_empty() {
                k -= 1;
            }
            let target = num_degs[k].pop().unwrap();
            edges.push(key(original[source], original[target]));
            n -= 1;
            if k > 1 {
                modstubs[mslen] = (k - 1, target);
                mslen += 1;
            }
        }
        for &(stubval, stubtarget) in &modstubs[..mslen] {
            num_degs[stubval].push(stubtarget);
            n += 1;
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

/// Directed configuration model: random pairing of out-stubs with in-stubs.
///
/// The shorter sequence is padded with zeros. The result is a multigraph: self-loops
/// and parallel edges are kept.
#[pyfunction]
#[pyo3(signature = (in_degree_sequence, out_degree_sequence, seed=None))]
pub fn directed_configuration_edges(
    in_degree_sequence: Vec<i64>,
    out_degree_sequence: Vec<i64>,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let ins = degree_list("in_degree_sequence", &in_degree_sequence)?;
    let outs = degree_list("out_degree_sequence", &out_degree_sequence)?;
    let (sum_in, sum_out) = (ins.iter().sum::<usize>(), outs.iter().sum::<usize>());
    if sum_in != sum_out {
        return value_error(format!(
            "the in-degree and out-degree sequences must have equal sums, got {} and {}",
            sum_in, sum_out
        ));
    }
    let stubs = |seq: &[usize]| -> Vec<usize> {
        seq.iter()
            .enumerate()
            .flat_map(|(v, &d)| std::iter::repeat(v).take(d))
            .collect()
    };
    let mut rng = make_rng(seed);
    let mut out_stubs = stubs(&outs);
    let mut in_stubs = stubs(&ins);
    out_stubs.shuffle(&mut rng);
    in_stubs.shuffle(&mut rng);
    Ok(out_stubs.into_iter().zip(in_stubs).collect())
}

/// Directed Havel-Hakimi graph (Kleitman and Wang 1973): deterministic simple digraph
/// with the given in- and out-degrees, following NetworkX's
/// `directed_havel_hakimi_graph()`.
#[pyfunction]
pub fn directed_havel_hakimi_edges(
    in_degree_sequence: Vec<i64>,
    out_degree_sequence: Vec<i64>,
) -> PyResult<EdgeList> {
    let ins = degree_list("in_degree_sequence", &in_degree_sequence)?;
    let outs = degree_list("out_degree_sequence", &out_degree_sequence)?;
    let n = ins.len().max(outs.len());
    let (sum_in, sum_out) = (ins.iter().sum::<usize>(), outs.iter().sum::<usize>());
    if sum_in != sum_out {
        return value_error(format!(
            "the in-degree and out-degree sequences must have equal sums, got {} and {}",
            sum_in, sum_out
        ));
    }
    let not_digraphical = || {
        value_error("the in- and out-degree sequences are not digraphical: no simple directed graph has these degrees")
    };
    // min-heaps keyed like NetworkX's: (-out, -in, node) and (-out, node)
    let mut stubheap: BinaryHeap<Reverse<(i64, i64, usize)>> = BinaryHeap::new();
    let mut zeroheap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut maxin = 0;
    for v in 0..n {
        let in_deg = ins.get(v).copied().unwrap_or(0) as i64;
        let out_deg = outs.get(v).copied().unwrap_or(0) as i64;
        maxin = maxin.max(in_deg as usize);
        if in_deg > 0 {
            stubheap.push(Reverse((-out_deg, -in_deg, v)));
        } else if out_deg > 0 {
            zeroheap.push(Reverse((-out_deg, v)));
        }
    }
    let mut edges = Vec::new();
    let mut modstubs: Vec<(i64, i64, usize)> = vec![(0, 0, 0); maxin + 1];
    while let Some(Reverse((freeout, freein, target))) = stubheap.pop() {
        let freein = (-freein) as usize;
        if freein > stubheap.len() + zeroheap.len() {
            return not_digraphical();
        }
        let mut mslen = 0;
        for _ in 0..freein {
            let take_zero = match (zeroheap.peek(), stubheap.peek()) {
                (Some(_), None) => true,
                (Some(Reverse(z)), Some(Reverse(s))) => s.0 > z.0,
                _ => false,
            };
            let (stubout, stubin, stubsource) = if take_zero {
                let Reverse((o, s)) = zeroheap.pop().unwrap();
                (o, 0, s)
            } else {
                let Reverse(t) = stubheap.pop().unwrap();
                t
            };
            if stubout == 0 {
                return not_digraphical();
            }
            edges.push((stubsource, target));
            if stubout + 1 < 0 || stubin < 0 {
                modstubs[mslen] = (stubout + 1, stubin, stubsource);
                mslen += 1;
            }
        }
        for &stub in &modstubs[..mslen] {
            if stub.1 < 0 {
                stubheap.push(Reverse(stub));
            } else {
                zeroheap.push(Reverse((stub.0, stub.2)));
            }
        }
        if freeout < 0 {
            zeroheap.push(Reverse((freeout, target)));
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

/// Random simple graph with a given degree sequence (Bayati, Kim and Saberi 2010).
///
/// Edges are added one at a time between nodes with free stubs, each candidate pair
/// `(u, v)` accepted with probability `1 - d_u d_v / 4m`, in three phases that follow
/// NetworkX's `random_degree_sequence_graph()`. An attempt can get stuck with stubs
/// that can't be joined; up to `tries` attempts are made.
#[pyfunction]
#[pyo3(signature = (degree_sequence, tries=10, seed=None))]
pub fn random_degree_sequence_edges(
    degree_sequence: Vec<i64>,
    tries: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let degree = degree_list("degree_sequence", &degree_sequence)?;
    if !is_graphical(&degree) {
        return value_error("degree_sequence is not graphical: no simple graph has these degrees");
    }
    if tries < 1 {
        return value_error("tries must be at least 1");
    }
    let mut rng = make_rng(seed);
    for _ in 0..tries {
        if let Some(mut edges) = BksAttempt::run(&degree, &mut rng) {
            edges.sort_unstable();
            return Ok(edges);
        }
    }
    value_error(format!(
        "failed to generate a graph with this degree sequence in {} tries",
        tries
    ))
}

/// One attempt of the Bayati-Kim-Saberi construction.
struct BksAttempt<'a> {
    degree: &'a [usize],
    four_m: f64,
    remaining: Vec<usize>,
    alive: Vec<usize>, // nodes with free stubs
    slot: Vec<usize>,  // position of each node in `alive`
    count: Vec<usize>, // alive nodes per remaining degree
    max_remaining: usize,
    edges: FxHashSet<(usize, usize)>,
}

impl<'a> BksAttempt<'a> {
    fn run(degree: &'a [usize], rng: &mut ChaCha8Rng) -> Option<EdgeList> {
        let n = degree.len();
        let dmax = degree.iter().copied().max().unwrap_or(0);
        let total: usize = degree.iter().sum();
        let mut s = BksAttempt {
            degree,
            four_m: 2.0 * total as f64,
            remaining: degree.to_vec(),
            alive: Vec::new(),
            slot: vec![usize::MAX; n],
            count: vec![0; dmax + 1],
            max_remaining: dmax,
            edges: FxHashSet::default(),
        };
        for (v, &d) in degree.iter().enumerate() {
            if d > 0 {
                s.slot[v] = s.alive.len();
                s.alive.push(v);
                s.count[d] += 1;
            }
        }
        if !s.alive.is_empty() {
            s.phase1(rng, dmax)?;
            s.phase2(rng, dmax)?;
            s.phase3(rng)?;
        }
        Some(s.edges.into_iter().collect())
    }

    fn p(&self, u: usize, v: usize) -> f64 {
        1.0 - (self.degree[u] * self.degree[v]) as f64 / self.four_m
    }

    fn q(&mut self, u: usize, v: usize) -> f64 {
        while self.max_remaining > 0 && self.count[self.max_remaining] == 0 {
            self.max_remaining -= 1;
        }
        let norm = (self.max_remaining * self.max_remaining) as f64;
        (self.remaining[u] * self.remaining[v]) as f64 / norm
    }

    fn decrement(&mut self, v: usize) {
        let r = self.remaining[v];
        self.count[r] -= 1;
        self.remaining[v] = r - 1;
        if r > 1 {
            self.count[r - 1] += 1;
        } else {
            let i = self.slot[v];
            self.alive.swap_remove(i);
            if i < self.alive.len() {
                self.slot[self.alive[i]] = i;
            }
            self.slot[v] = usize::MAX;
        }
    }

    fn join(&mut self, u: usize, v: usize) {
        self.edges.insert(key(u, v));
        self.decrement(u);
        self.decrement(v);
    }

    /// Pairs drawn with probability proportional to remaining degree (stub sampling).
    fn phase1(&mut self, rng: &mut ChaCha8Rng, dmax: usize) -> Option<()> {
        let mut stubs: Vec<usize> = self
            .alive
            .iter()
            .flat_map(|&v| std::iter::repeat(v).take(self.remaining[v]))
            .collect();
        let mut stall = 0;
        while stubs.len() >= 2 * dmax * dmax {
            stall += 1;
            if stall > STALL_LIMIT {
                return None;
            }
            let i = rng.gen_range(0..stubs.len());
            let mut j = rng.gen_range(0..stubs.len());
            while stubs[j] == stubs[i] {
                j = rng.gen_range(0..stubs.len());
            }
            let (u, v) = key(stubs[i], stubs[j]);
            if self.edges.contains(&(u, v)) {
                continue;
            }
            if rng.gen::<f64>() < self.p(u, v) {
                self.join(u, v);
                stubs.swap_remove(i.max(j));
                stubs.swap_remove(i.min(j));
                stall = 0;
            }
        }
        Some(())
    }

    /// Uniform pairs of nodes with free stubs, thinned by remaining degree.
    fn phase2(&mut self, rng: &mut ChaCha8Rng, dmax: usize) -> Option<()> {
        let mut stall = 0;
        while self.alive.len() >= 2 * dmax {
            let (u, v) = loop {
                stall += 1;
                if stall > STALL_LIMIT {
                    return None;
                }
                let picked: Vec<usize> = self.alive.choose_multiple(rng, 2).copied().collect();
                let (u, v) = key(picked[0], picked[1]);
                if self.edges.contains(&(u, v)) {
                    continue;
                }
                if rng.gen::<f64>() < self.q(u, v) {
                    break (u, v);
                }
            };
            if rng.gen::<f64>() < self.p(u, v) {
                self.join(u, v);
                stall = 0;
            }
        }
        Some(())
    }

    /// The last few stubs: draw from the explicit list of joinable pairs.
    fn phase3(&mut self, rng: &mut ChaCha8Rng) -> Option<()> {
        let mut nodes = self.alive.clone();
        nodes.sort_unstable();
        let mut pairs: EdgeList = Vec::new();
        for (i, &u) in nodes.iter().enumerate() {
            for &v in &nodes[i + 1..] {
                if !self.edges.contains(&(u, v)) {
                    pairs.push((u, v));
                }
            }
        }
        let mut stall = 0;
        while !self.alive.is_empty() {
            let u = *self.alive.iter().min().unwrap();
            if !self
                .alive
                .iter()
                .any(|&v| v != u && !self.edges.contains(&key(u, v)))
            {
                return None;
            }
            let idx = loop {
                stall += 1;
                if stall > STALL_LIMIT {
                    return None;
                }
                let idx = rng.gen_range(0..pairs.len());
                let (a, b) = pairs[idx];
                if rng.gen::<f64>() < self.q(a, b) {
                    break idx;
                }
            };
            let (a, b) = pairs[idx];
            if rng.gen::<f64>() < self.p(a, b) {
                self.join(a, b);
                pairs.swap_remove(idx);
                let remaining = &self.remaining;
                pairs.retain(|&(x, y)| remaining[x] > 0 && remaining[y] > 0);
                stall = 0;
            }
        }
        Some(())
    }
}

/// Random simple graph with a given joint degree matrix (Gjoka, Tillman and
/// Markopoulou 2015).
///
/// `joint_degrees[k][l]` is the number of edges between nodes of degree `k` and
/// nodes of degree `l` (counted twice when `k == l`). Nodes are numbered degree class
/// by degree class, in increasing order of degree. Returns the node count and the
/// edges.
#[pyfunction]
#[pyo3(signature = (joint_degrees, seed=None))]
pub fn joint_degree_edges(
    joint_degrees: BTreeMap<usize, BTreeMap<usize, usize>>,
    seed: Option<Seed>,
) -> PyResult<(usize, EdgeList)> {
    let seed = seed.map(u64::from);
    let entry = |k: usize, l: usize| {
        joint_degrees
            .get(&k)
            .and_then(|row| row.get(&l))
            .copied()
            .unwrap_or(0)
    };
    let mut class_size: BTreeMap<usize, usize> = BTreeMap::new();
    for (&k, row) in &joint_degrees {
        let total: usize = row.values().sum();
        if k == 0 {
            if total > 0 {
                return value_error(
                    "nodes of degree 0 can't have edges, but joint_degrees[0] has some",
                );
            }
            continue;
        }
        if total % k != 0 {
            return value_error(format!(
                "the entries of joint_degrees[{}] must sum to a multiple of {}, got {}",
                k, k, total
            ));
        }
        if total > 0 {
            class_size.insert(k, total / k);
        }
    }
    for (&k, row) in &joint_degrees {
        for (&l, &count) in row {
            if count == 0 {
                continue;
            }
            if entry(l, k) != count {
                return value_error(format!(
                    "joint_degrees must be symmetric, but joint_degrees[{}][{}] = {} and joint_degrees[{}][{}] = {}",
                    k, l, count, l, k, entry(l, k)
                ));
            }
            let (ck, cl) = (class_size[&k], class_size[&l]);
            if k == l {
                if count % 2 != 0 {
                    return value_error(format!(
                        "joint_degrees[{}][{}] counts each edge twice, so it must be even",
                        k, k
                    ));
                }
                if count > ck * (ck - 1) {
                    return value_error(format!(
                        "joint_degrees[{}][{}] = {} is more than the {} nodes of degree {} can hold",
                        k, k, count, ck, k
                    ));
                }
            } else if count > ck * cl {
                return value_error(format!(
                    "joint_degrees[{}][{}] = {} is more than {} nodes of degree {} and {} of degree {} can hold",
                    k, l, count, ck, k, cl, l
                ));
            }
        }
    }

    let n: usize = class_size.values().sum();
    let mut start: BTreeMap<usize, usize> = BTreeMap::new();
    let mut residual = vec![0usize; n];
    let mut next = 0;
    for (&k, &c) in &class_size {
        start.insert(k, next);
        residual[next..next + c].iter_mut().for_each(|r| *r = k);
        next += c;
    }
    let mut rng = make_rng(seed);
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let stuck = || value_error("could not realize joint_degrees");

    for (&k, row) in &joint_degrees {
        for (&l, &count) in row {
            if count == 0 || k < l {
                continue;
            }
            let same = k == l;
            let (k0, k_size) = (start[&k], class_size[&k]);
            let (l0, l_size) = (start[&l], class_size[&l]);
            let mut k_unsat: BTreeSet<usize> =
                (k0..k0 + k_size).filter(|&v| residual[v] > 0).collect();
            let mut l_unsat: BTreeSet<usize> = if same {
                BTreeSet::new()
            } else {
                (l0..l0 + l_size).filter(|&v| residual[v] > 0).collect()
            };
            let mut to_add = if same { count / 2 } else { count };
            let mut stall = 0;
            while to_add > 0 {
                stall += 1;
                if stall > STALL_LIMIT {
                    return stuck();
                }
                let v = k0 + rng.gen_range(0..k_size);
                let w = l0 + rng.gen_range(0..l_size);
                if v == w || adj[v].contains(&w) {
                    continue;
                }
                if residual[v] == 0
                    && !neighbor_switch(&mut adj, v, &mut k_unsat, &mut residual, None)
                {
                    return stuck();
                }
                if residual[w] == 0 {
                    let ok = if same {
                        neighbor_switch(&mut adj, w, &mut k_unsat, &mut residual, Some(v))
                    } else {
                        neighbor_switch(&mut adj, w, &mut l_unsat, &mut residual, None)
                    };
                    if !ok {
                        return stuck();
                    }
                }
                adj[v].insert(w);
                adj[w].insert(v);
                residual[v] -= 1;
                residual[w] -= 1;
                to_add -= 1;
                stall = 0;
                if residual[v] == 0 {
                    k_unsat.remove(&v);
                }
                if residual[w] == 0 {
                    if same {
                        k_unsat.remove(&w)
                    } else {
                        l_unsat.remove(&w)
                    };
                }
            }
        }
    }
    Ok((n, edges_from_adj(&adj)))
}

/// Free one stub of saturated node `w` without changing the joint degrees: move one of
/// its edges to an unsaturated node `w'` of the same degree. Returns `false` if no
/// such move exists (which a valid input rules out).
fn neighbor_switch(
    adj: &mut [FxHashSet<usize>],
    w: usize,
    unsat: &mut BTreeSet<usize>,
    residual: &mut [usize],
    avoid: Option<usize>,
) -> bool {
    let w_prime = match avoid {
        Some(a) if residual[a] <= 1 => unsat.iter().copied().find(|&x| x != a),
        _ => unsat.iter().next().copied(),
    };
    let Some(w_prime) = w_prime else { return false };
    let mut nbrs: Vec<usize> = adj[w].iter().copied().collect();
    nbrs.sort_unstable();
    let Some(switch) = nbrs
        .into_iter()
        .find(|&x| x != w_prime && !adj[w_prime].contains(&x))
    else {
        return false;
    };
    adj[w].remove(&switch);
    adj[switch].remove(&w);
    adj[w_prime].insert(switch);
    adj[switch].insert(w_prime);
    residual[w] += 1;
    residual[w_prime] -= 1;
    if residual[w_prime] == 0 {
        unsat.remove(&w_prime);
    }
    true
}

// --- directed growth -----------------------------------------------------------

/// Growing network (Krapivsky and Redner 2001): each new node links to one existing
/// node chosen with probability proportional to `kernel[degree]`.
///
/// `kernel` tabulates the attachment kernel by total degree; `None` is the linear
/// kernel (probability proportional to degree). Edges point from the new node.
#[pyfunction]
#[pyo3(signature = (n, kernel=None, seed=None))]
pub fn gn_edges(n: usize, kernel: Option<Vec<f64>>, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if let Some(table) = &kernel {
        if let Some(bad) = table.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return value_error(format!(
                "kernel must return finite, non-negative weights, got {}",
                bad
            ));
        }
    }
    let weight = |d: usize| match &kernel {
        Some(table) => table.get(d).copied().unwrap_or(0.0),
        None => d as f64,
    };
    if n <= 1 {
        return Ok(Vec::new());
    }
    let mut rng = make_rng(seed);
    let mut degree = vec![0usize; n];
    let mut tree = WeightTree::new(n);
    let mut edges = vec![(1, 0)];
    degree[0] = 1;
    degree[1] = 1;
    tree.set(0, weight(1));
    tree.set(1, weight(1));
    for source in 2..n {
        let Some(target) = tree.sample(&mut rng) else {
            return value_error(
                "kernel gives every existing node zero weight, so no target can be chosen",
            );
        };
        edges.push((source, target));
        degree[source] = 1;
        tree.set(source, weight(1));
        degree[target] += 1;
        tree.set(target, weight(degree[target]));
    }
    Ok(edges)
}

/// Growing network with redirection (Krapivsky and Redner 2001): each new node picks
/// a uniformly random existing node and, with probability `p`, links to that node's
/// target instead.
#[pyfunction]
#[pyo3(signature = (n, p, seed=None))]
pub fn gnr_edges(n: usize, p: f64, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_probability("p", p)?;
    let mut rng = make_rng(seed);
    let mut successor = vec![0usize; n];
    let mut edges = Vec::with_capacity(n.saturating_sub(1));
    for source in 1..n {
        let mut target = rng.gen_range(0..source);
        if rng.gen::<f64>() < p && target != 0 {
            target = successor[target];
        }
        successor[source] = target;
        edges.push((source, target));
    }
    Ok(edges)
}

/// Growing network with copying (Krapivsky and Redner 2005): each new node links to a
/// uniformly random existing node and to every node that one links to.
#[pyfunction]
#[pyo3(signature = (n, seed=None))]
pub fn gnc_edges(n: usize, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let mut rng = make_rng(seed);
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for source in 1..n {
        let target = rng.gen_range(0..source);
        let mut out = successors[target].clone();
        out.push(target);
        edges.extend(out.iter().map(|&t| (source, t)));
        successors[source] = out;
    }
    Ok(edges)
}

/// Scale-free directed multigraph (Bollobás, Borgs, Chayes and Riordan 2003).
///
/// Grows from the 3-cycle `0 -> 1 -> 2 -> 0`. Each step adds one edge: with
/// probability `alpha` from a new node to an existing node chosen by in-degree, with
/// `beta` between two existing nodes chosen by out-degree and in-degree, and with
/// `gamma` from an existing node chosen by out-degree to a new node. `delta_in` and
/// `delta_out` add a uniform share to the in- and out-degree choices.
#[pyfunction]
#[pyo3(signature = (n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0.0, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn scale_free_edges(
    n: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta_in: f64,
    delta_out: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if n < 3 {
        return value_error(format!(
            "n must be at least 3 (growth starts from a 3-cycle), got {}",
            n
        ));
    }
    for (name, x) in [("alpha", alpha), ("beta", beta), ("gamma", gamma)] {
        if x.is_nan() || x <= 0.0 {
            return value_error(format!("{} must be positive, got {}", name, x));
        }
    }
    if (alpha + beta + gamma - 1.0).abs() >= 1e-9 {
        return value_error(format!(
            "alpha + beta + gamma must equal 1, got {}",
            alpha + beta + gamma
        ));
    }
    for (name, x) in [("delta_in", delta_in), ("delta_out", delta_out)] {
        if x.is_nan() || x < 0.0 {
            return value_error(format!("{} must be non-negative, got {}", name, x));
        }
    }
    let mut rng = make_rng(seed);
    let mut edges: EdgeList = vec![(0, 1), (1, 2), (2, 0)];
    let mut sources: Vec<usize> = vec![0, 1, 2]; // one entry per out-edge
    let mut targets: Vec<usize> = vec![1, 2, 0]; // one entry per in-edge
    let mut cursor = 3;
    let choose = |rng: &mut ChaCha8Rng, candidates: &[usize], count: usize, delta: f64| {
        if delta > 0.0 {
            let bias = count as f64 * delta;
            if rng.gen::<f64>() < bias / (bias + candidates.len() as f64) {
                return rng.gen_range(0..count);
            }
        }
        candidates[rng.gen_range(0..candidates.len())]
    };
    while cursor < n {
        let r: f64 = rng.gen();
        let (v, w) = if r < alpha {
            cursor += 1;
            (cursor - 1, choose(&mut rng, &targets, cursor, delta_in))
        } else if r < alpha + beta {
            let v = choose(&mut rng, &sources, cursor, delta_out);
            (v, choose(&mut rng, &targets, cursor, delta_in))
        } else {
            let v = choose(&mut rng, &sources, cursor, delta_out);
            cursor += 1;
            (v, cursor - 1)
        };
        edges.push((v, w));
        sources.push(v);
        targets.push(w);
    }
    Ok(edges)
}

/// Random `k`-out graph (Peterson and Pittel): every node gets exactly `k` out-edges,
/// whose targets are chosen with probability proportional to `alpha + in-degree`.
#[pyfunction]
#[pyo3(signature = (n, k, alpha, self_loops=true, seed=None))]
pub fn random_k_out_edges(
    n: usize,
    k: usize,
    alpha: f64,
    self_loops: bool,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if !(alpha.is_finite() && alpha > 0.0) {
        return value_error(format!("alpha must be positive, got {}", alpha));
    }
    if !self_loops && n == 1 && k > 0 {
        return value_error("a single node can't have out-edges when self_loops=False");
    }
    let mut rng = make_rng(seed);
    let mut tree = WeightTree::new(n);
    for v in 0..n {
        tree.set(v, alpha);
    }
    let mut open: Vec<usize> = (0..n).collect(); // nodes still short of k out-edges
    let mut out = vec![0usize; n];
    let mut edges = Vec::with_capacity(n * k);
    if k == 0 {
        return Ok(edges);
    }
    for _ in 0..n * k {
        let i = rng.gen_range(0..open.len());
        let u = open[i];
        let own = tree.weight[u];
        if !self_loops {
            tree.set(u, 0.0);
        }
        let v = tree.sample(&mut rng).expect("weights are positive");
        if !self_loops {
            tree.set(u, own);
        }
        edges.push((u, v));
        tree.set(v, tree.weight[v] + 1.0);
        out[u] += 1;
        if out[u] == k {
            open.swap_remove(i);
        }
    }
    Ok(edges)
}

// --- geometric -----------------------------------------------------------------

/// Waxman graph (Waxman 1988): nodes placed uniformly in a rectangle, each pair joined
/// with probability `beta * exp(-d / (alpha * L))`.
///
/// `L` defaults to the largest distance between two nodes. If `L` is given, `d` is
/// drawn uniformly from `[0, L)` for each pair instead of measured (the positions are
/// still drawn and returned).
#[pyfunction]
#[pyo3(signature = (n, beta=0.4, alpha=0.1, l=None, domain=(0.0, 0.0, 1.0, 1.0), seed=None))]
pub fn waxman_edges(
    n: usize,
    beta: f64,
    alpha: f64,
    l: Option<f64>,
    domain: (f64, f64, f64, f64),
    seed: Option<Seed>,
) -> PyResult<(Positions, EdgeList)> {
    let seed = seed.map(u64::from);
    if !(beta > 0.0 && beta <= 1.0) {
        return value_error(format!(
            "beta must be greater than 0 and at most 1, got {}",
            beta
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0) {
        return value_error(format!("alpha must be positive, got {}", alpha));
    }
    if let Some(l) = l {
        if !(l.is_finite() && l > 0.0) {
            return value_error(format!("L must be positive, got {}", l));
        }
    }
    let (xmin, ymin, xmax, ymax) = domain;
    if ![xmin, ymin, xmax, ymax].iter().all(|v| v.is_finite()) || xmin > xmax || ymin > ymax {
        return value_error(
            "domain must be (xmin, ymin, xmax, ymax) with xmin <= xmax and ymin <= ymax",
        );
    }
    let mut rng = make_rng(seed);
    let pos: Positions = (0..n)
        .map(|_| {
            let x = xmin + (xmax - xmin) * rng.gen::<f64>();
            let y = ymin + (ymax - ymin) * rng.gen::<f64>();
            vec![x, y]
        })
        .collect();
    let scale = match l {
        Some(l) => l,
        None => {
            let mut longest: f64 = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    longest = longest.max(minkowski(&pos[i], &pos[j], 2.0));
                }
            }
            longest
        }
    };
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = match l {
                Some(l) => rng.gen::<f64>() * l,
                None => minkowski(&pos[i], &pos[j], 2.0),
            };
            let ratio = if scale > 0.0 {
                d / (alpha * scale)
            } else {
                0.0
            };
            if rng.gen::<f64>() < beta * (-ratio).exp() {
                edges.push((i, j));
            }
        }
    }
    Ok((pos, edges))
}

/// Soft random geometric graph (Penrose 2016): nodes placed uniformly in the unit
/// cube; each pair within `radius` is joined with probability `p_dist(d)`
/// (default `exp(-d)`).
#[pyfunction]
#[pyo3(signature = (n, radius, dim=2, p_norm=2.0, p_dist=None, seed=None))]
pub fn soft_random_geometric_edges(
    n: usize,
    radius: f64,
    dim: usize,
    p_norm: f64,
    p_dist: Option<Bound<'_, PyAny>>,
    seed: Option<Seed>,
) -> PyResult<(Positions, EdgeList)> {
    let seed = seed.map(u64::from);
    if radius.is_nan() || radius < 0.0 {
        return value_error(format!("radius must be non-negative, got {}", radius));
    }
    check_geometry(dim, p_norm)?;
    let mut rng = make_rng(seed);
    let pos = uniform_positions(&mut rng, n, dim);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = minkowski(&pos[i], &pos[j], p_norm);
            if d > radius {
                continue;
            }
            let prob = match &p_dist {
                Some(f) => f.call1((d,))?.extract::<f64>()?,
                None => (-d).exp(),
            };
            if rng.gen::<f64>() < prob {
                edges.push((i, j));
            }
        }
    }
    Ok((pos, edges))
}

/// Geographical threshold graph (Masuda, Miwa and Konno 2005): nodes get weights
/// drawn from an exponential distribution and positions in the unit cube; a pair is
/// joined when `(w_u + w_v) * p_dist(d) >= theta` (default `p_dist(d) = d^-2`).
/// Returns the positions, the weights, and the edges.
#[pyfunction]
#[pyo3(signature = (n, theta, dim=2, p_norm=2.0, p_dist=None, seed=None))]
pub fn geographical_threshold_edges(
    n: usize,
    theta: f64,
    dim: usize,
    p_norm: f64,
    p_dist: Option<Bound<'_, PyAny>>,
    seed: Option<Seed>,
) -> PyResult<(Positions, Vec<f64>, EdgeList)> {
    let seed = seed.map(u64::from);
    if theta.is_nan() {
        return value_error("theta must be a number");
    }
    check_geometry(dim, p_norm)?;
    let mut rng = make_rng(seed);
    let weight: Vec<f64> = (0..n).map(|_| -(1.0 - rng.gen::<f64>()).ln()).collect();
    let pos = uniform_positions(&mut rng, n, dim);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = minkowski(&pos[i], &pos[j], p_norm);
            let factor = match &p_dist {
                Some(f) => f.call1((d,))?.extract::<f64>()?,
                None => d.powi(-2),
            };
            if (weight[i] + weight[j]) * factor >= theta {
                edges.push((i, j));
            }
        }
    }
    Ok((pos, weight, edges))
}

/// Kleinberg's navigable small world: an `n^dim` grid in which every node links to
/// all nodes within lattice (Manhattan) distance `p` and to `q` long-range contacts
/// chosen with probability proportional to `d^-r`.
///
/// Nodes are numbered in lexicographic order of their coordinates. Links are
/// directed; with `directed=false` each pair is returned once.
#[pyfunction]
#[pyo3(signature = (n, p=1, q=1, r=2.0, dim=2, directed=true, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn navigable_small_world_edges(
    n: usize,
    p: usize,
    q: usize,
    r: f64,
    dim: usize,
    directed: bool,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    if p < 1 {
        return value_error("p must be at least 1");
    }
    if r.is_nan() || r < 0.0 {
        return value_error(format!("r must be non-negative, got {}", r));
    }
    if dim < 1 {
        return value_error("dim must be at least 1");
    }
    let total = u32::try_from(dim)
        .ok()
        .and_then(|d| n.checked_pow(d))
        .filter(|&t| t <= 1_000_000)
        .ok_or_else(|| PyValueError::new_err(format!("n^dim = {}^{} nodes is too many", n, dim)))?;
    let coords: Vec<Vec<usize>> = (0..total)
        .map(|mut idx| {
            let mut c = vec![0; dim];
            for slot in c.iter_mut().rev() {
                *slot = idx % n;
                idx /= n;
            }
            c
        })
        .collect();
    let lattice_dist = |a: usize, b: usize| -> usize {
        coords[a]
            .iter()
            .zip(&coords[b])
            .map(|(x, y)| x.abs_diff(*y))
            .sum()
    };
    let mut rng = make_rng(seed);
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut edges = Vec::new();
    let mut push = |u: usize, v: usize| {
        let e = if directed { (u, v) } else { key(u, v) };
        if seen.insert(e) {
            edges.push(e);
        }
    };
    let mut others = Vec::with_capacity(total);
    let mut cdf = Vec::with_capacity(total);
    for u in 0..total {
        others.clear();
        cdf.clear();
        let mut acc = 0.0;
        for v in 0..total {
            if v == u {
                continue;
            }
            let d = lattice_dist(u, v);
            if d <= p {
                push(u, v);
            }
            acc += (d as f64).powf(-r);
            others.push(v);
            cdf.push(acc);
        }
        if others.is_empty() {
            continue;
        }
        for _ in 0..q {
            let x = rng.gen::<f64>() * acc;
            let i = cdf.partition_point(|&c| c <= x).min(others.len() - 1);
            push(u, others[i]);
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erdos_gallai() {
        assert!(is_graphical(&[]));
        assert!(is_graphical(&[0, 0]));
        assert!(is_graphical(&[1, 1]));
        assert!(is_graphical(&[3, 3, 3, 3]));
        assert!(is_graphical(&[2, 2, 2]));
        assert!(!is_graphical(&[1]));
        assert!(!is_graphical(&[3, 3, 1, 1]));
        assert!(!is_graphical(&[2, 0]));
        assert!(!is_graphical(&[4, 1, 1, 1]));
    }

    #[test]
    fn prufer_round_trip() {
        // the sequence [3, 3, 3, 4] decodes to the tree with edges below
        let edges = prufer_tree(6, &[3, 3, 3, 4]);
        assert_eq!(edges, vec![(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]);
    }

    #[test]
    fn weight_tree_samples_only_positive_weights() {
        let mut rng = make_rng(Some(1));
        let mut tree = WeightTree::new(5);
        tree.set(1, 2.0);
        tree.set(3, 1.0);
        for _ in 0..1000 {
            let i = tree.sample(&mut rng).unwrap();
            assert!(i == 1 || i == 3);
        }
        tree.set(1, 0.0);
        tree.set(3, 0.0);
        assert!(tree.sample(&mut rng).is_none());
    }
}
