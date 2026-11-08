//! Bipartite, duplication, clustered, internet, expander, and other special generators.
//!
//! Like `generators` and `random_models`, these return plain index structures over
//! nodes `0..n` that the Python wrappers turn into nodes and edges. Symmetric models
//! return each edge as `(lower, higher)`, so that in a directed graph it points from
//! the earlier-created node to the later one. Bipartite models number the first part
//! `0..n1` and the second `n1..`, so their edges point from the first part to the
//! second.
//!
//! - bipartite: `bipartite_configuration_edges`, `bipartite_havel_hakimi_edges`,
//!   `bipartite_preferential_attachment_edges`, `bipartite_gnp_edges`,
//!   `bipartite_gnmk_edges`
//! - duplication and clustering: `partial_duplication_edges`,
//!   `duplication_divergence_edges`, `random_clustered_edges`
//! - internet: `internet_as_graph`
//! - expanders: `margulis_gabber_galil_edges`, `chordal_cycle_edges`, `paley_edges`,
//!   `maybe_regular_expander_edges`, `random_regular_expander_edges`,
//!   `adjacency_second_eigenvalue`
//! - deterministic: `sudoku_edges`, `visibility_edges`, `interval_edges`

use crate::utils::Seed;
use std::collections::BTreeSet;

use nalgebra::{DMatrix, SymmetricEigen};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use super::generators::{check_probability, key, make_rng, sample_range, EdgeList};

/// Consecutive failed duplications after which `duplication_divergence_edges` gives up.
const STALL_LIMIT: usize = 1_000_000;

fn value_error<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn degree_list(name: &str, seq: &[i64]) -> PyResult<Vec<usize>> {
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

/// Two degree sequences for the two parts of a bipartite graph, with equal sums.
fn bipartite_sequences(aseq: &[i64], bseq: &[i64]) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let a = degree_list("aseq", aseq)?;
    let b = degree_list("bseq", bseq)?;
    let (suma, sumb): (usize, usize) = (a.iter().sum(), b.iter().sum());
    if suma != sumb {
        return value_error(format!(
            "aseq and bseq must have the same sum, got {} and {}",
            suma, sumb
        ));
    }
    Ok((a, b))
}

/// Python's `round()`: halfway cases go to the even neighbor.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

// --- bipartite -------------------------------------------------------------------

/// Bipartite configuration model: shuffle each part's stubs and pair them in order.
///
/// Node `i < len(aseq)` gets degree `aseq[i]` and node `len(aseq) + j` degree
/// `bseq[j]`. Parallel edges are possible, as in NetworkX's
/// `bipartite.configuration_model()`.
#[pyfunction]
#[pyo3(signature = (aseq, bseq, seed=None))]
pub fn bipartite_configuration_edges(
    aseq: Vec<i64>,
    bseq: Vec<i64>,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let (a, b) = bipartite_sequences(&aseq, &bseq)?;
    let lena = a.len();
    let mut astubs: Vec<usize> = (0..lena)
        .flat_map(|v| std::iter::repeat_n(v, a[v]))
        .collect();
    let mut bstubs: Vec<usize> = (0..b.len())
        .flat_map(|v| std::iter::repeat_n(lena + v, b[v]))
        .collect();
    let mut rng = make_rng(seed);
    astubs.shuffle(&mut rng);
    bstubs.shuffle(&mut rng);
    Ok(astubs.into_iter().zip(bstubs).collect())
}

/// Deterministic bipartite graph with the given degrees, following NetworkX's
/// `bipartite.havel_hakimi_graph()` (`variant="largest"`),
/// `reverse_havel_hakimi_graph()` (`"smallest"`), or
/// `alternating_havel_hakimi_graph()` (`"alternating"`).
///
/// Each step takes the first-part node of largest remaining degree `d` and joins it to
/// `d` second-part nodes: those of largest remaining degree, of smallest, or alternately
/// one of each. NetworkX silently joins fewer nodes when not enough remain; here that
/// raises, since the result would not have the requested degrees.
#[pyfunction]
pub fn bipartite_havel_hakimi_edges(
    aseq: Vec<i64>,
    bseq: Vec<i64>,
    variant: &str,
) -> PyResult<EdgeList> {
    if !matches!(variant, "largest" | "smallest" | "alternating") {
        return value_error(format!(
            "variant must be 'largest', 'smallest' or 'alternating', got {:?}",
            variant
        ));
    }
    let (a, b) = bipartite_sequences(&aseq, &bseq)?;
    let lena = a.len();
    // (remaining degree, node) pairs, sorted ascending like NetworkX's lists
    let mut astubs: Vec<(usize, usize)> = a.iter().enumerate().map(|(v, &d)| (d, v)).collect();
    let mut bstubs: Vec<(usize, usize)> =
        b.iter().enumerate().map(|(v, &d)| (d, lena + v)).collect();
    // NetworkX keeps zero-degree nodes in the list until a step joins one of them (and
    // gives it degree -1); leaving them out keeps every degree exact
    bstubs.retain(|&(d, _)| d > 0);
    astubs.sort_unstable();
    if variant == "smallest" {
        bstubs.sort_unstable();
    }
    let mut edges = Vec::new();
    loop {
        if variant == "alternating" {
            astubs.sort_unstable();
        }
        let Some((degree, u)) = astubs.pop() else {
            break;
        };
        if degree == 0 {
            break;
        }
        if degree > bstubs.len() {
            return value_error(format!(
                "the degree sequences can't be realized by this construction: node {} needs {} \
                 neighbors but only {} nodes of the other part have degree left",
                u,
                degree,
                bstubs.len()
            ));
        }
        if variant != "smallest" {
            bstubs.sort_unstable();
        }
        let nb = bstubs.len();
        let targets: Vec<usize> = match variant {
            "largest" => (nb - degree..nb).collect(),
            "smallest" => (0..degree).collect(),
            _ => {
                let small: Vec<usize> = (0..degree / 2).collect();
                let mut large: Vec<usize> = (nb - (degree - degree / 2)..nb).collect();
                let mut stubs: Vec<usize> = large
                    .iter()
                    .zip(&small)
                    .flat_map(|(&l, &s)| [l, s])
                    .collect();
                if stubs.len() < small.len() + large.len() {
                    stubs.push(large.pop().unwrap());
                }
                stubs
            }
        };
        for &t in &targets {
            edges.push((u, bstubs[t].1));
            bstubs[t].0 -= 1;
        }
        bstubs.retain(|&(d, _)| d > 0);
    }
    Ok(edges)
}

/// Bipartite preferential attachment, following NetworkX's
/// `bipartite.preferential_attachment_graph()`.
///
/// Each stub of first-part node `v` (`aseq[v]` of them) goes, with probability `p`
/// (or when there are no second-part nodes yet), to a new second-part node, and
/// otherwise to an existing one chosen with probability proportional to its degree.
/// Returns the total node count and the edges; parallel edges are possible.
#[pyfunction]
#[pyo3(signature = (aseq, p, seed=None))]
pub fn bipartite_preferential_attachment_edges(
    aseq: Vec<i64>,
    p: f64,
    seed: Option<Seed>,
) -> PyResult<(usize, EdgeList)> {
    let seed = seed.map(u64::from);
    let a = degree_list("aseq", &aseq)?;
    check_probability("p", p)?;
    let mut rng = make_rng(seed);
    let mut n = a.len();
    let mut edges = Vec::new();
    // one entry per edge end in the second part, so a uniform draw is by degree
    let mut bottom_stubs: Vec<usize> = Vec::new();
    for (v, &d) in a.iter().enumerate() {
        for _ in 0..d {
            let target = if rng.gen::<f64>() < p || bottom_stubs.is_empty() {
                n += 1;
                n - 1
            } else {
                bottom_stubs[rng.gen_range(0..bottom_stubs.len())]
            };
            bottom_stubs.push(target);
            edges.push((v, target));
        }
    }
    Ok((n, edges))
}

/// Bipartite G(n, m, p): each of the `n * m` pairs across the parts is an edge with
/// probability `p`. With `directed`, a second independent pass adds edges from the
/// second part to the first, as in NetworkX's `bipartite.random_graph()`.
#[pyfunction]
#[pyo3(signature = (n, m, p, directed=false, seed=None))]
pub fn bipartite_gnp_edges(
    n: usize,
    m: usize,
    p: f64,
    directed: bool,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_probability("p", p)?;
    let mut rng = make_rng(seed);
    let mut edges = Vec::new();
    for v in 0..n {
        sample_range(&mut rng, p, 0, m, |w| edges.push((v, n + w)));
    }
    if directed {
        for v in 0..n {
            sample_range(&mut rng, p, 0, m, |w| edges.push((n + w, v)));
        }
    }
    Ok(edges)
}

/// Bipartite G(n, m, k): `k` distinct pairs across the parts drawn uniformly (all of
/// them if `k >= n * m`), sorted.
#[pyfunction]
#[pyo3(signature = (n, m, k, seed=None))]
pub fn bipartite_gnmk_edges(
    n: usize,
    m: usize,
    k: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let total = n
        .checked_mul(m)
        .ok_or_else(|| PyValueError::new_err("n * m is too large"))?;
    let mut picks: Vec<usize> = if k >= total {
        (0..total).collect()
    } else {
        rand::seq::index::sample(&mut make_rng(seed), total, k).into_vec()
    };
    picks.sort_unstable();
    Ok(picks.into_iter().map(|i| (i / m, n + i % m)).collect())
}

// --- duplication and clustering --------------------------------------------------

/// Partial duplication graph, following NetworkX's `partial_duplication_graph()`.
///
/// Starts from a complete graph on `n_initial` nodes. Each new node copies a random
/// existing node: it joins each of that node's neighbors with probability `p`, and
/// the node itself with probability `q`.
#[pyfunction]
#[pyo3(signature = (n, n_initial, p, q, seed=None))]
pub fn partial_duplication_edges(
    n: usize,
    n_initial: usize,
    p: f64,
    q: f64,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_probability("p", p)?;
    check_probability("q", q)?;
    if n_initial > n {
        return value_error(format!(
            "n_initial must be at most n, got {} > {}",
            n_initial, n
        ));
    }
    if n_initial == 0 && n > 0 {
        return value_error("n_initial must be at least 1 when n > 0");
    }
    let mut rng = make_rng(seed);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for b in 1..n_initial {
        for a in 0..b {
            adj[a].push(b);
            adj[b].push(a);
            edges.push((a, b));
        }
    }
    for new in n_initial..n {
        let src = rng.gen_range(0..new);
        let nbrs = adj[src].clone();
        for nbr in nbrs {
            if rng.gen::<f64>() < p {
                adj[nbr].push(new);
                adj[new].push(nbr);
                edges.push((nbr, new));
            }
        }
        if rng.gen::<f64>() < q {
            adj[src].push(new);
            adj[new].push(src);
            edges.push((src, new));
        }
    }
    Ok(edges)
}

/// Duplication-divergence graph, following NetworkX's `duplication_divergence_graph()`.
///
/// Starts from one edge. Each step copies a random node and keeps each of its edges
/// with probability `p`; a copy that keeps no edge is discarded and the step retried.
#[pyfunction]
#[pyo3(signature = (n, p, seed=None))]
pub fn duplication_divergence_edges(n: usize, p: f64, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_probability("p", p)?;
    if n < 2 {
        return value_error(format!("n must be at least 2, got {}", n));
    }
    if p == 0.0 && n > 2 {
        return value_error("p must be positive when n > 2: with p=0 no copy keeps an edge");
    }
    let mut rng = make_rng(seed);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    adj[0].push(1);
    adj[1].push(0);
    let mut edges = vec![(0, 1)];
    let mut i = 2;
    let mut stalls = 0;
    while i < n {
        let original = rng.gen_range(0..i);
        let kept: Vec<usize> = adj[original]
            .iter()
            .copied()
            .filter(|_| rng.gen::<f64>() < p)
            .collect();
        if kept.is_empty() {
            stalls += 1;
            if stalls >= STALL_LIMIT {
                return value_error(format!(
                    "no copy kept an edge in {} tries; p={} is too small",
                    STALL_LIMIT, p
                ));
            }
            continue;
        }
        stalls = 0;
        for nbr in kept {
            adj[nbr].push(i);
            adj[i].push(nbr);
            edges.push((nbr, i));
        }
        i += 1;
    }
    Ok(edges)
}

/// Random clustered graph (Newman 2009), following NetworkX's `random_clustered_graph()`.
///
/// Node `v` gets `joint[v].0` independent-edge stubs and `joint[v].1` triangle stubs.
/// Both lists are shuffled; edge stubs are joined in pairs and triangle stubs in
/// triples. Self-loops and parallel edges are possible.
#[pyfunction]
#[pyo3(signature = (joint, seed=None))]
pub fn random_clustered_edges(joint: Vec<(i64, i64)>, seed: Option<Seed>) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    let mut ilist = Vec::new();
    let mut tlist = Vec::new();
    for (v, &(e, t)) in joint.iter().enumerate() {
        let (e, t) = match (usize::try_from(e), usize::try_from(t)) {
            (Ok(e), Ok(t)) => (e, t),
            _ => {
                return value_error(format!(
                    "joint_degree_sequence must hold non-negative integers, got ({}, {})",
                    e, t
                ))
            }
        };
        ilist.extend(std::iter::repeat_n(v, e));
        tlist.extend(std::iter::repeat_n(v, t));
    }
    if ilist.len() % 2 != 0 {
        return value_error(format!(
            "the independent-edge degrees must sum to an even number, got {}",
            ilist.len()
        ));
    }
    if tlist.len() % 3 != 0 {
        return value_error(format!(
            "the triangle degrees must sum to a multiple of 3, got {}",
            tlist.len()
        ));
    }
    let mut rng = make_rng(seed);
    ilist.shuffle(&mut rng);
    tlist.shuffle(&mut rng);
    let mut edges = Vec::with_capacity(ilist.len() / 2 + tlist.len());
    for pair in ilist.chunks(2) {
        edges.push(key(pair[0], pair[1]));
    }
    for t in tlist.chunks(3) {
        edges.push(key(t[0], t[1]));
        edges.push(key(t[0], t[2]));
        edges.push(key(t[1], t[2]));
    }
    Ok(edges)
}

// --- internet autonomous systems -------------------------------------------------

/// Each node's kind, and the edges as `(u, v, transit)`.
type AsGraphParts = (Vec<&'static str>, Vec<(usize, usize, bool)>);

const KIND_T: usize = 0;
const KIND_M: usize = 1;
const KIND_CP: usize = 2;
const KIND_C: usize = 3;
const KIND_NAMES: [&str; 4] = ["T", "M", "CP", "C"];

/// A candidate drawn with probability proportional to its weight (uniformly if every
/// weight is zero), scanning the candidates in order, as NetworkX's
/// `choose_pref_attach()` does.
fn choose_pref_attach(rng: &mut ChaCha8Rng, candidates: &[usize], weights: &[usize]) -> usize {
    let s: usize = weights.iter().sum();
    if s == 0 {
        return candidates[rng.gen_range(0..candidates.len())];
    }
    let v = rng.gen::<f64>() * s as f64;
    let mut acc = 0.0;
    for (&c, &w) in candidates.iter().zip(weights) {
        acc += w as f64;
        if v <= acc {
            return c;
        }
    }
    candidates[candidates.len() - 1]
}

/// State of the Elmokashfi et al. AS-graph construction (NetworkX's
/// `AS_graph_generator`), with sets iterated in ascending order.
struct AsGraph {
    rng: ChaCha8Rng,
    kind: Vec<usize>,
    adj: Vec<BTreeSet<usize>>,
    peers: Vec<usize>,
    customers: Vec<BTreeSet<usize>>,
    providers: Vec<BTreeSet<usize>>,
    regions: Vec<BTreeSet<usize>>,
    nodes: [BTreeSet<usize>; 4],
    edges: Vec<(usize, usize, bool)>,
}

impl AsGraph {
    /// NetworkX's `uniform_int_from_avg(a, m)`: an integer with mean about `m`.
    fn uniform_int_from_avg(&mut self, a: f64, m: f64) -> usize {
        let b = 2.0 * m - a;
        let p = (b - b.floor()) / 2.0;
        let x1 = round_half_even(self.rng.gen::<f64>() * (b.floor() - a) + a);
        let x2 = if self.rng.gen::<f64>() < p { 1.0 } else { 0.0 };
        (x1 + x2) as usize
    }

    fn add_node(&mut self, kind: usize) -> usize {
        let i = self.kind.len();
        self.kind.push(kind);
        self.adj.push(BTreeSet::new());
        self.peers.push(0);
        self.customers.push(BTreeSet::new());
        self.providers.push(BTreeSet::new());
        self.nodes[kind].insert(i);
        i
    }

    fn add_edge(&mut self, i: usize, j: usize, transit: bool) {
        self.adj[i].insert(j);
        self.adj[j].insert(i);
        let (u, v) = if transit { (i, j) } else { key(i, j) };
        self.edges.push((u, v, transit));
    }

    fn add_customer(&mut self, i: usize, j: usize) {
        self.customers[j].insert(i);
        self.providers[i].insert(j);
        let above: Vec<usize> = self.providers[j].iter().copied().collect();
        for z in above {
            self.customers[z].insert(i);
            self.providers[i].insert(z);
        }
    }

    /// Add an M, CP, or C node and connect it to providers in its region(s).
    fn add_regional_node(&mut self, kind: usize, reg2prob: f64, avg_deg: f64, t_edge_prob: f64) {
        let regs = if self.rng.gen::<f64>() < reg2prob {
            2
        } else {
            1
        };
        let i = self.add_node(kind);
        let mut options = BTreeSet::new();
        for r in rand::seq::index::sample(&mut self.rng, self.regions.len(), regs).into_iter() {
            options.extend(self.regions[r].iter().copied());
            self.regions[r].insert(i);
        }
        let edge_num = self.uniform_int_from_avg(1.0, avg_deg);
        let mut t_options: BTreeSet<usize> =
            options.intersection(&self.nodes[KIND_T]).copied().collect();
        let mut m_options: BTreeSet<usize> =
            options.intersection(&self.nodes[KIND_M]).copied().collect();
        m_options.remove(&i);
        let mut d = 0;
        while d < edge_num && (!t_options.is_empty() || !m_options.is_empty()) {
            let to_t = m_options.is_empty()
                || (!t_options.is_empty() && self.rng.gen::<f64>() < t_edge_prob);
            let pool: Vec<usize> = if to_t { &t_options } else { &m_options }
                .iter()
                .copied()
                .collect();
            let degrees: Vec<usize> = pool.iter().map(|&c| self.adj[c].len()).collect();
            let j = choose_pref_attach(&mut self.rng, &pool, &degrees);
            if to_t {
                t_options.remove(&j);
            } else {
                m_options.remove(&j);
            }
            self.add_edge(i, j, true);
            self.add_customer(i, j);
            d += 1;
        }
    }

    fn add_m_peering_link(&mut self, m: usize) {
        let options: Vec<usize> = self.nodes[KIND_M]
            .iter()
            .copied()
            .filter(|&j| {
                j != m
                    && !self.customers[m].contains(&j)
                    && !self.providers[m].contains(&j)
                    && !self.adj[m].contains(&j)
            })
            .collect();
        if options.is_empty() {
            return;
        }
        let peers: Vec<usize> = options.iter().map(|&c| self.peers[c]).collect();
        let j = choose_pref_attach(&mut self.rng, &options, &peers);
        self.add_edge(m, j, false);
        self.peers[m] += 1;
        self.peers[j] += 1;
    }

    fn add_cp_peering_link(&mut self, cp: usize, to_kind: usize) {
        let mut region_nodes = BTreeSet::new();
        for region in &self.regions {
            if region.contains(&cp) {
                region_nodes.extend(region.iter().copied());
            }
        }
        let options: Vec<usize> = region_nodes
            .intersection(&self.nodes[to_kind])
            .copied()
            .filter(|&j| j != cp && !self.providers[cp].contains(&j) && !self.adj[cp].contains(&j))
            .collect();
        if options.is_empty() {
            return;
        }
        let j = options[self.rng.gen_range(0..options.len())];
        self.add_edge(cp, j, false);
        self.peers[cp] += 1;
        self.peers[j] += 1;
    }

    fn add_peering_links(&mut self, from_kind: usize, to_kind: usize, avg: f64) {
        let from: Vec<usize> = self.nodes[from_kind].iter().copied().collect();
        for i in from {
            let num = self.uniform_int_from_avg(0.0, avg);
            for _ in 0..num {
                if from_kind == KIND_M {
                    self.add_m_peering_link(i);
                } else {
                    self.add_cp_peering_link(i, to_kind);
                }
            }
        }
    }
}

/// Random internet autonomous-system graph (Elmokashfi, Kvalbein and Dovrolis 2010),
/// following NetworkX's `random_internet_as_graph()`.
///
/// Returns each node's kind (`"T"`, `"M"`, `"CP"` or `"C"`) and the edges as
/// `(u, v, transit)`. A transit edge points from the customer to its provider; a peer
/// edge is `(lower, higher)`. NetworkX can return more than `n` nodes when `n < 8`;
/// here the M and CP counts are capped so there are always exactly `n`.
#[pyfunction]
#[pyo3(signature = (n, seed=None))]
pub fn internet_as_graph(n: usize, seed: Option<Seed>) -> PyResult<AsGraphParts> {
    let seed = seed.map(u64::from);
    let mut rng = make_rng(seed);
    let nf = n as f64;
    let n_t = n.min(round_half_even(rng.gen::<f64>() * 2.0 + 4.0) as usize);
    let n_m = (round_half_even(0.15 * nf) as usize).min(n - n_t);
    let n_cp = (round_half_even(0.05 * nf) as usize).min(n - n_t - n_m);
    let n_c = n - n_t - n_m - n_cp;

    let d_m = 2.0 + 2.5 * nf / 10000.0;
    let d_cp = 2.0 + 1.5 * nf / 10000.0;
    let d_c = 1.0 + 5.0 * nf / 100000.0;
    let p_m_m = 1.0 + 2.0 * nf / 10000.0;
    let p_cp_m = 0.2 + 2.0 * nf / 10000.0;
    let p_cp_cp = 0.05 + 2.0 * nf / 100000.0;
    let (t_m, t_cp, t_c) = (0.375, 0.375, 0.125);

    let mut g = AsGraph {
        rng,
        kind: Vec::with_capacity(n),
        adj: Vec::with_capacity(n),
        peers: Vec::with_capacity(n),
        customers: Vec::with_capacity(n),
        providers: Vec::with_capacity(n),
        regions: vec![BTreeSet::new(); 5],
        nodes: Default::default(),
        edges: Vec::new(),
    };
    for _ in 0..n_t {
        let i = g.add_node(KIND_T);
        for region in g.regions.iter_mut() {
            region.insert(i);
        }
        for j in 0..i {
            g.add_edge(j, i, false);
        }
    }
    for _ in 0..n_m {
        g.add_regional_node(KIND_M, 0.2, d_m, t_m);
    }
    for _ in 0..n_cp {
        g.add_regional_node(KIND_CP, 0.05, d_cp, t_cp);
    }
    for _ in 0..n_c {
        g.add_regional_node(KIND_C, 0.0, d_c, t_c);
    }
    g.add_peering_links(KIND_M, KIND_M, p_m_m);
    g.add_peering_links(KIND_CP, KIND_M, p_cp_m);
    g.add_peering_links(KIND_CP, KIND_CP, p_cp_cp);

    let kinds = g.kind.iter().map(|&k| KIND_NAMES[k]).collect();
    Ok((kinds, g.edges))
}

// --- expanders -------------------------------------------------------------------

/// Margulis-Gabber-Galil expander on the `n x n` torus, following NetworkX.
///
/// Node `(x, y)` is index `x * n + y`. Each node adds an edge to each of its four
/// images, so the multigraph is 8-regular (self-loops counting twice) with `4 n^2`
/// edges, each pointing from a node to its image.
#[pyfunction]
pub fn margulis_gabber_galil_edges(n: usize) -> EdgeList {
    let mut edges = Vec::with_capacity(4 * n * n);
    for x in 0..n {
        for y in 0..n {
            let targets = [
                ((x + 2 * y) % n, y),
                ((x + 2 * y + 1) % n, y),
                (x, (y + 2 * x) % n),
                (x, (y + 2 * x + 1) % n),
            ];
            for (u, v) in targets {
                edges.push((x * n + y, u * n + v));
            }
        }
    }
    edges
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        base = ((base as u128 * base as u128) % m as u128) as u64;
        exp >>= 1;
    }
    result
}

/// Chordal cycle graph on `Z/pZ`: the cycle `x -> x + 1`, plus a chord from each `x` to
/// its inverse `x^(p-2) mod p` (`0` to itself). Each cycle edge and each chord appears
/// once, so for a prime `p` the self-inverse nodes `0`, `1` and `p - 1` carry self-loops
/// and the graph is 3-regular counting a loop once. NetworkX's `chordal_cycle_graph()`
/// adds every edge from both ends, doubling all but the self-loops.
#[pyfunction]
pub fn chordal_cycle_edges(p: usize) -> EdgeList {
    let mut edges: EdgeList = (0..p).map(|x| (x, (x + 1) % p)).collect();
    let mut seen = FxHashSet::default();
    for x in 0..p {
        let chord = if x > 0 {
            pow_mod(x as u64, p as u64 - 2, p as u64) as usize
        } else {
            0
        };
        if seen.insert(key(x, chord)) {
            edges.push(key(x, chord));
        }
    }
    edges
}

/// Paley graph on `Z/pZ`: `x -> x + s` for every nonzero square `s`, as in NetworkX's
/// `paley_graph()`. With `directed=false` each pair appears once, as `(lower, higher)`.
#[pyfunction]
#[pyo3(signature = (p, directed=true))]
pub fn paley_edges(p: usize, directed: bool) -> EdgeList {
    let squares: BTreeSet<usize> = (1..p)
        .map(|x| (x as u128 * x as u128 % p as u128) as usize)
        .filter(|&s| s != 0)
        .collect();
    let mut seen = FxHashSet::default();
    let mut edges = Vec::new();
    for x in 0..p {
        for &s in &squares {
            let y = (x + s) % p;
            if directed {
                edges.push((x, y));
            } else if seen.insert(key(x, y)) {
                edges.push(key(x, y));
            }
        }
    }
    edges
}

fn check_expander_args(n: usize, d: usize) -> PyResult<()> {
    if n < 1 {
        return value_error("n must be a positive integer");
    }
    if d < 2 {
        return value_error(format!("d must be at least 2, got {}", d));
    }
    if d % 2 != 0 {
        return value_error(format!("d must be even, got {}", d));
    }
    if n - 1 < d {
        return value_error(format!(
            "need n - 1 >= d to fit {} edge-disjoint cycles on {} nodes",
            d / 2,
            n
        ));
    }
    Ok(())
}

/// `d / 2` edge-disjoint random Hamiltonian cycles, each retried up to `max_tries`
/// times until it shares no edge with the earlier ones; `None` if one can't be placed.
fn union_of_cycles(rng: &mut ChaCha8Rng, n: usize, d: usize, max_tries: usize) -> Option<EdgeList> {
    let mut edges: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut out = Vec::with_capacity(n * d / 2);
    for _ in 0..d / 2 {
        let mut placed = false;
        for _ in 0..max_tries {
            let mut cycle: Vec<usize> = (0..n - 1).collect();
            cycle.shuffle(rng);
            cycle.push(n - 1);
            let new: Vec<(usize, usize)> =
                (0..n).map(|k| key(cycle[k], cycle[(k + 1) % n])).collect();
            if new.iter().all(|e| !edges.contains(e)) {
                edges.extend(new.iter().copied());
                out.extend(new);
                placed = true;
                break;
            }
        }
        if !placed {
            return None;
        }
    }
    out.sort_unstable();
    Some(out)
}

/// A random `d`-regular graph that is likely an expander: the union of `d / 2`
/// edge-disjoint random Hamiltonian cycles, as in NetworkX's `maybe_regular_expander()`.
#[pyfunction]
#[pyo3(signature = (n, d, max_tries=100, seed=None))]
pub fn maybe_regular_expander_edges(
    n: usize,
    d: usize,
    max_tries: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_expander_args(n, d)?;
    match union_of_cycles(&mut make_rng(seed), n, d, max_tries) {
        Some(edges) => Ok(edges),
        None => value_error(format!(
            "no edge-disjoint cycle found in {} tries; raise max_tries",
            max_tries
        )),
    }
}

/// Largest absolute value among the adjacency eigenvalues after the first (sorted by
/// absolute value), for an undirected graph on `0..n`. A self-loop adds `1` to the
/// diagonal, as in NetworkX's `adjacency_matrix()`. Returns 0 for fewer than 2 nodes.
#[pyfunction]
pub fn adjacency_second_eigenvalue(n: usize, edges: Vec<(usize, usize)>) -> PyResult<f64> {
    if edges.iter().any(|&(u, v)| u >= n || v >= n) {
        return value_error("edge endpoint out of range");
    }
    Ok(second_eigenvalue(n, &edges))
}

fn second_eigenvalue(n: usize, edges: &[(usize, usize)]) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let mut a = DMatrix::<f64>::zeros(n, n);
    for &(u, v) in edges {
        a[(u, v)] += 1.0;
        if u != v {
            a[(v, u)] += 1.0;
        }
    }
    let mut mags: Vec<f64> = SymmetricEigen::new(a)
        .eigenvalues
        .iter()
        .map(|x| x.abs())
        .collect();
    mags.sort_unstable_by(|x, y| y.total_cmp(x));
    mags[1]
}

/// Like `maybe_regular_expander_edges`, retried (up to `max_tries` times) until the
/// second-largest adjacency eigenvalue magnitude is below `2 sqrt(d - 1) + epsilon`,
/// as in NetworkX's `random_regular_expander_graph()`.
#[pyfunction]
#[pyo3(signature = (n, d, epsilon=0.0, max_tries=100, seed=None))]
pub fn random_regular_expander_edges(
    n: usize,
    d: usize,
    epsilon: f64,
    max_tries: usize,
    seed: Option<Seed>,
) -> PyResult<EdgeList> {
    let seed = seed.map(u64::from);
    check_expander_args(n, d)?;
    if epsilon.is_nan() || epsilon < 0.0 {
        return value_error(format!("epsilon must be non-negative, got {}", epsilon));
    }
    // a small margin keeps rounding error from passing a graph that sits exactly on the
    // bound, such as an even cycle (eigenvalue -2 against the bound 2 for d = 2)
    let bound = 2.0 * ((d - 1) as f64).sqrt() + epsilon - 1e-9 * d as f64;
    let mut rng = make_rng(seed);
    // a candidate whose cycles can't be placed counts as a failed try
    for _ in 0..max_tries.max(1) {
        if let Some(edges) = union_of_cycles(&mut rng, n, d, max_tries) {
            if second_eigenvalue(n, &edges) < bound {
                return Ok(edges);
            }
        }
    }
    value_error(format!(
        "no candidate passed the expander test in {} tries; raise max_tries or epsilon",
        max_tries
    ))
}

// --- deterministic ---------------------------------------------------------------

/// Sudoku graph of order `n`, following NetworkX's `sudoku_graph()`: the `n^4` cells of
/// an `n^2 x n^2` grid, row by row, joined when they share a row, column or box.
#[pyfunction]
pub fn sudoku_edges(n: usize) -> EdgeList {
    let (n2, n3) = (n * n, n * n * n);
    let n4 = n3 * n;
    let mut edges = Vec::new();
    if n < 2 {
        return edges;
    }
    for row in 0..n2 {
        let start = row * n2;
        for j in 1..n2 {
            for i in 0..j {
                edges.push((start + i, start + j));
            }
        }
    }
    for col in 0..n2 {
        for j in (col..n4).step_by(n2) {
            for i in (col..j).step_by(n2) {
                edges.push((i, j));
            }
        }
    }
    for band in 0..n {
        for stack in 0..n {
            let start = n3 * band + n * stack;
            for j in 1..n2 {
                for i in 0..j {
                    // pairs in the same row or column of the box are already joined
                    if i / n == j / n || i % n == j % n {
                        continue;
                    }
                    edges.push((start + i % n + n2 * (i / n), start + j % n + n2 * (j / n)));
                }
            }
        }
    }
    edges
}

/// Natural visibility graph of a series: `i < j` are joined when every value strictly
/// between them lies below the straight line from `(i, series[i])` to `(j, series[j])`.
///
/// For each `i` the steepest slope seen so far is tracked, so the scan is `O(n^2)`;
/// slopes are compared by cross-multiplying, which is exact for integer values.
#[pyfunction]
pub fn visibility_edges(series: Vec<f64>) -> PyResult<EdgeList> {
    if let Some(v) = series.iter().find(|v| !v.is_finite()) {
        return value_error(format!("series values must be finite, got {}", v));
    }
    let n = series.len();
    let mut edges = Vec::new();
    for i in 0..n {
        let mut best = i + 1; // the point with the steepest slope from i so far
        if best < n {
            edges.push((i, best));
        }
        for j in i + 2..n {
            // slope(i, j) > slope(i, best)
            let lhs = (series[j] - series[i]) * (best - i) as f64;
            let rhs = (series[best] - series[i]) * (j - i) as f64;
            if lhs > rhs {
                edges.push((i, j));
                best = j;
            }
        }
    }
    Ok(edges)
}

/// Interval graph: closed intervals `i < j` are joined when they overlap (touching
/// endpoints count). Returns the sorted pairs.
#[pyfunction]
pub fn interval_edges(intervals: Vec<(f64, f64)>) -> PyResult<EdgeList> {
    for &(lo, hi) in &intervals {
        if lo.is_nan() || hi.is_nan() {
            return value_error("interval endpoints must not be NaN");
        }
        if lo > hi {
            return value_error(format!(
                "each interval must have its lower value first, got ({}, {})",
                lo, hi
            ));
        }
    }
    let mut order: Vec<usize> = (0..intervals.len()).collect();
    order.sort_by(|&a, &b| intervals[a].0.total_cmp(&intervals[b].0).then(a.cmp(&b)));
    let mut active: Vec<usize> = Vec::new();
    let mut edges = Vec::new();
    for &i in &order {
        let lo = intervals[i].0;
        active.retain(|&a| intervals[a].1 >= lo);
        for &a in &active {
            edges.push(key(a, i));
        }
        active.push(i);
    }
    edges.sort_unstable();
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_matches_python() {
        assert_eq!(round_half_even(4.5), 4.0);
        assert_eq!(round_half_even(5.5), 6.0);
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(4.4), 4.0);
        assert_eq!(round_half_even(4.6), 5.0);
    }

    #[test]
    fn sudoku_edge_count() {
        for n in 2..5 {
            let n4 = n * n * n * n;
            let expected = n4 * (3 * n * n - 2 * n - 1) / 2;
            let edges = sudoku_edges(n);
            assert_eq!(edges.len(), expected);
            let distinct: FxHashSet<_> = edges.iter().copied().collect();
            assert_eq!(distinct.len(), expected);
        }
    }

    #[test]
    fn visibility_of_a_valley() {
        // 3 at both ends sees over the dip
        let edges = visibility_edges(vec![3.0, 1.0, 3.0]).unwrap();
        assert_eq!(edges, vec![(0, 1), (0, 2), (1, 2)]);
        // collinear points block each other
        let edges = visibility_edges(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn interval_overlaps() {
        let edges = interval_edges(vec![(0.0, 1.0), (1.0, 2.0), (3.0, 4.0), (0.5, 3.0)]).unwrap();
        assert_eq!(edges, vec![(0, 1), (0, 3), (1, 3), (2, 3)]);
    }

    #[test]
    fn expander_union_is_regular() {
        let edges = maybe_regular_expander_edges(20, 4, 100, Some(Seed::from(3))).unwrap();
        let mut degree = [0; 20];
        for (u, v) in edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        assert!(degree.iter().all(|&d| d == 4));
    }

    #[test]
    fn internet_as_has_n_nodes() {
        for n in [0, 1, 4, 7, 50, 300] {
            let (kinds, _) = internet_as_graph(n, Some(Seed::from(1))).unwrap();
            assert_eq!(kinds.len(), n);
        }
    }
}
