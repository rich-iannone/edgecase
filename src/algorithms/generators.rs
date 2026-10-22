//! Random generators for community benchmark graphs.
//!
//! These return plain index structures over nodes `0..n` (edge pairs and, where
//! the model has them, each node's planted community). The Python wrappers add
//! the nodes, store the ground truth, and wire the edges by ID.
//!
//! - `random_partition_edges`: planted partition on given block sizes
//! - `gaussian_random_partition`: block sizes drawn from a normal distribution
//! - `relaxed_caveman_edges`: disjoint cliques with each edge rewired with probability p
//! - `lfr_benchmark`: the Lancichinetti-Fortunato-Radicchi benchmark
//!
//! The random, degree-sequence, directed, and geometric models live in
//! `random_models`, which shares the helpers below.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) type EdgeList = Vec<(usize, usize)>;

pub(crate) fn make_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

pub(crate) fn key(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn check_probability(name: &str, p: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!("{} must be between 0.0 and 1.0, got {}", name, p)));
    }
    Ok(())
}

/// Call `f(b)` for each `b` in `lo..hi` kept independently with probability `p`.
///
/// Gaps between kept values are drawn from a geometric distribution, so a sparse
/// range costs time proportional to the values kept rather than to its length.
fn sample_range(rng: &mut ChaCha8Rng, p: f64, lo: usize, hi: usize, mut f: impl FnMut(usize)) {
    if p <= 0.0 || lo >= hi {
        return;
    }
    if p >= 1.0 {
        (lo..hi).for_each(f);
        return;
    }
    let log_q = (1.0 - p).ln();
    let mut b = lo;
    while b < hi {
        let r: f64 = rng.gen();
        let skip = ((1.0 - r).ln() / log_q).floor();
        if skip >= (hi - b) as f64 {
            break;
        }
        b += skip as usize;
        f(b);
        b += 1;
    }
}

/// Edges of a planted partition: blocks are consecutive runs of nodes.
fn partition_edges(
    rng: &mut ChaCha8Rng,
    sizes: &[usize],
    p_in: f64,
    p_out: f64,
    directed: bool,
) -> EdgeList {
    let n: usize = sizes.iter().sum();
    let mut out = Vec::new();
    let mut start = 0;
    for &size in sizes {
        let end = start + size;
        for a in start..end {
            if directed {
                sample_range(rng, p_out, 0, start, |b| out.push((a, b)));
                sample_range(rng, p_in, start, a, |b| out.push((a, b)));
                sample_range(rng, p_in, a + 1, end, |b| out.push((a, b)));
            } else {
                sample_range(rng, p_in, a + 1, end, |b| out.push((a, b)));
            }
            sample_range(rng, p_out, end, n, |b| out.push((a, b)));
        }
        start = end;
    }
    out
}

/// Planted partition (random partition) graph on consecutive blocks of `sizes`.
///
/// Each pair inside a block is joined with probability `p_in` and each pair across
/// blocks with `p_out`; with `directed`, ordered pairs are drawn independently.
#[pyfunction]
#[pyo3(signature = (sizes, p_in, p_out, directed=false, seed=None))]
pub fn random_partition_edges(
    sizes: Vec<usize>,
    p_in: f64,
    p_out: f64,
    directed: bool,
    seed: Option<u64>,
) -> PyResult<EdgeList> {
    check_probability("p_in", p_in)?;
    check_probability("p_out", p_out)?;
    let mut rng = make_rng(seed);
    Ok(partition_edges(&mut rng, &sizes, p_in, p_out, directed))
}

/// Standard normal draw (Box-Muller).
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>(); // (0, 1], so the log is finite
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gaussian random partition graph (Brandes, Gaertler and Wagner 2003).
///
/// Block sizes are drawn from a normal distribution with mean `mean` and variance
/// `mean / shape` (rounded, sizes below 1 redrawn) until they cover `n` nodes; the
/// last block takes whatever is left. Returns the sizes and the edges.
#[pyfunction]
#[pyo3(signature = (n, mean, shape, p_in, p_out, directed=false, seed=None))]
pub fn gaussian_random_partition(
    n: usize,
    mean: f64,
    shape: f64,
    p_in: f64,
    p_out: f64,
    directed: bool,
    seed: Option<u64>,
) -> PyResult<(Vec<usize>, EdgeList)> {
    if !(mean.is_finite() && mean >= 1.0) {
        return Err(PyValueError::new_err(format!("mean block size must be at least 1, got {}", mean)));
    }
    if !(shape.is_finite() && shape > 0.0) {
        return Err(PyValueError::new_err(format!("shape must be positive, got {}", shape)));
    }
    check_probability("p_in", p_in)?;
    check_probability("p_out", p_out)?;
    let mut rng = make_rng(seed);
    let std = (mean / shape).sqrt();
    let mut sizes = Vec::new();
    let mut assigned = 0;
    while assigned < n {
        let size = (mean + std * normal(&mut rng)).round();
        if size < 1.0 {
            continue;
        }
        let size = (size as usize).min(n - assigned);
        sizes.push(size);
        assigned += size;
    }
    let edges = partition_edges(&mut rng, &sizes, p_in, p_out, directed);
    Ok((sizes, edges))
}

/// Relaxed caveman graph: `l` cliques of `k` nodes, each edge rewired with probability `p`.
///
/// Edges are visited in clique order; a rewired edge `(u, v)` becomes `(u, x)` for a
/// uniformly random node `x`, and the rewiring is skipped when `x` is `u` or already
/// adjacent to it, so the edge count never changes.
#[pyfunction]
#[pyo3(signature = (l, k, p, seed=None))]
pub fn relaxed_caveman_edges(l: usize, k: usize, p: f64, seed: Option<u64>) -> PyResult<EdgeList> {
    check_probability("p", p)?;
    let mut rng = make_rng(seed);
    let n = l * k;
    let mut edges: EdgeList = Vec::with_capacity(l * k * k.saturating_sub(1) / 2);
    for c in 0..l {
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((c * k + i, c * k + j));
            }
        }
    }
    let mut present: FxHashSet<(usize, usize)> = edges.iter().copied().collect();
    for edge in edges.iter_mut() {
        if rng.gen::<f64>() >= p {
            continue;
        }
        let (u, v) = *edge;
        let x = rng.gen_range(0..n);
        if x == u || present.contains(&key(u, x)) {
            continue;
        }
        present.remove(&key(u, v));
        present.insert(key(u, x));
        *edge = key(u, x);
    }
    Ok(edges)
}

// --- LFR benchmark ------------------------------------------------------------

/// Integral of x^m over [a, b].
fn power_integral(m: f64, a: f64, b: f64) -> f64 {
    if (m + 1.0).abs() < 1e-12 {
        (b / a).ln()
    } else {
        (b.powf(m + 1.0) - a.powf(m + 1.0)) / (m + 1.0)
    }
}

/// Mean of the continuous power law x^-tau on [a, b].
fn power_law_mean(tau: f64, a: f64, b: f64) -> f64 {
    if b - a < 1e-12 {
        return a;
    }
    power_integral(1.0 - tau, a, b) / power_integral(-tau, a, b)
}

/// One draw from the continuous power law x^-tau on [a, b] (inverse CDF).
fn power_law_draw(rng: &mut ChaCha8Rng, tau: f64, a: f64, b: f64) -> f64 {
    let u: f64 = rng.gen();
    let m = 1.0 - tau;
    if m.abs() < 1e-12 {
        a * (b / a).powf(u)
    } else {
        (a.powf(m) + u * (b.powf(m) - a.powf(m))).powf(1.0 / m)
    }
}

/// Integer draw in [lo, hi] from a power law with exponent `tau`.
fn power_law_int(rng: &mut ChaCha8Rng, tau: f64, lo: usize, hi: usize) -> usize {
    // widen by half a unit on each side so rounding gives the end values their share
    let x = power_law_draw(rng, tau, lo as f64 - 0.5 + 1e-9, hi as f64 + 0.5 - 1e-9);
    (x.round() as usize).clamp(lo, hi)
}

/// Community sizes in [smin, smax] summing to exactly `n`.
fn community_sizes(rng: &mut ChaCha8Rng, n: usize, tau: f64, smin: usize, smax: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut total = 0;
    while total < n {
        let s = power_law_int(rng, tau, smin, smax);
        sizes.push(s);
        total += s;
    }
    // Trim the overshoot (or fill a gap after dropping a community) one node at a
    // time from random communities that can still shrink or grow. The caller has
    // checked that some count of communities can cover n, so this terminates.
    while total != n {
        if total > n {
            let shrinkable: Vec<usize> = (0..sizes.len()).filter(|&i| sizes[i] > smin).collect();
            if let Some(&i) = shrinkable.choose(rng) {
                sizes[i] -= 1;
                total -= 1;
            } else {
                total -= sizes.pop().unwrap();
            }
        } else {
            let growable: Vec<usize> = (0..sizes.len()).filter(|&i| sizes[i] < smax).collect();
            if let Some(&i) = growable.choose(rng) {
                sizes[i] += 1;
                total += 1;
            } else {
                sizes.push(smin);
                total += smin;
            }
        }
    }
    sizes
}

/// True if nodes with internal degrees `kin` can be packed into communities of
/// `sizes` with every node in a community larger than its internal degree.
fn communities_fit(kin_desc: &[usize], sizes: &[usize]) -> bool {
    // for every threshold t: #nodes with kin >= t <= total size of communities with size > t
    let mut sizes_desc = sizes.to_vec();
    sizes_desc.sort_unstable_by(|a, b| b.cmp(a));
    let mut capacity = 0;
    let mut j = 0;
    for (count, &t) in kin_desc.iter().enumerate() {
        while j < sizes_desc.len() && sizes_desc[j] > t {
            capacity += sizes_desc[j];
            j += 1;
        }
        if count + 1 > capacity {
            return false;
        }
    }
    true
}

/// Make `edges` simple by degree-preserving swaps.
///
/// An edge is bad if it is a self-loop, a repeat, or joins two nodes of the same
/// community when `comm` is given (the external wiring). Each bad edge is swapped
/// with a random partner, `(a, b), (c, d) -> (a, c), (b, d)` or `(a, d), (b, c)`,
/// when both new edges are good. Edges still bad after the rounds are dropped.
fn make_simple(rng: &mut ChaCha8Rng, edges: &mut EdgeList, comm: Option<&[usize]>) {
    let forbidden = |a: usize, b: usize| a == b || comm.is_some_and(|c| c[a] == c[b]);
    let mut count: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    for &(a, b) in edges.iter() {
        *count.entry(key(a, b)).or_insert(0) += 1;
    }
    let is_bad = |e: (usize, usize), count: &FxHashMap<(usize, usize), u32>| {
        forbidden(e.0, e.1) || count[&key(e.0, e.1)] > 1
    };
    let m = edges.len();
    let mut bad: Vec<usize> = (0..m).filter(|&i| is_bad(edges[i], &count)).collect();
    for _round in 0..100 {
        if bad.is_empty() || m < 2 {
            break;
        }
        for &i in &bad {
            if !is_bad(edges[i], &count) {
                continue; // fixed when it was some other edge's partner
            }
            for _attempt in 0..20 {
                let j = rng.gen_range(0..m);
                if j == i {
                    continue;
                }
                let (a, b) = edges[i];
                let (c, d) = edges[j];
                let (e1, e2) = if rng.gen::<bool>() { ((a, c), (b, d)) } else { ((a, d), (b, c)) };
                let free = |e: (usize, usize)| {
                    !forbidden(e.0, e.1) && count.get(&key(e.0, e.1)).copied().unwrap_or(0) == 0
                };
                if key(e1.0, e1.1) == key(e2.0, e2.1) || !free(e1) || !free(e2) {
                    continue;
                }
                for old in [edges[i], edges[j]] {
                    *count.get_mut(&key(old.0, old.1)).unwrap() -= 1;
                }
                for new in [e1, e2] {
                    *count.entry(key(new.0, new.1)).or_insert(0) += 1;
                }
                edges[i] = e1;
                edges[j] = e2;
                break;
            }
        }
        bad.retain(|&i| is_bad(edges[i], &count));
    }
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    edges.retain(|&(a, b)| !forbidden(a, b) && seen.insert(key(a, b)));
}

/// Pair up a stub list at random (configuration model).
fn pair_stubs(rng: &mut ChaCha8Rng, mut stubs: Vec<usize>) -> EdgeList {
    stubs.shuffle(rng);
    stubs.chunks_exact(2).map(|p| (p[0], p[1])).collect()
}

/// LFR benchmark graph (Lancichinetti, Fortunato and Radicchi 2008).
///
/// 1. Degrees follow a power law with exponent `tau1` on `[k_min, max_degree]`,
///    with `k_min` solved so the mean is `average_degree`.
/// 2. Each node splits its degree into internal `(1 - mu) k` and external `mu k`
///    parts (stochastic rounding, so the mixing is right on average).
/// 3. Community sizes follow a power law with exponent `tau2` on
///    `[min_community, max_community]` and sum to `n`; they are redrawn until every
///    node can sit in a community larger than its internal degree.
/// 4. Nodes are placed largest internal degree first into random communities with
///    room, then each community and the between-community remainder are wired by
///    the configuration model and made simple by degree-preserving swaps.
///
/// Nodes are numbered community by community. Returns the edges and each node's
/// community (0-based).
#[pyfunction]
#[pyo3(signature = (n, mu, average_degree, max_degree, tau1, tau2, min_community=None, max_community=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn lfr_benchmark(
    n: usize,
    mu: f64,
    average_degree: f64,
    max_degree: usize,
    tau1: f64,
    tau2: f64,
    min_community: Option<usize>,
    max_community: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(EdgeList, Vec<usize>)> {
    check_probability("mu", mu)?;
    for (name, tau) in [("tau1", tau1), ("tau2", tau2)] {
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(PyValueError::new_err(format!("{} must be a non-negative number, got {}", name, tau)));
        }
    }
    if n < 2 {
        return Err(PyValueError::new_err("an LFR graph needs at least 2 nodes"));
    }
    if max_degree < 1 || max_degree > n - 1 {
        return Err(PyValueError::new_err(format!(
            "max_degree must be between 1 and n - 1 = {}, got {}",
            n - 1,
            max_degree
        )));
    }
    let kmax = max_degree as f64;
    let lowest_mean = power_law_mean(tau1, 1.0, kmax);
    if !(average_degree.is_finite() && average_degree >= lowest_mean && average_degree <= kmax) {
        return Err(PyValueError::new_err(format!(
            "average_degree must be between {:.2} (the mean of the degree power law when the \
             smallest degree is 1) and max_degree = {}, got {}",
            lowest_mean, max_degree, average_degree
        )));
    }

    let mut rng = make_rng(seed);

    // 1. degrees: bisect for the lower cutoff whose mean matches average_degree
    let (mut lo, mut hi) = (1.0, kmax);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if power_law_mean(tau1, mid, kmax) < average_degree {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let kmin = 0.5 * (lo + hi);
    let mut degree: Vec<usize> = (0..n)
        .map(|_| (power_law_draw(&mut rng, tau1, kmin, kmax).round() as usize).clamp(1, max_degree))
        .collect();
    if degree.iter().sum::<usize>() % 2 == 1 {
        // even degree sum: nudge one node that has room
        let i = (0..n).filter(|&i| degree[i] < max_degree).choose(&mut rng);
        match i {
            Some(i) => degree[i] += 1,
            None => degree[0] -= 1,
        }
    }

    // 2. internal / external split
    let mut kin: Vec<usize> = degree
        .iter()
        .map(|&k| {
            let x = (1.0 - mu) * k as f64;
            let base = x.floor();
            let extra = if rng.gen::<f64>() < x - base { 1 } else { 0 };
            (base as usize + extra).min(k)
        })
        .collect();
    let kin_max = kin.iter().copied().max().unwrap_or(0);

    // 3. community sizes
    let smin = min_community.unwrap_or_else(|| (kmin.round() as usize).max(1));
    let smax = max_community.unwrap_or_else(|| max_degree.max(kin_max + 1)).min(n);
    if smin < 1 || smin > smax {
        return Err(PyValueError::new_err(format!(
            "min_community ({}) must be at least 1 and no larger than max_community ({})",
            smin, smax
        )));
    }
    if (n + smax - 1) / smax > n / smin {
        return Err(PyValueError::new_err(format!(
            "{} nodes can't be split into communities of {} to {} nodes",
            n, smin, smax
        )));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    order.sort_by(|&a, &b| kin[b].cmp(&kin[a])); // stable: ties stay shuffled
    let kin_desc: Vec<usize> = order.iter().map(|&v| kin[v]).collect();
    let mut sizes = Vec::new();
    for attempt in 0..=200 {
        if attempt == 200 {
            return Err(PyValueError::new_err(
                "could not draw community sizes large enough for the nodes' internal degrees; \
                 raise max_community, lower max_degree, or raise mu",
            ));
        }
        sizes = community_sizes(&mut rng, n, tau2, smin, smax);
        if communities_fit(&kin_desc, &sizes) {
            break;
        }
    }

    // 4. placement: largest internal degree first, into a random community that
    // has room and is larger than the node's internal degree. The fit check above
    // guarantees such a community always exists.
    let ncomm = sizes.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncomm];
    let mut open: Vec<usize> = (0..ncomm).collect(); // communities with free slots
    let mut comm = vec![0usize; n];
    for &v in &order {
        let fits = |c: usize| sizes[c] > kin[v];
        let mut slot = None;
        for _ in 0..32 {
            let i = rng.gen_range(0..open.len());
            if fits(open[i]) {
                slot = Some(i);
                break;
            }
        }
        let candidates: Vec<usize>;
        let i = match slot {
            Some(i) => i,
            None => {
                candidates = (0..open.len()).filter(|&i| fits(open[i])).collect();
                *candidates.choose(&mut rng).expect("fit check guarantees a community")
            }
        };
        let c = open[i];
        members[c].push(v);
        comm[v] = c;
        if members[c].len() == sizes[c] {
            open.swap_remove(i);
        }
    }

    // internal stub sums must be even: move one stub between internal and external
    for (c, group) in members.iter().enumerate() {
        if group.iter().map(|&v| kin[v]).sum::<usize>() % 2 == 0 {
            continue;
        }
        if let Some(&v) = group.iter().find(|&&v| kin[v] < degree[v] && kin[v] + 1 < sizes[c]) {
            kin[v] += 1;
        } else if let Some(&v) = group.iter().find(|&&v| kin[v] > 0) {
            kin[v] -= 1;
        }
    }

    // wiring
    let mut edges: EdgeList = Vec::new();
    for group in &members {
        let stubs: Vec<usize> = group.iter().flat_map(|&v| std::iter::repeat(v).take(kin[v])).collect();
        let mut internal = pair_stubs(&mut rng, stubs);
        make_simple(&mut rng, &mut internal, None);
        edges.extend(internal);
    }
    let stubs: Vec<usize> =
        (0..n).flat_map(|v| std::iter::repeat(v).take(degree[v] - kin[v])).collect();
    let mut external = pair_stubs(&mut rng, stubs);
    make_simple(&mut rng, &mut external, Some(&comm));
    edges.extend(external);

    // renumber community by community
    let mut new_id = vec![0usize; n];
    let mut communities = Vec::with_capacity(n);
    let mut next = 0;
    for (c, group) in members.iter_mut().enumerate() {
        group.sort_unstable();
        for &v in group.iter() {
            new_id[v] = next;
            communities.push(c);
            next += 1;
        }
    }
    let mut edges: EdgeList = edges.into_iter().map(|(a, b)| key(new_id[a], new_id[b])).collect();
    edges.sort_unstable();
    Ok((edges, communities))
}
