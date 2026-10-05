//! Trees and branchings: Edmonds' optimum branchings and spanning arborescences, the
//! matrix-tree theorem count of spanning trees, weighted uniform spanning tree sampling
//! (Wilson's algorithm), branching recognition and Prüfer sequences.
//!
//! Edge values come from the edge attribute named by `weight_attr`, else the edge weight
//! (1.0 when unset); missing or non-numeric attribute values count as 1.0, as in
//! `minimum_spanning_tree()`.

use crate::graph::{Edge, Graph, Node};
use nalgebra as na;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn edge_value(py: Python, e: &Edge, weight_attr: Option<&str>) -> PyResult<f64> {
    Ok(match weight_attr {
        None => e.weight.unwrap_or(1.0),
        Some(attr) => match &e.attributes {
            Some(d) => match d.bind(py).downcast::<PyDict>()?.get_item(attr)? {
                Some(v) => v.extract::<f64>().unwrap_or(1.0),
                None => 1.0,
            },
            None => 1.0,
        },
    })
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// True when the graph's edges (direction ignored) connect all `n` indexed nodes.
fn spans(n: usize, pairs: impl Iterator<Item = (usize, usize)>) -> bool {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut parts = n;
    for (a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
            parts -= 1;
        }
    }
    parts <= 1
}

// ---------------------------------------------------------------------------------------------
// Edmonds' algorithm
// ---------------------------------------------------------------------------------------------

/// A branching weight ordered lexicographically: first the edge count `k`, then the total
/// `w`. Optimum branchings use `k = 0` (pure weight); spanning arborescences use `k = 1`
/// per edge, so the maximum first takes as many edges as possible (`n - 1` when a spanning
/// arborescence exists) and only then the best weight. Karp's contraction only needs an
/// ordered group, so the pair works exactly, without a large additive constant.
#[derive(Clone, Copy, Debug)]
struct W {
    k: i64,
    w: f64,
}

impl W {
    fn cmp(&self, other: &W) -> Ordering {
        self.k.cmp(&other.k).then(self.w.partial_cmp(&other.w).unwrap_or(Ordering::Equal))
    }
    fn positive(&self) -> bool {
        self.k > 0 || (self.k == 0 && self.w > 0.0)
    }
    fn shift(&self, minus: W, plus: W) -> W {
        W { k: self.k - minus.k + plus.k, w: self.w - minus.w + plus.w }
    }
}

/// Maximum-weight branching (Edmonds 1967, with Karp's 1971 cycle contraction) on nodes
/// `0..n`. Returns indices into `edges`. Only positive-weight edges can be chosen; ties go
/// to the earliest edge.
fn max_branching(n: usize, edges: &[(usize, usize, W)]) -> Vec<usize> {
    // Heaviest positive in-edge of every node
    let mut best: Vec<Option<usize>> = vec![None; n];
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        if u == v || !w.positive() {
            continue;
        }
        match best[v] {
            Some(j) if edges[j].2.cmp(&w) != Ordering::Less => {}
            _ => best[v] = Some(i),
        }
    }

    // Cycles of the chosen in-edges (each node has at most one, so each is a simple cycle)
    let mut state = vec![0u8; n]; // 0 new, 1 on the current walk, 2 done
    let mut cycle_of = vec![usize::MAX; n];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        let mut walk = Vec::new();
        let mut x = start;
        while state[x] == 0 {
            state[x] = 1;
            walk.push(x);
            match best[x] {
                Some(e) => x = edges[e].0,
                None => break,
            }
        }
        if state[x] == 1 && best[x].is_some() && walk.contains(&x) {
            let pos = walk.iter().position(|&y| y == x).unwrap();
            let cycle: Vec<usize> = walk[pos..].to_vec();
            for &y in &cycle {
                cycle_of[y] = cycles.len();
            }
            cycles.push(cycle);
        }
        for y in walk {
            state[y] = 2;
        }
    }
    if cycles.is_empty() {
        return best.into_iter().flatten().collect();
    }

    // Contract every cycle into one node
    let mut comp = vec![usize::MAX; n];
    let mut m = cycles.len();
    for (c, cycle) in cycles.iter().enumerate() {
        for &y in cycle {
            comp[y] = c;
        }
    }
    for x in 0..n {
        if comp[x] == usize::MAX {
            comp[x] = m;
            m += 1;
        }
    }
    let min_edge: Vec<usize> = cycles
        .iter()
        .map(|cycle| {
            let mut pick = best[cycle[0]].unwrap();
            for &y in &cycle[1..] {
                let e = best[y].unwrap();
                if edges[e].2.cmp(&edges[pick].2) == Ordering::Less {
                    pick = e;
                }
            }
            pick
        })
        .collect();
    let mut sub_edges = Vec::new();
    let mut origin = Vec::new();
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        let (cu, cv) = (comp[u], comp[v]);
        if cu == cv {
            continue;
        }
        let w2 = match cycle_of[v] {
            usize::MAX => w,
            c => w.shift(edges[best[v].unwrap()].2, edges[min_edge[c]].2),
        };
        sub_edges.push((cu, cv, w2));
        origin.push(i);
    }

    let chosen: Vec<usize> = max_branching(m, &sub_edges).into_iter().map(|j| origin[j]).collect();
    let mut entered: Vec<Option<usize>> = vec![None; cycles.len()];
    for &e in &chosen {
        let v = edges[e].1;
        if cycle_of[v] != usize::MAX {
            entered[cycle_of[v]] = Some(v);
        }
    }
    let mut result = chosen;
    for (c, cycle) in cycles.iter().enumerate() {
        let drop = match entered[c] {
            Some(v) => best[v].unwrap(),
            None => min_edge[c],
        };
        result.extend(cycle.iter().map(|&y| best[y].unwrap()).filter(|&e| e != drop));
    }
    result
}

/// Optimum branching or spanning arborescence: the chosen edge IDs, in source edge order.
///
/// `kind` is one of `"max_branching"`, `"min_branching"`, `"max_arborescence"` and
/// `"min_arborescence"`. A minimum branching maximizes the negated weights, so it only uses
/// negative-weight edges (as in NetworkX); spanning arborescences raise when none exists.
#[pyfunction]
#[pyo3(signature = (graph, kind, weight_attr=None))]
pub fn optimum_branching(
    py: Python,
    graph: &Graph,
    kind: &str,
    weight_attr: Option<&str>,
) -> PyResult<Vec<u64>> {
    if !graph.is_directed() {
        return value_err("branchings and arborescences require a directed graph");
    }
    let (spanning, sign) = match kind {
        "max_branching" => (false, 1.0),
        "min_branching" => (false, -1.0),
        "max_arborescence" => (true, 1.0),
        "min_arborescence" => (true, -1.0),
        _ => return value_err(format!("unknown branching kind {kind:?}")),
    };
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut edges = Vec::new();
    let mut edge_ids = Vec::new();
    for e in graph.get_edges_internal() {
        let w = edge_value(py, e, weight_attr)?;
        if w.is_nan() {
            return value_err("edge values must not be NaN");
        }
        edges.push((idx[&e.source], idx[&e.target], W { k: spanning as i64, w: sign * w }));
        edge_ids.push(e.id);
    }
    let mut chosen = max_branching(n, &edges);
    if spanning && n > 0 && chosen.len() != n - 1 {
        return value_err("no spanning arborescence exists: not every node can be reached from a single root");
    }
    chosen.sort_unstable();
    Ok(chosen.into_iter().map(|i| edge_ids[i]).collect())
}

/// The values of the given edges (see the module notes), in the given order.
#[pyfunction]
#[pyo3(signature = (graph, edge_ids, weight_attr=None))]
pub fn edge_values(py: Python, graph: &Graph, edge_ids: Vec<u64>, weight_attr: Option<&str>) -> PyResult<Vec<f64>> {
    let by_id: FxHashMap<u64, &Edge> = graph.get_edges_internal().iter().map(|e| (e.id, e)).collect();
    edge_ids
        .iter()
        .map(|eid| match by_id.get(eid) {
            Some(e) => edge_value(py, e, weight_attr),
            None => value_err(format!("Edge with ID {eid} not found in graph")),
        })
        .collect()
}

/// A graph with every node of `graph` (IDs, labels and attributes kept) and only the given
/// edges (weights and attributes copied), with the same directedness and weighting.
#[pyfunction]
pub fn spanning_subgraph(py: Python, graph: &Graph, edge_ids: Vec<u64>) -> PyResult<Graph> {
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let (ids, _) = sorted_ids(graph);
    let nodes = graph.get_nodes();
    for id in ids {
        let node = &nodes[&id];
        let attributes = match &node.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_node_rust(Node { id, label: node.label.clone(), attributes })?;
    }
    let by_id: FxHashMap<u64, &Edge> = graph.get_edges_internal().iter().map(|e| (e.id, e)).collect();
    for eid in edge_ids {
        let Some(e) = by_id.get(&eid) else {
            return value_err(format!("Edge with ID {eid} not found in graph"));
        };
        let attributes = match &e.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_edge_rust(Edge { id: 0, source: e.source, target: e.target, weight: e.weight, attributes })?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Counting and sampling spanning trees
// ---------------------------------------------------------------------------------------------

/// Exact determinant of an integer matrix by Bareiss' fraction-free elimination, or None on
/// i128 overflow.
fn bareiss(mut a: Vec<Vec<i128>>) -> Option<i128> {
    let n = a.len();
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n {
        if a[k][k] == 0 {
            let Some(swap) = (k + 1..n).find(|&r| a[r][k] != 0) else {
                return Some(0);
            };
            a.swap(k, swap);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let x = a[i][j].checked_mul(a[k][k])?;
                let y = a[i][k].checked_mul(a[k][j])?;
                a[i][j] = x.checked_sub(y)? / prev;
            }
        }
        prev = a[k][k];
    }
    if n == 0 {
        return Some(1);
    }
    Some(sign * a[n - 1][n - 1])
}

/// Number of spanning trees (undirected) or of spanning arborescences rooted at `root` and
/// directed away from it, by Kirchhoff's matrix-tree theorem. With `weighted`, each tree
/// counts as the product of its edge weights. Unweighted counts are exact integers.
#[pyfunction]
#[pyo3(signature = (graph, root=None, weighted=false))]
pub fn number_of_spanning_trees(
    py: Python,
    graph: &Graph,
    root: Option<u64>,
    weighted: bool,
) -> PyResult<PyObject> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n == 0 {
        return value_err("the graph must contain at least one node");
    }
    let directed = graph.is_directed();
    let skip = match (directed, root) {
        (true, None) => return value_err("root= must be given for a directed graph"),
        (true, Some(r)) => match idx.get(&r) {
            Some(&i) => i,
            None => return value_err(format!("Root node {r} does not exist")),
        },
        (false, _) => 0,
    };
    let edges: Vec<(usize, usize, f64)> = graph
        .get_edges_internal()
        .iter()
        .filter(|e| e.source != e.target)
        .map(|e| (idx[&e.source], idx[&e.target], if weighted { e.weight.unwrap_or(1.0) } else { 1.0 }))
        .collect();
    if !spans(n, edges.iter().map(|&(a, b, _)| (a, b))) {
        return Ok(if weighted { 0.0f64.into_py(py) } else { 0i64.into_py(py) });
    }
    // Laplacian (out-degree convention for undirected, in-degree for arborescences)
    let mut lap = vec![vec![0.0f64; n]; n];
    for &(a, b, w) in &edges {
        lap[a][b] -= w;
        lap[b][b] += w;
        if !directed {
            lap[b][a] -= w;
            lap[a][a] += w;
        }
    }
    let keep: Vec<usize> = (0..n).filter(|&i| i != skip).collect();
    let m = keep.len();
    if !weighted {
        let minor: Vec<Vec<i128>> =
            keep.iter().map(|&i| keep.iter().map(|&j| lap[i][j] as i128).collect()).collect();
        if let Some(det) = bareiss(minor.clone()) {
            return Ok(det.into_py(py));
        }
        // Too large for 128 bits: residues modulo enough 61-bit primes to pin the value down
        // (Hadamard's bound), combined into an exact integer by the Python wrapper (CRT)
        let bits: f64 = minor
            .iter()
            .map(|row| row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt().max(1.0).log2())
            .sum();
        let count = (bits / 60.0).ceil() as usize + 2;
        let mut residues = Vec::with_capacity(count);
        let mut p = (1u64 << 61) - 1;
        while residues.len() < count {
            if is_prime(p) {
                residues.push((p, det_mod(&minor, p)));
            }
            p -= 2;
        }
        return Ok(residues.into_py(py));
    }
    let det = na::DMatrix::from_fn(m, m, |i, j| lap[keep[i]][keep[j]]).determinant();
    Ok(det.into_py(py))
}

fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, a, p);
        }
        a = mul_mod(a, a, p);
        e >>= 1;
    }
    r
}

/// Deterministic Miller-Rabin for 64-bit integers.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let bases = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &b in &bases {
        if n % b == 0 {
            return n == b;
        }
    }
    let (mut d, mut r) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    'outer: for &a in &bases {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Determinant of an integer matrix modulo the prime `p` (Gaussian elimination).
fn det_mod(a: &[Vec<i128>], p: u64) -> u64 {
    let n = a.len();
    let mut m: Vec<Vec<u64>> =
        a.iter().map(|row| row.iter().map(|&x| x.rem_euclid(p as i128) as u64).collect()).collect();
    let mut det = 1u64;
    for k in 0..n {
        let Some(piv) = (k..n).find(|&r| m[r][k] != 0) else {
            return 0;
        };
        if piv != k {
            m.swap(piv, k);
            det = p - det;
        }
        det = mul_mod(det, m[k][k], p);
        let inv = pow_mod(m[k][k], p - 2, p);
        for i in k + 1..n {
            if m[i][k] == 0 {
                continue;
            }
            let f = mul_mod(m[i][k], inv, p);
            for j in k..n {
                let sub = mul_mod(f, m[k][j], p);
                m[i][j] = (m[i][j] + p - sub) % p;
            }
        }
    }
    det % p
}

/// A random spanning tree of a connected undirected graph, chosen with probability
/// proportional to the product of its edge weights (uniformly when unweighted), by
/// Wilson's loop-erased random walk algorithm. Returns edge IDs.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, seed=None))]
pub fn random_spanning_tree(graph: &Graph, weighted: bool, seed: Option<u64>) -> PyResult<Vec<u64>> {
    if graph.is_directed() {
        return value_err("random_spanning_tree() requires an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    // adjacency: (neighbor, edge id, cumulative weight)
    let mut adj: Vec<Vec<(usize, u64, f64)>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let w = if weighted { e.weight.unwrap_or(1.0) } else { 1.0 };
        if !(w >= 0.0) {
            return value_err("edge weights must be non-negative");
        }
        if w == 0.0 {
            continue;
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        adj[a].push((b, e.id, w));
        adj[b].push((a, e.id, w));
    }
    if n == 0 || !spans(n, adj.iter().enumerate().flat_map(|(a, l)| l.iter().map(move |&(b, _, _)| (a, b)))) {
        return value_err("the graph must be non-empty and connected (positive-weight edges)");
    }
    for list in adj.iter_mut() {
        let mut total = 0.0;
        for item in list.iter_mut() {
            total += item.2;
            item.2 = total;
        }
    }
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let mut in_tree = vec![false; n];
    let mut next: Vec<(usize, u64)> = vec![(0, 0); n];
    in_tree[rng.gen_range(0..n)] = true;
    let mut tree = Vec::with_capacity(n.saturating_sub(1));
    for start in 0..n {
        let mut u = start;
        while !in_tree[u] {
            let list = &adj[u];
            let r = rng.gen::<f64>() * list[list.len() - 1].2;
            let k = list.partition_point(|item| item.2 <= r).min(list.len() - 1);
            next[u] = (list[k].0, list[k].1);
            u = list[k].0;
        }
        let mut u = start;
        while !in_tree[u] {
            in_tree[u] = true;
            tree.push(next[u].1);
            u = next[u].0;
        }
    }
    Ok(tree)
}

// ---------------------------------------------------------------------------------------------
// Recognition and Prüfer sequences
// ---------------------------------------------------------------------------------------------

/// True for a directed forest in which every node has at most one incoming edge.
#[pyfunction]
pub fn is_branching(graph: &Graph) -> bool {
    if !graph.is_directed() || graph.get_node_ids().is_empty() {
        return false;
    }
    let (_, idx) = sorted_ids(graph);
    let n = idx.len();
    let mut indeg = vec![0usize; n];
    let mut parent: Vec<usize> = (0..n).collect();
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        indeg[b] += 1;
        if indeg[b] > 1 {
            return false;
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            return false;
        }
        parent[ra] = rb;
    }
    true
}

/// Prüfer sequence of an undirected tree, ranking nodes by ascending ID. Returns node IDs.
#[pyfunction]
pub fn to_prufer_sequence(graph: &Graph) -> PyResult<Vec<u64>> {
    if graph.is_directed() {
        return value_err("to_prufer_sequence() requires an undirected tree");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n < 2 {
        return value_err("the Prüfer sequence is undefined for trees with fewer than two nodes");
    }
    let edges = graph.get_edges_internal();
    let pairs: Vec<(usize, usize)> = edges.iter().map(|e| (idx[&e.source], idx[&e.target])).collect();
    if pairs.len() != n - 1 || pairs.iter().any(|&(a, b)| a == b) || !spans(n, pairs.iter().copied()) {
        return value_err("the graph is not a tree");
    }
    let mut adj = vec![Vec::new(); n];
    let mut degree = vec![0usize; n];
    for &(a, b) in &pairs {
        adj[a].push(b);
        adj[b].push(a);
        degree[a] += 1;
        degree[b] += 1;
    }
    let mut removed = vec![false; n];
    let mut index = (0..n).find(|&k| degree[k] == 1).unwrap();
    let mut u = index;
    let mut result = Vec::with_capacity(n - 2);
    for _ in 0..n - 2 {
        let v = *adj[u].iter().find(|&&v| !removed[v]).unwrap();
        removed[u] = true;
        result.push(ids[v]);
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            u = v;
        } else {
            index = (index + 1..n).find(|&k| degree[k] == 1 && !removed[k]).unwrap();
            u = index;
        }
    }
    Ok(result)
}

/// The tree on nodes `0..len + 2` encoded by a Prüfer sequence, as edge pairs.
#[pyfunction]
pub fn from_prufer_sequence(sequence: Vec<i64>) -> PyResult<Vec<(usize, usize)>> {
    let n = sequence.len() + 2;
    let mut degree = vec![1usize; n];
    for &v in &sequence {
        if v < 0 || v as usize >= n {
            return value_err(format!(
                "invalid Prüfer sequence: values must be between 0 and {}, got {v}",
                n - 1
            ));
        }
        degree[v as usize] += 1;
    }
    let mut edges = Vec::with_capacity(n - 1);
    let mut index = (0..n).find(|&k| degree[k] == 1).unwrap();
    let mut u = index;
    for &v in &sequence {
        let v = v as usize;
        edges.push((u, v));
        degree[u] = 0;
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            u = v;
        } else {
            index = (index + 1..n).find(|&k| degree[k] == 1).unwrap();
            u = index;
        }
    }
    degree[u] = 0;
    let last: Vec<usize> = (0..n).filter(|&k| degree[k] == 1).collect();
    edges.push((u, last[0]));
    Ok(edges)
}
