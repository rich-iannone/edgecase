//! Exact (sub)graph isomorphism by VF2-style backtracking, and SimRank similarity.
//!
//! The matcher maps the nodes of a pattern graph into a target graph in one of three modes:
//! `"iso"` (bijection preserving adjacency both ways), `"induced"` (injective, pattern edges
//! and non-edges both preserved: an induced subgraph isomorphism) and `"mono"` (injective,
//! pattern edges preserved: a monomorphism). Parallel edges and self-loops are respected: an
//! edge "label" between an ordered pair is the sorted list of its edge colors, one entry per
//! parallel edge, which must be equal (iso, induced) or contained (mono). Node and edge colors
//! are integers computed by the caller from attributes (all zero to ignore attributes).

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

type Label = Vec<i64>;

struct Side {
    ids: Vec<u64>,
    color: Vec<i64>,
    out: Vec<FxHashMap<usize, Label>>,
    inn: Vec<FxHashMap<usize, Label>>,
    deg_out: Vec<usize>,
    deg_in: Vec<usize>,
    /// undirected neighbors in either direction (for ordering and candidate generation)
    any: Vec<Vec<usize>>,
}

impl Side {
    fn new(graph: &Graph, node_colors: &FxHashMap<u64, i64>, edge_colors: &FxHashMap<u64, i64>) -> Self {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let directed = graph.is_directed();
        let mut out: Vec<FxHashMap<usize, Label>> = vec![FxHashMap::default(); n];
        let mut inn: Vec<FxHashMap<usize, Label>> = vec![FxHashMap::default(); n];
        for e in graph.get_edges_internal() {
            let (a, b) = (idx[&e.source], idx[&e.target]);
            let c = edge_colors.get(&e.id).copied().unwrap_or(0);
            out[a].entry(b).or_default().push(c);
            if directed {
                inn[b].entry(a).or_default().push(c);
            } else if a != b {
                out[b].entry(a).or_default().push(c);
            }
        }
        for map in out.iter_mut().chain(inn.iter_mut()) {
            for label in map.values_mut() {
                label.sort_unstable();
            }
        }
        let deg_out = out.iter().map(|m| m.values().map(|l| l.len()).sum()).collect();
        let deg_in = if directed {
            inn.iter().map(|m| m.values().map(|l| l.len()).sum()).collect()
        } else {
            vec![0; n]
        };
        let any = (0..n)
            .map(|u| {
                let mut v: Vec<usize> = out[u].keys().chain(inn[u].keys()).copied().filter(|&w| w != u).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let color = ids.iter().map(|id| node_colors.get(id).copied().unwrap_or(0)).collect();
        Side { ids, color, out, inn, deg_out, deg_in, any }
    }
}

/// Multiset inclusion of sorted lists.
fn contains(big: &[i64], small: &[i64]) -> bool {
    let mut i = 0;
    for &x in small {
        while i < big.len() && big[i] < x {
            i += 1;
        }
        if i == big.len() || big[i] != x {
            return false;
        }
        i += 1;
    }
    true
}

/// 1-dimensional Weisfeiler-Leman color refinement run jointly on both sides, so that equal
/// colors mean equal refined classes across the two graphs.
fn refine(p: &Side, t: &Side, directed: bool) -> (Vec<i64>, Vec<i64>) {
    let mut cp = p.color.clone();
    let mut ct = t.color.clone();
    let classes = |a: &[i64], b: &[i64]| {
        let mut all: Vec<i64> = a.iter().chain(b.iter()).copied().collect();
        all.sort_unstable();
        all.dedup();
        all.len()
    };
    let mut count = classes(&cp, &ct);
    loop {
        let mut table: FxHashMap<(i64, Vec<(u8, i64, Label)>), i64> = FxHashMap::default();
        let mut sig = |side: &Side, col: &[i64]| -> Vec<i64> {
            (0..side.ids.len())
                .map(|u| {
                    let mut parts: Vec<(u8, i64, Label)> =
                        side.out[u].iter().map(|(&w, l)| (0u8, col[w], l.clone())).collect();
                    if directed {
                        parts.extend(side.inn[u].iter().map(|(&w, l)| (1u8, col[w], l.clone())));
                    }
                    parts.sort_unstable();
                    let key = (col[u], parts);
                    let next = table.len() as i64;
                    *table.entry(key).or_insert(next)
                })
                .collect()
        };
        let np = sig(p, &cp);
        let nt = sig(t, &ct);
        let new_count = classes(&np, &nt);
        cp = np;
        ct = nt;
        if new_count == count {
            break;
        }
        count = new_count;
    }
    (cp, ct)
}

struct Matcher<'a> {
    p: &'a Side,
    t: &'a Side,
    mode: u8, // 0 iso, 1 induced, 2 mono
    directed: bool,
    pc: Vec<i64>,
    tc: Vec<i64>,
    order: Vec<usize>,
    map: Vec<usize>,
    inv: Vec<usize>,
    /// number of mapped neighbors (either direction) of every node, per side
    touch_p: Vec<usize>,
    touch_t: Vec<usize>,
    limit: usize,
    results: Vec<Vec<usize>>,
}

const NONE: usize = usize::MAX;

impl<'a> Matcher<'a> {
    fn label_ok(&self, lp: Option<&Label>, lt: Option<&Label>) -> bool {
        match self.mode {
            2 => match lp {
                None => true,
                Some(lp) => lt.map_or(false, |lt| contains(lt, lp)),
            },
            _ => lp == lt,
        }
    }

    fn feasible(&self, u: usize, x: usize) -> bool {
        let (p, t) = (self.p, self.t);
        if self.pc[u] != self.tc[x] {
            return false;
        }
        if self.mode == 0 {
            if p.deg_out[u] != t.deg_out[x] || p.deg_in[u] != t.deg_in[x] {
                return false;
            }
        } else if p.deg_out[u] > t.deg_out[x] || p.deg_in[u] > t.deg_in[x] {
            return false;
        }
        // self-loops
        if !self.label_ok(p.out[u].get(&u), t.out[x].get(&x)) {
            return false;
        }
        for (dir, pm, tm) in [(0, &p.out, &t.out), (1, &p.inn, &t.inn)] {
            if dir == 1 && !self.directed {
                continue;
            }
            let mut mapped_p = 0;
            for (&w, lp) in &pm[u] {
                if w == u || self.map[w] == NONE {
                    continue;
                }
                mapped_p += 1;
                if !self.label_ok(Some(lp), tm[x].get(&self.map[w])) {
                    return false;
                }
            }
            if self.mode != 2 {
                let mapped_t = tm[x].keys().filter(|&&y| y != x && self.inv[y] != NONE).count();
                if mapped_t != mapped_p {
                    return false;
                }
            }
        }
        self.lookahead(u, x)
    }

    /// VF2++-style cutting rule: compare how the unmapped neighbors of `u` and `x` split between
    /// the search frontier (adjacent to the mapping) and the untouched rest, by refined color.
    fn lookahead(&self, u: usize, x: usize) -> bool {
        if self.mode == 2 {
            return true;
        }
        let split = |any: &[usize], col: &[i64], touch: &[usize], mapped: &dyn Fn(usize) -> bool| {
            let mut front = Vec::new();
            let mut rest = Vec::new();
            for &w in any {
                if mapped(w) {
                    continue;
                }
                if touch[w] > 0 { front.push(col[w]) } else { rest.push(col[w]) }
            }
            front.sort_unstable();
            rest.sort_unstable();
            (front, rest)
        };
        let (fp, rp) = split(&self.p.any[u], &self.pc, &self.touch_p, &|w| self.map[w] != NONE);
        let (ft, rt) = split(&self.t.any[x], &self.tc, &self.touch_t, &|y| self.inv[y] != NONE);
        if self.mode == 0 {
            fp == ft && rp == rt
        } else {
            ft.len() >= fp.len() && ft.len() + rt.len() >= fp.len() + rp.len()
        }
    }

    fn assign(&mut self, u: usize, x: usize, on: bool) {
        if on {
            self.map[u] = x;
            self.inv[x] = u;
        } else {
            self.map[u] = NONE;
            self.inv[x] = NONE;
        }
        for &w in &self.p.any[u] {
            if on { self.touch_p[w] += 1 } else { self.touch_p[w] -= 1 }
        }
        for &y in &self.t.any[x] {
            if on { self.touch_t[y] += 1 } else { self.touch_t[y] -= 1 }
        }
    }

    fn search(&mut self, depth: usize) -> bool {
        if depth == self.order.len() {
            self.results.push(self.map.clone());
            return self.results.len() >= self.limit;
        }
        let u = self.order[depth];
        // candidates: neighbors of the image of an already mapped neighbor, else everything
        let parent = self.p.any[u].iter().copied().find(|&w| self.map[w] != NONE);
        let candidates: Vec<usize> = match parent {
            Some(w) => {
                let img = self.map[w];
                let mut c: Vec<usize> = Vec::new();
                if self.p.out[w].contains_key(&u) {
                    c.extend(self.t.out[img].keys().copied());
                } else if self.p.inn[w].contains_key(&u) {
                    c.extend(self.t.inn[img].keys().copied());
                } else {
                    c.extend(self.t.any[img].iter().copied());
                }
                c.sort_unstable();
                c
            }
            None => (0..self.t.ids.len()).collect(),
        };
        for x in candidates {
            if self.inv[x] != NONE || !self.feasible(u, x) {
                continue;
            }
            self.assign(u, x, true);
            let stop = self.search(depth + 1);
            self.assign(u, x, false);
            if stop {
                return true;
            }
        }
        false
    }
}

/// Mappings `(pattern node ID, target node ID)` from `pattern` into `target`.
#[pyfunction]
#[pyo3(signature = (pattern, target, mode, node_colors_p, node_colors_t, edge_colors_p, edge_colors_t, limit=None))]
#[allow(clippy::too_many_arguments)]
pub fn match_graphs(
    pattern: &Graph,
    target: &Graph,
    mode: &str,
    node_colors_p: FxHashMap<u64, i64>,
    node_colors_t: FxHashMap<u64, i64>,
    edge_colors_p: FxHashMap<u64, i64>,
    edge_colors_t: FxHashMap<u64, i64>,
    limit: Option<usize>,
) -> PyResult<Vec<Vec<(u64, u64)>>> {
    let mode = match mode {
        "iso" => 0u8,
        "induced" => 1,
        "mono" => 2,
        _ => return value_err(format!("unknown mode {mode:?}")),
    };
    if pattern.is_directed() != target.is_directed() {
        return value_err("both graphs must be directed, or both undirected");
    }
    let directed = pattern.is_directed();
    let p = Side::new(pattern, &node_colors_p, &edge_colors_p);
    let t = Side::new(target, &node_colors_t, &edge_colors_t);
    let (np, nt) = (p.ids.len(), t.ids.len());
    let limit = limit.unwrap_or(usize::MAX).max(1);
    if np > nt || (mode == 0 && (np != nt || pattern.num_edges() != target.num_edges())) {
        return Ok(Vec::new());
    }
    let (pc, tc) = if mode == 0 {
        let (pc, tc) = refine(&p, &t, directed);
        let mut a = pc.clone();
        let mut b = tc.clone();
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            return Ok(Vec::new());
        }
        (pc, tc)
    } else {
        (p.color.clone(), t.color.clone())
    };
    // VF2++ matching order: BFS layers from a root of rarest color and highest degree; within
    // a layer, most already-ordered neighbors first, then highest degree, then rarest color
    let mut rarity: FxHashMap<i64, i64> = FxHashMap::default();
    for &c in &tc {
        *rarity.entry(c).or_default() += 1;
    }
    let degree = |u: usize| p.deg_out[u] + p.deg_in[u];
    let mut order = Vec::with_capacity(np);
    let mut placed = vec![false; np];
    let mut used = vec![0usize; np];
    while order.len() < np {
        let root = (0..np)
            .filter(|&u| !placed[u])
            .min_by_key(|&u| (rarity.get(&pc[u]).copied().unwrap_or(0), std::cmp::Reverse(degree(u)), u))
            .unwrap();
        let mut seen = vec![false; np];
        seen[root] = true;
        let mut layer = vec![root];
        while !layer.is_empty() {
            let mut todo = layer.clone();
            while !todo.is_empty() {
                let k = (0..todo.len())
                    .max_by_key(|&k| {
                        let u = todo[k];
                        (
                            used[u],
                            degree(u),
                            std::cmp::Reverse(rarity.get(&pc[u]).copied().unwrap_or(0)),
                            std::cmp::Reverse(u),
                        )
                    })
                    .unwrap();
                let u = todo.swap_remove(k);
                order.push(u);
                placed[u] = true;
                *rarity.entry(pc[u]).or_default() -= 1;
                for &w in &p.any[u] {
                    used[w] += 1;
                }
            }
            let mut next = Vec::new();
            for &u in &layer {
                for &w in &p.any[u] {
                    if !seen[w] {
                        seen[w] = true;
                        next.push(w);
                    }
                }
            }
            layer = next;
        }
    }
    let mut m = Matcher {
        p: &p,
        t: &t,
        mode,
        directed,
        pc,
        tc,
        order,
        map: vec![NONE; np],
        inv: vec![NONE; nt],
        touch_p: vec![0; np],
        touch_t: vec![0; nt],
        limit,
        results: Vec::new(),
    };
    m.search(0);
    Ok(m.results
        .into_iter()
        .map(|map| (0..np).map(|u| (p.ids[u], t.ids[map[u]])).collect())
        .collect())
}

/// SimRank similarity (Jeh and Widom), iterated as in NetworkX:
/// `S = c * Wᵀ S W` with column-normalized adjacency `W`, diagonal reset to 1, until
/// `|S_new - S_old| <= tolerance + 1e-5 * |S_new|` everywhere. Returns `(node IDs, matrix)`.
#[pyfunction]
#[pyo3(signature = (graph, importance_factor=0.9, max_iterations=1000, tolerance=1e-4, weighted=false))]
pub fn simrank(
    graph: &Graph,
    importance_factor: f64,
    max_iterations: usize,
    tolerance: f64,
    weighted: bool,
) -> PyResult<(Vec<u64>, Vec<Vec<f64>>)> {
    use nalgebra::DMatrix;
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let n = ids.len();
    let mut a = DMatrix::<f64>::zeros(n, n);
    for e in graph.get_edges_internal() {
        let (u, v) = (idx[&e.source], idx[&e.target]);
        let w = if weighted { e.weight.unwrap_or(1.0) } else { 1.0 };
        a[(u, v)] += w;
        if !graph.is_directed() && u != v {
            a[(v, u)] += w;
        }
    }
    for j in 0..n {
        let s: f64 = a.column(j).sum();
        if s != 0.0 {
            a.column_mut(j).scale_mut(1.0 / s);
        }
    }
    let at = a.transpose();
    let mut sim = DMatrix::<f64>::identity(n, n);
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut next = (&at * &sim) * &a * importance_factor;
        for i in 0..n {
            next[(i, i)] = 1.0;
        }
        let close = sim.iter().zip(next.iter()).all(|(&old, &new)| (old - new).abs() <= tolerance + 1e-5 * new.abs());
        sim = next;
        if close {
            converged = true;
            break;
        }
    }
    if !converged && n > 0 {
        return value_err(format!("SimRank did not converge after {max_iterations} iterations"));
    }
    let rows = (0..n).map(|i| (0..n).map(|j| sim[(i, j)]).collect()).collect();
    Ok((ids, rows))
}
