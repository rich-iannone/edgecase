//! Exact matchings and covers: Edmonds' weighted blossom algorithm (maximum-weight and
//! minimum-weight maximum-cardinality matchings), Hopcroft-Karp bipartite matching, the
//! assignment problem (minimum-weight full bipartite matching), König vertex covers, minimum
//! edge covers and the local-ratio 2-approximate weighted vertex cover.
//!
//! Matchings ignore edge direction and self-loops. Among parallel edges, only the best one
//! (heaviest for maximum, lightest for minimum) can be chosen.

use crate::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

/// One edge per unordered node pair (no self-loops): `(i, j, weight, edge id)` with `i < j`.
/// `prefer_max` keeps the heaviest parallel edge, otherwise the lightest; ties keep the
/// lowest edge ID.
fn simple_edges(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    weighted: bool,
    prefer_max: bool,
) -> PyResult<Vec<(usize, usize, f64, u64)>> {
    let mut best: FxHashMap<(usize, usize), (f64, u64)> = FxHashMap::default();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        if !w.is_finite() {
            return value_err("edge weights must be finite numbers");
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        let key = (a.min(b), a.max(b));
        match best.get(&key) {
            Some(&(bw, bid)) => {
                let better = if prefer_max { w > bw } else { w < bw };
                if better || (w == bw && e.id < bid) {
                    best.insert(key, (w, e.id));
                }
            }
            None => {
                best.insert(key, (w, e.id));
            }
        }
    }
    let mut out: Vec<_> = best
        .into_iter()
        .map(|((a, b), (w, id))| (a, b, w, id))
        .collect();
    out.sort_unstable_by_key(|x| x.3);
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Edmonds' weighted blossom algorithm (after Joris van Rantwijk's mwmatching, as in NetworkX)
// ---------------------------------------------------------------------------------------------

struct Blossom<'a> {
    n: usize,
    edges: &'a [(usize, usize, f64)],
    endpoint: Vec<usize>,
    neighbend: Vec<Vec<usize>>,
    mate: Vec<i64>,
    label: Vec<u8>,
    labelend: Vec<i64>,
    inblossom: Vec<usize>,
    blossomparent: Vec<i64>,
    blossomchilds: Vec<Vec<usize>>,
    blossombase: Vec<i64>,
    blossomendps: Vec<Vec<usize>>,
    bestedge: Vec<i64>,
    blossombestedges: Vec<Option<Vec<usize>>>,
    unusedblossoms: Vec<usize>,
    dualvar: Vec<f64>,
    allowedge: Vec<bool>,
    queue: Vec<usize>,
}

fn wrap(j: i64, len: usize) -> usize {
    j.rem_euclid(len as i64) as usize
}

impl<'a> Blossom<'a> {
    fn new(n: usize, edges: &'a [(usize, usize, f64)]) -> Self {
        let maxweight = edges.iter().fold(0.0f64, |m, e| m.max(e.2));
        let mut endpoint = Vec::with_capacity(2 * edges.len());
        let mut neighbend = vec![Vec::new(); n];
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            endpoint.push(i);
            endpoint.push(j);
            neighbend[i].push(2 * k + 1);
            neighbend[j].push(2 * k);
        }
        let mut dualvar = vec![maxweight; n];
        dualvar.extend(std::iter::repeat(0.0).take(n));
        Blossom {
            n,
            edges,
            endpoint,
            neighbend,
            mate: vec![-1; n],
            label: vec![0; 2 * n],
            labelend: vec![-1; 2 * n],
            inblossom: (0..n).collect(),
            blossomparent: vec![-1; 2 * n],
            blossomchilds: vec![Vec::new(); 2 * n],
            blossombase: (0..n as i64).chain(std::iter::repeat(-1).take(n)).collect(),
            blossomendps: vec![Vec::new(); 2 * n],
            bestedge: vec![-1; 2 * n],
            blossombestedges: vec![None; 2 * n],
            unusedblossoms: (n..2 * n).collect(),
            dualvar,
            allowedge: vec![false; edges.len()],
            queue: Vec::new(),
        }
    }

    fn slack(&self, k: usize) -> f64 {
        let (i, j, w) = self.edges[k];
        self.dualvar[i] + self.dualvar[j] - 2.0 * w
    }

    fn leaves(&self, b: usize, out: &mut Vec<usize>) {
        if b < self.n {
            out.push(b);
        } else {
            for &t in &self.blossomchilds[b] {
                self.leaves(t, out);
            }
        }
    }

    fn leaves_of(&self, b: usize) -> Vec<usize> {
        let mut out = Vec::new();
        self.leaves(b, &mut out);
        out
    }

    fn assign_label(&mut self, w: usize, t: u8, p: i64) {
        let b = self.inblossom[w];
        self.label[w] = t;
        self.label[b] = t;
        self.labelend[w] = p;
        self.labelend[b] = p;
        self.bestedge[w] = -1;
        self.bestedge[b] = -1;
        if t == 1 {
            let leaves = self.leaves_of(b);
            self.queue.extend(leaves);
        } else if t == 2 {
            let base = self.blossombase[b] as usize;
            let m = self.mate[base] as usize;
            self.assign_label(self.endpoint[m], 1, (m ^ 1) as i64);
        }
    }

    fn scan_blossom(&mut self, mut v: i64, mut w: i64) -> i64 {
        let mut path = Vec::new();
        let mut base = -1;
        while v != -1 || w != -1 {
            let mut b = self.inblossom[v as usize];
            if self.label[b] & 4 != 0 {
                base = self.blossombase[b];
                break;
            }
            path.push(b);
            self.label[b] = 5;
            if self.labelend[b] == -1 {
                v = -1;
            } else {
                v = self.endpoint[self.labelend[b] as usize] as i64;
                b = self.inblossom[v as usize];
                v = self.endpoint[self.labelend[b] as usize] as i64;
            }
            if w != -1 {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label[b] = 1;
        }
        base
    }

    fn add_blossom(&mut self, base: usize, k: usize) {
        let (mut v, mut w, _) = self.edges[k];
        let bb = self.inblossom[base];
        let mut bv = self.inblossom[v];
        let mut bw = self.inblossom[w];
        let b = self.unusedblossoms.pop().expect("blossom pool exhausted");
        self.blossombase[b] = base as i64;
        self.blossomparent[b] = -1;
        self.blossomparent[bb] = b as i64;
        let mut path = Vec::new();
        let mut endps = Vec::new();
        while bv != bb {
            self.blossomparent[bv] = b as i64;
            path.push(bv);
            endps.push(self.labelend[bv] as usize);
            v = self.endpoint[self.labelend[bv] as usize];
            bv = self.inblossom[v];
        }
        path.push(bb);
        path.reverse();
        endps.reverse();
        endps.push(2 * k);
        while bw != bb {
            self.blossomparent[bw] = b as i64;
            path.push(bw);
            endps.push((self.labelend[bw] ^ 1) as usize);
            w = self.endpoint[self.labelend[bw] as usize];
            bw = self.inblossom[w];
        }
        self.blossomchilds[b] = path.clone();
        self.blossomendps[b] = endps;
        self.label[b] = 1;
        self.labelend[b] = self.labelend[bb];
        self.dualvar[b] = 0.0;
        for leaf in self.leaves_of(b) {
            if self.label[self.inblossom[leaf]] == 2 {
                self.queue.push(leaf);
            }
            self.inblossom[leaf] = b;
        }
        let mut bestedgeto = vec![-1i64; 2 * self.n];
        for &sub in &path {
            let nblists: Vec<Vec<usize>> = match self.blossombestedges[sub].take() {
                Some(list) => vec![list],
                None => self
                    .leaves_of(sub)
                    .into_iter()
                    .map(|leaf| self.neighbend[leaf].iter().map(|p| p / 2).collect())
                    .collect(),
            };
            for nblist in nblists {
                for k in nblist {
                    let (mut i, mut j, _) = self.edges[k];
                    if self.inblossom[j] == b {
                        std::mem::swap(&mut i, &mut j);
                    }
                    let _ = i;
                    let bj = self.inblossom[j];
                    if bj != b
                        && self.label[bj] == 1
                        && (bestedgeto[bj] == -1
                            || self.slack(k) < self.slack(bestedgeto[bj] as usize))
                    {
                        bestedgeto[bj] = k as i64;
                    }
                }
            }
            self.blossombestedges[sub] = None;
            self.bestedge[sub] = -1;
        }
        let list: Vec<usize> = bestedgeto
            .into_iter()
            .filter(|&k| k != -1)
            .map(|k| k as usize)
            .collect();
        self.bestedge[b] = -1;
        for &k in &list {
            if self.bestedge[b] == -1 || self.slack(k) < self.slack(self.bestedge[b] as usize) {
                self.bestedge[b] = k as i64;
            }
        }
        self.blossombestedges[b] = Some(list);
    }

    fn expand_blossom(&mut self, b: usize, endstage: bool) {
        let childs = self.blossomchilds[b].clone();
        for &s in &childs {
            self.blossomparent[s] = -1;
            if s < self.n {
                self.inblossom[s] = s;
            } else if endstage && self.dualvar[s] == 0.0 {
                self.expand_blossom(s, endstage);
            } else {
                for leaf in self.leaves_of(s) {
                    self.inblossom[leaf] = s;
                }
            }
        }
        if !endstage && self.label[b] == 2 {
            let len = childs.len();
            let endps = self.blossomendps[b].clone();
            let entrychild = self.inblossom[self.endpoint[(self.labelend[b] ^ 1) as usize]];
            let mut j = childs.iter().position(|&c| c == entrychild).unwrap() as i64;
            let (jstep, endptrick): (i64, usize) = if j & 1 == 1 {
                j -= len as i64;
                (1, 0)
            } else {
                (-1, 1)
            };
            let mut p = self.labelend[b] as usize;
            while j != 0 {
                self.label[self.endpoint[p ^ 1]] = 0;
                let q = endps[wrap(j - endptrick as i64, len)];
                self.label[self.endpoint[q ^ endptrick ^ 1]] = 0;
                self.assign_label(self.endpoint[p ^ 1], 2, p as i64);
                self.allowedge[q / 2] = true;
                j += jstep;
                p = endps[wrap(j - endptrick as i64, len)] ^ endptrick;
                self.allowedge[p / 2] = true;
                j += jstep;
            }
            let bv = childs[wrap(j, len)];
            let w = self.endpoint[p ^ 1];
            self.label[w] = 2;
            self.label[bv] = 2;
            self.labelend[w] = p as i64;
            self.labelend[bv] = p as i64;
            self.bestedge[bv] = -1;
            j += jstep;
            while childs[wrap(j, len)] != entrychild {
                let bv = childs[wrap(j, len)];
                if self.label[bv] == 1 {
                    j += jstep;
                    continue;
                }
                let leaves = self.leaves_of(bv);
                let mut v = *leaves.last().unwrap();
                for &leaf in &leaves {
                    v = leaf;
                    if self.label[leaf] != 0 {
                        break;
                    }
                }
                if self.label[v] != 0 {
                    self.label[v] = 0;
                    let m = self.mate[self.blossombase[bv] as usize] as usize;
                    self.label[self.endpoint[m]] = 0;
                    let le = self.labelend[v];
                    self.assign_label(v, 2, le);
                }
                j += jstep;
            }
        }
        self.label[b] = 0;
        self.labelend[b] = -1;
        self.blossomchilds[b].clear();
        self.blossomendps[b].clear();
        self.blossombase[b] = -1;
        self.blossombestedges[b] = None;
        self.bestedge[b] = -1;
        self.unusedblossoms.push(b);
    }

    fn augment_blossom(&mut self, b: usize, v: usize) {
        let mut t = v;
        while self.blossomparent[t] != b as i64 {
            t = self.blossomparent[t] as usize;
        }
        if t >= self.n {
            self.augment_blossom(t, v);
        }
        let len = self.blossomchilds[b].len();
        let i = self.blossomchilds[b].iter().position(|&c| c == t).unwrap();
        let mut j = i as i64;
        let (jstep, endptrick): (i64, usize) = if i & 1 == 1 {
            j -= len as i64;
            (1, 0)
        } else {
            (-1, 1)
        };
        while j != 0 {
            j += jstep;
            let t = self.blossomchilds[b][wrap(j, len)];
            let p = self.blossomendps[b][wrap(j - endptrick as i64, len)] ^ endptrick;
            if t >= self.n {
                self.augment_blossom(t, self.endpoint[p]);
            }
            j += jstep;
            let t = self.blossomchilds[b][wrap(j, len)];
            if t >= self.n {
                self.augment_blossom(t, self.endpoint[p ^ 1]);
            }
            self.mate[self.endpoint[p]] = (p ^ 1) as i64;
            self.mate[self.endpoint[p ^ 1]] = p as i64;
        }
        self.blossomchilds[b].rotate_left(i);
        self.blossomendps[b].rotate_left(i);
        self.blossombase[b] = self.blossombase[self.blossomchilds[b][0]];
    }

    fn augment_matching(&mut self, k: usize) {
        let (v, w, _) = self.edges[k];
        for (mut s, mut p) in [(v, 2 * k + 1), (w, 2 * k)] {
            loop {
                let bs = self.inblossom[s];
                if bs >= self.n {
                    self.augment_blossom(bs, s);
                }
                self.mate[s] = p as i64;
                if self.labelend[bs] == -1 {
                    break;
                }
                let t = self.endpoint[self.labelend[bs] as usize];
                let bt = self.inblossom[t];
                s = self.endpoint[self.labelend[bt] as usize];
                let j = self.endpoint[(self.labelend[bt] ^ 1) as usize];
                if bt >= self.n {
                    self.augment_blossom(bt, j);
                }
                self.mate[j] = self.labelend[bt];
                p = (self.labelend[bt] ^ 1) as usize;
            }
        }
    }

    fn solve(&mut self, maxcardinality: bool) {
        let n = self.n;
        for _ in 0..n {
            self.label.iter_mut().for_each(|x| *x = 0);
            self.bestedge.iter_mut().for_each(|x| *x = -1);
            for b in n..2 * n {
                self.blossombestedges[b] = None;
            }
            self.allowedge.iter_mut().for_each(|x| *x = false);
            self.queue.clear();
            for v in 0..n {
                if self.mate[v] == -1 && self.label[self.inblossom[v]] == 0 {
                    self.assign_label(v, 1, -1);
                }
            }
            let mut augmented = false;
            loop {
                while !augmented {
                    let Some(v) = self.queue.pop() else { break };
                    for idx in 0..self.neighbend[v].len() {
                        let p = self.neighbend[v][idx];
                        let k = p / 2;
                        let w = self.endpoint[p];
                        if self.inblossom[v] == self.inblossom[w] {
                            continue;
                        }
                        let mut kslack = 0.0;
                        if !self.allowedge[k] {
                            kslack = self.slack(k);
                            if kslack <= 0.0 {
                                self.allowedge[k] = true;
                            }
                        }
                        if self.allowedge[k] {
                            if self.label[self.inblossom[w]] == 0 {
                                self.assign_label(w, 2, (p ^ 1) as i64);
                            } else if self.label[self.inblossom[w]] == 1 {
                                let base = self.scan_blossom(v as i64, w as i64);
                                if base >= 0 {
                                    self.add_blossom(base as usize, k);
                                } else {
                                    self.augment_matching(k);
                                    augmented = true;
                                    break;
                                }
                            } else if self.label[w] == 0 {
                                self.label[w] = 2;
                                self.labelend[w] = (p ^ 1) as i64;
                            }
                        } else if self.label[self.inblossom[w]] == 1 {
                            let b = self.inblossom[v];
                            if self.bestedge[b] == -1
                                || kslack < self.slack(self.bestedge[b] as usize)
                            {
                                self.bestedge[b] = k as i64;
                            }
                        } else if self.label[w] == 0
                            && (self.bestedge[w] == -1
                                || kslack < self.slack(self.bestedge[w] as usize))
                        {
                            self.bestedge[w] = k as i64;
                        }
                    }
                }
                if augmented {
                    break;
                }
                // Dual adjustment
                let mut deltatype = -1;
                let mut delta = 0.0f64;
                let mut deltaedge = 0usize;
                let mut deltablossom = 0usize;
                let min_dual = self.dualvar[..n]
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                if !maxcardinality {
                    deltatype = 1;
                    delta = min_dual;
                }
                for v in 0..n {
                    if self.label[self.inblossom[v]] == 0 && self.bestedge[v] != -1 {
                        let d = self.slack(self.bestedge[v] as usize);
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 2;
                            deltaedge = self.bestedge[v] as usize;
                        }
                    }
                }
                for b in 0..2 * n {
                    if self.blossomparent[b] == -1 && self.label[b] == 1 && self.bestedge[b] != -1 {
                        let d = self.slack(self.bestedge[b] as usize) / 2.0;
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 3;
                            deltaedge = self.bestedge[b] as usize;
                        }
                    }
                }
                for b in n..2 * n {
                    if self.blossombase[b] >= 0
                        && self.blossomparent[b] == -1
                        && self.label[b] == 2
                        && (deltatype == -1 || self.dualvar[b] < delta)
                    {
                        delta = self.dualvar[b];
                        deltatype = 4;
                        deltablossom = b;
                    }
                }
                if deltatype == -1 {
                    deltatype = 1;
                    delta = min_dual.max(0.0);
                }
                for v in 0..n {
                    match self.label[self.inblossom[v]] {
                        1 => self.dualvar[v] -= delta,
                        2 => self.dualvar[v] += delta,
                        _ => {}
                    }
                }
                for b in n..2 * n {
                    if self.blossombase[b] >= 0 && self.blossomparent[b] == -1 {
                        match self.label[b] {
                            1 => self.dualvar[b] += delta,
                            2 => self.dualvar[b] -= delta,
                            _ => {}
                        }
                    }
                }
                match deltatype {
                    1 => break,
                    2 => {
                        self.allowedge[deltaedge] = true;
                        let (mut i, j, _) = self.edges[deltaedge];
                        if self.label[self.inblossom[i]] == 0 {
                            i = j;
                        }
                        self.queue.push(i);
                    }
                    3 => {
                        self.allowedge[deltaedge] = true;
                        self.queue.push(self.edges[deltaedge].0);
                    }
                    _ => self.expand_blossom(deltablossom, false),
                }
            }
            if !augmented {
                break;
            }
            for b in n..2 * n {
                if self.blossomparent[b] == -1
                    && self.blossombase[b] >= 0
                    && self.label[b] == 1
                    && self.dualvar[b] == 0.0
                {
                    self.expand_blossom(b, true);
                }
            }
        }
    }
}

/// Indices into `edges` of a maximum-weight matching (maximum cardinality first when
/// `maxcardinality`).
pub(crate) fn blossom_matching(
    n: usize,
    edges: &[(usize, usize, f64)],
    maxcardinality: bool,
) -> Vec<usize> {
    if edges.is_empty() {
        return Vec::new();
    }
    let mut solver = Blossom::new(n, edges);
    solver.solve(maxcardinality);
    (0..n)
        .filter(|&v| solver.mate[v] >= 0)
        .map(|v| solver.mate[v] as usize / 2)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Maximum-weight matching (Edmonds' blossom algorithm), or with `minimize` the
/// minimum-weight matching among the maximum-cardinality ones. Returns edge IDs, ascending.
#[pyfunction]
#[pyo3(signature = (graph, maxcardinality=false, weighted=true, minimize=false))]
pub fn blossom_max_weight_matching(
    graph: &Graph,
    maxcardinality: bool,
    weighted: bool,
    minimize: bool,
) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let simple = simple_edges(graph, &idx, weighted, !minimize)?;
    let edges: Vec<(usize, usize, f64)> = if minimize {
        let top = simple.iter().fold(f64::NEG_INFINITY, |m, e| m.max(e.2)) + 1.0;
        simple.iter().map(|&(a, b, w, _)| (a, b, top - w)).collect()
    } else {
        simple.iter().map(|&(a, b, w, _)| (a, b, w)).collect()
    };
    let chosen = blossom_matching(ids.len(), &edges, maxcardinality || minimize);
    let mut out: Vec<u64> = chosen.into_iter().map(|k| simple[k].3).collect();
    out.sort_unstable();
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Bipartite matchings and covers
// ---------------------------------------------------------------------------------------------

/// Side of each node (0 = top, 1 = bottom), from `top` or by 2-coloring (direction ignored).
pub(crate) fn bipartition(
    n: usize,
    adj: &[Vec<(usize, usize)>],
    idx: &FxHashMap<u64, usize>,
    top: Option<Vec<u64>>,
) -> PyResult<Vec<u8>> {
    let mut side = vec![u8::MAX; n];
    match top {
        Some(nodes) => {
            side.iter_mut().for_each(|s| *s = 1);
            for id in nodes {
                match idx.get(&id) {
                    Some(&i) => side[i] = 0,
                    None => return value_err(format!("Node {id} does not exist")),
                }
            }
            for (u, list) in adj.iter().enumerate() {
                if list.iter().any(|&(v, _)| side[v] == side[u]) {
                    return value_err("top_nodes= does not split the graph into two sides: some edge joins two nodes on the same side");
                }
            }
        }
        None => {
            for s in 0..n {
                if side[s] != u8::MAX {
                    continue;
                }
                side[s] = 0;
                let mut queue = VecDeque::from([s]);
                while let Some(u) = queue.pop_front() {
                    for &(v, _) in &adj[u] {
                        if side[v] == u8::MAX {
                            side[v] = 1 - side[u];
                            queue.push_back(v);
                        } else if side[v] == side[u] {
                            return value_err("the graph is not bipartite");
                        }
                    }
                }
            }
        }
    }
    Ok(side)
}

/// Undirected adjacency without self-loops: `adj[u] = [(v, simple edge index)]`.
fn adjacency(n: usize, simple: &[(usize, usize, f64, u64)]) -> Vec<Vec<(usize, usize)>> {
    let mut adj = vec![Vec::new(); n];
    for (k, &(a, b, _, _)) in simple.iter().enumerate() {
        adj[a].push((b, k));
        adj[b].push((a, k));
    }
    adj
}

/// Hopcroft-Karp maximum matching: `mate[u]` = (partner, simple edge index).
pub(crate) fn hopcroft_karp(
    n: usize,
    adj: &[Vec<(usize, usize)>],
    side: &[u8],
) -> Vec<Option<(usize, usize)>> {
    let mut mate: Vec<Option<(usize, usize)>> = vec![None; n];
    let left: Vec<usize> = (0..n).filter(|&u| side[u] == 0).collect();
    let mut dist = vec![usize::MAX; n];
    loop {
        // BFS layering from free left nodes
        let mut queue = VecDeque::new();
        for &u in &left {
            if mate[u].is_none() {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = usize::MAX;
            }
        }
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for &(v, _) in &adj[u] {
                match mate[v] {
                    None => found = true,
                    Some((w, _)) => {
                        if dist[w] == usize::MAX {
                            dist[w] = dist[u] + 1;
                            queue.push_back(w);
                        }
                    }
                }
            }
        }
        if !found {
            break;
        }
        // DFS augmentations along the layering (iterative)
        let mut it = vec![0usize; n];
        for &root in &left {
            if mate[root].is_some() {
                continue;
            }
            let mut stack: Vec<usize> = vec![root];
            let mut path: Vec<(usize, usize, usize)> = Vec::new(); // (u, v, edge)
            while let Some(&u) = stack.last() {
                if it[u] == adj[u].len() {
                    dist[u] = usize::MAX;
                    stack.pop();
                    path.pop();
                    continue;
                }
                let (v, k) = adj[u][it[u]];
                it[u] += 1;
                match mate[v] {
                    None => {
                        path.push((u, v, k));
                        for &(a, b, e) in &path {
                            mate[a] = Some((b, e));
                            mate[b] = Some((a, e));
                        }
                        break;
                    }
                    Some((w, _)) => {
                        if dist[w] == dist[u] + 1 {
                            path.push((u, v, k));
                            stack.push(w);
                        }
                    }
                }
            }
        }
    }
    mate
}

/// Maximum-cardinality bipartite matching (Hopcroft-Karp). Returns edge IDs, ascending.
#[pyfunction]
#[pyo3(signature = (graph, top_nodes=None))]
pub fn hopcroft_karp_matching(graph: &Graph, top_nodes: Option<Vec<u64>>) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let simple = simple_edges(graph, &idx, false, true)?;
    let adj = adjacency(ids.len(), &simple);
    let side = bipartition(ids.len(), &adj, &idx, top_nodes)?;
    let mate = hopcroft_karp(ids.len(), &adj, &side);
    let mut out: Vec<u64> = (0..ids.len())
        .filter(|&u| side[u] == 0)
        .filter_map(|u| mate[u].map(|(_, k)| simple[k].3))
        .collect();
    out.sort_unstable();
    Ok(out)
}

/// Minimum vertex cover of a bipartite graph by König's theorem: `(cover node IDs, matching
/// edge IDs)`.
#[pyfunction]
#[pyo3(signature = (graph, top_nodes=None))]
pub fn bipartite_vertex_cover(
    graph: &Graph,
    top_nodes: Option<Vec<u64>>,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let simple = simple_edges(graph, &idx, false, true)?;
    let adj = adjacency(n, &simple);
    let side = bipartition(n, &adj, &idx, top_nodes)?;
    let mate = hopcroft_karp(n, &adj, &side);
    // Z: reachable from free top nodes along alternating paths
    let mut seen = vec![false; n];
    let mut queue: VecDeque<usize> = (0..n)
        .filter(|&u| side[u] == 0 && mate[u].is_none())
        .collect();
    for &u in &queue {
        seen[u] = true;
    }
    while let Some(u) = queue.pop_front() {
        for &(v, k) in &adj[u] {
            let via_matching = mate[u].map(|(_, e)| e) == Some(k);
            let allowed = if side[u] == 0 {
                !via_matching
            } else {
                via_matching
            };
            if allowed && !seen[v] {
                seen[v] = true;
                queue.push_back(v);
            }
        }
    }
    let cover: Vec<u64> = (0..n)
        .filter(|&u| (side[u] == 0 && !seen[u]) || (side[u] == 1 && seen[u]))
        .map(|u| ids[u])
        .collect();
    let mut matching: Vec<u64> = (0..n)
        .filter(|&u| side[u] == 0)
        .filter_map(|u| mate[u].map(|(_, k)| simple[k].3))
        .collect();
    matching.sort_unstable();
    Ok((cover, matching))
}

/// Minimum-weight full matching of a bipartite graph (the assignment problem): every node on
/// the smaller side is matched. Hungarian algorithm with potentials, O(n^2 m).
#[pyfunction]
#[pyo3(signature = (graph, top_nodes=None, weighted=true))]
pub fn min_weight_full_matching(
    graph: &Graph,
    top_nodes: Option<Vec<u64>>,
    weighted: bool,
) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let simple = simple_edges(graph, &idx, weighted, false)?;
    let adj = adjacency(n, &simple);
    let side = bipartition(n, &adj, &idx, top_nodes)?;
    let mut rows: Vec<usize> = (0..n).filter(|&u| side[u] == 0).collect();
    let mut cols: Vec<usize> = (0..n).filter(|&u| side[u] == 1).collect();
    if rows.len() > cols.len() {
        std::mem::swap(&mut rows, &mut cols);
    }
    let (r, c) = (rows.len(), cols.len());
    if r == 0 {
        return Ok(Vec::new());
    }
    let col_of: FxHashMap<usize, usize> = cols.iter().enumerate().map(|(j, &u)| (u, j)).collect();
    let mut cost: Vec<Vec<Option<(f64, usize)>>> = vec![vec![None; c]; r];
    for (i, &u) in rows.iter().enumerate() {
        for &(v, k) in &adj[u] {
            if let Some(&j) = col_of.get(&v) {
                cost[i][j] = Some((simple[k].2, k));
            }
        }
    }
    // Hungarian algorithm (shortest augmenting paths with potentials), 1-based as usual
    let inf = f64::INFINITY;
    let mut u_pot = vec![0.0f64; r + 1];
    let mut v_pot = vec![0.0f64; c + 1];
    let mut p = vec![0usize; c + 1];
    let mut way = vec![0usize; c + 1];
    for i in 1..=r {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; c + 1];
        let mut used = vec![false; c + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = usize::MAX;
            for j in 1..=c {
                if used[j] {
                    continue;
                }
                if let Some((w, _)) = cost[i0 - 1][j - 1] {
                    let cur = w - u_pot[i0] - v_pot[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if j1 == usize::MAX {
                return value_err(
                    "no full matching exists: some node on the smaller side cannot be matched",
                );
            }
            for j in 0..=c {
                if used[j] {
                    u_pot[p[j]] += delta;
                    v_pot[j] -= delta;
                } else if minv[j] < inf {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out: Vec<u64> = (1..=c)
        .filter(|&j| p[j] != 0)
        .map(|j| simple[cost[p[j] - 1][j - 1].unwrap().1].3)
        .collect();
    out.sort_unstable();
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Covers
// ---------------------------------------------------------------------------------------------

/// Minimum edge cover: a maximum matching plus one edge for each unmatched node. Edge IDs.
#[pyfunction]
pub fn min_edge_cover(graph: &Graph) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let simple = simple_edges(graph, &idx, false, true)?;
    let edges: Vec<(usize, usize, f64)> = simple.iter().map(|&(a, b, _, _)| (a, b, 1.0)).collect();
    let mut covered = vec![false; n];
    let mut out: Vec<u64> = Vec::new();
    for k in blossom_matching(n, &edges, true) {
        covered[simple[k].0] = true;
        covered[simple[k].1] = true;
        out.push(simple[k].3);
    }
    // first incident edge (by ID) for every node still uncovered; self-loops count
    let mut first: Vec<Option<u64>> = vec![None; n];
    let mut sorted_edges: Vec<_> = graph.get_edges_internal().iter().collect();
    sorted_edges.sort_unstable_by_key(|e| e.id);
    for e in sorted_edges {
        for end in [e.source, e.target] {
            let i = idx[&end];
            if first[i].is_none() {
                first[i] = Some(e.id);
            }
        }
    }
    for v in 0..n {
        if !covered[v] {
            match first[v] {
                Some(e) => out.push(e),
                None => return value_err("no edge cover exists: the graph has an isolated node"),
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Local-ratio 2-approximation of the minimum-weight vertex cover (Bar-Yehuda and Even).
/// Node costs come from the node attribute `weight_attr` (default 1; non-numeric counts as 1).
#[pyfunction]
#[pyo3(signature = (graph, weight_attr=None))]
pub fn min_weighted_vertex_cover(
    py: Python,
    graph: &Graph,
    weight_attr: Option<&str>,
) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let nodes = graph.get_nodes();
    let mut cost = Vec::with_capacity(ids.len());
    for id in &ids {
        let c = match (weight_attr, &nodes[id].attributes) {
            (Some(attr), Some(d)) => match d.bind(py).downcast::<PyDict>()?.get_item(attr)? {
                Some(v) => v.extract::<f64>().unwrap_or(1.0),
                None => 1.0,
            },
            _ => 1.0,
        };
        if c < 0.0 || c.is_nan() {
            return value_err("node weights must be non-negative");
        }
        cost.push(c);
    }
    let mut in_cover = vec![false; ids.len()];
    let mut sorted_edges: Vec<_> = graph.get_edges_internal().iter().collect();
    sorted_edges.sort_unstable_by_key(|e| e.id);
    for e in sorted_edges {
        let (u, v) = (idx[&e.source], idx[&e.target]);
        if in_cover[u] || in_cover[v] {
            continue;
        }
        if cost[u] <= cost[v] {
            in_cover[u] = true;
            cost[v] -= cost[u];
        } else {
            in_cover[v] = true;
            cost[u] -= cost[v];
        }
    }
    Ok((0..ids.len())
        .filter(|&i| in_cover[i])
        .map(|i| ids[i])
        .collect())
}
