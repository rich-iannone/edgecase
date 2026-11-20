//! Clique kernels with NetworkX's exact output order.
//!
//! NetworkX's `find_cliques` (Tomita-style pivoting over Python sets) picks nodes in the
//! iteration order of Python sets: `set.pop()`, iteration for `max(...)`, and the sets built
//! by `&` and `-`. That order follows CPython's hash tables, so it is reproduced here with a
//! model of CPython's set (`PySetModel`: the same open addressing, probing, resizing, dummy
//! entries and the same algorithms for copy, intersection and difference), fed with the
//! node keys' Python hashes. The cliques, and the order of the nodes in each, then come out
//! as NetworkX's do. `enumerate_all_cliques` and `max_weight_clique` only depend on list
//! order and are direct ports.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::algorithms::path_search::PathAdjacency;

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const EMPTY: usize = usize::MAX;
const DUMMY: usize = usize::MAX - 1;
const MINSIZE: usize = 8;

/// A model of a CPython (3.13) `set` of node positions, whose table layout, and so whose
/// iteration and `pop()` order, follows CPython's exactly given the keys' hashes
#[derive(Clone)]
pub(crate) struct PySetModel {
    keys: Vec<usize>,
    hashes: Vec<i64>,
    mask: usize,
    fill: usize,
    used: usize,
    finger: usize,
}

impl PySetModel {
    pub(crate) fn new() -> Self {
        PySetModel {
            keys: vec![EMPTY; MINSIZE],
            hashes: vec![0; MINSIZE],
            mask: MINSIZE - 1,
            fill: 0,
            used: 0,
            finger: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.used
    }

    /// `set_lookkey`: the slot holding `key`, or the empty slot that ends its probe sequence
    fn lookkey(&self, key: usize, hash: i64) -> usize {
        let mask = self.mask;
        let mut perturb = hash as u64;
        let mut i = (hash as u64 as usize) & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                if self.keys[j] == EMPTY {
                    return j;
                }
                if self.hashes[j] == hash && self.keys[j] == key {
                    return j;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    pub(crate) fn contains(&self, key: usize, hash: i64) -> bool {
        self.keys[self.lookkey(key, hash)] != EMPTY
    }

    /// `set_add_entry`
    pub(crate) fn add(&mut self, key: usize, hash: i64) {
        let mask = self.mask;
        let mut perturb = hash as u64;
        let mut i = (hash as u64 as usize) & mask;
        let mut freeslot = None;
        let unused;
        'probe: loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                if self.keys[j] == EMPTY {
                    unused = j;
                    break 'probe;
                }
                if self.hashes[j] == hash {
                    if self.keys[j] == key {
                        return; // already present
                    }
                } else if self.hashes[j] == -1 {
                    freeslot = Some(j); // (a dummy: the last one seen is reused)
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
        if let Some(f) = freeslot {
            self.keys[f] = key;
            self.hashes[f] = hash;
            self.used += 1;
            return;
        }
        self.fill += 1;
        self.used += 1;
        self.keys[unused] = key;
        self.hashes[unused] = hash;
        if self.fill * 5 < mask * 3 {
            return;
        }
        let minused = if self.used > 50000 {
            self.used * 2
        } else {
            self.used * 4
        };
        self.resize(minused);
    }

    /// `set_insert_clean` into a table without dummies
    fn insert_clean(keys: &mut [usize], hashes: &mut [i64], mask: usize, key: usize, hash: i64) {
        let mut perturb = hash as u64;
        let mut i = (hash as u64 as usize) & mask;
        loop {
            if keys[i] == EMPTY {
                keys[i] = key;
                hashes[i] = hash;
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in i + 1..=i + LINEAR_PROBES {
                    if keys[j] == EMPTY {
                        keys[j] = key;
                        hashes[j] = hash;
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    /// `set_table_resize`
    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == MINSIZE && self.mask == MINSIZE - 1 && self.fill == self.used {
            return; // (the small table, without dummies: nothing to do)
        }
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY; newsize]);
        let old_hashes = std::mem::replace(&mut self.hashes, vec![0; newsize]);
        self.mask = newsize - 1;
        self.fill = self.used;
        for (&k, &h) in old_keys.iter().zip(&old_hashes) {
            if k != EMPTY && k != DUMMY {
                Self::insert_clean(&mut self.keys, &mut self.hashes, self.mask, k, h);
            }
        }
    }

    /// `set_discard_entry`
    pub(crate) fn discard(&mut self, key: usize, hash: i64) -> bool {
        let j = self.lookkey(key, hash);
        if self.keys[j] == EMPTY {
            return false;
        }
        self.keys[j] = DUMMY;
        self.hashes[j] = -1;
        self.used -= 1;
        true
    }

    /// `set.pop()` (None if empty)
    pub(crate) fn pop(&mut self) -> Option<usize> {
        if self.used == 0 {
            return None;
        }
        let mut i = self.finger & self.mask;
        while self.keys[i] == EMPTY || self.keys[i] == DUMMY {
            i += 1;
            if i > self.mask {
                i = 0;
            }
        }
        let key = self.keys[i];
        self.keys[i] = DUMMY;
        self.hashes[i] = -1;
        self.used -= 1;
        self.finger = i + 1;
        Some(key)
    }

    /// The members in iteration order
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        self.keys
            .iter()
            .zip(&self.hashes)
            .filter(|(&k, _)| k != EMPTY && k != DUMMY)
            .map(|(&k, &h)| (k, h))
    }

    /// `set_merge` (`update` with another set)
    fn merge(&mut self, other: &PySetModel) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.mask == other.mask && other.fill == other.used {
            self.keys.clone_from(&other.keys);
            self.hashes.clone_from(&other.hashes);
            self.fill = other.fill;
            self.used = other.used;
            return;
        }
        if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            for (k, h) in other.iter() {
                Self::insert_clean(&mut self.keys, &mut self.hashes, self.mask, k, h);
            }
            return;
        }
        for (k, h) in other.iter() {
            self.add(k, h);
        }
    }

    /// `set.copy()`
    pub(crate) fn copy(&self) -> PySetModel {
        let mut result = PySetModel::new();
        result.merge(self);
        result
    }

    /// `a & b` (`set_intersection`)
    pub(crate) fn intersection(a: &PySetModel, b: &PySetModel) -> PySetModel {
        let (so, other) = if b.used > a.used { (b, a) } else { (a, b) };
        let mut result = PySetModel::new();
        for (k, h) in other.iter() {
            if so.contains(k, h) {
                result.add(k, h);
            }
        }
        result
    }

    /// `a &= b` (the result's table replaces a's; a keeps its pop finger)
    pub(crate) fn intersection_update(&mut self, other: &PySetModel) {
        let tmp = PySetModel::intersection(self, other);
        self.keys = tmp.keys;
        self.hashes = tmp.hashes;
        self.mask = tmp.mask;
        self.fill = tmp.fill;
        self.used = tmp.used;
    }

    /// `a -= b` (`set_difference_update_internal`)
    pub(crate) fn difference_update(&mut self, other: &PySetModel) {
        let smaller;
        let other = if (other.used >> 3) > self.used {
            smaller = PySetModel::intersection(self, other);
            &smaller
        } else {
            other
        };
        let members: Vec<(usize, i64)> = other.iter().collect();
        for (k, h) in members {
            self.discard(k, h);
        }
        if self.fill - self.used <= self.mask / 4 {
            return;
        }
        let minused = if self.used > 50000 {
            self.used * 2
        } else {
            self.used * 4
        };
        self.resize(minused);
    }

    /// `a.difference_update(iterable)` with an iterable that is not a set: each key is
    /// discarded, then dummies are cleared away as for a set
    pub(crate) fn difference_update_iterable(&mut self, keys: impl Iterator<Item = (usize, i64)>) {
        for (k, h) in keys {
            self.discard(k, h);
        }
        if self.fill - self.used <= self.mask / 4 {
            return;
        }
        let minused = if self.used > 50000 {
            self.used * 2
        } else {
            self.used * 4
        };
        self.resize(minused);
    }

    /// `a - b` (`set_difference`)
    pub(crate) fn difference(a: &PySetModel, b: &PySetModel) -> PySetModel {
        if (a.used >> 2) > b.used {
            let mut result = a.copy();
            result.difference_update(b);
            return result;
        }
        let mut result = PySetModel::new();
        for (k, h) in a.iter() {
            if !b.contains(k, h) {
                result.add(k, h);
            }
        }
        result
    }

    /// The number of members of `self & other`
    fn intersection_size(&self, other: &PySetModel) -> usize {
        let (small, large) = if self.used <= other.used {
            (self, other)
        } else {
            (other, self)
        };
        small.iter().filter(|&(k, h)| large.contains(k, h)).count()
    }
}

/// A Python-side test hook: apply `ops` to model sets and return each set's iteration order
/// (to compare with real Python sets). Each op is `(name, target, other_or_key)`: "add",
/// "discard", "pop", "copy", "and", "iand", "sub", "isub" (`other_or_key` a key for add and
/// discard, else a set index).
#[pyfunction]
pub fn nx_pyset_model_check(
    hashes: Vec<i64>,
    sets: usize,
    ops: Vec<(String, usize, usize)>,
) -> PyResult<(Vec<Vec<usize>>, Vec<Option<usize>>)> {
    let mut models = vec![PySetModel::new(); sets];
    let mut popped = Vec::new();
    for (op, target, arg) in ops {
        match op.as_str() {
            "add" => models[target].add(arg, hashes[arg]),
            "discard" => {
                models[target].discard(arg, hashes[arg]);
            }
            "pop" => popped.push(models[target].pop()),
            "copy" => models[target] = models[arg].copy(),
            "and" => models[target] = PySetModel::intersection(&models[target], &models[arg]),
            "iand" => {
                let other = models[arg].clone();
                models[target].intersection_update(&other);
            }
            "sub" => models[target] = PySetModel::difference(&models[target], &models[arg]),
            "isub" => {
                let other = models[arg].clone();
                models[target].difference_update(&other);
            }
            _ => return Err(PyValueError::new_err(format!("unknown op {:?}", op))),
        }
    }
    Ok((
        models
            .iter()
            .map(|m| m.iter().map(|(k, _)| k).collect())
            .collect(),
        popped,
    ))
}

// ============================================================================
// find_cliques
// ============================================================================

/// NetworkX's `find_cliques` (its iterative version), producing maximal cliques in batches
#[pyclass(module = "edgecase._core")]
pub struct CliqueSearch {
    hashes: Vec<i64>,
    adj: Vec<PySetModel>,
    q: Vec<usize>,
    subg: PySetModel,
    cand: PySetModel,
    ext_u: PySetModel,
    stack: Vec<(PySetModel, PySetModel, PySetModel)>,
    /// A clique to yield before searching (when the given nodes leave no candidates)
    pending: Option<Vec<usize>>,
    done: bool,
}

impl CliqueSearch {
    /// `max(subg, key=lambda u: len(cand & adj[u]))`
    fn pivot(&self) -> usize {
        let mut best = EMPTY;
        let mut best_size = 0;
        for (u, _) in self.subg.iter() {
            let size = self.cand.intersection_size(&self.adj[u]);
            if best == EMPTY || size > best_size {
                best = u;
                best_size = size;
            }
        }
        best
    }
}

#[pymethods]
impl CliqueSearch {
    /// Start NetworkX's `find_cliques(G, nodes)` (`nodes` positions, None for []) on the
    /// graph's adjacency, with the node keys' Python `hash()`es. Returns the search and
    /// whether `nodes` form a clique (if not, NetworkX raises ValueError).
    #[staticmethod]
    #[pyo3(signature = (adjacency, hashes, nodes=None))]
    fn start(
        adjacency: &PathAdjacency,
        hashes: Vec<i64>,
        nodes: Option<Vec<usize>>,
    ) -> PyResult<(CliqueSearch, bool)> {
        let n = adjacency.len();
        if hashes.len() != n {
            return Err(PyValueError::new_err("hashes must have one entry per node"));
        }
        let (indptr, indices, _) = adjacency.rows();
        // adj = {u: {v for v in G[u] if v != u} for u in G}
        let adj: Vec<PySetModel> = (0..n)
            .map(|u| {
                let mut s = PySetModel::new();
                for &v in &indices[indptr[u]..indptr[u + 1]] {
                    if v != u {
                        s.add(v, hashes[v]);
                    }
                }
                s
            })
            .collect();
        let mut search = CliqueSearch {
            hashes,
            adj,
            q: nodes.unwrap_or_default(),
            subg: PySetModel::new(),
            cand: PySetModel::new(),
            ext_u: PySetModel::new(),
            stack: Vec::new(),
            pending: None,
            done: n == 0,
        };
        if n == 0 {
            return Ok((search, true));
        }
        let mut cand = PySetModel::new();
        for v in 0..n {
            cand.add(v, search.hashes[v]);
        }
        for i in 0..search.q.len() {
            let node = search.q[i];
            if node >= n || !cand.contains(node, search.hashes[node]) {
                search.done = true;
                return Ok((search, false));
            }
            let adj_node = search.adj[node].clone();
            cand.intersection_update(&adj_node);
        }
        if cand.len() == 0 {
            search.pending = Some(search.q.clone());
            search.done = true;
            return Ok((search, true));
        }
        search.subg = cand.copy();
        search.cand = cand;
        search.q.push(EMPTY);
        let u = search.pivot();
        search.ext_u = PySetModel::difference(&search.cand, &search.adj[u]);
        Ok((search, true))
    }

    /// Up to `limit` more cliques (node positions); an empty list when the search is over
    fn next_batch(&mut self, py: Python<'_>, limit: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        if let Some(clique) = self.pending.take() {
            out.push(clique);
        }
        if self.done {
            return out;
        }
        py.allow_threads(|| {
            while out.len() < limit {
                if self.ext_u.len() > 0 {
                    let q = self.ext_u.pop().expect("not empty");
                    self.cand.discard(q, self.hashes[q]);
                    *self.q.last_mut().expect("not empty") = q;
                    let adj_q = &self.adj[q];
                    let subg_q = PySetModel::intersection(&self.subg, adj_q);
                    if subg_q.len() == 0 {
                        out.push(self.q.clone());
                    } else {
                        let cand_q = PySetModel::intersection(&self.cand, adj_q);
                        if cand_q.len() > 0 {
                            let subg = std::mem::replace(&mut self.subg, subg_q);
                            let cand = std::mem::replace(&mut self.cand, cand_q);
                            let ext_u = std::mem::replace(&mut self.ext_u, PySetModel::new());
                            self.stack.push((subg, cand, ext_u));
                            self.q.push(EMPTY);
                            let u = self.pivot();
                            self.ext_u = PySetModel::difference(&self.cand, &self.adj[u]);
                        }
                    }
                } else {
                    self.q.pop();
                    match self.stack.pop() {
                        Some((subg, cand, ext_u)) => {
                            self.subg = subg;
                            self.cand = cand;
                            self.ext_u = ext_u;
                        }
                        None => {
                            self.done = true;
                            break;
                        }
                    }
                }
            }
        });
        out
    }
}

// ============================================================================
// enumerate_all_cliques and max_weight_clique
// ============================================================================

/// NetworkX's `enumerate_all_cliques`: every clique, by size and then in NetworkX's order,
/// produced in batches
#[pyclass(module = "edgecase._core")]
pub struct AllCliques {
    /// Each node's neighbors that come after it in node order
    later: Vec<FxHashSet<usize>>,
    queue: std::collections::VecDeque<(Vec<usize>, Vec<usize>)>,
}

#[pymethods]
impl AllCliques {
    #[new]
    fn new(adjacency: &PathAdjacency) -> Self {
        let n = adjacency.len();
        let (indptr, indices, _) = adjacency.rows();
        // nbrs[u] = {v for v in G[u] if v not in index}: the neighbors after u (u itself is
        // already indexed, so a self-loop does not count)
        let later: Vec<FxHashSet<usize>> = (0..n)
            .map(|u| {
                indices[indptr[u]..indptr[u + 1]]
                    .iter()
                    .copied()
                    .filter(|&v| v > u)
                    .collect()
            })
            .collect();
        let queue = (0..n)
            .map(|u| {
                let mut cnbrs: Vec<usize> = later[u].iter().copied().collect();
                cnbrs.sort_unstable();
                (vec![u], cnbrs)
            })
            .collect();
        AllCliques { later, queue }
    }

    /// Up to `limit` more cliques; an empty list when all are done
    fn next_batch(&mut self, py: Python<'_>, limit: usize) -> Vec<Vec<usize>> {
        py.allow_threads(|| {
            let mut out = Vec::new();
            while out.len() < limit {
                let Some((base, cnbrs)) = self.queue.pop_front() else {
                    break;
                };
                for (i, &u) in cnbrs.iter().enumerate() {
                    let mut next = base.clone();
                    next.push(u);
                    let rest: Vec<usize> = cnbrs[i + 1..]
                        .iter()
                        .copied()
                        .filter(|w| self.later[u].contains(w))
                        .collect();
                    self.queue.push_back((next, rest));
                }
                out.push(base);
            }
            out
        })
    }
}

/// NetworkX's `MaxWeightClique` branch and bound over `nodes` (positions, in NetworkX's
/// starting order: by degree, positive weights only) with integer `weights`. Returns the
/// best clique found first and its weight.
#[pyfunction]
pub fn nx_max_weight_clique(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    nodes: Vec<usize>,
    weights: Vec<i64>,
) -> PyResult<(Vec<usize>, i64)> {
    let n = adjacency.len();
    if weights.len() != n || nodes.iter().any(|&v| v >= n) {
        return Err(PyValueError::new_err(
            "one weight per node; nodes are positions",
        ));
    }
    let (indptr, indices, _) = adjacency.rows();
    let neighbors: Vec<FxHashSet<usize>> = (0..n)
        .map(|u| indices[indptr[u]..indptr[u + 1]].iter().copied().collect())
        .collect();
    struct Search<'a> {
        neighbors: &'a [FxHashSet<usize>],
        weights: &'a [i64],
        best: Vec<usize>,
        best_weight: i64,
    }
    impl Search<'_> {
        fn has_edge(&self, v: usize, w: usize) -> bool {
            self.neighbors[v].contains(&w)
        }
        fn independent_set(&self, p: &[usize]) -> Vec<usize> {
            let mut set = Vec::new();
            let mut p = p.to_vec();
            while let Some(&v) = p.first() {
                set.push(v);
                p.retain(|&w| v != w && !self.has_edge(v, w));
            }
            set
        }
        fn branching_nodes(&self, p: &[usize], target: i64) -> Vec<usize> {
            let mut residual: rustc_hash::FxHashMap<usize, i64> =
                p.iter().map(|&v| (v, self.weights[v])).collect();
            let mut total = 0i64;
            let mut p = p.to_vec();
            while !p.is_empty() {
                let set = self.independent_set(&p);
                let min_wt = set.iter().map(|v| residual[v]).min().expect("not empty");
                total += min_wt;
                if total > target {
                    break;
                }
                for v in &set {
                    *residual.get_mut(v).expect("in P") -= min_wt;
                }
                p.retain(|v| residual[v] != 0);
            }
            p
        }
        fn expand(&mut self, c: &[usize], c_weight: i64, mut p: Vec<usize>) {
            if c_weight > self.best_weight {
                self.best = c.to_vec();
                self.best_weight = c_weight;
            }
            let mut branching = self.branching_nodes(&p, self.best_weight - c_weight);
            while let Some(v) = branching.pop() {
                if let Some(pos) = p.iter().position(|&x| x == v) {
                    p.remove(pos);
                }
                let mut new_c = c.to_vec();
                new_c.push(v);
                let new_weight = c_weight + self.weights[v];
                let new_p: Vec<usize> =
                    p.iter().copied().filter(|&w| self.has_edge(v, w)).collect();
                self.expand(&new_c, new_weight, new_p);
            }
        }
    }
    Ok(py.allow_threads(|| {
        let mut search = Search {
            neighbors: &neighbors,
            weights: &weights,
            best: Vec::new(),
            best_weight: 0,
        };
        search.expand(&[], 0, nodes);
        (search.best, search.best_weight)
    }))
}
