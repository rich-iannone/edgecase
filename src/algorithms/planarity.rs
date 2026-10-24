//! Planarity testing, planar embeddings, and straight-line planar drawings.
//!
//! A port of NetworkX's `check_planarity()` (Brandes' left-right planarity test, which yields a
//! combinatorial embedding when the graph is planar), its edge-removal search for a Kuratowski
//! subgraph, and `combinatorial_embedding_to_pos()` (triangulation, canonical ordering, and the
//! Chrobak-Payne shift method, giving integer grid positions without edge crossings).
//!
//! Everything works on node indices `0..n` and a simple undirected edge list prepared by the
//! Python side (no self-loops, no parallel edges). Where NetworkX picks from a hash set or keeps
//! insertion order, this module picks the smallest index, so results are deterministic but can
//! differ from NetworkX's in which (equally valid) embedding or drawing comes out.

use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

// --- embedding -----------------------------------------------------------------------------

/// A rotation system: for each node, its neighbors in a cyclic order, stored as clockwise and
/// counterclockwise links, plus the "leftmost" neighbor where the clockwise listing starts.
#[derive(Clone)]
struct Embedding {
    nbrs: Vec<HashMap<usize, (usize, usize)>>, // node -> neighbor -> (cw, ccw)
    leftmost: Vec<Option<usize>>,
}

impl Embedding {
    fn new(n: usize) -> Self {
        Embedding {
            nbrs: vec![HashMap::new(); n],
            leftmost: vec![None; n],
        }
    }

    fn n(&self) -> usize {
        self.nbrs.len()
    }

    fn cw(&self, v: usize, w: usize) -> usize {
        self.nbrs[v][&w].0
    }

    fn ccw(&self, v: usize, w: usize) -> usize {
        self.nbrs[v][&w].1
    }

    fn has_edge(&self, v: usize, w: usize) -> bool {
        self.nbrs[v].contains_key(&w)
    }

    /// Add the half-edge `s -> t`, directly counterclockwise of `cw` or clockwise of `ccw`
    /// (exactly one of them, unless `t` is the first neighbor of `s`).
    fn add_half_edge(&mut self, s: usize, t: usize, cw: Option<usize>, ccw: Option<usize>) {
        if self.nbrs[s].is_empty() {
            self.nbrs[s].insert(t, (t, t));
            self.leftmost[s] = Some(t);
            return;
        }
        match (cw, ccw) {
            (Some(c), None) => {
                let ref_ccw = self.ccw(s, c);
                self.nbrs[s].insert(t, (c, ref_ccw));
                self.nbrs[s].get_mut(&ref_ccw).unwrap().0 = t;
                self.nbrs[s].get_mut(&c).unwrap().1 = t;
                if self.leftmost[s] == Some(c) {
                    self.leftmost[s] = Some(t);
                }
            }
            (None, Some(c)) => {
                let ref_cw = self.cw(s, c);
                self.nbrs[s].insert(t, (ref_cw, c));
                self.nbrs[s].get_mut(&ref_cw).unwrap().1 = t;
                self.nbrs[s].get_mut(&c).unwrap().0 = t;
            }
            _ => unreachable!("a reference neighbor is required"),
        }
    }

    /// Add `s -> t` and make `t` the leftmost neighbor of `s`.
    fn add_half_edge_first(&mut self, s: usize, t: usize) {
        let lm = self.leftmost[s];
        self.add_half_edge(s, t, lm, None);
    }

    /// Join two nodes of different components with an edge.
    fn connect_components(&mut self, v: usize, w: usize) {
        let r = self.leftmost[v];
        self.add_half_edge(v, w, r, None);
        let r = self.leftmost[w];
        self.add_half_edge(w, v, r, None);
    }

    /// The half-edge that follows `v -> w` around their shared face.
    fn next_face_half_edge(&self, v: usize, w: usize) -> (usize, usize) {
        (w, self.ccw(w, v))
    }

    fn neighbors_cw(&self, v: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.nbrs[v].len());
        if let Some(start) = self.leftmost[v] {
            out.push(start);
            let mut cur = self.cw(v, start);
            while cur != start {
                out.push(cur);
                cur = self.cw(v, cur);
            }
        }
        out
    }
}

// --- LR planarity test ---------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Interval {
    low: Option<usize>, // oriented edge ids
    high: Option<usize>,
}

impl Interval {
    fn empty(&self) -> bool {
        self.low.is_none() && self.high.is_none()
    }
}

#[derive(Clone, Copy)]
struct ConflictPair {
    left: Interval,
    right: Interval,
    uid: usize, // identity, for the stack-bottom markers
}

impl ConflictPair {
    fn swap(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right);
    }
}

struct Lr {
    // oriented edges
    src: Vec<usize>,
    dst: Vec<usize>,
    out: Vec<Vec<usize>>, // node -> oriented edges leaving it, in orientation order
    height: Vec<Option<usize>>,
    parent_edge: Vec<Option<usize>>,
    lowpt: Vec<usize>,
    lowpt2: Vec<usize>,
    nesting: Vec<i64>,
    refs: Vec<Option<usize>>,
    side: Vec<i64>,
    stack: Vec<ConflictPair>,
    stack_bottom: Vec<Option<usize>>,
    lowpt_edge: Vec<Option<usize>>,
    next_uid: usize,
    roots: Vec<usize>,
}

impl Lr {
    fn top_uid(&self) -> Option<usize> {
        self.stack.last().map(|p| p.uid)
    }

    fn pair(&mut self, left: Interval, right: Interval) -> ConflictPair {
        self.next_uid += 1;
        ConflictPair {
            left,
            right,
            uid: self.next_uid,
        }
    }

    fn conflicting(&self, i: &Interval, b: usize) -> bool {
        !i.empty() && i.high.is_some_and(|h| self.lowpt[h] > self.lowpt[b])
    }

    fn lowest(&self, p: &ConflictPair) -> usize {
        match (p.left.low, p.right.low) {
            (None, Some(r)) => self.lowpt[r],
            (Some(l), None) => self.lowpt[l],
            (Some(l), Some(r)) => self.lowpt[l].min(self.lowpt[r]),
            (None, None) => usize::MAX,
        }
    }

    fn set_ref(&mut self, e: Option<usize>, to: Option<usize>) {
        if let Some(e) = e {
            self.refs[e] = to;
        }
    }

    fn orient(&mut self, adj: &[Vec<usize>]) {
        let n = adj.len();
        let mut oriented: HashMap<(usize, usize), usize> = HashMap::new();
        let mut ind = vec![0usize; n];
        let mut skip_init: HashSet<usize> = HashSet::new();
        for root in 0..n {
            if self.height[root].is_some() {
                continue;
            }
            self.height[root] = Some(0);
            self.roots.push(root);
            let mut dfs = vec![root];
            while let Some(v) = dfs.pop() {
                let e = self.parent_edge[v];
                while ind[v] < adj[v].len() {
                    let w = adj[v][ind[v]];
                    let vw = match oriented.get(&(v, w)) {
                        Some(&id) if skip_init.contains(&id) => id,
                        _ => {
                            if oriented.contains_key(&(v, w)) || oriented.contains_key(&(w, v)) {
                                ind[v] += 1;
                                continue;
                            }
                            let id = self.src.len();
                            self.src.push(v);
                            self.dst.push(w);
                            self.out[v].push(id);
                            oriented.insert((v, w), id);
                            let hv = self.height[v].unwrap();
                            self.lowpt.push(hv);
                            self.lowpt2.push(hv);
                            self.nesting.push(0);
                            if self.height[w].is_none() {
                                self.parent_edge[w] = Some(id);
                                self.height[w] = Some(hv + 1);
                                dfs.push(v);
                                dfs.push(w);
                                skip_init.insert(id);
                                break;
                            }
                            self.lowpt[id] = self.height[w].unwrap();
                            id
                        }
                    };
                    // nesting order, and the lowpoints of the parent edge
                    let hv = self.height[v].unwrap();
                    self.nesting[vw] = 2 * self.lowpt[vw] as i64;
                    if self.lowpt2[vw] < hv {
                        self.nesting[vw] += 1;
                    }
                    if let Some(e) = e {
                        if self.lowpt[vw] < self.lowpt[e] {
                            self.lowpt2[e] = self.lowpt[e].min(self.lowpt2[vw]);
                            self.lowpt[e] = self.lowpt[vw];
                        } else if self.lowpt[vw] > self.lowpt[e] {
                            self.lowpt2[e] = self.lowpt2[e].min(self.lowpt[vw]);
                        } else {
                            self.lowpt2[e] = self.lowpt2[e].min(self.lowpt2[vw]);
                        }
                    }
                    ind[v] += 1;
                }
            }
        }
    }

    fn sort_out(&mut self) {
        for v in 0..self.out.len() {
            let mut list = std::mem::take(&mut self.out[v]);
            list.sort_by_key(|&e| self.nesting[e]); // stable, like Python's sorted()
            self.out[v] = list;
        }
    }

    fn testing(&mut self) -> bool {
        let n = self.out.len();
        let mut ind = vec![0usize; n];
        let mut skip_init: HashSet<usize> = HashSet::new();
        for ri in 0..self.roots.len() {
            let mut dfs = vec![self.roots[ri]];
            while let Some(v) = dfs.pop() {
                let e = self.parent_edge[v];
                let mut skip_final = false;
                while ind[v] < self.out[v].len() {
                    let ei = self.out[v][ind[v]];
                    let w = self.dst[ei];
                    if !skip_init.contains(&ei) {
                        self.stack_bottom[ei] = self.top_uid();
                        if self.parent_edge[w] == Some(ei) {
                            dfs.push(v);
                            dfs.push(w);
                            skip_init.insert(ei);
                            skip_final = true;
                            break;
                        }
                        self.lowpt_edge[ei] = Some(ei);
                        let p = self.pair(
                            Interval::default(),
                            Interval {
                                low: Some(ei),
                                high: Some(ei),
                            },
                        );
                        self.stack.push(p);
                    }
                    if self.lowpt[ei] < self.height[v].unwrap() {
                        if ind[v] == 0 {
                            let e = e.expect("a non-root node has a parent edge");
                            self.lowpt_edge[e] = self.lowpt_edge[ei];
                        } else if !self.add_constraints(ei, e.expect("non-root")) {
                            return false;
                        }
                    }
                    ind[v] += 1;
                }
                if !skip_final {
                    if let Some(e) = e {
                        self.remove_back_edges(e);
                    }
                }
            }
        }
        true
    }

    fn add_constraints(&mut self, ei: usize, e: usize) -> bool {
        let mut p = self.pair(Interval::default(), Interval::default());
        // merge return edges of e_i into P.right
        loop {
            let Some(mut q) = self.stack.pop() else {
                return false;
            };
            if !q.left.empty() {
                q.swap();
            }
            if !q.left.empty() {
                return false;
            }
            let qlow = q.right.low.expect("non-empty interval");
            if self.lowpt[qlow] > self.lowpt[e] {
                if p.right.empty() {
                    p.right = q.right;
                } else {
                    self.set_ref(p.right.low, q.right.high);
                }
                p.right.low = q.right.low;
            } else {
                self.refs[qlow] = self.lowpt_edge[e];
            }
            if self.top_uid() == self.stack_bottom[ei] {
                break;
            }
        }
        // merge conflicting return edges of e_1, ..., e_{i-1} into P.left
        while let Some(top) = self.stack.last() {
            if !(self.conflicting(&top.left, ei) || self.conflicting(&top.right, ei)) {
                break;
            }
            let mut q = self.stack.pop().unwrap();
            if self.conflicting(&q.right, ei) {
                q.swap();
            }
            if self.conflicting(&q.right, ei) {
                return false;
            }
            self.set_ref(p.right.low, q.right.high);
            if q.right.low.is_some() {
                p.right.low = q.right.low;
            }
            if p.left.empty() {
                p.left = q.left;
            } else {
                self.set_ref(p.left.low, q.left.high);
            }
            p.left.low = q.left.low;
        }
        if !(p.left.empty() && p.right.empty()) {
            self.stack.push(p);
        }
        true
    }

    fn remove_back_edges(&mut self, e: usize) {
        let u = self.src[e];
        let hu = self.height[u].unwrap();
        while let Some(top) = self.stack.last() {
            if self.lowest(top) != hu {
                break;
            }
            let p = self.stack.pop().unwrap();
            if let Some(l) = p.left.low {
                self.side[l] = -1;
            }
        }
        if let Some(mut p) = self.stack.pop() {
            // trim the left interval
            while let Some(h) = p.left.high {
                if self.dst[h] != u {
                    break;
                }
                p.left.high = self.refs[h];
            }
            if p.left.high.is_none() {
                if let Some(l) = p.left.low {
                    self.refs[l] = p.right.low;
                    self.side[l] = -1;
                    p.left.low = None;
                }
            }
            // trim the right interval
            while let Some(h) = p.right.high {
                if self.dst[h] != u {
                    break;
                }
                p.right.high = self.refs[h];
            }
            if p.right.high.is_none() {
                if let Some(r) = p.right.low {
                    self.refs[r] = p.left.low;
                    self.side[r] = -1;
                    p.right.low = None;
                }
            }
            self.stack.push(p);
        }
        // the side of e is the side of a highest return edge
        if self.lowpt[e] < hu {
            if let Some(top) = self.stack.last() {
                let (hl, hr) = (top.left.high, top.right.high);
                self.refs[e] = match (hl, hr) {
                    (Some(l), None) => Some(l),
                    (Some(l), Some(r)) if self.lowpt[l] > self.lowpt[r] => Some(l),
                    _ => hr,
                };
            }
        }
    }

    /// Resolve the side of `e` relative to its reference chain into an absolute side.
    fn sign(&mut self, e: usize) -> i64 {
        let mut chain = vec![e];
        let mut cur = e;
        while let Some(r) = self.refs[cur] {
            chain.push(r);
            cur = r;
        }
        // unwind from the far end: side[x] *= side[ref[x]]
        for i in (0..chain.len() - 1).rev() {
            let (x, r) = (chain[i], chain[i + 1]);
            self.side[x] *= self.side[r];
            self.refs[x] = None;
        }
        self.side[e]
    }
}

/// The LR planarity test. Returns a planar embedding, or `None` if the graph is not planar.
/// `adj` must be a simple undirected adjacency (symmetric, no loops, no duplicates).
fn lr_planarity(adj: &[Vec<usize>]) -> Option<Embedding> {
    let n = adj.len();
    let m: usize = adj.iter().map(Vec::len).sum::<usize>() / 2;
    if n > 2 && m > 3 * n - 6 {
        return None;
    }
    let mut s = Lr {
        src: Vec::with_capacity(m),
        dst: Vec::with_capacity(m),
        out: vec![Vec::new(); n],
        height: vec![None; n],
        parent_edge: vec![None; n],
        lowpt: Vec::with_capacity(m),
        lowpt2: Vec::with_capacity(m),
        nesting: Vec::with_capacity(m),
        refs: Vec::new(),
        side: Vec::new(),
        stack: Vec::new(),
        stack_bottom: Vec::new(),
        lowpt_edge: Vec::new(),
        next_uid: 0,
        roots: Vec::new(),
    };
    s.orient(adj);
    let m = s.src.len();
    s.refs = vec![None; m];
    s.side = vec![1; m];
    s.stack_bottom = vec![None; m];
    s.lowpt_edge = vec![None; m];
    s.sort_out();
    if !s.testing() {
        return None;
    }
    for e in 0..m {
        s.nesting[e] *= s.sign(e);
    }
    s.sort_out();

    let mut emb = Embedding::new(n);
    for v in 0..n {
        let mut prev = None;
        for &e in &s.out[v] {
            let w = s.dst[e];
            emb.add_half_edge(v, w, None, prev);
            prev = Some(w);
        }
    }
    // complete the embedding with the reverse half-edges
    let mut left_ref: Vec<usize> = vec![usize::MAX; n];
    let mut right_ref: Vec<usize> = vec![usize::MAX; n];
    let mut ind = vec![0usize; n];
    for &root in &s.roots {
        let mut dfs = vec![root];
        while let Some(v) = dfs.pop() {
            while ind[v] < s.out[v].len() {
                let ei = s.out[v][ind[v]];
                ind[v] += 1;
                let w = s.dst[ei];
                if s.parent_edge[w] == Some(ei) {
                    emb.add_half_edge_first(w, v);
                    left_ref[v] = w;
                    right_ref[v] = w;
                    dfs.push(v);
                    dfs.push(w);
                    break;
                } else if s.side[ei] == 1 {
                    emb.add_half_edge(w, v, None, Some(right_ref[w]));
                } else {
                    emb.add_half_edge(w, v, Some(left_ref[w]), None);
                    left_ref[w] = v;
                }
            }
        }
    }
    Some(emb)
}

fn simple_adjacency(n: usize, edges: &[(usize, usize)]) -> PyResult<Vec<Vec<usize>>> {
    let mut adj = vec![Vec::new(); n];
    let mut seen = HashSet::new();
    for &(u, v) in edges {
        if u >= n || v >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "edge endpoint out of range",
            ));
        }
        if u == v || !seen.insert((u.min(v), u.max(v))) {
            continue;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    Ok(adj)
}

/// Whether the graph on `0..n` with `edges` is planar; self-loops and repeated edges are fine.
pub(crate) fn is_planar_edges(n: usize, edges: &[(usize, usize)]) -> bool {
    simple_adjacency(n, edges).is_ok_and(|adj| lr_planarity(&adj).is_some())
}

// --- planar drawing ------------------------------------------------------------------------

/// Add edges inside the face to the right of `v1 -> v2` until it is triangulated.
fn triangulate_face(emb: &mut Embedding, mut v1: usize, mut v2: usize) {
    let (_, mut v3) = emb.next_face_half_edge(v1, v2);
    let (_, mut v4) = emb.next_face_half_edge(v2, v3);
    if v1 == v2 || v1 == v3 {
        return; // the component has fewer than three nodes
    }
    while v1 != v4 {
        if emb.has_edge(v1, v3) {
            (v1, v2, v3) = (v2, v3, v4);
        } else {
            emb.add_half_edge(v1, v3, None, Some(v2));
            emb.add_half_edge(v3, v1, Some(v2), None);
            (v2, v3) = (v3, v4);
        }
        v4 = emb.next_face_half_edge(v2, v3).1;
    }
}

/// Walk the face of `start -> out`, adding an edge wherever a node repeats (so the graph
/// becomes 2-connected), and return the face's nodes; empty if the face was seen before.
fn make_bi_connected(
    emb: &mut Embedding,
    start: usize,
    out: usize,
    counted: &mut HashSet<(usize, usize)>,
) -> Vec<usize> {
    if !counted.insert((start, out)) {
        return Vec::new();
    }
    let (mut v1, mut v2) = (start, out);
    let mut face = vec![start];
    let mut face_set: HashSet<usize> = HashSet::from([start]);
    let (_, mut v3) = emb.next_face_half_edge(v1, v2);
    while v2 != start || v3 != out {
        if face_set.contains(&v2) {
            emb.add_half_edge(v1, v3, None, Some(v2));
            emb.add_half_edge(v3, v1, Some(v2), None);
            counted.insert((v2, v3));
            counted.insert((v3, v1));
            v2 = v1;
        } else {
            face_set.insert(v2);
            face.push(v2);
        }
        v1 = v2;
        (v2, v3) = emb.next_face_half_edge(v2, v3);
        counted.insert((v1, v2));
    }
    face
}

/// Triangulate every face but the largest, which becomes the outer face.
fn triangulate_embedding(emb: &Embedding) -> (Embedding, Vec<usize>) {
    let mut emb = emb.clone();
    let n = emb.n();
    // one representative (the smallest node) per connected component
    let mut comp = vec![usize::MAX; n];
    let mut reps = Vec::new();
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        reps.push(s);
        comp[s] = s;
        let mut stack = vec![s];
        while let Some(v) = stack.pop() {
            for &w in emb.nbrs[v].keys() {
                if comp[w] == usize::MAX {
                    comp[w] = s;
                    stack.push(w);
                }
            }
        }
    }
    for pair in reps.windows(2) {
        emb.connect_components(pair[0], pair[1]);
    }
    let mut outer: Vec<usize> = Vec::new();
    let mut outer_idx = usize::MAX;
    let mut faces = Vec::new();
    let mut counted = HashSet::new();
    for v in 0..n {
        // iterate lazily, as edges may be added at v along the way
        let Some(start) = emb.leftmost[v] else {
            continue;
        };
        let mut w = start;
        loop {
            let face = make_bi_connected(&mut emb, v, w, &mut counted);
            if !face.is_empty() {
                if face.len() > outer.len() {
                    outer = face.clone();
                    outer_idx = faces.len();
                }
                faces.push(face);
            }
            w = emb.cw(v, w);
            if w == start {
                break;
            }
        }
    }
    for (i, face) in faces.iter().enumerate() {
        if i != outer_idx {
            triangulate_face(&mut emb, face[0], face[1]);
        }
    }
    (emb, outer)
}

/// A canonical ordering of a triangulated embedding: `(v_k, contour neighbors)` pairs.
fn canonical_ordering(emb: &Embedding, outer: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let n = emb.n();
    let (v1, v2) = (outer[0], outer[1]);
    let mut chords = vec![0i64; n];
    let mut marked = vec![false; n];
    let mut ready: BTreeSet<usize> = outer.iter().copied().collect();

    let mut ccw_nbr: HashMap<usize, usize> = HashMap::new();
    let mut prev = v2;
    for &x in &outer[2..] {
        ccw_nbr.insert(prev, x);
        prev = x;
    }
    ccw_nbr.insert(prev, v1);
    let mut cw_nbr: HashMap<usize, usize> = HashMap::new();
    let mut prev = v1;
    for idx in (1..outer.len()).rev() {
        cw_nbr.insert(prev, outer[idx]);
        prev = outer[idx];
    }

    let is_outer_nbr = |ccw_nbr: &HashMap<usize, usize>,
                        cw_nbr: &HashMap<usize, usize>,
                        x: usize,
                        y: usize| {
        match (ccw_nbr.get(&x), cw_nbr.get(&x)) {
            (None, c) => c == Some(&y),
            (a, None) => a == Some(&y),
            (a, c) => a == Some(&y) || c == Some(&y),
        }
    };
    let on_outer = |ccw_nbr: &HashMap<usize, usize>, marked: &[bool], x: usize| {
        !marked[x] && (ccw_nbr.contains_key(&x) || x == v1)
    };

    for &v in outer {
        for nbr in emb.neighbors_cw(v) {
            if on_outer(&ccw_nbr, &marked, nbr) && !is_outer_nbr(&ccw_nbr, &cw_nbr, v, nbr) {
                chords[v] += 1;
                ready.remove(&v);
            }
        }
    }

    let mut order: Vec<(usize, Vec<usize>)> = vec![(usize::MAX, Vec::new()); n];
    order[0] = (v1, Vec::new());
    order[1] = (v2, Vec::new());
    ready.remove(&v1);
    ready.remove(&v2);
    for k in (2..n).rev() {
        let v = ready
            .pop_first()
            .expect("a triangulated embedding has a removable node");
        marked[v] = true;
        let (mut wp, mut wq) = (None, None);
        for nbr in emb.neighbors_cw(v) {
            if marked[nbr] {
                continue;
            }
            if on_outer(&ccw_nbr, &marked, nbr) {
                if nbr == v1 {
                    wp = Some(v1);
                } else if nbr == v2 {
                    wq = Some(v2);
                } else if cw_nbr.get(&nbr) == Some(&v) {
                    wp = Some(nbr);
                } else {
                    wq = Some(nbr);
                }
            }
            if wp.is_some() && wq.is_some() {
                break;
            }
        }
        let (wp, wq) = (wp.unwrap(), wq.unwrap());
        let mut wp_wq = vec![wp];
        let mut nbr = wp;
        while nbr != wq {
            let next = emb.ccw(v, nbr);
            wp_wq.push(next);
            cw_nbr.insert(nbr, next);
            ccw_nbr.insert(next, nbr);
            nbr = next;
        }
        if wp_wq.len() == 2 {
            for w in [wp, wq] {
                chords[w] -= 1;
                if chords[w] == 0 {
                    ready.insert(w);
                }
            }
        } else {
            let new_face: BTreeSet<usize> = wp_wq[1..wp_wq.len() - 1].iter().copied().collect();
            for &w in &new_face {
                ready.insert(w);
                for nb in emb.neighbors_cw(w) {
                    if on_outer(&ccw_nbr, &marked, nb) && !is_outer_nbr(&ccw_nbr, &cw_nbr, w, nb) {
                        chords[w] += 1;
                        ready.remove(&w);
                        if !new_face.contains(&nb) {
                            chords[nb] += 1;
                            ready.remove(&nb);
                        }
                    }
                }
            }
        }
        order[k] = (v, wp_wq);
    }
    order
}

/// Integer grid positions for a planar embedding (Chrobak-Payne), without edge crossings.
fn embedding_to_pos(emb: &Embedding) -> Vec<(i64, i64)> {
    let n = emb.n();
    if n < 4 {
        return [(0, 0), (2, 0), (1, 1)][..n].to_vec();
    }
    let (emb, outer) = triangulate_embedding(emb);
    let order = canonical_ordering(&emb, &outer);
    let mut left: Vec<Option<usize>> = vec![None; n];
    let mut right: Vec<Option<usize>> = vec![None; n];
    let mut dx = vec![0i64; n];
    let mut y = vec![0i64; n];
    let (v1, v2, v3) = (order[0].0, order[1].0, order[2].0);
    dx[v1] = 0;
    y[v1] = 0;
    right[v1] = Some(v3);
    dx[v2] = 1;
    y[v2] = 0;
    dx[v3] = 1;
    y[v3] = 1;
    right[v3] = Some(v2);
    for (vk, contour) in &order[3..] {
        let vk = *vk;
        let (wp, wp1) = (contour[0], contour[1]);
        let (wq, wq1) = (contour[contour.len() - 1], contour[contour.len() - 2]);
        let multi = contour.len() > 2;
        dx[wp1] += 1;
        dx[wq] += 1;
        let span: i64 = contour[1..].iter().map(|&x| dx[x]).sum();
        dx[vk] = (-y[wp] + span + y[wq]).div_euclid(2);
        y[vk] = (y[wp] + span + y[wq]).div_euclid(2);
        dx[wq] = span - dx[vk];
        if multi {
            dx[wp1] -= dx[vk];
        }
        right[wp] = Some(vk);
        right[vk] = Some(wq);
        if multi {
            left[vk] = Some(wp1);
            right[wq1] = None;
        } else {
            left[vk] = None;
        }
    }
    let mut pos = vec![(0i64, 0i64); n];
    pos[v1] = (0, y[v1]);
    let mut stack = vec![v1];
    while let Some(p) = stack.pop() {
        for child in [left[p], right[p]].into_iter().flatten() {
            pos[child] = (pos[p].0 + dx[child], y[child]);
            stack.push(child);
        }
    }
    pos
}

/// The indices of a minimal non-planar edge set (a Kuratowski subgraph) of a non-planar graph,
/// found as NetworkX does: drop each edge in turn, and put it back if the rest became planar.
fn kuratowski_edges(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut alive: Vec<bool> = vec![true; edges.len()];
    let adj_of = |alive: &[bool]| {
        let mut adj = vec![Vec::new(); n];
        for (i, &(u, v)) in edges.iter().enumerate() {
            if alive[i] {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
        adj
    };
    for i in 0..edges.len() {
        alive[i] = false;
        if lr_planarity(&adj_of(&alive)).is_some() {
            alive[i] = true;
        }
    }
    (0..edges.len()).filter(|&i| alive[i]).collect()
}

// --- Python entry points -------------------------------------------------------------------

/// Clockwise neighbor lists of a planar embedding of the simple graph on `0..n` with `edges`,
/// or `None` if the graph is not planar. Self-loops and repeated edges are ignored.
#[pyfunction]
pub fn planar_embedding(n: usize, edges: Vec<(usize, usize)>) -> PyResult<Option<Vec<Vec<usize>>>> {
    let adj = simple_adjacency(n, &edges)?;
    Ok(lr_planarity(&adj).map(|emb| (0..n).map(|v| emb.neighbors_cw(v)).collect()))
}

/// Indices into `edges` of a Kuratowski subgraph (a subdivision of K5 or K3,3), or `None` if
/// the graph is planar. `edges` must be simple: no self-loops and no repeated pairs.
#[pyfunction]
pub fn planarity_counterexample(
    n: usize,
    edges: Vec<(usize, usize)>,
) -> PyResult<Option<Vec<usize>>> {
    let adj = simple_adjacency(n, &edges)?;
    if adj.iter().map(Vec::len).sum::<usize>() != 2 * edges.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges must be simple",
        ));
    }
    if lr_planarity(&adj).is_some() {
        return Ok(None);
    }
    Ok(Some(kuratowski_edges(n, &edges)))
}

/// The faces of a planar embedding of the simple graph on `0..n` with `edges` (the embedding
/// `planar_embedding` returns), or `None` if the graph is not planar. Each face is the walk of
/// nodes around it, starting from the half-edge first met when visiting nodes in order and
/// their neighbors clockwise; every half-edge lies on exactly one face. Walks repeat nodes
/// around bridges and cut vertices; isolated nodes are on no face, and each connected component
/// contributes its own outer face.
#[pyfunction]
pub fn planar_faces(n: usize, edges: Vec<(usize, usize)>) -> PyResult<Option<Vec<Vec<usize>>>> {
    let adj = simple_adjacency(n, &edges)?;
    let Some(emb) = lr_planarity(&adj) else {
        return Ok(None);
    };
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut faces = Vec::new();
    for v in 0..n {
        for w in emb.neighbors_cw(v) {
            if seen.contains(&(v, w)) {
                continue;
            }
            let mut face = Vec::new();
            let (mut a, mut b) = (v, w);
            while seen.insert((a, b)) {
                face.push(a);
                (a, b) = emb.next_face_half_edge(a, b);
            }
            faces.push(face);
        }
    }
    Ok(Some(faces))
}

/// Integer grid positions drawing the simple graph on `0..n` without edge crossings, or `None`
/// if the graph is not planar.
#[pyfunction]
pub fn planar_layout(n: usize, edges: Vec<(usize, usize)>) -> PyResult<Option<Vec<(i64, i64)>>> {
    let adj = simple_adjacency(n, &edges)?;
    Ok(lr_planarity(&adj).map(|emb| embedding_to_pos(&emb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adj(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        simple_adjacency(n, edges).unwrap()
    }

    fn complete(n: usize) -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
            .collect()
    }

    fn segments_cross(a: (i64, i64), b: (i64, i64), c: (i64, i64), d: (i64, i64)) -> bool {
        let orient = |p: (i64, i64), q: (i64, i64), r: (i64, i64)| {
            ((q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)).signum()
        };
        let (o1, o2, o3, o4) = (
            orient(a, b, c),
            orient(a, b, d),
            orient(c, d, a),
            orient(c, d, b),
        );
        o1 * o2 < 0 && o3 * o4 < 0
    }

    #[test]
    fn complete_graphs() {
        assert!(lr_planarity(&adj(4, &complete(4))).is_some());
        assert!(lr_planarity(&adj(5, &complete(5))).is_none());
    }

    #[test]
    fn k33_and_its_counterexample() {
        let e: Vec<_> = (0..3).flat_map(|u| (3..6).map(move |v| (u, v))).collect();
        assert!(lr_planarity(&adj(6, &e)).is_none());
        assert_eq!(kuratowski_edges(6, &e).len(), 9);
    }

    #[test]
    fn drawing_has_no_crossings() {
        // K4 plus a triangulated ring around it
        let e = [
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (1, 3),
            (2, 3),
            (0, 4),
            (1, 4),
            (4, 5),
            (5, 1),
            (5, 2),
        ];
        let emb = lr_planarity(&adj(6, &e)).unwrap();
        let pos = embedding_to_pos(&emb);
        let distinct: HashSet<_> = pos.iter().collect();
        assert_eq!(distinct.len(), 6);
        for (i, &(a, b)) in e.iter().enumerate() {
            for &(c, d) in &e[i + 1..] {
                if [a, b].contains(&c) || [a, b].contains(&d) {
                    continue;
                }
                assert!(!segments_cross(pos[a], pos[b], pos[c], pos[d]));
            }
        }
    }

    #[test]
    fn embedding_lists_every_neighbor() {
        let e = complete(4);
        let emb = lr_planarity(&adj(4, &e)).unwrap();
        for v in 0..4 {
            let mut nb = emb.neighbors_cw(v);
            nb.sort();
            assert_eq!(nb, (0..4).filter(|&w| w != v).collect::<Vec<_>>());
        }
    }
}
