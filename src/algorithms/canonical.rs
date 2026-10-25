//! Canonical labeling by individualization-refinement (McKay and Piperno, 2014).
//!
//! Nodes are numbered 0..n so that isomorphic graphs, numbered this way, become identical.
//! The search tree starts from the coarsest equitable partition refining the node colors;
//! each tree node individualizes one node of a non-singleton cell and refines again, and the
//! leaves are numberings. The canonical numbering is the leaf with the largest (path
//! invariant, certificate) pair, where the certificate is the graph rewritten under the
//! numbering. Automorphisms found between equivalent leaves prune the search, as in nauty.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Edge kinds in adjacency signatures: an undirected edge or out-arc, an in-arc, a self-loop.
const OUT: u8 = 0;
const IN: u8 = 1;
const LOOP: u8 = 2;

/// A node's sorted (neighbor color, kind, label) list.
type Signature = Vec<(u32, u8, u32)>;

struct Structure {
    n: usize,
    color: Vec<i64>,
    /// (neighbor, kind, label); the label ranks the multiset of edge colors on the pair
    adj: Vec<Vec<(usize, u8, u32)>>,
    /// one entry per ordered pair (directed) or unordered pair (undirected) with edges
    pairs: Vec<(usize, usize, u32)>,
    directed: bool,
}

impl Structure {
    fn new(graph: &Graph, ids: &[u64], node_colors: &FxHashMap<u64, i64>,
           edge_colors: &FxHashMap<u64, i64>) -> Self {
        let n = ids.len();
        let directed = graph.is_directed();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut multi: FxHashMap<(usize, usize), Vec<i64>> = FxHashMap::default();
        for e in graph.get_edges_internal() {
            let (mut a, mut b) = (idx[&e.source], idx[&e.target]);
            if !directed && a > b {
                std::mem::swap(&mut a, &mut b);
            }
            multi.entry((a, b)).or_default().push(edge_colors.get(&e.id).copied().unwrap_or(0));
        }
        for colors in multi.values_mut() {
            colors.sort_unstable();
        }
        // Labels rank the distinct color multisets by value, so they don't depend on IDs
        let mut distinct: Vec<&Vec<i64>> = multi.values().collect();
        distinct.sort_unstable();
        distinct.dedup();
        let rank: FxHashMap<&Vec<i64>, u32> =
            distinct.into_iter().enumerate().map(|(i, c)| (c, i as u32)).collect();
        let mut pairs: Vec<(usize, usize, u32)> =
            multi.iter().map(|(&(a, b), c)| (a, b, rank[c])).collect();
        pairs.sort_unstable();
        let mut adj = vec![Vec::new(); n];
        for &(a, b, l) in &pairs {
            if a == b {
                adj[a].push((a, LOOP, l));
            } else if directed {
                adj[a].push((b, OUT, l));
                adj[b].push((a, IN, l));
            } else {
                adj[a].push((b, OUT, l));
                adj[b].push((a, OUT, l));
            }
        }
        let color = ids.iter().map(|id| node_colors.get(id).copied().unwrap_or(0)).collect();
        Structure { n, color, adj, pairs, directed }
    }

    fn signature(&self, u: usize, col: &[u32]) -> Signature {
        let mut sig: Signature =
            self.adj[u].iter().map(|&(w, k, l)| (col[w], k, l)).collect();
        sig.sort_unstable();
        sig
    }

    /// The graph rewritten under the numbering `lab`: node colors by number, then the pairs.
    fn certificate(&self, lab: &[u32]) -> (Vec<i64>, Vec<(u32, u32, u32)>) {
        let mut colors = vec![0; self.n];
        for (u, &l) in lab.iter().enumerate() {
            colors[l as usize] = self.color[u];
        }
        let mut pairs: Vec<(u32, u32, u32)> = self
            .pairs
            .iter()
            .map(|&(a, b, l)| {
                let (x, y) = (lab[a], lab[b]);
                if self.directed || x <= y { (x, y, l) } else { (y, x, l) }
            })
            .collect();
        pairs.sort_unstable();
        (colors, pairs)
    }
}

/// An ordered partition: each node's color is the offset of its cell in `order`.
#[derive(Clone)]
struct Partition {
    col: Vec<u32>,
    order: Vec<usize>,
    /// `end[s]` is the end of the cell starting at offset `s`
    end: Vec<u32>,
}

impl Partition {
    fn new(s: &Structure) -> Self {
        let mut order: Vec<usize> = (0..s.n).collect();
        order.sort_by_key(|&u| s.color[u]);
        let mut col = vec![0; s.n];
        let mut end = vec![0; s.n];
        let mut start = 0;
        for i in 0..s.n {
            if i > 0 && s.color[order[i]] != s.color[order[i - 1]] {
                end[start] = i as u32;
                start = i;
            }
            col[order[i]] = start as u32;
        }
        if s.n > 0 {
            end[start] = s.n as u32;
        }
        Partition { col, order, end }
    }

    /// Refine to the coarsest equitable partition, given the nodes whose color changed. A
    /// cell is split by its members' sorted (neighbor color, kind, label) lists, the parts
    /// ordered by those lists, so the result depends only on the colors, not on node IDs.
    /// Every split is hashed into `trace`, in cell order, which makes the trace an
    /// isomorphism invariant of the starting partition.
    fn refine(&mut self, s: &Structure, changed: Vec<usize>, trace: &mut FxHasher) {
        let mut changed = changed;
        let mut dirty_mark = vec![false; s.n];
        loop {
            let mut dirty = Vec::new();
            for &u in &changed {
                for &(w, _, _) in &s.adj[u] {
                    let c = self.col[w] as usize;
                    if !dirty_mark[c] && self.end[c] as usize - c > 1 {
                        dirty_mark[c] = true;
                        dirty.push(c);
                    }
                }
            }
            if dirty.is_empty() {
                return;
            }
            dirty.sort_unstable();
            let mut updates = Vec::new();
            for c in dirty {
                dirty_mark[c] = false;
                let b = self.end[c] as usize;
                let mut keyed: Vec<(Signature, usize)> =
                    self.order[c..b].iter().map(|&u| (s.signature(u, &self.col), u)).collect();
                keyed.sort_unstable();
                if keyed[0].0 == keyed[keyed.len() - 1].0 {
                    continue;
                }
                c.hash(trace);
                let mut start = c;
                for (i, (_, u)) in keyed.iter().enumerate() {
                    if i > 0 && keyed[i].0 != keyed[i - 1].0 {
                        self.end[start] = (c + i) as u32;
                        (c + i, &keyed[i - 1].0).hash(trace);
                        start = c + i;
                    }
                    self.order[c + i] = *u;
                    if start != c {
                        updates.push((*u, start as u32));
                    }
                }
                self.end[start] = b as u32;
            }
            changed = updates.iter().map(|&(u, _)| u).collect();
            for (u, c) in updates {
                self.col[u] = c;
            }
        }
    }

    /// Give `v` a cell of its own, at the front of its current cell.
    fn individualize(&mut self, v: usize) -> Vec<usize> {
        let a = self.col[v] as usize;
        let b = self.end[a] as usize;
        let at = self.order[a..b].iter().position(|&u| u == v).unwrap() + a;
        self.order.swap(a, at);
        self.end[a] = (a + 1) as u32;
        self.end[a + 1] = b as u32;
        let rest = self.order[a + 1..b].to_vec();
        for &u in &rest {
            self.col[u] = (a + 1) as u32;
        }
        rest
    }

    fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut s = 0;
        std::iter::from_fn(move || {
            (s < self.order.len()).then(|| {
                let c = (s, self.end[s] as usize);
                s = c.1;
                c
            })
        })
    }

    /// Whether every permutation within cells is an automorphism: each member of a cell sees
    /// all or none of every other cell (all or none of its own), with one label per kind.
    /// Every numbering below such a tree node is then equivalent, so it acts as a leaf.
    fn is_uniform(&self, s: &Structure) -> bool {
        for (a, b) in self.cells() {
            if b - a == 1 {
                continue;
            }
            let mut seen: FxHashMap<(u32, u8), (u32, usize)> = FxHashMap::default();
            for &(w, k, l) in &s.adj[self.order[a]] {
                if k == LOOP {
                    continue;
                }
                let entry = seen.entry((self.col[w], k)).or_insert((l, 0));
                if entry.0 != l {
                    return false;
                }
                entry.1 += 1;
            }
            for (&(c, _), &(_, count)) in &seen {
                let c = c as usize;
                let size = self.end[c] as usize - c;
                if count != if c == a { size - 1 } else { size } {
                    return false;
                }
            }
        }
        true
    }

    /// The numbering given by positions in `order` (a leaf when the partition is uniform).
    fn numbering(&self) -> Vec<u32> {
        let mut lab = vec![0; self.order.len()];
        for (i, &u) in self.order.iter().enumerate() {
            lab[u] = i as u32;
        }
        lab
    }

    /// Whether the members of cell `a..b` are twins: the same neighbors (and labels) outside
    /// the cell, and all or none of each other. Any permutation of them is an automorphism.
    fn twins(&self, s: &Structure, a: usize, b: usize) -> bool {
        let outside = |u: usize| -> Vec<(usize, u8, u32)> {
            let mut v: Vec<(usize, u8, u32)> = s.adj[u]
                .iter()
                .filter(|&&(w, k, _)| k == LOOP || (self.col[w] as usize) != a)
                .map(|&(w, k, l)| if k == LOOP { (usize::MAX, k, l) } else { (w, k, l) })
                .collect();
            v.sort_unstable();
            v
        };
        let first = outside(self.order[a]);
        // equitable: inside degrees (and labels) already agree, so all-or-none is one check
        let inside = s.adj[self.order[a]].len() - first.len();
        let size = b - a;
        let kinds = if s.directed { 2 } else { 1 };
        (inside == 0 || inside == kinds * (size - 1) && {
            let mut labels: Vec<u32> = s.adj[self.order[a]]
                .iter()
                .filter(|&&(w, k, _)| k != LOOP && self.col[w] as usize == a)
                .map(|&(_, _, l)| l)
                .collect();
            labels.sort_unstable();
            labels.dedup();
            labels.len() == 1
        }) && self.order[a + 1..b].iter().all(|&u| outside(u) == first)
    }

    /// The target cell: the first smallest non-singleton cell.
    fn target(&self) -> (usize, usize) {
        self.cells()
            .filter(|&(a, b)| b - a > 1)
            .min_by_key(|&(a, b)| (b - a, a))
            .unwrap()
    }
}

type Certificate = (Vec<i64>, Vec<(u32, u32, u32)>);

struct Leaf {
    path: Vec<(u64, u32)>,
    cert: Certificate,
    lab: Vec<u32>,
    inv: Vec<usize>,
    seq: Vec<usize>,
}

struct Search<'a> {
    s: &'a Structure,
    first: Option<Leaf>,
    best: Option<Leaf>,
    gens: Vec<Vec<usize>>,
    path: Vec<(u64, u32)>,
    seq: Vec<usize>,
}

impl Search<'_> {
    /// Explore the tree node with partition `p`; return the depth to unwind to when an
    /// automorphism shows the rest of an ancestor's subtree to be equivalent to explored parts.
    fn visit(&mut self, p: &Partition, inv: u64) -> Option<usize> {
        let depth = self.seq.len();
        let uniform = p.is_uniform(self.s);
        let target = if uniform { None } else { Some(p.target()) };
        self.path.push((inv, target.map_or(u32::MAX, |t| t.0 as u32)));
        // Prune when the path is already worse than the best leaf's
        if let Some(best) = &self.best {
            let k = self.path.len().min(best.path.len());
            if self.path[..k] < best.path[..k] {
                self.path.pop();
                return None;
            }
        }
        let result = match target {
            None => self.leaf(p),
            Some((a, b)) => self.branch(p, a, b, depth),
        };
        self.path.pop();
        result
    }

    fn leaf(&mut self, p: &Partition) -> Option<usize> {
        let lab = p.numbering();
        let cert = self.s.certificate(&lab);
        let mut inv = vec![0; lab.len()];
        for (u, &l) in lab.iter().enumerate() {
            inv[l as usize] = u;
        }
        let leaf = Leaf { path: self.path.clone(), cert, lab, inv, seq: self.seq.clone() };
        for stored in [&self.first, &self.best].into_iter().flatten() {
            if stored.path == leaf.path && stored.cert == leaf.cert {
                // Equal leaves differ by an automorphism, which fixes their common prefix
                let gamma: Vec<usize> = leaf.lab.iter().map(|&l| stored.inv[l as usize]).collect();
                let common = leaf.seq.iter().zip(&stored.seq).take_while(|(x, y)| x == y).count();
                if gamma.iter().enumerate().any(|(u, &g)| u != g) {
                    self.gens.push(gamma);
                }
                return Some(common);
            }
        }
        let better = match &self.best {
            None => true,
            Some(best) => match leaf.path.cmp(&best.path) {
                Ordering::Equal => leaf.cert > best.cert,
                o => o == Ordering::Greater,
            },
        };
        if self.first.is_none() {
            self.first = Some(Leaf {
                path: leaf.path.clone(),
                cert: leaf.cert.clone(),
                lab: leaf.lab.clone(),
                inv: leaf.inv.clone(),
                seq: leaf.seq.clone(),
            });
        }
        if better {
            self.best = Some(leaf);
        }
        None
    }

    fn branch(&mut self, p: &Partition, a: usize, b: usize, depth: usize) -> Option<usize> {
        let mut members = p.order[a..b].to_vec();
        members.sort_unstable();
        if p.twins(self.s, a, b) {
            members.truncate(1);
        }
        let mut explored: Vec<usize> = Vec::new();
        let mut orbit: Vec<usize> = Vec::new();
        let mut gens_seen = usize::MAX;
        for v in members {
            if !explored.is_empty() {
                if gens_seen != self.gens.len() {
                    orbit = self.orbits();
                    gens_seen = self.gens.len();
                }
                if explored.iter().any(|&x| find(&mut orbit, x) == find(&mut orbit, v)) {
                    continue;
                }
            }
            let mut child = p.clone();
            let changed = child.individualize(v);
            let mut trace = FxHasher::default();
            child.refine(self.s, changed, &mut trace);
            self.seq.push(v);
            let jump = self.visit(&child, trace.finish());
            self.seq.pop();
            explored.push(v);
            if let Some(k) = jump {
                if k < depth {
                    return Some(k);
                }
            }
        }
        None
    }

    /// Orbits (as a union-find forest) of the automorphisms found so far that fix the current
    /// prefix pointwise.
    fn orbits(&self) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..self.s.n).collect();
        for g in &self.gens {
            if self.seq.iter().all(|&v| g[v] == v) {
                for (u, &w) in g.iter().enumerate() {
                    let (x, y) = (find(&mut parent, u), find(&mut parent, w));
                    if x != y {
                        parent[x.max(y)] = x.min(y);
                    }
                }
            }
        }
        parent
    }
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// A canonical numbering of the nodes, as `(node ID, number)` pairs in ID order. Node and
/// edge colors are integers ranked by the caller from attribute values (empty to ignore
/// attributes); edge direction, parallel edges and self-loops all count.
#[pyfunction]
pub fn canonical_labeling(
    graph: &Graph,
    node_colors: FxHashMap<u64, i64>,
    edge_colors: FxHashMap<u64, i64>,
) -> Vec<(u64, u32)> {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    if ids.is_empty() {
        return Vec::new();
    }
    let s = Structure::new(graph, &ids, &node_colors, &edge_colors);
    let mut root = Partition::new(&s);
    let mut trace = FxHasher::default();
    root.refine(&s, (0..s.n).collect(), &mut trace);
    let mut search =
        Search { s: &s, first: None, best: None, gens: Vec::new(), path: Vec::new(), seq: Vec::new() };
    search.visit(&root, trace.finish());
    let best = search.best.expect("the search reaches a leaf");
    ids.into_iter().zip(best.lab).collect()
}
