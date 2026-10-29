//! Modular decomposition of undirected graphs.
//!
//! A *module* is a set of nodes that every outside node sees uniformly: adjacent to all of
//! them or to none. The strong modules (those overlapping no other module) form a tree, the
//! modular decomposition, whose internal nodes are parallel (the module is disconnected),
//! series (its complement is disconnected) or prime (neither).

use crate::graph::Graph;
use pyo3::prelude::*;
use std::collections::VecDeque;

struct Adjacency {
    n: usize,
    words: usize,
    bits: Vec<u64>,
    lists: Vec<Vec<usize>>,
}

impl Adjacency {
    fn new(graph: &Graph, ids: &[u64]) -> Self {
        let n = ids.len();
        let words = (n + 63) / 64;
        let idx: rustc_hash::FxHashMap<u64, usize> =
            ids.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut adj = Adjacency {
            n,
            words,
            bits: vec![0; n * words],
            lists: vec![Vec::new(); n],
        };
        for e in graph.get_edges_internal() {
            let (a, b) = (idx[&e.source], idx[&e.target]);
            if a != b && !adj.has(a, b) {
                adj.bits[a * words + b / 64] |= 1 << (b % 64);
                adj.bits[b * words + a / 64] |= 1 << (a % 64);
                adj.lists[a].push(b);
                adj.lists[b].push(a);
            }
        }
        adj
    }

    fn has(&self, a: usize, b: usize) -> bool {
        self.bits[a * self.words + b / 64] >> (b % 64) & 1 == 1
    }
}

/// Components of `set` in the graph (`complement = false`) or in its complement, each sorted,
/// ordered by smallest member. `mark` is scratch space: entries for `set` must equal `stamp`.
fn components(
    adj: &Adjacency,
    set: &[usize],
    complement: bool,
    mark: &mut [usize],
    stamp: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if complement {
        // BFS in the complement: scan the not-yet-reached nodes for non-neighbours.
        let mut left: Vec<usize> = set.to_vec();
        while let Some(start) = left.first().copied() {
            left.swap_remove(0);
            let mut comp = vec![start];
            let mut queue = VecDeque::from([start]);
            while let Some(v) = queue.pop_front() {
                let mut keep = Vec::with_capacity(left.len());
                for &u in &left {
                    if adj.has(v, u) {
                        keep.push(u);
                    } else {
                        comp.push(u);
                        queue.push_back(u);
                    }
                }
                left = keep;
            }
            left.sort_unstable();
            comp.sort_unstable();
            out.push(comp);
        }
    } else {
        let done = stamp + 1;
        for &s in set {
            if mark[s] != stamp {
                continue;
            }
            mark[s] = done;
            let mut comp = vec![s];
            let mut i = 0;
            while i < comp.len() {
                for &u in &adj.lists[comp[i]] {
                    if mark[u] == stamp {
                        mark[u] = done;
                        comp.push(u);
                    }
                }
                i += 1;
            }
            comp.sort_unstable();
            out.push(comp);
        }
    }
    out.sort_by_key(|c| c[0]);
    out
}

/// Children of a prime module `set` (sorted, with the graph and its complement both
/// connected on it): its maximal proper modules. `part_of` is scratch space holding
/// `usize::MAX` outside calls.
fn prime_children(adj: &Adjacency, set: &[usize], part_of: &mut [usize]) -> Vec<Vec<usize>> {
    let v = set[0];
    // Partition refinement: the coarsest partition of set - {v}, starting from v's
    // neighbours and non-neighbours, in which no node splits a part other than its own.
    // Its parts are the maximal modules not containing v. Parts are ranges of `order`, so
    // a pivot w splits them in O(deg w) by moving its neighbours to the front of each.
    let mut order: Vec<usize> = set[1..].to_vec();
    order.sort_by_key(|&u| !adj.has(v, u));
    let near = order.iter().take_while(|&&u| adj.has(v, u)).count();
    let mut ranges: Vec<(usize, usize)> = [(0, near), (near, order.len())]
        .into_iter()
        .filter(|&(a, b)| a < b)
        .collect();
    let mut pos = vec![0; adj.n];
    for (i, &u) in order.iter().enumerate() {
        pos[u] = i;
    }
    for (p, &(a, b)) in ranges.iter().enumerate() {
        for &u in &order[a..b] {
            part_of[u] = p;
        }
    }
    let mut moved = vec![0usize; ranges.len()];
    let mut touched: Vec<usize> = Vec::new();
    let mut queued = vec![false; adj.n];
    let mut queue: VecDeque<usize> = order.iter().copied().collect();
    for &u in &order {
        queued[u] = true;
    }
    while let Some(w) = queue.pop_front() {
        queued[w] = false;
        let own = part_of[w];
        for &u in &adj.lists[w] {
            let p = part_of[u];
            if p == usize::MAX || p == own {
                continue;
            }
            if moved[p] == 0 {
                touched.push(p);
            }
            let slot = ranges[p].0 + moved[p];
            let other = order[slot];
            order.swap(slot, pos[u]);
            pos[other] = pos[u];
            pos[u] = slot;
            moved[p] += 1;
        }
        for p in touched.drain(..) {
            let (a, b) = ranges[p];
            let cut = a + moved[p];
            moved[p] = 0;
            if cut == b {
                continue;
            }
            let q = ranges.len();
            ranges[p] = (a, cut);
            ranges.push((cut, b));
            moved.push(0);
            for &u in &order[cut..b] {
                part_of[u] = q;
            }
            for &u in &order[a..b] {
                if !queued[u] {
                    queued[u] = true;
                    queue.push_back(u);
                }
            }
        }
    }
    for &u in &order {
        part_of[u] = usize::MAX;
    }
    // Forcing digraph on the parts: X -> Y when Y's nodes tell X apart from v, so Y lies in
    // the smallest module holding v and X. Parts outside v's maximal proper module reach
    // every part, while parts inside reach only parts inside, so the outside parts form the
    // unique source component: everything that reaches the last node to finish a DFS.
    let k = ranges.len();
    let rep: Vec<usize> = ranges.iter().map(|&(a, _)| order[a]).collect();
    let forces = |x: usize, y: usize| x != y && adj.has(rep[y], rep[x]) != adj.has(rep[y], v);
    let mut seen = vec![false; k];
    let mut last = 0;
    for root in 0..k {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root, 0)];
        while let Some(&mut (x, ref mut y)) = stack.last_mut() {
            while *y < k && (seen[*y] || !forces(x, *y)) {
                *y += 1;
            }
            if *y == k {
                last = x;
                stack.pop();
            } else {
                let next = *y;
                seen[next] = true;
                stack.push((next, 0));
            }
        }
    }
    let mut outside = vec![false; k];
    outside[last] = true;
    let mut queue = vec![last];
    while let Some(y) = queue.pop() {
        for (x, out) in outside.iter_mut().enumerate() {
            if !*out && forces(x, y) {
                *out = true;
                queue.push(x);
            }
        }
    }
    let mut with_v = vec![v];
    let mut children = Vec::new();
    for (p, &(a, b)) in ranges.iter().enumerate() {
        if outside[p] {
            let mut part = order[a..b].to_vec();
            part.sort_unstable();
            children.push(part);
        } else {
            with_v.extend(&order[a..b]);
        }
    }
    with_v.sort_unstable();
    children.push(with_v);
    children.sort_by_key(|c| c[0]);
    children
}

/// The modular decomposition tree in preorder: `(kind, parent, members)` per tree node, with
/// `kind` one of "parallel", "series", "prime" or "leaf", `parent` the index of the parent
/// entry (-1 for the root) and `members` the sorted node IDs of the module. Children follow
/// their smallest member.
#[pyfunction]
pub fn modular_decomposition(graph: &Graph) -> PyResult<Vec<(String, i64, Vec<u64>)>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "modular_decomposition() requires an undirected graph",
        ));
    }
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let adj = Adjacency::new(graph, &ids);
    let mut out = Vec::new();
    if ids.is_empty() {
        return Ok(out);
    }
    let mut mark = vec![usize::MAX; adj.n];
    let mut part_of = vec![usize::MAX; adj.n];
    let mut stack: Vec<(Vec<usize>, i64)> = vec![((0..adj.n).collect(), -1)];
    let mut stamp = 0;
    while let Some((set, parent)) = stack.pop() {
        let me = out.len() as i64;
        let members = set.iter().map(|&i| ids[i]).collect();
        if set.len() == 1 {
            out.push(("leaf".to_string(), parent, members));
            continue;
        }
        stamp += 2;
        for &u in &set {
            mark[u] = stamp;
        }
        let mut kind = "parallel";
        let mut children = components(&adj, &set, false, &mut mark, stamp);
        if children.len() == 1 {
            kind = "series";
            children = components(&adj, &set, true, &mut mark, stamp);
            if children.len() == 1 {
                kind = "prime";
                children = prime_children(&adj, &set, &mut part_of);
            }
        }
        out.push((kind.to_string(), parent, members));
        stack.extend(children.into_iter().rev().map(|c| (c, me)));
    }
    Ok(out)
}
