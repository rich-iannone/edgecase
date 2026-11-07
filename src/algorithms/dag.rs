//! Directed-acyclic-graph queries: ancestors/descendants, topological generations, longest
//! paths, condensation (strongly connected components as a DAG), and lowest common ancestors.

use crate::algorithms::path_queries::IndexedGraph;
use crate::graph::Graph;
use pyo3::prelude::*;
use std::collections::VecDeque;

fn require_directed(graph: &Graph, what: &str) -> PyResult<()> {
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} requires a directed graph",
            what
        )));
    }
    Ok(())
}

fn not_a_dag(what: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{} requires a directed acyclic graph (DAG), but the graph contains a cycle",
        what
    ))
}

/// Nodes reachable from `start` along `adj` (excluding `start` itself), as sorted node IDs.
fn reach(g: &IndexedGraph, adj: &[Vec<(usize, f64)>], start: usize) -> Vec<bool> {
    let mut seen = vec![false; g.len()];
    let mut queue = VecDeque::from([start]);
    seen[start] = true;
    while let Some(v) = queue.pop_front() {
        for &(w, _) in &adj[v] {
            if !seen[w] {
                seen[w] = true;
                queue.push_back(w);
            }
        }
    }
    seen[start] = false;
    seen
}

fn ids_where(g: &IndexedGraph, mask: &[bool]) -> Vec<u64> {
    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| g.ids[i])
        .collect()
}

/// All nodes that can reach `node` (sorted IDs). On an undirected graph: the rest of its
/// component.
#[pyfunction]
pub fn ancestors(graph: &Graph, node: u64) -> PyResult<Vec<u64>> {
    let g = IndexedGraph::new(graph, false);
    let s = g.index_of(node)?;
    Ok(ids_where(&g, &reach(&g, &g.inn, s)))
}

/// All nodes reachable from `node` (sorted IDs). On an undirected graph: the rest of its
/// component.
#[pyfunction]
pub fn descendants(graph: &Graph, node: u64) -> PyResult<Vec<u64>> {
    let g = IndexedGraph::new(graph, false);
    let s = g.index_of(node)?;
    Ok(ids_where(&g, &reach(&g, &g.out, s)))
}

/// Kahn layering: generation 0 holds the sources, generation k the nodes whose longest path
/// from a source has k edges. Returns None if the graph has a cycle.
fn generations(g: &IndexedGraph) -> Option<Vec<Vec<usize>>> {
    let n = g.len();
    let mut indeg: Vec<usize> = g.inn.iter().map(|p| p.len()).collect();
    let mut current: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut gens = Vec::new();
    let mut seen = 0;
    while !current.is_empty() {
        seen += current.len();
        let mut next = Vec::new();
        for &v in &current {
            for &(w, _) in &g.out[v] {
                indeg[w] -= 1;
                if indeg[w] == 0 {
                    next.push(w);
                }
            }
        }
        next.sort_unstable();
        gens.push(std::mem::take(&mut current));
        current = next;
    }
    (seen == n).then_some(gens)
}

fn has_self_loop(graph: &Graph) -> bool {
    graph
        .get_edges_internal()
        .iter()
        .any(|e| e.source == e.target)
}

/// Topological generations of a DAG, each a sorted list of node IDs.
#[pyfunction]
pub fn topological_generations(graph: &Graph) -> PyResult<Vec<Vec<u64>>> {
    require_directed(graph, "topological_generations()")?;
    if has_self_loop(graph) {
        return Err(not_a_dag("topological_generations()"));
    }
    let g = IndexedGraph::new(graph, false);
    let gens = generations(&g).ok_or_else(|| not_a_dag("topological_generations()"))?;
    Ok(gens
        .into_iter()
        .map(|layer| layer.into_iter().map(|i| g.ids[i]).collect())
        .collect())
}

/// Width of the partial order a DAG defines: the size of its largest antichain (a set of nodes
/// none of which can reach another). By Dilworth's theorem this equals the fewest chains that
/// cover the nodes, which is `n` minus a maximum matching between the two copies of the nodes
/// in the transitive closure (Fulkerson 1956).
#[pyfunction]
pub fn antichain_width(graph: &Graph) -> PyResult<usize> {
    require_directed(graph, "antichain_width()")?;
    if has_self_loop(graph) {
        return Err(not_a_dag("antichain_width()"));
    }
    let g = IndexedGraph::new(graph, false);
    let n = g.len();
    let gens = generations(&g).ok_or_else(|| not_a_dag("antichain_width()"))?;
    // Transitive closure as bitsets, filled in reverse topological order
    let words = n / 64 + 1;
    let mut below: Vec<Vec<u64>> = vec![vec![0; words]; n];
    for &v in gens.iter().flatten().rev() {
        let mut row = vec![0u64; words];
        for &(w, _) in &g.out[v] {
            row[w / 64] |= 1 << (w % 64);
            for (r, b) in row.iter_mut().zip(&below[w]) {
                *r |= b;
            }
        }
        below[v] = row;
    }
    // Bipartite graph: left copy 0..n, right copy n..2n, u -> v for every comparable pair
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 2 * n];
    for (u, row) in below.iter().enumerate() {
        for (k, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let v = k * 64 + bits.trailing_zeros() as usize;
                adj[u].push((n + v, 0));
                bits &= bits - 1;
            }
        }
    }
    let side: Vec<u8> = (0..2 * n).map(|i| u8::from(i >= n)).collect();
    let mate = crate::algorithms::matching_cover::hopcroft_karp(2 * n, &adj, &side);
    let matched = mate[..n].iter().filter(|m| m.is_some()).count();
    Ok(n - matched)
}

/// Longest path in a DAG and its length.
///
/// Uses edge weights on weighted graphs (a missing weight counts as 1.0; for parallel edges
/// the heaviest counts), hop counts otherwise. Ties prefer the path found from the lower node
/// ID. An empty graph gives an empty path of length 0.
#[pyfunction]
pub fn dag_longest_path(graph: &Graph) -> PyResult<(Vec<u64>, f64)> {
    require_directed(graph, "dag_longest_path()")?;
    if has_self_loop(graph) {
        return Err(not_a_dag("dag_longest_path()"));
    }
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let n = g.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let gens = generations(&g).ok_or_else(|| not_a_dag("dag_longest_path()"))?;
    let mut best = vec![0.0_f64; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    for v in gens.into_iter().flatten() {
        // best[v] is final once every predecessor has been processed (topological order)
        // Extend the best incoming path if that is at least as long as starting fresh at v
        // (matters only for negative weights), as NetworkX does
        let mut top: Option<(f64, usize)> = None;
        for &(u, w) in &g.inn[v] {
            let cand = best[u] + w;
            let better = match top {
                None => true,
                Some((c, pu)) => cand > c || (cand == c && u < pu),
            };
            if better {
                top = Some((cand, u));
            }
        }
        if let Some((c, u)) = top {
            if c >= 0.0 {
                best[v] = c;
                prev[v] = Some(u);
            }
        }
    }
    let end = (0..n)
        .max_by(|&a, &b| best[a].total_cmp(&best[b]).then(b.cmp(&a)))
        .unwrap();
    let mut path = vec![g.ids[end]];
    let mut cur = end;
    while let Some(p) = prev[cur] {
        path.push(g.ids[p]);
        cur = p;
    }
    path.reverse();
    Ok((path, best[end]))
}

/// Strongly connected components (iterative Tarjan), returned as component index per node.
fn scc(g: &IndexedGraph) -> (Vec<usize>, usize) {
    let n = g.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp = vec![usize::MAX; n];
    let mut n_comp = 0;
    let mut counter = 0;
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = counter;
        low[root] = counter;
        counter += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut k)) = call.last_mut() {
            if *k < g.out[v].len() {
                let w = g.out[v][*k].0;
                *k += 1;
                if index[w] == usize::MAX {
                    index[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == index[v] {
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        comp[w] = n_comp;
                        if w == v {
                            break;
                        }
                    }
                    n_comp += 1;
                }
            }
        }
    }
    (comp, n_comp)
}

/// Condensation: the DAG of strongly connected components.
///
/// Returns `(components, edges)`: components as sorted node-ID lists in topological order
/// (ties by smallest member ID), and edges as `(from, to)` component indices, deduplicated.
#[pyfunction]
pub fn condensation(graph: &Graph) -> PyResult<(Vec<Vec<u64>>, Vec<(usize, usize)>)> {
    require_directed(graph, "condensation()")?;
    let g = IndexedGraph::new(graph, false);
    let (comp, k) = scc(&g);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (v, &c) in comp.iter().enumerate() {
        members[c].push(v);
    }
    // Component-level DAG
    let mut cadj: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut cindeg = vec![0usize; k];
    for v in 0..g.len() {
        for &(w, _) in &g.out[v] {
            let (a, b) = (comp[v], comp[w]);
            if a != b && !cadj[a].contains(&b) {
                cadj[a].push(b);
                cindeg[b] += 1;
            }
        }
    }
    // Topological order of components, smallest-member first among ready ones
    let key = |c: usize| members[c][0];
    let mut ready: std::collections::BTreeSet<(usize, usize)> = (0..k)
        .filter(|&c| cindeg[c] == 0)
        .map(|c| (key(c), c))
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(k);
    while let Some(&first) = ready.iter().next() {
        ready.remove(&first);
        let c = first.1;
        order.push(c);
        for &d in &cadj[c] {
            cindeg[d] -= 1;
            if cindeg[d] == 0 {
                ready.insert((key(d), d));
            }
        }
    }
    let mut pos = vec![0usize; k];
    for (i, &c) in order.iter().enumerate() {
        pos[c] = i;
    }
    let components: Vec<Vec<u64>> = order
        .iter()
        .map(|&c| members[c].iter().map(|&v| g.ids[v]).collect())
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for a in 0..k {
        for &b in &cadj[a] {
            edges.push((pos[a], pos[b]));
        }
    }
    edges.sort_unstable();
    Ok((components, edges))
}

/// Lowest common ancestor of `u` and `v` in a DAG, or None if they share no ancestor.
///
/// A node counts as its own ancestor. Among common ancestors, the lowest are those with no
/// descendant that is also a common ancestor; if several qualify, the deepest (longest path
/// from a source) wins, then the smallest node ID.
#[pyfunction]
pub fn lowest_common_ancestor(graph: &Graph, u: u64, v: u64) -> PyResult<Option<u64>> {
    require_directed(graph, "lowest_common_ancestor()")?;
    if has_self_loop(graph) {
        return Err(not_a_dag("lowest_common_ancestor()"));
    }
    let g = IndexedGraph::new(graph, false);
    let a = g.index_of(u)?;
    let b = g.index_of(v)?;
    let gens = generations(&g).ok_or_else(|| not_a_dag("lowest_common_ancestor()"))?;
    let mut depth = vec![0usize; g.len()];
    for (d, layer) in gens.iter().enumerate() {
        for &x in layer {
            depth[x] = d;
        }
    }
    let mut anc_a = reach(&g, &g.inn, a);
    anc_a[a] = true;
    let mut anc_b = reach(&g, &g.inn, b);
    anc_b[b] = true;
    let common: Vec<bool> = anc_a.iter().zip(&anc_b).map(|(&x, &y)| x && y).collect();
    let lowest = (0..g.len())
        .filter(|&x| common[x] && !g.out[x].iter().any(|&(w, _)| common[w]))
        .max_by(|&x, &y| depth[x].cmp(&depth[y]).then(y.cmp(&x)));
    Ok(lowest.map(|x| g.ids[x]))
}

/// Every topological ordering of a DAG, in lexicographic order of node IDs (Knuth & Szwarcfiter
/// style backtracking). `max_sorts` caps the enumeration, since a DAG with few edges can have
/// up to n! orderings.
#[pyfunction]
#[pyo3(signature = (graph, max_sorts=None))]
pub fn all_topological_sorts(graph: &Graph, max_sorts: Option<usize>) -> PyResult<Vec<Vec<u64>>> {
    require_directed(graph, "all_topological_sorts()")?;
    if has_self_loop(graph) {
        return Err(not_a_dag("all_topological_sorts()"));
    }
    let g = IndexedGraph::new(graph, false);
    if generations(&g).is_none() {
        return Err(not_a_dag("all_topological_sorts()"));
    }
    let n = g.len();
    let limit = max_sorts.unwrap_or(usize::MAX);
    let mut out: Vec<Vec<u64>> = Vec::new();
    if n == 0 {
        out.push(Vec::new());
        return Ok(out);
    }
    let mut indeg: Vec<usize> = g.inn.iter().map(|p| p.len()).collect();
    let mut placed = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    // One frame per depth: the nodes available at that depth and the next one to try
    let ready = |indeg: &[usize], placed: &[bool]| -> Vec<usize> {
        (0..n).filter(|&v| !placed[v] && indeg[v] == 0).collect()
    };
    let mut frames: Vec<(Vec<usize>, usize)> = vec![(ready(&indeg, &placed), 0)];
    while let Some((avail, pos)) = frames.last_mut() {
        if *pos > 0 {
            // undo this depth's previous choice
            let v = order.pop().unwrap();
            placed[v] = false;
            for &(w, _) in &g.out[v] {
                indeg[w] += 1;
            }
        }
        if *pos == avail.len() {
            frames.pop();
            continue;
        }
        let v = avail[*pos];
        *pos += 1;
        order.push(v);
        placed[v] = true;
        for &(w, _) in &g.out[v] {
            indeg[w] -= 1;
        }
        if order.len() == n {
            out.push(order.iter().map(|&i| g.ids[i]).collect());
            if out.len() >= limit {
                break;
            }
        } else {
            frames.push((ready(&indeg, &placed), 0));
        }
    }
    Ok(out)
}

/// A reachability index from pruned landmark labeling: calling it with `(u, v)` node IDs
/// says whether `v` can be reached from `u`.
#[pyclass(module = "edgecase._core")]
pub struct ReachIndex {
    /// node ID -> component (strongly connected; connected when undirected)
    comp: rustc_hash::FxHashMap<u64, usize>,
    /// `lout[c]`: ranks of the hubs `c` reaches; `lin[c]`: ranks of the hubs reaching `c`
    /// (both sorted, since hubs are added in rank order)
    lout: Vec<Vec<u32>>,
    lin: Vec<Vec<u32>>,
}

fn share_hub(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

#[pymethods]
impl ReachIndex {
    fn __call__(&self, u: u64, v: u64) -> PyResult<bool> {
        let find = |x: u64| {
            self.comp.get(&x).copied().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Node {x} is not in the reachability index (rebuild it after changing the graph)"
                ))
            })
        };
        let (a, b) = (find(u)?, find(v)?);
        Ok(a == b || share_hub(&self.lout[a], &self.lin[b]))
    }

    /// Number of components the index was built on.
    #[getter]
    fn n_components(&self) -> usize {
        self.lout.len()
    }

    /// Total number of hub entries over all labels.
    #[getter]
    fn n_labels(&self) -> usize {
        self.lout.iter().chain(&self.lin).map(Vec::len).sum()
    }
}

/// Build a reachability index by pruned landmark labeling (2-hop cover) on the condensation.
/// Components are taken as hubs in decreasing `(in + 1) * (out + 1)` degree order, ties by
/// Tarjan component index (reverse topological); each runs a forward and a backward BFS,
/// pruned wherever an earlier hub already answers the query.
#[pyfunction]
pub fn reachability_index(graph: &Graph) -> ReachIndex {
    let g = IndexedGraph::new(graph, false);
    let (comp, k) = scc(&g);
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); k];
    if graph.is_directed() {
        for v in 0..g.len() {
            for &(w, _) in &g.out[v] {
                if comp[v] != comp[w] {
                    succ[comp[v]].push(comp[w]);
                }
            }
        }
        for (c, out) in succ.iter_mut().enumerate() {
            out.sort_unstable();
            out.dedup();
            for &d in out.iter() {
                pred[d].push(c);
            }
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&c| std::cmp::Reverse((pred[c].len() + 1) * (succ[c].len() + 1)));
    let mut lout: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut lin: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut seen = vec![usize::MAX; k];
    let mut queue = VecDeque::new();
    for (rank, &h) in order.iter().enumerate() {
        let r = rank as u32;
        for forward in [true, false] {
            let stamp = 2 * rank + forward as usize;
            seen[h] = stamp;
            queue.push_back(h);
            while let Some(c) = queue.pop_front() {
                let covered = if forward {
                    share_hub(&lout[h], &lin[c])
                } else {
                    share_hub(&lout[c], &lin[h])
                };
                if covered {
                    continue;
                }
                if forward {
                    lin[c].push(r)
                } else {
                    lout[c].push(r)
                }
                for &d in if forward { &succ[c] } else { &pred[c] } {
                    if seen[d] != stamp {
                        seen[d] = stamp;
                        queue.push_back(d);
                    }
                }
            }
        }
    }
    let comp = g
        .ids
        .iter()
        .enumerate()
        .map(|(v, &id)| (id, comp[v]))
        .collect();
    ReachIndex { comp, lout, lin }
}
