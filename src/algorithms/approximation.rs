//! Approximation and optimization: traveling salesperson heuristics on the metric closure
//! (Christofides, nearest neighbor, simulated annealing, threshold accepting), the metric
//! closure itself, tree decompositions (min-degree and min-fill-in elimination), the densest
//! subgraph (exact via Goldberg's max-flow reduction, or Greedy++ peeling) and max-cut local
//! search.

use crate::algorithms::matching_cover::blossom_matching;
use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn rng_from(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Outgoing adjacency `(target, weight)` following edge direction on directed graphs.
fn out_adjacency(graph: &Graph, idx: &FxHashMap<u64, usize>, weighted: bool) -> PyResult<Vec<Vec<(usize, f64)>>> {
    let directed = graph.is_directed();
    let mut adj = vec![Vec::new(); idx.len()];
    for e in graph.get_edges_internal() {
        let w = if weighted { e.weight.unwrap_or(1.0) } else { 1.0 };
        if !(w >= 0.0) || !w.is_finite() {
            return value_err("edge weights must be finite and non-negative");
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        adj[a].push((b, w));
        if !directed {
            adj[b].push((a, w));
        }
    }
    Ok(adj)
}

/// Dijkstra from `s`: distances and predecessors (usize::MAX for none).
fn dijkstra(adj: &[Vec<(usize, f64)>], s: usize) -> (Vec<f64>, Vec<usize>) {
    #[derive(PartialEq)]
    struct Item(f64, usize);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            o.0.partial_cmp(&self.0).unwrap().then(o.1.cmp(&self.1))
        }
    }
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut pred = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    heap.push(Item(0.0, s));
    while let Some(Item(d, u)) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                pred[v] = u;
                heap.push(Item(nd, v));
            }
        }
    }
    (dist, pred)
}

// ---------------------------------------------------------------------------------------------
// Traveling salesperson
// ---------------------------------------------------------------------------------------------

fn tour_cost(d: &[Vec<f64>], tour: &[usize]) -> f64 {
    tour.windows(2).map(|w| d[w[0]][w[1]]).sum()
}

fn greedy_tour(d: &[Vec<f64>]) -> Vec<usize> {
    let k = d.len();
    let mut left: Vec<bool> = vec![true; k];
    left[0] = false;
    let mut tour = vec![0];
    let mut cur = 0;
    for _ in 1..k {
        let mut best = usize::MAX;
        for v in 0..k {
            if left[v] && (best == usize::MAX || d[cur][v] < d[cur][best]) {
                best = v;
            }
        }
        left[best] = false;
        tour.push(best);
        cur = best;
    }
    tour.push(0);
    tour
}

fn christofides_tour(d: &[Vec<f64>]) -> Vec<usize> {
    let k = d.len();
    // Prim's minimum spanning tree on the complete graph
    let mut in_tree = vec![false; k];
    let mut best = vec![f64::INFINITY; k];
    let mut parent = vec![usize::MAX; k];
    let mut multi: Vec<(usize, usize)> = Vec::new();
    best[0] = 0.0;
    for _ in 0..k {
        let u = (0..k).filter(|&v| !in_tree[v]).min_by(|&a, &b| best[a].partial_cmp(&best[b]).unwrap().then(a.cmp(&b))).unwrap();
        in_tree[u] = true;
        if parent[u] != usize::MAX {
            multi.push((parent[u], u));
        }
        for v in 0..k {
            if !in_tree[v] && d[u][v] < best[v] {
                best[v] = d[u][v];
                parent[v] = u;
            }
        }
    }
    // Minimum-weight perfect matching on the odd-degree nodes
    let mut degree = vec![0usize; k];
    for &(a, b) in &multi {
        degree[a] += 1;
        degree[b] += 1;
    }
    let odd: Vec<usize> = (0..k).filter(|&v| degree[v] % 2 == 1).collect();
    let mut pairs = Vec::new();
    let mut top = f64::NEG_INFINITY;
    for (i, &a) in odd.iter().enumerate() {
        for &b in &odd[i + 1..] {
            pairs.push((i, odd.iter().position(|&x| x == b).unwrap(), d[a][b]));
            top = top.max(d[a][b]);
        }
    }
    let flipped: Vec<(usize, usize, f64)> = pairs.iter().map(|&(i, j, w)| (i, j, top + 1.0 - w)).collect();
    for e in blossom_matching(odd.len(), &flipped, true) {
        multi.push((odd[pairs[e].0], odd[pairs[e].1]));
    }
    // Eulerian circuit (Hierholzer) and shortcutting
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k];
    for (i, &(a, b)) in multi.iter().enumerate() {
        adj[a].push((b, i));
        adj[b].push((a, i));
    }
    let mut used = vec![false; multi.len()];
    let mut ptr = vec![0usize; k];
    let mut stack = vec![0usize];
    let mut circuit = Vec::new();
    while let Some(&u) = stack.last() {
        while ptr[u] < adj[u].len() && used[adj[u][ptr[u]].1] {
            ptr[u] += 1;
        }
        if ptr[u] == adj[u].len() {
            circuit.push(u);
            stack.pop();
        } else {
            let (v, e) = adj[u][ptr[u]];
            used[e] = true;
            stack.push(v);
        }
    }
    circuit.reverse();
    let mut seen = vec![false; k];
    let mut tour = Vec::with_capacity(k + 1);
    for v in circuit {
        if !seen[v] {
            seen[v] = true;
            tour.push(v);
        }
    }
    tour.push(tour[0]);
    tour
}

/// Simulated annealing (`anneal`) or threshold accepting local search from a starting cycle.
#[allow(clippy::too_many_arguments)]
fn local_search(
    d: &[Vec<f64>],
    start: Vec<usize>,
    anneal: bool,
    move_one: bool,
    temperature: f64,
    max_iterations: usize,
    n_inner: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let len = start.len();
    if len < 4 {
        return start;
    }
    let mut cycle = start;
    let mut cost = tour_cost(d, &cycle);
    let mut best = cycle.clone();
    let mut best_cost = cost;
    let mut temp = temperature;
    let alpha = if anneal { 0.01 } else { 0.1 };
    let mut count = 0;
    while count <= max_iterations && (!anneal || temp > 0.0) {
        count += 1;
        let mut accepted = false;
        for _ in 0..n_inner {
            let a = rng.gen_range(1..len - 1);
            let mut b = rng.gen_range(1..len - 2);
            if b >= a {
                b += 1;
            }
            let mut cand = cycle.clone();
            if move_one {
                let x = cand.remove(a);
                cand.insert(b, x);
            } else {
                cand.swap(a, b);
            }
            let cand_cost = tour_cost(d, &cand);
            let delta = cand_cost - cost;
            let take = if anneal {
                delta <= 0.0 || (-delta / temp).exp() >= rng.gen::<f64>()
            } else {
                delta <= temp
            };
            if take {
                accepted = true;
                cycle = cand;
                cost = cand_cost;
                if cost < best_cost {
                    count = 0;
                    best = cycle.clone();
                    best_cost = cost;
                }
            }
        }
        if anneal || accepted {
            temp -= temp * alpha;
        }
    }
    best
}

/// A traveling salesperson tour (or path) through `nodes` (default: all), on the metric
/// closure: `(walk of node IDs, total cost)`. Consecutive tour nodes are joined by shortest
/// paths, so the walk may pass through other nodes.
#[pyfunction]
#[pyo3(signature = (graph, nodes=None, cycle=true, method="christofides", weighted=true, seed=None,
                    max_iterations=10, n_inner=100, move_kind="1-1", temperature=None))]
#[allow(clippy::too_many_arguments)]
pub fn traveling_salesman(
    graph: &Graph,
    nodes: Option<Vec<u64>>,
    cycle: bool,
    method: &str,
    weighted: bool,
    seed: Option<u64>,
    max_iterations: usize,
    n_inner: usize,
    move_kind: &str,
    temperature: Option<f64>,
) -> PyResult<(Vec<u64>, f64)> {
    let (ids, idx) = sorted_ids(graph);
    let targets: Vec<usize> = match nodes {
        Some(list) => {
            let mut out = Vec::new();
            let mut seen = FxHashSet::default();
            for id in list {
                match idx.get(&id) {
                    Some(&i) => {
                        if seen.insert(i) {
                            out.push(i)
                        }
                    }
                    None => return value_err(format!("Node {id} does not exist")),
                }
            }
            out
        }
        None => (0..ids.len()).collect(),
    };
    let k = targets.len();
    if k == 0 {
        return value_err("the tour needs at least one node");
    }
    if method == "christofides" && graph.is_directed() {
        return value_err("method='christofides' requires an undirected graph");
    }
    let move_one = match move_kind {
        "1-1" => false,
        "1-0" => true,
        _ => return value_err(format!("move= must be '1-1' or '1-0', got {move_kind:?}")),
    };
    let adj = out_adjacency(graph, &idx, weighted)?;
    let mut d = vec![vec![0.0; k]; k];
    let mut preds = Vec::with_capacity(k);
    for (i, &s) in targets.iter().enumerate() {
        let (dist, pred) = dijkstra(&adj, s);
        for (j, &t) in targets.iter().enumerate() {
            if !dist[t].is_finite() {
                return value_err(if graph.is_directed() {
                    "every tour node must be reachable from every other (the graph is not strongly connected)"
                } else {
                    "every tour node must be reachable from every other (the graph is not connected)"
                });
            }
            d[i][j] = dist[t];
        }
        preds.push(pred);
    }
    let mut tour: Vec<usize> = if k == 1 {
        vec![0, 0]
    } else if k == 2 {
        vec![0, 1, 0]
    } else {
        match method {
            "christofides" => christofides_tour(&d),
            "greedy" => greedy_tour(&d),
            "simulated_annealing" | "threshold_accepting" => {
                let anneal = method == "simulated_annealing";
                let mut rng = rng_from(seed);
                let t = temperature.unwrap_or(if anneal { 100.0 } else { 1.0 });
                local_search(&d, greedy_tour(&d), anneal, move_one, t, max_iterations, n_inner, &mut rng)
            }
            _ => return value_err(format!("unknown method {method:?}")),
        }
    };
    if !cycle {
        // drop the most expensive leg, as NetworkX does
        let mut worst = 0;
        for i in 1..tour.len() - 1 {
            if d[tour[i]][tour[i + 1]] > d[tour[worst]][tour[worst + 1]] {
                worst = i;
            }
        }
        let body = &tour[..tour.len() - 1];
        tour = body[worst + 1..].iter().chain(body[..worst + 1].iter()).copied().collect();
    }
    let cost = tour_cost(&d, &tour);
    // expand legs into shortest paths
    let mut walk = vec![ids[targets[tour[0]]]];
    for w in tour.windows(2) {
        let (a, b) = (targets[w[0]], targets[w[1]]);
        let pred = &preds[w[0]];
        let mut leg = Vec::new();
        let mut x = b;
        if a == b {
            leg.push(b); // a one-stop round trip
        }
        while x != a {
            leg.push(x);
            x = pred[x];
        }
        walk.extend(leg.into_iter().rev().map(|v| ids[v]));
    }
    Ok((walk, cost))
}

/// The metric closure: a complete graph on the same nodes (IDs, labels and attributes kept)
/// whose edge weights are shortest-path distances.
#[pyfunction]
#[pyo3(signature = (graph, weighted=true))]
pub fn metric_closure(py: Python, graph: &Graph, weighted: bool) -> PyResult<Graph> {
    let (ids, idx) = sorted_ids(graph);
    let adj = out_adjacency(graph, &idx, weighted)?;
    let directed = graph.is_directed();
    let mut result = Graph::new_rust(directed, true);
    let nodes = graph.get_nodes();
    for &id in &ids {
        let node = &nodes[&id];
        let attributes = match &node.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_node_rust(Node { id, label: node.label.clone(), attributes })?;
    }
    for s in 0..ids.len() {
        let (dist, _) = dijkstra(&adj, s);
        for t in 0..ids.len() {
            if t == s || (!directed && t < s) {
                continue;
            }
            if !dist[t].is_finite() {
                return value_err(if directed {
                    "the metric closure needs a strongly connected graph"
                } else {
                    "the metric closure needs a connected graph"
                });
            }
            result.add_edge_rust(Edge { id: 0, source: ids[s], target: ids[t], weight: Some(dist[t]), attributes: None })?;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Tree decompositions
// ---------------------------------------------------------------------------------------------

/// Underlying simple undirected adjacency sets (no self-loops, no parallel edges).
fn simple_sets(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<FxHashSet<usize>> {
    let mut adj = vec![FxHashSet::default(); idx.len()];
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a != b {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    }
    adj
}

/// Tree decomposition by greedy elimination: `(width, bags as node IDs, bag edges)`.
#[pyfunction]
#[pyo3(signature = (graph, heuristic="min_fill_in"))]
pub fn tree_decomposition(graph: &Graph, heuristic: &str) -> PyResult<(i64, Vec<Vec<u64>>, Vec<(usize, usize)>)> {
    let min_fill = match heuristic {
        "min_fill_in" => true,
        "min_degree" => false,
        _ => return value_err(format!("heuristic= must be 'min_fill_in' or 'min_degree', got {heuristic:?}")),
    };
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut adj = simple_sets(graph, &idx);
    let mut alive: Vec<usize> = (0..n).collect();
    let mut is_alive = vec![true; n];
    let mut heap: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();
    let mut counter = 0;
    if !min_fill {
        for v in 0..n {
            heap.push(Reverse((adj[v].len(), counter, v)));
            counter += 1;
        }
    }
    let mut stack: Vec<(usize, Vec<usize>)> = Vec::new();
    loop {
        let remaining = alive.len();
        let pick = if remaining == 0 {
            None
        } else if min_fill {
            let mut order = alive.clone();
            order.sort_by_key(|&v| adj[v].len()); // stable: ties keep ID order
            if adj[order[0]].len() == remaining - 1 {
                None
            } else {
                let mut best: Option<usize> = None;
                let mut best_fill = usize::MAX;
                for &v in &order {
                    let nbrs = &adj[v];
                    let mut fill = 0usize;
                    for &u in nbrs {
                        fill += nbrs.iter().filter(|&&x| x != u && !adj[u].contains(&x)).count();
                    }
                    let fill = fill / 2;
                    if fill < best_fill {
                        best_fill = fill;
                        best = Some(v);
                        if fill == 0 {
                            break;
                        }
                    }
                }
                best
            }
        } else {
            let mut found = None;
            while let Some(Reverse((deg, _, v))) = heap.pop() {
                if !is_alive[v] || adj[v].len() != deg {
                    continue;
                }
                if deg == remaining - 1 {
                    break;
                }
                found = Some(v);
                break;
            }
            found
        };
        let Some(v) = pick else { break };
        let mut nbrs: Vec<usize> = adj[v].iter().copied().collect();
        nbrs.sort_unstable();
        for (i, &a) in nbrs.iter().enumerate() {
            for &b in &nbrs[i + 1..] {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        for &u in &nbrs {
            adj[u].remove(&v);
        }
        adj[v].clear();
        is_alive[v] = false;
        alive.retain(|&x| x != v);
        if !min_fill {
            for &u in &nbrs {
                heap.push(Reverse((adj[u].len(), counter, u)));
                counter += 1;
            }
        }
        stack.push((v, nbrs));
    }
    let mut bags: Vec<Vec<usize>> = vec![alive.clone()];
    let mut bag_sets: Vec<FxHashSet<usize>> = vec![alive.iter().copied().collect()];
    let mut width = alive.len() as i64 - 1;
    let mut edges = Vec::new();
    while let Some((v, nbrs)) = stack.pop() {
        let old = bag_sets.iter().position(|bag| nbrs.iter().all(|x| bag.contains(x))).unwrap_or(0);
        let mut bag = nbrs.clone();
        bag.push(v);
        bag.sort_unstable();
        width = width.max(bag.len() as i64 - 1);
        bag_sets.push(bag.iter().copied().collect());
        bags.push(bag);
        edges.push((old, bags.len() - 1));
    }
    let bags = bags.into_iter().map(|b| b.into_iter().map(|i| ids[i]).collect()).collect();
    Ok((width, bags, edges))
}

// ---------------------------------------------------------------------------------------------
// Densest subgraph
// ---------------------------------------------------------------------------------------------

struct Dinic {
    adj: Vec<Vec<usize>>,
    to: Vec<usize>,
    cap: Vec<i64>,
}

impl Dinic {
    fn new(n: usize) -> Self {
        Dinic { adj: vec![Vec::new(); n], to: Vec::new(), cap: Vec::new() }
    }
    fn add(&mut self, u: usize, v: usize, c: i64, back: i64) {
        self.adj[u].push(self.to.len());
        self.to.push(v);
        self.cap.push(c);
        self.adj[v].push(self.to.len());
        self.to.push(u);
        self.cap.push(back);
    }
    fn max_flow(&mut self, s: usize, t: usize) -> i64 {
        let n = self.adj.len();
        let mut flow = 0;
        loop {
            let mut level = vec![usize::MAX; n];
            level[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for &e in &self.adj[u] {
                    if self.cap[e] > 0 && level[self.to[e]] == usize::MAX {
                        level[self.to[e]] = level[u] + 1;
                        q.push_back(self.to[e]);
                    }
                }
            }
            if level[t] == usize::MAX {
                return flow;
            }
            let mut it = vec![0usize; n];
            loop {
                let f = self.dfs(s, t, i64::MAX, &level, &mut it);
                if f == 0 {
                    break;
                }
                flow += f;
            }
        }
    }
    fn dfs(&mut self, u: usize, t: usize, f: i64, level: &[usize], it: &mut [usize]) -> i64 {
        if u == t {
            return f;
        }
        while it[u] < self.adj[u].len() {
            let e = self.adj[u][it[u]];
            let v = self.to[e];
            if self.cap[e] > 0 && level[v] == level[u] + 1 {
                let d = self.dfs(v, t, f.min(self.cap[e]), level, it);
                if d > 0 {
                    self.cap[e] -= d;
                    self.cap[e ^ 1] += d;
                    return d;
                }
            }
            it[u] += 1;
        }
        0
    }
    fn source_side(&self, s: usize) -> Vec<bool> {
        let mut seen = vec![false; self.adj.len()];
        seen[s] = true;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &e in &self.adj[u] {
                if self.cap[e] > 0 && !seen[self.to[e]] {
                    seen[self.to[e]] = true;
                    q.push_back(self.to[e]);
                }
            }
        }
        seen
    }
}

/// Densest subgraph (most edges per node), on the underlying simple undirected graph:
/// `(density, node IDs)`. `method` is `"exact"` (Goldberg's reduction with Dinkelbach
/// iterations) or `"greedy++"` (Boob et al. peeling, `iterations` rounds).
#[pyfunction]
#[pyo3(signature = (graph, method="exact", iterations=1))]
pub fn densest_subgraph(graph: &Graph, method: &str, iterations: usize) -> PyResult<(f64, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let adj = simple_sets(graph, &idx);
    let pairs: Vec<(usize, usize)> =
        (0..n).flat_map(|u| adj[u].iter().filter(move |&&v| u < v).map(move |&v| (u, v))).collect();
    let m = pairs.len();
    if m == 0 {
        return Ok((0.0, Vec::new()));
    }
    let deg: Vec<usize> = adj.iter().map(|s| s.len()).collect();
    let mut best: Vec<usize> = match method {
        "exact" => {
            // current candidate: all nodes, density p / q
            let mut set: Vec<usize> = (0..n).collect();
            let (mut p, mut q) = (m as i64, n as i64);
            loop {
                let (s, t) = (n, n + 1);
                let mut net = Dinic::new(n + 2);
                for v in 0..n {
                    net.add(s, v, m as i64 * q, 0);
                    net.add(v, t, m as i64 * q + 2 * p - deg[v] as i64 * q, 0);
                }
                for &(a, b) in &pairs {
                    net.add(a, b, q, q);
                }
                net.max_flow(s, t);
                let side = net.source_side(s);
                let cand: Vec<usize> = (0..n).filter(|&v| side[v]).collect();
                if cand.is_empty() {
                    break;
                }
                let inside: FxHashSet<usize> = cand.iter().copied().collect();
                let e = pairs.iter().filter(|(a, b)| inside.contains(a) && inside.contains(b)).count() as i64;
                if q * e - p * cand.len() as i64 > 0 {
                    p = e;
                    q = cand.len() as i64;
                    set = cand;
                } else {
                    break;
                }
            }
            set
        }
        "greedy++" => {
            if iterations < 1 {
                return value_err("iterations= must be at least 1");
            }
            let mut loads = vec![0usize; n];
            let mut best_density = 0.0f64;
            let mut best_set: Vec<usize> = Vec::new();
            for _ in 0..iterations {
                let mut cur = deg.clone();
                let mut alive = vec![true; n];
                let mut count = n;
                let mut edges = m;
                let mut heap: BinaryHeap<Reverse<(usize, usize)>> = (0..n).map(|v| Reverse((loads[v] + cur[v], v))).collect();
                let mut order = Vec::with_capacity(n);
                let mut best_round = (0.0f64, 0usize); // density, nodes removed before it
                while count > 0 {
                    let density = edges as f64 / count as f64;
                    if density > best_round.0 {
                        best_round = (density, order.len());
                    }
                    let Some(Reverse((key, v))) = heap.pop() else { break };
                    if !alive[v] || key != loads[v] + cur[v] {
                        continue;
                    }
                    loads[v] += cur[v];
                    for &u in &adj[v] {
                        if alive[u] {
                            cur[u] -= 1;
                            edges -= 1;
                            heap.push(Reverse((loads[u] + cur[u], u)));
                        }
                    }
                    alive[v] = false;
                    count -= 1;
                    order.push(v);
                }
                if best_round.0 > best_density {
                    best_density = best_round.0;
                    let removed: FxHashSet<usize> = order[..best_round.1].iter().copied().collect();
                    best_set = (0..n).filter(|v| !removed.contains(v)).collect();
                }
            }
            best_set
        }
        _ => return value_err(format!("method= must be 'exact' or 'greedy++', got {method:?}")),
    };
    best.sort_unstable();
    let inside: FxHashSet<usize> = best.iter().copied().collect();
    let e = pairs.iter().filter(|(a, b)| inside.contains(a) && inside.contains(b)).count();
    Ok((e as f64 / best.len() as f64, best.into_iter().map(|v| ids[v]).collect()))
}

// ---------------------------------------------------------------------------------------------
// Max-cut
// ---------------------------------------------------------------------------------------------

/// Large cut by single-node flips (`"one_exchange"`) or random sides (`"randomized"`):
/// `(cut weight, side S, side T)`. Edge direction is ignored and self-loops never count.
#[pyfunction]
#[pyo3(signature = (graph, method="one_exchange", initial=None, seed=None, weighted=false, p=0.5))]
pub fn max_cut(
    graph: &Graph,
    method: &str,
    initial: Option<Vec<u64>>,
    seed: Option<u64>,
    weighted: bool,
    p: f64,
) -> PyResult<(f64, Vec<u64>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a == b {
            continue;
        }
        let w = if weighted { e.weight.unwrap_or(1.0) } else { 1.0 };
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    let mut rng = rng_from(seed);
    let mut side = vec![false; n];
    match method {
        "one_exchange" => {
            if let Some(list) = initial {
                for id in list {
                    match idx.get(&id) {
                        Some(&i) => side[i] = true,
                        None => return value_err(format!("Node {id} does not exist")),
                    }
                }
            }
            let mut gain: Vec<f64> = (0..n)
                .map(|v| adj[v].iter().map(|&(u, w)| if side[u] == side[v] { w } else { -w }).sum())
                .collect();
            let mut order: Vec<usize> = (0..n).collect();
            loop {
                order.shuffle(&mut rng);
                let mut pick = None;
                for &v in &order {
                    if pick.map_or(true, |b: usize| gain[v] > gain[b]) {
                        pick = Some(v);
                    }
                }
                let Some(v) = pick else { break };
                if gain[v] <= 0.0 {
                    break;
                }
                for &(u, w) in &adj[v] {
                    gain[u] += if side[u] == side[v] { -2.0 * w } else { 2.0 * w };
                }
                gain[v] = -gain[v];
                side[v] = !side[v];
            }
        }
        "randomized" => {
            if !(0.0..=1.0).contains(&p) {
                return value_err("p= must be between 0 and 1");
            }
            for s in side.iter_mut() {
                *s = rng.gen::<f64>() < p;
            }
        }
        _ => return value_err(format!("method= must be 'one_exchange' or 'randomized', got {method:?}")),
    }
    let value: f64 = (0..n).map(|v| adj[v].iter().filter(|&&(u, _)| side[u] != side[v]).map(|&(_, w)| w).sum::<f64>()).sum::<f64>() / 2.0;
    let s: Vec<u64> = (0..n).filter(|&v| side[v]).map(|v| ids[v]).collect();
    let t: Vec<u64> = (0..n).filter(|&v| !side[v]).map(|v| ids[v]).collect();
    Ok((value, s, t))
}
