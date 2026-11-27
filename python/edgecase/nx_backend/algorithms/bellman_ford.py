"""Bellman-Ford, Goldberg-Radzik, negative cycles, Johnson's algorithm and all shortest paths.

The searches run in Rust (`PathAdjacency.bellman_ford`, `GoldbergRadzik`, and Dijkstra over
Johnson's reweighted costs) and follow NetworkX's code step by step, including its queue
order, its "recent update" heuristic for negative cycles and the Python sets whose order
decides Goldberg-Radzik's search order (kept in Python here), so distances, predecessor
lists, paths and their order are NetworkX's.
"""

from __future__ import annotations

import networkx as nx

from edgecase.nx_backend.algorithms.shortest_paths import (
    _adjacency,
    _batches,
    _contradiction,
    _keys,
    _paths,
    _require,
    _weighted,
    dijkstra_predecessor_and_distance,
    predecessor,
)

__all__ = [
    "all_pairs_all_shortest_paths",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "all_shortest_paths",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "bellman_ford_predecessor_and_distance",
    "find_negative_cycle",
    "goldberg_radzik",
    "johnson",
    "negative_edge_cycle",
    "single_source_all_shortest_paths",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
]


def _build_paths_from_predecessors(sources, target, pred):
    """NetworkX's `_build_paths_from_predecessors` (the same paths in the same order)."""
    if target not in pred:
        raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")
    stack = {target: iter(pred[target])}
    path = [target]
    while stack:
        node = path[-1]
        if node in sources:
            yield path[::-1]
        for predecessor_ in stack[node]:
            if predecessor_ in stack:
                continue
            stack[predecessor_] = iter(pred[predecessor_])
            path.append(predecessor_)
            break
        else:
            stack.popitem()
            path.pop()


def _negative_selfloop(G, weight):
    """Whether a self-loop (any one of a multigraph's) has a negative weight."""
    if weight is None:
        return False
    values = G.edge_values_of(weight, 1)
    return any(values[i] < 0 for i, (s, d) in enumerate(zip(G.src, G.dst)) if s == d)


def _bellman_ford(G, sources, weight, *, heuristic=True, paths=False, target=None, reverse=False):
    """NetworkX's `_bellman_ford` from `sources` (positions): `(dist, pred, paths)` keyed by
    node, `paths` only with `paths=True` (all, or only `target`'s).
    """
    adj = _adjacency(G, weight, reverse)
    order, dist, preds, cycle = adj.bellman_ford(sources, heuristic, reverse)
    if cycle is not None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    keys = _keys(G, order)
    roots = len(set(sources))
    if not adj.integral:
        dist[:roots] = [0] * roots  # (NetworkX's sources start at the int 0)
    dist = dict(zip(keys, dist))
    pred = dict(zip(keys, (_keys(G, p) for p in preds)))
    result_paths = None
    if paths:
        source_keys = keys[:roots]
        result_paths = {s: [s] for s in source_keys}
        source_set = set(source_keys)
        for dst in [target] if target is not None else pred:
            result_paths[dst] = next(_build_paths_from_predecessors(source_set, dst, pred))
    return dist, pred, result_paths


@_weighted
def bellman_ford_predecessor_and_distance(G, source, target=None, weight="weight", heuristic=False):
    s = _require(G, source, f"Node {source} is not found in the graph")
    if _negative_selfloop(G, weight):
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    if len(G) == 1:
        return ({source: []}, {source: 0})
    dist, pred, _ = _bellman_ford(G, [s], weight, heuristic=heuristic)
    return (pred, dist)


@_weighted
def bellman_ford_path(G, source, target, weight="weight"):
    length, path = single_source_bellman_ford(G, source, target=target, weight=weight)
    return path


@_weighted
def bellman_ford_path_length(G, source, target, weight="weight"):
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        return 0
    s = _require(G, source, f"Source {source} not in G")
    dist, _, _ = _bellman_ford(G, [s], weight)
    try:
        return dist[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"node {target} not reachable from {source}") from err


@_weighted
def single_source_bellman_ford_path(G, source, weight="weight"):
    length, path = single_source_bellman_ford(G, source, weight=weight)
    return path


@_weighted
def single_source_bellman_ford_path_length(G, source, weight="weight"):
    s = _require(G, source, f"Source {source} not in G")
    return _bellman_ford(G, [s], weight)[0]


def _single_source_bellman_ford(G, source, target=None, weight="weight", reverse=False):
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        return (0, [source])
    s = _require(G, source, f"Source {source} not in G")
    dist, _, paths = _bellman_ford(G, [s], weight, paths=True, target=target, reverse=reverse)
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err


@_weighted
def single_source_bellman_ford(G, source, target=None, weight="weight"):
    return _single_source_bellman_ford(G, source, target, weight)


@_weighted
def all_pairs_bellman_ford_path_length(G, weight="weight"):
    for n in G.nodes:
        yield (n, dict(single_source_bellman_ford_path_length(G, n, weight=weight)))


@_weighted
def all_pairs_bellman_ford_path(G, weight="weight"):
    for n in G.nodes:
        yield (n, single_source_bellman_ford_path(G, n, weight=weight))


@_weighted
def negative_edge_cycle(G, weight="weight", heuristic=True):
    if G.number_of_edges() == 0:
        return False
    # (NetworkX adds a node joined to every node and runs Bellman-Ford from it)
    if _negative_selfloop(G, weight):
        return True
    cycle = _adjacency(G, weight).bellman_ford([], heuristic, False, True)[3]
    return cycle is not None


@_weighted
def find_negative_cycle(G, source, weight="weight"):
    s = _require(G, source, f"Source {source} not in G")
    adj = _adjacency(G, weight)
    order, _, preds, cycle = adj.bellman_ford([s], True)
    if cycle is None:
        raise nx.NetworkXError("No negative cycles detected.")
    nodes = G.nodes
    pred = dict(zip(_keys(G, order), (_keys(G, p) for p in preds)))
    v = nodes[cycle]
    # NetworkX's search for the cycle through `v` in the predecessor lists
    neg_cycle = []
    stack = [(v, list(pred[v]))]
    seen = {v}
    while stack:
        node, preds_ = stack[-1]
        if v in preds_:
            neg_cycle.extend([node, v])
            return list(reversed(neg_cycle))
        if preds_:
            nbr = preds_.pop()
            if nbr not in seen:
                stack.append((nbr, list(pred[nbr])))
                neg_cycle.append(node)
                seen.add(nbr)
        else:
            stack.pop()
            if neg_cycle:
                neg_cycle.pop()
            else:
                if _negative_selfloop_at(G, weight, cycle):
                    return [v, v]
                raise nx.NetworkXError("Negative cycle is detected but not found")
    raise nx.NetworkXUnbounded("negative cycle detected but not identified")


def _negative_selfloop_at(G, weight, position):
    values = None if weight is None else G.edge_values_of(weight, 1)
    node_id = position + 1
    return any(
        values is not None and values[i] < 0
        for i, (s, d) in enumerate(zip(G.src, G.dst))
        if s == d == node_id
    )


@_weighted
def goldberg_radzik(G, source, weight="weight"):
    from edgecase._core import GoldbergRadzik

    s = _require(G, source, f"Node {source} is not found in the graph")
    if _negative_selfloop(G, weight):
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    if len(G) == 1:
        return {source: None}, {source: 0}
    adj = _adjacency(G, weight)
    nodes = G.nodes
    key_to_id = G.key_to_id
    state = GoldbergRadzik(adj, s)
    relabeled = {nodes[s]}
    while relabeled:
        # (NetworkX iterates over `relabeled - neg_count.keys()` with neg_count still empty;
        # the set's order decides where its depth-first searches start)
        start = relabeled - {}.keys()
        to_scan = state.topo_sort([key_to_id[u] - 1 for u in start])
        if to_scan is None:
            raise nx.NetworkXUnbounded("Negative cycle detected.")
        relabeled = set()
        for v in state.relax(to_scan):
            relabeled.add(nodes[v])
    order, preds, dist = state.result()
    keys = _keys(G, order)
    pred = {k: None if p is None else nodes[p] for k, p in zip(keys, preds)}
    if not adj.integral:
        dist[0] = 0
    return pred, dict(zip(keys, dist))


@_weighted
def johnson(G, weight="weight"):
    nodes = G.nodes
    n = len(nodes)
    if n == 0:
        return {}
    adj = _adjacency(G, weight)
    order, dist, _, cycle = adj.bellman_ford(list(range(n)), True)
    if cycle is not None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    potentials = [0.0] * n
    for v, d in zip(order, dist):
        potentials[v] = float(d)
    reweighted = adj.reweighted(potentials)
    result = {}
    for batch in _batches(n):
        found, failed = reweighted.dijkstra_many(list(batch), None, True)
        for s, (order_, _, parents) in zip(batch, found):
            result[nodes[s]] = _paths(_keys(G, order_), parents)
        if failed:
            raise _contradiction()
    return result


# All shortest paths (generic.py)


def _predecessors(G, source, weight, method):
    method = "unweighted" if weight is None else method
    if method == "unweighted":
        return predecessor(G, source)
    if method == "dijkstra":
        return dijkstra_predecessor_and_distance(G, source, weight=weight)[0]
    if method == "bellman-ford":
        return bellman_ford_predecessor_and_distance(G, source, weight=weight)[0]
    raise ValueError(f"method not supported: {method}")


@_weighted
def all_shortest_paths(G, source, target, weight=None, method="dijkstra"):
    pred = _predecessors(G, source, weight, method)
    return _build_paths_from_predecessors({source}, target, pred)


@_weighted
def single_source_all_shortest_paths(G, source, weight=None, method="dijkstra"):
    pred = _predecessors(G, source, weight, method)
    for n in pred:
        yield (n, list(_build_paths_from_predecessors({source}, n, pred)))


@_weighted
def all_pairs_all_shortest_paths(G, weight=None, method="dijkstra"):
    for n in G.nodes:
        yield (n, dict(single_source_all_shortest_paths(G, n, weight=weight, method=method)))


def _reverse_bellman_ford_paths(G, target, weight):
    """`single_source_bellman_ford_path` on NetworkX's reverse view, from `target`."""
    return _single_source_bellman_ford(G, target, None, weight, reverse=True)[1]


def _reverse_bellman_ford_lengths(G, target, weight):
    t = _require(G, target, f"Source {target} not in G")
    return _bellman_ford(G, [t], weight, reverse=True)[0]
