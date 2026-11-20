"""Shortest paths: NetworkX's `shortest_paths` package (unweighted, Dijkstra and generic).

The searches run in Rust (`edgecase._core.PathAdjacency`) over the graph's arcs in
NetworkX's adjacency order, and break ties the way NetworkX's own code does: breadth-first
search level by level in discovery order, and Dijkstra with a heap ordered by distance and
then insertion. So the results are NetworkX's exactly: the same paths among equally short
ones, the same predecessor lists and the same dict order.

A directed graph's backward searches (`single_target_*`, the backward half of the
bidirectional searches, `shortest_path(G, target=t)`) follow the predecessor order the
conversion recorded (`EdgecaseGraph.pred_order`, NetworkX's `G._pred`). Without it (with
`nx.config.backends.edgecase.preserve_pred_order = False`) predecessors are ordered by
source node, and ties among equally short paths can be broken differently.

Weights are read from the edge attribute named by `weight` (1 where absent; a value of None
hides the edge), as NetworkX does; on a multigraph a step costs the least of its parallel
edges. A callable `weight` is not supported (NetworkX runs such calls itself).
"""

from __future__ import annotations

import inspect
from math import inf

import networkx as nx

from edgecase._core import PathAdjacency
from edgecase.nx_backend.registry import networkx_algorithm, prefer_converted_graph

__all__ = [
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "average_shortest_path_length",
    "bidirectional_dijkstra",
    "bidirectional_shortest_path",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "has_path",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "predecessor",
    "shortest_path",
    "shortest_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
]

# Sources per Rust call in the all-pairs generators (the first calls do fewer, so taking
# only the first few results stays cheap); fewer on big graphs to bound memory
_MAX_BATCH = 256
_BATCH_ENTRIES = 4_000_000


# Helpers


def _adjacency(G, weight=None, reverse=False):
    """The `PathAdjacency` of `G`, weighted by edge attribute `weight` (None: all 1).

    Built on first use and kept on the graph (which is never mutated). With `reverse` (for
    searches over predecessors), a directed graph's recorded predecessor order is added.
    """
    cache = G.__dict__.get("_path_adjacencies")
    if cache is None:
        cache = G._path_adjacencies = {}
    adj = cache.get(weight)
    if adj is None:
        weights = None if weight is None else G.edge_values_of(weight, 1)
        adj = cache[weight] = PathAdjacency(
            len(G.nodes), G.src, G.dst, weights, directed=G.directed
        )
    if reverse and G.directed and not adj.has_predecessor_order:
        pred_order = getattr(G, "pred_order", None)
        if pred_order is not None:
            pred_nodes, counts, preds = pred_order
            get_id = G.key_to_id.__getitem__
            adj.set_predecessor_order(
                list(map(get_id, pred_nodes)), counts, list(map(get_id, preds))
            )
    return adj


def _position(G, node):
    """The 0-based position of `node` in `G` (-1 if `G` has no such node)."""
    try:
        return G.key_to_id[node] - 1
    except (KeyError, TypeError):
        return -1


def _require(G, node, message):
    position = _position(G, node)
    if position < 0:
        raise nx.NodeNotFound(message)
    return position


def _cutoff(cutoff):
    """A BFS cutoff for Rust (`inf` for None)."""
    return inf if cutoff is None else float(cutoff)


def _keys(G, positions):
    return list(map(G.nodes.__getitem__, positions))


def _paths(keys, parents, roots=1, reverse=False):
    """`{node: path}` from search-tree parents (positions in `keys`); the first `roots`
    nodes are roots. Paths run from a root, or towards it with `reverse`.
    """
    paths = [[key] for key in keys[:roots]]
    append = paths.append
    if reverse:
        for i in range(roots, len(keys)):
            append([keys[i], *paths[parents[i]]])
    else:
        for i in range(roots, len(keys)):
            append([*paths[parents[i]], keys[i]])
    return dict(zip(keys, paths))


def _distances(G, adj, order, dist, roots=1):
    """`{node: distance}`; roots get the int 0, as in NetworkX, even with float weights.

    The roots are the first `roots` nodes of `order`. Among them, a source that a negative
    arc reached first keeps its float distance (NetworkX finishes it from that heap entry).
    """
    if roots and not adj.integral:
        for i in range(min(roots, len(dist))):
            if dist[i] == 0:
                dist[i] = 0
    return dict(zip(_keys(G, order), dist))


def _batches(n):
    """Ranges covering `range(n)`, small first."""
    cap = max(1, min(_MAX_BATCH, _BATCH_ENTRIES // max(n, 1)))
    start = 0
    size = 1
    while start < n:
        stop = min(n, start + size)
        yield range(start, stop)
        start = stop
        size = min(cap, 2 * size)


def _contradiction():
    return ValueError("Contradictory paths found:", "negative weights?")


def _can_run_weight(func):
    """A `can_run` for `func` that refuses a callable `weight`."""
    signature = inspect.signature(func)

    def can_run(*args, **kwargs):
        try:
            weight = signature.bind(*args, **kwargs).arguments.get("weight")
        except TypeError:
            return True  # (the call raises the TypeError itself)
        if callable(weight):
            return "a callable weight is not supported; give an edge attribute name"
        return True

    return can_run


def _weighted(func=None, *, should_run=None):
    """Register a function that takes `weight` (an edge attribute name)."""
    if func is None:
        return lambda f: _weighted(f, should_run=should_run)
    return networkx_algorithm(func, can_run=_can_run_weight(func), should_run=should_run)


# Automatic dispatch (`should_run`): single-source breadth-first searches cost about as
# much as converting the graph, and single-pair searches usually look at a small part of
# it, so on a graph not yet converted NetworkX runs them itself (measured in the shortest-
# path benchmark: 0.3x and ~0.02x with conversion; 5x and ~9x on a converted graph).
# Single-source Dijkstra (1.4-1.9x with conversion) and the all-pairs functions always run.


def _when_target_given(G, *args, **kwargs):
    """`should_run` for functions that stop early once a `target` is reached."""
    target = kwargs.get("target", args[1] if len(args) > 1 else None)
    return True if target is None else prefer_converted_graph(G)


def _unless_all_pairs(G, source=None, target=None, *args, **kwargs):
    """`should_run` for `shortest_path(_length)`: always for all pairs, else as above."""
    if source is None and target is None:
        return True
    return prefer_converted_graph(G)


# Unweighted (breadth-first search)


def _bfs_lengths(G, source, cutoff, reverse=False):
    return _adjacency(G, reverse=reverse).bfs_lengths(G.nodes, source, _cutoff(cutoff), reverse)


def _bfs_paths(G, source, cutoff, reverse=False):
    adj = _adjacency(G, reverse=reverse)
    order, _, parents = adj.bfs_levels(source, _cutoff(cutoff), reverse, True)
    return _paths(_keys(G, order), parents, reverse=reverse)


@networkx_algorithm(linear_time=True)
def single_source_shortest_path_length(G, source, cutoff=None):
    s = _require(G, source, f"Source {source} is not in G")
    return _bfs_lengths(G, s, cutoff)


@networkx_algorithm(linear_time=True)
def single_target_shortest_path_length(G, target, cutoff=None):
    t = _require(G, target, f"Target {target} is not in G")
    return _bfs_lengths(G, t, cutoff, reverse=True)


@networkx_algorithm
def all_pairs_shortest_path_length(G, cutoff=None):
    adj = _adjacency(G)
    nodes = G.nodes
    cutoff = _cutoff(cutoff)
    for batch in _batches(len(nodes)):
        yield from zip(
            map(nodes.__getitem__, batch), adj.bfs_lengths_many(nodes, list(batch), cutoff)
        )


@networkx_algorithm(linear_time=True)
def single_source_shortest_path(G, source, cutoff=None):
    s = _require(G, source, f"Source {source} not in G")
    return _bfs_paths(G, s, cutoff)


@networkx_algorithm(linear_time=True)
def single_target_shortest_path(G, target, cutoff=None):
    t = _require(G, target, f"Target {target} not in G")
    return _bfs_paths(G, t, cutoff, reverse=True)


@networkx_algorithm
def all_pairs_shortest_path(G, cutoff=None):
    adj = _adjacency(G)
    nodes = G.nodes
    cutoff = _cutoff(cutoff)
    for batch in _batches(len(nodes)):
        results = adj.bfs_levels_many(list(batch), cutoff, parents=True)
        for s, (order, _, parents) in zip(batch, results):
            yield nodes[s], _paths(_keys(G, order), parents)


def _bidirectional_bfs(G, source, target):
    """The node positions on NetworkX's bidirectional BFS path (None if there is none)."""
    s = _require(G, source, f"Source {source} is not in G")
    t = _require(G, target, f"Target {target} is not in G")
    return _adjacency(G, reverse=True).bidirectional_bfs(s, t)


@networkx_algorithm(linear_time=True)
def bidirectional_shortest_path(G, source, target):
    path = _bidirectional_bfs(G, source, target)
    if path is None:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    return _keys(G, path)


@networkx_algorithm(linear_time=True)
def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    s = _require(G, source, f"Source {source} not in G")
    # (NetworkX ignores a falsy cutoff such as 0)
    order, levels, preds = _adjacency(G).bfs_predecessors(s, float(cutoff) if cutoff else None)
    keys = _keys(G, order)
    pred = dict(zip(keys, (_keys(G, p) for p in preds)))
    if target is not None:
        if return_seen:
            if target not in pred:
                return ([], -1)
            return (pred[target], dict(zip(keys, levels))[target])
        if target not in pred:
            return []
        return pred[target]
    if return_seen:
        return (pred, dict(zip(keys, levels)))
    return pred


# Weighted (Dijkstra)


def _sources(G, sources):
    """Positions of `sources`, raising like NetworkX's multi-source functions."""
    if not sources:
        raise ValueError("sources must not be empty")
    positions = []
    for s in sources:
        position = _position(G, s)
        if position < 0:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
        positions.append(position)
    return positions


def _dijkstra_lengths(G, sources, cutoff, weight, reverse=False):
    adj = _adjacency(G, weight, reverse)
    return adj.dijkstra_lengths(G.nodes, sources, cutoff, reverse)


def _dijkstra_paths(G, sources, cutoff, weight, reverse=False):
    """`(dist, paths)` from `sources` (positions)."""
    adj = _adjacency(G, weight, reverse)
    order, dist, parents = adj.dijkstra(sources, cutoff, None, "paths", reverse)
    distinct = list(dict.fromkeys(sources))
    roots = len(distinct)
    # (only with negative weights can a node be finished before a source's turn)
    if order[:roots] != distinct and set(order[:roots]) != set(distinct):
        # NetworkX rebuilds paths assuming the sources come first: some go missing or it
        # raises KeyError
        raise NotImplementedError(
            "negative weights finish another node before a source; left to NetworkX"
        )
    paths = _paths(_keys(G, order), parents, roots, reverse=reverse)
    if order[:roots] != distinct:
        paths = dict.fromkeys(_keys(G, distinct)) | paths  # (NetworkX's key order)
    return (_distances(G, adj, order, dist, roots), paths)


@_weighted(should_run=prefer_converted_graph)
def dijkstra_path(G, source, target, weight="weight"):
    length, path = single_source_dijkstra(G, source, target=target, weight=weight)
    return path


@_weighted(should_run=prefer_converted_graph)
def dijkstra_path_length(G, source, target, weight="weight"):
    s = _require(G, source, f"Node {source} not found in graph")
    if source == target:
        return 0
    t = _position(G, target)
    if t >= 0:
        order, dist, _ = _adjacency(G, weight).dijkstra([s], None, t, "lengths")
        if order[-1] == t:
            return dist[-1]
    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


@_weighted
def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)


@_weighted
def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)


@_weighted(should_run=_when_target_given)
def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    return multi_source_dijkstra(G, {source}, cutoff=cutoff, target=target, weight=weight)


@_weighted
def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    length, path = multi_source_dijkstra(G, sources, cutoff=cutoff, weight=weight)
    return path


@_weighted
def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    return _dijkstra_lengths(G, _sources(G, sources), cutoff, weight)


@_weighted(should_run=_when_target_given)
def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    positions = _sources(G, sources)
    if target in sources:
        return (0, [target])
    if target is None:
        return _dijkstra_paths(G, positions, cutoff, weight)
    t = _position(G, target)
    if t >= 0:
        adj = _adjacency(G, weight)
        order, dist, path = adj.dijkstra(positions, cutoff, t, "paths")
        if path is not None:
            return (dist[-1], _keys(G, path))
    raise nx.NetworkXNoPath(f"No path to {target}.")


@_weighted
def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    s = _require(G, source, f"Node {source} is not found in the graph")
    adj = _adjacency(G, weight)
    order, dist, (seen, preds) = adj.dijkstra([s], cutoff, None, "preds")
    pred = dict(zip(_keys(G, seen), (_keys(G, p) for p in preds)))
    return (pred, _distances(G, adj, order, dist))


def _all_pairs_dijkstra(G, cutoff, weight):
    """Yield `(node, (dist, paths))` for every node, as NetworkX's generators."""
    adj = _adjacency(G, weight)
    nodes = G.nodes
    for batch in _batches(len(nodes)):
        results, failed = adj.dijkstra_many(list(batch), cutoff, True)
        for s, (order, dist, parents) in zip(batch, results):
            dist = _distances(G, adj, order, dist)
            yield nodes[s], (dist, _paths(_keys(G, order), parents))
        if failed:
            raise _contradiction()


@_weighted
def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    yield from _all_pairs_dijkstra(G, cutoff, weight)


@_weighted
def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    adj = _adjacency(G, weight)
    nodes = G.nodes
    for batch in _batches(len(nodes)):
        results, failed = adj.dijkstra_lengths_many(nodes, list(batch), cutoff)
        yield from zip(map(nodes.__getitem__, batch), results)
        if failed:
            raise _contradiction()


@_weighted
def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    for n, (_, paths) in _all_pairs_dijkstra(G, cutoff, weight):
        yield n, paths


@_weighted(should_run=prefer_converted_graph)
def bidirectional_dijkstra(G, source, target, weight="weight"):
    s = _require(G, source, f"Source {source} is not in G")
    t = _require(G, target, f"Target {target} is not in G")
    if source == target:
        return (0, [source])
    found = _adjacency(G, weight, reverse=True).bidirectional_dijkstra(s, t)
    if found is None:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    length, path = found
    return (length, _keys(G, path))


# Generic


def _can_run_generic(G, source=None, target=None, weight=None, method="dijkstra"):
    if callable(weight):
        return "a callable weight is not supported; give an edge attribute name"
    return True


def _bf():
    """The Bellman-Ford module (imported late: it builds on this one)."""
    from edgecase.nx_backend.algorithms import bellman_ford

    return bellman_ford


def _check_method(method, weight):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    return "unweighted" if weight is None else method


@networkx_algorithm(can_run=_can_run_generic, should_run=_unless_all_pairs)
def shortest_path(G, source=None, target=None, weight=None, method="dijkstra"):
    method = _check_method(method, weight)
    if source is None:
        if target is None:
            if method == "unweighted":
                return all_pairs_shortest_path(G)
            if method == "dijkstra":
                return all_pairs_dijkstra_path(G, weight=weight)
            return _bf().all_pairs_bellman_ford_path(G, weight=weight)
        # Paths to `target`: search backwards (predecessors) and reverse the paths
        if method == "unweighted":
            t = _require(G, target, f"Source {target} not in G")
            return _bfs_paths(G, t, None, reverse=True)
        if method == "dijkstra":
            t = _require(G, target, f"Node {target} not found in graph")
            return _dijkstra_paths(G, [t], None, weight, reverse=True)[1]
        paths = _bf()._reverse_bellman_ford_paths(G, target, weight)
        for t in paths:
            paths[t] = list(reversed(paths[t]))
        return paths
    if target is None:
        if method == "unweighted":
            return single_source_shortest_path(G, source)
        if method == "dijkstra":
            return single_source_dijkstra_path(G, source, weight=weight)
        return _bf().single_source_bellman_ford_path(G, source, weight=weight)
    if method == "unweighted":
        return bidirectional_shortest_path(G, source, target)
    if method == "dijkstra":
        return bidirectional_dijkstra(G, source, target, weight)[1]
    return _bf().bellman_ford_path(G, source, target, weight)


@networkx_algorithm(can_run=_can_run_generic, should_run=_unless_all_pairs)
def shortest_path_length(G, source=None, target=None, weight=None, method="dijkstra"):
    method = _check_method(method, weight)
    if source is None:
        if target is None:
            if method == "unweighted":
                return all_pairs_shortest_path_length(G)
            if method == "dijkstra":
                return all_pairs_dijkstra_path_length(G, weight=weight)
            return _bf().all_pairs_bellman_ford_path_length(G, weight=weight)
        # Lengths to `target`: search backwards (predecessors)
        if method == "unweighted":
            t = _require(G, target, f"Source {target} is not in G")
            return _bfs_lengths(G, t, None, reverse=True)
        if method == "dijkstra":
            t = _require(G, target, f"Node {target} not found in graph")
            return _dijkstra_lengths(G, [t], None, weight, reverse=True)
        return _bf()._reverse_bellman_ford_lengths(G, target, weight)
    if target is None:
        if method == "unweighted":
            return single_source_shortest_path_length(G, source)
        if method == "dijkstra":
            return single_source_dijkstra_path_length(G, source, weight=weight)
        return _bf().single_source_bellman_ford_path_length(G, source, weight=weight)
    if method == "unweighted":
        return len(bidirectional_shortest_path(G, source, target)) - 1
    if method == "dijkstra":
        return dijkstra_path_length(G, source, target, weight)
    return _bf().bellman_ford_path_length(G, source, target, weight)


@networkx_algorithm(linear_time=True)
def has_path(G, source, target):
    return _bidirectional_bfs(G, source, target) is not None


_SINGLE_SOURCE_METHODS = ["unweighted", "dijkstra", "bellman-ford"]
_ALL_PAIRS_METHODS = ["floyd-warshall", "floyd-warshall-numpy"]


def _can_run_average(G, weight=None, method=None):
    if callable(weight):
        return "a callable weight is not supported; give an edge attribute name"
    if method in _ALL_PAIRS_METHODS:
        return f"method {method!r} is not implemented by edgecase"
    return True


@networkx_algorithm(can_run=_can_run_average)
def average_shortest_path_length(G, weight=None, method=None):
    if method is None:
        method = "unweighted" if weight is None else "dijkstra"
    if method not in _SINGLE_SOURCE_METHODS + _ALL_PAIRS_METHODS:
        raise ValueError(f"method not supported: {method}")
    n = len(G)
    if n == 0:
        msg = "the null graph has no paths, thus there is no average shortest path length"
        raise nx.NetworkXPointlessConcept(msg)
    if n == 1:
        return 0
    if not _adjacency(G).is_strongly_connected():
        if G.is_directed():
            raise nx.NetworkXError("Graph is not strongly connected.")
        raise nx.NetworkXError("Graph is not connected.")
    if method == "unweighted":
        total = _adjacency(G).distance_sum(False)
    elif method == "dijkstra":
        total = _adjacency(G, weight).distance_sum(True)
    else:
        # (Python's own sum over the distance dicts, in NetworkX's order)
        path_length = _bf().single_source_bellman_ford_path_length
        total = sum(d for u in G.nodes for d in path_length(G, u, weight=weight).values())
    return total / (n * (n - 1))
