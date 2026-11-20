"""Degree, eigenvector and Katz centrality, edge betweenness and the subset betweenness
functions, with NetworkX's exact results.

The power iterations and Brandes' path counting run in Rust (`edgecase._core.nx_*` over the
graph's `PathAdjacency`) and follow NetworkX's own code operation by operation, including
its summation order and CPython's `math.hypot`, so the floats are equal to the bit.
"""

from __future__ import annotations

import inspect
from collections import Counter

import networkx as nx

from edgecase.nx_backend.algorithms.centrality import _rescale
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "degree_centrality",
    "edge_betweenness_centrality",
    "edge_betweenness_centrality_subset",
    "betweenness_centrality_subset",
    "eigenvector_centrality",
    "in_degree_centrality",
    "katz_centrality",
    "out_degree_centrality",
]

_CALLABLE = "a callable weight is not supported; give an edge attribute name"


def _bound(func):
    """A function binding `func`'s arguments by name (None if they do not bind)."""
    signature = inspect.signature(func)

    def bind(args, kwargs):
        try:
            return signature.bind(*args, **kwargs).arguments
        except TypeError:
            return None

    return bind


def _register(check):
    """Register a function with `can_run` given by `check(arguments)` (True or a reason)."""

    def decorate(func):
        bind = _bound(func)

        def can_run(*args, **kwargs):
            arguments = bind(args, kwargs)
            return True if arguments is None else check(arguments)

        return networkx_algorithm(func, can_run=can_run)

    return decorate


def _weight_check(arguments):
    return _CALLABLE if callable(arguments.get("weight")) else True


# Degree centrality


def _degrees(G, kind="degree"):
    """NetworkX's `G.degree()` (or `in_degree`/`out_degree`) values in node order."""
    out_counts = Counter(G.src)
    if kind == "out":
        counts = out_counts
    elif kind == "in":
        counts = Counter(G.dst)
    elif G.digraph_storage:
        counts = out_counts + Counter(G.dst)
    else:
        # Each undirected edge is an arc from both ends; a self-loop is one arc but counts
        # twice
        counts = out_counts
        counts.update(s for s, d in zip(G.src, G.dst) if s == d)
    return [counts[i] for i in range(1, len(G.nodes) + 1)]


def _degree_centrality(G, kind):
    if len(G) <= 1:
        return {n: 1 for n in G.nodes}
    s = 1.0 / (len(G) - 1.0)
    return {n: d * s for n, d in zip(G.nodes, _degrees(G, kind))}


@networkx_algorithm(linear_time=True)
def degree_centrality(G):
    return _degree_centrality(G, "degree")


@networkx_algorithm(linear_time=True)
def in_degree_centrality(G):
    return _degree_centrality(G, "in")


@networkx_algorithm(linear_time=True)
def out_degree_centrality(G):
    return _degree_centrality(G, "out")


# Eigenvector and Katz centrality


def _start_check(arguments):
    """`nstart` must name exactly the graph's nodes (NetworkX fails on anything else)."""
    nstart = arguments.get("nstart")
    if nstart is not None:
        G = arguments["G"]
        try:
            if set(nstart) != set(G):
                return "nstart does not have exactly the graph's nodes"
        except TypeError:
            return "nstart is not a dict of nodes"
    return True


def _positions(G, keys):
    key_to_id = G.key_to_id
    return [key_to_id[k] - 1 for k in keys]


def _by_position(order, values, n):
    vector = [0.0] * n
    for position, value in zip(order, values):
        vector[position] = float(value)
    return vector


@_register(_start_check)
def eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight=None):
    from edgecase._core import nx_eigenvector_centrality

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("cannot compute centrality for the null graph")
    if nstart is None:
        nstart = {v: 1 for v in G.nodes}
    if all(v == 0 for v in nstart.values()):
        raise nx.NetworkXError("initial vector cannot have all zero values")
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    keys = list(x)
    order = _positions(G, keys)
    # (NetworkX weighs a step by the edge's `weight` attribute only for a truthy `weight`)
    adjacency = _adjacency(G, weight if weight else None)
    result = nx_eigenvector_centrality(
        adjacency, order, _by_position(order, x.values(), len(G)), max_iter, tol
    )
    if result is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return {k: result[p] for k, p in zip(keys, order)}


@_register(_start_check)
def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
):
    from edgecase._core import nx_katz_centrality

    if len(G) == 0:
        return {}
    nodes = G.nodes
    x = {n: 0 for n in nodes} if nstart is None else nstart
    try:
        b = dict.fromkeys(nodes, float(beta))
    except (TypeError, ValueError, AttributeError) as err:
        b = beta
        if set(beta) != set(nodes):
            raise nx.NetworkXError("beta dictionary must have a value for every node") from err
    keys = list(x)
    order = _positions(G, keys)
    n = len(nodes)
    result = nx_katz_centrality(
        _adjacency(G, weight),
        order,
        _by_position(order, x.values(), n),
        [float(b[v]) for v in nodes],
        alpha,
        max_iter,
        tol,
        bool(normalized),
    )
    if result is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return {k: result[p] for k, p in zip(keys, order)}


# Edge betweenness and the subset betweenness functions


def _brandes(G, sources, weight, mode, targets=None):
    """`nx_brandes()` from `sources` (node keys, in order): `(edge dict, node sums)`."""
    from edgecase._core import nx_brandes

    nodes = G.nodes
    key_to_id = G.key_to_id
    positions = [key_to_id[s] - 1 for s in sources]
    mask = None
    if targets is not None:
        mask = [False] * len(nodes)
        for t in targets:
            position = key_to_id.get(t) if _hashable(t) else None
            if position is not None:
                mask[position - 1] = True
    pairs, edge_values, node_values = nx_brandes(
        _adjacency(G, weight), positions, weight is not None, mode, mask
    )
    edges = dict(zip(((nodes[u], nodes[v]) for u, v in pairs), edge_values))
    return edges, node_values


def _hashable(x):
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _add_edge_keys(G, betweenness, weight=None):
    """NetworkX's `_add_edge_keys`: share each multigraph edge's betweenness among its
    parallel edges of least weight.
    """
    nodes = G.nodes
    src, dst, keys = G.src, G.dst, G.keys
    values = None if weight is None else G.edge_values_of(weight, 1)
    edge_bc = {}
    groups = []  # per (u, v): [(key, weight)] in G.edges order
    last = None
    for i in G.edge_arcs:
        pair = (src[i], dst[i])
        if pair != last:
            groups.append((pair, []))
            last = pair
        groups[-1][1].append((keys[i], 1 if values is None else values[i]))
    for (u_id, v_id), members in groups:
        u = nodes[u_id - 1]
        v = nodes[v_id - 1]
        for k, _ in members:
            edge_bc[u, v, k] = 0.0
    for (u_id, v_id), members in groups:
        u = nodes[u_id - 1]
        v = nodes[v_id - 1]
        wt = min(w for _, w in members)
        least = [k for k, w in members if w == wt]
        bc = betweenness[u, v] / len(least)
        for k in least:
            edge_bc[u, v, k] = bc
    return edge_bc


@_register(_weight_check)
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    nodes = G.nodes
    if k is None:
        sources = nodes
        sampled = None
    else:
        # The same sample NetworkX draws (`seed` is a random.Random by now)
        sources = sampled = seed.sample(list(nodes), k)
    betweenness, _ = _brandes(G, sources, weight, "edges")
    betweenness = _rescale(
        betweenness,
        len(nodes),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=True,
        sampled_nodes=sampled,
    )
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)
    return betweenness


@_register(_weight_check)
def betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    _, node_values = _brandes(G, list(sources), weight, "nodes_subset", set(targets))
    b = dict(zip(G.nodes, node_values))
    return _rescale(
        b,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=False,
        sampled_nodes=None,
    )


@_register(_weight_check)
def edge_betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    b, _ = _brandes(G, list(sources), weight, "edges_subset", set(targets))
    b = _rescale(
        b,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=True,
        sampled_nodes=None,
    )
    if G.is_multigraph():
        b = _add_edge_keys(G, b, weight=weight)
    return b
