"""Cliques: maximal cliques, all cliques, clique numbers and the maximum-weight clique.

`find_cliques` follows NetworkX's search, whose order comes from Python sets; the Rust
search (`edgecase._core.CliqueSearch`) models CPython's set tables with the node keys' own
hashes, so the cliques and the order of their nodes are NetworkX's. Results are produced in
batches as the generators are consumed.
"""

from __future__ import annotations

from collections import defaultdict

from edgecase.nx_backend.algorithms.recognition import _degrees
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "enumerate_all_cliques",
    "find_cliques",
    "find_cliques_recursive",
    "max_weight_clique",
    "node_clique_number",
]

_MAX_BATCH = 4096


def _start(G, nodes):
    """A `CliqueSearch` for `find_cliques(G, nodes)`, and whether `nodes` form a clique."""
    from edgecase._core import CliqueSearch

    key_to_id = G.key_to_id
    n = len(G)
    positions = None
    if nodes is not None:
        positions = []
        for v in nodes:
            try:
                positions.append(key_to_id[v] - 1)
            except (KeyError, TypeError):
                positions.append(n)  # (not a node: not a candidate either)
    hashes = [hash(v) for v in G.nodes]
    return CliqueSearch.start(_adjacency(G), hashes, positions)


def _cliques(G, search):
    nodes = G.nodes
    size = 16
    while batch := search.next_batch(size):
        for clique in batch:
            yield [nodes[v] for v in clique]
        size = min(_MAX_BATCH, 2 * size)


@networkx_algorithm
def find_cliques(G, nodes=None):
    if len(G) == 0:
        return
    search, ok = _start(G, nodes)
    if not ok:
        raise ValueError(f"The given `nodes` {nodes} do not form a clique")
    yield from _cliques(G, search)


@networkx_algorithm
def find_cliques_recursive(G, nodes=None):
    # (the recursive search visits the cliques in the same order as find_cliques)
    if len(G) == 0:
        return iter([])
    search, ok = _start(G, nodes)
    if not ok:
        raise ValueError(f"The given `nodes` {nodes} do not form a clique")
    return _cliques(G, search)


@networkx_algorithm
def enumerate_all_cliques(G):
    from edgecase._core import AllCliques

    nodes = G.nodes
    search = AllCliques(_adjacency(G))
    size = 16
    while batch := search.next_batch(size):
        for clique in batch:
            yield [nodes[v] for v in clique]
        size = min(_MAX_BATCH, 2 * size)


def _largest_clique_with(G, node):
    """The size of the largest clique containing `node` (as NetworkX finds it: the largest
    maximal clique of `node`'s ego graph, which always contains `node`)."""
    search, _ = _start(G, [node])
    return max(len(c) for c in _cliques(G, search))


@networkx_algorithm
def node_clique_number(G, nodes=None, cliques=None, separate_nodes=False):
    if cliques is None:
        if nodes is not None:
            if nodes in G:
                return _largest_clique_with(G, nodes)
            return {n: _largest_clique_with(G, n) for n in nodes}
        cliques = list(find_cliques(G))
    if nodes in G:
        return max(len(c) for c in cliques if nodes in c)
    size_for_n = defaultdict(int)
    for c in cliques:
        size_of_c = len(c)
        for n in c:
            if size_for_n[n] < size_of_c:
                size_for_n[n] = size_of_c
    if nodes is None:
        return size_for_n
    return {n: size_for_n[n] for n in nodes}


@networkx_algorithm
def max_weight_clique(G, weight="weight"):
    from edgecase._core import nx_max_weight_clique
    from edgecase.nx_backend.graph import MISSING

    nodes = G.nodes
    if weight is None:
        weights = [1] * len(nodes)
    else:
        weights = G.node_values_of(weight, MISSING)
        for v, w in zip(nodes, weights):
            if w is MISSING:
                raise KeyError(f"Node {v!r} does not have the requested weight field.")
            if not isinstance(w, int):
                raise ValueError(f"The {weight!r} field of node {v!r} is not an integer.")
    degrees = _degrees(G)
    order = sorted(range(len(nodes)), key=degrees.__getitem__, reverse=True)
    order = [v for v in order if weights[v] > 0]
    clique, total = nx_max_weight_clique(_adjacency(G), order, [int(w) for w in weights])
    return ([nodes[v] for v in clique], total)
