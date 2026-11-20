"""Link prediction: NetworkX's `link_prediction` scores for pairs of nodes.

The scores are computed in Rust (`edgecase._core.nx_link_scores`), in batches as the
returned generator is consumed; the pairs (given, or `nx.non_edges(G)` in NetworkX's order)
and the order of the results are NetworkX's. Adamic-Adar and resource-allocation sums can
differ from NetworkX's in the last bit when three or more common neighbors contribute:
NetworkX adds those terms in the order of a Python set.

The community-based scores (Soundarajan-Hopcroft, within-inter-cluster) are left to
NetworkX.
"""

from __future__ import annotations

from itertools import islice
from math import inf

import networkx as nx

from edgecase.nx_backend.algorithms.shortest_paths import _adjacency
from edgecase.nx_backend.registry import networkx_algorithm, prefer_converted_graph

__all__ = [
    "adamic_adar_index",
    "common_neighbor_centrality",
    "jaccard_coefficient",
    "preferential_attachment",
    "resource_allocation_index",
]

_MAX_BATCH = 4096


def _should_run(G, ebunch=None, *args, **kwargs):
    """Score all non-edges (many pairs) always; a short list of pairs only on a graph that
    needs no conversion (D6), which would cost more than scoring the pairs.
    """
    if ebunch is None:
        return True
    try:
        if len(ebunch) >= len(G):
            return True
    except TypeError:
        pass
    return prefer_converted_graph(G)


def _non_edges(G):
    """`nx.non_edges(G)` for an undirected graph: the same pairs in the same order (the
    same Python set operations on the same keys, inserted in the same order).
    """
    nodes = G.nodes
    indptr, indices, _ = _adjacency(G).arcs()
    key_to_id = G.key_to_id
    remaining = set(nodes)
    while remaining:
        u = remaining.pop()
        i = key_to_id[u] - 1
        neighbors = set(map(nodes.__getitem__, indices[indptr[i] : indptr[i + 1]]))
        for v in remaining - neighbors:
            yield (u, v)


def _predictions(G, kind, ebunch, finish):
    """NetworkX's `_apply_prediction`: check `ebunch` now, then score lazily."""
    if ebunch is None:
        ebunch = _non_edges(G)
    else:
        for u, v in ebunch:
            if u not in G:
                raise nx.NodeNotFound(f"Node {u} not in G.")
            if v not in G:
                raise nx.NodeNotFound(f"Node {v} not in G.")
    return _scores(G, kind, ebunch, finish)


def _scores(G, kind, ebunch, finish):
    from edgecase._core import nx_link_scores

    adjacency = _adjacency(G)
    key_to_id = G.key_to_id
    pairs = iter(ebunch)
    size = 1
    while batch := list(islice(pairs, size)):
        us = [key_to_id[u] - 1 for u, _ in batch]
        vs = [key_to_id[v] - 1 for _, v in batch]
        values, counts = nx_link_scores(adjacency, us, vs, kind)
        for (u, v), value, count, i, j in zip(batch, values, counts, us, vs):
            yield (u, v, finish(i, j, value, count))
        size = min(_MAX_BATCH, 2 * size)


def _zero_if_none(i, j, value, count):
    """A float score, or the int 0 when nothing was added up (count 0)."""
    return 0 if count == 0 else value


@networkx_algorithm(should_run=_should_run)
def resource_allocation_index(G, ebunch=None):
    return _predictions(G, "resource_allocation", ebunch, _zero_if_none)


@networkx_algorithm(should_run=_should_run)
def jaccard_coefficient(G, ebunch=None):
    # (count is the size of the union of the two neighborhoods)
    return _predictions(G, "jaccard", ebunch, _zero_if_none)


@networkx_algorithm(should_run=_should_run)
def adamic_adar_index(G, ebunch=None):
    return _predictions(G, "adamic_adar", ebunch, _zero_if_none)


@networkx_algorithm(should_run=_should_run)
def preferential_attachment(G, ebunch=None):
    return _predictions(G, "preferential_attachment", ebunch, lambda i, j, value, count: count)


def _cnc_should_run(G, ebunch=None, alpha=0.8):
    # (with alpha != 1 NetworkX first finds all shortest-path lengths, which edgecase skips)
    return True if alpha != 1 else _should_run(G, ebunch)


@networkx_algorithm(should_run=_cnc_should_run)
def common_neighbor_centrality(G, ebunch=None, alpha=0.8):
    if alpha == 1:

        def finish(i, j, value, count):
            if i == j:
                raise nx.NetworkXAlgorithmError("Self loops are not supported")
            return count

    else:
        adjacency = _adjacency(G)
        n = len(G)
        lengths = {}  # source position -> {position: path length}, for the latest source

        def finish(i, j, value, count):
            if i == j:
                raise nx.NetworkXAlgorithmError("Self loops are not supported")
            dist = lengths.get(i)
            if dist is None:
                order, levels, _ = adjacency.bfs_levels(i)
                lengths.clear()
                dist = lengths[i] = dict(zip(order, levels))
            path_len = dist.get(j, inf)
            return alpha * count + (1 - alpha) * n / path_len

    return _predictions(G, "common_neighbors", ebunch, finish)
