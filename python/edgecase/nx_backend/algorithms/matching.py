"""Matchings: maximal (greedy), maximum- and minimum-weight (blossom), and checks.

Maximum- and minimum-weight matchings come from edgecase's blossom algorithm
(`blossom_max_weight_matching`), the same Edmonds algorithm NetworkX ports; the matchings
have NetworkX's weight and cardinality, but where several matchings tie, edgecase may pick
a different one. Edges come out oriented as in `G.edges()`. Non-finite weights are left to
NetworkX.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "is_matching",
    "is_maximal_matching",
    "is_perfect_matching",
    "max_weight_matching",
    "maximal_matching",
    "min_weight_matching",
]


@networkx_algorithm(linear_time=True)
def maximal_matching(G):
    nodes, src, dst = G.nodes, G.src, G.dst
    matching = set()
    matched = set()
    for k in G.edge_arcs:
        u, v = src[k], dst[k]
        if u not in matched and v not in matched and u != v:
            matching.add((nodes[u - 1], nodes[v - 1]))
            matched.update((u, v))
    return matching


def _blossom(G, weight, maxcardinality, minimize):
    from edgecase._core import blossom_max_weight_matching

    rg = G.rust_graph() if weight is None else G.rust_graph(weight, default=1)
    try:
        edge_ids = blossom_max_weight_matching(rg, maxcardinality, weight is not None, minimize)
    except ValueError as exc:
        raise NotImplementedError(f"weights edgecase cannot use: {exc}") from exc
    nodes, src, dst, arcs = G.nodes, G.src, G.dst, G.edge_arcs
    return {(nodes[src[arcs[e - 1]] - 1], nodes[dst[arcs[e - 1]] - 1]) for e in edge_ids}


@networkx_algorithm
def max_weight_matching(G, maxcardinality=False, weight="weight"):
    return _blossom(G, weight, maxcardinality, minimize=False)


@networkx_algorithm
def min_weight_matching(G, weight="weight"):
    if G.number_of_edges() == 0:
        return set()
    return _blossom(G, weight, True, minimize=True)


# Checks


def _pairs(G):
    """`(u, v)` node-ID pairs `G.has_edge(u, v)` is true for."""
    pairs = G.__dict__.get("_edge_pairs")
    if pairs is None:
        pairs = set(zip(G.src, G.dst))
        G._edge_pairs = pairs
    return pairs


def _check_edges(G, matching):
    """NetworkX's shared loop: the matched node IDs, or False if not a matching."""
    import networkx as nx
    from networkx.algorithms.matching import matching_dict_to_set

    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)
    key_to_id = G.key_to_id
    pairs = _pairs(G)
    nodes = set()
    edges = set()
    for edge in matching:
        if len(edge) != 2:
            raise nx.NetworkXError(f"matching has non-2-tuple edge {edge}")
        u, v = edge
        if u not in G or v not in G:
            raise nx.NetworkXError(f"matching contains edge {edge} with node not in G")
        if u == v:
            return False, None, None
        uid, vid = key_to_id[u], key_to_id[v]
        if (uid, vid) not in pairs:
            return False, None, None
        if uid in nodes or vid in nodes:
            return False, None, None
        nodes.update((uid, vid))
        edges.update(((uid, vid), (vid, uid)))
    return True, nodes, edges


@networkx_algorithm(linear_time=True)
def is_matching(G, matching):
    return _check_edges(G, matching)[0]


@networkx_algorithm(linear_time=True)
def is_maximal_matching(G, matching):
    ok, nodes, edges = _check_edges(G, matching)
    if not ok:
        return False
    src, dst = G.src, G.dst
    for k in G.edge_arcs:
        u, v = src[k], dst[k]
        if (u, v) not in edges and u not in nodes and v not in nodes and u != v:
            return False
    return True


@networkx_algorithm(linear_time=True)
def is_perfect_matching(G, matching):
    ok, nodes, _ = _check_edges(G, matching)
    return ok and len(nodes) == len(G)
