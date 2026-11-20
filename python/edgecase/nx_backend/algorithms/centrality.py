"""Betweenness centrality and PageRank."""

from __future__ import annotations

import math

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = ["betweenness_centrality", "pagerank"]


def _rescale(betweenness, n, *, normalized, directed, endpoints, sampled_nodes):
    """NetworkX's rescaling of accumulated betweenness (`betweenness._rescale`)."""
    k = None if sampled_nodes is None else len(sampled_nodes)
    # Valid (s, t) pairs with s != t that a path through v can join; v is not t unless
    # endpoints count
    N = n if endpoints else n - 1
    if N < 2:
        return betweenness
    K_source = N if k is None else k
    if k is None or endpoints:
        if normalized:
            scale = 1 / (K_source * (N - 1))
        else:
            # Undirected graphs count each unordered pair {s, t} once
            correction = 1 if directed else 2
            scale = N / (K_source * correction)
        if scale != 1:
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness
    # Sampled sources without endpoints: a source cannot be its own (s, t) pair
    if normalized:
        scale_source = 1 / ((K_source - 1) * (N - 1)) if K_source > 1 else math.nan
        scale_nonsource = 1 / (K_source * (N - 1))
    else:
        correction = 1 if directed else 2
        scale_source = N / ((K_source - 1) * correction) if K_source > 1 else math.nan
        scale_nonsource = N / (K_source * correction)
    sampled_nodes = set(sampled_nodes)
    for v in betweenness:
        betweenness[v] *= scale_source if v in sampled_nodes else scale_nonsource
    return betweenness


@networkx_algorithm
def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    from edgecase._core import nx_betweenness_centrality

    nodes = G.nodes
    if k == len(nodes):
        k = None
    if k is None:
        sampled = None
        sources = None
    else:
        # The same sample NetworkX draws (`seed` is a random.Random by now)
        sampled = seed.sample(list(nodes), k)
        key_to_id = G.key_to_id
        sources = [key_to_id[v] for v in sampled]
    if weight is None:
        rg = G.rust_graph()
    else:
        rg = G.rust_graph(weight, default=1.0)
    raw = nx_betweenness_centrality(rg, sources, weighted=weight is not None, endpoints=endpoints)
    return _rescale(
        dict(zip(nodes, raw)),
        len(nodes),
        normalized=normalized,
        directed=G.is_directed(),
        endpoints=endpoints,
        sampled_nodes=sampled,
    )


@networkx_algorithm
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    import networkx as nx

    from edgecase._core import nx_pagerank

    nodes = G.nodes
    if not nodes:
        return {}

    def per_node(values):
        return None if values is None else [float(values.get(n, 0)) for n in nodes]

    p = per_node(personalization)
    if p is not None and sum(p) == 0:
        raise ZeroDivisionError
    rg = G.rust_graph() if weight is None else G.rust_graph(weight, default=1.0)
    ranks = nx_pagerank(
        rg,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        weighted=weight is not None,
        personalization=p,
        nstart=per_node(nstart),
        dangling=per_node(dangling),
    )
    if ranks is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(nodes, ranks))
