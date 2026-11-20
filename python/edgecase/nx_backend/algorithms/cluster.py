"""Triangles, clustering coefficients and transitivity.

These run on edgecase's counting kernels (`Graph.count_triangles`,
`local_clustering_coefficient`, `transitivity`, `square_clustering`, `generalized_degree`),
which define degrees and triangles on the simple graph (no self-loops, parallel edges once)
as NetworkX does. Weighted clustering, and the calls NetworkX does not support on
multigraphs, are left to NetworkX.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "average_clustering",
    "clustering",
    "generalized_degree",
    "square_clustering",
    "transitivity",
    "triangles",
]


def nbunch_iter(G, nbunch):
    """NetworkX's `G.nbunch_iter(nbunch)` for an `EdgecaseGraph`."""
    import networkx as nx

    if nbunch is None:
        return iter(G.nodes)
    if nbunch in G:
        return iter([nbunch])

    def bunch_iter(nlist, key_to_id):
        try:
            for n in nlist:
                if n in key_to_id:
                    yield n
        except TypeError as err:
            exc, message = err, err.args[0]
            if "iter" in message:
                exc = nx.NetworkXError("nbunch is not a node or a sequence of nodes.")
            if "hashable" in message:
                exc = nx.NetworkXError(f"Node {n} in sequence nbunch is not a valid node.")
            raise exc

    return bunch_iter(nbunch, G.key_to_id)


def _per_node(G, values_by_id, nodes, default=0):
    """`{node: value}` for `nodes` (None: all, in node order), or the value alone for a
    single node; `default` for nodes missing from `values_by_id`"""
    if nodes is None:
        get = values_by_id.get
        return {v: get(i, default) for i, v in enumerate(G.nodes, 1)}
    key_to_id = G.key_to_id
    result = {v: values_by_id.get(key_to_id[v], default) for v in nbunch_iter(G, nodes)}
    if nodes in G:
        return result[nodes]
    return result


def _zero_as_int(x):
    """NetworkX writes `0` (an int) where a coefficient has no triangles or no room for any."""
    return 0 if x is None or x == 0 else x


def _multigraph_with_nodes(G, nodes=None):
    if G.is_multigraph() and nodes is not None:
        return "NetworkX raises for a multigraph with nodes="
    return True


@networkx_algorithm(can_run=_multigraph_with_nodes)
def triangles(G, nodes=None):
    return _per_node(G, G.rust_graph().count_triangles(), nodes)


def _simple_unweighted(G, nodes=None, weight=None, *args, **kwargs):
    if weight is not None:
        return "weighted clustering is not implemented in edgecase"
    if G.is_multigraph():
        return "NetworkX raises for multigraphs"
    return True


def _clustering(G, nodes):
    values = G.rust_graph().local_clustering_coefficient()
    return _per_node(G, {i: _zero_as_int(c) for i, c in values.items()}, nodes)


@networkx_algorithm(can_run=_simple_unweighted)
def clustering(G, nodes=None, weight=None):
    return _clustering(G, nodes)


@networkx_algorithm(can_run=_simple_unweighted)
def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    c = _clustering(G, nodes).values()  # (a single node fails here, as in NetworkX)
    if not count_zeros:
        c = [v for v in c if abs(v) > 0]
    return sum(c) / len(c)


def _undirected_simple(G, *args, **kwargs):
    if G.is_multigraph():
        return "NetworkX raises for multigraphs"
    if G.is_directed():
        return "NetworkX's definition on directed graphs differs from edgecase's"
    return True


@networkx_algorithm(can_run=_undirected_simple)
def transitivity(G):
    return _zero_as_int(G.rust_graph().transitivity())


def _undirected(G, *args, **kwargs):
    if G.is_directed():
        return "NetworkX's definition on directed graphs differs from edgecase's"
    return True


@networkx_algorithm(can_run=_undirected)
def square_clustering(G, nodes=None):
    from edgecase._core import square_clustering as rust_square_clustering

    values = {i: _zero_as_int(c) for i, c in rust_square_clustering(G.rust_graph()).items()}
    return _per_node(G, values, nodes)


def _not_multigraph(G, *args, **kwargs):
    return "NetworkX raises for multigraphs" if G.is_multigraph() else True


@networkx_algorithm(can_run=_not_multigraph)
def generalized_degree(G, nodes=None):
    from collections import Counter

    from edgecase._core import generalized_degree as rust_generalized_degree

    values = {i: Counter(c) for i, c in rust_generalized_degree(G.rust_graph()).items()}
    if nodes is None:
        return {v: values.get(i) or Counter() for i, v in enumerate(G.nodes, 1)}
    result = {v: values.get(G.key_to_id[v]) or Counter() for v in nbunch_iter(G, nodes)}
    if nodes in G:
        return result[nodes]
    return result
