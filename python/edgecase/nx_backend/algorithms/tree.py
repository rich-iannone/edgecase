"""Minimum and maximum spanning trees (Kruskal) and the condensation of a digraph.

Spanning trees use edgecase's `Graph.kruskal_mst_edges()`, which sorts the edges stably by
weight (in NetworkX's edge order) and grows the forest with a union-find, exactly as
NetworkX's Kruskal does, so the same edges are chosen among equal weights. Other
`algorithm` choices (Prim, Borůvka) are left to NetworkX, as are NaN weights.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "condensation",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
]


def _kruskal_only(algorithm):
    if algorithm != "kruskal":
        return f"algorithm={algorithm!r} is left to NetworkX"
    return True


def _edges_can_run(G, algorithm="kruskal", *args, **kwargs):
    return _kruskal_only(algorithm)


def _tree_can_run(G, weight="weight", algorithm="kruskal", *args, **kwargs):
    return _kruskal_only(algorithm)


def _spanning_edges(G, minimum, weight, keys, data):
    """The spanning-forest edges, computed now (so NotImplementedError reaches the
    dispatcher), as the tuples NetworkX yields."""
    import networkx as nx

    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")
    rg = G.rust_graph() if weight is None else G.rust_graph(weight, default=1)
    arcs = G.edge_arcs
    nodes, src, dst = G.nodes, G.src, G.dst
    multigraph = G.is_multigraph()
    keys_of = G.keys
    edge_dicts = G.edge_dicts
    if data and edge_dicts is None:
        # (only the weight was converted)
        values = None if weight is None else G.edge_values_of(weight)
    out = []
    for edge_id in rg.kruskal_mst_edges(minimum):
        arc = arcs[edge_id - 1]
        e = (nodes[src[arc] - 1], nodes[dst[arc] - 1])
        if multigraph and keys:
            e += (keys_of[arc],)
        if data:
            if edge_dicts is not None:
                d = edge_dicts[arc]
            else:
                d = {} if values is None or values[arc] is None else {weight: values[arc]}
            e += (d,)
        out.append(e)
    return out


@networkx_algorithm(can_run=_edges_can_run)
def minimum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    return iter(_spanning_edges(G, True, weight, keys, data))


@networkx_algorithm(can_run=_edges_can_run)
def maximum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    return iter(_spanning_edges(G, False, weight, keys, data))


def _spanning_tree(G, minimum, weight):
    import networkx as nx

    from edgecase.nx_backend.graph import _new_nx_graph

    edges = _spanning_edges(G, minimum, weight, keys=True, data=True)
    T = _new_nx_graph(G, nx)  # same graph class as G
    T.graph.update(G.graph)
    if G.node_dicts is not None:
        T.add_nodes_from(zip(G.nodes, G.node_dicts))
    else:
        T.add_nodes_from(G.nodes)
    T.add_edges_from(edges)
    return T


@networkx_algorithm(can_run=_tree_can_run)
def minimum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _spanning_tree(G, True, weight)


@networkx_algorithm(can_run=_tree_can_run)
def maximum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _spanning_tree(G, False, weight)


@networkx_algorithm(linear_time=True)
def condensation(G, scc=None):
    import networkx as nx

    from edgecase.nx_backend.algorithms.components import _strong_components

    if scc is None:
        scc = _strong_components(G)
    mapping = {}
    members = {}
    C = nx.DiGraph()
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    for i, component in enumerate(scc):
        members[i] = component
        mapping.update((n, i) for n in component)
    number_of_components = i + 1
    C.add_nodes_from(range(number_of_components))
    nodes = G.nodes
    component_of = [mapping[n] for n in nodes]
    C.add_edges_from(
        (cu, cv)
        for cu, cv in zip(
            (component_of[u - 1] for u in G.src), (component_of[v - 1] for v in G.dst)
        )
        if cu != cv
    )
    nx.set_node_attributes(C, members, "members")
    return C
