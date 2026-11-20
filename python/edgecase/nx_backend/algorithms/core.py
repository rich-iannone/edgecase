"""k-core decomposition: core numbers, k-cores, shells, crusts, coronas and trusses.

Core numbers come from edgecase's `Graph.get_coreness()` (Batagelj and Zaversnik's
algorithm; degrees count in- and out-edges, as NetworkX's do). The subgraph functions
keep NetworkX's own construction (`G.subgraph(nodes).copy()`) on the graph converted
back, so their results, attributes and order are NetworkX's.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = ["core_number", "k_core", "k_corona", "k_crust", "k_shell", "k_truss"]


def _self_loop_error(G, what="graph has"):
    import networkx as nx

    rg = G.rust_graph()
    if rg.count_loops():
        return nx.NetworkXNotImplemented(
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
    return None


def _core_number(G):
    error = _self_loop_error(G)
    if error is not None:
        raise error
    return G.dict_by_key(G.rust_graph().get_coreness())


@networkx_algorithm(linear_time=True)
def core_number(G):
    return _core_number(G)


def _core_subgraph(G, k_filter, k=None, core=None):
    from edgecase.nx_backend.graph import to_networkx

    if core is None:
        core = _core_number(G)
    if k is None:
        k = max(core.values())
    H = to_networkx(G)
    nodes = (v for v in core if k_filter(H, v, k, core))
    return H.subgraph(nodes).copy()


@networkx_algorithm(linear_time=True)
def k_core(G, k=None, core_number=None):
    return _core_subgraph(G, lambda H, v, k, c: c[v] >= k, k, core_number)


@networkx_algorithm(linear_time=True)
def k_shell(G, k=None, core_number=None):
    return _core_subgraph(G, lambda H, v, k, c: c[v] == k, k, core_number)


@networkx_algorithm(linear_time=True)
def k_crust(G, k=None, core_number=None):
    from edgecase.nx_backend.graph import to_networkx

    if core_number is None:
        core_number = _core_number(G)
    if k is None:
        k = max(core_number.values()) - 1
    nodes = (v for v in core_number if core_number[v] <= k)
    return to_networkx(G).subgraph(nodes).copy()


@networkx_algorithm(linear_time=True)
def k_corona(G, k, core_number=None):
    def corona(H, v, k, c):
        return c[v] == k and k == sum(1 for w in H[v] if c[w] >= k)

    return _core_subgraph(G, corona, k, core_number)


@networkx_algorithm
def k_truss(G, k):
    import networkx as nx

    from edgecase._core import k_truss as rust_k_truss
    from edgecase.nx_backend.graph import to_networkx

    error = _self_loop_error(G)
    if error is not None:
        raise error
    H = to_networkx(G).copy()
    if k >= 3:
        nodes = G.nodes
        kept = {
            frozenset((nodes[s - 1], nodes[t - 1]))
            for _, s, t in rust_k_truss(G.rust_graph(), k).edges()
        }
        H.remove_edges_from([(u, v) for u, v in H.edges if frozenset((u, v)) not in kept])
    # (for k < 3 every edge qualifies; NetworkX still drops the isolated nodes)
    H.remove_nodes_from(list(nx.isolates(H)))
    return H
