"""Connected, weakly connected and strongly connected components."""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "connected_components",
    "is_connected",
    "is_strongly_connected",
    "is_weakly_connected",
    "node_connected_component",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "strongly_connected_components",
    "weakly_connected_components",
]


def _null_graph_error():
    import networkx as nx

    return nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")


def _components(G):
    """Components (ignoring edge direction) as sets of node keys, in NetworkX's order.

    NetworkX lists components in order of their first node in node order; node IDs follow
    node order, so that is the order of each component's smallest ID.
    """
    components = G.rust_graph().weakly_connected_components()
    components.sort(key=min)
    nodes = G.nodes
    return [{nodes[i - 1] for i in component} for component in components]


def _component_of(G, n):
    node_id = G.key_to_id[n]  # KeyError for a missing node, as in NetworkX
    nodes = G.nodes
    for component in G.rust_graph().weakly_connected_components():
        if node_id in component:
            return {nodes[i - 1] for i in component}
    raise AssertionError("every node is in a component")  # pragma: no cover


@networkx_algorithm(linear_time=True)
def connected_components(G):
    yield from _components(G)


@networkx_algorithm(linear_time=True)
def number_connected_components(G):
    return G.rust_graph().count_weakly_connected_components()


@networkx_algorithm(linear_time=True)
def is_connected(G):
    if len(G) == 0:
        raise _null_graph_error()
    return G.rust_graph().count_weakly_connected_components() == 1


@networkx_algorithm(linear_time=True)
def node_connected_component(G, n):
    return _component_of(G, n)


@networkx_algorithm(linear_time=True)
def weakly_connected_components(G):
    yield from _components(G)


@networkx_algorithm(linear_time=True)
def number_weakly_connected_components(G):
    return G.rust_graph().count_weakly_connected_components()


@networkx_algorithm(linear_time=True)
def is_weakly_connected(G):
    if len(G) == 0:
        raise _null_graph_error()
    return G.rust_graph().count_weakly_connected_components() == 1


def _strong_components(G):
    from edgecase._core import nx_strongly_connected_components

    nodes = G.nodes
    for component in nx_strongly_connected_components(G.rust_graph()):
        yield {nodes[i - 1] for i in component}


@networkx_algorithm(linear_time=True)
def strongly_connected_components(G):
    yield from _strong_components(G)


@networkx_algorithm(linear_time=True)
def number_strongly_connected_components(G):
    from edgecase._core import nx_strongly_connected_components

    return len(nx_strongly_connected_components(G.rust_graph()))


@networkx_algorithm(linear_time=True)
def is_strongly_connected(G):
    if len(G) == 0:
        raise _null_graph_error()
    return len(next(_strong_components(G))) == len(G)
