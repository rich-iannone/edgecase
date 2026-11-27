"""Relabeling, ego graphs and small graph predicates.

`relabel_nodes` (with `copy=True`) and `convert_node_labels_to_integers` return edgecase
graphs, so a chain of NetworkX calls on an edgecase graph stays on edgecase. They are built
the way NetworkX builds them (the same node, neighbor and edge-key order, merged nodes and
renumbered keys included), so converting the result back gives NetworkX's graph exactly.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "convert_node_labels_to_integers",
    "ego_graph",
    "has_cycle",
    "is_empty",
    "is_negatively_weighted",
    "is_weighted",
    "relabel_nodes",
]


def _relabel_copy(H, mapping):
    """NetworkX's `_relabel_copy` (on a NetworkX graph)."""
    R = H.__class__()
    R.add_nodes_from(mapping.get(n, n) for n in H)
    R._node.update((mapping.get(n, n), d.copy()) for n, d in H.nodes.items())
    if H.is_multigraph():
        new_edges = [
            (mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
            for (n1, n2, k, d) in H.edges(keys=True, data=True)
        ]
        # conflicting edge keys get renumbered, as in NetworkX
        undirected = not H.is_directed()
        seen_edges = set()
        for i, (source, target, key, data) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, int | float):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)
        R.add_edges_from(new_edges)
    else:
        R.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in H.edges(data=True)
        )
    R.graph.update(H.graph)
    return R


def _as_edgecase(R):
    from edgecase.nx_backend.graph import from_networkx

    return from_networkx(
        R, preserve_edge_attrs=True, preserve_node_attrs=True, preserve_graph_attrs=True
    )


def _copy_only(G, mapping, copy=True):
    return True if copy else "relabeling in place mutates the input graph"


@networkx_algorithm(can_run=_copy_only)
def relabel_nodes(G, mapping, copy=True):
    from edgecase.nx_backend.graph import to_networkx

    H = to_networkx(G)
    m = {n: mapping(n) for n in H} if callable(mapping) else mapping
    return _as_edgecase(_relabel_copy(H, m))


@networkx_algorithm
def convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None):
    import networkx as nx

    from edgecase.nx_backend.algorithms.assortativity import _degrees
    from edgecase.nx_backend.graph import to_networkx

    nodes = G.nodes
    N = len(nodes) + first_label
    if ordering == "default":
        mapping = dict(zip(nodes, range(first_label, N)))
    elif ordering == "sorted":
        mapping = dict(zip(sorted(nodes), range(first_label, N)))
    elif ordering in ("increasing degree", "decreasing degree"):
        degrees = _degrees(G, None, "all")
        dv_pairs = [(d, n) for n, d in zip(nodes, degrees)]
        dv_pairs.sort()
        if ordering == "decreasing degree":
            dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    R = _relabel_copy(to_networkx(G), mapping)
    if label_attribute is not None:
        nx.set_node_attributes(R, {v: k for k, v in mapping.items()}, label_attribute)
    return _as_edgecase(R)


def _ego_can_run(G, n, radius=1, center=True, undirected=False, distance=None):
    if undirected and G.is_directed():
        return "the undirected neighborhood of a directed graph is left to NetworkX"
    return True


@networkx_algorithm(can_run=_ego_can_run)
def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    from edgecase.nx_backend.algorithms.shortest_paths import (
        single_source_dijkstra,
        single_source_shortest_path_length,
    )
    from edgecase.nx_backend.graph import to_networkx

    if distance is not None:
        sp, _ = single_source_dijkstra(G, n, None, radius, distance)
    else:
        sp = single_source_shortest_path_length(G, n, radius)
    H = to_networkx(G).subgraph(sp).copy()
    if not center:
        H.remove_node(n)
    return H


@networkx_algorithm(linear_time=True)
def has_cycle(G):
    import networkx as nx

    from edgecase._core import topological_generations_partial

    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
    return not topological_generations_partial(G.rust_graph())[1]


@networkx_algorithm(linear_time=True)
def is_empty(G):
    return not G.src


def _edge_data(G, edge):
    """`G.get_edge_data(*edge)`: the edge's attribute dict (a key dict for a multigraph's
    `(u, v)`), or None."""
    key_to_id = G.key_to_id
    u, v, *key = edge
    try:
        uid, vid = key_to_id[u], key_to_id[v]
    except (KeyError, TypeError):
        return None
    src, dst = G.src, G.dst
    arcs = [k for k in range(len(src)) if src[k] == uid and dst[k] == vid]
    if not arcs:
        return None
    dicts = G.edge_dicts
    if not G.is_multigraph():
        return dicts[arcs[0]]
    if key:
        for k in arcs:
            if G.keys[k] == key[0]:
                return dicts[k]
        return None
    return {G.keys[k]: dicts[k] for k in arcs}


@networkx_algorithm(linear_time=True)
def is_weighted(G, edge=None, weight="weight"):
    import networkx as nx

    if edge is not None:
        data = _edge_data(G, edge)
        if data is None:
            raise nx.NetworkXError(f"Edge {edge!r} does not exist.")
        return weight in data
    if not G.src:
        return False
    return all(weight in G.edge_dicts[k] for k in G.edge_arcs)


@networkx_algorithm(linear_time=True)
def is_negatively_weighted(G, edge=None, weight="weight"):
    import networkx as nx

    if edge is not None:
        if G.edge_dicts is None:
            raise NotImplementedError("needs the edge attributes")
        data = _edge_data(G, edge)
        if data is None:
            raise nx.NetworkXError(f"Edge {edge!r} does not exist.")
        return weight in data and data[weight] < 0
    if not G.src:
        return False
    from edgecase.nx_backend.graph import MISSING

    if weight is None and G.edge_dicts is None:
        return False  # (no attribute is named None)
    values = G.edge_values_of(weight, MISSING)
    return any(values[k] is not MISSING and values[k] < 0 for k in G.edge_arcs)
