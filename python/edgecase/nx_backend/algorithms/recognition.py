"""Bipartite coloring, tree and forest recognition, and Eulerian circuits and paths.

The searches run in Rust (`nx_bipartite_color`, `ArcAdjacency.eulerian_circuit`) and follow
NetworkX's code step by step, so colorings and circuits, and their order, are NetworkX's.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from edgecase.nx_backend.algorithms.components import (
    is_connected,
    is_strongly_connected,
    is_weakly_connected,
    number_connected_components,
    number_weakly_connected_components,
)
from edgecase.nx_backend.algorithms.dfs_structure import _arc_adjacency
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "color",
    "eulerian_circuit",
    "eulerian_path",
    "has_eulerian_path",
    "is_arborescence",
    "is_bipartite",
    "is_branching",
    "is_eulerian",
    "is_forest",
    "is_semieulerian",
    "is_tree",
    "sets",
]


def _out_degrees(G):
    counts = Counter(G.src)
    return [counts[i] for i in range(1, len(G.nodes) + 1)]


def _in_degrees(G):
    counts = Counter(G.dst)
    return [counts[i] for i in range(1, len(G.nodes) + 1)]


def _degrees(G):
    """`G.degree()` values in node order (a self-loop counts twice; directed: in + out)."""
    if G.digraph_storage:
        return [a + b for a, b in zip(_out_degrees(G), _in_degrees(G))]
    counts = Counter(G.src)
    counts.update(s for s, d in zip(G.src, G.dst) if s == d)
    return [counts[i] for i in range(1, len(G.nodes) + 1)]


# Bipartite


def _color(G):
    from edgecase._core import nx_bipartite_color

    found = nx_bipartite_color(_adjacency(G, reverse=True))
    if found is None:
        raise nx.NetworkXError("Graph is not bipartite.")
    nodes = G.nodes
    order, colors = found
    result = {nodes[v]: c for v, c in zip(order, colors)}
    # (and the isolated nodes 0, as nx.isolates lists them)
    degrees = _degrees(G)
    result.update(dict.fromkeys((n for n, d in zip(nodes, degrees) if d == 0), 0))
    return result


@networkx_algorithm(name="color", linear_time=True)
def color(G):
    return _color(G)


@networkx_algorithm(linear_time=True)
def is_bipartite(G):
    try:
        _color(G)
        return True
    except nx.NetworkXError:
        return False


@networkx_algorithm(name="sets", linear_time=True)
def sets(G, top_nodes=None):
    connected = is_weakly_connected if G.is_directed() else is_connected
    if top_nodes is not None:
        X = set(top_nodes)
        Y = set(G.nodes) - X
    else:
        if not connected(G):
            msg = "Disconnected graph: Ambiguous solution for bipartite sets."
            raise nx.AmbiguousSolution(msg)
        c = _color(G)
        X = {n for n, is_top in c.items() if is_top}
        Y = {n for n, is_top in c.items() if not is_top}
    return (X, Y)


# Trees and forests


@networkx_algorithm(linear_time=True)
def is_tree(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    connected = is_weakly_connected if G.is_directed() else is_connected
    return len(G) - 1 == G.number_of_edges() and connected(G)


@networkx_algorithm(linear_time=True)
def is_forest(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    # (every component has at least one edge fewer than nodes; all have exactly that)
    count = (
        number_weakly_connected_components(G) if G.is_directed() else number_connected_components(G)
    )
    return G.number_of_edges() == len(G) - count


@networkx_algorithm(linear_time=True)
def is_arborescence(G):
    return is_tree(G) and max(_in_degrees(G)) <= 1


@networkx_algorithm(linear_time=True)
def is_branching(G):
    return is_forest(G) and max(_in_degrees(G)) <= 1


# Eulerian


def _is_eulerian(G):
    if G.is_directed():
        balanced = all(i == o for i, o in zip(_in_degrees(G), _out_degrees(G)))
        return balanced and is_strongly_connected(G)
    return all(d % 2 == 0 for d in _degrees(G)) and is_connected(G)


@networkx_algorithm(linear_time=True)
def is_eulerian(G):
    return _is_eulerian(G)


def _has_eulerian_path(G, source=None):
    if _is_eulerian(G):
        return True
    if G.is_directed():
        ins = _in_degrees(G)
        outs = _out_degrees(G)
        if source is not None:
            s = G.key_to_id[source] - 1
            if outs[s] - ins[s] != 1:
                return False
        unbalanced_ins = 0
        unbalanced_outs = 0
        for i, o in zip(ins, outs):
            if i - o == 1:
                unbalanced_ins += 1
            elif o - i == 1:
                unbalanced_outs += 1
            elif i != o:
                return False
        return unbalanced_ins <= 1 and unbalanced_outs <= 1 and is_weakly_connected(G)
    degrees = _degrees(G)
    if source is not None and degrees[G.key_to_id[source] - 1] % 2 != 1:
        return False
    return sum(d % 2 == 1 for d in degrees) == 2 and is_connected(G)


@networkx_algorithm(linear_time=True)
def has_eulerian_path(G, source=None):
    return _has_eulerian_path(G, source)


@networkx_algorithm(linear_time=True)
def is_semieulerian(G):
    return _has_eulerian_path(G) and not _is_eulerian(G)


def _walk(G, source, reverse):
    """NetworkX's Eulerian walk from `source` (a key): `(u, v, key)` per step."""
    if source not in G:
        raise nx.NetworkXError(f"Node {source} is not in the graph.")
    nodes = G.nodes
    keys = G.keys if G.multigraph else None
    steps = _arc_adjacency(G).eulerian_circuit(G.key_to_id[source] - 1, reverse)
    for u, v, arc in steps:
        yield nodes[u], nodes[v], None if keys is None else keys[arc]


def _find_path_start(G, reversed_graph):
    """NetworkX's `_find_path_start` (on the reversed graph for a directed one)."""
    nodes = G.nodes
    if G.is_directed():
        ins, outs = _in_degrees(G), _out_degrees(G)
        if reversed_graph:
            ins, outs = outs, ins
        # (has_eulerian_path and is_eulerian are the same on the reverse)
        if not _has_eulerian_path(G):
            return None
        if _is_eulerian(G):
            return nodes[0]
        v1, v2 = (i for i in range(len(nodes)) if ins[i] != outs[i])
        return nodes[v1] if outs[v1] > ins[v1] else nodes[v2]
    if not _has_eulerian_path(G):
        return None
    if _is_eulerian(G):
        return nodes[0]
    degrees = _degrees(G)
    return [n for n, d in zip(nodes, degrees) if d % 2 != 0][0]


def _source_can_run(G, source=None, keys=False):
    if source is not None and source not in G:
        try:
            iter(source)
        except TypeError:
            return True  # (`_walk` raises NetworkX's error)
        # NetworkX's degree and edge views then treat `source` as a bunch of nodes
        return f"source {source!r} is not a node but iterable; left to NetworkX"
    return True


@networkx_algorithm(can_run=_source_can_run, linear_time=True)
def eulerian_circuit(G, source=None, keys=False):
    if not _is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian.")
    if source is None:
        source = G.nodes[0]
    multigraph = G.is_multigraph()
    for u, v, k in _walk(G, source, G.is_directed()):
        if multigraph and keys:
            yield (u, v, k)
        else:
            yield (u, v)


@networkx_algorithm(can_run=_source_can_run, linear_time=True)
def eulerian_path(G, source=None, keys=False):
    if not _has_eulerian_path(G, source):
        raise nx.NetworkXError("Graph has no Eulerian paths.")
    multigraph = G.is_multigraph()
    if G.is_directed():
        # NetworkX walks G.reverse(), and picks its own start unless that is Eulerian
        if source is None or not _is_eulerian(G):
            source = _find_path_start(G, True)
        for u, v, k in _walk(G, source, True):
            if multigraph and keys:
                yield (u, v, k)
            else:
                yield (u, v)
    else:
        if source is None:
            source = _find_path_start(G, False)
        steps = list(_walk(G, source, False))
        if multigraph and keys:
            yield from reversed([(v, u, k) for u, v, k in steps])
        else:
            yield from reversed([(v, u) for u, v, _ in steps])
