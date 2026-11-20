"""Bridges, biconnected components, chains, cycle bases, girth, edge DFS and find_cycle.

The depth-first searches run in Rust (`edgecase._core.nx_*` in `nx_structure.rs`) and follow
NetworkX's code step by step, so components, chains, cycles and their order are NetworkX's.
Like NetworkX's, the functions that yield their results are generators that raise errors
when iteration starts.
"""

from __future__ import annotations

from itertools import chain
from math import inf

import networkx as nx

from edgecase.nx_backend.algorithms.distance_measures import _nbunch_iter
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency, _position
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "articulation_points",
    "biconnected_component_edges",
    "biconnected_components",
    "bridges",
    "chain_decomposition",
    "cycle_basis",
    "edge_dfs",
    "find_cycle",
    "girth",
    "has_bridges",
    "is_biconnected",
    "local_bridges",
]


def _edge_pairs(G):
    """An undirected graph's edges as `G.edges()` lists them (from the end listed first),
    each pair of nodes once: `[u_id, v_id, number of parallel edges]`.
    """
    src, dst = G.src, G.dst
    pairs = []
    last = None
    for i in G.edge_arcs:
        pair = (src[i], dst[i])
        if pair == last:
            pairs[-1][2] += 1
        else:
            pairs.append([pair[0], pair[1], 1])
            last = pair
    return pairs


# Biconnected components


def _biconnected(G, components):
    from edgecase._core import nx_biconnected_dfs

    return nx_biconnected_dfs(_adjacency(G), components)


@networkx_algorithm(linear_time=True)
def biconnected_component_edges(G):
    nodes = G.nodes
    for comp in _biconnected(G, True)[0]:
        yield [(nodes[u], nodes[v]) for u, v in comp]


@networkx_algorithm(linear_time=True)
def biconnected_components(G):
    nodes = G.nodes
    for comp in _biconnected(G, True)[0]:
        yield set(chain.from_iterable((nodes[u], nodes[v]) for u, v in comp))


@networkx_algorithm(linear_time=True)
def is_biconnected(G):
    bccs = biconnected_components(G)
    try:
        bcc = next(bccs)
    except StopIteration:
        return False
    try:
        next(bccs)
    except StopIteration:
        return len(bcc) == len(G)
    else:
        return False


@networkx_algorithm(linear_time=True)
def articulation_points(G):
    nodes = G.nodes
    seen = set()
    for a in _biconnected(G, False)[1]:
        if a not in seen:
            seen.add(a)
            yield nodes[a]


# Chains and bridges


def _chains(G, root):
    """The chains of `chain_decomposition(G, root)` as position pairs."""
    from edgecase._core import nx_chain_decomposition

    r = None
    if root is not None:
        r = _position(G, root)
        if r < 0:
            raise nx.NodeNotFound(f"Root node {root} is not in graph")
    return nx_chain_decomposition(_adjacency(G), r), r


@networkx_algorithm(linear_time=True)
def chain_decomposition(G, root=None):
    nodes = G.nodes
    chains, _ = _chains(G, root)
    for chain_ in chains:
        yield [(nodes[u], nodes[v]) for u, v in chain_]


@networkx_algorithm(linear_time=True)
def bridges(G, root=None):
    # (Every edge that is in no chain is a bridge; for a multigraph NetworkX decomposes
    # nx.Graph(G), whose chains cover the same edges)
    chains, r = _chains(G, root)
    chain_edges = set(chain.from_iterable(chains))
    component = None
    if r is not None:
        component = set(_adjacency(G).bfs_levels(r)[0])
    multigraph = G.is_multigraph()
    nodes = G.nodes
    for u_id, v_id, count in _edge_pairs(G):
        u, v = u_id - 1, v_id - 1
        if component is not None and u not in component:
            continue
        if (u, v) not in chain_edges and (v, u) not in chain_edges:
            if multigraph and count > 1:
                continue
            yield (nodes[u], nodes[v])


@networkx_algorithm(linear_time=True)
def has_bridges(G, root=None):
    try:
        next(bridges(G, root=root))
    except StopIteration:
        return False
    else:
        return True


def _can_run_local_bridges(G, with_span=True, weight=None):
    if callable(weight):
        return "a callable weight is not supported; give an edge attribute name"
    return True


@networkx_algorithm(can_run=_can_run_local_bridges)
def local_bridges(G, with_span=True, weight=None):
    from edgecase._core import nx_local_bridges

    pairs = _edge_pairs(G)
    us = [u - 1 for u, _, _ in pairs]
    vs = [v - 1 for _, v, _ in pairs]
    span = with_span is True
    adjacency = _adjacency(G, weight)
    flags, spans = nx_local_bridges(adjacency, us, vs, span)
    nodes = G.nodes
    integral = adjacency.integral
    for u, v, flag, length in zip(us, vs, flags, spans):
        if not flag:
            continue
        if not span:
            yield (nodes[u], nodes[v])
        elif length == inf:
            yield (nodes[u], nodes[v], inf)
        else:
            yield (nodes[u], nodes[v], int(length) if integral else length)


# Cycles


@networkx_algorithm(linear_time=True)
def cycle_basis(G, root=None):
    from edgecase._core import nx_cycle_basis

    r = None
    if root is not None:
        r = _position(G, root)
        if r < 0:
            raise KeyError(root)  # (NetworkX looks up G[root])
    return nx_cycle_basis(_adjacency(G), G.nodes, r)


@networkx_algorithm
def girth(G):
    from edgecase._core import nx_girth

    length = nx_girth(_adjacency(G))
    return inf if length is None else length


# Edge-by-edge depth-first search and find_cycle


def _arc_adjacency(G):
    """The graph's `ArcAdjacency` (every arc on its own), built on first use."""
    from edgecase._core import ArcAdjacency

    arcs = G.__dict__.get("_arc_adjacency")
    if arcs is None:
        pred_nodes = pred_counts = preds = None
        if G.directed and getattr(G, "pred_order", None) is not None:
            get_id = G.key_to_id.__getitem__
            nodes_, pred_counts, pred_keys = G.pred_order
            pred_nodes = list(map(get_id, nodes_))
            preds = list(map(get_id, pred_keys))
        arcs = G._arc_adjacency = ArcAdjacency(
            len(G.nodes), G.src, G.dst, G.directed, pred_nodes, pred_counts, preds
        )
    return arcs


def _edge_dfs(G, source, orientation):
    """NetworkX's `edge_dfs` edges: `(u, v[, key][, "forward" or "reverse"])`."""
    starts = list(_nbunch_iter(G, source))
    if not starts:
        return
    directed = G.is_directed()
    if directed and orientation not in (None, "original", "reverse", "ignore"):
        raise nx.NetworkXError("invalid orientation argument.")
    key_to_id = G.key_to_id
    found = _arc_adjacency(G).edge_dfs(
        [key_to_id[s] - 1 for s in starts], orientation if directed else None
    )
    nodes, src, dst = G.nodes, G.src, G.dst
    keys = G.keys if G.multigraph else None
    for arc, reverse in found:
        edge = (nodes[src[arc] - 1], nodes[dst[arc] - 1])
        if keys is not None:
            edge += (keys[arc],)
        if orientation is not None:
            edge += ("reverse" if reverse else "forward",)
        yield edge


@networkx_algorithm(linear_time=True)
def edge_dfs(G, source=None, orientation=None):
    yield from _edge_dfs(G, source, orientation)


@networkx_algorithm(linear_time=True)
def find_cycle(G, source=None, orientation=None):
    # NetworkX's find_cycle, on edgecase's edge_dfs
    if not G.is_directed() or orientation in (None, "original"):

        def tailhead(edge):
            return edge[:2]

    elif orientation == "reverse":

        def tailhead(edge):
            return (edge[1], edge[0])

    elif orientation == "ignore":

        def tailhead(edge):
            if edge[-1] == "reverse":
                return (edge[1], edge[0])
            return edge[:2]

    explored = set()
    cycle = []
    final_node = None
    for start_node in _nbunch_iter(G, source):
        if start_node in explored:
            continue
        edges = []
        seen = {start_node}
        active_nodes = {start_node}
        previous_head = None
        for edge in _edge_dfs(G, start_node, orientation):
            tail, head = tailhead(edge)
            if head in explored:
                continue
            if previous_head is not None and tail != previous_head:
                while True:
                    try:
                        popped_edge = edges.pop()
                    except IndexError:
                        edges = []
                        active_nodes = {tail}
                        break
                    else:
                        popped_head = tailhead(popped_edge)[1]
                        active_nodes.remove(popped_head)
                    if edges:
                        last_head = tailhead(edges[-1])[1]
                        if tail == last_head:
                            break
            edges.append(edge)
            if head in active_nodes:
                cycle.extend(edges)
                final_node = head
                break
            else:
                seen.add(head)
                active_nodes.add(head)
                previous_head = head
        if cycle:
            break
        else:
            explored.update(seen)
    else:
        assert len(cycle) == 0
        raise nx.NetworkXNoCycle("No cycle found.")
    for i, edge in enumerate(cycle):
        tail, head = tailhead(edge)
        if tail == final_node:
            break
    return cycle[i:]
