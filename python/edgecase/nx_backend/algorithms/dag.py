"""More DAG algorithms: lexicographical and all topological sorts, longest paths, transitive
closure and reduction, aperiodicity, edge BFS and dominators.

The searches run in Rust (`ArcAdjacency`, `nx_immediate_dominators`, `nx_is_aperiodic`)
and follow NetworkX's steps, so results and their order are NetworkX's. Functions that
return graphs build NetworkX graphs (with the same Python steps as NetworkX, on top of the
faster searches).
"""

from __future__ import annotations

from collections import deque

import networkx as nx

from edgecase.nx_backend.algorithms.components import is_strongly_connected
from edgecase.nx_backend.algorithms.dfs_structure import _arc_adjacency
from edgecase.nx_backend.algorithms.distance_measures import _nbunch_iter
from edgecase.nx_backend.algorithms.recognition import _in_degrees
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency, _position
from edgecase.nx_backend.algorithms.traversal import (
    dfs_edges,
    is_directed_acyclic_graph,
    topological_sort,
)
from edgecase.nx_backend.registry import networkx_algorithm, prefer_converted_graph

__all__ = [
    "all_topological_sorts",
    "dag_longest_path",
    "dag_longest_path_length",
    "dominance_frontiers",
    "edge_bfs",
    "immediate_dominators",
    "is_aperiodic",
    "lexicographical_topological_sort",
    "transitive_closure",
    "transitive_reduction",
]


def _out_targets(G):
    """Each node's out-edge targets (keys), one per edge (parallel edges repeat), in order."""
    nodes = G.nodes
    targets = [[] for _ in nodes]
    for s, d in zip(G.src, G.dst):
        targets[s - 1].append(nodes[d - 1])
    return targets


# Topological sorts


@networkx_algorithm(linear_time=True)
def lexicographical_topological_sort(G, key=None):
    if key is None:

        def key(node):
            return node

    nodes = G.nodes
    # NetworkX's heap compares (key(node), node position); ranking the nodes by that once
    # gives the same order (and keys that do not compare are left to NetworkX)
    try:
        by_key = sorted(range(len(nodes)), key=lambda i: (key(nodes[i]), i))
    except TypeError:
        raise NotImplementedError("node keys that do not compare are left to NetworkX") from None
    rank = [0] * len(nodes)
    for r, i in enumerate(by_key):
        rank[i] = r
    return _lexicographical_order(G, rank)


def _lexicographical_order(G, rank):
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
    order, left_over = _arc_adjacency(G).lexicographical_topological_sort(rank)
    nodes = G.nodes
    for v in order:
        yield nodes[v]
    if left_over:
        raise nx.NetworkXUnfeasible("Graph contains a cycle or graph changed during iteration")


@networkx_algorithm(should_run=prefer_converted_graph)
def all_topological_sorts(G):
    # NetworkX's algorithm (Knuth's), on precomputed out-edge lists
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
    nodes = G.nodes
    out_targets = dict(zip(nodes, _out_targets(G)))
    count = dict(zip(nodes, _in_degrees(G)))
    D = deque([v for v in nodes if count[v] == 0])
    bases = []
    current_sort = []
    while True:
        if len(current_sort) == len(nodes):
            yield list(current_sort)
            while len(current_sort) > 0:
                q = current_sort.pop()
                for j in out_targets[q]:
                    count[j] += 1
                while len(D) > 0 and count[D[-1]] > 0:
                    D.pop()
                D.appendleft(q)
                if D[-1] == bases[-1]:
                    bases.pop()
                else:
                    break
        else:
            if len(D) == 0:
                raise nx.NetworkXUnfeasible("Graph contains a cycle.")
            q = D.pop()
            for j in out_targets[q]:
                count[j] -= 1
                if count[j] == 0:
                    D.append(j)
            current_sort.append(q)
            if len(bases) < len(current_sort):
                bases.append(q)
        if len(bases) == 0:
            break


# Longest paths


def _longest_path(G, weight, default_weight, topo_order):
    """`(path, length)` as NetworkX's `dag_longest_path(_length)` find them."""
    if topo_order is None:
        topo_order = topological_sort(G)
    key_to_id = G.key_to_id
    positions = [key_to_id[v] - 1 for v in topo_order]
    if weight is None:
        weights = [default_weight] * len(G.src)
    else:
        weights = G.edge_values_of(weight, default_weight)
    found = _arc_adjacency(G).dag_longest_path(positions, [float(w) for w in weights])
    if found is None:
        raise KeyError("topo_order lists a node before one of its predecessors")
    path, length = found
    nodes = G.nodes
    integral = all(isinstance(w, int) for w in weights)
    if len(path) < 2 or integral:
        length = int(length)
    return [nodes[v] for v in path], length


@networkx_algorithm(linear_time=True)
def dag_longest_path(G, weight="weight", default_weight=1, topo_order=None):
    if len(G) == 0:
        return []
    return _longest_path(G, weight, default_weight, topo_order)[0]


@networkx_algorithm(linear_time=True)
def dag_longest_path_length(G, weight="weight", default_weight=1):
    if len(G) == 0:
        return 0
    return _longest_path(G, weight, default_weight, None)[1]


# Transitive closure and reduction


def _bfs_descendants(adjacency, nodes, v):
    """`nx.descendants(G, v)`: a set built in breadth-first order (so it iterates alike)."""
    order = adjacency.bfs_levels(v)[0]
    return {nodes[u] for u in order[1:]}


@networkx_algorithm
def transitive_closure(G, reflexive=False):
    TC = G.to_networkx().copy()
    if reflexive not in {None, True, False}:
        raise nx.NetworkXError("Incorrect value for the parameter `reflexive`")
    adjacency = _adjacency(G)
    nodes = G.nodes
    for i, v in enumerate(nodes):
        if reflexive is None:
            descendants = _bfs_descendants(adjacency, nodes, i)
            TC.add_edges_from((v, u) for u in descendants if u not in TC[v])
        elif reflexive is True:
            descendants = _bfs_descendants(adjacency, nodes, i) | {v}
            TC.add_edges_from((v, u) for u in descendants if u not in TC[v])
        elif reflexive is False:
            TC.add_edges_from((v, e[1]) for e in _edge_bfs(G, v, None) if e[1] not in TC[v])
    return TC


@networkx_algorithm
def transitive_reduction(G):
    if not is_directed_acyclic_graph(G):
        msg = "Directed Acyclic Graph required for transitive_reduction"
        raise nx.NetworkXError(msg)
    nodes = G.nodes
    TR = nx.DiGraph()
    TR.add_nodes_from(nodes)
    indptr, indices, _ = _adjacency(G).arcs()
    descendants = {}
    check_count = dict(zip(nodes, _in_degrees(G)))
    for i, u in enumerate(nodes):
        u_succ = [nodes[k] for k in indices[indptr[i] : indptr[i + 1]]]
        u_nbrs = set(u_succ)
        for v in u_succ:
            if v in u_nbrs:
                if v not in descendants:
                    descendants[v] = {y for x, y in dfs_edges(G, v)}
                u_nbrs -= descendants[v]
            check_count[v] -= 1
            if check_count[v] == 0:
                del descendants[v]
        TR.add_edges_from((u, v) for v in u_nbrs)
    return TR


# Aperiodicity


@networkx_algorithm(linear_time=True)
def is_aperiodic(G):
    from edgecase._core import nx_is_aperiodic

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    if not is_strongly_connected(G):
        raise nx.NetworkXError("Graph is not strongly connected.")
    return nx_is_aperiodic(_adjacency(G), 0)


# Edge BFS


def _edge_bfs(G, source, orientation):
    """NetworkX's `edge_bfs` edges: `(u, v[, key][, "forward" or "reverse"])`."""
    starts = list(_nbunch_iter(G, source))
    if not starts:
        return
    directed = G.is_directed()
    if directed and orientation not in (None, "original", "reverse", "ignore"):
        raise nx.NetworkXError("invalid orientation argument.")
    key_to_id = G.key_to_id
    found = _arc_adjacency(G).edge_bfs(
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
def edge_bfs(G, source=None, orientation=None):
    yield from _edge_bfs(G, source, orientation)


# Dominators


def _immediate_dominators(G, start):
    from edgecase._core import nx_immediate_dominators

    s = _position(G, start)
    if s < 0:
        raise nx.NetworkXError("start is not in G")
    assigned, dominators = nx_immediate_dominators(_adjacency(G, reverse=True), s)
    nodes = G.nodes
    return {nodes[u]: nodes[d] for u, d in zip(assigned, dominators)}


@networkx_algorithm(linear_time=True)
def immediate_dominators(G, start):
    return _immediate_dominators(G, start)


@networkx_algorithm(linear_time=True)
def dominance_frontiers(G, start):
    idom = _immediate_dominators(G, start) | {start: None}
    nodes = G.nodes
    key_to_id = G.key_to_id
    indptr, indices, _ = _adjacency(G, reverse=True).reverse_arcs()
    df = {u: set() for u in idom}
    for u in idom:
        i = key_to_id[u] - 1
        preds = [nodes[k] for k in indices[indptr[i] : indptr[i + 1]]]
        if u == start or len(preds) >= 2:
            for v in preds:
                if v in idom:
                    while v != idom[u]:
                        df[v].add(u)
                        v = idom[v]
    return df
