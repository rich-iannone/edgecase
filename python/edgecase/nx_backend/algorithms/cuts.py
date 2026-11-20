"""Edge and node boundaries, cut sizes, volumes and expansion measures.

These follow NetworkX's definitions and iteration orders (edges come out in the order
NetworkX's `G.edges(nbunch)` gives them, node sets built the same way), so sums of float
weights round the same. They cost about as much as the conversion, so automatic dispatch
uses them only on graphs already converted.
"""

from __future__ import annotations

from itertools import chain

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "boundary_expansion",
    "conductance",
    "cut_size",
    "edge_boundary",
    "edge_expansion",
    "inter_community_edges",
    "intra_community_edges",
    "mixing_expansion",
    "node_boundary",
    "node_expansion",
    "normalized_cut_size",
    "volume",
]


def _starts(G):
    """Per node ID, where its arcs start (arcs are grouped by node, in node order)."""
    starts = G.__dict__.get("_arc_starts")
    if starts is None:
        starts = [0] * (len(G.nodes) + 2)
        for s in G.src:
            starts[s + 1] += 1
        for i in range(1, len(starts)):
            starts[i] += starts[i - 1]
        G._arc_starts = starts
    return starts


def _neighbor_ids(G, node_id):
    """Distinct successors (neighbors) of a node, in NetworkX's order."""
    starts, dst = _starts(G), G.dst
    out = []
    previous = None
    for k in range(starts[node_id], starts[node_id + 1]):
        v = dst[k]
        if v != previous:  # (a multigraph's parallel arcs are consecutive)
            out.append(v)
            previous = v
    return out


def _edges_of(G, nbunch, data=False, keys=False, default=None):
    """`G.edges(nbunch, data=data, keys=keys, default=default)` (nbunch: nodes in G)."""
    starts, src, dst, nodes = _starts(G), G.src, G.dst, G.nodes
    key_to_id = G.key_to_id
    multigraph = G.is_multigraph()
    directed = G.is_directed()
    if data is True:
        values = G.edge_dicts
    elif data is None:
        values = [default] * len(src)  # (`dd.get(None, default)`)
    elif data is not False:
        values = G.edge_values_of(data, default) if G.number_of_edges() else []
    seen = set()
    for n in nbunch:
        nid = key_to_id[n]
        for k in range(starts[nid], starts[nid + 1]):
            v = dst[k]
            if not directed and v in seen:
                continue
            e = (n, nodes[v - 1])
            if multigraph and keys:
                e += (G.keys[k],)
            if data is not False:
                e += (values[k],)
            yield e
        if not directed:
            seen.add(nid)


def _edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    nset1 = {n for n in nbunch1 if n in G}
    edges = _edges_of(G, nset1, data=data, keys=keys, default=default)
    if nbunch2 is None:
        return (e for e in edges if (e[0] in nset1) ^ (e[1] in nset1))
    nset2 = set(nbunch2)
    return (
        e for e in edges if (e[0] in nset1 and e[1] in nset2) or (e[1] in nset1 and e[0] in nset2)
    )


@networkx_algorithm(linear_time=True)
def edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    return _edge_boundary(G, nbunch1, nbunch2, data, keys, default)


def _node_boundary(G, nbunch1, nbunch2=None):
    nodes, key_to_id = G.nodes, G.key_to_id
    nset1 = {n for n in nbunch1 if n in G}
    bdy = {nodes[v - 1] for n in nset1 for v in _neighbor_ids(G, key_to_id[n])} - nset1
    if nbunch2 is not None:
        bdy &= set(nbunch2)
    return bdy


@networkx_algorithm(linear_time=True)
def node_boundary(G, nbunch1, nbunch2=None):
    return _node_boundary(G, nbunch1, nbunch2)


def _cut_size(G, S, T=None, weight=None):
    edges = _edge_boundary(G, S, T, data=weight, default=1)
    if G.is_directed():
        edges = chain(edges, _edge_boundary(G, T, S, data=weight, default=1))
    return sum(weight for u, v, weight in edges)


@networkx_algorithm(linear_time=True)
def cut_size(G, S, T=None, weight=None):
    return _cut_size(G, S, T, weight)


def _volume(G, S, weight=None):
    from edgecase.nx_backend.algorithms.cluster import nbunch_iter
    from edgecase.nx_backend.algorithms.community import _degrees

    weights = None if weight is None else G.edge_values_of(weight, 1)
    degrees = _degrees(G, weights)[0]  # (out-degrees for a directed graph)
    key_to_id = G.key_to_id
    if S in G:
        degree = degrees[key_to_id[S] - 1]
        return sum(d for v, d in degree)  # (a single node: fails as in NetworkX)
    return sum(degrees[key_to_id[v] - 1] for v in nbunch_iter(G, S))


@networkx_algorithm(linear_time=True)
def volume(G, S, weight=None):
    return _volume(G, S, weight)


def _complement(G, S, T):
    return set(G.nodes) - set(S) if T is None else T


@networkx_algorithm(linear_time=True)
def normalized_cut_size(G, S, T=None, weight=None):
    T = _complement(G, S, T)
    num_cut_edges = _cut_size(G, S, T=T, weight=weight)
    volume_S = _volume(G, S, weight=weight)
    volume_T = _volume(G, T, weight=weight)
    return num_cut_edges * ((1 / volume_S) + (1 / volume_T))


@networkx_algorithm(linear_time=True)
def conductance(G, S, T=None, weight=None):
    T = _complement(G, S, T)
    num_cut_edges = _cut_size(G, S, T, weight=weight)
    volume_S = _volume(G, S, weight=weight)
    volume_T = _volume(G, T, weight=weight)
    return num_cut_edges / min(volume_S, volume_T)


@networkx_algorithm(linear_time=True)
def edge_expansion(G, S, T=None, weight=None):
    T = _complement(G, S, T)
    num_cut_edges = _cut_size(G, S, T=T, weight=weight)
    return num_cut_edges / min(len(S), len(T))


@networkx_algorithm(linear_time=True)
def mixing_expansion(G, S, T=None, weight=None):
    num_cut_edges = _cut_size(G, S, T=T, weight=weight)
    num_total_edges = G.number_of_edges()
    return num_cut_edges / (2 * num_total_edges)


@networkx_algorithm(linear_time=True)
def node_expansion(G, S):
    import networkx as nx

    nodes, key_to_id = G.nodes, G.key_to_id
    neighborhood = set()
    for v in S:
        try:
            v_id = key_to_id[v]
        except KeyError:
            kind = "digraph" if G.is_directed() else "graph"
            raise nx.NetworkXError(f"The node {v} is not in the {kind}.") from None
        neighborhood.update(nodes[u - 1] for u in _neighbor_ids(G, v_id))
    return len(neighborhood) / len(S)


@networkx_algorithm(linear_time=True)
def boundary_expansion(G, S):
    return len(_node_boundary(G, S)) / len(S)


@networkx_algorithm(linear_time=True)
def intra_community_edges(G, partition):
    # sum(G.subgraph(block).size() for block in partition): edges inside each block
    starts, dst, key_to_id = _starts(G), G.dst, G.key_to_id
    directed = G.is_directed()
    total = 0
    for block in partition:
        ids = {key_to_id[n] for n in block if n in G}
        arcs = loops = 0
        for u in ids:
            for k in range(starts[u], starts[u + 1]):
                v = dst[k]
                if v in ids:
                    arcs += 1
                    loops += v == u
        total += arcs if directed else (arcs + loops) // 2
    return total


def _not_multigraph(G, *args, **kwargs):
    if G.is_multigraph():
        return "NetworkX counts a multigraph's quotient differently from its edges"
    return True


@networkx_algorithm(linear_time=True, can_run=_not_multigraph)
def inter_community_edges(G, partition):
    # The size of the multigraph quotient: the edges joining different blocks
    key_to_id = G.key_to_id
    block_of = {}
    for i, block in enumerate(partition):
        for n in block:
            block_of[key_to_id[n]] = i
    src, dst = G.src, G.dst
    return sum(block_of[src[k]] != block_of[dst[k]] for k in G.edge_arcs)
