"""HITS hubs and authorities (NetworkX's `hits`, power iteration).

NetworkX builds a SciPy adjacency matrix with a Python loop over `G.edges(data=...)` and
then iterates with NumPy. Here the same matrix is built from the converted graph's arc
arrays, as the same COO triples in the same order, and the same NumPy iteration runs on it,
so the results are NetworkX's to the bit; the saving is the matrix construction.
"""

from __future__ import annotations

import networkx as nx

from edgecase.nx_backend.registry import networkx_algorithm, prefer_converted_graph

__all__ = ["hits"]


def _scipy_adjacency(G, weight="weight", dtype=float):
    """`nx.to_scipy_sparse_array(G, nodelist=list(G), dtype=dtype, weight=weight)`."""
    import scipy as sp

    n = len(G)
    if n == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")
    src, dst = G.src, G.dst
    values = G.edge_values_of(weight, 1) if weight is not None else None
    # `G.edges(data=weight, default=1)`: each edge once (an undirected edge from the end
    # NetworkX lists first), with parallel edges each on their own
    arcs = G.edge_arcs
    row = [src[i] - 1 for i in arcs]
    col = [dst[i] - 1 for i in arcs]
    data = [1 if values is None else values[i] for i in arcs]
    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(n, n), dtype=dtype)
    else:
        d = data + data
        r = row + col
        c = col + row
        # (a self-loop is in both halves; NetworkX takes one copy back out)
        selfloops = [(src[i] - 1, data[k]) for k, i in enumerate(arcs) if src[i] == dst[i]]
        if selfloops:
            d += [-wt for _, wt in selfloops]
            diagonal = [u for u, _ in selfloops]
            r += diagonal
            c += diagonal
        A = sp.sparse.coo_array((d, (r, c)), shape=(n, n), dtype=dtype)
    return A.asformat("csr")


def _can_run(
    G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True, *, method="power_iteration"
):
    if method == "svd":
        return "method='svd' is not implemented by edgecase"
    return True


# (Only the matrix construction is faster than NetworkX's, by about converting the graph:
# measured 0.7-0.8x with the conversion, 1.4-2.8x without, so D6 applies)
@networkx_algorithm(can_run=_can_run, should_run=prefer_converted_graph)
def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True, *, method="power_iteration"):
    if method == "power_iteration":
        return _hits_power_iteration(G, max_iter, tol, nstart, normalized)
    raise ValueError(f"method not supported: {method}")


def _hits_power_iteration(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """NetworkX's `_hits_power_iteration`, line for line, on the same matrix."""
    import numpy as np

    N = len(G)
    if N == 0:
        return {}, {}
    nodelist = list(G.nodes)
    A = _scipy_adjacency(G, dtype=float)
    if nstart is None:
        h = np.full(N, 1.0 / N)
    else:
        missing = {node for node in nodelist if node not in nstart.keys()}
        if missing:
            raise nx.NetworkXError(f"nstart must have a value for every node; missing: {missing}")
        h = np.array([nstart[node] for node in nodelist], dtype=float)
        s = h.sum()
        if s == 0:
            raise nx.NetworkXError("nstart values must not sum to zero")
        h = h / s
    if max_iter <= 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    for _ in range(max_iter):
        hlast = h
        a = h @ A
        h = A @ a
        h /= h.max()
        if np.abs(h - hlast).sum() < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        h /= h.sum()
        a /= a.sum()
    else:
        a /= a.max()
    hubs = dict(zip(nodelist, map(float, h)))
    authorities = dict(zip(nodelist, map(float, a)))
    return hubs, authorities
