"""Weisfeiler-Lehman graph and subgraph hashes.

edgecase's `NxWlGraph` builds each round's aggregated node labels in Rust (the node's label,
then its neighbors' labels sorted, with NetworkX's prefixes); hashing them with BLAKE2b uses
Python's `hashlib`, as NetworkX does, so the hashes are NetworkX's, warnings included.
Hashing dominates the time, so this is only modestly faster than NetworkX (Measured: about
1.2x on an already converted graph of 20,000 nodes); automatic dispatch therefore uses it
only for converted graphs.
"""

from __future__ import annotations

import warnings
from collections import Counter
from hashlib import blake2b

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = ["weisfeiler_lehman_graph_hash", "weisfeiler_lehman_subgraph_hashes"]


def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def _init_node_labels(G, edge_attr, node_attr):
    from edgecase.nx_backend.algorithms.assortativity import _degrees
    from edgecase.nx_backend.graph import MISSING

    if node_attr:
        values = G.node_values_of(node_attr, MISSING)
        if any(v is MISSING for v in values):
            raise KeyError(node_attr)
        return [str(v) for v in values]
    if edge_attr:
        return [""] * len(G.nodes)
    warnings.warn(
        "The hashes produced for graphs without node or edge attributes "
        "changed in v3.5 due to a bugfix (see documentation).",
        UserWarning,
        stacklevel=4,
    )
    if G.is_directed():
        ins = _degrees(G, None, "in")
        outs = _degrees(G, None, "out")
        return [str(i) + "_" + str(o) for i, o in zip(ins, outs)]
    return [str(d) for d in _degrees(G, None, "all")]


def _warn_directed(G, version="version v3.5"):
    # (NetworkX words the two functions' warnings slightly differently)
    if G.is_directed():
        warnings.warn(
            f"The hashes produced for directed graphs changed in {version}"
            " to track in and out edges separately, and in version 3.8 to include"
            " direction prefixes when edge_attr is used (see documentation).",
            UserWarning,
            stacklevel=4,
        )


def _wl_graph(G, edge_attr):
    from edgecase._core import NxWlGraph
    from edgecase.nx_backend.graph import MISSING

    edge_labels = None
    if edge_attr is not None:
        values = G.edge_values_of(edge_attr, MISSING) if G.src else []
        if any(v is MISSING for v in values):
            raise KeyError(edge_attr)
        edge_labels = [str(v) for v in values]
    return NxWlGraph(len(G.nodes), G.src, G.dst, G.is_directed(), edge_labels)


@networkx_algorithm(linear_time=True)
def weisfeiler_lehman_graph_hash(G, edge_attr=None, node_attr=None, iterations=3, digest_size=16):
    _warn_directed(G)
    if iterations <= 0:
        raise ValueError("The WL algorithm requires that `iterations` be positive")
    labels = _init_node_labels(G, edge_attr, node_attr)
    wl = _wl_graph(G, edge_attr)
    if not edge_attr and not node_attr:
        iterations -= 1
    subgraph_hash_counts = []
    for _ in range(iterations):
        labels = [_hash_label(label, digest_size) for label in wl.aggregate(labels)]
        counter = Counter(labels)
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)


@networkx_algorithm(linear_time=True)
def weisfeiler_lehman_subgraph_hashes(
    G,
    edge_attr=None,
    node_attr=None,
    iterations=3,
    digest_size=16,
    include_initial_labels=False,
):
    _warn_directed(G, "v3.5")
    if iterations <= 0:
        raise ValueError("The WL algorithm requires that `iterations` be positive")
    labels = _init_node_labels(G, edge_attr, node_attr)
    wl = _wl_graph(G, edge_attr)
    nodes = G.nodes
    if include_initial_labels:
        hashes = [[_hash_label(label, digest_size)] for label in labels]
    else:
        hashes = [[] for _ in nodes]
    if not edge_attr and not node_attr:
        iterations -= 1
        for i, label in enumerate(labels):
            hashes[i].append(_hash_label(label, digest_size))
    for _ in range(iterations):
        labels = [_hash_label(label, digest_size) for label in wl.aggregate(labels)]
        for i, label in enumerate(labels):
            hashes[i].append(label)
    return dict(zip(nodes, hashes))
