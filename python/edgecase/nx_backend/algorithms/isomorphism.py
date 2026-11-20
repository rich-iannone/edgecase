"""Graph isomorphism tests: is_isomorphic, vf2pp_is_isomorphic and the could_be_isomorphic
family.

The isomorphism tests use edgecase's VF2-style matcher (`edgecase._core.match_graphs`,
which respects parallel edges and self-loops); their answers are booleans, so any correct
search gives NetworkX's. Node and edge match functions are left to NetworkX; node labels
for `vf2pp_is_isomorphic` become integer colors (equal labels, equal colors).
"""

from __future__ import annotations

from collections import Counter
from itertools import chain

import networkx as nx

from edgecase.nx_backend.algorithms.recognition import _degrees
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "could_be_isomorphic",
    "fast_could_be_isomorphic",
    "faster_could_be_isomorphic",
    "is_isomorphic",
    "vf2pp_is_isomorphic",
]


def _isomorphic(G1, G2, colors1=None, colors2=None):
    from edgecase._core import match_graphs

    if len(G1) != len(G2) or G1.number_of_edges() != G2.number_of_edges():
        return False
    found = match_graphs(
        G1.rust_graph(),
        G2.rust_graph(),
        "iso",
        colors1 or {},
        colors2 or {},
        {},
        {},
        limit=1,
    )
    return bool(found)


def _is_isomorphic_can_run(G1, G2, node_match=None, edge_match=None):
    if node_match is not None or edge_match is not None:
        return "node_match and edge_match functions are left to NetworkX"
    return True


@networkx_algorithm(can_run=_is_isomorphic_can_run)
def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    if G1.is_directed() != G2.is_directed():
        raise nx.NetworkXError("Graphs G1 and G2 are not of the same type.")
    return _isomorphic(G1, G2)


@networkx_algorithm
def vf2pp_is_isomorphic(FG, SG, node_label=None, default_label=None):
    if len(SG) == 0 or len(FG) == 0:
        return False
    if SG.is_directed() != FG.is_directed():
        raise nx.NetworkXError("SG and FG must have the same directedness")
    colors1 = colors2 = None
    if node_label is not None:
        codes = {}

        def colors(G):
            labels = G.node_values_of(node_label, default_label)
            return {i: codes.setdefault(label, len(codes)) for i, label in enumerate(labels, 1)}

        try:
            colors1, colors2 = colors(FG), colors(SG)
        except TypeError:
            raise NotImplementedError("unhashable node labels are left to NetworkX") from None
    return _isomorphic(FG, SG, colors1, colors2)


@networkx_algorithm
def could_be_isomorphic(G1, G2, *, properties="dtc"):
    if len(G1) != len(G2):
        return False
    properties_to_check = set(properties)
    G1_props, G2_props = [], []

    def _properties_consistent():
        G1_ptable = [tuple(p[n] for p in G1_props) for n in G1.nodes]
        G2_ptable = [tuple(p[n] for p in G2_props) for n in G2.nodes]
        return sorted(G1_ptable) == sorted(G2_ptable)

    if "d" in properties_to_check:
        G1_props.append(dict(zip(G1.nodes, _degrees(G1))))
        G2_props.append(dict(zip(G2.nodes, _degrees(G2))))
        if not _properties_consistent():
            return False
    if "t" in properties_to_check:
        G1_props.append(nx.triangles(G1))
        G2_props.append(nx.triangles(G2))
        if not _properties_consistent():
            return False
    if "c" in properties_to_check:
        G1_props.append(Counter(chain.from_iterable(nx.find_cliques(G1))))
        G2_props.append(Counter(chain.from_iterable(nx.find_cliques(G2))))
        if not _properties_consistent():
            return False
    return True


@networkx_algorithm
def fast_could_be_isomorphic(G1, G2):
    return could_be_isomorphic(G1, G2, properties="dt")


@networkx_algorithm
def faster_could_be_isomorphic(G1, G2):
    return could_be_isomorphic(G1, G2, properties="d")
