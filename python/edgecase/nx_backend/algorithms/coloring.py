"""Greedy coloring, maximal independent sets and dominating sets.

The Rust kernels (`edgecase._core.nx_greedy_color`, `nx_maximal_independent_set`,
`nx_dominating_set`) model the Python sets NetworkX's strategies draw nodes from (with the
node keys' hashes) and CPython's random generator, so colorings and sets are NetworkX's.
Directed graphs, multigraphs and graphs with self-loops, custom strategy functions and
`interchange=True` are left to NetworkX for greedy coloring.
"""

from __future__ import annotations

import random

import networkx as nx

from edgecase.nx_backend.algorithms.community import _rng_state, _set_rng_state
from edgecase.nx_backend.algorithms.shortest_paths import _adjacency
from edgecase.nx_backend.registry import networkx_algorithm, prefer_converted_graph

__all__ = [
    "dominating_set",
    "greedy_color",
    "is_dominating_set",
    "maximal_independent_set",
]

_STRATEGIES = {
    "largest_first": "largest_first",
    "random_sequential": "given",
    "smallest_last": "smallest_last",
    "independent_set": "independent_set",
    "connected_sequential_bfs": "connected_sequential_bfs",
    "connected_sequential_dfs": "connected_sequential_dfs",
    "connected_sequential": "connected_sequential_bfs",
    "saturation_largest_first": "saturation_largest_first",
    "DSATUR": "saturation_largest_first",
}
_LINEAR = {"largest_first", "random_sequential", "smallest_last", "connected_sequential"}


def _hashes(G):
    return [hash(v) for v in G.nodes]


def _has_selfloops(G):
    if hasattr(G, "src"):  # (an edgecase graph)
        return any(s == d for s, d in zip(G.src, G.dst))
    return any(u in nbrs for u, nbrs in G._adj.items())


def _greedy_can_run(G, strategy="largest_first", interchange=False):
    if G.is_directed() or G.is_multigraph():
        return "greedy coloring of directed graphs and multigraphs is left to NetworkX"
    if not isinstance(strategy, str):
        return "custom strategy functions are left to NetworkX"
    if interchange and strategy in _STRATEGIES:
        if _STRATEGIES[strategy] not in ("independent_set", "saturation_largest_first"):
            return "interchange=True is left to NetworkX"
    if _has_selfloops(G):
        return "greedy coloring of graphs with self-loops is left to NetworkX"
    return True


def _greedy_should_run(G, strategy="largest_first", interchange=False):
    if isinstance(strategy, str) and strategy.startswith(tuple(_LINEAR)):
        return prefer_converted_graph(G)
    return True


@networkx_algorithm(can_run=_greedy_can_run, should_run=_greedy_should_run)
def greedy_color(G, strategy="largest_first", interchange=False):
    from edgecase._core import nx_greedy_color

    if len(G) == 0:
        return {}
    if strategy not in _STRATEGIES:
        raise nx.NetworkXError(
            f"strategy must be callable or a valid string. {strategy} not valid."
        )
    name = _STRATEGIES[strategy]
    if interchange:
        if name == "independent_set":
            raise nx.NetworkXPointlessConcept("interchange cannot be used with independent_set")
        raise nx.NetworkXPointlessConcept(
            "interchange cannot be used with saturation_largest_first"
        )
    nodes = G.nodes
    order = None
    if name == "given":
        # NetworkX's own strategy: shuffles with the global random generator
        from networkx.algorithms.coloring.greedy_coloring import strategy_random_sequential

        key_to_id = G.key_to_id
        order = [key_to_id[v] - 1 for v in strategy_random_sequential(G, {})]
    colored, colors = nx_greedy_color(_adjacency(G), _hashes(G), name, order)
    return {nodes[v]: c for v, c in zip(colored, colors)}


def _random_can_run(G, nodes=None, seed=None):
    if type(seed) is not random.Random:
        return f"seed is a {type(seed).__name__}, not a random.Random"
    return True


@networkx_algorithm(can_run=_random_can_run)
def maximal_independent_set(G, nodes=None, seed=None):
    from edgecase._core import nx_maximal_independent_set

    if not nodes:
        nodes = {seed.choice(list(G.nodes))}
    else:
        nodes = set(nodes)
    if not nodes.issubset(G):
        raise nx.NetworkXUnfeasible(f"{nodes} is not a subset of the nodes of G")
    key_to_id = G.key_to_id
    start = [key_to_id[v] - 1 for v in nodes]
    indptr, indices, _ = _adjacency(G).arcs()
    neighbors = {k for i in start for k in indices[indptr[i] : indptr[i + 1]]}
    if neighbors.intersection(start):
        raise nx.NetworkXUnfeasible(f"{nodes} is not an independent set of G")
    words, index, extra = _rng_state(seed)
    chosen, words, index = nx_maximal_independent_set(
        _adjacency(G), _hashes(G), start, words, index
    )
    _set_rng_state(seed, words, index, extra)
    indep_nodes = set(nodes)
    all_nodes = G.nodes
    for v in chosen:
        indep_nodes.add(all_nodes[v])
    return indep_nodes


@networkx_algorithm(linear_time=True)
def dominating_set(G, start_with=None):
    from edgecase._core import nx_dominating_set

    if start_with is None:
        start_with = nx.utils.arbitrary_element(set(G.nodes))
    if start_with not in G:
        raise nx.NetworkXError(f"node {start_with} is not in G")
    added = nx_dominating_set(_adjacency(G), _hashes(G), G.key_to_id[start_with] - 1)
    result = {start_with}
    nodes = G.nodes
    for v in added[1:]:
        result.add(nodes[v])
    return result


@networkx_algorithm(linear_time=True)
def is_dominating_set(G, nbunch):
    key_to_id = G.key_to_id
    testset = {key_to_id[n] - 1 for n in nbunch if n in G}
    indptr, indices, _ = _adjacency(G).arcs()
    covered = set(testset)
    for i in testset:
        covered.update(indices[indptr[i] : indptr[i + 1]])
    return len(covered) == len(G)
