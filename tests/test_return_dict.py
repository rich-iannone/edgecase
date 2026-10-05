"""return_dict=True returns the result without writing anything to the graph.

Regression: 22 node-level methods (pagerank, betweenness, eigen_centrality, the cmty_* family,
the bipartite centralities, ...) wrote their column even with return_dict=True, while others
(katz, closeness, ...) didn't.
"""

import pytest

import edgecase as ec

# A six-cycle: connected, undirected and bipartite, so every method below accepts it
CALLS = {
    "bipartite_degree_centrality": {},
    "bipartite_closeness_centrality": {},
    "bipartite_betweenness_centrality": {},
    "cmty_fast_greedy": {},
    "cmty_edge_btwns": {},
    "cmty_girvan_newman": {},
    "cmty_l_eigenvec": {},
    "cmty_louvain": {},
    "cmty_walktrap": {},
    "label_propagation": {"known_labels": {1: 0, 4: 1}},
    "degrees": {},
    "pagerank": {},
    "personalized_pagerank": {"personalization": {1: 1.0}},
    "random_walk_with_restart": {"seeds": {1: 1.0}},
    "heat_diffusion": {"seeds": {1: 1.0}},
    "betweenness_centrality": {},
    "load_centrality": {},
    "second_order_centrality": {},
    "communicability_betweenness_centrality": {},
    "percolation_centrality": {},
    "average_neighbor_degree": {},
    "harmonic_centrality": {},
    "eigen_centrality": {},
}


def _state(g):
    return g.get_node_df().columns, len(g._computations)


@pytest.mark.parametrize("method", sorted(CALLS))
def test_return_dict_true_leaves_graph_untouched(method):
    g = ec.Graph().add_cycle_graph(6)
    before = _state(g)
    result = getattr(g, method)(return_dict=True, **CALLS[method])
    assert isinstance(result, dict) and result
    assert _state(g) == before


@pytest.mark.parametrize("method", sorted(CALLS))
def test_return_dict_false_still_writes(method):
    g = ec.Graph().add_cycle_graph(6)
    columns, n_computations = _state(g)
    assert getattr(g, method)(attr_name="out", **CALLS[method]) is g
    assert "out" in g.get_node_df().columns and "out" not in columns
    assert len(g._computations) == n_computations + 1
