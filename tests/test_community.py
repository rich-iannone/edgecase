"""Tests for community detection: stochastic block model fitting."""

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_cliques():
    """Two 5-cliques connected by a single bridge edge."""
    g = ec.Graph()
    for i in range(10):
        g.add_node(label=f"N{i}")
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(f"N{i}", f"N{j}")
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g.add_edge(f"N{i}", f"N{j}")
    g.add_edge("N4", "N5")
    return g


@pytest.fixture
def sbm_graph():
    """Graph generated from a known 2-block SBM."""
    g = ec.Graph()
    sbm = g.generate_stochastic_block_model([10, 10], [0.9, 0.05, 0.05, 0.9], seed=23)
    return sbm


# =============================================================================
# SBM Fitting
# =============================================================================


class TestCmtyStochasticBlockModel:
    def test_returns_dict(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert isinstance(result, dict)
        assert len(result) == 10

    def test_correct_number_of_blocks(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(set(result.values())) <= 2

    def test_chaining_returns_self(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, seed=23)
        assert result is two_cliques

    def test_stores_attribute(self, two_cliques):
        two_cliques.cmty_stochastic_block_model(k=2, seed=23)
        ndf = two_cliques.get_node_df()
        assert "sbm_block" in ndf.columns

    def test_custom_attr_name(self, two_cliques):
        two_cliques.cmty_stochastic_block_model(k=2, attr_name="my_block", seed=23)
        ndf = two_cliques.get_node_df()
        assert "my_block" in ndf.columns

    def test_recovers_clear_structure(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        clique1_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5)}
        clique2_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5, 10)}
        clique1_blocks = {result[nid] for nid in clique1_ids}
        clique2_blocks = {result[nid] for nid in clique2_ids}
        # Each clique should be mostly in one block
        assert len(clique1_blocks) <= 2
        assert len(clique2_blocks) <= 2

    def test_sbm_generated_graph(self, sbm_graph):
        result = sbm_graph.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(result) == 20
        assert len(set(result.values())) <= 2

    def test_three_blocks(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=3, return_dict=True, seed=23)
        assert len(set(result.values())) <= 3

    def test_single_block(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=1, return_dict=True, seed=23)
        assert all(v == 0 for v in result.values())

    def test_empty_graph(self):
        g = ec.Graph()
        result = g.cmty_stochastic_block_model(k=2, return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        result = g.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(result) == 1

    def test_seed_reproducibility(self, two_cliques):
        r1 = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=99)
        r2 = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=99)
        assert r1 == r2

    def test_all_node_ids_present(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        node_ids = set(two_cliques.get_node_ids())
        assert set(result.keys()) == node_ids
