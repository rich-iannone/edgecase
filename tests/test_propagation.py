"""Tests for network propagation algorithms: personalized PageRank, RWR, heat diffusion."""

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def triangle_directed():
    return (
        ec.Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
    )


@pytest.fixture
def path_graph():
    g = ec.Graph()
    for label in ["A", "B", "C", "D", "E"]:
        g.add_node(label=label)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "D").add_edge("D", "E")
    return g


@pytest.fixture
def weighted_path():
    g = ec.Graph(weighted=True)
    for label in ["A", "B", "C", "D"]:
        g.add_node(label=label)
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=5.0)
    g.add_edge("C", "D", weight=1.0)
    return g


@pytest.fixture
def disconnected_graph():
    g = ec.Graph()
    for label in ["A", "B", "C", "X", "Y"]:
        g.add_node(label=label)
    g.add_edge("A", "B").add_edge("B", "C")
    g.add_edge("X", "Y")
    return g


# =============================================================================
# Personalized PageRank
# =============================================================================


class TestPersonalizedPageRank:
    def test_single_seed_returns_all_nodes(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        assert len(scores) == 3

    def test_scores_sum_to_one(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_directed_seed_has_highest_score(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        seed_score = scores[seed_id]
        assert all(seed_score >= s for s in scores.values())

    def test_seed_scores_higher_than_distant(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        a_score = scores[seed_id]
        e_score = scores[g.get_id_from_label("E")]
        assert a_score > e_score

    def test_multiple_seeds(self, triangle_directed):
        g = triangle_directed
        a_id = g.get_id_from_label("A")
        c_id = g.get_id_from_label("C")
        scores = g.personalized_pagerank(
            personalization={a_id: 1.0, c_id: 1.0}, return_dict=True
        )
        assert len(scores) == 3
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_chaining_stores_attr(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        result = g.personalized_pagerank(personalization={seed_id: 1.0})
        assert result is g
        ndf = g.get_node_df()
        assert "personalized_pagerank" in ndf.columns

    def test_custom_attr_name(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        g.personalized_pagerank(
            personalization={seed_id: 1.0}, attr_name="gwas_score"
        )
        ndf = g.get_node_df()
        assert "gwas_score" in ndf.columns

    def test_empty_personalization_raises(self, triangle_directed):
        with pytest.raises(Exception):
            triangle_directed.personalized_pagerank(
                personalization={}, return_dict=True
            )

    def test_disconnected_graph_returns_all_nodes(self, disconnected_graph):
        g = disconnected_graph
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        assert len(scores) == 5
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_disconnected_seed_unreachable_low_score(self, disconnected_graph):
        g = disconnected_graph
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        a_score = scores[seed_id]
        x_score = scores[g.get_id_from_label("X")]
        assert a_score > x_score

    def test_weighted_graph(self, weighted_path):
        g = weighted_path
        seed_id = g.get_id_from_label("A")
        scores = g.personalized_pagerank(
            personalization={seed_id: 1.0}, return_dict=True
        )
        assert len(scores) == 4
        assert abs(sum(scores.values()) - 1.0) < 1e-6


# =============================================================================
# Random Walk with Restart
# =============================================================================


class TestRandomWalkWithRestart:
    def test_returns_all_nodes(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        scores = g.random_walk_with_restart(seeds={seed_id: 1.0}, return_dict=True)
        assert len(scores) == 3

    def test_scores_sum_to_one(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        scores = g.random_walk_with_restart(seeds={seed_id: 1.0}, return_dict=True)
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_seed_bias(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        scores = g.random_walk_with_restart(
            seeds={seed_id: 1.0}, restart_prob=0.3, return_dict=True
        )
        a_score = scores[g.get_id_from_label("A")]
        e_score = scores[g.get_id_from_label("E")]
        assert a_score > e_score

    def test_high_restart_concentrates_on_seed(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        scores = g.random_walk_with_restart(
            seeds={seed_id: 1.0}, restart_prob=0.9, return_dict=True
        )
        a_score = scores[seed_id]
        assert a_score > 0.5

    def test_stores_rwr_attr(self, triangle_directed):
        g = triangle_directed
        seed_id = g.get_id_from_label("A")
        g.random_walk_with_restart(seeds={seed_id: 1.0})
        ndf = g.get_node_df()
        assert "rwr" in ndf.columns

    def test_equivalence_with_personalized_pagerank(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        restart_prob = 0.2

        rwr_scores = g.random_walk_with_restart(
            seeds={seed_id: 1.0}, restart_prob=restart_prob, return_dict=True
        )
        ppr_scores = g.personalized_pagerank(
            personalization={seed_id: 1.0},
            damping=1.0 - restart_prob,
            return_dict=True,
        )

        for node_id in rwr_scores:
            assert abs(rwr_scores[node_id] - ppr_scores[node_id]) < 1e-10

    def test_disconnected_graph(self, disconnected_graph):
        g = disconnected_graph
        seed_id = g.get_id_from_label("X")
        scores = g.random_walk_with_restart(seeds={seed_id: 1.0}, return_dict=True)
        assert len(scores) == 5
        x_score = scores[seed_id]
        a_score = scores[g.get_id_from_label("A")]
        assert x_score > a_score

    def test_weighted_graph(self, weighted_path):
        g = weighted_path
        seed_id = g.get_id_from_label("A")
        scores = g.random_walk_with_restart(seeds={seed_id: 1.0}, return_dict=True)
        assert len(scores) == 4
        assert abs(sum(scores.values()) - 1.0) < 1e-6


# =============================================================================
# Heat Diffusion
# =============================================================================


class TestHeatDiffusion:
    def test_returns_all_nodes(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, return_dict=True)
        assert len(heat) == 5

    def test_heat_spreads_from_seed(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("C")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, t=0.5, return_dict=True)
        assert heat[seed_id] > 0
        assert heat[g.get_id_from_label("B")] > 0

    def test_small_t_stays_local(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, t=0.01, return_dict=True)
        a_heat = heat[seed_id]
        e_heat = heat[g.get_id_from_label("E")]
        assert a_heat > e_heat * 10

    def test_approximate_heat_conservation(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("C")
        initial_heat = 10.0
        heat = g.heat_diffusion(
            seeds={seed_id: initial_heat}, t=0.5, max_iter=200, return_dict=True
        )
        total_heat = sum(heat.values())
        assert abs(total_heat - initial_heat) / initial_heat < 0.15

    def test_multiple_seeds(self, path_graph):
        g = path_graph
        a_id = g.get_id_from_label("A")
        e_id = g.get_id_from_label("E")
        heat = g.heat_diffusion(seeds={a_id: 5.0, e_id: 5.0}, t=0.5, return_dict=True)
        assert all(h >= 0 for h in heat.values())

    def test_negative_t_raises(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        with pytest.raises(Exception):
            g.heat_diffusion(seeds={seed_id: 10.0}, t=-1.0, return_dict=True)

    def test_empty_seeds_raises(self, path_graph):
        with pytest.raises(Exception):
            path_graph.heat_diffusion(seeds={}, return_dict=True)

    def test_stores_heat_attr(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        g.heat_diffusion(seeds={seed_id: 10.0})
        ndf = g.get_node_df()
        assert "heat" in ndf.columns

    def test_all_values_non_negative(self, path_graph):
        g = path_graph
        seed_id = g.get_id_from_label("A")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, t=2.0, return_dict=True)
        assert all(h >= 0 for h in heat.values())

    def test_disconnected_graph_no_heat_crosses_components(self, disconnected_graph):
        g = disconnected_graph
        seed_id = g.get_id_from_label("A")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, t=0.5, return_dict=True)
        x_heat = heat[g.get_id_from_label("X")]
        y_heat = heat[g.get_id_from_label("Y")]
        assert x_heat == 0.0
        assert y_heat == 0.0

    def test_weighted_graph(self, weighted_path):
        g = weighted_path
        seed_id = g.get_id_from_label("A")
        heat = g.heat_diffusion(seeds={seed_id: 10.0}, t=0.5, return_dict=True)
        assert len(heat) == 4
        assert all(h >= 0 for h in heat.values())
