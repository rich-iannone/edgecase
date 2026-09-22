"""Tests for SimRank* similarity."""

import pytest
import edgecase as ec


@pytest.fixture
def diamond_directed():
    """Diamond DAG: 1->2, 1->3, 2->4, 3->4."""
    g = ec.Graph(directed=True)
    for i in range(1, 5):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("1", "3")
    g.add_edge("2", "4").add_edge("3", "4")
    return g


@pytest.fixture
def simple_directed():
    """Two nodes pointing to a third: 1->3, 2->3."""
    g = ec.Graph(directed=True)
    for i in range(1, 4):
        g.add_node(label=str(i))
    g.add_edge("1", "3").add_edge("2", "3")
    return g


@pytest.fixture
def triangle_undirected():
    """Triangle: 1-2, 2-3, 3-1."""
    g = ec.Graph(directed=False)
    for i in range(1, 4):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("2", "3").add_edge("3", "1")
    return g


@pytest.fixture
def chain_directed():
    """Chain: 1->2->3->4->5."""
    g = ec.Graph(directed=True)
    for i in range(1, 6):
        g.add_node(label=str(i))
    for i in range(1, 5):
        g.add_edge(str(i), str(i + 1))
    return g


class TestSimRankStarSimilarity:
    def test_returns_dict(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_self_similarity_is_one(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        for nid, inner in result.items():
            assert abs(inner.get(nid, 0.0) - 1.0) < 1e-10

    def test_symmetric_nodes_positive(self, simple_directed):
        """Nodes 1 and 2 both point to 3; they should have positive similarity."""
        result = simple_directed.simrank_star_similarity(return_dict=True)
        id1 = simple_directed.get_id_from_label("1")
        id2 = simple_directed.get_id_from_label("2")
        sim = result[id1].get(id2, 0.0)
        assert sim > 0.0, f"Expected positive similarity, got {sim}"

    def test_diamond_nodes_2_3_similar(self, diamond_directed):
        """Nodes 2 and 3 share in-neighbor (1) and out-neighbor (4)."""
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        sim = result[id2].get(id3, 0.0)
        assert sim > 0.0, f"Expected positive similarity for symmetric nodes, got {sim}"

    def test_values_in_unit_range(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        for inner in result.values():
            for val in inner.values():
                assert -1e-10 <= val <= 1.0 + 1e-10, f"Value out of range: {val}"

    def test_symmetry(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        ids = list(result.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                val_ab = result[a].get(b, 0.0)
                val_ba = result[b].get(a, 0.0)
                assert abs(val_ab - val_ba) < 1e-10, (
                    f"Not symmetric: sim({a},{b})={val_ab} vs sim({b},{a})={val_ba}"
                )

    def test_methods_agree(self, diamond_directed):
        cf = diamond_directed.simrank_star_similarity(method="closed_form", return_dict=True)
        it = diamond_directed.simrank_star_similarity(
            method="iterative", max_iterations=500, tolerance=1e-8, return_dict=True
        )
        for a in cf:
            for b in cf[a]:
                v_cf = cf[a][b]
                v_it = it.get(a, {}).get(b, 0.0)
                assert abs(v_cf - v_it) < 1e-3, f"Methods disagree for ({a},{b}): {v_cf} vs {v_it}"

    def test_triangle_equal_similarity(self, triangle_undirected):
        result = triangle_undirected.simrank_star_similarity(return_dict=True)
        ids = sorted(result.keys())
        sim_01 = result[ids[0]].get(ids[1], 0.0)
        sim_02 = result[ids[0]].get(ids[2], 0.0)
        sim_12 = result[ids[1]].get(ids[2], 0.0)
        assert abs(sim_01 - sim_02) < 1e-6
        assert abs(sim_02 - sim_12) < 1e-6

    def test_empty_graph(self):
        g = ec.Graph(directed=True)
        result = g.simrank_star_similarity(return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        result = g.simrank_star_similarity(return_dict=True)
        assert len(result) == 1
        nid = list(result.keys())[0]
        assert abs(result[nid][nid] - 1.0) < 1e-10

    def test_disconnected_nodes(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B")
        result = g.simrank_star_similarity(return_dict=True)
        id_a = g.get_id_from_label("A")
        id_b = g.get_id_from_label("B")
        sim = result[id_a].get(id_b, 0.0)
        assert sim == 0.0, f"Disconnected nodes should have 0 similarity, got {sim}"

    def test_chain_decay(self, chain_directed):
        """Similarity should decrease with distance."""
        result = chain_directed.simrank_star_similarity(return_dict=True)
        id1 = chain_directed.get_id_from_label("1")
        id2 = chain_directed.get_id_from_label("2")
        id5 = chain_directed.get_id_from_label("5")
        sim_12 = result[id1].get(id2, 0.0)
        sim_15 = result[id1].get(id5, 0.0)
        assert sim_12 >= sim_15, (
            f"Closer nodes should be more similar: sim(1,2)={sim_12}, sim(1,5)={sim_15}"
        )


class TestSimRankStarSimilarityPair:
    def test_pair_matches_full(self, diamond_directed):
        full = diamond_directed.simrank_star_similarity(return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        pair = diamond_directed.simrank_star_similarity_pair("2", "3")
        expected = full[id2].get(id3, 0.0)
        assert abs(pair - expected) < 1e-10

    def test_self_pair(self, diamond_directed):
        assert diamond_directed.simrank_star_similarity_pair("1", "1") == 1.0

    def test_invalid_node(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity_pair("1", "999")


class TestSimRankStarGraphMethod:
    def test_returns_graph(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity()
        assert isinstance(result, ec.Graph)

    def test_does_not_mutate_structure(self, diamond_directed):
        n = diamond_directed.count_nodes()
        e = diamond_directed.count_edges()
        diamond_directed.simrank_star_similarity()
        assert diamond_directed.count_nodes() == n
        assert diamond_directed.count_edges() == e

    def test_importance_factor_validation(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(importance_factor=1.0, return_dict=True)
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(importance_factor=-0.1, return_dict=True)

    def test_invalid_method(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(method="bad", return_dict=True)

    def test_different_importance_factors(self, diamond_directed):
        low = diamond_directed.simrank_star_similarity(importance_factor=0.3, return_dict=True)
        high = diamond_directed.simrank_star_similarity(importance_factor=0.9, return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        assert low[id2].get(id3, 0.0) < high[id2].get(id3, 0.0), (
            "Higher importance factor should give higher similarity"
        )
