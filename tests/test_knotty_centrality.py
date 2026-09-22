"""Tests for knotty centrality."""

import pytest
import edgecase as ec


@pytest.fixture
def two_triangles():
    """Two triangles connected by a single bridge edge (C-D)."""
    return (
        ec.Graph(directed=False)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_node(label="F")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
        .add_edge("C", "D")  # bridge
        .add_edge("D", "E")
        .add_edge("E", "F")
        .add_edge("F", "D")
    )


@pytest.fixture
def star_graph():
    """Star with center node 1 and 6 leaves."""
    g = ec.Graph(directed=False)
    for i in range(1, 8):
        g.add_node(label=f"N{i}")
    for i in range(2, 8):
        g.add_edge("N1", f"N{i}")
    return g


@pytest.fixture
def two_cliques():
    """Two 5-cliques connected by two bridge edges."""
    g = ec.Graph(directed=False)
    for i in range(10):
        g.add_node(label=f"N{i}")
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(f"N{i}", f"N{j}")
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g.add_edge(f"N{i}", f"N{j}")
    g.add_edge("N3", "N7")
    g.add_edge("N4", "N5")
    return g


class TestKnottyCentrality:
    def test_returns_tuple(self, two_triangles):
        result = two_triangles.knotty_centrality()
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, core = result
        assert isinstance(score, float)
        assert isinstance(core, list)

    def test_bridge_nodes_are_core(self, two_triangles):
        score, core = two_triangles.knotty_centrality()
        labels = {two_triangles.get_label_from_id(nid) for nid in core}
        assert "C" in labels
        assert "D" in labels

    def test_score_in_range(self, two_triangles):
        score, _ = two_triangles.knotty_centrality()
        assert 0.0 <= score <= 1.0

    def test_core_has_at_least_two_nodes(self, two_triangles):
        _, core = two_triangles.knotty_centrality()
        assert len(core) >= 2

    def test_core_nodes_sorted(self, two_triangles):
        _, core = two_triangles.knotty_centrality()
        assert core == sorted(core)

    def test_compact_penalizes(self, two_triangles):
        score, _ = two_triangles.knotty_centrality(compact=False)
        score_c, _ = two_triangles.knotty_centrality(compact=True)
        assert score_c <= score

    def test_star_center_in_core(self, star_graph):
        _, core = star_graph.knotty_centrality()
        center_id = star_graph.get_id_from_label("N1")
        assert center_id in core

    def test_two_cliques_bridge_nodes(self, two_cliques):
        _, core = two_cliques.knotty_centrality()
        labels = {two_cliques.get_label_from_id(nid) for nid in core}
        bridge_nodes = {"N3", "N4", "N5", "N7"}
        assert len(labels & bridge_nodes) >= 2

    def test_empty_graph(self):
        g = ec.Graph()
        score, core = g.knotty_centrality()
        assert score == 0.0
        assert core == []

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        score, core = g.knotty_centrality()
        assert score == 0.0
        assert core == []

    def test_two_connected_nodes(self):
        g = ec.Graph(directed=False).add_node(label="A").add_node(label="B")
        g.add_edge("A", "B")
        score, core = g.knotty_centrality()
        # Two nodes with an edge: density=1, but BC is 0 for both
        assert score == 0.0

    def test_triangle(self):
        g = (
            ec.Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        score, core = g.knotty_centrality()
        # All nodes have equal BC in a triangle; all subsets of size 2 or 3
        # are valid. With 3 nodes all connected, density=1.
        assert score >= 0.0

    def test_top_m_parameter(self, two_cliques):
        score_small, _ = two_cliques.knotty_centrality(top_m=3)
        score_large, _ = two_cliques.knotty_centrality(top_m=8)
        # Larger top_m explores more subsets, should find >= score
        assert score_large >= score_small

    def test_does_not_mutate_graph(self, two_triangles):
        edges_before = two_triangles.count_edges()
        nodes_before = two_triangles.count_nodes()
        two_triangles.knotty_centrality()
        assert two_triangles.count_edges() == edges_before
        assert two_triangles.count_nodes() == nodes_before
