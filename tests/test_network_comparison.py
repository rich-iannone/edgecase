"""Tests for network comparison: edge overlap, differential edges, graph edit distance."""

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph_abc():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )


@pytest.fixture
def graph_abd():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("B", "D")
    )


@pytest.fixture
def empty_graph():
    return ec.Graph()


@pytest.fixture
def triangle():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
    )


# =============================================================================
# Edge Overlap
# =============================================================================


class TestEdgeOverlap:
    def test_identical_graphs(self, triangle):
        ov = triangle.edge_overlap(triangle)
        assert ov["jaccard"] == pytest.approx(1.0)
        assert ov["g1_only"] == 0.0
        assert ov["g2_only"] == 0.0

    def test_disjoint_graphs(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")
        g2 = ec.Graph().add_node(label="C").add_node(label="D").add_edge("C", "D")
        ov = g1.edge_overlap(g2)
        assert ov["jaccard"] == pytest.approx(0.0)
        assert ov["overlap"] == 0.0

    def test_partial_overlap(self, graph_abc, graph_abd):
        ov = graph_abc.edge_overlap(graph_abd)
        assert ov["overlap"] == 1.0
        assert ov["g1_only"] == 1.0
        assert ov["g2_only"] == 1.0
        assert ov["union_size"] == 3.0
        assert ov["jaccard"] == pytest.approx(1.0 / 3.0)

    def test_empty_graphs(self, empty_graph):
        ov = empty_graph.edge_overlap(empty_graph)
        assert ov["jaccard"] == pytest.approx(1.0)

    def test_one_empty_graph(self, triangle, empty_graph):
        ov = triangle.edge_overlap(empty_graph)
        assert ov["jaccard"] == pytest.approx(0.0)
        assert ov["g1_only"] == 3.0
        assert ov["g2_only"] == 0.0

    def test_superset_graph(self, graph_abc, triangle):
        ov = graph_abc.edge_overlap(triangle)
        assert ov["overlap"] == 2.0
        assert ov["g1_only"] == 0.0
        assert ov["g2_only"] == 1.0


# =============================================================================
# Differential Edges
# =============================================================================


class TestDifferentialEdges:
    def test_identical_graphs(self, triangle):
        diff = triangle.differential_edges(triangle)
        assert len(diff["added"]) == 0
        assert len(diff["removed"]) == 0

    def test_disjoint_graphs(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")
        g2 = ec.Graph().add_node(label="C").add_node(label="D").add_edge("C", "D")
        diff = g1.differential_edges(g2)
        assert len(diff["removed"]) == 1
        assert len(diff["added"]) == 1

    def test_partial_overlap(self, graph_abc, graph_abd):
        diff = graph_abc.differential_edges(graph_abd)
        assert len(diff["removed"]) == 1
        assert len(diff["added"]) == 1
        removed_labels = {tuple(sorted(e)) for e in diff["removed"]}
        added_labels = {tuple(sorted(e)) for e in diff["added"]}
        assert ("B", "C") in removed_labels
        assert ("B", "D") in added_labels

    def test_empty_vs_nonempty(self, triangle, empty_graph):
        diff = triangle.differential_edges(empty_graph)
        assert len(diff["removed"]) == 3
        assert len(diff["added"]) == 0

    def test_nonempty_vs_empty(self, empty_graph, triangle):
        diff = empty_graph.differential_edges(triangle)
        assert len(diff["removed"]) == 0
        assert len(diff["added"]) == 3

    def test_superset_only_adds(self, graph_abc, triangle):
        diff = graph_abc.differential_edges(triangle)
        assert len(diff["removed"]) == 0
        assert len(diff["added"]) == 1


# =============================================================================
# Graph Edit Distance
# =============================================================================


class TestGraphEditDistance:
    def test_identical_graphs(self, triangle):
        ged = triangle.graph_edit_distance(triangle)
        assert ged["distance"] == pytest.approx(0.0)

    def test_empty_graphs(self, empty_graph):
        ged = empty_graph.graph_edit_distance(empty_graph)
        assert ged["distance"] == pytest.approx(0.0)

    def test_one_edge_difference(self, graph_abc, graph_abd):
        ged = graph_abc.graph_edit_distance(graph_abd)
        assert ged["node_insertions"] == 1.0
        assert ged["node_deletions"] == 1.0
        assert ged["edge_insertions"] == 1.0
        assert ged["edge_deletions"] == 1.0
        assert ged["distance"] == pytest.approx(4.0)

    def test_custom_costs(self, graph_abc, graph_abd):
        ged = graph_abc.graph_edit_distance(graph_abd, node_cost=2.0, edge_cost=0.5)
        expected = 2 * 2.0 + 2 * 0.5
        assert ged["distance"] == pytest.approx(expected)

    def test_empty_to_nonempty(self, empty_graph, triangle):
        ged = empty_graph.graph_edit_distance(triangle)
        assert ged["node_insertions"] == 3.0
        assert ged["node_deletions"] == 0.0
        assert ged["edge_insertions"] == 3.0
        assert ged["edge_deletions"] == 0.0
        assert ged["distance"] == pytest.approx(6.0)

    def test_nonempty_to_empty(self, triangle, empty_graph):
        ged = triangle.graph_edit_distance(empty_graph)
        assert ged["node_deletions"] == 3.0
        assert ged["node_insertions"] == 0.0
        assert ged["edge_deletions"] == 3.0
        assert ged["edge_insertions"] == 0.0
        assert ged["distance"] == pytest.approx(6.0)

    def test_symmetry(self, graph_abc, graph_abd):
        ged1 = graph_abc.graph_edit_distance(graph_abd)
        ged2 = graph_abd.graph_edit_distance(graph_abc)
        assert ged1["distance"] == pytest.approx(ged2["distance"])

    def test_superset_distance(self, graph_abc, triangle):
        ged = graph_abc.graph_edit_distance(triangle)
        assert ged["node_insertions"] == 0.0
        assert ged["node_deletions"] == 0.0
        assert ged["edge_insertions"] == 1.0
        assert ged["edge_deletions"] == 0.0
