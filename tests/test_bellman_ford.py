"""Tests for Bellman-Ford shortest path algorithm."""

import pytest
from edgecase import Graph


class TestBellmanFordBasic:
    """Basic Bellman-Ford functionality tests."""

    def test_simple_path(self):
        """Test shortest path in simple weighted graph."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert distance == 3.0

    def test_negative_weights(self):
        """Test with negative edge weights."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=4.0)
            .add_edge("A", "C", weight=2.0)
            .add_edge("C", "B", weight=-3.0)  # Negative weight
            .add_edge("B", "D", weight=1.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "D")
        # Should go A -> C -> B -> D to take advantage of negative weight
        assert path == ["A", "C", "B", "D"]
        assert distance == 0.0  # 2 + (-3) + 1 = 0

    def test_direct_connection(self):
        """Test direct edge between nodes."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B", weight=5.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "B")
        assert path == ["A", "B"]
        assert distance == 5.0

    def test_no_path(self):
        """Test when no path exists."""
        graph = (
            Graph(directed=True).add_node(label="A").add_node(label="B").add_node(label="isolated")
        )

        result = graph.bellman_ford_shortest_path("A", "isolated")
        assert result is None


class TestBellmanFordNegativeCycles:
    """Tests for negative cycle detection."""

    def test_negative_cycle_detection(self):
        """Test that negative cycles are detected."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=-3.0)
            .add_edge("C", "A", weight=1.0)  # Creates negative cycle: total = -1
        )

        with pytest.raises(ValueError, match="negative weight cycle"):
            graph.bellman_ford_shortest_path("A", "B")

    def test_negative_cycle_self_loop(self):
        """Test negative self-loop detection."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "A", weight=-1.0)  # Negative self-loop
            .add_edge("A", "B", weight=1.0)
        )

        with pytest.raises(ValueError, match="negative weight cycle"):
            graph.bellman_ford_shortest_path("A", "B")


class TestBellmanFordComplexGraphs:
    """Tests with more complex graph structures."""

    def test_multiple_paths_chooses_shortest(self):
        """Test that algorithm finds truly shortest path."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=10.0)
            .add_edge("A", "C", weight=1.0)
            .add_edge("C", "D", weight=1.0)
            .add_edge("D", "B", weight=1.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "B")
        # Should go A -> C -> D -> B (total: 3) instead of A -> B (10)
        assert path == ["A", "C", "D", "B"]
        assert distance == 3.0

    def test_star_graph(self):
        """Test with star topology."""
        graph = (
            Graph(directed=True)
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A", weight=1.0)
            .add_edge("hub", "B", weight=2.0)
            .add_edge("hub", "C", weight=3.0)
        )

        path, distance = graph.bellman_ford_shortest_path("hub", "C")
        assert path == ["hub", "C"]
        assert distance == 3.0

    def test_mixed_positive_negative_weights(self):
        """Test graph with both positive and negative weights."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=5.0)
            .add_edge("B", "C", weight=-2.0)
            .add_edge("C", "D", weight=3.0)
            .add_edge("A", "D", weight=10.0)  # Alternative longer path
        )

        path, distance = graph.bellman_ford_shortest_path("A", "D")
        # Should be A -> B -> C -> D = 5 + (-2) + 3 = 6
        assert path == ["A", "B", "C", "D"]
        assert distance == 6.0


class TestBellmanFordEdgeCases:
    """Edge case tests."""

    def test_single_node(self):
        """Test path from node to itself."""
        graph = Graph(directed=True).add_node(label="A")

        path, distance = graph.bellman_ford_shortest_path("A", "A")
        assert path == ["A"]
        assert distance == 0.0

    def test_zero_weight_edges(self):
        """Test with zero-weight edges."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=0.0)
            .add_edge("B", "C", weight=0.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert distance == 0.0

    def test_large_negative_weights(self):
        """Test with large negative weights (no cycle)."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=100.0)
            .add_edge("B", "C", weight=-50.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert distance == 50.0


class TestBellmanFordUndirected:
    """Test Bellman-Ford on undirected graphs."""

    def test_undirected_graph(self):
        """Test on undirected graph."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert distance == 3.0

    def test_undirected_negative_weights(self):
        """Test undirected graph with negative weights."""
        # Note: Negative weights in undirected graphs can easily create cycles
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=5.0)
            .add_edge("B", "C", weight=3.0)
        )

        path, distance = graph.bellman_ford_shortest_path("A", "C")
        assert distance == 8.0
