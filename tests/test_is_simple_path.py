import pytest
from edgecase import Graph


class TestBasicValidation:
    """Test basic path validation."""

    def test_valid_linear_path(self):
        """Valid linear path A->B->C."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert g.is_simple_path(["A", "B", "C"]) is True

    def test_valid_two_node_path(self):
        """Valid path with two nodes."""
        g = Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")

        assert g.is_simple_path(["A", "B"]) is True

    def test_invalid_no_edge(self):
        """Invalid: nodes exist but no edge between them."""
        g = Graph().add_node(label="A").add_node(label="B").add_node(label="C").add_edge("A", "B")

        # No edge from A to C
        assert g.is_simple_path(["A", "C"]) is False

    def test_invalid_nonexistent_node(self):
        """Invalid: one node doesn't exist."""
        g = Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")

        assert g.is_simple_path(["A", "B", "Z"]) is False

    def test_invalid_repeated_node(self):
        """Invalid: node appears twice (not simple)."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # Cycle: A appears at start and end
        assert g.is_simple_path(["A", "B", "C", "A"]) is False


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_path(self):
        """Empty path is valid."""
        g = Graph().add_node(label="A")

        assert g.is_simple_path([]) is True

    def test_single_node_exists(self):
        """Single node path is valid if node exists."""
        g = Graph().add_node(label="A")

        assert g.is_simple_path(["A"]) is True

    def test_single_node_not_exists(self):
        """Single node path is invalid if node doesn't exist."""
        g = Graph().add_node(label="A")

        assert g.is_simple_path(["Z"]) is False

    def test_empty_graph(self):
        """Empty path in empty graph is valid."""
        g = Graph()

        assert g.is_simple_path([]) is True


class TestDirectedGraphs:
    """Test directed graph behavior."""

    def test_directed_valid_path(self):
        """Valid path in directed graph."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert g.is_simple_path(["A", "B", "C"]) is True

    def test_directed_wrong_direction(self):
        """Invalid: path goes against edge direction."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # No edge from C to B (only B->C exists)
        assert g.is_simple_path(["C", "B", "A"]) is False

    def test_directed_diamond(self):
        """Valid paths in directed diamond graph."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )

        assert g.is_simple_path(["A", "B", "D"]) is True
        assert g.is_simple_path(["A", "C", "D"]) is True
        # Invalid: no edge from B to C
        assert g.is_simple_path(["A", "B", "C", "D"]) is False


class TestUndirectedGraphs:
    """Test undirected graph behavior."""

    def test_undirected_bidirectional(self):
        """Both directions valid in undirected graph."""
        g = (
            Graph()  # undirected
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Forward
        assert g.is_simple_path(["A", "B", "C"]) is True
        # Backward
        assert g.is_simple_path(["C", "B", "A"]) is True

    def test_undirected_triangle(self):
        """Triangle graph with multiple valid paths."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        assert g.is_simple_path(["A", "B"]) is True
        assert g.is_simple_path(["B", "C"]) is True
        assert g.is_simple_path(["C", "A"]) is True
        assert g.is_simple_path(["A", "B", "C"]) is True
        assert g.is_simple_path(["C", "B", "A"]) is True


class TestComplexPaths:
    """Test more complex path scenarios."""

    def test_longer_path(self):
        """Valid longer path."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "E")
        )

        assert g.is_simple_path(["A", "B", "C", "D", "E"]) is True
        assert g.is_simple_path(["A", "B", "C"]) is True
        assert g.is_simple_path(["C", "D", "E"]) is True

    def test_skip_node_invalid(self):
        """Invalid: skip intermediate node."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        # No edge from A to C (must go through B)
        assert g.is_simple_path(["A", "C", "D"]) is False

    def test_grid_path(self):
        """Valid path through grid."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge("A", "B")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "E")
            .add_edge("C", "F")
            .add_edge("D", "E")
            .add_edge("E", "F")
        )

        # Valid L-shaped path
        assert g.is_simple_path(["A", "B", "E", "F"]) is True
        # Valid path with turn
        assert g.is_simple_path(["A", "D", "E", "B"]) is True


class TestIntegrationWithAllSimplePaths:
    """Test that is_simple_path validates paths from get_all_simple_paths."""

    def test_all_returned_paths_are_valid(self):
        """All paths from get_all_simple_paths should validate."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
            .add_edge("B", "C")
        )

        paths = g.get_all_simple_paths("A", "D")
        assert len(paths) > 0

        for path in paths:
            assert g.is_simple_path(path) is True

    def test_invalid_path_not_in_results(self):
        """Invalid paths should not be returned by get_all_simple_paths."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        paths = g.get_all_simple_paths("A", "C")

        # Invalid path with nonexistent node
        invalid_path = ["A", "Z", "C"]
        assert g.is_simple_path(invalid_path) is False
        assert invalid_path not in paths


class TestNodeIDsAndLabels:
    """Test with both integer IDs and string labels."""

    def test_integer_ids(self):
        """Test with integer node IDs - must use labels for this method."""
        # Note: is_simple_path works with labels, not raw integer IDs without labels
        # When nodes don't have explicit labels, use labeled nodes or access by ID as string
        g = (
            Graph()
            .add_node(id=1, label="1")
            .add_node(id=2, label="2")
            .add_node(id=3, label="3")
            .add_edge("1", "2")
            .add_edge("2", "3")
        )

        assert g.is_simple_path(["1", "2", "3"]) is True
        assert g.is_simple_path(["1", "3"]) is False  # no edge

    def test_string_labels(self):
        """Test with string labels."""
        g = (
            Graph()
            .add_node(label="Node1")
            .add_node(label="Node2")
            .add_node(label="Node3")
            .add_edge("Node1", "Node2")
            .add_edge("Node2", "Node3")
        )

        assert g.is_simple_path(["Node1", "Node2", "Node3"]) is True

    def test_mixed_ids_labels(self):
        """Test with mix of IDs and labels."""
        g = (
            Graph()
            .add_node(id=1, label="A")
            .add_node(id=2, label="B")
            .add_node(id=3, label="C")
            .add_edge(1, 2)
            .add_edge(2, 3)
        )

        # Can use either IDs or labels
        assert g.is_simple_path([1, 2, 3]) is True
        assert g.is_simple_path(["A", "B", "C"]) is True
        # Mixed (both resolve to same nodes)
        assert g.is_simple_path([1, "B", 3]) is True


class TestWeightedGraphs:
    """Test that weights don't affect path validation."""

    def test_weighted_path_valid(self):
        """Weights should not affect validation."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        assert g.is_simple_path(["A", "B", "C"]) is True

    def test_weighted_graph_invalid_path(self):
        """Invalid path still invalid regardless of weights."""
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
        )

        # No edge from B to C
        assert g.is_simple_path(["A", "B", "C"]) is False
