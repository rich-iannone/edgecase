import pytest
from edgecase import Graph


def labeled_trails(G, *args, **kwargs):
    """`enumerate_trails()` with each node ID mapped to its label."""
    return [[G.get_label_from_id(n) for n in t] for t in G.enumerate_trails(*args, **kwargs)]


class TestCountTrails:
    """Test count_trails() method."""

    def test_count_trails_triangle_with_labels(self):
        """Test counting trails in a triangle using labels."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        # Two trails from A to C: direct (A->C) and via B (A->B->C)
        count = G.count_trails("A", "C")
        assert count == 2

    def test_count_trails_triangle_with_ids(self):
        """Test counting trails in a triangle using node IDs."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        # Get node IDs
        a_id = G.get_id_from_label("A")
        c_id = G.get_id_from_label("C")

        # Should get same result as with labels
        count = G.count_trails(a_id, c_id)
        assert count == 2

    def test_count_trails_mixed_ids_and_labels(self):
        """Test counting trails with mixed ID and label inputs."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        a_id = G.get_id_from_label("A")

        # Mix ID and label
        count1 = G.count_trails(a_id, "C")
        count2 = G.count_trails("A", "C")
        assert count1 == count2 == 1

    def test_count_trails_simple_path(self):
        """Test counting trails in a simple path."""
        G = Graph()
        for i in range(4):
            G.add_node(label=str(i))
        for i in range(3):
            G.add_edge(str(i), str(i + 1))

        # Only one trail from 0 to 3 (linear path)
        count = G.count_trails("0", "3")
        assert count == 1

    def test_count_trails_no_path(self):
        """Test counting trails when no path exists."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        # No edges, no trails

        count = G.count_trails("A", "B")
        assert count == 0

    def test_count_trails_self_loop(self):
        """Test counting trails with same source and target."""
        G = Graph()
        G.add_node(label="A")
        G.add_edge("A", "A")

        # Trivial empty trail
        count = G.count_trails("A", "A")
        assert count == 1

    def test_count_trails_with_limit(self):
        """Test counting trails with max_trails limit."""
        G = Graph()
        # Create a graph with many trails
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_node(label="D")
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        # Multiple paths from A to D
        count_unlimited = G.count_trails("A", "D", max_trails=None)
        count_limited = G.count_trails("A", "D", max_trails=1)

        assert count_unlimited == 2
        # When limit is exceeded, should return None
        assert count_limited is None  # Exceeded limit of 1

    def test_count_trails_multiple_paths(self):
        """Test counting trails with multiple alternative paths."""
        G = Graph()
        # Diamond graph: A -> B,C -> D
        for label in ["A", "B", "C", "D"]:
            G.add_node(label=label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        count = G.count_trails("A", "D")
        assert count == 2  # Two paths: A->B->D and A->C->D

    def test_count_trails_nonexistent_node_label(self):
        """Test that nonexistent label raises error."""
        G = Graph()
        G.add_node(label="A")

        with pytest.raises(ValueError, match="not found"):
            G.count_trails("A", "nonexistent")

    def test_count_trails_nonexistent_node_id(self):
        """Test that nonexistent node ID raises error."""
        G = Graph()
        G.add_node(label="A")
        a_id = G.get_id_from_label("A")

        with pytest.raises(ValueError, match="does not exist"):
            G.count_trails(a_id, 99999)


class TestEnumerateTrails:
    """Test enumerate_trails() method."""

    def test_enumerate_trails_triangle_with_labels(self):
        """Test enumerating trails in a triangle using labels."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        trails = labeled_trails(G, "A", "C")

        assert len(trails) == 2
        # Check that we have both trails
        trail_strs = [" -> ".join(trail) for trail in trails]
        assert any("A" in t and "C" in t and "B" not in t for t in trail_strs)  # Direct
        assert any("A" in t and "B" in t and "C" in t for t in trail_strs)  # Via B

    def test_enumerate_trails_triangle_with_ids(self):
        """Test enumerating trails in a triangle using node IDs."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        # Get node IDs
        a_id = G.get_id_from_label("A")
        c_id = G.get_id_from_label("C")

        trails = G.enumerate_trails(a_id, c_id)

        assert len(trails) == 2
        # Results are node IDs
        assert all(trail[0] == a_id and trail[-1] == c_id for trail in trails)

    def test_enumerate_trails_mixed_ids_and_labels(self):
        """Test enumerating trails with mixed ID and label inputs."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_edge("A", "B")

        a_id = G.get_id_from_label("A")

        trails1 = labeled_trails(G, a_id, "B")
        trails2 = labeled_trails(G, "A", "B")

        assert len(trails1) == len(trails2) == 1
        assert trails1[0] == trails2[0]

    def test_enumerate_trails_simple_path(self):
        """Test enumerating trails in a simple path."""
        G = Graph()
        for i in range(4):
            G.add_node(label=str(i))
        for i in range(3):
            G.add_edge(str(i), str(i + 1))

        trails = labeled_trails(G, "0", "3")

        assert len(trails) == 1
        assert trails[0] == ["0", "1", "2", "3"]

    def test_enumerate_trails_no_path(self):
        """Test enumerating trails when no path exists."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")

        trails = labeled_trails(G, "A", "B")
        assert len(trails) == 0

    def test_enumerate_trails_with_limit(self):
        """Test enumerating trails with max_trails limit."""
        G = Graph()
        # Create graph with multiple trails
        for label in ["A", "B", "C", "D"]:
            G.add_node(label=label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        trails_all = labeled_trails(G, "A", "D", max_trails=None)
        trails_limited = labeled_trails(G, "A", "D", max_trails=1)

        assert len(trails_all) == 2
        assert len(trails_limited) == 1

    def test_enumerate_trails_returns_node_sequences(self):
        """Test that enumerate_trails returns node sequences."""
        G = Graph()
        G.add_node(label="start")
        G.add_node(label="middle")
        G.add_node(label="end")
        G.add_edge("start", "middle")
        G.add_edge("middle", "end")

        trails = labeled_trails(G, "start", "end")

        assert len(trails) == 1
        assert trails[0] == ["start", "middle", "end"]
        assert all(isinstance(node, str) for node in trails[0])

    def test_enumerate_trails_default_limit(self):
        """Test that enumerate_trails has a reasonable default limit."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_edge("A", "B")

        # Default limit should be 1000
        trails = labeled_trails(G, "A", "B")
        assert isinstance(trails, list)
        assert len(trails) <= 1000

    def test_enumerate_trails_multiple_paths(self):
        """Test enumerating multiple distinct trails."""
        G = Graph()
        # Create square with diagonal
        for label in ["A", "B", "C", "D"]:
            G.add_node(label=label)
        G.add_edge("A", "B")
        G.add_edge("B", "D")
        G.add_edge("A", "C")
        G.add_edge("C", "D")

        trails = labeled_trails(G, "A", "D")

        assert len(trails) == 2
        # Both trails should end at D
        assert all(trail[-1] == "D" for trail in trails)
        # Both trails should start at A
        assert all(trail[0] == "A" for trail in trails)

    def test_enumerate_trails_nonexistent_node_label(self):
        """Test that nonexistent label raises error."""
        G = Graph()
        G.add_node(label="A")

        with pytest.raises(ValueError, match="not found"):
            G.enumerate_trails("A", "nonexistent")

    def test_enumerate_trails_nonexistent_node_id(self):
        """Test that nonexistent node ID raises error."""
        G = Graph()
        G.add_node(label="A")
        a_id = G.get_id_from_label("A")

        with pytest.raises(ValueError, match="does not exist"):
            G.enumerate_trails(a_id, 99999)


class TestTrailsEdgeCases:
    """Test edge cases for trail functions."""

    def test_empty_graph(self):
        """Test trail functions on empty graph."""
        G = Graph()

        # Should raise error for nonexistent nodes
        with pytest.raises(ValueError):
            G.count_trails("A", "B")

        with pytest.raises(ValueError):
            G.enumerate_trails("A", "B")

    def test_single_node(self):
        """Test trail functions with single node."""
        G = Graph()
        G.add_node(label="A")

        # Same source and target
        count = G.count_trails("A", "A")
        assert count == 1  # Trivial empty trail

        trails = labeled_trails(G, "A", "A")
        assert len(trails) == 1
        assert trails[0] == ["A"]

    def test_disconnected_graph(self):
        """Test trail functions on disconnected components."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_node(label="D")
        G.add_edge("A", "B")
        G.add_edge("C", "D")

        # No path between components
        count = G.count_trails("A", "C")
        assert count == 0

        trails = labeled_trails(G, "A", "C")
        assert len(trails) == 0

    def test_directed_vs_undirected(self):
        """Test that trail counting respects graph directionality."""
        # Undirected graph
        G_undirected = Graph(directed=False)
        G_undirected.add_node(label="A")
        G_undirected.add_node(label="B")
        G_undirected.add_edge("A", "B")

        # Should work in both directions
        count_ab = G_undirected.count_trails("A", "B")
        count_ba = G_undirected.count_trails("B", "A")
        assert count_ab == count_ba == 1

        # Directed graph
        G_directed = Graph(directed=True)
        G_directed.add_node(label="A")
        G_directed.add_node(label="B")
        G_directed.add_edge("A", "B")

        # Should only work in one direction
        count_ab = G_directed.count_trails("A", "B")
        count_ba = G_directed.count_trails("B", "A")
        assert count_ab == 1
        assert count_ba == 0


class TestTrailsIntegration:
    """Integration tests for trail functions."""

    def test_count_and_enumerate_consistency(self):
        """Test that count_trails and enumerate_trails are consistent."""
        G = Graph()
        for i in range(5):
            G.add_node(label=str(i))
        G.add_edge("0", "1")
        G.add_edge("1", "2")
        G.add_edge("0", "3")
        G.add_edge("3", "2")
        G.add_edge("2", "4")

        # Count and enumeration should match
        count = G.count_trails("0", "4")
        trails = labeled_trails(G, "0", "4")
        assert count == len(trails)

    def test_trails_with_custom_labels(self):
        """Test trail functions with non-numeric labels."""
        G = Graph()
        labels = ["Alice", "Bob", "Charlie", "Diana"]
        for label in labels:
            G.add_node(label=label)
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Charlie")
        G.add_edge("Alice", "Diana")
        G.add_edge("Diana", "Charlie")

        count = G.count_trails("Alice", "Charlie")
        assert count == 2

        trails = labeled_trails(G, "Alice", "Charlie")
        assert len(trails) == 2
        assert all("Alice" in str(trail) and "Charlie" in str(trail) for trail in trails)

    def test_trails_preserve_path_structure(self):
        """Test that enumerated trails preserve correct path structure."""
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        trails = labeled_trails(G, "A", "C")

        assert len(trails) == 1
        trail = trails[0]
        # Check sequential connectivity
        assert trail[0] == "A"
        assert trail[1] == "B"
        assert trail[2] == "C"
