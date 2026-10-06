import pytest
from edgecase import Graph


class TestBasicPaths:
    """Test basic path finding scenarios."""

    def test_single_path(self):
        """Test graph with only one path between nodes."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        paths = g.get_all_simple_paths( "A", "C")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]

    def test_two_paths(self):
        """Test diamond graph with two paths."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        paths = g.get_all_simple_paths( "A", "D")
        assert len(paths) == 2
        # Sort paths for consistent comparison
        paths_sorted = sorted([tuple(p) for p in paths])
        expected = sorted([("A", "B", "D"), ("A", "C", "D")])
        assert paths_sorted == expected

    def test_multiple_paths(self):
        """Test graph with multiple alternative paths."""
        g = Graph()
        # Create: A -> B -> D
        #         A -> C -> D
        #         A -> D (shortcut)
        #         B -> C (cross connection enabling A->B->C->D and A->C->B->D)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        paths = g.get_all_simple_paths( "A", "D")
        assert len(paths) == 5
        # Paths: A->D, A->B->D, A->C->D, A->B->C->D, A->C->B->D

    def test_direct_edge(self):
        """Test direct connection between source and target."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        paths = g.get_all_simple_paths( "A", "B")
        assert len(paths) == 1
        assert paths[0] == ["A", "B"]

    def test_self_loop(self):
        """Test source equals target."""
        g = Graph()
        g.add_node(label="A")

        paths = g.get_all_simple_paths( "A", "A")
        # Should return path with just the node itself
        assert len(paths) == 1
        assert paths[0] == ["A"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_path(self):
        """Test disconnected nodes."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        # No edge between them

        paths = g.get_all_simple_paths( "A", "B")
        assert len(paths) == 0

    def test_nonexistent_source(self):
        """Test with nonexistent source node."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(Exception):  # Should raise ValueError or similar
            g.get_all_simple_paths( "B", "A")

    def test_nonexistent_target(self):
        """Test with nonexistent target node."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(Exception):  # Should raise ValueError or similar
            g.get_all_simple_paths( "A", "B")

    def test_empty_graph(self):
        """Test with empty graph."""
        g = Graph()

        with pytest.raises(Exception):  # Should raise error
            g.get_all_simple_paths( "A", "B")


class TestDirectedGraphs:
    """Test with directed graphs."""

    def test_directed_path(self):
        """Test directed graph respects edge direction."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # Forward direction should work
        paths_forward = g.get_all_simple_paths( "A", "C")
        assert len(paths_forward) == 1
        assert paths_forward[0] == ["A", "B", "C"]

        # Reverse direction should not work
        paths_backward = g.get_all_simple_paths( "C", "A")
        assert len(paths_backward) == 0

    def test_directed_diamond(self):
        """Test directed diamond with specific flow."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        # A -> B -> D
        # A -> C -> D
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        paths = g.get_all_simple_paths( "A", "D")
        assert len(paths) == 2

    def test_directed_cycle(self):
        """Test directed graph with cycle doesn't create infinite paths."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Cycle back

        # Should not get stuck in cycle
        paths = g.get_all_simple_paths( "A", "C")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]


class TestUndirectedGraphs:
    """Test with undirected graphs."""

    def test_undirected_bidirectional(self):
        """Test undirected graph allows both directions."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # Both directions should work
        paths_forward = g.get_all_simple_paths( "A", "C")
        assert len(paths_forward) == 1

        paths_backward = g.get_all_simple_paths( "C", "A")
        assert len(paths_backward) == 1

    def test_undirected_triangle(self):
        """Test undirected triangle has multiple paths."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Should find 2 paths in triangle: direct and going around
        paths = g.get_all_simple_paths( "A", "C")
        assert len(paths) == 2


class TestCutoff:
    """Test cutoff parameter for limiting path length."""

    def test_cutoff_limits_length(self):
        """Test cutoff restricts maximum path length."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("A", "D")  # Shortcut

        # Without cutoff: should find 2 paths
        all_paths = g.get_all_simple_paths( "A", "D")
        assert len(all_paths) == 2

        # With cutoff=1: only direct path
        short_paths = g.get_all_simple_paths( "A", "D", cutoff=1)
        assert len(short_paths) == 1
        assert len(short_paths[0]) == 2  # 2 nodes = 1 edge

        # With cutoff=2: both paths should be found
        medium_paths = g.get_all_simple_paths( "A", "D", cutoff=3)
        assert len(medium_paths) == 2

    def test_cutoff_zero(self):
        """Test cutoff=0 (no edges allowed)."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # cutoff=0 should not allow any traversal
        paths = g.get_all_simple_paths( "A", "B", cutoff=0)
        assert len(paths) == 0

    def test_cutoff_exact_length(self):
        """Test cutoff matching exact path length."""
        g = Graph()
        # Create linear path A-B-C-D (3 edges)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        # cutoff=3 should find the path (exactly 3 edges)
        paths = g.get_all_simple_paths( "A", "D", cutoff=3)
        assert len(paths) == 1
        assert len(paths[0]) == 4  # 4 nodes = 3 edges

        # cutoff=2 should not find the path (need 3 edges)
        paths_short = g.get_all_simple_paths( "A", "D", cutoff=2)
        assert len(paths_short) == 0


class TestComplexGraphs:
    """Test with more complex graph structures."""

    def test_grid_graph(self):
        """Test on a small grid graph."""
        # Create 2x3 grid:
        # A - B - C
        # |   |   |
        # D - E - F
        g = Graph()
        for label in ["A", "B", "C", "D", "E", "F"]:
            g.add_node(label=label)

        # Horizontal edges
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("D", "E")
        g.add_edge("E", "F")

        # Vertical edges
        g.add_edge("A", "D")
        g.add_edge("B", "E")
        g.add_edge("C", "F")

        # Find all paths from A to F
        paths = g.get_all_simple_paths( "A", "F")
        # Many possible paths in a grid
        assert len(paths) > 0

    def test_complete_graph(self):
        """Test on complete graph (all nodes connected)."""
        # K4 complete graph
        g = Graph()
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            g.add_node(label=node)

        # Connect all pairs
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        # In K4, there should be many paths between any two nodes
        paths = g.get_all_simple_paths( "A", "D")
        # Direct + via B + via C + via B,C
        assert len(paths) >= 4

    def test_tree_structure(self):
        """Test on tree (no cycles)."""
        # Binary tree
        #       A
        #      / \
        #     B   C
        #    / \   \
        #   D   E   F
        g = Graph()
        for label in ["A", "B", "C", "D", "E", "F"]:
            g.add_node(label=label)

        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("B", "E")
        g.add_edge("C", "F")

        # In a tree, there's exactly one path between any two nodes
        paths = g.get_all_simple_paths( "D", "F")
        assert len(paths) == 1
        assert "A" in paths[0]  # Must go through root


class TestLargerGraphs:
    """Test performance and correctness on larger graphs."""

    def test_linear_chain(self):
        """Test on long linear chain."""
        g = Graph()
        n = 20
        labels = [str(i) for i in range(n)]

        for label in labels:
            g.add_node(label=label)

        for i in range(n - 1):
            g.add_edge(str(i), str(i + 1))

        # Only one path in a chain
        paths = g.get_all_simple_paths( "0", str(n - 1))
        assert len(paths) == 1
        assert len(paths[0]) == n

    def test_star_graph(self):
        """Test on star graph (one central node)."""
        g = Graph()
        g.add_node(label="center")
        n_spokes = 10

        for i in range(n_spokes):
            g.add_node(label=f"spoke_{i}")
            g.add_edge("center", f"spoke_{i}")

        # Path from one spoke to another must go through center
        paths = g.get_all_simple_paths( "spoke_0", "spoke_5")
        assert len(paths) == 1
        assert "center" in paths[0]
        assert len(paths[0]) == 3  # spoke_0 -> center -> spoke_5


class TestWeightedGraphs:
    """Test that simple paths work with weighted graphs."""

    def test_weighted_graph(self):
        """Test that weights don't affect simple path enumeration."""
        g = Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")

        # Create paths with different weights
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "D", weight=1.0)
        g.add_edge("A", "C", weight=10.0)
        g.add_edge("C", "D", weight=10.0)

        # Should find both paths regardless of weight
        paths = g.get_all_simple_paths( "A", "D")
        assert len(paths) == 2
        # Simple path enumeration doesn't consider weights


class TestUnlabelledAndMissingEndpoints:
    """Regression: TypeError for an unlabelled endpoint or a nonexistent integer ID."""

    def test_unlabelled_endpoints(self):
        g = Graph().add_cycle_graph(4, label=False)
        a, b, c, d = sorted(g.get_node_ids())
        paths = g.get_all_simple_paths(a, c)
        assert sorted(paths) == sorted([[str(a), str(b), str(c)], [str(a), str(d), str(c)]])

    def test_missing_integer_id_raises_value_error(self):
        g = Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="Node 999 does not exist"):
            g.get_all_simple_paths(999, 1)
        with pytest.raises(ValueError, match="Node 999 does not exist"):
            g.get_all_simple_paths(1, 999)
