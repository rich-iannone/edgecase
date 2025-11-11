"""
Tests for Anytime Dynamic A* (ADA*) algorithm.

Tests cover:
- Basic pathfinding
- Anytime behavior (epsilon refinement)
- Dynamic graph updates
- Custom heuristics
- Edge cases
"""

import pytest
import edgecase as ec


def add_weighted_edges(graph, edge_list):
    """Helper to add edges with weights, creating nodes as needed."""
    nodes = set()
    for src, dst, attrs in edge_list:
        nodes.add(src)
        nodes.add(dst)

    for node in nodes:
        try:
            graph.add_node(label=node)
        except:
            pass

    for src, dst, attrs in edge_list:
        graph.add_edge(src, dst, **attrs)


class TestAdaStarBasic:
    """Basic pathfinding tests."""

    def test_simple_path(self):
        """Test finding a simple path."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("C", "D", {"weight": 1.0}),
            ],
        )

        path = g.ada_star_search("A", "D", epsilon=1.0)

        assert path == ["A", "B", "C", "D"]

    def test_shortest_of_two_paths(self):
        """Test choosing shortest of two paths."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("A", "D", {"weight": 5.0}),
                ("D", "C", {"weight": 5.0}),
            ],
        )

        path = g.ada_star_search("A", "C", epsilon=1.0)

        # Should take shorter path A -> B -> C (cost 2.0) over A -> D -> C (cost 10.0)
        assert path == ["A", "B", "C"]

    def test_weighted_path(self):
        """Test path with different edge weights."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 10.0}),
                ("A", "C", {"weight": 2.0}),
                ("C", "D", {"weight": 2.0}),
                ("D", "B", {"weight": 2.0}),
            ],
        )

        path = g.ada_star_search("A", "B", epsilon=1.0)

        # Should take A -> C -> D -> B (cost 6.0) over A -> B (cost 10.0)
        assert path == ["A", "C", "D", "B"]


class TestAdaStarAnytime:
    """Test anytime behavior with epsilon refinement."""

    def test_epsilon_refinement(self):
        """Test that lower epsilon gives better or equal paths."""
        g = ec.Graph()
        # Create a graph where greedy choice differs from optimal
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("C", "D", {"weight": 1.0}),
                ("A", "X", {"weight": 0.5}),
                ("X", "Y", {"weight": 0.5}),
                ("Y", "Z", {"weight": 0.5}),
                ("Z", "D", {"weight": 0.5}),
            ],
        )

        # High epsilon - may find suboptimal path quickly
        path_high = g.ada_star_search("A", "D", epsilon=10.0, reuse_planner=True)
        assert isinstance(path_high, list)
        assert path_high[0] == "A"
        assert path_high[-1] == "D"

        # Lower epsilon - should find better or equal path
        path_low = g.ada_star_search("A", "D", epsilon=1.5, reuse_planner=True)
        assert isinstance(path_low, list)

        # Optimal epsilon - should find best path
        path_optimal = g.ada_star_search("A", "D", epsilon=1.0, reuse_planner=True)
        assert path_optimal == ["A", "X", "Y", "Z", "D"]

    def test_repeated_calls_same_epsilon(self):
        """Test that repeated calls with same epsilon return same path."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
            ],
        )

        path1 = g.ada_star_search("A", "C", epsilon=1.0, reuse_planner=True)
        path2 = g.ada_star_search("A", "C", epsilon=1.0, reuse_planner=True)

        assert path1 == path2

    def test_reuse_planner_parameter(self):
        """Test reuse_planner parameter for iterative refinement without separate class."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("C", "D", {"weight": 1.0}),
                ("A", "X", {"weight": 0.5}),
                ("X", "Y", {"weight": 0.5}),
                ("Y", "Z", {"weight": 0.5}),
                ("Z", "D", {"weight": 0.5}),
            ],
        )

        # Use reuse_planner=True for iterative refinement
        path_high = g.ada_star_search("A", "D", epsilon=10.0, reuse_planner=True)
        assert isinstance(path_high, list)
        assert path_high[0] == "A"
        assert path_high[-1] == "D"

        # Lower epsilon - should reuse planner state
        path_optimal = g.ada_star_search("A", "D", epsilon=1.0, reuse_planner=True)
        assert path_optimal == ["A", "X", "Y", "Z", "D"]

        # Without reuse_planner, creates fresh planner each time
        path_fresh = g.ada_star_search("A", "D", epsilon=1.0, reuse_planner=False)
        assert path_fresh == ["A", "X", "Y", "Z", "D"]


class TestAdaStarDynamic:
    """Test dynamic graph updates."""

    def test_edge_weight_increase(self):
        """Test handling edge weight increase."""
        # Skip for now - need to implement edge weight update API
        pytest.skip("Edge weight updates require API to modify existing edges")

    def test_edge_weight_decrease(self):
        """Test handling edge weight decrease."""
        # Skip for now - need to implement edge weight update API
        pytest.skip("Edge weight updates require API to modify existing edges")


class TestAdaStarHeuristics:
    """Test custom heuristics."""

    def test_zero_heuristic(self):
        """Test with zero heuristic (equivalent to Dijkstra)."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("A", "C", {"weight": 3.0}),
            ],
        )

        def zero_heuristic(u, v):
            return 0.0

        path = g.ada_star_search("A", "C", epsilon=1.0, heuristic=zero_heuristic)

        assert path == ["A", "B", "C"]

    def test_constant_heuristic(self):
        """Test with constant heuristic."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("A", "C", {"weight": 3.0}),
            ],
        )

        def constant_heuristic(u, v):
            return 1.0

        path = g.ada_star_search("A", "C", epsilon=1.0, heuristic=constant_heuristic)

        # Still should find optimal path
        assert path == ["A", "B", "C"]


class TestAdaStarEdgeCases:
    """Test edge cases and error handling."""

    def test_single_node_path(self):
        """Test path from node to itself."""
        g = ec.Graph()
        g.add_node("A")

        path = g.ada_star_search("A", "A", epsilon=1.0)

        assert path == ["A"]

    def test_two_node_path(self):
        """Test simple two-node path."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=1.0)

        path = g.ada_star_search("A", "B", epsilon=1.0)

        assert path == ["A", "B"]

    def test_no_path_exists(self):
        """Test when no path exists between nodes."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        # C is disconnected

        with pytest.raises(ValueError, match="No path exists"):
            g.ada_star_search("A", "C", epsilon=1.0)


class TestAdaStarComparison:
    """Compare ADA* with other pathfinding algorithms."""

    def test_compare_with_dijkstra(self):
        """Test that ADA* with epsilon=1.0 finds optimal paths."""
        g = ec.Graph()
        add_weighted_edges(
            g,
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 2.0}),
                ("A", "D", {"weight": 2.0}),
                ("D", "C", {"weight": 1.0}),
                ("B", "E", {"weight": 1.0}),
                ("E", "C", {"weight": 1.0}),
            ],
        )

        # ADA* with epsilon=1.0 should find optimal path
        ada_path = g.ada_star_search("A", "C", epsilon=1.0)

        # EdgeCase's shortest_path uses Dijkstra
        dijkstra_path = ec.shortest_path(g, "A", "C")

        # Both should find optimal paths
        # Multiple optimal paths may exist, so we check they're both valid
        assert ada_path[0] == "A" and ada_path[-1] == "C"
        assert dijkstra_path[0] == "A" and dijkstra_path[-1] == "C"

        # For this graph, optimal path is A->B->E->C or A->D->C (both cost 3.0)
        assert len(ada_path) == 3  # Should be a 3-hop path
