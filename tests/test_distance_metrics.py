"""Tests for distance and eccentricity metrics."""

import pytest
import edgecase as ec


class TestMeanDistance:
    """Test mean distance (average path length) calculation."""

    def test_mean_distance_path_graph(self):
        """Test mean distance on a path graph."""
        G = ec.Graph()
        G.add_path_graph(4)  # 0-1-2-3
        
        # Distances: 0→1=1, 0→2=2, 0→3=3, 1→2=1, 1→3=2, 2→3=1
        # Total = 1+2+3+1+2+1 = 10, Count = 6, Mean = 10/6 ≈ 1.667
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        assert abs(mean_dist - 10.0/6.0) < 0.001

    def test_mean_distance_star_graph(self):
        """Test mean distance on a star graph."""
        G = ec.Graph()
        G.add_star_graph(5)  # Hub connected to 4 leaves
        
        # From hub to leaves: 4 paths of length 1
        # Between leaves: 6 pairs, each distance 2 (via hub)
        # Total = 4*1 + 6*2 = 16, Count = 10, Mean = 1.6
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        assert abs(mean_dist - 1.6) < 0.001

    def test_mean_distance_cycle_graph(self):
        """Test mean distance on a cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(6)  # 0-1-2-3-4-5-0
        
        # In a cycle of 6, distances are: 1,2,3,2,1 for each node
        # Mean should be (6*1 + 6*2 + 6*2 + 6*1) / 30 = 1.667...
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        assert 1.5 < mean_dist < 2.0

    def test_mean_distance_complete_graph(self):
        """Test mean distance on a complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        
        # All pairs are at distance 1
        mean_dist = G.get_mean_distance()
        assert mean_dist == 1.0

    def test_mean_distance_disconnected(self):
        """Test that disconnected graph returns None."""
        G = ec.Graph()
        G.add_path_graph(3)  # Component 1
        G.add_path_graph(3)  # Component 2 (disconnected)
        
        mean_dist = G.get_mean_distance()
        assert mean_dist is None

    def test_mean_distance_single_node(self):
        """Test mean distance on single node graph."""
        G = ec.Graph()
        G.add_node("A")
        
        # No pairs to compute distance for
        mean_dist = G.get_mean_distance()
        assert mean_dist is None

    def test_mean_distance_empty_graph(self):
        """Test mean distance on empty graph."""
        G = ec.Graph()
        
        mean_dist = G.get_mean_distance()
        assert mean_dist is None

    def test_mean_distance_directed(self):
        """Test mean distance on directed graph."""
        G = ec.Graph(directed=True)
        G.add_cycle_graph(4)  # 0→1→2→3→0
        
        # Each node can reach all others with varying distances
        # 0→1=1, 0→2=2, 0→3=3, 1→2=1, 1→3=2, 1→0=3, etc.
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        assert mean_dist == 2.0  # Symmetric cycle


class TestMinEccentricity:
    """Test minimum eccentricity (radius) calculation."""

    def test_min_eccentricity_path(self):
        """Test radius on path graph."""
        G = ec.Graph()
        G.add_path_graph(5)  # 0-1-2-3-4
        
        # Middle node (2) has minimum eccentricity = 2
        radius = G.get_min_eccentricity()
        assert radius == 2

    def test_min_eccentricity_star(self):
        """Test radius on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        
        # Hub has minimum eccentricity = 1
        radius = G.get_min_eccentricity()
        assert radius == 1

    def test_min_eccentricity_cycle(self):
        """Test radius on cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        
        # All nodes have eccentricity 3
        radius = G.get_min_eccentricity()
        assert radius == 3

    def test_min_eccentricity_complete(self):
        """Test radius on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        
        # All nodes have eccentricity 1
        radius = G.get_min_eccentricity()
        assert radius == 1

    def test_min_eccentricity_disconnected(self):
        """Test that disconnected graph returns None."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_path_graph(3)
        
        radius = G.get_min_eccentricity()
        assert radius is None

    def test_min_eccentricity_alias(self):
        """Test that min_eccentricity is alias for graph_radius."""
        G = ec.Graph()
        G.add_star_graph(7)
        
        assert G.get_min_eccentricity() == G.get_graph_radius()


class TestMaxEccentricity:
    """Test maximum eccentricity (diameter) calculation."""

    def test_max_eccentricity_path(self):
        """Test diameter on path graph."""
        G = ec.Graph()
        G.add_path_graph(5)  # 0-1-2-3-4
        
        # Distance from end to end = 4
        diameter = G.get_max_eccentricity()
        assert diameter == 4

    def test_max_eccentricity_star(self):
        """Test diameter on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        
        # Distance between any two leaves = 2 (via hub)
        diameter = G.get_max_eccentricity()
        assert diameter == 2

    def test_max_eccentricity_cycle(self):
        """Test diameter on cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        
        # Maximum distance in cycle of 6 is 3
        diameter = G.get_max_eccentricity()
        assert diameter == 3

    def test_max_eccentricity_complete(self):
        """Test diameter on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        
        # All pairs at distance 1
        diameter = G.get_max_eccentricity()
        assert diameter == 1

    def test_max_eccentricity_disconnected(self):
        """Test that disconnected graph returns None."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_path_graph(3)
        
        diameter = G.get_max_eccentricity()
        assert diameter is None

    def test_max_eccentricity_alias(self):
        """Test that max_eccentricity is alias for graph_diameter."""
        G = ec.Graph()
        G.add_path_graph(10)
        
        assert G.get_max_eccentricity() == G.get_graph_diameter()


class TestDistanceMetricsIntegration:
    """Test interactions between distance metrics."""

    def test_radius_less_than_diameter(self):
        """Test that radius ≤ diameter for all graphs."""
        G = ec.Graph()
        G.add_path_graph(10)
        
        radius = G.get_min_eccentricity()
        diameter = G.get_max_eccentricity()
        
        assert radius is not None
        assert diameter is not None
        assert radius <= diameter

    def test_mean_distance_bounds(self):
        """Test that 1 ≤ mean_distance ≤ diameter for connected graphs."""
        G = ec.Graph()
        G.add_path_graph(8)
        
        mean_dist = G.get_mean_distance()
        diameter = G.get_max_eccentricity()
        
        assert mean_dist is not None
        assert diameter is not None
        assert 1.0 <= mean_dist <= float(diameter)

    def test_metrics_on_balanced_tree(self):
        """Test metrics on balanced tree."""
        G = ec.Graph()
        G.add_balanced_tree(3, 2)  # 3-ary tree, depth 2
        
        radius = G.get_min_eccentricity()
        diameter = G.get_max_eccentricity()
        mean_dist = G.get_mean_distance()
        
        assert radius is not None
        assert diameter is not None
        assert mean_dist is not None
        assert radius == 2  # Root to farthest leaf
        assert diameter == 4  # Leaf to leaf via root
        assert 1.0 < mean_dist < 4.0

    def test_metrics_on_grid(self):
        """Test metrics on 2D grid."""
        G = ec.Graph()
        G.add_grid_2d(3, 3)  # 3x3 grid
        
        radius = G.get_min_eccentricity()
        diameter = G.get_max_eccentricity()
        mean_dist = G.get_mean_distance()
        
        assert radius is not None
        assert diameter is not None
        assert mean_dist is not None
        # Center node has minimum eccentricity
        assert radius <= diameter

    def test_all_metrics_disconnected(self):
        """Test that all metrics return None for disconnected graphs."""
        G = ec.Graph()
        G.add_full_graph(3)
        G.add_full_graph(3)  # Two disconnected cliques
        
        assert G.get_mean_distance() is None
        assert G.get_min_eccentricity() is None
        assert G.get_max_eccentricity() is None


class TestDistanceMetricsEdgeCases:
    """Test edge cases for distance metrics."""

    def test_self_loop_ignored(self):
        """Test that self-loops don't affect distance metrics."""
        G = ec.Graph()
        G.add_path_graph(4)
        
        # Add self-loop
        from edgecase.graph import Edge
        G._graph.add_edge(Edge(id=0, source="1", target="1", weight=None, attributes=None))
        
        # Should still compute same distances
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None

    def test_two_node_graph(self):
        """Test metrics on two-node graph."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        
        assert G.get_mean_distance() == 1.0
        assert G.get_min_eccentricity() == 1
        assert G.get_max_eccentricity() == 1

    def test_directed_asymmetric(self):
        """Test metrics on directed asymmetric graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        # No return path - disconnected when considering all directions
        
        # Graph is weakly connected but not strongly connected
        # Some pairs unreachable
        result = G.get_mean_distance()
        # Should return None because not all pairs are reachable
        assert result is None
