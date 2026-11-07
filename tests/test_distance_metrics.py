import random

import edgecase as ec
import networkx as nx
import pytest


class TestMeanDistance:
    """Test mean distance (average path length) calculation."""

    def test_mean_distance_path_graph(self):
        """Test mean distance on a path graph."""
        G = ec.Graph()
        G.add_path_graph(4)  # 0-1-2-3

        # Distances: 0->1=1, 0->2=2, 0->3=3, 1->2=1, 1->3=2, 2->3=1
        # Total = 1+2+3+1+2+1 = 10, Count = 6, Mean = 10/6 ≈ 1.667
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        assert abs(mean_dist - 10.0 / 6.0) < 0.001

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
        """Test disconnected graph behavior with unconnected parameter."""
        G = ec.Graph()
        G.add_path_graph(3)  # Component 1
        G.add_path_graph(3)  # Component 2 (disconnected)

        # With unconnected=False (strict mode), should return None
        mean_dist = G.get_mean_distance(unconnected=False)
        assert mean_dist is None

        # With unconnected=True (default), computes mean within reachable pairs
        mean_dist = G.get_mean_distance()
        assert mean_dist is not None
        # Each path: (0,1)=1, (0,2)=2, (1,2)=1 -> sum=4, 3 pairs
        # Two paths: sum=8, 6 pairs -> mean = 4/3 ≈ 1.333
        assert abs(mean_dist - 4 / 3) < 0.01

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
        G.add_cycle_graph(4)  # 0->1->2->3->0

        # Each node can reach all others with varying distances
        # 0->1=1, 0->2=2, 0->3=3, 1->2=1, 1->3=2, 1->0=3, etc.
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
        """Test disconnected graph behavior with unconnected parameter."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_path_graph(3)

        # With unconnected=False (strict mode), should return None
        radius = G.get_min_eccentricity(unconnected=False)
        assert radius is None

        # With unconnected=True (default), returns min radius across components
        radius = G.get_min_eccentricity()
        assert radius == 1  # Center nodes of each path have eccentricity 1

    def test_min_eccentricity_alias(self):
        """Test that min_eccentricity is alias for graph_radius."""
        G = ec.Graph()
        G.add_star_graph(7)

        assert G.get_min_eccentricity() == G.get_radius()


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
        """Test disconnected graph behavior with unconnected parameter."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_path_graph(3)

        # With unconnected=False (strict mode), should return None
        diameter = G.get_max_eccentricity(unconnected=False)
        assert diameter is None

        # With unconnected=True (default), returns max diameter across components
        diameter = G.get_max_eccentricity()
        assert diameter == 2  # Paths of 3 nodes have diameter 2

    def test_max_eccentricity_alias(self):
        """Test that max_eccentricity is alias for graph_diameter."""
        G = ec.Graph()
        G.add_path_graph(10)

        assert G.get_max_eccentricity() == G.get_diameter()


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
        """Test metrics behavior for disconnected graphs."""
        G = ec.Graph()
        G.add_full_graph(3)
        G.add_full_graph(3)  # Two disconnected cliques

        # With unconnected=False (strict mode), all return None
        assert G.get_mean_distance(unconnected=False) is None
        assert G.get_min_eccentricity(unconnected=False) is None
        assert G.get_max_eccentricity(unconnected=False) is None

        # With unconnected=True (default), compute metrics within components
        assert G.get_mean_distance() == 1.0  # All pairs in cliques at distance 1
        assert G.get_min_eccentricity() == 1  # All nodes in cliques have ecc 1
        assert G.get_max_eccentricity() == 1  # All nodes in cliques have ecc 1


class TestDistanceMetricsEdgeCases:
    """Test edge cases for distance metrics."""

    def test_self_loop_ignored(self):
        """Test that self-loops don't affect distance metrics."""
        G = ec.Graph()
        G.add_path_graph(4)

        # Add self-loop (using node ID 1, not string "1")
        from edgecase.graph import Edge

        G._graph.add_edge(Edge(id=0, source=1, target=1, weight=None, attributes=None))

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
        # With unconnected=True (default), computes mean for reachable pairs
        result = G.get_mean_distance()
        # A->B=1, A->C=2, B->C=1, so mean = (1+2+1)/3 = 1.33...
        assert result is not None
        assert abs(result - 4.0 / 3.0) < 0.001

        # With unconnected=False, returns None for unreachable pairs
        result_strict = G.get_mean_distance(unconnected=False)
        assert result_strict is None


class TestUnconnectedParameter:
    """Test the unconnected parameter functionality (matching igraph's unconn parameter)."""

    def test_unconnected_parameter_diameter(self):
        """Test diameter with unconnected parameter on various graph types."""
        # Simple disconnected graph: path + isolated node
        G = ec.Graph()
        G.add_path_graph(4)  # 0-1-2-3, diameter 3
        G.add_node("isolated")

        # unconnected=True (default): max diameter across components
        assert G.get_diameter() == 3
        assert G.get_diameter(unconnected=True) == 3

        # unconnected=False: None for disconnected
        assert G.get_diameter(unconnected=False) is None

    def test_unconnected_parameter_radius(self):
        """Test radius with unconnected parameter on various graph types."""
        # Disconnected: star + triangle
        G = ec.Graph()
        G.add_star_graph(5)  # Hub has radius 1
        G.add_cycle_graph(3)  # All nodes have radius 1

        # unconnected=True (default): min radius across components
        assert G.get_radius() == 1
        assert G.get_radius(unconnected=True) == 1

        # unconnected=False: None for disconnected
        assert G.get_radius(unconnected=False) is None

    def test_unconnected_parameter_mean_distance(self):
        """Test mean distance with unconnected parameter."""
        # Two complete graphs
        G = ec.Graph()
        G.add_full_graph(4)  # All pairs at distance 1
        G.add_full_graph(3)  # All pairs at distance 1

        # unconnected=True (default): mean of all reachable pairs
        mean = G.get_mean_distance()
        assert mean == 1.0  # All reachable pairs at distance 1

        # unconnected=False: None for disconnected
        assert G.get_mean_distance(unconnected=False) is None

    def test_unconnected_with_isolated_nodes(self):
        """Test metrics with multiple isolated nodes."""
        G = ec.Graph()
        G.add_path_graph(3)  # diameter 2, radius 1
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Isolated nodes reach nothing, so they don't count toward the radius
        assert G.get_diameter() == 2  # Max ecc from path
        assert G.get_radius() == 1  # Path's middle node

        # Mean distance computed only for reachable pairs
        mean = G.get_mean_distance()
        # Path has pairs: (0,1)=1, (0,2)=2, (1,2)=1 -> sum=4, 3 pairs -> mean = 4/3
        assert abs(mean - 4 / 3) < 0.01

    def test_unconnected_connected_graph(self):
        """Test that connected graphs work the same with both parameter values."""
        G = ec.Graph()
        G.add_cycle_graph(6)

        # Connected graph: unconnected parameter shouldn't matter
        assert G.get_diameter() == G.get_diameter(unconnected=False) == 3
        assert G.get_radius() == G.get_radius(unconnected=False) == 3
        assert G.get_mean_distance() == G.get_mean_distance(unconnected=False)

    def test_unconnected_matches_igraph_behavior(self):
        """Test that default behavior matches igraph's unconn=True."""
        # Create graph with multiple components of different sizes
        G = ec.Graph()
        G.add_path_graph(5)  # diameter 4, radius 2 (middle node)
        G.add_cycle_graph(4)  # diameter 2, radius 2 (all nodes)
        G.add_star_graph(6)  # diameter 2, radius 1 (center)

        # Default should return max/min across all components
        diameter = G.get_diameter()
        radius = G.get_radius()

        # Diameter = max(4, 2, 2) = 4, Radius = min(2, 2, 1) = 1
        assert diameter == 4  # From the path
        assert radius == 1  # From the star center

        # With unconnected=False, should be None
        assert G.get_diameter(unconnected=False) is None
        assert G.get_radius(unconnected=False) is None


class TestWienerIndex:
    """Test Wiener index calculation."""

    def test_wiener_index_path_graph(self):
        """Test Wiener index on a path graph P_n.

        For a path graph with n nodes: W(P_n) = n(n^2-1)/6
        """
        # P_4: 0-1-2-3
        G = ec.Graph()
        G.add_path_graph(4)

        # W(P_4) = 4*(16-1)/6 = 4*15/6 = 10
        # Distances: 0->1=1, 0->2=2, 0->3=3, 1->2=1, 1->3=2, 2->3=1
        # Sum = 1+2+3+1+2+1 = 10
        wiener = G.get_wiener_index()
        assert wiener == 10.0

    def test_wiener_index_complete_graph(self):
        """Test Wiener index on a complete graph K_n.

        For a complete graph with n nodes: W(K_n) = n(n-1)/2
        since all pairs have distance 1.
        """
        # K_5
        G = ec.Graph()
        G.add_full_graph(5)

        # W(K_5) = 5*4/2 = 10 (all 10 pairs at distance 1)
        wiener = G.get_wiener_index()
        assert wiener == 10.0

    def test_wiener_index_cycle_graph(self):
        """Test Wiener index on a cycle graph C_n.

        For a cycle graph with n nodes:
        - If n even: W(C_n) = n^3/8
        - If n odd: W(C_n) = n(n^2-1)/8
        """
        # C_6 (even)
        G = ec.Graph()
        G.add_cycle_graph(6)

        # W(C_6) = 6^3/8 = 216/8 = 27
        wiener = G.get_wiener_index()
        assert wiener == 27.0

    def test_wiener_index_star_graph(self):
        """Test Wiener index on a star graph S_n (hub + n-1 leaves)."""
        # S_5 (1 hub + 4 leaves = 5 nodes)
        G = ec.Graph()
        G.add_star_graph(5)

        # Hub to leaves: 4 paths of distance 1 = 4
        # Leaf to leaf: 6 pairs of distance 2 = 12
        # Total = 4 + 12 = 16
        wiener = G.get_wiener_index()
        assert wiener == 16.0

    def test_wiener_index_single_node(self):
        """Test Wiener index on single node graph."""
        G = ec.Graph()
        G.add_node("A")

        # No pairs, sum is 0
        wiener = G.get_wiener_index()
        assert wiener == 0.0

    def test_wiener_index_two_nodes_connected(self):
        """Test Wiener index on two connected nodes."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_edge("A", "B")

        # Single pair at distance 1
        wiener = G.get_wiener_index()
        assert wiener == 1.0

    def test_wiener_index_disconnected_graph(self):
        """Test Wiener index on disconnected graph returns infinity."""
        G = ec.Graph()
        G.add_path_graph(3)  # Component 1
        G.add_path_graph(3)  # Component 2

        # Disconnected graphs have infinite Wiener index
        wiener = G.get_wiener_index()
        assert wiener == float("inf")

    def test_wiener_index_directed_graph(self):
        """Test Wiener index on directed graph."""
        G = ec.Graph(directed=True)
        G.add_path_graph(3)  # 0->1->2

        # 0->1=1, 0->2=2, 1->2=1 (only forward paths exist)
        # 1->0=inf, 2->0=inf, 2->1=inf
        wiener = G.get_wiener_index()
        assert wiener == float("inf")

    def test_wiener_index_directed_cycle(self):
        """Test Wiener index on directed cycle."""
        G = ec.Graph(directed=True)
        G.add_cycle_graph(4)  # 0->1->2->3->0

        # All pairs reachable: this forms a strongly connected component
        # 0->1=1, 0->2=2, 0->3=3, 1->2=1, 1->3=2, 1->0=3
        # 2->3=1, 2->0=2, 2->1=3, 3->0=1, 3->1=2, 3->2=3
        # Total = (1+2+3)*4 = 24
        wiener = G.get_wiener_index()
        assert wiener == 24.0

    def test_wiener_index_weighted_graph(self):
        """Test Wiener index respects edge weights when weight parameter is specified."""
        G = ec.Graph(weighted=True)
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B", weight=2.0)
        G.add_edge("B", "C", weight=3.0)

        # A->B=2, A->C=5, B->C=3
        # Total = 2 + 5 + 3 = 10
        wiener = G.get_wiener_index(weight="weight")
        assert wiener == 10.0

        # Without weight parameter, uses unweighted (hop count)
        # A->B=1, A->C=2, B->C=1
        # Total = 1 + 2 + 1 = 4
        wiener_unweighted = G.get_wiener_index()
        assert wiener_unweighted == 4.0


class TestEstradaIndex:
    """Test Estrada index calculation."""

    def test_estrada_index_single_node(self):
        """Test Estrada index on single node graph.

        Single node has adjacency matrix [0], eigenvalue = 0
        EE = exp(0) = 1
        """
        G = ec.Graph()
        G.add_node("A")

        estrada = G.get_estrada_index()
        assert abs(estrada - 1.0) < 0.001

    def test_estrada_index_two_connected_nodes(self):
        """Test Estrada index on K_2.

        Adjacency matrix: [[0,1],[1,0]]
        Eigenvalues: -1, 1
        EE = exp(-1) + exp(1) ≈ 0.368 + 2.718 ≈ 3.086
        """
        G = ec.Graph()
        G.add_full_graph(2)

        import math

        expected = math.exp(-1) + math.exp(1)
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.001

    def test_estrada_index_path_graph(self):
        """Test Estrada index on path graph P_3.

        Adjacency matrix: [[0,1,0],[1,0,1],[0,1,0]]
        Eigenvalues: -sqrt(2), 0, sqrt(2)
        EE = exp(-sqrt(2)) + exp(0) + exp(sqrt(2))
        """
        G = ec.Graph()
        G.add_path_graph(3)

        import math

        sqrt2 = math.sqrt(2)
        expected = math.exp(-sqrt2) + math.exp(0) + math.exp(sqrt2)
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.01

    def test_estrada_index_complete_graph(self):
        """Test Estrada index on complete graph K_4.

        K_n has eigenvalues: (n-1) with multiplicity 1, and -1 with multiplicity (n-1)
        For K_4: eigenvalues are 3, -1, -1, -1
        EE = exp(3) + 3*exp(-1)
        """
        G = ec.Graph()
        G.add_full_graph(4)

        import math

        expected = math.exp(3) + 3 * math.exp(-1)
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.01

    def test_estrada_index_cycle_graph(self):
        """Test Estrada index on cycle graph C_4.

        C_4 eigenvalues: 2, 0, 0, -2
        EE = exp(2) + 2*exp(0) + exp(-2)
        """
        G = ec.Graph()
        G.add_cycle_graph(4)

        import math

        expected = math.exp(2) + 2 * math.exp(0) + math.exp(-2)
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.01

    def test_estrada_index_star_graph(self):
        """Test Estrada index on star graph S_4 (hub + 3 leaves).

        S_n (hub + n-1 leaves) has eigenvalues: sqrt(n-1), 0 (n-2 times), -sqrt(n-1)
        For S_4: eigenvalues are sqrt(3), 0, 0, -sqrt(3)
        EE = exp(sqrt(3)) + 2*exp(0) + exp(-sqrt(3))
        """
        G = ec.Graph()
        G.add_star_graph(4)

        import math

        sqrt3 = math.sqrt(3)
        expected = math.exp(sqrt3) + 2 * math.exp(0) + math.exp(-sqrt3)
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.01

    def test_estrada_index_empty_graph(self):
        """Test Estrada index on empty graph."""
        G = ec.Graph()

        estrada = G.get_estrada_index()
        assert estrada == 0.0

    def test_estrada_index_disconnected_graph(self):
        """Test Estrada index on disconnected graph.

        Estrada index is well-defined for disconnected graphs.
        It's just the sum of eigenvalue exponentials of the adjacency matrix.
        """
        G = ec.Graph()
        G.add_full_graph(2)  # Component 1 (2 nodes)
        G.add_full_graph(2)  # Component 2 (2 nodes)

        import math

        # Each K_2 contributes exp(-1) + exp(1), two components
        expected = 2 * (math.exp(-1) + math.exp(1))
        estrada = G.get_estrada_index()
        assert abs(estrada - expected) < 0.01

    def test_estrada_index_increases_with_edges(self):
        """Test that Estrada index increases when edges are added.

        Adding edges generally increases the Estrada index as it
        increases the number of closed walks.
        """
        G1 = ec.Graph()
        G1.add_path_graph(4)  # 3 edges

        G2 = ec.Graph()
        G2.add_cycle_graph(4)  # 4 edges (closed cycle)

        assert G2.get_estrada_index() > G1.get_estrada_index()

    def test_estrada_index_positive(self):
        """Test that Estrada index is always positive for non-empty graphs."""
        for n in [3, 5, 10]:
            G = ec.Graph()
            G.add_full_graph(n)
            assert G.get_estrada_index() > 0

            G2 = ec.Graph()
            G2.add_cycle_graph(n)
            assert G2.get_estrada_index() > 0

            G3 = ec.Graph()
            G3.add_path_graph(n)
            assert G3.get_estrada_index() > 0


class TestResistanceDistance:
    """Test resistance distance calculation."""

    def test_resistance_distance_same_node(self):
        """Test resistance distance from a node to itself is zero."""
        G = ec.Graph()
        G.add_path_graph(3)  # Creates nodes 1-2-3
        assert G.get_resistance_distance(1, 1) == 0.0

    def test_resistance_distance_adjacent_nodes(self):
        """Test resistance distance between adjacent nodes equals 1."""
        G = ec.Graph()
        G.add_full_graph(2)  # Two nodes with one edge (IDs 1, 2)
        result = G.get_resistance_distance(1, 2)
        assert abs(result - 1.0) < 1e-10

    def test_resistance_distance_path_graph(self):
        """Test resistance distance in a path graph (resistors in series)."""
        G = ec.Graph()
        G.add_path_graph(4)  # 1-2-3-4

        # In a path, resistance = number of edges (series)
        assert abs(G.get_resistance_distance(1, 2) - 1.0) < 1e-10
        assert abs(G.get_resistance_distance(1, 3) - 2.0) < 1e-10
        assert abs(G.get_resistance_distance(1, 4) - 3.0) < 1e-10
        assert abs(G.get_resistance_distance(2, 3) - 1.0) < 1e-10
        assert abs(G.get_resistance_distance(2, 4) - 2.0) < 1e-10

    def test_resistance_distance_complete_graph(self):
        """Test resistance distance in complete graph K_n.

        For complete graph Kn: Ω(i,j) = 2/n for all pairs i≠j
        """
        # K3
        G3 = ec.Graph()
        G3.add_full_graph(3)  # Nodes 1, 2, 3
        assert abs(G3.get_resistance_distance(1, 2) - 2.0 / 3.0) < 1e-10

        # K4
        G4 = ec.Graph()
        G4.add_full_graph(4)  # Nodes 1, 2, 3, 4
        assert abs(G4.get_resistance_distance(1, 2) - 0.5) < 1e-10

        # K5
        G5 = ec.Graph()
        G5.add_full_graph(5)  # Nodes 1, 2, 3, 4, 5
        assert abs(G5.get_resistance_distance(1, 2) - 0.4) < 1e-10

    def test_resistance_distance_cycle_graph(self):
        """Test resistance distance in cycle graph (parallel paths).

        For cycle C4: paths of length 1 and 3 in parallel
        R = 1*3/(1+3) = 0.75
        """
        G = ec.Graph()
        G.add_cycle_graph(4)  # 1-2-3-4-1
        # Adjacent nodes in C4: parallel paths of 1 and 3
        assert abs(G.get_resistance_distance(1, 2) - 0.75) < 1e-10
        # Opposite nodes in C4: two parallel paths of 2
        assert abs(G.get_resistance_distance(1, 3) - 1.0) < 1e-10

    def test_resistance_distance_star_graph(self):
        """Test resistance distance in star graph."""
        G = ec.Graph()
        G.add_star_graph(5)  # Hub (1) + 4 leaves (2, 3, 4, 5)

        # Hub to leaf: distance 1
        assert abs(G.get_resistance_distance(1, 2) - 1.0) < 1e-10

        # Leaf to leaf: through hub (two resistors in series)
        assert abs(G.get_resistance_distance(2, 3) - 2.0) < 1e-10

    def test_resistance_distance_symmetric(self):
        """Test that resistance distance is symmetric."""
        G = ec.Graph()
        G.add_path_graph(4)  # 1-2-3-4
        assert abs(G.get_resistance_distance(1, 3) - G.get_resistance_distance(3, 1)) < 1e-10

    def test_resistance_distance_triangle_inequality(self):
        """Test that resistance distance satisfies triangle inequality."""
        G = ec.Graph()
        G.add_path_graph(4)  # 1-2-3-4

        r12 = G.get_resistance_distance(1, 2)
        r23 = G.get_resistance_distance(2, 3)
        r13 = G.get_resistance_distance(1, 3)

        # r13 <= r12 + r23 (actually equals for path)
        assert r13 <= r12 + r23 + 1e-10

    def test_resistance_distance_with_labels(self):
        """Test resistance distance using node labels."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        # Path A-B-C
        assert abs(G.get_resistance_distance("A", "B") - 1.0) < 1e-10
        assert abs(G.get_resistance_distance("A", "C") - 2.0) < 1e-10

    def test_resistance_distance_directed_error(self):
        """Test that directed graphs raise an error."""
        G = ec.Graph(directed=True)
        G.add_path_graph(3)
        with pytest.raises(ValueError, match="undirected"):
            G.get_resistance_distance(1, 2)

    def test_resistance_distance_node_not_found(self):
        """Test error when node doesn't exist."""
        G = ec.Graph()
        G.add_path_graph(3)  # Nodes 1, 2, 3
        with pytest.raises(ValueError):
            G.get_resistance_distance(1, 99)


class TestResistanceDistanceMatrix:
    """Test resistance distance matrix calculation."""

    def test_resistance_distance_matrix_path(self):
        """Test resistance distance matrix for path graph."""
        G = ec.Graph()
        G.add_path_graph(3)  # 1-2-3
        matrix = G.resistance_distance_matrix()

        # Check diagonal is zero
        assert matrix[(1, 1)] == 0.0
        assert matrix[(2, 2)] == 0.0
        assert matrix[(3, 3)] == 0.0

        # Check off-diagonal values
        assert abs(matrix[(1, 2)] - 1.0) < 1e-10
        assert abs(matrix[(1, 3)] - 2.0) < 1e-10
        assert abs(matrix[(2, 3)] - 1.0) < 1e-10

        # Check symmetry
        assert matrix[(1, 2)] == matrix[(2, 1)]
        assert matrix[(1, 3)] == matrix[(3, 1)]

    def test_resistance_distance_matrix_complete(self):
        """Test resistance distance matrix for complete graph."""
        G = ec.Graph()
        G.add_full_graph(3)  # K3 with nodes 1, 2, 3
        matrix = G.resistance_distance_matrix()

        # All off-diagonal values should be 2/3
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    assert matrix[(i, j)] == 0.0
                else:
                    assert abs(matrix[(i, j)] - 2.0 / 3.0) < 1e-10

    def test_resistance_distance_matrix_single_node(self):
        """Test resistance distance matrix for single node."""
        G = ec.Graph()
        G.add_node(label="A")
        matrix = G.resistance_distance_matrix()
        a_id = G.get_id_from_label("A")
        assert matrix[(a_id, a_id)] == 0.0

    def test_resistance_distance_matrix_empty(self):
        """Test resistance distance matrix for empty graph."""
        G = ec.Graph()
        matrix = G.resistance_distance_matrix()
        assert len(matrix) == 0

    def test_resistance_distance_matrix_key_order(self):
        # The dict came from a Rust HashMap, so the key order changed from call to call
        G = ec.Graph().add_gnp_graph(10, 0.3, seed=2)
        ids = G.get_node_ids()
        expected = [(u, v) for u in ids for v in ids]
        for _ in range(5):
            assert list(G.resistance_distance_matrix()) == expected


class TestResistanceDistanceAccuracy:
    """Exact values on spectra with repeated eigenvalues, and infinite cross-component pairs."""

    @staticmethod
    def _cycle(n):
        g = ec.Graph()
        for i in range(n):
            g.add_node(label=str(i))
        for i in range(n):
            g.add_edge(str(i), str((i + 1) % n))
        return g

    @pytest.mark.parametrize("n", range(3, 16))
    def test_cycle_closed_form(self, n):
        """On C_n, nodes k steps apart have resistance k(n - k)/n (odd n used to be off ~1e-4)."""
        g = self._cycle(n)
        matrix = g.resistance_distance_matrix()
        zero = g.get_id_from_label("0")
        for k in range(1, n):
            expected = k * (n - k) / n
            node = g.get_id_from_label(str(k))
            assert g.get_resistance_distance("0", str(k)) == pytest.approx(expected, abs=1e-12)
            assert matrix[(zero, node)] == pytest.approx(expected, abs=1e-12)

    def test_different_components_are_infinite(self):
        g = ec.Graph()
        for label in "ABCDE":
            g.add_node(label=label)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("D", "E")
        ids = {label: g.get_id_from_label(label) for label in "ABCDE"}

        assert g.get_resistance_distance("A", "D") == float("inf")
        matrix = g.resistance_distance_matrix()
        assert matrix[(ids["A"], ids["E"])] == float("inf")
        assert matrix[(ids["E"], ids["C"])] == float("inf")

        # Pairs inside each component are unaffected by the other component
        assert g.get_resistance_distance("A", "C") == pytest.approx(2.0)
        assert matrix[(ids["D"], ids["E"])] == pytest.approx(1.0)

    def test_isolated_node_is_infinitely_far(self):
        g = ec.Graph()
        for label in "ABC":
            g.add_node(label=label)
        g.add_edge("A", "B")

        assert g.get_resistance_distance("A", "C") == float("inf")
        assert g.get_resistance_distance("A", "B") == pytest.approx(1.0)

    def test_matches_networkx_per_component(self):
        nx = pytest.importorskip("networkx")
        import math
        import random

        rng = random.Random(4)
        for seed in range(40):
            reference = nx.gnp_random_graph(rng.randint(2, 12), rng.random() * 0.5, seed=seed)
            g = ec.Graph()
            for node in reference.nodes():
                g.add_node(label=str(node))
            for u, v in reference.edges():
                g.add_edge(str(u), str(v))
            matrix = g.resistance_distance_matrix()

            for u in reference:
                for v in reference:
                    if u == v:
                        continue
                    got = matrix[(g.get_id_from_label(str(u)), g.get_id_from_label(str(v)))]
                    if nx.has_path(reference, u, v):
                        component = reference.subgraph(nx.node_connected_component(reference, u))
                        expected = nx.resistance_distance(component, u, v)
                        assert got == pytest.approx(expected, abs=1e-10)
                    else:
                        assert math.isinf(got)


class TestKirchhoffIndex:
    """Test Kirchhoff index calculation."""

    def test_kirchhoff_index_single_node(self):
        """Test Kirchhoff index for single node is zero."""
        G = ec.Graph()
        G.add_node(label="A")
        assert G.get_kirchhoff_index() == 0.0

    def test_kirchhoff_index_two_nodes(self):
        """Test Kirchhoff index for two connected nodes."""
        G = ec.Graph()
        G.add_full_graph(2)
        # Single pair with resistance 1
        assert abs(G.get_kirchhoff_index() - 1.0) < 1e-10

    def test_kirchhoff_index_path_graph(self):
        """Test Kirchhoff index for path graph.

        For path Pn: Kf = n(n²-1)/6
        """
        # P3: Kf = 3*(9-1)/6 = 24/6 = 4
        G3 = ec.Graph()
        G3.add_path_graph(3)
        assert abs(G3.get_kirchhoff_index() - 4.0) < 1e-10

        # P4: Kf = 4*(16-1)/6 = 60/6 = 10
        G4 = ec.Graph()
        G4.add_path_graph(4)
        assert abs(G4.get_kirchhoff_index() - 10.0) < 1e-10

        # P5: Kf = 5*(25-1)/6 = 120/6 = 20
        G5 = ec.Graph()
        G5.add_path_graph(5)
        assert abs(G5.get_kirchhoff_index() - 20.0) < 1e-10

    def test_kirchhoff_index_complete_graph(self):
        """Test Kirchhoff index for complete graph.

        For complete Kn: Kf = n-1
        """
        # K3: Kf = 2
        G3 = ec.Graph()
        G3.add_full_graph(3)
        assert abs(G3.get_kirchhoff_index() - 2.0) < 1e-10

        # K4: Kf = 3
        G4 = ec.Graph()
        G4.add_full_graph(4)
        assert abs(G4.get_kirchhoff_index() - 3.0) < 1e-10

        # K5: Kf = 4
        G5 = ec.Graph()
        G5.add_full_graph(5)
        assert abs(G5.get_kirchhoff_index() - 4.0) < 1e-10

    def test_kirchhoff_index_cycle_graph(self):
        """Test Kirchhoff index for cycle graph.

        For cycle Cn: Kf = n(n²-1)/12
        """
        # C3: Kf = 3*(9-1)/12 = 24/12 = 2
        G3 = ec.Graph()
        G3.add_cycle_graph(3)
        assert abs(G3.get_kirchhoff_index() - 2.0) < 1e-10

        # C4: Kf = 4*(16-1)/12 = 60/12 = 5
        G4 = ec.Graph()
        G4.add_cycle_graph(4)
        assert abs(G4.get_kirchhoff_index() - 5.0) < 1e-10

        # C6: Kf = 6*(36-1)/12 = 210/12 = 17.5
        G6 = ec.Graph()
        G6.add_cycle_graph(6)
        assert abs(G6.get_kirchhoff_index() - 17.5) < 1e-10

    def test_kirchhoff_index_disconnected(self):
        """Test Kirchhoff index for disconnected graph returns infinity."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_path_graph(3)  # Disconnected second component
        assert G.get_kirchhoff_index() == float("inf")

    def test_kirchhoff_index_directed_error(self):
        """Test that directed graphs raise an error."""
        G = ec.Graph(directed=True)
        G.add_path_graph(3)
        with pytest.raises(ValueError, match="undirected"):
            G.get_kirchhoff_index()

    def test_kirchhoff_index_empty_graph(self):
        """Test Kirchhoff index for empty graph."""
        G = ec.Graph()
        assert G.get_kirchhoff_index() == 0.0

    def test_kirchhoff_vs_sum_resistance(self):
        """Test that Kirchhoff index equals sum of pairwise resistances."""
        G = ec.Graph()
        G.add_path_graph(4)  # Nodes 1, 2, 3, 4

        # Manual sum of resistance distances for i < j
        total = 0.0
        for i in range(1, 5):
            for j in range(i + 1, 5):
                total += G.get_resistance_distance(i, j)

        assert abs(G.get_kirchhoff_index() - total) < 1e-10


# Center, periphery and barycenter, validated against NetworkX.


def _random_undirected():
    for seed in range(25):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(18, rnd.choice([0.1, 0.15, 0.25]), seed=seed)
        G.add_nodes_from(range(18))
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
        yield G


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _labels(g: ec.Graph, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


def _lab(g: ec.Graph, nid: int) -> int:
    return int(g.get_label_from_id(nid))


class TestCenterAndBarycenter:
    def test_center_matches_networkx_on_connected_graphs(self):
        for G in _random_undirected():
            big = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            if big.number_of_nodes() < 2:
                continue
            g = _build(big)
            assert sorted(_lab(g, n) for n in g.get_center()) == sorted(nx.center(big))

    def test_center_and_periphery_are_counterparts(self):
        g = ec.Graph().add_path_graph(5)
        assert g.get_center() == [3]
        assert sorted(g.get_periphery()) == [1, 5]
        assert ec.Graph().get_center() == []

    @pytest.mark.parametrize("weighted", [False, True])
    def test_barycenter_matches_networkx(self, weighted):
        for G in _random_undirected():
            big = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            if big.number_of_nodes() < 2:
                continue
            g = _build(big, weighted)
            ref = nx.barycenter(big, weight="weight" if weighted else None)
            assert sorted(_lab(g, n) for n in g.get_barycenter()) == sorted(ref)

    def test_center_and_barycenter_differ(self):
        g = ec.Graph().add_nodes(["hub", "A", "B", "C", "X", "D", "E", "F", "G"])
        g.add_edges(
            [("hub", x) for x in ["A", "B", "C", "X", "D"]] + [("D", "E"), ("E", "F"), ("F", "G")]
        )
        assert _labels(g, g.get_center()) == ["D", "E"]
        assert _labels(g, g.get_barycenter()) == ["hub"]

    def test_barycenter_requires_connectivity(self):
        with pytest.raises(ValueError, match="connected"):
            ec.Graph().add_nodes(["A", "B"]).get_barycenter()
        one_way = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="strongly connected"):
            one_way.get_barycenter()
        cycle = ec.Graph(directed=True).add_nodes(list("ABC"))
        cycle.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        assert cycle.get_barycenter() == [1, 2, 3]

    def test_steps(self):
        g = ec.Graph().add_path_graph(5)
        g.get_center()
        assert g.steps[-1].narration == "Center: 3 (eccentricity 2)."
        g.get_barycenter()
        assert g.steps[-1].narration == "Barycenter: 3 (total distance 6)."


class TestRadiusIgnoresNodesThatReachNothing:
    """Regression: an isolated node, or a sink in a directed graph, made the radius 0."""

    def test_isolated_node_skipped(self):
        g = ec.Graph().add_path_graph(5).add_node(label="loner")
        assert g.get_radius() == 2
        assert g.get_min_eccentricity() == 2
        assert g.get_center() == [3]
        assert g.get_radius(unconnected=False) is None

    def test_smallest_radius_across_components(self):
        # Path of 5 (radius 2) plus a separate edge (radius 1)
        g = ec.Graph().add_path_graph(5).add_nodes(["X", "Y"]).add_edges([("X", "Y")])
        assert g.get_radius() == 1
        assert sorted(_labels(g, g.get_center())) == ["X", "Y"]

    def test_graph_without_edges(self):
        g = ec.Graph().add_nodes(["A", "B"])
        assert g.get_radius() == 0
        assert len(g.get_center()) == 2
        assert ec.Graph().get_radius() is None
        assert ec.Graph().add_nodes(["A"]).get_radius() == 0

    def test_directed_path_and_sink(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B"), ("B", "C")])
        # Only A reaches the whole component; C (a sink) no longer gives 0
        assert g.get_radius() == 2
        assert _labels(g, g.get_center()) == ["A"]
        # Only weakly connected, so unconnected=False gives None (strong connectivity needed)
        assert g.get_radius(unconnected=False) is None

    def test_directed_out_star_and_in_star(self):
        out_star = ec.Graph(directed=True).add_star_graph(4)
        assert out_star.get_radius() == 1
        assert out_star.get_center() == [1]
        # No node of an in-star reaches the others
        in_star = ec.Graph(directed=True).add_nodes(["H", "A", "B"])
        in_star.add_edges([("A", "H"), ("B", "H")])
        assert in_star.get_radius() is None
        assert in_star.get_center() == []

    def test_strongly_connected_matches_networkx(self):
        for seed in range(15):
            G = nx.gnp_random_graph(10, 0.35, seed=seed, directed=True)
            if not nx.is_strongly_connected(G):
                continue
            g = ec.Graph(directed=True).add_nodes([str(n) for n in G])
            g.add_edges([(str(u), str(v)) for u, v in G.edges])
            assert g.get_radius() == nx.radius(G)
            assert sorted(_lab(g, n) for n in g.get_center()) == sorted(nx.center(G))
