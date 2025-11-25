from edgecase import Graph


class TestAllPairsShortestPathLength:
    """Tests for all_pairs_shortest_path_length method."""

    def test_simple_path_graph(self):
        """Test all pairs distances on a simple path graph."""
        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        distances = graph.all_pairs_shortest_path_length()

        # Check some specific distances
        assert distances[1][1] == 0.0  # Distance to self
        assert distances[1][2] == 1.0
        assert distances[1][3] == 2.0
        assert distances[1][4] == 3.0
        assert distances[2][4] == 2.0
        assert distances[3][4] == 1.0

        # Check symmetry for undirected graph
        assert distances[1][4] == distances[4][1]

    def test_weighted_graph(self):
        """Test all pairs distances on a weighted graph."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="A", target="C", weight=5.0)  # Longer direct route
        )

        distances = graph.all_pairs_shortest_path_length()
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")

        # A->C should go via B (1+2=3) rather than direct (5)
        assert distances[a_id][c_id] == 3.0
        assert distances[a_id][b_id] == 1.0
        assert distances[b_id][c_id] == 2.0

    def test_disconnected_graph(self):
        """Test all pairs distances with disconnected components."""
        graph = Graph()
        graph.add_path_graph(3)  # 1-2-3
        graph.add_node(4)  # Isolated node

        distances = graph.all_pairs_shortest_path_length()

        # Check connected component
        assert 1 in distances
        assert 2 in distances[1]
        assert 3 in distances[1]

        # Isolated node shouldn't be reachable from others
        assert 4 not in distances[1]
        assert 4 not in distances[2]
        assert 4 not in distances[3]

        # But isolated node should have entry for itself
        assert 4 in distances
        assert distances[4][4] == 0.0

    def test_star_graph(self):
        """Test all pairs distances on a star graph."""
        graph = Graph()
        graph.add_star_graph(5)  # Creates 5 nodes total

        distances = graph.all_pairs_shortest_path_length()
        node_ids = graph.get_node_ids()
        center = 1  # Center is node 1
        leaves = [n for n in node_ids if n != center]

        # Center to leaves is distance 1
        for leaf in leaves:
            assert distances[center][leaf] == 1.0

        # Leaf to leaf is distance 2 (through center)
        for i, leaf1 in enumerate(leaves):
            for leaf2 in leaves[i + 1 :]:
                assert distances[leaf1][leaf2] == 2.0

    def test_complete_graph(self):
        """Test all pairs distances on a complete graph."""
        graph = Graph()
        # Create a complete graph manually
        for i in range(1, 6):
            graph.add_node(i)
        for i in range(1, 6):
            for j in range(i + 1, 6):
                graph.add_edge(source=i, target=j)

        distances = graph.all_pairs_shortest_path_length()

        # All pairs should have distance 1 (direct connection)
        for i in range(1, 6):
            for j in range(1, 6):
                if i == j:
                    assert distances[i][j] == 0.0
                else:
                    assert distances[i][j] == 1.0

    def test_directed_graph(self):
        """Test all pairs distances on a directed graph."""
        graph = Graph(directed=True)
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_edge(source=1, target=2)
        graph.add_edge(source=2, target=3)

        distances = graph.all_pairs_shortest_path_length()

        # Forward paths exist
        assert distances[1][2] == 1.0
        assert distances[1][3] == 2.0
        assert distances[2][3] == 1.0

        # Backward paths don't exist
        assert 1 not in distances[2]
        assert 1 not in distances[3]
        assert 2 not in distances[3]


class TestAllPairsShortestPath:
    """Tests for all_pairs_shortest_path method."""

    def test_simple_path_graph(self):
        """Test all pairs paths on a simple path graph."""
        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        paths = graph.all_pairs_shortest_path()

        # Check some specific paths
        assert paths[1][4] == [1, 2, 3, 4]
        assert paths[2][4] == [2, 3, 4]
        assert paths[1][2] == [1, 2]

        # Check that paths include source and target
        for source in range(1, 5):
            for target in range(1, 5):
                path = paths[source][target]
                assert path[0] == source
                assert path[-1] == target

    def test_weighted_graph_chooses_shortest(self):
        """Test that weighted graph chooses shortest path by weight."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)  # A->B->D = 2.0
            .add_edge(source="A", target="C", weight=1.0)
            .add_edge(source="C", target="D", weight=5.0)  # A->C->D = 6.0
        )

        paths = graph.all_pairs_shortest_path()
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        d_id = graph.get_id_from_label("D")

        # Should take A->B->D route (shorter by weight)
        path = paths[a_id][d_id]
        assert b_id in path
        assert len(path) == 3

    def test_disconnected_components(self):
        """Test paths with disconnected components."""
        graph = Graph()
        graph.add_path_graph(3)  # 1-2-3
        graph.add_path_graph(3)  # Creates 4-5-6

        paths = graph.all_pairs_shortest_path()

        # Within first component
        assert paths[1][3] == [1, 2, 3]

        # Within second component
        assert paths[4][6] == [4, 5, 6]

        # Between components - no path
        assert 4 not in paths[1]
        assert 1 not in paths[4]

    def test_path_lengths_match_distances(self):
        """Test that path lengths match distance calculations."""
        graph = Graph()
        graph.add_cycle_graph(6)

        paths = graph.all_pairs_shortest_path()
        distances = graph.all_pairs_shortest_path_length()

        # For unweighted graphs, path length - 1 should equal distance
        for source in paths:
            for target in paths[source]:
                path_length = len(paths[source][target]) - 1
                assert path_length == distances[source][target]

    def test_self_loops(self):
        """Test that paths to self are just [self]."""
        graph = Graph()
        graph.add_star_graph(5)

        paths = graph.all_pairs_shortest_path()
        node_ids = graph.get_node_ids()

        for node in node_ids:
            assert paths[node][node] == [node]


class TestKShortestPaths:
    """Tests for k_shortest_paths method."""

    def test_simple_alternative_routes(self):
        """Test finding k alternative routes."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)  # Route 1: A->B->D (2.0)
            .add_edge(source="A", target="C", weight=1.5)
            .add_edge(source="C", target="D", weight=1.0)  # Route 2: A->C->D (2.5)
            .add_edge(source="A", target="D", weight=4.0)  # Route 3: A->D (4.0)
        )

        paths = graph.k_shortest_paths("A", "D", k=3)

        # Should find all 3 paths
        assert len(paths) == 3

        # Check they're sorted by distance
        assert paths[0][1] == 2.0  # A->B->D
        assert paths[1][1] == 2.5  # A->C->D
        assert paths[2][1] == 4.0  # A->D

        # Check paths are correct
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")
        d_id = graph.get_id_from_label("D")

        assert paths[0][0] == [a_id, b_id, d_id]
        assert paths[1][0] == [a_id, c_id, d_id]
        assert paths[2][0] == [a_id, d_id]

    def test_k_larger_than_available_paths(self):
        """Test when k is larger than number of available paths."""
        graph = Graph()
        graph.add_path_graph(3)  # Only one path from 1 to 3

        paths = graph.k_shortest_paths(1, 3, k=10)

        # Should return only the available path
        assert len(paths) == 1
        assert paths[0][0] == [1, 2, 3]

    def test_k_equals_zero(self):
        """Test k=0 returns empty list."""
        graph = Graph()
        graph.add_path_graph(3)

        paths = graph.k_shortest_paths(1, 3, k=0)
        assert len(paths) == 0

    def test_no_path_exists(self):
        """Test when no path exists between nodes."""
        graph = Graph()
        graph.add_node(1)
        graph.add_node(2)
        # No edges - disconnected

        paths = graph.k_shortest_paths(1, 2, k=5)
        assert len(paths) == 0

    def test_grid_multiple_paths(self):
        """Test k shortest paths on a grid with multiple routes."""
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)
        # Grid:
        # 1 - 2 - 3
        # |   |   |
        # 4 - 5 - 6
        # |   |   |
        # 7 - 8 - 9

        paths = graph.k_shortest_paths(1, 9, k=10)

        # Should find multiple paths
        assert len(paths) >= 2

        # All paths should have same length (4 hops) for shortest paths
        # in an unweighted grid
        assert paths[0][1] == 4.0

        # First path should be 4 edges
        assert len(paths[0][0]) == 5  # 5 nodes = 4 edges

    def test_directed_graph(self):
        """Test k shortest paths on directed graph."""
        graph = Graph(directed=True, weighted=True)
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_edge(source=1, target=2, weight=1.0)
        graph.add_edge(source=2, target=3, weight=1.0)
        graph.add_edge(source=1, target=3, weight=3.0)

        paths = graph.k_shortest_paths(1, 3, k=2)

        assert len(paths) == 2
        assert paths[0][1] == 2.0  # 1->2->3
        assert paths[1][1] == 3.0  # 1->3

    def test_single_path_k_equals_1(self):
        """Test k=1 returns only shortest path."""
        graph = (
            Graph(weighted=True)
            .add_node(1)
            .add_node(2)
            .add_node(3)
            .add_edge(source=1, target=2, weight=1.0)
            .add_edge(source=2, target=3, weight=1.0)
            .add_edge(source=1, target=3, weight=5.0)
        )

        paths = graph.k_shortest_paths(1, 3, k=1)

        assert len(paths) == 1
        assert paths[0][1] == 2.0
        assert paths[0][0] == [1, 2, 3]

    def test_paths_are_simple(self):
        """Test that returned paths are simple (no repeated nodes)."""
        graph = Graph(weighted=True)
        graph.add_cycle_graph(5)

        paths = graph.k_shortest_paths(1, 3, k=5)

        # Check no path has repeated nodes
        for path, _ in paths:
            assert len(path) == len(set(path)), f"Path {path} has repeated nodes"

    def test_unweighted_counts_hops(self):
        """Test that unweighted graphs count hops correctly."""
        graph = Graph()
        graph.add_path_graph(5)  # 1-2-3-4-5

        paths = graph.k_shortest_paths(1, 5, k=1)

        # Distance should be 4.0 (4 hops)
        assert paths[0][1] == 4.0
        assert len(paths[0][0]) == 5  # 5 nodes

    def test_invalid_nodes(self):
        """Test with non-existent nodes."""
        graph = Graph()
        graph.add_path_graph(3)

        # Non-existent node
        paths = graph.k_shortest_paths(1, 999, k=5)
        assert len(paths) == 0

        paths = graph.k_shortest_paths(999, 1, k=5)
        assert len(paths) == 0


class TestAdvancedPathsIntegration:
    """Integration tests using all three methods together."""

    def test_consistency_across_methods(self):
        """Test that all three methods give consistent results."""
        graph = Graph(weighted=True)
        graph.add_cycle_graph(6)

        # Get results from all methods
        distances = graph.all_pairs_shortest_path_length()
        paths = graph.all_pairs_shortest_path()
        k_paths_1_to_4 = graph.k_shortest_paths(1, 4, k=1)

        # Check consistency
        assert distances[1][4] == k_paths_1_to_4[0][1]
        assert paths[1][4] == k_paths_1_to_4[0][0]

    def test_large_graph_performance(self):
        """Test that methods work on larger graphs."""
        graph = Graph()
        graph.add_grid_2d(rows=10, cols=10)  # 100 nodes

        # These should complete without timing out
        distances = graph.all_pairs_shortest_path_length()
        assert len(distances) == 100

        # Test a few paths
        paths = graph.k_shortest_paths(1, 100, k=3)
        assert len(paths) >= 1

    def test_weighted_vs_unweighted(self):
        """Test difference between weighted and unweighted graphs."""
        # Create same structure, one weighted one not
        unweighted = Graph()
        unweighted.add_path_graph(4)

        weighted = Graph(weighted=True)
        weighted.add_node(1)
        weighted.add_node(2)
        weighted.add_node(3)
        weighted.add_node(4)
        weighted.add_edge(source=1, target=2, weight=10.0)
        weighted.add_edge(source=2, target=3, weight=10.0)
        weighted.add_edge(source=3, target=4, weight=10.0)

        uw_dist = unweighted.all_pairs_shortest_path_length()
        w_dist = weighted.all_pairs_shortest_path_length()

        # Unweighted: distance is hop count
        assert uw_dist[1][4] == 3.0

        # Weighted: distance is sum of weights
        assert w_dist[1][4] == 30.0
