from edgecase import Graph


class TestIsMatching:
    """Tests for is_matching() method."""

    def test_valid_matching_simple(self):
        """Test valid matching with disjoint edges."""
        graph = Graph()
        for i in range(6):
            graph.add_node(label=str(i))

        # Create edges: 0-1, 2-3, 4-5
        graph.add_edge("0", "1")
        graph.add_edge("2", "3")
        graph.add_edge("4", "5")

        # Get edge IDs
        edge_ids = graph.get_edge_ids()

        assert graph.is_matching(edge_ids)

    def test_invalid_matching_shared_node(self):
        """Test invalid matching with shared endpoint."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # B is shared between edges
        edge_ids = graph.get_edge_ids()

        assert not graph.is_matching(edge_ids)

    def test_empty_matching(self):
        """Test empty matching is valid."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_matching([])

    def test_single_edge_matching(self):
        """Test single edge is valid matching."""
        graph = Graph()
        graph.add_path_graph(3)

        edge_ids = graph.get_edge_ids()

        assert graph.is_matching([edge_ids[0]])

    def test_perfect_matching(self):
        """Test perfect matching in complete graph."""
        graph = Graph()
        graph.add_full_graph(4)

        # Get nodes
        nodes = list(graph.get_node_ids())

        # Create perfect matching: pick 2 non-adjacent edges
        # In K4, find edges that don't share endpoints
        all_edges = graph.get_edge_ids()
        edges = []
        for i, e1 in enumerate(all_edges):
            edge1 = graph.get_edge(e1)
            for e2 in all_edges[i + 1 :]:
                edge2 = graph.get_edge(e2)
                # Check if edges don't share endpoints
                if (
                    edge1.source != edge2.source
                    and edge1.source != edge2.target
                    and edge1.target != edge2.source
                    and edge1.target != edge2.target
                ):
                    edges = [e1, e2]
                    break
            if edges:
                break

        assert len(edges) == 2
        assert graph.is_matching(edges)

    def test_matching_directed_graph(self):
        """Test matching in directed graph."""
        graph = Graph(directed=True)
        for i in range(4):
            graph.add_node(label=str(i))

        graph.add_edge("0", "1")
        graph.add_edge("2", "3")

        edge_ids = graph.get_edge_ids()

        assert graph.is_matching(edge_ids)


class TestMaximalMatching:
    """Tests for maximal_matching() method."""

    def test_path_graph(self):
        """Test maximal matching in path graph."""
        graph = Graph()
        graph.add_path_graph(6)  # 1-2-3-4-5-6

        matching = graph.maximal_matching()

        # Should match 3 edges
        assert len(matching) >= 2  # At least 2 edges
        assert graph.is_matching(matching)

    def test_complete_graph(self):
        """Test maximal matching in complete graph."""
        graph = Graph()
        graph.add_full_graph(8)

        matching = graph.maximal_matching()

        # Perfect matching: 4 edges covering all 8 nodes
        assert len(matching) == 4
        assert graph.is_matching(matching)

    def test_star_graph(self):
        """Test maximal matching in star graph."""
        graph = Graph()
        graph.add_star_graph(5)  # 1 center, 5 leaves

        matching = graph.maximal_matching()

        # Can only match 1 edge (center is matched once)
        assert len(matching) == 1
        assert graph.is_matching(matching)

    def test_cycle_graph(self):
        """Test maximal matching in cycle graph."""
        graph = Graph()
        graph.add_cycle_graph(6)

        matching = graph.maximal_matching()

        # Should match 3 edges
        assert len(matching) == 3
        assert graph.is_matching(matching)

    def test_disconnected_components(self):
        """Test maximal matching with multiple components."""
        graph = Graph()
        graph.add_path_graph(4)

        # Add second component
        offset = graph.count_nodes()
        for i in range(4):
            graph.add_node(label=f"comp2_{i}")

        nodes = list(graph.get_node_ids())
        graph.add_edge(nodes[offset], nodes[offset + 1])
        graph.add_edge(nodes[offset + 1], nodes[offset + 2])
        graph.add_edge(nodes[offset + 2], nodes[offset + 3])

        matching = graph.maximal_matching()

        # Should match edges from both components
        assert len(matching) >= 2
        assert graph.is_matching(matching)

    def test_empty_graph(self):
        """Test maximal matching in graph with no edges."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        matching = graph.maximal_matching()

        assert len(matching) == 0

    def test_single_edge(self):
        """Test maximal matching with single edge."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        matching = graph.maximal_matching()

        assert len(matching) == 1
        assert graph.is_matching(matching)


class TestMaxWeightMatching:
    """Tests for max_weight_matching() method."""

    def test_weighted_graph_simple(self):
        """Test maximum weight matching with different weights."""
        graph = Graph(weighted=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")

        # Heavy edge
        graph.add_edge("A", "B", weight=10.0)
        # Light edges
        graph.add_edge("B", "C", weight=1.0)
        graph.add_edge("C", "D", weight=1.0)

        matching = graph.max_weight_matching()

        # Should include heavy edge (first edge ID = 1)
        edge_ids = graph.get_edge_ids()
        heavy_edge_id = edge_ids[0]  # A-B edge

        assert heavy_edge_id in matching
        assert graph.is_matching(matching)

    def test_weighted_graph_balanced(self):
        """Test when weights are similar."""
        graph = Graph(weighted=True)
        for i in range(6):
            graph.add_node(label=str(i))

        # Create path with equal weights
        for i in range(5):
            graph.add_edge(str(i), str(i + 1), weight=1.0)

        matching = graph.max_weight_matching()

        # Should match 2-3 edges
        assert len(matching) >= 2
        assert graph.is_matching(matching)

    def test_unweighted_graph(self):
        """Test max weight matching on unweighted graph."""
        graph = Graph()
        graph.add_full_graph(6)

        matching = graph.max_weight_matching()

        # Perfect matching: 3 edges
        assert len(matching) == 3
        assert graph.is_matching(matching)

    def test_triangle_heavy_edge(self):
        """Test greedy behavior with triangle."""
        graph = Graph(weighted=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        graph.add_edge("A", "B", weight=5.0)
        graph.add_edge("B", "C", weight=3.0)
        graph.add_edge("A", "C", weight=2.0)

        matching = graph.max_weight_matching()

        # Greedy picks heaviest first (first edge A-B)
        edge_ids = graph.get_edge_ids()
        heavy_edge_id = edge_ids[0]  # A-B edge

        assert heavy_edge_id in matching
        assert len(matching) == 1
        assert graph.is_matching(matching)

    def test_weighted_perfect_matching(self):
        """Test perfect matching with weights."""
        graph = Graph(weighted=True)
        for i in range(4):
            graph.add_node(label=str(i))

        # Create weighted complete graph
        weights = {
            ("0", "1"): 5.0,
            ("0", "2"): 2.0,
            ("0", "3"): 1.0,
            ("1", "2"): 3.0,
            ("1", "3"): 4.0,
            ("2", "3"): 6.0,
        }

        for (src, tgt), weight in weights.items():
            graph.add_edge(src, tgt, weight=weight)

        matching = graph.max_weight_matching()

        # Should get 2 edges (perfect matching)
        assert len(matching) == 2
        assert graph.is_matching(matching)

        # Calculate total weight
        total_weight = sum(graph.get_edge(eid).weight for eid in matching)

        assert total_weight >= 9.0  # At least decent solution

    def test_grid_graph(self):
        """Test max weight matching on grid."""
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)

        matching = graph.max_weight_matching()

        # Should get reasonable matching
        assert len(matching) >= 3
        assert graph.is_matching(matching)

    def test_compare_with_maximal(self):
        """Test that max_weight_matching finds valid maximal matching."""
        graph = Graph()
        graph.add_path_graph(10)

        matching1 = graph.maximal_matching()
        matching2 = graph.max_weight_matching()

        # Both should be valid matchings
        assert graph.is_matching(matching1)
        assert graph.is_matching(matching2)

        # Should have same cardinality for unweighted
        assert len(matching1) == len(matching2)

    def test_directed_graph_weighted(self):
        """Test max weight matching in directed graph."""
        graph = Graph(directed=True, weighted=True)
        for i in range(4):
            graph.add_node(label=str(i))

        graph.add_edge("0", "1", weight=5.0)
        graph.add_edge("1", "2", weight=2.0)
        graph.add_edge("2", "3", weight=3.0)

        matching = graph.max_weight_matching()

        assert len(matching) >= 1
        assert graph.is_matching(matching)


class TestMatchingEdgeCases:
    """Test edge cases for matching algorithms."""

    def test_self_loop(self):
        """Test matching with self-loop."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "A")  # Self-loop
        graph.add_edge("A", "B")

        matching = graph.maximal_matching()

        assert graph.is_matching(matching)

    def test_multi_edges(self):
        """Test matching with multi-edges."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")
        graph.add_edge("A", "B")  # Duplicate edge

        # Can't use both edges (share endpoints)
        edge_ids = graph.get_edge_ids()

        assert not graph.is_matching(edge_ids)

    def test_large_complete_graph(self):
        """Test matching on larger complete graph."""
        graph = Graph()
        graph.add_full_graph(20)

        matching = graph.maximal_matching()

        # Perfect matching: 10 edges
        assert len(matching) == 10
        assert graph.is_matching(matching)

    def test_matching_preserves_graph(self):
        """Test that finding matching doesn't modify graph."""
        graph = Graph()
        graph.add_path_graph(6)

        original_edges = graph.count_edges()
        original_nodes = graph.count_nodes()

        _ = graph.maximal_matching()
        _ = graph.max_weight_matching()

        assert graph.count_edges() == original_edges
        assert graph.count_nodes() == original_nodes


class TestMatchingIntegration:
    """Integration tests combining matching with other graph operations."""

    def test_matching_after_node_removal(self):
        """Test matching after removing nodes."""
        graph = Graph()
        graph.add_full_graph(6)

        # Remove a node
        node_ids = list(graph.get_node_ids())
        graph.delete_node(node_ids[0])

        matching = graph.maximal_matching()
        assert graph.is_matching(matching)

    def test_matching_with_node_attributes(self):
        """Test matching preserves node attributes."""
        graph = Graph()
        for i in range(4):
            graph.add_node(label=str(i), color=f"color_{i}")

        graph.add_edge("0", "1")
        graph.add_edge("2", "3")

        matching = graph.maximal_matching()

        # Verify attributes still exist
        for node_id in graph.get_node_ids():
            node = graph.get_node(node_id)
            assert "color" in node.attributes

    def test_weighted_matching_different_weights(self):
        """Test that weights affect matching selection."""
        graph = Graph(weighted=True)
        for i in range(6):
            graph.add_node(label=str(i))

        # Create two paths with different weights
        graph.add_edge("0", "1", weight=1.0)
        graph.add_edge("1", "2", weight=1.0)
        graph.add_edge("2", "3", weight=1.0)

        graph.add_edge("3", "4", weight=10.0)  # Heavy edge
        graph.add_edge("4", "5", weight=1.0)

        matching = graph.max_weight_matching()

        # Should include the heavy edge
        heavy_edge = None
        for edge_id in graph.get_edge_ids():
            edge = graph.get_edge(edge_id)
            if edge.weight == 10.0:
                heavy_edge = edge.id
                break

        assert heavy_edge in matching
