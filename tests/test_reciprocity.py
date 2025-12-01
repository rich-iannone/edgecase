import pytest
import edgecase as ec


class TestWhichMutual:
    """Test which_mutual() function."""

    def test_empty_graph(self):
        """Empty graph has no mutual edges."""
        g = ec.Graph(directed=True)
        mutual = g.which_mutual()
        assert len(mutual) == 0

    def test_single_edge_no_mutual(self):
        """Single directed edge has no mutual pair."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        mutual = g.which_mutual()
        assert len(mutual) == 0

    def test_simple_mutual_pair(self):
        """Two edges A->B and B->A are mutual."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")  # Edge 1
        g.add_edge("B", "A")  # Edge 2

        mutual = g.which_mutual()
        assert len(mutual) == 1
        assert mutual[0][0] < mutual[0][1]  # Edge IDs in order

    def test_multiple_mutual_pairs(self):
        """Multiple mutual pairs should all be detected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Mutual 1
        g.add_edge("B", "C")
        g.add_edge("C", "B")  # Mutual 2

        mutual = g.which_mutual()
        assert len(mutual) == 2

    def test_mixed_mutual_and_one_way(self):
        """Graph with both mutual and one-way edges."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Mutual
        g.add_edge("A", "C")  # One-way, no reciprocal

        mutual = g.which_mutual()
        assert len(mutual) == 1

    def test_triangle_with_all_mutual(self):
        """Triangle where all edges are mutual."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        # Create bidirectional triangle
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("B", "C")
        g.add_edge("C", "B")
        g.add_edge("C", "A")
        g.add_edge("A", "C")

        mutual = g.which_mutual()
        assert len(mutual) == 3  # Three mutual pairs

    def test_self_loop_ignored(self):
        """Self-loops should be ignored in mutual detection."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Mutual

        mutual = g.which_mutual()
        assert len(mutual) == 1  # Only A-B mutual, not self-loop

    def test_undirected_graph_raises_error(self):
        """Should raise error for undirected graphs."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="directed"):
            g.which_mutual()

    def test_multi_edges_mutual(self):
        """Multiple edges between same nodes should be detected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        # Two edges A->B
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        # One edge B->A
        g.add_edge("B", "A")

        mutual = g.which_mutual()
        # Should find 2 mutual pairs (each forward edge pairs with reverse)
        assert len(mutual) == 2


class TestReciprocity:
    """Test get_reciprocity() function."""

    def test_empty_graph(self):
        """Empty graph has reciprocity 0.0."""
        g = ec.Graph(directed=True)
        assert g.get_reciprocity() == 0.0

    def test_single_edge(self):
        """Single edge has reciprocity 0.0."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert g.get_reciprocity() == 0.0

    def test_perfect_reciprocity(self):
        """All edges mutual gives reciprocity 1.0."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        assert g.get_reciprocity() == 1.0

    def test_partial_reciprocity(self):
        """Mixed mutual and one-way edges."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Mutual (2 edges)
        g.add_edge("A", "C")  # One-way (1 edge)

        recip = g.get_reciprocity()
        # 2 mutual edges out of 3 total
        assert abs(recip - (2.0 / 3.0)) < 1e-10

    def test_no_reciprocity(self):
        """All one-way edges gives reciprocity 0.0."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # One-way cycle, no mutual pairs
        assert g.get_reciprocity() == 0.0

    def test_ratio_mode(self):
        """Test 'ratio' mode (should be same as default)."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("A", "C")

        recip_default = g.get_reciprocity("default")
        recip_ratio = g.get_reciprocity("ratio")
        assert abs(recip_default - recip_ratio) < 1e-10

    def test_multiple_mutual_pairs(self):
        """Multiple mutual pairs calculation."""
        g = ec.Graph(directed=True)
        # Create 3 nodes with all mutual connections
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Mutual
        g.add_edge("B", "C")
        g.add_edge("C", "B")  # Mutual
        g.add_edge("A", "C")
        g.add_edge("C", "A")  # Mutual

        # All 6 edges are mutual
        assert g.get_reciprocity() == 1.0

    def test_star_graph_no_reciprocity(self):
        """Star graph (hub with out-edges) has no reciprocity."""
        g = ec.Graph(directed=True)
        g.add_node(label="hub")
        for i in range(5):
            g.add_node(label=str(i))
            g.add_edge("hub", str(i))

        assert g.get_reciprocity() == 0.0

    def test_undirected_graph_raises_error(self):
        """Should raise error for undirected graphs."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="directed"):
            g.get_reciprocity()

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise error."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="Invalid mode"):
            g.get_reciprocity("invalid")

    def test_self_loops_ignored_in_reciprocity(self):
        """Self-loops should not affect reciprocity."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")

        # Only 1 non-self-loop edge, no mutual
        assert g.get_reciprocity() == 0.0


class TestReciprocityUseCases:
    """Test realistic use cases for reciprocity analysis."""

    def test_social_network_friendship(self):
        """Friendship network should have high reciprocity."""
        g = ec.Graph(directed=True)
        # Friends typically reciprocate
        friends = [
            ("Alice", "Bob"),
            ("Bob", "Alice"),
            ("Bob", "Carol"),
            ("Carol", "Bob"),
            ("Alice", "Carol"),
            ("Carol", "Alice"),
        ]

        for a, b in friends:
            if not g.has_node(a):
                g.add_node(label=a)
            if not g.has_node(b):
                g.add_node(label=b)
            g.add_edge(a, b)

        assert g.get_reciprocity() == 1.0

    def test_citation_network_low_reciprocity(self):
        """Citation networks typically have low reciprocity."""
        g = ec.Graph(directed=True)
        # Papers cite older papers, rarely reciprocal
        citations = [
            ("Paper2020", "Paper2015"),
            ("Paper2020", "Paper2018"),
            ("Paper2021", "Paper2020"),
            ("Paper2021", "Paper2015"),
        ]

        for a, b in citations:
            if not g.has_node(a):
                g.add_node(label=a)
            if not g.has_node(b):
                g.add_node(label=b)
            g.add_edge(a, b)

        # No reciprocal citations
        assert g.get_reciprocity() == 0.0

    def test_communication_network_partial(self):
        """Communication network with some bidirectional communication."""
        g = ec.Graph(directed=True)
        edges = [
            ("Alice", "Bob"),
            ("Bob", "Alice"),  # Mutual
            ("Alice", "Carol"),  # One-way
            ("Bob", "Dave"),  # One-way
            ("Dave", "Carol"),
            ("Carol", "Dave"),  # Mutual
        ]

        for a, b in edges:
            if not g.has_node(a):
                g.add_node(label=a)
            if not g.has_node(b):
                g.add_node(label=b)
            g.add_edge(a, b)

        recip = g.get_reciprocity()
        # 4 mutual out of 6 total edges
        assert abs(recip - (4.0 / 6.0)) < 1e-10

    def test_which_mutual_returns_edge_ids(self):
        """Verify which_mutual returns valid edge IDs."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        mutual = g.which_mutual()
        assert len(mutual) == 1

        # Verify the edge IDs are valid
        edge1_id, edge2_id = mutual[0]
        assert g.has_edge_id(edge1_id)
        assert g.has_edge_id(edge2_id)

        # Verify they point to the right nodes (returns node IDs, not labels)
        e1 = g.get_node_ids_by_edge_id(edge1_id)
        e2 = g.get_node_ids_by_edge_id(edge2_id)

        # Both should be between the same two nodes
        nodes_e1 = set(e1)
        nodes_e2 = set(e2)
        assert nodes_e1 == nodes_e2

        # They should be opposite directions
        assert e1[0] == e2[1] and e1[1] == e2[0]


class TestReciprocityEdgeCases:
    """Test edge cases for reciprocity functions."""

    def test_single_node_graph(self):
        """Single node with no edges."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        assert len(g.which_mutual()) == 0
        assert g.get_reciprocity() == 0.0

    def test_two_nodes_no_edges(self):
        """Two nodes with no edges."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")

        assert len(g.which_mutual()) == 0
        assert g.get_reciprocity() == 0.0

    def test_complete_directed_graph(self):
        """Complete directed graph (all possible edges)."""
        g = ec.Graph(directed=True)
        nodes = ["A", "B", "C"]
        for n in nodes:
            g.add_node(label=n)

        # Add all possible directed edges
        for n1 in nodes:
            for n2 in nodes:
                if n1 != n2:
                    g.add_edge(n1, n2)

        # All edges should be mutual
        assert g.get_reciprocity() == 1.0
        mutual = g.which_mutual()
        # 3 pairs of mutual edges (A-B, A-C, B-C)
        assert len(mutual) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
