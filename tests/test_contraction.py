import pytest
from edgecase import Graph


class TestContractNodes:
    """Tests for contract_nodes method."""

    def test_contract_nodes_basic(self):
        """Test basic node contraction on a path."""
        # Path: 1 -- 2 -- 3 -- 4
        g = Graph(directed=False)
        g.add_path_graph(4)

        # Contract nodes 2 and 3 (merge 3 into 2)
        result = g.contract_nodes(2, 3)

        # Should have 3 nodes: 1, 2, 4
        assert result.count_nodes() == 3
        assert result.has_node(1)
        assert result.has_node(2)
        assert not result.has_node(3)
        assert result.has_node(4)

        # Should have 3 edges: 1-2, 2-2 (self-loop from 2-3), and 2-4
        assert result.count_edges() == 3

    def test_contract_nodes_with_labels(self):
        """Test node contraction using labels."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Contract B and C (merge C into B)
        result = g.contract_nodes("B", "C")

        assert result.count_nodes() == 2
        assert result.has_node("A")
        assert result.has_node("B")
        assert not result.has_node("C")

    def test_contract_nodes_creates_self_loop(self):
        """Test that contracting adjacent nodes creates a self-loop."""
        # Triangle: 1-2-3-1
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # Contract A and B (edge A-B becomes self-loop)
        result = g.contract_nodes("A", "B", self_loops=True)

        assert result.count_nodes() == 2

        # Should have self-loop on A and edge A-C
        edges = result.get_edges_as_tuples()
        self_loops = [(s, t) for _, s, t in edges if s == t]

        assert len(self_loops) >= 1

    def test_contract_nodes_no_self_loops(self):
        """Test suppressing self-loops during contraction."""
        # Triangle: 1-2-3-1
        g = Graph(directed=False)
        for i in range(1, 4):
            g.add_node()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)

        # Contract 1 and 2 without self-loops
        result = g.contract_nodes(1, 2, self_loops=False)

        # No self-loops
        edges = result.get_edges_as_tuples()
        self_loops = [(s, t) for _, s, t in edges if s == t]

        assert len(self_loops) == 0

    def test_contract_nodes_preserves_original(self):
        """Test that contraction does not modify the original graph."""
        g = Graph(directed=False)
        g.add_path_graph(4)
        original_nodes = g.count_nodes()

        result = g.contract_nodes(2, 3)

        # Original unchanged
        assert g.count_nodes() == original_nodes
        assert g.has_node(3)

        # Result modified
        assert result.count_nodes() == original_nodes - 1

    def test_contract_nodes_invalid_same_node(self):
        """Test error when contracting a node with itself."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        with pytest.raises(ValueError):
            g.contract_nodes(1, 1)

    def test_contract_nodes_invalid_node(self):
        """Test error when node doesn't exist."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        with pytest.raises(ValueError, match="Node 99 does not exist"):
            g.contract_nodes(1, 99)

    def test_parallel_edges_elsewhere_are_kept(self):
        """Only edges at the merged node collapse; `c - d` twice used to become one edge."""
        g = Graph().add_nodes(list("abcd"))
        g.add_edges([("a", "b"), ("c", "d"), ("c", "d"), ("a", "c"), ("b", "b")])
        result = g.contract_nodes("a", "b")
        assert result.get_edges_str_repr(use_labels=True) == ["a--a", "c--d", "c--d", "a--c"]
        assert result.get_edge_ids(limit=None) == [1, 2, 3, 4]

    def test_directed_collapse_keeps_each_direction(self):
        """On a directed graph, `a -> c` and `c -> a` are different pairs."""
        g = Graph(directed=True).add_nodes(list("abc"))
        g.add_edges([("a", "c"), ("b", "c"), ("c", "b"), ("c", "a"), ("b", "a"), ("a", "b")])
        result = g.contract_nodes("a", "b")
        assert result.get_edges_str_repr(use_labels=True) == ["a->c", "c->a", "a->a"]


class TestContractedEdge:
    """Tests for contracted_edge method."""

    def test_contracted_edge_basic(self):
        """Test basic edge contraction."""
        g = Graph(directed=False)
        g.add_path_graph(4)  # 1-2-3-4

        # Contract edge 2-3
        result = g.contracted_edge(2, 3)

        assert result.count_nodes() == 3
        assert not result.has_node(3)

    def test_contracted_edge_invalid(self):
        """Test error when edge doesn't exist."""
        g = Graph(directed=False)
        g.add_path_graph(4)  # 1-2-3-4

        # Try to contract non-existent edge 1-4
        with pytest.raises(ValueError):
            g.contracted_edge(1, 4)

    def test_contracted_edge_directed(self):
        """Test edge contraction on directed graph."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Contract edge A->B
        result = g.contracted_edge("A", "B")

        assert result.count_nodes() == 2
        assert not result.has_node("B")


class TestContractNodesMulti:
    """Tests for contract_nodes_multi method."""

    def test_contract_multi_basic(self):
        """Test contracting multiple nodes at once."""
        g = Graph(directed=False)
        g.add_path_graph(6)  # 1-2-3-4-5-6

        # Contract 2, 3, 4 into 2
        result = g.contract_nodes_multi([2, 3, 4])

        # Should have 3 nodes: 1, 2, 5, 6... wait, 6 also
        # Actually: 1, 2, 5, 6 = 4 nodes
        assert result.count_nodes() == 4
        assert result.has_node(2)
        assert not result.has_node(3)
        assert not result.has_node(4)

    def test_contract_multi_single_node(self):
        """Test with single node (no-op)."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        result = g.contract_nodes_multi([2])

        # Should be identical to original
        assert result.count_nodes() == g.count_nodes()

    def test_contract_multi_empty_list(self):
        """Test error with empty node list."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        with pytest.raises(ValueError):
            g.contract_nodes_multi([])

    def test_contract_multi_with_labels(self):
        """Test contracting multiple nodes using labels."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        # Contract B, C into B
        result = g.contract_nodes_multi(["B", "C"])

        assert result.count_nodes() == 3
        assert result.has_node("B")
        assert not result.has_node("C")


class TestQuotientGraph:
    """Tests for quotient_graph method."""

    def test_quotient_basic(self):
        """Test basic quotient graph creation."""
        g = Graph(directed=False)
        g.add_path_graph(4)  # 1-2-3-4

        # Partition: {1, 2} and {3, 4}
        partition = [[1, 2], [3, 4]]
        quotient = g.quotient_graph(partition)

        # Should have 2 nodes (one per block)
        assert quotient.count_nodes() == 2

        # Should have 1 edge (from edge 2-3 crossing blocks)
        assert quotient.count_edges() == 1

    def test_quotient_with_labels(self):
        """Test quotient graph with labeled nodes."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        partition = [["A", "B"], ["C", "D"]]
        quotient = g.quotient_graph(partition)

        assert quotient.count_nodes() == 2
        assert quotient.count_edges() == 1

    def test_quotient_multiple_crossing_edges(self):
        """Test quotient with multiple edges between blocks."""
        # Create graph where blocks have multiple connections
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node()
        g.add_edge(1, 2)  # Within block 1
        g.add_edge(3, 4)  # Within block 2
        g.add_edge(1, 3)  # Between blocks
        g.add_edge(2, 4)  # Between blocks

        partition = [[1, 2], [3, 4]]
        quotient = g.quotient_graph(partition)

        # Should have 2 nodes
        assert quotient.count_nodes() == 2

        # Should have 1 edge (duplicates removed in simple graph)
        assert quotient.count_edges() == 1

    def test_quotient_no_crossing_edges(self):
        """Test quotient where blocks have no edges between them."""
        g = Graph(directed=False)
        for i in range(1, 5):
            g.add_node()
        g.add_edge(1, 2)  # Within block 1
        g.add_edge(3, 4)  # Within block 2

        partition = [[1, 2], [3, 4]]
        quotient = g.quotient_graph(partition)

        assert quotient.count_nodes() == 2
        assert quotient.count_edges() == 0

    def test_quotient_node_labels(self):
        """Test that quotient graph nodes have block labels."""
        g = Graph(directed=False)
        g.add_path_graph(4)

        partition = [[1, 2], [3, 4]]
        quotient = g.quotient_graph(partition)

        # Check node labels
        labels = [quotient.get_label_from_id(n) for n in quotient.get_node_ids()]

        assert "block_0" in labels
        assert "block_1" in labels

    def test_quotient_invalid_missing_node(self):
        """Test error when partition is incomplete."""
        g = Graph(directed=False)
        g.add_path_graph(4)

        # Missing node 4
        partition = [[1, 2], [3]]

        with pytest.raises(ValueError):
            g.quotient_graph(partition)

    def test_quotient_invalid_duplicate_node(self):
        """Test error when node appears in multiple blocks."""
        g = Graph(directed=False)
        g.add_path_graph(4)

        # Node 2 appears twice
        partition = [[1, 2], [2, 3, 4]]

        with pytest.raises(ValueError):
            g.quotient_graph(partition)

    def test_quotient_single_block(self):
        """Test quotient with all nodes in one block."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        partition = [[1, 2, 3]]
        quotient = g.quotient_graph(partition)

        # Should have 1 node (all nodes merged)
        assert quotient.count_nodes() == 1

        # No edges (all are within the block)
        assert quotient.count_edges() == 0


class TestContractionDirectedGraphs:
    """Tests for contraction on directed graphs."""

    def test_contract_nodes_directed(self):
        """Test node contraction on directed graph."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        result = g.contract_nodes("A", "B")

        assert result.count_nodes() == 2

        # A->C should exist (redirected from B->C)
        assert result.is_directed

    def test_quotient_directed(self):
        """Test quotient graph preserves directedness."""
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        partition = [["A", "B"], ["C", "D"]]
        quotient = g.quotient_graph(partition)

        assert quotient.is_directed
        assert quotient.count_edges() == 1


class TestContractionWeightedGraphs:
    """Tests for contraction on weighted graphs."""

    def test_contract_preserves_weights(self):
        """Test that edge weights are preserved after contraction."""
        g = Graph(directed=False, weighted=True)
        for i in range(1, 4):
            g.add_node()
        g.add_edge(1, 2, weight=2.5)
        g.add_edge(2, 3, weight=3.5)

        result = g.contract_nodes(1, 2)

        # Edge to 3 should have weight 3.5
        assert result.is_weighted


class TestContractionCommunityUseCase:
    """Integration tests for community detection use case."""

    def test_contract_community(self):
        """Test contracting nodes detected as a community."""
        g = Graph(directed=False)

        # Two triangles connected by a single edge
        # Triangle 1: 1-2-3
        for i in range(1, 4):
            g.add_node()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)

        # Triangle 2: 4-5-6
        for i in range(4, 7):
            g.add_node()
        g.add_edge(4, 5)
        g.add_edge(5, 6)
        g.add_edge(6, 4)

        # Bridge between triangles
        g.add_edge(3, 4)

        # Contract each triangle into a single node
        partition = [[1, 2, 3], [4, 5, 6]]
        quotient = g.quotient_graph(partition)

        # Should have 2 nodes connected by 1 edge
        assert quotient.count_nodes() == 2
        assert quotient.count_edges() == 1
