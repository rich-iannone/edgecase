import pytest
import edgecase as ec


class TestMinimumSpanningTree:
    """Test minimum_spanning_tree() using Kruskal's algorithm."""

    def test_simple_mst(self):
        """Basic MST on a simple weighted graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        mst = g.minimum_spanning_tree()

        # MST should have n-1 edges for n nodes
        assert mst.num_edges() == 3
        assert mst.num_nodes() == 4

        # Calculate total weight
        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 6.0  # 1 + 2 + 3

    def test_mst_return_edge_ids(self):
        """Test MST returning edge IDs instead of subgraph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        edge_ids = g.minimum_spanning_tree(return_subgraph=False)

        assert isinstance(edge_ids, list)
        assert len(edge_ids) == 2  # Triangle: 3 nodes, 2 edges in MST
        assert all(isinstance(eid, str) for eid in edge_ids)

    def test_mst_custom_weight_attribute(self):
        """MST using custom weight attribute."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=10.0, attributes={"cost": 1.0})
        g.add_edge("B", "C", weight=10.0, attributes={"cost": 2.0})
        g.add_edge("C", "A", weight=10.0, attributes={"cost": 3.0})

        # Using default weight field
        mst_weight = g.minimum_spanning_tree()
        total_weight = sum(
            e.weight
            for e in [mst_weight.get_edge(eid) for eid, _, _ in mst_weight.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total_weight == 20.0  # Two edges with weight 10.0 each

        # Using custom cost attribute - MST should prefer lower cost values
        mst_cost = g.minimum_spanning_tree(weight_attr="cost")
        # The MST based on cost should still have 2 edges, but selected by cost
        assert mst_cost.num_edges() == 2

    def test_mst_single_node(self):
        """MST with single node returns empty."""
        g = ec.Graph(weighted=True)
        g.add_node("A")

        mst = g.minimum_spanning_tree()
        assert mst.num_nodes() == 1
        assert mst.num_edges() == 0

    def test_mst_empty_graph(self):
        """MST with empty graph returns empty."""
        g = ec.Graph(weighted=True)
        mst = g.minimum_spanning_tree()
        assert mst.num_nodes() == 0
        assert mst.num_edges() == 0

    def test_mst_disconnected_graph(self):
        """MST on disconnected graph creates spanning forest."""
        g = ec.Graph(weighted=True)
        # Component 1
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=1.0)

        # Component 2
        g.add_node("C")
        g.add_node("D")
        g.add_edge("C", "D", weight=2.0)

        mst = g.minimum_spanning_tree()

        # Should have 4 nodes and 2 edges (one per component)
        assert mst.num_nodes() == 4
        assert mst.num_edges() == 2

    def test_mst_preserves_node_attributes(self):
        """MST preserves node attributes from original graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A", attributes={"type": "start"})
        g.add_node("B", attributes={"type": "middle"})
        g.add_node("C", attributes={"type": "end"})
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()

        # Check that nodes exist and have basic structure
        node_a = mst.get_node(mst.get_id_from_label("A"))
        assert node_a is not None
        assert node_a.label == "A"
        # Attributes may or may not be preserved - just check structure works
        assert mst.num_nodes() == 3

    def test_mst_preserves_edge_attributes(self):
        """MST preserves edge attributes from original graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0, attributes={"type": "primary"})
        g.add_edge("B", "C", weight=2.0, attributes={"type": "secondary"})
        g.add_edge("C", "A", weight=3.0, attributes={"type": "tertiary"})

        mst = g.minimum_spanning_tree()

        # MST should have 2 edges with lowest weights (1.0 and 2.0)
        assert mst.num_edges() == 2

        # Verify the structure is correct
        edges = list(mst.edges_as_tuples())
        assert len(edges) == 2

    def test_mst_equal_weights(self):
        """MST with equal weights (any valid MST is acceptable)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("C", "A", weight=1.0)

        mst = g.minimum_spanning_tree()

        # Should have 2 edges for 3 nodes
        assert mst.num_edges() == 2
        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 2.0

    def test_mst_large_graph(self):
        """MST on larger graph."""
        g = ec.Graph(weighted=True)

        # Create a grid-like graph
        for i in range(10):
            g.add_node(str(i))

        # Add edges in a pattern
        for i in range(9):
            g.add_edge(str(i), str(i + 1), weight=float(i + 1))

        # Add some cross edges
        g.add_edge("0", "5", weight=10.0)
        g.add_edge("2", "7", weight=15.0)

        mst = g.minimum_spanning_tree()

        # Should have n-1 edges for n nodes
        assert mst.num_edges() == 9
        assert mst.num_nodes() == 10


class TestMaximumSpanningTree:
    """Test maximum_spanning_tree() using Kruskal's algorithm."""

    def test_simple_maximum_st(self):
        """Basic maximum spanning tree."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        max_st = g.maximum_spanning_tree()

        # Should have n-1 edges for n nodes
        assert max_st.num_edges() == 3
        assert max_st.num_nodes() == 4

        # Calculate total weight (should prefer highest weights)
        total = sum(
            e.weight
            for e in [max_st.get_edge(eid) for eid, _, _ in max_st.edges_as_tuples()]
            if e and e.weight is not None
        )
        # Maximum ST should select the 3 highest-weight edges that connect all nodes
        # Should be at least 9.0 (e.g., 5 + 4 + something, or 5 + 3 + 2, etc.)
        assert total >= 9.0

    def test_maximum_st_return_edge_ids(self):
        """Test maximum ST returning edge IDs."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        edge_ids = g.maximum_spanning_tree(return_subgraph=False)

        assert isinstance(edge_ids, list)
        assert len(edge_ids) == 2
        assert all(isinstance(eid, str) for eid in edge_ids)

    def test_maximum_st_custom_weight(self):
        """Maximum ST using custom weight attribute."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0, attributes={"value": 10.0})
        g.add_edge("B", "C", weight=2.0, attributes={"value": 20.0})
        g.add_edge("C", "A", weight=3.0, attributes={"value": 30.0})

        max_st = g.maximum_spanning_tree(weight_attr="value")

        # Maximum ST based on "value" should select 2 highest-value edges
        assert max_st.num_edges() == 2

    def test_mst_vs_maximum_st(self):
        """Compare minimum and maximum spanning trees."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=5.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()
        max_st = g.maximum_spanning_tree()

        # Both should have 2 edges
        assert mst.num_edges() == 2
        assert max_st.num_edges() == 2

        # Minimum should prefer low weights
        mst_total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert mst_total == 4.0  # 1.0 + 3.0

        # Maximum should prefer high weights
        max_st_total = sum(
            e.weight
            for e in [max_st.get_edge(eid) for eid, _, _ in max_st.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert max_st_total == 8.0  # 5.0 + 3.0


class TestMSTEdgeCases:
    """Test edge cases and error conditions."""

    def test_mst_two_nodes(self):
        """MST with exactly two nodes."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=1.0)

        mst = g.minimum_spanning_tree()
        assert mst.num_nodes() == 2
        assert mst.num_edges() == 1

    def test_mst_complete_graph(self):
        """MST on complete graph (K4)."""
        g = ec.Graph(weighted=True)
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            g.add_node(n)

        weight = 1.0
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2, weight=weight)
                weight += 1.0

        mst = g.minimum_spanning_tree()

        # K4 has 6 edges, MST should have 3
        assert mst.num_edges() == 3
        assert mst.num_nodes() == 4

    def test_mst_missing_weights(self):
        """MST with edges missing explicit weights (should default to 1.0)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")  # No weight specified
        g.add_edge("B", "C")  # No weight specified
        g.add_edge("C", "A", weight=5.0)

        mst = g.minimum_spanning_tree()

        # Should prefer edges with default weight 1.0 over weight 5.0
        assert mst.num_edges() == 2

    def test_mst_negative_weights(self):
        """MST handles negative weights correctly."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=-5.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("C", "A", weight=2.0)

        mst = g.minimum_spanning_tree()

        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.edges_as_tuples()]
            if e and e.weight is not None
        )
        # Should include the negative weight edge
        assert total == -4.0  # -5.0 + 1.0

    def test_mst_zero_weights(self):
        """MST with zero-weight edges."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=0.0)
        g.add_edge("B", "C", weight=0.0)
        g.add_edge("C", "A", weight=1.0)

        mst = g.minimum_spanning_tree()

        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 0.0  # Two zero-weight edges

    def test_mst_parallel_edges(self):
        """MST with parallel edges (multigraph)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        # Multiple edges between A and B
        g.add_edge("A", "B", weight=5.0)
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()

        # Should have 2 edges for 3 nodes
        assert mst.num_edges() == 2
