"""
Tests for select_last_nodes_created() and select_last_edges_created().
"""

import pytest
from edgecase import Graph


class TestSelectLastNodesCreated:
    """Test select_last_nodes_created() functionality."""

    def test_select_last_nodes_basic(self):
        """Select the last N nodes created."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_node("E")

        # Select last 2 nodes (D=4, E=5)
        G.select_last_nodes_created(2)

        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 4 in selected  # D
        assert 5 in selected  # E

    def test_select_last_nodes_order_preserved(self):
        """Verify creation order is preserved."""
        G = Graph()
        nodes = ["Node_1", "Node_2", "Node_3", "Node_4", "Node_5"]
        for node in nodes:
            G.add_node(node)

        # Get creation order - returns node IDs
        order = G.get_node_creation_order()
        assert order == [1, 2, 3, 4, 5]  # Integer node IDs

        # Select last 3 (Node_3=3, Node_4=4, Node_5=5)
        G.select_last_nodes_created(3)
        selected = G.get_selected_nodes()

        assert len(selected) == 3
        assert 3 in selected  # Node_3
        assert 4 in selected  # Node_4
        assert 5 in selected  # Node_5

    def test_select_last_nodes_more_than_exist(self):
        """Requesting more nodes than exist selects all."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Request more than exist
        G.select_last_nodes_created(10)

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        assert set(selected) == {1, 2, 3}  # Integer node IDs

    def test_select_last_nodes_zero(self):
        """Selecting 0 nodes selects nothing."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")

        G.select_last_nodes_created(0)

        selected = G.get_selected_nodes()
        assert len(selected) == 0

    def test_select_last_nodes_with_edges(self):
        """Creation order tracks nodes even when edges are added."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        G.add_node("C")

        order = G.get_node_creation_order()
        assert order == [1, 2, 3]  # Integer node IDs for A, B, C

        G.select_last_nodes_created(2)  # Last 2: B=2, C=3
        selected = G.get_selected_nodes()

        assert len(selected) == 2
        assert 2 in selected  # B
        assert 3 in selected  # C

    def test_select_last_nodes_chaining(self):
        """Chain with other selection methods."""
        G = Graph()
        G.add_node("A", priority=1)
        G.add_node("B", priority=2)
        G.add_node("C", priority=3)
        G.add_node("D", priority=4)

        # Select last 2, then add one more by ID
        G.select_last_nodes_created(2)  # C=3, D=4
        first_count = G.count_selected_nodes()
        assert first_count == 2

        G.select_nodes_by_id([1])  # A = node ID 1

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        assert 1 in selected  # A
        assert 3 in selected  # C
        assert 4 in selected  # D

    def test_select_last_nodes_incremental_building(self):
        """Test incremental graph building workflow."""
        G = Graph()

        # Build first batch
        G.add_node("Batch1_A")
        G.add_node("Batch1_B")

        # Build second batch
        G.add_node("Batch2_A")
        G.add_node("Batch2_B")
        G.add_node("Batch2_C")

        # Select only second batch (nodes 3, 4, 5)
        G.select_last_nodes_created(3)

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        # Nodes 3, 4, 5 are the Batch2 nodes
        assert 3 in selected
        assert 4 in selected
        assert 5 in selected


class TestSelectLastEdgesCreated:
    """Test select_last_edges_created() functionality."""

    def test_select_last_edges_basic(self):
        """Select the last N edges created."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "E")

        # Select last 2 edges
        G.select_last_edges_created(2)

        selected = G.get_selected_edges()
        assert len(selected) == 2

        # Verify they're the last edges
        all_edge_ids = G.edge_ids()
        assert all_edge_ids[-2] in selected
        assert all_edge_ids[-1] in selected

    def test_select_last_edges_order_preserved(self):
        """Verify edge creation order is preserved."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
        ]

        for src, tgt in edges:
            G.add_edge(src, tgt)

        # Get creation order
        order = G.get_edge_creation_order()
        assert len(order) == 4

        # Edge IDs should be in sequential order
        assert order == [1, 2, 3, 4]

        # Select last 2
        G.select_last_edges_created(2)
        selected = G.get_selected_edges()

        assert len(selected) == 2
        assert 3 in selected
        assert 4 in selected

    def test_select_last_edges_more_than_exist(self):
        """Requesting more edges than exist selects all."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        # Request more than exist
        G.select_last_edges_created(10)

        selected = G.get_selected_edges()
        assert len(selected) == 2

    def test_select_last_edges_zero(self):
        """Selecting 0 edges selects nothing."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        G.select_last_edges_created(0)

        selected = G.get_selected_edges()
        assert len(selected) == 0

    def test_select_last_edges_with_weights(self):
        """Creation order works with weighted edges."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)
        G.add_edge("C", "D", weight=3.0)

        G.select_last_edges_created(2)
        selected = G.get_selected_edges()

        assert len(selected) == 2

        # Verify the weights of selected edges
        edges_info = G.edges_as_tuples()
        selected_weights = []
        for eid, src, tgt in edges_info:
            if eid in selected:
                edge = G.get_edge(eid)
                selected_weights.append(edge.weight)

        assert 2.0 in selected_weights
        assert 3.0 in selected_weights

    def test_select_last_edges_chaining(self):
        """Chain with other edge selection methods."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "E")

        # Select last 2 edges
        G.select_last_edges_created(2)

        # Then select edges connected to A (node ID 1)
        G.select_edges_by_node_id([1])  # A = node ID 1

        selected = G.get_selected_edges()
        # Should have 3 edges: last 2 + A-B
        assert len(selected) == 3

    def test_select_last_edges_incremental_building(self):
        """Test incremental edge building workflow."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)

        # Build initial structure
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        # Add more edges later
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        G.add_edge("E", "F")

        # Select only the last 3 edges
        G.select_last_edges_created(3)

        selected = G.get_selected_edges()
        assert len(selected) == 3

        # Verify these are edges involving C, D, E, F (node IDs 3, 4, 5, 6)
        edges_info = G.edges_as_tuples()
        selected_nodes = set()
        for eid, src, tgt in edges_info:
            if eid in selected:
                selected_nodes.add(src)  # Integer node IDs
                selected_nodes.add(tgt)

        # C=3, D=4, E=5, F=6
        assert 3 in selected_nodes  # C
        assert 4 in selected_nodes  # D
        assert 5 in selected_nodes  # E
        assert 6 in selected_nodes  # F


class TestCreationOrderHelpers:
    """Test get_node_creation_order() and get_edge_creation_order()."""

    def test_get_node_creation_order_empty(self):
        """Empty graph returns empty order."""
        G = Graph()
        order = G.get_node_creation_order()
        assert order == []

    def test_get_node_creation_order_sequence(self):
        """Node creation order matches addition sequence."""
        G = Graph()
        nodes = ["First", "Second", "Third", "Fourth"]

        for node in nodes:
            G.add_node(node)

        order = G.get_node_creation_order()
        assert order == [1, 2, 3, 4]  # Integer node IDs

    def test_get_edge_creation_order_empty(self):
        """Empty graph returns empty order."""
        G = Graph()
        order = G.get_edge_creation_order()
        assert order == []

    def test_get_edge_creation_order_sequence(self):
        """Edge creation order matches addition sequence."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)

        G.add_edge("A", "B")  # Edge ID 1
        G.add_edge("B", "C")  # Edge ID 2
        G.add_edge("C", "D")  # Edge ID 3

        order = G.get_edge_creation_order()
        assert order == [1, 2, 3]

    def test_creation_order_with_explicit_edge_ids(self):
        """Creation order works with user-specified edge IDs."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)

        G.add_edge("A", "B", edge_id=100)
        G.add_edge("B", "C", edge_id=200)
        G.add_edge("C", "D")  # Auto-assigned (likely 201)

        order = G.get_edge_creation_order()
        assert order == [100, 200, 201]


class TestCombinedWorkflows:
    """Test realistic workflows combining creation order selection."""

    def test_batch_processing_workflow(self):
        """Process graph in batches, selecting recent additions."""
        G = Graph()

        # Batch 1: Initial setup
        G.add_node("Server1", status="active")
        G.add_node("Server2", status="active")
        G.add_edge("Server1", "Server2", latency=10)

        # Batch 2: Add more nodes
        G.add_node("Server3", status="active")
        G.add_node("Server4", status="maintenance")
        G.add_edge("Server2", "Server3", latency=15)
        G.add_edge("Server3", "Server4", latency=20)

        # Select nodes from Batch 2 (Server3=3, Server4=4)
        G.select_last_nodes_created(2)
        batch2_nodes = G.get_selected_nodes()
        assert len(batch2_nodes) == 2
        assert 3 in batch2_nodes  # Server3
        assert 4 in batch2_nodes  # Server4

        # Select edges from Batch 2
        G.clear_selection()
        G.select_last_edges_created(2)
        batch2_edges = G.get_selected_edges()
        assert len(batch2_edges) == 2

    def test_pipeline_workflow(self):
        """Select recent additions, then filter by attributes."""
        G = Graph()

        # Add nodes with attributes
        G.add_node("Node1", priority=1, processed=True)
        G.add_node("Node2", priority=2, processed=True)
        G.add_node("Node3", priority=3, processed=False)
        G.add_node("Node4", priority=4, processed=False)

        # Select last 2 nodes
        G.select_last_nodes_created(2)

        # Further filter by predicate - only unprocessed
        recent_unprocessed = set(G.get_selected_nodes())
        G.clear_selection()
        G.select_nodes(
            lambda n: (
                n.id in recent_unprocessed
                and n.attributes
                and n.attributes.get("processed") == False
            )
        )

        final_selection = G.get_selected_nodes()
        assert len(final_selection) == 2
        assert 3 in final_selection  # Node3 = node ID 3
        assert 4 in final_selection  # Node4 = node ID 4

    def test_streaming_data_scenario(self):
        """Simulate streaming data where we only care about recent entries."""
        G = Graph(directed=True)

        # Simulate time-series data
        timestamps = range(1, 11)

        for t in timestamps:
            node_id = f"Event_{t}"
            G.add_node(node_id, timestamp=t, value=t * 10)

        # Connect sequential events
        for i in range(1, 10):
            G.add_edge(f"Event_{i}", f"Event_{i + 1}")

        # Select only last 3 events (most recent) - Event_8=8, Event_9=9, Event_10=10
        G.select_last_nodes_created(3)
        recent_events = G.get_selected_nodes()

        assert len(recent_events) == 3
        assert 8 in recent_events  # Event_8
        assert 9 in recent_events  # Event_9
        assert 10 in recent_events  # Event_10

        # Select last 2 edges (connections between recent events)
        G.clear_selection()
        G.select_last_edges_created(2)
        recent_connections = G.get_selected_edges()

        assert len(recent_connections) == 2


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_select_last_nodes_empty_graph(self):
        """Selecting from empty graph returns nothing."""
        G = Graph()

        G.select_last_nodes_created(5)
        assert G.count_selected_nodes() == 0

    def test_select_last_edges_empty_graph(self):
        """Selecting from empty graph returns nothing."""
        G = Graph()

        G.select_last_edges_created(5)
        assert G.count_selected_edges() == 0

    def test_select_last_nodes_after_clear(self):
        """Selection works after clearing previous selection."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Select and clear
        G.select_last_nodes_created(2)
        G.clear_selection()

        # Select again (C = node ID 3)
        G.select_last_nodes_created(1)
        selected = G.get_selected_nodes()

        assert len(selected) == 1
        assert 3 in selected  # C

    def test_creation_order_persists(self):
        """Creation order is maintained throughout graph lifetime."""
        G = Graph()

        # Add nodes
        for i in range(5):
            G.add_node(f"Node_{i}")

        # Do various operations
        G.select_last_nodes_created(2)
        G.clear_selection()
        G.select_nodes_by_id([1])  # Node_0 = node ID 1
        G.invert_node_selection()

        # Creation order should still be intact (integer IDs)
        order = G.get_node_creation_order()
        assert order == [1, 2, 3, 4, 5]  # Integer node IDs

    def test_method_chaining_with_last_created(self):
        """Verify method chaining returns Graph object."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        # Chain multiple operations
        result = G.select_last_nodes_created(2).select_last_edges_created(1).clear_selection()

        assert result is G
