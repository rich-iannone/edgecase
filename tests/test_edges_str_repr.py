"""Tests for edges_str_repr() method."""

import pytest
from edgecase import Graph


class TestEdgesStrRepr:
    """Test suite for edges_str_repr() method."""

    def test_basic_directed_graph_with_ids(self):
        """Test basic directed graph returns node IDs by default."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        edges = graph.get_edges_str_repr()
        assert len(edges) == 2
        assert "1->2" in edges
        assert "2->3" in edges

    def test_basic_undirected_graph_with_ids(self):
        """Test basic undirected graph returns node IDs by default."""
        graph = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        edges = graph.get_edges_str_repr()
        assert len(edges) == 2
        assert "1--2" in edges
        assert "2--3" in edges

    def test_directed_graph_with_labels(self):
        """Test directed graph with use_labels=True."""
        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
            .add_edge("Alice", "Bob")
            .add_edge("Bob", "Charlie")
        )

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 2
        assert "Alice->Bob" in edges
        assert "Bob->Charlie" in edges

    def test_undirected_graph_with_labels(self):
        """Test undirected graph with use_labels=True."""
        graph = Graph(directed=False)
        graph.add_node(label="Dog")
        graph.add_node(label="Cat")
        graph.add_edge("Dog", "Cat")

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 1
        assert "Dog--Cat" in edges

    def test_mixed_labels_and_no_labels(self):
        """Test graph where some nodes have labels and others don't."""
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node()  # No label
        graph.add_node(label="Charlie")

        # Get the ID of the unlabeled node
        all_ids = graph.get_node_ids(limit=None)
        unlabeled_id = [nid for nid in all_ids if graph.get_label_from_id(nid) is None][0]

        graph.add_edge("Alice", unlabeled_id)
        graph.add_edge(unlabeled_id, "Charlie")

        # With use_labels=True, should fall back to IDs for unlabeled nodes
        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 2

        # Check that labeled nodes show labels, unlabeled shows ID
        assert any(f"Alice--{unlabeled_id}" in edge for edge in edges)
        assert any(f"{unlabeled_id}--Charlie" in edge for edge in edges)

    def test_no_labels_at_all(self):
        """Test graph with no labels using use_labels=True."""
        graph = Graph()
        graph.add_node()  # ID 1, no label
        graph.add_node()  # ID 2, no label
        graph.add_edge(1, 2)

        # Should fall back to IDs
        edges_with_labels = graph.get_edges_str_repr(use_labels=True)
        edges_default = graph.get_edges_str_repr(use_labels=False)

        assert edges_with_labels == edges_default
        assert "1--2" in edges_with_labels

    def test_limit_parameter(self):
        """Test limit parameter restricts number of edges returned."""
        graph = Graph()
        for i in range(10):
            graph.add_node(label=f"N{i}")

        # Create a path graph
        for i in range(9):
            graph.add_edge(f"N{i}", f"N{i + 1}")

        # Test with limit
        edges_limited = graph.get_edges_str_repr(limit=5)
        assert len(edges_limited) == 5

        # Test with no limit
        edges_all = graph.get_edges_str_repr(limit=None)
        assert len(edges_all) == 9

    def test_limit_with_labels(self):
        """Test limit parameter works with use_labels=True."""
        graph = Graph(directed=True)
        for i in range(10):
            graph.add_node(label=f"Node{i}")

        for i in range(9):
            graph.add_edge(f"Node{i}", f"Node{i + 1}")

        edges = graph.get_edges_str_repr(limit=3, use_labels=True)
        assert len(edges) == 3
        # Should be formatted with labels
        assert all("->" in edge for edge in edges)
        assert all("Node" in edge for edge in edges)

    def test_empty_graph(self):
        """Test edges_str_repr on empty graph."""
        graph = Graph()
        edges = graph.get_edges_str_repr()
        assert edges == []

        edges_with_labels = graph.get_edges_str_repr(use_labels=True)
        assert edges_with_labels == []

    def test_graph_with_nodes_no_edges(self):
        """Test graph with nodes but no edges."""
        graph = Graph().add_node(label="A").add_node(label="B")

        edges = graph.get_edges_str_repr()
        assert edges == []

        edges_with_labels = graph.get_edges_str_repr(use_labels=True)
        assert edges_with_labels == []

    def test_self_loop(self):
        """Test edge representation for self-loops."""
        graph = Graph(directed=True)
        graph.add_node(label="SelfNode")
        graph.add_edge("SelfNode", "SelfNode")

        edges_ids = graph.get_edges_str_repr()
        assert len(edges_ids) == 1
        assert edges_ids[0] == "1->1"

        edges_labels = graph.get_edges_str_repr(use_labels=True)
        assert len(edges_labels) == 1
        assert edges_labels[0] == "SelfNode->SelfNode"

    def test_multigraph(self):
        """Test multiple edges between same nodes."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")
        graph.add_edge("A", "B")

        edges_ids = graph.get_edges_str_repr()
        assert len(edges_ids) == 2
        # Both should be the same representation
        assert all(edge == "1--2" for edge in edges_ids)

        edges_labels = graph.get_edges_str_repr(use_labels=True)
        assert len(edges_labels) == 2
        assert all(edge == "A--B" for edge in edges_labels)

    def test_large_graph_default_limit(self):
        """Test that default limit of 1000 is applied."""
        graph = Graph()

        # Create a graph with more than 1000 edges
        num_nodes = 50
        for i in range(num_nodes):
            graph.add_node(label=f"N{i}")

        # Create a complete graph (lots of edges)
        edge_count = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                graph.add_edge(f"N{i}", f"N{j}")
                edge_count += 1
                if edge_count >= 1500:  # Create > 1000 edges
                    break
            if edge_count >= 1500:
                break

        # Default should return 1000 edges
        edges_default = graph.get_edges_str_repr()
        assert len(edges_default) == 1000

        # Explicit limit=None should return all
        edges_all = graph.get_edges_str_repr(limit=None)
        assert len(edges_all) == edge_count

    def test_directed_vs_undirected_separators(self):
        """Test that correct separators are used for directed vs undirected."""
        # Directed graph should use ->
        graph_directed = (
            Graph(directed=True).add_node(label="A").add_node(label="B").add_edge("A", "B")
        )

        edges = graph_directed.get_edges_str_repr(use_labels=True)
        assert "->" in edges[0]
        assert "--" not in edges[0]

        # Undirected graph should use --
        graph_undirected = (
            Graph(directed=False).add_node(label="X").add_node(label="Y").add_edge("X", "Y")
        )

        edges = graph_undirected.get_edges_str_repr(use_labels=True)
        assert "--" in edges[0]
        assert "->" not in edges[0]

    def test_special_characters_in_labels(self):
        """Test labels with special characters."""
        graph = Graph()
        graph.add_node(label="Node-1")
        graph.add_node(label="Node_2")
        graph.add_node(label="Node 3")
        graph.add_edge("Node-1", "Node_2")
        graph.add_edge("Node_2", "Node 3")

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 2
        assert "Node-1--Node_2" in edges
        assert "Node_2--Node 3" in edges

    def test_numeric_string_labels(self):
        """Test labels that are numeric strings."""
        graph = Graph()
        graph.add_node(label="100")
        graph.add_node(label="200")
        graph.add_edge("100", "200")

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 1
        assert "100--200" in edges

    def test_consistency_between_calls(self):
        """Test that multiple calls return consistent results."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        edges1 = graph.get_edges_str_repr(use_labels=True)
        edges2 = graph.get_edges_str_repr(use_labels=True)

        assert edges1 == edges2

    def test_use_labels_false_explicit(self):
        """Test that use_labels=False explicitly uses IDs."""
        graph = Graph().add_node(label="Alice").add_node(label="Bob").add_edge("Alice", "Bob")

        edges = graph.get_edges_str_repr(use_labels=False)
        assert len(edges) == 1
        assert "1--2" in edges
        assert "Alice" not in edges[0]
        assert "Bob" not in edges[0]

    def test_combined_parameters(self):
        """Test using limit and use_labels together."""
        graph = Graph(directed=True)
        for i in range(20):
            graph.add_node(label=f"Node{i}")

        for i in range(19):
            graph.add_edge(f"Node{i}", f"Node{i + 1}")

        edges = graph.get_edges_str_repr(limit=10, use_labels=True)
        assert len(edges) == 10
        assert all("Node" in edge for edge in edges)
        assert all("->" in edge for edge in edges)

    def test_node_with_none_label(self):
        """Test node explicitly set with None as label."""
        graph = Graph()
        graph.add_node(label=None)
        graph.add_node(label="HasLabel")
        # Get the ID of the first node (no label)
        node_ids = graph.get_node_ids()
        first_node_id = node_ids[0]
        graph.add_edge(first_node_id, "HasLabel")

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 1
        # Node with None label should show ID (whatever it was auto-assigned)
        assert "--HasLabel" in edges[0] or "HasLabel--" in edges[0]
        # Should not show "None" as the label
        assert "None" not in edges[0]

    def test_after_node_deletion(self):
        """Test edges_str_repr after deleting nodes."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Delete middle node
        graph.delete_node("B")

        edges = graph.get_edges_str_repr(use_labels=True)
        # Should have no edges now (B connected both)
        assert len(edges) == 0

    def test_after_edge_deletion(self):
        """Test edges_str_repr after deleting edges."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        graph.delete_edge(source="A", target="B")

        edges = graph.get_edges_str_repr(use_labels=True)
        assert len(edges) == 1
        assert "B--C" in edges
        assert "A--B" not in edges
