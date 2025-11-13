import pytest
from edgecase import Graph


class TestDeleteNodeBasics:
    """Test basic delete_node functionality."""

    def test_delete_node_by_label(self):
        """Test deleting a node by its label."""
        graph = Graph().add_node(label="A").add_node(label="B").add_node(label="C")

        assert graph.has_node("B")
        graph.delete_node("B")
        assert not graph.has_node("B")
        assert graph.num_nodes() == 2

    def test_delete_node_by_id(self):
        """Test deleting a node by its ID."""
        graph = Graph().add_node(label="Alice")
        node_id = graph.get_id_from_label("Alice")

        assert graph.has_node(node_id)
        graph.delete_node(node_id)
        assert not graph.has_node(node_id)
        assert graph.num_nodes() == 0

    def test_delete_nonexistent_node_raises_error(self):
        """Test that deleting non-existent node raises KeyError."""
        graph = Graph().add_node(label="A")

        with pytest.raises(KeyError):
            graph.delete_node("NonExistent")

        with pytest.raises(KeyError):
            graph.delete_node(9999)

    def test_delete_node_returns_self(self):
        """Test that delete_node returns self for chaining."""
        graph = Graph().add_node(label="A").add_node(label="B")

        result = graph.delete_node("A")
        assert result is graph
        assert not graph.has_node("A")


class TestDeleteNodeWithEdges:
    """Test delete_node with incident edges."""

    def test_delete_node_removes_incident_edges(self):
        """Test that deleting a node removes all incident edges."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        initial_edges = graph.num_edges()
        assert initial_edges == 2

        graph.delete_node("B")

        assert graph.num_edges() == 0  # Both edges removed
        # Note: Cannot use has_edge with deleted node label "B"

    def test_delete_node_directed_graph(self):
        """Test deleting node in directed graph removes incoming and outgoing edges."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")  # A -> B
            .add_edge("B", "C")  # B -> C
            .add_edge("C", "A")  # C -> A
        )

        assert graph.num_edges() == 3

        graph.delete_node("B")

        assert graph.num_edges() == 1  # Only C -> A remains
        # Note: Cannot use has_edge with deleted node label "B"
        assert graph.has_edge("C", "A")

    def test_delete_node_with_self_loop(self):
        """Test deleting node that has a self-loop."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "A")  # Self-loop
            .add_edge("A", "B")
        )

        assert graph.num_edges() == 2

        graph.delete_node("A")

        assert graph.num_edges() == 0
        assert graph.num_nodes() == 1
        assert graph.has_node("B")

    def test_delete_node_with_multiple_edges(self):
        """Test deleting node in multigraph with parallel edges."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", edge_id=1)
            .add_edge("A", "B", edge_id=2)
            .add_edge("B", "C", edge_id=3)
        )

        assert graph.num_edges() == 3

        graph.delete_node("B")

        assert graph.num_edges() == 0  # All three edges removed
        assert graph.num_nodes() == 2


class TestDeleteNodeIsolated:
    """Test deleting isolated nodes."""

    def test_delete_isolated_node(self):
        """Test deleting a node with no edges."""
        graph = (
            Graph().add_node(label="A").add_node(label="B").add_node(label="C").add_edge("A", "B")
        )

        assert graph.num_nodes() == 3
        assert graph.num_edges() == 1

        graph.delete_node("C")  # Isolated node

        assert graph.num_nodes() == 2
        assert graph.num_edges() == 1  # Edge count unchanged

    def test_delete_all_nodes(self):
        """Test deleting all nodes one by one."""
        graph = Graph().add_node(label="A").add_node(label="B").add_node(label="C")

        graph.delete_node("A")
        assert graph.num_nodes() == 2

        graph.delete_node("B")
        assert graph.num_nodes() == 1

        graph.delete_node("C")
        assert graph.num_nodes() == 0


class TestDeleteNodeChaining:
    """Test method chaining with delete_node."""

    def test_delete_multiple_nodes_chained(self):
        """Test chaining multiple delete_node calls."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .delete_node("B")
            .delete_node("D")
        )

        assert graph.num_nodes() == 2
        assert graph.has_node("A")
        assert not graph.has_node("B")
        assert graph.has_node("C")
        assert not graph.has_node("D")

    def test_delete_node_mixed_with_add(self):
        """Test chaining delete_node with add_node."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .delete_node("B")
            .add_node(label="D")
        )

        assert graph.num_nodes() == 3
        assert not graph.has_node("B")
        assert graph.has_node("D")


class TestDeleteNodeWithAttributes:
    """Test deleting nodes with attributes."""

    def test_delete_node_with_attributes(self):
        """Test that node attributes are properly cleaned up."""
        graph = Graph().add_node(label="A", age=30, role="engineer").add_node(label="B", age=25)

        graph.delete_node("A")

        assert not graph.has_node("A")
        assert graph.num_nodes() == 1

    def test_delete_node_label_removed(self):
        """Test that label mapping is removed after deletion."""
        graph = Graph().add_node(label="Alice")

        assert graph.get_id_from_label("Alice") is not None

        node_id = graph.get_id_from_label("Alice")
        graph.delete_node(node_id)

        assert graph.get_id_from_label("Alice") is None


class TestDeleteNodeEdgeCases:
    """Test edge cases for delete_node."""

    def test_delete_node_from_empty_graph(self):
        """Test deleting from an empty graph raises error."""
        graph = Graph()

        with pytest.raises(KeyError):
            graph.delete_node(1)

    def test_delete_node_twice_raises_error(self):
        """Test that deleting the same node twice raises error."""
        graph = Graph().add_node(label="A")

        graph.delete_node("A")

        with pytest.raises(KeyError):
            graph.delete_node("A")

    def test_delete_node_in_complete_graph(self):
        """Test deleting node from a complete graph."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("A", "C")
        )

        assert graph.num_edges() == 3

        graph.delete_node("B")

        # Only A-C edge remains
        assert graph.num_edges() == 1
        assert graph.num_nodes() == 2
        assert graph.has_edge("A", "C")


class TestDeleteNodeWeightedGraph:
    """Test delete_node on weighted graphs."""

    def test_delete_node_weighted_edges_removed(self):
        """Test that weighted edges are properly removed."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.5)
            .add_edge("B", "C", weight=2.5)
        )

        assert graph.num_edges() == 2

        graph.delete_node("B")

        assert graph.num_edges() == 0

    def test_delete_node_preserves_other_weighted_edges(self):
        """Test that other weighted edges are preserved."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("A", "C", weight=2.0)
        )

        graph.delete_node("B")

        assert graph.num_edges() == 1
        assert graph.has_edge("A", "C")


class TestDeleteNodeDegree:
    """Test delete_node affects degree calculations."""

    def test_delete_node_updates_neighbor_degrees(self):
        """Test that neighbor degrees update after node deletion."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Before deletion
        assert graph.degree("A") == 1
        assert graph.degree("B") == 2
        assert graph.degree("C") == 1

        graph.delete_node("B")

        # After deletion
        assert graph.degree("A") == 0
        assert graph.degree("C") == 0

    def test_delete_hub_node(self):
        """Test deleting a hub node (high degree)."""
        graph = (
            Graph()
            .add_node(label="Hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("Hub", "A")
            .add_edge("Hub", "B")
            .add_edge("Hub", "C")
            .add_edge("Hub", "D")
        )

        assert graph.degree("Hub") == 4
        assert graph.num_edges() == 4

        graph.delete_node("Hub")

        assert graph.num_edges() == 0
        assert graph.num_nodes() == 4
        # All other nodes now isolated
        assert all(graph.degree(n) == 0 for n in ["A", "B", "C", "D"])


class TestDeleteNodeConnectivity:
    """Test delete_node affects connectivity."""

    def test_delete_node_disconnects_graph(self):
        """Test that deleting a bridge node disconnects the graph."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert graph.is_connected()

        graph.delete_node("B")  # B is a bridge

        assert not graph.is_connected()  # Graph now disconnected

    def test_delete_leaf_node_preserves_connectivity(self):
        """Test that deleting a leaf node may preserve connectivity."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        graph.delete_node("A")  # A is a leaf

        assert graph.is_connected()  # B-C still connected


class TestDeleteNodeCycles:
    """Test delete_node with cycles."""

    def test_delete_node_breaks_cycle(self):
        """Test deleting a node in a cycle."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        assert graph.has_cycles()

        graph.delete_node("B")

        assert not graph.has_cycles()  # Cycle broken
