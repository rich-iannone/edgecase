"""Tests for bulk modification operations (_ws methods)."""

import pytest
import edgecase as ec


class TestDeleteNodesWS:
    """Test delete_nodes_ws() method."""

    def test_delete_nodes_basic(self):
        """Delete selected nodes and their edges."""
        G = ec.Graph()
        G.add_path_graph(5)  # Creates nodes 1,2,3,4,5

        assert G.num_nodes() == 5
        assert G.num_edges() == 4

        # Select and delete middle nodes (use integer IDs)
        G.select_nodes_by_id([2, 3])
        G.delete_nodes_ws()

        assert G.num_nodes() == 3
        assert set(G.get_node_ids(limit=None)) == {1, 4, 5}  # Returns integer IDs
        # Edges incident to deleted nodes should also be removed
        assert G.num_edges() == 1  # Only 4-5 edge remains

    def test_delete_nodes_clears_selection(self):
        """Deleting nodes clears the selection."""
        G = ec.Graph()
        G.add_path_graph(3)

        G.select_nodes_by_id([1, 2])  # Integer IDs
        assert G.count_selected_nodes() == 2

        G.delete_nodes_ws()
        assert G.count_selected_nodes() == 0

    def test_delete_nodes_isolated(self):
        """Delete isolated nodes (no edges)."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        G.select_nodes_by_id([2])  # B = node ID 2
        G.delete_nodes_ws()

        assert G.num_nodes() == 2
        assert set(G.get_node_ids(limit=None)) == {1, 3}  # A=1, C=3

    def test_delete_nodes_in_directed_graph(self):
        """Delete nodes in directed graph."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        G.select_nodes_by_id([2])  # B = node ID 2
        G.delete_nodes_ws()

        assert G.num_nodes() == 2
        assert G.num_edges() == 1  # Only C->A remains

    def test_delete_all_nodes(self):
        """Delete all nodes in graph."""
        G = ec.Graph()
        G.add_star_graph(5)

        all_nodes = G.get_node_ids(limit=None)
        G.select_nodes_by_id(all_nodes)
        G.delete_nodes_ws()

        assert G.num_nodes() == 0
        assert G.num_edges() == 0


class TestDeleteEdgesWS:
    """Test delete_edges_ws() method."""

    def test_delete_edges_basic(self):
        """Delete selected edges."""
        G = ec.Graph()
        G.add_path_graph(4)  # 3 edges

        assert G.num_edges() == 3

        # Select and delete first two edges
        G.select_edges_by_id([1, 2])
        G.delete_edges_ws()

        assert G.num_edges() == 1
        assert G.num_nodes() == 4  # Nodes remain

    def test_delete_edges_clears_selection(self):
        """Deleting edges clears edge selection."""
        G = ec.Graph()
        G.add_cycle_graph(4)

        G.select_edges_by_id([1, 2])
        assert G.count_selected_edges() == 2

        G.delete_edges_ws()
        assert G.count_selected_edges() == 0

    def test_delete_all_edges(self):
        """Delete all edges leaving nodes."""
        G = ec.Graph()
        G.add_full_graph(4)

        initial_edges = G.num_edges()
        edge_ids = G.get_edge_ids()

        G.select_edges_by_id(edge_ids)
        G.delete_edges_ws()

        assert G.num_edges() == 0
        assert G.num_nodes() == 4  # Nodes still exist


class TestSelectionCleanup:
    """Test that deletion operations properly clean up selections."""

    def test_delete_single_edge_removes_from_selection(self):
        """Deleting an edge via delete_edge_by_id removes it from selection."""
        G = ec.Graph()
        G.add_path_graph(4)  # Creates edges 1, 2, 3

        # Select all edges
        G.select_edges_by_id([1, 2, 3])
        assert G.count_selected_edges() == 3

        # Delete edge 2 directly (not via _ws)
        G.delete_edge_by_id(2)

        # Edge should be removed from selection
        assert G.count_selected_edges() == 2
        assert 2 not in G.get_selected_edges()
        assert set(G.get_selected_edges()) == {1, 3}

    def test_delete_single_node_removes_from_selection(self):
        """Deleting a node via Rust method removes it from selection."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Select multiple nodes (integer IDs)
        G.select_nodes_by_id([1, 2, 3])
        assert G.count_selected_nodes() == 3

        # Delete node 2 directly via Rust method (use integer ID)
        G._graph.delete_node(2)

        # Node should be removed from selection
        assert G.count_selected_nodes() == 2
        assert 2 not in G.get_selected_nodes()
        assert set(G.get_selected_nodes()) == {1, 3}

    def test_delete_multiple_nodes_updates_selection(self):
        """Deleting multiple nodes removes them all from selection."""
        G = ec.Graph()
        G.add_path_graph(6)

        # Select nodes 1-5 (integer IDs)
        G.select_nodes_by_id([1, 2, 3, 4, 5])
        assert G.count_selected_nodes() == 5

        # Delete nodes 2 and 4 (use integer IDs)
        G._graph.delete_node(2)
        G._graph.delete_node(4)

        # Only 1, 3, 5 should remain selected
        assert G.count_selected_nodes() == 3
        assert set(G.get_selected_nodes()) == {1, 3, 5}

    def test_delete_multiple_edges_updates_selection(self):
        """Deleting multiple edges removes them all from selection."""
        G = ec.Graph()
        G.add_path_graph(6)  # Creates edges 1-5

        # Select edges 1-5
        G.select_edges_by_id([1, 2, 3, 4, 5])
        assert G.count_selected_edges() == 5

        # Delete edges 2 and 4
        G.delete_edge_by_id(2)
        G.delete_edge_by_id(4)

        # Only 1, 3, 5 should remain selected
        assert G.count_selected_edges() == 3
        assert set(G.get_selected_edges()) == {1, 3, 5}


class TestSetNodeAttrsWS:
    """Test set_node_attrs_ws() method."""

    def test_set_single_attribute(self):
        """Set one attribute on selected nodes."""
        G = ec.Graph()
        G.add_path_graph(3)

        G.select_nodes_by_id([1, 2])  # Integer IDs
        G.set_node_attrs_ws(color="red")

        assert G.get_node(1).attributes["color"] == "red"
        assert G.get_node(2).attributes["color"] == "red"
        node3 = G.get_node(3)
        assert node3.attributes is None or "color" not in node3.attributes

    def test_set_multiple_attributes(self):
        """Set multiple attributes at once."""
        G = ec.Graph()
        G.add_star_graph(4)

        # Center node is 1 (first node created)
        G.select_nodes_by_id([1])  # Integer ID
        G.set_node_attrs_ws(importance=0.9, color="blue", hub=True)

        node = G.get_node(1)
        assert node.attributes["importance"] == 0.9
        assert node.attributes["color"] == "blue"
        assert node.attributes["hub"] is True

    def test_set_attrs_overwrites_existing(self):
        """Setting attributes overwrites existing values."""
        G = ec.Graph()
        G.add_node("A", score=10)

        G.select_nodes_by_id([1])  # A = node ID 1
        G.set_node_attrs_ws(score=20)

        assert G.get_node(1).attributes["score"] == 20

    def test_set_attrs_preserves_other_attrs(self):
        """Setting attributes preserves other existing attributes."""
        G = ec.Graph()
        G.add_node("A", name="Alice", age=30)

        G.select_nodes_by_id([1])  # A = node ID 1
        G.set_node_attrs_ws(city="NYC")

        node = G.get_node(1)
        assert node.attributes["name"] == "Alice"
        assert node.attributes["age"] == 30
        assert node.attributes["city"] == "NYC"

    def test_set_attrs_with_chaining(self):
        """Use set_node_attrs_ws in method chain."""
        G = ec.Graph()
        G.add_path_graph(5)

        result = G.select_nodes_by_degree(min_degree=2).set_node_attrs_ws(type="internal")

        assert result is G  # Returns self
        # Middle nodes have degree 2
        assert G.get_node(2).attributes["type"] == "internal"


class TestSetEdgeAttrsWS:
    """Test set_edge_attrs_ws() method."""

    def test_set_edge_weight(self):
        """Set weight on selected edges."""
        G = ec.Graph()
        G.add_path_graph(3)

        G.select_edges_by_id([1])
        G.set_edge_attrs_ws(weight=5.0)

        assert G.get_edge(1).weight == 5.0
        assert G.get_edge(2).weight is None

    def test_set_edge_regular_attributes(self):
        """Set non-weight attributes on edges."""
        G = ec.Graph()
        G.add_cycle_graph(3)

        G.select_edges_by_id([1, 2])
        G.set_edge_attrs_ws(type="strong", color="red")

        assert G.get_edge(1).attributes["type"] == "strong"
        assert G.get_edge(1).attributes["color"] == "red"
        assert G.get_edge(3).attributes is None or "type" not in G.get_edge(3).attributes

    def test_set_weight_and_attrs_together(self):
        """Set weight and other attributes simultaneously."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")

        G.select_edges_by_id([1])
        G.set_edge_attrs_ws(weight=2.5, type="link", active=True)

        edge = G.get_edge(1)
        assert edge.weight == 2.5
        assert edge.attributes["type"] == "link"
        assert edge.attributes["active"] is True


class TestMutateNodeAttrsWS:
    """Test mutate_node_attrs_ws() method."""

    def test_mutate_numeric_attribute(self):
        """Transform numeric attribute with function."""
        G = ec.Graph()
        G.add_node("A", score=10)
        G.add_node("B", score=20)
        G.add_node("C", score=30)

        G.select_nodes_by_id([1, 2])  # A=1, B=2
        G.mutate_node_attrs_ws("score", lambda x: x * 2)

        assert G.get_node(1).attributes["score"] == 20
        assert G.get_node(2).attributes["score"] == 40
        assert G.get_node(3).attributes["score"] == 30  # Unchanged

    def test_mutate_string_attribute(self):
        """Transform string attribute."""
        G = ec.Graph()
        G.add_node("A", name="alice")
        G.add_node("B", name="bob")

        G.select_nodes_by_id([1, 2])  # A=1, B=2
        G.mutate_node_attrs_ws("name", str.upper)

        assert G.get_node(1).attributes["name"] == "ALICE"
        assert G.get_node(2).attributes["name"] == "BOB"

    def test_mutate_with_complex_function(self):
        """Use complex transformation function."""
        G = ec.Graph()
        G.add_node("A", value=5)
        G.add_node("B", value=10)

        def normalize(x):
            return x / 15.0  # Normalize by sum

        G.select_nodes_by_id([1, 2])  # A=1, B=2
        G.mutate_node_attrs_ws("value", normalize)

        assert abs(G.get_node(1).attributes["value"] - 0.333) < 0.01
        assert abs(G.get_node(2).attributes["value"] - 0.667) < 0.01

    def test_mutate_missing_attribute_ignored(self):
        """Mutating missing attribute has no effect."""
        G = ec.Graph()
        G.add_node("A", score=10)
        G.add_node("B")  # No score attribute

        G.select_nodes_by_id([1, 2])  # A=1, B=2
        G.mutate_node_attrs_ws("score", lambda x: x * 2)

        assert G.get_node(1).attributes["score"] == 20
        assert G.get_node(2).attributes is None or "score" not in G.get_node(2).attributes


class TestMutateEdgeAttrsWS:
    """Test mutate_edge_attrs_ws() method."""

    def test_mutate_edge_weight(self):
        """Transform edge weight."""
        G = ec.Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.5)

        G.select_edges_by_id([1])
        G.mutate_edge_attrs_ws("weight", lambda x: x * 2)

        assert G.get_edge(1).weight == 3.0
        assert G.get_edge(2).weight == 2.5

    def test_mutate_edge_regular_attribute(self):
        """Transform non-weight edge attribute."""
        G = ec.Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B", strength=10)
        G.add_edge("B", "C", strength=20)

        G.select_edges_by_id([1, 2])
        G.mutate_edge_attrs_ws("strength", lambda x: x / 10)

        assert G.get_edge(1).attributes["strength"] == 1.0
        assert G.get_edge(2).attributes["strength"] == 2.0


class TestFullyConnectNodesWS:
    """Test fully_connect_nodes_ws() method."""

    def test_fully_connect_undirected(self):
        """Create clique in undirected graph."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        G.select_nodes_by_id([1, 2, 3])  # A=1, B=2, C=3
        G.fully_connect_nodes_ws()

        # Should have 3 edges for 3 nodes: A-B, B-C, A-C
        assert G.num_edges() == 3

    def test_fully_connect_directed(self):
        """Create clique in directed graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        G.select_nodes_by_id([1, 2, 3])  # A=1, B=2, C=3
        G.fully_connect_nodes_ws()

        # Should have 6 edges (both directions for each pair)
        assert G.num_edges() == 6

    def test_fully_connect_with_attributes(self):
        """Create clique with edge attributes."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")

        G.select_nodes_by_id([1, 2])  # A=1, B=2
        G.fully_connect_nodes_ws(type="clique", weight=1.0)

        edge = G.get_edge(1)
        assert edge.weight == 1.0
        assert edge.attributes["type"] == "clique"

    def test_fully_connect_preserves_existing_edges(self):
        """Fully connecting doesn't duplicate existing edges."""
        G = ec.Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")

        initial_edges = G.num_edges()
        G.select_nodes_by_id([1, 2, 3])  # A=1, B=2, C=3
        G.fully_connect_nodes_ws()

        # Should add A-C and B-C (A-B already exists)
        assert G.num_edges() == 3


class TestAddReverseEdgesWS:
    """Test add_reverse_edges_ws() method."""

    def test_add_reverse_edges_basic(self):
        """Add reverse edges for selected edges."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        G.select_edges_by_id([1])
        G.add_reverse_edges_ws()

        # Should have original 2 + 1 reverse = 3 edges
        assert G.num_edges() == 3
        # Check reverse edge exists (edges_as_tuples() returns (id, source, target) as IDs now)
        edges = [(s, t) for _, s, t in G.edges_as_tuples()]
        assert (2, 1) in edges  # B->A (reverse of A->B)

    def test_add_reverse_with_attributes(self):
        """Add reverse edges with attributes."""
        G = ec.Graph(directed=True)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.0)

        G.select_edges_by_id([1])
        G.add_reverse_edges_ws(weight=2.0, type="reverse")

        # Find the reverse edge (source/target are now IDs)
        reverse_edge = None
        for edge_id, s, t in G.edges_as_tuples():
            if s == 2 and t == 1:  # B=2, A=1
                reverse_edge = G.get_edge(edge_id)
                break

        assert reverse_edge is not None
        assert reverse_edge.weight == 2.0
        assert reverse_edge.attributes["type"] == "reverse"

    def test_add_reverse_undirected_error(self):
        """Adding reverse edges on undirected graph raises error."""
        G = ec.Graph(directed=False)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")

        G.select_edges_by_id([1])
        with pytest.raises(ValueError, match="directed"):
            G.add_reverse_edges_ws()

    def test_add_reverse_skips_existing(self):
        """Don't add reverse edge if it already exists."""
        G = ec.Graph(directed=True)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reverse already exists

        G.select_edges_by_id([1])
        G.add_reverse_edges_ws()

        assert G.num_edges() == 2  # No new edge added


class TestRevEdgeDirWS:
    """Test rev_edge_dir_ws() method."""

    def test_reverse_edge_direction(self):
        """Reverse direction of selected edges."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        G.select_edges_by_id([1])
        G.rev_edge_dir_ws()

        # Edge 1 should now be B->A (returns IDs: 2, 1)
        edge_tuple = G.get_node_ids_by_edge_id(1)
        assert edge_tuple == (2, 1)  # B=2, A=1

    def test_reverse_preserves_attributes(self):
        """Reversing preserves edge attributes."""
        G = ec.Graph(directed=True)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.5, type="link")

        G.select_edges_by_id([1])
        G.rev_edge_dir_ws()

        edge = G.get_edge(1)
        assert edge.weight == 1.5
        assert edge.attributes["type"] == "link"
        assert edge.source == 2  # B
        assert edge.target == 1  # A

    def test_reverse_undirected_error(self):
        """Reversing on undirected graph raises error."""
        G = ec.Graph(directed=False)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")

        G.select_edges_by_id([1])
        with pytest.raises(ValueError, match="directed"):
            G.rev_edge_dir_ws()

    def test_reverse_multiple_edges(self):
        """Reverse multiple edges at once."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")

        G.select_edges_by_id([1, 3])
        G.rev_edge_dir_ws()

        # Returns IDs: B=2, A=1, C=3, D=4
        assert G.get_node_ids_by_edge_id(1) == (2, 1)  # B->A
        assert G.get_node_ids_by_edge_id(2) == (2, 3)  # B->C (Unchanged)
        assert G.get_node_ids_by_edge_id(3) == (4, 3)  # D->C


class TestBulkModificationWorkflows:
    """Test complex workflows combining multiple _ws operations."""

    def test_workflow_clean_and_label_subgraph(self):
        """Select subgraph, label nodes, delete unwanted edges."""
        G = ec.Graph()
        G.add_full_graph(5)

        # Select high-degree nodes
        G.select_nodes_by_degree(min_degree=3).set_node_attrs_ws(hub=True, color="red")

        # Check attributes were set
        hubs = [
            n
            for n in G.get_node_ids(limit=None)
            if G.get_node(n).attributes and G.get_node(n).attributes.get("hub")
        ]
        assert len(hubs) >= 3

    def test_workflow_transform_weights(self):
        """Select edges and normalize their weights."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "C", weight=20)
        G.add_edge("C", "D", weight=30)

        # Normalize weights to [0, 1]
        G.select_edges_by_id([1, 2, 3]).mutate_edge_attrs_ws("weight", lambda x: x / 60.0)

        weights = [G.get_edge(i).weight for i in [1, 2, 3]]
        assert abs(sum(weights) - 1.0) < 0.01

    def test_workflow_bidirectional_network(self):
        """Create directed network and make selected edges bidirectional."""
        G = ec.Graph(directed=True)
        G.add_path_graph(4)  # A->B->C->D

        # Make first two edges bidirectional
        G.select_edges_by_id([1, 2]).add_reverse_edges_ws(type="bidirectional")

        assert G.num_edges() == 5  # 3 original + 2 reverse

    def test_workflow_delete_and_reconnect(self):
        """Delete nodes and reconnect remaining ones."""
        G = ec.Graph()
        G.add_cycle_graph(6)

        # Delete every other node (use integer IDs)
        G.select_nodes_by_id([1, 3, 5])
        G.delete_nodes_ws()

        # Fully connect remaining nodes
        remaining = G.get_node_ids(limit=None)
        G.select_nodes_by_id(remaining).fully_connect_nodes_ws(type="rebuilt")

        assert G.num_nodes() == 3
        assert G.num_edges() == 3  # Complete graph on 3 nodes
