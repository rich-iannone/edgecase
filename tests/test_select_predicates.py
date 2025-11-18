"""
Tests for attribute-based selection methods.
"""

import pytest
from edgecase import Graph
from edgecase.graph import Node, Edge


class TestSelectNodesByPredicate:
    """Test select_nodes() with callable predicates."""

    def test_select_nodes_by_id_pattern(self):
        """Select nodes matching label pattern."""
        G = Graph()
        G.add_node("user_1", type="user")
        G.add_node("user_2", type="user")
        G.add_node("admin_1", type="admin")
        G.add_node("user_3", type="user")

        # Select nodes whose label starts with "user_"
        G.select_nodes(lambda n: n.label and n.label.startswith("user_"))

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        # Selected nodes are IDs 1, 2, 4 (user_1, user_2, user_3)
        assert 1 in selected  # user_1
        assert 2 in selected  # user_2
        assert 4 in selected  # user_3
        assert 3 not in selected  # admin_1

    def test_select_nodes_with_attributes(self):
        """Select nodes based on attribute values."""
        G = Graph()
        G.add_node("A", importance=0.9, active=True)
        G.add_node("B", importance=0.3, active=True)
        G.add_node("C", importance=0.7, active=False)
        G.add_node("D", importance=0.95, active=True)

        # Select nodes with high importance
        G.select_nodes(lambda n: n.attributes and n.attributes.get("importance", 0) > 0.5)

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        assert 1 in selected  # A
        assert 3 in selected  # C
        assert 4 in selected  # D
        assert 2 not in selected  # B

    def test_select_nodes_complex_predicate(self):
        """Select nodes with complex multi-condition predicate."""
        G = Graph()
        G.add_node("A", score=85, category="premium", active=True)
        G.add_node("B", score=45, category="basic", active=True)
        G.add_node("C", score=90, category="premium", active=False)
        G.add_node("D", score=70, category="premium", active=True)

        # Select premium active nodes with score > 60
        def premium_active_high_score(node):
            if not node.attributes:
                return False
            attrs = node.attributes
            return (
                attrs.get("category") == "premium"
                and attrs.get("active") == True
                and attrs.get("score", 0) > 60
            )

        G.select_nodes(premium_active_high_score)

        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 1 in selected  # A
        assert 4 in selected  # D

    def test_select_nodes_without_attributes(self):
        """Select nodes without any attributes."""
        G = Graph()
        G.add_node("A", type="user")
        G.add_node("B")  # No attributes
        G.add_node("C", type="admin")
        G.add_node("D")  # No attributes

        # Select nodes with no attributes
        G.select_nodes(lambda n: n.attributes is None)

        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 2 in selected  # B
        assert 4 in selected  # D

    def test_select_nodes_chaining_with_predicate(self):
        """Chain predicate-based selection with other selections."""
        G = Graph()
        G.add_node("A", priority=1)
        G.add_node("B", priority=2)
        G.add_node("C", priority=3)
        G.add_node("D", priority=4)

        # First select by predicate, then add more by ID
        G.select_nodes(lambda n: n.attributes and n.attributes.get("priority", 0) > 2)
        first_count = G.count_selected_nodes()

        G.select_nodes_by_id([1])  # Select node ID 1 (A)

        selected = G.get_selected_nodes()
        assert len(selected) == 3
        assert 1 in selected  # A
        assert 3 in selected  # C
        assert 4 in selected  # D


class TestSelectEdgesByPredicate:
    """Test select_edges() with callable predicates."""

    def test_select_edges_by_weight(self):
        """Select edges based on weight."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=0.3)
        G.add_edge("B", "C", weight=0.8)
        G.add_edge("C", "D", weight=0.5)
        G.add_edge("D", "E", weight=0.9)

        # Select edges with weight > 0.5
        G.select_edges(lambda e: e.weight and e.weight > 0.5)

        selected = G.get_selected_edges()
        assert len(selected) == 2

        # Verify correct edges selected
        edges = G.get_edges_as_tuples()
        selected_edges = [e for e in edges if e[0] in selected]
        weights = [G.get_edge(e[0]).weight for e in selected_edges]
        assert all(w > 0.5 for w in weights)

    def test_select_edges_by_relationship_type(self):
        """Select edges based on relationship attribute."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", rel="friend")
        G.add_edge("B", "C", rel="colleague")
        G.add_edge("C", "D", rel="friend")
        G.add_edge("D", "E", rel="family")

        # Select friendship edges
        G.select_edges(lambda e: e.attributes and e.attributes.get("rel") == "friend")

        selected = G.get_selected_edges()
        assert len(selected) == 2

    def test_select_edges_by_source_target_pattern(self):
        """Select edges based on source/target node label patterns."""
        G = Graph()
        for label in ["server_1", "server_2", "client_1", "client_2", "client_3"]:
            G.add_node(label)
        G.add_edge("server_1", "client_1")
        G.add_edge("server_1", "client_2")
        G.add_edge("client_1", "client_2")
        G.add_edge("server_2", "client_3")

        # Pre-compute node mappings to avoid borrowing issues in predicate
        node_labels = {
            node.id: node.label
            for node_id in G.get_node_ids(limit=None)
            if (node := G.get_node(node_id))
        }

        # Select edges from servers to clients - check label patterns
        def is_server_to_client(edge):
            source_label = node_labels.get(edge.source, "")
            target_label = node_labels.get(edge.target, "")
            return source_label.startswith("server_") and target_label.startswith("client_")

        G.select_edges(is_server_to_client)

        selected = G.get_selected_edges()
        assert len(selected) == 3

    def test_select_edges_complex_predicate(self):
        """Select edges with complex multi-condition predicate."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=0.8, confidence=0.9, verified=True)
        G.add_edge("B", "C", weight=0.3, confidence=0.7, verified=False)
        G.add_edge("C", "D", weight=0.9, confidence=0.85, verified=True)
        G.add_edge("D", "E", weight=0.6, confidence=0.5, verified=True)

        # Select high-quality verified edges
        def high_quality_verified(edge):
            if not edge.attributes:
                return False
            attrs = edge.attributes
            return (
                attrs.get("verified") == True
                and (edge.weight or 0) > 0.5
                and attrs.get("confidence", 0) > 0.8
            )

        G.select_edges(high_quality_verified)

        selected = G.get_selected_edges()
        assert len(selected) == 2

    def test_select_edges_without_weight(self):
        """Select unweighted edges."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C")  # No weight
        G.add_edge("C", "D", weight=2.0)
        G.add_edge("D", "E")  # No weight

        # Select edges without weight
        G.select_edges(lambda e: e.weight is None)

        selected = G.get_selected_edges()
        assert len(selected) == 2

    def test_select_edges_chaining_with_predicate(self):
        """Chain predicate-based selection with other selections."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=0.8)
        G.add_edge("B", "C", weight=0.3)
        G.add_edge("C", "D", weight=0.9)
        G.add_edge("D", "E", weight=0.5)

        # First select by predicate
        G.select_edges(lambda e: e.weight and e.weight > 0.5)

        # Then select additional edges by node (use integer ID)
        G.select_edges_by_node_id([2])  # B = node ID 2

        selected = G.get_selected_edges()
        # Should have high-weight edges plus edges connected to B
        assert len(selected) == 3


class TestGetNodeAndEdge:
    """Test get_node() and get_edge() helper methods."""

    def test_get_node_exists(self):
        """Get an existing node."""
        G = Graph()
        G.add_node("A", type="user", score=100)

        node = G.get_node("A")  # Can look up by label
        assert node is not None
        assert node.id == 1  # ID is integer
        assert node.label == "A"  # Label is string
        assert node.attributes is not None
        assert node.attributes.get("type") == "user"
        assert node.attributes.get("score") == 100

    def test_get_node_not_exists(self):
        """Get a non-existent node returns None."""
        G = Graph()
        G.add_node("A")

        node = G.get_node(999)  # Try with non-existent node ID
        assert node is None

    def test_get_edge_exists(self):
        """Get an existing edge."""
        G = Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=0.5, rel="friend")

        edge_ids = G.get_edge_ids()
        edge = G.get_edge(edge_ids[0])

        assert edge is not None
        assert edge.id == edge_ids[0]
        assert edge.source == 1  # Node ID for A
        assert edge.target == 2  # Node ID for B
        assert edge.weight == 0.5
        assert edge.attributes.get("rel") == "friend"

    def test_get_edge_not_exists(self):
        """Get a non-existent edge returns None."""
        G = Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")

        edge = G.get_edge(999)
        assert edge is None


class TestCombinedPredicateSelection:
    """Test combining predicate-based and other selection methods."""

    def test_select_nodes_then_filter_by_degree(self):
        """Select nodes by attributes, then filter by degree."""
        G = Graph()
        # Build a network
        G.add_node("A", category="premium")
        G.add_node("B", category="premium")
        G.add_node("C", category="basic")
        G.add_node("D", category="premium")

        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")  # A has degree 3
        G.add_edge("B", "C")  # B has degree 2
        G.add_edge("D", "C")  # D has degree 2

        # Select premium nodes
        G.select_nodes(lambda n: n.attributes and n.attributes.get("category") == "premium")
        premium_nodes = set(G.get_selected_nodes())

        # Now select by degree
        G.clear_selection()
        G.select_nodes_by_degree(min_degree=2)
        high_degree = set(G.get_selected_nodes())

        # The intersection would be premium nodes with high degree
        premium_high_degree = premium_nodes & high_degree
        assert len(premium_high_degree) == 3  # A, B, D

    def test_select_edges_by_attributes_then_endpoints(self):
        """Select edges by attributes, then filter by connected nodes."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B", importance=0.9)
        G.add_edge("B", "C", importance=0.3)
        G.add_edge("C", "D", importance=0.8)
        G.add_edge("D", "E", importance=0.5)

        # Select important edges
        G.select_edges(lambda e: e.attributes and e.attributes.get("importance", 0) > 0.5)

        selected = G.get_selected_edges()
        assert len(selected) == 2

        # Verify they involve the right nodes (using integer IDs)
        edges = G.get_edges_as_tuples()
        selected_edge_info = [e for e in edges if e[0] in selected]
        involved_nodes = set()
        for e in selected_edge_info:
            involved_nodes.add(e[1])  # source node ID
            involved_nodes.add(e[2])  # target node ID

        # A=1, B=2, C=3, D=4
        assert 1 in involved_nodes  # A
        assert 2 in involved_nodes  # B
        assert 3 in involved_nodes  # C
        assert 4 in involved_nodes  # D

    def test_workflow_find_important_subgraph(self):
        """Real workflow: Find important subgraph based on attributes."""
        G = Graph(directed=True)

        # Build a network
        G.add_node("User1", reputation=100)
        G.add_node("User2", reputation=50)
        G.add_node("User3", reputation=80)
        G.add_node("User4", reputation=30)

        G.add_edge("User1", "User2", interaction_count=20)
        G.add_edge("User2", "User3", interaction_count=5)
        G.add_edge("User1", "User3", interaction_count=15)
        G.add_edge("User3", "User4", interaction_count=3)

        # Step 1: Find high-reputation users
        G.select_nodes(lambda n: n.attributes and n.attributes.get("reputation", 0) >= 80)
        high_rep_users = G.get_selected_nodes()
        assert len(high_rep_users) == 2  # User1, User3

        # Step 2: Find strong interactions between them
        G.select_edges(
            lambda e: (
                e.source in high_rep_users
                and e.target in high_rep_users
                and e.attributes
                and e.attributes.get("interaction_count", 0) >= 10
            )
        )

        strong_interactions = G.get_selected_edges()
        assert len(strong_interactions) == 1  # User1 -> User3


class TestPredicateErrorHandling:
    """Test error handling in predicate-based selection."""

    def test_predicate_exception_propagates(self):
        """Exceptions in predicates are propagated."""
        G = Graph()
        G.add_node("A", value=10)
        G.add_node("B", value="invalid")

        # Predicate that will fail on string value
        with pytest.raises(Exception):
            G.select_nodes(lambda n: n.attributes.get("value") > 5)

    def test_predicate_must_return_boolean(self):
        """Predicate must return strict boolean (not truthy value)."""
        G = Graph()
        G.add_node("A", score=10)

        # Rust requires strict boolean, not truthy value
        with pytest.raises(TypeError, match="cannot be converted to 'PyBool'"):
            G.select_nodes(lambda n: n.attributes.get("score"))

        # Must use explicit boolean comparison
        G.select_nodes(lambda n: n.attributes.get("score") is not None)
        selected = G.get_selected_nodes()
        assert 1 in selected  # A = node ID 1

    def test_empty_graph_predicate_selection(self):
        """Predicate selection on empty graph."""
        G = Graph()

        # Should not crash
        G.select_nodes(lambda n: True)
        assert G.count_selected_nodes() == 0

        G.select_edges(lambda e: True)
        assert G.count_selected_edges() == 0
