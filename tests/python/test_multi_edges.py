"""Tests for multi-edge graph functionality (edge IDs)."""

import pytest
import polars as pl
import edgecase as ec


class TestMultiEdgeGraphs:
    """Test that graphs now support multiple edges between same node pairs."""

    def test_basic_multi_edges(self):
        """Test adding multiple edges between same nodes."""
        g = ec.Graph(directed=True)
        g.add_node("A")
        g.add_node("B")

        # Add first edge
        g.add_edge("A", "B", rel="follows")
        assert g.num_edges() == 1

        # Add second edge (same nodes, different relationship)
        g.add_edge("A", "B", rel="likes")
        assert g.num_edges() == 2

        # Check edge IDs are different
        edge_ids = g.edge_ids()
        assert len(edge_ids) == 2
        assert edge_ids[0] != edge_ids[1]

    def test_multi_edges_auto_increment_ids(self):
        """Test that edge IDs auto-increment correctly."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")

        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "B")  # Third A->B edge

        edge_ids = g.edge_ids()
        assert edge_ids == [1, 2, 3, 4]

        # Check edges returns (id, source_id, target_id) - IDs not labels
        edges = g.edges()
        assert len(edges) == 4
        # Nodes get auto-assigned integer IDs (A=1, B=2, C=3)
        assert edges[0] == (1, 1, 2)  # edge 1: A->B
        assert edges[1] == (2, 1, 2)  # edge 2: A->B
        assert edges[2] == (3, 2, 3)  # edge 3: B->C
        assert edges[3] == (4, 1, 2)  # edge 4: A->B

    def test_delete_edge_by_id(self):
        """Test deleting specific edges by ID."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")

        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)
        g.add_edge("A", "B", weight=3.0)

        assert g.num_edges() == 3
        edge_ids = g.edge_ids()

        # Delete middle edge
        g.delete_edge_by_id(edge_ids[1])
        assert g.num_edges() == 2

        # Remaining edge IDs should be first and third
        remaining = g.edge_ids()
        assert edge_ids[0] in remaining
        assert edge_ids[2] in remaining
        assert edge_ids[1] not in remaining

    def test_get_node_ids_by_edge_id(self):
        """Test retrieving edge node IDs by edge ID."""
        g = ec.Graph()
        g.add_node("Alice")
        g.add_node("Bob")
        g.add_node("Charlie")

        g.add_edge("Alice", "Bob")
        g.add_edge("Bob", "Charlie")

        edge_ids = g.edge_ids()

        # Get first edge - returns (source_id, target_id)
        edge1 = g.get_node_ids_by_edge_id(edge_ids[0])
        # Alice=1, Bob=2, Charlie=3
        assert edge1 == (1, 2)

        # Get second edge
        edge2 = g.get_node_ids_by_edge_id(edge_ids[1])
        assert edge2 == (2, 3)

        # Non-existent ID
        assert g.get_node_ids_by_edge_id(999) is None

    def test_has_edge_id(self):
        """Test checking if edge ID exists."""
        g = ec.Graph()
        g.add_node("X")
        g.add_node("Y")

        g.add_edge("X", "Y")
        edge_ids = g.edge_ids()

        assert g.has_edge_id(edge_ids[0])
        assert not g.has_edge_id(999)
        assert not g.has_edge_id(0)

    def test_multi_edges_with_attributes(self):
        """Test multi-edges with different attributes."""
        g = ec.Graph(directed=True)
        g.add_node("User1")
        g.add_node("User2")

        # Multiple relationships with different attributes
        g.add_edge("User1", "User2", rel="follows", since="2020")
        g.add_edge("User1", "User2", rel="likes", count=42)
        g.add_edge("User1", "User2", rel="mentions", frequency="high")

        assert g.num_edges() == 3
        assert len(g.edge_ids()) == 3

    def test_create_graph_with_edge_ids(self):
        """Test creating graph from DataFrames with explicit edge IDs."""
        nodes = pl.DataFrame({"id": ["A", "B", "C"], "type": ["person", "person", "person"]})

        edges = pl.DataFrame(
            {
                "id": [10, 20, 30],
                "from": ["A", "B", "A"],
                "to": ["B", "C", "B"],
                "rel": ["follows", "follows", "likes"],
            }
        )

        g = ec.Graph.from_dataframes(nodes_df=nodes, edges_df=edges, directed=True)

        # Check edge IDs were preserved
        edge_ids = g.edge_ids()
        assert 10 in edge_ids
        assert 20 in edge_ids
        assert 30 in edge_ids

    def test_create_graph_without_edge_ids(self):
        """Test creating graph from DataFrames without edge IDs (auto-assigned)."""
        nodes = pl.DataFrame({"id": ["A", "B", "C"]})

        edges = pl.DataFrame(
            {
                "from": ["A", "B", "A"],
                "to": ["B", "C", "B"],
            }
        )

        g = ec.Graph.from_dataframes(nodes_df=nodes, edges_df=edges)

        # Edge IDs should be auto-assigned starting from 1
        edge_ids = g.edge_ids()
        assert edge_ids == [1, 2, 3]

    def test_social_network_multi_edges(self):
        """Real-world example: social network with multiple relationship types."""
        g = ec.Graph(directed=True)

        # Add users
        for user in ["Alice", "Bob", "Charlie"]:
            g.add_node(user, type="user")

        # Alice interacts with Bob in multiple ways
        g.add_edge("Alice", "Bob", rel="follows", timestamp="2024-01-01")
        g.add_edge("Alice", "Bob", rel="messages", count=15)
        g.add_edge("Alice", "Bob", rel="tagged_in", posts=3)

        # Bob follows Charlie
        g.add_edge("Bob", "Charlie", rel="follows", timestamp="2024-01-15")

        # Charlie follows Alice back
        g.add_edge("Charlie", "Alice", rel="follows", timestamp="2024-02-01")

        assert g.num_nodes() == 3
        assert g.num_edges() == 5

        # Get all edge IDs
        edge_ids = g.edge_ids()
        assert len(edge_ids) == 5
        assert len(set(edge_ids)) == 5  # All unique

    def test_transaction_network_multi_edges(self):
        """Real-world example: financial transactions (multiple between same parties)."""
        g = ec.Graph(directed=True)

        accounts = ["Account_A", "Account_B", "Account_C"]
        for acc in accounts:
            g.add_node(acc, type="account")

        # Multiple transactions from A to B
        g.add_edge("Account_A", "Account_B", amount=100.0, date="2024-01-01")
        g.add_edge("Account_A", "Account_B", amount=250.0, date="2024-01-05")
        g.add_edge("Account_A", "Account_B", amount=75.0, date="2024-01-10")

        # B to C
        g.add_edge("Account_B", "Account_C", amount=200.0, date="2024-01-12")

        # C back to A (potential cycle)
        g.add_edge("Account_C", "Account_A", amount=300.0, date="2024-01-15")

        assert g.num_edges() == 5

        # Each transaction has unique ID
        assert len(set(g.edge_ids())) == 5

    def test_delete_specific_multi_edge(self):
        """Test deleting one specific edge among multiple edges."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")

        g.add_edge("A", "B", label="edge1")
        g.add_edge("A", "B", label="edge2")
        g.add_edge("A", "B", label="edge3")

        edge_ids = g.edge_ids()
        assert len(edge_ids) == 3

        # Delete the second edge
        middle_id = edge_ids[1]
        g.delete_edge_by_id(middle_id)

        # Should have 2 edges left
        assert g.num_edges() == 2
        remaining_ids = g.edge_ids()
        assert edge_ids[0] in remaining_ids
        assert edge_ids[2] in remaining_ids
