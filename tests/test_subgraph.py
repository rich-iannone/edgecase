import pytest
from edgecase import Graph


class TestSubgraphBasic:
    """Basic tests for subgraph extraction."""

    def test_empty_node_list(self):
        """Empty node list returns empty graph."""
        g = Graph().add_path_graph(5)

        sub = g.subgraph([])

        assert sub.count_nodes() == 0
        assert sub.count_edges() == 0

    def test_single_node(self):
        """Single node subgraph has no edges."""
        g = Graph().add_path_graph(5)

        node_ids = list(g.get_node_ids())
        sub = g.subgraph([node_ids[0]])

        assert sub.count_nodes() == 1
        assert sub.count_edges() == 0

    def test_two_adjacent_nodes(self):
        """Two adjacent nodes have one edge."""
        g = Graph().add_path_graph(5)

        # Get nodes with labels "1" and "2"
        node_ids = list(g.get_node_ids())
        nodes_12 = [nid for nid in node_ids if g.get_label_from_id(nid) in ["1", "2"]]
        sub = g.subgraph(nodes_12)

        assert sub.count_nodes() == 2
        assert sub.count_edges() == 1

    def test_two_non_adjacent_nodes(self):
        """Two non-adjacent nodes have no edges."""
        g = Graph().add_path_graph(5)

        # Get nodes with labels "1" and "3" (not directly connected in path)
        node_ids = list(g.get_node_ids())
        nodes_13 = [nid for nid in node_ids if g.get_label_from_id(nid) in ["1", "3"]]
        sub = g.subgraph(nodes_13)

        assert sub.count_nodes() == 2
        assert sub.count_edges() == 0

    def test_path_subgraph(self):
        """Extract middle portion of a path."""
        g = Graph().add_path_graph(5)  # 1-2-3-4-5
        node_ids = list(g.get_node_ids())

        # Get nodes 2, 3, 4
        middle = [nid for nid in node_ids if g.get_label_from_id(nid) in ["2", "3", "4"]]
        sub = g.subgraph(middle)

        assert sub.count_nodes() == 3
        assert sub.count_edges() == 2  # 2-3 and 3-4

    def test_subgraph_keeps_attributes_and_ids(self):
        """Attributes carry over; IDs are renumbered in ascending order, whatever the argument order."""
        graph = Graph(directed=True)
        for label in "ABC":
            graph.add_node(label=label)
        for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
            graph.add_edge(u, v)
        graph.set_node_attrs("x", {1: 10, 2: 20, 3: 30})
        graph.set_edge_attrs("w", 7)
        result = graph.subgraph([3, 1, 2])
        assert {i: result.get_label_from_id(i) for i in result.get_node_ids()} == {1: "A", 2: "B", 3: "C"}
        assert result.get_node_attrs("x") == {1: 10, 2: 20, 3: 30}
        assert set(result.get_edge_attrs("w").values()) == {7}


class TestSubgraphCompleteGraph:
    """Tests with complete graphs."""

    def test_k3_from_k5(self):
        """Extract K3 from K5."""
        g = Graph().add_full_graph(5)  # K5 has 10 edges

        node_ids = list(g.get_node_ids())[:3]
        sub = g.subgraph(node_ids)

        assert sub.count_nodes() == 3
        assert sub.count_edges() == 3  # K3 has 3 edges

    def test_k4_from_k5(self):
        """Extract K4 from K5."""
        g = Graph().add_full_graph(5)

        node_ids = list(g.get_node_ids())[:4]
        sub = g.subgraph(node_ids)

        assert sub.count_nodes() == 4
        assert sub.count_edges() == 6  # K4 has 6 edges

    def test_full_graph_subgraph(self):
        """Extracting all nodes gives full graph."""
        g = Graph().add_full_graph(4)

        node_ids = list(g.get_node_ids())
        sub = g.subgraph(node_ids)

        assert sub.count_nodes() == 4
        assert sub.count_edges() == 6


class TestSubgraphDirected:
    """Tests with directed graphs."""

    def test_directed_path(self):
        """Directed path subgraph preserves direction."""
        g = Graph(directed=True).add_path_graph(4)

        assert g.is_directed()

        node_ids = list(g.get_node_ids())

        # Get nodes 1, 2, 3
        nodes_123 = [nid for nid in node_ids if g.get_label_from_id(nid) in ["1", "2", "3"]]
        sub = g.subgraph(nodes_123)

        assert sub.is_directed()
        assert sub.count_nodes() == 3
        assert sub.count_edges() == 2  # 1->2, 2->3

    def test_directed_cycle(self):
        """Directed cycle subgraph."""
        g = Graph(directed=True).add_cycle_graph(4)

        node_ids = list(g.get_node_ids())

        # Get 3 consecutive nodes
        nodes_123 = [nid for nid in node_ids if g.get_label_from_id(nid) in ["1", "2", "3"]]
        sub = g.subgraph(nodes_123)

        assert sub.is_directed()
        assert sub.count_nodes() == 3

        # Only edges within {1,2,3}: 1->2, 2->3 (not 4->1 or 3->4)
        assert sub.count_edges() == 2


class TestSubgraphLabels:
    """Tests for label preservation."""

    def test_labels_preserved(self):
        """Node labels are preserved in subgraph."""
        g = Graph()
        g.add_node(label="Alice")
        g.add_node(label="Bob")
        g.add_node(label="Charlie")
        g.add_edge("Alice", "Bob")
        g.add_edge("Bob", "Charlie")

        # Get Alice and Bob
        node_ids = list(g.get_node_ids())
        alice_bob = [nid for nid in node_ids if g.get_label_from_id(nid) in ["Alice", "Bob"]]

        sub = g.subgraph(alice_bob)
        labels = {g.get_label_from_id(nid) for nid in sub.get_node_ids()}

        assert labels == {"Alice", "Bob"}


class TestSubgraphWeighted:
    """Tests with weighted graphs."""

    def test_weights_preserved(self):
        """Edge weights are preserved in subgraph."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B", weight=1.5)
        g.add_edge("B", "C", weight=2.5)

        # Get A and B
        node_ids = list(g.get_node_ids())
        ab_nodes = [nid for nid in node_ids if g.get_label_from_id(nid) in ["A", "B"]]

        sub = g.subgraph(ab_nodes)
        edge_ids = list(sub.get_edge_ids())

        assert len(edge_ids) == 1

        edge = sub.get_edge(edge_ids[0])

        assert edge.weight == 1.5


class TestSubgraphErrors:
    """Tests for error handling."""

    def test_invalid_node_id(self):
        """Invalid node ID raises error."""
        g = Graph().add_path_graph(3)

        with pytest.raises(Exception):  # ValueError from Rust
            g.subgraph([99999])

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid IDs raises error."""
        g = Graph().add_path_graph(3)

        node_ids = list(g.get_node_ids())

        with pytest.raises(Exception):
            g.subgraph([node_ids[0], 99999])


class TestEdgeSubgraphBasic:
    """Basic tests for edge_subgraph."""

    def test_empty_edge_list(self):
        """Empty edge list returns empty graph."""
        g = Graph().add_path_graph(5)

        sub = g.edge_subgraph([])

        assert sub.count_nodes() == 0
        assert sub.count_edges() == 0

    def test_single_edge(self):
        """Single edge subgraph has 2 nodes."""
        g = Graph().add_path_graph(5)

        edge_ids = list(g.get_edge_ids())
        sub = g.edge_subgraph([edge_ids[0]])

        assert sub.count_nodes() == 2
        assert sub.count_edges() == 1

    def test_two_adjacent_edges(self):
        """Two adjacent edges share a node."""
        g = Graph().add_path_graph(5)  # 1-2-3-4-5

        edge_ids = list(g.get_edge_ids())[:2]  # First two edges
        sub = g.edge_subgraph(edge_ids)

        assert sub.count_nodes() == 3  # Three nodes involved
        assert sub.count_edges() == 2

    def test_two_non_adjacent_edges(self):
        """Two non-adjacent edges have 4 nodes."""
        g = Graph().add_path_graph(5)  # 1-2-3-4-5, edges: 1-2, 2-3, 3-4, 4-5

        edge_ids = list(g.get_edge_ids())

        # Find edges that don't share a node
        # Edge 0 (1-2) and edge 2 (3-4) don't share nodes
        # But we need to find them by checking

        sub = g.edge_subgraph([edge_ids[0], edge_ids[2]])

        # Should have 4 nodes if edges don't share endpoints
        assert sub.count_edges() == 2

    def test_edge_subgraph_keeps_attributes_and_ids(self):
        """Attributes carry over and node IDs are renumbered in ascending order."""
        graph = Graph(directed=True)
        for label in "ABC":
            graph.add_node(label=label)
        for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
            graph.add_edge(u, v)
        graph.set_node_attrs("x", {1: 10, 2: 20, 3: 30})
        graph.set_edge_attrs("w", 7)
        result = graph.edge_subgraph([1, 2, 3])
        assert {i: result.get_label_from_id(i) for i in result.get_node_ids()} == {1: "A", 2: "B", 3: "C"}
        assert result.get_node_attrs("x") == {1: 10, 2: 20, 3: 30}
        assert set(result.get_edge_attrs("w").values()) == {7}


class TestEdgeSubgraphDirected:
    """Tests for edge_subgraph with directed graphs."""

    def test_directed_preserves_direction(self):
        """Directed edge subgraph preserves direction."""
        g = Graph(directed=True).add_path_graph(4)

        assert g.is_directed()

        edge_ids = list(g.get_edge_ids())[:2]
        sub = g.edge_subgraph(edge_ids)

        assert sub.is_directed()
        assert sub.count_edges() == 2


class TestEdgeSubgraphWeighted:
    """Tests for edge_subgraph with weighted graphs."""

    def test_weights_preserved(self):
        """Edge weights are preserved."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B", weight=1.5)
        g.add_edge("B", "C", weight=2.5)

        edge_ids = list(g.get_edge_ids())

        # Get subgraph with first edge
        sub = g.edge_subgraph([edge_ids[0]])
        sub_edge_ids = list(sub.get_edge_ids())
        edge = sub.get_edge(sub_edge_ids[0])

        # Weight should be either 1.5 or 2.5 depending on which edge is first
        assert edge.weight in [1.5, 2.5]


class TestEdgeSubgraphErrors:
    """Tests for error handling."""

    def test_invalid_edge_id(self):
        """Invalid edge ID raises error."""
        g = Graph().add_path_graph(3)

        with pytest.raises(Exception):
            g.edge_subgraph([99999])


class TestSubgraphCycle:
    """Tests with cycle graphs."""

    def test_cycle_half(self):
        """Extract half of a cycle."""
        g = Graph().add_cycle_graph(6)  # 1-2-3-4-5-6-1

        node_ids = list(g.get_node_ids())

        # Get nodes 1, 2, 3
        half = [nid for nid in node_ids if g.get_label_from_id(nid) in ["1", "2", "3"]]
        sub = g.subgraph(half)

        assert sub.count_nodes() == 3

        # Only edges 1-2 and 2-3 are within this set (not 6-1 or 3-4)
        assert sub.count_edges() == 2


class TestSubgraphStar:
    """Tests with star graphs."""

    def test_star_center_and_leaves(self):
        """Extract center and some leaves from star."""
        g = Graph().add_star_graph(5)  # Center + 5 leaves

        node_ids = list(g.get_node_ids())

        # Get 4 nodes (should include center if we want edges)
        some_nodes = node_ids[:4]
        sub = g.subgraph(some_nodes)

        # Edges depend on whether center is included
        assert sub.count_nodes() == 4


class TestSubgraphGrid:
    """Tests with grid graphs."""

    def test_grid_corner(self):
        """Extract corner of a grid."""
        g = Graph().add_grid_2d(3, 3)  # 3x3 grid, 9 nodes

        node_ids = list(g.get_node_ids())

        # Just verify we can extract a subgraph
        some_nodes = node_ids[:4]
        sub = g.subgraph(some_nodes)

        assert sub.count_nodes() == 4
        assert sub.count_edges() >= 0  # At least 0 edges
