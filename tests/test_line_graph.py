import pytest
from edgecase import Graph


class TestLineGraphBasic:
    """Basic tests for line graph construction."""

    def test_empty_graph(self):
        """Line graph of empty graph is empty."""
        g = Graph(directed=False)
        lg = g.line_graph()

        assert lg.count_nodes() == 0
        assert lg.count_edges() == 0

    def test_single_node_no_edges(self):
        """Line graph of graph with only nodes (no edges) is empty."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        lg = g.line_graph()

        assert lg.count_nodes() == 0
        assert lg.count_edges() == 0

    def test_single_edge(self):
        """Line graph of single edge has 1 node and 0 edges."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()

        assert lg.count_nodes() == 1
        assert lg.count_edges() == 0

    def test_two_disconnected_edges(self):
        """Line graph of two disconnected edges has 2 nodes and 0 edges."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        lg = g.line_graph()

        assert lg.count_nodes() == 2
        assert lg.count_edges() == 0  # No shared vertices


class TestLineGraphUndirected:
    """Tests for undirected line graphs."""

    def test_path_graph(self):
        """Line graph of path A-B-C has 2 nodes connected by 1 edge."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        lg = g.line_graph()

        assert lg.count_nodes() == 2
        assert lg.count_edges() == 1  # A-B and B-C share vertex B

    def test_longer_path(self):
        """Line graph of path A-B-C-D has 3 nodes and 2 edges."""
        g = Graph(directed=False)

        for label in ["A", "B", "C", "D"]:
            g.add_node(label=label)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        lg = g.line_graph()

        assert lg.count_nodes() == 3
        assert lg.count_edges() == 2  # A-B~B-C and B-C~C-D

    def test_triangle(self):
        """Line graph of triangle K3 is also K3."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        lg = g.line_graph()

        assert lg.count_nodes() == 3
        assert lg.count_edges() == 3  # Each pair shares a vertex

    def test_star_graph(self):
        """Line graph of star K_{1,n} is complete graph K_n."""
        g = Graph(directed=False)
        g.add_node(label="center")

        n = 4
        for i in range(n):
            g.add_node(label=f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        lg = g.line_graph()

        assert lg.count_nodes() == n

        # Complete graph K_n has n*(n-1)/2 edges
        expected_edges = n * (n - 1) // 2

        assert lg.count_edges() == expected_edges

    def test_star_graph_5(self):
        """Line graph of star with 5 leaves is K5."""
        g = Graph(directed=False)
        g.add_node(label="center")

        for i in range(5):
            g.add_node(label=f"L{i}")
            g.add_edge("center", f"L{i}")

        lg = g.line_graph()

        assert lg.count_nodes() == 5
        assert lg.count_edges() == 10  # C(5,2) = 10

    def test_cycle_graph(self):
        """Line graph of cycle Cn is also cycle Cn."""
        n = 5
        g = Graph(directed=False)

        for i in range(n):
            g.add_node(label=str(i))
        for i in range(n):
            g.add_edge(str(i), str((i + 1) % n))

        lg = g.line_graph()

        assert lg.count_nodes() == n
        assert lg.count_edges() == n  # Also a cycle

    def test_complete_graph_k4(self):
        """Line graph of K4 has 6 nodes and 12 edges."""
        g = Graph(directed=False)

        nodes = ["A", "B", "C", "D"]

        for node in nodes:
            g.add_node(label=node)

        # Add all edges for K4
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        assert g.count_edges() == 6  # K4 has 6 edges

        lg = g.line_graph()

        assert lg.count_nodes() == 6
        # Each edge in K4 shares vertices with 4 other edges
        # Total: 6 * 4 / 2 = 12 edges in line graph

        assert lg.count_edges() == 12


class TestLineGraphDirected:
    """Tests for directed line graphs."""

    def test_directed_chain(self):
        """Line graph of directed chain A->B->C has 2 nodes and 1 edge."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        lg = g.line_graph()

        assert lg.count_nodes() == 2
        assert lg.count_edges() == 1  # A->B connects to B->C
        assert lg.is_directed()

    def test_directed_diverging(self):
        """Line graph of A->B and A->C has 2 nodes and 0 edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("A", "C")

        lg = g.line_graph()

        assert lg.count_nodes() == 2

        # No edges: neither edge's target equals the other's source
        assert lg.count_edges() == 0

    def test_directed_converging(self):
        """Line graph of A->C and B->C has 2 nodes and 0 edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "C")
        g.add_edge("B", "C")

        lg = g.line_graph()

        assert lg.count_nodes() == 2

        # No edges: neither edge's target equals the other's source
        assert lg.count_edges() == 0

    def test_directed_cycle(self):
        """Line graph of directed cycle has same number of edges."""
        n = 4
        g = Graph(directed=True)

        for i in range(n):
            g.add_node(label=str(i))
        for i in range(n):
            g.add_edge(str(i), str((i + 1) % n))

        lg = g.line_graph()

        assert lg.count_nodes() == n
        assert lg.count_edges() == n  # Forms another directed cycle
        assert lg.is_directed()

    def test_directed_triangle(self):
        """Line graph of directed triangle A->B->C->A."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        lg = g.line_graph()

        assert lg.count_nodes() == 3
        assert lg.count_edges() == 3  # Also a directed cycle
        assert lg.is_directed()


class TestLineGraphDirectedness:
    """Tests for controlling directedness of line graph."""

    def test_inherits_undirected(self):
        """Line graph inherits undirected from source by default."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()

        assert not lg.is_directed()

    def test_inherits_directed(self):
        """Line graph inherits directed from source by default."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()

        assert lg.is_directed()

    def test_force_directed_from_undirected(self):
        """Can force directed line graph from undirected source."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph(create_using_directed=True)

        assert lg.is_directed()

    def test_force_undirected_from_directed(self):
        """Can force undirected line graph from directed source."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        lg = g.line_graph(create_using_directed=False)

        assert not lg.is_directed()


class TestLineGraphAttributes:
    """Tests for node attributes in line graph."""

    def test_original_edge_id_stored(self):
        """Line graph nodes store original edge ID."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()
        node_ids = lg.get_node_ids()

        assert len(node_ids) == 1

        node = lg.get_node(node_ids[0])
        attrs = node.attributes

        assert "original_edge_id" in attrs

        # Edge ID should be a positive integer
        assert isinstance(attrs["original_edge_id"], int)
        assert attrs["original_edge_id"] > 0

    def test_original_source_target_stored(self):
        """Line graph nodes store original source and target."""
        g = Graph(directed=False)
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge("X", "Y")

        lg = g.line_graph()
        node_ids = lg.get_node_ids()
        node = lg.get_node(node_ids[0])
        attrs = node.attributes

        assert "original_source" in attrs
        assert "original_target" in attrs
        # Source and target are stored as node IDs

    def test_original_weight_stored(self):
        """Line graph nodes store original edge weight."""
        g = Graph(directed=False, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=3.14)

        lg = g.line_graph()
        node_ids = lg.get_node_ids()
        node = lg.get_node(node_ids[0])
        attrs = node.attributes

        assert "original_weight" in attrs
        assert attrs["original_weight"] == pytest.approx(3.14)

    def test_multiple_edges_attributes(self):
        """Each line graph node has correct attributes for its edge."""
        g = Graph(directed=False, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)

        lg = g.line_graph()

        assert lg.count_nodes() == 2

        # Collect edge IDs from attributes
        edge_ids = set()
        weights = set()
        for node_id in lg.get_node_ids():
            node = lg.get_node(node_id)
            edge_ids.add(node.attributes["original_edge_id"])
            weights.add(node.attributes["original_weight"])

        # Should have 2 distinct edge IDs
        assert len(edge_ids) == 2
        assert weights == {1.0, 2.0}


class TestLineGraphWeighted:
    """Tests for weighted graph handling."""

    def test_weighted_graph_line_graph_is_weighted(self):
        """Line graph of weighted graph is also weighted."""
        g = Graph(directed=False, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=5.0)

        lg = g.line_graph()

        assert lg.is_weighted()

    def test_unweighted_graph_line_graph_is_unweighted(self):
        """Line graph of unweighted graph is also unweighted."""
        g = Graph(directed=False, weighted=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()

        assert not lg.is_weighted()


class TestLineGraphNodeLabels:
    """Tests for node labels in line graph."""

    def test_node_labels_format(self):
        """Line graph node labels are formatted as 'source-target'."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        lg = g.line_graph()
        node_ids = lg.get_node_ids()
        node = lg.get_node(node_ids[0])

        # Label should be in format "id1-id2"
        assert "-" in node.label

        parts = node.label.split("-")

        assert len(parts) == 2


class TestLineGraphSpecialCases:
    """Tests for special graph structures."""

    def test_self_loop(self):
        """Line graph handles self-loops."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")

        lg = g.line_graph()

        # Self-loop creates a node, and it's adjacent to A-B (share vertex A)
        assert lg.count_nodes() == 2
        assert lg.count_edges() == 1

    def test_multi_edges(self):
        """Line graph handles multi-edges between same nodes."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Second edge between same nodes

        lg = g.line_graph()

        # Each edge becomes a node, both share vertices A and B
        assert lg.count_nodes() == 2
        assert lg.count_edges() == 1

    def test_isolated_and_connected(self):
        """Line graph handles mix of isolated nodes and edges."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")  # Isolated
        g.add_node(label="D")  # Isolated
        g.add_edge("A", "B")

        lg = g.line_graph()

        assert lg.count_nodes() == 1  # Only the edge becomes a node
        assert lg.count_edges() == 0
