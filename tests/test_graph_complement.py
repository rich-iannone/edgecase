from edgecase import Graph


class TestComplementBasic:
    """Test basic complement operations."""

    def test_complement_empty_graph(self):
        """An empty graph complement is also empty."""
        g = Graph()
        comp = g.complement()

        assert comp.count_nodes() == 0
        assert comp.count_edges() == 0

    def test_complement_single_node(self):
        """A single node has no edges in its complement (no self-loops)."""
        g = Graph()
        g.add_node(label="A")
        comp = g.complement()

        assert comp.count_nodes() == 1
        assert comp.count_edges() == 0

    def test_complement_two_nodes_no_edge(self):
        """Two unconnected nodes: complement has one edge."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")

        comp = g.complement()

        assert comp.count_nodes() == 2
        assert comp.count_edges() == 1
        assert comp.has_edge("A", "B")

    def test_complement_two_nodes_with_edge(self):
        """Two connected nodes: complement has no edges."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        comp = g.complement()

        assert comp.count_nodes() == 2
        assert comp.count_edges() == 0
        assert not comp.has_edge("A", "B")

    def test_complement_triangle_to_empty(self):
        """Complete triangle: complement has no edges."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")

        comp = g.complement()

        assert comp.count_nodes() == 3
        assert comp.count_edges() == 0

    def test_complement_path_graph(self):
        """Path graph A-B-C: complement has A-C."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        comp = g.complement()

        assert comp.count_nodes() == 3
        assert comp.count_edges() == 1
        assert comp.has_edge("A", "C")
        assert not comp.has_edge("A", "B")
        assert not comp.has_edge("B", "C")


class TestComplementDirected:
    """Test complement on directed graphs."""

    def test_complement_directed_two_nodes(self):
        """Directed: A->B complement has B->A."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        comp = g.complement()

        assert comp.count_nodes() == 2
        assert comp.count_edges() == 1
        assert comp.has_edge("B", "A")
        assert not comp.has_edge("A", "B")

    def test_complement_directed_bidirectional(self):
        """Directed: A<->B (both edges) has no complement edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        comp = g.complement()

        assert comp.count_nodes() == 2
        assert comp.count_edges() == 0

    def test_complement_directed_three_nodes(self):
        """Directed: A->B, B->C; complement has all other directed edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        comp = g.complement()

        assert comp.count_nodes() == 3

        # Total possible: 3*2 = 6 directed edges (no self-loops)
        # Original: 2 edges
        # Complement: 4 edges
        assert comp.count_edges() == 4
        assert comp.has_edge("A", "C")
        assert comp.has_edge("B", "A")
        assert comp.has_edge("C", "A")
        assert comp.has_edge("C", "B")
        assert not comp.has_edge("A", "B")
        assert not comp.has_edge("B", "C")


class TestComplementAttributes:
    """Test attribute preservation in complement."""

    def test_complement_preserves_node_attributes(self):
        """Node attributes are preserved in complement."""
        g = Graph()
        g.add_node(label="A", color="red", value=10)
        g.add_node(label="B", color="blue", value=20)
        g.add_edge("A", "B")

        comp = g.complement()
        node_a = comp.get_node("A")

        assert node_a is not None
        assert node_a.attributes["color"] == "red"
        assert node_a.attributes["value"] == 10

    def test_complement_new_edges_no_attributes(self):
        """New edges in complement have no attributes."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B", label="original", type="strong")

        comp = g.complement()

        # Complement should have edges A-C and B-C (no attributes)
        edge_ids = comp.find_edge_ids("A", "C")

        assert len(edge_ids) > 0

        edge = comp.get_edge(edge_ids[0])

        # New edges have no custom attributes (attributes can be None or empty)
        assert (
            edge.attributes is None or len(edge.attributes) == 0 or "label" not in edge.attributes
        )


class TestComplementProperties:
    """Test graph properties of complements."""

    def test_complement_is_same_type(self):
        """Complement preserves directed/undirected type."""
        g_undirected = Graph(directed=False)
        g_undirected.add_node(label="A")
        g_undirected.add_node(label="B")
        comp_undirected = g_undirected.complement()

        assert not comp_undirected.is_directed()

        g_directed = Graph(directed=True)
        g_directed.add_node(label="A")
        g_directed.add_node(label="B")
        comp_directed = g_directed.complement()

        assert comp_directed.is_directed()

    def test_complement_is_same_weighted(self):
        """Complement preserves weighted property."""
        g_weighted = Graph(weighted=True)
        g_weighted.add_node(label="A")
        g_weighted.add_node(label="B")
        comp_weighted = g_weighted.complement()

        # Check edges have weight
        if comp_weighted.count_edges() > 0:
            edge_ids = comp_weighted.find_edge_ids("A", "B")
            edge = comp_weighted.get_edge(edge_ids[0])
            assert edge.weight is not None

    def test_complement_no_self_loops(self):
        """Complement never contains self-loops."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        # No edges in original graph

        comp = g.complement()

        # Should have A-B, A-C, B-C but no self-loops
        loop_counts = comp.count_loops()

        assert len(loop_counts) == 0  # No nodes should have self-loops


class TestComplementDoubleComplement:
    """Test taht double complement returns to original."""

    def test_double_complement_undirected(self):
        """Complement of complement equals original (undirected)."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")

        comp = g.complement()
        comp_comp = comp.complement()

        assert comp_comp.count_nodes() == g.count_nodes()
        assert comp_comp.count_edges() == g.count_edges()
        assert comp_comp.has_edge("A", "B")
        assert comp_comp.has_edge("B", "C") == g.has_edge("B", "C")
        assert comp_comp.has_edge("A", "C") == g.has_edge("A", "C")

    def test_double_complement_directed(self):
        """Complement of complement equals original (directed)."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        comp = g.complement()
        comp_comp = comp.complement()

        assert comp_comp.count_nodes() == g.count_nodes()
        assert comp_comp.count_edges() == g.count_edges()
        assert comp_comp.has_edge("A", "B")
        assert comp_comp.has_edge("B", "C")


class TestComplementCompleteGraphs:
    """Test complement with complete graphs."""

    def test_complement_complete_graph(self):
        """Complete graph complement is empty (no edges)."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")

        # Add all possible edges (complete K4)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        comp = g.complement()

        assert comp.count_nodes() == 4
        assert comp.count_edges() == 0

    def test_complement_empty_to_complete(self):
        """Empty graph (nodes only) complement is complete."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        # This graph has no edges

        comp = g.complement()

        assert comp.count_nodes() == 3

        # Complete K3 has 3 edges
        assert comp.count_edges() == 3
        assert comp.has_edge("A", "B")
        assert comp.has_edge("B", "C")
        assert comp.has_edge("A", "C")


class TestComplementEdgeCases:
    """Test edge cases for complement."""

    def test_complement_star_graph(self):
        """Star graph complement is complete graph minus center."""
        g = Graph(directed=False)
        g.add_node(label="center")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("center", "A")
        g.add_edge("center", "B")
        g.add_edge("center", "C")

        comp = g.complement()

        # Complement should have edges between A, B, C (not center)
        assert comp.count_edges() == 3
        assert comp.has_edge("A", "B")
        assert comp.has_edge("B", "C")
        assert comp.has_edge("A", "C")
        assert not comp.has_edge("center", "A")

    def test_complement_bipartite(self):
        """Bipartite graph complement."""
        g = Graph(directed=False)

        # Two sets: {A, B} and {X, Y}
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="X")
        g.add_node(label="Y")

        # Only cross-edges in bipartite
        g.add_edge("A", "X")
        g.add_edge("A", "Y")
        g.add_edge("B", "X")
        g.add_edge("B", "Y")

        comp = g.complement()

        # Complement has within-set edges: A-B, X-Y
        assert comp.count_edges() == 2
        assert comp.has_edge("A", "B")
        assert comp.has_edge("X", "Y")
