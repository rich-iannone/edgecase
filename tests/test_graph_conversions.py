from edgecase import Graph


class TestToUndirected:
    """Test converting graphs to undirected."""

    def test_directed_to_undirected_simple(self):
        """Convert simple directed graph to undirected."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        undirected = g.to_undirected()

        assert not undirected.is_directed()
        assert undirected.count_nodes() == 3
        assert undirected.count_edges() == 3

        # Check edges exist (in undirected, both directions work)
        assert undirected.has_edge("A", "B")
        assert undirected.has_edge("B", "A")

    def test_directed_to_undirected_with_reverse_edges(self):
        """Convert directed graph with bidirectional edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Reverse edge
        g.add_edge("B", "C")

        undirected = g.to_undirected()

        assert not undirected.is_directed()

        # Should collapse A<->B into one edge
        assert undirected.count_edges() == 2  # A-B and B-C

    def test_undirected_to_undirected_copy(self):
        """Converting undirected to undirected creates a copy."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        copy = g.to_undirected()

        assert not copy.is_directed()
        assert copy.count_nodes() == 3
        assert copy.count_edges() == 2

        # Verify it's a separate object
        g.add_node(label="D")
        g.add_edge("C", "D")

        assert g.count_edges() == 3
        assert copy.count_edges() == 2  # Copy unchanged

    def test_preserves_node_attributes(self):
        """Node attributes are preserved in conversion."""
        g = Graph(directed=True)
        g.add_node(label="A", color="red", value=10)
        g.add_node(label="B", color="blue", value=20)
        g.add_edge("A", "B")

        undirected = g.to_undirected()

        node_a = undirected.get_node("A")

        assert node_a.attributes["color"] == "red"
        assert node_a.attributes["value"] == 10

    def test_preserves_edge_attributes(self):
        """Edge attributes are preserved in conversion."""
        g = Graph(directed=True, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

        undirected = g.to_undirected()

        edge_id = undirected.find_edge_ids("A", "B")[0]
        edge = undirected.get_edge(edge_id)

        assert edge.attributes["label"] == "edge1"
        assert edge.attributes["type"] == "strong"

    def test_empty_graph(self):
        """Convert empty directed graph to undirected."""
        g = Graph(directed=True)
        undirected = g.to_undirected()

        assert not undirected.is_directed()
        assert undirected.count_nodes() == 0
        assert undirected.count_edges() == 0

    def test_single_node(self):
        """Convert single node graph."""
        g = Graph(directed=True)
        g.add_node(label="A")

        undirected = g.to_undirected()

        assert not undirected.is_directed()
        assert undirected.count_nodes() == 1
        assert undirected.count_edges() == 0


class TestToDirected:
    """Test converting graphs to directed."""

    def test_undirected_to_directed_simple(self):
        """Convert simple undirected graph to directed."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        directed = g.to_directed()

        assert directed.is_directed()
        assert directed.count_nodes() == 3

        # Each undirected edge becomes two directed edges
        assert directed.count_edges() == 4  # A->B, B->A, B->C, C->B

    def test_undirected_to_directed_bidirectional(self):
        """Undirected edges become bidirectional."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        directed = g.to_directed()

        assert directed.has_edge("A", "B")
        assert directed.has_edge("B", "A")

    def test_directed_to_directed_copy(self):
        """Converting directed to directed creates a copy."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        copy = g.to_directed()

        assert copy.is_directed()
        assert copy.count_nodes() == 4
        assert copy.count_edges() == 2

        # Verify it's a separate object
        g.add_edge("C", "D")

        assert g.count_edges() == 3
        assert copy.count_edges() == 2  # Copy unchanged

    def test_preserves_node_attributes(self):
        """Node attributes are preserved in conversion."""
        g = Graph(directed=False)
        g.add_node(label="A", color="red", value=10)
        g.add_node(label="B", color="blue", value=20)
        g.add_edge("A", "B")

        directed = g.to_directed()

        node_a = directed.get_node("A")

        assert node_a.attributes["color"] == "red"
        assert node_a.attributes["value"] == 10

    def test_preserves_edge_attributes(self):
        """Edge attributes are preserved in conversion."""
        g = Graph(directed=False, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

        directed = g.to_directed()

        # Both directions should have the attributes
        edge_ids = directed.find_edge_ids("A", "B")

        assert len(edge_ids) >= 1

        edge = directed.get_edge(edge_ids[0])

        assert edge.attributes["label"] == "edge1"
        assert edge.attributes["type"] == "strong"

    def test_empty_graph(self):
        """Convert empty undirected graph to directed."""
        g = Graph(directed=False)
        directed = g.to_directed()

        assert directed.is_directed()
        assert directed.count_nodes() == 0
        assert directed.count_edges() == 0

    def test_single_node(self):
        """Convert single node graph."""
        g = Graph(directed=False)
        g.add_node(label="A")

        directed = g.to_directed()

        assert directed.is_directed()
        assert directed.count_nodes() == 1
        assert directed.count_edges() == 0


class TestReverse:
    """Test reversing edge directions."""

    def test_reverse_directed_simple(self):
        """Reverse simple directed graph."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        reversed_g = g.reverse()

        assert reversed_g.is_directed()
        assert reversed_g.count_nodes() == 3
        assert reversed_g.count_edges() == 2

        # Original edges
        assert g.has_edge("A", "B")
        assert g.has_edge("B", "C")

        # Reversed edges
        assert reversed_g.has_edge("B", "A")
        assert reversed_g.has_edge("C", "B")
        assert not reversed_g.has_edge("A", "B")
        assert not reversed_g.has_edge("B", "C")

    def test_reverse_directed_cycle(self):
        """Reverse directed cycle."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        reversed_g = g.reverse()

        # Original: A->B->C->A
        # Reversed: A->C->B->A
        assert reversed_g.has_edge("B", "A")
        assert reversed_g.has_edge("C", "B")
        assert reversed_g.has_edge("A", "C")

    def test_reverse_undirected_unchanged(self):
        """Reversing undirected graph returns copy."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        reversed_g = g.reverse()

        assert not reversed_g.is_directed()
        assert reversed_g.count_nodes() == 3
        assert reversed_g.count_edges() == 2

        # Undirected edges work both ways
        assert reversed_g.has_edge("A", "B")
        assert reversed_g.has_edge("B", "A")

    def test_preserves_node_attributes(self):
        """Node attributes are preserved in reversal."""
        g = Graph(directed=True)
        g.add_node(label="A", color="red", value=10)
        g.add_node(label="B", color="blue", value=20)
        g.add_edge("A", "B")

        reversed_g = g.reverse()

        node_a = reversed_g.get_node("A")

        assert node_a.attributes["color"] == "red"
        assert node_a.attributes["value"] == 10

    def test_preserves_edge_attributes(self):
        """Edge attributes are preserved in reversal."""
        g = Graph(directed=True, weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

        reversed_g = g.reverse()

        # Edge should be B->A now
        edge_ids = reversed_g.find_edge_ids("B", "A")

        assert len(edge_ids) >= 1

        edge = reversed_g.get_edge(edge_ids[0])

        assert edge.attributes["label"] == "edge1"
        assert edge.attributes["type"] == "strong"

    def test_empty_graph(self):
        """Reverse empty directed graph."""
        g = Graph(directed=True)
        reversed_g = g.reverse()

        assert reversed_g.is_directed()
        assert reversed_g.count_nodes() == 0
        assert reversed_g.count_edges() == 0

    def test_single_node(self):
        """Reverse single node graph."""
        g = Graph(directed=True)
        g.add_node(label="A")

        reversed_g = g.reverse()

        assert reversed_g.is_directed()
        assert reversed_g.count_nodes() == 1
        assert reversed_g.count_edges() == 0

    def test_self_loop(self):
        """Reverse self-loop remains self-loop."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_edge("A", "A")

        reversed_g = g.reverse()

        assert reversed_g.has_edge("A", "A")
        assert reversed_g.count_edges() == 1


class TestRoundTripConversions:
    """Test round-trip conversions."""

    def test_directed_to_undirected_to_directed(self):
        """Convert directed->undirected->directed."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        undirected = g.to_undirected()
        back_to_directed = undirected.to_directed()

        assert back_to_directed.is_directed()

        # After round-trip through undirected, we get bidirectional edges
        assert back_to_directed.has_edge("A", "B")
        assert back_to_directed.has_edge("B", "A")
        assert back_to_directed.has_edge("B", "C")
        assert back_to_directed.has_edge("C", "B")

    def test_undirected_to_directed_to_undirected(self):
        """Convert undirected->directed->undirected."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        directed = g.to_directed()
        back_to_undirected = directed.to_undirected()

        assert not back_to_undirected.is_directed()

        # It should have same structure as the original
        assert back_to_undirected.count_edges() == 2

    def test_reverse_twice_returns_original(self):
        """Reversing twice returns to original structure."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        reversed_once = g.reverse()
        reversed_twice = reversed_once.reverse()

        # Should have same edge structure as original
        assert reversed_twice.has_edge("A", "B")
        assert reversed_twice.has_edge("B", "C")
        assert not reversed_twice.has_edge("B", "A")
        assert not reversed_twice.has_edge("C", "B")
