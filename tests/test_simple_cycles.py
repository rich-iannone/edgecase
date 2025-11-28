from edgecase import Graph


class TestSimpleCyclesDirected:
    """Tests for simple_cycles on directed graphs."""

    def test_empty_graph(self):
        """Empty graph has no cycles."""
        g = Graph(directed=True)
        cycles = g.simple_cycles()

        assert cycles == []

    def test_single_node_no_self_loop(self):
        """Single node without self-loop has no cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        cycles = g.simple_cycles()

        assert cycles == []

    def test_single_node_self_loop(self):
        """Single node with self-loop has one cycle."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_edge("A", "A")
        cycles = g.simple_cycles()

        assert len(cycles) == 1
        assert len(cycles[0]) == 1

    def test_simple_triangle(self):
        """Triangle cycle A -> B -> C -> A."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 1
        assert len(cycles[0]) == 3

    def test_two_node_cycle(self):
        """Two-node cycle A -> B -> A."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 1
        assert len(cycles[0]) == 2

    def test_multiple_cycles(self):
        """Graph with multiple independent cycles."""
        g = Graph(directed=True)

        # Two separate 2-cycles
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")

        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("C", "D")
        g.add_edge("D", "C")

        cycles = g.simple_cycles()

        assert len(cycles) == 2

    def test_overlapping_cycles(self):
        """Graph with overlapping cycles sharing edges."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # Creates both A->B->A and A->B->C->A
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 2

        # Should have one 2-cycle and one 3-cycle
        cycle_lengths = sorted([len(c) for c in cycles])

        assert cycle_lengths == [2, 3]

    def test_dag_no_cycles(self):
        """DAG (directed acyclic graph) has no cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        cycles = g.simple_cycles()

        assert cycles == []

    def test_length_bound(self):
        """Length bound filters out longer cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # 2-cycle and 3-cycle
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # All cycles
        all_cycles = g.simple_cycles()

        assert len(all_cycles) == 2

        # Only short cycles
        short_cycles = g.simple_cycles(length_bound=2)

        assert len(short_cycles) == 1
        assert len(short_cycles[0]) == 2


class TestSimpleCyclesUndirected:
    """Tests for simple_cycles on undirected graphs."""

    def test_empty_graph(self):
        """Empty undirected graph has no cycles."""
        g = Graph(directed=False)
        cycles = g.simple_cycles()

        assert cycles == []

    def test_single_node(self):
        """Single node has no cycles."""
        g = Graph(directed=False)
        g.add_node(label="A")
        cycles = g.simple_cycles()

        assert cycles == []

    def test_simple_triangle(self):
        """Triangle in undirected graph is one cycle."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 1
        assert len(cycles[0]) == 3

    def test_square_cycle(self):
        """Square cycle in undirected graph."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 1
        assert len(cycles[0]) == 4

    def test_path_no_cycles(self):
        """Path graph (tree) has no cycles."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        cycles = g.simple_cycles()

        assert cycles == []

    def test_two_triangles_sharing_edge(self):
        """Two triangles sharing one edge."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")

        # Triangle 1: A-B-C
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Triangle 2: A-B-D (shares edge A-B)
        g.add_edge("B", "D")
        g.add_edge("D", "A")

        cycles = g.simple_cycles()

        # Should find: A-B-C, A-B-D, and possibly A-B-D-...-C-A
        assert len(cycles) >= 2

    def test_length_bound_undirected(self):
        """Length bound works for undirected graphs."""
        g = Graph(directed=False)

        # Create a square with diagonal (has 3-cycle and 4-cycle)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "A")
        g.add_edge("A", "C")  # Diagonal

        all_cycles = g.simple_cycles()

        assert len(all_cycles) >= 2  # At least two triangles

        # Only triangles
        triangles = g.simple_cycles(length_bound=3)
        for cycle in triangles:
            assert len(cycle) <= 3

    def test_complete_k4(self):
        """Complete graph K4 has multiple cycles."""
        g = Graph(directed=False)
        for label in ["A", "B", "C", "D"]:
            g.add_node(label=label)

        # All edges in K4
        for i, l1 in enumerate(["A", "B", "C", "D"]):
            for l2 in ["A", "B", "C", "D"][i + 1 :]:
                g.add_edge(l1, l2)

        cycles = g.simple_cycles()

        # K4 has: 4 triangles + 3 squares = 7 cycles
        assert len(cycles) >= 4  # At least the 4 triangles


class TestSimpleCyclesEdgeCases:
    """Edge case tests for simple_cycles."""

    def test_disconnected_components(self):
        """Cycles in disconnected components are all found."""
        g = Graph(directed=False)

        # Component 1: triangle
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Component 2: square
        g.add_node(label="D")
        g.add_node(label="E")
        g.add_node(label="F")
        g.add_node(label="G")
        g.add_edge("D", "E")
        g.add_edge("E", "F")
        g.add_edge("F", "G")
        g.add_edge("G", "D")

        cycles = g.simple_cycles()

        assert len(cycles) == 2  # One triangle, one square

    def test_length_bound_zero(self):
        """Length bound of 0 returns no cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        cycles = g.simple_cycles(length_bound=0)

        assert cycles == []

    def test_length_bound_one(self):
        """Length bound of 1 returns only self-loops (directed)."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        cycles = g.simple_cycles(length_bound=1)

        assert len(cycles) == 1
        assert len(cycles[0]) == 1

    def test_returns_node_ids(self):
        """Cycles contain node IDs, not labels."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        cycles = g.simple_cycles()

        assert len(cycles) == 1

        # Check that we got integers (node IDs)
        for node_id in cycles[0]:
            assert isinstance(node_id, int)


class TestSimpleCyclesComparison:
    """Tests comparing simple_cycles with get_cycles."""

    def test_directed_graph_same_results(self):
        """For directed graphs, simple_cycles should find same cycles as get_cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("B", "A")

        simple = g.simple_cycles()
        traditional = g.get_cycles()

        # Same number of cycles
        assert len(simple) == len(traditional)

        # Convert to sets for comparison (order may differ)
        simple_sets = {frozenset(c) for c in simple}
        traditional_sets = {frozenset(c) for c in traditional}

        assert simple_sets == traditional_sets
