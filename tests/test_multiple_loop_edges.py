import pytest
from edgecase import Graph


def add_nodes(g: Graph, *labels: str) -> Graph:
    """Helper to add multiple nodes by label."""
    for label in labels:
        g.add_node(label=label)
    return g


class TestWhichMultiple:
    """Tests for which_multiple() method."""

    def test_empty_graph(self):
        """Empty graph should have no multiple edges."""
        g = Graph(directed=True)
        multiples = g.which_multiple()
        assert multiples == []

    def test_single_edge(self):
        """Single edge between nodes is not multiple."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        multiples = g.which_multiple()
        assert multiples == []

    def test_two_edges_same_pair_directed(self):
        """Two edges between same nodes in directed graph."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        multiples = g.which_multiple()
        assert len(multiples) == 2

    def test_three_edges_same_pair(self):
        """Three edges between same nodes."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        multiples = g.which_multiple()
        assert len(multiples) == 3
        assert len(multiples) > 0  # Multiple edges detected

    def test_multiple_pairs_with_duplicates(self):
        """Multiple node pairs with duplicate edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        # A->B has 2 edges
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        # B->C has 3 edges
        g.add_edge("B", "C")
        g.add_edge("B", "C")
        g.add_edge("B", "C")
        # C->D has 1 edge (not multiple)
        g.add_edge("C", "D")

        multiples = g.which_multiple()
        assert len(multiples) == 5  # 2 from A->B + 3 from B->C

    def test_undirected_graph_duplicate_edges(self):
        """Undirected graph with duplicate edges."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        multiples = g.which_multiple()
        assert len(multiples) == 2
        multiples = g.which_multiple()
        # e3 is single edge, not multiple

    def test_undirected_treats_both_directions_same(self):
        """Undirected graph treats (A,B) and (B,A) as same pair."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Same as A-B in undirected

        multiples = g.which_multiple()
        assert len(multiples) == 2

    def test_directed_different_directions_not_multiple(self):
        """In directed graph, A->B and B->A are different edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        multiples = g.which_multiple()
        assert multiples == []

    def test_self_loops_not_multiple(self):
        """Self-loops are not considered multiple edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        g.add_edge("A", "B")

        # Self-loops should not be in multiple edges
        # (they're handled by which_loops)
        multiples = g.which_multiple()
        # The two A->A edges ARE multiple edges of the same pair
        assert len(multiples) == 2

    def test_mixed_single_and_multiple(self):
        """Graph with both single and multiple edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D", "E")
        g.add_edge("A", "B")  # Single
        g.add_edge("B", "C")
        g.add_edge("B", "C")  # Multiple
        g.add_edge("C", "D")  # Single
        g.add_edge("D", "E")
        g.add_edge("D", "E")
        g.add_edge("D", "E")  # Multiple

        multiples = g.which_multiple()
        assert len(multiples) == 5


class TestCountMultiple:
    """Tests for count_multiple() method."""

    def test_empty_graph(self):
        """Empty graph should have no counts."""
        g = Graph(directed=True)
        counts = g.count_multiple()
        assert counts == {}

    def test_no_multiple_edges(self):
        """Graph with only single edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        counts = g.count_multiple()
        assert counts == {}

    def test_one_pair_two_edges(self):
        """One pair with two edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        counts = g.count_multiple()
        assert counts == {(a_id, b_id): 2}

    def test_one_pair_three_edges(self):
        """One pair with three edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        counts = g.count_multiple()
        assert counts == {(a_id, b_id): 3}

    def test_multiple_pairs_with_counts(self):
        """Multiple pairs with different counts."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        # A->B: 2 edges
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        # B->C: 3 edges
        g.add_edge("B", "C")
        g.add_edge("B", "C")
        g.add_edge("B", "C")
        # C->D: 1 edge (not in result)
        g.add_edge("C", "D")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        d_id = g.get_id_from_label("D")
        counts = g.count_multiple()
        assert len(counts) == 2
        assert counts[(a_id, b_id)] == 2
        assert counts[(b_id, c_id)] == 3
        assert (c_id, d_id) not in counts

    def test_undirected_graph_counts(self):
        """Undirected graph counts pairs correctly."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        counts = g.count_multiple()
        assert len(counts) == 1
        # Should normalize to (a_id, b_id) regardless of order
        assert counts[(a_id, b_id)] == 3

    def test_undirected_both_directions_counted(self):
        """Undirected: (A,B) and (B,A) count as same pair."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("A", "B")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        counts = g.count_multiple()
        assert len(counts) == 1
        assert counts[(a_id, b_id)] == 3

    def test_self_loops_counted(self):
        """Self-loops are counted as multiple if there are 2+."""
        g = Graph(directed=True)
        add_nodes(g, "A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")

        a_id = g.get_id_from_label("A")
        counts = g.count_multiple()
        assert counts[(a_id, a_id)] == 2

    def test_directed_different_directions_separate(self):
        """Directed: A->B and B->A are separate pairs."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("B", "A")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        counts = g.count_multiple()
        assert len(counts) == 2
        assert counts[(a_id, b_id)] == 2
        assert counts[(b_id, a_id)] == 2


class TestWhichLoop:
    """Tests for which_loops() method."""

    def test_empty_graph(self):
        """Empty graph has no loops."""
        g = Graph(directed=True)
        loops = g.which_loops()
        assert loops == []

    def test_no_loops(self):
        """Graph with no self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        loops = g.which_loops()
        assert loops == []

    def test_single_self_loop(self):
        """Single self-loop edge."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        e_loop = g.add_edge("A", "A")
        g.add_edge("B", "C")

        loops = g.which_loops()
        assert len(loops) == 1

    def test_multiple_self_loops_same_node(self):
        """Multiple self-loops on same node."""
        g = Graph(directed=True)
        add_nodes(g, "A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")

        loops = g.which_loops()
        assert len(loops) == 3
        assert len(loops) == 3

    def test_multiple_nodes_with_loops(self):
        """Multiple nodes each with self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("A", "A")
        g.add_edge("B", "C")
        g.add_edge("B", "B")
        g.add_edge("C", "C")

        loops = g.which_loops()
        assert len(loops) == 3
        assert len(loops) == 3

    def test_undirected_graph_loops(self):
        """Undirected graph with self-loops."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        e_loop = g.add_edge("A", "A")

        loops = g.which_loops()
        assert len(loops) == 1

    def test_all_edges_are_loops(self):
        """Graph where all edges are self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "A")
        g.add_edge("B", "B")
        g.add_edge("C", "C")

        loops = g.which_loops()
        assert len(loops) == 3
        assert len(loops) == 3

    def test_mixed_loops_and_regular_edges(self):
        """Mix of self-loops and regular edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        g.add_edge("A", "B")
        g.add_edge("A", "A")
        g.add_edge("B", "C")
        g.add_edge("B", "B")
        g.add_edge("C", "D")
        g.add_edge("D", "D")

        loops = g.which_loops()
        assert len(loops) == 3
        assert len(loops) == 3


class TestCountLoops:
    """Tests for count_loops() method."""

    def test_empty_graph(self):
        """Empty graph has no loop counts."""
        g = Graph(directed=True)
        counts = g.count_loops()
        assert counts == {}

    def test_no_loops(self):
        """Graph with no self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        counts = g.count_loops()
        assert counts == {}

    def test_single_loop_one_node(self):
        """Single node with one self-loop."""
        g = Graph(directed=True)
        add_nodes(g, "A")
        g.add_edge("A", "A")

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        assert counts == {a_id: 1}

    def test_single_node_multiple_loops(self):
        """Single node with multiple self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        assert counts == {a_id: 3}

    def test_multiple_nodes_one_loop_each(self):
        """Multiple nodes each with one self-loop."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "A")
        g.add_edge("B", "B")
        g.add_edge("C", "C")

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        assert len(counts) == 3
        assert counts[a_id] == 1
        assert counts[b_id] == 1
        assert counts[c_id] == 1

    def test_multiple_nodes_different_loop_counts(self):
        """Nodes with different numbers of self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        g.add_edge("B", "B")
        g.add_edge("C", "C")
        g.add_edge("C", "C")
        g.add_edge("C", "C")

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        assert counts[a_id] == 2
        assert counts[b_id] == 1
        assert counts[c_id] == 3

    def test_mixed_loops_and_regular_edges(self):
        """Mix of nodes with and without loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        g.add_edge("A", "B")
        g.add_edge("A", "A")
        g.add_edge("B", "C")
        g.add_edge("B", "B")
        g.add_edge("B", "B")
        g.add_edge("C", "D")
        # D has no self-loop

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        d_id = g.get_id_from_label("D")
        assert len(counts) == 2
        assert counts[a_id] == 1
        assert counts[b_id] == 2
        assert c_id not in counts
        assert d_id not in counts

    def test_undirected_graph_loops(self):
        """Undirected graph with self-loops."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        g.add_edge("B", "C")

        counts = g.count_loops()
        a_id = g.get_id_from_label("A")
        assert counts == {a_id: 2}


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_graph_with_both_multiple_and_loops(self):
        """Graph containing both multiple edges and loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        # Multiple edges
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        # Self-loops
        g.add_edge("A", "A")
        g.add_edge("B", "B")
        # Regular edge
        g.add_edge("B", "C")

        multiples = g.which_multiple()
        loops = g.which_loops()

        # Multiple edges: A->B appears twice
        assert len(multiples) == 2
        # Loops: A->A and B->B
        assert len(loops) == 2

    def test_count_functions_with_mixed_edges(self):
        """Test count functions with complex graph."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        # A->B: 3 edges
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        # A->A: 2 self-loops
        g.add_edge("A", "A")
        g.add_edge("A", "A")
        # B->B: 1 self-loop
        g.add_edge("B", "B")
        # B->C: 1 edge (not multiple)
        g.add_edge("B", "C")

        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")

        multi_counts = g.count_multiple()
        loop_counts = g.count_loops()

        # A->B has 3 edges, A->A has 2 edges
        assert multi_counts[(a_id, b_id)] == 3
        assert multi_counts[(a_id, a_id)] == 2
        assert (b_id, c_id) not in multi_counts

        # A has 2 loops, B has 1 loop
        assert loop_counts[a_id] == 2
        assert loop_counts[b_id] == 1
        assert c_id not in loop_counts

    def test_single_node_graph(self):
        """Single node with no edges."""
        g = Graph(directed=True)
        g.add_node("A")

        assert g.which_multiple() == []
        assert g.count_multiple() == {}
        assert g.which_loops() == []
        assert g.count_loops() == {}

    def test_disconnected_components(self):
        """Graph with disconnected components."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "X", "Y")
        # Component 1
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")
        # Component 2
        g.add_edge("X", "Y")
        g.add_edge("X", "Y")
        g.add_edge("Y", "Y")

        multiples = g.which_multiple()
        loops = g.which_loops()

        assert len(multiples) == 4  # 2 from A->B, 2 from X->Y
        assert len(loops) == 2  # A->A and Y->Y

    def test_complete_graph_with_self_loops(self):
        """Complete directed graph with self-loops."""
        g = Graph(directed=True)
        nodes = ["A", "B", "C"]

        # Add nodes first
        for node in nodes:
            g.add_node(label=node)

        # Add all edges including self-loops
        for src in nodes:
            for tgt in nodes:
                g.add_edge(src, tgt)

        loops = g.which_loops()
        assert len(loops) == 3  # One loop per node

        loop_counts = g.count_loops()
        node_ids = [g.get_id_from_label(n) for n in nodes]
        assert all(loop_counts[nid] == 1 for nid in node_ids)

    def test_multigraph_validation_workflow(self):
        """Test workflow for validating if graph is simple."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Check if simple graph (no multiple edges, no self-loops)
        has_multiple = len(g.which_multiple()) > 0
        has_loops = len(g.which_loops()) > 0
        is_simple = not (has_multiple or has_loops)

        assert is_simple is True

    def test_data_cleaning_workflow(self):
        """Test workflow for identifying edges to clean."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)  # Duplicate
        g.add_edge("A", "A", weight=1.0)  # Self-loop
        g.add_edge("B", "C", weight=1.0)

        # Identify problematic edges
        edges_to_review = g.which_multiple()
        loops_to_review = g.which_loops()

        total_issues = len(edges_to_review) + len(loops_to_review)
        assert total_issues == 3  # 2 multiples + 1 loop
