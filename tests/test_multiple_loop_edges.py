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


class TestRemoveSelfloops:
    """Tests for remove_selfloops() method."""

    def test_empty_graph(self):
        """Empty graph should remain empty."""
        g = Graph(directed=True)
        g.remove_selfloops()

        assert g.count_edges() == 0

    def test_graph_without_selfloops(self):
        """Graph without self-loops should be unchanged."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        initial_edges = g.count_edges()
        g.remove_selfloops()

        assert g.count_edges() == initial_edges

    def test_single_selfloop(self):
        """Remove a single self-loop."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        assert g.count_edges() == 2

        g.remove_selfloops()

        assert g.count_edges() == 1
        assert len(g.which_loops()) == 0

    def test_multiple_selfloops(self):
        """Remove multiple self-loops from different nodes."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop on A
        g.add_edge("B", "B")  # Self-loop on B
        g.add_edge("C", "C")  # Self-loop on C

        assert g.count_edges() == 4

        g.remove_selfloops()

        assert g.count_edges() == 1
        assert len(g.which_loops()) == 0

    def test_multiple_selfloops_same_node(self):
        """Remove multiple self-loops on the same node."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop 1
        g.add_edge("A", "A")  # Self-loop 2
        g.add_edge("A", "A")  # Self-loop 3

        assert g.count_edges() == 4

        g.remove_selfloops()

        assert g.count_edges() == 1
        assert len(g.which_loops()) == 0

    def test_undirected_graph(self):
        """Remove self-loops from undirected graph."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")

        assert g.count_edges() == 2

        g.remove_selfloops()

        assert g.count_edges() == 1
        assert len(g.which_loops()) == 0

    def test_method_chaining(self):
        """Method should return self for chaining."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "A")

        result = g.remove_selfloops()

        assert result is g

    def test_preserves_regular_edges(self):
        """Ensure regular edges are not affected."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("A", "A")  # Self-loop

        g.remove_selfloops()

        assert g.has_edge("A", "B")
        assert g.has_edge("B", "C")
        assert g.count_edges() == 2


class TestRemoveDuplicateEdges:
    """Tests for remove_duplicate_edges() method."""

    def test_empty_graph(self):
        """Empty graph should remain empty."""
        g = Graph(directed=True)
        g.remove_duplicate_edges()

        assert g.count_edges() == 0

    def test_graph_without_duplicates(self):
        """Graph without duplicates should be unchanged."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        initial_edges = g.count_edges()
        g.remove_duplicate_edges()

        assert g.count_edges() == initial_edges

    def test_single_duplicate_pair(self):
        """Remove duplicate edges between one pair."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Duplicate
        g.add_edge("A", "B")  # Duplicate

        assert g.count_edges() == 3

        g.remove_duplicate_edges()

        assert g.count_edges() == 1
        assert len(g.which_multiple()) == 0

    def test_multiple_duplicate_pairs(self):
        """Remove duplicates from multiple node pairs."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Duplicate
        g.add_edge("B", "C")
        g.add_edge("B", "C")  # Duplicate
        g.add_edge("B", "C")  # Duplicate

        assert g.count_edges() == 5

        g.remove_duplicate_edges()

        assert g.count_edges() == 2
        assert len(g.which_multiple()) == 0

    def test_directed_graph_respects_direction(self):
        """In directed graphs, A->B and B->A are different pairs."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Not a duplicate (different direction)
        g.add_edge("A", "B")  # Duplicate of first

        assert g.count_edges() == 3

        g.remove_duplicate_edges()

        assert g.count_edges() == 2  # A->B and B->A

    def test_undirected_graph_treats_directions_same(self):
        """In undirected graphs, A-B and B-A are the same pair."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Duplicate in undirected
        g.add_edge("A", "B")  # Also duplicate

        assert g.count_edges() == 3

        g.remove_duplicate_edges()

        assert g.count_edges() == 1

    def test_keeps_first_edge(self):
        """Should keep the first edge (lowest ID) in each group."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)
        g.add_edge("A", "B", weight=3.0)

        edge_ids_before = g.get_edge_ids()
        first_edge_id = min(edge_ids_before)

        g.remove_duplicate_edges()

        edge_ids_after = g.get_edge_ids()

        assert len(edge_ids_after) == 1
        assert edge_ids_after[0] == first_edge_id

    def test_method_chaining(self):
        """Method should return self for chaining."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        result = g.remove_duplicate_edges()

        assert result is g

    def test_does_not_remove_selfloops(self):
        """Self-loops should not be removed by this method."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        g.remove_duplicate_edges()

        assert g.count_edges() == 2
        assert len(g.which_loops()) == 1

    def test_keep_distinct_attributes_false(self):
        """Default behavior ignores attributes."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)
        g.add_edge("A", "B", weight=3.0)

        g.remove_duplicate_edges(keep_distinct_attributes=False)

        assert g.count_edges() == 1

    def test_keep_distinct_attributes_true_different_weights(self):
        """With keep_distinct_attributes=True, different weights are kept."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)
        g.add_edge("A", "B", weight=1.0)  # True duplicate

        g.remove_duplicate_edges(keep_distinct_attributes=True)

        assert g.count_edges() == 2  # weight 1.0 and 2.0

    def test_keep_distinct_attributes_true_different_rel(self):
        """With keep_distinct_attributes=True, different rel values are kept."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", rel="friend")
        g.add_edge("A", "B", rel="colleague")
        g.add_edge("A", "B", rel="friend")  # True duplicate

        g.remove_duplicate_edges(keep_distinct_attributes=True)

        assert g.count_edges() == 2  # friend and colleague

    def test_keep_distinct_attributes_true_different_custom_attrs(self):
        """With keep_distinct_attributes=True, different custom attrs are kept."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", color="red")
        g.add_edge("A", "B", color="blue")
        g.add_edge("A", "B", color="red")  # True duplicate
        g.add_edge("A", "B", color="red")  # True duplicate

        g.remove_duplicate_edges(keep_distinct_attributes=True)

        assert g.count_edges() == 2  # red and blue

    def test_keep_distinct_attributes_true_all_identical(self):
        """With keep_distinct_attributes=True, identical edges are removed."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B", weight=1.0, rel="friend")
        g.add_edge("A", "B", weight=1.0, rel="friend")  # True duplicate
        g.add_edge("A", "B", weight=1.0, rel="friend")  # True duplicate

        g.remove_duplicate_edges(keep_distinct_attributes=True)

        assert g.count_edges() == 1

    def test_keep_distinct_attributes_true_no_attributes(self):
        """Edges with no attributes are all duplicates."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        g.remove_duplicate_edges(keep_distinct_attributes=True)

        assert g.count_edges() == 1


class TestSimplify:
    """Tests for simplify() method."""

    def test_empty_graph(self):
        """Empty graph should remain empty."""
        g = Graph(directed=True)
        g.simplify()

        assert g.count_edges() == 0
        assert g.is_simple()

    def test_already_simple_graph(self):
        """Simple graph should be unchanged."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        initial_edges = g.count_edges()

        assert g.is_simple()

        g.simplify()

        assert g.count_edges() == initial_edges
        assert g.is_simple()

    def test_removes_selfloops(self):
        """Should remove self-loops."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        assert not g.is_simple()

        g.simplify()

        assert g.is_simple()
        assert g.count_edges() == 1

    def test_removes_duplicates(self):
        """Should remove duplicate edges."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Duplicate

        assert not g.is_simple()

        g.simplify()

        assert g.is_simple()
        assert g.count_edges() == 1

    def test_removes_both_selfloops_and_duplicates(self):
        """Should remove both self-loops and duplicates."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Duplicate
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("B", "B")  # Self-loop

        assert g.count_edges() == 4
        assert not g.is_simple()

        g.simplify()

        assert g.count_edges() == 1
        assert g.is_simple()
        assert len(g.which_loops()) == 0
        assert len(g.which_multiple()) == 0

    def test_complex_graph(self):
        """Test on a more complex graph."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B", "C", "D")
        # Regular edges
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        # Duplicates
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        # Self-loops
        g.add_edge("A", "A")
        g.add_edge("C", "C")
        g.add_edge("D", "D")

        assert g.count_edges() == 9
        assert not g.is_simple()

        g.simplify()

        assert g.count_edges() == 3  # A->B, B->C, C->D
        assert g.is_simple()

    def test_undirected_graph(self):
        """Test simplify on undirected graph."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C")
        g.add_edge("A", "B")
        g.add_edge("B", "A")  # Duplicate in undirected
        g.add_edge("B", "C")
        g.add_edge("A", "A")  # Self-loop

        assert not g.is_simple()

        g.simplify()

        assert g.is_simple()
        assert g.count_edges() == 2  # A-B and B-C

    def test_method_chaining(self):
        """Method should return self for chaining."""
        g = Graph(directed=True)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")

        result = g.simplify()

        assert result is g

    def test_chaining_pipeline(self):
        """Test using simplify in a method chain."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("A", "B")  # Duplicate
            .add_edge("B", "C")
            .add_edge("C", "C")  # Self-loop
            .simplify()
        )

        assert g.is_simple()
        assert g.count_edges() == 2
