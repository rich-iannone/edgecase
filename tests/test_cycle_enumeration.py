"""Tests for Johnson's cycle enumeration algorithm."""

import pytest
import edgecase as ec


class TestCycleEnumeration:
    """Test find_cycles() method using Johnson's algorithm."""

    def test_empty_graph(self):
        """Empty graph has no cycles."""
        G = ec.Graph(directed=True)
        cycles = G.get_cycles()
        assert len(cycles) == 0

    def test_single_node_no_loop(self):
        """Single node without self-loop has no cycles."""
        G = ec.Graph(directed=True).add_node("A")
        cycles = G.get_cycles()
        assert len(cycles) == 0

    def test_self_loop(self):
        """Self-loop creates a 1-node cycle."""
        G = ec.Graph(directed=True).add_node("A").add_edge("A", "A")
        cycles = G.get_cycles(min_length=1)
        assert len(cycles) == 1
        assert len(cycles[0]) == 1  # Self-loop is a single node

    def test_self_loop_min_length_filter(self):
        """Self-loop filtered out by min_length=2."""
        G = ec.Graph(directed=True).add_node("A").add_edge("A", "A")
        cycles = G.get_cycles(min_length=2)
        assert len(cycles) == 0

    def test_simple_triangle_cycle(self):
        """A -> B -> C -> A forms one 3-node cycle."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 1
        assert len(cycles[0]) == 3
        # Check all three node IDs are in the cycle
        node_ids = G.node_ids(limit=None)
        assert set(cycles[0]) == set(node_ids)

    def test_two_node_cycle(self):
        """A <-> B forms a 2-node cycle."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_edge("A", "B")
            .add_edge("B", "A")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 1
        assert len(cycles[0]) == 2
        # Check both node IDs are in the cycle
        node_ids = G.node_ids(limit=None)
        assert set(cycles[0]) == set(node_ids)

    def test_multiple_separate_cycles(self):
        """Two disconnected cycles should both be found."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            # First cycle: A <-> B
            .add_edge("A", "B")
            .add_edge("B", "A")
            # Second cycle: C <-> D
            .add_edge("C", "D")
            .add_edge("D", "C")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 2
        # Both cycles should be length 2
        assert all(len(cycle) == 2 for cycle in cycles)

    def test_overlapping_cycles(self):
        """Shared edges create multiple cycles: A<->B and A->B->C->A."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            # Short cycle: A <-> B
            .add_edge("A", "B")
            .add_edge("B", "A")
            # Longer cycle: A -> B -> C -> A (shares A->B with short cycle)
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 2
        cycle_lengths = sorted([len(c) for c in cycles])
        assert cycle_lengths == [2, 3]

    def test_min_length_filtering(self):
        """min_length parameter filters short cycles."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            # Two-node cycle
            .add_edge("A", "B")
            .add_edge("B", "A")
            # Three-node cycle
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # All cycles
        all_cycles = G.get_cycles()
        assert len(all_cycles) == 2

        # Only cycles of length 3+
        long_cycles = G.get_cycles(min_length=3)
        assert len(long_cycles) == 1
        assert len(long_cycles[0]) == 3

    def test_dag_no_cycles(self):
        """Directed acyclic graph has no cycles."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 0

    def test_undirected_raises_error(self):
        """Undirected graphs should raise ValueError."""
        G = ec.Graph(directed=False).add_node("A").add_node("B").add_edge("A", "B")
        with pytest.raises(ValueError, match="requires a directed graph"):
            G.get_cycles()

    def test_complex_graph_multiple_cycles(self):
        """Complex graph with multiple overlapping cycles."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            # Create a diamond with cycles
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
            .add_edge("D", "A")  # Closes multiple cycles
        )
        cycles = G.get_cycles()
        # Should find: A->B->D->A, A->C->D->A
        assert len(cycles) >= 2

    def test_cycle_with_attributes(self):
        """Cycles work correctly with node attributes."""
        G = (
            ec.Graph(directed=True)
            .add_node("A", type="user")
            .add_node("B", type="admin")
            .add_node("C", type="user")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
            .add_edge("C", "A", weight=3.0)
        )
        cycles = G.get_cycles()
        assert len(cycles) == 1
        # Check all three node IDs are in the cycle
        node_ids = G.node_ids(limit=None)
        assert set(cycles[0]) == set(node_ids)

    def test_standalone_function(self):
        """Standalone find_cycles function works."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_edge("A", "B")
            .add_edge("B", "A")
        )
        cycles = ec.get_cycles(G)
        assert len(cycles) == 1
        assert len(cycles[0]) == 2

    def test_standalone_function_min_length(self):
        """Standalone function accepts min_length parameter."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B")
            .add_edge("B", "A")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        cycles = ec.get_cycles(G, min_length=3)
        assert len(cycles) == 1
        assert len(cycles[0]) == 3

    def test_large_cycle(self):
        """Long cycle is detected correctly."""
        G = ec.Graph(directed=True)
        nodes = [f"N{i}" for i in range(10)]
        for node in nodes:
            G.add_node(node)

        # Create cycle: N0 -> N1 -> N2 -> ... -> N9 -> N0
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

        cycles = G.get_cycles()
        assert len(cycles) == 1
        assert len(cycles[0]) == 10

    def test_multiple_cycles_different_lengths(self):
        """Graph with cycles of different lengths."""
        G = (
            ec.Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_node("E")
            # 2-cycle: A <-> B
            .add_edge("A", "B")
            .add_edge("B", "A")
            # 3-cycle: C -> D -> E -> C
            .add_edge("C", "D")
            .add_edge("D", "E")
            .add_edge("E", "C")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 2
        cycle_lengths = sorted([len(c) for c in cycles])
        assert cycle_lengths == [2, 3]

    def test_disconnected_components_with_cycles(self):
        """Multiple disconnected components each with cycles."""
        G = (
            ec.Graph(directed=True)
            # Component 1: triangle
            .add_node("A1")
            .add_node("A2")
            .add_node("A3")
            .add_edge("A1", "A2")
            .add_edge("A2", "A3")
            .add_edge("A3", "A1")
            # Component 2: square
            .add_node("B1")
            .add_node("B2")
            .add_node("B3")
            .add_node("B4")
            .add_edge("B1", "B2")
            .add_edge("B2", "B3")
            .add_edge("B3", "B4")
            .add_edge("B4", "B1")
        )
        cycles = G.get_cycles()
        assert len(cycles) == 2
        cycle_lengths = sorted([len(c) for c in cycles])
        assert cycle_lengths == [3, 4]
