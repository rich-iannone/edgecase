import pytest
import edgecase as ec


class TestTransitiveClosure:
    """Test transitive_closure() method."""

    def test_simple_chain(self):
        """Test transitive closure on a simple chain A -> B -> C."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        closure = g.transitive_closure()

        # Should have same nodes
        assert closure.count_nodes() == 3

        # Should have 3 edges: A->B, B->C, A->C
        assert closure.count_edges() == 3

    def test_diamond_dag(self):
        """Test transitive closure on diamond-shaped DAG."""
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        closure = g.transitive_closure()

        assert closure.count_nodes() == 4

        # A can reach B, C, D (3 edges)
        # B can reach D (1 edge)
        # C can reach D (1 edge)
        # Total: 5 edges
        assert closure.count_edges() == 5

    def test_disconnected_components(self):
        """Test transitive closure with disconnected components."""
        g = ec.Graph(directed=True)

        # Component 1: A -> B
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Component 2: C -> D
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("C", "D")

        closure = g.transitive_closure()

        assert closure.count_nodes() == 4

        # Just the direct edges, no cross-component edges
        assert closure.count_edges() == 2

    def test_complete_dag(self):
        """Test transitive closure on a complete DAG (all forward edges)."""
        # A -> B -> C with all transitive edges already present
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")  # Already has transitive edge

        closure = g.transitive_closure()

        assert closure.count_nodes() == 3
        assert closure.count_edges() == 3  # Same as original

    def test_empty_graph(self):
        """Test transitive closure on empty graph."""
        g = ec.Graph(directed=True)

        closure = g.transitive_closure()

        assert closure.count_nodes() == 0
        assert closure.count_edges() == 0

    def test_single_node(self):
        """Test transitive closure on single node graph."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        closure = g.transitive_closure()

        assert closure.count_nodes() == 1
        assert closure.count_edges() == 0

    def test_undirected_raises_error(self):
        """Test that transitive_closure raises ValueError on undirected graph."""
        g = ec.Graph()  # Undirected
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="only works on directed graphs"):
            g.transitive_closure()

    def test_cyclic_graph(self):
        """Test transitive closure works on cyclic graphs (just computes reachability)."""
        # A -> B -> C -> A (cycle)
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        closure = g.transitive_closure()

        # In a cycle, everyone can reach everyone else including themselves
        # A reaches A, B, C (3 edges)
        # B reaches A, B, C (3 edges)
        # C reaches A, B, C (3 edges)
        # Total: 9 edges (all pairs including self-loops)
        assert closure.count_nodes() == 3
        assert closure.count_edges() == 9  # All pairs including self-loops

    def test_longer_chain(self):
        """Test transitive closure on longer chain."""
        g = ec.Graph(directed=True)
        for i in range(1, 6):
            g.add_node(label=str(i))
        for i in range(1, 5):
            g.add_edge(str(i), str(i + 1))

        closure = g.transitive_closure()

        assert closure.count_nodes() == 5

        # Number of edges = sum of (n-1) + (n-2) + ... + 1 = n(n-1)/2 = 5*4/2 = 10
        assert closure.count_edges() == 10


class TestTransitiveReduction:
    """Test transitive_reduction() method."""

    def test_simple_redundant_edge(self):
        """Test removing a single redundant edge."""
        # A -> B -> C with redundant A -> C
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")  # Redundant

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 3
        assert reduction.count_edges() == 2  # Only A->B, B->C

    def test_no_redundant_edges(self):
        """Test graph with no redundant edges stays the same."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 3
        assert reduction.count_edges() == 2  # Same as original

    def test_diamond_with_redundant(self):
        """Test diamond DAG with redundant edge."""
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        # Plus redundant A -> D
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        g.add_edge("A", "D")  # Redundant

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 4
        assert reduction.count_edges() == 4  # A->D removed

    def test_fully_connected_dag(self):
        """Test reduction of fully connected DAG (all forward edges)."""
        # 1 -> 2 -> 3 with all transitive edges
        g = ec.Graph(directed=True)
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("1", "3")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 3
        assert reduction.count_edges() == 2

    def test_multiple_redundant_edges(self):
        """Test removal of multiple redundant edges."""
        # Chain with all transitive edges
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")

        # Chain edges
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        # Redundant edges
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "D")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 4
        assert reduction.count_edges() == 3  # Only chain edges

    def test_empty_graph(self):
        """Test transitive reduction on empty graph."""
        g = ec.Graph(directed=True)

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 0
        assert reduction.count_edges() == 0

    def test_single_node(self):
        """Test transitive reduction on single node graph."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 1
        assert reduction.count_edges() == 0

    def test_undirected_raises_error(self):
        """Test that transitive_reduction raises ValueError on undirected graph."""
        g = ec.Graph()  # Undirected
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="only works on directed graphs"):
            g.transitive_reduction()

    def test_cyclic_raises_error(self):
        """Test that transitive_reduction raises ValueError on cyclic graph."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        with pytest.raises(ValueError, match="cycle"):
            g.transitive_reduction()

    def test_parallel_paths(self):
        """Test reduction with parallel paths of different lengths."""
        # A -> B -> D and A -> C -> D
        # Both paths have length 2, neither is redundant
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "D")
        g.add_edge("A", "C")
        g.add_edge("C", "D")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 4
        assert reduction.count_edges() == 4  # All edges needed

    def test_disconnected_components(self):
        """Test transitive reduction with disconnected components."""
        g = ec.Graph(directed=True)

        # Component 1: A -> B -> C with A -> C redundant
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")  # Redundant

        # Component 2: D -> E
        g.add_node(label="D")
        g.add_node(label="E")
        g.add_edge("D", "E")

        reduction = g.transitive_reduction()

        assert reduction.count_nodes() == 5
        assert reduction.count_edges() == 3  # A->B, B->C, D->E


class TestTransitiveRoundTrip:
    """Test that closure(reduction(G)) == closure(G) for DAGs."""

    def test_round_trip_simple_dag(self):
        """Test that reduction preserves reachability."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")

        original_closure = g.transitive_closure()
        reduction = g.transitive_reduction()
        reduction_closure = reduction.transitive_closure()

        # Both closures should have the same number of edges
        assert original_closure.count_edges() == reduction_closure.count_edges()

    def test_round_trip_complex_dag(self):
        """Test round trip on more complex DAG."""
        g = ec.Graph(directed=True)

        for i in range(1, 6):
            g.add_node(label=str(i))

        # Chain: 1 -> 2 -> 3 -> 4 -> 5
        for i in range(1, 5):
            g.add_edge(str(i), str(i + 1))

        # Add some redundant edges
        g.add_edge("1", "3")
        g.add_edge("1", "5")
        g.add_edge("2", "4")

        original_closure = g.transitive_closure()
        reduction = g.transitive_reduction()
        reduction_closure = reduction.transitive_closure()

        assert original_closure.count_edges() == reduction_closure.count_edges()
