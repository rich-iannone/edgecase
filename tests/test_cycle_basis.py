import pytest
from edgecase import Graph


class TestCycleBasisBasic:
    """Basic tests for cycle_basis()."""

    def test_empty_graph(self):
        """Empty graph has no cycles."""
        g = Graph(directed=False)
        basis = g.cycle_basis()

        assert basis == []

    def test_single_node(self):
        """Single node has no cycles."""
        g = Graph(directed=False).add_node(label="A")
        basis = g.cycle_basis()

        assert basis == []

    def test_two_nodes_no_cycle(self):
        """Two connected nodes but no cycle."""
        g = Graph(directed=False).add_node(label="A").add_node(label="B").add_edge("A", "B")
        basis = g.cycle_basis()

        assert basis == []

    def test_triangle(self):
        """Simple triangle has one cycle in basis."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        basis = g.cycle_basis()

        assert len(basis) == 1
        assert len(basis[0]) == 3  # Triangle has 3 nodes

    def test_square(self):
        """Square has one cycle."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )
        basis = g.cycle_basis()

        assert len(basis) == 1
        assert len(basis[0]) == 4  # Square has 4 nodes

    def test_square_with_diagonal(self):
        """Square with diagonal has 2 cycles in basis."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
            .add_edge("A", "C")  # Diagonal
        )
        basis = g.cycle_basis()

        # V=4, E=5, so cyclomatic complexity = E - V + 1 = 5 - 4 + 1 = 2
        assert len(basis) == 2

    def test_directed_graph_raises_error(self):
        """cycle_basis should raise error for directed graphs."""
        g = Graph(directed=True).add_node(label="A")
        with pytest.raises(ValueError, match="undirected"):
            g.cycle_basis()


class TestCycleBasisCyclomaticComplexity:
    """Tests verifying cyclomatic complexity (E - V + C)."""

    def test_single_component_tree(self):
        """Tree has no cycles (E = V - 1)."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("B", "D")
        )

        basis = g.cycle_basis()

        assert len(basis) == 0

    def test_complete_k4(self):
        """Complete graph K4 has C(4,2) - 3 = 3 fundamental cycles."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )
        basis = g.cycle_basis()

        # K4: V=4, E=6, cyclomatic = 6 - 4 + 1 = 3
        assert len(basis) == 3

    def test_two_triangles_sharing_edge(self):
        """Two triangles sharing an edge: 2 cycles."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            .add_edge("B", "D")
            .add_edge("D", "C")
        )
        basis = g.cycle_basis()

        # V=4, E=5, cyclomatic = 5 - 4 + 1 = 2
        assert len(basis) == 2


class TestCycleBasisRoot:
    """Tests for the root parameter."""

    def test_with_root_node(self):
        """Specifying root node should work."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        node_ids = g.get_node_ids()
        node_a = node_ids[0]  # First node
        basis = g.cycle_basis(root=node_a)

        assert len(basis) == 1

    def test_invalid_root_raises_error(self):
        """Invalid root node should raise error."""
        g = Graph(directed=False).add_node(label="A").add_node(label="B").add_edge("A", "B")
        with pytest.raises(ValueError, match="not found"):
            g.cycle_basis(root=99999)


class TestCycleBasisDisconnected:
    """Tests for disconnected graphs."""

    def test_two_disconnected_triangles(self):
        """Two disconnected triangles should have 2 cycles."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            .add_edge("D", "E")
            .add_edge("E", "F")
            .add_edge("F", "D")
        )
        basis = g.cycle_basis()

        # Two components, each with one cycle
        assert len(basis) == 2

    def test_isolated_nodes_plus_cycle(self):
        """Isolated nodes don't affect cycle basis."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="X")  # Isolated
            .add_node(label="Y")  # Isolated
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        basis = g.cycle_basis()

        assert len(basis) == 1


class TestMinimumCycleBasisBasic:
    """Basic tests for minimum_cycle_basis()."""

    def test_empty_graph(self):
        """Empty graph has no cycles."""
        g = Graph(directed=False)

        basis = g.minimum_cycle_basis()

        assert basis == []

    def test_single_node(self):
        """Single node has no cycles."""
        g = Graph(directed=False).add_node(label="A")

        basis = g.minimum_cycle_basis()

        assert basis == []

    def test_triangle(self):
        """Triangle has one cycle in minimum basis."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        basis = g.minimum_cycle_basis()

        assert len(basis) == 1
        assert len(basis[0]) == 3

    def test_directed_graph_raises_error(self):
        """minimum_cycle_basis should raise error for directed graphs."""
        g = Graph(directed=True).add_node(label="A")

        with pytest.raises(ValueError, match="undirected"):
            g.minimum_cycle_basis()


class TestMinimumCycleBasisWeighted:
    """Tests for minimum_cycle_basis with weights."""

    def test_prefers_shorter_cycles(self):
        """Minimum basis should prefer shorter cycles in unweighted graphs."""
        # Create a graph where there are multiple cycle options
        # but the minimum basis should pick the shortest ones
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            .add_edge("B", "D")
            .add_edge("D", "C")
        )
        basis = g.minimum_cycle_basis()

        # Should find 2 cycles (triangles)
        assert len(basis) == 2

        # Total nodes across all cycles should be minimized
        # Two triangles = 3 + 3 = 6 total (better than alternatives)

    def test_with_weight_attribute(self):
        """Minimum basis should use weight attribute when specified."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
            .add_edge("C", "A", weight=3.0)
        )

        basis = g.minimum_cycle_basis(weight="weight")

        assert len(basis) == 1

    def test_weighted_vs_unweighted(self):
        """Weighted minimum basis might differ from unweighted."""
        # Square with one heavy diagonal
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=1.0)
            .add_edge("C", "D", weight=1.0)
            .add_edge("D", "A", weight=1.0)
            .add_edge("A", "C", weight=100.0)  # Heavy diagonal
        )

        basis = g.minimum_cycle_basis(weight="weight")

        assert len(basis) == 2

    def test_weight_named_weight_reads_weight_field(self):
        """`weight="weight"` used to count every edge as 1.0 and pick the two triangles."""
        g = Graph(weighted=True).add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0), ("D", "A", 1.0)])
        g.add_edge("A", "C", weight=100.0)
        basis = g.minimum_cycle_basis(weight="weight")
        assert sorted(map(len, basis)) == [3, 4]
        assert basis == g.minimum_cycle_basis()

    def test_missing_weight_attribute_raises(self):
        """`weight="cots"` used to count every edge as 1.0."""
        g = Graph().add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", {"cost": 1.0}), ("B", "C", {"cost": 1.0}), ("C", "A")])
        with pytest.raises(ValueError, match="no numeric 'cots' attribute, which weight='cots'"):
            g.minimum_cycle_basis(weight="cots")
        with pytest.raises(ValueError, match="no numeric 'cost' attribute"):
            g.minimum_cycle_basis(weight="cost")


class TestMinimumCycleBasisComplexGraphs:
    """Tests on more complex graph structures."""

    def test_k4_minimum_basis(self):
        """K4 minimum basis should have 3 cycles."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )

        basis = g.minimum_cycle_basis()
        # K4: 3 fundamental cycles

        assert len(basis) == 3

    def test_grid_graph(self):
        """2x2 grid should have 1 cycle."""
        g = (
            Graph(directed=False)
            .add_node(label="00")
            .add_node(label="01")
            .add_node(label="10")
            .add_node(label="11")
            .add_edge("00", "01")
            .add_edge("00", "10")
            .add_edge("01", "11")
            .add_edge("10", "11")
        )

        basis = g.minimum_cycle_basis()

        # 2x2 grid: V=4, E=4, cyclomatic = 4 - 4 + 1 = 1
        assert len(basis) == 1

    def test_3x3_grid(self):
        """3x3 grid structure."""
        # Build a 3x3 grid
        g = Graph(directed=False)

        # Add nodes
        for i in range(3):
            for j in range(3):
                g.add_node(label=f"{i}{j}")

        # Add horizontal edges
        for i in range(3):
            for j in range(2):
                g.add_edge(f"{i}{j}", f"{i}{j + 1}")

        # Add vertical edges
        for i in range(2):
            for j in range(3):
                g.add_edge(f"{i}{j}", f"{i + 1}{j}")

        basis = g.minimum_cycle_basis()

        # 3x3 grid: V=9, E=12, cyclomatic = 12 - 9 + 1 = 4
        assert len(basis) == 4


class TestMinimumCycleBasisCorrectness:
    """Size and total weight of the basis, checked against NetworkX."""

    def test_isolated_node_does_not_drop_a_cycle(self):
        """An isolated node made E < V, which used to return no cycles at all."""
        g = Graph().add_nodes(list("ABCDEFG"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D")])
        assert g.minimum_cycle_basis() == [[1, 3, 2], [4, 6, 5]]

    def test_dependent_cycles_are_rejected(self):
        """Used to return 3 cycles, one the sum of two others, instead of 4 independent ones."""
        g = Graph().add_nodes([str(i) for i in range(5)])
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3)]
        g.add_edges([(str(a), str(b)) for a, b in pairs])
        assert g.minimum_cycle_basis() == [[1, 3, 2], [1, 4, 2], [1, 5, 2], [1, 4, 3]]

    def test_self_loops_and_parallel_edges(self):
        """A self-loop is a one-node cycle and a parallel pair a two-node one."""
        g = Graph().add_nodes(list("ABC"))
        g.add_edges([("A", "B"), ("A", "B"), ("B", "C"), ("C", "A"), ("C", "C")])
        assert g.minimum_cycle_basis() == [[3], [1, 2], [1, 3, 2]]

    def test_zero_weights(self):
        """Zero-weight edges still give simple cycles, shortest first."""
        g = Graph(weighted=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B", 0.0), ("B", "C", 0.0), ("C", "A", 0.0), ("A", "D", 0.0)])
        g.add_edge("D", "C", weight=0.0)
        assert g.minimum_cycle_basis() == [[1, 3, 2], [1, 3, 4]]

    @pytest.mark.parametrize("bad", [-1.0, float("nan")])
    def test_negative_or_nan_weight_raises(self, bad):
        """Negative weights have no minimum basis; NaN ones can't be compared."""
        g = Graph().add_nodes(list("ABC"))
        g.add_edge("A", "B", cost=1.0).add_edge("B", "C", cost=bad).add_edge("C", "A", cost=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            g.minimum_cycle_basis(weight="cost")

    @pytest.mark.parametrize("weighted", [False, True])
    def test_matches_networkx(self, weighted):
        """Same number of cycles and same total weight as `nx.minimum_cycle_basis`."""
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(7)
        for seed in range(60):
            n = rng.randint(2, 14)
            G = nx.gnm_random_graph(n, rng.randint(0, 3 * n), seed=seed)
            for u, v in G.edges:
                G[u][v]["weight"] = rng.choice([0, 0.5, 1, 2, 5]) if weighted else 1
            g = Graph(weighted=weighted).add_nodes([str(i) for i in range(n)])
            for u, v, w in G.edges(data="weight"):
                g.add_edge(str(u), str(v), **({"weight": w} if weighted else {}))
            basis = g.minimum_cycle_basis(weight="weight")

            def total(cycles):
                return sum(G[c[i - 1]][c[i]]["weight"] for c in cycles for i in range(len(c)))

            ours = [[int(g.get_node(x).label) for x in c] for c in basis]
            assert all(c[0] == min(c) for c in basis)
            size = G.number_of_edges() - n + nx.number_connected_components(G)
            assert len(basis) == size
            assert total(ours) == total(nx.minimum_cycle_basis(G, weight="weight"))


class TestCycleBasisComparison:
    """Compare cycle_basis and minimum_cycle_basis."""

    def test_same_count_different_selection(self):
        """Both methods should return same number of cycles."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
            .add_edge("A", "C")
        )

        basis1 = g.cycle_basis()
        basis2 = g.minimum_cycle_basis()

        # Both should have same number of cycles
        assert len(basis1) == len(basis2) == 2

    def test_tree_no_cycles(self):
        """Both methods return empty for trees."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("B", "E")
        )

        assert g.cycle_basis() == []
        assert g.minimum_cycle_basis() == []


class TestCycleBasisEdgeCases:
    """Edge cases and boundary conditions."""

    def test_large_cycle(self):
        """Large simple cycle."""
        n = 10
        g = Graph(directed=False)

        for i in range(n):
            g.add_node(label=str(i))
        for i in range(n):
            g.add_edge(str(i), str((i + 1) % n))

        basis = g.cycle_basis()

        assert len(basis) == 1
        assert len(basis[0]) == n

    def test_wheel_graph(self):
        """Wheel graph: central hub connected to all nodes in a cycle."""
        n = 5  # 5 outer nodes + 1 center
        g = Graph(directed=False)
        g.add_node(label="center")

        for i in range(n):
            g.add_node(label=f"outer_{i}")

        # Connect outer nodes in a cycle
        for i in range(n):
            g.add_edge(f"outer_{i}", f"outer_{(i + 1) % n}")

        # Connect center to all outer nodes
        for i in range(n):
            g.add_edge("center", f"outer_{i}")

        basis = g.cycle_basis()

        # Wheel: V = n+1, E = 2n, cyclomatic = 2n - (n+1) + 1 = n
        assert len(basis) == n

    def test_petersen_like_structure(self):
        """A structure similar to petersen graph."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            # Outer pentagon
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "E")
            .add_edge("E", "A")
        )

        basis = g.cycle_basis()

        assert len(basis) == 1  # Just the pentagon
