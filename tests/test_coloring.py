import pytest
from edgecase import Graph


class TestGreedyColor:
    """Tests for greedy_color method."""

    def test_empty_graph(self):
        """Empty graph should return empty coloring."""
        g = Graph(directed=False)

        coloring = g.greedy_color()

        assert coloring == {}

    def test_single_node(self):
        """Single node should get color 0."""
        g = Graph(directed=False)
        g.add_node(label="A")

        coloring = g.greedy_color()

        assert len(coloring) == 1
        assert list(coloring.values())[0] == 0

    def test_path_graph(self):
        """Path graph should be 2-colorable."""
        g = Graph(directed=False)

        for i in range(5):
            g.add_node(label=f"N{i}")
        for i in range(4):
            g.add_edge(f"N{i}", f"N{i + 1}")

        coloring = g.greedy_color()

        assert len(coloring) == 5
        assert len(set(coloring.values())) == 2  # 2 colors
        assert g.is_valid_coloring(coloring)

    def test_cycle_graph_even(self):
        """Even cycle should be 2-colorable."""
        g = Graph(directed=False)

        n = 6
        for i in range(n):
            g.add_node(label=f"N{i}")
        for i in range(n):
            g.add_edge(f"N{i}", f"N{(i + 1) % n}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 2
        assert g.is_valid_coloring(coloring)

    def test_cycle_graph_odd(self):
        """Odd cycle should need 3 colors."""
        g = Graph(directed=False)

        n = 5
        for i in range(n):
            g.add_node(label=f"N{i}")
        for i in range(n):
            g.add_edge(f"N{i}", f"N{(i + 1) % n}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 3
        assert g.is_valid_coloring(coloring)

    def test_complete_graph_k4(self):
        """Complete K4 should need 4 colors."""
        g = Graph(directed=False)

        for i in range(4):
            g.add_node(label=f"V{i}")
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(f"V{i}", f"V{j}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 4
        assert g.is_valid_coloring(coloring)

    def test_triangle(self):
        """Triangle (K3) should need 3 colors."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 3
        assert g.is_valid_coloring(coloring)

    def test_bipartite_graph(self):
        """Bipartite graph should be 2-colorable."""
        g = Graph(directed=False)

        # Create K_{3,3}
        for i in range(3):
            g.add_node(label=f"L{i}")
            g.add_node(label=f"R{i}")
        for i in range(3):
            for j in range(3):
                g.add_edge(f"L{i}", f"R{j}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 2
        assert g.is_valid_coloring(coloring)

    def test_petersen_graph(self):
        """Petersen graph has chromatic number 3."""
        g = Graph(directed=False)

        for i in range(10):
            g.add_node(label=f"P{i}")

        # Outer cycle
        for i in range(5):
            g.add_edge(f"P{i}", f"P{(i + 1) % 5}")

        # Inner star
        for i in range(5):
            g.add_edge(f"P{i + 5}", f"P{((i + 2) % 5) + 5}")

        # Spokes
        for i in range(5):
            g.add_edge(f"P{i}", f"P{i + 5}")

        # Use dsatur for best results
        coloring = g.greedy_color(strategy="dsatur")

        assert len(set(coloring.values())) == 3
        assert g.is_valid_coloring(coloring)

    def test_disconnected_components(self):
        """Coloring should work on disconnected graphs."""
        g = Graph(directed=False)

        # Two triangles
        for i in range(3):
            g.add_node(label=f"A{i}")
            g.add_node(label=f"B{i}")

        # Triangle A
        g.add_edge("A0", "A1")
        g.add_edge("A1", "A2")
        g.add_edge("A2", "A0")

        # Triangle B
        g.add_edge("B0", "B1")
        g.add_edge("B1", "B2")
        g.add_edge("B2", "B0")

        coloring = g.greedy_color()

        assert len(coloring) == 6
        assert g.is_valid_coloring(coloring)


class TestColoringStrategies:
    """Tests for different coloring strategies."""

    @pytest.fixture
    def star_graph(self):
        """Create a star graph with 6 nodes."""
        g = Graph(directed=False)

        for i in range(6):
            g.add_node(label=f"N{i}")

        # N0 is center
        for i in range(1, 6):
            g.add_edge("N0", f"N{i}")

        return g

    @pytest.mark.parametrize(
        "strategy",
        [
            "largest_first",
            "smallest_last",
            "dsatur",
            "saturation_largest_first",
            "independent_set",
            "random",
            "arbitrary",
        ],
    )
    def test_strategy_produces_valid_coloring(self, star_graph, strategy):
        """All strategies should produce valid colorings."""
        coloring = star_graph.greedy_color(strategy=strategy)

        assert star_graph.is_valid_coloring(coloring)

    @pytest.mark.parametrize(
        "strategy",
        [
            "largest_first",
            "smallest_last",
            "dsatur",
            "independent_set",
            "arbitrary",
        ],
    )
    def test_deterministic_strategies(self, star_graph, strategy):
        """Deterministic strategies should be consistent."""
        coloring1 = star_graph.greedy_color(strategy=strategy)
        coloring2 = star_graph.greedy_color(strategy=strategy)

        assert coloring1 == coloring2

    def test_invalid_strategy_raises(self):
        """Invalid strategy should raise an error."""
        g = Graph(directed=False)
        g.add_node(label="A")

        with pytest.raises(ValueError):
            g.greedy_color(strategy="invalid_strategy")


class TestDirectedGraphColoring:
    """Tests for coloring on directed graphs."""

    def test_directed_path(self):
        """Directed path should be colorable as undirected."""
        g = Graph(directed=True)

        for i in range(4):
            g.add_node(label=f"N{i}")
        for i in range(3):
            g.add_edge(f"N{i}", f"N{i + 1}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 2
        assert g.is_valid_coloring(coloring)

    def test_directed_cycle(self):
        """Directed cycle (even) should be 2-colorable."""
        g = Graph(directed=True)

        n = 4
        for i in range(n):
            g.add_node(label=f"N{i}")
        for i in range(n):
            g.add_edge(f"N{i}", f"N{(i + 1) % n}")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 2
        assert g.is_valid_coloring(coloring)

    def test_bidirectional_edges(self):
        """Bidirectional edges should be handled correctly."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        coloring = g.greedy_color()

        assert len(set(coloring.values())) == 2
        assert g.is_valid_coloring(coloring)


class TestIsValidColoring:
    """Tests for is_valid_coloring method."""

    def test_valid_coloring(self):
        """Valid coloring should return True."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        node_ids = g.get_node_ids()
        coloring = {node_ids[0]: 0, node_ids[1]: 1}

        assert g.is_valid_coloring(coloring) is True

    def test_invalid_coloring(self):
        """Invalid coloring (adjacent same color) should return False."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        node_ids = g.get_node_ids()
        coloring = {node_ids[0]: 0, node_ids[1]: 0}  # Same color!

        assert g.is_valid_coloring(coloring) is False

    def test_partial_coloring(self):
        """Partial coloring (missing nodes) should be checked correctly."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        node_ids = g.get_node_ids()

        # Only color A and B, skip C
        partial = {node_ids[0]: 0, node_ids[1]: 1}

        assert g.is_valid_coloring(partial) is True


class TestChromaticNumberUpperBound:
    """Tests for chromatic_number_upper_bound method."""

    def test_with_provided_coloring(self):
        """Should count colors in provided coloring."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        node_ids = g.get_node_ids()
        coloring = {node_ids[0]: 0, node_ids[1]: 1}

        assert g.chromatic_number_upper_bound(coloring) == 2

    def test_without_coloring(self):
        """Should compute coloring and count colors."""
        g = Graph(directed=False)

        for i in range(3):
            g.add_node(label=f"N{i}")
        g.add_edge("N0", "N1")
        g.add_edge("N1", "N2")
        g.add_edge("N2", "N0")

        bound = g.chromatic_number_upper_bound()

        assert bound == 3

    def test_complete_graph(self):
        """K_n should have chromatic number n."""
        n = 5
        g = Graph(directed=False)

        for i in range(n):
            g.add_node(label=f"V{i}")
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(f"V{i}", f"V{j}")

        bound = g.chromatic_number_upper_bound()

        assert bound == n


class TestSelfLoops:
    """Tests for graphs with self-loops."""

    def test_self_loop_ignored(self):
        """Self-loops should be ignored in coloring."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        coloring = g.greedy_color()

        assert g.is_valid_coloring(coloring)
        assert len(set(coloring.values())) == 2


class TestGreedyColorInterchangeAndSeed:
    """interchange= swaps colors along Kempe chains; strategy="random" honours seed=."""

    @staticmethod
    def _is_proper(g, coloring):
        for edge_id in g.get_edge_ids():
            edge = g.get_edge(edge_id)
            if edge.source != edge.target and coloring[edge.source] == coloring[edge.target]:
                return False
        return True

    @staticmethod
    def _gnm(seed):
        g = Graph()
        g.add_gnm_graph(n=30, m=80, seed=seed)
        return g

    def test_interchange_saves_a_color(self):
        g = self._gnm(23)
        plain = g.greedy_color()
        swapped = g.greedy_color(interchange=True)

        assert len(set(plain.values())) == 5
        assert len(set(swapped.values())) == 4
        assert self._is_proper(g, swapped)

    @pytest.mark.parametrize("strategy", ["largest_first", "smallest_last", "random", "arbitrary"])
    def test_interchange_is_always_proper_and_gap_free(self, strategy):
        import random

        rng = random.Random(3)
        for seed in range(40):
            g = Graph()
            g.add_gnp_graph(n=rng.randint(5, 30), p=rng.random() * 0.5, seed=seed)
            coloring = g.greedy_color(strategy=strategy, interchange=True, seed=seed)

            assert set(coloring) == set(g.get_node_ids())
            assert self._is_proper(g, coloring)
            assert set(coloring.values()) == set(range(len(set(coloring.values()))))

    @pytest.mark.parametrize("strategy", ["dsatur", "saturation_largest_first", "independent_set"])
    def test_interchange_rejected_for_self_ordering_strategies(self, strategy):
        with pytest.raises(ValueError, match="interchange"):
            self._gnm(1).greedy_color(strategy=strategy, interchange=True)

    def test_random_seed_is_reproducible(self):
        g = self._gnm(5)
        assert g.greedy_color(strategy="random", seed=7) == g.greedy_color(
            strategy="random", seed=7
        )

    def test_random_varies_across_seeds_and_without_seed(self):
        g = self._gnm(5)
        seeded = {tuple(sorted(g.greedy_color(strategy="random", seed=s).items())) for s in range(8)}
        unseeded = {tuple(sorted(g.greedy_color(strategy="random").items())) for _ in range(8)}

        assert len(seeded) > 1
        assert len(unseeded) > 1

    def test_random_sequential_is_random_ordering(self):
        g = self._gnm(5)
        assert g.greedy_color(strategy="random_sequential", seed=3) == g.greedy_color(
            strategy="random", seed=3
        )
