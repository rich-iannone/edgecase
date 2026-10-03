import pytest
import math
from edgecase import Graph


class TestConstraintBasic:
    """Basic constraint calculation tests."""

    def test_isolated_vertex(self):
        """Isolated vertices should return None."""
        g = Graph()
        g.add_node(label="A")

        result = g.constraint(return_dict=True)
        assert result[g.get_id_from_label("A")] is None

    def test_single_edge(self):
        """Two connected vertices have maximum constraint (1.0)."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        result = g.constraint(return_dict=True)
        assert result[g.get_id_from_label("A")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("B")] == pytest.approx(1.0)

    def test_simple_broker(self):
        """Broker connecting unconnected nodes has lower constraint."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")  # Broker
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        # A and C not connected - B spans structural hole

        result = g.constraint(return_dict=True)
        a_id, b_id, c_id = (
            g.get_id_from_label("A"),
            g.get_id_from_label("B"),
            g.get_id_from_label("C"),
        )

        # Broker has lower constraint than peripheral nodes
        assert result[b_id] < result[a_id]
        assert result[b_id] < result[c_id]

        # Peripheral nodes have maximum constraint
        assert result[a_id] == pytest.approx(1.0)
        assert result[c_id] == pytest.approx(1.0)

        # Broker constraint: (0.5 + 0)^2 + (0.5 + 0)^2 = 0.5
        assert result[b_id] == pytest.approx(0.5)

    def test_closed_triad(self):
        """Closed triad has high constraint (no structural holes)."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")

        result = g.constraint(return_dict=True)

        # All nodes have higher constraint due to closure
        # Each node sees others are connected
        for node in ["A", "B", "C"]:
            assert result[g.get_id_from_label(node)] > 0.5  # Higher than simple broker


class TestConstraintStructuralHoles:
    """Tests focusing on structural hole identification."""

    def test_star_network_center(self):
        """Star center (hub) with unconnected leaves has low constraint."""
        g = Graph()
        center = "hub"
        g.add_node(label=center)

        # Add 4 peripheral nodes connected only to center
        for i in range(4):
            leaf = f"leaf{i}"
            g.add_node(label=leaf)
            g.add_edge(center, leaf)

        result = g.constraint(return_dict=True)

        # Center has many structural holes
        hub_constraint = result[g.get_id_from_label(center)]

        # Each leaf has maximum constraint (only one connection)
        for i in range(4):
            assert result[g.get_id_from_label(f"leaf{i}")] == pytest.approx(1.0)

        # Hub constraint: each leaf contributes (0.25 + 0)^2 = 0.0625
        # Total: 4 * 0.0625 = 0.25
        assert hub_constraint == pytest.approx(0.25)

    def test_star_network_leaf(self):
        """Star leaf has maximum constraint."""
        g = Graph()
        g.add_node(label="hub")
        g.add_node(label="leaf1")
        g.add_node(label="leaf2")
        g.add_edge("hub", "leaf1")
        g.add_edge("hub", "leaf2")

        result = g.constraint(return_dict=True)

        # Leaves have only one connection
        assert result[g.get_id_from_label("leaf1")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("leaf2")] == pytest.approx(1.0)

    def test_bridge_position(self):
        """Node bridging two complete subgraphs has lower constraint than members."""
        g = Graph()

        # Left clique: A-B-C all connected
        for n in ["A", "B", "C"]:
            g.add_node(label=n)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")

        # Right clique: D-E-F all connected
        for n in ["D", "E", "F"]:
            g.add_node(label=n)
        g.add_edge("D", "E")
        g.add_edge("E", "F")
        g.add_edge("D", "F")

        # Bridge connecting cliques
        g.add_node(label="X")
        g.add_edge("B", "X")
        g.add_edge("X", "E")

        result = g.constraint(return_dict=True)

        # Bridge connects two separate groups
        # B and E are embedded in cliques, so have more indirect connections
        # This actually increases their constraint despite being bridge endpoints
        # The pure bridge node X has exactly 2 connections with no indirect paths
        assert result[g.get_id_from_label("X")] == pytest.approx(0.5)  # Two unconnected neighbors

        # Peripheral nodes in cliques have highest constraint
        assert result[g.get_id_from_label("C")] > result[g.get_id_from_label("B")]
        assert result[g.get_id_from_label("F")] > result[g.get_id_from_label("E")]


class TestConstraintWeighted:
    """Tests for weighted networks."""

    def test_weighted_simple(self):
        """Weighted edges affect constraint calculation."""
        g = Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # Strong tie to B, weak tie to C
        g.add_edge("A", "B", weight=9.0)
        g.add_edge("A", "C", weight=1.0)

        result = g.constraint(return_dict=True)

        # A's constraint dominated by strong tie to B
        # p_AB = 9/10 = 0.9, p_AC = 1/10 = 0.1
        # C_A = 0.9^2 + 0.1^2 = 0.81 + 0.01 = 0.82
        assert result[g.get_id_from_label("A")] == pytest.approx(0.82)

    def test_weighted_with_closure(self):
        """Weighted triad with closure."""
        g = Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        g.add_edge("A", "B", weight=2.0)
        g.add_edge("A", "C", weight=2.0)
        g.add_edge("B", "C", weight=1.0)

        result = g.constraint(return_dict=True)

        # All nodes have positive constraint
        assert all(v > 0 for v in result.values())

        # With closure, constraint increases
        # Each node sees its neighbors are connected


class TestConstraintDirected:
    """Tests for directed networks."""

    def test_directed_out_broker(self):
        """Directed broker with out-edges to unconnected nodes."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # B points to A and C, but they don't connect
        g.add_edge("B", "A")
        g.add_edge("B", "C")

        result = g.constraint(return_dict=True)

        # B has outgoing structural holes
        assert result[g.get_id_from_label("B")] is not None
        assert result[g.get_id_from_label("B")] < 1.0  # Has multiple connections

    def test_directed_in_broker(self):
        """Directed broker with in-edges from unconnected nodes."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # A and C point to B, but they don't connect
        g.add_edge("A", "B")
        g.add_edge("C", "B")

        result = g.constraint(return_dict=True)

        # B receives from multiple sources
        assert result[g.get_id_from_label("B")] is not None

    def test_directed_mutual(self):
        """Directed edges with mutual connections."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")

        # Mutual connection
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        result = g.constraint(return_dict=True)

        # Both should have constraint = 1.0 (only connection to each other)
        assert result[g.get_id_from_label("A")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("B")] == pytest.approx(1.0)


class TestConstraintEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_graph(self):
        """Empty graph returns empty dict."""
        g = Graph()
        result = g.constraint(return_dict=True)
        assert result == {}

    def test_all_isolated(self):
        """Graph with only isolated vertices."""
        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))

        result = g.constraint(return_dict=True)

        # All isolated vertices return None
        assert all(v is None for v in result.values())

    def test_mixed_isolated_and_connected(self):
        """Mix of isolated and connected vertices."""
        g = Graph()

        # Isolated
        g.add_node(label="A")

        # Connected pair
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("B", "C")

        result = g.constraint(return_dict=True)

        assert result[g.get_id_from_label("A")] is None
        assert result[g.get_id_from_label("B")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("C")] == pytest.approx(1.0)

    def test_self_loop(self):
        """Self-loops should not affect constraint."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        result = g.constraint(return_dict=True)

        # Self-loop shouldn't change constraint significantly
        assert result[g.get_id_from_label("A")] is not None
        assert result[g.get_id_from_label("B")] is not None


class TestConstraintInterpretation:
    """Tests verifying interpretation of constraint values."""

    def test_constraint_range(self):
        """Constraint values should be in reasonable range."""
        g = Graph()

        # Create various network positions
        for i in range(10):
            g.add_node(label=str(i))

        # Dense connections
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("0", "2")

        # Sparse connections
        g.add_edge("3", "4")
        g.add_edge("3", "5")

        # Star
        for i in range(6, 10):
            g.add_edge("4", str(i))

        result = g.constraint(return_dict=True)

        # All non-None values should be positive
        for node, value in result.items():
            if value is not None:
                assert value > 0
                # Typically constraint ≤ number of neighbors for well-connected networks

    def test_broker_lower_than_closure(self):
        """Broker positions have lower constraint than closed positions."""
        # Create two scenarios side by side
        g = Graph()

        # Broker position (lower constraint)
        g.add_node(label="broker")
        g.add_node(label="b1")
        g.add_node(label="b2")
        g.add_edge("broker", "b1")
        g.add_edge("broker", "b2")
        # b1 and b2 NOT connected

        # Closed position (higher constraint)
        g.add_node(label="closed")
        g.add_node(label="c1")
        g.add_node(label="c2")
        g.add_edge("closed", "c1")
        g.add_edge("closed", "c2")
        g.add_edge("c1", "c2")  # Closure

        result = g.constraint(return_dict=True)

        # Broker has lower constraint than closed position
        assert result[g.get_id_from_label("broker")] < result[g.get_id_from_label("closed")]


class TestConstraintComplex:
    """Complex network scenarios."""

    def test_karate_club_style(self):
        """Small network inspired by Zachary's karate club."""
        g = Graph()

        # Create two factions with a bridge
        # Faction 1
        for i in range(5):
            g.add_node(label=f"f1_{i}")

        # Dense connections in faction 1
        g.add_edge("f1_0", "f1_1")
        g.add_edge("f1_0", "f1_2")
        g.add_edge("f1_1", "f1_2")
        g.add_edge("f1_3", "f1_0")
        g.add_edge("f1_4", "f1_0")

        # Faction 2
        for i in range(5):
            g.add_node(label=f"f2_{i}")

        # Dense connections in faction 2
        g.add_edge("f2_0", "f2_1")
        g.add_edge("f2_0", "f2_2")
        g.add_edge("f2_1", "f2_2")
        g.add_edge("f2_3", "f2_0")
        g.add_edge("f2_4", "f2_0")

        # Bridge between factions
        g.add_node(label="bridge")
        g.add_edge("bridge", "f1_0")
        g.add_edge("bridge", "f2_0")

        result = g.constraint(return_dict=True)

        # Bridge has two unconnected neighbors (structural hole)
        assert result[g.get_id_from_label("bridge")] == pytest.approx(0.5)

        # Faction leaders have many connections within their groups
        # giving them lower constraint than pure periphery
        assert result[g.get_id_from_label("f1_0")] < result[g.get_id_from_label("f1_4")]
        assert result[g.get_id_from_label("f2_0")] < result[g.get_id_from_label("f2_4")]

        # Peripheral nodes should have high constraint
        assert result[g.get_id_from_label("f1_4")] > result[g.get_id_from_label("f1_0")]
        assert result[g.get_id_from_label("f2_4")] > result[g.get_id_from_label("f2_0")]

    def test_line_graph(self):
        """Chain/path graph constraint."""
        g = Graph()
        nodes = [str(i) for i in range(5)]

        for node in nodes:
            g.add_node(label=node)

        # Create chain: 0-1-2-3-4
        for i in range(4):
            g.add_edge(str(i), str(i + 1))

        result = g.constraint(return_dict=True)

        # End nodes have maximum constraint (one connection)
        assert result[g.get_id_from_label("0")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("4")] == pytest.approx(1.0)

        # Middle nodes have lower constraint (two unconnected neighbors)
        assert result[g.get_id_from_label("2")] < result[g.get_id_from_label("0")]
        assert result[g.get_id_from_label("2")] < result[g.get_id_from_label("4")]

        # Middle nodes have exactly 2 unconnected neighbors
        # Constraint = (0.5 + 0)^2 + (0.5 + 0)^2 = 0.5
        assert result[g.get_id_from_label("2")] == pytest.approx(0.5)


class TestConstraintValidation:
    """Validation and consistency tests."""

    def test_undirected_symmetry(self):
        """In undirected graphs, constraint should use undirected ties."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.constraint(return_dict=True)

        # All values should be positive and reasonable
        assert all(v is not None and v > 0 for v in result.values())

    def test_constraint_formula_simple(self):
        """Verify constraint calculation for simple case."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # A connects to B and C (no B-C edge)
        g.add_edge("A", "B")
        g.add_edge("A", "C")

        result = g.constraint(return_dict=True)

        # For A: two neighbors with equal weights
        # p_AB = 0.5, p_AC = 0.5
        # No indirect connections (B and C not connected)
        # C_A = (0.5 + 0)^2 + (0.5 + 0)^2 = 0.25 + 0.25 = 0.5
        assert result[g.get_id_from_label("A")] == pytest.approx(0.5)

        # For B and C: one connection each
        assert result[g.get_id_from_label("B")] == pytest.approx(1.0)
        assert result[g.get_id_from_label("C")] == pytest.approx(1.0)

    def test_constraint_with_indirect(self):
        """Verify constraint with indirect connections."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # Closed triad
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")

        result = g.constraint(return_dict=True)

        # For A: two neighbors B and C who are connected
        # This creates indirect connections, increasing constraint
        # Closed triad has higher constraint than open triad
        # p_AB = p_AC = 1/2 and each indirect path adds 1/2 * 1/2, so
        # C_A = 2 * (1/2 + 1/4)^2 = 1.125 (Burt 1992; matches networkx.constraint).
        # This used to assert 1.25, the old implementation's (wrong) output.
        assert result[g.get_id_from_label("A")] == pytest.approx(1.125)

        # Verify all nodes have increased constraint due to closure
        assert all(v > 0.5 for v in result.values())


class TestConstraintMatchesNetworkX:
    """constraint() follows Burt's definition exactly as networkx.constraint() computes it."""

    @staticmethod
    def _compare(reference, g, weights=None, weight_key=None):
        nx = pytest.importorskip("networkx")
        expected = nx.constraint(reference, weight=weight_key)
        got = g.constraint(weights=weights, return_dict=True)
        for node, value in expected.items():
            ours = got[g.get_id_from_label(str(node))]
            if set(nx.all_neighbors(reference, node)) <= {node}:
                # No contacts besides itself: undefined. NetworkX gives nan for undirected
                # graphs but a number for directed ones (its matrix path treats the self-loop
                # differently); edgecase returns None for both
                assert ours is None
                continue
            if value != value:  # nan: isolated (or, in NetworkX, a directed node w/o successors)
                if reference.is_directed() and reference.degree(node) > 0:
                    continue
                assert ours is None
            else:
                assert ours == pytest.approx(value, abs=1e-12)

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_random_graphs(self, directed, weighted):
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(1)
        for seed in range(30):
            reference = nx.gnp_random_graph(rng.randint(3, 10), 0.4, seed=seed, directed=directed)
            for v in rng.sample(list(reference), 2):
                reference.add_edge(v, v)  # self-loops too
            g = Graph(directed=directed, weighted=weighted)
            for node in reference:
                g.add_node(label=str(node))
            for u, v in reference.edges():
                if weighted:
                    w = round(rng.uniform(0.5, 3.0), 1)
                    reference[u][v]["weight"] = w
                    g.add_edge(str(u), str(v), weight=w)
                else:
                    g.add_edge(str(u), str(v))
            self._compare(reference, g, weight_key="weight" if weighted else None)

    def test_named_weight_attribute(self):
        nx = pytest.importorskip("networkx")
        reference = nx.Graph()
        reference.add_edge("A", "B", strength=5.0)
        reference.add_edge("A", "C", strength=1.0)
        reference.add_edge("B", "C", strength=3.0)
        g = Graph()
        for label in "ABC":
            g.add_node(label=label)
        for u, v, data in reference.edges(data=True):
            g.add_edge(u, v, strength=data["strength"])

        self._compare(reference, g, weights="strength", weight_key="strength")

    def test_parallel_edges_add_up(self):
        """A pair joined by edges of weight 1 and 2 behaves like one edge of weight 3."""
        multi = Graph(weighted=True)
        simple = Graph(weighted=True)
        for g in (multi, simple):
            for label in "ABCD":
                g.add_node(label=label)
        for s, t, w in [("A", "B", 1.0), ("B", "A", 2.0), ("A", "C", 1.0), ("C", "D", 1.0),
                        ("B", "C", 1.0)]:
            multi.add_edge(s, t, weight=w)
        for s, t, w in [("A", "B", 3.0), ("A", "C", 1.0), ("C", "D", 1.0), ("B", "C", 1.0)]:
            simple.add_edge(s, t, weight=w)

        m = multi.constraint(return_dict=True)
        s = simple.constraint(return_dict=True)
        for label in "ABCD":
            assert m[multi.get_id_from_label(label)] == pytest.approx(
                s[simple.get_id_from_label(label)]
            )

    def test_self_loop_only_node_is_undefined(self):
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")
        assert g.constraint(return_dict=True)[g.get_id_from_label("A")] is None

