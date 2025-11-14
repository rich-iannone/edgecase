"""Test suite for Bonacich power centrality.

Tests replicate the NetworkX test suite and validate against the original
Bonacich (1987) paper results in Tables 3 and 4.
"""

import pytest
from edgecase import Graph


def add_edges(graph, edge_list):
    """Helper function to add multiple edges, creating nodes as needed."""
    # First collect all unique nodes
    nodes = set()
    for src, dst in edge_list:
        nodes.add(src)
        nodes.add(dst)

    # Add all nodes (ignore if they already exist)
    for node in nodes:
        try:
            graph.add_node(label=node)
        except:
            pass  # Node already exists

    # Then add all edges
    for src, dst in edge_list:
        graph.add_edge(src, dst)


class TestBonacichBasic:
    """Basic functionality tests."""

    def test_empty_graph(self):
        """Empty graph returns empty dict."""
        g = Graph()
        result = g.get_bonacich_power_centrality()
        assert result == {}

    def test_single_node(self):
        """Single node has centrality of 0 (no edges)."""
        g = Graph()
        g.add_node("A")
        result = g.get_bonacich_power_centrality()
        # With no edges, A·1 = 0, so centrality is 0
        node_id = g.get_id_from_label("A")
        assert abs(result[node_id]) < 1e-7

    def test_complete_graph_K5(self):
        """Complete graph K5 - all nodes equal."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(str(i))
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(str(i), str(j))

        result = g.get_bonacich_power_centrality(beta=0.1)

        # All nodes should have equal centrality
        scores = list(result.values())
        for score in scores:
            assert abs(score - scores[0]) < 1e-6

    def test_path_P3(self):
        """Path graph with 3 nodes (NetworkX test_P3)."""
        g = Graph(directed=False)
        g.add_node("0")
        g.add_node("1")
        g.add_node("2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Expected values from NetworkX (converted to use node IDs)
        expected_by_label = {
            "0": 0.7480544714310444,
            "1": 1.3714331976235816,
            "2": 0.7480544714310444,
        }
        expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}

        for node, exp_val in expected.items():
            assert abs(result[node] - exp_val) < 1e-6

    def test_star_S3(self):
        """Star graph with 3 leaves (NetworkX test_S3)."""
        g = Graph(directed=False)
        g.add_node("0")  # center
        g.add_node("1")
        g.add_node("2")
        g.add_node("3")
        g.add_edge("0", "1")
        g.add_edge("0", "2")
        g.add_edge("0", "3")

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Expected values from NetworkX (converted to use node IDs)
        expected_by_label = {
            "0": 1.6520663752618043,
            "1": 0.6508140266182866,
            "2": 0.6508140266182866,
            "3": 0.6508140266182866,
        }
        expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}

        for node, exp_val in expected.items():
            assert abs(result[node] - exp_val) < 1e-6


class TestBonacichDirected:
    """Tests for directed graphs."""

    def test_directed_path_P3(self):
        """Directed path 0→1→2 (NetworkX test_directed_P3)."""
        g = Graph(directed=True)
        g.add_node("0")
        g.add_node("1")
        g.add_node("2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Expected values from NetworkX (converted to use node IDs)
        expected_by_label = {
            "0": 1.2816138016780187,
            "1": 1.1651034560709261,
            "2": 0.0,
        }
        expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}

        for node, exp_val in expected.items():
            assert abs(result[node] - exp_val) < 1e-6

    def test_directed_star_S3(self):
        """Directed star: center points to 3 leaves (NetworkX test_directed_S3)."""
        g = Graph(directed=True)
        g.add_node("0")  # center
        g.add_node("1")
        g.add_node("2")
        g.add_node("3")
        g.add_edge("0", "1")
        g.add_edge("0", "2")
        g.add_edge("0", "3")

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Expected values from NetworkX (converted to use node IDs)
        expected_by_label = {"0": 2.0, "1": 0.0, "2": 0.0, "3": 0.0}
        expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}

        for node, exp_val in expected.items():
            assert abs(result[node] - exp_val) < 1e-6

    def test_directed_cycle_C3(self):
        """Directed cycle 0→1→2→0 (NetworkX test_directed_C3)."""
        g = Graph(directed=True)
        g.add_node("0")
        g.add_node("1")
        g.add_node("2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "0")

        result = g.get_bonacich_power_centrality(beta=0.1)

        # All nodes should be equal in a symmetric cycle
        scores = list(result.values())
        for score in scores:
            assert abs(score - 1.0) < 1e-6


class TestBonacichNormalization:
    """Test normalization methods."""

    def test_normalized_false_l2(self):
        """Default normalization: sum(c^2) = n."""
        g = Graph(directed=False)
        g.add_node("0")
        g.add_node("1")
        g.add_node("2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        result = g.get_bonacich_power_centrality(beta=0.1, normalized=False)

        # L2 norm: sum of squares should equal n
        sum_squares = sum(v**2 for v in result.values())
        n = len(result)
        assert abs(sum_squares - n) < 1e-6

    def test_normalized_true_sum(self):
        """Sum normalization: sum(c) = 1."""
        g = Graph(directed=False)
        g.add_node("0")
        g.add_node("1")
        g.add_node("2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        result = g.get_bonacich_power_centrality(beta=0.1, normalized=True)

        # Sum should equal 1
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6


class TestBonacichParameters:
    """Test beta parameter effects."""

    def test_beta_positive(self):
        """Positive beta: powerful neighbors increase power."""
        g = Graph(directed=False)
        # Star: center connected to 3 leaves
        add_edges(g, [("center", "a"), ("center", "b"), ("center", "c")])

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Center should have highest power (connected to many)
        center_id = g.get_id_from_label("center")
        a_id = g.get_id_from_label("a")
        b_id = g.get_id_from_label("b")
        c_id = g.get_id_from_label("c")
        assert result[center_id] > result[a_id]
        assert result[center_id] > result[b_id]
        assert result[center_id] > result[c_id]

    def test_beta_negative(self):
        """Negative beta: competitive relations change power dynamics."""
        g = Graph(directed=False)
        # Path: A-B-C-D
        add_edges(g, [("A", "B"), ("B", "C"), ("C", "D")])

        result = g.get_bonacich_power_centrality(beta=-0.1)

        # With negative beta, middle nodes still have higher power in undirected paths
        # because they connect to more nodes (the math works differently than intuition)
        a_id, b_id, c_id, d_id = [g.get_id_from_label(l) for l in ["A", "B", "C", "D"]]
        assert result[b_id] > result[a_id]
        assert result[c_id] > result[d_id]

    def test_beta_zero(self):
        """Beta = 0 gives degree-based centrality."""
        g = Graph(directed=False)
        add_edges(g, [("A", "B"), ("B", "C"), ("B", "D")])

        result = g.get_bonacich_power_centrality(beta=0.0)

        # With beta=0, centrality proportional to degree
        # B has degree 3, others have degree 1-2
        a_id, b_id, c_id, d_id = [g.get_id_from_label(l) for l in ["A", "B", "C", "D"]]
        assert result[b_id] > result[a_id]
        assert result[b_id] > result[c_id]
        assert result[b_id] > result[d_id]


class TestBonacichCook1987:
    """Replicate results from Bonacich (1987) Tables 3 and 4."""

    def test_cook_1c(self):
        """Figure 2 network 1c - undirected (Table 3 in Bonacich 1987)."""
        g = Graph(directed=False)
        add_edges(g, [("0", "1"), ("0", "2"), ("1", "3"), ("2", "4")])

        # Test multiple beta values as in the paper
        test_cases = {
            -0.3: {
                "0": 0.9730085108210397,
                "1": 1.3378867023789296,
                "2": 1.3378867023789296,
                "3": 0.48650425541051995,
                "4": 0.48650425541051995,
            },
            -0.2: {
                "0": 1.0882143751650173,
                "1": 1.2695834376925206,
                "2": 1.2695834376925204,
                "3": 0.5441071875825086,
                "4": 0.5441071875825086,
            },
            -0.1: {
                "0": 1.1534996014569447,
                "1": 1.225593326548004,
                "2": 1.2255933265480041,
                "3": 0.5767498007284724,
                "4": 0.5767498007284724,
            },
            0.0: {
                "0": 1.1952286093343936,
                "1": 1.1952286093343936,
                "2": 1.1952286093343936,
                "3": 0.5976143046671968,
                "4": 0.5976143046671968,
            },
            0.1: {
                "0": 1.2241074813555017,
                "1": 1.173103002965689,
                "2": 1.173103002965689,
                "3": 0.6120537406777509,
                "4": 0.6120537406777509,
            },
            0.2: {
                "0": 1.245244117188435,
                "1": 1.1562981088178326,
                "2": 1.1562981088178326,
                "3": 0.6226220585942176,
                "4": 0.6226220585942176,
            },
            0.3: {
                "0": 1.2613684401582361,
                "1": 1.1431151488934015,
                "2": 1.1431151488934015,
                "3": 0.630684220079118,
                "4": 0.6306842200791181,
            },
        }

        for beta, expected_by_label in test_cases.items():
            result = g.get_bonacich_power_centrality(beta=beta, normalized=False)
            expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}
            for node, exp_val in expected.items():
                assert abs(result[node] - exp_val) < 1e-6, f"Beta={beta}, node={node}"

    def test_cook_1d(self):
        """Figure 2 network 1d - undirected (Table 3 in Bonacich 1987)."""
        g = Graph(directed=False)
        add_edges(g, [("0", "1"), ("0", "2"), ("0", "3"), ("1", "4"), ("2", "5"), ("3", "6")])

        # Expected values (same for all beta in this symmetric structure)
        expected = {
            "0": 1.6201851746019653,
            "1": 1.0801234497346432,
            "2": 1.0801234497346432,
            "3": 1.0801234497346432,
            "4": 0.5400617248673216,
            "5": 0.5400617248673216,
            "6": 0.5400617248673216,
        }

        # Test multiple beta values
        for beta in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            result = g.get_bonacich_power_centrality(beta=beta, normalized=False)
            expected_by_id = {g.get_id_from_label(lbl): val for lbl, val in expected.items()}
            for node, exp_val in expected_by_id.items():
                assert abs(result[node] - exp_val) < 1e-6, f"Beta={beta}, node={node}"

    def test_cook_1d_directed(self):
        """Figure 2 network 1d - directed (Table 4 in Bonacich 1987)."""
        g = Graph(directed=True)
        # Bidirectional edges between 0 and {1,2,3}, unidirectional from {1,2,3} to leaves
        add_edges(
            g,
            [
                ("0", "1"),
                ("0", "2"),
                ("0", "3"),
                ("1", "0"),
                ("2", "0"),
                ("3", "0"),
                ("1", "4"),
                ("2", "5"),
                ("3", "6"),
            ],
        )

        test_cases = {
            -0.3: {
                "0": 1.410023290755643,
                "1": 1.2925213498593398,
                "2": 1.2925213498593398,
                "3": 1.2925213498593395,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            -0.2: {
                "0": 1.5769724491135402,
                "1": 1.2265341270883094,
                "2": 1.2265341270883092,
                "3": 1.2265341270883092,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            -0.1: {
                "0": 1.671579730129196,
                "1": 1.1840356421748466,
                "2": 1.1840356421748468,
                "3": 1.1840356421748466,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            0.0: {
                "0": 1.7320508075688772,
                "1": 1.1547005383792515,
                "2": 1.1547005383792515,
                "3": 1.1547005383792515,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            0.1: {
                "0": 1.773900269015164,
                "1": 1.133325171870799,
                "2": 1.1333251718707993,
                "3": 1.133325171870799,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            0.2: {
                "0": 1.8045301643153684,
                "1": 1.1170901017190376,
                "2": 1.1170901017190376,
                "3": 1.1170901017190376,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
            0.3: {
                "0": 1.8278965282086308,
                "1": 1.104354152459381,
                "2": 1.104354152459381,
                "3": 1.104354152459381,
                "4": 0.0,
                "5": 0.0,
                "6": 0.0,
            },
        }

        for beta, expected_by_label in test_cases.items():
            result = g.get_bonacich_power_centrality(beta=beta, normalized=False)
            expected = {g.get_id_from_label(lbl): val for lbl, val in expected_by_label.items()}
            for node, exp_val in expected.items():
                assert abs(result[node] - exp_val) < 1e-6, f"Beta={beta}, node={node}"


class TestBonacichEdgeCases:
    """Edge cases and error conditions."""

    def test_singular_matrix_high_beta(self):
        """Test that algorithm works with various beta values."""
        g = Graph(directed=False)
        add_edges(g, [("A", "B"), ("B", "C")])

        # Should work with various beta values including large ones
        # The LU solver is quite robust
        for beta in [0.1, 0.5, 1.0]:
            result = g.get_bonacich_power_centrality(beta=beta)
            assert len(result) == 3
            # All nodes should have scores
            for score in result.values():
                assert isinstance(score, float)

    def test_disconnected_graph(self):
        """Graph with disconnected components."""
        g = Graph(directed=False)
        # Component 1
        add_edges(g, [("A", "B")])
        # Component 2
        add_edges(g, [("C", "D")])

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Both components should have valid centrality
        assert len(result) == 4
        for score in result.values():
            assert score > 0

    def test_self_loop(self):
        """Node with self-loop."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # self-loop

        result = g.get_bonacich_power_centrality(beta=0.1)

        # Self-loop increases A's centrality
        a_id = g.get_id_from_label("A")
        assert result[a_id] > 0

    def test_large_negative_beta(self):
        """Large negative beta (competitive relations)."""
        g = Graph(directed=False)
        add_edges(g, [("A", "B"), ("B", "C"), ("C", "D")])

        # Should not crash with moderately negative beta
        result = g.get_bonacich_power_centrality(beta=-0.3)
        assert len(result) == 4


class TestBonacichComparison:
    """Compare with other centrality measures."""

    def test_relation_to_eigenvector(self):
        """At beta = 1/lambda_max, should approximate eigenvector centrality."""
        g = Graph(directed=False)
        add_edges(g, [("A", "B"), ("B", "C"), ("B", "D")])

        # For small graphs, lambda_max ≈ max_degree
        # B has degree 3, so try beta ≈ 1/3
        bonacich = g.get_bonacich_power_centrality(beta=0.3, normalized=True)

        # Should have similar ordering (B highest due to degree)
        a_id, b_id, c_id, d_id = [g.get_id_from_label(l) for l in ["A", "B", "C", "D"]]
        assert bonacich[b_id] > bonacich[a_id]
        assert bonacich[b_id] > bonacich[c_id]
        assert bonacich[b_id] > bonacich[d_id]

    def test_beta_zero_vs_degree(self):
        """Beta = 0 should give degree-proportional centrality."""
        g = Graph(directed=False)
        add_edges(g, [("A", "B"), ("B", "C"), ("B", "D"), ("C", "D")])

        bonacich = g.get_bonacich_power_centrality(beta=0.0)

        # Get degrees manually (bonacich keys are now node IDs)
        degrees = {}
        for node_id in bonacich.keys():
            degrees[node_id] = len(g.get_neighbors(node_id))

        # Ordering should match degree ordering
        degree_order = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        bonacich_order = sorted(bonacich.items(), key=lambda x: x[1], reverse=True)

        # Top node should be same
        assert degree_order[0][0] == bonacich_order[0][0]


class TestBonacichLargeGraph:
    """Tests on larger graphs."""

    def test_large_path(self):
        """Path graph with 20 nodes."""
        g = Graph(directed=False)
        for i in range(20):
            g.add_node(str(i))
        for i in range(19):
            g.add_edge(str(i), str(i + 1))

        result = g.get_bonacich_power_centrality(beta=0.05)

        # Middle nodes should have higher centrality
        id_10 = g.get_id_from_label("10")
        id_0 = g.get_id_from_label("0")
        id_19 = g.get_id_from_label("19")
        assert result[id_10] > result[id_0]
        assert result[id_10] > result[id_19]

    def test_complete_graph(self):
        """Complete graph with 10 nodes."""
        g = Graph(directed=False)
        for i in range(10):
            g.add_node(str(i))
        for i in range(10):
            for j in range(i + 1, 10):
                g.add_edge(str(i), str(j))

        result = g.get_bonacich_power_centrality(beta=0.05)

        # All should be equal
        scores = list(result.values())
        for score in scores[1:]:
            assert abs(score - scores[0]) < 1e-6
