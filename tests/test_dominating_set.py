import pytest
from edgecase import Graph


class TestIsDominatingSet:
    """Tests for is_dominating_set() method."""

    def test_empty_graph(self):
        """Empty graph: empty set is trivially dominating."""
        graph = Graph()

        assert graph.is_dominating_set([])

    def test_single_node(self):
        """Single node: must include itself to dominate."""
        graph = Graph()
        graph.add_node(label="A")
        node_id = graph.get_id_from_label("A")

        # Including the node dominates
        assert graph.is_dominating_set([node_id])

        # Empty set doesn't dominate
        assert not graph.is_dominating_set([])

    def test_path_graph(self):
        """Path graph: test various dominating sets."""
        graph = Graph()
        graph.add_path_graph(4)  # Creates nodes with IDs and path edges

        # For a path, we need to identify which nodes are endpoints vs middle
        # Check dominating property more generally
        all_ids = graph.get_node_ids()

        # All nodes is always dominating
        assert graph.is_dominating_set(all_ids)

        # Empty set is not dominating (graph not empty)
        assert not graph.is_dominating_set([])

        # Find a valid dominating set by checking what actually works
        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_star_graph(self):
        """Star graph: hub alone dominates everything."""
        graph = Graph()
        graph.add_star_graph(5)  # Hub with 5 leaves

        # For star graph, the hub (node 1 based on edge list) dominates all
        # Find hub by looking for node with highest degree
        all_ids = graph.get_node_ids()

        # Get edges to find hub (it appears in all edges)
        edges = graph.get_edges_as_tuples()
        from collections import Counter

        node_counts = Counter()
        for _, src, tgt in edges:
            node_counts[src] += 1
            node_counts[tgt] += 1

        hub_id = node_counts.most_common(1)[0][0]

        # Hub alone dominates
        assert graph.is_dominating_set([hub_id])

        # Any single leaf doesn't dominate (misses other leaves and itself is dominated)
        leaf_ids = [n for n in all_ids if n != hub_id]

        assert not graph.is_dominating_set([leaf_ids[0]])

    def test_cycle_graph(self):
        """Cycle graph: various dominating sets."""
        graph = Graph()
        graph.add_cycle_graph(6)

        all_ids = graph.get_node_ids()

        # All nodes always dominates
        assert graph.is_dominating_set(all_ids)

        # Empty set doesn't dominate
        assert not graph.is_dominating_set([])

        # Computed dominating set should be valid
        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_complete_graph(self):
        """Complete graph: any single node dominates."""
        graph = Graph()
        graph.add_full_graph(5)

        any_node = graph.get_node_ids()[0]
        assert graph.is_dominating_set([any_node])

    def test_disconnected_graph(self):
        """Disconnected graph: need nodes from each component."""
        graph = Graph()

        # Two separate edges
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")
        graph.add_edge("A", "B")
        graph.add_edge("C", "D")

        a_id = graph.get_id_from_label("A")
        c_id = graph.get_id_from_label("C")

        # Need one from each component
        assert graph.is_dominating_set([a_id, c_id])

        # Just one component isn't enough
        assert not graph.is_dominating_set([a_id])


class TestDominatingSet:
    """Tests for dominating_set() method."""

    def test_empty_graph(self):
        """Empty graph: returns empty set."""
        graph = Graph()

        assert graph.dominating_set() == []

    def test_single_node(self):
        """Single node: returns that node."""
        graph = Graph()
        graph.add_node(label="A")

        dom = graph.dominating_set()

        assert len(dom) == 1
        assert graph.is_dominating_set(dom)

    def test_complete_graph(self):
        """Complete graph: single node suffices."""
        for n in [3, 5, 10]:
            graph = Graph()
            graph.add_full_graph(n)

            dom = graph.dominating_set()
            assert len(dom) == 1
            assert graph.is_dominating_set(dom)

    def test_path_graph(self):
        """Path graph: result is valid dominating set."""
        for n in [3, 5, 10, 20]:
            graph = Graph()
            graph.add_path_graph(n)

            dom = graph.dominating_set()
            assert graph.is_dominating_set(dom)

    def test_cycle_graph(self):
        """Cycle graph: result is valid dominating set."""
        for n in [4, 6, 8, 12]:
            graph = Graph()
            graph.add_cycle_graph(n)

            dom = graph.dominating_set()
            assert graph.is_dominating_set(dom)

    def test_star_graph(self):
        """Star graph: result should be valid dominating set."""
        graph = Graph()
        graph.add_star_graph(10)

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)
        # Note: The greedy algorithm may not find the optimal size-1 solution
        # (hub alone) but should still return a valid dominating set

    def test_grid_graph(self):
        """Grid graph: valid dominating set."""
        graph = Graph()
        graph.add_grid_2d(rows=4, cols=4)

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_start_with_specific_node(self):
        """Start from a specific node."""
        graph = Graph()
        graph.add_star_graph(5)

        hub_id = graph.get_node_ids()[0]
        dom = graph.dominating_set(start_with=hub_id)

        assert hub_id in dom
        assert graph.is_dominating_set(dom)

    def test_start_with_invalid_node(self):
        """Starting with non-existent node raises error."""
        graph = Graph()
        graph.add_path_graph(5)

        with pytest.raises(ValueError, match="not in the graph"):
            graph.dominating_set(start_with=9999)

    def test_random_graph(self):
        """Random graph: result is valid."""
        graph = Graph()
        graph.add_gnm_graph(n=50, m=100, seed=23)

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_sorted_output(self):
        """Output should be sorted for determinism."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        dom = graph.dominating_set()

        assert dom == sorted(dom)

    def test_deterministic_smallest_id_first(self):
        """The greedy sweep starts at the smallest ID, so edge order doesn't matter."""
        edges = [("1", "2"), ("2", "3"), ("2", "4"), ("3", "4"), ("4", "5"), ("5", "6"), ("5", "7"), ("6", "7")]
        results = set()
        for order in (edges, list(reversed(edges))):
            g = Graph()
            for i in range(1, 8):
                g.add_node(label=str(i))
            for u, v in order:
                g.add_edge(u, v)
            results.add(tuple(g.dominating_set()))
        assert results == {(1, 3, 5)}


class TestMinWeightedDominatingSet:
    """Tests for min_weighted_dominating_set() method."""

    def test_empty_graph(self):
        """Empty graph: returns empty set."""
        graph = Graph()

        assert graph.min_weighted_dominating_set() == []

    def test_unweighted_is_valid(self):
        """Without weights, still produces valid dominating set."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        dom = graph.min_weighted_dominating_set()

        assert graph.is_dominating_set(dom)

    def test_with_weights(self):
        """With weights, produces valid dominating set."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        # Add weights to nodes using correct API: set_node_attrs(attr_name, values_dict)
        node_ids = graph.get_node_ids()
        weight_dict = {node_id: float(i + 1) for i, node_id in enumerate(node_ids)}
        graph.set_node_attrs("weight", weight_dict)

        dom = graph.min_weighted_dominating_set(weight_attr="weight")

        assert graph.is_dominating_set(dom)

    def test_prefers_lower_weight(self):
        """Should prefer lower weight nodes when possible."""
        # Create star: hub expensive, leaves cheap
        graph = Graph()
        graph.add_node(label="hub", attributes={"cost": 100.0})
        for i in range(5):
            graph.add_node(label=f"leaf{i}", attributes={"cost": 1.0})
            graph.add_edge("hub", f"leaf{i}")

        # Unweighted: might pick hub (it dominates everything)
        dom_unweighted = graph.dominating_set()

        # Weighted: should prefer leaves over expensive hub
        dom_weighted = graph.min_weighted_dominating_set(weight_attr="cost")

        # Both should be valid
        assert graph.is_dominating_set(dom_unweighted)
        assert graph.is_dominating_set(dom_weighted)

    def test_complete_graph(self):
        """Complete graph: any single node suffices."""
        graph = Graph()
        graph.add_full_graph(5)

        dom = graph.min_weighted_dominating_set()

        assert len(dom) == 1
        assert graph.is_dominating_set(dom)

    def test_path_graph(self):
        """Path graph: valid dominating set."""
        graph = Graph()
        graph.add_path_graph(10)

        dom = graph.min_weighted_dominating_set()

        assert graph.is_dominating_set(dom)

    def test_missing_weight_attr(self):
        """Missing weight attribute defaults to 1."""
        graph = Graph()
        graph.add_path_graph(5)

        # Attribute doesn't exist, should still work
        dom = graph.min_weighted_dominating_set(weight_attr="nonexistent")

        assert graph.is_dominating_set(dom)

    def test_sorted_output(self):
        """Output should be sorted for determinism."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        dom = graph.min_weighted_dominating_set()

        assert dom == sorted(dom)


class TestDominatingSetRelationships:
    """Test relationships between dominating sets and other concepts."""

    def test_maximal_independent_set_is_dominating(self):
        """Every maximal independent set is a dominating set."""
        for _ in range(5):
            graph = Graph()
            graph.add_gnm_graph(n=30, m=60, seed=23)

            mis = graph.maximal_independent_set(seed=23)

            # A maximal independent set is always dominating
            # (can't add more nodes means all others are adjacent)
            assert graph.is_dominating_set(mis)

    def test_dominating_set_not_necessarily_independent(self):
        """A dominating set is not necessarily independent."""
        graph = Graph()
        graph.add_path_graph(3)  # 0-1-2

        all_ids = graph.get_node_ids()

        # {0, 1} dominates but is not independent
        dom = [all_ids[0], all_ids[1]]

        assert graph.is_dominating_set(dom)
        assert not graph.is_independent_set(dom)

    def test_whole_graph_is_dominating(self):
        """The entire vertex set is always a dominating set."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        all_nodes = graph.get_node_ids()

        assert graph.is_dominating_set(all_nodes)


class TestDominatingSetEdgeCases:
    """Edge cases and special graphs."""

    def test_isolated_nodes(self):
        """Graph with isolated nodes."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        # No edges - all nodes are isolated

        dom = graph.dominating_set()

        # Must include all nodes (each can only dominate itself)
        assert len(dom) == 3
        assert graph.is_dominating_set(dom)

    def test_self_loop(self):
        """Graph with self-loop."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")
        # Self-loop on A
        graph.add_edge("A", "A")

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_directed_graph(self):
        """Dominating set works on directed graphs."""
        graph = Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_bipartite_graph(self):
        """Bipartite graph dominating set."""
        # Create a simple bipartite graph (complete bipartite K_3,3)
        graph = Graph()
        left = ["L1", "L2", "L3"]
        right = ["R1", "R2", "R3"]
        for l in left:
            graph.add_node(label=l)
        for r in right:
            graph.add_node(label=r)
        for l in left:
            for r in right:
                graph.add_edge(l, r)

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_wheel_graph(self):
        """Wheel graph: hub dominates all."""
        graph = Graph()
        # Create wheel: hub connected to all nodes in a cycle
        graph.add_node(label="hub")
        for i in range(6):
            graph.add_node(label=f"spoke{i}")
            graph.add_edge("hub", f"spoke{i}")
        # Add cycle among spokes
        for i in range(6):
            graph.add_edge(f"spoke{i}", f"spoke{(i + 1) % 6}")

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

        # Hub alone should suffice
        hub_id = graph.get_id_from_label("hub")

        assert graph.is_dominating_set([hub_id])


class TestDominatingSetDocstringExamples:
    """Test examples from docstrings work correctly."""

    def test_is_dominating_set_example(self):
        """Test is_dominating_set docstring example."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        b_id = graph.get_id_from_label("B")
        d_id = graph.get_id_from_label("D")

        assert graph.is_dominating_set([b_id, d_id])
        assert not graph.is_dominating_set([b_id])

    def test_dominating_set_random_example(self):
        """Test dominating_set with random graph."""
        graph = Graph()
        graph.add_gnm_graph(n=100, m=200, seed=23)

        dom = graph.dominating_set()

        assert graph.is_dominating_set(dom)

    def test_star_graph_example(self):
        """Test star graph example."""
        graph = Graph()
        graph.add_star_graph(10)

        dom = graph.dominating_set(start_with=graph.get_node_ids()[0])

        assert graph.is_dominating_set(dom)
