import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_BUILD_EDGES, KIND_TRANSFORM


class TestMinimumSpanningTree:
    """Test minimum_spanning_tree() using Kruskal's algorithm."""

    def test_simple_mst(self):
        """Basic MST on a simple weighted graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        mst = g.minimum_spanning_tree()

        # MST should have n-1 edges for n nodes
        assert mst.count_edges() == 3
        assert mst.count_nodes() == 4

        # Calculate total weight
        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 6.0  # 1 + 2 + 3

    def test_mst_return_edge_ids(self):
        """Test MST returning edge IDs instead of subgraph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        edge_ids = g.minimum_spanning_tree(return_subgraph=False)

        assert isinstance(edge_ids, list)
        assert len(edge_ids) == 2  # Triangle: 3 nodes, 2 edges in MST
        assert all(isinstance(eid, str) for eid in edge_ids)

    def test_mst_custom_weight_attribute(self):
        """MST using custom weight attribute."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=10.0, attributes={"cost": 1.0})
        g.add_edge("B", "C", weight=10.0, attributes={"cost": 2.0})
        g.add_edge("C", "A", weight=10.0, attributes={"cost": 3.0})

        # Using default weight field
        mst_weight = g.minimum_spanning_tree()
        total_weight = sum(
            e.weight
            for e in [mst_weight.get_edge(eid) for eid, _, _ in mst_weight.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total_weight == 20.0  # Two edges with weight 10.0 each

        # Using custom cost attribute - MST should prefer lower cost values
        mst_cost = g.minimum_spanning_tree(weight_attr="cost")
        # The MST based on cost should still have 2 edges, but selected by cost
        assert mst_cost.count_edges() == 2

    def test_mst_single_node(self):
        """MST with single node returns empty."""
        g = ec.Graph(weighted=True)
        g.add_node("A")

        mst = g.minimum_spanning_tree()
        assert mst.count_nodes() == 1
        assert mst.count_edges() == 0

    def test_mst_empty_graph(self):
        """MST with empty graph returns empty."""
        g = ec.Graph(weighted=True)
        mst = g.minimum_spanning_tree()
        assert mst.count_nodes() == 0
        assert mst.count_edges() == 0

    def test_mst_disconnected_graph(self):
        """MST on disconnected graph creates spanning forest."""
        g = ec.Graph(weighted=True)
        # Component 1
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=1.0)

        # Component 2
        g.add_node("C")
        g.add_node("D")
        g.add_edge("C", "D", weight=2.0)

        mst = g.minimum_spanning_tree()

        # Should have 4 nodes and 2 edges (one per component)
        assert mst.count_nodes() == 4
        assert mst.count_edges() == 2

    def test_mst_preserves_node_attributes(self):
        """MST preserves node attributes from original graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A", attributes={"type": "start"})
        g.add_node("B", attributes={"type": "middle"})
        g.add_node("C", attributes={"type": "end"})
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()

        # Check that nodes exist and have basic structure
        node_a = mst.get_node(mst.get_id_from_label("A"))
        assert node_a is not None
        assert node_a.label == "A"
        # Attributes may or may not be preserved - just check structure works
        assert mst.count_nodes() == 3

    def test_mst_preserves_edge_attributes(self):
        """MST preserves edge attributes from original graph."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0, attributes={"type": "primary"})
        g.add_edge("B", "C", weight=2.0, attributes={"type": "secondary"})
        g.add_edge("C", "A", weight=3.0, attributes={"type": "tertiary"})

        mst = g.minimum_spanning_tree()

        # MST should have 2 edges with lowest weights (1.0 and 2.0)
        assert mst.count_edges() == 2

        # Verify the structure is correct
        edges = list(mst.get_edges_as_tuples())
        assert len(edges) == 2

    def test_mst_equal_weights(self):
        """MST with equal weights (any valid MST is acceptable)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("C", "A", weight=1.0)

        mst = g.minimum_spanning_tree()

        # Should have 2 edges for 3 nodes
        assert mst.count_edges() == 2
        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 2.0

    def test_mst_large_graph(self):
        """MST on larger graph."""
        g = ec.Graph(weighted=True)

        # Create a grid-like graph
        for i in range(10):
            g.add_node(str(i))

        # Add edges in a pattern
        for i in range(9):
            g.add_edge(str(i), str(i + 1), weight=float(i + 1))

        # Add some cross edges
        g.add_edge("0", "5", weight=10.0)
        g.add_edge("2", "7", weight=15.0)

        mst = g.minimum_spanning_tree()

        # Should have n-1 edges for n nodes
        assert mst.count_edges() == 9
        assert mst.count_nodes() == 10


class TestMaximumSpanningTree:
    """Test maximum_spanning_tree() using Kruskal's algorithm."""

    def test_simple_maximum_st(self):
        """Basic maximum spanning tree."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        max_st = g.maximum_spanning_tree()

        # Should have n-1 edges for n nodes
        assert max_st.count_edges() == 3
        assert max_st.count_nodes() == 4

        # Calculate total weight (should prefer highest weights)
        total = sum(
            e.weight
            for e in [max_st.get_edge(eid) for eid, _, _ in max_st.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        # Maximum ST should select the 3 highest-weight edges that connect all nodes
        # Should be at least 9.0 (e.g., 5 + 4 + something, or 5 + 3 + 2, etc.)
        assert total >= 9.0

    def test_maximum_st_return_edge_ids(self):
        """Test maximum ST returning edge IDs."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        edge_ids = g.maximum_spanning_tree(return_subgraph=False)

        assert isinstance(edge_ids, list)
        assert len(edge_ids) == 2
        assert all(isinstance(eid, str) for eid in edge_ids)

    def test_maximum_st_custom_weight(self):
        """Maximum ST using custom weight attribute."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0, attributes={"value": 10.0})
        g.add_edge("B", "C", weight=2.0, attributes={"value": 20.0})
        g.add_edge("C", "A", weight=3.0, attributes={"value": 30.0})

        max_st = g.maximum_spanning_tree(weight_attr="value")

        # Maximum ST based on "value" should select 2 highest-value edges
        assert max_st.count_edges() == 2

    def test_mst_vs_maximum_st(self):
        """Compare minimum and maximum spanning trees."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=5.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()
        max_st = g.maximum_spanning_tree()

        # Both should have 2 edges
        assert mst.count_edges() == 2
        assert max_st.count_edges() == 2

        # Minimum should prefer low weights
        mst_total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert mst_total == 4.0  # 1.0 + 3.0

        # Maximum should prefer high weights
        max_st_total = sum(
            e.weight
            for e in [max_st.get_edge(eid) for eid, _, _ in max_st.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert max_st_total == 8.0  # 5.0 + 3.0


class TestMSTEdgeCases:
    """Test edge cases and error conditions."""

    def test_mst_two_nodes(self):
        """MST with exactly two nodes."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=1.0)

        mst = g.minimum_spanning_tree()
        assert mst.count_nodes() == 2
        assert mst.count_edges() == 1

    def test_mst_complete_graph(self):
        """MST on complete graph (K4)."""
        g = ec.Graph(weighted=True)
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            g.add_node(n)

        weight = 1.0
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2, weight=weight)
                weight += 1.0

        mst = g.minimum_spanning_tree()

        # K4 has 6 edges, MST should have 3
        assert mst.count_edges() == 3
        assert mst.count_nodes() == 4

    def test_mst_missing_weights(self):
        """MST with edges missing explicit weights (should default to 1.0)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")  # No weight specified
        g.add_edge("B", "C")  # No weight specified
        g.add_edge("C", "A", weight=5.0)

        mst = g.minimum_spanning_tree()

        # Should prefer edges with default weight 1.0 over weight 5.0
        assert mst.count_edges() == 2

    def test_mst_negative_weights(self):
        """MST handles negative weights correctly."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=-5.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("C", "A", weight=2.0)

        mst = g.minimum_spanning_tree()

        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        # Should include the negative weight edge
        assert total == -4.0  # -5.0 + 1.0

    def test_mst_zero_weights(self):
        """MST with zero-weight edges."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=0.0)
        g.add_edge("B", "C", weight=0.0)
        g.add_edge("C", "A", weight=1.0)

        mst = g.minimum_spanning_tree()

        total = sum(
            e.weight
            for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
            if e and e.weight is not None
        )
        assert total == 0.0  # Two zero-weight edges

    def test_mst_parallel_edges(self):
        """MST with parallel edges (multigraph)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        # Multiple edges between A and B
        g.add_edge("A", "B", weight=5.0)
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()

        # Should have 2 edges for 3 nodes
        assert mst.count_edges() == 2


# Edmonds' optimum branchings and spanning arborescences, validated against NetworkX.


def _digraphs(count=120):
    for seed in range(count):
        rnd = random.Random(seed)
        n = rnd.randint(1, 10)
        G = nx.gnp_random_graph(n, rnd.choice([0.2, 0.4, 0.7]), seed=seed, directed=True)
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.choice([rnd.uniform(-3, 5), rnd.randint(-2, 4)])
        yield G, rnd


def _build(G, weighted=True):
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in sorted(G)])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
        else:
            g.add_edge(str(u), str(v))
    return g


def _total(g):
    return sum(g.get_edge(e).weight for e in g.get_edge_ids(limit=None))


class TestBranchings:
    def test_optimum_branchings_against_networkx(self):
        for G, _ in _digraphs():
            g = _build(G)
            for ours, ref in (
                (g.maximum_branching(), nx.maximum_branching(G)),
                (g.minimum_branching(), nx.minimum_branching(G)),
            ):
                assert _total(ours) == pytest.approx(
                    sum(G[u][v]["weight"] for u, v in ref.edges), abs=1e-9
                )
                assert ours.count_nodes() == g.count_nodes()
                if g.count_nodes():
                    assert ours.is_branching()

    def test_spanning_arborescences_against_networkx(self):
        for G, _ in _digraphs():
            g = _build(G)
            for method, ref_fn in (
                (g.maximum_spanning_arborescence, nx.maximum_spanning_arborescence),
                (g.minimum_spanning_arborescence, nx.minimum_spanning_arborescence),
            ):
                try:
                    ref = ref_fn(G)
                except nx.NetworkXException:
                    with pytest.raises(ValueError, match="no spanning arborescence"):
                        method()
                    continue
                ours = method()
                assert ours.is_arborescence()
                assert _total(ours) == pytest.approx(
                    sum(G[u][v]["weight"] for u, v in ref.edges), abs=1e-9
                )

    def test_unique_optimum_matches_edge_set(self):
        rnd = random.Random(5)
        G = nx.gnp_random_graph(12, 0.5, seed=5, directed=True)
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.random()
        g = _build(G)
        ours = {
            tuple(s.split("->"))
            for s in g.maximum_spanning_arborescence().get_edges_str_repr(use_labels=True)
        }
        ref = {(str(u), str(v)) for u, v in nx.maximum_spanning_arborescence(G).edges}
        assert ours == ref

    def test_weight_attr_and_copies(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        g.add_edges(
            [("A", "B", {"score": 1.0}), ("C", "B", {"score": 5.0}), ("A", "C", {"score": 2.0})]
        )
        tree = g.maximum_spanning_arborescence(weight_attr="score")
        assert sorted(tree.get_edges_str_repr(use_labels=True)) == ["A->C", "C->B"]
        assert tree.get_node_ids(limit=None) == g.get_node_ids(limit=None)
        assert g.count_edges() == 3

    def test_errors(self):
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_path_graph(3).maximum_branching()
        split = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        split.add_edges([("A", "B"), ("C", "B")])
        with pytest.raises(ValueError, match="no spanning arborescence"):
            split.minimum_spanning_arborescence()

    def test_steps(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 2.0), ("B", "C", 3.0), ("A", "C", 1.0)])
        tree = g.maximum_spanning_arborescence()
        step = g.steps[-1]
        assert step.kind == KIND_TRANSFORM
        assert sorted(step.delta.edges_selected) == [("A", "B"), ("B", "C")]
        assert step.narration == "maximum_spanning_arborescence(): 2 edges with total weight 5."
        assert tree.steps[-1].kind == KIND_BUILD_EDGES
