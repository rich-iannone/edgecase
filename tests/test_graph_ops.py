import collections
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_BUILD_EDGES, KIND_MUTATE, KIND_TRANSFORM


class TestGraphUnion:
    """Tests for graph union operation."""

    def test_union_basic(self):
        """Test basic union of two graphs."""
        # Graph 1: path 1-2-3
        g1 = Graph(directed=False)
        g1.add_path_graph(3)

        # Graph 2: uses same node IDs, adds node 4
        g2 = Graph(directed=False)
        for i in range(1, 5):
            g2.add_node()
        g2.add_edge(2, 3)
        g2.add_edge(3, 4)

        result = g1.union(g2)

        # Should have 4 nodes
        assert result.count_nodes() == 4

        # Edges: 1-2, 2-3 (shared), 3-4 = 3 edges
        assert result.count_edges() == 3

    def test_union_disjoint(self):
        """Test union of non-overlapping graphs."""
        g1 = Graph(directed=False)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=False)
        g2.add_node()  # Node 1
        g2.add_node()  # Node 2
        g2.add_node()  # Node 3
        g2.add_node()  # Node 4
        g2.add_edge(3, 4)

        result = g1.union(g2)

        assert result.count_nodes() == 4
        assert result.count_edges() == 2

    def test_union_identical(self):
        """Test union of identical graphs."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        result = g.union(g)

        # Should be same as original
        assert result.count_nodes() == g.count_nodes()
        assert result.count_edges() == g.count_edges()

    def test_union_preserves_labels(self):
        """Test that union preserves node labels from first graph."""
        g1 = Graph(directed=False)
        g1.add_node(label="Alice")
        g1.add_node(label="Bob")
        g1.add_edge("Alice", "Bob")

        g2 = Graph(directed=False)
        g2.add_node(label="Alice2")  # Same node ID, different label
        g2.add_node(label="Carol")
        g2.add_edge("Alice2", "Carol")

        result = g1.union(g2)

        # Node 1 should have Alice (from g1), not Alice2
        assert result.get_label_from_id(1) == "Alice"

    def test_union_directed(self):
        """Test union with directed graphs."""
        g1 = Graph(directed=True)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=True)
        g2.add_node(label="A")
        g2.add_node(label="B")
        g2.add_edge("B", "A")  # Reverse direction

        result = g1.union(g2)

        # Should have both directed edges
        assert result.count_edges() == 2
        assert result.is_directed

    def test_union_incompatible_error(self):
        """Test error when unioning directed and undirected graphs."""
        g1 = Graph(directed=True)
        g1.add_path_graph(3)

        g2 = Graph(directed=False)
        g2.add_path_graph(3)

        with pytest.raises(ValueError):
            g1.union(g2)

    def test_union_keeps_node_attributes_calling_graph_wins(self):
        """Union merges node attributes; on a clash the calling graph wins."""
        g1 = Graph()
        g2 = Graph()
        for g in (g1, g2):
            for label in "ABC":
                g.add_node(label=label)
            g.add_edge("A", "B")
        g2.add_edge("B", "C")
        g1.set_node_attrs("x", {1: 1})
        g1.set_node_attrs("shared", {1: "g1"})
        g2.set_node_attrs("y", {1: 2})
        g2.set_node_attrs("shared", {1: "g2"})
        result = g1.union(g2)
        assert result.get_node_attrs("x") == {1: 1}
        assert result.get_node_attrs("y") == {1: 2}
        assert result.get_node_attrs("shared") == {1: "g1"}

    def test_union_result_does_not_share_attribute_dicts(self):
        """Changing the result's attributes leaves the inputs untouched."""
        g1 = Graph()
        g2 = Graph()
        for g in (g1, g2):
            for label in "ABC":
                g.add_node(label=label)
            g.add_edge("A", "B")
        g2.add_edge("B", "C")
        g1.set_node_attrs("x", {1: 1})
        g1.set_node_attrs("shared", {1: "g1"})
        g2.set_node_attrs("y", {1: 2})
        g2.set_node_attrs("shared", {1: "g2"})
        result = g1.union(g2)
        result.set_node_attrs("x", {1: 99})
        assert g1.get_node_attrs("x") == {1: 1}


class TestGraphIntersection:
    """Tests for graph intersection operation."""

    def test_intersection_basic(self):
        """Test basic intersection of two graphs."""
        # Triangle: 1-2, 2-3, 3-1
        g1 = Graph(directed=False)

        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        # Square: 1-2, 2-3, 3-4, 4-1
        g2 = Graph(directed=False)

        for i in range(1, 5):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(3, 4)
        g2.add_edge(4, 1)

        result = g1.intersection(g2)

        # Common edges: 1-2, 2-3
        assert result.count_edges() == 2
        # Only nodes 1, 2, 3 (endpoints of common edges)
        assert result.count_nodes() == 3

    def test_intersection_empty(self):
        """Test intersection with no common edges."""
        g1 = Graph(directed=False)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=False)
        g2.add_node()  # Node 1
        g2.add_node()  # Node 2
        g2.add_node()  # Node 3
        g2.add_node()  # Node 4
        g2.add_edge(3, 4)

        result = g1.intersection(g2)

        assert result.count_edges() == 0
        assert result.count_nodes() == 0

    def test_intersection_identical(self):
        """Test intersection of identical graphs."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        result = g.intersection(g)

        # Should be same as original
        assert result.count_edges() == g.count_edges()

    def test_intersection_directed(self):
        """Test intersection with directed graphs."""
        g1 = Graph(directed=True)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")
        g1.add_edge("B", "A")

        g2 = Graph(directed=True)
        g2.add_node(label="A")
        g2.add_node(label="B")
        g2.add_edge("A", "B")  # Only this direction

        result = g1.intersection(g2)

        # Only A->B is common
        assert result.count_edges() == 1

    def test_intersection_keeps_attributes_calling_graph_wins(self):
        """Intersection merges attributes from both graphs; the calling graph wins clashes."""
        g1 = Graph()
        for label in "ABC":
            g1.add_node(label=label)
        g1.add_edge("A", "B")
        g1.add_edge("B", "C")
        g2 = Graph()
        for label in "ABCD":
            g2.add_node(label=label)
        g2.add_edge("A", "B")
        g2.add_edge("C", "D")
        for g, tag in ((g1, "g1"), (g2, "g2")):
            g.set_node_attrs("src", {nid: tag for nid in g.get_node_ids()})
            g.set_edge_attrs("tag", tag)
        g2.set_node_attrs("only2", {1: True})
        result = g1.intersection(g2)
        assert set(result.get_node_attrs("src").values()) == {"g1"}
        assert result.get_node_attrs("only2") == {1: True}
        assert set(result.get_edge_attrs("tag").values()) == {"g1"}


class TestGraphDifference:
    """Tests for graph difference operation."""

    def test_difference_basic(self):
        """Test basic difference of two graphs."""
        # Triangle: 1-2, 2-3, 3-1
        g1 = Graph(directed=False)

        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        # Square: 1-2, 2-3, 3-4, 4-1
        g2 = Graph(directed=False)

        for i in range(1, 5):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(3, 4)
        g2.add_edge(4, 1)

        result = g1.difference(g2)

        # Edge 3-1 is in g1 but not g2
        assert result.count_edges() == 1

    def test_difference_asymmetric(self):
        """Test that difference is not symmetric."""
        g1 = Graph(directed=False)
        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)

        g2 = Graph(directed=False)
        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(2, 3)

        diff1 = g1.difference(g2)  # 1-2 only
        diff2 = g2.difference(g1)  # Nothing

        assert diff1.count_edges() == 1
        assert diff2.count_edges() == 0

    def test_difference_identical(self):
        """Test difference of identical graphs is empty."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        result = g.difference(g)

        assert result.count_edges() == 0

    def test_difference_directed(self):
        """Test difference with directed graphs."""
        g1 = Graph(directed=True)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")
        g1.add_edge("B", "A")

        g2 = Graph(directed=True)
        g2.add_node(label="A")
        g2.add_node(label="B")
        g2.add_edge("A", "B")

        result = g1.difference(g2)

        # Only B->A is in g1 but not g2
        assert result.count_edges() == 1

    def test_difference_keeps_calling_graph_attributes(self):
        """Difference keeps the calling graph's node and edge attributes."""
        g1 = Graph()
        for label in "ABC":
            g1.add_node(label=label)
        g1.add_edge("A", "B")
        g1.add_edge("B", "C")
        g2 = Graph()
        for label in "ABCD":
            g2.add_node(label=label)
        g2.add_edge("A", "B")
        g2.add_edge("C", "D")
        for g, tag in ((g1, "g1"), (g2, "g2")):
            g.set_node_attrs("src", {nid: tag for nid in g.get_node_ids()})
            g.set_edge_attrs("tag", tag)
        g2.set_node_attrs("only2", {1: True})
        result = g1.difference(g2)
        assert set(result.get_edge_attrs("tag").values()) == {"g1"}
        assert set(result.get_node_attrs("src").values()) == {"g1"}


class TestGraphSymmetricDifference:
    """Tests for graph symmetric difference operation."""

    def test_symmetric_difference_basic(self):
        """Test basic symmetric difference."""
        # Triangle: 1-2, 2-3, 3-1
        g1 = Graph(directed=False)

        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        # Path: 1-2, 2-3
        g2 = Graph(directed=False)

        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)

        result = g1.symmetric_difference(g2)

        # Edge 3-1 is in g1 only
        assert result.count_edges() == 1

    def test_symmetric_difference_is_symmetric(self):
        """Test that symmetric difference is commutative."""
        g1 = Graph(directed=False)

        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        g2 = Graph(directed=False)

        for i in range(1, 5):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(3, 4)
        g2.add_edge(4, 1)

        sym1 = g1.symmetric_difference(g2)
        sym2 = g2.symmetric_difference(g1)

        # Should have same number of edges
        assert sym1.count_edges() == sym2.count_edges()

    def test_symmetric_difference_identical(self):
        """Test symmetric difference of identical graphs is empty."""
        g = Graph(directed=False)
        g.add_path_graph(3)

        result = g.symmetric_difference(g)

        assert result.count_edges() == 0

    def test_symmetric_difference_edges_keep_their_own_attributes(self):
        """Each surviving edge keeps the attributes it had in its own graph."""
        g1 = Graph()
        for label in "ABC":
            g1.add_node(label=label)
        g1.add_edge("A", "B")
        g1.add_edge("B", "C")
        g2 = Graph()
        for label in "ABCD":
            g2.add_node(label=label)
        g2.add_edge("A", "B")
        g2.add_edge("C", "D")
        for g, tag in ((g1, "g1"), (g2, "g2")):
            g.set_node_attrs("src", {nid: tag for nid in g.get_node_ids()})
            g.set_edge_attrs("tag", tag)
        g2.set_node_attrs("only2", {1: True})
        result = g1.symmetric_difference(g2)
        lab = result.get_label_from_id
        tags = {
            tuple(sorted((lab(s), lab(t)))): result.get_edge_attrs("tag")[e]
            for e, s, t in result.get_edges_as_tuples()
        }
        assert tags == {("B", "C"): "g1", ("C", "D"): "g2"}


class TestGraphCompose:
    """Tests for graph compose operation."""

    def test_compose_basic(self):
        """Test basic composition."""
        g1 = Graph(directed=False)
        g1.add_node(label="Alice")
        g1.add_node(label="Bob")
        g1.add_edge("Alice", "Bob")

        g2 = Graph(directed=False)
        g2.add_node(label="Updated_Alice")  # Node 1 - will override Alice
        g2.add_node(label="Carol")  # Node 2 - will override Bob

        result = g1.compose(g2)

        # Should have 2 nodes (same IDs)
        assert result.count_nodes() == 2

        # Node 1 should have g2's label (Updated_Alice)
        assert result.get_label_from_id(1) == "Updated_Alice"

        # Node 2 should have g2's label (Carol)
        assert result.get_label_from_id(2) == "Carol"

    def test_compose_edges(self):
        """Test that compose combines edges."""
        g1 = Graph(directed=False)
        g1.add_path_graph(3)

        g2 = Graph(directed=False)
        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(1, 3)  # New edge

        result = g1.compose(g2)

        # Original 2 edges + new edge
        assert result.count_edges() == 3

    def test_compose_keeps_node_attributes_other_graph_wins(self):
        """Compose merges node attributes; on a clash the other graph wins."""
        g1 = Graph()
        g2 = Graph()
        for g in (g1, g2):
            for label in "ABC":
                g.add_node(label=label)
            g.add_edge("A", "B")
        g2.add_edge("B", "C")
        g1.set_node_attrs("x", {1: 1})
        g1.set_node_attrs("shared", {1: "g1"})
        g2.set_node_attrs("y", {1: 2})
        g2.set_node_attrs("shared", {1: "g2"})
        result = g1.compose(g2)
        assert result.get_node_attrs("x") == {1: 1}
        assert result.get_node_attrs("shared") == {1: "g2"}


class TestGraphDisjointUnion:
    """Tests for graph disjoint union operation."""

    def test_disjoint_union_basic(self):
        """Test basic disjoint union."""
        g1 = Graph(directed=False)

        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        g2 = Graph(directed=False)

        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(3, 1)

        result = g1.disjoint_union(g2)

        # Should have 6 nodes (3 + 3, no overlap)
        assert result.count_nodes() == 6

        # Should have 6 edges (3 + 3)
        assert result.count_edges() == 6

    def test_disjoint_union_preserves_labels(self):
        """Test that labels are preserved."""
        g1 = Graph(directed=False)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=False)
        g2.add_node(label="C")
        g2.add_node(label="D")
        g2.add_edge("C", "D")

        result = g1.disjoint_union(g2)

        # All labels should be present
        labels = [result.get_label_from_id(nid) for nid in result.get_node_ids()]

        assert "A" in labels
        assert "B" in labels
        assert "C" in labels
        assert "D" in labels

    def test_disjoint_union_multiple(self):
        """Test chaining multiple disjoint unions."""
        g1 = Graph(directed=False)
        g1.add_star_graph(3)  # 3 nodes

        g2 = Graph(directed=False)
        g2.add_cycle_graph(4)  # 4 nodes

        g3 = Graph(directed=False)
        g3.add_path_graph(3)  # 3 nodes

        # Chain: (g1 disjoint g2) disjoint g3
        result = g1.disjoint_union(g2).disjoint_union(g3)

        # Star: 3 nodes, Cycle: 4 nodes, Path: 3 nodes = 10 nodes
        assert result.count_nodes() == 10

    def test_disjoint_union_directed(self):
        """Test disjoint union with directed graphs."""
        g1 = Graph(directed=True)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=True)
        g2.add_node(label="C")
        g2.add_node(label="D")
        g2.add_edge("C", "D")

        result = g1.disjoint_union(g2)

        assert result.is_directed
        assert result.count_edges() == 2

    def test_disjoint_union_keeps_attributes(self):
        """Disjoint union carries node attributes from both graphs."""
        g1 = Graph()
        g2 = Graph()
        for g in (g1, g2):
            for label in "ABC":
                g.add_node(label=label)
            g.add_edge("A", "B")
        g2.add_edge("B", "C")
        g1.set_node_attrs("x", {1: 1})
        g1.set_node_attrs("shared", {1: "g1"})
        g2.set_node_attrs("y", {1: 2})
        g2.set_node_attrs("shared", {1: "g2"})
        result = g1.disjoint_union(g2)
        assert result.get_node_attrs("x") == {1: 1}
        assert sorted(result.get_node_attrs("y").values()) == [2]


class TestGraphAlgebraWeighted:
    """Tests for graph algebra with weighted graphs."""

    def test_union_weighted(self):
        """Test that union handles weighted graphs."""
        g1 = Graph(directed=False, weighted=True)
        g1.add_node()  # Node 1
        g1.add_node()  # Node 2
        g1.add_edge(1, 2, weight=1.5)

        g2 = Graph(directed=False, weighted=True)
        g2.add_node()  # Node 1
        g2.add_node()  # Node 2
        g2.add_node()  # Node 3
        g2.add_edge(2, 3, weight=2.5)

        result = g1.union(g2)

        assert result.is_weighted
        assert result.count_edges() == 2

    def test_intersection_preserves_weights(self):
        """Test that intersection preserves edge weights from first graph."""
        g1 = Graph(directed=False, weighted=True)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B", weight=1.5)

        g2 = Graph(directed=False, weighted=True)
        g2.add_node(label="A")
        g2.add_node(label="B")
        g2.add_edge("A", "B", weight=2.5)

        result = g1.intersection(g2)

        assert result.count_edges() == 1

        # Should have g1's weight
        edges = result.get_edges_as_tuples()

        assert len(edges) == 1


# Node relabeling.


class TestRelabelNodes:
    def _graph(self):
        g = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("b", "c")])
        g.set_node_attrs("score", {1: 0.5})
        return g

    def test_dict_by_label_and_id_keeps_ids_attrs_edges(self):
        g = self._graph().relabel_nodes({"a": "alpha", 2: "beta"})
        assert g.get_node_df()["label"].to_list() == ["alpha", "beta", "c"]
        assert g.get_id_from_label("alpha") == 1
        assert g.get_node_attrs("score") == {1: 0.5}
        assert g.get_edges_str_repr(use_labels=True) == ["alpha--beta", "beta--c"]
        assert not g.has_node("a")

    def test_callable_and_swap(self):
        g = self._graph().relabel_nodes(str.upper)
        assert g.get_node_df()["label"].to_list() == ["A", "B", "C"]
        g.relabel_nodes({"A": "B", "B": "A"})
        assert g.get_id_from_label("B") == 1 and g.get_id_from_label("A") == 2

    def test_collisions_are_atomic(self):
        g = self._graph()
        with pytest.raises(ValueError, match="already used"):
            g.relabel_nodes({"a": "c"})
        with pytest.raises(ValueError, match="both be labeled"):
            g.relabel_nodes({"a": "x", "b": "x"})
        assert g.get_node_df()["label"].to_list() == ["a", "b", "c"]  # unchanged

    def test_type_and_existence_errors(self):
        g = self._graph()
        with pytest.raises(TypeError, match="strings"):
            g.relabel_nodes({"a": 5})
        with pytest.raises(ValueError):
            g.relabel_nodes({"zzz": "q"})
        with pytest.raises(TypeError, match="dict or a callable"):
            g.relabel_nodes([("a", "b")])

    def test_step_records_rename_for_the_player(self):
        g = self._graph().relabel_nodes({"a": "alpha"})
        step = g.steps[-1]
        assert step.kind == KIND_MUTATE
        assert step.delta.nodes_removed == ["a"] and step.delta.nodes_added == ["alpha"]
        assert step.delta.edges_removed == [("a", "b")]
        assert step.delta.edges_added == [("alpha", "b")]
        assert step.narration == "relabel_nodes() renamed a → alpha."
        html = g.play()._html  # the player can replay across the rename
        assert '"alpha"' in html
        n = len(g.steps)
        g.relabel_nodes({"alpha": "alpha"})  # no-op rename records nothing
        assert len(g.steps) == n


# Graph products, powers, full joins and n-ary operators, validated against NetworkX (product
# nodes are compared through their "(g, h)" labels).


def _nx_build(G, weighted=False):
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
    return g


def _edge_counter(g):
    out = collections.Counter()
    for _, s, t in g.get_edges_as_tuples():
        a, b = g.get_label_from_id(s), g.get_label_from_id(t)
        out[(a, b) if g.is_directed() else tuple(sorted((a, b)))] += 1
    return out


def _nx_counter(P, directed):
    def label(x):
        return f"({x[0]}, {x[1]})" if isinstance(x, tuple) else str(x)

    out = collections.Counter()
    for a, b in P.edges():
        a, b = label(a), label(b)
        out[(a, b) if directed else tuple(sorted((a, b)))] += 1
    return out


class TestGraphProducts:
    PRODUCTS = {
        "cartesian_product": nx.cartesian_product,
        "tensor_product": nx.tensor_product,
        "strong_product": nx.strong_product,
        "lexicographic_product": nx.lexicographic_product,
        "corona_product": nx.corona_product,
        "modular_product": nx.modular_product,
    }

    def test_against_networkx(self):
        for seed in range(80):
            rnd = random.Random(seed)
            directed = seed % 3 == 0
            G = nx.gnp_random_graph(rnd.randint(1, 5), 0.5, seed=seed, directed=directed)
            H = nx.gnp_random_graph(rnd.randint(1, 4), 0.5, seed=seed + 7, directed=directed)
            H = nx.relabel_nodes(H, {i: "abcd"[i] for i in H})
            if seed % 5 == 0 and len(G):
                G.add_edge(0, 0)
            g, h = _nx_build(G), _nx_build(H)
            for method, ref in self.PRODUCTS.items():
                if directed and method in ("corona_product", "modular_product"):
                    with pytest.raises(ValueError, match="undirected"):
                        getattr(g, method)(h)
                    continue
                product = getattr(g, method)(h)
                expected = ref(G, H)
                assert product.count_nodes() == expected.number_of_nodes()
                assert _edge_counter(product) == _nx_counter(expected, directed)
            if len(H) and not directed:
                root = sorted(H)[0]
                assert _edge_counter(g.rooted_product(h, root)) == _nx_counter(
                    nx.rooted_product(G, H, root), False
                )

    def test_multigraph_factors_keep_parallel_edges(self):
        G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
        H = nx.MultiGraph([("a", "b"), ("b", "c"), ("b", "c")])
        g, h = _nx_build(G), _nx_build(H)
        for method, ref in list(self.PRODUCTS.items())[:4]:
            assert getattr(g, method)(h).count_edges() == ref(G, H).number_of_edges()

    def test_power_against_networkx(self):
        for seed in range(40):
            G = nx.gnp_random_graph(random.Random(seed).randint(1, 9), 0.3, seed=seed)
            g = _nx_build(G)
            for k in (1, 2, 3):
                assert _edge_counter(g.power(k)) == _nx_counter(nx.power(G, k), False)
        with pytest.raises(ValueError, match="positive"):
            ec.Graph().add_path_graph(3).power(0)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).power(2)

    def test_numbering_attributes_and_weights(self):
        a = ec.Graph(weighted=True).add_nodes([("A", {"c": "x"}), "B"])
        a.add_edge("A", "B", weight=2.0)
        b = ec.Graph(weighted=True).add_nodes(["p", "q"]).add_edge("p", "q", weight=3.0)
        cart = a.cartesian_product(b)
        assert [cart.get_label_from_id(i) for i in range(1, 5)] == [
            "(A, p)",
            "(A, q)",
            "(B, p)",
            "(B, q)",
        ]
        assert cart.get_node_attrs("c")[1] == ("x", None)
        assert sorted(cart.get_edge(e).weight for e in cart.get_edge_ids()) == [2, 2, 3, 3]
        tensor = a.tensor_product(b)
        assert {tensor.get_edge(e).weight for e in tensor.get_edge_ids()} == {6.0}
        with pytest.raises(ValueError, match="directed"):
            a.cartesian_product(ec.Graph(directed=True).add_path_graph(2))
        with pytest.raises(ValueError, match="not found"):
            a.rooted_product(b, root="nope")

    def test_examples(self):
        edge = ec.Graph().add_nodes(["0", "1"]).add_edge("0", "1")
        cube = edge.cartesian_product(edge).cartesian_product(edge)
        assert cube.count_edges() == 12 and cube.is_k_regular(3)
        triangle = ec.Graph().add_cycle_graph(3)
        assert triangle.tensor_product(edge).is_isomorphic(ec.Graph().add_cycle_graph(6))
        p3 = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("b", "c")])
        assert p3.strong_product(p3).count_edges() == 20
        comb = ec.Graph().add_path_graph(4).rooted_product(edge, root="0")
        assert comb.is_tree() and comb.count_nodes() == 8
        sunlet = triangle.corona_product(ec.Graph().add_nodes(["x"]))
        assert sunlet.count_nodes() == 6 and sunlet.count_edges() == 6


class TestJoinsAndNaryOperators:
    def test_full_join(self):
        wheel = ec.Graph().add_cycle_graph(5).full_join(ec.Graph().add_nodes(["hub"]))
        assert wheel.count_edges() == 10 and wheel.get_degree("hub") == 5
        k23 = ec.Graph().add_nodes(["a", "b"]).full_join(ec.Graph().add_nodes(["x", "y", "z"]))
        assert k23.count_edges() == 6 and k23.is_bipartite()
        d = (
            ec.Graph(directed=True)
            .add_nodes(["a"])
            .full_join(ec.Graph(directed=True).add_nodes(["b", "c"]))
        )
        assert d.count_edges() == 4
        G = nx.full_join(nx.cycle_graph(4), nx.path_graph(3), rename=("g", "h"))
        g = ec.Graph().add_cycle_graph(4).full_join(ec.Graph().add_path_graph(3))
        assert g.count_edges() == G.number_of_edges()
        with pytest.raises(ValueError, match="directedness"):
            ec.Graph().full_join(ec.Graph(directed=True))

    def test_nary_operators(self):
        def snapshot(pairs):
            g = ec.Graph()
            for _ in range(4):
                g.add_node()
            return g.add_edges(pairs)

        days = [snapshot([(1, 2), (2, 3)]), snapshot([(1, 2), (3, 4)]), snapshot([(1, 2), (2, 3)])]
        assert ec.Graph.union_all(days).count_edges() == 3
        assert ec.Graph.compose_all(days).count_edges() == 3
        assert ec.Graph.intersection_all(days).count_edges() == 1
        combined = ec.Graph.disjoint_union_all(days)
        assert combined.count_nodes() == 12 and combined.count_edges() == 6
        single = ec.Graph.union_all([days[0]])
        assert single is not days[0] and single.count_edges() == 2
        with pytest.raises(ValueError, match="at least one"):
            ec.Graph.union_all([])
        with pytest.raises(TypeError, match="Graph"):
            ec.Graph.union_all([days[0], "nope"])

    def test_steps(self):
        edge = ec.Graph().add_nodes(["0", "1"]).add_edge("0", "1")
        square = edge.cartesian_product(edge)
        assert edge.steps[-1].kind == KIND_TRANSFORM
        assert edge.steps[-1].narration == "cartesian_product(): 4 nodes and 4 edges."
        assert square.steps[-1].kind == KIND_BUILD_EDGES
        path = ec.Graph().add_path_graph(4)
        path.power(2)
        assert path.steps[-1].narration == "power(2): 2 edges added for pairs within distance 2."
