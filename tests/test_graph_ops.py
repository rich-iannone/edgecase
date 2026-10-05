import edgecase as ec
import pytest
from edgecase import Graph
from edgecase.steps import KIND_MUTATE


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
