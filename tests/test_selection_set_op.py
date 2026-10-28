"""Tests for `set_op=` on the `select_*` methods, and for holding one kind of selection."""

import edgecase as ec
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _graph():
    """A -> B -> C -> D -> E plus A -> C. Node IDs 1 to 5, edge IDs 1 to 5."""
    g = ec.Graph(directed=True).add_nodes(["A", "B", "C", "D", "E"])
    return g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("A", "C")])


NODE_SELECTORS = {
    "select_nodes": lambda g, **k: g.select_nodes(lambda n: n.label in "ABC", **k),
    "select_nodes_by_id": lambda g, **k: g.select_nodes_by_id([1, 2, 3], **k),
    "select_last_nodes_created": lambda g, **k: g.select_last_nodes_created(3, **k),
    "select_nodes_in_neighborhood": lambda g, **k: g.select_nodes_in_neighborhood("A", **k),
    "select_nodes_by_degree": lambda g, **k: g.select_nodes_by_degree(min_degree=3, **k),
    "select_nodes_by_centrality": lambda g, **k: g.select_nodes_by_centrality(
        "degree", k=2, include_ties=True, **k
    ),
}

EDGE_SELECTORS = {
    "select_edges": lambda g, **k: g.select_edges(lambda e: e.source == 1, **k),
    "select_edges_by_id": lambda g, **k: g.select_edges_by_id([1, 2], **k),
    "select_edges_by_node": lambda g, **k: g.select_edges_by_node(["C"], **k),
    "select_last_edges_created": lambda g, **k: g.select_last_edges_created(2, **k),
}

COMBINE = {
    "replace": lambda prev, new: new,
    "union": lambda prev, new: prev | new,
    "intersect": lambda prev, new: prev & new,
    "difference": lambda prev, new: prev - new,
}


def _nodes(g):
    return set(g.get_selected_node_ids())


def _edges(g):
    return set(g.get_selected_edge_ids())


# =============================================================================
# The set operations
# =============================================================================


class TestSetOps:
    @pytest.mark.parametrize("set_op", list(COMBINE))
    @pytest.mark.parametrize("name", list(NODE_SELECTORS))
    def test_node_selectors(self, name, set_op):
        select = NODE_SELECTORS[name]
        new = _nodes(select(_graph()))
        assert new  # each selector picks something on its own
        g = _graph().select_nodes_by_id([1, 4])
        select(g, set_op=set_op)
        assert _nodes(g) == COMBINE[set_op]({1, 4}, new)
        assert _edges(g) == set()

    @pytest.mark.parametrize("set_op", list(COMBINE))
    @pytest.mark.parametrize("name", list(EDGE_SELECTORS))
    def test_edge_selectors(self, name, set_op):
        select = EDGE_SELECTORS[name]
        new = _edges(select(_graph()))
        assert new
        g = _graph().select_edges_by_id([1, 4])
        select(g, set_op=set_op)
        assert _edges(g) == COMBINE[set_op]({1, 4}, new)
        assert _nodes(g) == set()

    def test_union_is_the_default(self):
        g = _graph().select_nodes_by_id([1, 2]).select_nodes_by_id([3])
        assert _nodes(g) == {1, 2, 3}
        g.select_edges_by_id([1]).select_edges_by_id([2])
        assert _edges(g) == {1, 2}

    def test_chained_rules(self):
        """Each set operation in turn, on the selection the last call left."""
        g = (
            _graph()
            .select_nodes_by_degree(min_degree=2)  # A, B, C, D
            .select_nodes(lambda n: n.label != "B", set_op="intersect")  # A, C, D
            .select_nodes_by_id([5], set_op="union")  # A, C, D, E
            .select_nodes_by_id([1], set_op="difference")  # C, D, E
        )
        assert _nodes(g) == {3, 4, 5}

    def test_empty_list_with_replace_clears(self):
        g = _graph().select_nodes_by_id([1, 2]).select_nodes_by_id([], set_op="replace")
        assert _nodes(g) == set()

    def test_empty_list_with_union_keeps(self):
        g = _graph().select_nodes_by_id([1, 2]).select_nodes_by_id([])
        assert _nodes(g) == {1, 2}

    def test_difference_matches_deselect(self):
        g = _graph().select_nodes_by_id([1, 2, 3]).select_nodes_by_id([2], set_op="difference")
        h = _graph().select_nodes_by_id([1, 2, 3]).deselect_nodes([2])
        assert _nodes(g) == _nodes(h) == {1, 3}


# =============================================================================
# One kind of selection at a time
# =============================================================================


class TestOneKind:
    @pytest.mark.parametrize("name", list(NODE_SELECTORS))
    def test_node_selectors_clear_edges(self, name):
        g = _graph().select_edges_by_id([1, 2])
        NODE_SELECTORS[name](g)
        assert _nodes(g) and _edges(g) == set()

    @pytest.mark.parametrize("name", list(EDGE_SELECTORS))
    def test_edge_selectors_clear_nodes(self, name):
        g = _graph().select_nodes_by_id([1, 2])
        EDGE_SELECTORS[name](g)
        assert _edges(g) and _nodes(g) == set()

    @pytest.mark.parametrize(
        "set_op, expected", [("union", {2, 3}), ("intersect", set()), ("difference", set())]
    )
    def test_set_ops_start_from_no_nodes_after_edges(self, set_op, expected):
        """With edges selected, there's no node selection to combine with."""
        g = _graph().select_edges_by_id([1]).select_nodes_by_id([2, 3], set_op=set_op)
        assert _nodes(g) == expected and _edges(g) == set()

    def test_invert_node_selection_clears_edges(self):
        g = _graph().select_edges_by_id([1]).invert_node_selection()
        assert _nodes(g) == {1, 2, 3, 4, 5} and _edges(g) == set()

    def test_invert_edge_selection_clears_nodes(self):
        g = _graph().select_nodes_by_id([1]).invert_edge_selection()
        assert _edges(g) == {1, 2, 3, 4, 5} and _nodes(g) == set()

    def test_node_traversal_with_only_edges_does_nothing(self):
        g = _graph().select_edges_by_id([1]).trav_out()
        assert _edges(g) == {1} and _nodes(g) == set()

    def test_traversals_switch_kinds(self):
        g = _graph().select_nodes_by_id([1]).trav_out_edge()
        assert _edges(g) == {1, 5} and _nodes(g) == set()

    def test_trav_out_until_add_to_selection_still_adds(self):
        g = _graph().select_nodes_by_id([1])
        g.trav_out_until(lambda nid: nid == 3, add_to_selection=True)
        assert {1, 3} <= _nodes(g)
        g = _graph().select_nodes_by_id([1]).trav_out_until(lambda nid: nid == 3)
        assert _nodes(g) == {3}


# =============================================================================
# Errors
# =============================================================================


class TestErrors:
    @pytest.mark.parametrize("set_op", ["add", "Union", None, True])
    @pytest.mark.parametrize("name", list(NODE_SELECTORS) + list(EDGE_SELECTORS))
    def test_bad_set_op(self, name, set_op):
        select = {**NODE_SELECTORS, **EDGE_SELECTORS}[name]
        g = _graph().select_nodes_by_id([4])
        with pytest.raises(ValueError, match=r"set_op= must be one of 'union', 'intersect'"):
            select(g, set_op=set_op)
        assert _nodes(g) == {4}
        assert len(g.steps) == 3  # nothing recorded for the failed call

    def test_missing_id_keeps_the_selection(self):
        g = _graph().select_nodes_by_id([4])
        with pytest.raises(KeyError):
            g.select_nodes_by_id([1, 99])
        assert _nodes(g) == {4}
        g.select_edges_by_id([2])
        with pytest.raises(KeyError):
            g.select_edges_by_id([1, 99], set_op="union")
        assert _edges(g) == {2}

    def test_unknown_label_keeps_the_selection(self):
        g = _graph().select_edges_by_id([2])
        with pytest.raises(ValueError):
            g.select_edges_by_node(["nope"])
        with pytest.raises(ValueError):
            g.select_nodes_in_neighborhood(["nope"])
        assert _edges(g) == {2}

    def test_failing_predicate_keeps_the_selection(self):
        g = _graph().select_nodes_by_id([4])

        def boom(n):
            if n.id == 3:
                raise RuntimeError("boom")
            return True

        with pytest.raises(RuntimeError):
            g.select_nodes(boom)
        assert _nodes(g) == {4}
        with pytest.raises(TypeError):
            g.select_edges(lambda e: None)
        assert _nodes(g) == {4} and _edges(g) == set()


# =============================================================================
# Steps
# =============================================================================


class TestSteps:
    @pytest.mark.parametrize(
        "call, code",
        [
            (lambda g: g.select_nodes(lambda n: n.id < 3), "G.select_nodes(predicate)  # 2 selected"),
            (
                lambda g: g.select_nodes(lambda n: n.id == 5, set_op="replace"),
                "G.select_nodes(predicate, set_op='replace')  # 1 selected",
            ),
            (
                lambda g: g.select_nodes_by_id([1, 2], set_op="intersect"),
                "G.select_nodes_by_id(..., set_op='intersect')  # 1 selected",
            ),
            (
                lambda g: g.select_last_nodes_created(2, set_op="difference"),
                "G.select_last_nodes_created(2, set_op='difference')",
            ),
            (
                lambda g: g.select_nodes_in_neighborhood("A"),
                "G.select_nodes_in_neighborhood(..., k=1)  # 3 selected",
            ),
            (
                lambda g: g.select_nodes_by_centrality("degree", k=1),
                "G.select_nodes_by_centrality('degree', k=1)  # 2 selected",
            ),
            (
                lambda g: g.select_edges_by_node(["A"], set_op="replace"),
                "G.select_edges_by_node(..., set_op='replace')  # 2 selected",
            ),
        ],
    )
    def test_code(self, call, code):
        g = _graph().select_nodes_by_id([2])
        assert call(g).steps[-1].code == code

    def test_count_is_the_resulting_selection(self):
        g = _graph().select_nodes_by_id([1, 2]).select_nodes_by_id([2, 3])
        assert g.steps[-1].code == "G.select_nodes_by_id(...)  # 3 selected"
