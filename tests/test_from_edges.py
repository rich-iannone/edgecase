"""Tests for `Graph.from_edges()`."""

import edgecase as ec
import numpy as np
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _nodes(g):
    """`{label: attributes}` for every node, in node ID order."""
    return {g.get_node(n).label: dict(g.get_node(n).attributes or {})
            for n in sorted(g.get_node_ids(limit=None))}


def _edges(g):
    """`(source label, target label, weight, attributes)` for every edge, in edge ID order."""
    out = []
    for e in sorted(g.get_edge_ids(limit=None)):
        x = g.get_edge(e)
        s, t = g.get_label_from_id(x.source), g.get_label_from_id(x.target)
        out.append((s, t, x.weight, dict(x.attributes or {})))
    return out


def _pairs(g):
    return g.get_edges_str_repr(use_labels=True)


# =============================================================================
# Building
# =============================================================================


class TestBasics:
    def test_pairs(self):
        g = ec.Graph.from_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
        assert isinstance(g, ec.Graph)
        assert not g.is_directed() and not g.is_weighted()
        assert list(_nodes(g)) == ["A", "B", "C", "D"]
        assert _pairs(g) == ["A--B", "B--C", "C--A", "C--D"]

    def test_nodes_in_order_of_first_appearance(self):
        g = ec.Graph.from_edges([("c", "a"), ("b", "a"), ("d", "c")])
        assert list(_nodes(g)) == ["c", "a", "b", "d"]

    def test_directed(self):
        g = ec.Graph.from_edges([("a", "b"), ("b", "a")], directed=True)
        assert g.is_directed() and _pairs(g) == ["a->b", "b->a"]

    def test_parallel_edges_and_self_loops(self):
        g = ec.Graph.from_edges([("a", "b"), ("a", "b"), ("a", "a")])
        assert g.count_nodes() == 2 and _pairs(g) == ["a--b", "a--b", "a--a"]

    def test_graph_name(self):
        g = ec.Graph.from_edges([("a", "b")], graph_name="small")
        assert g.get_graph_name() == "small"
        assert ec.Graph.from_edges([("a", "b")]).get_graph_name() is None

    def test_empty(self):
        g = ec.Graph.from_edges([])
        assert g.count_nodes() == 0 and g.count_edges() == 0 and not g.is_weighted()

    def test_lists_as_items(self):
        g = ec.Graph.from_edges([["a", "b"], ["b", "c", 2]])
        assert _edges(g) == [("a", "b", None, {}), ("b", "c", 2.0, {})]

    def test_any_iterable(self):
        g = ec.Graph.from_edges((s, t) for s, t in zip("abc", "bcd"))
        assert _pairs(g) == ["a--b", "b--c", "c--d"]
        assert _pairs(ec.Graph.from_edges(zip("ab", "bc"))) == ["a--b", "b--c"]

    def test_graph_can_grow_afterwards(self):
        g = ec.Graph.from_edges([("a", "b")])
        g.add_node("c")
        g.add_edge("b", "c")
        assert list(_nodes(g)) == ["a", "b", "c"] and _pairs(g) == ["a--b", "b--c"]


class TestLabels:
    def test_integers_become_labels(self):
        g = ec.Graph.from_edges([(0, 1), (1, 2)])
        assert list(_nodes(g)) == ["0", "1", "2"]
        assert _pairs(g) == ["0--1", "1--2"]

    def test_number_and_string_are_the_same_node(self):
        g = ec.Graph.from_edges([(1, "2"), ("1", 2)])
        assert list(_nodes(g)) == ["1", "2"] and g.count_edges() == 2

    def test_other_values_use_str(self):
        g = ec.Graph.from_edges([(1.5, True), ("é", "two words")])
        assert list(_nodes(g)) == ["1.5", "True", "é", "two words"]


class TestWeights:
    def test_third_element(self):
        g = ec.Graph.from_edges([("a", "b", 2), ("b", "c", 0.5), ("c", "a")])
        assert g.is_weighted()
        assert [e[2] for e in _edges(g)] == [2.0, 0.5, None]

    def test_weight_in_the_dict(self):
        g = ec.Graph.from_edges([("a", "b", {"weight": 3, "k": "v"}), ("b", "c", 1.5, {"j": 1})])
        assert g.is_weighted()
        assert _edges(g) == [("a", "b", 3.0, {"k": "v"}), ("b", "c", 1.5, {"j": 1})]

    def test_none_weight_in_the_dict(self):
        g = ec.Graph.from_edges([("a", "b", {"weight": None, "rel": "x"})])
        assert not g.is_weighted() and _edges(g) == [("a", "b", None, {"rel": "x"})]

    def test_numpy_number_weights(self):
        g = ec.Graph.from_edges([("a", "b", np.float32(0.5)), ("b", "c", np.int64(4))])
        assert [e[2] for e in _edges(g)] == [0.5, 4.0]

    def test_weighted_argument(self):
        g = ec.Graph.from_edges([("a", "b")], weighted=True)
        assert g.is_weighted() and _edges(g)[0][2] is None
        # as with `add_edges()`, an unweighted graph still keeps each edge's weight
        g = ec.Graph.from_edges([("a", "b", 2)], weighted=False)
        assert not g.is_weighted() and _edges(g)[0][2] == 2.0


class TestAttributes:
    def test_edge_attributes_and_rel(self):
        g = ec.Graph.from_edges(
            [("ann", "bob", {"rel": "friend", "since": 2019}), ("bob", "cat", {"rel": "boss"})]
        )
        assert _edges(g) == [
            ("ann", "bob", None, {"rel": "friend", "since": 2019}),
            ("bob", "cat", None, {"rel": "boss"}),
        ]
        assert g.get_edge_df()["rel"].to_list() == ["friend", "boss"]

    def test_none_attributes_are_dropped(self):
        g = ec.Graph.from_edges([("a", "b", {"k": None, "j": 1})], nodes={"a": {"x": None}})
        assert _edges(g)[0][3] == {"j": 1}
        assert _nodes(g)["a"] == {}

    def test_input_dicts_are_not_changed(self):
        attrs = {"weight": 2, "k": 1}
        node_attrs = {"age": 3}
        g = ec.Graph.from_edges([("a", "b", attrs)], nodes={"a": node_attrs})
        assert attrs == {"weight": 2, "k": 1} and node_attrs == {"age": 3}
        g.set_node_attrs("age", {g.get_id_from_label("a"): 4})
        assert node_attrs == {"age": 3}


class TestNodes:
    def test_list_adds_isolated_nodes_first(self):
        g = ec.Graph.from_edges([("a", "b")], nodes=["x", "b"])
        assert list(_nodes(g)) == ["x", "b", "a"]
        assert g.isolates() == [g.get_id_from_label("x")]

    def test_integers_in_nodes(self):
        g = ec.Graph.from_edges([(0, 1), (1, 2)], directed=True, nodes=[0, 1, 2, 3])
        assert list(_nodes(g)) == ["0", "1", "2", "3"]
        assert _pairs(g) == ["0->1", "1->2"]

    def test_tuples_with_attributes(self):
        g = ec.Graph.from_edges([("a", "b")], nodes=[("a", {"age": 31}), "c"])
        assert _nodes(g) == {"a": {"age": 31}, "c": {}, "b": {}}

    def test_dict(self):
        g = ec.Graph.from_edges(
            [("ann", "bob"), ("bob", "cat")], nodes={"ann": {"age": 31}, "cat": {"age": 45}}
        )
        assert _nodes(g) == {"ann": {"age": 31}, "cat": {"age": 45}, "bob": {}}
        assert g.get_node_df().select(["label", "age"]).rows() == [
            ("ann", 31), ("cat", 45), ("bob", None),
        ]

    def test_only_nodes(self):
        g = ec.Graph.from_edges([], nodes=["a", "b"])
        assert list(_nodes(g)) == ["a", "b"] and g.count_edges() == 0

    @pytest.mark.parametrize("nodes", [["a", "b", "a"], ["1", 1], [("a", {}), "a"]])
    def test_duplicates(self, nodes):
        with pytest.raises(ValueError, match="is listed more than once in nodes="):
            ec.Graph.from_edges([], nodes=nodes)

    def test_none_label(self):
        with pytest.raises(ValueError, match="a node label can't be None"):
            ec.Graph.from_edges([("a", "b")], nodes=[None])

    def test_attributes_must_be_a_dict(self):
        with pytest.raises(TypeError, match="attributes for node 'a' must be a dict"):
            ec.Graph.from_edges([("a", "b")], nodes={"a": 3})


class TestNumPy:
    def test_integer_array(self):
        g = ec.Graph.from_edges(np.array([[0, 1], [1, 2], [2, 0]]))
        assert not g.is_weighted()
        assert list(_nodes(g)) == ["0", "1", "2"]
        assert _pairs(g) == ["0--1", "1--2", "2--0"]

    def test_float_array_with_weights(self):
        g = ec.Graph.from_edges(np.array([[0, 1, 0.5], [1, 2, 2]]), directed=True)
        assert g.is_weighted()
        assert _edges(g) == [("0", "1", 0.5, {}), ("1", "2", 2.0, {})]

    def test_fractional_float_endpoints_stay_floats(self):
        g = ec.Graph.from_edges(np.array([[0.5, 1.0]]))
        assert list(_nodes(g)) == ["0.5", "1"]

    def test_string_array(self):
        g = ec.Graph.from_edges(np.array([["a", "b"], ["b", "c"]]))
        assert _pairs(g) == ["a--b", "b--c"]


# =============================================================================
# Errors
# =============================================================================


class TestErrors:
    @pytest.mark.parametrize(
        "edges, error, message",
        [
            ([("a",)], TypeError, r"each edge must have 2 to 4 elements"),
            ([("a", "b", 1, {}, 2)], TypeError, r"each edge must have 2 to 4 elements"),
            (["ab"], TypeError, r"each edge must be a tuple like"),
            ([("a", "b", True)], TypeError, r"must be a numeric weight or an attribute dict"),
            ([("a", "b", "x")], TypeError, r"must be a numeric weight or an attribute dict"),
            ([("a", "b", {"weight": "x"})], TypeError,
             r"the weight of edge \('a', 'b'\) must be a number; got 'x'"),
            ([("a", "b", 2, {"weight": 3})], ValueError, r"gives its weight twice"),
            ([(None, "b")], ValueError, r"an edge endpoint can't be None"),
            ([("a", None)], ValueError, r"an edge endpoint can't be None"),
        ],
    )
    def test_bad_edges(self, edges, error, message):
        with pytest.raises(error, match=message):
            ec.Graph.from_edges(edges)

    def test_not_iterable(self):
        with pytest.raises(TypeError):
            ec.Graph.from_edges(42)


# =============================================================================
# Steps
# =============================================================================


class TestSteps:
    def test_one_step(self):
        g = ec.Graph.from_edges([("a", "b"), ("b", "c")], nodes=["x"])
        assert len(g.steps) == 1
        step = g.steps[-1]
        assert step.code == "Graph.from_edges([('a', 'b'), ('b', 'c')], nodes=<1 node>)"
        assert step.delta.nodes_added == ["x", "a", "b", "c"]
        assert step.delta.edges_added == [("a", "b"), ("b", "c")]

    @pytest.mark.parametrize(
        "args, code",
        [
            (([("a", "b")],), "Graph.from_edges([('a', 'b')])"),
            (([("a", "b")], True), "Graph.from_edges([('a', 'b')], directed=True)"),
            (([],), "Graph.from_edges([])"),
            ((tuple(),), "Graph.from_edges(<0 edges>)"),
            (([("a", "b")] * 20,), "Graph.from_edges(<20 edges>)"),
            ((iter([("a", "b")]),), "Graph.from_edges(<1 edge>)"),
            ((np.array([[0, 1], [1, 2]]),), "Graph.from_edges(<2 edges>)"),
        ],
    )
    def test_code(self, args, code):
        assert ec.Graph.from_edges(*args).steps[-1].code == code

    def test_code_with_node_dict(self):
        g = ec.Graph.from_edges([("a", "b")], nodes={"a": {}, "z": {}})
        assert g.steps[-1].code == "Graph.from_edges([('a', 'b')], nodes=<2 nodes>)"
