"""Tests for the item shapes accepted by add_edges() and add_nodes()."""

import edgecase as ec
import pytest


def _edges(g):
    """(source label, target label, weight, attributes) for every edge, in ID order."""
    out = []
    for edge_id in sorted(g.get_edge_ids()):
        e = g.get_edge(edge_id)
        out.append(
            (
                g.get_label_from_id(e.source),
                g.get_label_from_id(e.target),
                e.weight,
                dict(e.attributes or {}),
            )
        )
    return out


def _abc():
    return ec.Graph(weighted=True).add_nodes(["A", "B", "C"])


class TestAddEdgesShapes:
    def test_bulk_matches_chained_calls(self):
        """add_nodes() + add_edges() build exactly the graph that chained calls build."""
        pairs = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "C")]
        for directed in (False, True):
            chained = ec.Graph(directed=directed)
            for label in "ABCDE":
                chained.add_node(label=label)
            for s, t in pairs:
                chained.add_edge(s, t)
            bulk = ec.Graph(directed=directed).add_nodes(list("ABCDE")).add_edges(pairs)

            assert sorted(chained.get_node_ids()) == sorted(bulk.get_node_ids())
            for node_id in chained.get_node_ids():
                assert chained.get_label_from_id(node_id) == bulk.get_label_from_id(node_id)
            assert _edges(chained) == _edges(bulk)

    def test_numeric_third_element_is_the_weight(self):
        g = _abc().add_edges([("A", "B", 2.0), ("B", "C", 0.5)])
        assert _edges(g) == [("A", "B", 2.0, {}), ("B", "C", 0.5, {})]

    def test_int_and_numpy_weights(self):
        np = pytest.importorskip("numpy")
        g = _abc().add_edges([("A", "B", 3), ("B", "C", np.float32(1.5)), ("A", "C", np.int64(4))])
        assert [w for *_, w, _ in _edges(g)] == [3.0, 1.5, 4.0]

    def test_per_edge_weight_overrides_shared_weight(self):
        g = _abc().add_edges([("A", "B", 2.0), ("B", "C")], weight=9.0)
        assert [w for *_, w, _ in _edges(g)] == [2.0, 9.0]

    def test_weight_and_attrs(self):
        g = _abc().add_edges([("A", "B", 2.0, {"color": "red"})], rel="link")
        (source, target, weight, attrs), = _edges(g)
        assert (source, target, weight, attrs["color"]) == ("A", "B", 2.0, "red")
        assert g.get_edge_df()["rel"].to_list() == ["link"]

    def test_attrs_dict_still_overrides_weight_and_rel(self):
        g = _abc().add_edges(
            [("A", "B"), ("A", "C", {"weight": 2.0, "rel": "strong"})], rel="weak", weight=1.0
        )
        assert [w for *_, w, _ in _edges(g)] == [1.0, 2.0]
        assert g.get_edge_df()["rel"].to_list() == ["weak", "strong"]

    def test_lists_work_like_tuples(self):
        g = _abc().add_edges([["A", "B"], ["B", "C", 0.5]])
        assert _edges(g) == [("A", "B", None, {}), ("B", "C", 0.5, {})]

    @pytest.mark.parametrize(
        "bad",
        [
            ("A", "B", "heavy"),  # third element neither number nor dict
            ("A", "B", True),  # bools aren't weights
            ("A",),  # too short
            ("A", "B", 1.0, {}, "extra"),  # too long
            ("A", "B", {}, 1.0),  # 4-tuple in the wrong order
            "AB",  # a string isn't an edge, even with two characters
        ],
    )
    def test_malformed_items_raise_and_add_nothing(self, bad):
        g = _abc()
        with pytest.raises(TypeError):
            g.add_edges([("A", "B"), bad])
        assert g.count_edges() == 0

    def test_weight_given_twice_raises(self):
        g = _abc()
        with pytest.raises(ValueError, match="twice"):
            g.add_edges([("A", "B", 1.0, {"weight": 2.0})])
        assert g.count_edges() == 0

    def test_single_step_recorded(self):
        g = _abc()
        before = len(g.steps)
        g.add_edges([("A", "B", 1.0), ("B", "C")])
        assert len(g.steps) == before + 1


class TestAddNodesShapes:
    def test_dict_maps_labels_to_attrs(self):
        g = ec.Graph().add_nodes(
            {"Alice": {"age": 34}, "Bob": {"type": "bot"}, "Carol": None}, type="person"
        )
        rows = g.get_node_df().select("label", "type", "age").to_dicts()
        assert rows == [
            {"label": "Alice", "type": "person", "age": 34},
            {"label": "Bob", "type": "bot", "age": None},
            {"label": "Carol", "type": "person", "age": None},
        ]

    def test_generator_input_is_read_once(self):
        g = ec.Graph().add_nodes(f"n{i}" for i in range(3))
        assert g.get_node_df()["label"].to_list() == ["n0", "n1", "n2"]
        assert "['n0', 'n1', 'n2']" in g.steps[-1].code

    def test_list_pair_works_like_tuple(self):
        g = ec.Graph().add_nodes([["A", {"team": "red"}], "B"])
        assert g.get_node_df().select("label", "team").to_dicts() == [
            {"label": "A", "team": "red"},
            {"label": "B", "team": None},
        ]

    @pytest.mark.parametrize(
        "bad",
        [
            [("A", 5)],  # tuple that isn't (label, dict)
            [("A", {}, "extra")],
            [{"label": "A"}],  # a dict element
            {"A": 5},  # dict form with non-dict attributes
        ],
    )
    def test_malformed_input_raises_and_adds_nothing(self, bad):
        g = ec.Graph()
        with pytest.raises(TypeError):
            g.add_nodes(bad)
        assert g.count_nodes() == 0


# Step recording for add_edges_by_id().


class TestAddEdgesByIdSteps:
    def test_add_edges_by_id_records_labels_and_respects_compounds(self):
        g = ec.Graph().add_nodes(["A", "B"])
        g.add_edges_by_id([(1, 2)])
        assert g.steps[-1].delta.edges_added == [("A", "B")]
