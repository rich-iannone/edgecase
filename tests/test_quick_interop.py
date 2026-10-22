"""Tests for the dict-of-dicts and dict-of-lists conversions, the classic social datasets,
`Graph.attr_matrix()` and `Graph.rescale_layout()`, checked against NetworkX."""

import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_BUILD_NODES, KIND_MUTATE

from tests.test_io_interop import assert_same, edges_of, nodes_of, nx_pairs, pairs, rich_graph

np = pytest.importorskip("numpy")


# =============================================================================
# Helpers
# =============================================================================


def random_graph(seed, directed=False, weighted=True, n=12, m=30, loops=True):
    """A random simple graph on nodes labeled 'n0', 'n1', ... with integer-valued weights."""
    rng = random.Random(seed)
    g = ec.Graph(directed=directed, weighted=weighted)
    g.add_nodes([f"n{i}" for i in range(n)])
    seen = set()
    while len(seen) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v and not loops:
            continue
        key = (u, v) if directed else tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        if weighted:
            g.add_edge(f"n{u}", f"n{v}", weight=float(rng.randint(1, 9)))
        else:
            g.add_edge(f"n{u}", f"n{v}")
    return g


def to_nx(g, multigraph=False, node_attrs=False):
    """The NetworkX version of `g`, labels as nodes, edges in ID order with their data."""
    cls = {
        (False, False): nx.Graph,
        (True, False): nx.DiGraph,
        (False, True): nx.MultiGraph,
        (True, True): nx.MultiDiGraph,
    }[(g.is_directed(), multigraph)]
    G = cls()
    for label, attrs in nodes_of(g):
        G.add_node(label, **(attrs if node_attrs else {}))
    for s, t, w, attrs in edges_of(g):
        data = dict(attrs)
        if w is not None:
            data["weight"] = w
        G.add_edge(s, t, **data)
    return G


def last_step(g):
    return g._steps[-1]


# =============================================================================
# to_dict_of_dicts / from_dict_of_dicts
# =============================================================================


class TestToDictOfDicts:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_networkx(self, directed, seed):
        g = random_graph(seed, directed)
        assert ec.to_dict_of_dicts(g) == nx.to_dict_of_dicts(to_nx(g))

    @pytest.mark.parametrize("directed", [False, True])
    def test_nodelist(self, directed):
        g = random_graph(7, directed)
        nodelist = ["n3", "n0", "n5", "n9", "n1"]
        ours = ec.to_dict_of_dicts(g, nodelist=nodelist)
        assert list(ours) == nodelist
        assert ours == nx.to_dict_of_dicts(to_nx(g), nodelist=nodelist)

    def test_nodelist_accepts_ids(self):
        g = random_graph(1)
        ids = [g.get_id_from_label("n2"), g.get_id_from_label("n4")]
        assert ec.to_dict_of_dicts(g, nodelist=ids) == ec.to_dict_of_dicts(g, nodelist=["n2", "n4"])

    @pytest.mark.parametrize("directed", [False, True])
    def test_edge_data(self, directed):
        g = random_graph(3, directed)
        assert ec.to_dict_of_dicts(g, edge_data=1) == nx.to_dict_of_dicts(to_nx(g), edge_data=1)

    def test_attributes_and_unweighted(self):
        g = ec.Graph().add_nodes(["a", "b"])
        g.add_edge("a", "b", rel="knows")
        assert ec.to_dict_of_dicts(g) == {
            "a": {"b": {"rel": "knows"}},
            "b": {"a": {"rel": "knows"}},
        }

    def test_unlabeled_nodes_use_ids(self):
        g = ec.Graph().add_path_graph(3)
        assert ec.to_dict_of_dicts(g) == {"1": {"2": {}}, "2": {"1": {}, "3": {}}, "3": {"2": {}}}

    def test_sides_are_separate_dicts(self):
        d = ec.to_dict_of_dicts(ec.Graph().add_path_graph(2))
        d["1"]["2"]["x"] = 1
        assert d["2"]["1"] == {}

    @pytest.mark.parametrize("directed", [False, True])
    def test_multigraph_layout(self, directed):
        g = rich_graph(directed)
        ours = ec.to_dict_of_dicts(g)
        theirs = nx.to_dict_of_dicts(to_nx(g, multigraph=True))
        assert ours == theirs
        assert ours["A"]["B"] == {
            0: {"weight": 2.5, "rel": "knows"},
            1: {"weight": 3.5, "rel": "likes"},
        }
        assert ours["A"]["A"] == {0: {"weight": 0.5}}

    def test_multigraph_flag(self):
        with pytest.raises(ValueError, match="multigraph=True"):
            ec.to_dict_of_dicts(rich_graph(), multigraph=False)
        simple = ec.Graph().add_path_graph(2)
        assert ec.to_dict_of_dicts(simple, multigraph=True) == {
            "1": {"2": {0: {}}},
            "2": {"1": {0: {}}},
        }

    def test_parallel_edges_outside_nodelist_dont_count(self):
        g = rich_graph()
        d = ec.to_dict_of_dicts(g, nodelist=["B", "Lonely"])
        assert d == {"B": {}, "Lonely": {}}

    def test_empty_graph(self):
        assert ec.to_dict_of_dicts(ec.Graph()) == {}


class TestFromDictOfDicts:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_round_trip(self, directed, seed):
        g = random_graph(seed, directed)
        h = ec.from_dict_of_dicts(ec.to_dict_of_dicts(g), directed=directed)
        assert [label for label, _ in nodes_of(h)] == [label for label, _ in nodes_of(g)]
        assert pairs(h) == pairs(g)
        weights = {(s, t): w for s, t, w, _ in edges_of(g)}
        if not directed:
            weights.update({(t, s): w for (s, t), w in list(weights.items())})
        assert all(weights[(s, t)] == w for s, t, w, _ in edges_of(h))
        assert h.is_weighted()

    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        d = {
            "a": {"b": {"weight": 2}, "c": {"x": 1}},
            "b": {"a": {"weight": 2}, "d": {}},
            "c": {"a": {"y": 2}},
        }
        G = nx.from_dict_of_dicts(d, create_using=nx.DiGraph if directed else nx.Graph)
        h = ec.from_dict_of_dicts(d, directed=directed)
        assert [label for label, _ in nodes_of(h)] == [str(n) for n in G]
        assert pairs(h) == nx_pairs(G)
        for s, t, w, attrs in edges_of(h):
            data = dict(attrs)
            if w is not None:
                data["weight"] = w
            assert data == G[s][t]

    def test_undirected_listings_merge(self):
        h = ec.from_dict_of_dicts(
            {"a": {"b": {"x": 1, "weight": 1}}, "b": {"a": {"y": 2, "weight": 5}}}
        )
        assert edges_of(h) == [("a", "b", 5.0, {"x": 1, "y": 2})]

    @pytest.mark.parametrize("directed", [False, True])
    def test_multigraph_round_trip(self, directed):
        g = rich_graph(directed)
        h = ec.from_dict_of_dicts(ec.to_dict_of_dicts(g), directed=directed, multigraph_input=True)
        assert [label for label, _ in nodes_of(h)] == [label for label, _ in nodes_of(g)]
        assert sorted(map(repr, edges_of(h))) == sorted(map(repr, edges_of(g)))

    @pytest.mark.parametrize("directed", [False, True])
    def test_multigraph_input_matches_networkx(self, directed):
        d = {
            "a": {"b": {0: {"weight": 1}, 1: {"weight": 2}}, "a": {0: {}, 1: {}}},
            "b": {"a": {0: {"weight": 1}, 1: {"weight": 2}}, "c": {0: {}}},
            "c": {"b": {0: {}}},
        }
        cls = nx.MultiDiGraph if directed else nx.MultiGraph
        G = nx.from_dict_of_dicts(d, create_using=cls, multigraph_input=True)
        h = ec.from_dict_of_dicts(d, directed=directed, multigraph_input=True)
        assert pairs(h) == nx_pairs(G)
        assert sorted(w for _, _, w, _ in edges_of(h) if w is not None) == sorted(
            w for _, _, w in G.edges(data="weight") if w is not None
        )

    def test_neighbor_only_nodes_come_last(self):
        h = ec.from_dict_of_dicts({"b": {"z": {}}, "a": {"b": {}}})
        assert [label for label, _ in nodes_of(h)] == ["b", "a", "z"]
        assert [label for label, _ in nodes_of(h)] == list(
            nx.from_dict_of_dicts({"b": {"z": {}}, "a": {"b": {}}})
        )

    def test_non_string_keys(self):
        h = ec.from_dict_of_dicts({0: {1: {}}, 1: {0: {}}})
        assert pairs(h) == [("0", "1")]

    def test_weighted_detection(self):
        assert not ec.from_dict_of_dicts({"a": {"b": {}}}).is_weighted()
        assert ec.from_dict_of_dicts({"a": {"b": {"weight": 1}}}).is_weighted()
        assert not ec.from_dict_of_dicts({"a": {"b": {"weight": 1}}}, weighted=False).is_weighted()
        h = ec.from_dict_of_dicts({"a": {"b": {"weight": "heavy"}}})
        assert edges_of(h) == [("a", "b", None, {"weight": "heavy"})]

    def test_single_build_step(self):
        h = ec.from_dict_of_dicts({"a": {"b": {}}, "b": {"a": {}}}, graph_name="G")
        assert len(h._steps) == 1
        assert last_step(h).kind == KIND_BUILD_NODES
        assert h.get_graph_name() == "G"

    def test_errors(self):
        with pytest.raises(TypeError, match="dict of dicts"):
            ec.from_dict_of_dicts([("a", "b")])
        with pytest.raises(TypeError, match="neighbors of 'a'"):
            ec.from_dict_of_dicts({"a": ["b"]})
        with pytest.raises(TypeError, match="Edge data"):
            ec.from_dict_of_dicts({"a": {"b": 1}})
        with pytest.raises(TypeError, match="parallel edge"):
            ec.from_dict_of_dicts({"a": {"b": {0: 1}}}, multigraph_input=True)


# =============================================================================
# to_dict_of_lists / from_dict_of_lists
# =============================================================================


class TestDictOfLists:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_networkx(self, directed, seed):
        g = random_graph(seed, directed, weighted=False)
        assert ec.to_dict_of_lists(g) == nx.to_dict_of_lists(to_nx(g))

    def test_nodelist(self):
        g = random_graph(5)
        nodelist = ["n4", "n2", "n8", "n0"]
        assert ec.to_dict_of_lists(g, nodelist=nodelist) == nx.to_dict_of_lists(
            to_nx(g), nodelist=nodelist
        )

    @pytest.mark.parametrize("directed", [False, True])
    def test_parallel_edges_listed_once(self, directed):
        g = rich_graph(directed)
        assert ec.to_dict_of_lists(g) == nx.to_dict_of_lists(to_nx(g, multigraph=True))
        assert ec.to_dict_of_lists(g)["A"] == ["B", "A"]

    @pytest.mark.parametrize("directed", [False, True])
    def test_round_trip(self, directed):
        g = random_graph(2, directed, weighted=False)
        h = ec.from_dict_of_lists(ec.to_dict_of_lists(g), directed=directed)
        assert [label for label, _ in nodes_of(h)] == [label for label, _ in nodes_of(g)]
        assert pairs(h) == pairs(g)

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("parallel_edges", [False, True])
    def test_from_matches_networkx(self, directed, parallel_edges):
        d = {"a": ["b", "c", "b", "a"], "b": ["a", "d"], "c": ["a", "a"], "e": []}
        cls = {
            (False, False): nx.Graph,
            (True, False): nx.DiGraph,
            (False, True): nx.MultiGraph,
            (True, True): nx.MultiDiGraph,
        }[(directed, parallel_edges)]
        G = nx.from_dict_of_lists(d, create_using=cls)
        h = ec.from_dict_of_lists(d, directed=directed, parallel_edges=parallel_edges)
        assert [label for label, _ in nodes_of(h)] == list(G)
        assert pairs(h) == nx_pairs(G)

    def test_iterables_and_name(self):
        h = ec.from_dict_of_lists({"a": ("b",), "b": {"c"}}, graph_name="G")
        assert pairs(h) == [("a", "b"), ("b", "c")]
        assert h.get_graph_name() == "G"
        assert len(h._steps) == 1
        assert not h.is_weighted()

    def test_errors(self):
        with pytest.raises(TypeError, match="dict of lists"):
            ec.from_dict_of_lists([["a", "b"]])
        with pytest.raises(TypeError, match="neighbors of 'a'"):
            ec.from_dict_of_lists({"a": "bc"})
        with pytest.raises(TypeError, match="neighbors of 'a'"):
            ec.from_dict_of_lists({"a": 3})


# =============================================================================
# Datasets
# =============================================================================


DATASETS = [
    (ec.davis_southern_women, nx.davis_southern_women_graph, "Davis Southern Women", False),
    (ec.florentine_families, nx.florentine_families_graph, "Florentine Families", False),
    (ec.les_miserables, nx.les_miserables_graph, "Les Misérables", True),
]


class TestDatasets:
    @pytest.mark.parametrize("ours, theirs, name, weighted", DATASETS)
    def test_matches_networkx(self, ours, theirs, name, weighted):
        g, G = ours(), theirs()
        assert [label for label, _ in nodes_of(g)] == list(G)
        assert pairs(g) == nx_pairs(G)
        assert g.get_graph_name() == name
        assert g.is_weighted() == weighted
        assert not g.is_directed()
        assert len(g._steps) == 1
        assert last_step(g).kind == KIND_BUILD_NODES

    def test_sizes(self):
        assert (
            ec.davis_southern_women().count_nodes(),
            ec.davis_southern_women().count_edges(),
        ) == (32, 89)
        assert (ec.florentine_families().count_nodes(), ec.florentine_families().count_edges()) == (
            15,
            20,
        )
        assert (ec.les_miserables().count_nodes(), ec.les_miserables().count_edges()) == (77, 254)

    def test_davis_bipartite_attribute(self):
        g, G = ec.davis_southern_women(), nx.davis_southern_women_graph()
        assert [attrs for _, attrs in nodes_of(g)] == [d for _, d in G.nodes(data=True)]
        women = [label for label, attrs in nodes_of(g) if attrs["bipartite"] == 0]
        assert women == G.graph["top"]
        assert g.is_bipartite()

    def test_les_miserables_weights(self):
        g, G = ec.les_miserables(), nx.les_miserables_graph()
        for s, t, w, attrs in edges_of(g):
            assert w == G[s][t]["weight"]
            assert attrs == {}

    def test_classic_results(self):
        florentine = ec.florentine_families()
        scores = florentine.betweenness_centrality(return_dict=True)
        assert florentine.get_label_from_id(max(scores, key=scores.get)) == "Medici"
        assert ec.les_miserables().get_degree("Valjean") == 36
        assert ec.davis_southern_women().bipartite_projection(which=0).count_nodes() == 18

    def test_fresh_graph_each_call(self):
        a = ec.florentine_families()
        a.delete_node("Medici")
        assert ec.florentine_families().count_nodes() == 15


# =============================================================================
# Graph.attr_matrix()
# =============================================================================


def colored(g, seed, k=3):
    rng = random.Random(seed)
    colors = ["red", "green", "blue"][:k]
    for nid in g.get_node_ids(limit=None):
        g.set_node_attrs("color", {nid: rng.choice(colors)})
    return g


class TestAttrMatrix:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("edge_attr", [None, "weight", "cost"])
    @pytest.mark.parametrize("normalized", [False, True])
    def test_grouped_matches_networkx(self, directed, edge_attr, normalized):
        g = colored(random_graph(11, directed), 4)
        g.set_edge_attrs("cost", {eid: eid % 4 + 1 for eid, _, _ in g._graph.edges()})
        order = ["red", "green", "blue"]
        matrix, labels = g.attr_matrix(
            edge_attr=edge_attr, node_attr="color", normalized=normalized, rc_order=order
        )
        expected = nx.attr_matrix(
            to_nx(g, node_attrs=True),
            edge_attr=edge_attr,
            node_attr="color",
            normalized=normalized,
            rc_order=order,
        )
        assert labels == order
        assert np.allclose(matrix, expected)

    @pytest.mark.parametrize("directed", [False, True])
    def test_per_node_matches_networkx(self, directed):
        g = random_graph(12, directed)
        ids = sorted(g.get_node_ids(limit=None))
        labels = [g.get_label_from_id(nid) for nid in ids]
        matrix, order = g.attr_matrix(edge_attr="weight")
        assert order == ids
        expected = nx.attr_matrix(to_nx(g), edge_attr="weight", rc_order=labels)
        assert np.allclose(matrix, expected)
        assert np.allclose(matrix, ec.to_numpy_array(g))

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("edge_attr", [None, "weight"])
    def test_parallel_edges_match_networkx_multigraph(self, directed, edge_attr):
        g = rich_graph(directed)
        labels = [label for label, _ in nodes_of(g)]
        matrix, _ = g.attr_matrix(edge_attr=edge_attr)
        expected = nx.attr_matrix(to_nx(g, multigraph=True), edge_attr=edge_attr, rc_order=labels)
        assert np.allclose(matrix, expected)

    def test_default_order_is_sorted(self):
        g = colored(random_graph(3), 1)
        _, labels = g.attr_matrix(node_attr="color")
        assert labels == sorted(set(g.get_node_attrs("color").values()))

    def test_unsortable_values_sorted_by_text(self):
        g = ec.Graph().add_nodes(["a", "b"]).add_edges([("a", "b")])
        g.set_node_attrs("k", {1: 1, 2: "x"})
        _, labels = g.attr_matrix(node_attr="k")
        assert labels == sorted([1, "x"], key=repr)

    def test_rc_order_labels_and_subset(self):
        g = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("b", "c")])
        matrix, order = g.attr_matrix(rc_order=["c", "b"])
        assert order == [3, 2]
        assert matrix == [[0.0, 1.0], [1.0, 0.0]]

    def test_callables(self):
        g = ec.Graph(weighted=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b", 2.0), ("b", "c", 3.0)])
        matrix, labels = g.attr_matrix(
            edge_attr=lambda e: e.weight * 10, node_attr=lambda n: n.label == "b"
        )
        assert labels == [False, True]
        assert matrix == [[0.0, 50.0], [50.0, 0.0]]

    def test_self_loop_and_within_group_counted_once(self):
        g = ec.Graph().add_nodes(["a", "b"]).add_edges([("a", "a"), ("a", "b")])
        g.set_node_attrs("t", {1: 0, 2: 0})
        assert g.attr_matrix()[0] == [[1.0, 1.0], [1.0, 0.0]]
        assert g.attr_matrix(node_attr="t")[0] == [[2.0]]

    def test_normalized_zero_row(self):
        g = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b")])
        matrix, _ = g.attr_matrix(normalized=True)
        assert matrix == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_errors(self):
        g = ec.Graph().add_nodes(["a", "b"]).add_edges([("a", "b")])
        with pytest.raises(ValueError, match="no 'color' attribute"):
            g.attr_matrix(node_attr="color")
        with pytest.raises(ValueError, match="no 'cost' attribute"):
            g.attr_matrix(edge_attr="cost")
        with pytest.raises(ValueError, match="more than once"):
            g.attr_matrix(rc_order=["a", "a"])
        with pytest.raises(ValueError, match="non-numeric"):
            g.attr_matrix(edge_attr=lambda e: "x")

    def test_records_query_step(self):
        g = ec.Graph().add_path_graph(3)
        n = len(g._steps)
        g.attr_matrix()
        assert len(g._steps) == n + 1
        assert last_step(g).method == "attr_matrix"


# =============================================================================
# Graph.rescale_layout()
# =============================================================================


class TestRescaleLayout:
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("scale", [1.0, 3.5])
    def test_matches_networkx(self, seed, scale):
        rng = random.Random(seed)
        g = ec.Graph().add_path_graph(9)
        pos = {nid: (rng.uniform(-5, 20), rng.uniform(3, 8)) for nid in g.get_node_ids()}
        ours = g.rescale_layout(scale=scale, pos=pos, return_dict=True)
        theirs = nx.rescale_layout_dict({k: np.array(v) for k, v in pos.items()}, scale=scale)
        assert set(ours) == set(theirs)
        assert all(np.allclose(ours[k], theirs[k]) for k in ours)

    def test_center(self):
        g = ec.Graph().add_path_graph(3)
        pos = {1: (0, 0), 2: (10, 5), 3: (20, 10)}
        assert g.rescale_layout(pos=pos, center=(1, 2), return_dict=True) == {
            1: (0.0, 1.5),
            2: (1.0, 2.0),
            3: (2.0, 2.5),
        }

    def test_in_place_on_attribute(self):
        g = ec.Graph().add_path_graph(3)
        g.set_node_attrs("xy", {1: [0, 0], 2: [6, 0], 3: [6, 3]})
        assert g.rescale_layout(scale=2, pos="xy") is g
        assert g.get_node_attrs("xy") == {1: [-2.0, -0.5], 2: [1.0, -0.5], 3: [1.0, 1.0]}
        assert g.get_node_attrs("pos") == {}
        assert last_step(g).kind == KIND_MUTATE
        assert last_step(g).method == "rescale_layout"

    def test_default_pos_attribute(self):
        g = ec.Graph().add_cycle_graph(5).layout_circular(scale=4)
        g.rescale_layout()
        expected = ec.Graph().add_cycle_graph(5).layout_circular(return_dict=True)
        stored = g.get_node_attrs("pos")
        assert all(np.allclose(stored[k], expected[k]) for k in expected)

    def test_dict_pos_stored_in_pos_or_attr_name(self):
        g = ec.Graph().add_path_graph(2)
        g.rescale_layout(pos={"1": (0, 0), "2": (2, 0)})
        assert g.get_node_attrs("pos") == {1: [-1.0, 0.0], 2: [1.0, 0.0]}
        g.rescale_layout(pos="pos", attr_name="big", scale=5)
        assert g.get_node_attrs("big") == {1: [-5.0, 0.0], 2: [5.0, 0.0]}
        assert g.get_node_attrs("pos") == {1: [-1.0, 0.0], 2: [1.0, 0.0]}

    def test_only_positioned_nodes(self):
        g = ec.Graph().add_path_graph(3)
        g.set_node_attrs("pos", {1: [0, 0], 2: [4, 0]})
        g.rescale_layout()
        assert g.get_node_attrs("pos") == {1: [-1.0, 0.0], 2: [1.0, 0.0]}

    def test_single_point(self):
        g = ec.Graph().add_path_graph(2)
        out = g.rescale_layout(pos={1: (3, 3), 2: (3, 3)}, center=(1, 0), return_dict=True)
        assert out == {1: (1.0, 0.0), 2: (1.0, 0.0)}

    def test_return_dict_leaves_graph_alone(self):
        g = ec.Graph().add_path_graph(2).layout_circular(scale=3)
        before = g.get_node_attrs("pos")
        n = len(g._steps)
        g.rescale_layout(return_dict=True)
        assert g.get_node_attrs("pos") == before
        assert len(g._steps) == n

    def test_errors(self):
        g = ec.Graph().add_path_graph(2)
        with pytest.raises(ValueError, match="which no node has"):
            g.rescale_layout()
        with pytest.raises(ValueError, match="scale"):
            g.rescale_layout(scale=0, pos={1: (0, 0)})
        with pytest.raises(ValueError, match="scale"):
            g.rescale_layout(scale=None, pos={1: (0, 0)})
        with pytest.raises(ValueError, match="Position of node 1"):
            g.rescale_layout(pos={1: (0, float("nan"))})
        with pytest.raises(ValueError, match="Position of node 1"):
            g.rescale_layout(pos={1: "ab"})
        with pytest.raises(TypeError):
            g.rescale_layout(pos=None)


def test_exports():
    for name in [
        "to_dict_of_dicts",
        "from_dict_of_dicts",
        "to_dict_of_lists",
        "from_dict_of_lists",
        "davis_southern_women",
        "florentine_families",
        "les_miserables",
    ]:
        assert name in ec.__all__
        assert callable(getattr(ec, name))


def test_rich_graph_round_trip_keeps_edges():
    g = rich_graph()
    h = ec.from_dict_of_dicts(
        ec.to_dict_of_dicts(g), multigraph_input=True, graph_name="Rich graph"
    )
    with pytest.raises(AssertionError):
        assert_same(h, g)  # node attributes aren't part of the format
    assert sorted(map(repr, edges_of(h))) == sorted(map(repr, edges_of(g)))
