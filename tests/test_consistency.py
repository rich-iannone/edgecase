"""Regression tests for the "Inconsistencies" section of BUGS.md."""

import networkx as nx
import pytest

import edgecase as ec


def _dir_ab():
    return ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "B")])


class TestUnknownIdsRaiseLikeLabels:
    """Unknown integer IDs used to return quiet defaults while unknown labels raised."""

    @pytest.mark.parametrize("bad", [999, "Z"])
    @pytest.mark.parametrize(
        "call",
        [
            lambda g, x: g.get_in_degree(x),
            lambda g, x: g.get_out_degree(x),
            lambda g, x: g.deselect_nodes([x]),
            lambda g, x: g.select_edges_by_node([x]),
            lambda g, x: g.find_edge_ids(x, "A"),
            lambda g, x: g.k_shortest_paths(x, "A", 2),
            lambda g, x: g.get_similar_neighbors(x, "v"),
        ],
    )
    def test_unknown_node_raises(self, call, bad):
        with pytest.raises(ValueError):
            call(_dir_ab(), bad)

    def test_has_edge_is_false_for_unknown(self):
        g = _dir_ab()
        assert g.has_edge("A", "B")
        for args in [(999, "A"), ("Z", "A"), ("A", -1), (True, "B")]:
            assert g.has_edge(*args) is False

    def test_nothing_changes_before_raising(self):
        g = _dir_ab()
        g.select_nodes_by_id([1, 2])
        with pytest.raises(ValueError):
            g.deselect_nodes([1, 999])
        assert sorted(g.get_selected_node_ids()) == [1, 2]
        with pytest.raises(ValueError):
            g.select_edges_by_node([1, 999])
        assert g.get_selected_edge_ids() == []


class TestNoPartialWork:
    def test_select_by_id_is_atomic(self):
        """select_nodes_by_id / select_edges_by_id left earlier IDs selected when raising."""
        g = _dir_ab()
        with pytest.raises(KeyError):
            g.select_nodes_by_id([1, 999])
        assert g.get_selected_node_ids() == []
        edge = g.get_edge_ids()[0]
        with pytest.raises(KeyError):
            g.select_edges_by_id([edge, 999])
        assert g.get_selected_edge_ids() == []

    @pytest.mark.parametrize("ws", [False, True])
    def test_delete_drops_incident_edges_from_selection(self, ws):
        """delete_node / delete_nodes_ws left stale edge IDs in the selection."""
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        g.select_edges_by_id(g.get_edge_ids())
        if ws:
            g.select_nodes_by_id([g.get_id_from_label("A")]).delete_nodes_ws()
        else:
            g.delete_node("A")
        assert g.get_selected_edge_ids() == g.find_edge_ids("B", "C")


def test_successors_have_no_duplicates():
    """get_successors repeated targets of parallel edges; get_predecessors didn't."""
    g = _dir_ab().add_node(label="C").add_edge("A", "C")
    assert g.get_successors("A") == [2, 3]
    assert g.get_predecessors("B") == [1]


class TestAttributeDictsAreNotShared:
    def _graph(self, directed):
        g = ec.Graph(directed=directed).add_node(label="A", x=1).add_node(label="B")
        return g.add_edge("A", "B", w=1)

    @pytest.mark.parametrize(
        "derive",
        [
            lambda g: g.to_undirected(),
            lambda g: g.to_directed(),
            lambda g: g.reverse(),
            lambda g: g.k_core(1),
            lambda g: g.minimum_spanning_tree(),
        ],
    )
    def test_derived_graphs_have_their_own_dicts(self, derive):
        g = self._graph(directed=True)
        try:
            h = derive(g)
        except ValueError:  # e.g. to_directed() on a directed graph
            h = derive(self._graph(directed=False))
            g = self._graph(directed=False)
        h.set_node_attrs("x", {h.get_id_from_label("A"): 99})
        h.set_edge_attrs("w", {e: 5 for e in h.get_edge_ids()})
        assert g.get_node_attrs("x") == {1: 1}
        assert set(g.get_edge_attrs("w").values()) == {1}

    def test_get_node_and_get_edge_are_snapshots(self):
        g = self._graph(directed=False)
        g.get_node(1).attributes["x"] = 42
        g.get_edge(g.get_edge_ids()[0]).attributes["w"] = 42
        assert g.get_node_attrs("x") == {1: 1}
        assert set(g.get_edge_attrs("w").values()) == {1}


def _shifted_graph():
    """A graph whose IDs don't start at 1, with node and edge attributes."""
    g = ec.Graph().add_node(label="tmp")
    for x in "ABCDEF":
        g.add_node(label=x, color=x.lower())
    g.delete_node("tmp")
    edges = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D"),
             ("A", "D")]
    for i, (u, v) in enumerate(edges):
        g.add_edge(u, v, rel=f"e{i}")
    return g


class TestDerivedGraphsKeepIdsAndAttributes:
    @pytest.mark.parametrize(
        "derive",
        [
            lambda g: g.contract_nodes("A", "B"),
            lambda g: g.contracted_edge("A", "B"),
            lambda g: g.contract_nodes_multi(["A", "B"]),
            lambda g: g.k_core(2),
            lambda g: g.k_shell(2),
            lambda g: g.k_crust(2),
            lambda g: g.k_truss(3),
            lambda g: g.k_corona(2),
            lambda g: g.minimum_spanning_tree(),
            lambda g: g.maximum_spanning_tree(),
            lambda g: g.steiner_tree(["A", "E"]),
            lambda g: g.prize_collecting_steiner_tree({"A": 5.0, "F": 5.0}),
            lambda g: g.all_spanning_trees(max_trees=2)[0],
            lambda g: g.node_percolation(0.3, seed=1),
            lambda g: g.edge_percolation(0.3, seed=1),
        ],
    )
    def test_ids_labels_and_attributes_kept(self, derive):
        g = _shifted_graph()
        h = derive(g)
        assert h.count_nodes() > 0
        for nid in h.get_node_ids():
            assert h.get_label_from_id(nid) == g.get_label_from_id(nid)
            assert h.get_node(nid).attributes == g.get_node(nid).attributes
        for eid in h.get_edge_ids():
            original = g.get_edge(eid)
            assert original is not None
            assert h.get_edge(eid).attributes == original.attributes

    def test_directed_steiner_tree(self):
        g = ec.Graph(directed=True)
        g.add_node(label="tmp")
        for x in "RABC":
            g.add_node(label=x, kind=x)
        g.delete_node("tmp")
        g.add_edge("R", "A", rel="ra").add_edge("A", "B", rel="ab").add_edge("R", "C", rel="rc")
        h = g.directed_steiner_tree("R", ["B", "C"])
        assert {h.get_label_from_id(n) for n in h.get_node_ids()} == {"R", "A", "B", "C"}
        for nid in h.get_node_ids():
            assert h.get_node(nid).attributes == g.get_node(nid).attributes
        assert sorted(h.get_edge_ids()) == sorted(g.get_edge_ids())
        assert g.directed_steiner_tree("R", ["B"], return_subgraph=False) == [
            ("A", "B", 1.0), ("R", "A", 1.0)
        ]

    def test_steiner_accepts_unlabelled_terminals(self):
        g = ec.Graph().add_path_graph(4, label=False)
        a, _, _, d = sorted(g.get_node_ids())
        assert sorted(g.steiner_tree([a, d]).get_node_ids()) == sorted(g.get_node_ids())

    def test_quotient_blocks_record_members(self):
        g = _shifted_graph()
        q = g.quotient_graph([["A", "B", "C"], ["D", "E", "F"]])
        assert q.get_node_attrs("members") == {1: [2, 3, 4], 2: [5, 6, 7]}


class TestGeneratorLabels:
    def test_labels_match_ids_after_deletion(self):
        """Labels were numbered from count_nodes() + 1, so they drifted from IDs."""
        g = ec.Graph().add_nodes(["X", "Y"]).delete_node("X")
        for build in (
            lambda h: h.add_path_graph(3),
            lambda h: h.add_star_graph(3),
            lambda h: h.add_full_graph(3),
            lambda h: h.add_balanced_tree(2, 1),
        ):
            before = set(g.get_node_ids())
            build(g)
            for nid in set(g.get_node_ids()) - before:
                assert g.get_label_from_id(nid) == str(nid)

    @pytest.mark.parametrize(
        "build, n_nodes, n_edges",
        [
            (lambda g, **kw: g.add_balanced_tree(2, 2, **kw), 7, 6),
            (lambda g, **kw: g.add_grid_2d(2, 3, **kw), 6, 7),
            (lambda g, **kw: g.add_prism_graph(3, **kw), 6, 9),
        ],
    )
    def test_label_false_wires_only_new_nodes(self, build, n_nodes, n_edges):
        """tree/grid/prism with label=False wired edges to nodes computed from count_nodes()."""
        g = ec.Graph().add_nodes(["X", "Y", "Z"]).delete_node("X")
        before = set(g.get_node_ids())
        build(g, label=False)
        new = set(g.get_node_ids()) - before
        assert len(new) == n_nodes and g.count_edges() == n_edges
        assert all(s in new and t in new for _, s, t in g.get_edges_as_tuples())
        labelled = build(ec.Graph())
        bare = build(ec.Graph(), label=False)
        assert sorted(bare.get_edges_as_tuples()) == sorted(labelled.get_edges_as_tuples())

    def test_prism_and_grid_labels(self):
        assert ec.Graph().add_prism_graph(3).get_label_from_id(4) == "4_bottom"
        assert ec.Graph().add_grid_2d(2, 2).get_label_from_id(4) == "(1,1)"

    def test_label_clash_raises_before_adding(self):
        g = ec.Graph().add_nodes(["2"])
        with pytest.raises(ValueError, match="already exist"):
            g.add_path_graph(3)
        assert g.count_nodes() == 1


class TestOptInWeights:
    def _graph(self):
        g = ec.Graph(weighted=True).add_nodes(["S", "A", "B", "T"])
        g.add_edge("S", "A", weight=5.0).add_edge("A", "T", weight=1.0)
        g.add_edge("S", "B", weight=1.0).add_edge("B", "T", weight=4.0)
        return g

    def test_cut_methods_take_capacity(self):
        g = self._graph()
        assert g.get_min_cut_between("S", "T") == 2
        assert g.get_min_cut_between("S", "T", capacity="weight") == 2.0
        cut = g.minimum_edge_cut("S", "T", capacity="weight")
        labels = {tuple(g.get_label_from_id(n) for n in e) for e in cut}
        assert labels == {("A", "T"), ("S", "B")}
        with pytest.raises(ValueError, match="no numeric 'cap' attribute"):
            g.minimum_edge_cut("S", "T", capacity="cap")

    def test_weighted_vitality_matches_networkx(self):
        g = self._graph().add_edge("A", "B", weight=2.0)
        G = nx.Graph()
        for _, s, t in g.get_edges_as_tuples():
            G.add_edge(g.get_label_from_id(s), g.get_label_from_id(t),
                       weight=g.get_edge(g.find_edge_ids(s, t)[0]).weight)
        # edgecase sums over ordered pairs, so on an undirected graph every pair counts twice
        want = nx.closeness_vitality(G, weight="weight")
        got = g.closeness_vitality(weighted=True, return_dict=True)
        for nid, value in got.items():
            assert value == pytest.approx(2 * want[g.get_label_from_id(nid)])
            assert g.get_node_vitality(nid, weighted=True) == pytest.approx(value)
        # Default stays hop-based
        hops = nx.closeness_vitality(G)
        for nid, value in g.closeness_vitality(return_dict=True).items():
            assert value == pytest.approx(2 * hops[g.get_label_from_id(nid)])


def test_unconnected_false_needs_strong_connectivity():
    """Diameter/radius checked weak connectivity while mean distance required strong."""
    chain = ec.Graph(directed=True).add_path_graph(3)
    for method in ("get_diameter", "get_radius", "get_mean_distance"):
        assert getattr(chain, method)(unconnected=False) is None
    ring = ec.Graph(directed=True).add_cycle_graph(3)
    assert ring.get_diameter(unconnected=False) == 2
    assert ring.get_radius(unconnected=False) == 2
    assert ring.get_mean_distance(unconnected=False) == 1.5


class TestSmallerOddities:
    def test_enumerate_trails_zero(self):
        assert ec.Graph().add_cycle_graph(4).enumerate_trails(1, 3, max_trails=0) == []

    def test_is_tree_directed_and_self_loop(self):
        assert ec.Graph(directed=True).add_nodes(["A"]).is_tree() is False
        assert ec.Graph(directed=True).is_tree() is False
        assert ec.Graph().add_nodes(["A"]).is_tree() is True
        assert ec.Graph().add_nodes(["A"]).add_edge("A", "A").is_tree() is False

    def test_random_edge_walk_digit_labels_are_labels(self):
        g = ec.Graph().add_nodes(["10", "20", "30"]).add_edges([("10", "20"), ("20", "30")])
        walk = g.random_edge_walk("10", steps=1, seed=1, return_list=True)
        assert len(walk) == 1 and g.get_id_from_label("10") in walk[0][1:]
        with pytest.raises(ValueError):
            g.random_edge_walk(10, steps=1)


class TestRemainingDerivedGraphsKeepIds:
    """Regression: subgraph/ego_graph/graph algebra/transforms renumbered node or edge IDs."""

    def _assert_kept(self, g, h):
        for nid in h.get_node_ids():
            assert h.get_label_from_id(nid) == g.get_label_from_id(nid)
            assert h.get_node(nid).attributes == g.get_node(nid).attributes
        for eid, s, t in h.get_edges_as_tuples():
            original = g.get_edge(eid)
            assert original is not None and {s, t} == {original.source, original.target}
            assert h.get_edge(eid).attributes == original.attributes

    @pytest.mark.parametrize(
        "derive",
        [
            lambda g: g.subgraph([g.get_id_from_label(x) for x in "ACD"]),
            lambda g: g.edge_subgraph(g.get_edge_ids()[2:5]),
            lambda g: g.ego_graph("C"),
            lambda g: g.to_undirected(),
            lambda g: g.reverse(),
            lambda g: g.select_nodes_by_id([g.get_id_from_label(x) for x in "ACD"])
            .transform_to_subgraph_ws(),
            lambda g: g.select_edges_by_id(g.get_edge_ids()[:3]).transform_to_subgraph_ws(),
        ],
    )
    def test_nodes_and_edges_keep_ids(self, derive):
        g = _shifted_graph()
        self._assert_kept(g, derive(g))

    def test_to_directed_keeps_ids_for_stored_direction(self):
        g = _shifted_graph()
        d = g.to_directed()
        assert d.count_edges() == 2 * g.count_edges()
        top = max(g.get_edge_ids())
        for eid, s, t in d.get_edges_as_tuples():
            if eid <= top:
                assert (s, t) == (g.get_edge(eid).source, g.get_edge(eid).target)
        assert sorted(e for e in d.get_edge_ids() if e > top) == list(
            range(top + 1, top + 1 + g.count_edges())
        )

    def test_graph_algebra_keeps_this_graphs_edge_ids(self):
        g = _shifted_graph()
        other = g.subgraph(g.get_node_ids())  # same IDs
        a, f = g.get_id_from_label("A"), g.get_id_from_label("F")
        other.add_edge(a, f, rel="new")
        for op in ("union", "compose", "intersection", "difference", "symmetric_difference"):
            h = getattr(g, op)(other)
            for eid, s, t in h.get_edges_as_tuples():
                if {s, t} == {a, f}:
                    assert eid > max(g.get_edge_ids()), op
                else:
                    assert g.get_edge(eid) is not None, op
                    assert {g.get_edge(eid).source, g.get_edge(eid).target} == {s, t}, op

    def test_bipartite_projection_keeps_node_ids_and_attributes(self):
        g = ec.Graph().add_node(label="tmp")
        for x in ["a1", "a2", "b1", "b2"]:
            g.add_node(label=x, side=x[0])
        g.delete_node("tmp")
        g.add_edges([("a1", "b1"), ("a2", "b1"), ("a2", "b2")])
        ids = {x: g.get_id_from_label(x) for x in ["a1", "a2", "b1", "b2"]}
        which = 0 if ids["a1"] in g.bipartite_sets()[0] else 1
        proj = g.bipartite_projection(which=which)
        assert sorted(proj.get_node_ids()) == [ids["a1"], ids["a2"]]
        assert proj.get_node_attrs("side") == {ids["a1"]: "a", ids["a2"]: "a"}
        assert [(s, t) for _, s, t in proj.get_edges_as_tuples()] == [(ids["a1"], ids["a2"])]

    def test_transform_to_subgraph_ws_unlabelled_and_weighted(self):
        g = ec.Graph(weighted=True).add_path_graph(4, label=False)
        g.select_nodes_by_id(sorted(g.get_node_ids())[:3])
        h = g.transform_to_subgraph_ws()
        assert h.is_weighted() and h.count_edges() == 2
        assert ec.Graph().transform_to_subgraph_ws().count_nodes() == 0
