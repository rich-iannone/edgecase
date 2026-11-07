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

    @pytest.mark.parametrize("bad", [999, -1])
    def test_unknown_edge_ids(self, bad):
        # negative IDs used to raise OverflowError, and delete_edge_by_id() a plain KeyError
        g = _dir_ab()
        assert g.has_edge_id(bad) is False
        assert g.get_node_ids_by_edge_id(bad) is None
        for call in (g.get_edge, g.delete_edge_by_id, lambda x: g.delete_edge(id=x)):
            with pytest.raises(ec.EdgeNotFoundError, match=f"Edge with ID {bad} does not exist"):
                call(bad)
        assert g.count_edges() == 2

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.k_core(-1),
            lambda g: g.k_shell(-1),
            lambda g: g.k_crust(-1),
            lambda g: g.k_corona(-1),
            lambda g: g.k_truss(-1),
            lambda g: g.is_k_regular(-1),
            lambda g: g.k_shortest_paths("A", "B", -1),
            lambda g: g.power(-1),
            lambda g: g.random_walk("A", -1),
            lambda g: g.random_edge_walk("A", -1),
            lambda g: g.select_last_nodes_created(-1),
            lambda g: g.select_last_edges_created(-1),
            lambda g: g.cmty_fluid(-1),
            lambda g: g.cmty_fluid(1, max_iter=-1),
            lambda g: g.lukes_partitioning(-1),
            lambda g: g.influence_maximization(-1),
            lambda g: g.influence_maximization(1, n_simulations=-1),
            lambda g: g.prominent_group(-1),
            lambda g: g.add_gnp_graph(-1, 0.5),
            lambda g: g.add_gnm_graph(3, -1),
            lambda g: g.add_pa_graph(-1),
            lambda g: g.add_smallworld_graph(-1),
            lambda g: g.add_random_geometric_graph(5, 0.5, dim=-1),
            lambda g: g.add_johnson_graph(4, -1),
        ],
    )
    def test_negative_counts_raise_value_error(self, call):
        # pyo3 used to raise OverflowError converting them to unsigned integers
        g = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        steps = len(g.steps)
        with pytest.raises(ValueError, match="must be non-negative"):
            call(g)
        assert len(g.steps) == steps and g.count_nodes() == 2

    def test_negative_degree_raises_value_error(self):
        with pytest.raises(ValueError, match="degree_sequence must hold non-negative integers"):
            ec.Graph().configuration_model([2, -1, 1])


def _seeded_base():
    return ec.Graph().add_connected_watts_strogatz_graph(16, 4, 0.3, seed=1)


def _fingerprint(result):
    if isinstance(result, ec.Graph):
        return result.get_edges_str_repr(), result.get_node_df().to_dicts()
    if isinstance(result, dict):  # some come from a Rust HashMap, in no fixed key order
        return sorted((repr(k), _fingerprint(v)) for k, v in result.items())
    if callable(result):  # alt_heuristic() returns the heuristic h(u, v)
        return [result(1, v) for v in range(1, 17)]
    return repr(result)


_SEED_NO_ARGS = [
    "alt_heuristic",
    "approximate_current_flow_betweenness_centrality",
    "cmty_fast_label_propagation",
    "cmty_label_propagation",
    "cmty_leiden",
    "cmty_leiden_levels",
    "cmty_louvain_levels",
    "cmty_stochastic_block_model",
    "degree_preserving_rewire",
    "find_maximal_independent_sets",
    "greedy_color",
    "greedy_node_swap_bipartition",
    "kernighan_lin_bisection",
    "lattice_reference",
    "layout_arf",
    "layout_forceatlas2",
    "layout_random",
    "layout_spring",
    "max_cut",
    "maximal_independent_set",
    "motif_significance",
    "network_robustness",
    "percolation_threshold_analysis",
    "random_reference",
    "random_spanning_tree",
    "rich_club_coefficient",
    "small_world_omega",
    "small_world_sigma",
    "traveling_salesman",
]

_SEED_CALLS = [
    *[
        pytest.param(lambda g, s, name=name: getattr(g, name)(seed=s), id=name)
        for name in _SEED_NO_ARGS
    ],
    lambda g, s: g.add_bipartite_configuration_model([1, 1], [2], seed=s),
    lambda g, s: g.add_bipartite_gnm_graph(3, 3, 4, seed=s),
    lambda g, s: g.add_bipartite_gnp_graph(3, 3, 0.5, seed=s),
    lambda g, s: g.add_bipartite_preferential_attachment_graph([1, 2, 1], 0.5, seed=s),
    lambda g, s: g.add_connected_watts_strogatz_graph(10, 4, 0.3, seed=s),
    lambda g, s: g.add_directed_configuration_model([1, 1, 0], [0, 1, 1], seed=s),
    lambda g, s: g.add_dual_barabasi_albert_graph(10, 1, 2, 0.5, seed=s),
    lambda g, s: g.add_duplication_divergence_graph(10, 0.5, seed=s),
    lambda g, s: g.add_extended_barabasi_albert_graph(10, 1, 0.2, 0.2, seed=s),
    lambda g, s: g.add_gaussian_random_partition_graph(20, 5, 2, 0.5, 0.1, seed=s),
    lambda g, s: g.add_geographical_threshold_graph(10, 10, seed=s),
    lambda g, s: g.add_gn_graph(10, seed=s),
    lambda g, s: g.add_gnc_graph(10, seed=s),
    lambda g, s: g.add_gnm_graph(10, 15, seed=s),
    lambda g, s: g.add_gnp_graph(10, 0.3, seed=s),
    lambda g, s: g.add_gnr_graph(10, 0.5, seed=s),
    lambda g, s: g.add_joint_degree_graph(g.degree_mixing_dict(), seed=s),
    lambda g, s: g.add_lfr_graph(n=100, mu=0.2, average_degree=6, seed=s),
    lambda g, s: g.add_maybe_regular_expander(10, 4, seed=s),
    lambda g, s: g.add_navigable_small_world_graph(4, seed=s),
    lambda g, s: g.add_newman_watts_strogatz_graph(10, 2, 0.3, seed=s),
    lambda g, s: g.add_pa_graph(10, seed=s),
    lambda g, s: g.add_partial_duplication_graph(10, 3, 0.5, 0.5, seed=s),
    lambda g, s: g.add_planted_partition_graph(2, 5, 0.8, 0.1, seed=s),
    lambda g, s: g.add_powerlaw_cluster_graph(10, 2, 0.3, seed=s),
    lambda g, s: g.add_random_clustered_graph([(1, 2)] * 6, seed=s),
    lambda g, s: g.add_random_degree_sequence_graph([2, 2, 2, 2], seed=s),
    lambda g, s: g.add_random_geometric_graph(10, 0.5, seed=s),
    lambda g, s: g.add_random_internet_as_graph(30, seed=s),
    lambda g, s: g.add_random_k_out_graph(10, 2, 1.0, seed=s),
    lambda g, s: g.add_random_kernel_graph(10, lambda u, w, z: 5 * (z - w), seed=s),
    lambda g, s: g.add_random_labeled_tree(10, seed=s),
    lambda g, s: g.add_random_lobster(10, 0.5, 0.5, seed=s),
    lambda g, s: g.add_random_partition_graph([3, 4], 0.8, 0.1, seed=s),
    lambda g, s: g.add_random_powerlaw_tree(10, seed=s),
    lambda g, s: g.add_random_regular_expander_graph(10, 4, seed=s),
    lambda g, s: g.add_random_regular_graph(10, 3, seed=s),
    lambda g, s: g.add_relaxed_caveman_graph(3, 4, 0.2, seed=s),
    lambda g, s: g.add_scale_free_graph(10, seed=s),
    lambda g, s: g.add_smallworld_graph(10, seed=s),
    lambda g, s: g.add_soft_random_geometric_graph(10, 0.5, seed=s),
    lambda g, s: g.add_waxman_graph(10, seed=s),
    lambda g, s: g.cmty_fluid(2, seed=s),
    lambda g, s: g.configuration_model([2, 2, 2], seed=s),
    lambda g, s: g.edge_percolation(0.5, seed=s),
    lambda g, s: g.expected_degree_graph([1, 2, 2, 1], seed=s),
    lambda g, s: g.generate_random_paths(3, seed=s),
    lambda g, s: g.generate_stochastic_block_model([3, 3], [0.8, 0.1, 0.1, 0.8], seed=s),
    lambda g, s: g.influence_maximization(2, seed=s),
    lambda g, s: g.node_percolation(0.5, seed=s),
    lambda g, s: g.panther_similarity(1, seed=s),
    lambda g, s: g.panther_vector_similarity(1, seed=s),
    lambda g, s: g.random_edge_walk(1, 5, seed=s),
    lambda g, s: g.random_k_lift(2, seed=s),
    lambda g, s: g.random_walk(1, 5, seed=s),
    lambda g, s: g.transform_with_spanner(3, seed=s),
]


@pytest.mark.parametrize("seed", [-1, -(2**70), 2**70])
@pytest.mark.parametrize("call", _SEED_CALLS)
def test_any_integer_seed_is_accepted(call, seed):
    """Rust-backed methods raised OverflowError for a seed outside the u64 range."""
    first = _fingerprint(call(_seeded_base(), seed))
    assert _fingerprint(call(_seeded_base(), seed)) == first


def test_seed_is_taken_modulo_2_64():
    def walk(seed):
        return _seeded_base().random_walk(1, 20, seed=seed, return_list=True)

    assert walk(-1) == walk(2**64 - 1) == walk(2**65 - 1)
    assert walk(-1) != walk(1)


def test_non_integer_seed_still_raises_type_error():
    with pytest.raises(TypeError):
        _seeded_base().random_walk(1, 5, seed=1.5)


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

    def test_delete_drops_incident_edges_from_selection(self):
        """delete_node left stale edge IDs in the selection."""
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        g.select_edges_by_id(g.get_edge_ids())
        g.delete_node("A")
        assert g.get_selected_edge_ids() == g.find_edge_ids("B", "C")

    def test_delete_nodes_ws_leaves_no_selection(self):
        """Selecting nodes clears the edge selection, and deleting them empties the selection."""
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        g.select_edges_by_id(g.get_edge_ids())
        g.select_nodes_by_id([g.get_id_from_label("A")]).delete_nodes_ws()
        assert g.get_selected_edge_ids() == [] and g.get_selected_node_ids() == []


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
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "D"),
        ("A", "D"),
    ]
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
            ("A", "B", 1.0),
            ("R", "A", 1.0),
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
            G.add_edge(
                g.get_label_from_id(s),
                g.get_label_from_id(t),
                weight=g.get_edge(g.find_edge_ids(s, t)[0]).weight,
            )
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
            lambda g: g.select_nodes_by_id(
                [g.get_id_from_label(x) for x in "ACD"]
            ).transform_to_subgraph_ws(),
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


class TestPlotTraversalBfsTree:
    """Regression: tree edges attached each node to its most recently visited neighbor (the DFS
    rule), which is wrong for BFS, including the default BFS run from start=."""

    @staticmethod
    def _tree(player):
        import json
        import re

        html = player._repr_html_()
        steps = json.loads(re.search(r"var STEPS=(\[.*?\]);", html, re.S).group(1))
        return sorted(tuple(e) for e in steps[-1]["tree_edges"])

    def _graph(self):
        # Triangle 1-2-3 plus 3-4: BFS from 1 reaches 2 and 3 directly
        return ec.Graph().add_cycle_graph(3).add_node(label="4").add_edge("3", "4")

    def test_start_uses_real_bfs_tree(self):
        g = self._graph()
        assert self._tree(g.plot_traversal(start=1)) == [("1", "2"), ("1", "3"), ("3", "4")]

    def test_inference_follows_algorithm(self):
        g = self._graph()
        order = [1, 2, 3, 4]
        bfs = self._tree(g.plot_traversal(visit_order=order, algorithm="BFS"))
        dfs = self._tree(g.plot_traversal(visit_order=order, algorithm="DFS"))
        assert bfs == [("1", "2"), ("1", "3"), ("3", "4")]
        assert dfs == [("1", "2"), ("2", "3"), ("3", "4")]


class TestLabelOnlyRustEntryPoints:
    """Regression: expand_seed_set and is_simple_path worked on labels in Rust, so unlabelled
    nodes failed and digit-only labels were confused with IDs."""

    def test_expand_seed_set_accepts_ids_and_unlabelled_nodes(self):
        g = ec.Graph().add_full_graph(4, label=False)
        assert g.expand_seed_set([1, 2]) == [1, 2, 3, 4]
        labelled = ec.Graph().add_full_graph(4)
        assert labelled.expand_seed_set(["1", 2]) == [1, 2, 3, 4]
        with pytest.raises(ValueError, match="does not exist"):
            g.expand_seed_set([99])

    def test_is_simple_path_ids_vs_digit_labels(self):
        # Node 1 is unlabelled; node 2 is labelled "1"; only 1 -- 3 is an edge
        g = ec.Graph().add_node().add_node(label="1").add_node(label="C").add_edge(1, 3)
        assert g.is_simple_path([1, "C"]) is True
        assert g.is_simple_path(["1", "C"]) is False  # the node labelled "1" isn't adjacent
        assert g.is_simple_path([1, 99]) is False
        assert g.is_simple_path([1, "C", 1]) is False


class TestResolverChecksIntegerIds:
    """Regression: _resolve_node_id passed any integer through, so ~90 methods returned quiet
    defaults for a missing ID while raising for a missing label."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.get_node(99),
            lambda g: g.get_degree(99),
            lambda g: g.neighbors(99, mode="in"),
            lambda g: g.has_path(1, 99),
            lambda g: g.shortest_path(1, 99),
            lambda g: g.shortest_path_length(1, 99),
            lambda g: g.bfs(99),
            lambda g: g.dfs(99),
            lambda g: g.descendants(99),
            lambda g: g.common_neighbors([(1, 99)]),
            lambda g: g.get_non_neighbors(99),
            lambda g: g.single_source_shortest_path_length(99),
            lambda g: g.select_nodes_in_neighborhood([99]),
            lambda g: g.set_node_attrs("x", {99: 1}),
            lambda g: g.max_flow_value(1, 99),
            lambda g: g.contract_nodes(1, 99),
            lambda g: g.ego_graph(99),
        ],
    )
    def test_unknown_id_raises(self, call):
        g = ec.Graph(directed=True).add_path_graph(3)
        with pytest.raises(ValueError, match="Node 99 does not exist"):
            call(g)

    def test_bools_and_negative_ids(self):
        g = ec.Graph().add_path_graph(2)
        with pytest.raises(TypeError):
            g.get_degree(True)
        with pytest.raises(ValueError):
            g.get_degree(-1)
        assert g.has_node(True) is False and g.has_edge(True, 2) is False


class TestDegreesFromIndex:
    """get_degree / get_in_degree / degrees() used to scan every edge per node on directed
    graphs (O(V·E) for degrees()); they now read counts kept in the edge index, which must stay
    right as edges are added and deleted."""

    def _check(self, g):
        G = nx.MultiDiGraph()
        G.add_nodes_from(g.get_node_ids())
        G.add_edges_from((s, t) for _, s, t in g.get_edges_as_tuples())
        for nid in g.get_node_ids():
            assert g.get_in_degree(nid) == G.in_degree(nid)
            assert g.get_out_degree(nid) == G.out_degree(nid)
            assert g.get_degree(nid) == G.degree(nid)
        assert g.degrees(return_dict=True) == dict(G.degree())
        assert list(g.degrees(return_dict=True)) == g.get_node_ids(limit=None)

    def test_directed_with_parallel_edges_and_self_loops(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B"), ("A", "B"), ("B", "C"), ("C", "C"), ("C", "A")])
        self._check(g)
        g.add_edge("B", "A")  # index updated in place
        self._check(g)
        g.delete_edge("A", "B")  # index rebuilt
        self._check(g)
        g.delete_node("C")
        self._check(g)

    def test_undirected_unchanged(self):
        g = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "B"), ("B", "B")])
        assert g.get_degree("A") == 2 and g.get_degree("B") == 4
        assert g.degrees(return_dict=True) == {1: 2, 2: 4}


class TestUniqueLabelsAndIds:
    """Labels weren't checked for uniqueness: a repeated label left the earlier node unreachable
    by label, and re-adding an existing ID overwrote that node."""

    def test_add_node_rejects_existing_label(self):
        g = ec.Graph().add_node(label="A", x=1)
        with pytest.raises(ValueError, match="Node with label 'A' already exists"):
            g.add_node(label="A")
        assert g.count_nodes() == 1 and g.get_node_attrs("x") == {1: 1}

    def test_add_nodes_is_atomic(self):
        g = ec.Graph().add_nodes(["A"])
        with pytest.raises(ValueError, match="'A' already exists"):
            g.add_nodes(["B", "A"])
        with pytest.raises(ValueError, match="'C' already exists"):
            g.add_nodes(["C", "D", "C"])
        assert g.count_nodes() == 1

    def test_existing_id_is_not_overwritten(self):
        from edgecase import _core as rust

        g = ec.Graph().add_node(label="A", x=1)
        with pytest.raises(ValueError, match="Node with ID 1 already exists"):
            g._graph.add_node(rust.Node(id=1, label="B", attributes=None))
        assert g.get_label_from_id(1) == "A" and g.get_node_attrs("x") == {1: 1}

    @pytest.mark.parametrize(
        "build",
        [
            lambda g: g.add_johnson_graph(4, 2),
            lambda g: g.add_random_geometric_graph(5, 0.5, seed=1),
        ],
    )
    def test_content_labelled_generators_check_first(self, build):
        g = build(ec.Graph())
        n, m = g.count_nodes(), g.count_edges()
        with pytest.raises(ValueError, match="already exists"):
            build(g)
        assert (g.count_nodes(), g.count_edges()) == (n, m)

    def test_unlabelled_nodes_are_unaffected(self):
        g = ec.Graph().add_node().add_node().add_path_graph(3, label=False)
        assert g.count_nodes() == 5


class TestNotFoundErrors:
    """Missing nodes and edges raised a mix of ValueError and KeyError (and get_edge /
    get_label_from_id returned None); they now raise NodeNotFoundError / EdgeNotFoundError,
    which are both ValueError and KeyError."""

    def _graph(self):
        return ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.get_node(99),
            lambda g: g.get_node("Z"),
            lambda g: g.get_label_from_id(99),
            lambda g: g.delete_node(99),
            lambda g: g.delete_node("Z"),
            lambda g: g.select_nodes_by_id([1, 99]),
            lambda g: g.neighbors("Z"),
            lambda g: g.add_node(label="C", from_node="Z"),
        ],
    )
    def test_nodes(self, call):
        g = self._graph()
        for expected in (ec.NodeNotFoundError, ValueError, KeyError):
            with pytest.raises(expected):
                call(g)
        assert g.count_nodes() == 2 and g.get_selected_node_ids() == []

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.get_edge(99),
            lambda g: g.delete_edge(id=99),
            lambda g: g.select_edges_by_id([99]),
        ],
    )
    def test_edges(self, call):
        g = self._graph()
        for expected in (ec.EdgeNotFoundError, ValueError, KeyError):
            with pytest.raises(expected):
                call(g)
        assert g.count_edges() == 1

    def test_message_has_no_keyerror_quotes(self):
        with pytest.raises(ec.NodeNotFoundError) as info:
            self._graph().delete_node(99)
        assert str(info.value) == "Node 99 does not exist in the graph"

    def test_get_label_from_id_unlabelled_is_none(self):
        g = ec.Graph().add_node()
        assert g.get_label_from_id(1) is None

    def test_add_node_connects_existing_nodes(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B"])
        g.add_node(label="C", from_node="A", to_node=2)
        assert sorted(
            (g.get_label_from_id(s), g.get_label_from_id(t)) for _, s, t in g.get_edges_as_tuples()
        ) == [("B", "C"), ("C", "A")]


def test_delete_edge_with_no_edge_between_nodes():
    g = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
    with pytest.raises(ec.EdgeNotFoundError, match="No edge found from 'B' to 'A'"):
        g.delete_edge("B", "A")
    with pytest.raises(KeyError):
        g.delete_edge("B", "A")


class TestNaNWeightsRejected:
    """A NaN weight compares false with everything, so sorting and best-choice steps aborted
    the process (label propagation, Kernighan-Lin, degree mixing) or looped (eulerize). NaN
    weights are now rejected where they enter; infinite weights are still allowed."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.add_edge("A", "B", weight=float("nan")),
            lambda g: g.add_edges([("A", "B", {"weight": float("nan")})]),
            lambda g: g.set_edge_attrs("weight", float("nan")),
            lambda g: g.set_edge_attrs("weight", {1: float("nan")}),
        ],
    )
    def test_nan_weight_raises(self, call):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"]).add_edge("B", "C", weight=1.0)
        with pytest.raises(ValueError, match="NaN"):
            call(g)
        assert [g.get_edge(e).weight for e in g.get_edge_ids()] == [1.0]

    def test_infinite_weight_allowed(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=float("inf"))
        assert g.get_edge(g.get_edge_ids()[0]).weight == float("inf")
