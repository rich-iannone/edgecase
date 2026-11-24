"""Methods that compute a result record one step that carries it, so `play()` can draw it.

Scores travel as `metric_values`, partitions as `node_groups` and set-shaped results (paths,
bridges, a spanning tree, an odd cycle) as the selected nodes and edges. Writing a result to a
node attribute must not add a second, generic `set_node_attrs` step.
"""

import pytest

import edgecase as ec


def karate():
    return ec.karate_club()


def new_steps(graph, call):
    n = len(graph.steps)
    call(graph)
    return graph.steps[n:]


def only_step(graph, call):
    steps = new_steps(graph, call)
    assert [s.method for s in steps] == [steps[0].method], [s.method for s in steps]
    return steps[0]


SCORES = [
    ("pagerank", lambda g: g.pagerank(), "pagerank"),
    ("pagerank", lambda g: g.pagerank(damping=0.9, attr_name="pr"), "pr"),
    ("betweenness_centrality", lambda g: g.betweenness_centrality(), "betweenness_centrality"),
    ("harmonic_centrality", lambda g: g.harmonic_centrality(), "harmonic_centrality"),
    ("eigen_centrality", lambda g: g.eigen_centrality(), "eigen_centrality"),
    ("katz_centrality", lambda g: g.katz_centrality(), "katz_centrality"),
    ("degrees", lambda g: g.degrees(), "degree"),
    ("closeness_vitality", lambda g: g.closeness_vitality(), "closeness_vitality"),
    ("laplacian_centrality", lambda g: g.laplacian_centrality(), "laplacian_centrality"),
    ("bridging_centrality", lambda g: g.bridging_centrality(), "bridging_centrality"),
]


@pytest.mark.parametrize("method, call, attr", SCORES)
def test_scores_record_metric_values(method, call, attr):
    g = karate()
    step = only_step(g, call)
    assert (step.kind, step.method) == ("metric", method)
    assert step.delta.attrs_changed == {attr: 34}
    stored = {g._node_label(n): v for n, v in g.get_node_attrs(attr).items()}
    assert step.delta.metric_values == pytest.approx({k: float(v) for k, v in stored.items()})
    assert "highest" in step.narration


def test_score_code_repeats_the_call():
    g = karate()
    assert only_step(g, lambda g: g.pagerank()).code == "G.pagerank()"
    step = only_step(g, lambda g: g.pagerank(damping=0.9, attr_name="pr"))
    assert step.code == "G.pagerank(damping=0.9, attr_name='pr')"


GROUPS = [
    ("cmty_louvain", lambda g: g.cmty_louvain(), "cmty_louvain"),
    ("cmty_fast_greedy", lambda g: g.cmty_fast_greedy(), "cmty_fast_greedy"),
    ("cmty_walktrap", lambda g: g.cmty_walktrap(), "cmty_walktrap"),
    ("cmty_l_eigenvec", lambda g: g.cmty_l_eigenvec(), "cmty_l_eigenvec"),
    (
        "cmty_stochastic_block_model",
        lambda g: g.cmty_stochastic_block_model(k=2, seed=1),
        "sbm_block",
    ),
]


@pytest.mark.parametrize("method, call, attr", GROUPS)
def test_partitions_record_node_groups(method, call, attr):
    g = karate()
    step = only_step(g, call)
    assert (step.kind, step.method) == ("metric", method)
    stored = {g._node_label(n): v for n, v in g.get_node_attrs(attr).items()}
    assert step.delta.node_groups == stored
    assert "grouped 34 nodes" in step.narration


def test_return_dict_records_nothing():
    g = karate()
    assert new_steps(g, lambda g: g.pagerank(return_dict=True)) == []
    assert new_steps(g, lambda g: g.cmty_stochastic_block_model(k=2, return_dict=True)) == []
    assert "sbm_block" not in g.get_node_df().columns


def test_shortest_path_selects_the_route():
    g = ec.metro()
    step = only_step(g, lambda g: g.shortest_path("Westgate", "Airport"))
    route = [g._node_label(n) for n in g.shortest_path("Westgate", "Airport", return_list=True)]
    assert step.kind == "query"
    assert step.delta.nodes_selected == route
    assert step.delta.edges_selected == list(zip(route, route[1:]))
    assert step.delta.attrs_changed == {"shortest_path_order": len(route)}
    hops = len(route) - 1
    assert step.narration == f"shortest_path() followed {hops} edges from Westgate to Airport."


@pytest.mark.parametrize("method", ["random_walk", "random_edge_walk"])
def test_walks_select_the_walk(method):
    g = ec.metro()
    step = only_step(g, lambda g: getattr(g, method)("Central", steps=4, seed=3))
    assert step.kind == "query"
    assert step.delta.nodes_selected[0] == "Central"
    assert len(step.delta.edges_selected) >= 1
    assert step.narration.startswith(f"{method}() followed")


def test_bridges_select_edges():
    g = ec.metro()
    step = only_step(g, lambda g: g.bridges())
    want = sorted(g._edge_ends(e) for e in g.bridges())
    assert sorted(step.delta.edges_selected) == want
    assert step.narration == f"bridges() found {len(want)} bridges."


def test_minimum_spanning_tree_selects_tree_edges():
    g = ec.metro()
    step = only_step(g, lambda g: g.minimum_spanning_tree())
    assert len(step.delta.edges_selected) == g.count_nodes() - 1
    assert step.code == "G.minimum_spanning_tree()"


def test_maximum_flow_records_edge_flows():
    g = ec.pipeline()
    step = only_step(g, lambda g: g.maximum_flow("Reservoir", "City"))
    flows = g.maximum_flow("Reservoir", "City")
    assert step.delta.edge_metric_values == {g._edge_ends(e): float(f) for e, f in flows.items()}
    total = g.maximum_flow_value("Reservoir", "City")
    assert step.narration.endswith(f"total {total:g}.")


def test_greedy_color_records_classes():
    g = ec.exams()
    step = only_step(g, lambda g: g.greedy_color(strategy="dsatur"))
    colors = g.greedy_color(strategy="dsatur")
    assert step.delta.node_groups == {g._node_label(n): c for n, c in colors.items()}
    assert step.code == "G.greedy_color(strategy='dsatur')"
    assert step.narration == f"greedy_color() used {len(set(colors.values()))} colors."


def test_is_bipartite_records_sides_or_an_odd_cycle():
    teams = ec.teams()
    step = only_step(teams, lambda g: g.is_bipartite())
    assert step.result is True
    assert sorted(set(step.delta.node_groups.values())) == [0, 1]

    g = ec.Graph.from_edges(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "E"), ("E", "B")]
    )
    step = only_step(g, lambda g: g.is_bipartite())
    assert step.result is False
    assert sorted(step.delta.nodes_selected) == ["A", "B", "E"]
    assert len(step.delta.edges_selected) == 3
    assert step.narration == "Not bipartite: odd cycle A–B–E."


def test_nothing_recorded_inside_a_compound_call():
    g = karate()
    with g._no_recording():
        g.pagerank()
        g.bridges()
        g.is_bipartite()
    assert g.get_node_attrs("pagerank")
    assert [s.method for s in g.steps][-1] != "pagerank"


def test_involves_counts_selected_nodes_but_not_scores():
    g = ec.metro()
    g.shortest_path("Westgate", "Airport")
    g.bridges()
    g.pagerank()
    assert [s.method for s in g.steps.filter(involves="Airport")] == [
        "metro", "shortest_path", "bridges",
    ]
    # Docks is only reached through a bridge, and every node gets a PageRank score
    assert [s.method for s in g.steps.filter(involves="Docks")] == ["metro", "bridges"]


def test_printing_a_graph_records_nothing():
    # The summary checks bipartiteness, which mustn't land in the step log
    g = ec.metro()
    before = [s.method for s in g.steps]
    repr(g)
    str(g)
    assert [s.method for s in g.steps] == before


def test_player_frames_carry_the_results():
    g = ec.pipeline()
    g.maximum_flow("Reservoir", "City")
    g.greedy_color()
    g.pagerank()
    html = g.play(seed=1)._repr_html_()
    for key in ('"emv"', '"ng"', '"mv"'):
        assert key in html


def test_player_shows_each_voterank_round():
    g = ec.Graph().add_path_graph(5)
    g.voterank(count=2)
    html = g.play(seed=1)._repr_html_()
    assert '"code": "elect 2"' in html and '"code": "elect 4"' in html
    assert '"sn": ["2"]' in html and '"sn": ["2", "4"]' in html
    assert '"parent_code": "G.voterank(count=2)"' in html
