"""Tests for closeness_centrality(), degree_centrality(), and edge_betweenness_centrality().

Each method is validated against NetworkX on random graphs (directed and undirected, including
disconnected ones), checked on hand-computed cases, and exercised through the step log and the
step-through player.
"""

import networkx as nx
import pytest

from edgecase import Graph
from edgecase.steps import KIND_METRIC

TOL = 1e-9


def _from_nx(G: nx.Graph) -> Graph:
    g = Graph(directed=G.is_directed())
    g.add_nodes([str(n) for n in G.nodes])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
    return g


def _by_label(g: Graph, scores: dict[int, float]) -> dict[int, float]:
    return {int(g.get_label_from_id(nid)): v for nid, v in scores.items()}


def _edges_by_label(g: Graph, scores: dict[int, float]) -> dict[tuple[int, int], float]:
    out = {}
    for eid, value in scores.items():
        s, t = g.get_node_ids_by_edge_id(eid)
        out[(int(g.get_label_from_id(s)), int(g.get_label_from_id(t)))] = value
    return out


def _random_graphs(directed: bool):
    # p is low enough that many of these graphs are disconnected
    for seed in range(12):
        yield nx.gnp_random_graph(20, 0.12, seed=seed, directed=directed)


def _strip() -> Graph:
    """The seven-node graph used by the doc figures."""
    return (
        Graph()
        .add_nodes(list("ABCDEFG"))
        .add_edges(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
                ("C", "E"),
                ("D", "E"),
                ("E", "F"),
                ("F", "G"),
            ]
        )
    )


# --- closeness_centrality ---------------------------------------------------------


class TestClosenessCentrality:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("wf_improved", [True, False])
    def test_matches_networkx(self, directed, wf_improved):
        for G in _random_graphs(directed):
            g = _from_nx(G)
            # NetworkX measures incoming distance on digraphs, i.e. mode="in"
            mode = "in" if directed else "out"
            ours = _by_label(
                g, g.closeness_centrality(mode=mode, wf_improved=wf_improved, return_dict=True)
            )
            ref = nx.closeness_centrality(G, wf_improved=wf_improved)
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_out_mode_matches_networkx_on_reverse(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours = _by_label(g, g.closeness_centrality(mode="out", return_dict=True))
            ref = nx.closeness_centrality(G.reverse())
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_all_mode_ignores_direction(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours = _by_label(g, g.closeness_centrality(mode="all", return_dict=True))
            ref = nx.closeness_centrality(G.to_undirected())
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_path_hand_computed(self):
        g = Graph().add_nodes(list("ABCDE"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        c = g.closeness_centrality(return_dict=True)
        lab = {g.get_label_from_id(k): v for k, v in c.items()}
        # C: distances 2+1+1+2 = 6 -> 4/6; A: 1+2+3+4 = 10 -> 4/10
        assert lab["C"] == pytest.approx(4 / 6)
        assert lab["A"] == pytest.approx(0.4)
        assert lab["A"] == lab["E"] and lab["B"] == lab["D"]

    def test_strip_values_used_in_figure(self):
        g = _strip()
        lab = {
            g.get_label_from_id(k): round(v, 2)
            for k, v in g.closeness_centrality(return_dict=True).items()
        }
        assert lab == {"A": 0.46, "B": 0.5, "C": 0.67, "D": 0.6, "E": 0.67, "F": 0.5, "G": 0.35}

    def test_wf_correction_on_disconnected_graph(self):
        g = Graph().add_nodes(["Hub", "A", "B", "C", "X", "Y"])
        g.add_edges([("Hub", "A"), ("Hub", "B"), ("Hub", "C"), ("X", "Y")])
        corrected = g.closeness_centrality(return_dict=True)
        raw = g.closeness_centrality(wf_improved=False, return_dict=True)
        x, hub = g.get_id_from_label("X"), g.get_id_from_label("Hub")
        assert raw[x] == pytest.approx(1.0)
        assert corrected[x] == pytest.approx(0.2)  # 1 * (1 / 5)
        assert corrected[hub] > corrected[x]

    def test_empty_single_and_isolated(self):
        assert Graph().closeness_centrality(return_dict=True) == {}
        g = Graph().add_node(label="A")
        assert list(g.closeness_centrality(return_dict=True).values()) == [0.0]
        g = Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        assert g.closeness_centrality(return_dict=True)[g.get_id_from_label("C")] == 0.0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _strip().closeness_centrality(mode="sideways")

    def test_stores_attribute_and_chains(self):
        g = _strip().closeness_centrality()
        assert "closeness_centrality" in g.get_node_df().columns
        g.closeness_centrality(attr_name="close", mode="all")
        assert "close" in g.get_node_df().columns
        assert ("closeness_centrality", {"mode": "all"}, "close") in g._computations

    def test_return_dict_does_not_store(self):
        g = _strip()
        g.closeness_centrality(return_dict=True)
        assert "closeness_centrality" not in g.get_node_df().columns


# --- degree_centrality ------------------------------------------------------------


class TestDegreeCentrality:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        for G in _random_graphs(directed):
            g = _from_nx(G)
            ours = _by_label(g, g.degree_centrality(return_dict=True))
            for n, v in nx.degree_centrality(G).items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_in_and_out_match_networkx(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours_in = _by_label(g, g.degree_centrality(mode="in", return_dict=True))
            ours_out = _by_label(g, g.degree_centrality(mode="out", return_dict=True))
            for n, v in nx.in_degree_centrality(G).items():
                assert ours_in[n] == pytest.approx(v, abs=TOL)
            for n, v in nx.out_degree_centrality(G).items():
                assert ours_out[n] == pytest.approx(v, abs=TOL)

    def test_mode_ignored_for_undirected(self):
        g = _strip()
        assert g.degree_centrality(mode="in", return_dict=True) == g.degree_centrality(
            return_dict=True
        )

    def test_star(self):
        g = Graph().add_nodes(["Hub", "A", "B", "C", "D"])
        g.add_edges([("Hub", x) for x in "ABCD"])
        c = g.degree_centrality(return_dict=True)
        assert c[g.get_id_from_label("Hub")] == pytest.approx(1.0)
        assert c[g.get_id_from_label("A")] == pytest.approx(0.25)

    def test_self_loop_counts_twice_like_networkx(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 2)])
        g = _from_nx(G)
        ours = _by_label(g, g.degree_centrality(return_dict=True))
        for n, v in nx.degree_centrality(G).items():
            assert ours[n] == pytest.approx(v)

    def test_empty_and_single(self):
        assert Graph().degree_centrality(return_dict=True) == {}
        g = Graph().add_node(label="A")
        assert list(g.degree_centrality(return_dict=True).values()) == [1.0]

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _strip().degree_centrality(mode="both")

    def test_stores_attribute(self):
        g = Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        g.degree_centrality(mode="in", attr_name="indeg")
        df = g.get_node_df().sort("label")
        assert df["indeg"].to_list() == [0.0, 1.0]


# --- edge_betweenness_centrality --------------------------------------------------


class TestEdgeBetweennessCentrality:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_networkx(self, directed, normalized):
        for G in _random_graphs(directed):
            g = _from_nx(G)
            ours = _edges_by_label(
                g, g.edge_betweenness_centrality(normalized=normalized, return_dict=True)
            )
            ref = nx.edge_betweenness_centrality(G, normalized=normalized)
            for (u, v), val in ref.items():
                got = ours.get((u, v))
                if got is None and not directed:
                    got = ours[(v, u)]
                assert got == pytest.approx(val, abs=TOL)

    def test_bridge_between_triangles(self):
        g = Graph().add_nodes(list("ABCDEF"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
        g.add_edges([("D", "E"), ("E", "F"), ("F", "D")])
        raw = g.edge_betweenness_centrality(normalized=False, return_dict=True)
        bridge = g.find_edge_ids("C", "D")[0]
        assert raw[bridge] == pytest.approx(9.0)  # all 3 x 3 cross pairs
        assert raw[bridge] == max(raw.values())

    def test_strip_values_used_in_figure(self):
        g = _strip()
        ours = {
            "".join(g.get_label_from_id(n) for n in g.get_node_ids_by_edge_id(eid)): round(v, 2)
            for eid, v in g.edge_betweenness_centrality(return_dict=True).items()
        }
        assert ours == {
            "AB": 0.07,
            "AC": 0.21,
            "BC": 0.12,
            "BD": 0.14,
            "CD": 0.07,
            "CE": 0.36,
            "DE": 0.21,
            "EF": 0.48,
            "FG": 0.29,
        }

    def test_parallel_edges_split_paths(self):
        g = Graph().add_nodes(["A", "B"])
        g.add_edges([("A", "B"), ("A", "B")])
        raw = g.edge_betweenness_centrality(normalized=False, return_dict=True)
        assert sorted(raw.values()) == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_self_loop_scores_zero(self):
        g = Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "A")])
        raw = g.edge_betweenness_centrality(return_dict=True)
        assert raw[g.find_edge_ids("A", "A")[0]] == 0.0

    def test_empty(self):
        assert Graph().edge_betweenness_centrality(return_dict=True) == {}
        assert Graph().add_nodes(["A", "B"]).edge_betweenness_centrality(return_dict=True) == {}

    def test_stores_edge_attribute(self):
        g = _strip().edge_betweenness_centrality(attr_name="ebc")
        df = g.get_edge_df()
        assert "ebc" in df.columns
        assert df["ebc"].max() == pytest.approx(10 / 21)
        assert "ebc" not in g.get_node_df().columns

    def test_describe_reports_edge_coverage(self, capsys):
        _strip().edge_betweenness_centrality().describe()
        out = capsys.readouterr().out
        assert "edge_betweenness_centrality() -> attr: 'edge_betweenness_centrality'" in out
        assert "9 of 9 edges" in out


# --- step log and player ----------------------------------------------------------


class TestStepRecording:
    def test_node_metrics_record_metric_steps(self):
        g = _strip().closeness_centrality().degree_centrality(mode="all")
        closeness, degree = g.steps[-2], g.steps[-1]
        assert closeness.kind == KIND_METRIC and closeness.method == "closeness_centrality"
        assert closeness.code == "G.closeness_centrality(attr_name='closeness_centrality')"
        assert round(closeness.delta.metric_values["E"], 2) == 0.67
        assert set(closeness.delta.metric_values) == set("ABCDEFG")
        assert degree.method == "degree_centrality"
        assert degree.delta.metric_values["C"] == pytest.approx(4 / 6)
        assert "highest: C" in degree.narration

    def test_non_default_params_appear_in_code(self):
        g = Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        g.closeness_centrality(mode="in", wf_improved=False)
        assert g.steps[-1].code == (
            "G.closeness_centrality(mode='in', wf_improved=False, attr_name='closeness_centrality')"
        )

    def test_edge_betweenness_records_edge_metric_values(self):
        g = _strip().edge_betweenness_centrality()
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and step.method == "edge_betweenness_centrality"
        assert step.delta.metric_values == {}
        assert len(step.delta.edge_metric_values) == 9
        assert step.delta.edge_metric_values[("E", "F")] == pytest.approx(10 / 21)
        assert "highest: E–F" in step.narration

    def test_parallel_edges_are_summed_for_the_player(self):
        g = Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "B")])
        g.edge_betweenness_centrality(normalized=False)
        assert g.steps[-1].delta.edge_metric_values == {("A", "B"): pytest.approx(1.0)}

    def test_return_dict_records_no_step(self):
        g = _strip()
        n = len(g.steps)
        g.closeness_centrality(return_dict=True)
        g.degree_centrality(return_dict=True)
        g.edge_betweenness_centrality(return_dict=True)
        assert len(g.steps) == n

    def test_player_serializes_metric_values(self):
        g = _strip().closeness_centrality().edge_betweenness_centrality()
        html = g.play()._html
        assert '"mv"' in html  # node heat for closeness
        assert '"emv"' in html  # edge heat for edge betweenness
        assert '["E", "F", 0.4762]' in html
