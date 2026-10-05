import warnings

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_METRIC

from .correctness_utils import (
    IGRAPH_AVAILABLE,
    align_results,
    check_validation_passes,
    compute_comparison_metrics,
    format_validation_error,
    nx_to_edgecase,
    nx_to_igraph,
)

warnings.filterwarnings("ignore")

# Tolerance thresholds (relaxed since betweenness can have larger variance)
MAX_ABS_DIFF = 1e-3
MAX_REL_DIFF = 0.05  # 5%
MIN_CORRELATION = 0.995


def create_test_graphs():
    """Create diverse graph types for validation."""
    graphs = []

    # 1. Complete graph
    nx_g = nx.complete_graph(10)
    graphs.append(("complete_10", nx_g, False))

    # 2. Star graph
    nx_g = nx.star_graph(15)
    graphs.append(("star_16", nx_g, False))

    # 3. Path graph
    nx_g = nx.path_graph(15)
    graphs.append(("path_15", nx_g, False))

    # 4. Cycle graph
    nx_g = nx.cycle_graph(12)
    graphs.append(("cycle_12", nx_g, False))

    # 5. Random sparse graph
    nx_g = nx.erdos_renyi_graph(30, 0.1, seed=23, directed=False)
    graphs.append(("random_sparse_30", nx_g, False))

    # 6. Scale-free (Barabási-Albert)
    nx_g = nx.barabasi_albert_graph(40, 2, seed=23)
    graphs.append(("scale_free_40", nx_g, False))

    # 7. Karate club (real-world network)
    nx_g = nx.karate_club_graph()
    graphs.append(("karate_club", nx_g, False))

    # 8. Grid graph (small for speed)
    nx_g = nx.grid_2d_graph(4, 4)
    nx_g = nx.relabel_nodes(nx_g, {node: str(node) for node in nx_g.nodes()})
    graphs.append(("grid_4x4", nx_g, False))

    # 9. Directed acyclic graph
    nx_g = nx.DiGraph()
    nx_g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4)])
    graphs.append(("dag_5", nx_g, True))

    return graphs


def compare_betweenness_networkx(ec_graph, nx_graph, directed=False):
    """Compare Betweenness Centrality between EdgeCase and NetworkX."""
    # Run Betweenness Centrality
    ec_btw = ec_graph.betweenness_centrality(return_dict=True)
    nx_btw = nx.betweenness_centrality(nx_graph, normalized=True)

    # Align results
    node_list = list(nx_graph.nodes())
    ec_array, nx_array = align_results(ec_btw, nx_btw, node_list, ec_graph, is_dict=True)

    # Compute metrics
    return compute_comparison_metrics(ec_array, nx_array)


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
def compare_betweenness_igraph(ec_graph, nx_graph, directed=False):
    """Compare Betweenness Centrality between EdgeCase and igraph."""
    ig_graph, node_list = nx_to_igraph(nx_graph, directed)

    # Run Betweenness Centrality
    ec_btw = ec_graph.betweenness_centrality(return_dict=True)
    ig_btw_raw = ig_graph.betweenness(directed=directed)

    # igraph returns unnormalized betweenness, need to normalize
    n = len(node_list)
    if n <= 2:
        norm_factor = 1.0
    else:
        norm_factor = (n - 1) * (n - 2) / (2.0 if not directed else 1.0)

    ig_btw = [score / norm_factor for score in ig_btw_raw]

    # Align results
    ec_array, ig_array = align_results(ec_btw, ig_btw, node_list, ec_graph, is_dict=False)

    # Compute metrics
    return compute_comparison_metrics(ec_array, ig_array)


@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_betweenness_vs_networkx(name, nx_graph, directed):
    """Test Betweenness Centrality correctness against NetworkX."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_betweenness_networkx(ec_graph, nx_graph, directed)

    passes = check_validation_passes(metrics, MAX_ABS_DIFF, MAX_REL_DIFF, MIN_CORRELATION)

    assert passes, format_validation_error("Betweenness Centrality", name, metrics)


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_betweenness_vs_igraph(name, nx_graph, directed):
    """Test Betweenness Centrality correctness against igraph."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_betweenness_igraph(ec_graph, nx_graph, directed)

    passes = check_validation_passes(metrics, MAX_ABS_DIFF, MAX_REL_DIFF, MIN_CORRELATION)

    assert passes, format_validation_error(
        "Betweenness Centrality", name, metrics, vs_library="igraph"
    )


# Edge betweenness centrality, validated against NetworkX and on hand-computed cases.


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


def _from_nx(G: nx.Graph) -> Graph:
    g = Graph(directed=G.is_directed())
    g.add_nodes([str(n) for n in G.nodes])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
    return g


def _edges_by_label(g: Graph, scores: dict[int, float]) -> dict[tuple[int, int], float]:
    out = {}
    for eid, value in scores.items():
        s, t = g.get_node_ids_by_edge_id(eid)
        out[(int(g.get_label_from_id(s)), int(g.get_label_from_id(t)))] = value
    return out


TOL = 1e-9


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
        g.edge_betweenness_centrality(return_dict=True)
        assert len(g.steps) == n

    def test_player_serializes_edge_metric_values(self):
        html = _strip().edge_betweenness_centrality().play()._html
        assert '"emv"' in html
        assert '["E", "F", 0.4762]' in html
