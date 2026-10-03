"""describe() lists every logged computation, including graph-level results.

`network_robustness`, `influence_maximization`, `network_smooth`, and `soft_threshold_power`
used to log 2-tuples, so a later describe() (and therefore repr()) raised
`ValueError: not enough values to unpack`.
"""

import contextlib
import io

import pytest

import edgecase as ec


@pytest.fixture
def graph():
    g = ec.Graph(weighted=True)
    for label in "ABCDEFG":
        g.add_node(label=label)
    for source, target in ["AB", "AC", "BD", "CD", "DE", "DF", "EG", "FG"]:
        g.add_edge(source, target, weight=0.5)
    return g


def _describe(g):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g.describe()
    return out.getvalue()


GRAPH_LEVEL = {
    "network_robustness": lambda g: g.network_robustness(strategy="degree"),
    "influence_maximization": lambda g: g.influence_maximization(k=2, seed=1),
    "soft_threshold_power": lambda g: g.soft_threshold_power(),
}


@pytest.mark.parametrize("run", GRAPH_LEVEL.values(), ids=GRAPH_LEVEL.keys())
def test_graph_level_computation_is_described(graph, run):
    name = next(k for k, v in GRAPH_LEVEL.items() if v is run)

    assert run(graph) is graph
    method_name, params, attr_name = graph._computations[-1]
    assert method_name == name
    assert isinstance(params, dict)
    assert attr_name is None

    text = _describe(graph)
    assert f"{name}(" in text
    assert "graph-level result (not stored)" in text
    repr(graph)


def test_network_smooth_stores_default_column(graph):
    a_id = graph.get_id_from_label("A")

    assert graph.network_smooth({a_id: 1.0}) is graph

    assert graph._computations[-1][2] == "network_smooth"
    scores = graph.get_node_df()["network_smooth"].to_list()
    assert len(scores) == 7 and all(s is not None for s in scores)
    assert "attr: 'network_smooth' (7 of 7 nodes)" in _describe(graph)


def test_network_smooth_custom_attr_name(graph):
    a_id = graph.get_id_from_label("A")

    graph.network_smooth({a_id: 1.0}, attr_name="smoothed")

    assert "smoothed" in graph.get_node_df().columns
    assert "network_smooth" not in graph.get_node_df().columns
    assert "attr: 'smoothed' (7 of 7 nodes)" in _describe(graph)


def test_all_four_chained_then_described(graph):
    a_id = graph.get_id_from_label("A")
    (
        graph.network_robustness()
        .influence_maximization(k=1, seed=1)
        .network_smooth({a_id: 1.0})
        .soft_threshold_power()
    )

    text = _describe(graph)
    assert text.count("graph-level result (not stored)") == 3
    assert "network_smooth(" in text
