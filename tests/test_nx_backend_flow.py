"""Maximum flows and minimum cuts of the NetworkX backend.

Flow values and minimum cuts equal NetworkX's (the cut is the side that can still reach the
sink, the same for every maximum flow); flow assignments are valid maximum flows.
"""

import random

import pytest

nx = pytest.importorskip("networkx")

EC = {"backend": "edgecase"}


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_network(cls, seed, kind):
    rng = random.Random(seed)
    G = cls()
    nodes = list(range(25))
    rng.shuffle(nodes)
    G.add_nodes_from(nodes)
    for _ in range(70):
        u, v = rng.choice(nodes), rng.choice(nodes)
        r = rng.random()
        if kind == "int":
            cap = rng.choice([0, 1, 2, 3, 5, 8])
        else:
            cap = rng.choice([0.5, 1.25, 2.0, 3.5])
        if r < 0.1:
            G.add_edge(u, v)  # infinite capacity
        elif r < 0.15:
            G.add_edge(u, v, capacity=-1)
        else:
            G.add_edge(u, v, capacity=cap)
    return G, nodes


def check_flow(G, s, t, value, flow_dict):
    assert set(flow_dict) == set(G)
    excess = dict.fromkeys(G, 0)
    for u in G:
        assert set(flow_dict[u]) == set(G[u])
        for v, f in flow_dict[u].items():
            cap = G[u][v].get("capacity", float("inf"))
            assert f >= 0
            assert f == 0 or (u != v and cap > 0 and f <= cap)
            excess[u] -= f
            excess[v] += f
    for u, e in excess.items():
        expected = -value if u == s else value if u == t else 0
        assert e == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("cls", [nx.Graph, nx.DiGraph])
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("kind", ["int", "float"])
def test_flows_and_cuts(cls, seed, kind):
    G, nodes = random_network(cls, seed, kind)
    s, t = nodes[0], nodes[1]
    try:
        expected = nx.maximum_flow_value(G, s, t)
    except nx.NetworkXUnbounded:
        for func in [nx.maximum_flow_value, nx.maximum_flow, nx.minimum_cut]:
            with pytest.raises(nx.NetworkXUnbounded):
                func(G, s, t, **EC)
        return
    value = nx.maximum_flow_value(G, s, t, **EC)
    assert value == pytest.approx(expected, rel=1e-12)
    if kind == "int":
        assert value == expected and type(value) is type(expected)
    assert nx.minimum_cut_value(G, s, t, **EC) == value
    cut_value, partition = nx.minimum_cut(G, s, t, **EC)
    expected_cut = nx.minimum_cut(G, s, t)
    assert cut_value == value
    assert partition == expected_cut[1]
    flow_value, flow_dict = nx.maximum_flow(G, s, t, **EC)
    assert flow_value == value
    check_flow(G, s, t, flow_value, flow_dict)
    for func in [nx.algorithms.flow.edmonds_karp, nx.algorithms.flow.dinitz]:
        assert nx.maximum_flow_value(G, s, t, flow_func=func, **EC) == value


def test_errors_and_fallbacks():
    from edgecase.nx_backend.interface import backend_interface as bi

    G = nx.DiGraph([(0, 1, {"capacity": 2})])
    with pytest.raises(nx.NetworkXError, match="not in graph"):
        nx.maximum_flow_value(G, 0, 9, **EC)
    with pytest.raises(nx.NetworkXError, match="same node"):
        nx.maximum_flow_value(G, 0, 0, **EC)
    assert isinstance(bi.can_run("maximum_flow", (nx.MultiDiGraph(G), 0, 1), {}), str)
    assert isinstance(bi.can_run("maximum_flow_value", (G, 0, 1), {"cutoff": 1}), str)
    assert isinstance(bi.can_run("maximum_flow_value", (G, 0, 1), {"flow_func": max}), str)
    assert isinstance(
        bi.can_run("maximum_flow_value", (G, 0, 1), {"capacity": lambda u, v, d: 1}), str
    )
