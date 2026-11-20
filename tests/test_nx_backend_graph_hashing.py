"""Weisfeiler-Lehman hashes of the NetworkX backend are NetworkX's."""

import random
import warnings

import pytest

nx = pytest.importorskip("networkx")

EC = {"backend": "edgecase"}


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_graph(cls, seed):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(100), 12)
    for v in nodes:
        G.add_node(v, color=rng.choice(["red", "blue", 3, 1.5]))
    for _ in range(25):
        G.add_edge(rng.choice(nodes), rng.choice(nodes), kind=rng.choice(["a", "b", 2]))
    return G


def both(func, G, **kw):
    """Results and warnings of NetworkX and of edgecase."""
    out = []
    for extra in [{}, EC]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(G, **kw, **extra)
        out.append((result, [str(x.message) for x in w]))
    return out


@pytest.mark.parametrize("cls", [nx.Graph, nx.DiGraph])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"iterations": 1},
        {"node_attr": "color"},
        {"edge_attr": "kind", "digest_size": 8},
        {"edge_attr": "kind", "node_attr": "color", "iterations": 5},
    ],
)
def test_hashes(cls, seed, kw):
    G = random_graph(cls, seed)
    expected, actual = both(nx.weisfeiler_lehman_graph_hash, G, **kw)
    assert actual == expected
    for include in [False, True]:
        expected, actual = both(
            nx.weisfeiler_lehman_subgraph_hashes, G, include_initial_labels=include, **kw
        )
        assert actual == expected
        assert list(actual[0]) == list(expected[0])


def test_errors():
    G = nx.path_graph(3)
    G.add_edge(0, 1, kind="a")
    G.nodes[0]["color"] = "red"
    for func in [nx.weisfeiler_lehman_graph_hash, nx.weisfeiler_lehman_subgraph_hashes]:
        with pytest.raises(ValueError, match="positive"):
            func(G, iterations=0, edge_attr="kind", **EC)
        with pytest.raises(KeyError):
            func(G, edge_attr="kind", **EC)
        with pytest.raises(KeyError):
            func(G, node_attr="color", **EC)
        assert func(nx.Graph(), **EC) == func(nx.Graph())
