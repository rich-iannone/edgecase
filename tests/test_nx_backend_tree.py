"""Spanning trees and condensation of the NetworkX backend match NetworkX."""

import math
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


def random_graph(cls, n, m, seed, ties=True):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    G.add_nodes_from(nodes, color="c")
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if rng.random() < 0.2:
            G.add_edge(u, v, other=1)  # no weight: counts as 1
        else:
            G.add_edge(u, v, weight=rng.choice([1, 2, 3]) if ties else rng.random())
    G.graph["name"] = "g"
    return G


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ties", [True, False])
def test_spanning_trees(cls, seed, ties):
    G = random_graph(cls, 40, 70, seed, ties)  # usually a forest of several trees
    for weight in ["weight", None, "other"]:
        for func in [nx.minimum_spanning_tree, nx.maximum_spanning_tree]:
            expected = func(G, weight=weight)
            result = func(G, weight=weight, **EC)
            assert type(result) is type(expected)
            assert result.graph == expected.graph
            assert list(result.nodes(data=True)) == list(expected.nodes(data=True))
            assert list(result.edges(data=True)) == list(expected.edges(data=True))
        for func in [nx.minimum_spanning_edges, nx.maximum_spanning_edges]:
            for keys in [True, False]:
                for data in [True, False]:
                    kw = {"weight": weight, "keys": keys, "data": data}
                    assert list(func(G, **kw, **EC)) == list(func(G, **kw))


def test_spanning_tree_errors_and_fallbacks():
    from edgecase.nx_backend.interface import backend_interface as bi

    D = nx.DiGraph([(0, 1)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.minimum_spanning_tree(D, **EC)
    G = nx.path_graph(3)
    assert isinstance(bi.can_run("minimum_spanning_tree", (G,), {"algorithm": "prim"}), str)
    assert isinstance(bi.can_run("minimum_spanning_edges", (G, "boruvka"), {}), str)
    assert bi.can_run("minimum_spanning_tree", (G, "weight", "kruskal"), {}) is True
    # NaN weights are NetworkX's business: refused when called, not when iterated
    G[0][1]["weight"] = math.nan
    with pytest.raises(NotImplementedError):
        nx.minimum_spanning_edges(G, **EC)


@pytest.mark.parametrize("cls", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("seed", range(6))
def test_condensation(cls, seed):
    G = random_graph(cls, 30, 50, seed)
    expected = nx.condensation(G)
    result = nx.condensation(G, **EC)
    assert result.graph == expected.graph
    assert list(result.nodes(data=True)) == list(expected.nodes(data=True))
    assert list(result.edges) == list(expected.edges)
    scc = list(nx.strongly_connected_components(G))
    assert nx.utils.graphs_equal(nx.condensation(G, scc, **EC), nx.condensation(G, scc))
    assert nx.condensation(nx.DiGraph(), **EC).graph == {"mapping": {}}


def test_generators_refuse_unusable_weights_when_called():
    # NotImplementedError must come from the call, so the dispatcher can fall back
    G = nx.path_graph(4)
    G[0][1]["weight"] = "heavy"
    with pytest.raises(NotImplementedError):
        nx.community.louvain_partitions(G, **EC)
    with pytest.raises(NotImplementedError):
        nx.community.asyn_lpa_communities(G, weight="weight", **EC)
    with pytest.raises(NotImplementedError):
        nx.community.leiden_partitions(G, **EC)
