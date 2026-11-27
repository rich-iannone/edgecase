"""The NetworkX backend's isomorphism tests agree with NetworkX."""

import random

import pytest

nx = pytest.importorskip("networkx")

GRAPH_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


@pytest.fixture(autouse=True)
def no_conversion_cache():
    cache = nx.config.cache_converted_graphs
    nx.config.cache_converted_graphs = False
    yield
    nx.config.cache_converted_graphs = cache


def both(func, *args, **kwargs):
    results = []
    for backend in ("networkx", "edgecase"):
        try:
            result = func(*args, backend=backend, **kwargs)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert actual == expected
    return expected


def random_graph(cls, seed, n=12, p=0.3):
    rng = random.Random(seed)
    G = cls()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if (u < v or G.is_directed()) and u != v and rng.random() < p:
                G.add_edge(u, v)
    if G.is_multigraph():
        for u, v in rng.sample(list(G.edges()), 3):
            G.add_edge(u, v)
    G.add_edge(1, 1)
    for v in G:
        G.nodes[v]["color"] = rng.choice("ab")
    return G


def relabeled(G, seed):
    nodes = list(G)
    perm = nodes[:]
    random.Random(seed).shuffle(perm)
    return nx.relabel_nodes(G, dict(zip(nodes, perm)))


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("seed", range(5))
def test_isomorphism(cls, seed):
    G = random_graph(cls, seed)
    H = relabeled(G, seed)
    K = H.copy()
    K.remove_edge(*next(iter(K.edges())))
    K.add_edge(*random.Random(seed).sample(list(K), 2))  # (as many edges, maybe not iso)
    for other in [H, K, random_graph(cls, seed + 1)]:
        expected = assert_same(nx.is_isomorphic, G, other)
        assert_same(nx.vf2pp_is_isomorphic, G, other)
        assert_same(nx.vf2pp_is_isomorphic, G, other, node_label="color")
        assert_same(nx.vf2pp_is_isomorphic, G, other, node_label="shade", default_label=0)
        assert_same(nx.faster_could_be_isomorphic, G, other)
        if not G.is_directed():
            assert_same(nx.could_be_isomorphic, G, other)
            assert_same(nx.fast_could_be_isomorphic, G, other)
        if expected:
            assert nx.could_be_isomorphic(G, other, properties="d", backend="edgecase")
    assert_same(nx.is_isomorphic, G, nx.DiGraph() if not G.is_directed() else nx.Graph())
    assert_same(nx.vf2pp_is_isomorphic, cls(), cls())
    assert_same(nx.is_isomorphic, cls(), cls())
