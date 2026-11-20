"""Matchings of the NetworkX backend: NetworkX's greedy maximal matching exactly, and
maximum/minimum-weight matchings with NetworkX's weight and cardinality."""

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


def random_graph(seed, n=18, m=35, weights="int"):
    rng = random.Random(seed)
    G = nx.Graph()
    nodes = rng.sample(range(100), n)
    G.add_nodes_from(nodes)
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if weights == "int":
            G.add_edge(u, v, weight=rng.choice([-1, 1, 2, 3, 7]))
        elif weights == "float":
            G.add_edge(u, v, weight=rng.random())
        else:
            G.add_edge(u, v)
    return G


def weight_of(G, matching, weight="weight"):
    return sum(G[u][v].get(weight, 1) for u, v in matching)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("weights", ["int", "float", "none"])
def test_weighted_matchings(seed, weights):
    G = random_graph(seed, weights=weights)
    for maxcardinality in [False, True]:
        expected = nx.max_weight_matching(G, maxcardinality=maxcardinality)
        result = nx.max_weight_matching(G, maxcardinality=maxcardinality, **EC)
        assert nx.is_matching(G, result)
        assert (
            len(result) == len(expected)
            or not maxcardinality
            and (weight_of(G, result) == pytest.approx(weight_of(G, expected)))
        )
        assert weight_of(G, result) == pytest.approx(weight_of(G, expected))
    expected = nx.min_weight_matching(G)
    result = nx.min_weight_matching(G, **EC)
    assert nx.is_matching(G, result) and len(result) == len(expected)
    assert weight_of(G, result) == pytest.approx(weight_of(G, expected))
    assert nx.max_weight_matching(G, weight=None, **EC) and len(
        nx.max_weight_matching(G, weight=None, **EC)
    ) == len(nx.max_weight_matching(G, weight=None))


@pytest.mark.parametrize("seed", range(10))
def test_maximal_matching_and_checks(seed):
    G = random_graph(seed)
    G.add_edge(*list(G)[:1] * 2)
    M = nx.maximal_matching(G, **EC)
    assert M == nx.maximal_matching(G)
    D = nx.DiGraph(G)
    rng = random.Random(seed)
    candidates = [
        M,
        set(list(M)[1:]),
        {(v, u) for u, v in M},
        {u: v for u, v in M} | {v: u for u, v in M},
        set(rng.sample(list(G.edges), 4)),
        nx.max_weight_matching(G, maxcardinality=True),
    ]
    for matching in candidates:
        for H in [G, D]:
            for func in [nx.is_matching, nx.is_maximal_matching, nx.is_perfect_matching]:
                assert func(H, matching, **EC) == func(H, matching)
    with pytest.raises(nx.NetworkXError):
        nx.is_matching(G, {(1234567, list(G)[0])}, **EC)
    with pytest.raises(nx.NetworkXError):
        nx.is_matching(G, {(1, 2, 3)}, **EC)
