"""The NetworkX backend's greedy coloring, maximal independent sets and dominating sets
give NetworkX's results (whose order comes from Python sets and the random generator).
"""

import random

import pytest

nx = pytest.importorskip("networkx")


@pytest.fixture(autouse=True)
def no_conversion_cache():
    cache = nx.config.cache_converted_graphs
    nx.config.cache_converted_graphs = False
    yield
    nx.config.cache_converted_graphs = cache


def both(func, *args, **kwargs):
    """`(networkx result, edgecase result)`, or the exceptions they raise."""
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
    assert type(actual) is type(expected)
    if isinstance(expected, dict | set):
        assert list(actual) == list(expected)  # (the same order too)
    return expected


LABELS = [None, str, lambda i: (i % 4, f"v{i}"), lambda i: i * 64 - 300]


def random_graph(seed, n=40, p=0.15, labels=None):
    rng = random.Random(seed)
    nodes = list(range(n)) if labels is None else [labels(i) for i in range(n)]
    G = nx.Graph()
    G.add_nodes_from(rng.sample(nodes, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                u, v = (nodes[i], nodes[j]) if rng.random() < 0.5 else (nodes[j], nodes[i])
                G.add_edge(u, v)
    G.add_nodes_from(nodes[-3:] if labels is None else [labels(n + i) for i in range(3)])
    return G


STRATEGIES = [
    "largest_first",
    "smallest_last",
    "independent_set",
    "connected_sequential_bfs",
    "connected_sequential_dfs",
    "connected_sequential",
    "saturation_largest_first",
    "DSATUR",
    "nope",
]


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("seed", range(4))
def test_greedy_color(seed, labels):
    G = random_graph(seed, labels=labels)
    for strategy in STRATEGIES:
        assert_same(nx.greedy_color, G, strategy)
    for strategy in ["independent_set", "DSATUR"]:
        assert_same(nx.greedy_color, G, strategy, interchange=True)
    # random_sequential shuffles with the global random generator: give both the same state
    state = random.getstate()
    expected = nx.greedy_color(G, "random_sequential", backend="networkx")
    after = random.getstate()
    random.setstate(state)
    actual = nx.greedy_color(G, "random_sequential", backend="edgecase")
    assert list(actual.items()) == list(expected.items())
    assert random.getstate() == after


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("seed", range(4))
def test_maximal_independent_set_and_dominating_set(seed, labels):
    G = random_graph(seed, labels=labels)
    nodes = list(G)
    for rng_seed in range(3):
        rngs = [random.Random(rng_seed), random.Random(rng_seed)]
        results = [
            nx.maximal_independent_set(G, seed=rng, backend=backend)
            for rng, backend in zip(rngs, ["networkx", "edgecase"])
        ]
        assert list(results[1]) == list(results[0])
        assert rngs[1].getstate() == rngs[0].getstate()  # (the same numbers drawn)
        independent = nx.maximal_independent_set(G, seed=rng_seed)
        start = [next(iter(independent))]
        assert_same(nx.maximal_independent_set, G, start, rng_seed)
    adjacent = [nodes[0], next(iter(G[nodes[0]]), nodes[0])]
    assert_same(nx.maximal_independent_set, G, adjacent, 1)
    assert_same(nx.maximal_independent_set, G, [nodes[0], "missing"], 1)
    assert_same(nx.dominating_set, G)
    for start in nodes[:5] + ["missing"]:
        assert_same(nx.dominating_set, G, start)
    for nbunch in [nodes, nodes[:5], list(nx.dominating_set(G))]:
        assert_same(nx.is_dominating_set, G, nbunch)
