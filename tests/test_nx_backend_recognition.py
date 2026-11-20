"""The NetworkX backend's bipartite coloring, tree recognition and Eulerian circuits and
paths give NetworkX's results in NetworkX's order.
"""

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
    """`(networkx result, edgecase result)`, or the exceptions they raise."""
    results = []
    for backend in ("networkx", "edgecase"):
        try:
            result = func(*args, backend=backend, **kwargs)
            if not isinstance(result, dict | list | tuple | int | float | bool):
                result = list(result)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert actual == expected
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
    return expected


def eulerian_graph(cls, seed, n=12, cycles=5):
    """A union of random cycles (so every node is balanced), edges added in random order."""
    rng = random.Random(seed)
    edges = []
    for _ in range(cycles):
        cycle = rng.sample(range(n), rng.randint(2 if cls().is_multigraph() else 3, n))
        edges += list(zip(cycle, cycle[1:] + cycle[:1]))
    G = cls()
    G.add_nodes_from(rng.sample(range(n), n))
    for u, v in edges:
        if not G.is_multigraph() and G.has_edge(u, v):
            continue
        G.add_edge(u, v)
    return G


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("seed", range(6))
def test_eulerian(cls, seed):
    G = eulerian_graph(cls, seed)
    for func in [nx.is_eulerian, nx.has_eulerian_path, nx.is_semieulerian]:
        assert_same(func, G)
    for keys in [False, True]:
        assert_same(nx.eulerian_circuit, G, keys=keys)
        assert_same(nx.eulerian_circuit, G, source=list(G)[3], keys=keys)
        assert_same(nx.eulerian_path, G, keys=keys)
        assert_same(nx.eulerian_path, G, source=list(G)[2], keys=keys)
    # One edge fewer: an Eulerian path but no circuit (if still connected)
    H = G.copy()
    H.remove_edge(*next(iter(H.edges())))
    for func in [nx.is_eulerian, nx.has_eulerian_path, nx.is_semieulerian]:
        assert_same(func, H)
    for keys in [False, True]:
        assert_same(nx.eulerian_circuit, H, keys=keys)
        assert_same(nx.eulerian_path, H, keys=keys)
        for source in list(H)[:4]:
            assert_same(nx.eulerian_path, H, source=source, keys=keys)
            assert_same(nx.has_eulerian_path, H, source=source)


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("seed", range(4))
def test_bipartite_and_trees(cls, seed):
    rng = random.Random(seed)
    B = nx.bipartite.random_graph(8, 9, 0.25, seed=seed, directed=cls().is_directed())
    B = cls(B)
    B.add_nodes_from([100, 101])
    for G in [B, nx.cycle_graph(5, create_using=cls), nx.path_graph(6, create_using=cls)]:
        assert_same(nx.bipartite.color, G)
        assert_same(nx.is_bipartite, G)
        assert_same(nx.bipartite.sets, G)
        assert_same(nx.bipartite.sets, G, top_nodes=list(G)[:3])
    T = nx.random_labeled_tree(15, seed=rng)
    T = cls(nx.bfs_tree(T, 0)) if cls().is_directed() else cls(T)
    for G in [T, B, nx.empty_graph(0, create_using=cls), nx.empty_graph(3, create_using=cls)]:
        for func in [nx.is_tree, nx.is_forest]:
            assert_same(func, G)
        if G.is_directed():
            for func in [nx.is_arborescence, nx.is_branching]:
                assert_same(func, G)
