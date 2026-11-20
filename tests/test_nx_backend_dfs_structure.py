"""The NetworkX backend's bridges, biconnected components, chains, cycle bases and girth
give NetworkX's results in NetworkX's order.
"""

import math
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
            if not isinstance(result, dict | list | tuple | int | float | bool):
                result = list(result)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert actual == expected
    assert type(actual) is type(expected)
    if isinstance(expected, list):
        assert [type(x) for x in actual] == [type(x) for x in expected]
    return expected


def random_graph(cls, n=30, p=0.08, seed=0, selfloops=True):
    """Sparse, with cut vertices and bridges, nodes and edges in a random order."""
    rng = random.Random(seed)
    nodes = list(range(n))
    rng.shuffle(nodes)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    edges += [(i, i + 1) for i in range(0, n - 1, 3)]  # (a sprinkle of bridges)
    if selfloops:
        edges += [(u, u) for u in range(n) if rng.random() < 0.05]
    if cls().is_multigraph():
        edges += rng.sample(edges, len(edges) // 5)
    rng.shuffle(edges)
    G = cls()
    G.add_nodes_from(nodes)
    for u, v in edges:
        if rng.random() < 0.5:
            u, v = v, u
        G.add_edge(u, v, weight=rng.choice([1, 2, 3]), cost=rng.random())
    return G


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize("seed", range(5))
def test_biconnected_and_bridges(cls, seed):
    G = random_graph(cls, seed=seed)
    assert_same(nx.biconnected_component_edges, G)
    assert_same(nx.biconnected_components, G)
    assert_same(nx.articulation_points, G)
    assert_same(nx.is_biconnected, G)
    assert_same(nx.is_biconnected, nx.cycle_graph(5, create_using=cls))
    assert_same(nx.bridges, G)
    assert_same(nx.has_bridges, G)
    for root in list(G)[:3] + ["missing"]:
        assert_same(nx.bridges, G, root=root)
        assert_same(nx.has_bridges, G, root=root)
    if cls is nx.Graph:
        assert_same(nx.chain_decomposition, G)
        for root in list(G)[:3] + ["missing"]:
            assert_same(nx.chain_decomposition, G, root=root)
        for weight in [None, "weight", "cost"]:
            assert_same(nx.local_bridges, G, weight=weight)
        assert_same(nx.local_bridges, G, with_span=False)
        assert_same(nx.cycle_basis, G)
        for root in list(G)[:3]:
            assert_same(nx.cycle_basis, G, root=root)
        assert_same(nx.cycle_basis, G, root="missing")
        assert_same(nx.girth, G)


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_small_graphs(n):
    for G in [nx.path_graph(n), nx.cycle_graph(n), nx.complete_graph(n), nx.star_graph(n)]:
        for func in [
            nx.biconnected_components,
            nx.articulation_points,
            nx.is_biconnected,
            nx.bridges,
            nx.chain_decomposition,
            nx.cycle_basis,
            nx.girth,
            nx.local_bridges,
        ]:
            assert_same(func, G)


def test_girth():
    for G in [nx.petersen_graph(), nx.heawood_graph(), nx.grid_2d_graph(4, 5), nx.empty_graph(3)]:
        assert_same(nx.girth, G)
    G = nx.path_graph(4)
    G.add_edge(2, 2)
    assert_same(nx.girth, G)
    assert nx.girth(nx.path_graph(4), backend="edgecase") == math.inf


GRAPH_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("seed", range(4))
def test_edge_dfs_and_find_cycle(cls, seed):
    G = random_graph(cls, n=20, p=0.1, seed=seed)
    nodes = list(G)
    for orientation in [None, "original", "reverse", "ignore", "nope"]:
        assert_same(nx.edge_dfs, G, orientation=orientation)
        assert_same(nx.edge_dfs, G, nodes[3], orientation=orientation)
        assert_same(nx.edge_dfs, G, nodes[2:6], orientation=orientation)
        assert_same(nx.find_cycle, G, orientation=orientation)
        assert_same(nx.find_cycle, G, nodes[5], orientation=orientation)
    # Acyclic: no cycle to find
    T = nx.bfs_tree(nx.Graph(G), nodes[0])
    T = cls(T) if cls().is_directed() else cls(T.to_undirected())
    for orientation in [None, "original", "reverse", "ignore"]:
        assert_same(nx.find_cycle, T, orientation=orientation)
    assert_same(nx.edge_dfs, G, [])


def test_edge_dfs_follows_predecessor_order():
    # 3's in-edges were added from 2, 0, 1; its arcs come from 0, 1, 2 (node order)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(2, 3), (0, 3), (1, 3), (1, 3), (4, 0), (3, 4)])
    for orientation in ["reverse", "ignore"]:
        assert_same(nx.edge_dfs, G, 3, orientation=orientation)
        assert_same(nx.find_cycle, G, 3, orientation=orientation)
