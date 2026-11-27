"""Clustering, triangle and k-core functions of the NetworkX backend match NetworkX."""

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


def random_graph(cls, n, m, seed, selfloops=False):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    G.add_nodes_from(nodes)
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v or selfloops:
            G.add_edge(u, v, weight=rng.choice([1, 2, 0.5]))
    return G


SIZES = [(1, 0), (6, 8), (20, 50), (40, 160), (60, 120)]


def assert_same(a, b):
    assert type(a) is type(b) or {type(a), type(b)} <= {int, float}
    if isinstance(b, dict):
        assert list(a) == list(b)
        for k in b:
            assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-15)
    else:
        assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize(("n", "m"), SIZES)
@pytest.mark.parametrize("seed", range(3))
def test_undirected_counts(cls, n, m, seed):
    G = random_graph(cls, n, m, seed, selfloops=True)
    assert_same(nx.triangles(G, **EC), nx.triangles(G))
    assert_same(nx.square_clustering(G, **EC), nx.square_clustering(G))
    if not G.is_multigraph():
        some = list(G)[: max(1, n // 3)]
        assert_same(nx.triangles(G, some, **EC), nx.triangles(G, some))
        assert_same(nx.triangles(G, some[0], **EC), nx.triangles(G, some[0]))
        assert_same(nx.clustering(G, **EC), nx.clustering(G))
        assert_same(nx.clustering(G, some, **EC), nx.clustering(G, some))
        assert_same(nx.clustering(G, some[0], **EC), nx.clustering(G, some[0]))
        assert_same(nx.transitivity(G, **EC), nx.transitivity(G))
        assert nx.generalized_degree(G, **EC) == nx.generalized_degree(G)
        assert nx.generalized_degree(G, some[0], **EC) == nx.generalized_degree(G, some[0])
        if n > 1:
            for count_zeros in [True, False]:
                kw = {"count_zeros": count_zeros}
                if not count_zeros and not any(nx.clustering(G).values()):
                    continue
                assert_same(nx.average_clustering(G, **kw, **EC), nx.average_clustering(G, **kw))


@pytest.mark.parametrize(("n", "m"), SIZES)
@pytest.mark.parametrize("seed", range(3))
def test_directed_clustering(n, m, seed):
    G = random_graph(nx.DiGraph, n, m, seed, selfloops=True)
    assert_same(nx.clustering(G, **EC), nx.clustering(G))
    if n > 1:
        assert_same(nx.average_clustering(G, **EC), nx.average_clustering(G))


def test_left_to_networkx():
    from edgecase.nx_backend.interface import backend_interface

    G = nx.path_graph(3)
    assert isinstance(backend_interface.can_run("clustering", (G,), {"weight": "w"}), str)
    M = nx.MultiGraph(G)
    assert isinstance(backend_interface.can_run("transitivity", (M,), {}), str)
    assert isinstance(backend_interface.can_run("triangles", (M, [0]), {}), str)
    assert backend_interface.can_run("triangles", (M,), {}) is True
    assert isinstance(backend_interface.can_run("transitivity", (nx.DiGraph(G),), {}), str)


@pytest.mark.parametrize("cls", [nx.Graph, nx.DiGraph])
@pytest.mark.parametrize(("n", "m"), SIZES)
@pytest.mark.parametrize("seed", range(4))
def test_cores(cls, n, m, seed):
    G = random_graph(cls, n, m, seed)
    G.graph["name"] = "g"
    core = nx.core_number(G)
    assert nx.core_number(G, **EC) == core
    assert list(nx.core_number(G, **EC)) == list(core)
    top = max(core.values())
    for func in [nx.k_core, nx.k_shell, nx.k_crust]:
        for k in [None, 0, 1, 2, top]:
            expected = func(G, k)
            result = func(G, k, **EC)
            assert nx.utils.graphs_equal(result, expected)
            assert list(result) == list(expected)
            assert list(result.edges(data=True)) == list(expected.edges(data=True))
        assert nx.utils.graphs_equal(func(G, None, core, **EC), func(G, None, core))
    for k in range(top + 1):
        assert nx.utils.graphs_equal(nx.k_corona(G, k, **EC), nx.k_corona(G, k))
    if not G.is_directed():
        for k in [1, 2, 3, 4, 5]:
            expected = nx.k_truss(G, k)
            result = nx.k_truss(G, k, **EC)
            assert nx.utils.graphs_equal(result, expected)
            assert list(result) == list(expected)


def test_core_number_scales_linearly():
    # The old peeling was quadratic on long paths
    G = nx.path_graph(200_000)
    core = nx.core_number(G, **EC)
    assert core[0] == 1 and core[100_000] == 1


def test_core_errors():
    G = nx.path_graph(3)
    G.add_edge(1, 1)
    for func in [nx.core_number, nx.k_core, nx.k_truss]:
        with pytest.raises(nx.NetworkXNotImplemented, match="self loops"):
            func(G, **EC) if func is not nx.k_truss else func(G, 3, **EC)
    with pytest.raises(ValueError):
        nx.k_core(nx.Graph(), **EC)
