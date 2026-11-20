"""The NetworkX backend's algorithms give NetworkX's answers (`backend="edgecase"`)."""

import random

import pytest

nx = pytest.importorskip("networkx")


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_graph(cls, n, m, seed, weighted=True, selfloops=True):
    rng = random.Random(seed)
    G = cls()
    G.add_nodes_from(rng.sample(range(10 * n), n))  # nodes not in sorted order
    nodes = list(G)
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u == v and not selfloops:
            continue
        if weighted and rng.random() < 0.9:  # some edges lack a weight
            G.add_edge(u, v, weight=rng.choice([0.5, 1, 2, 3.25]))
        else:
            G.add_edge(u, v)
    return G


ALL_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
GRAPHS = [
    pytest.param(cls, n, m, seed, id=f"{cls.__name__}-{n}-{m}-{seed}")
    for cls in ALL_CLASSES
    for n, m, seed in [(1, 0, 0), (2, 1, 1), (12, 10, 2), (30, 60, 3), (60, 200, 4)]
]


def assert_close(a, b, rel=1e-9, abs=1e-12):
    assert list(a) == list(b)  # same keys in the same order
    assert a == pytest.approx(b, rel=rel, abs=abs)


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
def test_components(cls, n, m, seed):
    G = random_graph(cls, n, m, seed)
    kw = {"backend": "edgecase"}
    if G.is_directed():
        expected = list(nx.weakly_connected_components(G))
        assert list(nx.weakly_connected_components(G, **kw)) == expected
        assert nx.number_weakly_connected_components(G, **kw) == len(expected)
        assert nx.is_weakly_connected(G, **kw) == nx.is_weakly_connected(G)
        # Strongly connected components come in NetworkX's order
        expected = list(nx.strongly_connected_components(G))
        assert list(nx.strongly_connected_components(G, **kw)) == expected
        assert nx.number_strongly_connected_components(G, **kw) == len(expected)
        assert nx.is_strongly_connected(G, **kw) == nx.is_strongly_connected(G)
    else:
        expected = list(nx.connected_components(G))
        assert list(nx.connected_components(G, **kw)) == expected
        assert nx.number_connected_components(G, **kw) == len(expected)
        assert nx.is_connected(G, **kw) == nx.is_connected(G)
        for node in list(G)[:5]:
            assert nx.node_connected_component(G, node, **kw) == (
                nx.node_connected_component(G, node)
            )


def test_component_errors():
    kw = {"backend": "edgecase"}
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_connected(nx.Graph(), **kw)
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_weakly_connected(nx.DiGraph(), **kw)
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_strongly_connected(nx.DiGraph(), **kw)
    with pytest.raises(KeyError):
        nx.node_connected_component(nx.path_graph(3), 7, **kw)
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.connected_components(nx.DiGraph(), **kw))
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.strongly_connected_components(nx.Graph(), **kw))


def test_strongly_connected_components_long_path():
    # A long cycle used to be a deep recursion; the search is iterative
    G = nx.DiGraph()
    nx.add_cycle(G, range(200_000))
    G.add_edge(0, "tail")
    result = list(nx.strongly_connected_components(G, backend="edgecase"))
    assert result == [{"tail"}, set(range(200_000))]


def test_weakly_connected_components_scale():
    # Used to scan every edge for each node of a directed graph (O(V * E))
    G = nx.gnm_random_graph(20_000, 40_000, seed=1, directed=True)
    expected = list(nx.weakly_connected_components(G))
    assert list(nx.weakly_connected_components(G, backend="edgecase")) == expected


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
@pytest.mark.parametrize("weight", ["weight", None])
def test_pagerank(cls, n, m, seed, weight):
    G = random_graph(cls, n, m, seed)
    expected = nx.pagerank(G, weight=weight, tol=1e-10, max_iter=1000)
    result = nx.pagerank(G, weight=weight, tol=1e-10, max_iter=1000, backend="edgecase")
    assert_close(result, expected, rel=1e-8, abs=1e-10)
    assert all(type(v) is float for v in result.values())


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_pagerank_options(cls):
    G = random_graph(cls, 40, 90, 7)
    rng = random.Random(3)
    nodes = list(G)
    personalization = {v: rng.random() for v in rng.sample(nodes, 10)}
    nstart = {v: rng.random() for v in nodes}
    dangling = {v: rng.random() for v in rng.sample(nodes, 5)}
    kwargs = {
        "alpha": 0.7,
        "personalization": personalization,
        "nstart": nstart,
        "dangling": dangling,
        "tol": 1e-10,
        "max_iter": 500,
    }
    expected = nx.pagerank(G, **kwargs)
    assert_close(nx.pagerank(G, **kwargs, backend="edgecase"), expected, rel=1e-8)


def test_pagerank_errors():
    G = nx.DiGraph(nx.path_graph(4))
    with pytest.raises(nx.PowerIterationFailedConvergence):
        nx.pagerank(G, max_iter=1, backend="edgecase")
    with pytest.raises(ZeroDivisionError):
        nx.pagerank(G, personalization={"missing": 1}, backend="edgecase")
    assert nx.pagerank(nx.Graph(), backend="edgecase") == {}


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
@pytest.mark.parametrize("weight", ["weight", None])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness_centrality(cls, n, m, seed, weight, normalized, endpoints):
    G = random_graph(cls, n, m, seed)
    kwargs = {"weight": weight, "normalized": normalized, "endpoints": endpoints}
    expected = nx.betweenness_centrality(G, **kwargs)
    result = nx.betweenness_centrality(G, **kwargs, backend="edgecase")
    assert_close(result, expected)
    assert all(type(v) is float for v in result.values())


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness_centrality_sampled(cls, k, endpoints):
    import math

    G = random_graph(cls, 20, 50, 11)
    expected = nx.betweenness_centrality(G, k=k, endpoints=endpoints, seed=42)
    result = nx.betweenness_centrality(G, k=k, endpoints=endpoints, seed=42, backend="edgecase")
    assert list(result) == list(expected)
    for v in G:
        if math.isnan(expected[v]):
            assert math.isnan(result[v])
        else:
            assert result[v] == pytest.approx(expected[v], rel=1e-9, abs=1e-12)


def test_betweenness_zero_weight_ties():
    # With zero-weight edges NetworkX's Dijkstra also counts paths through finished nodes, so
    # its answer depends on neighbor order. A directed graph keeps NetworkX's neighbor order
    # in edgecase, so the answers agree; an undirected one may list neighbors differently.
    G = nx.DiGraph()
    edges = [(0, 1, 0), (1, 2, 0), (0, 2, 0), (2, 3, 1), (3, 3, 0), (2, 0, 0), (1, 0, 0)]
    G.add_weighted_edges_from(edges)
    expected = nx.betweenness_centrality(G, weight="weight")
    assert_close(nx.betweenness_centrality(G, weight="weight", backend="edgecase"), expected)
