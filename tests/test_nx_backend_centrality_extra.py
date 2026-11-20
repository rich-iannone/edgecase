"""The NetworkX backend's degree/eigenvector/Katz centrality, edge and subset betweenness,
HITS and link prediction give NetworkX's results: the same order and floats equal to the
last bit (except where noted).
"""

import math
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
            if not isinstance(result, dict | list | tuple | int | float):
                result = list(result)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def same(x, y):
    """Equal, with the same dict order and types, and NaN equal to NaN."""
    if isinstance(x, dict):
        return (
            isinstance(y, dict) and list(x) == list(y) and all(same(v, y[k]) for k, v in x.items())
        )
    if isinstance(x, list | tuple):
        return type(x) is type(y) and len(x) == len(y) and all(map(same, x, y))
    if isinstance(x, float) and math.isnan(x):
        return isinstance(y, float) and math.isnan(y)
    return type(x) is type(y) and x == y


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert same(expected, actual), (expected, actual)
    return expected


def random_graph(cls, n=18, p=0.2, seed=0, selfloops=True):
    rng = random.Random(seed)
    nodes = list(range(n))
    rng.shuffle(nodes)
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
    if selfloops:
        edges += [(u, u) for u in range(n) if rng.random() < 0.1]
    if cls().is_multigraph():
        edges += rng.sample(edges, len(edges) // 4)
    rng.shuffle(edges)
    G = cls()
    G.add_nodes_from(nodes)
    for u, v in edges:
        G.add_edge(u, v, weight=rng.choice([1, 2, 3]), cost=rng.random() + 0.1)
    return G


GRAPHS = [random_graph(cls, seed=seed) for cls in GRAPH_CLASSES for seed in range(3)]


def graph_id(G):
    return f"{type(G).__name__}-{G.number_of_nodes()}-{G.number_of_edges()}"


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
def test_degree_centrality(G):
    assert_same(nx.degree_centrality, G)
    if G.is_directed():
        assert_same(nx.in_degree_centrality, G)
        assert_same(nx.out_degree_centrality, G)
    for n in [0, 1]:
        assert_same(nx.degree_centrality, nx.empty_graph(n, create_using=type(G)))


@pytest.mark.parametrize("G", [G for G in GRAPHS if not G.is_multigraph()], ids=graph_id)
@pytest.mark.parametrize("weight", [None, "weight", "cost"])
def test_eigenvector_and_katz(G, weight):
    assert_same(nx.eigenvector_centrality, G, weight=weight, max_iter=1000)
    assert_same(nx.eigenvector_centrality, G, weight=weight, max_iter=3)  # (no convergence)
    nstart = {n: i + 1 for i, n in enumerate(reversed(list(G)))}
    assert_same(nx.eigenvector_centrality, G, weight=weight, max_iter=1000, nstart=nstart)
    for alpha in [0.01, 0.05]:
        assert_same(nx.katz_centrality, G, alpha=alpha, weight=weight)
        assert_same(nx.katz_centrality, G, alpha=alpha, weight=weight, normalized=False)
    beta = {n: (i % 3) + 0.5 for i, n in enumerate(G)}
    assert_same(nx.katz_centrality, G, alpha=0.02, beta=beta, weight=weight, nstart=nstart)
    assert_same(nx.katz_centrality, G, alpha=0.9, weight=weight, max_iter=50)
    assert_same(nx.katz_centrality, G, beta={0: 1})  # (beta for some nodes only)


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
@pytest.mark.parametrize("weight", [None, "weight", "cost"])
def test_edge_and_subset_betweenness(G, weight):
    for normalized in [True, False]:
        assert_same(nx.edge_betweenness_centrality, G, normalized=normalized, weight=weight)
    assert_same(nx.edge_betweenness_centrality, G, k=5, seed=3, weight=weight)
    nodes = list(G)
    sources, targets = nodes[:6], nodes[4:12]
    for normalized in [True, False]:
        assert_same(
            nx.betweenness_centrality_subset, G, sources, targets, normalized, weight=weight
        )
        assert_same(
            nx.edge_betweenness_centrality_subset, G, sources, targets, normalized, weight=weight
        )


def test_zero_weight_edges_in_betweenness():
    # Equal-length paths through zero-weight edges (and a zero-weight self-loop)
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 0), (1, 2, 1), (0, 2, 1), (2, 3, 0), (3, 3, 0), (1, 3, 1)])
    assert_same(nx.edge_betweenness_centrality, G, weight="weight")
    assert_same(nx.edge_betweenness_centrality_subset, G, [0, 1], [3, 2], weight="weight")
    assert_same(nx.betweenness_centrality_subset, G, [0, 1], [3, 2], weight="weight")


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
def test_hits(G):
    for normalized in [True, False]:
        assert_same(nx.hits, G, normalized=normalized, max_iter=1000)
    nstart = {n: i + 1 for i, n in enumerate(G)}
    assert_same(nx.hits, G, nstart=nstart, max_iter=1000)
    assert_same(nx.hits, G, max_iter=2)
    assert_same(nx.hits, G, nstart={0: 1})
    assert_same(nx.hits, G, method="nope")


@pytest.mark.parametrize("seed", range(3))
def test_link_prediction(seed):
    G = random_graph(nx.Graph, n=25, p=0.2, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))  # (common_neighbor_centrality refuses them)
    G.add_edge(3, 3)
    pairs = [(0, 1), (2, 5), (7, 7), (3, 4), (3, 3)]
    for func in [nx.jaccard_coefficient, nx.preferential_attachment]:
        assert_same(func, G)
        assert_same(func, G, pairs)
    for alpha in [0.8, 1]:
        G2 = G.copy()
        G2.remove_edge(3, 3)
        assert_same(nx.common_neighbor_centrality, G2, alpha=alpha)
        assert_same(nx.common_neighbor_centrality, G2, [(0, 1), (2, 2)], alpha=alpha)
        assert_same(nx.common_neighbor_centrality, G2, [(0, 1), (2, 99)], alpha=alpha)
    # Adamic-Adar and resource allocation: the same pairs, and sums equal up to the order
    # of their terms
    for func in [nx.adamic_adar_index, nx.resource_allocation_index]:
        expected, actual = both(func, G)
        assert [p[:2] for p in actual] == [p[:2] for p in expected]
        for (_, _, x), (_, _, y) in zip(expected, actual):
            assert type(x) is type(y)
            assert math.isclose(x, y, rel_tol=1e-15)
        assert_same(func, G, [(0, 99)])


def test_link_prediction_iterator_ebunch_is_consumed_by_the_check():
    # (NetworkX checks every pair of an iterator first, which leaves nothing to score)
    G = nx.path_graph(4)
    for backend in ["networkx", "edgecase"]:
        result = nx.jaccard_coefficient(G, iter([(0, 2), (1, 3)]), backend=backend)
        assert list(result) == []


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
@pytest.mark.parametrize("weight", [None, "weight", "cost"])
def test_load_centrality(G, weight):
    for normalized in [True, False]:
        assert_same(nx.load_centrality, G, normalized=normalized, weight=weight)
    assert_same(nx.load_centrality, G, cutoff=2, weight=weight)
    assert_same(nx.load_centrality, G, v=list(G)[3], weight=weight)
    assert_same(nx.load_centrality, G, v="not a node", weight=weight)
    H = nx.relabel_nodes(G, {n: f"n{n:02d}" for n in G})  # (ties ordered by string keys)
    assert_same(nx.load_centrality, H, weight=weight)


def test_load_centrality_with_unsortable_keys_is_left_to_networkx():
    G = nx.path_graph(4)
    G = nx.relabel_nodes(G, {0: "a", 1: 1, 2: (2,), 3: 3.5})
    with pytest.raises(NotImplementedError):
        nx.load_centrality(G, backend="edgecase")
    # Dispatched automatically, NetworkX then runs it (and fails the same way)
    with nx.config.backend_priority(algos=["edgecase"]):
        with pytest.raises(TypeError):
            nx.load_centrality(G)
    G = nx.relabel_nodes(G, {"a": 0, (2,): 2})
    assert_same(nx.load_centrality, G)  # (ints and floats compare)


@pytest.mark.parametrize("seed", range(3))
def test_community_link_prediction(seed):
    rng = random.Random(seed)
    G = random_graph(nx.Graph, n=25, p=0.25, seed=seed)
    for n in G:
        G.nodes[n]["community"] = rng.choice(["a", "b", 3])
    G.nodes[5]["group"] = 1
    pairs = [(0, 1), (2, 5), (3, 4), (7, 7)]
    for func in [nx.cn_soundarajan_hopcroft, nx.ra_index_soundarajan_hopcroft]:
        expected, actual = both(func, G)
        assert [p[:2] for p in actual] == [p[:2] for p in expected]
        for (_, _, x), (_, _, y) in zip(expected, actual):
            assert type(x) is type(y)
            assert math.isclose(x, y, rel_tol=1e-15)
        assert_same(func, G, pairs)
    assert_same(nx.within_inter_cluster, G)
    assert_same(nx.within_inter_cluster, G, pairs, delta=0.5)
    assert_same(nx.within_inter_cluster, G, delta=0)
    # Missing communities raise as NetworkX does (and name the same node)
    del G.nodes[2]["community"]
    for func in [nx.cn_soundarajan_hopcroft, nx.within_inter_cluster]:
        assert_same(func, G, [(0, 1), (2, 3)])
        assert_same(func, G, [(0, 1), (3, 2)])
        assert_same(func, G)
    assert_same(nx.cn_soundarajan_hopcroft, G, [(0, 1)], community="group")
