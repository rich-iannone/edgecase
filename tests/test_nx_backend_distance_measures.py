"""The NetworkX backend's distance measures and closeness/harmonic centrality.

Each check calls a NetworkX function with and without `backend="edgecase"` and expects the
same result: the same dict order and floats equal to the last bit.
"""

import math
import random

import pytest

nx = pytest.importorskip("networkx")

from edgecase.nx_backend.interface import backend_interface  # noqa: E402

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


def connected_graph(cls, n=16, p=0.2, seed=0, weights=(1, 2, 3)):
    """A strongly connected graph (a cycle plus random edges, in a random order)."""
    rng = random.Random(seed)
    nodes = list(range(n))
    rng.shuffle(nodes)
    edges = list(zip(nodes, nodes[1:] + nodes[:1]))
    edges += [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
    if cls().is_multigraph():
        edges += rng.sample(edges, len(edges) // 4)
    rng.shuffle(edges)
    G = cls()
    G.add_nodes_from(nodes)
    for u, v in edges:
        G.add_edge(u, v, weight=rng.choice(weights), cost=rng.random())
    return G


def graphs():
    for cls in GRAPH_CLASSES:
        for seed in range(3):
            yield connected_graph(cls, seed=seed)
    yield connected_graph(nx.Graph, n=40, p=0.05, seed=9)


GRAPHS = list(graphs())


def graph_id(G):
    return f"{type(G).__name__}-{G.number_of_nodes()}-{G.number_of_edges()}"


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
@pytest.mark.parametrize("weight", [None, "weight", "cost"])
def test_distance_measures(G, weight):
    assert_same(nx.eccentricity, G, weight=weight)
    assert_same(nx.eccentricity, G, v=list(G)[3], weight=weight)
    assert_same(nx.eccentricity, G, v=list(G)[:5], weight=weight)
    for func in [nx.diameter, nx.radius, nx.center, nx.periphery]:
        expected = assert_same(func, G, weight=weight)
        if weight == "cost":
            # With float weights NetworkX's bounds can loop forever (rounding); edgecase
            # then finishes with exact eccentricities
            actual = func(G, weight=weight, usebounds=True, backend="edgecase")
            assert same(expected, actual)
        else:
            assert_same(func, G, weight=weight, usebounds=True)
    assert_same(nx.harmonic_diameter, G, weight=weight)
    assert_same(nx.barycenter, G, weight=weight)


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
@pytest.mark.parametrize("distance", [None, "weight", "cost"])
def test_closeness_and_harmonic(G, distance):
    nodes = list(G)
    for wf_improved in (True, False):
        assert_same(nx.closeness_centrality, G, distance=distance, wf_improved=wf_improved)
    assert_same(nx.closeness_centrality, G, nodes[2], distance=distance)
    assert_same(nx.harmonic_centrality, G, distance=distance)
    assert_same(nx.harmonic_centrality, G, nbunch=nodes[:3], distance=distance)
    assert_same(nx.harmonic_centrality, G, sources=nodes[4:6], distance=distance)
    assert_same(nx.harmonic_centrality, G, nodes[1:7], distance, nodes[5:9])


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
def test_disconnected_and_trivial_graphs(cls):
    G = nx.empty_graph(0, create_using=cls)
    for func in [nx.eccentricity, nx.diameter, nx.radius, nx.center, nx.periphery]:
        assert_same(func, G)
        assert_same(func, G, usebounds=True)
    for func in [nx.harmonic_diameter, nx.barycenter, nx.closeness_centrality]:
        assert_same(func, G)
    assert_same(nx.harmonic_centrality, G)
    G = nx.empty_graph(1, create_using=cls)
    for func in [nx.eccentricity, nx.diameter, nx.harmonic_diameter, nx.closeness_centrality]:
        assert_same(func, G)
    G = nx.path_graph(5, create_using=cls)
    G.add_edge(7, 8)
    for func in [nx.eccentricity, nx.diameter, nx.radius, nx.center, nx.barycenter]:
        assert_same(func, G)
        assert_same(func, G, weight="weight")
    assert_same(nx.diameter, G, usebounds=True)
    for func in [nx.closeness_centrality, nx.harmonic_centrality, nx.harmonic_diameter]:
        assert_same(func, G)
    assert_same(nx.eccentricity, G, v=99)
    assert_same(nx.eccentricity, G, v=[[1]])
    assert_same(nx.closeness_centrality, G, u=99)
    assert_same(nx.closeness_centrality, G, u=99, distance="weight")


def test_negative_weight_contradiction():
    G = nx.DiGraph()
    nx.add_cycle(G, range(4), weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(2, 1, weight=-5)
    for func in [nx.eccentricity, nx.closeness_centrality, nx.harmonic_diameter]:
        kwargs = {"distance": "weight"} if func is nx.closeness_centrality else {}
        kwargs = kwargs or {"weight": "weight"}
        nx_result, ec_result = both(func, G, **kwargs)
        assert ec_result == nx_result
        assert ec_result[0] is ValueError


def test_zero_distances_and_infinite_weights():
    G = nx.Graph()
    G.add_edge(0, 1, weight=0)
    G.add_edge(1, 2, weight=0.0)
    G.add_edge(2, 3, weight=math.inf)
    for func in [nx.eccentricity, nx.harmonic_diameter, nx.barycenter]:
        assert_same(func, G, weight="weight")
    assert_same(nx.closeness_centrality, G, distance="weight")
    assert_same(nx.harmonic_centrality, G, distance="weight")


def test_trees_and_refused_arguments():
    T = nx.balanced_tree(2, 3)
    assert_same(nx.center, T)
    # A tree's centroid comes from NetworkX's tree algorithm (which orders two centroids)
    assert_same(nx.barycenter, T)
    rng = random.Random(1)
    for n in [1, 2, 7, 30]:
        for seed in range(5):
            T = nx.random_labeled_tree(n, seed=rng)
            assert_same(nx.barycenter, nx.relabel_nodes(T, {7: "x"}))
    # Writing node attributes is NetworkX's job
    assert isinstance(backend_interface.can_run("centroid", (T,), {"attr": "b"}), str)
    assert isinstance(backend_interface.can_run("eccentricity", (T,), {"sp": {}}), str)

    def length(u, v, d):
        return 1

    assert isinstance(backend_interface.can_run("diameter", (T, None, False, length), {}), str)
    assert backend_interface.can_run("diameter", (T,), {"weight": "w"}) is True


def test_should_run_declines_single_searches_on_unconverted_graphs():
    from edgecase.nx_backend import from_networkx

    G = nx.path_graph(5)
    declines = [
        ("eccentricity", (G, 0), {}),
        ("eccentricity", (G,), {"v": 3}),
        ("closeness_centrality", (G, 1), {}),
        ("harmonic_centrality", (G,), {"nbunch": 2}),
        ("harmonic_centrality", (G,), {"sources": 2}),
        ("diameter", (G,), {"e": {0: 1}}),
        ("center", (G, {0: 1}), {}),
    ]
    for name, args, kwargs in declines:
        assert isinstance(backend_interface.should_run(name, args, kwargs), str), name
        args = (from_networkx(G), *args[1:])
        assert backend_interface.should_run(name, args, kwargs) is True, name
    runs = [
        ("eccentricity", (G,), {}),
        ("eccentricity", (G, [0, 1]), {}),
        ("closeness_centrality", (G,), {}),
        ("harmonic_centrality", (G,), {"nbunch": [2]}),
        ("diameter", (G,), {"usebounds": True}),
        ("barycenter", (G,), {}),
    ]
    for name, args, kwargs in runs:
        name = "centroid" if name == "barycenter" else name
        assert backend_interface.should_run(name, args, kwargs) is True, name
