"""Degree correlations, assortativity, reciprocity, efficiency, Wiener index and rich
clubs of the NetworkX backend equal NetworkX's (exactly, floats included)."""

import math
import random

import pytest

nx = pytest.importorskip("networkx")
pytest.importorskip("scipy")

ALL_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
EC = {"backend": "edgecase"}


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_graph(cls, n, m, seed, selfloops=True):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    for v in nodes:
        G.add_node(v, color=rng.choice("abc"), size=rng.choice([1, 2, 5]))
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u == v and not selfloops:
            continue
        if rng.random() < 0.2:
            G.add_edge(u, v)
        else:
            G.add_edge(u, v, weight=rng.random())
    return G


def same(a, b):
    if isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and math.isnan(a)
    if isinstance(b, dict):
        return list(a.items()) == list(b.items())
    return a == b and type(a) is type(b)


GRAPHS = [
    pytest.param(cls, seed, id=f"{cls.__name__}-{seed}") for cls in ALL_CLASSES for seed in range(4)
]


@pytest.mark.parametrize(("cls", "seed"), GRAPHS)
def test_neighbor_degrees(cls, seed):
    G = random_graph(cls, 30, 60, seed)
    some = list(G)[:7]
    directions = ["in", "out", "in+out"] if G.is_directed() else [None]
    for weight in [None, "weight"]:
        for nodes in [None, some]:
            for s in directions:
                for t in directions:
                    kw = {"weight": weight, "nodes": nodes}
                    if s is not None:
                        kw.update(source=s, target=t)
                    if not G.is_multigraph() or weight is None:
                        assert same(
                            nx.average_neighbor_degree(G, **kw, **EC),
                            nx.average_neighbor_degree(G, **kw),
                        )
                    assert same(
                        nx.average_degree_connectivity(G, **kw, **EC),
                        nx.average_degree_connectivity(G, **kw),
                    )


@pytest.mark.parametrize(("cls", "seed"), GRAPHS)
def test_degree_assortativity(cls, seed):
    np = pytest.importorskip("numpy")
    G = random_graph(cls, 30, 60, seed)
    pairs = [("out", "in"), ("in", "out"), ("out", "out"), ("in", "in")]
    for x, y in pairs if G.is_directed() else [("out", "in")]:
        for weight in [None, "weight"]:
            for nodes in [None, list(G)[:12]]:
                kw = {"x": x, "y": y, "weight": weight, "nodes": nodes}
                assert same(
                    nx.degree_assortativity_coefficient(G, **kw, **EC),
                    nx.degree_assortativity_coefficient(G, **kw),
                )
                assert same(
                    nx.degree_pearson_correlation_coefficient(G, **kw, **EC),
                    nx.degree_pearson_correlation_coefficient(G, **kw),
                )
                assert nx.degree_mixing_dict(G, **kw, **EC) == nx.degree_mixing_dict(G, **kw)
                assert list(nx.node_degree_xy(G, **kw, **EC)) == list(nx.node_degree_xy(G, **kw))
                np.testing.assert_array_equal(
                    nx.degree_mixing_matrix(G, **kw, **EC), nx.degree_mixing_matrix(G, **kw)
                )


@pytest.mark.parametrize(("cls", "seed"), GRAPHS)
def test_attribute_assortativity(cls, seed):
    np = pytest.importorskip("numpy")
    G = random_graph(cls, 30, 60, seed)
    some = list(G)[:15]
    for nodes in [None, some]:
        assert same(
            nx.attribute_assortativity_coefficient(G, "color", nodes, **EC),
            nx.attribute_assortativity_coefficient(G, "color", nodes),
        )
        assert same(
            nx.numeric_assortativity_coefficient(G, "size", nodes, **EC),
            nx.numeric_assortativity_coefficient(G, "size", nodes),
        )
        assert list(nx.node_attribute_xy(G, "color", nodes, **EC)) == list(
            nx.node_attribute_xy(G, "color", nodes)
        )
        assert nx.attribute_mixing_dict(G, "color", nodes, **EC) == (
            nx.attribute_mixing_dict(G, "color", nodes)
        )
        np.testing.assert_array_equal(
            nx.attribute_mixing_matrix(G, "color", nodes, **EC),
            nx.attribute_mixing_matrix(G, "color", nodes),
        )


@pytest.mark.parametrize("seed", range(5))
def test_reciprocity(seed):
    G = random_graph(nx.DiGraph, 30, 70, seed)
    G.add_node("isolated")
    assert same(nx.overall_reciprocity(G, **EC), nx.overall_reciprocity(G))
    assert same(nx.reciprocity(G, **EC), nx.reciprocity(G))
    some = [n for n in list(G)[:10] if n != "isolated"]
    assert same(nx.reciprocity(G, some, **EC), nx.reciprocity(G, some))
    assert same(nx.reciprocity(G, some[0], **EC), nx.reciprocity(G, some[0]))
    with pytest.raises(nx.NetworkXError):
        nx.reciprocity(G, "isolated", **EC)


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize("seed", range(5))
def test_efficiency_and_wiener(cls, seed):
    G = random_graph(cls, 25, 45, seed)
    assert same(nx.global_efficiency(G, **EC), nx.global_efficiency(G))
    assert same(nx.local_efficiency(G, **EC), nx.local_efficiency(G))
    nodes = list(G)
    for u, v in zip(nodes[:5], nodes[5:10]):
        assert same(nx.efficiency(G, u, v, **EC), nx.efficiency(G, u, v))
    for weight in [None, "weight"]:
        assert same(nx.wiener_index(G, weight, **EC), nx.wiener_index(G, weight))
    H = nx.connected_watts_strogatz_graph(30, 4, 0.3, seed=seed)
    assert same(nx.wiener_index(H, **EC), nx.wiener_index(H))
    assert same(nx.global_efficiency(H, **EC), nx.global_efficiency(H))
    assert same(nx.local_efficiency(H, **EC), nx.local_efficiency(H))


def test_efficiency_small_cases():
    for n in range(4):
        G = nx.complete_graph(n)
        assert same(nx.global_efficiency(G, **EC), nx.global_efficiency(G))
        if n:
            assert same(nx.local_efficiency(G, **EC), nx.local_efficiency(G))
    D = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    assert nx.wiener_index(D, **EC) == nx.wiener_index(D)


@pytest.mark.parametrize("seed", range(5))
def test_rich_club(seed):
    G = random_graph(nx.Graph, 40, 100, seed, selfloops=False)
    assert same(
        nx.rich_club_coefficient(G, normalized=False, **EC),
        nx.rich_club_coefficient(G, normalized=False),
    )
    from edgecase.nx_backend.interface import backend_interface as bi

    assert isinstance(bi.can_run("rich_club_coefficient", (G,), {}), str)
