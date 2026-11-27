"""Boundaries, cuts and community edge counts of the NetworkX backend match NetworkX."""

import random

import pytest

nx = pytest.importorskip("networkx")

ALL_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
EC = {"backend": "edgecase"}


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_graph(cls, seed):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(300), 30)
    G.add_nodes_from(nodes)
    for _ in range(70):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if rng.random() < 0.3:
            G.add_edge(u, v, color="red")
        else:
            G.add_edge(u, v, weight=rng.random(), color="blue")
    return G, rng


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(5))
def test_boundaries_and_cuts(cls, seed):
    G, rng = random_graph(cls, seed)
    nodes = list(G)
    S = set(rng.sample(nodes, 10))
    T = set(rng.sample([n for n in nodes if n not in S], 8))
    for data in [False, True, "weight", "color"]:
        for keys in [False, True]:
            for default in [None, 7]:
                kw = {"data": data, "keys": keys, "default": default}
                for nb2 in [None, T]:
                    assert list(nx.edge_boundary(G, S, nb2, **kw, **EC)) == list(
                        nx.edge_boundary(G, S, nb2, **kw)
                    )
    assert nx.node_boundary(G, S, **EC) == nx.node_boundary(G, S)
    assert nx.node_boundary(G, S, T, **EC) == nx.node_boundary(G, S, T)
    for weight in [None, "weight"]:
        for func in [nx.cut_size, nx.normalized_cut_size, nx.conductance, nx.edge_expansion]:
            for t in [None, T] if func is not nx.cut_size or not G.is_directed() else [T]:
                try:
                    expected = func(G, S, t, weight=weight)
                except ZeroDivisionError:
                    with pytest.raises(ZeroDivisionError):
                        func(G, S, t, weight=weight, **EC)
                    continue
                assert func(G, S, t, weight=weight, **EC) == expected
        assert nx.volume(G, S, weight=weight, **EC) == nx.volume(G, S, weight=weight)
        assert nx.mixing_expansion(G, S, T, weight=weight, **EC) == (
            nx.mixing_expansion(G, S, T, weight=weight)
        )
    assert nx.node_expansion(G, S, **EC) == nx.node_expansion(G, S)
    assert nx.boundary_expansion(G, S, **EC) == nx.boundary_expansion(G, S)


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(5))
def test_community_edges(cls, seed):
    G, rng = random_graph(cls, seed)
    partition = nx.community.louvain_communities(G, seed=seed)
    assert nx.algorithms.community.quality.intra_community_edges(G, partition, **EC) == (
        nx.algorithms.community.quality.intra_community_edges(G, partition)
    )
    if not G.is_multigraph():
        assert nx.algorithms.community.quality.inter_community_edges(G, partition, **EC) == (
            nx.algorithms.community.quality.inter_community_edges(G, partition)
        )
