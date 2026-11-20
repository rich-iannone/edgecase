"""Community detection and partition quality of the NetworkX backend match NetworkX.

The randomized algorithms must give NetworkX's exact partitions for the same seed and
leave the seed generator in the same state.
"""

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


def random_graph(cls, n, m, seed, float_weights=False):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    G.add_nodes_from(nodes)
    for _ in range(m):
        w = rng.random() * 3 if float_weights else rng.choice([1, 2, 3])
        G.add_edge(rng.choice(nodes), rng.choice(nodes), weight=w)
    return G


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("float_weights", [False, True])
def test_louvain(cls, seed, float_weights):
    G = random_graph(cls, 60, 150, seed, float_weights)
    for kw in [{}, {"weight": None}, {"resolution": 0.6}, {"threshold": 1e-3}]:
        r1, r2 = random.Random(seed), random.Random(seed)
        expected = list(nx.community.louvain_partitions(G, seed=r1, **kw))
        assert list(nx.community.louvain_partitions(G, seed=r2, **kw, **EC)) == expected
        assert r1.getstate() == r2.getstate()
        assert nx.community.louvain_communities(G, seed=seed, **kw, **EC) == (
            nx.community.louvain_communities(G, seed=seed, **kw)
        )
    for max_level in [1, 2]:
        kw = {"seed": seed, "max_level": max_level}
        assert nx.community.louvain_communities(G, **kw, **EC) == (
            nx.community.louvain_communities(G, **kw)
        )


def test_louvain_edge_cases():
    G = nx.empty_graph(4)
    assert nx.community.louvain_communities(G, seed=1, **EC) == [{0}, {1}, {2}, {3}]
    with pytest.raises(ValueError, match="max_level"):
        nx.community.louvain_communities(nx.path_graph(3), max_level=0, **EC)


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("weight", [None, "weight"])
def test_asyn_lpa(cls, seed, weight):
    G = random_graph(cls, 50, 90, seed, float_weights=seed % 2 == 1)
    G.add_node("isolated")
    r1, r2 = random.Random(seed), random.Random(seed)
    expected = list(nx.community.asyn_lpa_communities(G, weight=weight, seed=r1))
    result = list(nx.community.asyn_lpa_communities(G, weight=weight, seed=r2, **EC))
    assert result == expected
    assert r1.getstate() == r2.getstate()


@pytest.mark.parametrize("cls", [nx.Graph])
@pytest.mark.parametrize("seed", range(10))
def test_label_propagation(cls, seed):
    G = random_graph(cls, 40, 70, seed)
    G.add_edge(*list(G)[:1] * 2)
    assert list(nx.community.label_propagation_communities(G, **EC)) == list(
        nx.community.label_propagation_communities(G)
    )


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(5))
def test_quality(cls, seed):
    G = random_graph(cls, 30, 60, seed, float_weights=True)
    partition = nx.community.louvain_communities(G, seed=seed)
    for kw in [{}, {"weight": None}, {"resolution": 0.5}]:
        assert nx.community.modularity(G, partition, **kw, **EC) == pytest.approx(
            nx.community.modularity(G, partition, **kw), rel=1e-12, abs=1e-15
        )
    assert nx.community.partition_quality(G, partition, **EC) == (
        nx.community.partition_quality(G, partition)
    )
    assert nx.community.is_partition(G, partition, **EC)
    assert not nx.community.is_partition(G, partition[1:], **EC)
    assert not nx.community.is_partition(G, [*partition, {list(G)[0]}], **EC)
    with pytest.raises(nx.NetworkXError):
        nx.community.modularity(G, partition[1:], **EC)


def test_modularity_without_edges():
    assert nx.community.modularity(nx.empty_graph(3), [{0, 1, 2}], **EC) == 0


def test_left_to_networkx():
    import numpy as np

    from edgecase.nx_backend.interface import backend_interface as bi

    G = nx.path_graph(4)
    numpy_seed = nx.utils.create_py_random_state(np.random.default_rng(1))
    assert isinstance(bi.can_run("louvain_communities", (G,), {"seed": numpy_seed}), str)
    assert isinstance(bi.can_run("asyn_lpa_communities", (G, None, numpy_seed), {}), str)
    assert bi.can_run("louvain_communities", (G,), {"seed": random.Random(1)}) is True
    assert bi.can_run("louvain_communities", (G,), {}) is True
    assert isinstance(bi.can_run("label_propagation_communities", (nx.MultiGraph(G),), {}), str)


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(6))
def test_greedy_modularity(cls, seed):
    rng = random.Random(seed)
    G = random_graph(cls, 50, rng.choice([30, 90, 160]), seed, float_weights=seed % 2 == 1)
    for kw in [
        {},
        {"weight": "weight"},
        {"resolution": 0.5},
        {"resolution": 1.7, "weight": "weight"},
        {"cutoff": 5},
        {"best_n": 3},
        {"cutoff": 2, "best_n": 10},
    ]:
        expected = nx.community.greedy_modularity_communities(G, **kw)
        result = nx.community.greedy_modularity_communities(G, **kw, **EC)
        assert result == expected
        assert [type(c) for c in result] == [type(c) for c in expected]


def test_greedy_modularity_edge_cases():
    G = nx.karate_club_graph()
    for kw in [{"cutoff": 0}, {"cutoff": 40}, {"best_n": 0}, {"cutoff": 4, "best_n": 3}]:
        with pytest.raises(ValueError):
            nx.community.greedy_modularity_communities(G, **kw, **EC)
    assert nx.community.greedy_modularity_communities(G, best_n=1, **EC) == [set(G)]
    assert nx.community.greedy_modularity_communities(nx.empty_graph(3), **EC) == [
        {0},
        {1},
        {2},
    ]
    # Ties between equal gains are broken by comparing nodes, as NetworkX does
    G = nx.cycle_graph(["d", "a", "c", "b", "e", "f"])
    assert nx.community.greedy_modularity_communities(G, **EC) == (
        nx.community.greedy_modularity_communities(G)
    )
