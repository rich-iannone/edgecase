"""Relabeling, ego graphs and small predicates of the NetworkX backend match NetworkX."""

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
    nodes = rng.sample(range(100), 15)
    for v in nodes:
        G.add_node(v, size=rng.random())
    for _ in range(30):
        G.add_edge(rng.choice(nodes), rng.choice(nodes), weight=rng.choice([-1, 1, 2]))
    G.graph["name"] = "g"
    return G


def identical(H, R):
    """Same class, attributes and iteration order of nodes, neighbors and keys."""
    assert type(H) is type(R)
    assert H.graph == R.graph
    assert list(H.nodes(data=True)) == list(R.nodes(data=True))
    for u in R:
        assert list(H._adj[u].items()) == list(R._adj[u].items())


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(4))
def test_relabel(cls, seed):
    from edgecase.nx_backend import EdgecaseGraph, to_networkx

    G = random_graph(cls, seed)
    nodes = list(G)
    mappings = [
        {n: f"v{n}" for n in nodes},
        {nodes[0]: nodes[1], nodes[2]: "new"},  # merges two nodes
        str,
    ]
    for mapping in mappings:
        result = nx.relabel_nodes(G, mapping, **EC)
        assert isinstance(result, EdgecaseGraph)
        identical(to_networkx(result), nx.relabel_nodes(G, mapping))
    for ordering in ["default", "sorted", "increasing degree", "decreasing degree"]:
        kw = {"first_label": 3, "ordering": ordering, "label_attribute": "old"}
        result = nx.convert_node_labels_to_integers(G, **kw, **EC)
        identical(to_networkx(result), nx.convert_node_labels_to_integers(G, **kw))
    # A chain of calls on an edgecase graph stays on edgecase
    H = nx.convert_node_labels_to_integers(G, **EC)
    assert isinstance(nx.relabel_nodes(H, str), EdgecaseGraph)


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("seed", range(4))
def test_ego_and_predicates(cls, seed):
    G = random_graph(cls, seed)
    n = list(G)[0]
    for kw in [{}, {"radius": 2}, {"center": False}, {"distance": "weight", "radius": 1.5}]:
        if kw.get("distance") and any(d["weight"] < 0 for *_, d in G.edges(data=True)):
            continue
        identical(nx.ego_graph(G, n, **kw, **EC), nx.ego_graph(G, n, **kw))
    if not G.is_directed():
        identical(nx.ego_graph(G, n, undirected=True, **EC), nx.ego_graph(G, n, undirected=True))
    assert nx.is_empty(G, **EC) == nx.is_empty(G)
    assert nx.is_empty(nx.empty_graph(3), **EC)
    for weight in ["weight", "size"]:
        assert nx.is_weighted(G, weight=weight, **EC) == nx.is_weighted(G, weight=weight)
        assert nx.is_negatively_weighted(G, weight=weight, **EC) == (
            nx.is_negatively_weighted(G, weight=weight)
        )
    if G.is_directed():
        assert nx.algorithms.dag.has_cycle(G, **EC) == nx.algorithms.dag.has_cycle(G)
        assert nx.algorithms.dag.has_cycle(nx.DiGraph([(0, 1), (1, 2)]), **EC) is False


def test_negative_weight_check_without_a_weight():
    G = nx.path_graph(3)
    assert nx.is_negatively_weighted(G, weight=None, **EC) is False
