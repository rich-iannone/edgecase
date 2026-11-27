"""Traversal, reachability, DAG and isolate functions of the NetworkX backend match NetworkX."""

import itertools
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


def random_graph(cls, n, m, seed, dag=False):
    """Nodes in shuffled order and edges added in random order, so NetworkX's neighbor and
    predecessor orders differ from node order"""
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    G.add_nodes_from(nodes)
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if dag:
            if u == v:
                continue
            u, v = sorted((u, v))
        G.add_edge(u, v)
    return G


GRAPHS = [
    pytest.param(cls, n, m, seed, id=f"{cls.__name__}-{n}-{m}-{seed}")
    for cls in ALL_CLASSES
    for n, m, seed in [(1, 0, 0), (5, 4, 1), (15, 20, 2), (40, 60, 3), (40, 120, 4)]
]


def sources_of(G, seed, k=4):
    rng = random.Random(seed)
    return rng.sample(list(G), min(k, len(G)))


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
def test_bfs(cls, n, m, seed):
    G = random_graph(cls, n, m, seed)
    for source in sources_of(G, seed):
        for depth_limit in [None, -1, 0, 1, 2, 5]:
            for reverse in [False, True]:
                kw = {"reverse": reverse, "depth_limit": depth_limit}
                expected = list(nx.bfs_edges(G, source, **kw))
                assert list(nx.bfs_edges(G, source, **kw, **EC)) == expected
                T, T_ec = nx.bfs_tree(G, source, **kw), nx.bfs_tree(G, source, **kw, **EC)
                assert list(T.edges) == list(T_ec.edges) and list(T) == list(T_ec)
            kw = {"depth_limit": depth_limit}
            assert list(nx.bfs_successors(G, source, **kw, **EC)) == list(
                nx.bfs_successors(G, source, **kw)
            )
        for distance in [-1, 0, 1, 2, 3, 50]:
            assert nx.descendants_at_distance(G, source, distance, **EC) == (
                nx.descendants_at_distance(G, source, distance)
            )
        assert nx.descendants(G, source, **EC) == nx.descendants(G, source)
        assert nx.ancestors(G, source, **EC) == nx.ancestors(G, source)
    sources = sources_of(G, seed + 1, 3)
    for srcs in [sources, sources + sources[:1], sources[:1], sources[0] if sources else []]:
        assert list(nx.bfs_layers(G, srcs, **EC)) == list(nx.bfs_layers(G, srcs))


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
def test_dfs(cls, n, m, seed):
    G = random_graph(cls, n, m, seed)
    for source, depth_limit in itertools.product(
        [None, *sources_of(G, seed, 3)], [None, -1, 0, 1, 2, 4]
    ):
        kw = {"source": source, "depth_limit": depth_limit}
        assert list(nx.dfs_labeled_edges(G, **kw, **EC)) == list(nx.dfs_labeled_edges(G, **kw))
        assert list(nx.dfs_edges(G, **kw, **EC)) == list(nx.dfs_edges(G, **kw))
        assert list(nx.dfs_preorder_nodes(G, **kw, **EC)) == list(nx.dfs_preorder_nodes(G, **kw))
        assert list(nx.dfs_postorder_nodes(G, **kw, **EC)) == list(nx.dfs_postorder_nodes(G, **kw))
        assert nx.dfs_predecessors(G, **kw, **EC) == nx.dfs_predecessors(G, **kw)
        assert list(nx.dfs_successors(G, **kw, **EC).items()) == list(
            nx.dfs_successors(G, **kw).items()
        )
        T, T_ec = nx.dfs_tree(G, **kw), nx.dfs_tree(G, **kw, **EC)
        assert list(T) == list(T_ec) and list(T.edges) == list(T_ec.edges)


@pytest.mark.parametrize("cls", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dag", [True, False])
def test_topological(cls, seed, dag):
    G = random_graph(cls, 30, 50, seed, dag=dag)
    if not dag and seed % 2:
        G.add_edge(*list(G)[:1] * 2)  # a self-loop
    assert nx.is_directed_acyclic_graph(G, **EC) == nx.is_directed_acyclic_graph(G)

    def run(func, **kw):
        out = []
        try:
            for x in func(G, **kw):
                out.append(x)
        except nx.NetworkXUnfeasible:
            out.append("unfeasible")
        return out

    assert run(nx.topological_sort, **EC) == run(nx.topological_sort)
    assert run(nx.topological_generations, **EC) == run(nx.topological_generations)


def test_topological_errors():
    with pytest.raises(nx.NetworkXError, match="undirected"):
        list(nx.topological_sort(nx.path_graph(3), **EC))
    with pytest.raises(nx.NetworkXUnfeasible):
        list(nx.topological_generations(nx.DiGraph([(0, 1), (1, 0)]), **EC))
    assert nx.is_directed_acyclic_graph(nx.path_graph(3), **EC) is False


@pytest.mark.parametrize(("cls", "n", "m", "seed"), GRAPHS)
def test_isolates(cls, n, m, seed):
    G = random_graph(cls, n, m, seed)
    G.add_nodes_from(["lonely", "alone"])
    G.add_edge("self", "self")
    assert list(nx.isolates(G, **EC)) == list(nx.isolates(G))
    assert nx.number_of_isolates(G, **EC) == nx.number_of_isolates(G)
    for node in G:
        assert nx.is_isolate(G, node, **EC) == nx.is_isolate(G, node)


def test_missing_nodes():
    G = nx.path_graph(3)
    D = nx.DiGraph([(0, 1)])
    for func in [nx.bfs_edges, nx.dfs_edges, nx.bfs_successors]:
        with pytest.raises(nx.NetworkXError, match="not in the graph"):
            list(func(G, 7, **EC))
    with pytest.raises(nx.NetworkXError, match="not in the digraph"):
        list(nx.bfs_edges(D, 7, reverse=True, **EC))
    with pytest.raises(nx.NetworkXError):
        nx.descendants(G, 7, **EC)
    with pytest.raises(nx.NetworkXError):
        nx.ancestors(D, 7, **EC)
    with pytest.raises(nx.NetworkXError):
        nx.descendants_at_distance(G, 7, 1, **EC)
    with pytest.raises(nx.NetworkXError):
        list(nx.bfs_layers(G, [0, 7], **EC))
    # NetworkX's labeled DFS reports the start before failing on it
    steps = nx.dfs_labeled_edges(G, 7, **EC)
    assert next(steps) == (7, 7, "forward")
    with pytest.raises(nx.NetworkXError):
        next(steps)


def test_reverse_bfs_follows_predecessor_order():
    # Predecessors of 1 are [2, 0] (insertion order), not [0, 2] (node order)
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(2, 1)
    G.add_edge(0, 1)
    G.add_edge(3, 0)
    G.add_edge(3, 2)
    assert (
        list(nx.bfs_edges(G, 1, reverse=True, **EC))
        == list(nx.bfs_edges(G, 1, reverse=True))
        == [(1, 2), (1, 0), (2, 3)]
    )


def test_sort_neighbors_is_left_to_networkx():
    from edgecase.nx_backend.interface import backend_interface

    G = nx.path_graph(4)
    assert isinstance(
        backend_interface.can_run("bfs_edges", (G, 0), {"sort_neighbors": sorted}), str
    )
    assert isinstance(backend_interface.can_run("dfs_edges", (G,), {"sort_neighbors": sorted}), str)
    assert backend_interface.can_run("dfs_edges", (G,), {}) is True
