"""The NetworkX backend's DAG algorithms, edge BFS and dominators give NetworkX's results
in NetworkX's order.
"""

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
            if isinstance(result, nx.Graph):
                result = (
                    type(result),
                    list(result.nodes(data=True)),
                    [(u, list(nbrs.items())) for u, nbrs in result.adj.items()],
                )
            elif not isinstance(result, dict | list | tuple | int | float | bool):
                result = list(result)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert actual == expected
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        assert [list(v) if isinstance(v, set) else v for v in actual.values()] == [
            list(v) if isinstance(v, set) else v for v in expected.values()
        ]
    return expected


def random_dag(cls, seed, n=16, p=0.2):
    """Edges from lower to higher labels (a DAG), nodes and edges in a random order."""
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    if cls().is_multigraph():
        edges += rng.sample(edges, len(edges) // 4)
    rng.shuffle(edges)
    G = cls()
    G.add_nodes_from(rng.sample(range(n), n))
    for u, v in edges:
        G.add_edge(u, v, weight=rng.choice([-1, 1, 2, 3]), cost=rng.random(), attr="x")
    return G


DIRECTED = [nx.DiGraph, nx.MultiDiGraph]


@pytest.mark.parametrize("cls", DIRECTED)
@pytest.mark.parametrize("seed", range(5))
def test_dag_functions(cls, seed):
    G = random_dag(cls, seed)
    assert_same(nx.lexicographical_topological_sort, G)
    assert_same(nx.lexicographical_topological_sort, G, key=lambda n: -n)
    assert_same(nx.lexicographical_topological_sort, G, key=lambda n: n % 3)
    small = random_dag(cls, seed, n=7, p=0.3)
    assert_same(nx.all_topological_sorts, small)
    for weight in ["weight", "cost", None]:
        for default_weight in [1, 2.5]:
            assert_same(nx.dag_longest_path, G, weight, default_weight)
            assert_same(nx.dag_longest_path_length, G, weight, default_weight)
    assert_same(nx.dag_longest_path, G, topo_order=list(nx.topological_sort(G)))
    for reflexive in [False, True, None, "nope"]:
        assert_same(nx.transitive_closure, G, reflexive=reflexive)
    assert_same(nx.transitive_reduction, G)
    for orientation in [None, "original", "reverse", "ignore", "nope"]:
        assert_same(nx.edge_bfs, G, orientation=orientation)
        assert_same(nx.edge_bfs, G, list(G)[:3], orientation=orientation)
    # Cycles: the errors (and what comes before them)
    H = G.copy()
    H.add_edges_from([(15, 0), (0, 15)])  # (a cycle)
    assert_same(nx.lexicographical_topological_sort, H)
    assert_same(nx.transitive_reduction, H)
    assert_same(nx.all_topological_sorts, H)
    assert_same(nx.transitive_closure, H)


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize("seed", range(3))
def test_undirected(cls, seed):
    G = nx.Graph(random_dag(nx.DiGraph, seed))
    G = cls(G)
    assert_same(nx.lexicographical_topological_sort, G)
    for reflexive in [False, True, None]:
        assert_same(nx.transitive_closure, G, reflexive=reflexive)
    assert_same(nx.edge_bfs, G)
    assert_same(nx.edge_bfs, G, orientation="ignore")


@pytest.mark.parametrize("cls", DIRECTED)
@pytest.mark.parametrize("seed", range(5))
def test_dominators_and_aperiodicity(cls, seed):
    rng = random.Random(seed)
    G = random_dag(cls, seed, n=14, p=0.2)
    for _ in range(5):  # back edges: loops and irreducible flow graphs
        u, v = rng.sample(range(14), 2)
        G.add_edge(max(u, v), min(u, v))
    for start in list(G)[:4] + ["missing"]:
        assert_same(nx.immediate_dominators, G, start)
        assert_same(nx.dominance_frontiers, G, start)
    S = nx.DiGraph(nx.cycle_graph(6, create_using=nx.DiGraph))
    S.add_edges_from([(0, 3), (2, 5)])
    for H in [S, nx.cycle_graph(6, create_using=cls), G]:
        assert_same(nx.is_aperiodic, H)
    assert_same(nx.is_aperiodic, cls())
