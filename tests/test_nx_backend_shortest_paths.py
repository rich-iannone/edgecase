"""The NetworkX backend's shortest paths give NetworkX's exact answers.

Each check calls a NetworkX function twice, once with `backend="edgecase"`, and compares
the results including dict order and which of several equally short paths is reported.
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
            if not isinstance(result, dict | list | tuple | int | float | bool):
                result = list(result)  # (a generator)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def ordered(x):
    """`x` with every dict replaced by its items, so that comparisons see the order."""
    if isinstance(x, dict):
        return [(k, ordered(v)) for k, v in x.items()]
    if isinstance(x, list | tuple):
        return [ordered(v) for v in x]
    return x


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert ordered(actual) == ordered(expected)
    return expected


def random_graph(cls, n=14, p=0.25, seed=0, weights=(0, 1, 2, 3)):
    """A graph with ties among shortest paths, edges added in a random order."""
    rng = random.Random(seed)
    G = cls()
    nodes = list(range(n))
    rng.shuffle(nodes)
    G.add_nodes_from(nodes)
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
    edges += [(u, u) for u in range(n) if rng.random() < 0.1]
    if G.is_multigraph():
        edges += rng.sample(edges, len(edges) // 4)  # parallel edges
    rng.shuffle(edges)
    for u, v in edges:
        G.add_edge(u, v, weight=rng.choice(weights))
    return G


def graphs():
    for cls in GRAPH_CLASSES:
        for seed in range(4):
            yield random_graph(cls, seed=seed)
    yield random_graph(nx.Graph, seed=7, weights=(0.1, 0.2, 0.30000000000000004, 1.5))
    yield random_graph(nx.DiGraph, n=30, p=0.1, seed=8)


GRAPHS = list(graphs())


def graph_id(G):
    return f"{type(G).__name__}-{G.number_of_nodes()}-{G.number_of_edges()}"


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
def test_unweighted(G):
    for source in list(G)[:5]:
        for cutoff in (None, 0, 1, 2.5):
            assert_same(nx.single_source_shortest_path_length, G, source, cutoff=cutoff)
            assert_same(nx.single_source_shortest_path, G, source, cutoff=cutoff)
        assert_same(nx.predecessor, G, source)
        assert_same(nx.predecessor, G, source, cutoff=2)
        assert_same(nx.predecessor, G, source, return_seen=True)
        for target in list(G)[-4:]:
            assert_same(nx.predecessor, G, source, target=target, return_seen=True)
            assert_same(nx.has_path, G, source, target)
            assert_same(nx.shortest_path_length, G, source, target)
    assert_same(nx.all_pairs_shortest_path_length, G, cutoff=2)
    assert_same(nx.all_pairs_shortest_path, G)
    assert_same(nx.shortest_path, G)
    assert_same(nx.shortest_path_length, G)


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
def test_unweighted_towards_a_target(G):
    # (a directed graph's predecessors follow the recorded `G._pred` order)
    for node in list(G)[:5]:
        assert_same(nx.single_target_shortest_path, G, node, cutoff=2)
        assert_same(nx.shortest_path, G, target=node)
        for source in list(G)[-4:]:
            assert_same(nx.bidirectional_shortest_path, G, source, node)
        assert_same(nx.single_target_shortest_path_length, G, node)
        assert_same(nx.shortest_path_length, G, target=node)


def test_predecessor_order_differs_from_arc_order():
    # 3's predecessors were added as 2, 0, 1; its arcs come from 0, 1, 2 (node order)
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(2, 3), (0, 3), (1, 3), (4, 0), (4, 1), (4, 2)])
    assert list(G.pred[3]) == [2, 0, 1]
    assert_same(nx.single_target_shortest_path, G, 3)
    assert_same(nx.shortest_path, G, target=3)
    assert_same(nx.shortest_path, G, target=3, weight="weight")
    assert_same(nx.bidirectional_shortest_path, G, 4, 3)
    assert_same(nx.bidirectional_dijkstra, G, 4, 3)
    assert nx.single_target_shortest_path(G, 3, backend="edgecase")[4] == [4, 2, 3]
    # Without the recorded order, predecessors come in node order: another valid answer
    config = nx.config.backends.edgecase
    previous = config.preserve_pred_order
    config.preserve_pred_order = False
    try:
        paths = nx.single_target_shortest_path(G, 3, backend="edgecase")
    finally:
        config.preserve_pred_order = previous
    assert paths[4] == [4, 0, 3]


@pytest.mark.parametrize("G", GRAPHS, ids=graph_id)
@pytest.mark.parametrize("weight", ["weight", None, "missing"])
def test_dijkstra(G, weight):
    nodes = list(G)
    for source in nodes[:4]:
        for cutoff in (None, 0, 2):
            assert_same(nx.single_source_dijkstra, G, source, cutoff=cutoff, weight=weight)
            assert_same(nx.single_source_dijkstra_path_length, G, source, cutoff, weight)
            assert_same(nx.dijkstra_predecessor_and_distance, G, source, cutoff, weight)
        for target in nodes[-4:]:
            assert_same(nx.single_source_dijkstra, G, source, target, weight=weight)
            assert_same(nx.dijkstra_path, G, source, target, weight=weight)
            assert_same(nx.dijkstra_path_length, G, source, target, weight=weight)
            assert_same(nx.shortest_path_length, G, source, target, weight=weight)
            assert_same(nx.bidirectional_dijkstra, G, source, target, weight)
            assert_same(nx.shortest_path, G, source, target, weight=weight)
    sources = set(nodes[2:5])
    assert_same(nx.multi_source_dijkstra, G, sources, weight=weight)
    assert_same(nx.multi_source_dijkstra, G, sources, target=nodes[-1], weight=weight)
    assert_same(nx.multi_source_dijkstra_path_length, G, sources, cutoff=1, weight=weight)
    assert_same(nx.all_pairs_dijkstra, G, weight=weight)
    assert_same(nx.all_pairs_dijkstra_path_length, G, cutoff=2, weight=weight)
    assert_same(nx.shortest_path, G, weight=weight)
    assert_same(nx.shortest_path, G, target=nodes[0], weight=weight)
    assert_same(nx.shortest_path_length, G, target=nodes[0], weight=weight)


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("weight", [None, "weight"])
def test_average_shortest_path_length(cls, weight):
    rng = random.Random(5)
    G = nx.complete_graph(9, create_using=cls)
    G.remove_edges_from(rng.sample(list(G.edges), 12))
    for u, v, d in G.edges(data=True):
        d["weight"] = rng.random()
    expected = assert_same(nx.average_shortest_path_length, G, weight=weight)
    assert isinstance(expected, float)  # (and the floats are equal to the last bit)
    G = nx.path_graph(4, create_using=cls)
    assert_same(nx.average_shortest_path_length, G, weight=weight)  # not strongly connected
    assert_same(nx.average_shortest_path_length, nx.empty_graph(0, create_using=cls))
    assert_same(nx.average_shortest_path_length, nx.empty_graph(1, create_using=cls))
    assert_same(nx.average_shortest_path_length, G, method="nope")


def test_integer_weights_give_integer_distances():
    G = nx.Graph([(0, 1, {"weight": 2}), (1, 2, {"weight": 3}), (0, 2, {"weight": 9})])
    dist = nx.single_source_dijkstra_path_length(G, 0, backend="edgecase")
    assert dist == {0: 0, 1: 2, 2: 5}
    assert all(type(d) is int for d in dist.values())
    G[0][1]["weight"] = 2.5
    dist = nx.single_source_dijkstra_path_length(G, 0, backend="edgecase")
    assert dist == {0: 0, 1: 2.5, 2: 5.5}
    assert type(dist[0]) is int  # (a source's distance is the int 0, as in NetworkX)
    assert nx.dijkstra_path_length(G, 0, 0, backend="edgecase") == 0


def test_multigraph_uses_the_cheapest_parallel_edge():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b", weight=5)
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c")  # (weight 1 by default)
    G.add_edge("a", "c", weight=3)
    assert_same(nx.single_source_dijkstra, G, "a")
    assert nx.dijkstra_path(G, "a", "c", backend="edgecase") == ["a", "b", "c"]


def test_none_weight_hides_an_edge():
    G = nx.Graph()
    G.add_edge(0, 1, weight=None)
    G.add_edge(1, 2, weight=1)
    G.add_edge(0, 2, weight=5)
    assert_same(nx.single_source_dijkstra, G, 0)
    assert nx.dijkstra_path(G, 0, 1, backend="edgecase") == [0, 2, 1]
    assert_same(nx.bidirectional_dijkstra, G, 0, 1)


def test_negative_weights():
    G = nx.DiGraph()
    nx.add_path(G, [0, 1, 2, 3], weight=1)
    G.add_edge(0, 3, weight=1)
    G.add_edge(3, 1, weight=-5)
    for func in [
        nx.single_source_dijkstra,
        nx.single_source_dijkstra_path_length,
        nx.dijkstra_predecessor_and_distance,
    ]:
        nx_result, ec_result = both(func, G, 0)
        assert ec_result == nx_result
        assert ec_result[0] is ValueError
    nx_result, ec_result = both(nx.all_pairs_dijkstra_path_length, G)
    assert ec_result == nx_result
    assert_same(nx.bidirectional_dijkstra, G, 0, 2)
    G = nx.cycle_graph(4)
    G[0][1]["weight"] = -1
    assert_same(nx.bidirectional_dijkstra, G, 0, 2)
    assert_same(nx.single_source_dijkstra, G, 3)


@pytest.mark.parametrize(
    "func, args",
    [
        (nx.single_source_shortest_path_length, (9,)),
        (nx.single_source_shortest_path, (9,)),
        (nx.single_target_shortest_path_length, (9,)),
        (nx.single_target_shortest_path, (9,)),
        (nx.bidirectional_shortest_path, (9, 0)),
        (nx.bidirectional_shortest_path, (0, 9)),
        (nx.bidirectional_shortest_path, (9, 9)),
        (nx.predecessor, (9,)),
        (nx.has_path, (0, 9)),
        (nx.dijkstra_path, (9, 0)),
        (nx.dijkstra_path, (0, 9)),
        (nx.dijkstra_path, (0, 3)),
        (nx.dijkstra_path_length, (9, 9)),
        (nx.dijkstra_path_length, (0, 9)),
        (nx.single_source_dijkstra, (9,)),
        (nx.single_source_dijkstra, (0, 9)),
        (nx.multi_source_dijkstra, ([],)),
        (nx.multi_source_dijkstra, ([0, 9],)),
        (nx.dijkstra_predecessor_and_distance, (9,)),
        (nx.bidirectional_dijkstra, (0, 9)),
        (nx.bidirectional_dijkstra, (9, 0)),
        (nx.shortest_path, (9,)),
        (nx.shortest_path, (None, 9)),
        (nx.shortest_path, (None, 9, "weight")),
        (nx.shortest_path, (0, 1, None, "nope")),
        (nx.shortest_path_length, (None, 9)),
        (nx.shortest_path_length, (0, 9, "weight")),
        (nx.shortest_path_length, ([1],)),
    ],
)
def test_errors_match(func, args):
    G = nx.DiGraph([(0, 1), (1, 2)])
    G.add_node(3)
    nx_result, ec_result = both(func, G, *args)
    assert ec_result == nx_result


def test_no_path():
    G = nx.DiGraph([(0, 1), (2, 1)])
    for func in [nx.bidirectional_shortest_path, nx.dijkstra_path, nx.bidirectional_dijkstra]:
        nx_result, ec_result = both(func, G, 0, 2)
        assert ec_result == nx_result
        assert ec_result[0] is nx.NetworkXNoPath
    assert nx.has_path(G, 0, 2, backend="edgecase") is False
    assert nx.has_path(G, 2, 1, backend="edgecase") is True


def test_callable_weight_is_left_to_networkx():
    G = nx.path_graph(3)

    def weight(u, v, d):
        return 1

    reason = backend_interface.can_run("dijkstra_path", (G, 0, 2, weight), {})
    assert isinstance(reason, str)
    assert backend_interface.can_run("dijkstra_path", (G, 0, 2), {"weight": "w"}) is True
    assert isinstance(backend_interface.can_run("shortest_path", (G, 0, 2, weight), {}), str)
    assert backend_interface.can_run("shortest_path", (G, 0, 2, "w", "bellman-ford"), {}) is True
    # The dispatcher then runs NetworkX's own code
    assert nx.dijkstra_path(G, 0, 2, weight=weight) == [0, 1, 2]


def test_float_distance_sums_match_networkx_bits():
    rng = random.Random(3)
    G = nx.gnm_random_graph(40, 200, seed=3)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.random() * 1e3
    ec = nx.average_shortest_path_length(G, weight="weight", backend="edgecase")
    expected = nx.average_shortest_path_length(G, weight="weight", backend="networkx")
    assert ec == expected
    assert math.isfinite(ec)


def test_generators_are_lazy():
    G = nx.path_graph(1000)
    it = nx.all_pairs_shortest_path_length(G, backend="edgecase")
    node, lengths = next(it)
    assert node == 0
    assert lengths[999] == 999


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
def test_path_adjacency_from_graph_agrees_with_arcs(cls):
    """`PathAdjacency.from_graph()` (an edgecase Graph) equals the arc-array construction."""
    from edgecase._core import PathAdjacency
    from edgecase.nx_backend import from_networkx
    from edgecase.nx_backend.algorithms.shortest_paths import _adjacency

    G = random_graph(cls, seed=3, weights=(1, 2.5, 0.25))
    G.add_edge(5, 5, weight=4)  # a self-loop
    G.add_edge(2, 9, weight=0.5)
    G.add_edge(2, 9, weight=7)  # (a parallel edge on a multigraph)
    G.add_node("isolated")
    for weight in [None, "weight"]:
        E = from_networkx(G, edge_attrs=None if weight is None else {weight: 1})
        expected = _adjacency(E, weight).arcs()
        rust_graph = E.rust_graph(weight)
        adjacency, node_ids = PathAdjacency.from_graph(rust_graph, weighted=weight is not None)
        assert node_ids == list(range(1, len(G) + 1))
        assert adjacency.arcs() == expected
        assert adjacency.num_nodes == len(G)
        # The searches agree too
        assert adjacency.source_stats(list(range(len(G))), weight is not None) == _adjacency(
            E, weight
        ).source_stats(list(range(len(G))), weight is not None)


def test_path_adjacency_from_an_edgecase_graph():
    from edgecase._core import Graph, PathAdjacency

    # Undirected: 1-2 twice (not adjacent in the edge list), 2-3 without a weight, 3-3
    rust_graph = Graph.from_edge_arrays(
        3, [1, 2, 1, 3], [2, 3, 2, 3], [5.0, None, 2.0, 1.5], directed=False
    )
    adjacency, node_ids = PathAdjacency.from_graph(rust_graph)
    assert node_ids == [1, 2, 3]
    indptr, indices, weights = adjacency.arcs()
    assert indptr == [0, 1, 3, 5]
    # each neighbor once, at its first place; the cheapest parallel edge; missing weight 1
    assert indices == [1, 0, 2, 1, 2]
    assert weights == [2.0, 2.0, 1.0, 1.0, 1.5]
    _, levels, _ = adjacency.bfs_levels(0)
    assert levels == [0, 1, 2]
    _, unweighted = PathAdjacency.from_graph(rust_graph, weighted=False)[0].arcs()[1:]
    assert unweighted == [1.0] * 5


def negative_weight_graph(cls, seed, cycle=False, n=14):
    """A graph with negative weights: on a DAG's edges (no negative cycle) unless `cycle`."""
    rng = random.Random(seed)
    G = cls()
    nodes = list(range(n))
    rng.shuffle(nodes)
    G.add_nodes_from(nodes)
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < 0.25]
    if cls().is_multigraph():
        edges += rng.sample(edges, len(edges) // 4)
    rng.shuffle(edges)
    for u, v in edges:
        negative = G.is_directed() and u < v and rng.random() < 0.3
        G.add_edge(u, v, weight=rng.choice([-2, -1]) if negative else rng.choice([0, 1, 2, 3]))
        G.add_edge(u, v, cost=rng.random() * (-1 if negative else 1))
    if cycle:
        G.add_edge(n - 1, 0, weight=-50, cost=-50.0)
    return G


@pytest.mark.parametrize("cls", GRAPH_CLASSES)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weight", ["weight", "cost", None])
def test_bellman_ford_family(cls, seed, weight):
    G = negative_weight_graph(cls, seed)
    nodes = list(G)
    for source in nodes[:4]:
        for heuristic in [False, True]:
            assert_same(
                nx.bellman_ford_predecessor_and_distance,
                G,
                source,
                weight=weight,
                heuristic=heuristic,
            )
        assert_same(nx.single_source_bellman_ford, G, source, weight=weight)
        assert_same(nx.single_source_bellman_ford_path_length, G, source, weight=weight)
        assert_same(nx.goldberg_radzik, G, source, weight=weight)
        for target in nodes[-3:]:
            assert_same(nx.bellman_ford_path, G, source, target, weight=weight)
            assert_same(nx.bellman_ford_path_length, G, source, target, weight=weight)
            assert_same(nx.single_source_bellman_ford, G, source, target, weight=weight)
            assert_same(nx.all_shortest_paths, G, source, target, weight=weight)
            for method in ["dijkstra", "bellman-ford"]:
                assert_same(nx.shortest_path, G, source, target, weight, method)
                assert_same(nx.shortest_path_length, G, source, target, weight, method)
                assert_same(nx.all_shortest_paths, G, source, target, weight, method)
        assert_same(nx.single_source_all_shortest_paths, G, source, weight, "bellman-ford")
        assert_same(nx.shortest_path, G, source, weight=weight, method="bellman-ford")
        assert_same(nx.shortest_path, G, target=source, weight=weight, method="bellman-ford")
        assert_same(nx.shortest_path_length, G, target=source, weight=weight, method="bellman-ford")
    assert_same(nx.all_pairs_bellman_ford_path_length, G, weight=weight)
    assert_same(nx.all_pairs_bellman_ford_path, G, weight=weight)
    assert_same(nx.all_pairs_all_shortest_paths, G, weight=weight, method="bellman-ford")
    assert_same(nx.johnson, G, weight=weight)
    for heuristic in [False, True]:
        assert_same(nx.negative_edge_cycle, G, weight=weight, heuristic=heuristic)
    assert_same(nx.find_negative_cycle, G, nodes[0], weight=weight)


@pytest.mark.parametrize("cls", [nx.DiGraph, nx.MultiDiGraph, nx.Graph])
@pytest.mark.parametrize("seed", range(3))
def test_negative_cycles(cls, seed):
    G = negative_weight_graph(cls, seed, cycle=True)
    nodes = list(G)
    for weight in ["weight", "cost"]:
        for source in nodes[:4]:
            for heuristic in [False, True]:
                assert_same(
                    nx.bellman_ford_predecessor_and_distance,
                    G,
                    source,
                    weight=weight,
                    heuristic=heuristic,
                )
            assert_same(nx.goldberg_radzik, G, source, weight=weight)
            assert_same(nx.find_negative_cycle, G, source, weight=weight)
            assert_same(nx.single_source_bellman_ford, G, source, weight=weight)
        for heuristic in [False, True]:
            assert_same(nx.negative_edge_cycle, G, weight=weight, heuristic=heuristic)
        assert_same(nx.johnson, G, weight=weight)
    G.add_edge(nodes[2], nodes[2], weight=-1)  # (a negative self-loop)
    assert_same(nx.negative_edge_cycle, G)
    assert_same(nx.goldberg_radzik, G, nodes[0])
    assert_same(nx.find_negative_cycle, G, nodes[2])


def test_goldberg_radzik_follows_python_set_order():
    # String nodes: the order of NetworkX's sets of relabeled nodes depends on their hashes
    G = negative_weight_graph(nx.DiGraph, 5, n=30)
    G = nx.relabel_nodes(G, {n: f"node {n}" for n in G})
    for source in list(G)[:5]:
        assert_same(nx.goldberg_radzik, G, source)
        assert_same(nx.goldberg_radzik, G, source, weight="cost")


def test_bellman_ford_errors():
    G = nx.DiGraph([(0, 1, {"weight": -1}), (1, 2, {"weight": 2})])
    G.add_node(3)
    cases = [
        (nx.bellman_ford_path, (0, 3)),
        (nx.bellman_ford_path, (9, 0)),
        (nx.bellman_ford_path, (9, 9)),
        (nx.bellman_ford_path_length, (0, 3)),
        (nx.bellman_ford_path_length, (9, 9)),
        (nx.bellman_ford_path_length, (9, 0)),
        (nx.single_source_bellman_ford, (9,)),
        (nx.single_source_bellman_ford, (0, 3)),
        (nx.single_source_bellman_ford_path_length, (9,)),
        (nx.bellman_ford_predecessor_and_distance, (9,)),
        (nx.goldberg_radzik, (9,)),
        (nx.find_negative_cycle, (0,)),
        (nx.find_negative_cycle, (9,)),
        (nx.all_shortest_paths, (0, 3, "weight", "nope")),
        (nx.all_shortest_paths, (0, 3, "weight", "bellman-ford")),
    ]
    for func, args in cases:
        nx_result, ec_result = both(func, G, *args)
        assert ec_result == nx_result, func
    assert_same(nx.single_source_all_shortest_paths, G, 0, "weight", "nope")
    assert_same(nx.bellman_ford_predecessor_and_distance, nx.DiGraph([(0, 0)]), 0)
    assert_same(nx.goldberg_radzik, nx.empty_graph(1), 0)
    assert_same(nx.johnson, nx.DiGraph())
    assert_same(nx.negative_edge_cycle, nx.empty_graph(3))
    assert_same(nx.average_shortest_path_length, nx.cycle_graph(6), "weight", "bellman-ford")
