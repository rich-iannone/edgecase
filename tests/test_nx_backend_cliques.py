"""The NetworkX backend's clique functions give NetworkX's cliques in NetworkX's order
(which comes from Python sets, modeled exactly with the nodes' hashes).
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
            if not isinstance(result, dict | list | tuple | int | float | bool):
                result = list(result)
        except Exception as exc:
            result = (type(exc), str(exc))
        results.append(result)
    return results


def assert_same(func, *args, **kwargs):
    expected, actual = both(func, *args, **kwargs)
    assert actual == expected
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
    return expected


def random_graph(cls, seed, n=30, p=0.3, labels=None):
    rng = random.Random(seed)
    G = cls()
    nodes = list(range(n)) if labels is None else [labels(i) for i in range(n)]
    G.add_nodes_from(rng.sample(nodes, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                u, v = (nodes[i], nodes[j]) if rng.random() < 0.5 else (nodes[j], nodes[i])
                G.add_edge(u, v)
    for i in rng.sample(range(n), 3):
        G.add_edge(nodes[i], nodes[i])  # (self-loops do not count)
    return G


LABELS = [None, str, lambda i: (i % 3, f"x{i}"), lambda i: -i - 1, lambda i: i * 64]


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
def test_find_cliques(cls, seed, labels):
    G = random_graph(cls, seed, labels=labels)
    nodes = list(G)
    assert_same(nx.find_cliques, G)
    assert_same(nx.find_cliques_recursive, G)
    for given in [[nodes[0]], [nodes[0], nodes[1]], [nodes[3], "missing"]]:
        assert_same(nx.find_cliques, G, nodes=given)
        assert_same(nx.find_cliques_recursive, G, nodes=given)
    small = random_graph(cls, seed, n=12, p=0.4, labels=labels)
    assert_same(nx.enumerate_all_cliques, small)
    assert_same(nx.node_clique_number, G)
    assert_same(nx.node_clique_number, G, nodes=nodes[:5])
    assert_same(nx.node_clique_number, G, nodes=nodes[2])


@pytest.mark.parametrize("seed", range(5))
def test_max_weight_clique(seed):
    rng = random.Random(seed)
    G = random_graph(nx.Graph, seed, n=25, p=0.4)
    for v in G:
        G.nodes[v]["weight"] = rng.choice([0, 1, 2, 3, 7])
    assert_same(nx.max_weight_clique, G)
    assert_same(nx.max_weight_clique, G, weight=None)
    G.nodes[3]["weight"] = 1.5
    assert_same(nx.max_weight_clique, G)
    del G.nodes[5]["weight"]
    assert_same(nx.max_weight_clique, G)


def test_empty_and_edgeless():
    for G in [nx.Graph(), nx.empty_graph(4)]:
        for func in [nx.find_cliques, nx.find_cliques_recursive, nx.enumerate_all_cliques]:
            assert_same(func, G)
        assert_same(nx.max_weight_clique, G, weight=None)


@pytest.mark.parametrize("kind", ["ints", "negative", "strings", "colliding"])
def test_python_set_model_matches_python_sets(kind):
    """The CPython set model behind find_cliques orders members as real sets do."""
    from edgecase._core import nx_pyset_model_check

    for trial in range(150):
        rng = random.Random(trial)
        m = rng.choice([5, 20, 60, 300])
        keys = {
            "ints": lambda: list(range(m)),
            "negative": lambda: rng.sample(range(-(10**6), 10**6), m) + [-1, -2],
            "strings": lambda: [f"node{i}" for i in range(m)],
            "colliding": lambda: [i * 8 for i in range(m)],
        }[kind]()
        keys = list(dict.fromkeys(keys))
        real = [set() for _ in range(4)]
        ops, popped = [], []
        for _ in range(rng.choice([20, 200, 1000])):
            op = rng.choice(
                ["add"] * 6 + ["discard"] * 2 + ["pop", "copy", "and", "iand"] + ["sub", "isub"]
            )
            t = rng.randrange(4)
            if op in ("add", "discard"):
                k = rng.randrange(len(keys))
                ops.append((op, t, k))
                getattr(real[t], op)(keys[k])
            elif op == "pop":
                ops.append((op, t, 0))
                popped.append(keys.index(real[t].pop()) if real[t] else None)
            else:
                a = rng.choice([x for x in range(4) if x != t])
                ops.append((op, t, a))
                if op == "copy":
                    real[t] = real[a].copy()
                elif op == "and":
                    real[t] = real[t] & real[a]
                elif op == "iand":
                    real[t] &= real[a]
                elif op == "sub":
                    real[t] = real[t] - real[a]
                else:
                    real[t] -= real[a]
        orders, model_popped = nx_pyset_model_check([hash(k) for k in keys], 4, ops)
        index = {k: i for i, k in enumerate(keys)}
        assert orders == [[index[k] for k in s] for s in real]
        assert model_popped == popped
