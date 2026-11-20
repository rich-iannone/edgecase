"""Matrix exports of the NetworkX backend equal NetworkX's (values, dtype and format)."""

import random

import pytest

nx = pytest.importorskip("networkx")
np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy")

ALL_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
EC = {"backend": "edgecase"}


@pytest.fixture(autouse=True)
def _no_cache_warnings():
    ignored = set(nx.config.warnings_to_ignore)
    nx.config.warnings_to_ignore.add("cache")
    yield
    nx.config.warnings_to_ignore = ignored


def random_graph(cls, n, m, seed, kind):
    rng = random.Random(seed)
    G = cls()
    nodes = rng.sample(range(10 * n), n)
    G.add_nodes_from(nodes)
    for _ in range(m):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if kind == "int":
            G.add_edge(u, v, weight=rng.choice([1, 2, 3]))
        elif kind == "float":
            G.add_edge(u, v, weight=rng.random())
        elif kind == "partial":
            if rng.random() < 0.5:
                G.add_edge(u, v, weight=rng.random())
            else:
                G.add_edge(u, v)
        else:
            G.add_edge(u, v, weight=rng.random() < 0.5)
    return G


def assert_sparse_equal(A, B):
    assert type(A) is type(B)
    assert A.format == B.format
    assert A.dtype == B.dtype
    assert A.shape == B.shape
    assert (A != B).nnz == 0


GRAPHS = [
    pytest.param(cls, seed, kind, id=f"{cls.__name__}-{seed}-{kind}")
    for cls in ALL_CLASSES
    for seed in range(3)
    for kind in ["int", "float", "partial", "bool"]
]


@pytest.mark.parametrize(("cls", "seed", "kind"), GRAPHS)
def test_sparse(cls, seed, kind):
    G = random_graph(cls, 25, 60, seed, kind)
    nodes = list(G)
    rng = random.Random(seed)
    nodelists = [None, nodes[::-1], rng.sample(nodes, 10), nodes[:1]]
    for nodelist in nodelists:
        for kw in [{}, {"weight": None}, {"dtype": np.float32}, {"dtype": int}]:
            expected = nx.to_scipy_sparse_array(G, nodelist=nodelist, **kw)
            assert_sparse_equal(
                nx.to_scipy_sparse_array(G, nodelist=nodelist, **kw, **EC), expected
            )
        for fmt in ["csr", "csc", "coo", "lil", "dok", "bsr", "dia"]:
            expected = nx.adjacency_matrix(G, nodelist=nodelist, format=fmt)
            assert_sparse_equal(
                nx.adjacency_matrix(G, nodelist=nodelist, format=fmt, **EC), expected
            )
        if not G.is_directed():
            for func in [nx.laplacian_matrix, nx.normalized_laplacian_matrix]:
                for kw in [{}, {"weight": None}]:
                    expected = func(G, nodelist=nodelist, **kw)
                    result = func(G, nodelist=nodelist, **kw, **EC)
                    assert_sparse_equal(result, expected)
    # Same COO entries in the same order for the whole graph
    A = nx.to_scipy_sparse_array(G, format="coo", **EC)
    B = nx.to_scipy_sparse_array(G, format="coo")
    assert np.array_equal(A.row, B.row) and np.array_equal(A.col, B.col)
    assert np.array_equal(A.data, B.data)


@pytest.mark.parametrize(("cls", "seed", "kind"), GRAPHS)
def test_numpy(cls, seed, kind):
    G = random_graph(cls, 25, 60, seed, kind)
    nodes = list(G)
    rng = random.Random(seed)
    for nodelist in [None, nodes[::-1], rng.sample(nodes, 10)]:
        for kw in [
            {},
            {"weight": None},
            {"nonedge": np.nan},
            {"dtype": int},
            {"dtype": bool},
            {"order": "F"},
            {"multigraph_weight": min},
            {"multigraph_weight": max, "nonedge": -1.0},
        ]:
            expected = nx.to_numpy_array(G, nodelist=nodelist, **kw)
            result = nx.to_numpy_array(G, nodelist=nodelist, **kw, **EC)
            assert result.dtype == expected.dtype
            assert result.flags.f_contiguous == expected.flags.f_contiguous
            np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_dict_of_lists(cls):
    G = random_graph(cls, 20, 40, 1, "int")
    assert nx.to_dict_of_lists(G, **EC) == nx.to_dict_of_lists(G)
    some = list(G)[::3]
    assert nx.to_dict_of_lists(G, some, **EC) == nx.to_dict_of_lists(G, some)
    with pytest.raises(nx.NetworkXError):
        nx.to_dict_of_lists(G, [1234567], **EC)


def test_errors():
    G = nx.path_graph(4)
    with pytest.raises(nx.NetworkXError, match="no nodes or edges"):
        nx.to_scipy_sparse_array(nx.Graph(), **EC)
    with pytest.raises(nx.NetworkXError, match="has no nodes"):
        nx.to_scipy_sparse_array(G, nodelist=[], **EC)
    with pytest.raises(nx.NetworkXError, match="not in G"):
        nx.to_scipy_sparse_array(G, nodelist=[0, 9], **EC)
    with pytest.raises(nx.NetworkXError, match="duplicates"):
        nx.to_scipy_sparse_array(G, nodelist=[0, 0], **EC)
    with pytest.raises(nx.NetworkXError, match="Unknown sparse matrix format"):
        nx.to_scipy_sparse_array(G, format="nope", **EC)
    with pytest.raises(nx.NetworkXError, match="not in G"):
        nx.to_numpy_array(G, nodelist=[0, 9], **EC)
    with pytest.raises(nx.NetworkXError, match="duplicates"):
        nx.to_numpy_array(G, nodelist=[0, 0], **EC)
    assert nx.to_numpy_array(nx.empty_graph(2), **EC).tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_object_and_structured_dtypes_left_to_networkx():
    from edgecase.nx_backend.interface import backend_interface as bi

    G = nx.path_graph(3)
    dt = np.dtype([("weight", float), ("cost", int)])
    assert isinstance(bi.can_run("to_numpy_array", (G,), {"dtype": dt}), str)
    assert isinstance(bi.can_run("to_numpy_array", (G,), {"dtype": object}), str)
    assert isinstance(bi.can_run("to_scipy_sparse_array", (G, None, object), {}), str)
    assert bi.can_run("to_numpy_array", (G,), {"dtype": float}) is True
