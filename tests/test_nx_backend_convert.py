"""Round trips through the NetworkX backend's graph type (`edgecase.nx_backend`)."""

import math

import pytest

nx = pytest.importorskip("networkx")

from edgecase.nx_backend import EdgecaseGraph, from_networkx, to_networkx
from edgecase.nx_backend.graph import MISSING


def roundtrip(G, **kwargs):
    kwargs.setdefault("preserve_edge_attrs", True)
    kwargs.setdefault("preserve_node_attrs", True)
    kwargs.setdefault("preserve_graph_attrs", True)
    E = from_networkx(G, **kwargs)
    assert isinstance(E, EdgecaseGraph)
    return E, to_networkx(E)


def assert_identical(G, H):
    """Same class, attributes and iteration order of nodes, neighbors and edge keys."""
    assert type(H) is type(G)
    assert H.graph == G.graph
    assert list(H._node.items()) == list(G._node.items())
    assert list(H._adj) == list(G._adj)
    for u, nbrs in G._adj.items():
        assert list(H._adj[u]) == list(nbrs)
        for v, data in nbrs.items():
            if G.is_multigraph():
                assert list(H._adj[u][v].items()) == list(data.items())
            else:
                assert H._adj[u][v] == data
    if G.is_directed():
        assert {n: set(p) for n, p in H._pred.items()} == {n: set(p) for n, p in G._pred.items()}
    assert H.number_of_edges() == G.number_of_edges()


def mixed_key_graph(cls):
    G = cls()
    G.add_nodes_from([("t", 1), "s", 3, 2.5, frozenset({1}), None.__class__.__name__])
    G.add_edge(("t", 1), "s", weight=2.0, color="red")
    G.add_edge("s", 3)
    G.add_edge(3, ("t", 1), weight=-1)
    G.add_edge(2.5, 2.5, weight=0.5)
    G.add_node("isolated", size=4)
    return G


ALL_CLASSES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_empty_graph(cls):
    E, H = roundtrip(cls())
    assert len(E) == 0
    assert E.number_of_edges() == 0
    assert E.rust_graph().num_nodes() == 0
    assert_identical(cls(), H)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_mixed_keys_self_loops_and_isolated_nodes(cls):
    G = mixed_key_graph(cls)
    G.graph["name"] = "mixed"
    E, H = roundtrip(G)
    assert E.number_of_edges() == G.number_of_edges() == 4
    assert E.rust_graph().num_nodes() == G.number_of_nodes()
    assert E.rust_graph().num_edges() == G.number_of_edges()
    assert_identical(G, H)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_neighbor_order_survives(cls):
    # The closing edge of a cycle is added last, so node 0 lists 1 before 4 but node 4
    # lists 3 before 0; adding edges in `G.edges()` order would get node 4 wrong
    G = nx.cycle_graph(5, create_using=cls)
    G.add_edges_from([(2, 0), (4, 1)])
    _, H = roundtrip(G)
    assert_identical(G, H)


@pytest.mark.parametrize("cls", [nx.MultiGraph, nx.MultiDiGraph])
def test_multiedges_and_keys(cls):
    G = cls()
    G.add_edge(1, 2, key="a", weight=1)
    G.add_edge(1, 2, key="b", weight=2)
    G.add_edge(2, 1, key=7)
    G.add_edge(3, 3)
    G.add_edge(3, 3)
    E, H = roundtrip(G)
    assert E.number_of_edges() == 5
    assert E.rust_graph().num_edges() == 5
    assert_identical(G, H)
    if not G.is_directed():
        # Both ends of an undirected multiedge share the same keydict, as in NetworkX
        assert H._adj[1][2] is H._adj[2][1]


def test_undirected_ends_share_attribute_dict():
    G = nx.Graph([(1, 2, {"w": 1})])
    _, H = roundtrip(G)
    assert H._adj[1][2] is H._adj[2][1]
    H[1][2]["w"] = 5
    assert H[2][1]["w"] == 5
    assert G[1][2]["w"] == 1  # the original graph is not aliased


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_requested_edge_attrs_with_defaults(cls):
    G = cls()
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 2)
    G.add_edge(2, 0, weight=0.5, other="x")
    # Default 1: missing weights become 1, other attributes are dropped
    E = from_networkx(G, edge_attrs={"weight": 1})
    H = to_networkx(E)
    assert sorted(H.edges(data="weight")) == sorted(G.edges(data="weight", default=1))
    assert all(set(d) == {"weight"} for *_, d in H.edges(data=True))
    # Default None: missing weights stay missing
    E = from_networkx(G, edge_attrs={"weight": None})
    assert MISSING in E.edge_values["weight"]
    H = to_networkx(E)
    assert sorted(H.edges(data="weight")) == sorted(G.edges(data="weight"))
    assert not H.has_edge(1, 2) or "weight" not in (
        H[1][2] if not H.is_multigraph() else H[1][2][0]
    )


def test_no_attrs_requested():
    G = mixed_key_graph(nx.Graph)
    G.graph["name"] = "dropped"
    E = from_networkx(G)
    H = to_networkx(E)
    assert H.graph == {}
    assert all(not d for _, d in H.nodes(data=True))
    assert all(not d for *_, d in H.edges(data=True))
    assert sorted(map(str, H.edges)) == sorted(map(str, G.edges))


def test_node_attrs_with_none_default():
    G = nx.Graph()
    G.add_node("a", color="red", size=1)
    G.add_node("b")
    E = from_networkx(G, node_attrs={"color": None, "size": 0})
    assert E.node_values_of("color") == ["red", None]
    assert E.node_values_of("size") == [1, 0]
    H = to_networkx(E)
    assert dict(H.nodes(data=True)) == {"a": {"color": "red", "size": 1}, "b": {"size": 0}}


def test_rust_graph_weights():
    G = nx.Graph()
    G.add_edge("a", "b", weight=2.5)
    G.add_edge("b", "c")
    G.add_edge("c", "c", weight=-1)
    E = from_networkx(G, edge_attrs={"weight": None})
    rg = E.rust_graph("weight", default=1.0)
    assert rg.num_nodes() == 3
    assert rg.num_edges() == 3
    weights = sorted(w for *_, w in rg._weighted_edges())
    assert weights == [-1.0, 1.0, 2.5]
    # Missing weights may also stay unset in the Rust graph
    rg_none = E.rust_graph("weight", default=None)
    assert sorted((w is None, w or 0) for *_, w in rg_none._weighted_edges()) == [
        (False, -1.0),
        (False, 2.5),
        (True, 0),
    ]
    # Each (weight, default) pair is built once
    assert E.rust_graph("weight", default=1.0) is rg


def test_rust_graph_rejects_missing_attribute():
    E = from_networkx(nx.path_graph(3))
    with pytest.raises(KeyError, match="weight"):
        E.rust_graph("weight")


def test_edge_arcs_map_rust_edges_to_arcs():
    G = nx.cycle_graph(4)
    G.add_edge(2, 2)
    E = from_networkx(G)
    rg = E.rust_graph()
    for edge_id, source, target in rg.edges():
        arc = E.edge_arcs[edge_id - 1]
        assert (E.src[arc], E.dst[arc]) == (source, target)


def test_infinite_and_bool_weights():
    G = nx.DiGraph()
    G.add_edge(0, 1, capacity=math.inf)
    G.add_edge(1, 2, capacity=True)
    rg = from_networkx(G, edge_attrs={"capacity": 1}).rust_graph("capacity")
    assert sorted(w for *_, w in rg._weighted_edges()) == [1.0, math.inf]


def test_nan_weight_is_rejected():
    G = nx.Graph([(0, 1, {"weight": math.nan})])
    E = from_networkx(G, edge_attrs={"weight": 1})
    with pytest.raises(ValueError, match="NaN"):
        E.rust_graph("weight")


def test_graph_views_and_subclasses():
    G = nx.path_graph(5)
    view = nx.subgraph_view(G, filter_node=lambda n: n != 2)
    _, H = roundtrip(view)
    assert sorted(H.edges) == [(0, 1), (3, 4)]
    D = nx.DiGraph([(0, 1), (1, 2)])
    _, H = roundtrip(nx.reverse_view(D))
    assert sorted(H.edges) == [(1, 0), (2, 1)]
    P = nx.PlanarEmbedding()
    P.set_data({0: [1], 1: [0]})
    _, H = roundtrip(P)
    assert type(H) is nx.PlanarEmbedding
    assert H.get_data() == P.get_data()


def test_graph_presenting_a_different_adjacency():
    # `_AntiGraph` presents the complement of the graph it stores; convert what it presents
    from networkx.algorithms.approximation.kcomponents import _AntiGraph

    A = _AntiGraph()
    A.add_nodes_from(range(4))
    A.add_edge(0, 1)
    E = from_networkx(A)
    H = to_networkx(E)
    assert type(H) is nx.Graph
    assert sorted(map(sorted, H.edges)) == [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def test_node_view_and_bad_input():
    G = nx.Graph()
    G.add_node(1, a=2)
    E = from_networkx(G.nodes, preserve_node_attrs=True)
    assert E.nodes == [1]
    assert E.node_dicts == [{"a": 2}]
    with pytest.raises(TypeError, match="Bad type"):
        from_networkx([1, 2], name="f", graph_name="G")


def test_backend_interface_conversions():
    from edgecase.nx_backend.interface import backend_interface

    G = nx.path_graph(3)
    E = backend_interface.convert_from_nx(G, preserve_edge_attrs=True)
    assert backend_interface.convert_to_nx(E).edges == G.edges
    assert backend_interface.convert_to_nx(5) == 5
    pair = backend_interface.convert_to_nx((E, 1))
    assert isinstance(pair[0], nx.Graph) and pair[1] == 1


def test_backend_info_is_light_and_in_sync():
    import subprocess
    import sys

    code = (
        "import sys, networkx as nx; "
        "info = nx.utils.backends.backend_info['edgecase']; "
        "assert info['backend_name'] == 'edgecase'; "
        "assert 'edgecase' not in sys.modules and 'polars' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    from _nx_edgecase import get_info
    from edgecase.nx_backend.interface import BackendInterface
    from edgecase.nx_backend.registry import algorithms

    assert set(get_info()["functions"]) == set(algorithms), (
        "update _FUNCTIONS in python/_nx_edgecase/__init__.py"
    )
    for name in algorithms:
        assert hasattr(BackendInterface, name)
