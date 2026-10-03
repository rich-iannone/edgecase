"""Graphs returned by transformations, set operations, and generators are fully usable.

These methods build their result in Rust and wrap it with `Graph._from_rust()`. The result must
behave like any other `Graph`: it can be printed and described, can store computed results, and
records steps.
"""

import contextlib
import io

import pytest

import edgecase as ec


def _two_triangles(directed=False):
    g = ec.Graph(directed=directed)
    for label in "ABCDEF":
        g.add_node(label=label)
    for source, target in [
        ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"),
        ("D", "E"), ("E", "F"), ("F", "D"), ("A", "D"),
    ]:
        g.add_edge(source, target)
    return g


def _chain_dag():
    g = ec.Graph(directed=True)
    for label in "abc":
        g.add_node(label=label)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a", "c")
    return g


g = _two_triangles()
other = _two_triangles()
dg = _two_triangles(directed=True)

DERIVED = {
    "configuration_model": lambda: g.configuration_model([2, 2, 2, 2], seed=1),
    "degree_preserving_rewire": lambda: g.degree_preserving_rewire(n_swaps=3, seed=1),
    "expected_degree_graph": lambda: g.expected_degree_graph([1.0, 2.0, 2.0, 1.0], seed=1),
    "generate_stochastic_block_model": lambda: g.generate_stochastic_block_model(
        [3, 3], [0.9, 0.1, 0.1, 0.9], seed=1
    ),
    "to_undirected": lambda: dg.to_undirected(),
    "to_directed": lambda: g.to_directed(),
    "reverse": lambda: dg.reverse(),
    "complement": lambda: g.complement(),
    "line_graph": lambda: g.line_graph(),
    "contract_nodes": lambda: g.contract_nodes("A", "B"),
    "contracted_edge": lambda: g.contracted_edge("A", "B"),
    "contract_nodes_multi": lambda: g.contract_nodes_multi(["A", "B", "C"]),
    "quotient_graph": lambda: g.quotient_graph([["A", "B", "C"], ["D", "E", "F"]]),
    "union": lambda: g.union(other),
    "intersection": lambda: g.intersection(other),
    "difference": lambda: g.difference(other),
    "symmetric_difference": lambda: g.symmetric_difference(other),
    "compose": lambda: g.compose(other),
    "disjoint_union": lambda: g.disjoint_union(other),
    "subgraph": lambda: g.subgraph(sorted(g.get_node_ids())[:4]),
    "edge_subgraph": lambda: g.edge_subgraph(sorted(g.get_edge_ids())[:4]),
    "node_percolation": lambda: g.node_percolation(0.5, seed=1),
    "edge_percolation": lambda: g.edge_percolation(0.5, seed=1),
    "k_core": lambda: g.k_core(2),
    "k_shell": lambda: g.k_shell(2),
    "k_crust": lambda: g.k_crust(2),
    "k_truss": lambda: g.k_truss(3),
    "k_corona": lambda: g.k_corona(2),
    "transitive_closure": lambda: dg.transitive_closure(),
    "transitive_reduction": lambda: _chain_dag().transitive_reduction(),
    "from_adjacency_matrix": lambda: ec.from_adjacency_matrix([[0, 1], [1, 0]]),
}


@pytest.mark.parametrize("make", DERIVED.values(), ids=DERIVED.keys())
def test_derived_graph_is_fully_initialized(make):
    result = make()

    assert isinstance(result, ec.Graph)
    assert result.get_graph_name() is None

    # Printing and describing used to raise AttributeError (missing `_graph_name`)
    repr(result)
    with contextlib.redirect_stdout(io.StringIO()):
        result.describe()

    # Storing a result logs it in the computation history (used to raise on `_computations`)
    result.betweenness_centrality()
    assert any(name == "betweenness_centrality" for name, *_ in result._computations)

    # Step recording works on the new graph
    result.add_node(label="__new_node__")
    assert len(result.steps) >= 1


@pytest.mark.parametrize("directed", [False, True])
def test_from_rust_keeps_directedness_and_weighting(directed):
    source = ec.Graph(directed=directed, weighted=True)
    source.add_node(label="A")
    source.add_node(label="B")
    source.add_edge("A", "B", weight=2.5)

    result = source.subgraph(source.get_node_ids())

    assert result.is_directed() == directed
    assert result.is_weighted()


def test_describe_with_unlabelled_nodes():
    """Unlabelled nodes used to break describe()/repr() via a label lookup of str(node_id)."""
    g = ec.Graph()
    g.add_node()
    g.add_node()
    ids = g.get_node_ids()
    g.add_edge(ids[0], ids[1])

    repr(g)
    info = g._create_describe_info_dict()
    assert info["Average degree (min, max)"] == "1.00 (1, 1)"
