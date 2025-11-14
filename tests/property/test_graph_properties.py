# tests/property/test_graph_properties.py
"""
Property-based tests using Hypothesis
"""

import pytest
from hypothesis import given, strategies as st, settings
from edgecase import Graph


# Custom strategies for graph generation
@st.composite
def node_ids(draw):
    """Generate valid node IDs"""
    return draw(
        st.text(
            min_size=1,
            max_size=10,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=65, max_codepoint=122
            ),
        )
    )


@st.composite
def small_graphs(draw, directed=None):
    """Generate small random graphs using integer node IDs"""
    is_directed = draw(st.booleans()) if directed is None else directed
    is_weighted = draw(st.booleans())

    G = Graph(directed=is_directed, weighted=is_weighted)

    # Generate nodes - use labels so we can reference them, then get their IDs
    num_nodes = draw(st.integers(min_value=0, max_value=20))

    for i in range(num_nodes):
        G.add_node(label=f"n{i}")

    # Generate edges using node IDs
    if num_nodes > 1:
        node_ids = G.node_ids(limit=None)
        num_edges = draw(st.integers(min_value=0, max_value=num_nodes * 2))

        for _ in range(num_edges):
            src = draw(st.sampled_from(node_ids))
            tgt = draw(st.sampled_from(node_ids))

            if is_weighted:
                weight = draw(st.floats(min_value=0.1, max_value=10.0))
                G.add_edge(src, tgt, weight=weight)
            else:
                G.add_edge(src, tgt)

    return G


# Graph invariant tests


@given(G=small_graphs())
@settings(max_examples=100)
def test_node_count_nonnegative(G):
    """Node count should always be non-negative"""
    assert G.num_nodes() >= 0


@given(G=small_graphs())
@settings(max_examples=100)
def test_edge_count_nonnegative(G):
    """Edge count should always be non-negative"""
    assert G.num_edges() >= 0


@given(G=small_graphs())
@settings(max_examples=100)
def test_nodes_list_length(G):
    """len(nodes()) should equal num_nodes()"""
    assert len(G.node_ids(limit=None)) == G.num_nodes()


@given(G=small_graphs())
@settings(max_examples=100)
def test_edges_list_length(G):
    """len(edges_as_tuples()) should equal num_edges()"""
    assert len(G.edges_as_tuples()) == G.num_edges()


@given(G=small_graphs())
@settings(max_examples=100)
def test_add_node_increases_count(G):
    """Adding a new node should increase count by 1"""
    initial = G.num_nodes()
    G.add_node("new_unique_node_xyz")
    assert G.num_nodes() == initial + 1


@given(G=small_graphs())
@settings(max_examples=100)
def test_add_duplicate_node_no_change(G):
    """Adding duplicate node should not change count"""
    if G.num_nodes() > 0:
        node = G.node_ids(limit=None)[0]
        initial = G.num_nodes()
        G.add_node(node)
        assert G.num_nodes() == initial


# Undirected graph property tests


@given(G=small_graphs(directed=False))
@settings(max_examples=100)
def test_undirected_neighbors_symmetric(G):
    """In undirected graph, if B is neighbor of A, then A is neighbor of B"""
    if G.num_nodes() > 0:
        for node in G.node_ids(limit=None):
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                # Check symmetry
                reverse_neighbors = G.neighbors(neighbor)
                assert node in reverse_neighbors


# PageRank property tests


@given(G=small_graphs())
@settings(max_examples=50, deadline=1000)
def test_pagerank_sum_to_one(G):
    """PageRank scores should sum to 1.0"""
    if G.num_nodes() > 0:
        ranks = G.pagerank()

        if ranks:  # If PageRank succeeded
            total = sum(ranks.values())
            assert abs(total - 1.0) < 1e-4


@given(G=small_graphs())
@settings(max_examples=50, deadline=1000)
def test_pagerank_all_positive(G):
    """All PageRank scores should be positive"""
    if G.num_nodes() > 0:
        ranks = G.pagerank()

        if ranks:
            for score in ranks.values():
                assert score > 0


@given(G=small_graphs())
@settings(max_examples=50, deadline=1000)
def test_pagerank_covers_all_nodes(G):
    """PageRank should return scores for all nodes"""
    if G.num_nodes() > 0:
        ranks = G.pagerank()

        if ranks:
            assert len(ranks) == G.num_nodes()
            # PageRank returns dict with node labels as keys
            # Get all node labels
            node_labels = set()
            for node_id in G.node_ids(limit=None):
                label = G.get_label_from_id(node_id)
                node_labels.add(label)
            assert set(ranks.keys()) == node_labels


# Shortest path property tests


@given(G=small_graphs(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=50, deadline=1000)
def test_shortest_path_starts_and_ends_correctly(G, seed):
    """Shortest path should start at source and end at target"""
    if G.num_nodes() >= 2:
        nodes = G.node_ids(limit=None)
        source = nodes[seed % len(nodes)]
        target = nodes[(seed + 1) % len(nodes)]

        path = G.shortest_path(source, target)

        if path and len(path) > 0:
            # Path returns node labels, so convert source/target to labels
            source_label = G.get_label_from_id(source)
            target_label = G.get_label_from_id(target)
            assert path[0] == source_label
            assert path[-1] == target_label


@given(G=small_graphs(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=50, deadline=1000)
def test_shortest_path_consecutive_edges_exist(G, seed):
    """All consecutive nodes in shortest path should be connected"""
    if G.num_nodes() >= 2:
        nodes = G.node_ids(limit=None)
        source = nodes[seed % len(nodes)]
        target = nodes[(seed + 1) % len(nodes)]

        path = G.shortest_path(source, target)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                # Path contains labels, convert current node label to ID
                current_label = path[i]
                current_id = G.get_id_from_label(current_label)
                # Get neighbor IDs
                neighbor_ids = G.neighbors(current_id)
                # Convert next label to ID for comparison
                next_label = path[i + 1]
                next_id = G.get_id_from_label(next_label)
                # Next node should be in current node's neighbors
                assert next_id in neighbor_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
