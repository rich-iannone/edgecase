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
        node_ids = G.get_node_ids(limit=None)
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
    assert G.count_nodes() >= 0


@given(G=small_graphs())
@settings(max_examples=100)
def test_edge_count_nonnegative(G):
    """Edge count should always be non-negative"""
    assert G.count_edges() >= 0


@given(G=small_graphs())
@settings(max_examples=100)
def test_nodes_list_length(G):
    """len(nodes()) should equal num_nodes()"""
    assert len(G.get_node_ids(limit=None)) == G.count_nodes()


@given(G=small_graphs())
@settings(max_examples=100)
def test_edges_list_length(G):
    """len(edges_as_tuples()) should equal num_edges()"""
    assert len(G.get_edges_as_tuples()) == G.count_edges()


@given(G=small_graphs())
@settings(max_examples=100)
def test_add_node_increases_count(G):
    """Adding a new node should increase count by 1"""
    initial = G.count_nodes()
    G.add_node("new_unique_node_xyz")
    assert G.count_nodes() == initial + 1


@given(G=small_graphs())
@settings(max_examples=100)
def test_add_duplicate_node_no_change(G):
    """Adding duplicate node with same label should create a new node (different ID)"""
    if G.count_nodes() > 0:
        node_id = G.get_node_ids(limit=None)[0]
        label = G.get_label_from_id(node_id)
        initial = G.count_nodes()
        # Adding node with same label creates a new node (IDs are auto-managed)
        G.add_node(label=label)
        # Should create a new node since IDs are different
        assert G.count_nodes() == initial + 1


# Undirected graph property tests


@given(G=small_graphs(directed=False))
@settings(max_examples=100)
def test_undirected_neighbors_symmetric(G):
    """In undirected graph, if B is neighbor of A, then A is neighbor of B"""
    if G.count_nodes() > 0:
        for node in G.get_node_ids(limit=None):
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
    if G.count_nodes() > 0:
        ranks = G.pagerank(return_dict=True)

        if ranks:  # If PageRank succeeded
            total = sum(ranks.values())
            assert abs(total - 1.0) < 1e-4


@given(G=small_graphs())
@settings(max_examples=50, deadline=1000)
def test_pagerank_all_positive(G):
    """All PageRank scores should be positive"""
    if G.count_nodes() > 0:
        ranks = G.pagerank(return_dict=True)

        if ranks:
            for score in ranks.values():
                assert score > 0


@given(G=small_graphs())
@settings(max_examples=50, deadline=1000)
def test_pagerank_covers_all_nodes(G):
    """PageRank should return scores for all nodes"""
    if G.count_nodes() > 0:
        ranks = G.pagerank(return_dict=True)

        if ranks:
            assert len(ranks) == G.count_nodes()
            # PageRank returns dict with node IDs as keys
            node_ids = set(G.get_node_ids(limit=None))
            assert set(ranks.keys()) == node_ids


# Shortest path property tests


@given(G=small_graphs(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=50, deadline=1000)
def test_shortest_path_starts_and_ends_correctly(G, seed):
    """Shortest path should start at source and end at target"""
    if G.count_nodes() >= 2:
        nodes = G.get_node_ids(limit=None)
        source = nodes[seed % len(nodes)]
        target = nodes[(seed + 1) % len(nodes)]

        path = G.shortest_path(source, target, return_list=True)

        if path and len(path) > 0:
            # Path now returns node IDs
            assert path[0] == source
            assert path[-1] == target


@given(G=small_graphs(), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=50, deadline=1000)
def test_shortest_path_consecutive_edges_exist(G, seed):
    """All consecutive nodes in shortest path should be connected"""
    if G.count_nodes() >= 2:
        nodes = G.get_node_ids(limit=None)
        source = nodes[seed % len(nodes)]
        target = nodes[(seed + 1) % len(nodes)]

        path = G.shortest_path(source, target, return_list=True)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                # Path contains node IDs - use them directly
                current_id = path[i]
                # Get neighbor IDs
                neighbor_ids = G.neighbors(current_id)
                # Next node should be in current node's neighbors
                next_id = path[i + 1]
                assert next_id in neighbor_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
