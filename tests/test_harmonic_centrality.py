from edgecase import Graph


def test_harmonic_centrality_empty_graph():
    """Test harmonic centrality on an empty graph."""
    graph = Graph()
    scores = graph.harmonic_centrality(return_dict=True)

    assert len(scores) == 0


def test_harmonic_centrality_single_node():
    """Test harmonic centrality with a single node."""
    graph = Graph().add_node(label="A")
    scores = graph.harmonic_centrality(return_dict=True)
    node_id = graph.get_id_from_label("A")

    assert len(scores) == 1
    assert scores[node_id] == 0.0  # No other nodes to reach


def test_harmonic_centrality_star_graph():
    """Test harmonic centrality on a star graph."""
    graph = (
        Graph()
        .add_node(label="Center")
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("Center", "A")
        .add_edge("Center", "B")
        .add_edge("Center", "C")
    )

    scores = graph.harmonic_centrality(return_dict=True)
    center_id = graph.get_id_from_label("Center")

    # Center should have highest harmonic centrality
    assert scores[center_id] == max(scores.values())

    # Center is 1 hop from 3 nodes: 1/1 + 1/1 + 1/1 = 3
    assert scores[center_id] == 3.0

    # Peripheral nodes should have equal, lower scores
    a_id = graph.get_id_from_label("A")
    b_id = graph.get_id_from_label("B")
    c_id = graph.get_id_from_label("C")

    assert scores[a_id] == scores[b_id] == scores[c_id]

    # Each peripheral node: 1/1 (to center) + 1/2 + 1/2 = 2
    assert abs(scores[a_id] - 2.0) < 1e-6


def test_harmonic_centrality_normalized():
    """Test normalized harmonic centrality."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )

    # Unnormalized
    scores = graph.harmonic_centrality(normalized=False, return_dict=True)
    b_id = graph.get_id_from_label("B")
    unnorm_score = scores[b_id]

    # Normalized
    norm_scores = graph.harmonic_centrality(normalized=True, return_dict=True)
    norm_score = norm_scores[b_id]

    # Normalized should be divided by (n-1) = 2
    assert abs(norm_score - unnorm_score / 2.0) < 1e-6

    # Normalized scores should be in [0, 1]
    assert all(0 <= score <= 1.0 for score in norm_scores.values())


def test_harmonic_centrality_path_graph():
    """Test harmonic centrality on a path graph."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
        .add_edge("D", "E")
    )

    scores = graph.harmonic_centrality(return_dict=True)
    c_id = graph.get_id_from_label("C")

    # Middle node should have highest centrality
    assert scores[c_id] == max(scores.values())

    # End nodes should have lowest centrality
    a_id = graph.get_id_from_label("A")
    e_id = graph.get_id_from_label("E")

    assert scores[a_id] == scores[e_id]
    assert scores[a_id] < scores[c_id]


def test_harmonic_centrality_directed_mode():
    """Test harmonic centrality with different modes on directed graph."""
    graph = (
        Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )

    # Out mode (follow edges forward)
    out_scores = graph.harmonic_centrality(mode="out", return_dict=True)
    a_id = graph.get_id_from_label("A")
    c_id = graph.get_id_from_label("C")

    # A can reach B and C
    assert out_scores[a_id] > out_scores[c_id]

    # C cannot reach anyone
    assert out_scores[c_id] == 0.0

    # In mode (follow edges backward)
    in_scores = graph.harmonic_centrality(mode="in", return_dict=True)

    # C can reach B and A in reverse
    assert in_scores[c_id] > in_scores[a_id]

    # A cannot reach anyone in reverse
    assert in_scores[a_id] == 0.0


def test_harmonic_centrality_cutoff():
    """Test harmonic centrality with cutoff parameter."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
        .add_edge("D", "E")
    )

    # Full centrality
    full_scores = graph.harmonic_centrality(return_dict=True)

    # With cutoff=2 (only paths of length 2 or less)
    cutoff_scores = graph.harmonic_centrality(cutoff=2, return_dict=True)

    # Cutoff scores should be less than or equal to full scores
    for node_id in full_scores:
        assert cutoff_scores[node_id] <= full_scores[node_id]

    # Middle node should still have highest centrality even with cutoff
    c_id = graph.get_id_from_label("C")

    assert cutoff_scores[c_id] == max(cutoff_scores.values())


def test_harmonic_centrality_disconnected():
    """Test harmonic centrality on disconnected graph."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("C", "D")
    )

    scores = graph.harmonic_centrality(return_dict=True)

    # Each component should have equal scores within itself
    a_id = graph.get_id_from_label("A")
    b_id = graph.get_id_from_label("B")
    c_id = graph.get_id_from_label("C")
    d_id = graph.get_id_from_label("D")

    # A and B are in same component
    assert scores[a_id] == scores[b_id]

    # C and D are in same component
    assert scores[c_id] == scores[d_id]

    # All scores should be equal since both components have same structure
    assert scores[a_id] == scores[c_id]


def test_harmonic_centrality_dual_behavior():
    """Test that harmonic centrality has dual behavior (return_dict and attr_name)."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )

    # Test return_dict=True
    scores = graph.harmonic_centrality(return_dict=True)

    assert isinstance(scores, dict)
    assert len(scores) == 3

    # Test dual behavior: should store in NDF and return self
    result = graph.harmonic_centrality(attr_name="harmonic")

    assert result is graph

    ndf = graph.get_node_df()

    assert "harmonic" in ndf.columns
    assert len(ndf) == 3

    # Values in NDF should match return_dict values
    for row in ndf.iter_rows(named=True):
        node_id = row["id"]
        assert abs(row["harmonic"] - scores[node_id]) < 1e-6


def test_harmonic_centrality_auto_attr_name():
    """Test that harmonic centrality auto-generates attribute name."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_edge("A", "B")
        .harmonic_centrality()  # No attr_name provided
    )

    ndf = graph.get_node_df()

    # Should auto-generate "harmonic_centrality" as attribute name
    assert "harmonic_centrality" in ndf.columns


def test_harmonic_centrality_ring_graph():
    """Test harmonic centrality on a ring graph (should all be equal)."""
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
        .add_edge("D", "E")
        .add_edge("E", "A")
    )

    scores = graph.harmonic_centrality(return_dict=True)

    # In a symmetric ring, all nodes should have equal centrality
    unique_scores = set(scores.values())

    assert len(unique_scores) == 1
