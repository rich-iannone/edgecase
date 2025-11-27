import pytest
from edgecase import Graph


def test_second_order_centrality_path_graph():
    """Second order centrality on path graph (as in reference paper)."""
    graph = Graph()
    for i in range(3):
        graph.add_node(label=str(i))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")

    soc = graph.second_order_centrality(return_dict=True)

    # Get node IDs
    id_0 = graph.get_id_from_label("0")
    id_1 = graph.get_id_from_label("1")
    id_2 = graph.get_id_from_label("2")

    # Middle node should have lowest (best) centrality
    assert soc[id_1] < soc[id_0]
    assert soc[id_1] < soc[id_2]

    # End nodes should have similar centrality
    assert abs(soc[id_0] - soc[id_2]) < 0.1


def test_second_order_centrality_complete_graph():
    """All nodes in complete graph should have equal centrality."""
    graph = Graph()
    for i in range(4):
        graph.add_node(label=str(i))

    # Add all possible edges
    for i in range(4):
        for j in range(i + 1, 4):
            graph.add_edge(str(i), str(j))

    soc = graph.second_order_centrality(return_dict=True)

    # All centralities should be equal (within numerical precision)
    values = list(soc.values())
    for v in values:
        assert abs(v - values[0]) < 0.01


def test_second_order_centrality_star_graph():
    """Center node in star graph should have lowest centrality."""
    graph = Graph()
    # Create center node
    graph.add_node(label="center")
    # Create leaf nodes
    for i in range(5):
        graph.add_node(label=f"leaf_{i}")
        graph.add_edge("center", f"leaf_{i}")

    soc = graph.second_order_centrality(return_dict=True)

    # Get node IDs
    center_id = graph.get_id_from_label("center")
    leaf_ids = [graph.get_id_from_label(f"leaf_{i}") for i in range(5)]

    # Center should have lowest (best) centrality
    center_val = soc[center_id]
    for leaf_id in leaf_ids:
        assert center_val < soc[leaf_id]

    # All leaves should have similar centrality
    leaf_vals = [soc[leaf_id] for leaf_id in leaf_ids]
    for v in leaf_vals:
        assert abs(v - leaf_vals[0]) < 0.1


def test_second_order_centrality_cycle_graph():
    """Ring graph - all nodes should have equal centrality."""
    graph = Graph()
    n = 5
    for i in range(n):
        graph.add_node(label=str(i))

    for i in range(n):
        graph.add_edge(str(i), str((i + 1) % n))

    soc = graph.second_order_centrality(return_dict=True)

    # All nodes should have equal centrality
    values = list(soc.values())
    for v in values:
        assert abs(v - values[0]) < 0.01


def test_second_order_centrality_directed_graph_error():
    """Should raise error for directed graphs."""
    graph = Graph(directed=True)
    graph.add_node(label="A")
    graph.add_node(label="B")
    graph.add_edge("A", "B")

    with pytest.raises(ValueError, match="undirected"):
        graph.second_order_centrality(return_dict=True)


def test_second_order_centrality_empty_graph_error():
    """Should raise error for empty graph."""
    graph = Graph()

    with pytest.raises(ValueError, match="Empty"):
        graph.second_order_centrality(return_dict=True)


def test_second_order_centrality_disconnected_graph_error():
    """Should raise error for disconnected graph."""
    graph = Graph()
    graph.add_node(label="0")
    graph.add_node(label="1")
    graph.add_node(label="2")

    # Only connect two nodes (node 2 is disconnected)
    graph.add_edge("0", "1")

    with pytest.raises(ValueError, match="not connected"):
        graph.second_order_centrality(return_dict=True)


def test_second_order_centrality_single_node_with_self_loop():
    """Single node with self-loop."""
    graph = Graph()
    graph.add_node(label="A")
    graph.add_edge("A", "A")

    soc = graph.second_order_centrality(return_dict=True)

    node_id = graph.get_id_from_label("A")
    assert node_id in soc
    assert soc[node_id] == 0.0  # Single node should have centrality 0


def test_second_order_centrality_two_nodes():
    """Two connected nodes."""
    graph = Graph()
    graph.add_node(label="A")
    graph.add_node(label="B")
    graph.add_edge("A", "B")

    soc = graph.second_order_centrality(return_dict=True)

    # Get node IDs
    id_a = graph.get_id_from_label("A")
    id_b = graph.get_id_from_label("B")

    # Both nodes should have equal centrality
    assert abs(soc[id_a] - soc[id_b]) < 0.01


def test_second_order_centrality_store_in_graph():
    """Test storing results in graph without returning dict."""
    graph = Graph()
    for i in range(4):
        graph.add_node(label=str(i))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")
    graph.add_edge("2", "3")

    result = graph.second_order_centrality(attr_name="soc")

    # Should return self
    assert result is graph

    # Should have stored computation
    assert any("second_order_centrality" in str(c) for c in graph._computations)
    assert any("soc" in str(c) for c in graph._computations)


def test_second_order_centrality_custom_attr_name():
    """Test using custom attribute name."""
    graph = Graph()
    for i in range(3):
        graph.add_node(label=str(i))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")

    graph.second_order_centrality(attr_name="my_centrality")

    # Check custom attribute name is used in computations
    assert any("my_centrality" in str(c) for c in graph._computations)


def test_second_order_centrality_chaining():
    """Test method chaining."""
    graph = Graph()
    for i in range(4):
        graph.add_node(label=str(i))

    result = (
        graph.add_edge("0", "1")
        .add_edge("1", "2")
        .add_edge("2", "3")
        .second_order_centrality(attr_name="soc")
    )

    assert result is graph
    assert any("soc" in str(c) for c in graph._computations)


def test_second_order_centrality_barbell_graph():
    """Barbell graph - bridge node should have lower centrality."""
    graph = Graph()

    # Left clique
    for i in range(3):
        graph.add_node(label=f"L{i}")
    graph.add_edge("L0", "L1")
    graph.add_edge("L1", "L2")
    graph.add_edge("L0", "L2")

    # Right clique
    for i in range(3):
        graph.add_node(label=f"R{i}")
    graph.add_edge("R0", "R1")
    graph.add_edge("R1", "R2")
    graph.add_edge("R0", "R2")

    # Bridge
    graph.add_edge("L0", "R0")

    soc = graph.second_order_centrality(return_dict=True)

    # Get node IDs
    l0_id = graph.get_id_from_label("L0")
    l1_id = graph.get_id_from_label("L1")
    r0_id = graph.get_id_from_label("R0")
    r1_id = graph.get_id_from_label("R1")

    # Bridge nodes should have lower centrality than other clique members
    assert soc[l0_id] < soc[l1_id]
    assert soc[r0_id] < soc[r1_id]


def test_second_order_centrality_numerical_stability():
    """Test numerical stability with larger graph."""
    graph = Graph()
    n = 10

    # Create a grid-like structure
    for i in range(n):
        graph.add_node(label=str(i))

    # Connect in a ring
    for i in range(n):
        graph.add_edge(str(i), str((i + 1) % n))

    # Add some cross connections
    for i in range(0, n, 2):
        graph.add_edge(str(i), str((i + 2) % n))

    soc = graph.second_order_centrality(return_dict=True)

    # All values should be positive and finite
    for v in soc.values():
        assert v > 0
        assert not (v != v)  # Check for NaN
        assert v < float("inf")


def test_second_order_centrality_values_match_paper():
    """Test specific values from reference paper (Table 1)."""
    # Path graph P3: Expected values approximately [3.741, 1.414, 3.741]
    graph = Graph()
    for i in range(3):
        graph.add_node(label=str(i))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")

    soc = graph.second_order_centrality(return_dict=True)

    # Get node IDs
    id_0 = graph.get_id_from_label("0")
    id_1 = graph.get_id_from_label("1")
    id_2 = graph.get_id_from_label("2")

    # Check values are in expected range (allowing some numerical variation)
    assert 3.6 < soc[id_0] < 3.9
    assert 1.3 < soc[id_1] < 1.6
    assert 3.6 < soc[id_2] < 3.9

    # Complete graph K3: Expected values approximately [1.414, 1.414, 1.414]
    graph2 = Graph()
    for i in range(3):
        graph2.add_node(label=str(i))
    graph2.add_edge("0", "1")
    graph2.add_edge("1", "2")
    graph2.add_edge("0", "2")

    soc2 = graph2.second_order_centrality(return_dict=True)

    for v in soc2.values():
        assert 1.3 < v < 1.6


def test_second_order_centrality_return_types():
    """Test return types are correct."""
    graph = Graph()
    for i in range(3):
        graph.add_node(label=str(i))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")

    # With return_dict=True
    result_dict = graph.second_order_centrality(return_dict=True)
    assert isinstance(result_dict, dict)
    assert all(isinstance(k, int) for k in result_dict.keys())  # Keys should be node IDs (int)
    assert all(isinstance(v, float) for v in result_dict.values())

    # With return_dict=False (default)
    result_self = graph.second_order_centrality()

    assert result_self is graph
