from edgecase import Graph


def test_to_undirected_ndf_snapshot():
    """Node DataFrame structure is preserved after to_undirected conversion."""
    g = Graph(directed=True)
    g.add_node(label="A", color="red", value=10)
    g.add_node(label="B", color="blue", value=20)
    g.add_edge("A", "B")

    undirected = g.to_undirected()

    ndf = undirected.get_node_df()

    assert ndf.shape[0] == 2  # Two nodes
    assert "label" in ndf.columns
    assert "color" in ndf.columns
    assert "value" in ndf.columns


def test_to_undirected_edf_snapshot():
    """Edge DataFrame structure is preserved after to_undirected conversion."""
    g = Graph(directed=True, weighted=True)
    g.add_node(label="A")
    g.add_node(label="B")
    g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

    undirected = g.to_undirected()

    edf = undirected.get_edge_df()

    assert edf.shape[0] == 1  # One edge
    assert "label" in edf.columns
    assert "type" in edf.columns
    assert "weight" in edf.columns


def test_to_directed_ndf_snapshot():
    """Node DataFrame structure is preserved after to_directed conversion."""
    g = Graph(directed=False)
    g.add_node(label="A", color="red", value=10)
    g.add_node(label="B", color="blue", value=20)
    g.add_edge("A", "B")

    directed = g.to_directed()

    ndf = directed.get_node_df()

    assert ndf.shape[0] == 2  # Two nodes
    assert "label" in ndf.columns
    assert "color" in ndf.columns
    assert "value" in ndf.columns


def test_to_directed_edf_snapshot():
    """Edge DataFrame structure is preserved after to_directed conversion."""
    g = Graph(directed=False, weighted=True)
    g.add_node(label="A")
    g.add_node(label="B")
    g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

    directed = g.to_directed()

    edf = directed.get_edge_df()

    assert edf.shape[0] == 2  # Two edges (bidirectional)
    assert "label" in edf.columns
    assert "type" in edf.columns
    assert "weight" in edf.columns


def test_reverse_ndf_snapshot():
    """Node DataFrame structure is preserved after reverse conversion."""
    g = Graph(directed=True)
    g.add_node(label="A", color="red", value=10)
    g.add_node(label="B", color="blue", value=20)
    g.add_edge("A", "B")

    reversed_g = g.reverse()

    ndf = reversed_g.get_node_df()

    assert ndf.shape[0] == 2  # Two nodes
    assert "label" in ndf.columns
    assert "color" in ndf.columns
    assert "value" in ndf.columns


def test_reverse_edf_snapshot():
    """Edge DataFrame structure is preserved after reverse conversion."""
    g = Graph(directed=True, weighted=True)
    g.add_node(label="A")
    g.add_node(label="B")
    g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

    reversed_g = g.reverse()

    edf = reversed_g.get_edge_df()

    assert edf.shape[0] == 1  # One edge
    assert "label" in edf.columns
    assert "type" in edf.columns
    assert "weight" in edf.columns


def test_node_attributes_via_api():
    """Node attributes are accessible via get_node() API after conversion."""
    g = Graph(directed=True, weighted=True)
    g.add_node(label="A", color="red", value=10)
    g.add_node(label="B", color="blue", value=20)
    g.add_edge("A", "B", weight=1.5, label="edge1", type="strong")

    # Test original graph
    node_a = g.get_node("A")

    assert node_a is not None
    assert node_a.label == "A"
    assert node_a.attributes["color"] == "red"
    assert node_a.attributes["value"] == 10

    # Test after conversion
    undirected = g.to_undirected()
    node_a_conv = undirected.get_node("A")

    assert node_a_conv is not None
    assert node_a_conv.label == "A"
    assert node_a_conv.attributes["color"] == "red"
    assert node_a_conv.attributes["value"] == 10


def test_attributes_preserved_across_conversions():
    """Attributes are preserved before and after conversion."""
    g = Graph(directed=True)
    g.add_node(label="A", color="red", value=10)
    g.add_node(label="B", color="blue", value=20)
    g.add_edge("A", "B", weight=1.5)

    # Get original attributes
    node_orig = g.get_node("A")

    assert node_orig is not None

    orig_attrs = node_orig.attributes

    # Convert and check attributes preserved
    undirected = g.to_undirected()
    node_conv = undirected.get_node("A")

    assert node_conv is not None

    conv_attrs = node_conv.attributes

    # Verify attributes match
    assert conv_attrs["color"] == orig_attrs["color"]
    assert conv_attrs["value"] == orig_attrs["value"]
