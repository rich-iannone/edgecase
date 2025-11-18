import warnings

import networkx as nx
import pytest

import edgecase as ec

warnings.filterwarnings("ignore")

# Import igraph if available
try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


def convert_to_networkx(ec_graph):
    """Convert EdgeCase graph to NetworkX.

    Note: EdgeCase uses 1-indexed node IDs, NetworkX uses 0-indexed.
    We keep the original node IDs for consistency in traversal comparison.
    """
    is_directed = ec_graph.is_directed()
    nx_graph = nx.DiGraph() if is_directed else nx.Graph()

    for node_id in ec_graph.get_node_ids(limit=None):
        nx_graph.add_node(node_id)

    for edge_id, source, target in ec_graph.get_edges_as_tuples():
        nx_graph.add_edge(source, target)

    return nx_graph


def convert_to_igraph(ec_graph):
    """Convert EdgeCase graph to igraph.

    Note: EdgeCase uses 1-indexed node IDs, igraph uses 0-indexed.
    We create a mapping between EdgeCase node IDs and igraph indices.
    """
    if not IGRAPH_AVAILABLE:
        return None

    node_ids = sorted(ec_graph.get_node_ids(limit=None))  # Sort for consistent ordering
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edges = []
    for edge_id, source, target in ec_graph.get_edges_as_tuples():
        edges.append((node_to_idx[source], node_to_idx[target]))

    ig_graph = ig.Graph(n=len(node_ids), edges=edges, directed=ec_graph.is_directed())
    return ig_graph, node_ids  # Return mapping for index conversion


def create_test_graphs():
    """Create diverse graph types for validation."""
    graphs = []

    # 1. Path graph
    g = ec.Graph()
    g.add_path_graph(10)
    graphs.append(("path_10", g))

    # 2. Karate club (standard dataset)
    g = ec.load_dataset("karate")
    graphs.append(("karate", g))

    # 3. Star graph
    g = ec.Graph()
    g.add_star_graph(20)
    graphs.append(("star_20", g))

    # 4. Cycle graph
    g = ec.Graph()
    g.add_cycle_graph(15)
    graphs.append(("cycle_15", g))

    # 5. Complete graph
    g = ec.Graph()
    g.add_full_graph(8)
    graphs.append(("complete_8", g))

    # 6. Grid 2D
    g = ec.Graph()
    g.add_grid_2d(rows=4, cols=4)
    graphs.append(("grid_4x4", g))

    # 7. Random sparse graph
    g = ec.Graph()
    g.add_gnp_graph(n=30, p=0.1, seed=42)
    graphs.append(("random_sparse_30", g))

    # 8. Random dense graph
    g = ec.Graph()
    g.add_gnp_graph(n=20, p=0.4, seed=42)
    graphs.append(("random_dense_20", g))

    # 9. Small world
    g = ec.Graph()
    g.add_smallworld_graph(n=25, k=4, p=0.2, seed=42)
    graphs.append(("smallworld_25", g))

    # 10. Scale-free
    g = ec.Graph()
    g.add_pa_graph(n=30, m=2, seed=42)
    graphs.append(("scalefree_30", g))

    return graphs


@pytest.mark.parametrize("graph_name,ec_graph", create_test_graphs())
def test_bfs_vs_networkx(graph_name, ec_graph):
    """Test that BFS visits the same nodes as NetworkX."""
    nx_graph = convert_to_networkx(ec_graph)

    # Get first node
    start_node = ec_graph.get_node_ids(limit=1)[0]

    # EdgeCase BFS
    ec_result = ec_graph.bfs(start_node, return_list=True)

    # NetworkX BFS
    nx_result = list(nx.bfs_tree(nx_graph, start_node).nodes())

    # Should visit exactly the same nodes (order may differ)
    assert set(ec_result) == set(nx_result), (
        f"{graph_name}: EdgeCase visited {set(ec_result)}, NetworkX visited {set(nx_result)}"
    )

    # Same count
    assert len(ec_result) == len(nx_result), (
        f"{graph_name}: EdgeCase visited {len(ec_result)} nodes, "
        f"NetworkX visited {len(nx_result)} nodes"
    )

    # First node should be start node
    assert ec_result[0] == start_node
    assert nx_result[0] == start_node


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not installed")
@pytest.mark.parametrize("graph_name,ec_graph", create_test_graphs())
def test_bfs_vs_igraph(graph_name, ec_graph):
    """Test that BFS visits the same nodes as igraph."""
    ig_graph, node_ids = convert_to_igraph(ec_graph)
    idx_to_node = {idx: nid for idx, nid in enumerate(node_ids)}

    # Get first node
    start_node = ec_graph.get_node_ids(limit=1)[0]
    start_idx = node_ids.index(start_node)

    # EdgeCase BFS
    ec_result = ec_graph.bfs(start_node, return_list=True)

    # igraph BFS (returns indices)
    ig_result_indices = ig_graph.bfs(start_idx)[0]
    ig_result = [idx_to_node[idx] for idx in ig_result_indices]

    # Should visit exactly the same nodes (order may differ)
    assert set(ec_result) == set(ig_result), (
        f"{graph_name}: EdgeCase visited {set(ec_result)}, igraph visited {set(ig_result)}"
    )

    # Same count
    assert len(ec_result) == len(ig_result), (
        f"{graph_name}: EdgeCase visited {len(ec_result)} nodes, "
        f"igraph visited {len(ig_result)} nodes"
    )

    # First node should be start node
    assert ec_result[0] == start_node
    assert ig_result[0] == start_node


@pytest.mark.parametrize("graph_name,ec_graph", create_test_graphs())
def test_dfs_vs_networkx(graph_name, ec_graph):
    """Test that DFS visits the same nodes as NetworkX."""
    nx_graph = convert_to_networkx(ec_graph)

    # Get first node
    start_node = ec_graph.get_node_ids(limit=1)[0]

    # EdgeCase DFS
    ec_result = ec_graph.dfs(start_node, return_list=True)

    # NetworkX DFS
    nx_result = list(nx.dfs_preorder_nodes(nx_graph, start_node))

    # Should visit exactly the same nodes (order may differ)
    assert set(ec_result) == set(nx_result), (
        f"{graph_name}: EdgeCase visited {set(ec_result)}, NetworkX visited {set(nx_result)}"
    )

    # Same count
    assert len(ec_result) == len(nx_result), (
        f"{graph_name}: EdgeCase visited {len(ec_result)} nodes, "
        f"NetworkX visited {len(nx_result)} nodes"
    )

    # First node should be start node
    assert ec_result[0] == start_node
    assert nx_result[0] == start_node


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not installed")
@pytest.mark.parametrize("graph_name,ec_graph", create_test_graphs())
def test_dfs_vs_igraph(graph_name, ec_graph):
    """Test that DFS visits the same nodes as igraph."""
    ig_graph, node_ids = convert_to_igraph(ec_graph)
    idx_to_node = {idx: nid for idx, nid in enumerate(node_ids)}

    # Get first node
    start_node = ec_graph.get_node_ids(limit=1)[0]
    start_idx = node_ids.index(start_node)

    # EdgeCase DFS
    ec_result = ec_graph.dfs(start_node, return_list=True)

    # igraph DFS (returns indices)
    ig_result_indices = ig_graph.dfs(start_idx)[0]
    ig_result = [idx_to_node[idx] for idx in ig_result_indices]

    # Should visit exactly the same nodes (order may differ)
    assert set(ec_result) == set(ig_result), (
        f"{graph_name}: EdgeCase visited {set(ec_result)}, igraph visited {set(ig_result)}"
    )

    # Same count
    assert len(ec_result) == len(ig_result), (
        f"{graph_name}: EdgeCase visited {len(ec_result)} nodes, "
        f"igraph visited {len(ig_result)} nodes"
    )

    # First node should be start node
    assert ec_result[0] == start_node
    assert ig_result[0] == start_node


@pytest.mark.parametrize("graph_name,ec_graph", create_test_graphs())
def test_bfs_dfs_same_nodes(graph_name, ec_graph):
    """Test that BFS and DFS visit the same nodes (just different order)."""
    # Get first node
    start_node = ec_graph.get_node_ids(limit=1)[0]

    bfs_result = ec_graph.bfs(start_node, return_list=True)
    dfs_result = ec_graph.dfs(start_node, return_list=True)

    # Same nodes visited
    assert set(bfs_result) == set(dfs_result), (
        f"{graph_name}: BFS visited {set(bfs_result)}, DFS visited {set(dfs_result)}"
    )

    # Same count
    assert len(bfs_result) == len(dfs_result), (
        f"{graph_name}: BFS visited {len(bfs_result)} nodes, DFS visited {len(dfs_result)} nodes"
    )

    # Both start at same node
    assert bfs_result[0] == start_node
    assert dfs_result[0] == start_node


def test_bfs_visits_all_nodes():
    """Test that BFS visits all nodes in connected graph."""
    g = ec.load_dataset("karate")

    result = g.bfs(1, return_list=True)

    # Should visit all 34 nodes
    assert len(result) == 34
    # All nodes should be unique
    assert len(set(result)) == 34
    # Should start at node 1
    assert result[0] == 1


def test_dfs_visits_all_nodes():
    """Test that DFS visits all nodes in connected graph."""
    g = ec.load_dataset("karate")

    result = g.dfs(1, return_list=True)

    # Should visit all 34 nodes
    assert len(result) == 34
    # All nodes should be unique
    assert len(set(result)) == 34
    # Should start at node 1
    assert result[0] == 1


def test_disconnected_graph_only_visits_component():
    """Test that BFS/DFS only visit connected component."""
    g = ec.Graph()
    # Component 1
    g.add_path_graph(3)
    # Component 2
    g.add_node(10)
    g.add_node(11)
    g.add_edge(10, 11)

    bfs_result = g.bfs(1, return_list=True)
    dfs_result = g.dfs(1, return_list=True)

    # Both should only visit component containing node 1
    assert set(bfs_result) == {1, 2, 3}
    assert set(dfs_result) == {1, 2, 3}

    # Now start from component 2
    bfs_result = g.bfs(10, return_list=True)
    dfs_result = g.dfs(10, return_list=True)

    assert set(bfs_result) == {10, 11}
    assert set(dfs_result) == {10, 11}
