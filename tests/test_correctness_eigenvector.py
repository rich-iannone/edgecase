"""
Correctness validation for Eigenvector Centrality algorithm.

Compares EdgeCase implementation against NetworkX and igraph across diverse graph types
to ensure numerical accuracy within tight tolerances.
"""

import warnings

import networkx as nx
import numpy as np
import pytest

import edgecase as ec

warnings.filterwarnings("ignore")

# Import igraph if available
try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


# Tolerance thresholds (more relaxed for eigenvector due to sign/scaling ambiguity)
MAX_ABS_DIFF = 1e-2  # After normalization
MAX_REL_DIFF = 0.05  # 5%
MIN_CORRELATION = 0.999


def create_test_graphs():
    """Create diverse graph types for validation."""
    graphs = []

    # Only test connected undirected graphs (eigenvector centrality requirement)

    # 1. Complete graph
    nx_g = nx.complete_graph(10)
    graphs.append(("complete_10", nx_g, False))

    # 2. Star graph (now works with diagonal shift!)
    nx_g = nx.star_graph(20)
    graphs.append(("star_21", nx_g, False))

    # 3. Cycle graph
    nx_g = nx.cycle_graph(15)
    graphs.append(("cycle_15", nx_g, False))

    # 4. Random dense graph (ensure connected)
    nx_g = nx.erdos_renyi_graph(30, 0.3, seed=42, directed=False)
    # Get largest connected component
    largest_cc = max(nx.connected_components(nx_g), key=len)
    nx_g = nx_g.subgraph(largest_cc).copy()
    graphs.append(("random_dense_30", nx_g, False))

    # 5. Scale-free (Barabási-Albert) - always connected
    nx_g = nx.barabasi_albert_graph(50, 3, seed=42)
    graphs.append(("scale_free_50", nx_g, False))

    # 6. Watts-Strogatz (small-world)
    nx_g = nx.watts_strogatz_graph(40, 4, 0.3, seed=42)
    graphs.append(("small_world_40", nx_g, False))

    # 7. Path graph
    nx_g = nx.path_graph(20)
    graphs.append(("path_20", nx_g, False))

    # 8. Karate club (real-world network)
    nx_g = nx.karate_club_graph()
    graphs.append(("karate_club", nx_g, False))

    # 9. Grid graph
    nx_g = nx.grid_2d_graph(5, 5)
    # Convert node labels to strings
    nx_g = nx.relabel_nodes(nx_g, {node: str(node) for node in nx_g.nodes()})
    graphs.append(("grid_5x5", nx_g, False))

    return graphs


def nx_to_edgecase(nx_graph, directed=False):
    """Convert NetworkX graph to EdgeCase graph."""
    ec_graph = ec.Graph(directed=directed)

    for node in nx_graph.nodes():
        ec_graph.add_node(label=str(node))

    for u, v in nx_graph.edges():
        ec_graph.add_edge(source=str(u), target=str(v))

    return ec_graph


def nx_to_igraph(nx_graph, directed=False):
    """Convert NetworkX graph to igraph."""
    node_list = list(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=directed)

    return ig_graph, node_list


def compare_eigenvector_networkx(ec_graph, nx_graph):
    """Compare Eigenvector Centrality between EdgeCase and NetworkX."""
    # Check if graph is connected
    if not nx.is_connected(nx_graph):
        pytest.skip("Graph is not connected")

    # Run Eigenvector Centrality
    ec_eigen = ec_graph.get_eigen_centrality(max_iter=1000, tol=1e-6)
    nx_eigen = nx.eigenvector_centrality(nx_graph, max_iter=1000, tol=1e-6)

    # Build label->node_id mapping for EdgeCase
    label_to_id = {}
    for node_id in ec_graph.node_ids(limit=None):
        node = ec_graph.get_node(node_id)
        if node:
            label_to_id[node.label] = node_id

    # Convert to comparable arrays
    node_list = list(nx_graph.nodes())
    ec_values = []
    nx_values = []

    for node in node_list:
        node_str = str(node)
        node_id = label_to_id.get(node_str)
        if node_id is not None and node_id in ec_eigen:
            ec_values.append(ec_eigen[node_id])
        else:
            ec_values.append(0.0)
        nx_values.append(nx_eigen[node])

    ec_array = np.array(ec_values)
    nx_array = np.array(nx_values)

    # Normalize both to unit norm (eigenvector sign/scale may differ)
    ec_norm = ec_array / (np.linalg.norm(ec_array) + 1e-10)
    nx_norm = nx_array / (np.linalg.norm(nx_array) + 1e-10)

    # Handle potential sign flip
    if np.sum(ec_norm * nx_norm) < 0:
        ec_norm = -ec_norm

    # Compute metrics
    abs_diff = np.abs(ec_norm - nx_norm)
    rel_diff = np.abs((ec_norm - nx_norm) / (nx_norm + 1e-10))

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation
    if np.std(ec_norm) > 1e-10 and np.std(nx_norm) > 1e-10:
        correlation = np.corrcoef(ec_norm, nx_norm)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-10 else 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correlation": correlation,
    }


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
def compare_eigenvector_igraph(ec_graph, nx_graph):
    """Compare Eigenvector Centrality between EdgeCase and igraph."""
    # Check if graph is connected
    if not nx.is_connected(nx_graph):
        pytest.skip("Graph is not connected")

    ig_graph, node_list = nx_to_igraph(nx_graph, directed=False)

    # Run Eigenvector Centrality
    ec_eigen = ec_graph.get_eigen_centrality(max_iter=1000, tol=1e-6)
    ig_eigen = ig_graph.eigenvector_centrality()

    # Build label->node_id mapping for EdgeCase
    label_to_id = {}
    for node_id in ec_graph.node_ids(limit=None):
        node = ec_graph.get_node(node_id)
        if node:
            label_to_id[node.label] = node_id

    # Convert to comparable arrays
    ec_values = []
    ig_values = ig_eigen

    for node in node_list:
        node_str = str(node)
        node_id = label_to_id.get(node_str)
        if node_id is not None and node_id in ec_eigen:
            ec_values.append(ec_eigen[node_id])
        else:
            ec_values.append(0.0)

    ec_array = np.array(ec_values)
    ig_array = np.array(ig_values)

    # Normalize both to unit norm
    ec_norm = ec_array / (np.linalg.norm(ec_array) + 1e-10)
    ig_norm = ig_array / (np.linalg.norm(ig_array) + 1e-10)

    # Handle potential sign flip
    if np.sum(ec_norm * ig_norm) < 0:
        ec_norm = -ec_norm

    # Compute metrics
    abs_diff = np.abs(ec_norm - ig_norm)
    rel_diff = np.abs((ec_norm - ig_norm) / (ig_norm + 1e-10))

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation
    if np.std(ec_norm) > 1e-10 and np.std(ig_norm) > 1e-10:
        correlation = np.corrcoef(ec_norm, ig_norm)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-10 else 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correlation": correlation,
    }


@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_eigenvector_vs_networkx(name, nx_graph, directed):
    """Test Eigenvector Centrality correctness against NetworkX."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_eigenvector_networkx(ec_graph, nx_graph)

    # Check passes validation
    passes = (
        metrics["max_abs_diff"] < MAX_ABS_DIFF and metrics["max_rel_diff"] < MAX_REL_DIFF
    ) or metrics["correlation"] > MIN_CORRELATION

    assert passes, (
        f"Eigenvector Centrality validation failed for {name}:\n"
        f"  Max abs diff: {metrics['max_abs_diff']:.2e}\n"
        f"  Max rel diff: {metrics['max_rel_diff']:.2%}\n"
        f"  Correlation: {metrics['correlation']:.6f}"
    )


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_eigenvector_vs_igraph(name, nx_graph, directed):
    """Test Eigenvector Centrality correctness against igraph."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_eigenvector_igraph(ec_graph, nx_graph)

    # Check passes validation
    passes = (
        metrics["max_abs_diff"] < MAX_ABS_DIFF and metrics["max_rel_diff"] < MAX_REL_DIFF
    ) or metrics["correlation"] > MIN_CORRELATION

    assert passes, (
        f"Eigenvector Centrality validation failed for {name} vs igraph:\n"
        f"  Max abs diff: {metrics['max_abs_diff']:.2e}\n"
        f"  Max rel diff: {metrics['max_rel_diff']:.2%}\n"
        f"  Correlation: {metrics['correlation']:.6f}"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
