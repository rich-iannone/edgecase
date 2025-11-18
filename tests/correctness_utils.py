"""
Shared utilities for correctness validation tests.

Provides common functions for graph conversion, comparison, and validation
across EdgeCase, NetworkX, and igraph implementations.
"""

import networkx as nx
import numpy as np

import edgecase as ec

# Import igraph if available
try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


def nx_to_edgecase(nx_graph, directed=False):
    """Convert NetworkX graph to EdgeCase graph.

    Parameters
    ----------
    nx_graph : networkx.Graph
        NetworkX graph to convert
    directed : bool, default=False
        Whether the graph is directed

    Returns
    -------
    edgecase.Graph
        Converted EdgeCase graph
    """
    ec_graph = ec.Graph(directed=directed)

    for node in nx_graph.nodes():
        ec_graph.add_node(label=str(node))

    for u, v in nx_graph.edges():
        ec_graph.add_edge(source=str(u), target=str(v))

    return ec_graph


def nx_to_igraph(nx_graph, directed=False):
    """Convert NetworkX graph to igraph.

    Parameters
    ----------
    nx_graph : networkx.Graph
        NetworkX graph to convert
    directed : bool, default=False
        Whether the graph is directed

    Returns
    -------
    tuple of (igraph.Graph, list)
        Converted igraph graph and list of node labels in order
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph not available")

    node_list = list(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=directed)

    return ig_graph, node_list


def get_edgecase_node_mapping(ec_graph):
    """Build mapping from node labels to internal node IDs.

    Parameters
    ----------
    ec_graph : edgecase.Graph
        EdgeCase graph

    Returns
    -------
    dict
        Mapping from string labels to integer node IDs
    """
    label_to_id = {}
    for node_id in ec_graph.get_node_ids(limit=None):
        node = ec_graph.get_node(node_id)
        if node:
            label_to_id[node.label] = node_id
    return label_to_id


def align_results(ec_result, other_result, node_list, ec_graph, is_dict=True):
    """Align EdgeCase results with NetworkX/igraph results.

    Parameters
    ----------
    ec_result : dict or list
        EdgeCase algorithm output (node_id -> value)
    other_result : dict or list
        NetworkX/igraph algorithm output
    node_list : list
        List of node labels in order
    ec_graph : edgecase.Graph
        EdgeCase graph for mapping
    is_dict : bool, default=True
        Whether other_result is a dict (NetworkX) or list (igraph)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Aligned arrays of EdgeCase and other library values
    """
    label_to_id = get_edgecase_node_mapping(ec_graph)

    ec_values = []
    other_values = []

    for i, node in enumerate(node_list):
        node_str = str(node)
        node_id = label_to_id.get(node_str)

        # Get EdgeCase value
        if node_id is not None and node_id in ec_result:
            ec_values.append(ec_result[node_id])
        else:
            ec_values.append(0.0)

        # Get other library value
        if is_dict:
            other_values.append(other_result[node])
        else:
            other_values.append(other_result[i])

    return np.array(ec_values), np.array(other_values)


def compute_comparison_metrics(ec_array, other_array, normalize=False, allow_sign_flip=False):
    """Compute comparison metrics between two result arrays.

    Parameters
    ----------
    ec_array : np.ndarray
        EdgeCase results
    other_array : np.ndarray
        NetworkX/igraph results
    normalize : bool, default=False
        Whether to normalize to unit L2 norm (for eigenvectors)
    allow_sign_flip : bool, default=False
        Whether to allow sign flip (for eigenvectors)

    Returns
    -------
    dict
        Dictionary with metrics: max_abs_diff, mean_abs_diff, max_rel_diff, correlation
    """
    if normalize:
        ec_norm = ec_array / (np.linalg.norm(ec_array) + 1e-10)
        other_norm = other_array / (np.linalg.norm(other_array) + 1e-10)

        # Handle potential sign flip
        if allow_sign_flip and np.sum(ec_norm * other_norm) < 0:
            ec_norm = -ec_norm

        ec_array = ec_norm
        other_array = other_norm

    # Compute differences
    abs_diff = np.abs(ec_array - other_array)
    rel_diff = np.abs((ec_array - other_array) / (other_array + 1e-10))

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation (handle uniform distributions)
    if np.std(ec_array) > 1e-10 and np.std(other_array) > 1e-10:
        correlation = np.corrcoef(ec_array, other_array)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-10 else 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correlation": correlation,
    }


def check_validation_passes(metrics, max_abs_diff=1e-4, max_rel_diff=0.01, min_correlation=0.9999):
    """Check if validation metrics pass thresholds.

    Parameters
    ----------
    metrics : dict
        Metrics from compute_comparison_metrics
    max_abs_diff : float, default=1e-4
        Maximum absolute difference threshold
    max_rel_diff : float, default=0.01
        Maximum relative difference threshold (1%)
    min_correlation : float, default=0.9999
        Minimum correlation threshold

    Returns
    -------
    bool
        True if validation passes
    """
    return (
        metrics["max_abs_diff"] < max_abs_diff and metrics["max_rel_diff"] < max_rel_diff
    ) or metrics["correlation"] > min_correlation


def format_validation_error(algorithm, graph_name, metrics, vs_library="NetworkX"):
    """Format validation error message.

    Parameters
    ----------
    algorithm : str
        Name of algorithm
    graph_name : str
        Name of graph type
    metrics : dict
        Comparison metrics
    vs_library : str, default="NetworkX"
        Library being compared against

    Returns
    -------
    str
        Formatted error message
    """
    return (
        f"{algorithm} validation failed for {graph_name} vs {vs_library}:\n"
        f"  Max abs diff: {metrics['max_abs_diff']:.2e}\n"
        f"  Mean abs diff: {metrics['mean_abs_diff']:.2e}\n"
        f"  Max rel diff: {metrics['max_rel_diff']:.2%}\n"
        f"  Correlation: {metrics['correlation']:.6f}"
    )
