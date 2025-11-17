"""
Correctness validation for Betweenness Centrality algorithm.

Compares EdgeCase implementation against NetworkX and igraph across diverse graph types
to ensure numerical accuracy within tight tolerances.
"""

import warnings

import networkx as nx
import pytest

import edgecase as ec

from .correctness_utils import (
    IGRAPH_AVAILABLE,
    align_results,
    check_validation_passes,
    compute_comparison_metrics,
    format_validation_error,
    nx_to_edgecase,
    nx_to_igraph,
)

warnings.filterwarnings("ignore")

# Tolerance thresholds (relaxed since betweenness can have larger variance)
MAX_ABS_DIFF = 1e-3
MAX_REL_DIFF = 0.05  # 5%
MIN_CORRELATION = 0.995


def create_test_graphs():
    """Create diverse graph types for validation."""
    graphs = []

    # 1. Complete graph
    nx_g = nx.complete_graph(10)
    graphs.append(("complete_10", nx_g, False))

    # 2. Star graph
    nx_g = nx.star_graph(15)
    graphs.append(("star_16", nx_g, False))

    # 3. Path graph
    nx_g = nx.path_graph(15)
    graphs.append(("path_15", nx_g, False))

    # 4. Cycle graph
    nx_g = nx.cycle_graph(12)
    graphs.append(("cycle_12", nx_g, False))

    # 5. Random sparse graph
    nx_g = nx.erdos_renyi_graph(30, 0.1, seed=42, directed=False)
    graphs.append(("random_sparse_30", nx_g, False))

    # 6. Scale-free (Barabási-Albert)
    nx_g = nx.barabasi_albert_graph(40, 2, seed=42)
    graphs.append(("scale_free_40", nx_g, False))

    # 7. Karate club (real-world network)
    nx_g = nx.karate_club_graph()
    graphs.append(("karate_club", nx_g, False))

    # 8. Grid graph (small for speed)
    nx_g = nx.grid_2d_graph(4, 4)
    nx_g = nx.relabel_nodes(nx_g, {node: str(node) for node in nx_g.nodes()})
    graphs.append(("grid_4x4", nx_g, False))

    # 9. Directed acyclic graph
    nx_g = nx.DiGraph()
    nx_g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4)])
    graphs.append(("dag_5", nx_g, True))

    return graphs


def compare_betweenness_networkx(ec_graph, nx_graph, directed=False):
    """Compare Betweenness Centrality between EdgeCase and NetworkX."""
    # Run Betweenness Centrality
    ec_btw = ec_graph.betweenness_centrality()
    nx_btw = nx.betweenness_centrality(nx_graph, normalized=True)

    # Align results
    node_list = list(nx_graph.nodes())
    ec_array, nx_array = align_results(ec_btw, nx_btw, node_list, ec_graph, is_dict=True)

    # Compute metrics
    return compute_comparison_metrics(ec_array, nx_array)


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
def compare_betweenness_igraph(ec_graph, nx_graph, directed=False):
    """Compare Betweenness Centrality between EdgeCase and igraph."""
    ig_graph, node_list = nx_to_igraph(nx_graph, directed)

    # Run Betweenness Centrality
    ec_btw = ec_graph.betweenness_centrality()
    ig_btw_raw = ig_graph.betweenness(directed=directed)

    # igraph returns unnormalized betweenness, need to normalize
    n = len(node_list)
    if n <= 2:
        norm_factor = 1.0
    else:
        norm_factor = (n - 1) * (n - 2) / (2.0 if not directed else 1.0)

    ig_btw = [score / norm_factor for score in ig_btw_raw]

    # Align results
    ec_array, ig_array = align_results(ec_btw, ig_btw, node_list, ec_graph, is_dict=False)

    # Compute metrics
    return compute_comparison_metrics(ec_array, ig_array)


@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_betweenness_vs_networkx(name, nx_graph, directed):
    """Test Betweenness Centrality correctness against NetworkX."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_betweenness_networkx(ec_graph, nx_graph, directed)

    passes = check_validation_passes(metrics, MAX_ABS_DIFF, MAX_REL_DIFF, MIN_CORRELATION)

    assert passes, format_validation_error("Betweenness Centrality", name, metrics)


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_betweenness_vs_igraph(name, nx_graph, directed):
    """Test Betweenness Centrality correctness against igraph."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_betweenness_igraph(ec_graph, nx_graph, directed)

    passes = check_validation_passes(metrics, MAX_ABS_DIFF, MAX_REL_DIFF, MIN_CORRELATION)

    assert passes, format_validation_error(
        "Betweenness Centrality", name, metrics, vs_library="igraph"
    )
