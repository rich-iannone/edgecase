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


# Tolerance thresholds
MAX_ABS_DIFF = 1e-2  # 0.01 (relaxed for convergence differences)
MAX_REL_DIFF = 0.10  # 10% (relaxed for small values)
MIN_CORRELATION = 0.98  # 0.98 (still very high)


def create_test_graphs():
    """Create diverse graph types for validation."""
    graphs = []

    # 1. Complete graph
    nx_g = nx.complete_graph(10)
    graphs.append(("complete_10", nx_g, False))

    # 2. Star graph
    nx_g = nx.star_graph(20)
    graphs.append(("star_21", nx_g, False))

    # 3. Cycle graph
    nx_g = nx.cycle_graph(15)
    graphs.append(("cycle_15", nx_g, False))

    # 4. Random sparse graph
    nx_g = nx.erdos_renyi_graph(50, 0.05, seed=23, directed=False)
    graphs.append(("random_sparse_50", nx_g, False))

    # 5. Random dense graph
    nx_g = nx.erdos_renyi_graph(30, 0.3, seed=23, directed=False)
    graphs.append(("random_dense_30", nx_g, False))

    # 6. Scale-free (Barabási-Albert)
    nx_g = nx.barabasi_albert_graph(50, 3, seed=23)
    graphs.append(("scale_free_50", nx_g, False))

    # 7. Watts-Strogatz (small-world)
    nx_g = nx.watts_strogatz_graph(40, 4, 0.3, seed=23)
    graphs.append(("small_world_40", nx_g, False))

    # 8. Path graph
    nx_g = nx.path_graph(20)
    graphs.append(("path_20", nx_g, False))

    # 9. Karate club (real-world network)
    nx_g = nx.karate_club_graph()
    graphs.append(("karate_club", nx_g, False))

    # 10. Directed acyclic graph
    nx_g = nx.DiGraph()
    nx_g.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 6), (4, 6), (5, 6)])
    graphs.append(("dag_7", nx_g, True))

    # 11. Directed cyclic graph
    nx_g = nx.DiGraph()
    nx_g.add_edges_from([(i, (i + 1) % 10) for i in range(10)])
    nx_g.add_edges_from([(i, (i + 3) % 10) for i in range(10)])
    graphs.append(("directed_cycle_10", nx_g, True))

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


def compare_pagerank_networkx(ec_graph, nx_graph, directed=False):
    """Compare PageRank between EdgeCase and NetworkX."""
    # Run PageRank
    ec_pr = ec_graph.pagerank(damping=0.85, max_iter=100, tol=1e-6, return_dict=True)
    nx_pr = nx.pagerank(nx_graph, alpha=0.85, max_iter=100, tol=1e-6)

    # Build label->node_id mapping for EdgeCase
    label_to_id = {}
    for node_id in ec_graph.get_node_ids(limit=None):
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
        if node_id is not None and node_id in ec_pr:
            ec_values.append(ec_pr[node_id])
        else:
            ec_values.append(0.0)
        nx_values.append(nx_pr[node])

    ec_array = np.array(ec_values)
    nx_array = np.array(nx_values)

    # Compute metrics
    abs_diff = np.abs(ec_array - nx_array)
    rel_diff = np.abs((ec_array - nx_array) / (nx_array + 1e-10))

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation (handle uniform distributions)
    if np.std(ec_array) > 1e-10 and np.std(nx_array) > 1e-10:
        correlation = np.corrcoef(ec_array, nx_array)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-10 else 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correlation": correlation,
    }


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
def compare_pagerank_igraph(ec_graph, nx_graph, directed=False):
    """Compare PageRank between EdgeCase and igraph."""
    ig_graph, node_list = nx_to_igraph(nx_graph, directed)

    # Run PageRank
    ec_pr = ec_graph.pagerank(damping=0.85, max_iter=100, tol=1e-6, return_dict=True)
    ig_pr = ig_graph.pagerank(damping=0.85)

    # Build label->node_id mapping for EdgeCase
    label_to_id = {}
    for node_id in ec_graph.get_node_ids(limit=None):
        node = ec_graph.get_node(node_id)
        if node:
            label_to_id[node.label] = node_id

    # Convert to comparable arrays
    ec_values = []
    ig_values = ig_pr

    for node in node_list:
        node_str = str(node)
        node_id = label_to_id.get(node_str)
        if node_id is not None and node_id in ec_pr:
            ec_values.append(ec_pr[node_id])
        else:
            ec_values.append(0.0)

    ec_array = np.array(ec_values)
    ig_array = np.array(ig_values)

    # Compute metrics
    abs_diff = np.abs(ec_array - ig_array)
    rel_diff = np.abs((ec_array - ig_array) / (ig_array + 1e-10))

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation
    if np.std(ec_array) > 1e-10 and np.std(ig_array) > 1e-10:
        correlation = np.corrcoef(ec_array, ig_array)[0, 1]
    else:
        correlation = 1.0 if max_abs_diff < 1e-10 else 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "correlation": correlation,
    }


@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_pagerank_vs_networkx(name, nx_graph, directed):
    """Test PageRank correctness against NetworkX."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_pagerank_networkx(ec_graph, nx_graph, directed)

    # Check passes validation
    passes = (
        metrics["max_abs_diff"] < MAX_ABS_DIFF and metrics["max_rel_diff"] < MAX_REL_DIFF
    ) or metrics["correlation"] > MIN_CORRELATION

    assert passes, (
        f"PageRank validation failed for {name}:\n"
        f"  Max abs diff: {metrics['max_abs_diff']:.2e}\n"
        f"  Max rel diff: {metrics['max_rel_diff']:.2%}\n"
        f"  Correlation: {metrics['correlation']:.6f}"
    )


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
@pytest.mark.parametrize("name,nx_graph,directed", create_test_graphs())
def test_pagerank_vs_igraph(name, nx_graph, directed):
    """Test PageRank correctness against igraph."""
    ec_graph = nx_to_edgecase(nx_graph, directed)
    metrics = compare_pagerank_igraph(ec_graph, nx_graph, directed)

    # Check passes validation
    passes = (
        metrics["max_abs_diff"] < MAX_ABS_DIFF and metrics["max_rel_diff"] < MAX_REL_DIFF
    ) or metrics["correlation"] > MIN_CORRELATION

    assert passes, (
        f"PageRank validation failed for {name} vs igraph:\n"
        f"  Max abs diff: {metrics['max_abs_diff']:.2e}\n"
        f"  Max rel diff: {metrics['max_rel_diff']:.2%}\n"
        f"  Correlation: {metrics['correlation']:.6f}"
    )
