"""Tell NetworkX about the edgecase backend.

NetworkX calls `get_info()` (the `networkx.backend_info` entry point) while it is being
imported, so this package must stay tiny: it must not import `edgecase` (which loads the
Rust extension and Polars) or `networkx`. The backend itself lives in
`edgecase.nx_backend` and is loaded only when NetworkX first dispatches to it.

`_FUNCTIONS` lists the NetworkX functions the backend implements; a test in edgecase's
suite checks it against the backend's registry, so the two cannot drift apart.
"""

_FUNCTIONS = frozenset(
    {
        # BEGIN: functions
        "ancestors",
        "average_clustering",
        "betweenness_centrality",
        "bfs_edges",
        "bfs_layers",
        "bfs_successors",
        "bfs_tree",
        "clustering",
        "connected_components",
        "core_number",
        "descendants",
        "descendants_at_distance",
        "dfs_edges",
        "dfs_labeled_edges",
        "dfs_postorder_nodes",
        "dfs_predecessors",
        "dfs_preorder_nodes",
        "dfs_successors",
        "dfs_tree",
        "generalized_degree",
        "is_connected",
        "is_directed_acyclic_graph",
        "is_isolate",
        "is_strongly_connected",
        "is_weakly_connected",
        "isolates",
        "k_core",
        "k_corona",
        "k_crust",
        "k_shell",
        "k_truss",
        "node_connected_component",
        "number_connected_components",
        "number_of_isolates",
        "number_strongly_connected_components",
        "number_weakly_connected_components",
        "pagerank",
        "square_clustering",
        "strongly_connected_components",
        "topological_generations",
        "topological_sort",
        "transitivity",
        "triangles",
        "weakly_connected_components",
        # END: functions
    }
)


def get_info():
    """Backend metadata for NetworkX (the `networkx.backend_info` entry point)."""
    return {
        "backend_name": "edgecase",
        "project": "edgecase",
        "package": "edgecase",
        "url": "https://github.com/rich-iannone/edgecase",
        "short_summary": "Rust-accelerated graph algorithms from edgecase.",
        "functions": {name: {} for name in sorted(_FUNCTIONS)},
        "default_config": {
            # Record each node's predecessor order when converting directed graphs, so
            # converting back reproduces it exactly (False saves 10-15% of the time)
            "preserve_pred_order": True,
        },
    }
