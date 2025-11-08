__version__ = "0.1.0"

# Try to import Rust core, fall back to Python implementation
try:
    # Import Rust core with Python wrapper for API compatibility
    from edgecase._core_rust import (
        Graph,
        Node,
        Edge,
        shortest_path,
        pagerank,
        betweenness_centrality,
        detect_communities,
        find_cycles,
    )
    _USING_RUST = True
except ImportError:
    # Fall back to pure Python implementation
    from edgecase._core_simple import (
        Graph,
        shortest_path,
        pagerank,
        from_polars,
    )
    _USING_RUST = False
    Node = None
    Edge = None
    betweenness_centrality = None
    detect_communities = None
    find_cycles = None

# Python-layer functionality
from edgecase.polars_integration import from_polars, to_polars
from edgecase.analysis import NetworkAnalyzer
from edgecase.tables import tbl_centrality, tbl_network_summary
from edgecase.viz import plot_graph
from edgecase.widget import GraphWidget, graph_widget
from edgecase.aesthetics import (
    colorize_node_attrs,
    rescale_node_attrs,
    set_node_attr_to_display,
)
from edgecase.bio import (
    protein_network,
    patient_similarity_network,
    pathway_analysis,
)
from edgecase.fraud import (
    transaction_network,
    find_suspicious_patterns,
    temporal_fraud_detection,
)
from edgecase.builders import (
    add_path,
    add_cycle,
    add_star,
    add_full_graph,
    add_balanced_tree,
    add_prism,
    add_grid_2d,
    create_graph,
)

# NetworkX interoperability (optional)
try:
    from edgecase.networkx import to_networkx, from_networkx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False
    to_networkx = None
    from_networkx = None

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Graph",
    "Node",
    "Edge",
    # Core algorithms
    "shortest_path",
    "pagerank",
    "betweenness_centrality",
    "detect_communities",
    "find_cycles",
    # I/O
    "from_polars",
    "to_polars",
    "to_networkx",
    "from_networkx",
    # Analysis
    "NetworkAnalyzer",
    # Visualization
    "plot_graph",
    "GraphWidget",
    "graph_widget",
    # Aesthetics
    "colorize_node_attrs",
    "rescale_node_attrs",
    "set_node_attr_to_display",
    # Tables
    "tbl_centrality",
    "tbl_network_summary",
    # Graph builders
    "add_path",
    "add_cycle",
    "add_star",
    "add_full_graph",
    "add_balanced_tree",
    "add_prism",
    "add_grid_2d",
    "create_graph",
    # Domain-specific
    "protein_network",
    "patient_similarity_network",
    "pathway_analysis",
    "transaction_network",
    "find_suspicious_patterns",
    "temporal_fraud_detection",
]
