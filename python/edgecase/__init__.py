try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("edgecase")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import Rust core with Python wrapper for API compatibility
from edgecase.aesthetics import (
    colorize_node_attrs,
    rescale_node_attrs,
    set_node_attr_to_display,
)
from edgecase.analysis import NetworkAnalyzer
from edgecase.datasets import (
    karate_club,
)
from edgecase.graph import (
    Edge,
    Graph,
    Node,
    betweenness_centrality,
    detect_communities,
    find_cycles,
    pagerank,
    shortest_path,
)
from edgecase.io import (
    from_matrix_market,
)

# Python-layer functionality
from edgecase.polars_integration import from_polars, to_polars
from edgecase.tables import tbl_centrality, tbl_network_summary
from edgecase.viz import plot_graph
from edgecase.widget import GraphWidget, graph_widget

# NetworkX interoperability (optional)
try:
    from edgecase.networkx import from_networkx, to_networkx

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False
    to_networkx = None
    from_networkx = None

# igraph interoperability (optional)
try:
    from edgecase.igraph_interop import from_igraph, to_igraph

    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False
    to_igraph = None
    from_igraph = None

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
    "to_igraph",
    "from_igraph",
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
    # Datasets
    "from_matrix_market",
    "karate_club",
]
