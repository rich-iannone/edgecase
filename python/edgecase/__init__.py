try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("edgecase")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import Rust core with Python wrapper for API compatibility
from edgecase.colors import (
    COLOR_PALETTES,
    list_palettes,
)
from edgecase.datasets import (
    ca_grqc,
    get_dataset_info,
    karate_club,
    list_available_datasets,
    list_registered_datasets,
    load_dataset,
    soc_livejournal,
    web_google,
)
from edgecase.graph import (
    Edge,
    Graph,
    Node,
    detect_communities,
    get_cycles,
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
    "detect_communities",
    "get_cycles",
    # I/O
    "from_polars",
    "to_polars",
    "to_networkx",
    "from_networkx",
    "to_igraph",
    "from_igraph",
    # Visualization
    "plot_graph",
    "GraphWidget",
    "graph_widget",
    # Tables
    "tbl_centrality",
    "tbl_network_summary",
    # Datasets
    "from_matrix_market",
    "karate_club",
    "web_google",
    "soc_livejournal",
    "ca_grqc",
    "load_dataset",
    "list_available_datasets",
    # Colors
    "COLOR_PALETTES",
    "list_palettes",
]
