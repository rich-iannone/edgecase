try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib_metadata import version as _version

try:  # pragma: no cover
    __version__ = _version("edgecase")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import Rust core with Python wrapper for API compatibility
from edgecase.colors import (
    COLOR_PALETTES,
    list_palettes,
)
from edgecase.datasets import (
    ca_grqc,
    celegans,
    condmat,
    dolphins,
    enron,
    football,
    get_dataset_info,
    gnutella,
    internet,
    jazz,
    karate_club,
    list_available_datasets,
    list_registered_datasets,
    load_dataset,
    pgp,
    physics,
    polbooks,
    power_grid,
    soc_livejournal,
    web_google,
    wiki_vote,
)
from edgecase.graph import (
    Edge,
    Graph,
    Node,
)
from edgecase.io import (
    from_gml,
    from_graphml,
    from_matrix_market,
    from_sif,
    from_tsv_edge_list,
    to_gml,
    to_graphml,
    to_sif,
    to_tsv_edge_list,
)
from edgecase.multiplex import MultiplexGraph
from edgecase.svg_renderer import PlotSVG

# Python-layer functionality
# Polars integration: Use Graph(nodes_df=..., edges_df=...) to create from DataFrames
# Use graph.get_node_df() and graph.get_edge_df() to export to DataFrames
from edgecase.tables import tbl_centrality, tbl_network_summary
from edgecase.viz import plot_cosmo, plot_graph
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

# Matrix utilities (always available)
from edgecase.matrix_utils import from_adjacency_matrix, to_adjacency_matrix

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Graph",
    "MultiplexGraph",
    "Node",
    "Edge",
    # I/O
    "to_networkx",
    "from_networkx",
    "to_igraph",
    "from_igraph",
    "to_adjacency_matrix",
    "from_adjacency_matrix",
    # Visualization
    "PlotSVG",
    # "plot_graph",
    # "GraphWidget",
    # "graph_widget",
    # Tables
    "tbl_centrality",
    "tbl_network_summary",
    # File format I/O
    "from_graphml",
    "to_graphml",
    "from_gml",
    "to_gml",
    "from_sif",
    "to_sif",
    "from_tsv_edge_list",
    "to_tsv_edge_list",
    # Datasets
    "from_matrix_market",
    "load_dataset",
    "list_available_datasets",
    "karate_club",
    "web_google",
    "soc_livejournal",
    "ca_grqc",
    "celegans",
    "condmat",
    "dolphins",
    "enron",
    "football",
    "get_dataset_info",
    "gnutella",
    "internet",
    "jazz",
    "pgp",
    "physics",
    "polbooks",
    "power_grid",
    "wiki_vote",
]

# Exclude Rust types from documentation generation (they can't be introspected by quartodoc)
__gt_exclude__ = ["Node", "Edge"]
