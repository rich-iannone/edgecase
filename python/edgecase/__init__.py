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
    davis_southern_women,
    dolphins,
    enron,
    florentine_families,
    football,
    get_dataset_info,
    gnutella,
    internet,
    jazz,
    karate_club,
    les_miserables,
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
from edgecase.errors import EdgeNotFoundError, NodeNotFoundError
from edgecase.graph import (
    Edge,
    Graph,
    Node,
)
from edgecase.io import (
    from_adjacency_data,
    from_adjlist,
    from_biadjacency_matrix,
    from_cytoscape,
    from_dict_of_dicts,
    from_dict_of_lists,
    from_gexf,
    from_gml,
    from_graph6,
    from_graphml,
    from_leda,
    from_matrix_market,
    from_multiline_adjlist,
    from_node_link,
    from_numpy_array,
    from_pajek,
    from_pandas_adjacency,
    from_pandas_edgelist,
    from_scipy_sparse_array,
    from_sif,
    from_sparse6,
    from_tree_data,
    from_tsv_edge_list,
    to_adjacency_data,
    to_adjlist,
    to_cytoscape,
    to_dict_of_dicts,
    to_dict_of_lists,
    to_gexf,
    to_gml,
    to_graph6,
    to_graphml,
    to_latex,
    to_latex_raw,
    to_leda,
    to_multiline_adjlist,
    to_node_link,
    to_numpy_array,
    to_pajek,
    to_pandas_adjacency,
    to_pandas_edgelist,
    to_scipy_sparse_array,
    to_sif,
    to_sparse6,
    to_tree_data,
    to_tsv_edge_list,
)
from edgecase.multiplex import MultiplexGraph
from edgecase.svg_renderer import StepPlayer

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
    # Exceptions
    "NodeNotFoundError",
    "EdgeNotFoundError",
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
    "StepPlayer",
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
    "from_gexf",
    "to_gexf",
    "from_node_link",
    "to_node_link",
    "from_cytoscape",
    "to_cytoscape",
    "from_pajek",
    "to_pajek",
    "from_leda",
    "to_leda",
    "to_latex",
    "to_latex_raw",
    "from_graph6",
    "to_graph6",
    "from_sparse6",
    "to_sparse6",
    "to_numpy_array",
    "from_numpy_array",
    "to_scipy_sparse_array",
    "from_scipy_sparse_array",
    "from_biadjacency_matrix",
    "to_pandas_adjacency",
    "from_pandas_adjacency",
    "to_pandas_edgelist",
    "from_pandas_edgelist",
    "to_adjlist",
    "from_adjlist",
    "to_multiline_adjlist",
    "from_multiline_adjlist",
    "to_adjacency_data",
    "from_adjacency_data",
    "to_tree_data",
    "from_tree_data",
    "to_dict_of_dicts",
    "from_dict_of_dicts",
    "to_dict_of_lists",
    "from_dict_of_lists",
    # Datasets
    "from_matrix_market",
    "load_dataset",
    "list_available_datasets",
    "karate_club",
    "davis_southern_women",
    "florentine_families",
    "les_miserables",
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
