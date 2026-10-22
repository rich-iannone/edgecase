import os
from pathlib import Path
from typing import Optional

import polars as pl

from edgecase.graph import Graph
from edgecase.io import _build_graph, from_matrix_market

# Path to data directory (relative to this file)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Cache directory for downloaded datasets
CACHE_DIR = Path.home() / ".cache" / "edgecase" / "datasets"

# Base URL for dataset downloads (GitHub releases)
DATASET_BASE_URL = "https://github.com/rich-iannone/edgecase/releases/download/datasets-v1"

# Allow users to disable auto-download
ALLOW_DOWNLOAD = os.environ.get("EDGECASE_NO_DOWNLOAD", "0") != "1"

# Dataset registry with metadata for known datasets
# Includes location (bundled vs remote) and download info
DATASET_REGISTRY = {
    "karate": {
        "filename": "karate.mtx",
        "name": "Zachary's Karate Club",
        "nodes": 34,
        "edges": 78,
        "directed": False,
        "description": "Classic social network of karate club friendships",
        "location": "bundled",
        "size_kb": 2,
    },
    "dolphins": {
        "filename": "dolphins.mtx",
        "name": "Dolphin Social Network",
        "nodes": 62,
        "edges": 159,
        "directed": False,
        "description": "Social network of bottlenose dolphins in Doubtful Sound, New Zealand",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "dolphins.mtx.gz",
        "size_kb": 3,
        "compressed_kb": 1,
    },
    "football": {
        "filename": "football.mtx",
        "name": "American College Football",
        "nodes": 35,
        "edges": 119,
        "directed": False,
        "description": "Network of American football games",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "football.mtx.gz",
        "size_kb": 1,
        "compressed_kb": 0.4,
    },
    "polbooks": {
        "filename": "polbooks.mtx",
        "name": "Political Books",
        "nodes": 105,
        "edges": 441,
        "directed": False,
        "description": "Network of books about US politics sold by Amazon, edges link frequently co-purchased books",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "polbooks.mtx.gz",
        "size_kb": 6,
        "compressed_kb": 2,
    },
    "jazz": {
        "filename": "jazz.mtx",
        "name": "Jazz Musicians Network",
        "nodes": 198,
        "edges": 2743,
        "directed": False,
        "description": "Collaboration network of jazz musicians",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "jazz.mtx.gz",
        "size_kb": 19,
        "compressed_kb": 7,
    },
    "celegans": {
        "filename": "celegans.mtx",
        "name": "C. Elegans Metabolic Network",
        "nodes": 453,
        "edges": 2041,
        "directed": False,
        "description": "Metabolic network of C. Elegans (nematode worm)",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "celegans.mtx.gz",
        "size_kb": 16,
        "compressed_kb": 5,
    },
    "power_grid": {
        "filename": "power_grid.mtx",
        "name": "US Power Grid",
        "nodes": 4941,
        "edges": 6594,
        "directed": False,
        "description": "Topology of the Western States Power Grid of the United States",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "power_grid.mtx.gz",
        "size_kb": 63,
        "compressed_kb": 23,
    },
    "ca_grqc": {
        "filename": "ca_grqc.mtx",
        "name": "Arxiv GR-QC Collaboration Network",
        "nodes": 5242,
        "edges": 28980,
        "directed": False,
        "description": "Scientific collaboration network from arXiv General Relativity",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_grqc.mtx.gz",
        "size_kb": 330,
        "compressed_kb": 103,
    },
    "wiki_vote": {
        "filename": "wiki_vote.mtx",
        "name": "Wikipedia Voting Network",
        "nodes": 7115,
        "edges": 103689,
        "directed": True,
        "description": "Wikipedia adminship voting network",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "wiki_vote.mtx.gz",
        "size_kb": 900,
        "compressed_kb": 279,
    },
    "ca_hepph": {
        "filename": "ca_hepph.mtx",
        "name": "High Energy Physics Collaboration Network",
        "nodes": 12008,
        "edges": 237010,
        "directed": False,
        "description": "Collaboration network from arXiv High Energy Physics - Phenomenology",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_hepph.mtx.gz",
        "size_kb": 1900,
        "compressed_kb": 774,
    },
    "pgp": {
        "filename": "pgp.mtx",
        "name": "PGP Trust Network",
        "nodes": 10680,
        "edges": 24340,
        "directed": True,
        "description": "Pretty-Good-Privacy (PGP) web of trust network",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "pgp.mtx.gz",
        "size_kb": 220,
        "compressed_kb": 81,
    },
    "internet": {
        "filename": "internet.mtx",
        "name": "Internet Autonomous Systems",
        "nodes": 6474,
        "edges": 13895,
        "directed": False,
        "description": "Internet topology at autonomous system level from January 2000",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "internet.mtx.gz",
        "size_kb": 120,
        "compressed_kb": 39,
    },
    "ca_condmat": {
        "filename": "ca_condmat.mtx",
        "name": "Condensed Matter Collaboration Network",
        "nodes": 23133,
        "edges": 186936,
        "directed": False,
        "description": "Collaboration network from arXiv Condensed Matter category",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_condmat.mtx.gz",
        "size_kb": 1600,
        "compressed_kb": 694,
    },
    "enron": {
        "filename": "enron.mtx",
        "name": "Enron Email Network",
        "nodes": 36692,
        "edges": 367662,
        "directed": True,
        "description": "Email communication network from Enron corpus",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "enron.mtx.gz",
        "size_kb": 2900,
        "compressed_kb": 1013,
    },
    "gnutella": {
        "filename": "gnutella.mtx",
        "name": "Gnutella P2P Network",
        "nodes": 6301,
        "edges": 20777,
        "directed": True,
        "description": "Gnutella peer-to-peer network snapshot from August 8, 2002",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "gnutella.mtx.gz",
        "size_kb": 180,
        "compressed_kb": 61,
    },
    "roadnet_pa": {
        "filename": "roadnet_pa.mtx",
        "name": "Pennsylvania Road Network",
        "nodes": 1087562,
        "edges": 1541514,
        "directed": False,
        "description": "Road network of Pennsylvania",
        "location": "remote",
        "compressed": "roadnet_pa.mtx.gz",
        "size_mb": 20,
        "compressed_mb": 5,
    },
    "web_google": {
        "filename": "web_google.mtx",
        "name": "Google Web Graph",
        "nodes": 875713,
        "edges": 5105039,
        "directed": True,
        "description": "Web graph from Google (2002)",
        "source": "SNAP",
        "location": "remote",
        "compressed": "web_google.mtx.gz",
        "size_mb": 45,
        "compressed_mb": 10,
    },
    "soc_livejournal": {
        "filename": "soc_livejournal.mtx",
        "name": "LiveJournal Social Network",
        "nodes": 4847571,
        "edges": 68993773,
        "directed": True,
        "description": "LiveJournal online social network",
        "source": "SNAP",
        "location": "remote",
        "compressed": "soc_livejournal.mtx.gz",
        "size_mb": 600,
        "compressed_mb": 120,
    },
}


def get_dataset_info(name: str) -> dict:
    """
    Get metadata about a dataset.

    Parameters
    ----------
    name
        Dataset name (with or without .mtx extension)

    Returns
    -------
    dict
        Dataset metadata including nodes, edges, description, etc.

    Examples
    --------
    ```python
    from edgecase import get_dataset_info

    info = get_dataset_info("roadnet_pa")

    # Returns dataset metadata with nodes, edges, etc.
    info
    ```
    """
    # Remove .mtx extension if present
    name = name.replace(".mtx", "")

    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name].copy()

    # Check if file exists even if not in registry
    filename = f"{name}.mtx"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return {
            "filename": filename,
            "name": name,
            "description": "Custom dataset (no metadata available)",
        }

    raise ValueError(
        f"Dataset '{name}' not found.\n"
        f"Available registered datasets: {list(DATASET_REGISTRY.keys())}\n"
        f"Available files: {list_available_datasets()}"
    )


def list_registered_datasets() -> list[str]:
    """
    List all registered datasets with metadata.

    Returns
    -------
    list[str]
        List of registered dataset names

    Examples
    --------
    ```python
    from edgecase import list_registered_datasets, get_dataset_info

    for name in list_registered_datasets():
        info = get_dataset_info(name)
        pass  # Iterates through registered datasets with their info
    ```
    """
    return list(DATASET_REGISTRY.keys())


def _download_dataset(name: str, compressed_filename: str) -> Path:
    """
    Download a dataset from remote repository and decompress it.

    Downloads to ~/.cache/edgecase/datasets/ and decompresses if needed.

    Parameters
    ----------
    name
        Dataset name (for metadata lookup)
    compressed_filename
        Filename of compressed dataset (e.g., roadnet_pa.mtx.gz)

    Returns
    -------
    Path
        Path to decompressed .mtx file in cache
    """
    import gzip
    from urllib.request import urlretrieve

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output filename (remove .gz)
    output_filename = compressed_filename.replace(".gz", "")
    output_path = CACHE_DIR / output_filename

    # Return if already cached
    if output_path.exists():
        return output_path

    # Construct download URL
    url = f"{DATASET_BASE_URL}/{compressed_filename}"
    temp_path = CACHE_DIR / f"{compressed_filename}.download"

    print(f"Downloading {name} dataset...")
    print(f"  Source: {url}")
    print(f"  This is a one-time download, cached at: {CACHE_DIR}")

    try:
        # Download compressed file
        urlretrieve(url, temp_path)

        # Decompress if .gz
        if compressed_filename.endswith(".gz"):
            print("  Decompressing...")
            with gzip.open(temp_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    f_out.write(f_in.read())
            temp_path.unlink()  # Remove compressed file
        else:
            # Not compressed, just rename
            temp_path.rename(output_path)

        print("  ✓ Download complete!")
        return output_path

    except Exception as e:
        # Clean up on failure
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"Failed to download dataset '{name}'.\n"
            f"Error: {e}\n\n"
            f"You can manually download from:\n  {url}\n"
            f"And place it at:\n  {output_path}"
        ) from e


def _load_dataset_from_file(
    filename: str,
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Load a dataset from a file in the data directory or cache.

    Checks bundled data/ directory first, then cache directory.
    Supports Matrix Market (.mtx and .mtx.gz) format.

    Parameters
    ----------
    filename
        Name of the file (e.g., karate.mtx or karate.mtx.gz).
    directed
        Whether graph is directed (auto-detected from file if None).
    weighted
        Whether graph is weighted (auto-detected from file if None).
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        Loaded graph
    """
    import gzip
    import tempfile

    # Check bundled data directory first (try both .mtx and .mtx.gz)
    filepath = DATA_DIR / filename
    compressed_filepath = DATA_DIR / f"{filename}.gz"

    # Determine which file exists
    if filepath.exists():
        final_path = filepath
        is_compressed = False
    elif compressed_filepath.exists():
        final_path = compressed_filepath
        is_compressed = True
    else:
        # Check cache directory
        cache_path = CACHE_DIR / filename
        if cache_path.exists():
            final_path = cache_path
            is_compressed = False
        else:
            raise FileNotFoundError(
                f"Dataset file not found: {filename}\n"
                f"Checked locations:\n"
                f"  - Bundled: {DATA_DIR}\n"
                f"  - Bundled compressed: {DATA_DIR} (.gz)\n"
                f"  - Cache: {CACHE_DIR}\n"
                f"Available bundled datasets: {list_available_datasets()}"
            )

    # Handle compressed files - decompress to temp file
    if is_compressed or str(final_path).endswith(".gz"):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mtx", delete=False) as tmp_file:
            with gzip.open(final_path, "rb") as f_in:
                tmp_file.write(f_in.read())
            temp_path = Path(tmp_file.name)

        try:
            result = from_matrix_market(
                temp_path, directed=directed, weighted=weighted, graph_name=graph_name
            )
        finally:
            temp_path.unlink()  # Clean up temp file
        return result

    # Support Matrix Market format
    if filename.endswith(".mtx"):
        return from_matrix_market(
            final_path, directed=directed, weighted=weighted, graph_name=graph_name
        )

    # Add more format support here (e.g., .csv, .json, .graphml)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def list_available_datasets() -> list[str]:
    """
    List all available dataset files in the data directory.

    Returns
    -------
    list[str]
        List of dataset filenames
    """
    if not DATA_DIR.exists():
        return []

    return [f.name for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]


def karate_club() -> Graph:
    """
    Zachary's Karate Club network.

    A social network of friendships between 34 members of a karate club at a US university in the
    1970s. The network became famous when the club split into two factions due to a dispute between
    the administrator ("John A.") and instructor ("Mr. Hi"), and the split could be predicted by
    network analysis.

    Network Properties:

    - 34 nodes (club members)
    - 78 edges (friendships)
    - Undirected, unweighted
    - Density: 0.139
    - 2 communities (corresponding to the actual club split)

    Network Metrics:

    - Degree: max=17, min=1, avg=4.588
    - Assortativity: -0.476 (not yet implemented)
    - Triangles: 135 total, 3.97 avg per node, 18 max (not yet implemented)
    - Clustering coefficient: 0.571 (average) ✓ implemented
    - Transitivity (fraction of closed triangles): 0.256 (not yet implemented)
    - Maximum k-core: 5 (currently reports 4)

    Node Attributes:

    - `faction`: Which faction the member joined (1 = Mr. Hi, 2 = John A.)
    - `role`: Club role ("Administrator", "Instructor", or "Member")

    Returns
    -------
    Graph
        The karate club network

    Examples
    --------
    ```python
    from edgecase import karate_club

    # Load the dataset
    G = karate_club()

    # Returns 34 nodes, 78 edges
    G.count_nodes(), G.count_edges()

    # Check basic metrics
    n = G.count_nodes()
    e = G.count_edges()
    density = e / (n * (n - 1) / 2)

    # Returns approximately 0.139
    density

    # Access node attributes by label
    node_id = G.get_id_from_label("0")
    node = G.get_node(node_id)

    # Returns 1 (Mr. Hi's faction)
    node.attributes['faction']

    # Analyze community structure
    communities = G.cmty_louvain(return_dict=True)
    num_communities = len(set(communities.values()))

    # Returns number of detected communities
    num_communities

    # Find most central members
    betweenness = G.betweenness_centrality(return_dict=True)
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
    for node_id, score in top_nodes:
        label = G.get_label_from_id(node_id)
        pass  # Iterates through top 3 nodes with centrality scores

    # Check k-core decomposition
    coreness = G.coreness(return_dict=True)
    max_core = max(coreness.values())

    # Returns the maximum k-core value
    max_core

    # Analyze clustering
    avg_clustering = G.get_avg_clustering_coefficient()

    # Returns approximately 0.571
    avg_clustering
    ```

    Notes
    -----
    This is one of the most well-studied networks in social network analysis. The network split
    prediction problem has been validated multiple times, and the dataset serves as a standard
    benchmark for community detection algorithms.
    """

    # Node data with factions (1 = Mr. Hi's faction, 2 = John A.'s faction)
    # Based on the actual split that occurred
    nodes_df = pl.DataFrame(
        {
            "id": list(range(34)),
            "faction": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,  # 0-9
                1,
                1,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                1,  # 10-19
                2,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,  # 20-29
                2,
                2,
                2,
                2,  # 30-33
            ],
            "role": ["Instructor"] + ["Member"] * 32 + ["Administrator"],
        }
    )

    # Edge list (undirected friendships) - canonical Zachary karate club network
    # 78 undirected edges from the Matrix Market format file (soc-karate.mtx)
    # Edges are 0-indexed (converted from 1-indexed in the original file)
    edge_list = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ]

    edges_df = pl.DataFrame({"from": [e[0] for e in edge_list], "to": [e[1] for e in edge_list]})

    # Build graph using the Graph constructor
    graph = Graph(
        directed=False, nodes_df=nodes_df, edges_df=edges_df, graph_name="Zachary's Karate Club"
    )

    return graph


def _labeled_graph(
    nodes: list[tuple[str, dict]],
    edges: list[tuple[str, str, Optional[float]]],
    weighted: bool,
    graph_name: str,
    code: str,
) -> Graph:
    """An undirected graph whose nodes are labeled with their names, built as a single step."""
    return _build_graph(
        [(name, name, attrs) for name, attrs in nodes],
        [(u, v, w, {}) for u, v, w in edges],
        False,
        weighted,
        graph_name,
        code,
    )


def davis_southern_women() -> Graph:
    """
    Davis Southern Women network.

    A two-mode network of 18 women and the 14 social events they attended in a town in the
    American South in the 1930s, collected by Davis, Gardner and Gardner for their 1941 study
    *Deep South*. Each edge joins a woman to an event she attended. It's the standard example
    for bipartite network analysis: projecting it onto the women links those who met at the same
    events, and the women fall into two overlapping groups.

    Network Properties:

    - 32 nodes (18 women, then 14 events labeled `"E1"` to `"E14"`)
    - 89 edges (attendances)
    - Undirected, unweighted, bipartite

    Node Attributes:

    - `bipartite`: `0` for the women, `1` for the events

    The nodes, their order, labels and attributes match NetworkX's
    `davis_southern_women_graph()`.

    Returns
    -------
    Graph
        The Davis Southern Women network

    Examples
    --------
    ```python
    from edgecase import davis_southern_women

    G = davis_southern_women()

    # Returns (32, 89)
    G.count_nodes(), G.count_edges()

    # Returns 8 (the events Evelyn Jefferson attended)
    G.get_degree("Evelyn Jefferson")

    # Project onto the women: two women are linked when they attended the same event
    women = G.bipartite_projection(which=0)

    # Returns 18
    women.count_nodes()
    ```

    Notes
    -----
    Davis, A., Gardner, B. B. and Gardner, M. R. (1941). *Deep South: A Social Anthropological
    Study of Caste and Class*. University of Chicago Press.
    """
    attended = {
        "Evelyn Jefferson": [1, 2, 3, 4, 5, 6, 8, 9],
        "Laura Mandeville": [1, 2, 3, 5, 6, 7, 8],
        "Theresa Anderson": [2, 3, 4, 5, 6, 7, 8, 9],
        "Brenda Rogers": [1, 3, 4, 5, 6, 7, 8],
        "Charlotte McDowd": [3, 4, 5, 7],
        "Frances Anderson": [3, 5, 6, 8],
        "Eleanor Nye": [5, 6, 7, 8],
        "Pearl Oglethorpe": [6, 8, 9],
        "Ruth DeSand": [5, 7, 8, 9],
        "Verne Sanderson": [7, 8, 9, 12],
        "Myra Liddel": [8, 9, 10, 12],
        "Katherina Rogers": [8, 9, 10, 12, 13, 14],
        "Sylvia Avondale": [7, 8, 9, 10, 12, 13, 14],
        "Nora Fayette": [6, 7, 9, 10, 11, 12, 13, 14],
        "Helen Lloyd": [7, 8, 10, 11, 12],
        "Dorothy Murchison": [8, 9],
        "Olivia Carleton": [9, 11],
        "Flora Price": [9, 11],
    }
    nodes = [(woman, {"bipartite": 0}) for woman in attended]
    nodes += [(f"E{i}", {"bipartite": 1}) for i in range(1, 15)]
    edges = [(woman, f"E{i}", None) for woman, events in attended.items() for i in events]
    return _labeled_graph(nodes, edges, False, "Davis Southern Women", "davis_southern_women()")


def florentine_families() -> Graph:
    """
    Florentine families network.

    Marriage ties between 15 leading families of Renaissance Florence around 1430, from the data
    John Padgett collected to study the rise of the Medici. The Medici weren't the richest or the
    most powerful family, but they sat at the center of the marriage network, bridging families
    that had no ties to each other; the network is a classic example for betweenness centrality.

    Network Properties:

    - 15 nodes (families, labeled by name)
    - 20 edges (marriage ties)
    - Undirected, unweighted
    - Connected (the isolated Pucci family is left out)

    The nodes, their order and labels match NetworkX's `florentine_families_graph()`.

    Returns
    -------
    Graph
        The Florentine families marriage network

    Examples
    --------
    ```python
    from edgecase import florentine_families

    G = florentine_families()

    # Returns (15, 20)
    G.count_nodes(), G.count_edges()

    # Returns 'Medici' (the family with the highest betweenness centrality)
    scores = G.betweenness_centrality(return_dict=True)
    G.get_label_from_id(max(scores, key=scores.get))
    ```

    Notes
    -----
    Padgett, J. F. and Ansell, C. K. (1993). Robust Action and the Rise of the Medici,
    1400-1434. *American Journal of Sociology*, 98(6), 1259-1319. The edges here follow
    Breiger, R. L. and Pattison, P. E. (1986). Cumulated social roles: The duality of persons
    and their algebras. *Social Networks*, 8(3), 215-256.
    """
    edges = [
        ("Acciaiuoli", "Medici"),
        ("Castellani", "Peruzzi"),
        ("Castellani", "Strozzi"),
        ("Castellani", "Barbadori"),
        ("Medici", "Barbadori"),
        ("Medici", "Ridolfi"),
        ("Medici", "Tornabuoni"),
        ("Medici", "Albizzi"),
        ("Medici", "Salviati"),
        ("Salviati", "Pazzi"),
        ("Peruzzi", "Strozzi"),
        ("Peruzzi", "Bischeri"),
        ("Strozzi", "Ridolfi"),
        ("Strozzi", "Bischeri"),
        ("Ridolfi", "Tornabuoni"),
        ("Tornabuoni", "Guadagni"),
        ("Albizzi", "Ginori"),
        ("Albizzi", "Guadagni"),
        ("Bischeri", "Guadagni"),
        ("Guadagni", "Lamberteschi"),
    ]
    return _labeled_graph(
        [],
        [(u, v, None) for u, v in edges],
        False,
        "Florentine Families",
        "florentine_families()",
    )


# Les Misérables coappearances as "character character weight", separated by commas and newlines
_LES_MISERABLES = """
Napoleon Myriel 1, MlleBaptistine Myriel 8, MmeMagloire Myriel 10
MmeMagloire MlleBaptistine 6, CountessDeLo Myriel 1, Geborand Myriel 1
Champtercier Myriel 1, Cravatte Myriel 1, Count Myriel 2, OldMan Myriel 1
Valjean Labarre 1, Valjean MmeMagloire 3, Valjean MlleBaptistine 3, Valjean Myriel 5
Marguerite Valjean 1, MmeDeR Valjean 1, Isabeau Valjean 1, Gervais Valjean 1
Listolier Tholomyes 4, Fameuil Tholomyes 4, Fameuil Listolier 4
Blacheville Tholomyes 4, Blacheville Listolier 4, Blacheville Fameuil 4
Favourite Tholomyes 3, Favourite Listolier 3, Favourite Fameuil 3
Favourite Blacheville 4, Dahlia Tholomyes 3, Dahlia Listolier 3, Dahlia Fameuil 3
Dahlia Blacheville 3, Dahlia Favourite 5, Zephine Tholomyes 3, Zephine Listolier 3
Zephine Fameuil 3, Zephine Blacheville 3, Zephine Favourite 4, Zephine Dahlia 4
Fantine Tholomyes 3, Fantine Listolier 3, Fantine Fameuil 3, Fantine Blacheville 3
Fantine Favourite 4, Fantine Dahlia 4, Fantine Zephine 4, Fantine Marguerite 2
Fantine Valjean 9, MmeThenardier Fantine 2, MmeThenardier Valjean 7
Thenardier MmeThenardier 13, Thenardier Fantine 1, Thenardier Valjean 12
Cosette MmeThenardier 4, Cosette Valjean 31, Cosette Tholomyes 1
Cosette Thenardier 1, Javert Valjean 17, Javert Fantine 5, Javert Thenardier 5
Javert MmeThenardier 1, Javert Cosette 1, Fauchelevent Valjean 8
Fauchelevent Javert 1, Bamatabois Fantine 1, Bamatabois Javert 1
Bamatabois Valjean 2, Perpetue Fantine 1, Simplice Perpetue 2, Simplice Valjean 3
Simplice Fantine 2, Simplice Javert 1, Scaufflaire Valjean 1, Woman1 Valjean 2
Woman1 Javert 1, Judge Valjean 3, Judge Bamatabois 2, Champmathieu Valjean 3
Champmathieu Judge 3, Champmathieu Bamatabois 2, Brevet Judge 2
Brevet Champmathieu 2, Brevet Valjean 2, Brevet Bamatabois 1, Chenildieu Judge 2
Chenildieu Champmathieu 2, Chenildieu Brevet 2, Chenildieu Valjean 2
Chenildieu Bamatabois 1, Cochepaille Judge 2, Cochepaille Champmathieu 2
Cochepaille Brevet 2, Cochepaille Chenildieu 2, Cochepaille Valjean 2
Cochepaille Bamatabois 1, Pontmercy Thenardier 1, Boulatruelle Thenardier 1
Eponine MmeThenardier 2, Eponine Thenardier 3, Anzelma Eponine 2
Anzelma Thenardier 2, Anzelma MmeThenardier 1, Woman2 Valjean 3, Woman2 Cosette 1
Woman2 Javert 1, MotherInnocent Fauchelevent 3, MotherInnocent Valjean 1
Gribier Fauchelevent 2, MmeBurgon Jondrette 1, Gavroche MmeBurgon 2
Gavroche Thenardier 1, Gavroche Javert 1, Gavroche Valjean 1, Gillenormand Cosette 3
Gillenormand Valjean 2, Magnon Gillenormand 1, Magnon MmeThenardier 1
MlleGillenormand Gillenormand 9, MlleGillenormand Cosette 2
MlleGillenormand Valjean 2, MmePontmercy MlleGillenormand 1, MmePontmercy Pontmercy 1
MlleVaubois MlleGillenormand 1, LtGillenormand MlleGillenormand 2
LtGillenormand Gillenormand 1, LtGillenormand Cosette 1, Marius MlleGillenormand 6
Marius Gillenormand 12, Marius Pontmercy 1, Marius LtGillenormand 1
Marius Cosette 21, Marius Valjean 19, Marius Tholomyes 1, Marius Thenardier 2
Marius Eponine 5, Marius Gavroche 4, BaronessT Gillenormand 1, BaronessT Marius 1
Mabeuf Marius 1, Mabeuf Eponine 1, Mabeuf Gavroche 1, Enjolras Marius 7
Enjolras Gavroche 7, Enjolras Javert 6, Enjolras Mabeuf 1, Enjolras Valjean 4
Combeferre Enjolras 15, Combeferre Marius 5, Combeferre Gavroche 6
Combeferre Mabeuf 2, Prouvaire Gavroche 1, Prouvaire Enjolras 4
Prouvaire Combeferre 2, Feuilly Gavroche 2, Feuilly Enjolras 6, Feuilly Prouvaire 2
Feuilly Combeferre 5, Feuilly Mabeuf 1, Feuilly Marius 1, Courfeyrac Marius 9
Courfeyrac Enjolras 17, Courfeyrac Combeferre 13, Courfeyrac Gavroche 7
Courfeyrac Mabeuf 2, Courfeyrac Eponine 1, Courfeyrac Feuilly 6
Courfeyrac Prouvaire 3, Bahorel Combeferre 5, Bahorel Gavroche 5
Bahorel Courfeyrac 6, Bahorel Mabeuf 2, Bahorel Enjolras 4, Bahorel Feuilly 3
Bahorel Prouvaire 2, Bahorel Marius 1, Bossuet Marius 5, Bossuet Courfeyrac 12
Bossuet Gavroche 5, Bossuet Bahorel 4, Bossuet Enjolras 10, Bossuet Feuilly 6
Bossuet Prouvaire 2, Bossuet Combeferre 9, Bossuet Mabeuf 1, Bossuet Valjean 1
Joly Bahorel 5, Joly Bossuet 7, Joly Gavroche 3, Joly Courfeyrac 5, Joly Enjolras 5
Joly Feuilly 5, Joly Prouvaire 2, Joly Combeferre 5, Joly Mabeuf 1, Joly Marius 2
Grantaire Bossuet 3, Grantaire Enjolras 3, Grantaire Combeferre 1
Grantaire Courfeyrac 2, Grantaire Joly 2, Grantaire Gavroche 1, Grantaire Bahorel 1
Grantaire Feuilly 1, Grantaire Prouvaire 1, MotherPlutarch Mabeuf 3
Gueulemer Thenardier 5, Gueulemer Valjean 1, Gueulemer MmeThenardier 1
Gueulemer Javert 1, Gueulemer Gavroche 1, Gueulemer Eponine 1, Babet Thenardier 6
Babet Gueulemer 6, Babet Valjean 1, Babet MmeThenardier 1, Babet Javert 2
Babet Gavroche 1, Babet Eponine 1, Claquesous Thenardier 4, Claquesous Babet 4
Claquesous Gueulemer 4, Claquesous Valjean 1, Claquesous MmeThenardier 1
Claquesous Javert 1, Claquesous Eponine 1, Claquesous Enjolras 1
Montparnasse Javert 1, Montparnasse Babet 2, Montparnasse Gueulemer 2
Montparnasse Claquesous 2, Montparnasse Valjean 1, Montparnasse Gavroche 1
Montparnasse Eponine 1, Montparnasse Thenardier 1, Toussaint Cosette 2
Toussaint Javert 1, Toussaint Valjean 1, Child1 Gavroche 2, Child2 Gavroche 2
Child2 Child1 3, Brujon Babet 3, Brujon Gueulemer 3, Brujon Thenardier 3
Brujon Gavroche 1, Brujon Eponine 1, Brujon Claquesous 1, Brujon Montparnasse 1
MmeHucheloup Bossuet 1, MmeHucheloup Joly 1, MmeHucheloup Grantaire 1
MmeHucheloup Bahorel 1, MmeHucheloup Courfeyrac 1, MmeHucheloup Gavroche 1
MmeHucheloup Enjolras 1
"""


def les_miserables() -> Graph:
    """
    Les Misérables character coappearance network.

    The characters of Victor Hugo's novel *Les Misérables*, joined when they appear together in
    a chapter, with the number of chapters they share as the edge weight. Compiled by Donald
    Knuth for the Stanford GraphBase, it's a standard example for weighted community detection
    and for drawing graphs: Valjean sits at the center, with clusters for the Thénardiers, the
    students of the ABC society and Fantine's circle.

    Network Properties:

    - 77 nodes (characters, labeled by name as in the Stanford GraphBase, e.g. `"Valjean"`)
    - 254 edges (coappearances)
    - Undirected, weighted (weights from 1 to 31)

    The nodes, their order, labels and weights match NetworkX's `les_miserables_graph()`.

    Returns
    -------
    Graph
        The Les Misérables coappearance network

    Examples
    --------
    ```python
    from edgecase import les_miserables

    G = les_miserables()

    # Returns (77, 254)
    G.count_nodes(), G.count_edges()

    # Returns 36 (Valjean has the most coappearing characters)
    G.get_degree("Valjean")

    # Weighted communities
    communities = G.cmty_louvain(return_dict=True)
    ```

    Notes
    -----
    Knuth, D. E. (1993). *The Stanford GraphBase: A Platform for Combinatorial Computing*.
    Addison-Wesley.
    """
    edges = []
    for entry in _LES_MISERABLES.replace("\n", ",").split(","):
        if entry.strip():
            u, v, weight = entry.split()
            edges.append((u, v, float(weight)))
    return _labeled_graph([], edges, True, "Les Misérables", "les_miserables()")


def web_google() -> Graph:
    """
    Google web graph (released 2002).

    A web graph from Google where nodes represent web pages and edges represent
    hyperlinks between them. This is a commonly used benchmark for PageRank.

    **Expected file**: `data/web_google.mtx`

    Network Properties:

    - Nodes: 875,713
    - Edges: 5,105,039
    - Directed: Yes
    - Weighted: No
    - Avg degree: 11.6

    Returns
    -------
    Graph
        Google web graph
    """
    return _load_dataset_from_file(
        "web_google.mtx", directed=True, weighted=False, graph_name="Google Web Graph"
    )


def soc_livejournal() -> Graph:
    """
    LiveJournal social network.

    A large social network from LiveJournal, a free on-line blogging community.
    Nodes are users and edges represent friendships.

    **Expected file**: `data/soc_livejournal.mtx`

    Network Properties:

    - Nodes: 4,847,571
    - Edges: 68,993,773
    - Directed: Yes
    - Weighted: No
    - Avg degree: 28.5

    Returns
    -------
    Graph
        LiveJournal social network

    Notes
    -----
    This is a very large graph. Requires significant RAM (~10-15GB).
    """
    return _load_dataset_from_file(
        "soc_livejournal.mtx",
        directed=True,
        weighted=False,
        graph_name="LiveJournal Social Network",
    )


def dolphins() -> Graph:
    """
    Dolphin social network.

    Social network of frequent associations between 62 bottlenose dolphins in a community living off
    Doubtful Sound, New Zealand.

    Network Properties:

    - Nodes: 62
    - Edges: 159
    - Directed: No
    - Weighted: No
    - Avg degree: 5.13

    Returns
    -------
    Graph
        Dolphin social network

    Notes
    -----
    This network is often used for community detection benchmarks.
    The network has been observed to split into two communities.
    """
    return _load_dataset_from_file(
        "dolphins.mtx",
        directed=False,
        weighted=False,
        graph_name="Dolphin Social Network",
    )


def football() -> Graph:
    """
    American college football network.

    Network of American football games between Division IA colleges during regular season Fall 2000.
    Nodes represent teams, edges represent games.

    Network Properties:

    - Nodes: 115
    - Edges: 613
    - Directed: No
    - Weighted: No
    - Avg degree: 10.66

    Returns
    -------
    Graph
        College football network

    Notes
    -----
    Teams are divided into conferences, providing natural community structure.
    Excellent dataset for testing community detection algorithms.
    """
    return _load_dataset_from_file(
        "football.mtx",
        directed=False,
        weighted=False,
        graph_name="American College Football",
    )


def polbooks() -> Graph:
    """
    Political books network.

    Network of books about US politics published around the time of the 2004 presidential election
    and sold by Amazon. Edges represent frequent co-purchasing of books by the same buyers.

    Network Properties:

    - Nodes: 105
    - Edges: 441
    - Directed: No
    - Weighted: No
    - Avg degree: 8.40

    Returns
    -------
    Graph
        Political books co-purchasing network

    Notes
    -----
    Books have been classified as liberal, neutral, or conservative.
    The network shows clear community structure along political lines.
    """
    return _load_dataset_from_file(
        "polbooks.mtx",
        directed=False,
        weighted=False,
        graph_name="Political Books",
    )


def jazz() -> Graph:
    """
    Jazz musicians collaboration network.

    Network of jazz musicians where edges represent collaboration on albums.

    Network Properties:

    - Nodes: 198
    - Edges: 2,742
    - Directed: No
    - Weighted: No
    - Avg degree: 27.70

    Returns
    -------
    Graph
        Jazz musicians network

    Notes
    -----
    This is a highly clustered network with small-world properties.
    Good for testing clustering algorithms and visualization.
    """
    return _load_dataset_from_file(
        "jazz.mtx",
        directed=False,
        weighted=False,
        graph_name="Jazz Musicians Network",
    )


def celegans() -> Graph:
    """
    C. Elegans metabolic network.

    Metabolic network of the nematode worm Caenorhabditis elegans. Nodes represent metabolites,
    edges represent metabolic reactions.

    Network Properties:

    - Nodes: 453
    - Edges: 2,041
    - Directed: No
    - Weighted: No
    - Avg degree: 9.01

    Returns
    -------
    Graph
        C. elegans metabolic network

    Notes
    -----
    This metabolic network shows the biochemical pathways in C. elegans.
    Different from the neural network version.
    """
    return _load_dataset_from_file(
        "celegans.mtx",
        directed=False,
        weighted=False,
        graph_name="C. Elegans Metabolic Network",
    )


def power_grid() -> Graph:
    """
    US Power Grid network.

    Topology of the Western States Power Grid of the United States. An undirected, unweighted
    network representing high-voltage transmission lines.

    Network Properties:

    - Nodes: 4,941
    - Edges: 6,594
    - Directed: No
    - Weighted: No
    - Avg degree: 2.67

    Returns
    -------
    Graph
        US power grid network

    Notes
    -----
    Classic example of a sparse infrastructure network with small-world properties.
    Used to study robustness and cascading failures.
    """
    return _load_dataset_from_file(
        "power_grid.mtx",
        directed=False,
        weighted=False,
        graph_name="US Power Grid",
    )


def pgp() -> Graph:
    """
    PGP web of trust network.

    Pretty-Good-Privacy (PGP) encryption software users and their trust relationships. Directed
    edges represent trust/certification relationships between users.

    Network Properties:

    - Nodes: 10,680
    - Edges: 24,316
    - Directed: Yes
    - Weighted: No
    - Avg degree: 4.55

    Returns
    -------
    Graph
        PGP trust network

    Notes
    -----
    Real-world trust network from cryptographic key signing.
    Exhibits power-law degree distribution and strong clustering.
    """
    return _load_dataset_from_file(
        "pgp.mtx",
        directed=True,
        weighted=False,
        graph_name="PGP Trust Network",
    )


def internet() -> Graph:
    """
    Internet Autonomous Systems network.

    Graph of autonomous systems (AS) that comprise the Internet. Nodes represent autonomous systems,
    edges represent BGP connections.

    Network Properties:

    - Nodes: 6,474
    - Edges: 13,895
    - Directed: No
    - Weighted: No
    - Avg degree: 4.29

    Returns
    -------
    Graph
        Internet AS topology network

    Notes
    -----
    Snapshot from January 2, 2000 from University of Oregon Route Views Project.
    """
    return _load_dataset_from_file(
        "internet.mtx",
        directed=False,
        weighted=False,
        graph_name="Internet Autonomous Systems",
    )


def ca_grqc() -> Graph:
    """
    Arxiv GR-QC (General Relativity and Quantum Cosmology) collaboration network.

    A scientific collaboration network from e-print arXiv. Nodes are authors, edges represent
    co-authorship.

    **Expected file**: `data/ca_grqc.mtx`

    Network Properties:

    - Nodes: 5,242
    - Edges: 28,980
    - Directed: No
    - Weighted: No
    - Avg degree: 6.46

    Returns
    -------
    Graph
        Arxiv collaboration network
    """
    return _load_dataset_from_file(
        "ca_grqc.mtx",
        directed=False,
        weighted=False,
        graph_name="Arxiv GR-QC Collaboration Network",
    )


def physics() -> Graph:
    """
    High Energy Physics - Phenomenology collaboration network.

    A scientific collaboration network from e-print arXiv. Nodes are authors, edges represent
    co-authorship on papers in the High Energy Physics - Phenomenology category.

    Network Properties:

    - Nodes: 12,008
    - Edges: 237,010
    - Directed: No
    - Weighted: No
    - Avg degree: 39.47

    Returns
    -------
    Graph
        High Energy Physics collaboration network
    """
    return _load_dataset_from_file(
        "physics.mtx",
        directed=False,
        weighted=False,
        graph_name="High Energy Physics Collaboration Network",
    )


def wiki_vote() -> Graph:
    """
    Wikipedia adminship voting network.

    A directed network of votes for Wikipedia adminship elections. Nodes represent Wikipedia users
    and edges represent votes from one user to another.

    Network Properties:

    - Nodes: 7,115
    - Edges: 103,689
    - Directed: Yes
    - Weighted: No
    - Avg degree: 14.57

    Returns
    -------
    Graph
        Wikipedia voting network

    Notes
    -----
    Includes all Wikipedia voting data from inception to January 2008.
    """
    return _load_dataset_from_file(
        "wiki_vote.mtx",
        directed=True,
        weighted=False,
        graph_name="Wikipedia Voting Network",
    )


def condmat() -> Graph:
    """
    Condensed Matter Physics collaboration network.

    A scientific collaboration network from e-print arXiv. Nodes are authors, edges represent
    co-authorship on papers in the Condensed Matter category.

    Network Properties:

    - Nodes: 23,133
    - Edges: 186,936
    - Directed: No
    - Weighted: No
    - Avg degree: 16.17

    Returns
    -------
    Graph
        Condensed Matter collaboration network
    """
    return _load_dataset_from_file(
        "condmat.mtx",
        directed=False,
        weighted=False,
        graph_name="Condensed Matter Collaboration Network",
    )


def enron() -> Graph:
    """
    Enron email communication network.

    A directed network of email communications within the Enron corporation. Nodes are email
    addresses, edges represent emails sent from one address to another.

    Network Properties:

    - Nodes: 36,692
    - Edges: 367,662
    - Directed: Yes
    - Weighted: No
    - Avg degree: 20.04

    Returns
    -------
    Graph
        Enron email network

    Notes
    -----
    Data from the Enron email corpus made public during the FERC investigation.
    """
    return _load_dataset_from_file(
        "enron.mtx",
        directed=True,
        weighted=False,
        graph_name="Enron Email Network",
    )


def gnutella() -> Graph:
    """
    Gnutella peer-to-peer network snapshot.

    A directed graph of the Gnutella P2P file sharing network from August 8, 2002. Nodes represent
    hosts, edges represent connections between hosts.

    Network Properties:

    - Nodes: 6,301
    - Edges: 20,777
    - Directed: Yes
    - Weighted: No
    - Avg degree: 6.59

    Returns
    -------
    Graph
        Gnutella P2P network
    """
    return _load_dataset_from_file(
        "gnutella.mtx",
        directed=True,
        weighted=False,
        graph_name="Gnutella P2P Network",
    )


def load_dataset(name: str, directed: bool = None, weighted: bool = False) -> Graph:
    """
    Load a dataset by name from the data directory.

    Automatically adds '.mtx' extension if not present. For registered datasets,
    metadata like directed/weighted flags are applied automatically unless overridden.

    Parameters
    ----------
    name
        Name of the dataset file (with or without .mtx extension)
    directed
        Override the graph type. If None, uses registry metadata or auto-detects from file.
    weighted
        Whether to treat edge weights as significant

    Returns
    -------
    Graph
        The loaded graph

    Notes
    -----
    Small datasets (<100KB) are bundled with the package. Larger datasets
    are automatically downloaded on first use from GitHub releases and cached
    at ~/.cache/edgecase/datasets/. Set EDGECASE_NO_DOWNLOAD=1 to disable
    auto-download.
    """
    # Remove .mtx extension if present to check registry
    name_key = name.replace(".mtx", "")

    # Check registry for metadata
    if name_key in DATASET_REGISTRY:
        metadata = DATASET_REGISTRY[name_key]
        filename = metadata["filename"]
        location = metadata.get("location", "bundled")

        # Use registry metadata if not overridden
        if directed is None:
            directed = metadata.get("directed", None)
        graph_name = metadata.get("name", name_key)

        # Handle remote datasets - download if needed
        if location == "remote":
            compressed = metadata.get("compressed")
            if compressed:
                # Check if file exists locally in bundled data (for development)
                bundled_path = DATA_DIR / filename
                cache_path = CACHE_DIR / filename

                if not bundled_path.exists() and not cache_path.exists():
                    # Not bundled and not cached - need to download
                    if not ALLOW_DOWNLOAD:
                        raise RuntimeError(
                            f"Dataset '{name_key}' requires download but "
                            f"EDGECASE_NO_DOWNLOAD=1 is set.\n"
                            f"To enable downloads, unset the environment variable.\n"
                            f"Or manually download from: "
                            f"{DATASET_BASE_URL}/{compressed}"
                        )
                    # Download and decompress
                    _download_dataset(name_key, compressed)
    else:
        # Not in registry - use filename as-is (add .mtx if missing)
        filename = name if name.endswith(".mtx") else f"{name}.mtx"
        graph_name = name_key

    return _load_dataset_from_file(
        filename=filename, directed=directed, weighted=weighted, graph_name=graph_name
    )


__all__ = [
    "karate_club",
    "davis_southern_women",
    "florentine_families",
    "les_miserables",
    "dolphins",
    "football",
    "polbooks",
    "jazz",
    "celegans",
    "power_grid",
    "ca_grqc",
    "physics",
    "condmat",
    "wiki_vote",
    "pgp",
    "internet",
    "enron",
    "gnutella",
    "web_google",
    "soc_livejournal",
    "load_dataset",
    "list_available_datasets",
    "list_registered_datasets",
    "get_dataset_info",
]
