import os
from pathlib import Path
from typing import Optional

import polars as pl

from edgecase.graph import Graph
from edgecase.io import from_matrix_market

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
        "description": "Scientific collaboration network from arXiv",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_grqc.mtx.gz",
        "size_kb": 200,
        "compressed_kb": 60,
    },
    "erdos_collab": {
        "filename": "erdos_collab.mtx",
        "name": "Erdős Collaboration Network",
        "nodes": 6927,
        "edges": 11850,
        "directed": False,
        "description": "Collaboration network of scientists with Erdős number <= 2",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "erdos_collab.mtx.gz",
        "size_kb": 150,
        "compressed_kb": 40,
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
        "size_mb": 1.5,
        "compressed_kb": 450,
    },
    "ca_hepph": {
        "filename": "ca_hepph.mtx",
        "name": "High Energy Physics Collaboration Network",
        "nodes": 12008,
        "edges": 118521,
        "directed": False,
        "description": "Collaboration network from arXiv High Energy Physics - Phenomenology",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_hepph.mtx.gz",
        "size_mb": 2,
        "compressed_kb": 500,
    },
    "pgp": {
        "filename": "pgp.mtx",
        "name": "PGP Trust Network",
        "nodes": 10680,
        "edges": 24316,
        "directed": True,
        "description": "Pretty-Good-Privacy (PGP) web of trust network",
        "source": "NetworkRepository",
        "location": "bundled",
        "compressed_bundled": "pgp.mtx.gz",
        "size_kb": 300,
        "compressed_kb": 75,
    },
    "internet_as": {
        "filename": "internet_as.mtx",
        "name": "Internet Autonomous Systems",
        "nodes": 22963,
        "edges": 48436,
        "directed": False,
        "description": "Internet topology at autonomous system level from traceroute data",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "internet_as.mtx.gz",
        "size_kb": 600,
        "compressed_kb": 150,
    },
    "ca_condmat": {
        "filename": "ca_condmat.mtx",
        "name": "Condensed Matter Collaboration Network",
        "nodes": 23133,
        "edges": 93497,
        "directed": False,
        "description": "Collaboration network from arXiv Condensed Matter category",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "ca_condmat.mtx.gz",
        "size_mb": 1.5,
        "compressed_kb": 400,
    },
    "email_enron": {
        "filename": "email_enron.mtx",
        "name": "Enron Email Network",
        "nodes": 36692,
        "edges": 183831,
        "directed": True,
        "description": "Email communication network from Enron corpus",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "email_enron.mtx.gz",
        "size_mb": 2.5,
        "compressed_kb": 700,
    },
    "p2p_gnutella": {
        "filename": "p2p_gnutella.mtx",
        "name": "Gnutella P2P Network",
        "nodes": 62586,
        "edges": 147892,
        "directed": True,
        "description": "Gnutella peer-to-peer network snapshot from August 2002",
        "source": "SNAP",
        "location": "bundled",
        "compressed_bundled": "p2p_gnutella.mtx.gz",
        "size_mb": 2,
        "compressed_kb": 550,
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
    name : str
        Dataset name (with or without .mtx extension)

    Returns
    -------
    dict
        Dataset metadata including nodes, edges, description, etc.

    Examples
    --------
    ```python
    info = get_dataset_info("roadnet_pa")
    print(f"{info['name']}: {info['nodes']:,} nodes, {info['edges']:,} edges")
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
    for name in list_registered_datasets():
        info = get_dataset_info(name)
        print(f"{name}: {info['description']}")
    ```
    """
    return list(DATASET_REGISTRY.keys())


def _download_dataset(name: str, compressed_filename: str) -> Path:
    """
    Download a dataset from remote repository and decompress it.

    Downloads to ~/.cache/edgecase/datasets/ and decompresses if needed.

    Parameters
    ----------
    name : str
        Dataset name (for metadata lookup)
    compressed_filename : str
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
    filename : str
        Name of the file (e.g., karate.mtx or karate.mtx.gz)
    directed : bool, optional
        Whether graph is directed (auto-detected from file if None)
    weighted : bool, optional
        Whether graph is weighted (auto-detected from file if None)
    graph_name : str, optional
        Name for the graph

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

    A social network of friendships between 34 members of a karate club at a
    US university in the 1970s. The network became famous when the club split
    into two factions due to a dispute between the administrator ("John A.") and
    instructor ("Mr. Hi"), and the split could be predicted by network analysis.

    **Network Properties:**
    - 34 nodes (club members)
    - 78 edges (friendships)
    - Undirected, unweighted
    - Density: 0.139
    - 2 communities (corresponding to the actual club split)

    **Network Metrics** (from networkrepository.com):
    - Degree: max=17, min=1, avg=4.588
    - Assortativity: -0.476 (not yet implemented)
    - Triangles: 135 total, 3.97 avg per node, 18 max (not yet implemented)
    - Clustering coefficient: 0.571 (average) ✓ implemented
    - Transitivity (fraction of closed triangles): 0.256 (not yet implemented)
    - Maximum k-core: 5 (currently reports 4)

    **Node Attributes:**
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
    print(f"Nodes: {G.num_nodes()}, Edges: {G.num_edges()}")
    # Nodes: 34, Edges: 78

    # Check basic metrics
    n = G.num_nodes()
    e = G.num_edges()
    density = e / (n * (n - 1) / 2)
    print(f"Density: {density:.3f}")
    # Density: 0.139

    # Access node attributes by label
    node_id = G.get_id_from_label("0")
    node = G.get_node(node_id)
    print(f"Mr. Hi's faction: {node.attributes['faction']}")
    # Mr. Hi's faction: 1

    # Analyze community structure
    communities = G.get_cmty_louvain()
    num_communities = len(set(communities.values()))
    print(f"Communities detected: {num_communities}")

    # Find most central members
    betweenness = G.betweenness_centrality()
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
    for node_id, score in top_nodes:
        label = G.get_label_from_id(node_id)
        print(f"Node {label}: {score:.3f}")

    # Check k-core decomposition
    coreness = G.get_coreness()
    max_core = max(coreness.values())
    print(f"Maximum k-core: {max_core}")

    # Analyze clustering
    avg_clustering = G.get_avg_clustering_coefficient()
    print(f"Average clustering coefficient: {avg_clustering:.3f}")
    # Average clustering coefficient: 0.571
    ```

    Notes
    -----
    This is one of the most well-studied networks in social network analysis.
    The network split prediction problem has been validated multiple times,
    and the dataset serves as a standard benchmark for community detection
    algorithms.

    References
    ----------
    Zachary, W. W. (1977). An information flow model for conflict and fission in
    small groups. Journal of Anthropological Research, 33(4), 452-473.
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


# Example large dataset loaders - add your own here!
# These will work once you place the corresponding .mtx files in the data/ directory


def web_google() -> Graph:
    """
    Google web graph (released 2002).

    A web graph from Google where nodes represent web pages and edges represent
    hyperlinks between them. This is a commonly used benchmark for PageRank.

    **Expected file**: `data/web_google.mtx`

    **Network Properties** (from SNAP):
    - Nodes: 875,713
    - Edges: 5,105,039
    - Directed: Yes
    - Weighted: No
    - Avg degree: 11.6

    Returns
    -------
    Graph
        Google web graph

    Examples
    --------
    ```python
    # Load the dataset (requires web-Google.mtx in data/)
    G = web_google()

    # Run PageRank
    ranks = G.pagerank(damping=0.85, max_iter=100)
    top_pages = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 pages by PageRank:")
    for node_id, rank in top_pages:
        print(f"  Node {node_id}: {rank:.6f}")
    ```

    References
    ----------
    J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney.
    Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters.
    Internet Mathematics 6(1) 29--123, 2009.
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

    **Network Properties** (from SNAP):
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

    Social network of frequent associations between 62 bottlenose dolphins
    in a community living off Doubtful Sound, New Zealand.

    **Network Properties**:
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

    References
    ----------
    D. Lusseau et al. The bottlenose dolphin community of Doubtful Sound
    features a large proportion of long-lasting associations.
    Behavioral Ecology and Sociobiology 54, 396-405 (2003).
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

    Network of American football games between Division IA colleges during
    regular season Fall 2000. Nodes represent teams, edges represent games.

    **Network Properties**:
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

    References
    ----------
    M. Girvan and M. E. J. Newman, Community structure in social and
    biological networks, Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002).
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

    Network of books about US politics published around the time of the
    2004 presidential election and sold by Amazon. Edges represent frequent
    co-purchasing of books by the same buyers.

    **Network Properties**:
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

    References
    ----------
    V. Krebs, unpublished, http://www.orgnet.com/
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

    **Network Properties**:
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

    References
    ----------
    P. Gleiser and L. Danon, Adv. Complex Syst. 6, 565 (2003).
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

    Metabolic network of the nematode worm Caenorhabditis elegans.
    Nodes represent metabolites, edges represent metabolic reactions.

    **Network Properties**:
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

    References
    ----------
    NetworkRepository datasets collection.
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

    Topology of the Western States Power Grid of the United States.
    An undirected, unweighted network representing high-voltage transmission lines.

    **Network Properties**:
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

    References
    ----------
    D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998).
    """
    return _load_dataset_from_file(
        "power_grid.mtx",
        directed=False,
        weighted=False,
        graph_name="US Power Grid",
    )


def erdos_collab() -> Graph:
    """
    Erdős collaboration network.

    Collaboration network of scientists with Erdős number <= 2.
    Nodes are mathematicians, edges represent co-authorship on papers.

    **Network Properties**:
    - Nodes: 6,927
    - Edges: 11,850
    - Directed: No
    - Weighted: No
    - Avg degree: 3.42

    Returns
    -------
    Graph
        Erdős collaboration network

    Notes
    -----
    Named after Paul Erdős, one of the most prolific mathematicians.
    Erdős number measures collaborative distance from Erdős.

    References
    ----------
    SNAP datasets collection.
    """
    return _load_dataset_from_file(
        "erdos_collab.mtx",
        directed=False,
        weighted=False,
        graph_name="Erdős Collaboration Network",
    )


def pgp() -> Graph:
    """
    PGP web of trust network.

    Pretty-Good-Privacy (PGP) encryption software users and their trust relationships.
    Directed edges represent trust/certification relationships between users.

    **Network Properties**:
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

    References
    ----------
    NetworkRepository datasets collection.
    """
    return _load_dataset_from_file(
        "pgp.mtx",
        directed=True,
        weighted=False,
        graph_name="PGP Trust Network",
    )


def internet_as() -> Graph:
    """
    Internet autonomous systems network.

    Internet topology at the autonomous system (AS) level from traceroute data.
    Nodes are autonomous systems, edges represent communication paths.

    **Network Properties**:
    - Nodes: 22,963
    - Edges: 48,436
    - Directed: No
    - Weighted: No
    - Avg degree: 4.22

    Returns
    -------
    Graph
        Internet AS topology

    Notes
    -----
    Snapshot of Internet backbone structure. Shows scale-free properties
    with power-law degree distribution (few hubs, many small nodes).

    References
    ----------
    SNAP datasets collection.
    """
    return _load_dataset_from_file(
        "internet_as.mtx",
        directed=False,
        weighted=False,
        graph_name="Internet Autonomous Systems",
    )


def ca_grqc() -> Graph:
    """
    Arxiv GR-QC (General Relativity and Quantum Cosmology) collaboration network.

    A scientific collaboration network from e-print arXiv. Nodes are authors,
    edges represent co-authorship.

    **Expected file**: `data/ca_grqc.mtx`

    **Network Properties**:
    - Nodes: 5,242
    - Edges: 28,980
    - Directed: No
    - Weighted: No
    - Avg degree: 6.46

    Returns
    -------
    Graph
        Arxiv collaboration network

    References
    ----------
    J. Leskovec, J. Kleinberg and C. Faloutsos.
    Graph Evolution: Densification and Shrinking Diameters.
    ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.
    """
    return _load_dataset_from_file(
        "ca_grqc.mtx",
        directed=False,
        weighted=False,
        graph_name="Arxiv GR-QC Collaboration Network",
    )


def load_dataset(name: str, directed: bool = None, weighted: bool = False) -> Graph:
    """
    Load a dataset by name from the data directory.

    Automatically adds '.mtx' extension if not present. For registered datasets,
    metadata like directed/weighted flags are applied automatically unless overridden.

    Parameters
    ----------
    name : str
        Name of the dataset file (with or without .mtx extension)
    directed : bool, optional
        Override the graph type. If None, uses registry metadata or auto-detects from file.
    weighted : bool, default=False
        Whether to treat edge weights as significant

    Returns
    -------
    Graph
        The loaded graph

    Examples
    --------
    ```python
    # Load registered dataset (uses metadata)
    g = load_dataset("karate")

    # Load with overrides
    g = load_dataset("web_google", directed=True)

    # Load custom .mtx file from data/
    g = load_dataset("my_graph.mtx")

    # See what's available
    print("Registered:", list_registered_datasets())
    print("All files:", list_available_datasets())
    ```

    See Also
    --------
    list_registered_datasets : List datasets with metadata
    get_dataset_info : Get metadata for a dataset
    list_available_datasets : List all .mtx files in data/

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
    "dolphins",
    "football",
    "polbooks",
    "jazz",
    "celegans",
    "power_grid",
    "erdos_collab",
    "pgp",
    "internet_as",
    "ca_grqc",
    "web_google",
    "soc_livejournal",
    "load_dataset",
    "list_available_datasets",
    "list_registered_datasets",
    "get_dataset_info",
]
