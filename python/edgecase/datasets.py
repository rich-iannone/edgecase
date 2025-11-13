import polars as pl

from edgecase.graph import Graph


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
    betweenness = G.get_betweenness()
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
    for node_id, score in top_nodes:
        label = G.label_from_id(node_id)
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


__all__ = [
    "karate_club",
]
