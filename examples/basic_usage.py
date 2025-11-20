import edgecase as ec
import polars as pl


def main() -> None:
    """Run basic examples."""

    # Example 1: Create a simple graph
    print("Creating a simple social network...")

    edges = pl.DataFrame(
        {
            "from": ["Alice", "Bob", "Carol", "Dave"],
            "to": ["Bob", "Carol", "Dave", "Alice"],
            "weight": [10, 5, 8, 3],
        }
    )

    G = ec.Graph(
        edges_df=edges,
        directed=False,
        weighted=True,
    )

    print(f"Network: {G.count_nodes()} people, {G.count_edges()} connections")

    # Example 2: Describe the network
    print("\nNetwork description:")
    print(G.describe())

    # Example 3: Calculate PageRank
    print("\nPageRank scores:")
    pagerank_scores = G.pagerank(return_dict=True)
    for person, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {person}: {score:.4f}")

    # Example 4: Find shortest path
    print("\nFinding shortest path from Alice to Carol...")
    path = G.shortest_path("Alice", "Carol", return_list=True)
    if path:
        # Convert node IDs to labels for display
        path_labels = [G.get_label_from_id(nid) or str(nid) for nid in path]
        print(f"Path: {' -> '.join(path_labels)}")


if __name__ == "__main__":
    main()
