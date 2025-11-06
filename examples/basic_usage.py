"""Example demonstrating basic EdgeCase usage."""

import polars as pl
import edgecase as ec

def main() -> None:
    """Run basic examples."""
    
    # Example 1: Create a simple graph
    print("Creating a simple social network...")
    
    edges = pl.DataFrame({
        "person_a": ["Alice", "Bob", "Carol", "Dave"],
        "person_b": ["Bob", "Carol", "Dave", "Alice"],
        "interactions": [10, 5, 8, 3],
    })
    
    G = ec.from_polars(
        edges,
        source_col="person_a",
        target_col="person_b",
        weight_col="interactions",
        directed=False,
    )
    
    print(f"Network: {G.num_nodes()} people, {G.num_edges()} connections")
    
    # Example 2: Analyze the network
    print("\nAnalyzing network...")
    analyzer = ec.NetworkAnalyzer(G)
    results = analyzer.analyze()
    
    print(f"PageRank scores:")
    for person, score in sorted(
        results['centrality']['pagerank'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {person}: {score:.4f}")
    
    # Example 3: Find shortest path
    print("\nFinding shortest path from Alice to Carol...")
    path = ec.shortest_path(G, "Alice", "Carol")
    print(f"Path: {' -> '.join(path)}")

if __name__ == "__main__":
    main()
