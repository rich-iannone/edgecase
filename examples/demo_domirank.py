import edgecase as ec


def main():
    print("=" * 70)
    print("DomiRank Centrality Demonstration")
    print("=" * 70)
    print()

    # Example 1: Star graph - hub dominance
    print("Example 1: Star Graph (Hub with 5 leaves)")
    print("-" * 70)
    star = ec.Graph()
    star.add_node("hub")
    for i in range(5):
        leaf = f"leaf_{i}"
        star.add_node(leaf)
        star.add_edge("hub", leaf)

    dr_star = star.get_domirank_centrality()
    print(f"Hub dominance: {dr_star['hub']:.4f}")
    leaf_scores = [f"{dr_star[f'leaf_{i}']:.4f}" for i in range(5)]
    print(f"Leaf dominances: {leaf_scores}")
    print("\nKey insight: Hub has high positive dominance.")
    print("Leaves have negative dominance (submissive to hub).\n")

    # Example 2: Path graph - center dominance
    print("Example 2: Path Graph A-B-C-D-E")
    print("-" * 70)
    path = ec.Graph()
    nodes = ["A", "B", "C", "D", "E"]
    for node in nodes:
        path.add_node(node)
    for i in range(len(nodes) - 1):
        path.add_edge(nodes[i], nodes[i + 1])

    dr_path, sigma_used = path.get_domirank_centrality(return_sigma=True)
    print(f"Auto-computed sigma: {sigma_used:.6f}\n")
    for node in nodes:
        print(f"  {node}: {dr_path[node]:8.4f}")
    print("\nKey insight: Center node C has highest dominance.")
    print("Endpoints A and E have lowest (similar) dominance.\n")

    # Example 3: Sigma parameter effects
    print("Example 3: Sigma Parameter Effects (Simple Triangle)")
    print("-" * 70)
    triangle = ec.Graph()
    triangle.add_node("A")
    triangle.add_node("B")
    triangle.add_node("C")
    triangle.add_edge("A", "B")
    triangle.add_edge("B", "C")
    triangle.add_edge("C", "A")

    # Low sigma (local, degree-like)
    dr_low = triangle.get_domirank_centrality(sigma=0.01)
    print("Low sigma (σ=0.01) - Local information:")
    for node in ["A", "B", "C"]:
        print(f"  {node}: {dr_low[node]:8.4f}")

    # Medium sigma (auto-optimized)
    dr_auto, sigma_auto = triangle.get_domirank_centrality(return_sigma=True)
    print(f"\nAuto sigma (σ={sigma_auto:.4f}) - Balanced:")
    for node in ["A", "B", "C"]:
        print(f"  {node}: {dr_auto[node]:8.4f}")

    print("\nKey insight: Low sigma → degree-like behavior.")
    print("Symmetric graphs (like triangles) have equal dominance.\n")

    # Example 4: Comparing centrality measures
    print("Example 4: DomiRank vs Degree Centrality")
    print("-" * 70)
    g = ec.Graph()
    # Create a graph where degree and dominance differ
    # Two hubs connected to different neighborhoods
    g.add_node("hub1")
    g.add_node("hub2")
    g.add_edge("hub1", "hub2")

    # Hub1 has many leaves (3)
    for i in range(3):
        leaf = f"leaf1_{i}"
        g.add_node(leaf)
        g.add_edge("hub1", leaf)

    # Hub2 has few leaves (1) but connects to another well-connected node
    g.add_node("leaf2_0")
    g.add_edge("hub2", "leaf2_0")

    dr = g.get_domirank_centrality()
    print("Node          Degree    DomiRank")
    print("-" * 40)
    for node in ["hub1", "hub2", "leaf1_0", "leaf2_0"]:
        deg = g.get_degree(node)
        print(f"{node:12s}  {deg:3d}      {dr[node]:8.4f}")

    print("\nKey insight: DomiRank captures competition dynamics.")
    print("Hub1 has more leaves → higher dominance.")
    print("Leaves are submissive regardless of hub degree.\n")

    # Example 5: Fragility analysis (sequential node removal)
    print("Example 5: Fragility Analysis - Identifying Critical Nodes")
    print("-" * 70)
    print("DomiRank can identify structurally critical nodes for fragility analysis.")
    print("Removing high-dominance nodes typically has larger network impact.\n")

    # Create a small network
    frag = ec.Graph()
    frag.add_node("central")
    for i in range(3):
        branch_hub = f"branch_{i}"
        frag.add_node(branch_hub)
        frag.add_edge("central", branch_hub)
        # Each branch has 2 leaves
        for j in range(2):
            leaf = f"leaf_{i}_{j}"
            frag.add_node(leaf)
            frag.add_edge(branch_hub, leaf)

    dr_frag = frag.get_domirank_centrality()

    # Find top 3 most dominant nodes
    top_nodes = sorted(dr_frag.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 most dominant nodes (critical for network stability):")
    for rank, (node, score) in enumerate(top_nodes, 1):
        print(f"  {rank}. {node:15s} - dominance: {score:8.4f}")

    print("\nKey insight: 'central' node has highest dominance.")
    print("Removing it would fragment the network into isolated branches.\n")

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("DomiRank centrality provides:")
    print("  • Competition-based importance (not just connectivity)")
    print("  • Tunable local/global tradeoff via sigma parameter")
    print("  • Positive scores for dominant nodes, negative for submissive")
    print("  • Useful for fragility analysis and identifying critical nodes")
    print("\nReference: Engsig et al., Nature Communications 15, 56 (2024)")
    print("  https://doi.org/10.1038/s41467-023-44257-0")
    print("=" * 70)


if __name__ == "__main__":
    main()
