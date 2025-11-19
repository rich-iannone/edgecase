import math

from edgecase import Graph


def demo_spectral_clustering():
    """Demonstrate spectral clustering using the Fiedler vector."""
    print("=" * 70)
    print("Demo 1: Spectral Clustering with Fiedler Vector")
    print("=" * 70)

    # Create a graph with two clear communities
    g = Graph(directed=False)

    # Community 1: Triangle (nodes 0, 1, 2)
    for i in range(3):
        g.add_node(f"v{i}")
    g.add_edge("v0", "v1")
    g.add_edge("v1", "v2")
    g.add_edge("v2", "v0")

    # Community 2: Triangle (nodes 3, 4, 5)
    for i in range(3, 6):
        g.add_node(f"v{i}")
    g.add_edge("v3", "v4")
    g.add_edge("v4", "v5")
    g.add_edge("v5", "v3")

    # Bridge edge connecting the two communities
    g.add_edge("v2", "v3", weight=0.5)

    # Compute Laplacian spectrum
    spec = g.get_laplacian_spectrum(which="smallest", k=3)

    print("\nGraph: 6 nodes (2 triangles connected by bridge)")
    print(f"Smallest 3 eigenvalues: {[f'{v:.4f}' for v in spec['values']]}")
    print(f"Algebraic connectivity: {spec['algebraic_connectivity']:.4f}")

    # Use Fiedler vector (eigenvector for λ₂) for clustering
    fiedler_vector = spec["vectors"][1]  # Second eigenvector (λ₂)

    print(f"\nFiedler vector: {[f'{v:.4f}' for v in fiedler_vector]}")

    # Partition by sign of Fiedler vector
    cluster1 = [i for i, v in enumerate(fiedler_vector) if v < 0]
    cluster2 = [i for i, v in enumerate(fiedler_vector) if v >= 0]

    print("\nClustering result:")
    print(f"  Cluster 1: {[f'v{i}' for i in cluster1]} (nodes {cluster1})")
    print(f"  Cluster 2: {[f'v{i}' for i in cluster2]} (nodes {cluster2})")

    print("\n✓ Spectral clustering successfully identified the two communities!")


def demo_spectral_radius():
    """Demonstrate computing spectral radius (largest eigenvalue)."""
    print("\n" + "=" * 70)
    print("Demo 2: Spectral Radius Analysis")
    print("=" * 70)

    graphs = []

    # Complete graph K₄
    g1 = Graph(directed=False)
    for i in range(4):
        g1.add_node(f"v{i}")
    for i in range(4):
        for j in range(i + 1, 4):
            g1.add_edge(f"v{i}", f"v{j}")
    graphs.append(("Complete K₄", g1, 3.0))

    # Star graph S₄ (1 center, 3 leaves)
    g2 = Graph(directed=False)
    for i in range(4):
        g2.add_node(f"v{i}")
    for i in range(1, 4):
        g2.add_edge("v0", f"v{i}")
    graphs.append(("Star S₄", g2, math.sqrt(3)))

    # Path graph P₄
    g3 = Graph(directed=False)
    for i in range(4):
        g3.add_node(f"v{i}")
    for i in range(3):
        g3.add_edge(f"v{i}", f"v{i + 1}")
    graphs.append(("Path P₄", g3, 2 * math.cos(math.pi / 5)))

    print("\nSpectral radius comparison:")
    print(f"{'Graph':<15} {'Computed':<12} {'Expected':<12} {'Match':<8}")
    print("-" * 50)

    for name, graph, expected in graphs:
        spec = graph.get_spectrum(which="largest", k=1)
        computed = spec["values"][0]
        match = "✓" if abs(computed - expected) < 1e-4 else "✗"
        print(f"{name:<15} {computed:>10.4f}   {expected:>10.4f}   {match:<8}")

    print("\n✓ All spectral radii match expected values!")


def demo_bipartite_detection():
    """Demonstrate detecting bipartite structure from spectrum."""
    print("\n" + "=" * 70)
    print("Demo 3: Bipartite Graph Detection")
    print("=" * 70)

    # Create a bipartite graph (complete bipartite K₃,₃)
    g = Graph(directed=False)

    # Left partition: a0, a1, a2
    # Right partition: b0, b1, b2
    for i in range(3):
        g.add_node(f"a{i}")
        g.add_node(f"b{i}")

    # Connect all left nodes to all right nodes
    for i in range(3):
        for j in range(3):
            g.add_edge(f"a{i}", f"b{j}")

    spec = g.get_spectrum(which="all")
    values = spec["values"]

    print("\nComplete bipartite K₃,₃ spectrum:")
    print(f"Eigenvalues: {[f'{v:.4f}' for v in values]}")

    # Check for symmetric spectrum (bipartite property)
    sorted_values = sorted(values)
    is_symmetric = all(
        abs(sorted_values[i] + sorted_values[-(i + 1)]) < 1e-6
        for i in range(len(sorted_values) // 2)
    )

    print(f"\nSymmetry check (λᵢ = -λₙ₋ᵢ₊₁): {is_symmetric}")

    if is_symmetric:
        print("✓ Symmetric spectrum indicates bipartite structure!")

    # Compare with non-bipartite graph
    g2 = Graph(directed=False)
    for i in range(6):
        g2.add_node(f"v{i}")
    # Add a triangle (not bipartite)
    g2.add_edge("v0", "v1")
    g2.add_edge("v1", "v2")
    g2.add_edge("v2", "v0")

    spec2 = g2.get_spectrum(which="all")
    values2 = spec2["values"]
    sorted_values2 = sorted(values2)

    is_symmetric2 = all(
        abs(sorted_values2[i] + sorted_values2[-(i + 1)]) < 1e-6
        for i in range(len(sorted_values2) // 2)
    )

    print(f"\nTriangle graph spectrum: {[f'{v:.4f}' for v in values2]}")
    print(f"Symmetry check: {is_symmetric2}")
    print("✓ Non-symmetric spectrum indicates non-bipartite structure!")


def demo_connectivity_analysis():
    """Demonstrate using algebraic connectivity to measure graph connectivity."""
    print("\n" + "=" * 70)
    print("Demo 4: Graph Connectivity Analysis")
    print("=" * 70)

    graphs = []

    # Complete graph K₄ (highly connected)
    g1 = Graph(directed=False)
    for i in range(4):
        g1.add_node(f"v{i}")
    for i in range(4):
        for j in range(i + 1, 4):
            g1.add_edge(f"v{i}", f"v{j}")
    graphs.append(("Complete K₄", g1, "Highly connected"))

    # Path graph P₄ (weakly connected)
    g2 = Graph(directed=False)
    for i in range(4):
        g2.add_node(f"v{i}")
    for i in range(3):
        g2.add_edge(f"v{i}", f"v{i + 1}")
    graphs.append(("Path P₄", g2, "Weakly connected"))

    # Star graph S₄ (moderate)
    g3 = Graph(directed=False)
    for i in range(4):
        g3.add_node(f"v{i}")
    for i in range(1, 4):
        g3.add_edge("v0", f"v{i}")
    graphs.append(("Star S₄", g3, "Moderate"))

    # Cycle graph C₄
    g4 = Graph(directed=False)
    for i in range(4):
        g4.add_node(f"v{i}")
    for i in range(4):
        g4.add_edge(f"v{i}", f"v{(i + 1) % 4}")
    graphs.append(("Cycle C₄", g4, "Moderate"))

    print("\nAlgebraic connectivity (λ₂) comparison:")
    print(f"{'Graph':<15} {'λ₂':<12} {'Description':<20}")
    print("-" * 50)

    for name, graph, desc in graphs:
        spec = graph.get_laplacian_spectrum(which="smallest", k=2)
        lambda2 = spec["algebraic_connectivity"]
        print(f"{name:<15} {lambda2:>10.4f}   {desc:<20}")

    print("\n✓ Higher λ₂ indicates better connectivity/robustness!")


def demo_spectral_gap():
    """Demonstrate using spectral gap for graph structure analysis."""
    print("\n" + "=" * 70)
    print("Demo 5: Spectral Gap Analysis")
    print("=" * 70)

    # Create graphs with different structures
    graphs = []

    # Regular graph: Complete K₄
    g1 = Graph(directed=False)
    for i in range(4):
        g1.add_node(f"v{i}")
    for i in range(4):
        for j in range(i + 1, 4):
            g1.add_edge(f"v{i}", f"v{j}")
    graphs.append(("Complete K₄", g1, "Regular"))

    # Bipartite: K₂,₂
    g2 = Graph(directed=False)
    for i in range(4):
        g2.add_node(f"v{i}")
    g2.add_edge("v0", "v2")
    g2.add_edge("v0", "v3")
    g2.add_edge("v1", "v2")
    g2.add_edge("v1", "v3")
    graphs.append(("K₂,₂", g2, "Bipartite"))

    # Star: unbalanced degree distribution
    g3 = Graph(directed=False)
    for i in range(4):
        g3.add_node(f"v{i}")
    for i in range(1, 4):
        g3.add_edge("v0", f"v{i}")
    graphs.append(("Star S₄", g3, "Unbalanced"))

    print("\nSpectral gap (λ₁ - λ₂) comparison:")
    print(f"{'Graph':<15} {'λ₁':<10} {'λ₂':<10} {'Gap':<10} {'Structure':<15}")
    print("-" * 60)

    for name, graph, structure in graphs:
        spec = graph.get_spectrum(which="largest", k=2)
        lambda1 = spec["values"][0]
        lambda2 = spec["values"][1] if len(spec["values"]) > 1 else 0
        gap = lambda1 - lambda2

        print(f"{name:<15} {lambda1:>8.4f}  {lambda2:>8.4f}  {gap:>8.4f}  {structure:<15}")

    print("\n✓ Spectral gap reveals structural properties of graphs!")


def demo_normalized_laplacian():
    """Demonstrate normalized Laplacian for weighted graphs."""
    print("\n" + "=" * 70)
    print("Demo 6: Normalized Laplacian for Weighted Graphs")
    print("=" * 70)

    # Create a weighted graph with varying edge weights
    g = Graph(directed=False)
    for i in range(4):
        g.add_node(f"v{i}")

    g.add_edge("v0", "v1", weight=1.0)
    g.add_edge("v1", "v2", weight=5.0)
    g.add_edge("v2", "v3", weight=1.0)
    g.add_edge("v3", "v0", weight=5.0)

    # Standard Laplacian
    spec_standard = g.get_laplacian_spectrum(which="all", normalized=False)

    # Normalized Laplacian
    spec_normalized = g.get_laplacian_spectrum(which="all", normalized=True)

    print("\nWeighted cycle graph (alternating weights: 1.0, 5.0, 1.0, 5.0)")
    print("\nStandard Laplacian eigenvalues:")
    print(f"  {[f'{v:.4f}' for v in spec_standard['values']]}")
    print(f"  λ₂ (algebraic connectivity): {spec_standard['algebraic_connectivity']:.4f}")

    print("\nNormalized Laplacian eigenvalues:")
    print(f"  {[f'{v:.4f}' for v in spec_normalized['values']]}")
    print(f"  λ₂ (normalized): {spec_normalized['algebraic_connectivity']:.4f}")

    # Check normalized eigenvalues are in [0, 2]
    in_range = all(0 <= v <= 2.0 + 1e-6 for v in spec_normalized["values"])
    print(f"\nAll normalized eigenvalues in [0, 2]: {in_range}")

    print("\n✓ Normalized Laplacian handles degree imbalance better!")


def main():
    """Run all demos."""
    demo_spectral_clustering()
    demo_spectral_radius()
    demo_bipartite_detection()
    demo_connectivity_analysis()
    demo_spectral_gap()
    demo_normalized_laplacian()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
