import edgecase as ec


def example_1_basic_triangle_counting():
    """Example 1: Count triangles in a simple graph."""
    print("=" * 70)
    print("Example 1: Basic Triangle Counting")
    print("=" * 70)

    # Create a simple social network
    g = ec.Graph()
    g.add_edges(
        [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Alice"),  # Forms triangle
            ("Alice", "David"),
            ("David", "Eve"),
        ]
    )

    # Count triangles per node
    counts = g.count_triangles()
    print("\nTriangle counts per node:")
    for node, count in sorted(counts.items()):
        print(f"  {node:10s}: {count} triangle(s)")

    # List all triangles
    triangles = g.list_triangles()
    print(f"\nTotal triangles found: {len(triangles)}")
    for i, tri in enumerate(triangles, 1):
        print(f"  Triangle {i}: {tri}")

    print()


def example_2_clustering_coefficients():
    """Example 2: Compute clustering coefficients."""
    print("=" * 70)
    print("Example 2: Clustering Coefficients")
    print("=" * 70)

    # Create a network with varying clustering
    g = ec.Graph()
    g.add_edges(
        [
            # Tight cluster: Alice, Bob, Charlie (triangle)
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Alice"),
            # Hub with no clustering: David connects to Alice, Bob, Charlie
            # but they're already connected
            ("David", "Alice"),
            ("David", "Bob"),
            ("David", "Charlie"),
            # Isolated pair
            ("Eve", "Frank"),
        ]
    )

    # Global transitivity
    trans = g.get_transitivity()
    print(f"\nGlobal transitivity: {trans:.4f}")
    print("  (Fraction of connected triples that form triangles)")

    # Local clustering per node
    clustering = g.local_clustering_coefficient(return_dict=True)
    print("\nLocal clustering coefficients:")
    for node in ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]:
        coeff = clustering[node]
        if coeff is None:
            print(f"  {node:10s}: None (degree < 2)")
        else:
            print(f"  {node:10s}: {coeff:.4f}")

    print("\nInterpretation:")
    print("  - Alice, Bob, Charlie: 0.5 (half their neighbors are connected)")
    print("  - David: 1.0 (all neighbors form triangles)")
    print("  - Eve, Frank: None (degree too low)")
    print()


def example_3_social_network_analysis():
    """Example 3: Analyze a realistic social network."""
    print("=" * 70)
    print("Example 3: Social Network Analysis")
    print("=" * 70)

    # Karate club network (simplified)
    g = ec.Graph()

    # Core members (highly interconnected)
    core_edges = [
        ("Leader", "Member1"),
        ("Leader", "Member2"),
        ("Leader", "Member3"),
        ("Member1", "Member2"),
        ("Member1", "Member3"),
        ("Member2", "Member3"),
    ]

    # Bridge members (connect core to periphery)
    bridge_edges = [
        ("Member3", "Bridge1"),
        ("Member3", "Bridge2"),
        ("Bridge1", "Bridge2"),
    ]

    # Peripheral members (loosely connected)
    peripheral_edges = [
        ("Bridge2", "Peripheral1"),
        ("Bridge2", "Peripheral2"),
        ("Peripheral1", "Peripheral3"),
        ("Peripheral2", "Peripheral3"),
    ]

    g.add_edges(core_edges + bridge_edges + peripheral_edges)

    # Analyze structure
    print("\nNetwork structure analysis:")
    print(f"  Nodes: {g.count_nodes()}")
    print(f"  Edges: {g.count_edges()}")
    print(f"  Density: {g.get_density():.4f}")

    # Triangle analysis
    triangle_counts = g.count_triangles()
    total_triangles = sum(triangle_counts.values()) // 3
    print(f"\n  Total triangles: {total_triangles}")
    print(f"  Global transitivity: {g.get_transitivity():.4f}")

    # Identify highly clustered nodes (potential community cores)
    clustering = g.local_clustering_coefficient(return_dict=True)
    highly_clustered = [
        (node, coeff) for node, coeff in clustering.items() if coeff and coeff > 0.5
    ]
    highly_clustered.sort(key=lambda x: x[1], reverse=True)

    print("\n  Highly clustered nodes (clustering > 0.5):")
    for node, coeff in highly_clustered[:5]:
        print(f"    {node:15s}: {coeff:.4f}")

    # Identify bridge nodes (many triangles, lower clustering)
    triangle_rich = [(node, count) for node, count in triangle_counts.items() if count >= 2]
    triangle_rich.sort(key=lambda x: x[1], reverse=True)

    print("\n  Triangle-rich nodes (>= 2 triangles):")
    for node, count in triangle_rich[:5]:
        clust = clustering[node]
        clust_str = f"{clust:.4f}" if clust else "None"
        print(f"    {node:15s}: {count} triangles, clustering={clust_str}")

    print()


def example_4_compare_network_types():
    """Example 4: Compare clustering in different network types."""
    print("=" * 70)
    print("Example 4: Clustering in Different Network Types")
    print("=" * 70)

    # 1. Complete graph (fully connected)
    complete = ec.Graph()
    nodes = ["A", "B", "C", "D", "E"]
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1 :]:
            complete.add_edge(n1, n2)

    # 2. Star graph (hub-and-spoke)
    star = ec.Graph()
    for i in range(5):
        star.add_edge("Hub", f"Leaf{i}")

    # 3. Chain graph (linear)
    chain = ec.Graph()
    for i in range(4):
        chain.add_edge(f"Node{i}", f"Node{i + 1}")

    # 4. Ring graph (cycle)
    ring = ec.Graph()
    nodes = ["A", "B", "C", "D", "E"]
    for i in range(len(nodes)):
        ring.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

    # Compare clustering
    graphs = [
        ("Complete (K5)", complete),
        ("Star", star),
        ("Chain", chain),
        ("Ring (C5)", ring),
    ]

    print("\nClustering comparison:")
    print(
        f"{'Graph Type':<20s} {'Nodes':>6s} {'Edges':>6s} {'Triangles':>10s} {'Transitivity':>12s}"
    )
    print("-" * 70)

    for name, g in graphs:
        triangle_counts = g.count_triangles()
        total_tri = sum(triangle_counts.values()) // 3
        trans = g.get_transitivity()
        trans_str = f"{trans:.4f}" if trans is not None else "None"

        print(
            f"{name:<20s} {g.count_nodes():>6d} {g.count_edges():>6d} {total_tri:>10d} {trans_str:>12s}"
        )

    print("\nObservations:")
    print("  - Complete: Maximum clustering (every triple is closed)")
    print("  - Star: No clustering (hub's neighbors not connected)")
    print("  - Chain: No triangles possible in tree structure")
    print("  - Ring: No triangles (each node has exactly 2 neighbors)")
    print()


def example_5_fraud_detection_pattern():
    """Example 5: Use triangles to detect suspicious patterns."""
    print("=" * 70)
    print("Example 5: Triangle-Based Fraud Detection")
    print("=" * 70)

    # Transaction network where edges represent money transfers
    g = ec.Graph(directed=True)

    # Normal transactions (linear chains)
    g.add_edges(
        [
            ("Alice", "Merchant1"),
            ("Bob", "Merchant2"),
            ("Charlie", "Merchant1"),
        ]
    )

    # Suspicious circular transaction pattern (triangle)
    g.add_edges(
        [
            ("Suspect1", "Suspect2"),
            ("Suspect2", "Suspect3"),
            ("Suspect3", "Suspect1"),  # Money returns to origin!
        ]
    )

    # Another circular pattern
    g.add_edges(
        [
            ("Account_A", "Account_B"),
            ("Account_B", "Account_C"),
            ("Account_C", "Account_A"),
        ]
    )

    # Detect triangles (potential circular money flow)
    triangles = g.list_triangles()
    triangle_counts = g.count_triangles()

    print("\nTransaction network analysis:")
    print(f"  Total accounts: {g.count_nodes()}")
    print(f"  Total transactions: {g.count_edges()}")

    print(f"\n  Detected circular patterns (triangles): {len(triangles)}")
    for i, tri in enumerate(triangles, 1):
        print(f"    Pattern {i}: {' → '.join(tri)} → {tri[0]}")

    # Flag accounts involved in multiple circular patterns
    suspicious = [(node, count) for node, count in triangle_counts.items() if count > 0]
    suspicious.sort(key=lambda x: x[1], reverse=True)

    print("\n  Accounts in circular patterns:")
    for node, count in suspicious:
        print(f"    {node:15s}: involved in {count} circular pattern(s)")

    print("\n  ⚠️  These accounts should be flagged for review")
    print()


def example_6_recommendation_patterns():
    """Example 6: Use triangles for collaborative filtering."""
    print("=" * 70)
    print("Example 6: Triangle-Based Recommendations")
    print("=" * 70)

    # Bipartite-like graph: users and items
    g = ec.Graph()

    # User-item connections (who likes what)
    g.add_edges(
        [
            # Alice's preferences
            ("Alice", "Movie_A"),
            ("Alice", "Movie_B"),
            ("Alice", "Movie_C"),
            # Bob's preferences (overlaps with Alice)
            ("Bob", "Movie_A"),
            ("Bob", "Movie_B"),
            ("Bob", "Movie_D"),  # Bob also likes Movie_D
            # Charlie's preferences (similar to Alice and Bob)
            ("Charlie", "Movie_A"),
            ("Charlie", "Movie_C"),
            ("Charlie", "Movie_D"),
        ]
    )

    # Find triangles to identify recommendation patterns
    # Triangle (Alice, Bob, Movie_A) means both like Movie_A
    triangles = g.list_triangles()
    triangle_counts = g.count_triangles()

    print("\nUser-Item network:")
    print(f"  Nodes (users + items): {g.count_nodes()}")
    print(f"  Edges (likes): {g.count_edges()}")

    print(f"\n  Common preference patterns (triangles): {len(triangles)}")

    # Items in many triangles are popular
    item_triangles = {
        node: count for node, count in triangle_counts.items() if node.startswith("Movie_")
    }
    popular_items = sorted(item_triangles.items(), key=lambda x: x[1], reverse=True)

    print("\n  Popular items (many triangles = many user pairs like it):")
    for item, count in popular_items:
        print(f"    {item:15s}: {count} user pairs")

    # Users in many triangles have diverse tastes
    user_triangles = {
        node: count for node, count in triangle_counts.items() if not node.startswith("Movie_")
    }
    active_users = sorted(user_triangles.items(), key=lambda x: x[1], reverse=True)

    print("\n  Active users (preferences overlap with others):")
    for user, count in active_users:
        print(f"    {user:15s}: {count} shared preferences")

    print("\n  Recommendation strategy:")
    print("    - Recommend Movie_D to Alice (Bob and Charlie like it)")
    print("    - Recommend Movie_C to Bob (Alice and Charlie like it)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Triangle Counting and Clustering Analysis Demo")
    print("=" * 70)
    print()

    example_1_basic_triangle_counting()
    example_2_clustering_coefficients()
    example_3_social_network_analysis()
    example_4_compare_network_types()
    example_5_fraud_detection_pattern()
    example_6_recommendation_patterns()

    print("=" * 70)
    print("Demo completed! 🎉")
    print("=" * 70)
