import edgecase as ec
from edgecase import shortest_path


def demo_basic_usage():
    """Demonstrate basic usage of get_all_simple_paths() method."""
    print("=" * 70)
    print("DEMO 1: Basic Usage - Finding All Paths")
    print("=" * 70)

    # Create a diamond-shaped network
    graph = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("A", "C")
        .add_edge("B", "D")
        .add_edge("C", "D")
    )

    print("\nNetwork structure:")
    print("    A")
    print("   / \\")
    print("  B   C")
    print("   \\ /")
    print("    D")

    paths = graph.get_all_simple_paths("A", "D")
    print(f"\nFound {len(paths)} paths from A to D:")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' -> '.join(path)}")

    print("\n✓ Both alternative routes discovered")


def demo_network_redundancy():
    """Demonstrate analyzing network redundancy."""
    print("\n" + "=" * 70)
    print("DEMO 2: Network Redundancy Analysis")
    print("=" * 70)

    # Create a network with redundant connections
    graph = (
        ec.Graph()
        .add_node(label="Server1")
        .add_node(label="Router1")
        .add_node(label="Router2")
        .add_node(label="Router3")
        .add_node(label="Server2")
        .add_edge("Server1", "Router1")
        .add_edge("Server1", "Router2")
        .add_edge("Router1", "Router3")
        .add_edge("Router2", "Router3")
        .add_edge("Router3", "Server2")
    )

    print("\nData center network:")
    print("  Server1 ------- Router1")
    print("      |             |")
    print("      |             |")
    print("  Router2 ------- Router3 ----- Server2")

    paths = graph.get_all_simple_paths("Server1", "Server2")
    print(f"\nRedundancy analysis: {len(paths)} independent paths available")

    for i, path in enumerate(paths, 1):
        hops = len(path) - 1
        print(f"  Route {i} ({hops} hops): {' -> '.join(path)}")

    print(f"\n✓ Network has {len(paths)}x redundancy - resilient to failures")


def demo_cutoff_parameter():
    """Demonstrate using cutoff to limit path length."""
    print("\n" + "=" * 70)
    print("DEMO 3: Path Length Limitation with Cutoff")
    print("=" * 70)

    # Create a more complex network
    graph = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
        .add_edge("D", "E")
        .add_edge("A", "C")  # Shortcut
        .add_edge("C", "E")  # Another shortcut
    )

    print("\nNetwork with shortcuts:")
    print("  A -> B -> C -> D -> E")
    print("   \\      / \\      /")
    print("    \\____/   \\____/")

    # Find all paths without limit
    all_paths = graph.get_all_simple_paths("A", "E")
    print(f"\nAll paths (no limit): {len(all_paths)} found")
    for path in all_paths:
        print(f"  {' -> '.join(path)} ({len(path) - 1} edges)")

    # Find only short paths
    short_paths = graph.get_all_simple_paths("A", "E", cutoff=2)
    print(f"\nShort paths (cutoff=2): {len(short_paths)} found")
    for path in short_paths:
        print(f"  {' -> '.join(path)} ({len(path) - 1} edges)")

    print("\n✓ Cutoff parameter helps limit search to practical routes")


def demo_vs_shortest_path():
    """Compare all_simple_paths with shortest_path."""
    print("\n" + "=" * 70)
    print("DEMO 4: Comparison with Shortest Path")
    print("=" * 70)

    # Create network where shortest path might not be best
    graph = (
        ec.Graph()
        .add_node(label="Office")
        .add_node(label="Highway")
        .add_node(label="LocalRd")
        .add_node(label="Bridge")
        .add_node(label="Home")
        .add_edge("Office", "Highway")
        .add_edge("Highway", "Home")  # Fast but single route
        .add_edge("Office", "LocalRd")
        .add_edge("LocalRd", "Bridge")
        .add_edge("Bridge", "Home")  # Slower but alternative
    )

    print("\nCommute network:")
    print("  Office ---- Highway ---- Home")
    print("    |")
    print("  LocalRd ---- Bridge --/")

    # Find shortest path
    shortest = shortest_path(graph, "Office", "Home")
    print(f"\nShortest path: {' -> '.join(shortest)} ({len(shortest) - 1} hops)")

    # Find all paths
    all_routes = graph.get_all_simple_paths("Office", "Home")
    print("\nAll alternative routes:")
    for i, route in enumerate(all_routes, 1):
        print(f"  Route {i}: {' -> '.join(route)} ({len(route) - 1} hops)")

    print(f"\n✓ All paths shows {len(all_routes) - 1} backup route(s)")
    print("  (Useful if highway is congested!)")


def demo_directed_graph():
    """Demonstrate behavior with directed graphs."""
    print("\n" + "=" * 70)
    print("DEMO 5: Directed Graph - One-Way Streets")
    print("=" * 70)

    # Create directed graph representing one-way streets
    graph = ec.Graph(directed=True)
    graph.add_node(label="A")
    graph.add_node(label="B")
    graph.add_node(label="C")
    graph.add_node(label="D")

    # One-way streets
    graph.add_edge("A", "B")  # A -> B only
    graph.add_edge("B", "C")  # B -> C only
    graph.add_edge("C", "D")  # C -> D only
    graph.add_edge("A", "C")  # Shortcut: A -> C only
    graph.add_edge("D", "A")  # Return route: D -> A only

    print("\nOne-way street network:")
    print("  A → B → C → D")
    print("  ↑    ↘     ↓")
    print("  └───────────┘")

    # Find paths A to D (should work)
    paths_forward = graph.get_all_simple_paths("A", "D")
    print(f"\nPaths from A to D: {len(paths_forward)} found")
    for path in paths_forward:
        print(f"  {' → '.join(path)}")

    # Find paths D to A (should find different paths)
    paths_backward = graph.get_all_simple_paths("D", "A")
    print(f"\nPaths from D to A: {len(paths_backward)} found")
    for path in paths_backward:
        print(f"  {' → '.join(path)}")

    print("\n✓ Directed graphs respect edge direction (one-way streets)")


def demo_social_network():
    """Demonstrate use case in social network analysis."""
    print("\n" + "=" * 70)
    print("DEMO 6: Social Network - Connection Paths")
    print("=" * 70)

    # Create social network
    graph = (
        ec.Graph()
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_node(label="Carol")
        .add_node(label="Dave")
        .add_node(label="Eve")
        .add_edge("Alice", "Bob")
        .add_edge("Alice", "Carol")
        .add_edge("Bob", "Dave")
        .add_edge("Carol", "Dave")
        .add_edge("Dave", "Eve")
        .add_edge("Carol", "Eve")
    )

    print("\nSocial network:")
    print("  Alice knows Bob and Carol")
    print("  Bob knows Dave")
    print("  Carol knows Dave and Eve")
    print("  Dave knows Eve")

    # Find all connection paths
    paths = graph.get_all_simple_paths("Alice", "Eve")
    print(f"\nHow Alice can reach Eve: {len(paths)} paths")
    for i, path in enumerate(paths, 1):
        if len(path) == 2:
            print(f"  Path {i}: Direct connection via {path[1]}")
        else:
            intermediaries = path[1:-1]
            print(f"  Path {i}: Through {' and '.join(intermediaries)}")
            print(f"         {' -> '.join(path)}")

    # Analyze connection strength
    print(f"\n✓ {len(paths)} independent connection paths")
    print(f"  → Alice and Eve have {len(paths)}-degree social connectivity")


def demo_logistics_routes():
    """Demonstrate logistics/delivery route analysis."""
    print("\n" + "=" * 70)
    print("DEMO 7: Logistics - Alternative Delivery Routes")
    print("=" * 70)

    # Create delivery network
    graph = (
        ec.Graph()
        .add_node(label="Warehouse")
        .add_node(label="Hub1")
        .add_node(label="Hub2")
        .add_node(label="Hub3")
        .add_node(label="Store")
        .add_edge("Warehouse", "Hub1")
        .add_edge("Warehouse", "Hub2")
        .add_edge("Hub1", "Hub3")
        .add_edge("Hub2", "Hub3")
        .add_edge("Hub1", "Store")
        .add_edge("Hub3", "Store")
    )

    print("\nDelivery network:")
    print("            Hub1 ----")
    print("           /  |       \\")
    print("  Warehouse   |        Store")
    print("           \\  |       /")
    print("            Hub2 - Hub3")

    paths = graph.get_all_simple_paths("Warehouse", "Store")
    print(f"\nDelivery route options: {len(paths)} available")

    for i, path in enumerate(paths, 1):
        transfers = len(path) - 2  # Exclude start and end
        print(f"  Route {i}: {' -> '.join(path)}")
        print(f"           ({transfers} transfer{'s' if transfers != 1 else ''})")

    print("\n✓ Multiple route options provide flexibility for:")
    print("  • Load balancing across hubs")
    print("  • Avoiding congested routes")
    print("  • Maintaining service during hub maintenance")


def demo_complete_graph():
    """Demonstrate exponential path count in complete graphs."""
    print("\n" + "=" * 70)
    print("DEMO 8: Complete Graph - Exponential Path Growth")
    print("=" * 70)

    # Create small complete graph
    nodes = ["A", "B", "C", "D"]
    graph = ec.Graph()
    for node in nodes:
        graph.add_node(label=node)

    # Connect all pairs
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1 :]:
            graph.add_edge(n1, n2)

    print("\nComplete graph K4 (all nodes connected):")
    print("    A -------- B")
    print("    |\\        /|")
    print("    | \\      / |")
    print("    |  \\    /  |")
    print("    |   \\  /   |")
    print("    |    \\/    |")
    print("    |    /\\    |")
    print("    |   /  \\   |")
    print("    |  /    \\  |")
    print("    | /      \\ |")
    print("    |/        \\|")
    print("    C -------- D")

    # Count paths of different lengths
    all_paths = graph.get_all_simple_paths("A", "D")
    print(f"\nTotal paths from A to D: {len(all_paths)}")

    for length in range(2, 5):
        paths_of_length = [p for p in all_paths if len(p) == length]
        if paths_of_length:
            print(f"\n  Paths with {length - 1} edge(s): {len(paths_of_length)} found")
            for path in paths_of_length:
                print(f"    {' -> '.join(path)}")

    print("\n⚠ Warning: Path count grows exponentially in dense graphs!")
    print("  → Use cutoff parameter to limit search in production")


def main():
    """Run all demo scenarios."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  EdgeCase: graph.get_all_simple_paths() Method Demo               ║")
    print("║  Finding All Simple Paths Between Nodes                           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    demo_basic_usage()
    demo_network_redundancy()
    demo_cutoff_parameter()
    demo_vs_shortest_path()
    demo_directed_graph()
    demo_social_network()
    demo_logistics_routes()
    demo_complete_graph()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
graph.get_all_simple_paths(source, target, cutoff=None) is useful for:

• Network Redundancy: Find backup routes in communication networks
• Route Planning: Discover alternative paths for logistics/navigation
• Social Networks: Analyze indirect connections between people
• Knowledge Graphs: Trace reasoning paths between concepts
• Reliability Analysis: Assess system resilience with multiple paths

Key Parameters:
• source/target: Start and end nodes (by ID or label)
• cutoff: Optional max path length to limit search depth

Performance Notes:
• Uses DFS with backtracking
• Time complexity: O(n!) worst case (exponential)
• Best for: Small-to-medium graphs or with cutoff limit
• Use shortest_path() for single optimal path only

NetworkX Compatibility:
• Based on NetworkX's all_simple_paths()
• Similar API and behavior
• Native Rust performance
    """)

    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
