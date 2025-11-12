import random

from edgecase import Graph


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_1_network_backbone():
    """Extract network backbone from dense graph."""
    print_section("Demo 1: Network Backbone Extraction")

    # Create a dense communication network
    g = Graph()
    nodes = ["A", "B", "C", "D", "E", "F"]
    for node in nodes:
        g.add_node(label=node)

    # Add many redundant connections
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "E"),
        ("C", "D"),
        ("C", "E"),
        ("C", "F"),
        ("D", "F"),
        ("E", "F"),
    ]
    for u, v in edges:
        g.add_edge(u, v)

    print(f"Original network: {g.num_nodes()} nodes, {g.num_edges()} edges")

    # Extract 2-spanner backbone
    backbone = g.transform_with_spanner(stretch=2.0, seed=42)
    print(f"2-spanner backbone: {backbone.num_nodes()} nodes, {backbone.num_edges()} edges")
    print(f"Reduction: {g.num_edges() - backbone.num_edges()} edges removed")
    print(f"({(g.num_edges() - backbone.num_edges()) / g.num_edges() * 100:.1f}% reduction)")


def demo_2_graph_compression():
    """Compress large graph while preserving approximate distances."""
    print_section("Demo 2: Graph Compression")

    # Create a moderately large random graph
    random.seed(42)
    g = Graph()
    n = 30
    p = 0.15  # Edge probability

    for i in range(n):
        g.add_node(label=str(i))

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g.add_edge(str(i), str(j))

    print(f"Original graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

    # Create spanners with different stretch values
    for stretch in [2.0, 3.0, 5.0]:
        spanner = g.transform_with_spanner(stretch=stretch, seed=42)
        reduction = (g.num_edges() - spanner.num_edges()) / g.num_edges() * 100
        print(f"{stretch}-spanner: {spanner.num_edges()} edges ({reduction:.1f}% reduction)")


def demo_3_stretch_comparison():
    """Compare different stretch values."""
    print_section("Demo 3: Stretch Value Comparison")

    # Create a complete graph K8
    g = Graph()
    for i in range(8):
        g.add_node(label=str(i))

    for i in range(8):
        for j in range(i + 1, 8):
            g.add_edge(str(i), str(j))

    print(f"Complete graph K8: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print("\nSpanner sizes for different stretch values:")
    print(f"{'Stretch':<10} {'Edges':<10} {'Reduction':<15}")
    print("-" * 40)

    for stretch in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        spanner = g.transform_with_spanner(stretch=stretch, seed=42)
        reduction = (g.num_edges() - spanner.num_edges()) / g.num_edges() * 100
        print(f"{stretch:<10.1f} {spanner.num_edges():<10} {reduction:<15.1f}%")


def demo_4_weighted_graph():
    """Spanner of weighted graph."""
    print_section("Demo 4: Weighted Graph Spanner")

    # Create a weighted transportation network
    g = Graph()
    cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix"]
    for city in cities:
        g.add_node(label=city)

    # Add edges with travel costs (weighted)
    routes = [
        ("NYC", "Chicago", 800),
        ("NYC", "Houston", 1600),
        ("Chicago", "LA", 2000),
        ("Chicago", "Houston", 1000),
        ("Chicago", "Phoenix", 1500),
        ("LA", "Phoenix", 400),
        ("Houston", "Phoenix", 1200),
    ]

    for u, v, cost in routes:
        g.add_edge(u, v, weight=float(cost))

    print(f"Transportation network: {g.num_nodes()} cities, {g.num_edges()} routes")
    print("\nOriginal routes:")
    for u, v, cost in routes:
        print(f"  {u} - {v}: ${cost}")

    # Create 1.5-spanner (at most 1.5x original cost)
    spanner = g.transform_with_spanner(stretch=1.5, weight="weight", seed=42)
    print(f"\n1.5-spanner: {spanner.num_edges()} routes")
    print(f"Reduction: {g.num_edges() - spanner.num_edges()} routes removed")
    print("(Guarantees travel cost ≤ 1.5× original)")


def demo_5_path_vs_complete():
    """Compare path and complete graph spanners."""
    print_section("Demo 5: Path vs Complete Graph")

    # Path graph
    path = Graph()
    for i in range(10):
        path.add_node(label=str(i))
    for i in range(9):
        path.add_edge(str(i), str(i + 1))

    print(f"Path graph: {path.num_nodes()} nodes, {path.num_edges()} edges")
    path_spanner = path.transform_with_spanner(stretch=2.0, seed=42)
    print(f"2-spanner: {path_spanner.num_edges()} edges")
    print("  → Path graphs need most edges preserved")

    # Complete graph
    complete = Graph()
    for i in range(10):
        complete.add_node(label=str(i))
    for i in range(10):
        for j in range(i + 1, 10):
            complete.add_edge(str(i), str(j))

    print(f"\nComplete graph K10: {complete.num_nodes()} nodes, {complete.num_edges()} edges")
    complete_spanner = complete.transform_with_spanner(stretch=2.0, seed=42)
    print(f"2-spanner: {complete_spanner.num_edges()} edges")
    reduction = (complete.num_edges() - complete_spanner.num_edges()) / complete.num_edges() * 100
    print(f"  → {reduction:.1f}% reduction for dense graphs")


def demo_6_grid_graph():
    """Spanner of a grid graph."""
    print_section("Demo 6: Grid Graph Spanner")

    # Create 5x5 grid
    g = Graph()
    for i in range(5):
        for j in range(5):
            g.add_node(label=f"({i},{j})")

    # Add edges
    for i in range(5):
        for j in range(5):
            if i < 4:  # Vertical edge
                g.add_edge(f"({i},{j})", f"({i + 1},{j})")
            if j < 4:  # Horizontal edge
                g.add_edge(f"({i},{j})", f"({i},{j + 1})")

    print(f"5×5 grid: {g.num_nodes()} nodes, {g.num_edges()} edges")

    for stretch in [1.5, 2.0, 3.0]:
        spanner = g.transform_with_spanner(stretch=stretch, seed=42)
        reduction = (g.num_edges() - spanner.num_edges()) / g.num_edges() * 100
        print(f"{stretch}-spanner: {spanner.num_edges()} edges ({reduction:.1f}% reduction)")


def demo_7_connected_components():
    """Spanner with disconnected components."""
    print_section("Demo 7: Disconnected Components")

    g = Graph()

    # Component 1: Triangle
    for i in range(3):
        g.add_node(label=f"A{i}")
    g.add_edge("A0", "A1")
    g.add_edge("A1", "A2")
    g.add_edge("A2", "A0")

    # Component 2: Square
    for i in range(4):
        g.add_node(label=f"B{i}")
    g.add_edge("B0", "B1")
    g.add_edge("B1", "B2")
    g.add_edge("B2", "B3")
    g.add_edge("B3", "B0")

    print(f"Graph with 2 components: {g.num_nodes()} nodes, {g.num_edges()} edges")

    spanner = g.transform_with_spanner(stretch=2.0, seed=42)
    print(f"2-spanner: {spanner.num_nodes()} nodes, {spanner.num_edges()} edges")
    print("  → Spanner preserves connectivity within components")


def demo_8_star_graph():
    """Spanner of star graph."""
    print_section("Demo 8: Star Graph")

    g = Graph()
    g.add_node(label="center")
    for i in range(10):
        g.add_node(label=str(i))
        g.add_edge("center", str(i))

    print(f"Star graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

    for stretch in [2.0, 3.0, 5.0]:
        spanner = g.transform_with_spanner(stretch=stretch, seed=42)
        print(f"{stretch}-spanner: {spanner.num_edges()} edges")

    print("  → Star graphs are already optimal - all edges needed")


def demo_9_comparison_table():
    """Comprehensive comparison across graph types."""
    print_section("Demo 9: Graph Type Comparison")

    graphs = []

    # Path
    path = Graph()
    for i in range(12):
        path.add_node(label=str(i))
    for i in range(11):
        path.add_edge(str(i), str(i + 1))
    graphs.append(("Path", path))

    # Cycle
    cycle = Graph()
    for i in range(12):
        cycle.add_node(label=str(i))
    for i in range(12):
        cycle.add_edge(str(i), str((i + 1) % 12))
    graphs.append(("Cycle", cycle))

    # Complete
    complete = Graph()
    for i in range(12):
        complete.add_node(label=str(i))
    for i in range(12):
        for j in range(i + 1, 12):
            complete.add_edge(str(i), str(j))
    graphs.append(("Complete K12", complete))

    # Random
    random.seed(42)
    rand = Graph()
    for i in range(12):
        rand.add_node(label=str(i))
    for i in range(12):
        for j in range(i + 1, 12):
            if random.random() < 0.3:
                rand.add_edge(str(i), str(j))
    graphs.append(("Random (p=0.3)", rand))

    print(f"{'Graph Type':<20} {'Original':<12} {'2-Spanner':<12} {'Reduction':<12}")
    print("-" * 60)

    for name, g in graphs:
        original_edges = g.num_edges()
        spanner = g.transform_with_spanner(stretch=2.0, seed=42)
        spanner_edges = spanner.num_edges()
        reduction = (
            (original_edges - spanner_edges) / original_edges * 100 if original_edges > 0 else 0
        )
        print(f"{name:<20} {original_edges:<12} {spanner_edges:<12} {reduction:<12.1f}%")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  EDGECASE: Graph Spanner Demonstrations")
    print("=" * 70)

    demo_1_network_backbone()
    demo_2_graph_compression()
    demo_3_stretch_comparison()
    demo_4_weighted_graph()
    demo_5_path_vs_complete()
    demo_6_grid_graph()
    demo_7_connected_components()
    demo_8_star_graph()
    demo_9_comparison_table()

    print("\n" + "=" * 70)
    print("  All demonstrations complete!")
    print("=" * 70 + "\n")
