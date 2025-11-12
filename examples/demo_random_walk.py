from edgecase import Graph


def demo_basic_walk():
    """Basic random walk on a cycle graph."""
    print("=" * 60)
    print("Demo 1: Basic Random Walk on Cycle")
    print("=" * 60)

    g = Graph()
    for i in range(1, 7):
        g.add_node(label=str(i))
    for i in range(1, 7):
        g.add_edge(str(i), str((i % 6) + 1))

    print(f"Graph: cycle with {g.num_nodes()} nodes")

    # Take a 10-step walk
    path = g.random_walk("1", steps=10, seed=42)
    print(f"\nRandom walk (10 steps): {' -> '.join(path)}")
    print(f"Path length: {len(path)} (includes starting vertex)")

    # Count vertex visits
    from collections import Counter

    visits = Counter(path)
    print("\nVertex visit counts:")
    for vertex, count in sorted(visits.items()):
        print(f"  {vertex}: {count} visits")


def demo_edge_walk():
    """Edge-based random walk."""
    print("\n" + "=" * 60)
    print("Demo 2: Random Edge Walk")
    print("=" * 60)

    g = Graph()
    for i in range(1, 5):
        g.add_node(label=str(i))
    g.add_edge("1", "2")
    g.add_edge("2", "3")
    g.add_edge("3", "4")
    g.add_edge("4", "1")

    print(f"Graph: cycle with {g.num_nodes()} nodes")

    # Track edges traversed
    edges = g.random_edge_walk("1", steps=8, seed=42)
    print("\nEdges traversed (8 steps):")
    for i, (edge_id, source, target) in enumerate(edges, 1):
        print(f"  Step {i}: Edge {edge_id} ({source} -> {target})")

    # Count edge usage
    from collections import Counter

    edge_counts = Counter((src, tgt) for _, src, tgt in edges)
    print("\nEdge usage frequency:")
    for (src, tgt), count in sorted(edge_counts.items()):
        print(f"  {src} -> {tgt}: {count} times")


def demo_directed_modes():
    """Different modes on directed graphs."""
    print("\n" + "=" * 60)
    print("Demo 3: Modes on Directed Graph")
    print("=" * 60)

    g = Graph(directed=True)
    for node in ["A", "B", "C", "D"]:
        g.add_node(label=node)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("D", "A")

    print("Graph: directed cycle A -> B -> C -> D -> A")

    # Forward walk
    forward = g.random_walk("A", steps=8, mode="out", seed=42)
    print(f"\nForward walk (mode='out'): {' -> '.join(forward)}")

    # Backward walk
    backward = g.random_walk("A", steps=8, mode="in", seed=42)
    print(f"Backward walk (mode='in'): {' -> '.join(backward)}")

    # Undirected walk
    undirected = g.random_walk("A", steps=8, mode="all", seed=42)
    print(f"Undirected walk (mode='all'): {' -> '.join(undirected)}")


def demo_stuck_handling():
    """Handling dead ends in random walks."""
    print("\n" + "=" * 60)
    print("Demo 4: Stuck Handling")
    print("=" * 60)

    g = Graph(directed=True)
    for node in ["A", "B", "C", "D"]:
        g.add_node(label=node)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    # C has no outgoing edges - walk gets stuck here

    print("Graph: A -> B -> C (dead end)")

    # Return partial walk when stuck
    path = g.random_walk("A", steps=10, stuck="return")
    print(f"\nPartial walk (stuck='return'): {' -> '.join(path)}")
    print(f"Stopped after {len(path) - 1} steps (got stuck at {path[-1]})")

    # Raise error when stuck
    try:
        g.random_walk("A", steps=10, stuck="error")
        print("\nERROR: Should have raised RuntimeError!")
    except RuntimeError as e:
        print("\nWith stuck='error': Raised RuntimeError ✓")


def demo_reproducibility():
    """Reproducible walks with seed parameter."""
    print("\n" + "=" * 60)
    print("Demo 5: Reproducibility")
    print("=" * 60)

    g = Graph()
    for i in range(1, 6):
        g.add_node(label=str(i))
    for i in range(1, 6):
        g.add_edge(str(i), str((i % 5) + 1))

    print("Graph: cycle with 5 nodes")

    # Same seed = same walk
    walk1 = g.random_walk("1", steps=20, seed=123)
    walk2 = g.random_walk("1", steps=20, seed=123)
    print(f"\nWalk 1 (seed=123): {' -> '.join(walk1[:10])}...")
    print(f"Walk 2 (seed=123): {' -> '.join(walk2[:10])}...")
    print(f"Walks identical: {walk1 == walk2} ✓")

    # Different seed = different walk
    walk3 = g.random_walk("1", steps=20, seed=456)
    print(f"\nWalk 3 (seed=456): {' -> '.join(walk3[:10])}...")
    print(f"Walk 3 different: {walk1 != walk3} ✓")


def demo_star_graph():
    """Random walk on star graph (hub and spokes)."""
    print("\n" + "=" * 60)
    print("Demo 6: Star Graph")
    print("=" * 60)

    g = Graph()
    g.add_node(label="hub")
    for i in range(1, 6):
        g.add_node(label=f"spoke{i}")
        g.add_edge("hub", f"spoke{i}")

    print("Graph: star with 1 hub and 5 spokes")

    # Walk alternates between hub and spokes
    path = g.random_walk("hub", steps=10, seed=42)
    print("\nRandom walk from hub:")
    for i in range(0, len(path), 2):
        if i < len(path):
            print(f"  Step {i}: {path[i]}")

    from collections import Counter

    visits = Counter(path)
    print(f"\nHub visited {visits['hub']} times")
    print(f"Spokes visited: {sum(v for k, v in visits.items() if k != 'hub')} times")


if __name__ == "__main__":
    demo_basic_walk()
    demo_edge_walk()
    demo_directed_modes()
    demo_stuck_handling()
    demo_reproducibility()
    demo_star_graph()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
