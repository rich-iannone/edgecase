import edgecase as ec


def example_1_basic_optimal_path():
    """Find optimal path with epsilon=1.0 (equivalent to A*)."""
    print("Example 1: Basic Optimal Path")
    print("=" * 50)

    g = ec.Graph(directed=True)

    # Create a simple weighted graph
    g.add_node(label="A")
    g.add_node(label="B")
    g.add_node(label="C")
    g.add_node(label="D")

    g.add_edge("A", "B", attributes={"weight": 1.0})
    g.add_edge("A", "C", attributes={"weight": 4.0})
    g.add_edge("B", "C", attributes={"weight": 2.0})
    g.add_edge("C", "D", attributes={"weight": 1.0})
    g.add_edge("B", "D", attributes={"weight": 5.0})

    # Find optimal path from A to D
    path = g.ada_star_search("A", "D", epsilon=1.0)

    print(f"Optimal path from A to D: {' -> '.join(path)}")
    print(f"Path length: {len(path) - 1} hops")
    print()


def example_2_fast_suboptimal():
    """Use higher epsilon for faster (but potentially suboptimal) paths."""
    print("Example 2: Fast Suboptimal Path")
    print("=" * 50)

    g = ec.Graph(directed=False)

    # Create a larger graph
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for node in nodes:
        g.add_node(label=node)

    # Add edges
    edges = [
        ("A", "B", 1.0),
        ("A", "C", 2.0),
        ("B", "D", 1.0),
        ("C", "D", 1.0),
        ("D", "E", 1.0),
        ("D", "F", 2.0),
        ("E", "G", 1.0),
        ("F", "G", 1.0),
        ("G", "H", 1.0),
    ]

    for src, dst, weight in edges:
        g.add_edge(src, dst, attributes={"weight": weight})

    # Compare epsilon values
    for epsilon in [1.0, 2.0, 5.0]:
        path = g.ada_star_search("A", "H", epsilon=epsilon)
        print(f"epsilon={epsilon}: {' -> '.join(path)}")

    print()


def example_3_custom_heuristic():
    """Use custom heuristic for spatial graphs."""
    print("Example 3: Custom Euclidean Heuristic")
    print("=" * 50)

    # Graph with 2D coordinates
    positions = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (1, 1),
        "D": (2, 1),
        "E": (3, 1),
    }

    g = ec.Graph(directed=False)
    for node in positions:
        g.add_node(label=node)

    # Add edges with weights as Euclidean distances
    g.add_edge("A", "B", attributes={"weight": 1.0})
    g.add_edge("B", "C", attributes={"weight": 1.0})
    g.add_edge("C", "D", attributes={"weight": 1.0})
    g.add_edge("D", "E", attributes={"weight": 1.0})
    g.add_edge("A", "C", attributes={"weight": 1.5})  # Shortcut

    # Define Euclidean distance heuristic
    def euclidean_heuristic(node: str, target: str) -> float:
        """Estimate distance from node to target using Euclidean distance."""
        x1, y1 = positions[node]
        x2, y2 = positions[target]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Without heuristic (zero heuristic = Dijkstra)
    path_no_h = g.ada_star_search("A", "E", epsilon=1.0)
    print(f"Without heuristic: {' -> '.join(path_no_h)}")

    # With Euclidean heuristic (guides search toward target)
    path_with_h = g.ada_star_search("A", "E", epsilon=1.0, heuristic=euclidean_heuristic)
    print(f"With heuristic: {' -> '.join(path_with_h)}")

    print("\nNote: Both paths are optimal, but heuristic explores fewer nodes")
    print()


def example_4_advanced_iterative_refinement():
    """Use iterative refinement without separate class."""
    print("Example 4: Iterative Refinement (No Separate Class)")
    print("=" * 50)

    g = ec.Graph(directed=True)

    # Create graph
    for node in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(label=node)

    edges = [
        ("A", "B", 1.0),
        ("A", "C", 3.0),
        ("B", "D", 1.0),
        ("C", "D", 1.0),
        ("D", "E", 1.0),
        ("D", "F", 2.0),
        ("E", "F", 1.0),
    ]

    for src, dst, weight in edges:
        g.add_edge(src, dst, attributes={"weight": weight})

    # Iterative refinement with reuse_planner=True
    print("Anytime refinement (reuses planner state):")
    for epsilon in [5.0, 2.0, 1.0]:
        path = g.ada_star_search("A", "F", epsilon=epsilon, reuse_planner=True)
        print(f"  epsilon={epsilon}: {' -> '.join(path)}")

    print("\nNote: Each call reuses work from previous searches")
    print("      Set reuse_planner=True to enable this")
    print()


def main():
    """Run all examples."""
    example_1_basic_optimal_path()
    example_2_fast_suboptimal()
    example_3_custom_heuristic()
    example_4_advanced_iterative_refinement()

    print("=" * 50)
    print("Summary:")
    print("- Use graph.ada_star_search() for one-shot pathfinding")
    print("- epsilon=1.0 guarantees optimal paths")
    print("- epsilon>1.0 trades optimality for speed")
    print("- Custom heuristics guide search toward target")
    print("- Set reuse_planner=True for iterative refinement")
    print()
    print("No separate class needed - everything works on Graph!")


if __name__ == "__main__":
    main()
