from edgecase import Graph


def demo_basic_validation():
    """Demo 1: Basic path validation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Path Validation")
    print("=" * 70)

    # Create a simple linear graph
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
    )

    # Valid paths
    print("\nValid paths:")
    paths_to_test = [
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
        ["B", "C", "D"],
    ]

    for path in paths_to_test:
        is_valid = graph.is_simple_path(path)
        print(f"  {' -> '.join(path)}: {is_valid}")

    # Invalid paths
    print("\nInvalid paths (missing edges):")
    invalid_paths = [
        ["A", "C"],  # No direct edge A->C
        ["A", "D"],  # No direct edge A->D
        ["B", "D"],  # No direct edge B->D
    ]

    for path in invalid_paths:
        is_valid = graph.is_simple_path(path)
        print(f"  {' -> '.join(path)}: {is_valid}")


def demo_cycles_and_repeats():
    """Demo 2: Detecting cycles and repeated nodes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Detecting Cycles and Repeated Nodes")
    print("=" * 70)

    # Create a triangle
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
    )

    print("\nTriangle graph: A <-> B <-> C <-> A")

    # Valid simple paths
    print("\nValid simple paths (no repeated nodes):")
    valid_paths = [
        ["A", "B"],
        ["B", "C"],
        ["C", "A"],
        ["A", "B", "C"],
    ]

    for path in valid_paths:
        is_valid = graph.is_simple_path(path)
        print(f"  {' -> '.join(path)}: {is_valid}")

    # Invalid: paths with repeated nodes
    print("\nInvalid paths (repeated nodes = not simple):")
    invalid_paths = [
        ["A", "B", "C", "A"],  # A appears twice
        ["B", "C", "A", "B"],  # B appears twice
        ["A", "B", "C", "A", "B"],  # Multiple repeats
    ]

    for path in invalid_paths:
        is_valid = graph.is_simple_path(path)
        print(f"  {' -> '.join(path)}: {is_valid}")


def demo_edge_cases():
    """Demo 3: Edge cases (empty, single node)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Edge Cases")
    print("=" * 70)

    graph = Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")

    print("\nGraph with 2 nodes: A <-> B")

    # Edge cases
    test_cases = [
        ([], "Empty path"),
        (["A"], "Single existing node"),
        (["B"], "Single existing node"),
        (["Z"], "Single non-existent node"),
    ]

    for path, description in test_cases:
        is_valid = graph.is_simple_path(path)
        path_str = " -> ".join(path) if path else "[]"
        print(f"  {path_str:20} ({description}): {is_valid}")


def demo_directed_vs_undirected():
    """Demo 4: Directed vs undirected graph differences."""
    print("\n" + "=" * 70)
    print("DEMO 4: Directed vs Undirected Graphs")
    print("=" * 70)

    # Undirected graph
    undirected = (
        Graph()  # default is undirected
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )

    # Directed graph
    directed = (
        Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
    )

    paths_to_test = [
        (["A", "B", "C"], "Forward path"),
        (["C", "B", "A"], "Backward path"),
    ]

    print("\nUndirected graph (bidirectional):")
    for path, description in paths_to_test:
        is_valid = undirected.is_simple_path(path)
        print(f"  {' -> '.join(path)} ({description}): {is_valid}")

    print("\nDirected graph (one direction only):")
    for path, description in paths_to_test:
        is_valid = directed.is_simple_path(path)
        print(f"  {' -> '.join(path)} ({description}): {is_valid}")


def demo_complex_graphs():
    """Demo 5: Complex graph scenarios."""
    print("\n" + "=" * 70)
    print("DEMO 5: Complex Graph - Diamond Pattern")
    print("=" * 70)

    # Diamond graph with multiple paths
    graph = (
        Graph()
        .add_node(label="Start")
        .add_node(label="Left")
        .add_node(label="Right")
        .add_node(label="End")
        .add_edge("Start", "Left")
        .add_edge("Start", "Right")
        .add_edge("Left", "End")
        .add_edge("Right", "End")
        .add_edge("Left", "Right")  # Extra connection
    )

    print("\nDiamond graph:")
    print("       Left")
    print("      /   \\ ")
    print("  Start   End")
    print("      \\   /")
    print("      Right")

    test_paths = [
        (["Start", "Left", "End"], "Path through Left"),
        (["Start", "Right", "End"], "Path through Right"),
        (["Start", "Left", "Right", "End"], "Zigzag path"),
        (["Start", "Left", "Right"], "Start to bridge"),
        (["Start", "End"], "Direct (no edge)"),
    ]

    print("\nPath validation:")
    for path, description in test_paths:
        is_valid = graph.is_simple_path(path)
        path_str = " -> ".join(path)
        print(f"  {path_str:30} ({description}): {is_valid}")


def demo_integration_with_all_simple_paths():
    """Demo 6: Integration with get_all_simple_paths()."""
    print("\n" + "=" * 70)
    print("DEMO 6: Integration with get_all_simple_paths()")
    print("=" * 70)

    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("A", "C")
        .add_edge("B", "D")
        .add_edge("C", "D")
        .add_edge("B", "C")
    )

    print("\nGraph structure:")
    print("  A -> B -> D")
    print("  |    X    |")
    print("  +-> C ---+")

    # Find all paths
    all_paths = graph.get_all_simple_paths("A", "D")
    print(f"\nFound {len(all_paths)} paths from A to D:")

    for i, path in enumerate(all_paths, 1):
        # Validate each path
        is_valid = graph.is_simple_path(path)
        path_str = " -> ".join(path)
        status = "✓" if is_valid else "✗"
        print(f"  {i}. {path_str:20} {status} Valid: {is_valid}")

    # Test an invalid path
    print("\nTesting invalid path not in results:")
    invalid_path = ["A", "D"]  # No direct edge
    is_valid = graph.is_simple_path(invalid_path)
    is_in_results = invalid_path in all_paths
    print(f"  Path: {' -> '.join(invalid_path)}")
    print(f"  Valid: {is_valid}")
    print(f"  In results: {is_in_results}")


def demo_real_world_scenario():
    """Demo 7: Real-world scenario - delivery route validation."""
    print("\n" + "=" * 70)
    print("DEMO 7: Real-World Scenario - Delivery Route Validation")
    print("=" * 70)

    # City delivery network
    network = (
        Graph()
        .add_node(label="Warehouse")
        .add_node(label="Store1")
        .add_node(label="Store2")
        .add_node(label="Store3")
        .add_node(label="Store4")
        .add_edge("Warehouse", "Store1")
        .add_edge("Warehouse", "Store2")
        .add_edge("Store1", "Store3")
        .add_edge("Store2", "Store3")
        .add_edge("Store3", "Store4")
        .add_edge("Store2", "Store4")
    )

    print("\nDelivery network:")
    print("  Warehouse -> Store1 -> Store3 -> Store4")
    print("            \\-> Store2 ->/      /")
    print("                       +--------+")

    # Proposed delivery routes
    routes = [
        (["Warehouse", "Store1", "Store3", "Store4"], "Main route"),
        (["Warehouse", "Store2", "Store4"], "Express route"),
        (["Warehouse", "Store2", "Store3", "Store4"], "Alternative route"),
        (["Warehouse", "Store1", "Store2"], "Invalid: no direct connection"),
        (["Warehouse", "Store4"], "Invalid: no direct connection"),
    ]

    print("\nValidating delivery routes:")
    for route, description in routes:
        is_valid = network.is_simple_path(route)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        route_str = " -> ".join(route)
        print(f"  {status:10} {route_str}")
        print(f"             ({description})")


def demo_performance_note():
    """Demo 8: Performance comparison note."""
    print("\n" + "=" * 70)
    print("DEMO 8: Performance Note")
    print("=" * 70)

    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "D")
        .add_edge("D", "E")
        .add_edge("A", "C")
        .add_edge("B", "D")
        .add_edge("C", "E")
    )

    test_path = ["A", "B", "D", "E"]

    print("\nComparing validation approaches:")
    print(f"Test path: {' -> '.join(test_path)}")

    # Method 1: Direct validation (O(n))
    is_valid = graph.is_simple_path(test_path)
    print("\n  Method 1 - Direct validation (O(n)):")
    print(f"    graph.is_simple_path(path) -> {is_valid}")

    # Method 2: Generate all paths and check (exponential!)
    all_paths = graph.get_all_simple_paths("A", "E")
    is_in_paths = test_path in all_paths
    print("\n  Method 2 - Generate all & check (exponential):")
    print(f"    path in graph.get_all_simple_paths('A', 'E') -> {is_in_paths}")
    print(f"    (Had to generate {len(all_paths)} paths to check!)")

    print("\n  Recommendation:")
    print("    Use is_simple_path() for O(n) validation")
    print("    Use get_all_simple_paths() only when you need ALL paths")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" EDGECASE: is_simple_path() METHOD DEMO")
    print("=" * 70)
    print("\nFast O(n) validation for checking if node sequences form valid paths")

    demo_basic_validation()
    demo_cycles_and_repeats()
    demo_edge_cases()
    demo_directed_vs_undirected()
    demo_complex_graphs()
    demo_integration_with_all_simple_paths()
    demo_real_world_scenario()
    demo_performance_note()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
