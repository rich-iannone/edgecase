import edgecase as ec
import networkx as nx


def create_both_graphs(edges, directed=False):
    """Create identical graphs in both libraries."""
    # EdgeCase graph
    ec_graph = ec.Graph(directed=directed)
    nodes = set()
    for src, tgt in edges:
        nodes.add(src)
        nodes.add(tgt)

    for node in sorted(nodes):
        ec_graph.add_node(label=node)

    for src, tgt in edges:
        ec_graph.add_edge(src, tgt)

    # NetworkX graph
    if directed:
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()

    nx_graph.add_edges_from(edges)

    return ec_graph, nx_graph


def compare_paths(ec_paths, nx_paths, graph_name):
    """Compare path sets from both implementations."""
    # Convert to sets of tuples for comparison
    ec_set = {tuple(path) for path in ec_paths}
    nx_set = {tuple(path) for path in nx_paths}

    if ec_set == nx_set:
        print(f"✓ {graph_name}: MATCH ({len(ec_paths)} paths)")
        return True
    else:
        print(f"✗ {graph_name}: MISMATCH")
        print(f"  EdgeCase found {len(ec_paths)} paths")
        print(f"  NetworkX found {len(nx_paths)} paths")

        only_ec = ec_set - nx_set
        only_nx = nx_set - ec_set

        if only_ec:
            print(f"  Paths only in EdgeCase: {len(only_ec)}")
            for path in list(only_ec)[:3]:
                print(f"    {' -> '.join(path)}")

        if only_nx:
            print(f"  Paths only in NetworkX: {len(only_nx)}")
            for path in list(only_nx)[:3]:
                print(f"    {' -> '.join(path)}")

        return False


def test_diamond_graph():
    """Test 1: Diamond graph (2 paths)."""
    print("\n" + "=" * 70)
    print("TEST 1: Diamond Graph")
    print("=" * 70)

    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A -> B -> D")
    print("           A -> C -> D")

    ec_paths = ec_graph.get_all_simple_paths("A", "D")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "D"))

    return compare_paths(ec_paths, nx_paths, "Diamond graph")


def test_complete_graph():
    """Test 2: Complete graph K4."""
    print("\n" + "=" * 70)
    print("TEST 2: Complete Graph K4")
    print("=" * 70)

    nodes = ["A", "B", "C", "D"]
    edges = [(n1, n2) for i, n1 in enumerate(nodes) for n2 in nodes[i + 1 :]]

    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: All nodes connected (K4)")

    ec_paths = ec_graph.get_all_simple_paths("A", "D")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "D"))

    return compare_paths(ec_paths, nx_paths, "Complete graph K4")


def test_linear_chain():
    """Test 3: Linear chain."""
    print("\n" + "=" * 70)
    print("TEST 3: Linear Chain")
    print("=" * 70)

    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A -> B -> C -> D -> E")

    ec_paths = ec_graph.get_all_simple_paths("A", "E")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "E"))

    return compare_paths(ec_paths, nx_paths, "Linear chain")


def test_directed_graph():
    """Test 4: Directed graph."""
    print("\n" + "=" * 70)
    print("TEST 4: Directed Graph")
    print("=" * 70)

    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    ec_graph, nx_graph = create_both_graphs(edges, directed=True)

    print("Structure: A → B → D")
    print("           A → C → D")

    ec_paths = ec_graph.get_all_simple_paths("A", "D")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "D"))

    return compare_paths(ec_paths, nx_paths, "Directed graph")


def test_with_cutoff():
    """Test 5: Using cutoff parameter."""
    print("\n" + "=" * 70)
    print("TEST 5: Path Length Cutoff")
    print("=" * 70)

    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("A", "C"),
        ("C", "E"),
    ]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A -> B -> C -> D -> E")
    print("           with shortcuts A->C and C->E")
    print("Testing cutoff=2 (max 2 edges)")

    ec_paths = ec_graph.get_all_simple_paths("A", "E", cutoff=2)
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "E", cutoff=2))

    return compare_paths(ec_paths, nx_paths, "Graph with cutoff=2")


def test_grid_graph():
    """Test 6: Grid graph."""
    print("\n" + "=" * 70)
    print("TEST 6: 2x3 Grid Graph")
    print("=" * 70)

    edges = [
        ("A", "B"),
        ("B", "C"),  # Top row
        ("D", "E"),
        ("E", "F"),  # Bottom row
        ("A", "D"),
        ("B", "E"),
        ("C", "F"),  # Vertical
    ]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A - B - C")
    print("           |   |   |")
    print("           D - E - F")

    ec_paths = ec_graph.get_all_simple_paths("A", "F")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "F"))

    return compare_paths(ec_paths, nx_paths, "Grid graph")


def test_tree_structure():
    """Test 7: Tree (no cycles)."""
    print("\n" + "=" * 70)
    print("TEST 7: Tree Structure")
    print("=" * 70)

    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
        ("C", "G"),
    ]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: Binary tree")
    print("              A")
    print("            /   \\")
    print("           B     C")
    print("          / \\   / \\")
    print("         D   E F   G")

    ec_paths = ec_graph.get_all_simple_paths("D", "G")
    nx_paths = list(nx.all_simple_paths(nx_graph, "D", "G"))

    return compare_paths(ec_paths, nx_paths, "Tree structure")


def test_self_loop():
    """Test 8: Self-loop (source == target)."""
    print("\n" + "=" * 70)
    print("TEST 8: Self-Loop (source == target)")
    print("=" * 70)

    edges = [("A", "B"), ("B", "C")]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A -> B -> C")
    print("Finding paths from A to A")

    ec_paths = ec_graph.get_all_simple_paths("A", "A")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "A"))

    return compare_paths(ec_paths, nx_paths, "Self-loop")


def test_disconnected():
    """Test 9: Disconnected components."""
    print("\n" + "=" * 70)
    print("TEST 9: Disconnected Components")
    print("=" * 70)

    edges = [("A", "B"), ("C", "D")]  # Two separate components
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: A - B    C - D")
    print("           (disconnected)")

    ec_paths = ec_graph.get_all_simple_paths("A", "D")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "D"))

    return compare_paths(ec_paths, nx_paths, "Disconnected graph")


def test_triangle_with_chord():
    """Test 10: Triangle with chord."""
    print("\n" + "=" * 70)
    print("TEST 10: Triangle with Chord")
    print("=" * 70)

    edges = [("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("D", "A")]
    ec_graph, nx_graph = create_both_graphs(edges)

    print("Structure: Triangle ABC with chord through D")

    ec_paths = ec_graph.get_all_simple_paths("A", "C")
    nx_paths = list(nx.all_simple_paths(nx_graph, "A", "C"))

    return compare_paths(ec_paths, nx_paths, "Triangle with chord")


def test_petersen_graph():
    """Test 11: Petersen graph (famous test case)."""
    print("\n" + "=" * 70)
    print("TEST 11: Petersen Graph")
    print("=" * 70)

    # Create Petersen graph
    nx_graph = nx.petersen_graph()

    # Convert to EdgeCase (nodes are 0-9)
    ec_graph = ec.Graph()
    for node in nx_graph.get_node_ids(limit=None):
        ec_graph.add_node(label=str(node))

    for src, tgt in nx_graph.edges():
        ec_graph.add_edge(str(src), str(tgt))

    print("Structure: Petersen graph (10 nodes, 15 edges)")
    print("Finding paths from node 0 to node 5")

    ec_paths = ec_graph.get_all_simple_paths("0", "5")
    nx_paths = list(nx.all_simple_paths(nx_graph, 0, 5))

    # Convert nx paths to strings for comparison
    nx_paths_str = [[str(n) for n in path] for path in nx_paths]

    return compare_paths(ec_paths, nx_paths_str, "Petersen graph")


def test_karate_club():
    """Test 12: Karate Club graph (real-world network)."""
    print("\n" + "=" * 70)
    print("TEST 12: Karate Club Graph")
    print("=" * 70)

    # Load Zachary's karate club
    nx_graph = nx.karate_club_graph()

    # Convert to EdgeCase
    ec_graph = ec.Graph()
    for node in nx_graph.get_node_ids(limit=None):
        ec_graph.add_node(label=str(node))

    for src, tgt in nx_graph.edges():
        ec_graph.add_edge(str(src), str(tgt))

    print("Structure: Zachary's Karate Club (34 nodes, 78 edges)")
    print("Finding paths from node 0 to node 33 with cutoff=4")

    # Use cutoff to make it computationally tractable
    ec_paths = ec_graph.get_all_simple_paths("0", "33", cutoff=4)
    nx_paths = list(nx.all_simple_paths(nx_graph, 0, 33, cutoff=4))

    # Convert nx paths to strings for comparison
    nx_paths_str = [[str(n) for n in path] for path in nx_paths]

    return compare_paths(ec_paths, nx_paths_str, "Karate Club with cutoff")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("EdgeCase vs NetworkX: get_all_simple_paths() Validation")
    print("=" * 70)
    print("\nComparing EdgeCase implementation against NetworkX reference...")

    results = []

    results.append(test_diamond_graph())
    results.append(test_complete_graph())
    results.append(test_linear_chain())
    results.append(test_directed_graph())
    results.append(test_with_cutoff())
    results.append(test_grid_graph())
    results.append(test_tree_structure())
    results.append(test_self_loop())
    results.append(test_disconnected())
    results.append(test_triangle_with_chord())
    results.append(test_petersen_graph())
    results.append(test_karate_club())

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        print("EdgeCase get_all_simple_paths() matches NetworkX behavior.")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("Review differences above.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        import networkx

        main()
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        print("This validation script requires NetworkX for comparison.")
