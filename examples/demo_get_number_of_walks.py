from edgecase import Graph


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def print_walk_matrix(walks: dict, title: str, nodes: list = None):
    """Print walk counts as a matrix."""
    print(f"\n{title}")
    print("-" * len(title))

    if nodes is None:
        nodes = sorted(walks.keys())

    # Header
    print("    ", end="")
    for target in nodes:
        print(f"{target:>5}", end="")
    print()

    # Rows
    for source in nodes:
        print(f"{source:>3} ", end="")
        for target in nodes:
            print(f"{walks[source][target]:>5}", end="")
        print()


# =============================================================================
# Demo 1: Simple Path Graph
# =============================================================================

print_section("Demo 1: Path Graph - Walk Enumeration")

print("\nGraph structure: 0 — 1 — 2")
print("Analyzing walks of different lengths")

g1 = Graph()
for i in range(3):
    g1.add_node(label=str(i))
g1.add_edge("0", "1")
g1.add_edge("1", "2")

# Length 0: Identity matrix
walks0 = g1.get_number_of_walks(0)
print_walk_matrix(walks0, "Walks of length 0 (stay at same node):")

# Length 1: Adjacency matrix
walks1 = g1.get_number_of_walks(1)
print_walk_matrix(walks1, "Walks of length 1 (direct edges):")

# Length 2: Can reach neighbors of neighbors
walks2 = g1.get_number_of_walks(2)
print_walk_matrix(walks2, "Walks of length 2:")
print("  Examples:")
print("    0 → 2: 1 walk (0 → 1 → 2)")
print("    1 → 1: 2 walks (1 → 0 → 1 and 1 → 2 → 1)")

# Length 3
walks3 = g1.get_number_of_walks(3)
print_walk_matrix(walks3, "Walks of length 3:")
print("  Note: Walk counts grow as paths become more redundant")


# =============================================================================
# Demo 2: Cycle Graph
# =============================================================================

print_section("Demo 2: Triangle (3-Cycle) - Closed Walks")

print("\nGraph structure: Triangle with nodes 0, 1, 2")

g2 = Graph()
for i in range(3):
    g2.add_node(label=str(i))
g2.add_edge("0", "1")
g2.add_edge("1", "2")
g2.add_edge("2", "0")

walks3 = g2.get_number_of_walks(3)
print_walk_matrix(walks3, "Walks of length 3 (one complete lap):")
print("\nInterpretation:")
print("  - Diagonal elements = 2: Each node can return to itself in 2 ways")
print("  - E.g., from 0: (0 → 1 → 2 → 0) or (0 → 2 → 1 → 0)")
print("  - Closed walks indicate cycle structure")

walks6 = g2.get_number_of_walks(6)
print_walk_matrix(walks6, "Walks of length 6 (two complete laps):")
print("\nNote: Walk counts grow exponentially in cycles")


# =============================================================================
# Demo 3: Star Graph - Centrality
# =============================================================================

print_section("Demo 3: Star Graph - Hub Connectivity")

print("\nGraph structure: Hub connected to 4 leaves")
print("  hub — leaf0, leaf1, leaf2, leaf3")

g3 = Graph()
g3.add_node(label="hub")
for i in range(4):
    leaf = f"leaf{i}"
    g3.add_node(label=leaf)
    g3.add_edge("hub", leaf)

walks2 = g3.get_number_of_walks(2)
node_labels = ["hub", "leaf0", "leaf1", "leaf2", "leaf3"]
nodes = [g3.get_id_from_label(label) for label in node_labels]
print_walk_matrix(walks2, "Walks of length 2:", nodes)

print("\nInterpretation:")
print("  - Hub to hub: 4 walks (one through each leaf)")
print("  - Leaf to leaf: 1 walk (must go through hub)")
print("  - High walk count from hub shows centrality")

# Calculate total walks from hub
hub_id = g3.get_id_from_label("hub")
hub_walks = sum(walks2[hub_id].values())
leaf0_id = g3.get_id_from_label("leaf0")
leaf_walks = sum(walks2[leaf0_id].values())
print("\nTotal walks of length 2:")
print(f"  From hub: {hub_walks}")
print(f"  From leaf: {leaf_walks}")
print(f"  Hub has {hub_walks / leaf_walks:.1f}x more reachability")


# =============================================================================
# Demo 4: Complete Graph - Dense Connectivity
# =============================================================================

print_section("Demo 4: Complete Graph K4 - Maximum Connectivity")

print("\nGraph structure: All nodes connected to all others")

g4 = Graph()
for i in range(4):
    g4.add_node(label=str(i))

# Add all edges
for i in range(4):
    for j in range(i + 1, 4):
        g4.add_edge(str(i), str(j))

walks1 = g4.get_number_of_walks(1)
print_walk_matrix(walks1, "Walks of length 1 (adjacency matrix):")

walks2 = g4.get_number_of_walks(2)
print_walk_matrix(walks2, "Walks of length 2:")

print("\nInterpretation:")
print("  - Diagonal elements: Many ways to return (through all neighbors)")
print("  - Off-diagonal: Can reach any node through various paths")
print("  - Dense graphs have exponentially growing walk counts")

# Total walk count growth
for k in [1, 2, 3, 4]:
    walks = g4.get_number_of_walks(k)
    total = sum(sum(targets.values()) for targets in walks.values())
    print(f"  Total walks of length {k}: {total}")


# =============================================================================
# Demo 5: Disconnected Graph - Component Analysis
# =============================================================================

print_section("Demo 5: Disconnected Graph - Components")

print("\nGraph structure: Two separate triangles")
print("  Triangle A: 0-1-2")
print("  Triangle B: 3-4-5")

g5 = Graph()
for i in range(6):
    g5.add_node(label=str(i))

# Triangle A
g5.add_edge("0", "1")
g5.add_edge("1", "2")
g5.add_edge("2", "0")

# Triangle B
g5.add_edge("3", "4")
g5.add_edge("4", "5")
g5.add_edge("5", "3")

walks2 = g5.get_number_of_walks(2)
print_walk_matrix(walks2, "Walks of length 2:")

print("\nInterpretation:")
print("  - Block structure shows disconnected components")
print("  - 0 walks between components (nodes 0-2 and 3-5)")
print("  - Each triangle has identical walk structure")


# =============================================================================
# Demo 6: Directed Graph - Asymmetric Walks
# =============================================================================

print_section("Demo 6: Directed Graph - Asymmetric Connectivity")

print("\nGraph structure: Directed cycle 0 → 1 → 2 → 0")

g6 = Graph(directed=True)
for i in range(3):
    g6.add_node(label=str(i))
g6.add_edge("0", "1")
g6.add_edge("1", "2")
g6.add_edge("2", "0")

walks1 = g6.get_number_of_walks(1)
print_walk_matrix(walks1, "Walks of length 1 (directed adjacency):")

walks3 = g6.get_number_of_walks(3)
print_walk_matrix(walks3, "Walks of length 3 (one complete cycle):")

print("\nInterpretation:")
print("  - Asymmetric matrix (walks[i][j] ≠ walks[j][i])")
print("  - Can only walk in directed edges")
print("  - Length 3: Each node returns to itself exactly once")


# =============================================================================
# Demo 7: Network Reachability Analysis
# =============================================================================

print_section("Demo 7: Reachability - Information Diffusion")

print("\nScenario: Social network with information spreading")
print("Graph structure: Small social network")

g7 = Graph()
people = ["Alice", "Bob", "Carol", "Dave", "Eve"]
for person in people:
    g7.add_node(label=person)

# Connections
g7.add_edge("Alice", "Bob")
g7.add_edge("Alice", "Carol")
g7.add_edge("Bob", "Dave")
g7.add_edge("Carol", "Dave")
g7.add_edge("Dave", "Eve")

print("\nConnections:")
print("  Alice connects to Bob and Carol")
print("  Bob and Carol both connect to Dave")
print("  Dave connects to Eve")

# Get node IDs for later use
alice_id = g7.get_id_from_label("Alice")
eve_id = g7.get_id_from_label("Eve")

# Analyze reachability at different steps
for k in range(1, 5):
    walks = g7.get_number_of_walks(k)

    # Count how many people Alice can reach
    reachable = sum(
        1 for target, count in walks[alice_id].items() if count > 0 and target != alice_id
    )
    total_paths = sum(walks[alice_id].values()) - walks[alice_id][alice_id]

    print(f"\nAfter {k} step(s):")
    print(f"  Alice can reach {reachable} people")
    print(f"  Total paths from Alice: {total_paths}")
    print(f"  Paths to Eve: {walks[alice_id][eve_id]}")

print("\nInterpretation:")
print("  - More steps = more people reached")
print("  - Redundant paths (multiple walks) = stronger connectivity")
print("  - Information can flow through multiple channels")


# =============================================================================
# Demo 8: Multigraph - Multiple Edges
# =============================================================================

print_section("Demo 8: Multigraph - Parallel Edges")

print("\nGraph structure: Two nodes with 3 parallel edges")

g8 = Graph()
g8.add_node(label="A")
g8.add_node(label="B")

# Add 3 edges between same nodes
for _ in range(3):
    g8.add_edge("A", "B")

walks1 = g8.get_number_of_walks(1)
print("\nWalks of length 1:")
print(
    f"  A → B: {walks1[g8.get_id_from_label('A')][g8.get_id_from_label('B')]} (counts all 3 edges)"
)
print(f"  B → A: {walks1[g8.get_id_from_label('B')][g8.get_id_from_label('A')]} (undirected)")

walks2 = g8.get_number_of_walks(2)
print("\nWalks of length 2:")
print(
    f"  A → A: {walks2[g8.get_id_from_label('A')][g8.get_id_from_label('A')]} (3 edges × 3 edges = 9 paths)"
)
print(f"  A → B: {walks2[g8.get_id_from_label('A')][g8.get_id_from_label('B')]}")

print("\nInterpretation:")
print("  - Multiple edges multiply walk counts")
print("  - Models: Multiple transportation routes, communication channels")
print("  - Walk count reflects capacity/redundancy")


# =============================================================================
# Demo 9: Practical Application - Network Robustness
# =============================================================================

print_section("Demo 9: Network Robustness - Path Redundancy")

print("\nComparing two network designs:")

# Network 1: Linear chain (fragile)
g_chain = Graph()
for i in range(5):
    g_chain.add_node(label=str(i))
for i in range(4):
    g_chain.add_edge(str(i), str(i + 1))

# Network 2: Mesh (robust)
g_mesh = Graph()
for i in range(5):
    g_mesh.add_node(label=str(i))
g_mesh.add_edge("0", "1")
g_mesh.add_edge("1", "2")
g_mesh.add_edge("2", "3")
g_mesh.add_edge("3", "4")
g_mesh.add_edge("0", "2")  # Shortcut
g_mesh.add_edge("2", "4")  # Shortcut

walks_chain = g_chain.get_number_of_walks(4)
walks_mesh = g_mesh.get_number_of_walks(4)

node_0 = g_chain.get_id_from_label("0")
node_4 = g_chain.get_id_from_label("4")

print("\nChain network (0-1-2-3-4):")
print(f"  Walks from 0 to 4 (length 4): {walks_chain[node_0][node_4]}")

print("\nMesh network (with shortcuts):")
print(f"  Walks from 0 to 4 (length 4): {walks_mesh[node_0][node_4]}")

print("\nInterpretation:")
print(f"  Mesh has {walks_mesh[node_0][node_4] / walks_chain[node_0][node_4]:.1f}x more paths")
print("  - More walks = more redundancy")
print("  - Mesh network is more robust to failures")
print("  - Walk count measures path diversity")


# =============================================================================
# Summary
# =============================================================================

print_section("Summary: Applications of Walk Counting")

print("""
Key insights from get_number_of_walks:

1. **Network Reachability**
   - Walk counts measure how many ways information/resources can flow
   - Higher counts = stronger connectivity
   - Zero counts identify unreachable nodes

2. **Structural Analysis**
   - Diagonal elements: Closed walks (cycles)
   - Symmetric patterns: Undirected connectivity
   - Block structure: Disconnected components

3. **Centrality & Importance**
   - Nodes with high total walk counts are more central
   - Hub nodes show up in many walks
   - Peripheral nodes have limited walk options

4. **Robustness & Redundancy**
   - Multiple walks = multiple paths = redundancy
   - Networks with more walks are more robust
   - Single walk paths are vulnerable to failures

5. **Growth Patterns**
   - Walks grow exponentially with length in dense graphs
   - Linear growth in sparse graphs (paths, trees)
   - Growth rate indicates connectivity density

6. **Practical Applications**
   - Information diffusion modeling
   - Network reliability analysis
   - Social network analysis (indirect connections)
   - Transportation network capacity
   - Communication network redundancy

7. **Relationship to Matrix Powers**
   - A^k[i,j] = number of walks of length k from i to j
   - Eigenvalues determine growth rate
   - Trace(A^k) counts closed walks (cycles)

8. **Comparison with Paths**
   - Walks allow repetition (counted here)
   - Paths require no repetition (use shortest_path methods)
   - Walks better capture redundancy and capacity

For more details, see:
    Lovász, L. (1993). Random walks on graphs: A survey.
    Bollobás, B. (1998). Modern Graph Theory. Springer.
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)
