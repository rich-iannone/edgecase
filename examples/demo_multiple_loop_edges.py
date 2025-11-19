from edgecase import Graph


def example_1_data_cleaning():
    """Identifying and cleaning duplicate edges in imported data."""
    print("=" * 70)
    print("EXAMPLE 1: Data Cleaning - Finding Duplicate Edges")
    print("=" * 70)

    # Simulated social network data with accidental duplicates
    g = Graph(directed=True)
    for node in ["Alice", "Bob", "Carol", "Dave"]:
        g.add_node(label=node)

    # Add edges - some are duplicates from data import errors
    g.add_edge("Alice", "Bob")
    g.add_edge("Alice", "Bob")  # Duplicate!
    g.add_edge("Alice", "Bob")  # Another duplicate!
    g.add_edge("Bob", "Carol")
    g.add_edge("Carol", "Dave")
    g.add_edge("Carol", "Dave")  # Duplicate!

    print(f"\nTotal edges in graph: {g.count_edges()}")

    # Find duplicate edges
    multiples = g.which_multiple()
    print(f"Duplicate edge IDs: {multiples}")
    print(f"Number of duplicate edges: {len(multiples)}")

    # Get counts per pair
    counts = g.count_multiple()
    print("\nEdge counts per node pair:")
    for (src, tgt), count in sorted(counts.items()):
        print(f"  {src} → {tgt}: {count} edges (should be 1)")

    # Data quality report
    if multiples:
        print(f"\n⚠️  Data quality issue: Found {len(multiples)} duplicate edges")
        print("   Recommendation: Remove duplicates or merge with weights")
    else:
        print("\n✅ Data quality check passed: No duplicate edges")

    print()


def example_2_multigraph_validation():
    """Check if a graph is a simple graph (no multiples, no loops)."""
    print("=" * 70)
    print("EXAMPLE 2: Multigraph Validation")
    print("=" * 70)

    # Test two graphs
    graphs = {"Simple Graph": create_simple_graph(), "Multigraph": create_multigraph()}

    for name, g in graphs.items():
        print(f"\n{name}:")
        print(f"  Nodes: {g.count_nodes()}, Edges: {g.count_edges()}")

        # Check for multiple edges
        multiples = g.which_multiple()
        has_multiples = len(multiples) > 0

        # Check for self-loops
        loops = g.which_loop()
        has_loops = len(loops) > 0

        # Determine graph type
        if not has_multiples and not has_loops:
            graph_type = "Simple Graph ✅"
        elif has_multiples and not has_loops:
            graph_type = "Multigraph (has parallel edges)"
        elif not has_multiples and has_loops:
            graph_type = "Pseudograph (has self-loops)"
        else:
            graph_type = "Multigraph with self-loops"

        print(f"  Type: {graph_type}")
        if has_multiples:
            print(f"  Multiple edges: {len(multiples)}")
        if has_loops:
            print(f"  Self-loops: {len(loops)}")

    print()


def example_3_network_simplification():
    """Simplify a multigraph by aggregating parallel edges."""
    print("=" * 70)
    print("EXAMPLE 3: Network Simplification")
    print("=" * 70)

    # Transportation network with multiple routes
    g = Graph(directed=False)
    for city in ["CityA", "CityB", "CityC"]:
        g.add_node(label=city)

    # Multiple routes between cities
    print("\nOriginal multigraph (multiple routes):")
    g.add_edge("CityA", "CityB")  # Highway
    g.add_edge("CityA", "CityB")  # Railway
    g.add_edge("CityA", "CityB")  # Airline
    g.add_edge("CityB", "CityC")  # Highway
    print(f"  Total edges: {g.count_edges()}")

    # Analyze parallel edges
    multi_counts = g.count_multiple()
    print("\nParallel routes detected:")
    for (src, tgt), count in multi_counts.items():
        print(f"  {src} ↔ {tgt}: {count} routes")

    # Simplification plan
    print("\nSimplification strategy:")
    print("  → Merge parallel edges into single weighted edges")
    print("  → Weight = number of routes (connectivity strength)")

    for (src, tgt), count in multi_counts.items():
        print(f"  → {src}-{tgt}: Consolidate {count} routes → weight={count}")

    print()


def example_4_self_loop_analysis():
    """Analyze self-loops in a state machine or workflow."""
    print("=" * 70)
    print("EXAMPLE 4: Self-Loop Analysis (State Machine)")
    print("=" * 70)

    # Create a state machine
    g = Graph(directed=True)
    states = ["Idle", "Loading", "Processing", "Complete", "Error"]
    for state in states:
        g.add_node(label=state)

    # Add transitions
    g.add_edge("Idle", "Loading")
    g.add_edge("Loading", "Loading")  # Retry loop
    g.add_edge("Loading", "Loading")  # Another retry path
    g.add_edge("Loading", "Processing")
    g.add_edge("Processing", "Processing")  # Busy wait
    g.add_edge("Processing", "Complete")
    g.add_edge("Processing", "Error")
    g.add_edge("Error", "Idle")  # Restart

    print(f"\nState machine: {g.count_nodes()} states, {g.count_edges()} transitions")

    # Find self-loops (states that loop to themselves)
    loops = g.which_loop()
    loop_counts = g.count_loops()

    print(f"\nSelf-transitions found: {len(loops)}")
    print("\nStates with self-loops:")
    for state, count in sorted(loop_counts.items()):
        print(f"  {state}: {count} self-transition(s)")
        if state == "Loading":
            print("    → Indicates retry/waiting behavior")
        elif state == "Processing":
            print("    → Indicates iterative processing")

    print("\nInterpretation:")
    print("  Self-loops represent states that can remain active")
    print("  for multiple cycles (retries, waiting, iteration)")

    print()


def example_5_social_network_self_mentions():
    """Find users who mention/cite themselves."""
    print("=" * 70)
    print("EXAMPLE 5: Social Network - Self-Mentions")
    print("=" * 70)

    # Social media mention network
    g = Graph(directed=True)
    users = ["@alice", "@bob", "@carol", "@dave", "@eve"]
    for user in users:
        g.add_node(label=user)

    # Mention edges (who mentions whom)
    g.add_edge("@alice", "@bob")
    g.add_edge("@alice", "@carol")
    g.add_edge("@bob", "@alice")
    g.add_edge("@bob", "@bob")  # Self-mention
    g.add_edge("@carol", "@dave")
    g.add_edge("@dave", "@dave")  # Self-mention
    g.add_edge("@dave", "@dave")  # Another self-mention
    g.add_edge("@eve", "@alice")

    print(f"\nMention network: {g.count_nodes()} users, {g.count_edges()} mentions")

    # Find self-mentions
    self_mentions = g.which_loop()
    self_mention_counts = g.count_loops()

    print(f"\nUsers with self-mentions: {len(self_mention_counts)}")
    for user, count in sorted(self_mention_counts.items()):
        print(f"  {user}: {count} self-mention(s)")

    # Analysis
    total_users = g.count_nodes()
    self_mention_pct = (len(self_mention_counts) / total_users) * 100

    print("\nAnalysis:")
    print(f"  {self_mention_pct:.1f}% of users mention themselves")
    print("  Self-mentions might indicate:")
    print("    - Quoting own previous posts")
    print("    - Personal reflection/notes")
    print("    - Potential bot/spam behavior (if excessive)")

    print()


def create_simple_graph():
    """Create a simple graph (no multiples, no loops)."""
    g = Graph(directed=True)
    for node in ["A", "B", "C"]:
        g.add_node(label=node)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    return g


def create_multigraph():
    """Create a multigraph (has multiple edges and loops)."""
    g = Graph(directed=True)
    for node in ["A", "B", "C"]:
        g.add_node(label=node)
    g.add_edge("A", "B")
    g.add_edge("A", "B")  # Multiple
    g.add_edge("A", "A")  # Loop
    g.add_edge("B", "C")
    return g


def main():
    """Run all examples."""
    example_1_data_cleaning()
    example_2_multigraph_validation()
    example_3_network_simplification()
    example_4_self_loop_analysis()
    example_5_social_network_self_mentions()


if __name__ == "__main__":
    main()
