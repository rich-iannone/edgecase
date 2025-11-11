import edgecase as ec


def example_1_basic_reciprocity():
    """Example 1: Basic reciprocity calculation."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Reciprocity")
    print("=" * 70)

    # Create a simple directed graph
    g = ec.Graph(directed=True)
    g.add_node(label="Alice")
    g.add_node(label="Bob")
    g.add_node(label="Carol")

    # Add some edges
    g.add_edge("Alice", "Bob")
    g.add_edge("Bob", "Alice")  # Mutual with above
    g.add_edge("Alice", "Carol")  # One-way, no reciprocal

    print("\nGraph structure:")
    print("  Alice → Bob (mutual)")
    print("  Bob → Alice")
    print("  Alice → Carol (one-way)")

    # Calculate reciprocity
    recip = g.reciprocity()
    print(f"\nReciprocity: {recip:.3f}")
    print(f"Interpretation: {recip * 100:.1f}% of edges are reciprocal")

    # Find mutual edge pairs
    mutual = g.which_mutual()
    print(f"\nMutual edge pairs: {len(mutual)}")
    for edge1_id, edge2_id in mutual:
        e1 = g.get_edge_by_id(edge1_id)
        e2 = g.get_edge_by_id(edge2_id)
        print(f"  Edge {edge1_id} and {edge2_id} are mutual")
    print()


def example_2_social_networks():
    """Example 2: Comparing different social network types."""
    print("=" * 70)
    print("EXAMPLE 2: Social Network Comparison")
    print("=" * 70)

    # Friendship network (typically high reciprocity)
    friendship = ec.Graph(directed=True)
    friends = [
        ("Alice", "Bob"),
        ("Bob", "Alice"),
        ("Bob", "Carol"),
        ("Carol", "Bob"),
        ("Alice", "Carol"),
        ("Carol", "Alice"),
        ("David", "Alice"),
        ("Alice", "David"),
    ]
    for a, b in friends:
        if not friendship.has_node(a):
            friendship.add_node(label=a)
        if not friendship.has_node(b):
            friendship.add_node(label=b)
        friendship.add_edge(a, b)

    # Follower network (typically low reciprocity)
    followers = ec.Graph(directed=True)
    follows = [
        ("Fan1", "Celebrity"),
        ("Fan2", "Celebrity"),
        ("Fan3", "Celebrity"),
        ("Fan1", "Influencer"),
        ("Fan2", "Influencer"),
        ("Celebrity", "Influencer"),  # One-way
    ]
    for a, b in follows:
        if not followers.has_node(a):
            followers.add_node(label=a)
        if not followers.has_node(b):
            followers.add_node(label=b)
        followers.add_edge(a, b)

    print("\nFriendship Network:")
    print(f"  Nodes: {friendship.num_nodes()}")
    print(f"  Edges: {friendship.num_edges()}")
    print(f"  Reciprocity: {friendship.reciprocity():.3f}")
    print("  Interpretation: Symmetric relationships (friends)")

    print("\nFollower Network:")
    print(f"  Nodes: {followers.num_nodes()}")
    print(f"  Edges: {followers.num_edges()}")
    print(f"  Reciprocity: {followers.reciprocity():.3f}")
    print("  Interpretation: Asymmetric relationships (followers)")
    print()


def example_3_communication_patterns():
    """Example 3: Analyzing communication patterns."""
    print("=" * 70)
    print("EXAMPLE 3: Communication Pattern Analysis")
    print("=" * 70)

    # Email communication network
    email = ec.Graph(directed=True)
    communications = [
        # Team members who communicate bidirectionally
        ("Alice", "Bob"),
        ("Bob", "Alice"),
        ("Bob", "Carol"),
        ("Carol", "Bob"),
        # Manager sends to team (mostly one-way)
        ("Manager", "Alice"),
        ("Manager", "Bob"),
        ("Manager", "Carol"),
        ("Alice", "Manager"),  # Some reply back
        # External communication (one-way)
        ("Alice", "External1"),
        ("Bob", "External2"),
    ]

    for sender, receiver in communications:
        if not email.has_node(sender):
            email.add_node(label=sender)
        if not email.has_node(receiver):
            email.add_node(label=receiver)
        email.add_edge(sender, receiver)

    recip = email.reciprocity()
    mutual_pairs = email.which_mutual()

    print("\nEmail Network Statistics:")
    print(f"  Total edges (emails): {email.num_edges()}")
    print(f"  Mutual pairs: {len(mutual_pairs)}")
    print(f"  Reciprocity: {recip:.3f}")

    print("\nCommunication Pattern:")
    if recip > 0.5:
        print("  → High bidirectional communication (collaborative)")
    elif recip > 0.2:
        print("  → Moderate bidirectional communication")
    else:
        print("  → Mostly one-way communication (broadcast)")

    print("\nMutual communication pairs:")
    for edge1_id, edge2_id in mutual_pairs:
        e1 = email.get_edge_by_id(edge1_id)
        e2 = email.get_edge_by_id(edge2_id)
        # Get labels by node IDs
        node1_id = e1[0]
        node2_id = e1[1]
        # Try to get labels, fall back to IDs
        try:
            node1 = email.get_node(node1_id)
            node2 = email.get_node(node2_id)
            node1_label = node1.label if node1.label else str(node1_id)
            node2_label = node2.label if node2.label else str(node2_id)
        except Exception:
            node1_label = str(node1_id)
            node2_label = str(node2_id)
        print(f"  {node1_label} ↔ {node2_label}")
    print()


def example_4_citation_network():
    """Example 4: Citation network (typically low reciprocity)."""
    print("=" * 70)
    print("EXAMPLE 4: Citation Network Analysis")
    print("=" * 70)

    # Academic citation network
    citations = ec.Graph(directed=True)
    papers = [
        ("Paper2023", "Paper2020"),
        ("Paper2023", "Paper2018"),
        ("Paper2022", "Paper2020"),
        ("Paper2022", "Paper2019"),
        ("Paper2021", "Paper2018"),
        ("Paper2021", "Paper2019"),
        ("Paper2020", "Paper2018"),
        ("Paper2020", "Paper2019"),
    ]

    for citing, cited in papers:
        if not citations.has_node(citing):
            citations.add_node(label=citing)
        if not citations.has_node(cited):
            citations.add_node(label=cited)
        citations.add_edge(citing, cited)

    recip = citations.reciprocity()

    print("\nCitation Network:")
    print(f"  Papers (nodes): {citations.num_nodes()}")
    print(f"  Citations (edges): {citations.num_edges()}")
    print(f"  Reciprocity: {recip:.3f}")

    print("\nInterpretation:")
    print("  → Very low reciprocity is expected in citation networks")
    print("  → Papers cite older papers, which cannot cite back")
    print("  → Reciprocal citations are rare and interesting!")

    mutual = citations.which_mutual()
    if len(mutual) > 0:
        print(f"\n  Found {len(mutual)} rare reciprocal citation(s)!")
    else:
        print("\n  No reciprocal citations (as expected)")
    print()


def example_5_detecting_mutual_edges():
    """Example 5: Using which_mutual() to identify bidirectional connections."""
    print("=" * 70)
    print("EXAMPLE 5: Detecting Mutual Edges")
    print("=" * 70)

    # Create a network with mix of mutual and one-way edges
    g = ec.Graph(directed=True)

    connections = {
        "Alice": ["Bob", "Carol", "David"],
        "Bob": ["Alice", "Carol"],  # Mutual with Alice
        "Carol": ["Bob"],  # Mutual with Bob
        "David": ["Carol"],  # One-way from Alice
        "Eve": ["Alice", "Bob"],  # One-way to Alice and Bob
    }

    for source, targets in connections.items():
        if not g.has_node(source):
            g.add_node(label=source)
        for target in targets:
            if not g.has_node(target):
                g.add_node(label=target)
            g.add_edge(source, target)

    print(f"\nNetwork: {g.num_nodes()} nodes, {g.num_edges()} edges")

    # Find all mutual edges
    mutual_pairs = g.which_mutual()
    print(f"\nFound {len(mutual_pairs)} mutual edge pair(s):")

    # Display mutual relationships
    for edge1_id, edge2_id in mutual_pairs:
        e1 = g.get_edge_by_id(edge1_id)
        e2 = g.get_edge_by_id(edge2_id)
        # Get labels by node IDs
        node1_id = e1[0]
        node2_id = e1[1]
        # Try to get labels, fall back to IDs
        try:
            node1 = g.get_node(node1_id)
            node2 = g.get_node(node2_id)
            node1_label = node1.label if node1.label else str(node1_id)
            node2_label = node2.label if node2.label else str(node2_id)
        except Exception:
            node1_label = str(node1_id)
            node2_label = str(node2_id)
        print(f"  {node1_label} ↔ {node2_label} (edges {edge1_id}, {edge2_id})")

    # Calculate overall reciprocity
    recip = g.reciprocity()
    print(f"\nOverall reciprocity: {recip:.3f}")
    print(f"({recip * 100:.1f}% of edges are part of mutual pairs)")
    print()


def example_6_comparing_modes():
    """Example 6: Understanding reciprocity calculation modes."""
    print("=" * 70)
    print("EXAMPLE 6: Reciprocity Calculation Modes")
    print("=" * 70)

    g = ec.Graph(directed=True)
    g.add_node(label="A")
    g.add_node(label="B")
    g.add_node(label="C")
    g.add_node(label="D")

    # 2 mutual pairs + 2 one-way edges
    g.add_edge("A", "B")
    g.add_edge("B", "A")  # Mutual
    g.add_edge("C", "D")
    g.add_edge("D", "C")  # Mutual
    g.add_edge("A", "C")  # One-way
    g.add_edge("B", "D")  # One-way

    print(f"\nGraph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print("  2 mutual pairs (4 edges)")
    print("  2 one-way edges")

    # Default mode
    recip_default = g.reciprocity("default")
    print(f"\nReciprocity (default): {recip_default:.3f}")
    print(f"  Calculation: {4} mutual / {6} total = {recip_default:.3f}")

    # Ratio mode (same as default)
    recip_ratio = g.reciprocity("ratio")
    print(f"\nReciprocity (ratio): {recip_ratio:.3f}")
    print("  Calculation: same as default")

    print("\nInterpretation:")
    print(f"  {recip_default * 100:.1f}% of edges have a reciprocal counterpart")
    print()


def main():
    """Run all examples."""
    example_1_basic_reciprocity()
    example_2_social_networks()
    example_3_communication_patterns()
    example_4_citation_network()
    example_5_detecting_mutual_edges()
    example_6_comparing_modes()


if __name__ == "__main__":
    main()
