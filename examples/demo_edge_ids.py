import edgecase as ec
import polars as pl


def demo_basic_multi_edges():
    """Basic example of multiple edges between same nodes."""
    print("=" * 70)
    print("1. BASIC MULTI-EDGE GRAPH")
    print("=" * 70)

    g = ec.Graph(directed=True)
    g.add_node("Alice", type="user")
    g.add_node("Bob", type="user")

    # Add multiple relationships between Alice and Bob
    g.add_edge("Alice", "Bob", rel="follows", since="2020")
    g.add_edge("Alice", "Bob", rel="messages", count=42)
    g.add_edge("Alice", "Bob", rel="likes_posts", score=0.85)

    print(f"Nodes: {g.count_nodes()}")
    print(f"Edges: {g.count_edges()}")
    print(f"Edge IDs: {g.get_edge_ids()}")
    print(f"\nEdges (id, source, target): {g.get_edges_as_tuples()}")
    print()


def demo_auto_increment():
    """Show auto-incrementing edge IDs."""
    print("=" * 70)
    print("2. AUTO-INCREMENT EDGE IDs")
    print("=" * 70)

    g = ec.Graph()
    for node in ["A", "B", "C", "D"]:
        g.add_node(node)

    # Add edges - IDs auto-increment
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("A", "B")  # Second edge A->B
    g.add_edge("C", "D")
    g.add_edge("A", "B")  # Third edge A->B

    print("Added 5 edges total")
    print(f"Edge IDs: {g.get_edge_ids()}")
    print("\nMultiple A->B edges:")
    for edge_id, src, tgt in g.get_edges_as_tuples():
        if src == "A" and tgt == "B":
            print(f"  Edge {edge_id}: {src} -> {tgt}")
    print()


def demo_edge_operations():
    """Demonstrate edge ID operations."""
    print("=" * 70)
    print("3. EDGE ID OPERATIONS")
    print("=" * 70)

    g = ec.Graph()
    g.add_node("X")
    g.add_node("Y")
    g.add_node("Z")

    g.add_edge("X", "Y", weight=1.0, label="first")
    g.add_edge("X", "Y", weight=2.0, label="second")
    g.add_edge("Y", "Z", weight=3.0, label="third")

    edge_ids = g.get_edge_ids()
    print(f"All edge IDs: {edge_ids}")

    # Get specific edge
    print(f"\nEdge {edge_ids[0]}: {g.get_node_ids_by_edge_id(edge_ids[0])}")
    print(f"Edge {edge_ids[1]}: {g.get_node_ids_by_edge_id(edge_ids[1])}")

    # Check if edge exists
    print(f"\nHas edge {edge_ids[0]}? {g.has_edge_id(edge_ids[0])}")
    print(f"Has edge 999? {g.has_edge_id(999)}")

    # Delete specific edge
    print(f"\nDeleting edge {edge_ids[1]}...")
    g.delete_edge_by_id(edge_ids[1])
    print(f"Remaining edges: {g.count_edges()}")
    print(f"Remaining edge IDs: {g.get_edge_ids()}")
    print()


def demo_dataframe_edge_ids():
    """Create graph from DataFrame with explicit edge IDs."""
    print("=" * 70)
    print("4. DATAFRAME WITH EXPLICIT EDGE IDs")
    print("=" * 70)

    nodes = pl.DataFrame(
        {
            "id": ["User1", "User2", "User3"],
            "type": ["person", "person", "person"],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    # EDF with edge IDs
    edges = pl.DataFrame(
        {
            "id": [100, 101, 102, 103],
            "from": ["User1", "User2", "User1", "User2"],
            "to": ["User2", "User3", "User2", "User3"],
            "rel": ["follows", "follows", "likes", "mentions"],
        }
    )

    g = ec.Graph(nodes_df=nodes, edges_df=edges, directed=True)

    print("Created graph from DataFrames")
    print(f"Nodes: {g.count_nodes()}, Edges: {g.count_edges()}")
    print(f"Edge IDs preserved: {g.get_edge_ids()}")
    print("\nUser1 -> User2 edges:")
    for edge_id, src, tgt in g.get_edges_as_tuples():
        if src == "User1" and tgt == "User2":
            print(f"  Edge {edge_id}: {src} -> {tgt}")
    print()


def demo_social_network():
    """Real-world example: Social network with multiple relationship types."""
    print("=" * 70)
    print("5. REAL-WORLD: SOCIAL NETWORK")
    print("=" * 70)

    g = ec.Graph(directed=True)

    # Create users
    users = ["Alice", "Bob", "Charlie", "Diana"]
    for user in users:
        g.add_node(user, type="user", status="active")

    # Multiple interactions between users
    g.add_edge("Alice", "Bob", rel="follows", timestamp="2024-01-01")
    g.add_edge("Alice", "Bob", rel="messaged", count=15, last="2024-11-01")
    g.add_edge("Alice", "Bob", rel="tagged_in_photo", photos=3)

    g.add_edge("Bob", "Charlie", rel="follows", timestamp="2024-01-15")
    g.add_edge("Bob", "Charlie", rel="commented_on", posts=8)

    g.add_edge("Charlie", "Diana", rel="follows", timestamp="2024-02-01")
    g.add_edge("Diana", "Alice", rel="follows", timestamp="2024-02-15")

    print(f"Social network: {g.count_nodes()} users, {g.count_edges()} interactions")
    print(f"Edge IDs: {g.get_edge_ids()}")

    # Find all Alice -> Bob interactions
    alice_bob = [(eid, s, t) for eid, s, t in g.get_edges_as_tuples() if s == "Alice" and t == "Bob"]
    print(f"\nAlice -> Bob has {len(alice_bob)} different interaction types:")
    for eid, s, t in alice_bob:
        print(f"  Edge {eid}: {s} -> {t}")
    print()


def demo_transaction_network():
    """Real-world example: Financial transactions."""
    print("=" * 70)
    print("6. REAL-WORLD: TRANSACTION NETWORK")
    print("=" * 70)

    g = ec.Graph(directed=True)

    # Bank accounts
    accounts = ["Acct_001", "Acct_002", "Acct_003"]
    for acc in accounts:
        g.add_node(acc, type="account")

    # Multiple transactions between accounts (now possible with edge IDs!)
    transactions = [
        ("Acct_001", "Acct_002", 100.50, "2024-01-01"),
        ("Acct_001", "Acct_002", 250.00, "2024-01-05"),
        ("Acct_001", "Acct_002", 75.25, "2024-01-10"),
        ("Acct_002", "Acct_003", 200.00, "2024-01-12"),
        ("Acct_003", "Acct_001", 300.00, "2024-01-15"),  # Potential cycle
    ]

    for src, tgt, amount, date in transactions:
        g.add_edge(src, tgt, amount=amount, date=date, type="transaction")

    print(f"Transaction network: {g.count_nodes()} accounts, {g.count_edges()} transactions")
    print(f"Transaction IDs: {g.get_edge_ids()}")

    # Analyze transactions between Acct_001 and Acct_002
    acct1_to_2 = [
        (eid, s, t) for eid, s, t in g.get_edges_as_tuples() if s == "Acct_001" and t == "Acct_002"
    ]
    print(f"\nAcct_001 -> Acct_002: {len(acct1_to_2)} transactions")
    for eid, s, t in acct1_to_2:
        print(f"  Transaction {eid}: {s} -> {t}")

    print("\n💡 Before edge IDs, we couldn't have multiple transactions between")
    print("   the same accounts - duplicates were silently ignored!")
    print()


def demo_comparison():
    """Show the difference: before vs after edge IDs."""
    print("=" * 70)
    print("7. BEFORE vs AFTER: The Edge ID Impact")
    print("=" * 70)

    print("BEFORE (old behavior - duplicates ignored):")
    print("  - add_edge('A', 'B', rel='follows')")
    print("  - add_edge('A', 'B', rel='likes')    # Would be silently ignored!")
    print("  - Result: Only 1 edge")

    print("\nAFTER (with edge IDs - multi-edges supported):")
    g = ec.Graph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", rel="follows")
    g.add_edge("A", "B", rel="likes")
    print(f"  - add_edge('A', 'B', rel='follows')  # Edge ID: {g.get_edge_ids()[0]}")
    print(f"  - add_edge('A', 'B', rel='likes')    # Edge ID: {g.get_edge_ids()[1]}")
    print(f"  - Result: {g.count_edges()} edges ✅")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EDGE ID FUNCTIONALITY DEMO")
    print("=" * 70 + "\n")

    demo_basic_multi_edges()
    demo_auto_increment()
    demo_edge_operations()
    demo_dataframe_edge_ids()
    demo_social_network()
    demo_transaction_network()
    demo_comparison()

    print("=" * 70)
    print("✅ All demos completed successfully!")
    print("=" * 70)
