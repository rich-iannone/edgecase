from edgecase import Graph


def demo_basic_selection():
    """Demo 1: Basic node and edge selection."""
    print("=" * 70)
    print("Demo 1: Basic Node and Edge Selection")
    print("=" * 70)

    G = Graph()

    # Build a simple social network
    edges = [
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Bob", "David"),
        ("Charlie", "David"),
        ("David", "Eve"),
        ("Eve", "Frank"),
    ]

    for src, tgt in edges:
        G.add_edge(src, tgt)

    print(f"\nGraph: {G.num_nodes()} nodes, {G.num_edges()} edges")

    # Select specific nodes
    G.select_nodes_by_id(["Alice", "David"])
    print(f"\nSelected nodes: {G.get_selected_nodes()}")
    print(f"Count: {G.count_selected_nodes()}")

    # Select edges connected to selected nodes
    G.clear_selection()
    G.select_nodes_by_id(["David"])
    G.select_edges_by_node_id(["David"])

    print(f"\nEdges connected to David: {G.count_selected_edges()} edges")
    selected_edge_ids = G.get_selected_edges()
    edges_info = G.edges_as_tuples()
    for eid in selected_edge_ids:
        edge = [e for e in edges_info if e[0] == eid][0]
        print(f"  Edge {edge[0]}: {edge[1]} -> {edge[2]}")


def demo_degree_filtering():
    """Demo 2: Selecting nodes by degree (hub detection)."""
    print("\n" + "=" * 70)
    print("Demo 2: Hub Detection via Degree Filtering")
    print("=" * 70)

    G = Graph()

    # Create a star network (hub-and-spoke)
    hub = "Hub"
    spokes = [f"Node_{i}" for i in range(1, 11)]

    for spoke in spokes:
        G.add_edge(hub, spoke)

    # Add some additional connections
    G.add_edge("Node_1", "Node_2")
    G.add_edge("Node_1", "Node_3")
    G.add_edge("Node_2", "Node_3")

    print(f"\nGraph: {G.num_nodes()} nodes, {G.num_edges()} edges")

    # Find high-degree nodes (hubs)
    G.select_nodes_by_degree(min_degree=5)
    hubs = G.get_selected_nodes()

    print("\nNodes with degree >= 5 (hubs):")
    for node in hubs:
        degree = len(G.neighbors(node))
        print(f"  {node}: degree = {degree}")

    # Find low-degree nodes (peripheral)
    G.clear_selection()
    G.select_nodes_by_degree(max_degree=2)
    peripheral = G.get_selected_nodes()

    print(f"\nNodes with degree <= 2 (peripheral): {len(peripheral)} nodes")
    print(f"  Examples: {peripheral[:5]}")


def demo_directed_graph_degree():
    """Demo 3: In-degree and out-degree in directed graphs."""
    print("\n" + "=" * 70)
    print("Demo 3: Directed Graph - In-degree and Out-degree")
    print("=" * 70)

    G = Graph(directed=True)

    # Build a citation network
    papers = {
        "P1": [],  # No citations (new paper)
        "P2": ["P1"],  # Cites P1
        "P3": ["P1", "P2"],  # Review paper
        "P4": ["P1"],  # Cites P1
        "P5": ["P3", "P4"],  # Cites P3 and P4
        "P6": ["P1", "P2", "P3"],  # Comprehensive review
    }

    for paper, citations in papers.items():
        for cited in citations:
            G.add_edge(paper, cited)  # paper -> cited

    print(f"\nCitation network: {G.num_nodes()} papers, {G.num_edges()} citations")

    # Find highly cited papers (high in-degree)
    G.select_nodes_by_degree(min_degree=3, degree_type="in")
    highly_cited = G.get_selected_nodes()

    print("\nHighly cited papers (in-degree >= 3):")
    for paper in highly_cited:
        # Count incoming citations
        citations = [e for e in G.edges_as_tuples() if e[2] == paper]
        print(f"  {paper}: {len(citations)} citations")

    # Find papers that cite many others (high out-degree)
    G.clear_selection()
    G.select_nodes_by_degree(min_degree=3, degree_type="out")
    review_papers = G.get_selected_nodes()

    print("\nReview papers (out-degree >= 3):")
    for paper in review_papers:
        citations = [e for e in G.edges_as_tuples() if e[1] == paper]
        cited_papers = [e[2] for e in citations]
        print(f"  {paper}: cites {cited_papers}")


def demo_selection_chaining():
    """Demo 4: Chaining selection operations."""
    print("\n" + "=" * 70)
    print("Demo 4: Selection Chaining and Refinement")
    print("=" * 70)

    G = Graph()

    # Build a collaboration network
    collaborations = [
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Alice", "David"),
        ("Bob", "Charlie"),
        ("Bob", "Eve"),
        ("Charlie", "David"),
        ("Charlie", "Frank"),
        ("David", "Eve"),
        ("Eve", "Frank"),
        ("Frank", "George"),
    ]

    for src, tgt in collaborations:
        G.add_edge(src, tgt)

    print(f"\nCollaboration network: {G.num_nodes()} people")

    # Find well-connected people (degree >= 3)
    G.select_nodes_by_degree(min_degree=3)
    well_connected = G.get_selected_nodes()

    print(f"\nWell-connected people (degree >= 3): {well_connected}")

    # Now deselect one person
    G.deselect_nodes(["Bob"])
    remaining = G.get_selected_nodes()

    print(f"After removing Bob: {remaining}")

    # Invert to find less connected people
    G.invert_node_selection()
    less_connected = G.get_selected_nodes()

    print(f"\nLess connected people (degree < 3): {less_connected}")


def demo_complex_workflow():
    """Demo 5: Complex workflow - analyzing a transaction network."""
    print("\n" + "=" * 70)
    print("Demo 5: Transaction Network Analysis")
    print("=" * 70)

    G = Graph(directed=True)

    # Simulate financial transactions
    transactions = [
        ("Account_A", "Account_B"),
        ("Account_A", "Account_C"),
        ("Account_A", "Account_D"),
        ("Account_B", "Account_E"),
        ("Account_C", "Account_E"),
        ("Account_D", "Account_E"),
        ("Account_E", "Account_F"),  # E is a hub
        ("Account_E", "Account_G"),
        ("Account_E", "Account_H"),
        ("Account_F", "Account_I"),
        ("Account_G", "Account_I"),
    ]

    for src, tgt in transactions:
        G.add_edge(src, tgt)

    print(f"\nTransaction network: {G.num_nodes()} accounts, {G.num_edges()} transactions")

    # Step 1: Find accounts with high outgoing transactions (potential sources)
    G.select_nodes_by_degree(min_degree=3, degree_type="out")
    sources = G.get_selected_nodes()
    print(f"\nAccounts with many outgoing transactions: {sources}")

    # Step 2: Find accounts with high incoming transactions (potential sinks/aggregators)
    G.clear_selection()
    G.select_nodes_by_degree(min_degree=3, degree_type="in")
    sinks = G.get_selected_nodes()
    print(f"Accounts with many incoming transactions: {sinks}")

    # Step 3: Select all transactions involving these key accounts
    G.select_edges_by_node_id(sources + sinks)
    key_transactions = G.get_selected_edges()

    print(f"\nKey transactions involving sources/sinks: {len(key_transactions)} transactions")
    print(f"This represents {len(key_transactions) / G.num_edges() * 100:.1f}% of all transactions")


def demo_subgraph_extraction():
    """Demo 6: Using selection to extract subgraphs."""
    print("\n" + "=" * 70)
    print("Demo 6: Subgraph Extraction via Selection")
    print("=" * 70)

    G = Graph()

    # Build a larger network
    edges = [
        # Component 1: Triangle
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        # Component 2: Square
        ("D", "E"),
        ("E", "F"),
        ("F", "G"),
        ("G", "D"),
        # Component 3: Line
        ("H", "I"),
        ("I", "J"),
    ]

    for src, tgt in edges:
        G.add_edge(src, tgt)

    print(f"\nFull graph: {G.num_nodes()} nodes, {G.num_edges()} edges")

    # Select nodes from component 1 (triangle)
    G.select_nodes_by_id(["A", "B", "C"])
    component1_nodes = G.get_selected_nodes()

    # Select edges within this component
    G.select_edges_by_node_id(component1_nodes)
    component1_edges = G.get_selected_edges()

    print("\nComponent 1 (triangle):")
    print(f"  Nodes: {component1_nodes}")
    print(f"  Edges: {len(component1_edges)} edges")

    # Now select component 2 (square)
    G.clear_selection()
    G.select_nodes_by_id(["D", "E", "F", "G"])
    G.select_edges_by_node_id(G.get_selected_nodes())

    print("\nComponent 2 (square):")
    print(f"  Nodes: {G.get_selected_nodes()}")
    print(f"  Edges: {G.count_selected_edges()} edges")


def demo_method_chaining():
    """Demo 7: Fluent API with method chaining."""
    print("\n" + "=" * 70)
    print("Demo 7: Fluent API with Method Chaining")
    print("=" * 70)

    G = Graph(directed=True)

    # Build a simple network
    for i in range(1, 11):
        G.add_edge(f"Node_{i}", f"Node_{(i % 10) + 1}")
        if i <= 5:
            G.add_edge(f"Node_{i}", f"Node_{i + 5}")

    print(f"\nGraph: {G.num_nodes()} nodes, {G.num_edges()} edges")

    # Chain multiple operations
    result = (
        G.select_nodes_by_degree(min_degree=2, degree_type="out")
        .deselect_nodes(["Node_1"])
        .invert_node_selection()
    )

    print("\nAfter chained operations:")
    print(f"  Selected nodes: {result.get_selected_nodes()}")
    print(f"  Count: {result.count_selected_nodes()}")

    print("\nMethod chaining allows fluent, readable queries!")


def demo_select_by_attributes():
    """Demo 8: Selecting nodes and edges by attributes using predicates."""
    print("\n" + "=" * 70)
    print("Demo 8: Selection by Attributes (Predicate Functions)")
    print("=" * 70)

    G = Graph()

    # Build a knowledge graph with attributes
    G.add_node("Python", category="language", popularity=95, year=1991)
    G.add_node("JavaScript", category="language", popularity=98, year=1995)
    G.add_node("Rust", category="language", popularity=85, year=2010)
    G.add_node("NumPy", category="library", popularity=90, year=2006)
    G.add_node("React", category="library", popularity=92, year=2013)

    G.add_edge("NumPy", "Python", relationship="depends_on", strength=1.0)
    G.add_edge("React", "JavaScript", relationship="depends_on", strength=1.0)

    print(f"\nKnowledge graph: {G.num_nodes()} nodes, {G.num_edges()} edges")

    # Select popular languages (popularity > 90)
    G.select_nodes(
        lambda n: (
            n.attributes
            and n.attributes.get("category") == "language"
            and n.attributes.get("popularity", 0) > 90
        )
    )

    popular_languages = G.get_selected_nodes()
    print(f"\nPopular languages (popularity > 90): {popular_languages}")

    # Select modern technologies (created after 2000)
    G.clear_selection()
    G.select_nodes(lambda n: n.attributes and n.attributes.get("year", 0) >= 2000)

    modern_tech = G.get_selected_nodes()
    print(f"Modern technologies (year >= 2000): {modern_tech}")

    # Select strong dependency edges
    G.select_edges(
        lambda e: (
            e.attributes
            and e.attributes.get("relationship") == "depends_on"
            and e.attributes.get("strength", 0) == 1.0
        )
    )

    print(f"\nStrong dependencies: {G.count_selected_edges()} edges")


def demo_advanced_filtering():
    """Demo 9: Advanced filtering with complex predicates."""
    print("\n" + "=" * 70)
    print("Demo 9: Advanced Filtering - E-commerce Product Network")
    print("=" * 70)

    G = Graph(directed=True)

    # Products with attributes
    G.add_node("Laptop", price=1200, rating=4.5, category="electronics", in_stock=True)
    G.add_node("Mouse", price=25, rating=4.2, category="electronics", in_stock=True)
    G.add_node("Keyboard", price=80, rating=4.7, category="electronics", in_stock=False)
    G.add_node("Monitor", price=350, rating=4.6, category="electronics", in_stock=True)
    G.add_node("Desk", price=200, rating=4.1, category="furniture", in_stock=True)

    # "Frequently bought together" relationships
    G.add_edge("Laptop", "Mouse", frequency=150, conversion_rate=0.75)
    G.add_edge("Laptop", "Keyboard", frequency=120, conversion_rate=0.65)
    G.add_edge("Monitor", "Keyboard", frequency=80, conversion_rate=0.55)
    G.add_edge("Desk", "Monitor", frequency=60, conversion_rate=0.45)

    print(f"\nProduct network: {G.num_nodes()} products")

    # Find premium, highly-rated, in-stock electronics
    def premium_quality_available(node):
        if not node.attributes:
            return False
        attrs = node.attributes
        return (
            attrs.get("category") == "electronics"
            and attrs.get("price", 0) > 100
            and attrs.get("rating", 0) >= 4.5
            and attrs.get("in_stock")
        )

    G.select_nodes(premium_quality_available)
    premium_products = G.get_selected_nodes()

    print("\nPremium electronics (price>100, rating>=4.5, in stock):")
    for product in premium_products:
        node = G.get_node(product)
        print(f"  {product}: ${node.attributes['price']}, ★{node.attributes['rating']}")

    # Find high-conversion product pairs
    G.clear_selection()
    G.select_edges(
        lambda e: (
            e.attributes
            and e.attributes.get("conversion_rate", 0) >= 0.6
            and e.attributes.get("frequency", 0) >= 100
        )
    )

    high_conversion = G.get_selected_edges()
    print("\nHigh-conversion product pairs (conversion>=60%, frequency>=100):")
    for edge_id in high_conversion:
        edge = G.get_edge(edge_id)
        print(
            f"  {edge.source} → {edge.target}: {edge.attributes['conversion_rate'] * 100:.0f}% conversion"
        )


def demo_social_network_analysis():
    """Demo 10: Social network analysis with attribute-based selection."""
    print("\n" + "=" * 70)
    print("Demo 10: Social Network Analysis with Predicates")
    print("=" * 70)

    G = Graph()

    # Users with various attributes
    users = {
        "Alice": {"reputation": 95, "posts": 200, "verified": True, "country": "USA"},
        "Bob": {"reputation": 45, "posts": 50, "verified": False, "country": "Canada"},
        "Charlie": {"reputation": 88, "posts": 150, "verified": True, "country": "USA"},
        "Diana": {"reputation": 62, "posts": 80, "verified": False, "country": "UK"},
        "Eve": {"reputation": 91, "posts": 180, "verified": True, "country": "USA"},
    }

    for user, attrs in users.items():
        G.add_node(user, **attrs)

    # Interactions with strength
    G.add_edge("Alice", "Charlie", interaction_count=45, interaction_type="collaboration")
    G.add_edge("Alice", "Eve", interaction_count=60, interaction_type="collaboration")
    G.add_edge("Bob", "Diana", interaction_count=20, interaction_type="discussion")
    G.add_edge("Charlie", "Eve", interaction_count=35, interaction_type="collaboration")
    G.add_edge("Diana", "Bob", interaction_count=15, interaction_type="discussion")

    print(f"\nSocial network: {G.num_nodes()} users, {G.num_edges()} interactions")

    # Find influential users (high reputation, verified, active)
    def influential_user(node):
        if not node.attributes:
            return False
        attrs = node.attributes
        return (
            attrs.get("reputation", 0) >= 85
            and attrs.get("verified") == True
            and attrs.get("posts", 0) >= 150
        )

    G.select_nodes(influential_user)
    influencers = G.get_selected_nodes()

    print("\nInfluential users:")
    for user in influencers:
        node = G.get_node(user)
        attrs = node.attributes
        print(f"  {user}: {attrs['reputation']} reputation, {attrs['posts']} posts")

    # Find strong collaboration edges
    G.clear_selection()
    G.select_edges(
        lambda e: (
            e.attributes
            and e.attributes.get("interaction_type") == "collaboration"
            and e.attributes.get("interaction_count", 0) >= 40
        )
    )

    strong_collabs = G.get_selected_edges()
    print("\nStrong collaborations (40+ interactions):")
    for edge_id in strong_collabs:
        edge = G.get_edge(edge_id)
        count = edge.attributes.get("interaction_count")
        print(f"  {edge.source} ↔ {edge.target}: {count} interactions")

    # Complex query: US-based verified users who collaborate with influencers
    G.clear_selection()
    influencer_ids = {user for user in influencers}

    G.select_nodes(
        lambda n: (
            n.attributes
            and n.attributes.get("country") == "USA"
            and n.attributes.get("verified") == True
        )
    )
    us_verified = set(G.get_selected_nodes())

    # Find those who collaborate with influencers
    G.select_edges(
        lambda e: (
            e.source in us_verified
            and e.target in influencer_ids
            and e.attributes
            and e.attributes.get("interaction_type") == "collaboration"
        )
    )

    print(
        f"\nUS verified users collaborating with influencers: {G.count_selected_edges()} connections"
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EdgeCase Selection System Demo")
    print("=" * 70)

    demo_basic_selection()
    demo_degree_filtering()
    demo_directed_graph_degree()
    demo_selection_chaining()
    demo_complex_workflow()
    demo_subgraph_extraction()
    demo_method_chaining()
    demo_select_by_attributes()
    demo_advanced_filtering()
    demo_social_network_analysis()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
