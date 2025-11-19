import edgecase as ec

print("=" * 70)
print("Comprehensive Centrality Methods Demo")
print("=" * 70)

# Create a simple network for demonstrations
print("\n1. Creating test network (karate club)...")
graph = ec.karate_club()
print(f"   Nodes: {graph.count_nodes()}, Edges: {graph.count_edges()}")

# Test all centrality methods with attr_name parameter
print("\n2. Computing all centrality measures with method chaining...")
graph = (
    graph.get_eigen_centrality(attr_name="eigen")
    .get_alpha_centrality(alpha=0.1, attr_name="alpha")
    .get_leverage_centrality(attr_name="leverage")
    .get_bonacich_power_centrality(beta=0.05, attr_name="bonacich")
    .get_katz_centrality(alpha=0.05, attr_name="katz")
)

# Show stored attributes for a sample node
node_1 = graph.get_node("1")
print("\n3. Stored centrality values for node 1:")
print(f"   Eigenvector: {node_1.attributes['eigen']:.4f}")
print(f"   Alpha: {node_1.attributes['alpha']:.4f}")
print(f"   Leverage: {node_1.attributes['leverage']:.4f}")
print(f"   Bonacich: {node_1.attributes['bonacich']:.4f}")
print(f"   Katz: {node_1.attributes['katz']:.4f}")

# Create a visualization pipeline
print("\n4. Creating multi-centrality visualization...")
viz_graph = (
    ec.karate_club()
    .get_eigen_centrality(attr_name="eigen")
    .get_leverage_centrality(attr_name="leverage")
    .rescale_node_attrs(
        node_attr_from="eigen",
        node_attr_to="size",
        to_lower_bound=0.6,
        to_upper_bound=1.6,
    )
    .colorize_node_attrs(
        node_attr_from="leverage",
        node_attr_to="fillcolor",
        palette="RdBu",  # Diverging palette for negative/positive leverage
    )
)
print("   ✅ Size scaled by eigenvector centrality")
print("   ✅ Color mapped by leverage centrality (RdBu palette)")

# HITS centrality for directed graphs
print("\n5. HITS centrality (hub and authority) for directed graphs...")
directed_graph = (
    ec.Graph(directed=True)
    .add_node(label="A")
    .add_node(label="B")
    .add_node(label="C")
    .add_node(label="D")
    .add_edge("A", "B")
    .add_edge("A", "C")
    .add_edge("B", "C")
    .add_edge("C", "D")
    .get_hub_centrality(attr_name="hub")
    .get_authority_centrality(attr_name="authority")
)

node_a = directed_graph.get_node("A")
print(
    f"   Node A (source): hub={node_a.attributes['hub']:.4f}, "
    f"authority={node_a.attributes['authority']:.4f}"
)
node_d = directed_graph.get_node("D")
print(
    f"   Node D (sink): hub={node_d.attributes['hub']:.4f}, "
    f"authority={node_d.attributes['authority']:.4f}"
)

# Compare multiple centrality measures
print("\n6. Comparing centrality measures across top nodes...")
comparison_graph = (
    ec.karate_club()
    .get_eigen_centrality(attr_name="eigen")
    .get_bonacich_power_centrality(beta=0.05, attr_name="bonacich")
    .get_katz_centrality(alpha=0.05, attr_name="katz")
)

# Get top 3 nodes by eigenvector centrality
eigen_scores = comparison_graph.get_node_attrs("eigen")
top_nodes = sorted(eigen_scores.items(), key=lambda x: x[1], reverse=True)[:3]

print("\n   Top 3 nodes by eigenvector centrality:")
for node_id, eigen_val in top_nodes:
    node = comparison_graph.get_node(node_id)
    print(f"   Node {node_id}:")
    print(f"     Eigenvector: {node.attributes['eigen']:.4f}")
    print(f"     Bonacich:    {node.attributes['bonacich']:.4f}")
    print(f"     Katz:        {node.attributes['katz']:.4f}")

print("\n" + "=" * 70)
print("Key Benefits of attr_name Parameter:")
print("  ✅ Automatic storage - no manual set_node_attrs() calls")
print("  ✅ Method chaining - fluent API for complex workflows")
print("  ✅ Backwards compatible - omit attr_name for dict return")
print("  ✅ Works with rescale_node_attrs() and colorize_node_attrs()")
print("=" * 70)

print("\nUpdated centrality methods:")
print("  • get_eigen_centrality(attr_name=None)")
print("  • get_alpha_centrality(alpha=0.1, attr_name=None)")
print("  • get_leverage_centrality(attr_name=None)")
print("  • get_bonacich_power_centrality(beta=0.1, attr_name=None)")
print("  • get_katz_centrality(alpha=0.1, attr_name=None)")
print("  • get_hub_centrality(attr_name=None)")
print("  • get_authority_centrality(attr_name=None)")
