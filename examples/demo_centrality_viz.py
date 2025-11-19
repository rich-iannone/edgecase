import edgecase as ec
import polars as pl

# Example 1: Small social network with centrality visualization
print("=" * 60)
print("Example 1: Social Network with Centrality Visualization")
print("=" * 60)

nodes_df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7],
        "label": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"],
    }
)

edges_df = pl.DataFrame(
    {
        "from": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6],
        "to": [2, 3, 4, 3, 5, 4, 6, 7, 6, 7],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
)

# Streamlined approach: calculate centrality and apply visualization in one statement
graph = (
    ec.Graph(nodes_df=nodes_df, edges_df=edges_df, directed=False, weighted=True)
    .get_eigen_centrality(attr_name="centrality")  # Auto-store and chain
    .rescale_node_attrs(
        node_attr_from="centrality",
        node_attr_to="size",
        to_lower_bound=0.5,
        to_upper_bound=1.5,
    )
    .colorize_node_attrs(
        node_attr_from="centrality",
        node_attr_to="fillcolor",
        palette="RdBu",  # ColorBrewer diverging palette (blue to red)
    )
)

print("\nCentrality scores:")
for node_id in graph.get_node_ids(limit=None):
    node = graph.get_node(node_id)
    label = node.attributes.get("label", str(node_id))
    centrality_value = node.attributes["centrality"]
    print(f"  {label}: {centrality_value:.3f}")

print("\nVisualization settings applied:")
print("  - Node size: scaled by centrality (0.5 to 1.5)")
print("  - Node color: RdBu ColorBrewer palette (blue → white → red)")
print("  - Method chaining: no manual loops or normalization!")

# Show the graph (will display in browser or notebook)
# Uncomment the line below to display:
# graph.show(layout="neato")


# Example 2: Star network showing hub centrality
print("\n" + "=" * 60)
print("Example 2: Star Network (Hub has high centrality)")
print("=" * 60)

star_nodes = pl.DataFrame(
    {"id": list(range(1, 11)), "label": ["Hub"] + [f"Node{i}" for i in range(2, 11)]}
)

star_edges = pl.DataFrame({"from": [1] * 9, "to": list(range(2, 11))})

# Streamlined approach with more dramatic size scaling for the hub
star_graph = (
    ec.Graph(nodes_df=star_nodes, edges_df=star_edges, directed=False)
    .get_eigen_centrality(attr_name="centrality")
    .rescale_node_attrs(
        node_attr_from="centrality",
        node_attr_to="size",
        to_lower_bound=0.5,
        to_upper_bound=2.5,  # Larger range for dramatic hub effect
    )
    .colorize_node_attrs(
        node_attr_from="centrality",
        node_attr_to="fillcolor",
        palette="Reds",  # Sequential ColorBrewer palette (light to dark red)
    )
)

hub_centrality = star_graph.get_node(1).attributes["centrality"]
spoke_centrality = star_graph.get_node(2).attributes["centrality"]

print(f"\nHub centrality: {hub_centrality:.3f}")
print(f"Spoke centrality: {spoke_centrality:.3f}")

print("\nVisualization: Hub should be much larger and dark red, spokes smaller and light")

# Uncomment to display:
# star_graph.show(layout="neato")


# Example 3: Using different ColorBrewer palettes
print("\n" + "=" * 60)
print("Example 3: Exploring ColorBrewer Palettes")
print("=" * 60)

# Create a simple graph with labeled nodes
simple_nodes = pl.DataFrame({"id": [1, 2, 3, 4], "label": ["A", "B", "C", "D"]})
simple_edges = pl.DataFrame({"from": [1, 2, 3, 4, 1], "to": [2, 3, 4, 1, 3]})

# Apply Greens palette (sequential)
simple_graph = (
    ec.Graph(nodes_df=simple_nodes, edges_df=simple_edges, directed=False)
    .get_eigen_centrality(attr_name="centrality")
    .rescale_node_attrs("centrality", node_attr_to="size", to_lower_bound=0.6, to_upper_bound=1.2)
    .colorize_node_attrs("centrality", node_attr_to="fillcolor", palette="Greens")
)

print("\nApplying 'Greens' ColorBrewer palette (sequential):")
for node_id in simple_graph.get_node_ids(limit=None):
    node = simple_graph.get_node(node_id)
    label = node.attributes.get("label", node_id)
    centrality = node.attributes["centrality"]
    color = node.attributes["fillcolor"]
    size = node.attributes["size"]
    print(f"  Node {label}: centrality={centrality:.3f}, color={color}, size={size}")

# Uncomment to display:
# simple_graph.show(layout="circo")


# Example 4: Dual centrality measures
print("\n" + "=" * 60)
print("Example 4: Dual Centrality - Size + Color")
print("=" * 60)

# Use karate club network for richer example
karate = (
    ec.karate_club()
    .get_eigen_centrality(attr_name="eigen_centrality")  # Size mapping
    .get_alpha_centrality(attr_name="alpha_centrality")  # Color mapping
    .rescale_node_attrs(
        node_attr_from="eigen_centrality",
        node_attr_to="size",
        to_lower_bound=0.6,
        to_upper_bound=1.4,
    )
    .colorize_node_attrs(
        node_attr_from="alpha_centrality",
        node_attr_to="fillcolor",
        palette="viridis",  # Perceptually uniform, colorblind-friendly
    )
)

print("\nDual centrality visualization:")
print("  - Node size: eigenvector centrality (0.6 to 1.4)")
print("  - Node color: alpha centrality with viridis palette")
print("  - Zero manual normalization required!")

# Show first few nodes
print("\nSample nodes:")
for node_id in list(karate.get_node_ids(limit=None))[:5]:
    node = karate.get_node(node_id)
    eigen = node.attributes["eigen_centrality"]
    alpha = node.attributes["alpha_centrality"]
    print(f"  Node {node_id}: eigen={eigen:.3f}, alpha={alpha:.3f}")

# Uncomment to display:
# karate.show(layout="neato")


print("\n" + "=" * 60)
print("Available ColorBrewer palette categories:")
print("  • Sequential: Blues, Greens, Reds, Oranges, Purples, etc.")
print("  • Diverging: RdBu, RdYlGn, BrBG, PiYG, Spectral, etc.")
print("  • Qualitative: Set1, Set2, Paired, Dark2, etc.")
print("  • Viridis: viridis, magma, inferno, plasma, cividis")
print("\nTo see visualizations, uncomment the graph.show() lines")
print("Hover over nodes to see all attributes in tooltips!")
print("=" * 60)
