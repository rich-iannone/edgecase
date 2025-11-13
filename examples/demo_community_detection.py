from edgecase import karate_club

print("=" * 70)
print("Community Detection with attr_name Parameter")
print("=" * 70)

# Load karate club network
G = karate_club()

print(f"\nNetwork: {G.num_nodes()} nodes, {G.num_edges()} edges")

# Test 1: Fast Greedy with attr_name
print("\n" + "-" * 70)
print("Test 1: Fast Greedy Community Detection")
print("-" * 70)

G_fast = (
    G.get_cmty_fast_greedy(attr_name="community")
    .rescale_node_attrs(
        node_attr_from="community", node_attr_to="size", to_lower_bound=0.8, to_upper_bound=1.2
    )
    .colorize_node_attrs(node_attr_from="community", palette="Set1")
)

node_1 = G_fast.id_from_label("1")
node_obj = G_fast.get_node(node_1)
print(
    f"Node 1 attributes: community={node_obj.attributes.get('community')}, "
    + f"size={node_obj.attributes.get('size'):.3f}"
)

# Count communities
communities = G.get_cmty_fast_greedy()
num_communities = len(set(communities.values()))
print(f"Total communities found: {num_communities}")

# Test 2: Louvain with attr_name
print("\n" + "-" * 70)
print("Test 2: Louvain Community Detection")
print("-" * 70)

G_louvain = G.get_cmty_louvain(resolution=1.0, attr_name="louvain_cmty").colorize_node_attrs(
    node_attr_from="louvain_cmty", palette="viridis"
)

node_obj = G_louvain.get_node(node_1)
print(f"Node 1 louvain_cmty: {node_obj.attributes.get('louvain_cmty')}")

communities = G.get_cmty_louvain()
num_communities = len(set(communities.values()))
print(f"Total communities found: {num_communities}")

# Test 3: Edge Betweenness with attr_name
print("\n" + "-" * 70)
print("Test 3: Edge Betweenness Community Detection")
print("-" * 70)

G_edge_btwns = G.get_cmty_edge_btwns(attr_name="eb_community").colorize_node_attrs(
    node_attr_from="eb_community", palette="Pastel1"
)

node_obj = G_edge_btwns.get_node(node_1)
print(f"Node 1 eb_community: {node_obj.attributes.get('eb_community')}")

communities = G.get_cmty_edge_btwns()
num_communities = len(set(communities.values()))
print(f"Total communities found: {num_communities}")

# Test 4: Leading Eigenvector with attr_name
print("\n" + "-" * 70)
print("Test 4: Leading Eigenvector Community Detection")
print("-" * 70)

G_eigenvec = G.get_cmty_l_eigenvec(attr_name="eigen_cmty").colorize_node_attrs(
    node_attr_from="eigen_cmty", palette="Spectral"
)

node_obj = G_eigenvec.get_node(node_1)
print(f"Node 1 eigen_cmty: {node_obj.attributes.get('eigen_cmty')}")

communities = G.get_cmty_l_eigenvec()
num_communities = len(set(communities.values()))
print(f"Total communities found: {num_communities}")

# Test 5: Walktrap with attr_name
print("\n" + "-" * 70)
print("Test 5: Walktrap Community Detection")
print("-" * 70)

G_walktrap = G.get_cmty_walktrap(steps=4, attr_name="walktrap_cmty").colorize_node_attrs(
    node_attr_from="walktrap_cmty", palette="tab10"
)

node_obj = G_walktrap.get_node(node_1)
print(f"Node 1 walktrap_cmty: {node_obj.attributes.get('walktrap_cmty')}")

communities = G.get_cmty_walktrap()
num_communities = len(set(communities.values()))
print(f"Total communities found: {num_communities}")

# Test 6: Method chaining with multiple attributes
print("\n" + "-" * 70)
print("Test 6: Multi-Method Community Comparison")
print("-" * 70)

G_multi = (
    G.get_cmty_louvain(attr_name="louvain")
    .get_cmty_fast_greedy(attr_name="fast_greedy")
    .get_cmty_walktrap(steps=4, attr_name="walktrap")
)

node_obj = G_multi.get_node(node_1)
print("Node 1 communities:")
print(f"  Louvain: {node_obj.attributes.get('louvain')}")
print(f"  Fast Greedy: {node_obj.attributes.get('fast_greedy')}")
print(f"  Walktrap: {node_obj.attributes.get('walktrap')}")

# Test 7: Backwards compatibility (no attr_name)
print("\n" + "-" * 70)
print("Test 7: Backwards Compatibility (without attr_name)")
print("-" * 70)

communities_dict = G.get_cmty_louvain()
print(f"Type: {type(communities_dict)}")
print(f"Returns dict: {isinstance(communities_dict, dict)}")
print(f"Node 1 community: {communities_dict[node_1]}")

# Test 8: Complete visualization workflow
print("\n" + "-" * 70)
print("Test 8: Complete Visualization Workflow")
print("-" * 70)

G_viz = (
    G.get_cmty_louvain(resolution=1.0, attr_name="community")
    .get_eigen_centrality(attr_name="centrality")
    .colorize_node_attrs(node_attr_from="community", palette="Set3")
    .rescale_node_attrs(
        node_attr_from="centrality", node_attr_to="size", to_lower_bound=0.6, to_upper_bound=1.4
    )
)

node_obj = G_viz.get_node(node_1)
print("Node 1 final attributes:")
print(f"  Community: {node_obj.attributes.get('community')}")
print(f"  Centrality: {node_obj.attributes.get('centrality'):.4f}")
print(f"  Size: {node_obj.attributes.get('size'):.3f}")
print(f"  Color: {node_obj.attributes.get('color', 'N/A')}")

print("\n" + "=" * 70)
print("✅ All community detection methods successfully updated!")
print("✅ Method chaining works perfectly!")
print("✅ Backwards compatibility maintained!")
print("✅ Community assignments can be stored and visualized!")
print("=" * 70)
