import edgecase as ec


def main():
    print("=" * 60)
    print("Johnson Graph J(4,2) - 2-subsets of {0,1,2,3}")
    print("=" * 60)

    # Create J(4,2)
    g1 = ec.Graph(directed=False).add_johnson_graph(n=4, k=2)

    print(f"Nodes: {g1.num_nodes()} (C(4,2) = 6)")
    print(f"Edges: {g1.num_edges()}")

    # Show node labels (the 2-subsets)
    nodes = sorted([g1.label_from_id(nid) for nid in g1.nodes()])
    print(f"\nNode labels (2-subsets): {nodes}")

    # Show regularity
    degrees = [g1.degree(nid) for nid in g1.nodes()]
    print(f"Degree: {degrees[0]} (k(n-k) = 2*2 = 4)")
    print(f"Is regular: {len(set(degrees)) == 1}")

    # Show adjacency for one node
    node_01 = g1.get_id_from_label("0,1")
    neighbors = [g1.label_from_id(nid) for nid in g1.neighbors(node_01)]
    print(f"\nNeighbors of '0,1': {sorted(neighbors)}")
    print("Each neighbor intersects '0,1' in exactly 1 element")

    print("\n" + "=" * 60)
    print("Johnson Graph J(5,2) - 2-subsets of {0,1,2,3,4}")
    print("=" * 60)

    # Create J(5,2)
    g2 = ec.Graph(directed=False).add_johnson_graph(n=5, k=2)

    print(f"Nodes: {g2.num_nodes()} (C(5,2) = 10)")
    print(f"Edges: {g2.num_edges()}")

    # Show node labels
    nodes2 = sorted([g2.label_from_id(nid) for nid in g2.nodes()])
    print(f"\nNode labels: {nodes2}")

    # Show regularity
    degrees2 = [g2.degree(nid) for nid in g2.nodes()]
    print(f"Degree: {degrees2[0]} (k(n-k) = 2*3 = 6)")
    print(f"Is regular: {len(set(degrees2)) == 1}")

    print("\n" + "=" * 60)
    print("Johnson Graph J(5,1) - Singletons (Complete Graph K5)")
    print("=" * 60)

    # Create J(5,1) - singletons
    g3 = ec.Graph(directed=False).add_johnson_graph(n=5, k=1)

    print(f"Nodes: {g3.num_nodes()} (C(5,1) = 5)")
    print(f"Edges: {g3.num_edges()}")

    nodes3 = sorted([g3.label_from_id(nid) for nid in g3.nodes()])
    print(f"Node labels (singletons): {nodes3}")

    degrees3 = [g3.degree(nid) for nid in g3.nodes()]
    print(f"Degree: {degrees3[0]} (k(n-k) = 1*4 = 4)")
    print("This is the complete graph K5 (all pairs connected)")

    print("\n" + "=" * 60)
    print("Edge Cases")
    print("=" * 60)

    # J(5,0) - empty set
    g4 = ec.Graph(directed=False).add_johnson_graph(n=5, k=0)
    print(f"J(5,0): {g4.num_nodes()} node, {g4.num_edges()} edges (single empty set)")

    # J(5,5) - full set
    g5 = ec.Graph(directed=False).add_johnson_graph(n=5, k=5)
    print(f"J(5,5): {g5.num_nodes()} node, {g5.num_edges()} edges (single full set)")


if __name__ == "__main__":
    main()
