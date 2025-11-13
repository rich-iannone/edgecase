import edgecase as ec
import polars as pl


def demo_path_graph():
    """Demonstrate path graph creation."""
    print("=" * 60)
    print("PATH GRAPH")
    print("=" * 60)

    g = ec.Graph(directed=True)
    g = g.add_path(n=5, type="step", label=True, rel="next")

    print(f"Created path graph: {g}")
    print(f"Nodes: {g.node_ids(limit=None)}")
    print(f"Edges: {g.edges_as_tuples()}")
    print(f"Node '1' neighbors: {g.neighbors('1')}")
    print()


def demo_cycle_graph():
    """Demonstrate cycle graph creation."""
    print("=" * 60)
    print("CYCLE GRAPH")
    print("=" * 60)

    g = ec.Graph(directed=True)
    g = g.add_cycle(n=4, type="state", rel="transitions_to")

    print(f"Created cycle graph: {g}")
    print("Each node has exactly 1 neighbor (directed cycle):")
    for node_id in g.node_ids(limit=None):
        print(f"  Node '{node_id}': {g.neighbors(node_id)}")
    print()


def demo_star_graph():
    """Demonstrate star graph creation."""
    print("=" * 60)
    print("STAR GRAPH")
    print("=" * 60)

    g = ec.Graph(directed=False)
    g = g.add_star(n=6, type="network")

    print(f"Created star graph: {g}")
    center = g.node_ids(limit=None)[0]  # First node is center
    print(f"Center node '{center}' has {len(g.neighbors(center))} neighbors")
    print(f"Center neighbors: {g.neighbors(center)}")
    print()


def demo_complete_graph():
    """Demonstrate complete graph creation."""
    print("=" * 60)
    print("COMPLETE GRAPH (CLIQUE)")
    print("=" * 60)

    g = ec.Graph(directed=False)
    g = ec.add_full_graph(g, n=4, type="clique")

    print(f"Created complete graph: {g}")
    print("Expected edges for K4: 4*(4-1)/2 = 6")
    print(f"Actual edges: {g.num_edges()}")
    print("Each node connects to all others:")
    for node_id in g.node_ids(limit=None):
        print(f"  Node '{node_id}': {len(g.neighbors(node_id))} neighbors")
    print()


def demo_balanced_tree():
    """Demonstrate balanced tree creation."""
    print("=" * 60)
    print("BALANCED BINARY TREE")
    print("=" * 60)

    g = ec.Graph(directed=True)
    g = ec.add_balanced_tree(g, k=2, h=2, type="tree_node", rel="parent_of")

    print(f"Created binary tree (k=2, h=2): {g}")
    print(f"Nodes at each level: 1, 2, 4 (total: {g.num_nodes()})")
    print(f"Root '1' children: {g.neighbors('1')}")
    print()


def demo_grid_graph():
    """Demonstrate 2D grid creation."""
    print("=" * 60)
    print("2D GRID GRAPH")
    print("=" * 60)

    g = ec.Graph(directed=False)
    g = ec.add_grid_2d(g, x=3, y=3, type="cell")

    print(f"Created 3x3 grid: {g}")
    print("Expected edges: (3-1)*3 + 3*(3-1) = 12")
    print(f"Actual edges: {g.num_edges()}")

    # Analyze node degrees
    corners = sum(1 for nid in g.node_ids(limit=None) if len(g.neighbors(nid)) == 2)
    edges = sum(1 for nid in g.node_ids(limit=None) if len(g.neighbors(nid)) == 3)
    interior = sum(1 for nid in g.node_ids(limit=None) if len(g.neighbors(nid)) == 4)

    print(f"Corner nodes (degree 2): {corners}")
    print(f"Edge nodes (degree 3): {edges}")
    print(f"Interior nodes (degree 4): {interior}")
    print()


def demo_prism_graph():
    """Demonstrate prism graph creation."""
    print("=" * 60)
    print("PRISM GRAPH")
    print("=" * 60)

    g = ec.Graph(directed=False)
    g = ec.add_prism(g, n=4, type="prism_node")

    print(f"Created prism graph (2 cycles of 4): {g}")
    print("Two parallel cycles connected by matching edges")
    print("Expected edges: 4 + 4 + 4 = 12")
    print(f"Actual edges: {g.num_edges()}")
    print()


def demo_sequential_building():
    """Demonstrate building complex graphs by combining structures."""
    print("=" * 60)
    print("SEQUENTIAL GRAPH BUILDING")
    print("=" * 60)

    g = ec.Graph(directed=False)

    print("Step 1: Add a path")
    g = g.add_path(n=3)
    print(f"  {g}")

    print("Step 2: Add a star")
    g = g.add_star(n=4)
    print(f"  {g}")

    print("Step 3: Add a cycle")
    g = g.add_cycle(n=5)
    print(f"  {g}")

    print("\nFinal combined graph has multiple disconnected components")
    print()


def demo_dataframe_creation():
    """Demonstrate creating graphs from Polars DataFrames."""
    print("=" * 60)
    print("DATAFRAME-BASED GRAPH CREATION")
    print("=" * 60)

    # Create node DataFrame (NDF)
    nodes = pl.DataFrame(
        {
            "id": ["Alice", "Bob", "Charlie", "Diana"],
            "type": ["user", "user", "admin", "user"],
            "label": ["A", "B", "C", "D"],
            "age": [25, 30, 35, 28],
        }
    )

    # Create edge DataFrame (EDF)
    edges = pl.DataFrame(
        {
            "from": ["Alice", "Bob", "Charlie"],
            "to": ["Bob", "Charlie", "Diana"],
            "rel": ["follows", "follows", "manages"],
            "weight": [1.0, 1.0, 2.0],
        }
    )

    print("Node DataFrame (NDF):")
    print(nodes)
    print()

    print("Edge DataFrame (EDF):")
    print(edges)
    print()

    g = ec.Graph.from_dataframes(nodes_df=nodes, edges_df=edges, directed=True)

    print(f"Created graph from DataFrames: {g}")
    print(f"Alice follows: {g.neighbors('Alice')}")
    print(f"Charlie manages: {g.neighbors('Charlie')}")
    print()


def demo_all():
    """Run all demonstrations."""
    print("\n")
    print("=" * 60)
    print("EDGECASE GRAPH BUILDERS DEMONSTRATION")
    print("=" * 60)
    print("\n")

    demo_path_graph()
    demo_cycle_graph()
    demo_star_graph()
    demo_complete_graph()
    demo_balanced_tree()
    demo_grid_graph()
    demo_prism_graph()
    demo_sequential_building()
    demo_dataframe_creation()

    print("=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    demo_all()
