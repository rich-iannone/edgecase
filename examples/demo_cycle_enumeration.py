import edgecase as ec

# Create a directed graph with multiple cycles
G = ec.Graph(directed=True).add_node("A").add_node("B").add_node("C").add_node("D").add_node("E")

# Add edges to create various cycles
G.add_edge("A", "B")
G.add_edge("B", "A")  # 2-cycle: A <-> B

G.add_edge("B", "C")
G.add_edge("C", "D")
G.add_edge("D", "B")  # 3-cycle: B -> C -> D -> B

G.add_edge("C", "E")
G.add_edge("E", "A")  # Creates larger cycles through the network

print("Graph Analysis: Cycle Enumeration")
print("=" * 50)
print(f"Nodes: {G.num_nodes()}")
print(f"Edges: {G.num_edges()}")
print()

# Find all cycles
all_cycles = G.get_cycles()
print(f"All elementary cycles found: {len(all_cycles)}")
for i, cycle in enumerate(all_cycles, 1):
    print(f"  {i}. {' -> '.join(cycle)} -> {cycle[0]} (length: {len(cycle)})")
print()

# Find only longer cycles (length >= 3)
long_cycles = G.get_cycles(min_length=3)
print(f"Cycles of length 3+: {len(long_cycles)}")
for i, cycle in enumerate(long_cycles, 1):
    print(f"  {i}. {' -> '.join(cycle)} -> {cycle[0]} (length: {len(cycle)})")
print()

# Check a simple triangle
print("\nSimple Triangle Example:")
print("-" * 50)
triangle = (
    ec.Graph(directed=True)
    .add_node("X")
    .add_node("Y")
    .add_node("Z")
    .add_edge("X", "Y")
    .add_edge("Y", "Z")
    .add_edge("Z", "X")
)

cycles = triangle.get_cycles()
print(f"Triangle cycles: {cycles}")
print(f"Cycle: {' -> '.join(cycles[0])} -> {cycles[0][0]}")
