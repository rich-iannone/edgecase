import edgecase as ec

print("=" * 70)
print("ASSORTATIVITY DEMONSTRATION")
print("=" * 70)

# ==============================================================================
# 1. DEGREE ASSORTATIVITY
# ==============================================================================
print("\n1. DEGREE ASSORTATIVITY")
print("-" * 70)

# Disassortative network: hub-and-spoke
print("\n  Hub-and-Spoke Network (Client-Server)")
g = ec.Graph(directed=False)
g.add_node("Server")
for client in ["Client1", "Client2", "Client3", "Client4"]:
    g.add_node(client)
    g.add_edge("Server", client)

r = g.get_assortativity_degree()
print(f"     Degree assortativity: {r:.3f}")
print("     Interpretation: Disassortative (high-degree server connects to low-degree clients)")

# ==============================================================================
# 2. NOMINAL (CATEGORICAL) ASSORTATIVITY
# ==============================================================================
print("\n\n2. NOMINAL (CATEGORICAL) ASSORTATIVITY")
print("-" * 70)

# Perfect homophily: only same-type connections
print("\n  a) Perfect Homophily (Segregated Network)")
g = ec.Graph(directed=False)
g.add_node("A", group="X")
g.add_node("B", group="X")
g.add_node("C", group="Y")
g.add_node("D", group="Y")
g.add_edge("A", "B")
g.add_edge("C", "D")

r = g.get_assortativity_nominal("group")
print(f"     Nominal assortativity: {r:.3f}")
print("     Interpretation: Perfect homophily (r = 1.0) - only same-group connections")

# Heterophily: different types connect
print("\n  b) Heterophily (Bipartite Network)")
g = ec.Graph(directed=False)
g.add_node("A", type="X")
g.add_node("B", type="X")
g.add_node("C", type="Y")
g.add_node("D", type="Y")
g.add_edge("A", "C")
g.add_edge("A", "D")
g.add_edge("B", "C")
g.add_edge("B", "D")

r = g.get_assortativity_nominal("type")
print(f"     Nominal assortativity: {r:.3f}")
print(f"     Interpretation: Disassortative (r = {r:.2f}) - different types connect")

# ==============================================================================
# 3. NUMERIC ASSORTATIVITY
# ==============================================================================
print("\n\n3. NUMERIC ASSORTATIVITY")
print("-" * 70)

# Positive correlation: similar ages connect
print("\n  Age-Based Friendships")
g = ec.Graph(directed=False)
g.add_node("Alice", age=25)
g.add_node("Bob", age=27)
g.add_node("Carol", age=26)
g.add_node("Dave", age=55)
g.add_node("Eve", age=53)
g.add_node("Frank", age=54)
g.add_edge("Alice", "Bob")
g.add_edge("Bob", "Carol")
g.add_edge("Alice", "Carol")
g.add_edge("Dave", "Eve")
g.add_edge("Eve", "Frank")

r = g.get_assortativity("age")
print(f"     Numeric assortativity: {r:.3f}")
print(f"     Interpretation: Positive correlation (r = {r:.2f}) - similar ages connect")

# ==============================================================================
# 4. BIOLOGICAL NETWORK EXAMPLE
# ==============================================================================
print("\n\n4. BIOLOGICAL NETWORK EXAMPLE")
print("-" * 70)

print("\n  Protein Interaction Network")
g = ec.Graph(directed=False)
g.add_node("P1", weight=25.5, family="kinase")
g.add_node("P2", weight=27.3, family="kinase")
g.add_node("P3", weight=52.1, family="receptor")
g.add_node("P4", weight=51.8, family="receptor")
g.add_edge("P1", "P2")
g.add_edge("P3", "P4")

r_weight = g.get_assortativity("weight")
r_family = g.get_assortativity_nominal("family")
print(f"     Weight assortativity: {r_weight:.3f} (similar sizes interact)")
print(f"     Family assortativity: {r_family:.3f} (same families interact)")

print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
