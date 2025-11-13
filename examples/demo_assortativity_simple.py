import edgecase as ec

print("=" * 70)
print("ASSORTATIVITY DEMONSTRATION")
print("=" * 70)

# ==============================================================================
# 1. NOMINAL ASSORTATIVITY - Segregated Network
# ==============================================================================
print("\n1. NOMINAL ASSORTATIVITY (Categorical Attributes)")
print("-" * 70)

g = ec.Graph(directed=False)
g.add_node("A", type="X")
g.add_node("B", type="X")
g.add_node("C", type="Y")
g.add_node("D", type="Y")

# Only same-type connections
g.add_edge("A", "B")  # X-X
g.add_edge("C", "D")  # Y-Y

r = g.assortativity_nominal("type")
print(f"   Perfect homophily: r = {r:.3f}")
print("   Interpretation: Only same-type nodes connect (perfect segregation)")

# ==============================================================================
# 2. NOMINAL ASSORTATIVITY - Bipartite Network
# ==============================================================================
print("\n2. NOMINAL ASSORTATIVITY (Bipartite Structure)")
print("-" * 70)

g = ec.Graph(directed=False)
g.add_node("A", type="X")
g.add_node("B", type="X")
g.add_node("C", type="Y")
g.add_node("D", type="Y")

# Only cross-type connections (bipartite)
g.add_edge("A", "C")  # X-Y
g.add_edge("A", "D")  # X-Y
g.add_edge("B", "C")  # X-Y
g.add_edge("B", "D")  # X-Y

r = g.assortativity_nominal("type")
print(f"   Perfect heterophily: r = {r:.3f}")
print("   Interpretation: Only different-type nodes connect (bipartite)")

# ==============================================================================
# 3. NUMERIC ASSORTATIVITY - Age-Based Friendships
# ==============================================================================
print("\n3. NUMERIC ASSORTATIVITY (Continuous Attributes)")
print("-" * 70)

g = ec.Graph(directed=False)
g.add_node("Alice", age=25)
g.add_node("Bob", age=27)
g.add_node("Carol", age=26)
g.add_node("Dave", age=55)
g.add_node("Eve", age=53)
g.add_node("Frank", age=54)

# Similar ages connect
g.add_edge("Alice", "Bob")  # 25-27
g.add_edge("Bob", "Carol")  # 27-26
g.add_edge("Alice", "Carol")  # 25-26
g.add_edge("Dave", "Eve")  # 55-53
g.add_edge("Eve", "Frank")  # 53-54

r = g.assortativity("age")
print(f"   Age assortativity: r = {r:.3f}")
print("   Interpretation: Similar ages connect (positive correlation)")

# ==============================================================================
# 4. DEGREE ASSORTATIVITY - Hub-and-Spoke
# ==============================================================================
print("\n4. DEGREE ASSORTATIVITY")
print("-" * 70)

g = ec.Graph(directed=False)
# Create a star/hub-and-spoke network
g.add_node("Hub")
g.add_node("Spoke1")
g.add_node("Spoke2")
g.add_node("Spoke3")
g.add_node("Spoke4")

g.add_edge("Hub", "Spoke1")
g.add_edge("Hub", "Spoke2")
g.add_edge("Hub", "Spoke3")
g.add_edge("Hub", "Spoke4")

r = g.get_graph_assortativity_degree()
print(f"   Degree assortativity: r = {r:.3f}")
print("   Interpretation: High-degree hub connects to low-degree spokes (disassortative)")

# ==============================================================================
# 5. BIOLOGICAL NETWORK EXAMPLE
# ==============================================================================
print("\n5. PROTEIN INTERACTION NETWORK")
print("-" * 70)

g = ec.Graph(directed=False)
g.add_node("P1", weight=25.5, family="kinase")
g.add_node("P2", weight=27.3, family="kinase")
g.add_node("P3", weight=52.1, family="receptor")
g.add_node("P4", weight=51.8, family="receptor")

# Similar proteins interact
g.add_edge("P1", "P2")  # Similar weight & family
g.add_edge("P3", "P4")  # Similar weight & family

r_weight = g.assortativity("weight")
r_family = g.assortativity_nominal("family")
print(f"   Weight assortativity: r = {r_weight:.3f}")
print(f"   Family assortativity: r = {r_family:.3f}")
print("   Interpretation: Similar proteins interact")

print("\n" + "=" * 70)
print("DEMO COMPLETE - Assortativity measures mixing patterns in networks")
print("=" * 70)
