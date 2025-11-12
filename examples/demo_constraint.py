from edgecase import Graph


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def print_constraint_analysis(g: Graph, title: str):
    """Print constraint analysis for a graph."""
    print(f"\n{title}")
    print("-" * len(title))

    constraints = g.constraint()

    # Sort by constraint value (low to high = good to bad positions)
    sorted_nodes = sorted(
        [(node, value) for node, value in constraints.items() if value is not None],
        key=lambda x: x[1],
    )

    for node, constraint in sorted_nodes:
        interpretation = interpret_constraint(constraint)
        print(f"  {node:12} {constraint:6.3f}  {interpretation}")

    # Identify isolated nodes
    isolated = [node for node, value in constraints.items() if value is None]
    if isolated:
        print(f"\n  Isolated: {', '.join(isolated)}")


def interpret_constraint(value: float) -> str:
    """Interpret constraint value."""
    if value < 0.3:
        return "★★★ Excellent broker position, many structural holes"
    elif value < 0.5:
        return "★★  Good broker position, some structural holes"
    elif value < 0.7:
        return "★   Moderate position, limited structural holes"
    elif value < 1.0:
        return "⚠   High redundancy, few structural holes"
    else:
        return "⚠⚠  Maximum constraint, no structural holes"


# =============================================================================
# Demo 1: Simple Broker vs. Closed Triad
# =============================================================================

print_section("Demo 1: Simple Broker vs. Closed Triad")

print("\nScenario A: Broker Position (Structural Hole)")
print("  Network: A--B--C  (B connects A and C who are not connected)")

g1a = Graph()
g1a.add_node(label="A")
g1a.add_node(label="B")
g1a.add_node(label="C")
g1a.add_edge("A", "B")
g1a.add_edge("B", "C")

print_constraint_analysis(g1a, "Constraint values:")

print("\nInterpretation:")
print("  - B is in a BROKER position, spanning a structural hole")
print("  - B has access to non-redundant information from A and C")
print("  - A and C have high constraint (only one contact)")
print("  - B has low constraint (contacts are disconnected)")

print("\n" + "-" * 70)

print("\nScenario B: Closed Triad (No Structural Holes)")
print("  Network: A--B--C with A--C  (triangle)")

g1b = Graph()
g1b.add_node(label="A")
g1b.add_node(label="B")
g1b.add_node(label="C")
g1b.add_edge("A", "B")
g1b.add_edge("B", "C")
g1b.add_edge("A", "C")

print_constraint_analysis(g1b, "Constraint values:")

print("\nInterpretation:")
print("  - All nodes have HIGH constraint (contacts know each other)")
print("  - No structural holes - closed, redundant network")
print("  - Limited access to diverse information")
print("  - High social capital within group, low brokerage potential")


# =============================================================================
# Demo 2: Star Network - Hub vs. Periphery
# =============================================================================

print_section("Demo 2: Star Network - Hub vs. Periphery")

print("\nNetwork structure:")
print("  Hub connected to 5 peripheral nodes")
print("  Peripheral nodes NOT connected to each other")

g2 = Graph()
g2.add_node(label="Hub")
for i in range(5):
    leaf = f"Leaf{i + 1}"
    g2.add_node(label=leaf)
    g2.add_edge("Hub", leaf)

print_constraint_analysis(g2, "Constraint values:")

print("\nInterpretation:")
print("  - Hub: EXCELLENT position with many structural holes")
print("  - Hub spans 5 separate contacts, maximum brokerage")
print("  - Each leaf: High constraint (only connected to hub)")
print("  - Hub has informational and control advantages")


# =============================================================================
# Demo 3: Bridge Between Communities
# =============================================================================

print_section("Demo 3: Bridge Between Communities")

print("\nNetwork structure:")
print("  Two dense communities connected by a bridge node")
print("  Community 1: Alice, Bob, Carol (all connected)")
print("  Community 2: David, Eve, Frank (all connected)")
print("  Bridge: Xavier connects both communities")

g3 = Graph()

# Community 1 - dense
for name in ["Alice", "Bob", "Carol"]:
    g3.add_node(label=name)
g3.add_edge("Alice", "Bob")
g3.add_edge("Bob", "Carol")
g3.add_edge("Alice", "Carol")

# Community 2 - dense
for name in ["David", "Eve", "Frank"]:
    g3.add_node(label=name)
g3.add_edge("David", "Eve")
g3.add_edge("Eve", "Frank")
g3.add_edge("David", "Frank")

# Bridge
g3.add_node(label="Xavier")
g3.add_edge("Bob", "Xavier")
g3.add_edge("Xavier", "Eve")

print_constraint_analysis(g3, "Constraint values:")

print("\nInterpretation:")
print("  - Xavier: Bridge position connecting two communities")
print("  - Bob & Eve: Gatekeepers within their communities")
print("  - Peripheral members: High constraint, embedded in dense groups")
print("  - Xavier has access to information from both communities")
print("  - Bridge positions are valuable for innovation and coordination")


# =============================================================================
# Demo 4: Corporate Network - Information Flow
# =============================================================================

print_section("Demo 4: Corporate Network - Information Flow")

print("\nNetwork structure:")
print("  Simulating a corporate network with departments")
print("  - Engineering team (dense)")
print("  - Sales team (dense)")
print("  - Product Manager connects both teams")

g4 = Graph()

# Engineering team
for name in ["Eng1", "Eng2", "Eng3"]:
    g4.add_node(label=name)
g4.add_edge("Eng1", "Eng2")
g4.add_edge("Eng2", "Eng3")
g4.add_edge("Eng1", "Eng3")

# Sales team
for name in ["Sales1", "Sales2", "Sales3"]:
    g4.add_node(label=name)
g4.add_edge("Sales1", "Sales2")
g4.add_edge("Sales2", "Sales3")
g4.add_edge("Sales1", "Sales3")

# Product Manager bridges teams
g4.add_node(label="PM")
g4.add_edge("PM", "Eng2")
g4.add_edge("PM", "Sales2")

# Executive connects to PM
g4.add_node(label="Exec")
g4.add_edge("Exec", "PM")

print_constraint_analysis(g4, "Constraint values:")

print("\nInterpretation:")
print("  - Exec: Best broker position (connects to PM only)")
print("  - PM: Strong broker connecting two departments")
print("  - Team members: High constraint within departments")
print("  - Information flows through PM and Exec")
print("  - PM is critical for cross-functional coordination")


# =============================================================================
# Demo 5: Weighted Networks - Strong vs. Weak Ties
# =============================================================================

print_section("Demo 5: Weighted Networks - Strong vs. Weak Ties")

print("\nNetwork structure:")
print("  Person with mix of strong and weak ties")
print("  Strong ties (weight=10): Close friends who know each other")
print("  Weak ties (weight=1): Acquaintances in different circles")

g5 = Graph(weighted=True)

# Focal person
g5.add_node(label="You")

# Strong ties - close friend group (all connected)
g5.add_node(label="BestFriend1")
g5.add_node(label="BestFriend2")
g5.add_edge("You", "BestFriend1", weight=10.0)
g5.add_edge("You", "BestFriend2", weight=10.0)
g5.add_edge("BestFriend1", "BestFriend2", weight=10.0)

# Weak ties - acquaintances (not connected to each other)
for i in range(3):
    acquaintance = f"Acquaint{i + 1}"
    g5.add_node(label=acquaintance)
    g5.add_edge("You", acquaintance, weight=1.0)

print_constraint_analysis(g5, "Constraint values:")

print("\nInterpretation:")
print("  - Your position: Despite having weak ties, strong ties dominate")
print("  - Strong ties create HIGH constraint (friends know each other)")
print("  - Weak ties provide structural holes but are 'drowned out'")
print("  - Classic 'strength of weak ties' paradox (Granovetter 1973)")
print("  - Weak ties more valuable for novel information despite lower weight")


# =============================================================================
# Demo 6: Identifying Brokerage Opportunities
# =============================================================================

print_section("Demo 6: Identifying Brokerage Opportunities")

print("\nNetwork structure:")
print("  Analyzing who has the best broker positions")
print("  Various network positions compared")

g6 = Graph()

# Position 1: Isolated
g6.add_node(label="Isolated")

# Position 2: Single connection
g6.add_node(label="Peripheral")
g6.add_node(label="P_connect")
g6.add_edge("Peripheral", "P_connect")

# Position 3: Embedded in clique
g6.add_node(label="Clique1")
g6.add_node(label="Clique2")
g6.add_node(label="Clique3")
g6.add_edge("Clique1", "Clique2")
g6.add_edge("Clique2", "Clique3")
g6.add_edge("Clique1", "Clique3")

# Position 4: Broker connecting unconnected contacts
g6.add_node(label="Broker")
g6.add_node(label="Contact1")
g6.add_node(label="Contact2")
g6.add_node(label="Contact3")
g6.add_edge("Broker", "Contact1")
g6.add_edge("Broker", "Contact2")
g6.add_edge("Broker", "Contact3")

print_constraint_analysis(g6, "Constraint values:")

print("\nInterpretation:")
print("  - Broker: BEST position - multiple unconnected contacts")
print("  - Clique members: High constraint, redundant information")
print("  - Peripheral: Maximum constraint, dependent on one contact")
print("  - Isolated: Undefined constraint (no network)")
print("\n  Lesson: Brokerage requires connecting DISCONNECTED contacts")


# =============================================================================
# Summary
# =============================================================================

print_section("Summary: Using Constraint for Network Strategy")

print("""
Key insights from Burt's constraint measure:

1. **Low Constraint (< 0.5)**
   - Many structural holes
   - Diverse, non-redundant contacts
   - Access to novel information
   - Brokerage opportunities
   - Good for: Innovation, entrepreneurship, spanning boundaries

2. **High Constraint (> 0.7)**
   - Few structural holes
   - Redundant, interconnected contacts
   - Limited information diversity
   - Embedded in dense groups
   - Good for: Trust, cohesion, collective action

3. **Practical Applications**
   - Identify key brokers in organizations
   - Optimize network positions for information flow
   - Understand power dynamics in social networks
   - Design communication structures
   - Measure social capital and network advantages

4. **Strategic Considerations**
   - Brokers have information advantages but less social support
   - Dense networks have trust but redundant information
   - Optimal strategy depends on goals (innovation vs. cohesion)
   - Weak ties often provide more structural holes than strong ties

For more details, see:
    Burt, R. S. (1992). Structural Holes: The Social Structure of
    Competition. Harvard University Press.
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)
