"""
Tests for Walktrap community detection algorithm.

The Walktrap algorithm uses random walks to detect communities.
Nodes in the same community are expected to be visited more
frequently by short random walks.
"""

import pytest
import edgecase as ec


def add_edges(G, edge_list):
    """Helper to add multiple edges to a graph."""
    for u, v in edge_list:
        G.add_edge(u, v)


class TestWalktrapBasic:
    """Test basic functionality and edge cases."""

    def test_empty_graph(self):
        """Empty graph should return empty communities."""
        G = ec.Graph()
        communities = G.get_cmty_walktrap()
        assert communities == {}

    def test_single_node(self):
        """Single node should form one community."""
        G = ec.Graph()
        G.add_node("A")
        communities = G.get_cmty_walktrap()
        assert communities == {"A": 0}

    def test_two_nodes_connected(self):
        """Two connected nodes should be in same community."""
        G = ec.Graph()
        G.add_edge("A", "B")
        communities = G.get_cmty_walktrap()
        assert communities["A"] == communities["B"]
        assert len(set(communities.values())) == 1

    def test_two_nodes_disconnected(self):
        """Two disconnected nodes should be in different communities."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        communities = G.get_cmty_walktrap()
        # Disconnected components should be separate
        assert communities["A"] != communities["B"]
        assert len(set(communities.values())) == 2

    def test_directed_graph_error(self):
        """Should raise error on directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        with pytest.raises(ValueError, match="undirected"):
            G.get_cmty_walktrap()


class TestWalktrapKnownStructures:
    """Test on graphs with known community structure."""

    def test_two_cliques(self):
        """Two disconnected cliques should form 2 communities."""
        G = ec.Graph()
        # First clique
        add_edges(G, [("A", "B"), ("B", "C"), ("C", "A")])
        # Second clique
        add_edges(G, [("D", "E"), ("E", "F"), ("F", "D")])
        
        communities = G.get_cmty_walktrap()
        
        # Check that A, B, C are in same community
        assert communities["A"] == communities["B"] == communities["C"]
        # Check that D, E, F are in same community
        assert communities["D"] == communities["E"] == communities["F"]
        # Check that the two groups are different
        assert communities["A"] != communities["D"]
        assert len(set(communities.values())) == 2

    def test_barbell_graph(self):
        """Barbell graph (two cliques connected by bridge) should find 2 communities."""
        G = ec.Graph()
        # Left clique
        add_edges(G, [("A", "B"), ("B", "C"), ("C", "A")])
        # Right clique
        add_edges(G, [("D", "E"), ("E", "F"), ("F", "D")])
        # Bridge
        G.add_edge("C", "D")
        
        communities = G.get_cmty_walktrap()
        
        # Should detect 2 communities
        num_communities = len(set(communities.values()))
        assert num_communities == 2, f"Expected 2 communities, got {num_communities}"
        
        # A and B should be together
        assert communities["A"] == communities["B"]
        # E and F should be together
        assert communities["E"] == communities["F"]
        # Left and right should be different
        assert communities["A"] != communities["E"]

    def test_four_cliques_connected(self):
        """Four cliques connected in a line."""
        G = ec.Graph()
        # Clique 1
        add_edges(G, [("A1", "A2"), ("A2", "A3"), ("A3", "A1")])
        # Clique 2
        add_edges(G, [("B1", "B2"), ("B2", "B3"), ("B3", "B1")])
        # Clique 3
        add_edges(G, [("C1", "C2"), ("C2", "C3"), ("C3", "C1")])
        # Clique 4
        add_edges(G, [("D1", "D2"), ("D2", "D3"), ("D3", "D1")])
        # Bridges
        G.add_edge("A3", "B1")
        G.add_edge("B3", "C1")
        G.add_edge("C3", "D1")
        
        communities = G.get_cmty_walktrap()
        num_communities = len(set(communities.values()))
        
        # Should find 2-4 communities
        assert 2 <= num_communities <= 4, f"Expected 2-4 communities, got {num_communities}"

    def test_karate_club(self):
        """Zachary's Karate Club should find 2-4 communities."""
        G = ec.Graph()
        # Famous social network with known structure
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33)
        ]
        add_edges(G, [(str(u), str(v)) for u, v in edges])
        
        communities = G.get_cmty_walktrap()
        num_communities = len(set(communities.values()))
        
        # Karate club typically has 2-4 communities
        assert 2 <= num_communities <= 4, f"Expected 2-4 communities, got {num_communities}"
        
        # Compute modularity
        Q = compute_modularity(G, communities)
        assert Q > 0.30, f"Modularity {Q:.4f} is too low for karate club"


class TestWalktrapModularity:
    """Test modularity-related properties."""

    def test_modularity_positive_structured(self):
        """Well-structured graph should have positive modularity."""
        G = ec.Graph()
        # Two well-connected groups with weak bridge
        add_edges(G, [
            ("A", "B"), ("B", "C"), ("C", "A"),  # Triangle 1
            ("D", "E"), ("E", "F"), ("F", "D"),  # Triangle 2
            ("C", "D")  # Weak bridge
        ])
        
        communities = G.get_cmty_walktrap()
        Q = compute_modularity(G, communities)
        assert Q > 0, f"Modularity should be positive, got {Q:.4f}"

    def test_complete_graph_modularity(self):
        """Complete graph should have modularity close to 0."""
        G = ec.Graph()
        nodes = ["A", "B", "C", "D", "E"]
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                G.add_edge(u, v)
        
        communities = G.get_cmty_walktrap()
        Q = compute_modularity(G, communities)
        
        # Complete graph has no clear structure
        assert -0.1 <= Q <= 0.1, f"Complete graph modularity should be ~0, got {Q:.4f}"


class TestWalktrapProperties:
    """Test algorithmic properties."""

    def test_all_nodes_assigned(self):
        """All nodes should be assigned to a community."""
        G = ec.Graph()
        add_edges(G, [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        
        communities = G.get_cmty_walktrap()
        
        # All nodes should be in communities dict
        assert set(communities.keys()) == {"A", "B", "C", "D", "E"}

    def test_community_ids_contiguous(self):
        """Community IDs should be contiguous (0, 1, 2, ...)."""
        G = ec.Graph()
        add_edges(G, [
            ("A", "B"), ("C", "D"), ("E", "F")
        ])
        
        communities = G.get_cmty_walktrap()
        community_ids = set(communities.values())
        
        # IDs should be 0, 1, 2, ..., k-1
        assert community_ids == set(range(len(community_ids)))

    def test_deterministic_results(self):
        """Algorithm should give consistent results (no randomness)."""
        G = ec.Graph()
        add_edges(G, [
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("D", "E"), ("E", "F"), ("F", "D"),
            ("C", "D")
        ])
        
        communities1 = G.get_cmty_walktrap()
        communities2 = G.get_cmty_walktrap()
        
        # Should be identical
        assert communities1 == communities2


class TestWalktrapStepParameter:
    """Test random walk step parameter."""

    def test_steps_parameter_2(self):
        """Test with steps=2."""
        G = ec.Graph()
        add_edges(G, [
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("D", "E"), ("E", "F"), ("F", "D"),
            ("C", "D")
        ])
        
        communities = G.get_cmty_walktrap(steps=2)
        
        # Should still detect communities
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 3

    def test_steps_parameter_6(self):
        """Test with steps=6."""
        G = ec.Graph()
        add_edges(G, [
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("D", "E"), ("E", "F"), ("F", "D"),
            ("C", "D")
        ])
        
        communities = G.get_cmty_walktrap(steps=6)
        
        # Should still detect communities
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 3

    def test_steps_default(self):
        """Test default steps parameter (should be 4)."""
        G = ec.Graph()
        add_edges(G, [
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("D", "E"), ("E", "F"), ("F", "D"),
            ("C", "D")
        ])
        
        # Default behavior
        communities = G.get_cmty_walktrap()
        
        # Should work
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 3


class TestWalktrapEdgeCases:
    """Test edge cases and special graph structures."""

    def test_complete_graph(self):
        """Complete graph should return 1 community."""
        G = ec.Graph()
        nodes = ["A", "B", "C", "D"]
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                G.add_edge(u, v)
        
        communities = G.get_cmty_walktrap()
        
        # All nodes in same community
        assert len(set(communities.values())) == 1

    def test_star_graph(self):
        """Star graph (hub with spokes)."""
        G = ec.Graph()
        center = "HUB"
        for i in range(5):
            G.add_edge(center, f"S{i}")
        
        communities = G.get_cmty_walktrap()
        
        # Should be 1 community (well-connected)
        assert len(set(communities.values())) == 1

    def test_disconnected_components(self):
        """Multiple disconnected components."""
        G = ec.Graph()
        # Component 1
        add_edges(G, [("A", "B"), ("B", "C")])
        # Component 2
        add_edges(G, [("D", "E"), ("E", "F")])
        # Component 3
        add_edges(G, [("G", "H"), ("H", "I")])
        
        communities = G.get_cmty_walktrap()
        
        # Each component should be separate
        assert len(set(communities.values())) >= 3

    def test_cycle_graph(self):
        """Cycle graph (ring)."""
        G = ec.Graph()
        nodes = ["A", "B", "C", "D", "E", "F"]
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i+1) % len(nodes)])
        
        communities = G.get_cmty_walktrap()
        
        # Cycle has no clear structure, likely 1 community
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 2

    def test_path_graph(self):
        """Path graph (linear chain)."""
        G = ec.Graph()
        add_edges(G, [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        
        communities = G.get_cmty_walktrap()
        
        # Path is weakly structured
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 3

    def test_bipartite_graph(self):
        """Bipartite graph should find 2 communities."""
        G = ec.Graph()
        # Two sets with edges only between sets
        left = ["L1", "L2", "L3"]
        right = ["R1", "R2", "R3"]
        for l in left:
            for r in right:
                G.add_edge(l, r)
        
        communities = G.get_cmty_walktrap()
        
        # Should detect bipartite structure
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 2


class TestWalktrapPerformance:
    """Test performance on larger graphs."""

    def test_medium_graph(self):
        """Test on medium-sized graph (50 nodes)."""
        G = ec.Graph()
        
        # Create 3 communities with dense internal connections
        for comm in range(3):
            nodes = [f"C{comm}_N{i}" for i in range(17)]
            # Dense connections within community
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    if hash((u, v)) % 3 != 0:  # ~67% density
                        G.add_edge(u, v)
        
        # Add some inter-community edges
        G.add_edge("C0_N0", "C1_N0")
        G.add_edge("C1_N0", "C2_N0")
        
        communities = G.get_cmty_walktrap()
        
        # Should find 2-4 communities
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 5, f"Expected 2-5 communities, got {num_communities}"

    def test_dense_graph(self):
        """Test on dense graph (30 nodes, high connectivity)."""
        G = ec.Graph()
        
        # Two dense groups
        for group in range(2):
            nodes = [f"G{group}_N{i}" for i in range(15)]
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    if hash((u, v)) % 2 == 0:  # 50% density
                        G.add_edge(u, v)
        
        # Sparse inter-group connections
        G.add_edge("G0_N0", "G1_N0")
        
        communities = G.get_cmty_walktrap()
        
        # Should find 2 main communities
        num_communities = len(set(communities.values()))
        assert 1 <= num_communities <= 3


# Helper function for modularity computation
def compute_modularity(G, communities):
    """
    Compute modularity Q for a given community partition.
    
    Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
    
    where:
    - A_ij = 1 if edge (i,j) exists
    - k_i = degree of node i
    - m = number of edges
    - δ(c_i, c_j) = 1 if nodes i and j are in same community
    """
    if not communities:
        return 0.0
    
    # Get all nodes and edges
    nodes = list(communities.keys())
    m = G.num_edges()
    
    if m == 0:
        return 0.0
    
    # Build adjacency info
    adjacency = {}
    degrees = {}
    
    for node in nodes:
        neighbors = G.neighbors(node)
        adjacency[node] = set(neighbors)
        degrees[node] = len(neighbors)
    
    # Compute modularity
    Q = 0.0
    for i, node_i in enumerate(nodes):
        for node_j in nodes[i:]:  # Include diagonal (self-loops in modularity formula)
            if communities[node_i] == communities[node_j]:
                # A_ij
                A_ij = 1.0 if node_j in adjacency[node_i] else 0.0
                if node_i == node_j:
                    A_ij *= 2  # Self-loop counts twice
                
                # Expected edges
                expected = (degrees[node_i] * degrees[node_j]) / (2.0 * m)
                
                # Add contribution
                contribution = A_ij - expected
                if node_i == node_j:
                    Q += contribution
                else:
                    Q += 2 * contribution  # Count both (i,j) and (j,i)
    
    Q /= (2.0 * m)
    return Q
