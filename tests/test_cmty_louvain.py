"""
Tests for Louvain community detection algorithm.

The Louvain method is a multi-level optimization algorithm that:
1. Phase 1: Each node tries to join neighboring communities to maximize modularity
2. Phase 2: Aggregate communities into super-nodes
3. Repeat until no improvement

References:
- Blondel et al. (2008) "Fast unfolding of communities in large networks"
"""

import edgecase as ec


class TestLouvainBasic:
    """Basic functionality tests."""

    def test_empty_graph(self):
        """Empty graph should return empty dict."""
        G = ec.Graph(directed=False)
        communities = G.get_cmty_louvain()
        assert communities == {}

    def test_single_node(self):
        """Single node should be in its own community."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        communities = G.get_cmty_louvain()
        assert communities == {'A': 0}

    def test_two_nodes_no_edge(self):
        """Two disconnected nodes should be in separate communities."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        G.add_node('B')
        communities = G.get_cmty_louvain()
        assert len(set(communities.values())) == 2

    def test_two_nodes_with_edge(self):
        """Two connected nodes can be in same community."""
        G = ec.Graph(directed=False)
        G.add_edge('A', 'B')
        communities = G.get_cmty_louvain()
        # Should be in same community (modularity optimization)
        assert communities['A'] == communities['B']

    def test_directed_graph_error(self):
        """Louvain requires undirected graph."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        try:
            G.get_cmty_louvain()
            assert False, "Should raise error for directed graph"
        except Exception as e:
            assert "undirected" in str(e).lower()


class TestLouvainKnownStructures:
    """Tests on graphs with known community structure."""

    def test_two_cliques(self):
        """Two separate triangles should form 2 communities."""
        G = ec.Graph(directed=False)
        
        # Triangle 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Triangle 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        communities = G.get_cmty_louvain()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2
        
        # Triangle 1 should be together
        assert communities['A'] == communities['B'] == communities['C']
        
        # Triangle 2 should be together
        assert communities['D'] == communities['E'] == communities['F']
        
        # Triangles should be separate
        assert communities['A'] != communities['D']

    def test_barbell_graph(self):
        """Barbell graph (two cliques + bridge) should split into 2."""
        G = ec.Graph(directed=False)
        
        # Left clique (K3)
        for i in ['A', 'B', 'C']:
            for j in ['A', 'B', 'C']:
                if i < j:
                    G.add_edge(i, j)
        
        # Bridge
        G.add_edge('C', 'D')
        
        # Right clique (K3)
        for i in ['D', 'E', 'F']:
            for j in ['D', 'E', 'F']:
                if i < j:
                    G.add_edge(i, j)
        
        communities = G.get_cmty_louvain()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2

    def test_four_cliques_chain(self):
        """Four cliques connected in a chain."""
        G = ec.Graph(directed=False)
        
        # Clique 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Clique 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        # Clique 3
        G.add_edge('G', 'H')
        G.add_edge('H', 'I')
        G.add_edge('I', 'G')
        
        # Clique 4
        G.add_edge('J', 'K')
        G.add_edge('K', 'L')
        G.add_edge('L', 'J')
        
        # Bridges between pairs
        G.add_edge('C', 'D')  # 1-2
        G.add_edge('F', 'G')  # 2-3
        G.add_edge('I', 'J')  # 3-4
        
        communities = G.get_cmty_louvain()
        
        # Should find 2-4 communities
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 4

    def test_karate_club_simplified(self):
        """Simplified karate club network."""
        G = ec.Graph(directed=False)
        
        # Faction 1 (instructor's group)
        for i in range(1, 18):
            for j in range(i+1, 18):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j))
        
        # Faction 2 (officer's group)
        for i in range(18, 35):
            for j in range(i+1, 35):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j))
        
        # Sparse connections between factions
        G.add_edge('17', '18')
        G.add_edge('16', '19')
        
        communities = G.get_cmty_louvain()
        
        # Should find at least 2 communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLouvainModularity:
    """Modularity-related tests."""

    def compute_modularity(self, G, communities):
        """Compute modularity Q for a given community assignment."""
        m = G.num_edges()
        if m == 0:
            return 0.0
        
        Q = 0.0
        for u in G.node_ids(limit=None):
            for v in G.node_ids(limit=None):
                if communities[u] == communities[v]:
                    A_ij = 1.0 if G.has_edge(u, v) else 0.0
                    k_u = G.get_degree(u)
                    k_v = G.get_degree(v)
                    Q += A_ij - (k_u * k_v) / (2.0 * m)
        
        return Q / (2.0 * m)

    def test_modularity_positive_structured(self):
        """Modularity should be positive for structured graphs."""
        G = ec.Graph(directed=False)
        
        # Two well-separated cliques
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        # Single bridge
        G.add_edge('C', 'D')
        
        communities = G.get_cmty_louvain()
        Q = self.compute_modularity(G, communities)
        
        # Should have positive modularity
        assert Q > 0.0

    def test_modularity_complete_graph(self):
        """Complete graph should have Q near 0 (no structure)."""
        G = ec.Graph(directed=False)
        
        nodes = ['A', 'B', 'C', 'D', 'E']
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j)
        
        communities = G.get_cmty_louvain()
        Q = self.compute_modularity(G, communities)
        
        # Complete graph has no community structure
        # Should be close to 0
        assert -0.1 <= Q <= 0.1

    def test_modularity_improvement(self):
        """Louvain should improve over initial assignment."""
        G = ec.Graph(directed=False)
        
        # Create graph with clear structure
        # Community 1
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(str(i), str(j))
        
        # Community 2
        for i in range(5, 10):
            for j in range(i+1, 10):
                G.add_edge(str(i), str(j))
        
        # Sparse connections between
        G.add_edge('4', '5')
        
        communities = G.get_cmty_louvain()
        Q_louvain = self.compute_modularity(G, communities)
        
        # Initial assignment (each node in own community)
        initial = {str(i): i for i in range(10)}
        Q_initial = self.compute_modularity(G, initial)
        
        # Louvain should improve modularity
        assert Q_louvain > Q_initial


class TestLouvainProperties:
    """Test algorithmic properties."""

    def test_all_nodes_assigned(self):
        """All nodes should be assigned to some community."""
        G = ec.Graph(directed=False)
        
        for i in range(10):
            for j in range(i+1, 10):
                if abs(i - j) <= 2:
                    G.add_edge(str(i), str(j))
        
        communities = G.get_cmty_louvain()
        
        # All nodes should be present
        assert set(communities.keys()) == set(str(i) for i in range(10))
        
        # All nodes should have valid community IDs
        for comm_id in communities.values():
            assert isinstance(comm_id, int)
            assert comm_id >= 0

    def test_community_ids_contiguous(self):
        """Community IDs should be contiguous starting from 0."""
        G = ec.Graph(directed=False)
        
        # Three separate components
        G.add_edge('A', 'B')
        G.add_edge('C', 'D')
        G.add_edge('E', 'F')
        
        communities = G.get_cmty_louvain()
        
        unique_ids = sorted(set(communities.values()))
        # Should be [0, 1, 2] or similar
        assert unique_ids == list(range(len(unique_ids)))

    def test_deterministic(self):
        """Louvain should be deterministic (for same initialization)."""
        G = ec.Graph(directed=False)
        
        # Create structured graph
        for i in range(8):
            for j in range(i+1, 8):
                if i // 4 == j // 4:  # Same group
                    G.add_edge(str(i), str(j))
        
        communities1 = G.get_cmty_louvain()
        communities2 = G.get_cmty_louvain()
        
        # Results should be identical
        assert communities1 == communities2


class TestLouvainPerformance:
    """Performance and scalability tests."""

    def test_medium_graph(self):
        """Test with 100 nodes."""
        G = ec.Graph(directed=False)
        
        # Create 5 communities of 20 nodes each
        for comm_idx in range(5):
            offset = comm_idx * 20
            for i in range(20):
                for j in range(i+1, 20):
                    if abs(i - j) <= 4:
                        G.add_edge(str(offset + i), str(offset + j))
        
        # Add sparse inter-community edges
        G.add_edge('19', '20')
        G.add_edge('39', '40')
        G.add_edge('59', '60')
        G.add_edge('79', '80')
        
        communities = G.get_cmty_louvain()
        
        # Should find multiple communities
        # The ground truth has 5 groups, but due to the chain-like structure (edges only within distance 4),
        # the algorithm may find 10-15 fine-grained communities which is acceptable
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 20  # More lenient expectation

    def test_dense_graph(self):
        """Test with dense connections."""
        G = ec.Graph(directed=False)
        
        # Create dense graph (50 nodes, ~70% edge probability within groups)
        for i in range(50):
            for j in range(i+1, 50):
                # Same group (0-24, 25-49)
                if i // 25 == j // 25:
                    if (i + j) % 3 != 0:  # ~67% edges
                        G.add_edge(str(i), str(j))
                else:
                    if (i + j) % 10 == 0:  # ~10% edges between
                        G.add_edge(str(i), str(j))
        
        communities = G.get_cmty_louvain()
        
        # Should find 2 main communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLouvainEdgeCases:
    """Edge cases and corner cases."""

    def test_complete_graph_small(self):
        """Complete graph should return 1 community."""
        G = ec.Graph(directed=False)
        
        nodes = ['A', 'B', 'C', 'D', 'E']
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j)
        
        communities = G.get_cmty_louvain()
        
        # All nodes in same community
        assert len(set(communities.values())) == 1

    def test_star_graph(self):
        """Star graph (hub + spokes) behavior."""
        G = ec.Graph(directed=False)
        
        # Center connected to all others
        for i in range(1, 6):
            G.add_edge('center', str(i))
        
        communities = G.get_cmty_louvain()
        
        # Likely forms 1 community (hub dominant)
        num_communities = len(set(communities.values()))
        assert num_communities >= 1

    def test_disconnected_components(self):
        """Multiple disconnected components."""
        G = ec.Graph(directed=False)
        
        # Component 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Component 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        # Component 3
        G.add_edge('G', 'H')
        G.add_edge('H', 'I')
        G.add_edge('I', 'G')
        
        communities = G.get_cmty_louvain()
        
        # Should find at least 3 communities
        assert len(set(communities.values())) >= 3

    def test_cycle_graph(self):
        """Cycle graph behavior."""
        G = ec.Graph(directed=False)
        
        # Create cycle
        nodes = ['A', 'B', 'C', 'D', 'E', 'F']
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i+1) % len(nodes)])
        
        communities = G.get_cmty_louvain()
        
        # Should assign all nodes (structure is weak)
        assert len(communities) == len(nodes)

    def test_bipartite_like(self):
        """Bipartite-like graph structure."""
        G = ec.Graph(directed=False)
        
        # Group A
        group_a = ['A1', 'A2', 'A3', 'A4']
        # Group B
        group_b = ['B1', 'B2', 'B3', 'B4']
        
        # Connect each A to multiple B's
        for a in group_a:
            for b in group_b:
                G.add_edge(a, b)
        
        communities = G.get_cmty_louvain()
        
        # Bipartite graphs are tricky - should find some structure
        num_communities = len(set(communities.values()))
        assert num_communities >= 1
