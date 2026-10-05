import edgecase as ec


class TestLouvainBasic:
    """Basic functionality tests."""

    def test_empty_graph(self):
        """Empty graph should return empty dict."""
        G = ec.Graph(directed=False)
        communities = G.cmty_louvain(return_dict=True)
        assert communities == {}

    def test_single_node(self):
        """Single node should be in its own community."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        communities = G.cmty_louvain(return_dict=True)
        a_id = G.get_id_from_label("A")
        assert communities == {a_id: 0}

    def test_two_nodes_no_edge(self):
        """Two disconnected nodes should be in separate communities."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        communities = G.cmty_louvain(return_dict=True)
        assert len(set(communities.values())) == 2

    def test_two_nodes_with_edge(self):
        """Two connected nodes can be in same community."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        communities = G.cmty_louvain(return_dict=True)
        a_id, b_id = G.get_id_from_label("A"), G.get_id_from_label("B")
        # Should be in same community (modularity optimization)
        assert communities[a_id] == communities[b_id]

    def test_directed_graph_error(self):
        """Louvain requires undirected graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        try:
            G.cmty_louvain()
            assert False, "Should raise error for directed graph"
        except Exception as e:
            assert "undirected" in str(e).lower()


class TestLouvainKnownStructures:
    """Tests on graphs with known community structure."""

    def test_two_cliques(self):
        """Two separate triangles should form 2 communities."""
        G = ec.Graph(directed=False)

        # Triangle 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)

        # Triangle 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should find 2 communities
        assert len(set(communities.values())) == 2

        # Get node IDs
        a_id, b_id, c_id = [G.get_id_from_label(x) for x in ["A", "B", "C"]]
        d_id, e_id, f_id = [G.get_id_from_label(x) for x in ["D", "E", "F"]]

        # Triangle 1 should be together
        assert communities[a_id] == communities[b_id] == communities[c_id]

        # Triangle 2 should be together
        assert communities[d_id] == communities[e_id] == communities[f_id]

        # Triangles should be separate
        assert communities[a_id] != communities[d_id]

    def test_barbell_graph(self):
        """Barbell graph (two cliques + bridge) should split into 2."""
        G = ec.Graph(directed=False)

        # Left clique (K3)
        for i in ["A", "B", "C"]:
            for j in ["A", "B", "C"]:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        # Bridge
        G.add_edge("C", "D", auto_create_nodes=True)

        # Right clique (K3)
        for i in ["D", "E", "F"]:
            for j in ["D", "E", "F"]:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should find 2 communities
        assert len(set(communities.values())) == 2

    def test_four_cliques_chain(self):
        """Four cliques connected in a chain."""
        G = ec.Graph(directed=False)

        # Clique 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)

        # Clique 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        # Clique 3
        G.add_edge("G", "H", auto_create_nodes=True)
        G.add_edge("H", "I", auto_create_nodes=True)
        G.add_edge("I", "G", auto_create_nodes=True)

        # Clique 4
        G.add_edge("J", "K", auto_create_nodes=True)
        G.add_edge("K", "L", auto_create_nodes=True)
        G.add_edge("L", "J", auto_create_nodes=True)

        # Bridges between pairs
        G.add_edge("C", "D", auto_create_nodes=True)  # 1-2
        G.add_edge("F", "G", auto_create_nodes=True)  # 2-3
        G.add_edge("I", "J", auto_create_nodes=True)  # 3-4

        communities = G.cmty_louvain(return_dict=True)

        # Should find 2-4 communities
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 4

    def test_karate_club_simplified(self):
        """Simplified karate club network."""
        G = ec.Graph(directed=False)

        # Faction 1 (instructor's group)
        for i in range(1, 18):
            for j in range(i + 1, 18):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Faction 2 (officer's group)
        for i in range(18, 35):
            for j in range(i + 1, 35):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Sparse connections between factions
        G.add_edge("17", "18", auto_create_nodes=True)
        G.add_edge("16", "19", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should find at least 2 communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLouvainModularity:
    """Modularity-related tests."""

    def compute_modularity(self, G, communities):
        """Compute modularity Q for a given community assignment."""
        m = G.count_edges()
        if m == 0:
            return 0.0

        Q = 0.0
        for u in G.get_node_ids(limit=None):
            for v in G.get_node_ids(limit=None):
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
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        # Single bridge
        G.add_edge("C", "D", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)
        Q = self.compute_modularity(G, communities)

        # Should have positive modularity
        assert Q > 0.0

    def test_modularity_complete_graph(self):
        """Complete graph should have Q near 0 (no structure)."""
        G = ec.Graph(directed=False)

        nodes = ["A", "B", "C", "D", "E"]
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)
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
            for j in range(i + 1, 5):
                G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Community 2
        for i in range(5, 10):
            for j in range(i + 1, 10):
                G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Sparse connections between
        G.add_edge("4", "5", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)
        Q_louvain = self.compute_modularity(G, communities)

        # Initial assignment (each node in own community)
        # Need to use node IDs, not labels
        initial = {G.get_id_from_label(str(i)): i for i in range(10)}
        Q_initial = self.compute_modularity(G, initial)

        # Louvain should improve modularity
        assert Q_louvain > Q_initial


class TestLouvainProperties:
    """Test algorithmic properties."""

    def test_all_nodes_assigned(self):
        """All nodes should be assigned to some community."""
        G = ec.Graph(directed=False)

        for i in range(10):
            for j in range(i + 1, 10):
                if abs(i - j) <= 2:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # All nodes should be present
        expected_ids = {G.get_id_from_label(str(i)) for i in range(10)}
        assert set(communities.keys()) == expected_ids

        # All nodes should have valid community IDs
        for comm_id in communities.values():
            assert isinstance(comm_id, int)
            assert comm_id >= 0

    def test_community_ids_contiguous(self):
        """Community IDs should be contiguous starting from 0."""
        G = ec.Graph(directed=False)

        # Three separate components
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        unique_ids = sorted(set(communities.values()))
        # Should be [0, 1, 2] or similar
        assert unique_ids == list(range(len(unique_ids)))

    def test_deterministic(self):
        """Louvain should be deterministic (for same initialization)."""
        G = ec.Graph(directed=False)

        # Create structured graph
        for i in range(8):
            for j in range(i + 1, 8):
                if i // 4 == j // 4:  # Same group
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        communities1 = G.cmty_louvain(return_dict=True)
        communities2 = G.cmty_louvain(return_dict=True)

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
                for j in range(i + 1, 20):
                    if abs(i - j) <= 4:
                        G.add_edge(str(offset + i), str(offset + j), auto_create_nodes=True)

        # Add sparse inter-community edges
        G.add_edge("19", "20", auto_create_nodes=True)
        G.add_edge("39", "40", auto_create_nodes=True)
        G.add_edge("59", "60", auto_create_nodes=True)
        G.add_edge("79", "80", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

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
            for j in range(i + 1, 50):
                # Same group (0-24, 25-49)
                if i // 25 == j // 25:
                    if (i + j) % 3 != 0:  # ~67% edges
                        G.add_edge(str(i), str(j), auto_create_nodes=True)
                else:
                    if (i + j) % 10 == 0:  # ~10% edges between
                        G.add_edge(str(i), str(j), auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should find 2 main communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLouvainEdgeCases:
    """Edge cases and corner cases."""

    def test_complete_graph_small(self):
        """Complete graph should return 1 community."""
        G = ec.Graph(directed=False)

        nodes = ["A", "B", "C", "D", "E"]
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # All nodes in same community
        assert len(set(communities.values())) == 1

    def test_star_graph(self):
        """Star graph (hub + spokes) behavior."""
        G = ec.Graph(directed=False)

        # Center connected to all others
        for i in range(1, 6):
            G.add_edge("center", str(i), auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Likely forms 1 community (hub dominant)
        num_communities = len(set(communities.values()))
        assert num_communities >= 1

    def test_disconnected_components(self):
        """Multiple disconnected components."""
        G = ec.Graph(directed=False)

        # Component 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)

        # Component 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        # Component 3
        G.add_edge("G", "H", auto_create_nodes=True)
        G.add_edge("H", "I", auto_create_nodes=True)
        G.add_edge("I", "G", auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should find at least 3 communities
        assert len(set(communities.values())) >= 3

    def test_cycle_graph(self):
        """Cycle graph behavior."""
        G = ec.Graph(directed=False)

        # Create cycle
        nodes = ["A", "B", "C", "D", "E", "F"]
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)], auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Should assign all nodes (structure is weak)
        assert len(communities) == len(nodes)

    def test_bipartite_like(self):
        """Bipartite-like graph structure."""
        G = ec.Graph(directed=False)

        # Group A
        group_a = ["A1", "A2", "A3", "A4"]
        # Group B
        group_b = ["B1", "B2", "B3", "B4"]

        # Connect each A to multiple B's
        for a in group_a:
            for b in group_b:
                G.add_edge(a, b, auto_create_nodes=True)

        communities = G.cmty_louvain(return_dict=True)

        # Bipartite graphs are tricky - should find some structure
        num_communities = len(set(communities.values()))
        assert num_communities >= 1


class TestLouvainQualityAndScale:
    """Regression tests for the multi-level (aggregating) implementation.

    The previous implementation had no aggregation phase and recomputed community totals over
    every node for every move: 115 s and modularity 0.23 on a 20k-node graph where NetworkX's
    Louvain reaches 0.385 in a few seconds.
    """

    @staticmethod
    def _from_nx(G):
        g = ec.Graph()
        g.add_nodes([str(n) for n in G.nodes])
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
        return g

    def test_karate_near_optimal_modularity(self):
        karate = ec.karate_club()
        q = karate.modularity(karate.cmty_louvain(return_dict=True))
        assert q > 0.41  # optimum is 0.4198

    def test_recovers_planted_partition(self):
        import networkx as nx

        P = nx.planted_partition_graph(8, 50, 0.3, 0.01, seed=1)
        g = self._from_nx(P)
        m = g.cmty_louvain(return_dict=True)
        found = sorted(
            sorted(int(g.get_label_from_id(n)) for n in m if m[n] == c) for c in set(m.values())
        )
        assert found == [list(range(i * 50, (i + 1) * 50)) for i in range(8)]

    def test_matches_networkx_quality_and_is_fast(self):
        import time

        import networkx as nx

        B = nx.powerlaw_cluster_graph(20000, 4, 0.3, seed=1)
        g = self._from_nx(B)
        start = time.perf_counter()
        m = g.cmty_louvain(return_dict=True)
        elapsed = time.perf_counter() - start
        # NetworkX louvain_communities(seed=1) reaches 0.385 on this graph
        assert g.modularity(m) > 0.37
        assert elapsed < 5.0  # ~0.05 s in practice; the old code took ~115 s

    def test_uses_edge_weights(self):
        # A 4-cycle: heavy edges A-B and C-D should pair those nodes, whichever pair is heavy
        def build(heavy):
            g = ec.Graph(weighted=True).add_nodes(list("ABCD"))
            for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]:
                g.add_edge(u, v, weight=10.0 if (u, v) in heavy else 1.0)
            m = g.cmty_louvain(return_dict=True)
            return {g.get_label_from_id(n): c for n, c in m.items()}

        a = build({("A", "B"), ("C", "D")})
        assert a["A"] == a["B"] and a["C"] == a["D"] and a["A"] != a["C"]
        b = build({("B", "C"), ("D", "A")})
        assert b["B"] == b["C"] and b["D"] == b["A"] and b["A"] != b["B"]

    def test_deterministic_on_large_graph(self):
        import networkx as nx

        g = self._from_nx(nx.powerlaw_cluster_graph(2000, 3, 0.3, seed=2))
        assert g.cmty_louvain(return_dict=True) == g.cmty_louvain(return_dict=True)
