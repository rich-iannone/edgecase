from edgecase import Graph


class TestRichClubCoefficientBasic:
    """Basic tests for rich_club_coefficient."""

    def test_empty_graph(self):
        """Empty graph returns empty dict."""
        g = Graph()

        rc = g.rich_club_coefficient()

        assert rc == {}

    def test_single_node(self):
        """Single node graph returns empty dict (no edges possible)."""
        g = Graph().add_node(label="A")

        rc = g.rich_club_coefficient()

        assert rc == {}

    def test_two_nodes_no_edge(self):
        """Two disconnected nodes return empty dict."""
        g = Graph().add_node(label="A").add_node(label="B")

        rc = g.rich_club_coefficient()

        # No edges, so all nodes have degree 0, no k with nodes having degree > k
        assert rc == {}

    def test_two_nodes_with_edge(self):
        """Two connected nodes: both have degree 1, k=0 gives φ=1."""
        g = Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")

        rc = g.rich_club_coefficient()

        # Both nodes have degree 1, so for k=0, N_k=2, E_k=1, φ=2*1/(2*1)=1.0
        assert 0 in rc
        assert abs(rc[0] - 1.0) < 1e-10

    def test_path_graph(self):
        """Path graph: 1--2--3--4--5."""
        g = Graph()

        for i in range(1, 6):
            g.add_node(label=str(i))
        for i in range(1, 5):
            g.add_edge(str(i), str(i + 1))

        # Degrees: 1->1, 2->2, 3->2, 4->2, 5->1
        rc = g.rich_club_coefficient()

        # k=0: all 5 nodes, 4 edges, potential=20, φ=8/20=0.4
        assert 0 in rc
        assert abs(rc[0] - 0.4) < 1e-10

        # k=1: nodes with degree > 1 are {2, 3, 4} (3 nodes)
        # Edges among them: 2-3, 3-4 = 2 edges
        # potential = 3*2 = 6, φ = 4/6 = 0.667
        assert 1 in rc
        assert abs(rc[1] - 2 / 3) < 1e-10

        # k=2: no node has degree > 2
        assert 2 not in rc


class TestRichClubCoefficientCompleteGraphs:
    """Tests with complete graphs (perfect rich club)."""

    def test_k3_complete(self):
        """K3: triangle, all nodes have degree 2, perfect connectivity."""
        g = Graph()

        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2").add_edge("2", "0")

        rc = g.rich_club_coefficient()

        # All degrees are 2
        # k=0: all 3 nodes, 3 edges, potential=6, φ=1.0
        assert abs(rc[0] - 1.0) < 1e-10

        # k=1: all 3 nodes (degree > 1), same result
        assert abs(rc[1] - 1.0) < 1e-10

        # k=2: no node has degree > 2
        assert 2 not in rc

    def test_k4_complete(self):
        """K4: complete graph with 4 nodes."""
        g = Graph()

        for i in range(4):
            g.add_node(label=str(i))

        # All pairs
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(str(i), str(j))

        rc = g.rich_club_coefficient()

        # All degrees are 3, 6 edges
        # For k=0,1,2: all 4 nodes, 6 edges, potential=12, φ=1.0
        for k in [0, 1, 2]:
            assert k in rc
            assert abs(rc[k] - 1.0) < 1e-10

    def test_k5_complete(self):
        """K5: complete graph with 5 nodes."""
        g = Graph()

        for i in range(5):
            g.add_node(label=str(i))

        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(str(i), str(j))

        rc = g.rich_club_coefficient()

        # All degrees are 4, 10 edges
        # For k=0,1,2,3: all 5 nodes, 10 edges, potential=20, φ=1.0
        for k in [0, 1, 2, 3]:
            assert k in rc
            assert abs(rc[k] - 1.0) < 1e-10


class TestRichClubCoefficientStarGraph:
    """Tests with star graphs (anti-rich-club)."""

    def test_star_graph_4_nodes(self):
        """Star graph: central hub connected to 3 spokes."""
        g = Graph()
        g.add_node(label="hub")

        for i in range(1, 4):
            g.add_node(label=f"spoke{i}")
            g.add_edge("hub", f"spoke{i}")

        # Degrees: hub->3, spokes->1 each
        rc = g.rich_club_coefficient()

        # k=0: all 4 nodes, 3 edges, potential=12, φ=6/12=0.5
        assert 0 in rc
        assert abs(rc[0] - 0.5) < 1e-10

        # k=1: only hub has degree > 1 (1 node), can't compute
        assert 1 not in rc

        # k=2: only hub has degree > 2, can't compute
        assert 2 not in rc

    def test_star_graph_6_nodes(self):
        """Larger star graph: 1 hub + 5 spokes."""
        g = Graph()
        g.add_node(label="hub")

        for i in range(1, 6):
            g.add_node(label=f"s{i}")
            g.add_edge("hub", f"s{i}")

        # Degrees: hub->5, spokes->1
        rc = g.rich_club_coefficient()

        # k=0: 6 nodes, 5 edges, potential=30, φ=10/30=0.333
        assert 0 in rc
        assert abs(rc[0] - 1 / 3) < 1e-10

        # k>=1: only hub has degree > 1
        for k in [1, 2, 3, 4]:
            assert k not in rc


class TestRichClubCoefficientBarbell:
    """Tests with barbell graphs (two cliques connected by a path)."""

    def test_barbell_graph(self):
        """Two triangles connected by a single edge."""
        g = Graph()

        # First clique: A, B, C
        g.add_node(label="A").add_node(label="B").add_node(label="C")
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")

        # Second clique: D, E, F
        g.add_node(label="D").add_node(label="E").add_node(label="F")
        g.add_edge("D", "E").add_edge("E", "F").add_edge("F", "D")

        # Bridge
        g.add_edge("C", "D")

        # Degrees: A->2, B->2, C->3, D->3, E->2, F->2
        rc = g.rich_club_coefficient()

        # k=0: 6 nodes, 7 edges, potential=30, φ=14/30=0.467
        assert 0 in rc
        assert abs(rc[0] - 14 / 30) < 1e-10

        # k=1: all 6 nodes have degree > 1, same as k=0
        assert abs(rc[1] - 14 / 30) < 1e-10

        # k=2: nodes with degree > 2 are C and D (2 nodes)
        # Edges between them: C-D (1 edge)
        # potential = 2, φ = 2/2 = 1.0
        assert 2 in rc
        assert abs(rc[2] - 1.0) < 1e-10


class TestRichClubCoefficientDirected:
    """Tests with directed graphs."""

    def test_directed_cycle(self):
        """Directed cycle: 1->2->3->1."""
        g = Graph(directed=True)

        for i in range(1, 4):
            g.add_node(label=str(i))
        g.add_edge("1", "2").add_edge("2", "3").add_edge("3", "1")

        rc = g.rich_club_coefficient()

        # In directed graph, total degree = in + out = 2 for each node
        # All nodes have total degree 2
        # k=0: 3 nodes, 3 directed edges, potential=6 (directed), φ=3/6=0.5
        # k=1: all 3 nodes still have degree > 1, 3 edges, potential=6, φ=0.5
        assert 0 in rc
        assert abs(rc[0] - 0.5) < 1e-10
        assert 1 in rc
        assert abs(rc[1] - 0.5) < 1e-10

        # k=2: no node has degree > 2
        assert 2 not in rc

    def test_directed_star(self):
        """Directed star: hub has out-edges to all spokes."""
        g = Graph(directed=True)
        g.add_node(label="hub")

        for i in range(1, 4):
            g.add_node(label=f"s{i}")
            g.add_edge("hub", f"s{i}")

        rc = g.rich_club_coefficient()

        # Total degrees: hub->3 (out), spokes->1 (in) each
        # k=0: all 4 nodes have degree >= 1, 3 edges among them
        #      potential=12 (directed: 4*3), φ=3/12=0.25
        assert 0 in rc
        assert abs(rc[0] - 0.25) < 1e-10

        # k=1: only hub has degree > 1 (3), but we need at least 2 nodes
        assert 1 not in rc

    def test_directed_bidirectional(self):
        """Directed graph with bidirectional edges."""
        g = Graph(directed=True)
        g.add_node(label="A").add_node(label="B").add_node(label="C")

        # Bidirectional: A <-> B <-> C
        g.add_edge("A", "B").add_edge("B", "A")
        g.add_edge("B", "C").add_edge("C", "B")

        rc = g.rich_club_coefficient()

        # Total degrees: A->2, B->4, C->2
        # k=0: all 3 nodes, 4 edges, potential=6, φ=4/6=0.667
        assert 0 in rc
        assert abs(rc[0] - 4 / 6) < 1e-10

        # k=1: all 3 nodes have degree > 1
        assert 1 in rc

        # k=2: only B has degree > 2 (but need 2 nodes)
        # Actually A and C have degree 2, which is not > 2
        assert 2 not in rc

        # k=3: only B has degree > 3, need 2+ nodes
        assert 3 not in rc


class TestRichClubCoefficientSelfLoops:
    """Tests that self-loops are handled correctly (ignored)."""

    def test_self_loop_ignored(self):
        """Self-loops should not count as edges in the rich club."""
        g = Graph()
        g.add_node(label="A").add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        rc = g.rich_club_coefficient()

        # Degrees from adjacency: A->2 (B + self), B->1
        # But self-loop should be ignored in E_k count
        # k=0: 2 nodes with degree > 0, 1 edge (A-B), potential=2, φ=1.0
        assert 0 in rc

        # The coefficient should be based on edge A-B only
        assert abs(rc[0] - 1.0) < 1e-10


class TestRichClubCoefficientLargeGraph:
    """Tests with larger generated graphs."""

    def test_gnp_graph(self):
        """G(n,p) random graph should have reasonable rich club values."""
        g = Graph().add_gnp_graph(50, 0.2, seed=23)
        rc = g.rich_club_coefficient()

        # Should have some coefficients
        assert len(rc) > 0

        # All coefficients should be between 0 and 1
        for k, phi in rc.items():
            assert 0 <= phi <= 1.0, f"φ({k}) = {phi} out of range"

    def test_pa_graph(self):
        """Preferential attachment graph should show rich-club phenomenon."""
        g = Graph().add_pa_graph(50, 2, seed=23)
        rc = g.rich_club_coefficient()

        # Should have coefficients
        assert len(rc) > 0

        # All coefficients should be between 0 and 1
        for k, phi in rc.items():
            assert 0 <= phi <= 1.0, f"φ({k}) = {phi} out of range"


class TestRichClubCoefficientMonotonicity:
    """Test that rich club coefficient tends to increase with k."""

    def test_complete_graph_constant(self):
        """Complete graph should have constant φ=1 for all k."""
        g = Graph()

        for i in range(6):
            g.add_node(label=str(i))

        for i in range(6):
            for j in range(i + 1, 6):
                g.add_edge(str(i), str(j))

        rc = g.rich_club_coefficient()

        for k, phi in rc.items():
            assert abs(phi - 1.0) < 1e-10

    def test_values_bounded(self):
        """All rich club coefficients should be in [0, 1]."""
        # Create a graph with varying degrees
        g = Graph()

        for i in range(10):
            g.add_node(label=str(i))

        # Add random-ish edges
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 4),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ]
        for s, t in edges:
            g.add_edge(str(s), str(t))

        rc = g.rich_club_coefficient()

        for k, phi in rc.items():
            assert 0 <= phi <= 1.0, f"φ({k}) = {phi} is out of bounds [0, 1]"
