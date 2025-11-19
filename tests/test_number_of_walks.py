import pytest
from edgecase import Graph


class TestNumberOfWalksBasic:
    """Basic walk counting tests."""

    def test_empty_graph(self):
        """Empty graph has no walks."""
        g = Graph()
        walks = g.count_walks(0)
        assert walks == {}

    def test_single_node_length_0(self):
        """Single node has 1 walk of length 0 to itself."""
        g = Graph()
        g.add_node(label="A")

        walks = g.count_walks(0)
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1

    def test_single_node_length_1(self):
        """Single node has 0 walks of length 1 (no edges)."""
        g = Graph()
        g.add_node(label="A")

        walks = g.count_walks(1)
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("A")] == 0

    def test_single_edge_length_1(self):
        """Single edge creates walks of length 1."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        walks = g.count_walks(1)

        # In undirected graph, edge works both ways
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("B")] == 1
        assert walks[g.get_id_from_label("B")][g.get_id_from_label("A")] == 1
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("A")] == 0
        assert walks[g.get_id_from_label("B")][g.get_id_from_label("B")] == 0

    def test_path_graph_networkx_example(self):
        """Test matches NetworkX documentation example."""
        g = Graph()
        g.add_node(label="0")
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        walks = g.count_walks(2)

        # Expected from NetworkX:
        # {0: {0: 1, 1: 0, 2: 1}, 1: {0: 0, 1: 2, 2: 0}, 2: {0: 1, 1: 0, 2: 1}}
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] == 1  # 0 → 1 → 0
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("1")] == 0
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("2")] == 1  # 0 → 1 → 2

        assert walks[g.get_id_from_label("1")][g.get_id_from_label("0")] == 0
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("1")] == 2  # 1 → 0 → 1 and 1 → 2 → 1
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("2")] == 0

        assert walks[g.get_id_from_label("2")][g.get_id_from_label("0")] == 1  # 2 → 1 → 0
        assert walks[g.get_id_from_label("2")][g.get_id_from_label("1")] == 0
        assert walks[g.get_id_from_label("2")][g.get_id_from_label("2")] == 1  # 2 → 1 → 2

    def test_negative_walk_length_raises(self):
        """Negative walk_length should raise ValueError."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(ValueError, match="non-negative"):
            g.count_walks(-1)


class TestNumberOfWalksPaths:
    """Tests on path graphs (chains)."""

    def test_path_length_3(self):
        """Path of 4 nodes: 0-1-2-3."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))

        for i in range(3):
            g.add_edge(str(i), str(i + 1))

        # Walks of length 1 = adjacency matrix
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("0")][g.get_id_from_label("1")] == 1
        assert walks1[g.get_id_from_label("0")][g.get_id_from_label("2")] == 0

        # Walks of length 2
        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("0")][g.get_id_from_label("2")] == 1  # 0 → 1 → 2
        assert walks2[g.get_id_from_label("0")][g.get_id_from_label("0")] == 1  # 0 → 1 → 0

        # Walks of length 3
        walks3 = g.count_walks(3)
        assert walks3[g.get_id_from_label("0")][g.get_id_from_label("3")] == 1  # 0 → 1 → 2 → 3
        assert walks3[g.get_id_from_label("0")][g.get_id_from_label("1")] == 2  # 0 → 1 → 0 → 1 and 0 → 1 → 2 → 1

    def test_long_path_walk_count(self):
        """Verify walk counts grow correctly on longer paths."""
        g = Graph()
        for i in range(6):
            g.add_node(label=str(i))
        for i in range(5):
            g.add_edge(str(i), str(i + 1))

        # Length 5 walks from 0 to 5
        walks = g.count_walks(5)
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("5")] == 1  # Only one path of minimum length


class TestNumberOfWalksCycles:
    """Tests on cyclic graphs."""

    def test_triangle(self):
        """Triangle graph (3-cycle)."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "0")

        # Walks of length 3 (complete one lap)
        walks = g.count_walks(3)

        # Each node can return to itself in 2 ways:
        # 0 → 1 → 2 → 0 and 0 → 2 → 1 → 0
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] == 2
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("1")] == 2
        assert walks[g.get_id_from_label("2")][g.get_id_from_label("2")] == 2

    def test_square(self):
        """Square graph (4-cycle)."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "0")

        # Walks of length 4 return to start
        # More than just clockwise/counter-clockwise (can go back and forth)
        walks = g.count_walks(4)
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] == 8  # Multiple paths including back-and-forth

        # Walks of length 2 reach opposite node
        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("0")][g.get_id_from_label("2")] == 2  # 0→1→2 and 0→3→2

    def test_self_loop(self):
        """Self-loop in undirected graph."""
        g = Graph()
        g.add_node(label="A")
        g.add_edge("A", "A")

        # In undirected graph, self-loop is counted twice in adjacency matrix
        # (once for each direction)
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("A")][g.get_id_from_label("A")] == 2  # Self-loop counted twice

        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("A")][g.get_id_from_label("A")] == 4  # 2^2

        walks3 = g.count_walks(3)
        assert walks3[g.get_id_from_label("A")][g.get_id_from_label("A")] == 8  # 2^3


class TestNumberOfWalksDirected:
    """Tests on directed graphs."""

    def test_directed_path(self):
        """Directed path: 0 → 1 → 2."""
        g = Graph(directed=True)
        g.add_node(label="0")
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        walks = g.count_walks(2)

        # Can walk forward
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("2")] == 1

        # Cannot walk backward (directed)
        assert walks[g.get_id_from_label("2")][g.get_id_from_label("0")] == 0
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("0")] == 0

    def test_directed_cycle(self):
        """Directed cycle: 0 → 1 → 2 → 0."""
        g = Graph(directed=True)
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "0")

        # Complete one lap (length 3)
        walks = g.count_walks(3)
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] == 1  # Only one direction

        # Cannot go backward
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("1")][g.get_id_from_label("0")] == 0

    def test_bidirectional_edge(self):
        """Bidirectional edges act like undirected."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        walks = g.count_walks(2)

        # Can go back and forth
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1  # A → B → A
        assert walks[g.get_id_from_label("B")][g.get_id_from_label("B")] == 1  # B → A → B

    def test_directed_self_loop(self):
        """Self-loop in directed graph."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_edge("A", "A")

        # In directed graph, self-loop is counted once
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1

        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1  # 1^2

        walks3 = g.count_walks(3)
        assert walks3[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1  # 1^3

        # With another node, self-loop doesn't affect other paths
        g.add_node(label="B")
        g.add_edge("A", "B")

        walks1_mixed = g.count_walks(1)
        assert walks1_mixed[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1  # Self-loop
        assert walks1_mixed[g.get_id_from_label("A")][g.get_id_from_label("B")] == 1  # Regular edge
        assert walks1_mixed[g.get_id_from_label("B")][g.get_id_from_label("A")] == 0  # No reverse edge

    def test_directed_multiple_self_loops(self):
        """Multiple self-loops in directed graph."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_edge("A", "A")
        g.add_edge("A", "A")  # Second self-loop
        g.add_edge("A", "A")  # Third self-loop

        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("A")][g.get_id_from_label("A")] == 3  # All 3 self-loops counted

        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("A")][g.get_id_from_label("A")] == 9  # 3^2

        walks3 = g.count_walks(3)
        assert walks3[g.get_id_from_label("A")][g.get_id_from_label("A")] == 27  # 3^3


class TestNumberOfWalksComplete:
    """Tests on complete graphs."""

    def test_k3_complete(self):
        """Complete graph on 3 nodes."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))

        # Add all edges
        for i in range(3):
            for j in range(i + 1, 3):
                g.add_edge(str(i), str(j))

        # Walks of length 1 = number of edges
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("0")][g.get_id_from_label("1")] == 1
        assert walks1[g.get_id_from_label("0")][g.get_id_from_label("2")] == 1

        # Walks of length 2: many options
        walks2 = g.count_walks(2)
        # From 0 to 0: 0→1→0, 0→2→0
        assert walks2[g.get_id_from_label("0")][g.get_id_from_label("0")] == 2
        # From 0 to 1: 0→2→1
        assert walks2[g.get_id_from_label("0")][g.get_id_from_label("1")] == 1

    def test_k4_complete_growth(self):
        """Walk counts grow quickly in complete graphs."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))

        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(str(i), str(j))

        # Total walks should grow exponentially
        walks2 = g.count_walks(2)
        total2 = sum(sum(targets.values()) for targets in walks2.values())

        walks3 = g.count_walks(3)
        total3 = sum(sum(targets.values()) for targets in walks3.values())

        assert total3 > total2


class TestNumberOfWalksStar:
    """Tests on star graphs."""

    def test_star_k5(self):
        """Star with center and 5 leaves."""
        g = Graph()
        g.add_node(label="center")
        for i in range(5):
            leaf = f"leaf{i}"
            g.add_node(label=leaf)
            g.add_edge("center", leaf)

        # Length 1: direct connections
        walks1 = g.count_walks(1)
        assert walks1[g.get_id_from_label("center")][g.get_id_from_label("leaf0")] == 1
        assert walks1[g.get_id_from_label("leaf0")][g.get_id_from_label("leaf1")] == 0  # Leaves not connected

        # Length 2: leaves can reach each other via center
        walks2 = g.count_walks(2)
        assert walks2[g.get_id_from_label("leaf0")][g.get_id_from_label("leaf1")] == 1  # leaf0 → center → leaf1
        assert walks2[g.get_id_from_label("center")][g.get_id_from_label("center")] == 5  # center → any leaf → center


class TestNumberOfWalksSpecial:
    """Tests on special graph structures."""

    def test_disconnected_components(self):
        """Disconnected graph has 0 walks between components."""
        g = Graph()

        # Component 1: A-B
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Component 2: C-D
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("C", "D")

        walks = g.count_walks(2)

        # Within components: walks exist
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("A")] == 1  # A → B → A
        assert walks[g.get_id_from_label("C")][g.get_id_from_label("C")] == 1  # C → D → C

        # Between components: no walks
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("C")] == 0
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("D")] == 0
        assert walks[g.get_id_from_label("C")][g.get_id_from_label("A")] == 0

    def test_tree(self):
        """Binary tree structure."""
        g = Graph()
        # Root with two children, each with two children
        g.add_node(label="root")
        g.add_node(label="L")
        g.add_node(label="R")
        g.add_edge("root", "L")
        g.add_edge("root", "R")

        g.add_node(label="LL")
        g.add_node(label="LR")
        g.add_edge("L", "LL")
        g.add_edge("L", "LR")

        # Walks of length 2 from root
        walks = g.count_walks(2)
        assert walks[g.get_id_from_label("root")][g.get_id_from_label("LL")] == 1  # root → L → LL
        assert walks[g.get_id_from_label("root")][g.get_id_from_label("LR")] == 1  # root → L → LR
        assert walks[g.get_id_from_label("root")][g.get_id_from_label("root")] == 2  # root → L → root, root → R → root

    def test_multigraph(self):
        """Multiple edges between same nodes."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")

        # Add 3 edges between A and B
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        g.add_edge("A", "B")

        walks1 = g.count_walks(1)
        # Should count all 3 edges
        assert walks1[g.get_id_from_label("A")][g.get_id_from_label("B")] == 3
        assert walks1[g.get_id_from_label("B")][g.get_id_from_label("A")] == 3  # Undirected

        walks2 = g.count_walks(2)
        # A → B → A: 3 choices for first edge × 3 for second = 9
        assert walks2[g.get_id_from_label("A")][g.get_id_from_label("A")] == 9


class TestNumberOfWalksLength0:
    """Tests specifically for walk_length = 0."""

    def test_length_0_is_identity(self):
        """Length 0 returns identity matrix."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))

        # Add some edges (shouldn't matter for length 0)
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        walks = g.count_walks(0)

        # Diagonal: 1 (stay at same node)
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] == 1
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("1")] == 1
        assert walks[g.get_id_from_label("2")][g.get_id_from_label("2")] == 1
        assert walks[g.get_id_from_label("3")][g.get_id_from_label("3")] == 1

        # Off-diagonal: 0 (can't reach other nodes in 0 steps)
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("1")] == 0
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("2")] == 0
        assert walks[g.get_id_from_label("1")][g.get_id_from_label("0")] == 0


class TestNumberOfWalksLarge:
    """Tests with larger walk lengths."""

    def test_large_walk_length(self):
        """Test with large walk_length using fast exponentiation."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))

        # Create a cycle
        for i in range(4):
            g.add_edge(str(i), str((i + 1) % 4))

        # Large walk length (tests fast exponentiation)
        walks = g.count_walks(100)

        # After 100 steps on a 4-cycle, should return to start
        assert walks[g.get_id_from_label("0")][g.get_id_from_label("0")] > 0

        # Verify it actually computed something
        total = sum(sum(targets.values()) for targets in walks.values())
        assert total > 0

    def test_power_of_two(self):
        """Test with power-of-2 walk lengths (optimizes exponentiation)."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Powers of 2: 2, 4, 8, 16
        for k in [2, 4, 8, 16]:
            walks = g.count_walks(k)
            # Should return valid results
            assert isinstance(walks, dict)
            assert g.get_id_from_label("A") in walks
            assert g.get_id_from_label("B") in walks


class TestNumberOfWalksConsistency:
    """Consistency and property tests."""

    def test_symmetry_undirected(self):
        """In undirected graphs, walk counts are symmetric."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        walks = g.count_walks(3)

        # In undirected graph, walks[i][j] should equal walks[j][i]
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("C")] == walks[g.get_id_from_label("C")][g.get_id_from_label("A")]
        assert walks[g.get_id_from_label("A")][g.get_id_from_label("B")] == walks[g.get_id_from_label("B")][g.get_id_from_label("A")]

    def test_total_walks_consistency(self):
        """Total walk count should be consistent."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        walks1 = g.count_walks(1)
        walks2 = g.count_walks(2)

        # Should have same structure
        assert set(walks1.keys()) == set(walks2.keys())

        # All values should be non-negative
        for source, targets in walks2.items():
            for target, count in targets.items():
                assert count >= 0

    def test_all_nodes_present(self):
        """All nodes should appear in result."""
        g = Graph()
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            g.add_node(label=node)

        g.add_edge("A", "B")
        g.add_edge("C", "D")

        walks = g.count_walks(1)

        # All nodes should be keys
        assert set(walks.keys()) == set(g.get_id_from_label(n) for n in nodes)

        # All nodes should be in inner dicts
        for source, targets in walks.items():
            assert set(targets.keys()) == set(g.get_id_from_label(n) for n in nodes)
