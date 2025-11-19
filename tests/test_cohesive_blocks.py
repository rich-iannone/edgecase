import pytest
from edgecase import Graph


class TestCohesiveBlocksBasic:
    """Test cohesive blocks on basic graph structures."""

    def test_empty_graph(self):
        """Empty graph should return empty results."""
        G = Graph()
        blocks, cohesions, parents = G.find_cohesive_blocks()

        assert blocks == []
        assert cohesions == []
        assert parents == []

    def test_single_node(self):
        """Single node has cohesion 0."""
        G = Graph()
        G.add_node(label="1")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        assert len(blocks) == 1
        assert len(blocks[0]) == 1
        assert isinstance(blocks[0][0], int)  # Returns node IDs, not labels
        assert cohesions == [0]
        assert parents == [None]

    def test_two_nodes_connected(self):
        """Two connected nodes have cohesion 1."""
        G = Graph()
        G.add_node(label="1")
        G.add_node(label="2")
        G.add_edge("1", "2")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        assert len(blocks) == 1
        assert len(blocks[0]) == 2
        assert all(isinstance(nid, int) for nid in blocks[0])  # Returns node IDs
        assert cohesions == [1]
        assert parents == [None]

    def test_two_nodes_disconnected(self):
        """Two disconnected nodes form separate blocks."""
        G = Graph()
        G.add_node(label="1")
        G.add_node(label="2")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        assert len(blocks) == 2
        assert all(len(block) == 1 for block in blocks)
        assert all(isinstance(block[0], int) for block in blocks)  # Returns node IDs
        assert cohesions == [0, 0]
        assert parents == [None, None]


class TestCohesiveBlocksStandard:
    """Test cohesive blocks on standard graph types."""

    def test_path_graph(self):
        """Path graph has cohesion 1 (one articulation point disconnects it)."""
        G = Graph()
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 5):
            G.add_edge(str(i), str(i + 1))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Path graph should have cohesion 1
        assert len(blocks) >= 1
        assert 1 in cohesions

    def test_cycle_graph(self):
        """Cycle graph has cohesion 2 (need to remove 2 nodes to disconnect)."""
        G = Graph()
        n = 6
        for i in range(1, n + 1):
            G.add_node(label=str(i))
        for i in range(1, n + 1):
            G.add_edge(str(i), str((i % n) + 1))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Cycle graph should have cohesion 2
        assert len(blocks) >= 1
        assert max(cohesions) == 2

    def test_complete_graph_k5(self):
        """Complete graph K_5 has cohesion 4."""
        G = Graph()
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Complete K_5 should have cohesion 4
        assert len(blocks) == 1
        assert cohesions[0] == 4
        assert parents[0] is None

    def test_star_graph(self):
        """Star graph has cohesion 1 (center is articulation point)."""
        G = Graph()
        G.add_node(label="center")
        for i in range(1, 6):
            G.add_node(label=str(i))
            G.add_edge("center", str(i))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Star graph should have cohesion 1
        assert len(blocks) >= 1
        assert 1 in cohesions


class TestCohesiveBlocksHierarchy:
    """Test hierarchical structure of cohesive blocks."""

    def test_hierarchy_parent_relationships(self):
        """Test that child blocks have higher cohesion than parents."""
        G = Graph()
        # Create a graph with hierarchical structure:
        # A larger 2-cohesive block containing a 3-cohesive sub-block

        # Build K_4 (complete graph on 4 nodes) - cohesion 3
        for i in range(1, 5):
            G.add_node(label=str(i))
        for i in range(1, 5):
            for j in range(i + 1, 5):
                G.add_edge(str(i), str(j))

        # Add two more nodes connected to form larger structure
        G.add_node(label="5")
        G.add_node(label="6")
        G.add_edge("1", "5")
        G.add_edge("2", "5")
        G.add_edge("5", "6")
        G.add_edge("6", "1")
        G.add_edge("6", "2")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Should find hierarchy: larger block with cohesion 2, smaller K_4 with cohesion 3
        assert len(blocks) >= 2

        # Check that if there's a parent-child relationship, child has higher cohesion
        for i, parent_idx in enumerate(parents):
            if parent_idx is not None:
                assert cohesions[i] > cohesions[parent_idx], (
                    f"Child block {i} (cohesion {cohesions[i]}) should have higher cohesion than parent {parent_idx} (cohesion {cohesions[parent_idx]})"
                )

    def test_no_duplicate_blocks(self):
        """Ensure no duplicate blocks in results."""
        G = Graph()
        # Create cycle graph
        n = 8
        for i in range(1, n + 1):
            G.add_node(label=str(i))
        for i in range(1, n + 1):
            G.add_edge(str(i), str((i % n) + 1))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Check no duplicate blocks
        block_signatures = set()
        for block in blocks:
            assert all(isinstance(nid, int) for nid in block)  # Returns node IDs
            sig = tuple(sorted(block))
            assert sig not in block_signatures, f"Duplicate block found: {block}"
            block_signatures.add(sig)


class TestCohesiveBlocksDisconnected:
    """Test cohesive blocks on disconnected graphs."""

    def test_two_components(self):
        """Disconnected graph with two components."""
        G = Graph()

        # Component 1: triangle (cohesion 2)
        for i in range(1, 4):
            G.add_node(label=str(i))
        G.add_edge("1", "2")
        G.add_edge("2", "3")
        G.add_edge("3", "1")

        # Component 2: complete K_4 (cohesion 3)
        for i in range(4, 8):
            G.add_node(label=str(i))
        for i in range(4, 8):
            for j in range(i + 1, 8):
                G.add_edge(str(i), str(j))

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Should find both components
        assert len(blocks) >= 2
        assert 2 in cohesions  # Triangle has cohesion 2
        assert 3 in cohesions  # K_4 has cohesion 3

    def test_mixed_components(self):
        """Graph with connected and single-node components."""
        G = Graph()

        # Connected component
        for i in range(1, 4):
            G.add_node(label=str(i))
        G.add_edge("1", "2")
        G.add_edge("2", "3")

        # Isolated nodes
        G.add_node(label="4")
        G.add_node(label="5")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Should find 3 blocks total
        assert len(blocks) == 3
        # Two isolated nodes should have cohesion 0
        assert cohesions.count(0) == 2


class TestCohesiveBlocksComplexStructures:
    """Test cohesive blocks on more complex graph structures."""

    def test_karate_club_like(self):
        """Test on a structure similar to Zachary's karate club."""
        G = Graph()

        # Create a graph with two dense communities connected by bridge
        # Community 1 (5 nodes, highly connected)
        for i in range(1, 6):
            G.add_node(label=f"A{i}")
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(f"A{i}", f"A{j}")

        # Community 2 (5 nodes, highly connected)
        for i in range(1, 6):
            G.add_node(label=f"B{i}")
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(f"B{i}", f"B{j}")

        # Bridge edges between communities
        G.add_edge("A1", "B1")
        G.add_edge("A2", "B2")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Should find hierarchical structure
        assert len(blocks) >= 3

        # Each K_5 community has cohesion 4
        assert 4 in cohesions

    def test_tree_structure(self):
        """Trees have cohesion 1 and no sub-blocks."""
        G = Graph()

        # Create binary tree
        #       1
        #      / \
        #     2   3
        #    / \ / \
        #   4  5 6  7

        for i in range(1, 8):
            G.add_node(label=str(i))

        G.add_edge("1", "2")
        G.add_edge("1", "3")
        G.add_edge("2", "4")
        G.add_edge("2", "5")
        G.add_edge("3", "6")
        G.add_edge("3", "7")

        blocks, cohesions, parents = G.find_cohesive_blocks()

        # Tree should have cohesion 1, no higher-order sub-blocks
        assert len(blocks) == 1
        assert cohesions[0] == 1
