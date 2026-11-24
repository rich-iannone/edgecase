import random

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


class TestCohesiveBlocksExact:
    """Exact blocks, parents and agreement with connectivity and k-components."""

    def test_nested_blocks_of_figure_graph(self):
        """K_4 inside a ring inside the whole graph, as in the docstring figure."""
        G = Graph.from_edges(
            [
                ("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"),
                ("C", "D"), ("C", "E"), ("E", "F"), ("F", "G"), ("D", "G"),
                ("G", "H"),
            ]
        )  # fmt: skip
        blocks, cohesions, parents = G.find_cohesive_blocks()
        labels = [sorted(G.get_label_from_id(n) for n in b) for b in blocks]
        assert labels == [list("ABCDEFGH"), list("ABCDEFG"), list("ABCD")]
        assert cohesions == [1, 2, 3]
        assert parents == [None, 0, 1]

    def test_parents_offset_across_components(self):
        """Parent indices of a later component point into its own blocks."""
        G = Graph.from_edges(
            [
                ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"),
                ("E", "F"), ("F", "G"), ("G", "E"), ("G", "H"),
            ]
        )  # fmt: skip
        blocks, cohesions, parents = G.find_cohesive_blocks()
        assert cohesions == [1, 2, 1, 2]
        assert parents == [None, 0, None, 2]
        for child, parent in enumerate(parents):
            if parent is not None:
                assert set(blocks[child]) < set(blocks[parent])

    def test_edge_directions_ignored(self):
        """A directed triangle has the blocks of an undirected one."""
        G = Graph.from_edges([("A", "B"), ("B", "C"), ("C", "A")], directed=True)
        blocks, cohesions, parents = G.find_cohesive_blocks()
        assert [len(b) for b in blocks] == [3]
        assert cohesions == [2]
        assert parents == [None]

    def test_karate_club_matches_networkx_k_components(self):
        """The maximal blocks at each level are NetworkX's k-components."""
        nx = pytest.importorskip("networkx")
        import edgecase as ec

        G = ec.karate_club()
        blocks, cohesions, _ = G.find_cohesive_blocks()
        assert sorted(zip(map(len, blocks), cohesions)) == [
            (5, 3), (6, 2), (10, 4), (18, 3), (28, 2), (34, 1),
        ]  # fmt: skip

        nxg = nx.Graph([(u, v) for _, u, v in G.get_edges_as_tuples()])
        expected = {
            k: sorted(sorted(c) for c in comps) for k, comps in nx.k_components(nxg).items()
        }
        found = {k: sorted(comps) for k, comps in G.k_components().items()}
        assert found == expected

    @pytest.mark.parametrize("seed", range(40))
    def test_block_cohesion_is_its_connectivity(self, seed):
        """Each block's cohesion is the node connectivity of its subgraph."""
        nx = pytest.importorskip("networkx")
        rng = random.Random(seed)
        nxg = nx.gnp_random_graph(rng.randint(2, 12), rng.uniform(0.2, 0.8), seed=seed)
        G = Graph()
        for v in nxg:
            G.add_node(label=str(v))
        for u, v in nxg.edges():
            G.add_edge(str(u), str(v))
        to_nx = {G.get_id_from_label(str(v)): v for v in nxg}

        blocks, cohesions, parents = G.find_cohesive_blocks()
        for block, cohesion, parent in zip(blocks, cohesions, parents):
            sub = nxg.subgraph(to_nx[n] for n in block)
            if len(block) > 1:
                assert nx.node_connectivity(sub) == cohesion
            if parent is not None:
                assert cohesions[parent] < cohesion
                assert set(block) < set(blocks[parent])

    def test_k_components_are_maximal_blocks(self):
        """k_components() keeps the largest blocks of cohesion at least k."""
        import edgecase as ec

        G = ec.les_miserables()
        blocks, cohesions, _ = G.find_cohesive_blocks()
        for k, comps in G.k_components().items():
            candidates = [set(b) for b, c in zip(blocks, cohesions) if c >= k]
            maximal = [s for s in candidates if not any(s < t for t in candidates)]
            assert sorted(map(sorted, maximal)) == sorted(map(sorted, comps))
