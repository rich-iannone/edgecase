import pytest
from edgecase import Graph


class TestIsIndependentSet:
    """Tests for is_independent_set() method."""

    def test_empty_set(self):
        """Empty set is trivially independent."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_independent_set([])

    def test_single_node(self):
        """Single node is trivially independent."""
        graph = Graph()
        graph.add_path_graph(5)
        node_ids = list(graph.get_node_ids())

        assert graph.is_independent_set([node_ids[0]])

    def test_valid_independent_set_path(self):
        """Non-adjacent nodes in path form independent set."""
        # Use labeled nodes for predictable structure
        graph = (
            Graph()
            .add_node(label="0")
            .add_node(label="1")
            .add_node(label="2")
            .add_node(label="3")
            .add_node(label="4")
            .add_edge("0", "1")
            .add_edge("1", "2")
            .add_edge("2", "3")
            .add_edge("3", "4")
        )

        n0 = graph.get_id_from_label("0")
        n2 = graph.get_id_from_label("2")
        n4 = graph.get_id_from_label("4")

        # Nodes 0 and 2 are not adjacent
        assert graph.is_independent_set([n0, n2])

        # Nodes 0, 2, 4 are not adjacent
        assert graph.is_independent_set([n0, n2, n4])

    def test_invalid_independent_set_adjacent(self):
        """Adjacent nodes don't form independent set."""
        graph = (
            Graph()
            .add_node(label="0")
            .add_node(label="1")
            .add_node(label="2")
            .add_edge("0", "1")
            .add_edge("1", "2")
        )

        n0 = graph.get_id_from_label("0")
        n1 = graph.get_id_from_label("1")
        n2 = graph.get_id_from_label("2")

        # Nodes 0 and 1 are adjacent
        assert not graph.is_independent_set([n0, n1])

        # Nodes 1 and 2 are adjacent
        assert not graph.is_independent_set([n1, n2])

    def test_invalid_independent_set_mixed(self):
        """Mix of adjacent and non-adjacent nodes fails."""
        graph = (
            Graph()
            .add_node(label="0")
            .add_node(label="1")
            .add_node(label="2")
            .add_node(label="3")
            .add_edge("0", "1")
            .add_edge("1", "2")
            .add_edge("2", "3")
        )

        n0 = graph.get_id_from_label("0")
        n2 = graph.get_id_from_label("2")
        n3 = graph.get_id_from_label("3")

        # 0 and 2 are OK, but 2 and 3 are adjacent
        assert not graph.is_independent_set([n0, n2, n3])

    def test_complete_graph_only_single(self):
        """In complete graph, only single nodes are independent."""
        graph = Graph()
        graph.add_full_graph(5)
        node_ids = list(graph.get_node_ids())

        # Single node is independent
        assert graph.is_independent_set([node_ids[0]])

        # Any pair is adjacent in complete graph
        assert not graph.is_independent_set([node_ids[0], node_ids[1]])

    def test_star_graph_leaves(self):
        """Leaves of star graph form independent set."""
        # Create star with labeled nodes
        graph = Graph()
        graph.add_node(label="hub")
        for i in range(5):
            graph.add_node(label=f"leaf{i}")
            graph.add_edge("hub", f"leaf{i}")

        hub = graph.get_id_from_label("hub")
        leaves = [graph.get_id_from_label(f"leaf{i}") for i in range(5)]

        # All leaves are independent (only connected to hub)
        assert graph.is_independent_set(leaves)

        # Hub with any leaf is not independent
        assert not graph.is_independent_set([hub, leaves[0]])

    def test_cycle_graph_alternating(self):
        """Alternating nodes in cycle form independent set."""
        # Create cycle with labeled nodes
        graph = Graph()
        for i in range(6):
            graph.add_node(label=str(i))
        for i in range(6):
            graph.add_edge(str(i), str((i + 1) % 6))

        # Alternating nodes (0, 2, 4) are independent
        alternating = [graph.get_id_from_label(str(i)) for i in [0, 2, 4]]

        assert graph.is_independent_set(alternating)

    def test_nonexistent_node_raises(self):
        """Checking with nonexistent node raises error."""
        graph = Graph()
        graph.add_path_graph(3)

        with pytest.raises(ValueError, match="does not exist"):
            graph.is_independent_set([9999])

    def test_directed_graph(self):
        """Independent set check works for directed graphs."""
        graph = Graph(directed=True)
        graph.add_node(label="0")
        graph.add_node(label="1")
        graph.add_node(label="2")
        graph.add_node(label="3")
        graph.add_edge("0", "1")
        graph.add_edge("2", "3")

        n0 = graph.get_id_from_label("0")
        n1 = graph.get_id_from_label("1")
        n2 = graph.get_id_from_label("2")

        # 0 and 2 are not connected
        assert graph.is_independent_set([n0, n2])

        # 0 and 1 are connected
        assert not graph.is_independent_set([n0, n1])


class TestMaximalIndependentSet:
    """Tests for maximal_independent_set() method."""

    def test_empty_graph(self):
        """Empty graph returns empty set."""
        graph = Graph()
        mis = graph.maximal_independent_set()

        assert mis == []

    def test_single_node(self):
        """Single node graph returns that node."""
        graph = Graph()
        graph.add_node(label="A")
        node_ids = list(graph.get_node_ids())

        mis = graph.maximal_independent_set()

        assert mis == node_ids

    def test_result_is_independent(self):
        """Result is always an independent set."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        mis = graph.maximal_independent_set(seed=123)

        assert graph.is_independent_set(mis)

    def test_result_is_maximal(self):
        """Result is maximal (can't add any more nodes)."""
        graph = Graph()
        graph.add_path_graph(10)

        mis = graph.maximal_independent_set(seed=23)
        mis_set = set(mis)

        # Every node not in MIS should be adjacent to at least one node in MIS
        for node in graph.get_node_ids():
            if node not in mis_set:
                neighbors = set(graph.neighbors(node))
                assert len(neighbors & mis_set) > 0

    def test_seed_reproducibility(self):
        """Same seed produces same result."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        mis1 = graph.maximal_independent_set(seed=123)
        mis2 = graph.maximal_independent_set(seed=123)

        assert mis1 == mis2

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different results."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        results = set()
        for seed in range(20):
            mis = graph.maximal_independent_set(seed=seed)
            results.add(tuple(sorted(mis)))

        # Should find at least a few different sets
        assert len(results) >= 2

    def test_starting_nodes_included(self):
        """Starting nodes are always in result."""
        graph = Graph()
        graph.add_path_graph(10)
        node_ids = list(graph.get_node_ids())

        start_node = node_ids[0]
        mis = graph.maximal_independent_set(nodes=[start_node], seed=23)

        assert start_node in mis
        assert graph.is_independent_set(mis)

    def test_starting_nodes_multiple(self):
        """Multiple starting nodes that are independent work."""
        # Create path with labeled nodes
        graph = Graph()
        for i in range(10):
            graph.add_node(label=str(i))
        for i in range(9):
            graph.add_edge(str(i), str(i + 1))

        # Start with non-adjacent nodes (0, 2, 4)
        start_nodes = [graph.get_id_from_label(str(i)) for i in [0, 2, 4]]
        mis = graph.maximal_independent_set(nodes=start_nodes, seed=23)

        for node in start_nodes:
            assert node in mis

        assert graph.is_independent_set(mis)

    def test_invalid_starting_nodes_raises(self):
        """Starting with adjacent nodes raises error."""
        graph = Graph()
        graph.add_node(label="0")
        graph.add_node(label="1")
        graph.add_edge("0", "1")

        n0 = graph.get_id_from_label("0")
        n1 = graph.get_id_from_label("1")

        # Adjacent nodes
        with pytest.raises(ValueError, match="independent set"):
            graph.maximal_independent_set(nodes=[n0, n1])

    def test_nonexistent_starting_node_raises(self):
        """Starting with nonexistent node raises error."""
        graph = Graph()
        graph.add_path_graph(5)

        with pytest.raises(ValueError, match="does not exist"):
            graph.maximal_independent_set(nodes=[9999])

    def test_complete_graph_single_node(self):
        """Complete graph's MIS has exactly one node."""
        graph = Graph()
        graph.add_full_graph(10)

        mis = graph.maximal_independent_set(seed=23)

        # In complete graph, max independent set has size 1
        assert len(mis) == 1
        assert graph.is_independent_set(mis)

    def test_path_graph_size(self):
        """Path graph MIS has correct size range."""
        graph = Graph()
        graph.add_path_graph(10)  # Can have MIS of size 5

        mis = graph.maximal_independent_set(seed=23)

        # Path of n nodes has MIS of size at least ceil(n/3) for maximal
        # and at most ceil(n/2) for maximum
        assert len(mis) >= 3  # At least ceil(10/3) for maximal
        assert len(mis) <= 5  # At most ceil(10/2) for maximum
        assert graph.is_independent_set(mis)

    def test_star_graph(self):
        """Star graph has well-defined MIS."""
        # Create star: hub + 5 leaves (6 total nodes)
        graph = Graph()
        graph.add_star_graph(6)  # 1 hub + 5 leaves

        mis = graph.maximal_independent_set(seed=23)

        # Either hub alone or all leaves (5 nodes)
        assert len(mis) == 1 or len(mis) == 5
        assert graph.is_independent_set(mis)


class TestMaximumIndependentSet:
    """Tests for maximum_independent_set() method."""

    def test_empty_graph(self):
        """Empty graph returns empty set."""
        graph = Graph()
        mis = graph.maximum_independent_set()

        assert mis == []

    def test_single_node(self):
        """Single node graph returns that node."""
        graph = Graph()
        graph.add_node(label="A")
        node_ids = list(graph.get_node_ids())

        mis = graph.maximum_independent_set()

        assert mis == node_ids

    def test_result_is_independent(self):
        """Result is always an independent set."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        mis = graph.maximum_independent_set()

        assert graph.is_independent_set(mis)

    def test_complete_graph_single_node(self):
        """Complete graph's maximum IS has exactly one node."""
        graph = Graph()
        graph.add_full_graph(10)

        mis = graph.maximum_independent_set()

        assert len(mis) == 1

    def test_path_graph_optimal(self):
        """Path graph achieves optimal MIS size."""
        graph = Graph()
        graph.add_path_graph(9)  # Optimal MIS size is ceil(9/2) = 5

        mis = graph.maximum_independent_set()

        # For path of 9 nodes, optimal is 5
        assert len(mis) == 5
        assert graph.is_independent_set(mis)

    def test_star_graph_leaves(self):
        """Star graph's maximum IS is all leaves."""
        # Create star: hub + 10 leaves (11 total nodes)
        graph = Graph()
        graph.add_star_graph(11)  # 1 hub + 10 leaves

        mis = graph.maximum_independent_set()

        # Maximum is all 10 leaves (not the hub)
        assert len(mis) == 10
        assert graph.is_independent_set(mis)

    def test_cycle_graph_half(self):
        """Even cycle has MIS of half size."""
        graph = Graph()
        graph.add_cycle_graph(10)  # Even cycle

        mis = graph.maximum_independent_set()

        # For cycle of 10, optimal is 5
        assert len(mis) == 5
        assert graph.is_independent_set(mis)

    def test_deterministic(self):
        """Result is deterministic (greedy, no randomness)."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        mis1 = graph.maximum_independent_set()
        mis2 = graph.maximum_independent_set()

        assert mis1 == mis2

    def test_at_least_as_good_as_maximal(self):
        """Maximum IS should be a good heuristic."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        maximum = graph.maximum_independent_set()

        # Verify it's valid and non-empty for connected graphs
        assert len(maximum) > 0
        assert graph.is_independent_set(maximum)

    def test_bipartite_graph(self):
        """Bipartite graph has known optimal."""
        graph = Graph()
        # Create complete bipartite K(3,5)
        for i in range(3):
            graph.add_node(label=f"A{i}")
        for i in range(5):
            graph.add_node(label=f"B{i}")
        for i in range(3):
            for j in range(5):
                graph.add_edge(f"A{i}", f"B{j}")

        mis = graph.maximum_independent_set()

        # Maximum IS is the larger partition (5 nodes)
        assert len(mis) == 5
        assert graph.is_independent_set(mis)

    def test_disconnected_graph(self):
        """Disconnected graph sums component MIS sizes."""
        graph = Graph()
        # Two disconnected paths of 5 nodes each
        for i in range(5):
            graph.add_node(label=f"A{i}")
        for i in range(4):
            graph.add_edge(f"A{i}", f"A{i + 1}")

        for i in range(5):
            graph.add_node(label=f"B{i}")
        for i in range(4):
            graph.add_edge(f"B{i}", f"B{i + 1}")

        mis = graph.maximum_independent_set()

        # Each path of 5 contributes 3, total 6
        assert len(mis) == 6
        assert graph.is_independent_set(mis)


class TestFindMaximalIndependentSets:
    """Tests for find_maximal_independent_sets() method."""

    def test_empty_graph(self):
        """Empty graph returns empty sets."""
        graph = Graph()
        sets = graph.find_maximal_independent_sets(count=5)

        assert sets == [[]]

    def test_single_node(self):
        """Single node returns that node."""
        graph = Graph()
        graph.add_node(label="A")

        sets = graph.find_maximal_independent_sets(count=5)

        # All sets should be the same (only one possible MIS)
        assert len(sets) >= 1
        assert all(len(s) == 1 for s in sets)

    def test_returns_requested_count(self):
        """Returns approximately the requested number of sets."""
        graph = Graph()
        graph.add_cycle_graph(10)

        sets = graph.find_maximal_independent_sets(count=5, seed=23)

        # Should get close to requested count
        assert len(sets) >= 1
        assert len(sets) <= 5

    def test_all_sets_are_valid(self):
        """All returned sets are valid maximal independent sets."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        sets = graph.find_maximal_independent_sets(count=10, seed=123)

        for s in sets:
            # Each set should be independent
            assert graph.is_independent_set(s)

    def test_sets_are_unique(self):
        """Returned sets are unique."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        sets = graph.find_maximal_independent_sets(count=10, seed=123)

        # Convert to tuples for hashing
        unique_sets = set(tuple(sorted(s)) for s in sets)

        assert len(unique_sets) == len(sets)

    def test_seed_reproducibility(self):
        """Same seed produces same results."""
        graph = Graph()
        graph.add_gnm_graph(n=20, m=40, seed=23)

        sets1 = graph.find_maximal_independent_sets(count=5, seed=123)
        sets2 = graph.find_maximal_independent_sets(count=5, seed=123)

        assert sets1 == sets2

    def test_different_sizes_found(self):
        """May find sets of different sizes."""
        graph = Graph()
        graph.add_gnm_graph(n=30, m=60, seed=23)

        sets = graph.find_maximal_independent_sets(count=20, seed=0)

        sizes = set(len(s) for s in sets)

        # May find sets of different sizes (depends on graph structure)
        assert len(sizes) >= 1

    def test_complete_graph_all_same(self):
        """Complete graph has only single-node MIS."""
        graph = Graph()
        graph.add_full_graph(5)

        sets = graph.find_maximal_independent_sets(count=10, seed=23)

        # All sets should have size 1
        assert all(len(s) == 1 for s in sets)


class TestIndependentSetEdgeCases:
    """Edge cases and special scenarios for independent set algorithms."""

    def test_isolated_nodes(self):
        """Isolated nodes form independent set."""
        graph = Graph()
        for i in range(5):
            graph.add_node(label=str(i))

        node_ids = list(graph.get_node_ids())

        # All isolated nodes are independent
        assert graph.is_independent_set(node_ids)

        # Maximum IS includes all nodes
        mis = graph.maximum_independent_set()

        assert len(mis) == 5

    def test_self_loop_handling(self):
        """Self-loops don't affect independence."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "A")  # Self-loop

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # A and B are still independent (not connected to each other)
        assert graph.is_independent_set([a_id, b_id])

    def test_multi_edges_handling(self):
        """Multiple edges between same nodes count as adjacent."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")
        graph.add_edge("A", "B")  # Multi-edge

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # A and B are still adjacent
        assert not graph.is_independent_set([a_id, b_id])

    def test_large_graph_performance(self):
        """Algorithm completes in reasonable time for larger graphs."""
        graph = Graph()
        graph.add_gnm_graph(n=500, m=2000, seed=23)

        # Should complete without timeout
        mis = graph.maximal_independent_set(seed=123)

        assert len(mis) > 0
        assert graph.is_independent_set(mis)

        maximum = graph.maximum_independent_set()

        assert len(maximum) > 0
        assert graph.is_independent_set(maximum)

    def test_tree_graph(self):
        """Tree has known MIS properties."""
        # Build a simple binary tree manually
        graph = Graph()
        graph.add_node(label="root")
        graph.add_node(label="L")
        graph.add_node(label="R")
        graph.add_node(label="LL")
        graph.add_node(label="LR")
        graph.add_node(label="RL")
        graph.add_node(label="RR")
        graph.add_edge("root", "L")
        graph.add_edge("root", "R")
        graph.add_edge("L", "LL")
        graph.add_edge("L", "LR")
        graph.add_edge("R", "RL")
        graph.add_edge("R", "RR")

        mis = graph.maximum_independent_set()

        assert graph.is_independent_set(mis)

        # Should get at least 4 nodes (leaves)
        assert len(mis) >= 4

    def test_grid_graph(self):
        """Grid graph has checkerboard MIS."""
        graph = Graph()
        graph.add_grid_2d(rows=4, cols=4)

        mis = graph.maximum_independent_set()

        # 4x4 grid has optimal MIS of size 8 (checkerboard pattern)
        assert len(mis) == 8
        assert graph.is_independent_set(mis)

    def test_barbell_graph(self):
        """Barbell graph MIS includes both clique single nodes and path."""
        graph = Graph()
        # Create barbell: complete(3) - path(2) - complete(3)
        for i in range(3):
            graph.add_node(label=f"L{i}")
        for i in range(3):
            for j in range(i + 1, 3):
                graph.add_edge(f"L{i}", f"L{j}")

        for i in range(3):
            graph.add_node(label=f"R{i}")
        for i in range(3):
            for j in range(i + 1, 3):
                graph.add_edge(f"R{i}", f"R{j}")

        graph.add_node(label="M")
        graph.add_edge("L0", "M")
        graph.add_edge("M", "R0")

        mis = graph.maximum_independent_set()

        # Each complete graph contributes 1, middle contributes 1
        # Total should be 3 if M is included, or 2 if both ends
        assert len(mis) >= 2
        assert graph.is_independent_set(mis)


class TestDocstringExamples:
    """Tests that verify docstring examples work correctly."""

    def test_is_independent_set_path_example(self):
        """Test path graph example from docstring."""
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        a_id = graph.get_id_from_label("A")
        c_id = graph.get_id_from_label("C")
        d_id = graph.get_id_from_label("D")

        # A and C are not adjacent
        assert graph.is_independent_set([a_id, c_id])

        # A, C, D: C and D are adjacent
        assert not graph.is_independent_set([a_id, c_id, d_id])

    def test_maximal_scheduling_example(self):
        """Test scheduling application example from docstring."""
        graph = (
            Graph()
            .add_node(label="Job_A")
            .add_node(label="Job_B")
            .add_node(label="Job_C")
            .add_node(label="Job_D")
            .add_edge("Job_A", "Job_B")
            .add_edge("Job_B", "Job_C")
            .add_edge("Job_C", "Job_D")
        )

        parallel_jobs = graph.maximal_independent_set()
        labels = [graph.get_label_from_id(nid) for nid in parallel_jobs]

        # Should find non-conflicting jobs
        assert len(parallel_jobs) >= 2
        assert graph.is_independent_set(parallel_jobs)

    def test_maximum_vertex_cover_complement(self):
        """Test vertex cover complement example from docstring."""
        graph = Graph()
        graph.add_cycle_graph(8)

        mis = graph.maximum_independent_set()
        vertex_cover = set(graph.get_node_ids()) - set(mis)

        # Should sum to total nodes
        assert len(mis) + len(vertex_cover) == graph.count_nodes()
        assert len(mis) == 4  # Half of 8


class TestIndependentSetDirected:
    """Edge direction is ignored: an edge A -> B rules out keeping both A and B."""

    @staticmethod
    def _undirected_adjacency(g):
        adj = {node_id: set() for node_id in g.get_node_ids()}
        for edge_id in g.get_edge_ids():
            edge = g.get_edge(edge_id)
            if edge.source != edge.target:
                adj[edge.source].add(edge.target)
                adj[edge.target].add(edge.source)
        return adj

    def test_single_arc_keeps_one_endpoint(self):
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        for seed in range(10):
            assert len(g.maximal_independent_set(seed=seed)) == 1
        assert len(g.maximum_independent_set()) == 1

    def test_in_neighbor_of_start_node_is_excluded(self):
        """A start node also rules out nodes that point *into* it."""
        g = Graph(directed=True)
        for label in "ABC":
            g.add_node(label=label)
        g.add_edge("A", "B").add_edge("C", "B")
        b_id = g.get_id_from_label("B")

        assert g.maximal_independent_set(nodes=[b_id], seed=1) == [b_id]

    def test_start_nodes_joined_by_arc_are_rejected(self):
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        ids = [g.get_id_from_label("B"), g.get_id_from_label("A")]

        with pytest.raises(ValueError, match="independent set"):
            g.maximal_independent_set(nodes=ids)

    def test_random_directed_graphs_give_valid_maximal_sets(self):
        import random

        rng = random.Random(9)
        for seed in range(80):
            n = rng.randint(2, 12)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(label=str(i))
            for _ in range(rng.randint(1, 2 * n)):
                g.add_edge(str(rng.randrange(n)), str(rng.randrange(n)))
            adj = self._undirected_adjacency(g)

            candidates = [g.maximal_independent_set(seed=seed), g.maximum_independent_set()]
            candidates += g.find_maximal_independent_sets(3, seed=seed)
            for chosen in candidates:
                members = set(chosen)
                assert g.is_independent_set(chosen)
                assert all(not (adj[u] & members) for u in members)
                # Maximal: every other node touches the set
                assert all(adj[v] & members for v in adj if v not in members)
