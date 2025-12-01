import edgecase as ec


class TestWeaklyConnectedComponents:
    """Tests for weakly_connected_components() method."""

    def test_empty_graph(self):
        """Empty graph returns empty list."""
        g = ec.Graph(directed=True)

        wccs = g.weakly_connected_components()

        assert wccs == []

    def test_single_node(self):
        """Single node is one component."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        wccs = g.weakly_connected_components()

        assert len(wccs) == 1
        assert len(wccs[0]) == 1

    def test_disconnected_directed(self):
        """Disconnected directed graph has multiple WCCs."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge("A", "B")
        g.add_edge("X", "Y")

        wccs = g.weakly_connected_components()

        assert len(wccs) == 2

        # Check total nodes
        all_nodes = set()
        for wcc in wccs:
            all_nodes.update(wcc)

        assert len(all_nodes) == 4

    def test_connected_directed(self):
        """Connected directed graph has one WCC."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        wccs = g.weakly_connected_components()

        assert len(wccs) == 1
        assert len(wccs[0]) == 3

    def test_undirected_graph(self):
        """Undirected graph WCCs are regular connected components."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")  # C is isolated

        wccs = g.weakly_connected_components()

        assert len(wccs) == 2

    def test_returns_sets(self):
        """Components are returned as sets."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        wccs = g.weakly_connected_components()

        assert isinstance(wccs[0], set)

    def test_reverse_edges_still_connected(self):
        """Reverse edges don't affect WCC (direction ignored)."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("B", "A")  # Reverse direction
        g.add_edge("C", "B")

        wccs = g.weakly_connected_components()

        assert len(wccs) == 1


class TestStronglyConnectedComponents:
    """Tests for strongly_connected_components() method."""

    def test_empty_graph(self):
        """Empty graph returns empty list."""
        g = ec.Graph(directed=True)

        sccs = g.strongly_connected_components()

        assert sccs == []

    def test_single_node(self):
        """Single node is one SCC."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        sccs = g.strongly_connected_components()

        assert len(sccs) == 1
        assert len(sccs[0]) == 1

    def test_simple_cycle(self):
        """Cycle forms single SCC."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        sccs = g.strongly_connected_components()

        assert len(sccs) == 1
        assert len(sccs[0]) == 3

    def test_no_cycle_each_node_is_scc(self):
        """DAG: each node is its own SCC."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        sccs = g.strongly_connected_components()

        assert len(sccs) == 3

        for scc in sccs:
            assert len(scc) == 1

    def test_cycle_with_tail(self):
        """Cycle with outgoing edge to separate node."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Cycle: A, B, C
        g.add_edge("C", "D")  # D can be reached but can't reach back

        sccs = g.strongly_connected_components()

        assert len(sccs) == 2

        # Find the large and small components
        sizes = sorted([len(scc) for scc in sccs])

        assert sizes == [1, 3]

    def test_two_separate_cycles(self):
        """Two separate cycles form two SCCs."""
        g = ec.Graph(directed=True)

        # Cycle 1
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        # Cycle 2
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge("X", "Y")
        g.add_edge("Y", "X")

        sccs = g.strongly_connected_components()

        assert len(sccs) == 2

        for scc in sccs:
            assert len(scc) == 2

    def test_undirected_same_as_wcc(self):
        """Undirected graph: SCCs equal WCCs."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        # C is isolated

        sccs = g.strongly_connected_components()
        wccs = g.weakly_connected_components()

        assert len(sccs) == len(wccs) == 2

    def test_returns_sets(self):
        """Components are returned as sets."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        sccs = g.strongly_connected_components()

        assert isinstance(sccs[0], set)


class TestIsWeaklyConnected:
    """Tests for is_weakly_connected() method."""

    def test_empty_graph(self):
        """Empty graph is considered connected."""
        g = ec.Graph(directed=True)

        assert g.is_weakly_connected() is True

    def test_single_node(self):
        """Single node is connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        assert g.is_weakly_connected() is True

    def test_connected(self):
        """Connected graph returns True."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.is_weakly_connected() is True

    def test_disconnected(self):
        """Disconnected graph returns False."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="X")
        g.add_edge("A", "B")

        assert g.is_weakly_connected() is False

    def test_undirected(self):
        """Works same as is_connected for undirected graphs."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert g.is_weakly_connected() is True

        g.add_node(label="C")  # Isolated

        assert g.is_weakly_connected() is False


class TestIsStronglyConnected:
    """Tests for is_strongly_connected() method."""

    def test_empty_graph(self):
        """Empty graph is considered strongly connected."""
        g = ec.Graph(directed=True)

        assert g.is_strongly_connected() is True

    def test_single_node(self):
        """Single node is strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")

        assert g.is_strongly_connected() is True

    def test_simple_cycle(self):
        """Cycle is strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        assert g.is_strongly_connected() is True

    def test_dag_not_strongly_connected(self):
        """DAG (no cycles) is not strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.is_strongly_connected() is False

    def test_one_way_path(self):
        """One-way path is not strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert g.is_strongly_connected() is False

    def test_bidirectional_edge(self):
        """Two nodes with bidirectional edges are strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        assert g.is_strongly_connected() is True

    def test_cycle_with_unreachable_node(self):
        """Cycle with node that can't be reached is not strongly connected."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("D", "A")  # D can reach cycle but can't be reached

        assert g.is_strongly_connected() is False

    def test_undirected_same_as_connected(self):
        """Undirected graph: strongly connected = connected."""
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert g.is_strongly_connected() is True

        g.add_node(label="C")  # Isolated

        assert g.is_strongly_connected() is False


class TestComponentConsistency:
    """Test consistency between count and list methods."""

    def test_wcc_count_matches_list_length(self):
        """count_weakly_connected_components matches len(weakly_connected_components)."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge("A", "B")
        g.add_edge("X", "Y")

        count = g.count_weakly_connected_components()
        components = g.weakly_connected_components()

        assert count == len(components)

    def test_scc_count_matches_list_length(self):
        """count_strongly_connected_components matches len(strongly_connected_components)."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_node(label="D")

        count = g.count_strongly_connected_components()
        components = g.strongly_connected_components()

        assert count == len(components)

    def test_is_weakly_connected_consistent(self):
        """is_weakly_connected() consistent with count == 1."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert g.is_weakly_connected() == (g.count_weakly_connected_components() == 1)

        g.add_node(label="C")  # Isolated

        assert g.is_weakly_connected() == (g.count_weakly_connected_components() == 1)

    def test_is_strongly_connected_consistent(self):
        """is_strongly_connected() consistent with count == 1."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        assert g.is_strongly_connected() == (g.count_strongly_connected_components() == 1)

        g.add_node(label="C")  # Isolated

        assert g.is_strongly_connected() == (g.count_strongly_connected_components() == 1)

    def test_all_nodes_covered(self):
        """All nodes appear in exactly one component."""
        g = ec.Graph(directed=True)

        for i in range(10):
            g.add_node(label=str(i))

        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("5", "6")
        g.add_edge("6", "5")

        wccs = g.weakly_connected_components()
        all_wcc_nodes = set()
        for wcc in wccs:
            # Check no overlap
            assert all_wcc_nodes.isdisjoint(wcc)
            all_wcc_nodes.update(wcc)

        # Check all nodes covered
        assert len(all_wcc_nodes) == 10

        sccs = g.strongly_connected_components()
        all_scc_nodes = set()
        for scc in sccs:
            # Check no overlap
            assert all_scc_nodes.isdisjoint(scc)
            all_scc_nodes.update(scc)

        # Check all nodes covered
        assert len(all_scc_nodes) == 10
