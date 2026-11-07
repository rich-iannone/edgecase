import pytest
from edgecase import Graph


class TestDispersion:
    """Test dispersion metric between node pairs."""

    def test_basic_dispersion(self):
        """Test basic dispersion calculation."""
        graph = Graph()
        # Create a simple network
        for i in range(1, 6):
            graph.add_node()
        for s, t in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]:
            graph.add_edge(s, t)

        # Calculate dispersion
        disp = graph.dispersion(1, 2)

        assert isinstance(disp, float)
        assert disp >= 0.0

    def test_dispersion_normalized_vs_unnormalized(self):
        """Test normalized vs unnormalized dispersion."""
        graph = Graph()
        graph.add_cycle_graph(6)

        nodes = sorted(graph.get_node_ids())

        norm = graph.dispersion(nodes[0], nodes[1], normalized=True)
        unnorm = graph.dispersion(nodes[0], nodes[1], normalized=False)

        assert isinstance(norm, float)
        assert isinstance(unnorm, float)

        # Normalized should generally be <= unnormalized (except edge cases)
        assert norm >= 0.0
        assert unnorm >= 0.0

    def test_dispersion_no_common_neighbors(self):
        """Nodes with no common neighbors have zero dispersion."""
        graph = Graph()

        # Path graph: 1 - 2 - 3 - 4
        graph.add_path_graph(4)

        nodes = sorted(graph.get_node_ids())

        # Nodes at opposite ends have no common neighbors
        disp = graph.dispersion(nodes[0], nodes[3])

        assert disp == 0.0

    def test_dispersion_triangle(self):
        """Test dispersion in a triangle (complete graph K_3)."""
        graph = Graph()
        graph.add_cycle_graph(3)

        nodes = sorted(graph.get_node_ids())

        # In a triangle, all nodes are connected
        # The one common neighbor is connected to both
        disp = graph.dispersion(nodes[0], nodes[1], normalized=False)

        # Should be 0 because the common neighbor is connected to both
        assert disp == 0.0

    def test_dispersion_star_graph(self):
        """Test dispersion in star graph."""
        graph = Graph()
        graph.add_star_graph(5)

        nodes = sorted(graph.get_node_ids())
        hub = nodes[0]  # First node is the hub
        leaf1 = nodes[1]
        leaf2 = nodes[2]

        # Hub to leaf: no common neighbors
        disp_hub_leaf = graph.dispersion(hub, leaf1)

        assert disp_hub_leaf == 0.0

        # Leaf to leaf: hub is common neighbor, but leaves not connected
        # This should have positive dispersion
        disp_leaf_leaf = graph.dispersion(leaf1, leaf2, normalized=False)

        assert disp_leaf_leaf >= 0.0

    def test_dispersion_with_labels(self):
        """Test dispersion using node labels."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")
        d_id = graph.get_id_from_label("D")

        graph.add_edge(a_id, b_id)
        graph.add_edge(a_id, c_id)
        graph.add_edge(b_id, c_id)
        graph.add_edge(b_id, d_id)

        disp = graph.dispersion(a_id, b_id)

        assert isinstance(disp, float)
        assert disp >= 0.0

    def test_dispersion_nonexistent_node(self):
        """Test dispersion with nonexistent nodes raises error."""
        graph = Graph()
        graph.add_cycle_graph(5)

        with pytest.raises(ValueError, match="does not exist"):
            graph.dispersion(1, 9999)

        with pytest.raises(ValueError, match="does not exist"):
            graph.dispersion(9999, 1)

    def test_dispersion_complete_graph(self):
        """Test dispersion in complete graph."""
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        # In complete graph, all common neighbors are connected
        # So dispersion should be 0
        disp = graph.dispersion(nodes[0], nodes[1], normalized=False)

        assert disp == 0.0

    def test_dispersion_disconnected_nodes(self):
        """Test dispersion between disconnected nodes."""
        graph = Graph()

        # Two separate components
        for i in range(1, 7):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)

        # Nodes in different components have no common neighbors
        disp = graph.dispersion(1, 4)

        assert disp == 0.0

    def test_shared_neighbor_outside_ego_network(self):
        """A neighbor shared by s and t only counts if it is also a neighbor of u."""
        graph = Graph().add_nodes(["a", "b", "s", "t", "x"])
        graph.add_edges([("a", "s"), ("a", "t"), ("b", "s"), ("b", "t"), ("s", "x"), ("t", "x")])
        assert graph.dispersion("a", "b") == 0.5
        assert graph.dispersion("a", "b", normalized=False) == 1.0
        graph.add_edge("a", "x")
        assert graph.dispersion("a", "b") == 0.0
        assert graph.dispersion("b", "a") == 0.5  # x is not a neighbor of b

    def test_matches_networkx(self):
        import random

        import networkx as nx

        for seed in range(40):
            G = nx.gnp_random_graph(random.Random(seed).randint(4, 14), 0.4, seed=seed)
            graph = Graph().add_nodes([str(n) for n in G])
            graph.add_edges([(str(u), str(v)) for u, v in G.edges])
            for u in G:
                for v in G:
                    if u == v:
                        continue
                    for normalized in (True, False):
                        expected = nx.dispersion(G, u, v, normalized=normalized)
                        got = graph.dispersion(str(u), str(v), normalized=normalized)
                        assert got == pytest.approx(expected)


class TestGlobalEfficiency:
    """Test global efficiency metric."""

    def test_empty_graph(self):
        """Empty graph has efficiency 0."""
        graph = Graph()

        eff = graph.global_efficiency()

        assert eff == 0.0

    def test_single_node(self):
        """Single node graph has efficiency 0."""
        graph = Graph()
        graph.add_node(label="A")

        eff = graph.global_efficiency()

        assert eff == 0.0

    def test_complete_graph(self):
        """Complete graph has efficiency 1."""
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        eff = graph.global_efficiency()

        # Complete graph should have efficiency 1.0
        assert abs(eff - 1.0) < 1e-6

    def test_path_graph(self):
        """Test efficiency of path graph."""
        graph = Graph()
        graph.add_path_graph(5)

        eff = graph.global_efficiency()

        assert 0.0 < eff < 1.0

    def test_cycle_graph(self):
        """Test efficiency of cycle graph."""
        graph = Graph()
        graph.add_cycle_graph(6)

        eff = graph.global_efficiency()

        assert 0.0 < eff < 1.0

    def test_star_graph(self):
        """Test efficiency of star graph."""
        graph = Graph()
        graph.add_star_graph(5)

        eff = graph.global_efficiency()

        assert 0.0 < eff < 1.0
        # Star is fairly efficient due to central hub

    def test_disconnected_graph(self):
        """Test efficiency of disconnected graph."""
        graph = Graph()

        # Two separate triangles
        for i in range(1, 7):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(6, 4)

        eff = graph.global_efficiency()

        # Two perfect triangles have efficiency 1.0 within components
        # but unreachable pairs contribute 0, so overall < 1.0
        # Actually, NetworkX counts unreachable pairs as distance infinity (efficiency 0)
        # So: 6 pairs within triangles (efficiency 1.0) + 9 cross-component pairs (efficiency 0)
        # = 6/15 = 0.4 (the old implementation dropped unreachable pairs and returned 1.0)
        assert eff == pytest.approx(0.4)

    def test_two_separate_edges(self):
        """Only 2 of the 6 node pairs can reach each other, so efficiency is 1/3, not 1.0."""
        graph = Graph()
        for label in "ABCD":
            graph.add_node(label=label)
        graph.add_edge("A", "B")
        graph.add_edge("C", "D")

        assert graph.global_efficiency() == pytest.approx(1 / 3)

    def test_isolated_node_lowers_efficiency(self):
        """Adding an isolated node to a triangle: 6 of 12 ordered pairs remain reachable."""
        graph = Graph()
        for label in "ABCD":
            graph.add_node(label=label)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        assert graph.global_efficiency() == pytest.approx(0.5)

    def test_matches_networkx_including_disconnected(self):
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(5)
        for seed in range(60):
            reference = nx.gnp_random_graph(rng.randint(2, 15), rng.random() * 0.4, seed=seed)
            graph = Graph()
            for node in reference:
                graph.add_node(label=str(node))
            for u, v in reference.edges():
                graph.add_edge(str(u), str(v))

            assert graph.global_efficiency() == pytest.approx(nx.global_efficiency(reference))

    def test_directed_averages_over_ordered_pairs(self):
        """A -> B -> C: reachable ordered pairs give 1 + 1/2 + 1 = 2.5 over 6 pairs."""
        graph = Graph(directed=True)
        for label in "ABC":
            graph.add_node(label=label)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        assert graph.global_efficiency() == pytest.approx(2.5 / 6)

    def test_efficiency_ordering(self):
        """Complete > Random > Path in efficiency."""
        # Complete graph
        complete = Graph()
        n = 6
        for i in range(n):
            complete.add_node(label=f"n{i}")
        nodes = sorted(complete.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                complete.add_edge(nodes[i], nodes[j])

        # Path graph
        path = Graph()
        path.add_path_graph(6)

        complete_eff = complete.global_efficiency()
        path_eff = path.global_efficiency()

        # Complete should be more efficient than path
        assert complete_eff > path_eff

    def test_two_nodes(self):
        """Test efficiency with two connected nodes."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        graph.add_edge(a_id, b_id)

        eff = graph.global_efficiency()

        # Efficiency should be 1.0 (distance 1, efficiency 1/1 = 1)
        assert abs(eff - 1.0) < 1e-6


class TestLocalEfficiency:
    """Test local efficiency metric."""

    def test_empty_graph(self):
        """Empty graph has local efficiency 0."""
        graph = Graph()

        eff = graph.local_efficiency()

        assert eff == 0.0

    def test_single_node(self):
        """Single node graph has local efficiency 0."""
        graph = Graph()
        graph.add_node(label="A")

        eff = graph.local_efficiency()

        assert eff == 0.0

    def test_complete_graph(self):
        """Complete graph has local efficiency 1."""
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        eff = graph.local_efficiency()

        # Complete graph should have local efficiency 1.0
        assert abs(eff - 1.0) < 1e-6

    def test_star_graph(self):
        """Test local efficiency of star graph."""
        graph = Graph()
        graph.add_star_graph(5)

        eff = graph.local_efficiency()

        # Star has poor local efficiency (leaves not connected)
        # and should be low because leaf neighborhoods are empty
        assert eff >= 0.0

    def test_cycle_graph(self):
        """Test local efficiency of cycle graph."""
        graph = Graph()
        graph.add_cycle_graph(6)

        eff = graph.local_efficiency()

        assert 0.0 <= eff <= 1.0

    def test_path_graph(self):
        """Test local efficiency of path graph."""
        graph = Graph()
        graph.add_path_graph(5)

        eff = graph.local_efficiency()

        # Path has low local efficiency
        assert eff >= 0.0

    def test_triangle(self):
        """Test local efficiency of triangle."""
        graph = Graph()
        graph.add_cycle_graph(3)

        eff = graph.local_efficiency()

        # Triangle's neighborhoods are edges, which have eff 1
        assert eff > 0.0

    def test_two_nodes(self):
        """Two connected nodes have local efficiency 0."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        graph.add_edge(a_id, b_id)

        eff = graph.local_efficiency()

        # Each node has only 1 neighbor, so no pairs to measure
        assert eff == 0.0

    def test_disconnected_components(self):
        """Test local efficiency with disconnected components."""
        graph = Graph()

        # Two triangles
        for i in range(1, 7):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(6, 4)

        eff = graph.local_efficiency()

        # Each triangle has good local efficiency
        assert eff > 0.0

    def test_parallel_edges_count_once(self):
        """A triangle with a doubled A-B edge is still a triangle: every node scores 1.0."""
        graph = Graph()
        for label in "ABC":
            graph.add_node(label=label)
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        assert graph.local_efficiency() == pytest.approx(1.0)

    def test_self_loop_does_not_join_own_neighborhood(self):
        """A self-loop on the center of a star must not let its leaves route through it."""
        graph = Graph()
        for label in ["hub", "A", "B"]:
            graph.add_node(label=label)
        graph.add_edge("hub", "A")
        graph.add_edge("hub", "B")
        graph.add_edge("hub", "hub")

        assert graph.local_efficiency() == 0.0

    def test_matches_networkx_with_parallel_edges_and_self_loops(self):
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(2)
        for seed in range(60):
            reference = nx.gnp_random_graph(rng.randint(2, 14), rng.random() * 0.5, seed=seed)
            graph = Graph()
            for node in reference:
                graph.add_node(label=str(node))
            for u, v in reference.edges():
                graph.add_edge(str(u), str(v))
            for u, v in list(reference.edges())[:4]:
                graph.add_edge(str(v), str(u))
            for node in list(reference)[:2]:
                graph.add_edge(str(node), str(node))

            assert graph.local_efficiency() == pytest.approx(nx.local_efficiency(reference))

    def test_directed_uses_ordered_pairs(self):
        """Hub -> {A, B} with A -> B: within the neighborhood only A reaches B, so 1/2 of the
        ordered pairs are efficient; averaged over 3 nodes gives 1/6, whatever the edge order."""
        for edges in (
            [("Hub", "A"), ("Hub", "B"), ("A", "B")],
            [("A", "B"), ("Hub", "B"), ("Hub", "A")],
        ):
            graph = Graph(directed=True)
            for label in ["Hub", "A", "B"]:
                graph.add_node(label=label)
            for source, target in edges:
                graph.add_edge(source, target)

            assert graph.local_efficiency() == pytest.approx(1 / 6)


class TestEfficiencyIntegration:
    """Integration tests for efficiency metrics."""

    def test_global_vs_local_complete(self):
        """Complete graphs have global = local = 1."""
        graph = Graph()
        n = 4
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        global_eff = graph.global_efficiency()
        local_eff = graph.local_efficiency()

        assert abs(global_eff - 1.0) < 1e-6
        assert abs(local_eff - 1.0) < 1e-6

    def test_efficiencies_in_range(self):
        """All efficiencies should be in [0, 1]."""
        graph = Graph()

        # Create a random graph manually
        import random

        random.seed(23)
        n = 10
        for i in range(n):
            graph.add_node()
        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    graph.add_edge(nodes[i], nodes[j])

        global_eff = graph.global_efficiency()
        local_eff = graph.local_efficiency()

        assert 0.0 <= global_eff <= 1.0
        assert 0.0 <= local_eff <= 1.0

    def test_small_world_properties(self):
        """Small-world networks have high local, reasonable global efficiency."""
        graph = Graph()

        # Create a small-world-like graph
        graph.add_cycle_graph(10)

        # Add some random shortcuts
        nodes = sorted(graph.get_node_ids())
        graph.add_edge(nodes[0], nodes[5])
        graph.add_edge(nodes[2], nodes[7])

        global_eff = graph.global_efficiency()
        local_eff = graph.local_efficiency()

        # Should have reasonably high efficiencies
        assert global_eff > 0.0
        assert local_eff >= 0.0


class TestDispersionEfficiencyComparison:
    """Compare dispersion and efficiency metrics."""

    def test_complete_graph_metrics(self):
        """Complete graph has specific efficiency and dispersion values."""
        graph = Graph()
        n = 4
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        # Efficiency should be 1
        assert abs(graph.global_efficiency() - 1.0) < 1e-6
        assert abs(graph.local_efficiency() - 1.0) < 1e-6

        # Dispersion should be 0 (all neighbors connected)
        disp = graph.dispersion(nodes[0], nodes[1], normalized=False)

        assert disp == 0.0

    def test_star_graph_metrics(self):
        """Star graph has specific patterns in metrics."""
        graph = Graph()
        graph.add_star_graph(5)

        nodes = sorted(graph.get_node_ids())

        # Global efficiency should be moderate (hub connects all)
        global_eff = graph.global_efficiency()

        assert 0.0 < global_eff < 1.0

        # Dispersion between leaves should be positive
        # (hub is common neighbor, but leaves not connected)
        disp = graph.dispersion(nodes[1], nodes[2], normalized=False)

        assert disp >= 0.0
