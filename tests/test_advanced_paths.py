import itertools
import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_METRIC, KIND_QUERY


class TestAllPairsShortestPathLength:
    """Tests for all_pairs_shortest_path_length method."""

    def test_simple_path_graph(self):
        """Test all pairs distances on a simple path graph."""
        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        distances = graph.all_pairs_shortest_path_length()

        # Check some specific distances
        assert distances[1][1] == 0.0  # Distance to self
        assert distances[1][2] == 1.0
        assert distances[1][3] == 2.0
        assert distances[1][4] == 3.0
        assert distances[2][4] == 2.0
        assert distances[3][4] == 1.0

        # Check symmetry for undirected graph
        assert distances[1][4] == distances[4][1]

    def test_weighted_graph(self):
        """Test all pairs distances on a weighted graph."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="A", target="C", weight=5.0)  # Longer direct route
        )

        distances = graph.all_pairs_shortest_path_length()
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")

        # A->C should go via B (1+2=3) rather than direct (5)
        assert distances[a_id][c_id] == 3.0
        assert distances[a_id][b_id] == 1.0
        assert distances[b_id][c_id] == 2.0

    def test_disconnected_graph(self):
        """Test all pairs distances with disconnected components."""
        graph = Graph()
        graph.add_path_graph(3)  # 1-2-3
        graph.add_node(4)  # Isolated node

        distances = graph.all_pairs_shortest_path_length()

        # Check connected component
        assert 1 in distances
        assert 2 in distances[1]
        assert 3 in distances[1]

        # Isolated node shouldn't be reachable from others
        assert 4 not in distances[1]
        assert 4 not in distances[2]
        assert 4 not in distances[3]

        # But isolated node should have entry for itself
        assert 4 in distances
        assert distances[4][4] == 0.0

    def test_star_graph(self):
        """Test all pairs distances on a star graph."""
        graph = Graph()
        graph.add_star_graph(5)  # Creates 5 nodes total

        distances = graph.all_pairs_shortest_path_length()
        node_ids = graph.get_node_ids()
        center = 1  # Center is node 1
        leaves = [n for n in node_ids if n != center]

        # Center to leaves is distance 1
        for leaf in leaves:
            assert distances[center][leaf] == 1.0

        # Leaf to leaf is distance 2 (through center)
        for i, leaf1 in enumerate(leaves):
            for leaf2 in leaves[i + 1 :]:
                assert distances[leaf1][leaf2] == 2.0

    def test_complete_graph(self):
        """Test all pairs distances on a complete graph."""
        graph = Graph()
        # Create a complete graph manually
        for i in range(1, 6):
            graph.add_node(i)
        for i in range(1, 6):
            for j in range(i + 1, 6):
                graph.add_edge(source=i, target=j)

        distances = graph.all_pairs_shortest_path_length()

        # All pairs should have distance 1 (direct connection)
        for i in range(1, 6):
            for j in range(1, 6):
                if i == j:
                    assert distances[i][j] == 0.0
                else:
                    assert distances[i][j] == 1.0

    def test_directed_graph(self):
        """Test all pairs distances on a directed graph."""
        graph = Graph(directed=True)
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_edge(source=1, target=2)
        graph.add_edge(source=2, target=3)

        distances = graph.all_pairs_shortest_path_length()

        # Forward paths exist
        assert distances[1][2] == 1.0
        assert distances[1][3] == 2.0
        assert distances[2][3] == 1.0

        # Backward paths don't exist
        assert 1 not in distances[2]
        assert 1 not in distances[3]
        assert 2 not in distances[3]

    def test_undirected_weight_is_symmetric(self):
        """An undirected edge has the same weight whichever way it is walked."""
        G_ = Graph(weighted=True)
        G_.add_node(label="A")
        G_.add_node(label="B")
        G_.add_edge("A", "B", weight=5.0)
        d = G_.all_pairs_shortest_path_length()
        a, b = G_.get_id_from_label("A"), G_.get_id_from_label("B")
        assert d[a][b] == 5.0
        assert d[b][a] == 5.0

    def test_undirected_weighted_matches_networkx(self):
        """Weighted undirected distances match NetworkX's Dijkstra in both directions."""
        edges = [("A", "B", 4.0), ("B", "C", 1.0), ("A", "C", 7.0), ("C", "D", 2.0), ("B", "D", 6.0)]
        G_ = Graph(weighted=True)
        for label in "ABCD":
            G_.add_node(label=label)
        for u, v, w in edges:
            G_.add_edge(u, v, weight=w)
        ref = dict(nx.all_pairs_dijkstra_path_length(nx.Graph([(u, v, {"weight": w}) for u, v, w in edges])))
        for s, row in G_.all_pairs_shortest_path_length().items():
            for t, dist in row.items():
                assert dist == pytest.approx(ref[G_.get_label_from_id(s)][G_.get_label_from_id(t)])


class TestAllPairsShortestPath:
    """Tests for all_pairs_shortest_path method."""

    def test_simple_path_graph(self):
        """Test all pairs paths on a simple path graph."""
        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        paths = graph.all_pairs_shortest_path()

        # Check some specific paths
        assert paths[1][4] == [1, 2, 3, 4]
        assert paths[2][4] == [2, 3, 4]
        assert paths[1][2] == [1, 2]

        # Check that paths include source and target
        for source in range(1, 5):
            for target in range(1, 5):
                path = paths[source][target]
                assert path[0] == source
                assert path[-1] == target

    def test_weighted_graph_chooses_shortest(self):
        """Test that weighted graph chooses shortest path by weight."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)  # A->B->D = 2.0
            .add_edge(source="A", target="C", weight=1.0)
            .add_edge(source="C", target="D", weight=5.0)  # A->C->D = 6.0
        )

        paths = graph.all_pairs_shortest_path()
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        d_id = graph.get_id_from_label("D")

        # Should take A->B->D route (shorter by weight)
        path = paths[a_id][d_id]
        assert b_id in path
        assert len(path) == 3

    def test_disconnected_components(self):
        """Test paths with disconnected components."""
        graph = Graph()
        graph.add_path_graph(3)  # 1-2-3
        graph.add_path_graph(3)  # Creates 4-5-6

        paths = graph.all_pairs_shortest_path()

        # Within first component
        assert paths[1][3] == [1, 2, 3]

        # Within second component
        assert paths[4][6] == [4, 5, 6]

        # Between components - no path
        assert 4 not in paths[1]
        assert 1 not in paths[4]

    def test_path_lengths_match_distances(self):
        """Test that path lengths match distance calculations."""
        graph = Graph()
        graph.add_cycle_graph(6)

        paths = graph.all_pairs_shortest_path()
        distances = graph.all_pairs_shortest_path_length()

        # For unweighted graphs, path length - 1 should equal distance
        for source in paths:
            for target in paths[source]:
                path_length = len(paths[source][target]) - 1
                assert path_length == distances[source][target]

    def test_self_loops(self):
        """Test that paths to self are just [self]."""
        graph = Graph()
        graph.add_star_graph(5)

        paths = graph.all_pairs_shortest_path()
        node_ids = graph.get_node_ids()

        for node in node_ids:
            assert paths[node][node] == [node]


class TestKShortestPaths:
    """Tests for k_shortest_paths method."""

    def test_simple_alternative_routes(self):
        """Test finding k alternative routes."""
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)  # Route 1: A->B->D (2.0)
            .add_edge(source="A", target="C", weight=1.5)
            .add_edge(source="C", target="D", weight=1.0)  # Route 2: A->C->D (2.5)
            .add_edge(source="A", target="D", weight=4.0)  # Route 3: A->D (4.0)
        )

        paths = graph.k_shortest_paths("A", "D", k=3)

        # Should find all 3 paths
        assert len(paths) == 3

        # Check they're sorted by distance
        assert paths[0][1] == 2.0  # A->B->D
        assert paths[1][1] == 2.5  # A->C->D
        assert paths[2][1] == 4.0  # A->D

        # Check paths are correct
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")
        d_id = graph.get_id_from_label("D")

        assert paths[0][0] == [a_id, b_id, d_id]
        assert paths[1][0] == [a_id, c_id, d_id]
        assert paths[2][0] == [a_id, d_id]

    def test_k_larger_than_available_paths(self):
        """Test when k is larger than number of available paths."""
        graph = Graph()
        graph.add_path_graph(3)  # Only one path from 1 to 3

        paths = graph.k_shortest_paths(1, 3, k=10)

        # Should return only the available path
        assert len(paths) == 1
        assert paths[0][0] == [1, 2, 3]

    def test_k_equals_zero(self):
        """Test k=0 returns empty list."""
        graph = Graph()
        graph.add_path_graph(3)

        paths = graph.k_shortest_paths(1, 3, k=0)
        assert len(paths) == 0

    def test_no_path_exists(self):
        """Test when no path exists between nodes."""
        graph = Graph()
        graph.add_node(1)
        graph.add_node(2)
        # No edges - disconnected

        paths = graph.k_shortest_paths(1, 2, k=5)
        assert len(paths) == 0

    def test_grid_multiple_paths(self):
        """Test k shortest paths on a grid with multiple routes."""
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)
        # Grid:
        # 1 - 2 - 3
        # |   |   |
        # 4 - 5 - 6
        # |   |   |
        # 7 - 8 - 9

        paths = graph.k_shortest_paths(1, 9, k=10)

        # Should find multiple paths
        assert len(paths) >= 2

        # All paths should have same length (4 hops) for shortest paths
        # in an unweighted grid
        assert paths[0][1] == 4.0

        # First path should be 4 edges
        assert len(paths[0][0]) == 5  # 5 nodes = 4 edges

    def test_directed_graph(self):
        """Test k shortest paths on directed graph."""
        graph = Graph(directed=True, weighted=True)
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_edge(source=1, target=2, weight=1.0)
        graph.add_edge(source=2, target=3, weight=1.0)
        graph.add_edge(source=1, target=3, weight=3.0)

        paths = graph.k_shortest_paths(1, 3, k=2)

        assert len(paths) == 2
        assert paths[0][1] == 2.0  # 1->2->3
        assert paths[1][1] == 3.0  # 1->3

    def test_single_path_k_equals_1(self):
        """Test k=1 returns only shortest path."""
        graph = (
            Graph(weighted=True)
            .add_node(1)
            .add_node(2)
            .add_node(3)
            .add_edge(source=1, target=2, weight=1.0)
            .add_edge(source=2, target=3, weight=1.0)
            .add_edge(source=1, target=3, weight=5.0)
        )

        paths = graph.k_shortest_paths(1, 3, k=1)

        assert len(paths) == 1
        assert paths[0][1] == 2.0
        assert paths[0][0] == [1, 2, 3]

    def test_paths_are_simple(self):
        """Test that returned paths are simple (no repeated nodes)."""
        graph = Graph(weighted=True)
        graph.add_cycle_graph(5)

        paths = graph.k_shortest_paths(1, 3, k=5)

        # Check no path has repeated nodes
        for path, _ in paths:
            assert len(path) == len(set(path)), f"Path {path} has repeated nodes"

    def test_unweighted_counts_hops(self):
        """Test that unweighted graphs count hops correctly."""
        graph = Graph()
        graph.add_path_graph(5)  # 1-2-3-4-5

        paths = graph.k_shortest_paths(1, 5, k=1)

        # Distance should be 4.0 (4 hops)
        assert paths[0][1] == 4.0
        assert len(paths[0][0]) == 5  # 5 nodes

    def test_invalid_nodes(self):
        """An unknown ID raises, like an unknown label (it used to return [])."""
        graph = Graph()
        graph.add_path_graph(3)

        with pytest.raises(ValueError, match="Node 999 does not exist"):
            graph.k_shortest_paths(1, 999, k=5)
        with pytest.raises(ValueError, match="Node 999 does not exist"):
            graph.k_shortest_paths(999, 1, k=5)


class TestAdvancedPathsIntegration:
    """Integration tests using all three methods together."""

    def test_consistency_across_methods(self):
        """Test that all three methods give consistent results."""
        graph = Graph(weighted=True)
        graph.add_cycle_graph(6)

        # Get results from all methods
        distances = graph.all_pairs_shortest_path_length()
        paths = graph.all_pairs_shortest_path()
        k_paths_1_to_4 = graph.k_shortest_paths(1, 4, k=1)

        # Check consistency
        assert distances[1][4] == k_paths_1_to_4[0][1]
        assert paths[1][4] == k_paths_1_to_4[0][0]

    def test_large_graph_performance(self):
        """Test that methods work on larger graphs."""
        graph = Graph()
        graph.add_grid_2d(rows=10, cols=10)  # 100 nodes

        # These should complete without timing out
        distances = graph.all_pairs_shortest_path_length()
        assert len(distances) == 100

        # Test a few paths
        paths = graph.k_shortest_paths(1, 100, k=3)
        assert len(paths) >= 1

    def test_weighted_vs_unweighted(self):
        """Test difference between weighted and unweighted graphs."""
        # Create same structure, one weighted one not
        unweighted = Graph()
        unweighted.add_path_graph(4)

        weighted = Graph(weighted=True)
        weighted.add_node(1)
        weighted.add_node(2)
        weighted.add_node(3)
        weighted.add_node(4)
        weighted.add_edge(source=1, target=2, weight=10.0)
        weighted.add_edge(source=2, target=3, weight=10.0)
        weighted.add_edge(source=3, target=4, weight=10.0)

        uw_dist = unweighted.all_pairs_shortest_path_length()
        w_dist = weighted.all_pairs_shortest_path_length()

        # Unweighted: distance is hop count
        assert uw_dist[1][4] == 3.0

        # Weighted: distance is sum of weights
        assert w_dist[1][4] == 30.0



class TestAllPairsNegativeWeights:
    """Dijkstra-based all-pairs methods reject negative weights instead of hanging."""

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize(
        "method", ["all_pairs_shortest_path_length", "all_pairs_shortest_path"]
    )
    def test_negative_weight_raises(self, method, directed):
        """The undirected case used to loop forever (a negative edge is a negative cycle)."""
        import subprocess
        import sys

        code = (
            "import edgecase as ec\n"
            f"g = ec.Graph(directed={directed}, weighted=True)\n"
            "for label in 'ABC':\n"
            "    g.add_node(label=label)\n"
            "g.add_edge('A', 'B', weight=1.0)\n"
            "g.add_edge('B', 'C', weight=-2.0)\n"
            "try:\n"
            f"    g.{method}()\n"
            "except ValueError as e:\n"
            "    print('ValueError:', e)\n"
        )
        # Run in a subprocess so a regression shows up as a timeout, not a hung test run
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        ).stdout
        assert "ValueError" in out
        assert "bellman_ford_shortest_path" in out

    def test_zero_weights_are_allowed(self):
        g = Graph(weighted=True)
        for label in "ABC":
            g.add_node(label=label)
        g.add_edge("A", "B", weight=0.0)
        g.add_edge("B", "C", weight=2.0)

        a, c = g.get_id_from_label("A"), g.get_id_from_label("C")
        assert g.all_pairs_shortest_path_length()[a][c] == pytest.approx(2.0)


class TestEdgeWeightLookupAfterMutation:
    """Weighted Dijkstra reads weights through a cached edge index; edits must refresh it."""

    @staticmethod
    def _dist(g, a, b):
        ids = {g.get_label_from_id(n): n for n in g.get_node_ids()}
        return g.all_pairs_shortest_path_length()[ids[a]].get(ids[b])

    def _triangle(self, directed=False):
        g = Graph(directed=directed, weighted=True)
        for label in "ABC":
            g.add_node(label=label)
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("A", "C", weight=5.0)
        return g

    def test_added_edge_is_seen(self):
        g = self._triangle()
        assert self._dist(g, "A", "C") == pytest.approx(2.0)  # builds the index
        g.add_node(label="D")
        g.add_edge("A", "D", weight=0.5)
        g.add_edge("D", "C", weight=0.5)
        assert self._dist(g, "A", "C") == pytest.approx(1.0)

    def test_cheaper_parallel_edge_added_later_is_used(self):
        """The cheapest parallel edge sets the step cost, even if added later (or reversed)."""
        g = self._triangle()
        assert self._dist(g, "A", "C") == pytest.approx(2.0)
        g.add_edge("C", "A", weight=0.1)  # undirected parallel to A-C, stored the other way
        assert self._dist(g, "A", "C") == pytest.approx(0.1)

    def test_deleted_edge_is_forgotten(self):
        g = self._triangle()
        assert self._dist(g, "A", "C") == pytest.approx(2.0)
        b_c = g.find_edge_ids("B", "C")[0]
        g.delete_edge_by_id(b_c)
        assert self._dist(g, "A", "C") == pytest.approx(5.0)

    def test_deleted_node_is_forgotten(self):
        g = self._triangle()
        assert self._dist(g, "A", "C") == pytest.approx(2.0)
        g.delete_node("B")
        assert self._dist(g, "A", "C") == pytest.approx(5.0)

    def test_weight_update_is_seen(self):
        g = self._triangle()
        assert self._dist(g, "A", "C") == pytest.approx(2.0)
        g.set_edge_attrs("weight", {g.find_edge_ids("A", "C")[0]: 0.5})
        assert self._dist(g, "A", "C") == pytest.approx(0.5)
        assert g.get_edge(g.find_edge_ids("A", "C")[0]).weight == pytest.approx(0.5)

    def test_reversed_edge_is_seen(self):
        g = self._triangle(directed=True)
        assert self._dist(g, "A", "C") == pytest.approx(2.0)
        g.select_edges_by_id([g.find_edge_ids("B", "C")[0]])
        g.rev_edge_dir_ws()
        assert self._dist(g, "A", "C") == pytest.approx(5.0)
        assert self._dist(g, "C", "B") == pytest.approx(1.0)

    def test_get_edge_by_id_after_deletions(self):
        g = self._triangle()
        ids = g.get_edge_ids()
        g.get_edge(ids[0])  # builds the index
        g.delete_edge_by_id(ids[0])
        assert not g.has_edge_id(ids[0])
        with pytest.raises(ValueError, match="does not exist"):
            g.get_edge(ids[0])
        for edge_id in ids[1:]:
            assert g.get_edge(edge_id).id == edge_id


class TestParallelEdgeWeights:
    """Weighted path searches use the cheapest of several parallel edges (as NetworkX does)."""

    @staticmethod
    def _parallel(directed=False):
        g = Graph(directed=directed, weighted=True)
        for label in "ABC":
            g.add_node(label=label)
        g.add_edge("A", "B", weight=5.0)  # first, expensive
        g.add_edge("A", "B", weight=1.0)  # later, cheap
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("A", "C", weight=3.0)
        return g

    @pytest.mark.parametrize("directed", [False, True])
    def test_dijkstra_family_uses_cheapest(self, directed):
        g = self._parallel(directed)
        a, c = g.get_id_from_label("A"), g.get_id_from_label("C")
        assert g.all_pairs_shortest_path_length()[a][c] == pytest.approx(2.0)
        assert g.all_pairs_shortest_path()[a][c] == [a, g.get_id_from_label("B"), c]
        assert g.k_shortest_paths("A", "C", 1)[0][1] == pytest.approx(2.0)

    def test_bellman_ford_uses_cheapest(self):
        g = self._parallel(directed=True)
        path = g.bellman_ford_shortest_path("A", "C", return_list=True)
        assert [g.get_label_from_id(n) for n in path] == ["A", "B", "C"]

    @pytest.mark.parametrize("directed", [False, True])
    def test_ada_star_uses_cheapest(self, directed):
        g = self._parallel(directed)
        path = g.ada_star_search("A", "C")
        assert [g.get_label_from_id(n) for n in path] == ["A", "B", "C"]

    def test_steiner_tree_reports_cheapest_edge(self):
        g = self._parallel()
        edges = g.steiner_tree(["A", "C"], return_subgraph=False)
        assert sum(w for *_, w in edges) == pytest.approx(2.0)

    def test_matches_networkx_multigraph(self):
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(42)
        for seed in range(40):
            directed = seed % 2 == 0
            n = rng.randint(4, 10)
            g = Graph(directed=directed, weighted=True)
            multi = nx.MultiDiGraph() if directed else nx.MultiGraph()
            for i in range(n):
                g.add_node(label=str(i))
                multi.add_node(i)
            for _ in range(rng.randint(n, 3 * n)):
                u, v = rng.sample(range(n), 2)
                for _ in range(rng.choice([1, 2, 3])):
                    w = round(rng.uniform(0.5, 5), 1)
                    a, b = (u, v) if (directed or rng.random() < 0.5) else (v, u)
                    g.add_edge(str(a), str(b), weight=w)
                    multi.add_edge(a, b, weight=w)

            got = g.all_pairs_shortest_path_length()
            for u, dists in nx.all_pairs_dijkstra_path_length(multi):
                for v, d in dists.items():
                    uid, vid = g.get_id_from_label(str(u)), g.get_id_from_label(str(v))
                    assert got[uid][vid] == pytest.approx(d)

    def test_label_propagation_sums_parallel_edges(self):
        """Each parallel edge votes with its own weight: two 1.0 edges outweigh one 1.5 edge."""
        g = Graph(weighted=True)
        for label in ["X", "L0", "L1"]:
            g.add_node(label=label)
        g.add_edge("X", "L0", weight=1.0)
        g.add_edge("X", "L0", weight=1.0)
        g.add_edge("X", "L1", weight=1.5)
        seeds = {g.get_id_from_label("L0"): 0, g.get_id_from_label("L1"): 1}

        result = g.label_propagation(seeds, return_dict=True)
        assert result[g.get_id_from_label("X")] == 0


# Shortest-path queries (shortest_path_length, has_path, all_shortest_paths, single- and
# multi-source paths and lengths), validated against NetworkX on random graphs.


def _random_cases():
    """(networkx graph, edgecase graph, weight key) over many random configurations."""
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            for weighted in (False, True):
                G = nx.gnp_random_graph(14, 0.2, seed=seed, directed=directed)
                for u, v in G.edges:
                    G[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
                yield G, _build(G, weighted), ("weight" if weighted else None)


def _two_triangles() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _lab(g: ec.Graph, nid: int) -> int:
    return int(g.get_label_from_id(nid))


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _weighted_cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G = nx.gnp_random_graph(15, 0.22, seed=seed, directed=directed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.choice([0.5, 1, 1.5, 2, 3])
            yield G, _build_in_order(G, True)


def _triangle_detour() -> ec.Graph:
    """A-C is heavy, so the cheapest A-C route detours through B."""
    return (
        ec.Graph(weighted=True)
        .add_nodes(["A", "B", "C"])
        .add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0)])
    )


def _build_in_order(G: nx.Graph, weighted: bool) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


class TestPathQueriesAgainstNetworkX:
    def test_single_source_lengths(self):
        for G, g, w in _random_cases():
            s = 0
            ours = {
                _lab(g, k): v
                for k, v in g.single_source_shortest_path_length("0", return_dict=True).items()
            }
            ref = (
                nx.single_source_dijkstra_path_length(G, s, weight=w)
                if w
                else nx.single_source_shortest_path_length(G, s)
            )
            assert ours.keys() == ref.keys()
            for k, v in ref.items():
                assert ours[k] == pytest.approx(v)

    def test_cutoff(self):
        for G, g, w in _random_cases():
            ours = set(
                _lab(g, k)
                for k in g.single_source_shortest_path_length("0", cutoff=2, return_dict=True)
            )
            ref = (
                nx.single_source_dijkstra_path_length(G, 0, cutoff=2, weight=w)
                if w
                else nx.single_source_shortest_path_length(G, 0, cutoff=2)
            )
            assert ours == set(ref)

    def test_single_source_paths_are_shortest(self):
        for G, g, w in _random_cases():
            ref = nx.single_source_dijkstra_path_length(G, 0, weight=w or (lambda u, v, d: 1))
            paths = g.single_source_shortest_path("0")
            assert {_lab(g, t) for t in paths} == set(ref)
            for t, p in paths.items():
                p = [_lab(g, x) for x in p]
                assert p[0] == 0 and p[-1] == _lab(g, t)
                cost = sum(G[a][b]["weight"] if w else 1 for a, b in zip(p, p[1:]))
                assert cost == pytest.approx(ref[p[-1]])

    def test_multi_source(self):
        for G, g, w in _random_cases():
            srcs = [0, 1, 2]
            ref = nx.multi_source_dijkstra_path_length(
                G, set(srcs), weight=w or (lambda u, v, d: 1)
            )
            ours = g.multi_source_shortest_path_length([str(s) for s in srcs], return_dict=True)
            assert {_lab(g, k): v for k, v in ours.items()} == pytest.approx(ref)
            for t, p in g.multi_source_shortest_path([str(s) for s in srcs]).items():
                p = [_lab(g, x) for x in p]
                assert p[0] in srcs and p[-1] == _lab(g, t)
                cost = sum(G[a][b]["weight"] if w else 1 for a, b in zip(p, p[1:]))
                assert cost == pytest.approx(ref[p[-1]])

    def test_pairwise_length_has_path_and_all_shortest_paths(self):
        for G, g, w in _random_cases():
            for a, b in itertools.islice(itertools.permutations(G.nodes, 2), 0, 80, 9):
                reachable = nx.has_path(G, a, b)
                assert g.has_path(str(a), str(b)) is reachable
                length = g.shortest_path_length(str(a), str(b))
                paths = g.all_shortest_paths(str(a), str(b))
                if not reachable:
                    assert length is None and paths == []
                    continue
                assert length == pytest.approx(nx.shortest_path_length(G, a, b, weight=w))
                ours = sorted([_lab(g, x) for x in p] for p in paths)
                assert ours == sorted(nx.all_shortest_paths(G, a, b, weight=w))


class TestShortestPathQueries:
    def test_unweighted_lengths_are_ints(self):
        g = _two_triangles()
        assert g.shortest_path_length("A", "F") == 3
        assert isinstance(g.shortest_path_length("A", "F"), int)
        dist = g.single_source_shortest_path_length("A", return_dict=True)
        assert all(isinstance(v, int) for v in dist.values())

    def test_weighted_lengths_are_floats(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=2.5)
        assert g.shortest_path_length("A", "B") == 2.5

    def test_agrees_with_shortest_path(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(list("ABCDF"))
        g.add_edges(
            [("A", "B", 1.0), ("B", "C", 2.0), ("C", "F", 1.0), ("A", "D", 3.0), ("D", "F", 3.0)]
        )
        path = g.shortest_path("A", "F", return_list=True)
        assert g.shortest_path_length("A", "F") == 4.0
        assert path in g.all_shortest_paths("A", "F")

    def test_self_distance_and_self_path(self):
        g = _two_triangles()
        assert g.shortest_path_length("A", "A") == 0
        assert g.has_path("A", "A")
        a = g.get_id_from_label("A")
        assert g.all_shortest_paths("A", "A") == [[a]]
        assert g.single_source_shortest_path("A")[a] == [a]

    def test_directed_has_no_way_back(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        assert g.has_path("A", "B") and not g.has_path("B", "A")
        assert g.shortest_path_length("B", "A") is None

    def test_grid_has_binomial_number_of_paths_and_max_paths(self):
        grid = ec.Graph().add_grid_2d(rows=4, cols=4)
        assert len(grid.all_shortest_paths(1, 16)) == math.comb(6, 3)
        assert len(grid.all_shortest_paths(1, 16, max_paths=5)) == 5
        with pytest.raises(ValueError, match="max_paths"):
            grid.all_shortest_paths(1, 16, max_paths=0)

    def test_negative_weights_rejected(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            g.shortest_path_length("A", "B")

    def test_errors(self):
        g = _two_triangles()
        with pytest.raises(ValueError):
            g.has_path("A", "nope")
        with pytest.raises(ValueError, match="source"):
            g.multi_source_shortest_path([])

    def test_length_attributes_stored_with_nulls(self):
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        g.single_source_shortest_path_length("A", attr_name="d")
        df = g.get_node_df().sort("label")
        assert df["d"].to_list() == [0, 1, None]
        g.multi_source_shortest_path_length(["A", "C"])
        df = g.get_node_df().sort("label")
        assert df["multi_source_shortest_path_length"].to_list() == [0, 1, 0]

    def test_path_queries_highlight_results(self):
        g = _two_triangles()
        g.all_shortest_paths("A", "F")
        step = g.steps[-1]
        assert step.kind == KIND_QUERY and step.method == "all_shortest_paths"
        assert step.delta.nodes_selected == ["A", "C", "D", "F"]
        assert step.delta.edges_selected == [("A", "C"), ("C", "D"), ("D", "F")]
        assert step.narration == "1 shortest path from A to F."

        g.has_path("A", "F")
        assert g.steps[-1].narration == "F is reachable from A."
        g.shortest_path_length("A", "F")
        assert g.steps[-1].result == 3
        assert g.steps[-1].delta.nodes_selected == ["A", "F"]

    def test_distance_methods_record_metric_heat(self):
        g = _two_triangles().single_source_shortest_path_length("A")
        step = g.steps[-1]
        assert step.kind == KIND_METRIC
        assert step.delta.metric_values == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3, "F": 3}
        assert "farthest: E (3)" in step.narration

    def test_multi_source_path_records_catchments(self):
        g = _two_triangles()
        g.multi_source_shortest_path(["A", "F"])
        d = g.steps[-1].delta
        assert d.node_groups == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        assert d.nodes_selected == [] and len(d.edges_selected) == 4

    def test_return_dict_records_no_step(self):
        g = _two_triangles()
        n = len(g.steps)
        g.single_source_shortest_path_length("A", return_dict=True)
        g.multi_source_shortest_path_length(["A"], return_dict=True)
        assert len(g.steps) == n


# Bidirectional Dijkstra, validated against NetworkX.


class TestBidirectionalDijkstra:
    def test_matches_networkx(self):
        for G, g in _weighted_cases():
            for weighted in (True, False):
                gg = g if weighted else _build_in_order(G, False)
                for a, b in itertools.islice(itertools.permutations(G.nodes, 2), 0, 200, 7):
                    res = gg.bidirectional_dijkstra(str(a), str(b))
                    if not nx.has_path(G, a, b):
                        assert res is None
                        continue
                    length, path = res
                    ref = nx.shortest_path_length(G, a, b, weight="weight" if weighted else None)
                    labels = [int(gg.get_label_from_id(n)) for n in path]
                    cost = sum(
                        G[x][y]["weight"] if weighted else 1 for x, y in zip(labels, labels[1:])
                    )
                    assert length == pytest.approx(ref) and cost == pytest.approx(ref)
                    assert labels[0] == a and labels[-1] == b

    def test_types_and_trivial_cases(self):
        g = _triangle_detour()
        length, path = g.bidirectional_dijkstra("A", "C")
        assert length == 2.0 and [g.get_label_from_id(n) for n in path] == ["A", "B", "C"]
        hops = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        assert hops.bidirectional_dijkstra("A", "B") == (1, [1, 2])
        assert isinstance(hops.bidirectional_dijkstra("A", "B")[0], int)
        assert hops.bidirectional_dijkstra("A", "A") == (0, [1])
        lonely = ec.Graph().add_nodes(["A", "B"])
        assert lonely.bidirectional_dijkstra("A", "B") is None

    def test_bidirectional_dijkstra_highlights_path(self):
        g = _triangle_detour()
        g.bidirectional_dijkstra("A", "C")
        step = g.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.delta.nodes_selected == ["A", "B", "C"]
        assert step.delta.edges_selected == [("A", "B"), ("B", "C")]
        assert "length 2" in step.narration


# Shortest-path extras (A*, predecessors, single-target, all shortest paths, Floyd-Warshall,
# Johnson, Yen's simple paths, edge paths), validated against NetworkX.


def _extras_cases(count=120):
    for seed in range(count):
        rnd = random.Random(seed)
        directed = seed % 2 == 0
        kind = seed % 4  # unweighted, integer, float, possibly negative (directed)
        G = nx.gnp_random_graph(rnd.randint(1, 9), 0.35, seed=seed, directed=directed)
        for u, v in G.edges:
            G[u][v]["weight"] = [1, rnd.randint(1, 3), rnd.uniform(0.5, 3), rnd.randint(-1, 4)][kind]
        if kind == 3 and (not directed or nx.negative_edge_cycle(G)):
            for u, v in G.edges:
                G[u][v]["weight"] = abs(G[u][v]["weight"]) + 1
        weighted = kind != 0
        g = ec.Graph(directed=directed, weighted=weighted).add_nodes([str(n) for n in G])
        for u, v, d in G.edges(data=True):
            g.add_edge(str(u), str(v), weight=d["weight"])
        yield G, g, weighted


def _relabel(g, mapping):
    return {int(g.get_label_from_id(k)): v for k, v in mapping.items()}


class TestPathExtras:
    def test_predecessor_and_single_target(self):
        for G, g, weighted in _extras_cases():
            negative = any(w < 0 for *_, w in G.edges(data="weight"))
            for s in list(G)[:3]:
                pred, dist = g.predecessor(str(s), return_distances=True)
                if not weighted:
                    rp, rd = nx.predecessor(G, s), nx.single_source_shortest_path_length(G, s)
                elif negative:
                    rp, rd = nx.bellman_ford_predecessor_and_distance(G, s)
                else:
                    rp, rd = nx.dijkstra_predecessor_and_distance(G, s)
                ours = {k: sorted(int(g.get_label_from_id(p)) for p in v) for k, v in _relabel(g, pred).items()}
                assert ours == {k: sorted(v) for k, v in rp.items()}
                assert _relabel(g, dist) == pytest.approx(dict(rd))
                to_s = _relabel(g, g.single_target_shortest_path_length(str(s)))
                rev = G.reverse() if G.is_directed() else G
                ref = (
                    nx.single_source_bellman_ford_path_length(rev, s)
                    if weighted
                    else nx.single_source_shortest_path_length(rev, s)
                )
                assert to_s == pytest.approx(dict(ref))
                paths = g.single_target_shortest_path(str(s))
                assert set(paths) == set(g.single_target_shortest_path_length(str(s)))

    def test_all_pairs_negative_weights(self):
        for G, g, weighted in _extras_cases():
            w = "weight" if weighted else None
            dist = g.floyd_warshall()
            ref = nx.floyd_warshall(G, weight=w)
            for s, row in _relabel(g, dist).items():
                expected = {t: d for t, d in ref[s].items() if not math.isinf(d)}
                assert {int(g.get_label_from_id(t)): d for t, d in row.items()} == pytest.approx(expected)
            if weighted:
                cost = lambda p: sum(G[a][b]["weight"] for a, b in zip(p, p[1:]))  # noqa: E731
                ref = nx.johnson(G, weight="weight")
                for s, row in g.johnson().items():
                    for t, path in row.items():
                        path = [int(g.get_label_from_id(x)) for x in path]
                        expected = ref[int(g.get_label_from_id(s))][int(g.get_label_from_id(t))]
                        assert cost(path) == pytest.approx(cost(expected))

    def test_all_shortest_paths_and_simple_paths(self):
        for G, g, weighted in _extras_cases(80):
            if len(G) < 2 or any(w < 0 for *_, w in G.edges(data="weight")):
                continue
            w = "weight" if weighted else None
            a, b = list(G)[0], list(G)[-1]
            ours = g.single_source_all_shortest_paths(str(a))
            ref = dict(nx.single_source_all_shortest_paths(G, a, weight=w))
            for t, paths in ref.items():
                mine = ours[g.get_id_from_label(str(t))]
                assert sorted(tuple(int(g.get_label_from_id(x)) for x in p) for p in mine) == sorted(
                    map(tuple, paths)
                )
            try:
                expected = list(nx.shortest_simple_paths(G, a, b, weight=w))
            except nx.NetworkXNoPath:
                expected = []
            found = [[int(g.get_label_from_id(x)) for x in p] for p in g.shortest_simple_paths(str(a), str(b))]
            assert sorted(map(tuple, found)) == sorted(map(tuple, expected))
            cost = (lambda p: sum(G[x][y]["weight"] for x, y in zip(p, p[1:]))) if weighted else len  # noqa: E731
            assert [cost(p) for p in found] == pytest.approx([cost(p) for p in expected])
            assert len(g.all_simple_edge_paths(str(a), str(b))) == len(
                list(nx.all_simple_edge_paths(G, a, b))
            )

    def test_all_shortest_paths_with_zero_weights(self):
        # Zero-weight edges make predecessors mutual (a <- b and b <- a); this used to hang
        g = ec.Graph(weighted=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b", {"weight": 0.0}), ("b", "c", {"weight": 1.0})])
        lab = g.get_label_from_id
        paths = {lab(t): [[lab(x) for x in p] for p in ps]
                 for t, ps in g.single_source_all_shortest_paths("c").items()}  # fmt: skip
        assert paths == {"a": [["c", "b", "a"]], "b": [["c", "b"]], "c": [["c"]]}
        rnd = random.Random(5)
        for directed in (False, True):
            for _ in range(30):
                G = nx.gnp_random_graph(7, 0.4, seed=rnd.randrange(10**6), directed=directed)
                for u, v in G.edges:
                    G[u][v]["weight"] = float(rnd.choice([0, 0, 1, 2]))
                g = ec.Graph(directed=directed, weighted=True).add_nodes([str(n) for n in G])
                g.add_edges([(str(u), str(v), {"weight": w}) for u, v, w in G.edges(data="weight")])
                lab = g.get_label_from_id
                ours = g.all_pairs_all_shortest_paths()
                for s, row in nx.all_pairs_all_shortest_paths(G, weight="weight"):
                    mine = ours[g.get_id_from_label(str(s))]
                    for t, ps in row.items():
                        got = mine[g.get_id_from_label(str(t))]
                        # NetworkX can repeat a path when the source sits on a zero-weight cycle
                        assert sorted(tuple(int(lab(x)) for x in p) for p in got) == sorted(
                            set(map(tuple, ps))
                        )

    def test_astar(self):
        G = nx.grid_2d_graph(7, 7)
        rnd = random.Random(1)
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.uniform(1, 2)
        G = nx.convert_node_labels_to_integers(G, label_attribute="pos")
        g = ec.Graph(weighted=True).add_nodes([str(n) for n in G])
        for u, v, d in G.edges(data=True):
            g.add_edge(str(u), str(v), weight=d["weight"])
        pos = {g.get_id_from_label(str(n)): G.nodes[n]["pos"] for n in G}

        def manhattan(u, t):
            return abs(pos[u][0] - pos[t][0]) + abs(pos[u][1] - pos[t][1])

        for s, t in [(0, 48), (5, 30), (12, 12)]:
            expected = nx.astar_path_length(G, s, t, weight="weight")
            assert g.astar_path_length(str(s), str(t), heuristic=manhattan) == pytest.approx(expected)
            assert g.astar_path_length(str(s), str(t)) == pytest.approx(expected)
        lonely = ec.Graph().add_nodes(["A", "B"])
        assert lonely.astar_path("A", "B") is None
        assert ec.Graph().add_path_graph(4).astar_path_length(1, 4) == 3

    def test_examples_and_errors(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 4.0), ("A", "C", 2.0), ("C", "B", -1.0)])
        a, b = g.get_id_from_label("A"), g.get_id_from_label("B")
        dist, pred = g.floyd_warshall(return_predecessors=True)
        assert dist[a][b] == 1.0 and pred[a][b] == g.get_id_from_label("C")
        assert g.johnson()[a][b] == [a, g.get_id_from_label("C"), b]
        with pytest.raises(ValueError, match="negative"):
            g.astar_path("A", "B")
        cyc = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B"])
        cyc.add_edges([("A", "B", 1.0), ("B", "A", -2.0)])
        for method in (cyc.floyd_warshall, cyc.johnson):
            with pytest.raises(ValueError, match="negative cycle"):
                method()
        multi = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("A", "B"), ("B", "C")])
        assert len(multi.all_simple_edge_paths("A", "C")) == 2
        assert multi.all_simple_edge_paths("A", "C", cutoff=1) == []
        assert list(itertools.islice(multi.shortest_simple_paths("A", "C"), 5)) == [[1, 2, 3]]
        with pytest.raises(ValueError, match="different"):
            multi.shortest_simple_paths("A", "A")

    def test_steps(self):
        square = ec.Graph().add_nodes(["A", "B", "C", "D"])
        square.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        square.predecessor("A")
        step = square.steps[-1]
        assert step.kind == KIND_METRIC and step.delta.metric_values["C"] == 2.0
        square.astar_path("A", "C")
        assert square.steps[-1].kind == KIND_QUERY
        assert square.steps[-1].narration == "A* path of length 2 from A to C."
