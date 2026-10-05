import itertools
import random

import edgecase as ec
import networkx as nx
import pytest


class TestNodeConnectivity:
    """Test node connectivity functions."""

    def test_node_connectivity_path_graph(self):
        """Path graph has node connectivity 1."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Node connectivity between endpoints
        conn = G._graph.node_connectivity_between(1, 5)
        assert conn == 1  # Removing any intermediate node disconnects path

    def test_node_connectivity_cycle_graph(self):
        """Cycle graph has node connectivity 2."""
        G = ec.Graph()
        G.add_cycle_graph(5)

        # Any two nodes in a cycle
        conn = G._graph.node_connectivity_between(1, 3)
        assert conn == 2  # Need to remove 2 nodes to disconnect

    def test_node_connectivity_complete_graph(self):
        """Complete graph K_n has node connectivity n-1."""
        G = ec.Graph()
        # Build K_5 (complete graph with 5 nodes)
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        conn = G._graph.node_connectivity()
        assert conn == 4  # K_5 has connectivity 4

    def test_node_connectivity_star_graph(self):
        """Star graph has node connectivity 1."""
        G = ec.Graph()
        G.add_star_graph(5)  # 1 central node + 5 peripheral

        conn = G._graph.node_connectivity()
        assert conn == 1  # Removing center disconnects everything

    def test_node_connectivity_disconnected(self):
        """Disconnected graph has connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        # C is isolated

        conn = G._graph.node_connectivity()
        assert conn == 0


class TestEdgeConnectivity:
    """Test edge connectivity functions."""

    def test_edge_connectivity_path(self):
        """Path graph has edge connectivity 1."""
        G = ec.Graph()
        G.add_path_graph(5)

        conn = G._graph.edge_connectivity()
        assert conn == 1

    def test_edge_connectivity_cycle(self):
        """Cycle graph has edge connectivity 2."""
        G = ec.Graph()
        G.add_cycle_graph(5)

        conn = G._graph.edge_connectivity()
        assert conn == 2

    def test_edge_connectivity_complete(self):
        """Complete graph K_n has high edge connectivity."""
        G = ec.Graph()
        # Build K_5
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        conn = G._graph.edge_connectivity()
        # For K_5, edge connectivity equals vertex connectivity = 4
        assert conn == 4

    def test_edge_connectivity_star(self):
        """Star graph has edge connectivity 1."""
        G = ec.Graph()
        G.add_star_graph(5)

        conn = G._graph.edge_connectivity()
        assert conn == 1

    def test_edge_connectivity_disconnected(self):
        """Disconnected graph has edge connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")

        conn = G._graph.edge_connectivity()
        assert conn == 0


class TestConnectivityEdgeCases:
    """Test edge cases for connectivity."""

    def test_empty_graph(self):
        """Empty graph has connectivity 0."""
        G = ec.Graph()

        assert G._graph.node_connectivity() == 0
        assert G._graph.edge_connectivity() == 0

    def test_single_node(self):
        """Single node graph has connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")

        assert G._graph.node_connectivity() == 0
        assert G._graph.edge_connectivity() == 0

    def test_two_nodes_connected(self):
        """Two connected nodes have connectivity 1."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_edge("A", "B")

        # Node connectivity between same nodes should error
        with pytest.raises(Exception):
            G._graph.node_connectivity_between(1, 1)

        # Global connectivity
        assert G._graph.node_connectivity() == 1
        assert G._graph.edge_connectivity() == 1


class TestNodeConnectivityBetween:
    """Test get_node_connectivity_between() method with IDs and labels."""

    def test_path_graph_with_labels(self):
        """Test node connectivity using labels in path graph."""
        G = ec.Graph()
        G.add_path_graph(5)  # Creates nodes with labels "1"-"5"

        # Endpoints have connectivity 1 (any middle node disconnects them)
        conn = G.get_node_connectivity_between("1", "5")
        assert conn == 1

    def test_path_graph_with_ids(self):
        """Test node connectivity using node IDs in path graph."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Get node IDs
        node1_id = G.get_id_from_label("1")
        node5_id = G.get_id_from_label("5")

        # Use node IDs directly
        conn = G.get_node_connectivity_between(node1_id, node5_id)
        assert conn == 1

    def test_cycle_graph_with_labels(self):
        """Test node connectivity in cycle graph using labels."""
        G = ec.Graph()
        G.add_cycle_graph(6)  # Creates nodes with labels "1"-"6"

        # Any two nodes in a cycle have connectivity 2
        conn = G.get_node_connectivity_between("1", "4")
        assert conn == 2

    def test_cycle_graph_with_ids(self):
        """Test node connectivity in cycle graph using node IDs."""
        G = ec.Graph()
        G.add_cycle_graph(6)

        node1_id = G.get_id_from_label("1")
        node4_id = G.get_id_from_label("4")

        conn = G.get_node_connectivity_between(node1_id, node4_id)
        assert conn == 2

    def test_complete_graph_with_labels(self):
        """Test node connectivity in complete graph using labels."""
        G = ec.Graph()
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        # Any two nodes in K_5 have connectivity 4 (need to remove 4 nodes to disconnect)
        conn = G.get_node_connectivity_between("1", "5")
        assert conn == 4

    def test_complete_graph_with_ids(self):
        """Test node connectivity in complete graph using node IDs."""
        G = ec.Graph()
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        node1_id = G.get_id_from_label("1")
        node5_id = G.get_id_from_label("5")

        conn = G.get_node_connectivity_between(node1_id, node5_id)
        assert conn == 4

    def test_mixed_id_and_label(self):
        """Test using both ID and label in same call."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Mix ID and label
        node1_id = G.get_id_from_label("1")
        conn = G.get_node_connectivity_between(node1_id, "5")
        assert conn == 1

        # Reverse order
        node5_id = G.get_id_from_label("5")
        conn2 = G.get_node_connectivity_between("1", node5_id)
        assert conn2 == 1

    def test_star_graph_with_labels(self):
        """Test node connectivity in star graph using labels."""
        G = ec.Graph()
        G.add_star_graph(5)  # Creates center (label "1") + 5 leaves

        # Center to any leaf has connectivity 1
        conn = G.get_node_connectivity_between("1", "2")
        assert conn == 1

    def test_nonexistent_label_error(self):
        """Test error when using non-existent label."""
        G = ec.Graph()
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="not found"):
            G.get_node_connectivity_between("1", "nonexistent")

    def test_nonexistent_id_error(self):
        """Test error when using non-existent node ID."""
        G = ec.Graph()
        G.add_path_graph(3)

        # Use a node ID that doesn't exist
        with pytest.raises(Exception):
            G.get_node_connectivity_between(999, 1000)

    def test_custom_labels(self):
        """Test with custom node labels."""
        G = ec.Graph()
        G.add_node(label="Alice")
        G.add_node(label="Bob")
        G.add_node(label="Charlie")
        G.add_node(label="Dave")
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Charlie")
        G.add_edge("Charlie", "Dave")

        # Path: Alice-Bob-Charlie-Dave
        conn = G.get_node_connectivity_between("Alice", "Dave")
        assert conn == 1  # Removing Bob or Charlie disconnects them

    @staticmethod
    def _two_diamonds():
        """Two diamonds sharing the hub D (node connectivity 1)."""
        g = ec.Graph()
        for label in "ABCDEFG":
            g.add_node(label=label)
        for u, v in [
            ("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"),
            ("D", "E"), ("D", "F"), ("E", "F"), ("E", "G"), ("F", "G"),
        ]:
            g.add_edge(u, v)
        return g

    @pytest.mark.parametrize("pair", [("A", "G"), ("B", "F"), ("B", "E"), ("A", "E")])
    def test_split_node_ids_do_not_collide(self, pair):
        """Node IDs 2v / 2v+1 used for node splitting never collide with source/target IDs."""
        g = self._two_diamonds()
        i = g.get_id_from_label
        assert g.get_node_connectivity_between(i(pair[0]), i(pair[1])) == 1

    def test_matches_networkx(self):
        """Pairwise node connectivity matches NetworkX for non-adjacent pairs."""
        g = self._two_diamonds()
        ref = nx.Graph([
            ("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"),
            ("D", "E"), ("D", "F"), ("E", "F"), ("E", "G"), ("F", "G"),
        ])
        i = g.get_id_from_label
        for s, t in [("A", "F"), ("B", "G"), ("C", "E")]:
            assert g.get_node_connectivity_between(i(s), i(t)) == nx.node_connectivity(ref, s, t)


# Edge- and node-disjoint paths, validated against NetworkX.


def _bowtie() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(list("ABCDE"))
        .add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "C")])
    )


def _random(seed: int, directed: bool, n: int = 12):
    rnd = random.Random(seed)
    G0 = nx.gnp_random_graph(n, rnd.choice([0.2, 0.3, 0.45]), seed=seed, directed=directed)
    G = (nx.DiGraph if directed else nx.Graph)()
    G.add_nodes_from(range(n))
    for u, v in G0.edges:
        G.add_edge(u, v, capacity=rnd.choice([1, 2, 3, 5]))
    g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(n)])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), capacity=d["capacity"])
    return G, g


def _id(g, v):
    return g.get_id_from_label(str(v))


def _labs(g, ids):
    return [int(g.get_label_from_id(i)) for i in ids]


class TestDisjointPaths:
    @pytest.mark.parametrize("directed", [False, True])
    def test_counts_match_networkx_and_paths_are_disjoint(self, directed):
        for seed in range(20):
            G, g = _random(seed, directed)
            for s, t in itertools.islice(itertools.permutations(range(12), 2), 0, 132, 19):
                reachable = nx.has_path(G, s, t)
                edge_paths = g.edge_disjoint_paths(_id(g, s), _id(g, t))
                node_paths = g.node_disjoint_paths(_id(g, s), _id(g, t))
                ref_e = len(list(nx.edge_disjoint_paths(G, s, t))) if reachable else 0
                ref_n = len(list(nx.node_disjoint_paths(G, s, t))) if reachable else 0
                assert len(edge_paths) == ref_e and len(node_paths) == ref_n
                used = []
                for p in edge_paths:
                    labs = _labs(g, p)
                    assert labs[0] == s and labs[-1] == t
                    for a, b in zip(labs, labs[1:]):
                        assert G.has_edge(a, b)
                        used.append((a, b) if directed else tuple(sorted((a, b))))
                assert len(used) == len(set(used))
                inner = [v for p in node_paths for v in _labs(g, p)[1:-1]]
                assert len(inner) == len(set(inner))

    def test_bowtie_contrast_and_menger(self):
        g = _bowtie()
        assert len(g.edge_disjoint_paths("A", "E")) == g.get_min_cut_between("A", "E") == 2
        assert len(g.node_disjoint_paths("A", "E")) == 1
        assert g.get_node_connectivity_between("A", "E") == 1

    def test_cube_and_max_paths(self):
        cube = ec.Graph().add_nodes([str(i) for i in range(8)])
        cube.add_edges(
            [
                (str(a), str(b))
                for a in range(8)
                for b in range(a + 1, 8)
                if bin(a ^ b).count("1") == 1
            ]
        )
        paths = cube.node_disjoint_paths("0", "7")
        assert len(paths) == 3 and all(len(p) == 4 for p in paths)
        assert len(cube.edge_disjoint_paths("0", "7", max_paths=2)) == 2
        with pytest.raises(ValueError, match="max_paths"):
            cube.node_disjoint_paths("0", "7", max_paths=0)

    def test_unreachable_and_adjacent(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"]).add_edge("A", "B")
        assert g.edge_disjoint_paths("B", "A") == [] and g.node_disjoint_paths("B", "A") == []
        assert g.node_disjoint_paths("A", "B") == [[1, 2]]

    def test_steps(self):
        b = _bowtie()
        b.edge_disjoint_paths("A", "E")
        assert b.steps[-1].narration == "2 edge-disjoint paths from A to E."
        assert set(b.steps[-1].delta.nodes_selected) == set("ABCDE")
