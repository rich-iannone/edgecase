import itertools
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_QUERY
from networkx.algorithms.connectivity import local_edge_connectivity


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
    """Test node_connectivity_between() method with IDs and labels."""

    def test_path_graph_with_labels(self):
        """Test node connectivity using labels in path graph."""
        G = ec.Graph()
        G.add_path_graph(5)  # Creates nodes with labels "1"-"5"

        # Endpoints have connectivity 1 (any middle node disconnects them)
        conn = G.node_connectivity_between("1", "5")
        assert conn == 1

    def test_path_graph_with_ids(self):
        """Test node connectivity using node IDs in path graph."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Get node IDs
        node1_id = G.get_id_from_label("1")
        node5_id = G.get_id_from_label("5")

        # Use node IDs directly
        conn = G.node_connectivity_between(node1_id, node5_id)
        assert conn == 1

    def test_cycle_graph_with_labels(self):
        """Test node connectivity in cycle graph using labels."""
        G = ec.Graph()
        G.add_cycle_graph(6)  # Creates nodes with labels "1"-"6"

        # Any two nodes in a cycle have connectivity 2
        conn = G.node_connectivity_between("1", "4")
        assert conn == 2

    def test_cycle_graph_with_ids(self):
        """Test node connectivity in cycle graph using node IDs."""
        G = ec.Graph()
        G.add_cycle_graph(6)

        node1_id = G.get_id_from_label("1")
        node4_id = G.get_id_from_label("4")

        conn = G.node_connectivity_between(node1_id, node4_id)
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
        conn = G.node_connectivity_between("1", "5")
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

        conn = G.node_connectivity_between(node1_id, node5_id)
        assert conn == 4

    def test_mixed_id_and_label(self):
        """Test using both ID and label in same call."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Mix ID and label
        node1_id = G.get_id_from_label("1")
        conn = G.node_connectivity_between(node1_id, "5")
        assert conn == 1

        # Reverse order
        node5_id = G.get_id_from_label("5")
        conn2 = G.node_connectivity_between("1", node5_id)
        assert conn2 == 1

    def test_star_graph_with_labels(self):
        """Test node connectivity in star graph using labels."""
        G = ec.Graph()
        G.add_star_graph(5)  # Creates center (label "1") + 5 leaves

        # Center to any leaf has connectivity 1
        conn = G.node_connectivity_between("1", "2")
        assert conn == 1

    def test_nonexistent_label_error(self):
        """Test error when using non-existent label."""
        G = ec.Graph()
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="not found"):
            G.node_connectivity_between("1", "nonexistent")

    def test_nonexistent_id_error(self):
        """Test error when using non-existent node ID."""
        G = ec.Graph()
        G.add_path_graph(3)

        # Use a node ID that doesn't exist
        with pytest.raises(Exception):
            G.node_connectivity_between(999, 1000)

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
        conn = G.node_connectivity_between("Alice", "Dave")
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
        assert g.node_connectivity_between(i(pair[0]), i(pair[1])) == 1

    def test_matches_networkx(self):
        """Pairwise node connectivity matches NetworkX for non-adjacent pairs."""
        g = self._two_diamonds()
        ref = nx.Graph([
            ("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"),
            ("D", "E"), ("D", "F"), ("E", "F"), ("E", "G"), ("F", "G"),
        ])
        i = g.get_id_from_label
        for s, t in [("A", "F"), ("B", "G"), ("C", "E")]:
            assert g.node_connectivity_between(i(s), i(t)) == nx.node_connectivity(ref, s, t)


def _brute_node_connectivity(H):
    """Smallest node set whose removal leaves H not (strongly) connected, else n - 1."""
    connected = nx.is_strongly_connected if H.is_directed() else nx.is_connected
    if len(H) <= 1 or not connected(H):
        return 0
    for k in range(1, len(H) - 1):
        for cut in itertools.combinations(H, k):
            if not connected(H.subgraph(set(H) - set(cut))):
                return k
    return len(H) - 1


class TestNodeConnectivityDirectedAndMulti:
    """Digraphs need strong connectivity and ordered pairs; parallel edges count once."""

    @staticmethod
    def _pair(n, edges, directed):
        g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(n)])
        if edges:
            g.add_edges([(str(u), str(v)) for u, v in edges])
        H = (nx.DiGraph if directed else nx.Graph)()
        H.add_nodes_from(range(n))
        H.add_edges_from((u, v) for u, v in edges if u != v)
        return g, H

    @pytest.mark.parametrize("directed", [False, True])
    def test_against_brute_force(self, directed):
        for seed in range(120):
            rnd = random.Random(seed)
            n = rnd.randint(1, 7)
            edges = [(rnd.randrange(n), rnd.randrange(n)) for _ in range(rnd.randint(0, 3 * n))]
            if rnd.random() < 0.3:
                edges += [(u, v) for u in range(n) for v in range(n) if u != v and rnd.random() < 0.8]
            g, H = self._pair(n, edges, directed)
            assert g.node_connectivity() == _brute_node_connectivity(H), seed
            for s, t in itertools.permutations(range(n), 2):
                # a direct edge is one path that no node removal breaks
                H2 = H.copy()
                direct = H2.has_edge(s, t)
                if direct:
                    H2.remove_edge(s, t)
                want = nx.node_connectivity(H2, s, t) + direct
                assert g.node_connectivity_between(str(s), str(t)) == want, (seed, s, t)

    def test_weakly_connected_digraph_is_zero(self):
        # nothing reaches node 0, so the digraph is already not strongly connected
        g, _ = self._pair(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)], True)
        assert g.node_connectivity() == 0

    def test_digraph_pairs_are_ordered(self):
        # a bidirected 4-cycle plus the one-way chord 0 -> 2: from 0 to 2 there are the
        # chord and the routes via 1 and 3, but from 2 to 0 only the two routes
        cycle = [(i, (i + 1) % 4) for i in range(4)]
        edges = cycle + [(v, u) for u, v in cycle] + [(0, 2)]
        g, H = self._pair(4, edges, True)
        assert g.node_connectivity() == 2 == nx.node_connectivity(H)
        assert g.node_connectivity_between("0", "2") == 3
        assert g.node_connectivity_between("2", "0") == 2

    def test_complete_digraph(self):
        g, _ = self._pair(4, [(u, v) for u in range(4) for v in range(4) if u != v], True)
        assert g.node_connectivity() == 3
        # one arc missing: the pair it joined can be separated by the other two nodes
        g, _ = self._pair(4, [(u, v) for u in range(4) for v in range(4) if u != v][1:], True)
        assert g.node_connectivity() == 2

    def test_parallel_edges_and_self_loops(self):
        # a path is 1-connected however its edges are doubled or looped
        g, _ = self._pair(3, [(0, 1), (0, 1), (1, 2)], False)
        assert g.node_connectivity() == 1
        g, _ = self._pair(3, [(0, 0), (0, 1), (1, 2)], False)
        assert g.node_connectivity() == 1
        # a doubled direct edge is still a single path
        g, _ = self._pair(3, [(0, 1), (0, 1), (1, 2), (0, 2)], False)
        assert g.node_connectivity_between("0", "1") == 2


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
        assert len(g.edge_disjoint_paths("A", "E")) == g.minimum_cut_value("A", "E") == 2
        assert len(g.node_disjoint_paths("A", "E")) == 1
        assert g.node_connectivity_between("A", "E") == 1

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


# Chain decomposition, k-edge-connected components and subgraphs, and average node
# connectivity, validated against NetworkX (and brute force where NetworkX's k-edge components
# return nothing for one-node graphs).


def _simple_cases(count=120):
    for seed in range(count):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(rnd.randint(2, 12), rnd.choice([0.15, 0.3, 0.5]), seed=seed)
        g = ec.Graph().add_nodes([str(n) for n in G])
        for u, v in G.edges:
            g.add_edge(str(u), str(v))
        yield G, g


def _lab(g, nid):
    return int(g.get_label_from_id(nid))


class TestChainsAndKEdge:
    def test_chain_decomposition_matches_networkx(self):
        for G, g in _simple_cases():
            for root in (None, 0):
                ours = [
                    [(_lab(g, a), _lab(g, b)) for a, b in c]
                    for c in g.chain_decomposition(None if root is None else "0")
                ]
                assert ours == [list(c) for c in nx.chain_decomposition(G, root)]

    def test_k_edge_components_and_subgraphs(self):
        for G, g in _simple_cases(80):
            for k in (1, 2, 3):
                comps = sorted(sorted(_lab(g, x) for x in c) for c in g.k_edge_components(k))
                assert comps == sorted(sorted(c) for c in nx.k_edge_components(G, k))
                subs = sorted(sorted(_lab(g, x) for x in c) for c in g.k_edge_subgraphs(k))
                assert subs == sorted(sorted(c) for c in nx.k_edge_subgraphs(G, k))
                assert g.is_k_edge_connected(k) == nx.is_k_edge_connected(G, k)

    def test_single_node_and_brute_force(self):
        lonely = ec.Graph().add_nodes(["A"])
        assert lonely.k_edge_components(3) == [lonely.get_node_ids(limit=None)]
        G = nx.complete_bipartite_graph(2, 3)
        g = ec.Graph().add_nodes([str(n) for n in G])
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
        brute = {
            frozenset(v for v in G if v == u or local_edge_connectivity(G, u, v) >= 3) for u in G
        }
        ours = {frozenset(_lab(g, x) for x in c) for c in g.k_edge_components(3)}
        assert ours == brute and len(g.k_edge_subgraphs(3)) == 5
        with pytest.raises(ValueError, match="at least 1"):
            g.k_edge_components(0)
        # directed graphs are supported: a one-way path has no 2-edge-connected pairs
        assert len(ec.Graph(directed=True).add_path_graph(3).k_edge_components(2)) == 3

    def test_average_node_connectivity(self):
        for G, g in _simple_cases(30):
            if len(G) <= 9:
                assert g.average_node_connectivity() == pytest.approx(
                    nx.average_node_connectivity(G)
                )
        D = nx.gnp_random_graph(7, 0.4, seed=3, directed=True)
        d = ec.Graph(directed=True).add_nodes([str(n) for n in D])
        d.add_edges([(str(u), str(v)) for u, v in D.edges])
        assert d.average_node_connectivity() == pytest.approx(nx.average_node_connectivity(D))
        assert ec.Graph().add_full_graph(5).average_node_connectivity() == 4.0

    def test_steps(self):
        g = ec.Graph().add_nodes(list("ABCDEF"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
        g.chain_decomposition()
        step = g.steps[-1]
        assert step.kind == KIND_QUERY and step.narration == "2 chains."
        assert ("C", "D") not in step.delta.edges_selected
        g.k_edge_components(2)
        assert set(g.steps[-1].delta.node_groups.values()) == {0, 1}


# Directed k-edge components, all-pairs node connectivity, all minimum node cuts, local edge
# connectivity and k-edge augmentation, validated against NetworkX and brute force.


def _directed_cases(count=60):
    for seed in range(count):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(
            rnd.randint(2, 10), rnd.choice([0.2, 0.35, 0.5]), seed=seed, directed=True
        )
        g = ec.Graph(directed=True).add_nodes([str(n) for n in G])
        for u, v in G.edges:
            g.add_edge(str(u), str(v))
        yield G, g


def _brute_node_cuts(G):
    k = nx.node_connectivity(G)
    cuts = []
    for c in itertools.combinations(G, k):
        rest = G.subgraph(set(G) - set(c))
        if len(rest) and not nx.is_connected(rest):
            cuts.append(sorted(c))
    return sorted(cuts)


class TestDirectedKEdge:
    def test_matches_networkx(self):
        for G, g in _directed_cases():
            for k in (1, 2, 3):
                comps = sorted(sorted(_lab(g, x) for x in c) for c in g.k_edge_components(k))
                assert comps == sorted(sorted(c) for c in nx.k_edge_components(G, k))
                subs = sorted(sorted(_lab(g, x) for x in c) for c in g.k_edge_subgraphs(k))
                assert subs == sorted(sorted(c) for c in nx.k_edge_subgraphs(G, k))

    def test_two_way_ring(self):
        ring = ec.Graph(directed=True).add_nodes(list("ABCD"))
        ring.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        assert len(ring.k_edge_components(1)) == 1 and len(ring.k_edge_components(2)) == 4
        ring.add_edges([("B", "A"), ("C", "B"), ("D", "C"), ("A", "D")])
        assert ring.k_edge_components(2) == [sorted(ring.get_node_ids(limit=None))]


class TestNodeConnectivityTables:
    def test_all_pairs_matches_networkx(self):
        cases = [(G, g) for G, g in _simple_cases(40) if len(G) <= 9]
        cases += list(_directed_cases(20))
        for G, g in cases:
            ours = g.all_pairs_node_connectivity()
            ref = nx.all_pairs_node_connectivity(G)
            assert {
                _lab(g, u): {_lab(g, v): k for v, k in row.items()} for u, row in ours.items()
            } == ref
            some = [str(n) for n in list(G)[:3]]
            sub = g.all_pairs_node_connectivity(some)
            assert {_lab(g, u) for u in sub} == {int(n) for n in some}
            assert sub == {u: {v: ours[u][v] for v in row} for u, row in sub.items()}

    def test_all_node_cuts_brute_force_and_networkx(self):
        checked = 0
        for seed in range(250):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(rnd.randint(3, 10), rnd.choice([0.3, 0.5, 0.7]), seed=seed)
            if not nx.is_connected(G):
                continue
            g = ec.Graph().add_nodes([str(n) for n in G])
            g.add_edges([(str(u), str(v)) for u, v in G.edges])
            ours = sorted(sorted(_lab(g, x) for x in c) for c in g.all_node_cuts())
            if nx.density(G) == 1:
                assert ours == []
                continue
            assert ours == _brute_node_cuts(G)
            assert ours == sorted(sorted(c) for c in nx.all_node_cuts(G))
            checked += 1
        assert checked > 100

    def test_all_node_cuts_examples_and_errors(self):
        ring = ec.Graph().add_cycle_graph(6)
        assert len(ring.all_node_cuts()) == 9
        assert ring.steps[-1].narration == "9 minimum node cuts of size 2."
        grid = ec.Graph().add_grid_2d(8, 8)
        assert len(grid.all_node_cuts()) == 4  # the neighbors of each corner
        assert ec.Graph().add_full_graph(4).all_node_cuts() == []
        split = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        with pytest.raises(ValueError, match="connected"):
            split.all_node_cuts()
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).all_node_cuts()


class TestEdgeAugmentation:
    def test_is_locally_k_edge_connected(self):
        for G, g in list(_simple_cases(40)) + list(_directed_cases(20)):
            ids = g.get_node_ids(limit=None)
            for u, v in itertools.combinations(ids, 2):
                for k in (1, 2, 3):
                    ours = g.is_locally_k_edge_connected(u, v, k)
                    a, b = _lab(g, u), _lab(g, v)
                    assert ours == (local_edge_connectivity(G, a, b) >= k)
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="at least 1"):
            g.is_locally_k_edge_connected(1, 2, 0)
        with pytest.raises(ValueError, match="different"):
            g.is_locally_k_edge_connected(1, 1, 1)

    def test_against_networkx(self):
        for seed in range(150):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(rnd.randint(2, 9), rnd.choice([0.1, 0.25, 0.45]), seed=seed)
            g = ec.Graph().add_nodes([str(n) for n in G])
            g.add_edges([(str(u), str(v)) for u, v in G.edges])
            non_edges = [e for e in itertools.combinations(G, 2) if not G.has_edge(*e)]
            for k in (1, 2, 3):
                avail = None
                if rnd.random() < 0.5:
                    picked = rnd.sample(non_edges, min(len(non_edges), rnd.randint(0, 8)))
                    avail = {(u, v): rnd.choice([1, 2, 5]) for u, v in picked}
                for partial in (False, True):
                    try:
                        ref = list(nx.k_edge_augmentation(G, k, avail=avail, partial=partial))
                    except nx.NetworkXUnfeasible:
                        ref = None
                    ours_avail = (
                        None
                        if avail is None
                        else {(str(u), str(v)): w for (u, v), w in avail.items()}
                    )
                    try:
                        ours = [
                            (_lab(g, u), _lab(g, v))
                            for u, v in g.k_edge_augmentation(k, ours_avail, partial)
                        ]
                    except ValueError:
                        ours = None
                    assert (ours is None) == (ref is None)
                    if ours is None:
                        continue
                    if avail is not None:
                        assert all((u, v) in avail or (v, u) in avail for u, v in ours)
                    H, R = G.copy(), G.copy()
                    H.add_edges_from(ours)
                    R.add_edges_from(ref)
                    done = len(H) > k and nx.is_k_edge_connected(H, k)
                    assert done == (len(R) > k and nx.is_k_edge_connected(R, k))
                    if done:

                        def cost(edges):
                            return sum(
                                avail.get((u, v), avail.get((v, u))) if avail else 1
                                for u, v in edges
                            )

                        assert cost(ours) <= cost(ref)
                        if k <= 2 and avail is None:
                            assert len(ours) == len(ref)  # both optimal

    def test_examples_and_errors(self):
        path = ec.Graph().add_nodes(list("ABCDE"))
        path.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        added = path.k_edge_augmentation(2)
        assert [tuple(path.get_label_from_id(n) for n in e) for e in added] == [("A", "E")]
        assert path.steps[-1].narration == "1 edges to add for 2-edge-connectivity."
        star = ec.Graph().add_nodes(["hub", "a", "b", "c", "d"])
        star.add_edges([("hub", "a"), ("hub", "b"), ("hub", "c"), ("hub", "d")])
        options = [("a", "b", 1), ("c", "d", 1), ("a", "c", 5), ("b", "d", 5)]
        chosen = {
            frozenset(star.get_label_from_id(n) for n in e)
            for e in star.k_edge_augmentation(2, avail=options)
        }
        assert chosen == {frozenset("ab"), frozenset("cd")}
        assert ec.Graph().add_cycle_graph(5).k_edge_augmentation(2) == []
        with pytest.raises(ValueError, match="at least 3 nodes"):
            ec.Graph().add_path_graph(2).k_edge_augmentation(2)
        assert len(ec.Graph().add_path_graph(2).k_edge_augmentation(2, partial=True)) == 0
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).k_edge_augmentation(1)
        with pytest.raises(ValueError, match="cannot"):
            star.k_edge_augmentation(2, avail=[("a", "b")])
