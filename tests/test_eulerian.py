import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_QUERY, KIND_TRANSFORM


class TestIsEulerian:
    """Tests for is_eulerian method."""

    def test_empty_graph_is_eulerian(self):
        """Empty graph is considered Eulerian."""
        graph = Graph()

        assert graph.is_eulerian()

    def test_single_node_is_eulerian(self):
        """Single node with no edges is Eulerian."""
        graph = Graph()
        graph.add_node(0)

        assert graph.is_eulerian()

    def test_cycle_graph_is_eulerian(self):
        """Cycle graph has Eulerian circuit (all even degrees)."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

    def test_path_graph_not_eulerian(self):
        """Path graph is not Eulerian (endpoints have odd degree)."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

    def test_star_graph_not_eulerian(self):
        """Star graph is not Eulerian (center and leaves have odd degree)."""
        graph = Graph()
        graph.add_star_graph(4)

        # Star: center has degree 4 (even), but leaves have degree 1 (odd)
        # More than 2 odd nodes -> not Eulerian
        assert not graph.is_eulerian()

    def test_grid_2x2_is_eulerian(self):
        """2x2 grid is Eulerian."""
        graph = Graph()
        graph.add_grid_2d(2, 2)

        # 2x2 grid: corners have degree 2, but forms a cycle -> Eulerian
        assert graph.is_eulerian()

    def test_disconnected_graph_not_eulerian(self):
        """Disconnected graph is not Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(3)  # First component (nodes 1,2,3)

        # Add disconnected path
        before = set(graph.get_node_ids())
        graph.add_node(label="d1")
        graph.add_node(label="d2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])

        assert not graph.is_eulerian()

    def test_directed_cycle_is_eulerian(self):
        """Directed cycle is Eulerian (in-degree = out-degree)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

    def test_directed_path_not_eulerian(self):
        """Directed path is not Eulerian."""
        graph = Graph(directed=True)
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

    def test_directed_balanced_graph(self):
        """Directed graph with balanced degrees is Eulerian."""
        graph = Graph(directed=True)

        # Create directed cycle with balanced degrees
        graph.add_cycle_graph(4)

        assert graph.is_eulerian()


class TestIsSemiEulerian:
    """Tests for is_semieulerian method."""

    def test_empty_graph_not_semieulerian(self):
        """Empty graph is not semi-Eulerian."""
        graph = Graph()

        assert not graph.is_semieulerian()

    def test_single_node_not_semieulerian(self):
        """Single node is not semi-Eulerian."""
        graph = Graph()
        graph.add_node(0)

        assert not graph.is_semieulerian()

    def test_path_graph_is_semieulerian(self):
        """Path graph is semi-Eulerian (exactly 2 odd-degree nodes)."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_semieulerian()

    def test_cycle_graph_not_semieulerian(self):
        """Cycle graph is not semi-Eulerian (it's Eulerian)."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert not graph.is_semieulerian()

    def test_star_graph_not_semieulerian(self):
        """Star graph is not semi-Eulerian (center has odd degree, leaves have degree 1)."""
        graph = Graph()
        graph.add_star_graph(5)

        # Star: center has degree 5 (odd), 5 leaves have degree 1 (odd)
        # More than 2 odd-degree nodes -> not semi-Eulerian
        assert not graph.is_semieulerian()

    def test_directed_path_is_semieulerian(self):
        """Directed path is semi-Eulerian (one source, one sink)."""
        graph = Graph(directed=True)
        graph.add_path_graph(5)

        assert graph.is_semieulerian()

    def test_directed_cycle_not_semieulerian(self):
        """Directed cycle is not semi-Eulerian (it's Eulerian)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert not graph.is_semieulerian()

    def test_graph_with_two_odd_degrees(self):
        """Graph with exactly 2 odd-degree vertices is semi-Eulerian."""
        # Path graph has exactly 2 odd-degree nodes (the endpoints)
        graph = Graph()
        graph.add_path_graph(4)

        assert graph.is_semieulerian()


class TestEulerianCircuit:
    """Tests for eulerian_circuit method."""

    def test_empty_graph_returns_empty_circuit(self):
        """Empty graph returns empty circuit."""
        graph = Graph()
        circuit = graph.eulerian_circuit()

        assert circuit == []

    def test_single_node_returns_single_node_circuit(self):
        """Single node returns circuit with just that node."""
        graph = Graph()
        node_id = graph.add_node(label="only").get_node_ids()[0]
        circuit = graph.eulerian_circuit()

        assert circuit == [node_id]

    def test_path_graph_no_circuit(self):
        """Path graph has no Eulerian circuit."""
        graph = Graph()
        graph.add_path_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_cycle_graph_has_circuit(self):
        """Cycle graph has Eulerian circuit."""
        graph = Graph()
        graph.add_cycle_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is not None

        # Circuit should have exactly n+1 nodes for n-cycle (returns to start)
        assert len(circuit) == 6  # 5 edges + return to start
        assert circuit[0] == circuit[-1]  # Starts and ends at same node

    def test_circuit_visits_all_edges(self):
        """Circuit visits all edges."""
        graph = Graph()
        graph.add_cycle_graph(4)
        circuit = graph.eulerian_circuit()

        # Circuit should visit all 4 edges
        assert circuit is not None
        assert len(circuit) >= 5  # At least 4 edges + return

    def test_directed_cycle_has_circuit(self):
        """Directed cycle has Eulerian circuit."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) >= 6  # At least 5 edges + return
        assert circuit[0] == circuit[-1]

    def test_circuit_respects_direction(self):
        """Directed circuit follows edge directions."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(3)

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) == 4  # 3 edges + return
        assert circuit[0] == circuit[-1]

        # Verify edges exist in the direction traveled
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            neighbors = graph.neighbors(u)
            assert v in neighbors

    def test_disconnected_graph_no_circuit(self):
        """Disconnected graph has no Eulerian circuit."""
        graph = Graph()
        graph.add_cycle_graph(3)

        # Add disconnected path
        before = set(graph.get_node_ids())
        graph.add_node(label="d1")
        graph.add_node(label="d2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])

        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_grid_2x2_has_circuit(self):
        """2x2 grid has Eulerian circuit."""
        graph = Graph()
        graph.add_grid_2d(2, 2)
        circuit = graph.eulerian_circuit()

        assert circuit is not None

        # 2x2 grid: 4 nodes, but edges form a cycle
        assert len(circuit) >= 5  # At least visit all edges + return


class TestEulerianIntegration:
    """Integration tests combining multiple Eulerian methods."""

    def test_eulerian_implies_circuit_exists(self):
        """If is_eulerian is True, eulerian_circuit should return a circuit."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None

    def test_not_eulerian_implies_no_circuit(self):
        """If is_eulerian is False, eulerian_circuit should return None."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_semieulerian_not_eulerian(self):
        """Semi-Eulerian graph is not Eulerian."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_semieulerian()
        assert not graph.is_eulerian()

    def test_eulerian_not_semieulerian(self):
        """Eulerian graph is not semi-Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()
        assert not graph.is_semieulerian()

    def test_neither_eulerian_nor_semieulerian(self):
        """Graph can be neither Eulerian nor semi-Eulerian."""
        graph = Graph()
        graph.add_star_graph(5)

        assert not graph.is_eulerian()
        assert not graph.is_semieulerian()


class TestEulerianEdgeCases:
    """Edge cases and special scenarios."""

    def test_single_isolated_node(self):
        """Single isolated node is Eulerian."""
        graph = Graph()
        graph.add_node(label="solo")

        # Single node with no edges is Eulerian
        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) == 1

    def test_multiple_edges_between_nodes(self):
        """Multiple edges between same nodes (multigraph)."""
        graph = Graph()
        graph.add_path_graph(2)  # Two nodes with one edge
        node_ids = sorted(graph.get_node_ids())

        # Add more edges to make even degree
        graph.add_edge(node_ids[0], node_ids[1])  # Now 2 edges
        graph.add_edge(node_ids[0], node_ids[1])  # Now 3 edges (odd!)

        # Each node has degree 3 (odd) -> not Eulerian, but is semi-Eulerian
        assert not graph.is_eulerian()
        assert graph.is_semieulerian()

    def test_very_small_cycle(self):
        """Triangle is Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(3)

        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) >= 4  # At least 3 edges + return

    def test_directed_strongly_connected_components(self):
        """Directed graph with multiple SCCs is not Eulerian."""
        graph = Graph(directed=True)

        # Create first cycle
        graph.add_cycle_graph(3)

        # Add disconnected cycle
        before = set(graph.get_node_ids())
        graph.add_node(label="x1")
        graph.add_node(label="x2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])
        graph.add_edge(new_nodes[1], new_nodes[0])

        # Not weakly connected -> not Eulerian
        assert not graph.is_eulerian()


class TestEulerianCircuitIsolatedNodes:
    """Isolated nodes must not be picked as the start (that returned just [node])."""

    @staticmethod
    def _is_valid_circuit(g, circuit, directed):
        import collections

        edges = [g.get_edge(e) for e in g.get_edge_ids()]
        if len(circuit) != len(edges) + 1 or circuit[0] != circuit[-1]:
            return False
        remaining = collections.Counter(
            (e.source, e.target) if directed else frozenset((e.source, e.target)) for e in edges
        )
        for a, b in zip(circuit, circuit[1:]):
            key = (a, b) if directed else frozenset((a, b))
            if remaining[key] == 0:
                return False
            remaining[key] -= 1
        return True

    @pytest.mark.parametrize("directed", [False, True])
    def test_isolated_node_stored_first(self, directed):
        """With node IDs 1-4, ID 3 comes first in storage (hash) order, so the isolated node
        is created third to make it the old start node."""
        g = Graph(directed=directed)
        for label in ["A", "B", "lonely", "C"]:
            g.add_node(label=label)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        assert g.get_label_from_id(g.get_node_ids()[0]) == "lonely"  # the case under test

        circuit = g.eulerian_circuit()
        assert self._is_valid_circuit(g, circuit, directed)
        assert {g.get_label_from_id(n) for n in circuit} == {"A", "B", "C"}

    @pytest.mark.parametrize("directed", [False, True])
    def test_random_eulerian_graphs_with_isolated_nodes(self, directed):
        import random

        rng = random.Random(3 if directed else 4)
        for _ in range(100):
            n = rng.randint(3, 8)
            labels = [f"iso{i}" for i in range(rng.randint(1, 3))] + [str(i) for i in range(n)]
            rng.shuffle(labels)
            g = Graph(directed=directed)
            for label in labels:
                g.add_node(label=label)
            # A union of cycles through node "0" is connected and Eulerian
            for _ in range(rng.randint(1, 3)):
                k = rng.randint(2 if directed else 3, n)
                cycle = ["0"] + [str(x) for x in rng.sample(range(1, n), k - 1)]
                for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                    g.add_edge(a, b)

            circuit = g.eulerian_circuit()
            assert circuit is not None
            assert self._is_valid_circuit(g, circuit, directed)


# Eulerian paths and eulerize(), validated against NetworkX.


def _santa_house() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(["bl", "br", "tl", "tr", "roof"])
        .add_edges(
            [
                ("bl", "br"),
                ("bl", "tl"),
                ("br", "tr"),
                ("tl", "tr"),
                ("tl", "roof"),
                ("tr", "roof"),
                ("bl", "tr"),
                ("br", "tl"),
            ]
        )  # fmt: skip
    )


class TestEulerianPathAndEulerize:
    @pytest.mark.parametrize("directed", [False, True])
    def test_paths_match_networkx_existence_and_are_valid_trails(self, directed):
        for seed in range(40):
            G = nx.gnp_random_graph(8, 0.4, seed=seed, directed=directed)
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() < 2:
                continue
            G = (nx.DiGraph if directed else nx.Graph)(sorted(G.edges))
            g = ec.Graph(directed=directed).add_nodes([str(v) for v in G.nodes])
            g.add_edges([(str(u), str(v)) for u, v in G.edges])
            assert g.has_eulerian_path() == nx.has_eulerian_path(G)
            for v in G.nodes:
                vid = g.get_id_from_label(str(v))
                assert g.has_eulerian_path(vid) == nx.has_eulerian_path(G, source=v)
            nodes = g.eulerian_path()
            if nodes is None:
                assert not nx.has_eulerian_path(G)
                continue
            edges = g.eulerian_path(edges=True)
            assert len(edges) == len(set(edges)) == G.number_of_edges()
            assert len(nodes) == len(edges) + 1
            for k, e in enumerate(edges):
                s, t = g.get_node_ids_by_edge_id(e)
                assert (s, t) == (nodes[k], nodes[k + 1]) or (
                    not directed and (t, s) == (nodes[k], nodes[k + 1])
                )

    def test_house_of_santa_claus(self):
        g = _santa_house()
        path = [g.get_label_from_id(n) for n in g.eulerian_path()]
        assert len(path) == 9 and {path[0], path[-1]} == {"bl", "br"}
        assert g.eulerian_path(source="roof") is None
        assert g.get_label_from_id(g.eulerian_path(source="br")[0]) == "br"

    def test_circuit_case_and_parallel_edges(self):
        cycle = ec.Graph().add_cycle_graph(4)
        path = cycle.eulerian_path(source=3)
        assert path[0] == path[-1] == 3 and len(path) == 5
        doubled = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "B")])
        assert sorted(doubled.eulerian_path(edges=True)) == [1, 2]
        assert ec.Graph().add_star_graph(4).eulerian_path() is None

    def test_isolated_nodes_are_ignored(self):
        g = ec.Graph().add_path_graph(3).add_node("lonely")
        assert g.has_eulerian_path() and g.eulerian_path() is not None

    def test_eulerize_matches_networkx_count(self):
        for seed in range(20):
            U = nx.gnp_random_graph(12, 0.35, seed=seed)
            if not nx.is_connected(U):
                continue
            U = nx.Graph(sorted(U.edges))
            g = ec.Graph().add_nodes([str(v) for v in sorted(U.nodes)])
            g.add_edges([(str(u), str(v)) for u, v in U.edges])
            walk = g.eulerize()
            assert walk.is_eulerian()
            added = walk.count_edges() - g.count_edges()
            assert added == nx.eulerize(U).number_of_edges() - U.number_of_edges()

    def test_eulerize_marks_duplicates_and_keeps_ids(self):
        house = (
            ec.Graph()
            .add_nodes(["bl", "br", "tl", "tr", "roof"])
            .add_edges(
                [
                    ("bl", "br"),
                    ("bl", "tl"),
                    ("br", "tr"),
                    ("tl", "tr"),
                    ("tl", "roof"),
                    ("tr", "roof"),
                ]
            )
        )
        walk = house.eulerize()
        added = walk.get_edge_df().filter(eulerize_added=True)
        assert added.height == 1
        assert walk.get_edge_df().filter(eulerize_added=False).height == 6  # never null
        s, t = added.select(["source", "target"]).row(0)
        assert {walk.get_label_from_id(s), walk.get_label_from_id(t)} == {"tl", "tr"}
        assert walk.get_edge_ids(limit=None)[:6] == house.get_edge_ids(limit=None)
        assert house.count_edges() == 6  # original untouched

    def test_eulerize_weighted_prefers_light_paths(self):
        # Odd nodes A and C: the direct A-C edge is heavy, so duplicate the light detour A-B-C
        g = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        g.add_edges(
            [("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0), ("C", "D", 1.0), ("D", "A", 1.0)]
        )
        added = g.eulerize().get_edge_df().filter(eulerize_added=True)
        assert added.height == 2
        assert g.eulerize(weighted=False).get_edge_df().filter(eulerize_added=True).height == 1

    def test_eulerize_errors(self):
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).eulerize()
        two = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        with pytest.raises(ValueError, match="connected"):
            two.eulerize()
        inf = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        inf.add_edges([("A", "B", float("inf")), ("B", "C", 1.0), ("C", "D", float("inf"))])
        with pytest.raises(ValueError, match="finite"):  # the pairing traceback used to hang
            inf.eulerize()

    def test_steps(self):
        path = ec.Graph().add_path_graph(4)
        path.eulerian_path()
        step = path.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.delta.edges_selected == [("1", "2"), ("2", "3"), ("3", "4")]
        assert step.narration == "Eulerian path through 3 edges, from 1."
        walk = path.eulerize()
        step = path.steps[-1]
        assert step.kind == KIND_TRANSFORM and step.narration == "eulerize() duplicated 3 edges."
        assert len(walk.steps) == 1
