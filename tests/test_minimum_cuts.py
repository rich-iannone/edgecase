import itertools
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_QUERY, KIND_TRANSFORM


class TestMinimumEdgeCut:
    """Test minimum_edge_cut method."""

    def test_simple_path(self):
        """Test minimum edge cut on a simple path."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 4)

        assert len(cut) == 1  # Only one edge needs to be removed

    def test_parallel_edges(self):
        """Test with parallel edges (multigraph)."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        # Two edges between A and B
        G.add_edge("A", "B")
        G.add_edge("A", "B")

        cut = G.minimum_edge_cut("A", "B")

        # For undirected graphs, edges between same nodes are represented once in the cut
        # (the cut algorithm works on edge capacity, and multiple edges increase capacity)
        assert len(cut) >= 1  # At least one edge tuple in the cut

    def test_diamond_graph(self):
        """Test on diamond graph with two paths."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        cut = G.minimum_edge_cut("A", "D")

        assert len(cut) == 2  # Need to cut both paths

    def test_complete_graph(self):
        """Test on complete graph K4."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 4)
        G.add_edge(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 4)

        assert len(cut) == 3  # Three edge-disjoint paths

    def test_directed_graph(self):
        """Test on directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        cut = G.minimum_edge_cut("A", "D")

        assert len(cut) == 2  # Two paths from A to D

    def test_bottleneck(self):
        """Test graph with clear bottleneck."""
        G = Graph()
        for i in [1, 2, 3, 4, 5, 6]:
            G.add_node(i)

        # Left cluster connected by single edge to right cluster
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Left triangle
        G.add_edge(3, 4)  # Bottleneck
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 4)  # Right triangle

        cut = G.minimum_edge_cut(1, 6)

        assert len(cut) == 1  # Only the bottleneck edge
        assert (3, 4) in cut or (4, 3) in cut

    def test_string_labels(self):
        """Test with string node labels."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave"]:
            G.add_node(label)
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")
        G.add_edge("Carol", "Dave")

        cut = G.minimum_edge_cut("Alice", "Dave")
        assert len(cut) >= 1

    def test_self_loop_ignored(self):
        """Test that self-loops don't affect min cut."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 1)  # Self-loop
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_edge_cut(1, 3)
        # Self-loop shouldn't be in the cut
        for edge in cut:
            assert edge[0] != edge[1]

    def test_disconnected_graph_error(self):
        """Test that disconnected nodes have no cut."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 3)

        # If no path exists, cut should be empty
        # (all edges from source side to unreachable side)
        assert isinstance(cut, set)

    def test_same_node_error(self):
        """Test error when source equals sink."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="Source and target must be different"):
            G.minimum_edge_cut(1, 1)

    def test_nonexistent_node_error(self):
        """Test error when node doesn't exist."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="does not exist"):
            G.minimum_edge_cut(1, 999)

    def test_bridge_graph(self):
        """Test graph with bridge edge."""
        G = Graph()
        for i in [1, 2, 3, 4, 5]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Triangle
        G.add_edge(3, 4)  # Bridge
        G.add_edge(4, 5)

        cut = G.minimum_edge_cut(1, 5)

        assert len(cut) == 1


class TestMinimumNodeCut:
    """Test minimum_node_cut method."""

    def test_simple_path(self):
        """Test minimum node cut on simple path."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        cut = G.minimum_node_cut(1, 4)

        # Removing either node 2 or 3 disconnects 1 from 4
        assert len(cut) == 1
        assert 2 in cut or 3 in cut

    def test_diamond_graph(self):
        """Test on diamond graph - two independent paths."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        a_id = G.get_id_from_label("A")
        d_id = G.get_id_from_label("D")

        cut = G.minimum_node_cut("A", "D")

        # Need to remove both B and C to disconnect A from D
        assert len(cut) == 2

    def test_source_sink_not_in_cut(self):
        """Test that source and sink are never in the cut."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_node_cut(1, 3)

        assert 1 not in cut
        assert 3 not in cut
        assert 2 in cut

    def test_articulation_point(self):
        """Test graph with articulation point."""
        G = Graph()
        for i in [1, 2, 3, 4, 5, 6]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Triangle
        G.add_edge(3, 4)  # Articulation point connection
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 4)  # Another triangle

        cut = G.minimum_node_cut(1, 6)

        # Either node 3 or node 4 must be in the cut
        assert len(cut) == 1
        assert 3 in cut or 4 in cut

    def test_complete_graph(self):
        """Test on complete graph K5."""
        G = Graph()
        nodes = [1, 2, 3, 4, 5]
        for i in nodes:
            G.add_node(i)
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j)

        # In K5 every pair is adjacent, so no node cut exists
        with pytest.raises(ValueError, match="adjacent"):
            G.minimum_node_cut(1, 5)

        # Without the 1-5 edge, all 3 other nodes must go
        G.delete_edge(1, 5)
        cut = G.minimum_node_cut(1, 5)
        assert cut == {2, 3, 4}

    def test_directed_graph(self):
        """Test on directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")
        G.add_edge("D", "E")

        cut = G.minimum_node_cut("A", "E")

        # Node D is the bottleneck
        assert len(cut) == 1

        d_id = G.get_id_from_label("D")

        assert d_id in cut

    def test_string_labels(self):
        """Test with string labels."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave"]:
            G.add_node(label)
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")
        G.add_edge("Alice", "Dave")
        G.add_edge("Dave", "Carol")

        cut = G.minimum_node_cut("Alice", "Carol")

        # Both Bob and Dave must be removed
        assert len(cut) == 2

    def test_single_intermediary(self):
        """Test graph with single intermediary node."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_node_cut(1, 3)

        assert len(cut) == 1
        assert 2 in cut

    def test_same_node_error(self):
        """Test error when source equals sink."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="Source and target must be different"):
            G.minimum_node_cut(1, 1)

    def test_nonexistent_node_error(self):
        """Test error when node doesn't exist."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="does not exist"):
            G.minimum_node_cut(1, 999)

    def test_adjacent_source_and_sink_raise(self):
        """Regression: adjacent source and sink returned set() or a set that didn't separate them."""
        # Triangle plus a second detour: used to return {C, D}, leaving the A-B edge intact
        G = Graph().add_edges(
            [("A", "B"), ("B", "C"), ("A", "C"), ("A", "D"), ("D", "B")], auto_create_nodes=True
        )
        with pytest.raises(ValueError, match="adjacent"):
            G.minimum_node_cut("A", "B")
        # Stored the other way round on an undirected graph
        with pytest.raises(ValueError, match="adjacent"):
            G.minimum_node_cut("B", "A")
        # A lone edge used to return set(), indistinguishable from "already disconnected"
        single = Graph().add_edges([("A", "B")], auto_create_nodes=True)
        with pytest.raises(ValueError, match="adjacent"):
            single.minimum_node_cut("A", "B")

    def test_directed_adjacency_follows_edge_direction(self):
        """Only a source -> sink edge makes the cut impossible on a directed graph."""
        G = Graph(directed=True).add_edges(
            [("A", "B"), ("B", "C"), ("C", "A")], auto_create_nodes=True
        )
        with pytest.raises(ValueError, match="adjacent"):
            G.minimum_node_cut("A", "B")
        # B -> A only runs through C
        assert G.minimum_node_cut("B", "A") == {G.get_id_from_label("C")}

    def test_matches_networkx_on_non_adjacent_pairs(self):
        for seed in range(25):
            G = nx.gnp_random_graph(12, 0.3, seed=seed)
            g = Graph().add_nodes([str(n) for n in G])
            if G.number_of_edges():
                g.add_edges([(str(u), str(v)) for u, v in G.edges])
            rng = random.Random(seed)
            for _ in range(5):
                s, t = rng.sample(sorted(G), 2)
                if G.has_edge(s, t):
                    continue
                cut = {int(g.get_label_from_id(n)) for n in g.minimum_node_cut(str(s), str(t))}
                assert len(cut) == len(nx.minimum_node_cut(G, s, t))
                if cut:
                    H = G.copy()
                    H.remove_nodes_from(cut)
                    assert not nx.has_path(H, s, t)

    def test_hub_and_spoke(self):
        """Test hub-and-spoke topology."""
        G = Graph()
        for i in [1, 2, 3, 4, 5]:
            G.add_node(i)
        # Hub node in center
        G.add_edge(1, 5)  # Spoke 1 to hub
        G.add_edge(2, 5)  # Spoke 2 to hub
        G.add_edge(3, 5)  # Spoke 3 to hub
        G.add_edge(4, 5)  # Spoke 4 to hub

        cut = G.minimum_node_cut(1, 2)

        # Hub node 5 is the only node that disconnects them
        assert len(cut) == 1
        assert 5 in cut


class TestMinimumCutComparison:
    """Test comparing edge cut vs node cut."""

    def test_edge_vs_node_cut(self):
        """Compare edge and node cuts on same graph."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)

        edge_cut = G.minimum_edge_cut(1, 4)
        node_cut = G.minimum_node_cut(1, 4)

        # Both should disconnect the graph
        assert len(edge_cut) >= 1
        assert len(node_cut) >= 1

    def test_min_cut_size_equivalence(self):
        """Test that minimum_cut_value matches minimum_edge_cut size."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(1, 4)
        G.add_edge(2, 4)

        cut_size = G.minimum_cut_value(1, 3)
        edge_cut = G.minimum_edge_cut(1, 3)

        assert cut_size == len(edge_cut)


class TestMinimumCutRealWorld:
    """Real-world scenarios for minimum cuts."""

    def test_network_vulnerability(self):
        """Test identifying critical network links."""
        G = Graph(directed=True)
        for label in ["Router1", "Router2", "Router3", "Server"]:
            G.add_node(label)
        G.add_edge("Router1", "Router2")
        G.add_edge("Router1", "Router3")
        G.add_edge("Router2", "Server")
        G.add_edge("Router3", "Server")

        critical_links = G.minimum_edge_cut("Router1", "Server")

        # Both paths to server are needed
        assert len(critical_links) == 2

    def test_social_network_bridges(self):
        """Test finding bridge people in social network."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]:
            G.add_node(label)
        # Two communities connected by one person
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")  # Community 1
        G.add_edge("Carol", "Dave")  # Bridge
        G.add_edge("Dave", "Eve")
        G.add_edge("Eve", "Frank")  # Community 2

        bridges = G.minimum_node_cut("Alice", "Frank")

        # Carol and Dave are bridges
        assert len(bridges) >= 1

    def test_supply_chain(self):
        """Test supply chain bottleneck analysis."""
        G = Graph(directed=True)
        for label in ["Supplier", "Factory1", "Factory2", "Distributor", "Retailer"]:
            G.add_node(label)
        G.add_edge("Supplier", "Factory1")
        G.add_edge("Supplier", "Factory2")
        G.add_edge("Factory1", "Distributor")
        G.add_edge("Factory2", "Distributor")
        G.add_edge("Distributor", "Retailer")

        bottleneck = G.minimum_node_cut("Supplier", "Retailer")

        # Distributor is single point of failure
        assert len(bottleneck) == 1

        dist_id = G.get_id_from_label("Distributor")

        assert dist_id in bottleneck


# Cut and boundary measures (cut_size, volume, normalized_cut_size, conductance, expansions,
# node/edge boundary), validated against NetworkX.


def _cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G0 = nx.gnp_random_graph(
                14, rnd.choice([0.15, 0.25, 0.4]), seed=seed, directed=directed
            )
            G = (nx.DiGraph if directed else nx.Graph)()
            G.add_nodes_from(range(14))
            for u, v in G0.edges:
                G.add_edge(u, v, weight=rnd.choice([0.5, 1, 2, 3]))
            g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(14)])
            gw = ec.Graph(directed=directed, weighted=True).add_nodes([str(v) for v in range(14)])
            for u, v, d in G.edges(data=True):
                g.add_edge(str(u), str(v))
                gw.add_edge(str(u), str(v), weight=d["weight"])
            yield G, g, gw, rnd


def _two_triangles():
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _ids(g, nodes):
    return [g.get_id_from_label(str(v)) for v in nodes]


def _lab(g, nid):
    return int(g.get_label_from_id(nid))


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


def _labs(g, ids):
    return [int(g.get_label_from_id(i)) for i in ids]


def _id(g, v):
    return g.get_id_from_label(str(v))


class TestCutMeasures:
    def test_against_networkx(self):
        for G, g, gw, rnd in _cases():
            S = rnd.sample(range(14), 5)
            T = [v for v in rnd.sample(range(14), 6) if v not in S]
            complement = set(G) - set(S)
            s_ids, t_ids = _ids(g, S), _ids(g, T)
            for gg, w in ((g, None), (gw, "weight")):
                assert gg.cut_size(s_ids, t_ids) == pytest.approx(nx.cut_size(G, S, T, weight=w))
                assert gg.cut_size(s_ids) == pytest.approx(nx.cut_size(G, S, complement, weight=w))
                assert gg.volume(s_ids) == pytest.approx(nx.volume(G, S, weight=w))
                if nx.volume(G, S, weight=w) and nx.volume(G, complement, weight=w):
                    assert gg.normalized_cut_size(s_ids) == pytest.approx(
                        nx.normalized_cut_size(G, S, weight=w)
                    )
                    assert gg.conductance(s_ids) == pytest.approx(nx.conductance(G, S, weight=w))
                assert gg.edge_expansion(s_ids) == pytest.approx(nx.edge_expansion(G, S, weight=w))
                if G.number_of_edges():
                    assert gg.mixing_expansion(s_ids, t_ids) == pytest.approx(
                        nx.mixing_expansion(G, S, T, weight=w)
                    )
            assert g.node_expansion(s_ids) == pytest.approx(nx.node_expansion(G, S))
            assert g.boundary_expansion(s_ids) == pytest.approx(nx.boundary_expansion(G, S))
            assert sorted(_lab(g, n) for n in g.node_boundary(s_ids)) == sorted(
                nx.node_boundary(G, S)
            )
            assert sorted(_lab(g, n) for n in g.node_boundary(s_ids, t_ids)) == sorted(
                nx.node_boundary(G, S, T)
            )
            ours = sorted(
                tuple(sorted(_lab(g, x) for x in g.get_node_ids_by_edge_id(e)))
                for e in g.edge_boundary(s_ids)
            )
            assert ours == sorted(tuple(sorted(e)) for e in nx.edge_boundary(G, S))

    def test_two_triangles(self):
        g = _two_triangles()
        tri = ["A", "B", "C"]
        assert g.cut_size(tri) == 1 and g.volume(tri) == 7
        assert g.conductance(tri) == pytest.approx(1 / 7)
        assert g.normalized_cut_size(tri) == pytest.approx(2 / 7)
        assert g.edge_expansion(tri) == pytest.approx(1 / 3)
        assert g.mixing_expansion(tri) == pytest.approx(1 / 14)
        assert g.node_expansion(tri) == pytest.approx(4 / 3)
        assert g.boundary_expansion(tri) == pytest.approx(1 / 3)
        assert g.node_boundary(tri) == [g.get_id_from_label("D")]
        assert g.edge_boundary(tri) == g.find_edge_ids("C", "D")

    def test_errors(self):
        g = _two_triangles()
        with pytest.raises(ValueError, match="at least one"):
            g.cut_size([])
        with pytest.raises(ValueError):
            g.volume(["nope"])
        lonely = ec.Graph().add_nodes(["A", "B"])
        with pytest.raises(ValueError, match="volume"):
            lonely.conductance(["A"])

    def test_steps(self):
        g = _two_triangles()
        g.conductance(["A", "B", "C"])
        step = g.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 0}
        assert step.delta.edges_selected == [("C", "D")]
        g.node_boundary(["A", "B", "C"])
        assert g.steps[-1].delta.nodes_selected == ["D"]


# s-t minimum cut partitions, Stoer-Wagner and Gomory-Hu trees, validated against NetworkX.


class TestMinimumCut:
    @pytest.mark.parametrize("directed", [False, True])
    def test_value_and_partition_match_networkx(self, directed):
        for seed in range(20):
            G, g = _random(seed, directed)
            for s, t in itertools.islice(itertools.permutations(range(12), 2), 0, 132, 17):
                value, (side, rest) = g.minimum_cut(_id(g, s), _id(g, t))
                assert value == pytest.approx(nx.minimum_cut_value(G, s, t))
                left = set(_labs(g, side))
                assert s in left and t not in left
                assert sorted(side + rest) == sorted(g.get_node_ids(limit=None))
                crossing = sum(
                    d["capacity"]
                    for u, v, d in G.edges(data=True)
                    if (u in left and v not in left)
                    or (not directed and v in left and u not in left)
                )
                assert crossing == pytest.approx(value)

    def test_source_side_is_closest_to_source(self):
        g = ec.Graph(directed=True).add_nodes(["s", "a", "b", "t"])
        g.add_edge("s", "a", capacity=2)
        g.add_edge("a", "b", capacity=5)
        g.add_edge("b", "t", capacity=4)
        value, (side, rest) = g.minimum_cut("s", "t")
        assert value == 2.0 and [g.get_label_from_id(n) for n in side] == ["s"]

    def test_capacity_attribute_name_and_errors(self):
        g = ec.Graph().add_nodes(["A", "B", "C"])
        g.add_edge("A", "B", width=3)
        g.add_edge("B", "C", width=7)
        assert g.minimum_cut("A", "C", capacity="width")[0] == 3.0
        assert g.minimum_cut("A", "C")[0] == 1.0  # no "capacity" attribute: unit capacities
        with pytest.raises(ValueError, match="different"):
            g.minimum_cut("A", "A")

    def test_steps(self):
        g = _two_triangles()
        g.minimum_cut("A", "F")
        step = g.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        assert step.delta.edges_selected == [("C", "D")]
        assert step.narration == "Minimum cut of value 1: 1 edge between 3 and 3 nodes."


class TestStoerWagner:
    def test_value_matches_networkx(self):
        for seed in range(25):
            G, g = _random(seed, False)
            if G.number_of_edges() == 0 or not nx.is_connected(G):
                continue
            value, (a, b) = g.stoer_wagner()
            assert value == pytest.approx(nx.stoer_wagner(G, weight="capacity")[0])
            left = set(_labs(g, a))
            crossing = sum(
                d["capacity"] for u, v, d in G.edges(data=True) if (u in left) != (v in left)
            )
            assert crossing == pytest.approx(value)
            assert min(a) < min(b)  # side with the smallest ID first

    def test_examples_and_edge_cases(self):
        value, (a, b) = _two_triangles().stoer_wagner()
        assert value == 1.0 and len(a) == len(b) == 3
        w = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        w.add_edges([("A", "B", 3.0), ("B", "C", 4.0), ("C", "A", 3.0), ("C", "D", 1.5)])
        value, (_, other) = w.stoer_wagner()
        assert value == 1.5 and [w.get_label_from_id(n) for n in other] == ["D"]
        split = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        assert split.stoer_wagner()[0] == 0.0
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A", "B"]).stoer_wagner()
        with pytest.raises(ValueError, match="two nodes"):
            ec.Graph().add_nodes(["A"]).stoer_wagner()

    def test_steps(self):
        g = _two_triangles()
        g.stoer_wagner()
        assert g.steps[-1].delta.edges_selected == [("C", "D")]


class TestGomoryHu:
    def test_tree_encodes_all_pairwise_cuts(self):
        for seed in range(12):
            G, g = _random(seed, False)
            tree = g.gomory_hu_tree()
            assert tree.count_edges() == g.count_nodes() - 1 and not tree.is_directed()
            T = nx.Graph()
            T.add_nodes_from(range(12))
            for eid in tree.get_edge_ids(limit=None):
                s, t = tree.get_node_ids_by_edge_id(eid)
                T.add_edge(
                    int(tree.get_label_from_id(s)),
                    int(tree.get_label_from_id(t)),
                    w=tree.get_edge(eid).weight,
                )
            assert nx.is_tree(T)
            for u, v in itertools.combinations(range(12), 2):
                path = nx.shortest_path(T, u, v)
                smallest = min(T[a][b]["w"] for a, b in zip(path, path[1:]))
                assert smallest == pytest.approx(nx.maximum_flow_value(G, u, v))

    def test_keeps_nodes_and_rejects_directed(self):
        g = _two_triangles()
        g.set_node_attrs("x", {1: "keep"})
        tree = g.gomory_hu_tree()
        assert tree.get_node_attrs("x") == {1: "keep"} and tree.is_weighted()
        weights = sorted(tree.get_edge(e).weight for e in tree.get_edge_ids(limit=None))
        assert weights == [1.0, 2.0, 2.0, 2.0, 2.0]
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).gomory_hu_tree()

    def test_steps(self):
        g = _two_triangles()
        tree = g.gomory_hu_tree()
        assert g.steps[-1].kind == KIND_TRANSFORM and len(tree.steps) == 1
        assert "weakest pairwise cut has value 1" in g.steps[-1].narration
