import random
import warnings

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_QUERY
from networkx.algorithms import isomorphism as nx_iso


class TestIsPlanar:
    """Tests for is_planar() method."""

    def test_empty_graph_is_planar(self):
        """Empty graph is planar."""
        graph = Graph(directed=False)

        assert graph.is_planar()

    def test_single_node_is_planar(self):
        """Single node is planar."""
        graph = Graph(directed=False).add_node(label="A")

        assert graph.is_planar()

    def test_two_nodes_is_planar(self):
        """Two nodes with edge is planar."""
        graph = Graph(directed=False).add_node(label="A").add_node(label="B").add_edge("A", "B")

        assert graph.is_planar()

    def test_path_graph_is_planar(self):
        """Path graphs are planar."""
        graph = Graph(directed=False).add_path_graph(n=10)

        assert graph.is_planar()

    def test_cycle_graph_is_planar(self):
        """Cycle graphs are planar."""
        graph = Graph(directed=False).add_cycle_graph(n=8)

        assert graph.is_planar()

    def test_tree_is_planar(self):
        """Trees are always planar (e = n-1)."""
        tree = Graph(directed=False)
        tree.add_balanced_tree(k=3, h=3)

        assert tree.is_planar()

    def test_star_graph_is_planar(self):
        """Star graphs are planar."""
        star = Graph(directed=False)
        star.add_star_graph(n=10)

        assert star.is_planar()

    def test_grid_is_planar(self):
        """Grid graphs are planar."""
        grid = Graph(directed=False)
        grid.add_grid_2d(rows=5, cols=5)

        assert grid.is_planar()

    def test_k4_is_planar(self):
        """Complete graph K_4 is planar (6 edges ≤ 3*4-6 = 6)."""
        k4 = Graph(directed=False)
        k4.add_full_graph(n=4)

        assert k4.is_planar()

    def test_k5_not_planar(self):
        """Complete graph K_5 is not planar (10 edges > 3*5-6 = 9)."""
        k5 = Graph(directed=False)
        k5.add_full_graph(n=5)

        assert not k5.is_planar()

    def test_k6_not_planar(self):
        """Complete graph K_6 is not planar."""
        k6 = Graph(directed=False)
        k6.add_full_graph(n=6)

        assert not k6.is_planar()

    def test_bipartite_graph_k33_not_planar(self):
        """Complete bipartite K_{3,3} is not planar."""
        # K_{3,3} has 6 nodes, 9 edges
        # For graphs without triangles: e ≤ 2n-4 = 8
        # So 9 > 8, not planar
        k33 = (
            Graph(directed=False)
            .add_node(label="A1")
            .add_node(label="A2")
            .add_node(label="A3")
            .add_node(label="B1")
            .add_node(label="B2")
            .add_node(label="B3")
            .add_edge("A1", "B1")
            .add_edge("A1", "B2")
            .add_edge("A1", "B3")
            .add_edge("A2", "B1")
            .add_edge("A2", "B2")
            .add_edge("A2", "B3")
            .add_edge("A3", "B1")
            .add_edge("A3", "B2")
            .add_edge("A3", "B3")
        )

        assert not k33.is_planar()

    def test_triangle_is_planar(self):
        """Triangle is planar."""
        triangle = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        assert triangle.is_planar()


class TestIsIsomorphic:
    """Tests for is_isomorphic() method."""

    def test_empty_graphs_isomorphic(self):
        """Two empty graphs are isomorphic."""
        g1 = Graph(directed=False)
        g2 = Graph(directed=False)

        assert g1.is_isomorphic(g2)

    def test_single_node_graphs_isomorphic(self):
        """Two single-node graphs are isomorphic."""
        g1 = Graph(directed=False).add_node(label="A")
        g2 = Graph(directed=False).add_node(label="X")

        assert g1.is_isomorphic(g2)

    def test_same_path_graphs_isomorphic(self):
        """Two path graphs with same length are isomorphic."""
        g1 = Graph(directed=False).add_path_graph(n=5)
        g2 = Graph(directed=False).add_path_graph(n=5)

        assert g1.is_isomorphic(g2)

    def test_different_sized_graphs_not_isomorphic(self):
        """Graphs with different node counts are not isomorphic."""
        g1 = Graph(directed=False).add_path_graph(n=3)
        g2 = Graph(directed=False).add_path_graph(n=4)

        assert not g1.is_isomorphic(g2)

    def test_different_edge_counts_not_isomorphic(self):
        """Graphs with different edge counts are not isomorphic."""
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        assert not g1.is_isomorphic(g2)

    def test_different_directedness_not_isomorphic(self):
        """Directed and undirected graphs are not isomorphic."""
        g1 = Graph(directed=True).add_path_graph(n=3)
        g2 = Graph(directed=False).add_path_graph(n=3)

        assert not g1.is_isomorphic(g2)

    def test_path_and_star_not_isomorphic(self):
        """Path and star graphs with same size have different degree sequences."""
        path = Graph(directed=False).add_path_graph(n=5)
        star = Graph(directed=False).add_star_graph(n=5)

        assert not path.is_isomorphic(star)

    def test_same_complete_graphs_isomorphic(self):
        """Two complete graphs of same size are isomorphic."""
        k4_a = Graph(directed=False).add_full_graph(n=4)
        k4_b = Graph(directed=False).add_full_graph(n=4)

        assert k4_a.is_isomorphic(k4_b)

    def test_same_cycle_graphs_isomorphic(self):
        """Two cycle graphs of same size are isomorphic."""
        c6_a = Graph(directed=False).add_cycle_graph(n=6)
        c6_b = Graph(directed=False).add_cycle_graph(n=6)

        assert c6_a.is_isomorphic(c6_b)

    def test_triangles_with_different_labels_isomorphic(self):
        """Triangles with different labels are isomorphic."""
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
            .add_edge("Z", "X")
        )

        assert g1.is_isomorphic(g2)

    def test_different_degree_sequences_not_isomorphic(self):
        """Graphs with different degree sequences are not isomorphic."""
        # Graph 1: Star (degrees: 1,1,1,3)
        g1 = Graph(directed=False).add_star_graph(n=4)

        # Graph 2: Path (degrees: 1,2,2,1)
        g2 = Graph(directed=False).add_path_graph(n=4)

        assert not g1.is_isomorphic(g2)

    def test_isomorphic_with_itself(self):
        """Any graph is isomorphic with itself."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert graph.is_isomorphic(graph)

    def test_complex_graphs_same_structure(self):
        """Complex graphs with same structure are isomorphic."""
        # Both are "diamond" shapes: 4 nodes, 5 edges
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
            .add_edge("B", "C")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="W")
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("W", "X")
            .add_edge("W", "Y")
            .add_edge("X", "Z")
            .add_edge("Y", "Z")
            .add_edge("X", "Y")
        )

        assert g1.is_isomorphic(g2)

    def test_directed_graphs_isomorphic(self):
        """Directed graphs can be isomorphic."""
        g1 = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        g2 = (
            Graph(directed=True)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        assert g1.is_isomorphic(g2)


class TestIntegration:
    """Integration tests for both methods."""

    def test_planar_graphs_can_be_isomorphic(self):
        """Two planar graphs can be isomorphic."""
        g1 = Graph(directed=False).add_cycle_graph(n=4)
        g2 = Graph(directed=False).add_cycle_graph(n=4)

        assert g1.is_planar()
        assert g2.is_planar()
        assert g1.is_isomorphic(g2)

    def test_non_planar_graphs_can_be_isomorphic(self):
        """Two non-planar graphs can be isomorphic."""
        k5_a = Graph(directed=False).add_full_graph(n=5)
        k5_b = Graph(directed=False).add_full_graph(n=5)

        assert not k5_a.is_planar()
        assert not k5_b.is_planar()
        assert k5_a.is_isomorphic(k5_b)

    def test_trees_are_planar(self):
        """All trees are planar."""
        for n in [5, 10, 20]:
            tree = Graph(directed=False).add_path_graph(n=n)
            assert tree.is_planar()
            assert tree.is_tree()


# Exact isomorphism, subgraph isomorphism and Weisfeiler-Lehman hashes, validated against
# NetworkX (is_isomorphic, GraphMatcher counts, WL hashes).


def _build(G, order=None):
    g = ec.Graph(directed=G.is_directed())
    g.add_nodes([str(n) for n in (order or sorted(G))])
    for u, v in G.edges():
        g.add_edge(str(u), str(v))
    return g


def _pairs(count=200):
    for seed in range(count):
        rnd = random.Random(seed)
        directed = seed % 3 == 0
        n = rnd.randint(1, 9)
        G = nx.gnp_random_graph(n, rnd.choice([0.2, 0.4, 0.6]), seed=seed, directed=directed)
        if seed % 5 == 0 and not directed:
            G = nx.MultiGraph(G)
            for u, v in list(G.edges())[:2]:
                G.add_edge(u, v)
        if seed % 7 == 0 and n:
            G.add_edge(0, 0)
        H = nx.relabel_nodes(G, dict(zip(G, rnd.sample(list(G), n))))
        if rnd.random() < 0.5 and isinstance(H, nx.Graph) and not H.is_multigraph():
            H = H.copy()
            if not directed and H.number_of_edges() >= 2:
                try:
                    nx.double_edge_swap(H, nswap=1, max_tries=100, seed=seed)
                except nx.NetworkXException:
                    pass
        yield G, H, rnd


class TestExactIsomorphism:
    def test_against_networkx(self):
        for G, H, rnd in _pairs():
            g, h = _build(G), _build(H, rnd.sample(sorted(H), len(H)))
            expected = nx.is_isomorphic(G, H)
            assert g.is_isomorphic(h) == expected
            mapping = g.find_isomorphism(h)
            assert (mapping is not None) == expected
            if mapping:
                assert sorted(mapping) == sorted(g.get_node_ids(limit=None))
                for _, s, t in g.get_edges_as_tuples():
                    assert h.find_edge_ids(mapping[s], mapping[t])

    def test_beats_invariant_checks(self):
        hexagon = ec.Graph().add_cycle_graph(6)
        two_triangles = ec.Graph().add_cycle_graph(3).add_cycle_graph(3)
        assert hexagon.could_be_isomorphic(two_triangles, properties="d")
        assert not hexagon.could_be_isomorphic(two_triangles)
        assert not hexagon.is_isomorphic(two_triangles)
        rook = nx.convert_node_labels_to_integers(
            nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
        )
        shrikhande = nx.Graph()
        for a in range(4):
            for b in range(4):
                for da, db in [(0, 1), (1, 0), (1, 1), (0, 3), (3, 0), (3, 3)]:
                    shrikhande.add_edge(4 * a + b, 4 * ((a + da) % 4) + (b + db) % 4)
        r, sh = _build(rook), _build(shrikhande)
        assert r.could_be_isomorphic(sh, properties="dt") and not r.could_be_isomorphic(sh)
        assert not r.is_isomorphic(sh)

    def test_attributes_and_directedness(self):
        def chain(elements):
            g = ec.Graph().add_nodes([(f"n{i}", {"el": e}) for i, e in enumerate(elements)])
            return g.add_edges([("n0", "n1"), ("n1", "n2")])

        water, other = chain("HOH"), chain("OHO")
        assert water.is_isomorphic(other) and not water.is_isomorphic(other, node_attr="el")
        a = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B", {"k": 1})])
        b = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B", {"k": 2})])
        assert a.is_isomorphic(b) and not a.is_isomorphic(b, edge_attr="k")
        d = ec.Graph(directed=True).add_path_graph(3)
        assert not d.is_isomorphic(ec.Graph().add_path_graph(3))
        assert d.find_isomorphism(ec.Graph().add_path_graph(3)) is None
        with pytest.raises(ValueError, match="letters"):
            d.could_be_isomorphic(d, properties="x")

    def test_large_and_regular_graphs(self):
        for G in (nx.random_regular_graph(3, 200, seed=1), nx.hypercube_graph(7)):
            G = nx.convert_node_labels_to_integers(G)
            order = sorted(G)
            random.Random(2).shuffle(order)
            assert _build(G).is_isomorphic(_build(G, order))


class TestSubgraphIsomorphism:
    def test_counts_match_networkx(self):
        for seed in range(150):
            rnd = random.Random(seed)
            directed = seed % 3 == 0
            T = nx.gnp_random_graph(rnd.randint(2, 9), 0.45, seed=seed, directed=directed)
            P = nx.gnp_random_graph(rnd.randint(1, 4), 0.6, seed=seed + 1000, directed=directed)
            t, p = _build(T), _build(P)
            matcher = nx_iso.DiGraphMatcher if directed else nx_iso.GraphMatcher
            induced = sum(1 for _ in matcher(T, P).subgraph_isomorphisms_iter())
            mono = sum(1 for _ in matcher(T, P).subgraph_monomorphisms_iter())
            assert len(t.subgraph_isomorphisms(p)) == induced
            assert len(t.subgraph_isomorphisms(p, induced=False)) == mono
            assert t.is_subgraph_isomorphic(p) == (induced > 0)
            assert t.is_subgraph_isomorphic(p, induced=False) == (mono > 0)

    def test_examples_limit_and_errors(self):
        triangle, path = ec.Graph().add_cycle_graph(3), ec.Graph().add_path_graph(3)
        assert not triangle.is_subgraph_isomorphic(path)
        assert triangle.is_subgraph_isomorphic(path, induced=False)
        assert len(triangle.subgraph_isomorphisms(triangle)) == 6
        assert (
            len(ec.Graph().add_cycle_graph(4).subgraph_isomorphisms(ec.Graph().add_cycle_graph(4)))
            == 8
        )
        assert len(ec.Graph().add_full_graph(5).subgraph_isomorphisms(triangle, limit=4)) == 4
        assert ec.Graph().add_full_graph(3).subgraph_isomorphisms(ec.Graph()) == [{}]
        with pytest.raises(ValueError, match="directed"):
            triangle.is_subgraph_isomorphic(ec.Graph(directed=True).add_path_graph(2))
        labeled = ec.Graph().add_nodes([("a", {"c": 1}), ("b", {"c": 2})]).add_edge("a", "b")
        pattern = ec.Graph().add_nodes([("x", {"c": 2}), ("y", {"c": 2})]).add_edge("x", "y")
        assert labeled.is_subgraph_isomorphic(pattern)
        assert not labeled.is_subgraph_isomorphic(pattern, node_attr="c")


def _aggregate_directed_fixed(G, node, node_labels, edge_attr=None):
    """NetworkX's directed WL aggregation with the fix of NetworkX PR #8946, which keeps the
    s_/p_ direction prefixes when edge_attr is given."""
    succ = [
        "s_" + ("" if edge_attr is None else str(G[node][nbr][edge_attr])) + node_labels[nbr]
        for nbr in G.successors(node)
    ]
    pred = [
        "p_" + ("" if edge_attr is None else str(G[nbr][node][edge_attr])) + node_labels[nbr]
        for nbr in G.predecessors(node)
    ]
    return node_labels[node] + "".join(sorted(succ)) + "".join(sorted(pred))


class TestWeisfeilerLehman:
    def test_hashes_match_networkx(self, monkeypatch):
        from networkx.algorithms import graph_hashing

        warnings.simplefilter("ignore", UserWarning)
        monkeypatch.setattr(
            graph_hashing, "_neighborhood_aggregate_directed", _aggregate_directed_fixed
        )
        for seed in range(120):
            rnd = random.Random(seed)
            directed = seed % 2 == 0
            G = nx.gnp_random_graph(
                rnd.randint(1, 11), rnd.choice([0.2, 0.4]), seed=seed, directed=directed
            )
            if seed % 5 == 0 and len(G):
                G.add_edge(0, 0)
            for n in G:
                G.nodes[n]["c"] = rnd.choice("xyz")
            for u, v in G.edges:
                G[u][v]["e"] = rnd.choice([1, 2])
            g = ec.Graph(directed=directed).add_nodes([(str(n), {"c": G.nodes[n]["c"]}) for n in G])
            if G.number_of_edges():
                g.add_edges([(str(u), str(v), {"e": G[u][v]["e"]}) for u, v in G.edges])
            label = {g.get_id_from_label(str(n)): n for n in G}
            for node_attr, edge_attr in [(None, None), ("c", None), (None, "e"), ("c", "e")]:
                for iterations in (1, 3):
                    assert g.weisfeiler_lehman_graph_hash(
                        node_attr, edge_attr, iterations
                    ) == nx.weisfeiler_lehman_graph_hash(
                        G, edge_attr=edge_attr, node_attr=node_attr, iterations=iterations
                    )
                    ours = g.weisfeiler_lehman_subgraph_hashes(
                        node_attr, edge_attr, iterations, include_initial_labels=iterations == 3
                    )
                    assert {label[k]: v for k, v in ours.items()} == (
                        nx.weisfeiler_lehman_subgraph_hashes(
                            G,
                            edge_attr=edge_attr,
                            node_attr=node_attr,
                            iterations=iterations,
                            include_initial_labels=iterations == 3,
                        )
                    )

    @pytest.mark.parametrize("node_attr", [None, "c"])
    def test_directed_edge_labels_keep_direction(self, node_attr):
        # NetworkX #8945: with edge_attr=, an out-star and an in-star used to collide
        stars = []
        for edges in ([("0", "1"), ("0", "2")], [("1", "0"), ("2", "0")]):
            g = ec.Graph(directed=True).add_nodes([(n, {"c": "x"}) for n in "012"])
            g.add_edges([(u, v, {"k": "e"}) for u, v in edges])
            stars.append(g)
        out_star, in_star = stars
        kw = {"node_attr": node_attr, "edge_attr": "k"}
        assert out_star.weisfeiler_lehman_graph_hash(**kw) != in_star.weisfeiler_lehman_graph_hash(
            **kw
        )
        center = out_star.get_id_from_label("0")
        assert (
            out_star.weisfeiler_lehman_subgraph_hashes(**kw)[center]
            != in_star.weisfeiler_lehman_subgraph_hashes(**kw)[in_star.get_id_from_label("0")]
        )
        renamed = ec.Graph(directed=True).add_nodes([(n, {"c": "x"}) for n in "abc"])
        renamed.add_edges([("c", "a", {"k": "e"}), ("c", "b", {"k": "e"})])
        assert renamed.weisfeiler_lehman_graph_hash(**kw) == out_star.weisfeiler_lehman_graph_hash(
            **kw
        )

    def test_invariance_collisions_and_errors(self):
        a = (
            ec.Graph()
            .add_nodes(list("ABCD"))
            .add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
        )
        b = (
            ec.Graph()
            .add_nodes(list("wxyz"))
            .add_edges([("z", "y"), ("y", "x"), ("x", "z"), ("w", "x")])
        )
        assert a.weisfeiler_lehman_graph_hash() == b.weisfeiler_lehman_graph_hash()
        hexagon = ec.Graph().add_cycle_graph(6)
        two_triangles = ec.Graph().add_cycle_graph(3).add_cycle_graph(3)
        assert (
            hexagon.weisfeiler_lehman_graph_hash() == two_triangles.weisfeiler_lehman_graph_hash()
        )
        assert len(a.weisfeiler_lehman_graph_hash(digest_size=8)) == 16
        with pytest.raises(ValueError, match="positive"):
            a.weisfeiler_lehman_graph_hash(iterations=0)
        with pytest.raises(ValueError, match="attribute"):
            a.weisfeiler_lehman_graph_hash(node_attr="missing")


class TestIsomorphismStepHooks:
    def test_steps(self):
        host = ec.Graph(directed=True).add_nodes(["x", "y", "z", "w"])
        host.add_edges([("x", "y"), ("y", "z"), ("x", "z"), ("z", "w"), ("y", "w")])
        ffl = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        ffl.add_edges([("A", "B"), ("B", "C"), ("A", "C")])
        host.subgraph_isomorphisms(ffl)
        step = host.steps[-1]
        assert step.kind == KIND_QUERY and step.narration == "2 occurrences of the pattern."
        assert sorted(step.delta.edges_selected) == [("x", "y"), ("x", "z"), ("y", "z")]
        path = ec.Graph().add_nodes(list("ABCDE"))
        path.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        path.weisfeiler_lehman_subgraph_hashes(iterations=2)
        groups = path.steps[-1].delta.node_groups
        assert groups["A"] == groups["E"] and groups["B"] == groups["D"] != groups["C"]
        path.is_isomorphic(path)
        assert path.steps[-1].narration == "is_isomorphic(): True."
