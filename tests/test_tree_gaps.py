"""Tests for the tree additions: constrained spanning trees and arborescences
(include=/exclude=), iter_spanning_trees(), iter_spanning_arborescences(), the additive
random_spanning_tree(), to/from_nested_tuple(), join_trees() and junction_tree(), checked
against NetworkX."""

import random
from collections import Counter
from itertools import combinations, islice

import edgecase as ec
import networkx as nx
import pytest
from networkx.algorithms.chordal import chordal_graph_cliques, complete_to_chordal_graph
from networkx.algorithms.tree import ArborescenceIterator, EdgePartition, SpanningTreeIterator


def weighted_pair(n: int, edges: list, directed: bool = False):
    """The same weighted graph in both libraries: nodes "1".."n", edges `(u, v, w)` on 0..n-1."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    g = ec.Graph(directed=directed, weighted=True).add_nodes([str(i + 1) for i in range(n)])
    for u, v, w in edges:
        if G.has_edge(u, v) or u == v:
            continue
        G.add_edge(u, v, weight=w)
        g.add_edge(str(u + 1), str(v + 1), weight=w)
    return G, g


def random_weighted(count: int, seed: int, directed: bool = False, connected: bool = True):
    rng = random.Random(seed)
    made = 0
    while made < count:
        n = rng.randint(1, 7)
        edges = [
            (u, v, float(rng.randint(1, 6)))
            for u in range(n)
            for v in range(n)
            if u != v and (directed or u < v) and rng.random() < 0.55
        ]
        G, g = weighted_pair(n, edges, directed)
        if connected and not directed and n and not nx.is_connected(G):
            continue
        made += 1
        yield G, g


def total(t: ec.Graph) -> float:
    return sum(t.get_edge(e).weight for e in t.get_edge_ids(limit=None))


def pairs(t: ec.Graph, directed: bool = False) -> frozenset:
    out = []
    for e in t.get_edge_ids(limit=None):
        s, d = t.get_node_ids_by_edge_id(e)
        a, b = t.get_label_from_id(s), t.get_label_from_id(d)
        out.append((a, b) if directed else frozenset((a, b)))
    return frozenset(out)


def nx_pairs(T, directed: bool = False) -> frozenset:
    return frozenset(
        (str(u + 1), str(v + 1)) if directed else frozenset((str(u + 1), str(v + 1)))
        for u, v in T.edges()
    )


# ---------------------------------------------------------------------------------------------
# Constrained spanning trees
# ---------------------------------------------------------------------------------------------


class TestPartitionSpanningTree:
    @pytest.mark.parametrize("minimum", [True, False])
    def test_weights_match_networkx(self, minimum):
        rng = random.Random(3)
        for G, g in random_weighted(60, seed=11 if minimum else 12, connected=False):
            edges = list(G.edges())
            include, exclude = [], []
            for u, v in edges:
                r = rng.random()
                if r < 0.15:
                    include.append((u, v))
                elif r < 0.3:
                    exclude.append((u, v))
            # NetworkX needs an acyclic include set too
            if include and not nx.is_forest(nx.Graph(include)):
                include = include[:1]
            H = G.copy()
            for u, v in include:
                H[u][v]["partition"] = EdgePartition.INCLUDED
            for u, v in exclude:
                H[u][v]["partition"] = EdgePartition.EXCLUDED
            T = nx.algorithms.tree.mst.partition_spanning_tree(H, minimum=minimum)
            spec = lambda es: [(str(u + 1), str(v + 1)) for u, v in es]  # noqa: E731
            method = g.minimum_spanning_tree if minimum else g.maximum_spanning_tree
            t = method(include=spec(include), exclude=spec(exclude))
            assert t.count_edges() == T.number_of_edges()
            assert total(t) == pytest.approx(T.size(weight="weight"))
            assert nx_pairs(nx.Graph(include)) <= pairs(t)
            assert not (nx_pairs(nx.Graph(exclude)) & pairs(t))

    def test_unconstrained_matches_plain_kruskal(self):
        for _, g in random_weighted(30, seed=5, connected=False):
            assert g.minimum_spanning_tree(include=[], return_subgraph=False) == (
                g.minimum_spanning_tree(return_subgraph=False)
            )

    def test_edge_ids_and_pairs(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 9.0)])
        (diag,) = g.find_edge_ids("A", "C")
        a, c = g.get_id_from_label("A"), g.get_id_from_label("C")
        edges = g.minimum_spanning_tree(include=[diag], return_subgraph=False)
        assert (a, c, 9.0) in edges and len(edges) == 2
        assert g.minimum_spanning_tree(include=[str(diag)], return_subgraph=False) == edges
        assert g.minimum_spanning_tree(include=[("C", "A")], return_subgraph=False) == edges

    def test_directed_pairs_match_either_way(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 9.0)])
        t = g.minimum_spanning_tree(include=[("A", "C")])
        assert ("C", "A") in pairs(t, directed=True)

    def test_exclude_can_disconnect(self):
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        assert g.minimum_spanning_tree(exclude=[("A", "B")]).count_edges() == 1

    def test_errors(self):
        g = ec.Graph().add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(ValueError, match="cycle"):
            g.minimum_spanning_tree(include=[("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(ValueError, match="both included and excluded"):
            g.minimum_spanning_tree(include=[("A", "B")], exclude=[("B", "A")])
        with pytest.raises(ValueError, match="missing edge"):
            g.add_node("D")
            g.minimum_spanning_tree(include=[("A", "D")])
        with pytest.raises(ValueError, match="not found"):
            g.minimum_spanning_tree(exclude=[99])
        with pytest.raises(TypeError):
            g.minimum_spanning_tree(include=[1.5])
        loop = ec.Graph().add_nodes(["A"]).add_edges([("A", "A")])
        with pytest.raises(ValueError, match="cycle"):
            loop.minimum_spanning_tree(include=[("A", "A")])
        nan = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        with pytest.raises(ValueError, match="NaN"):  # rejected on entry
            nan.add_edges([("A", "B", float("nan")), ("B", "C", 1.0)])


# ---------------------------------------------------------------------------------------------
# Spanning tree iteration
# ---------------------------------------------------------------------------------------------


class TestIterSpanningTrees:
    @pytest.mark.parametrize("minimum", [True, False])
    def test_weight_order_matches_networkx(self, minimum):
        for G, g in random_weighted(40, seed=21 if minimum else 22):
            if len(G) < 2:
                continue
            expected = [T.size(weight="weight") for T in SpanningTreeIterator(G, minimum=minimum)]
            got = [total(t) for t in g.iter_spanning_trees(minimum=minimum)]
            assert got == pytest.approx(expected)

    def test_every_tree_once(self):
        for G, g in random_weighted(30, seed=23):
            trees = [pairs(t) for t in g.iter_spanning_trees()]
            assert len(trees) == len(set(trees)) == g.number_of_spanning_trees(weighted=False)
            assert all(len(t) == len(G) - 1 for t in trees)

    def test_constraints_filter_the_full_list(self):
        rng = random.Random(4)
        for G, g in random_weighted(30, seed=24):
            edges = list(G.edges())
            if len(edges) < 2:
                continue
            inc, exc = rng.sample(edges, 2)
            inc_pair = frozenset((str(inc[0] + 1), str(inc[1] + 1)))
            exc_pair = frozenset((str(exc[0] + 1), str(exc[1] + 1)))
            full = [pairs(t) for t in g.iter_spanning_trees()]
            kept = [
                pairs(t)
                for t in g.iter_spanning_trees(include=[tuple(inc_pair)], exclude=[tuple(exc_pair)])
            ]
            assert len(kept) == len(set(kept))
            assert set(kept) == {t for t in full if inc_pair in t and exc_pair not in t}
            weights = [total(t) for t in g.iter_spanning_trees(include=[tuple(inc_pair)])]
            assert weights == sorted(weights)

    def test_parallel_edges_are_different_trees(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("A", "B", 2.0), ("B", "C", 1.0), ("A", "A", 0.0)])
        trees = list(g.iter_spanning_trees(return_subgraphs=False))
        assert len(trees) == 2 and sorted(w for t in trees for _, _, w in t) == [1.0, 1.0, 1.0, 2.0]

    def test_lazy_and_small_cases(self):
        k = ec.Graph().add_full_graph(9)  # 9^7 trees: must not enumerate them all
        assert len(list(islice(k.iter_spanning_trees(), 5))) == 5
        one = ec.Graph().add_nodes(["A"])
        assert [t.count_edges() for t in one.iter_spanning_trees()] == [0]
        g = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B")])
        assert list(g.iter_spanning_trees(exclude=[("A", "B")])) == []

    def test_errors(self):
        split = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B")])
        with pytest.raises(ValueError, match="disconnected"):
            split.iter_spanning_trees()
        nan = ec.Graph(weighted=True).add_nodes(["A", "B"])
        with pytest.raises(ValueError, match="NaN"):  # rejected on entry
            nan.add_edge("A", "B", weight=float("nan"))


# ---------------------------------------------------------------------------------------------
# Arborescences
# ---------------------------------------------------------------------------------------------


class TestArborescences:
    @pytest.mark.parametrize("minimum", [True, False])
    def test_iteration_matches_networkx(self, minimum):
        count = skipped = 0
        for G, g in random_weighted(80, seed=31 if minimum else 32, directed=True):
            if not any(
                nx.is_arborescence(nx.bfs_tree(G, r)) and len(nx.bfs_tree(G, r)) == len(G)
                for r in G
            ):
                with pytest.raises(ValueError):
                    g.iter_spanning_arborescences()
                continue
            if len(G) < 2:
                continue
            n_arb = sum(g.number_of_spanning_trees(root=str(r + 1), weighted=False) for r in G)
            if n_arb > 400:  # NetworkX is slow here
                continue
            count += 1
            theirs = list(ArborescenceIterator(G, minimum=minimum))
            ours = list(g.iter_spanning_arborescences(minimum=minimum))
            got = [total(t) for t in ours]
            assert got == sorted(got, reverse=not minimum)
            assert len(ours) == round(n_arb)
            # NetworkX's iterator can skip arborescences on dense graphs (3890 of 3900 on one
            # 7-node case here), so its output must be contained in ours, and equal when complete
            assert {nx_pairs(T, True) for T in theirs} <= {pairs(t, True) for t in ours}
            if len(theirs) == len(ours):
                assert got == pytest.approx([T.size(weight="weight") for T in theirs])
            else:
                skipped += 1
        assert count > 10
        assert skipped < count / 4

    def test_every_arborescence_once(self):
        for G, g in random_weighted(60, seed=33, directed=True):
            if len(G) < 2 or not any(len(nx.descendants(G, r)) == len(G) - 1 for r in G):
                continue
            trees = [pairs(t, True) for t in g.iter_spanning_arborescences()]
            expected = sum(g.number_of_spanning_trees(root=str(r + 1), weighted=False) for r in G)
            assert len(trees) == len(set(trees)) == expected
            for t in g.iter_spanning_arborescences():
                assert t.is_arborescence()

    def test_constrained_optimum_is_best_of_filtered(self):
        rng = random.Random(9)
        for G, g in random_weighted(60, seed=34, directed=True):
            if len(G) < 3 or not any(len(nx.descendants(G, r)) == len(G) - 1 for r in G):
                continue
            edges = list(G.edges())
            inc, exc = rng.sample(edges, 2)
            inc_pair = (str(inc[0] + 1), str(inc[1] + 1))
            exc_pair = (str(exc[0] + 1), str(exc[1] + 1))
            feasible = [
                total(t)
                for t in g.iter_spanning_arborescences()
                if inc_pair in pairs(t, True) and exc_pair not in pairs(t, True)
            ]
            constrained = [
                total(t)
                for t in g.iter_spanning_arborescences(include=[inc_pair], exclude=[exc_pair])
            ]
            assert constrained == pytest.approx(sorted(feasible))
            if feasible:
                best = g.minimum_spanning_arborescence(include=[inc_pair], exclude=[exc_pair])
                assert total(best) == pytest.approx(min(feasible))
                worst = g.maximum_spanning_arborescence(include=[inc_pair], exclude=[exc_pair])
                assert total(worst) == pytest.approx(max(feasible))
            else:
                with pytest.raises(ValueError, match="included edge"):
                    g.minimum_spanning_arborescence(include=[inc_pair], exclude=[exc_pair])

    def test_branching_include_forces_negative_edges(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 2.0), ("B", "C", -1.0)])
        assert pairs(g.maximum_branching(), True) == {("A", "B")}
        assert pairs(g.maximum_branching(include=[("B", "C")]), True) == {
            ("A", "B"),
            ("B", "C"),
        }
        assert pairs(g.maximum_branching(exclude=[("A", "B")]), True) == frozenset()
        g.add_edge("A", "C", weight=5.0)
        with pytest.raises(ValueError, match="branching"):
            g.minimum_branching(include=[("B", "C"), ("A", "C")])

    def test_init_partition_matches_networkx(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 3), (2, 3, 6), (3, 2, 1)]
        )
        _, g = weighted_pair(4, [(u, v, d["weight"]) for u, v, d in G.edges(data=True)], True)
        for inc, exc in [([(0, 1)], []), ([], [(2, 1)]), ([(2, 3)], [(1, 3)])]:
            theirs = ArborescenceIterator(G, init_partition=(inc, exc))
            spec = lambda es: [(str(u + 1), str(v + 1)) for u, v in es]  # noqa: E731
            ours = g.iter_spanning_arborescences(include=spec(inc), exclude=spec(exc))
            assert [pairs(t, True) for t in ours] == [nx_pairs(T, True) for T in theirs]
        forced = g.minimum_spanning_arborescence(include=[("1", "2")])
        assert total(forced) == 8.0 and ("1", "2") in pairs(forced, True)

    def test_errors(self):
        und = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B")])
        with pytest.raises(ValueError, match="directed"):
            und.iter_spanning_arborescences()
        with pytest.raises(ValueError, match="directed"):
            und.maximum_spanning_arborescence(include=[("A", "B")])


# ---------------------------------------------------------------------------------------------
# weight_attr="weight"
# ---------------------------------------------------------------------------------------------


class TestWeightAttrNamedWeight:
    """`weight_attr="weight"` reads the weight field, as `None` does (it used to count every
    edge as 1.0, since the weight field isn't an attribute)."""

    @pytest.mark.parametrize(
        "name",
        [
            "minimum_spanning_tree",
            "maximum_spanning_tree",
            "minimum_branching",
            "maximum_branching",
        ],
    )
    def test_undirected_and_directed(self, name):
        directed = "branching" in name
        for _, g in random_weighted(30, seed=41, directed=directed):
            method = getattr(g, name)
            assert total(method(weight_attr="weight")) == pytest.approx(total(method()))

    @pytest.mark.parametrize("minimum", [True, False])
    def test_arborescences(self, minimum):
        checked = 0
        for G, g in random_weighted(60, seed=42, directed=True):
            if len(G) < 2 or not any(len(nx.descendants(G, r)) == len(G) - 1 for r in G):
                continue
            checked += 1
            method = g.minimum_spanning_arborescence if minimum else g.maximum_spanning_arborescence
            assert total(method(weight_attr="weight")) == pytest.approx(total(method()))
            ours = [total(t) for t in g.iter_spanning_arborescences(minimum, weight_attr="weight")]
            assert ours == pytest.approx([total(t) for t in g.iter_spanning_arborescences(minimum)])
        assert checked > 10

    def test_spanning_tree_lists(self):
        for _, g in random_weighted(20, seed=43):
            got = [total(t) for t in g.iter_spanning_trees(weight_attr="weight")]
            assert got == pytest.approx([total(t) for t in g.iter_spanning_trees()])
            edges = g.all_spanning_trees(weight_attr="weight", return_subgraphs=False)
            assert edges == g.all_spanning_trees(return_subgraphs=False)


# ---------------------------------------------------------------------------------------------
# Additive random spanning trees
# ---------------------------------------------------------------------------------------------


def additive_distribution(G: nx.Graph) -> dict:
    trees = list(SpanningTreeIterator(G))
    weights = {nx_pairs(T): T.size(weight="weight") for T in trees}
    norm = sum(weights.values())
    return {k: v / norm for k, v in weights.items()}


class TestAdditiveRandomSpanningTree:
    def test_distribution(self):
        rng = random.Random(2)
        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.choice([0.0, 1.0, 2.0, 5.0])
        _, g = weighted_pair(4, [(u, v, d["weight"]) for u, v, d in G.edges(data=True)])
        expected = additive_distribution(G)
        draws = 20000
        seen = Counter(
            pairs(g.random_spanning_tree(seed=s, multiplicative=False)) for s in range(draws)
        )
        assert set(seen) <= {k for k, p in expected.items() if p > 0}
        for k, p in expected.items():
            # four standard errors
            assert abs(seen[k] / draws - p) < 4 * (p * (1 - p) / draws) ** 0.5 + 1e-9

    def test_differs_from_multiplicative(self):
        tri = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        tri.add_edges([("A", "B", 4.0), ("B", "C", 4.0), ("C", "A", 1.0)])
        n = 6000
        add = sum(
            tri.random_spanning_tree(seed=s, multiplicative=False).find_edge_ids("C", "A") == []
            for s in range(n)
        )
        assert abs(add / n - 8 / 18) < 0.025
        mul = sum(tri.random_spanning_tree(seed=s).find_edge_ids("C", "A") == [] for s in range(n))
        assert abs(mul / n - 2 / 3) < 0.025

    def test_multigraph_and_reproducibility(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("A", "B", 3.0), ("B", "C", 2.0), ("C", "C", 7.0)])
        # Trees {AB1, BC} (3) and {AB3, BC} (5), plus {AB1, AB3} is not a tree
        n = 8000
        heavy_id = g.find_edge_ids("A", "B")[1]
        hits = sum(
            heavy_id in g.random_spanning_tree(seed=s, multiplicative=False).get_edge_ids()
            for s in range(n)
        )
        assert abs(hits / n - 5 / 8) < 0.03
        a = g.random_spanning_tree(seed=3, multiplicative=False).get_edge_ids()
        b = g.random_spanning_tree(seed=3, multiplicative=False).get_edge_ids()
        assert sorted(a) == sorted(b)

    def test_zero_weights_still_connect(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 0.0), ("B", "C", 1.0)])
        assert g.random_spanning_tree(seed=1, multiplicative=False).count_edges() == 2
        with pytest.raises(ValueError, match="connected"):
            g.random_spanning_tree(seed=1)
        zero = ec.Graph(weighted=True).add_nodes(["A", "B"])
        zero.add_edge("A", "B", weight=0.0)
        with pytest.raises(ValueError, match="zero"):
            zero.random_spanning_tree(multiplicative=False)

    def test_small_and_unweighted(self):
        one = ec.Graph(weighted=True).add_nodes(["A"])
        assert one.random_spanning_tree(multiplicative=False).count_edges() == 0
        grid = ec.Graph().add_grid_2d(3, 3)
        assert grid.random_spanning_tree(seed=1, multiplicative=False).is_tree()
        with pytest.raises(ValueError, match="negative"):
            neg = ec.Graph(weighted=True).add_nodes(["A", "B"])
            neg.add_edge("A", "B", weight=-1.0)
            neg.random_spanning_tree(multiplicative=False)


# ---------------------------------------------------------------------------------------------
# Nested tuples and joins
# ---------------------------------------------------------------------------------------------


def random_tree(n: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    T = nx.Graph()
    T.add_node(0)
    for v in range(1, n):
        T.add_edge(rng.randrange(v), v)
    return T


def sorted_nested(T: nx.Graph, root, parent=None) -> tuple:
    """`nx.to_nested_tuple()` with children visited in ascending order."""
    return tuple(sorted_nested(T, v, root) for v in sorted(set(T[root]) - {parent}))


def tree_pair(T: nx.Graph) -> ec.Graph:
    g = ec.Graph().add_nodes([str(v) for v in sorted(T)])
    g.add_edges([(str(u), str(v)) for u, v in T.edges()])
    return g


class TestNestedTuple:
    def test_to_nested_tuple_matches_networkx(self):
        for seed in range(80):
            T = random_tree(random.Random(seed).randint(1, 25), seed)
            g = tree_pair(T)
            root = random.Random(seed + 1).randrange(len(T))
            # NetworkX takes children in set order; ours are in ascending node-ID order
            assert g.to_nested_tuple(str(root)) == sorted_nested(T, root)
            assert g.to_nested_tuple(str(root), canonical_form=True) == nx.to_nested_tuple(
                T, root, canonical_form=True
            )

    def test_round_trip_and_isomorphism(self):
        for seed in range(60):
            T = random_tree(random.Random(seed).randint(1, 30), seed)
            code = nx.to_nested_tuple(T, 0)
            for sensible in (False, True):
                g = ec.Graph.from_nested_tuple(code, sensible_relabeling=sensible)
                assert g.to_nested_tuple("1") == code
                F = nx.from_nested_tuple(code, sensible_relabeling=sensible)
                if sensible:
                    assert pairs(g) == nx_pairs(F)
                assert g.to_nested_tuple("1", canonical_form=True) == nx.to_nested_tuple(
                    F, 0, canonical_form=True
                )

    def test_depth_first_labels(self):
        g = ec.Graph.from_nested_tuple(((), ((), ())))
        assert pairs(g) == {frozenset(p) for p in [("1", "2"), ("1", "3"), ("3", "4"), ("3", "5")]}
        g = ec.Graph.from_nested_tuple(((((),), ()), ()))
        assert sorted(g.get_edges_str_repr(use_labels=True)) == sorted(
            ["1--2", "2--3", "3--4", "2--5", "1--6"]
        )

    def test_canonical_form_identifies_rooted_shapes(self):
        a = ec.Graph.from_nested_tuple(((), ((), ())))
        b = ec.Graph.from_nested_tuple((((), ()), ()))
        assert a.to_nested_tuple("1") != b.to_nested_tuple("1")
        assert a.to_nested_tuple("1", canonical_form=True) == b.to_nested_tuple(
            "1", canonical_form=True
        )

    def test_deep_path_has_no_recursion_limit(self):
        code = ()
        for _ in range(3000):
            code = (code,)
        g = ec.Graph.from_nested_tuple(code)
        assert g.count_nodes() == 3001
        top = g.to_nested_tuple("1")
        depth = 0
        while top:
            (top,) = top
            depth += 1
        assert depth == 3000

    def test_errors(self):
        cycle = ec.Graph().add_nodes(["A", "B", "C"])
        cycle.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(ValueError, match="not a tree"):
            cycle.to_nested_tuple("A")
        with pytest.raises(ValueError):
            cycle.to_nested_tuple("Z")
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).to_nested_tuple("A")
        with pytest.raises(TypeError):
            ec.Graph.from_nested_tuple(((), "x"))
        assert ec.Graph.from_nested_tuple(()).count_nodes() == 1


class TestJoinTrees:
    def test_matches_networkx_labels(self):
        for seed in range(40):
            rng = random.Random(seed)
            nx_trees, ec_trees = [], []
            for k in range(rng.randint(1, 4)):
                T = random_tree(rng.randint(1, 8), seed * 10 + k)
                root = rng.randrange(len(T))
                nx_trees.append((T, root))
                ec_trees.append((tree_pair(T), str(root)))
            J = nx.join_trees(nx_trees, label_attribute="old")
            g = ec.Graph.join_trees(ec_trees, label_attribute="old", first_label=0)
            assert pairs(g) == {frozenset(map(str, e)) for e in J.edges()}
            for n in g.get_node_ids(limit=None):
                label = g.get_label_from_id(n)
                attrs = g.get_node(n).attributes
                if label == "0":
                    assert not attrs
                else:
                    assert attrs["old"] == str(J.nodes[int(label)]["old"])

    def test_default_labels_follow_ids(self):
        a = ec.Graph().add_nodes(["a", "b"]).add_edges([("a", "b")])
        j = ec.Graph.join_trees([(a, "b"), (a, "a")])
        assert all(j.get_label_from_id(n) == str(n) for n in j.get_node_ids(limit=None))
        assert pairs(j) == {frozenset(p) for p in [("2", "3"), ("4", "5"), ("1", "3"), ("1", "4")]}

    def test_copies_attributes_weights_and_direction(self):
        t = ec.Graph(directed=True, weighted=True).add_nodes([("r", {"color": "red"}), "s"])
        t.add_edge("r", "s", weight=2.5, kind="link")
        j = ec.Graph.join_trees([(t, "r")])
        assert j.is_directed() and j.is_weighted()
        assert j.get_node(2).attributes == {"color": "red"}
        assert t.count_nodes() == 2 and t.count_edges() == 1  # the input is untouched
        assert sorted(j.get_edges_str_repr(use_labels=True)) == ["1->2", "2->3"]
        (e,) = j.find_edge_ids("2", "3")
        assert j.get_edge(e).weight == 2.5 and j.get_edge(e).attributes == {"kind": "link"}
        (r,) = j.find_edge_ids("1", "2")
        assert j.get_edge(r).weight == 1.0

    def test_balanced_trees(self):
        half = ec.Graph.from_nested_tuple((((), ()), ((), ())))
        whole = ec.Graph.join_trees([(half, "1"), (half, "1")])
        B = nx.balanced_tree(2, 3)
        assert whole.to_nested_tuple("1", canonical_form=True) == nx.to_nested_tuple(
            B, 0, canonical_form=True
        )

    def test_empty_and_errors(self):
        j = ec.Graph.join_trees([])
        assert j.count_nodes() == 1 and j.get_label_from_id(1) == "1"
        a = ec.Graph().add_nodes(["a"])
        with pytest.raises(ValueError):
            ec.Graph.join_trees([(a, "zz")])
        d = ec.Graph(directed=True).add_nodes(["a"])
        with pytest.raises(ValueError, match="directed"):
            ec.Graph.join_trees([(a, "a"), (d, "a")])
        with pytest.raises(TypeError):
            ec.Graph.join_trees([("not a graph", "a")])


# ---------------------------------------------------------------------------------------------
# Junction trees
# ---------------------------------------------------------------------------------------------


def nx_clique_tree_pairs(G):
    """NetworkX's junction tree as the set of joined clique pairs (before sepsets merge)."""
    if G.is_directed():
        G = nx.moral_graph(G)
    C, _ = complete_to_chordal_graph(G)
    cliques = [tuple(sorted(c)) for c in chordal_graph_cliques(C)]
    CG = nx.Graph()
    CG.add_nodes_from(cliques)
    for a, b in combinations(cliques, 2):
        if set(a) & set(b):
            CG.add_edge(a, b, weight=len(set(a) & set(b)))
    T = nx.maximum_spanning_tree(CG)
    key = lambda c: frozenset(str(v) for v in c)  # noqa: E731
    return {key(c) for c in cliques}, {frozenset((key(a), key(b))) for a, b in T.edges()}


def junction_parts(jt: ec.Graph):
    members = {
        jt.get_label_from_id(n): frozenset(jt.get_node(n).attributes["nodes"])
        for n in jt.get_node_ids(limit=None)
    }
    neighbors: dict = {}
    for e in jt.get_edges_str_repr(use_labels=True):
        a, b = e.split("--")
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    cliques = {v for k, v in members.items() if k.startswith("C")}
    joined = set()
    for k, ns in neighbors.items():
        if k.startswith("S"):
            assert len(ns) == 2
            a, b = (members[x] for x in ns)
            assert members[k] == a & b
            joined.add(frozenset((a, b)))
    return cliques, joined


class TestJunctionTree:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        rng = random.Random(7 + directed)
        for _ in range(120):
            G = nx.gnp_random_graph(
                rng.randint(1, 10),
                rng.choice([0.2, 0.35, 0.5]),
                seed=rng.randrange(2**31),
                directed=directed,
            )
            g = ec.Graph(directed=directed).add_nodes([str(v) for v in G])
            g.add_edges([(str(u), str(v)) for u, v in G.edges()])
            cliques, joined = junction_parts(g.junction_tree())
            nx_cliques, nx_joined = nx_clique_tree_pairs(G)
            assert cliques == nx_cliques
            assert joined == nx_joined

    def test_running_intersection_and_shape(self):
        rng = random.Random(1)
        for _ in range(60):
            G = nx.gnp_random_graph(rng.randint(2, 12), 0.3, seed=rng.randrange(2**31))
            g = ec.Graph().add_nodes([str(v) for v in G])
            g.add_edges([(str(u), str(v)) for u, v in G.edges()])
            jt = g.junction_tree()
            n_cliques = sum(
                1 for n in jt.get_node_ids(limit=None) if jt.get_node(n).label[0] == "C"
            )
            assert jt.count_nodes() - jt.count_edges() == nx.number_connected_components(G)
            assert jt.count_nodes() == 2 * n_cliques - nx.number_connected_components(G)
            for v in G:
                holding = [
                    n
                    for n in jt.get_node_ids(limit=None)
                    if str(v) in jt.get_node(n).attributes["nodes"]
                ]
                assert jt.subgraph(holding).is_connected()

    def test_types_and_example(self):
        bn = ec.Graph(directed=True).add_nodes(["D", "I", "G", "S", "L"])
        bn.add_edges([("D", "G"), ("I", "G"), ("I", "S"), ("G", "L")])
        jt = bn.junction_tree()
        nodes = {jt.get_label_from_id(n): jt.get_node(n) for n in jt.get_node_ids(limit=None)}
        assert {k: v.attributes["nodes"] for k, v in nodes.items()} == {
            "C1": ["D", "I", "G"],
            "C2": ["I", "S"],
            "C3": ["G", "L"],
            "S1": ["I"],
            "S2": ["G"],
        }
        df = jt.get_node_df()
        types = dict(zip(df["label"].to_list(), df["type"].to_list()))
        assert types == {"C1": "clique", "C2": "clique", "C3": "clique"} | {
            "S1": "sepset",
            "S2": "sepset",
        }

    def test_isolated_nodes_and_empty(self):
        g = ec.Graph().add_nodes(["A", "B"])
        jt = g.junction_tree()
        assert jt.count_nodes() == 2 and jt.count_edges() == 0
        assert ec.Graph().junction_tree().count_nodes() == 0
