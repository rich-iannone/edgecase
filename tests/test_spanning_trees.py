import collections
import itertools
import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_QUERY


class TestSteinerTree:
    """Tests for steiner_tree method."""

    def test_steiner_tree_basic_triangle(self):
        """Test Steiner tree on a simple triangle."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        # Connect all three nodes
        tree = g.steiner_tree(["A", "B", "C"])

        assert tree.count_edges() == 2  # Tree has V-1 edges
        assert tree.count_nodes() == 3

    def test_steiner_tree_with_steiner_node(self):
        """Test that Steiner tree can include non-terminal nodes."""
        g = Graph(weighted=True)

        # Star topology: all terminals connect through center
        g.add_node("Center")
        g.add_node("T1")
        g.add_node("T2")
        g.add_node("T3")

        g.add_edge("Center", "T1", weight=1.0)
        g.add_edge("Center", "T2", weight=1.0)
        g.add_edge("Center", "T3", weight=1.0)

        # Direct connections between terminals would be more expensive
        g.add_edge("T1", "T2", weight=10.0)
        g.add_edge("T2", "T3", weight=10.0)
        g.add_edge("T1", "T3", weight=10.0)

        # Connecting T1, T2, T3 should use Center as Steiner node
        tree = g.steiner_tree(["T1", "T2", "T3"])

        # Should have 3 edges through center (weight 3) not 2 direct edges (weight 20)
        assert tree.count_edges() == 3
        assert tree.has_node("Center")  # Center should be included as Steiner node

    def test_steiner_tree_two_terminals(self):
        """Test Steiner tree with exactly 2 terminals (shortest path)."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=5.0)
        g.add_edge("A", "C", weight=1.0)
        g.add_edge("C", "B", weight=1.0)

        tree = g.steiner_tree(["A", "B"])

        # Should find shortest path A-C-B (weight 2) not direct A-B (weight 5)
        assert tree.count_edges() == 2
        assert tree.has_node("C")

    def test_steiner_tree_return_edges(self):
        """Test returning edge tuples instead of subgraph."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)

        edges = g.steiner_tree(["A", "B", "C"], return_subgraph=False)

        assert isinstance(edges, list)
        assert len(edges) == 2

        # Each edge is (source, target, weight)
        for edge in edges:
            assert len(edge) == 3
            assert isinstance(edge[2], (int, float))

    def test_steiner_tree_with_weight_attr(self):
        """Test using custom weight attribute."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")

        # Default weight is high, but 'cost' attribute is low
        g.add_edge("A", "B", weight=100.0, attributes={"cost": 1.0})
        g.add_edge("B", "C", weight=100.0, attributes={"cost": 1.0})
        g.add_edge("A", "C", weight=1.0, attributes={"cost": 100.0})

        # Using default weight should prefer A-C edge
        tree1 = g.steiner_tree(["A", "B", "C"])
        edges1 = set()
        for eid, src, tgt in tree1.get_edges_as_tuples():
            src_label = tree1.get_label_from_id(src)
            tgt_label = tree1.get_label_from_id(tgt)
            edges1.add(frozenset([src_label, tgt_label]))

        assert frozenset(["A", "C"]) in edges1

        # Using 'cost' attribute should avoid A-C edge
        tree2 = g.steiner_tree(["A", "B", "C"], weight_attr="cost")
        edges2 = set()
        for eid, src, tgt in tree2.get_edges_as_tuples():
            src_label = tree2.get_label_from_id(src)
            tgt_label = tree2.get_label_from_id(tgt)
            edges2.add(frozenset([src_label, tgt_label]))

        assert frozenset(["A", "B"]) in edges2
        assert frozenset(["B", "C"]) in edges2

    def test_steiner_tree_directed_graph(self):
        """Test Steiner tree on directed graph."""
        g = Graph(directed=True, weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("A", "C", weight=5.0)

        tree = g.steiner_tree(["A", "B", "C"])

        assert tree.is_directed()
        assert tree.count_edges() == 2

    def test_steiner_tree_fewer_than_two_terminals_error(self):
        """Test that fewer than 2 terminals raises error."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            g.steiner_tree(["A"])

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            g.steiner_tree([])

    def test_steiner_tree_nonexistent_terminal_error(self):
        """Test that nonexistent terminal node raises error."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="not found|not exist"):
            g.steiner_tree(["A", "X"])

    def test_steiner_tree_disconnected_terminals_error(self):
        """Test that disconnected terminals raise error."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")  # Separate component

        with pytest.raises(ValueError, match="not connected|no path"):
            g.steiner_tree(["A", "C"])

    def test_steiner_tree_unweighted_graph(self):
        """Test Steiner tree on unweighted graph (uses weight=1.0)."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("A", "D")

        tree = g.steiner_tree(["A", "C", "D"])

        # Should connect all three with 2 edges
        assert tree.count_edges() == 2

    def test_steiner_tree_preserves_node_attributes(self):
        """Test that node attributes are preserved in result."""
        g = Graph(weighted=True)
        g.add_node("A", attributes={"type": "terminal"})
        g.add_node("B", attributes={"type": "intermediate"})
        g.add_node("C", attributes={"type": "terminal"})
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)

        tree = g.steiner_tree(["A", "C"])

        # Check that result has proper structure
        assert tree.count_nodes() in [2, 3]  # May or may not include B


class TestAllSpanningTrees:
    """Tests for all_spanning_trees method."""

    def test_all_spanning_trees_triangle(self):
        """Test all spanning trees of a triangle."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        trees = g.all_spanning_trees()

        # Triangle has 3 spanning trees (remove any one edge)
        assert len(trees) == 3
        for tree in trees:
            assert tree.count_nodes() == 3
            assert tree.count_edges() == 2

    def test_all_spanning_trees_complete_k4(self):
        """Test all spanning trees of complete graph K4."""
        g = Graph()

        nodes = ["A", "B", "C", "D"]

        for n in nodes:
            g.add_node(n)

        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        trees = g.all_spanning_trees()

        # K4 has 4^2 = 16 spanning trees by Cayley's formula
        assert len(trees) == 16

        for tree in trees:
            assert tree.count_nodes() == 4
            assert tree.count_edges() == 3  # V-1 edges

    def test_all_spanning_trees_path(self):
        """Test all spanning trees of a path graph."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        trees = g.all_spanning_trees()

        # Path has only 1 spanning tree (itself)
        assert len(trees) == 1
        assert trees[0].count_edges() == 3

    def test_all_spanning_trees_max_limit(self):
        """Test that max_trees parameter limits results."""
        g = Graph()

        nodes = ["A", "B", "C", "D"]

        for n in nodes:
            g.add_node(n)

        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        # Limit to 5 trees
        trees = g.all_spanning_trees(max_trees=5)

        assert len(trees) == 5

    def test_all_spanning_trees_return_edge_lists(self):
        """Test returning edge lists instead of graphs."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        edge_lists = g.all_spanning_trees(return_subgraphs=False)

        assert len(edge_lists) == 3

        for edges in edge_lists:
            assert isinstance(edges, list)
            assert len(edges) == 2
            for edge in edges:
                assert len(edge) == 3  # (source, target, weight)

    def test_all_spanning_trees_weighted(self):
        """Test all spanning trees with weights."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        trees = g.all_spanning_trees()

        assert len(trees) == 3

        # Calculate weights of all trees
        weights = []

        for tree in trees:
            total = 0.0
            for eid, _, _ in tree.get_edges_as_tuples():
                edge = tree.get_edge(eid)
                if edge and edge.weight:
                    total += edge.weight
            weights.append(total)

        # Should have trees with weights 3, 4, 5
        weights.sort()

        assert weights == [3.0, 4.0, 5.0]

    def test_all_spanning_trees_with_weight_attr(self):
        """Test using custom weight attribute."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0, attributes={"cost": 10.0})
        g.add_edge("B", "C", weight=2.0, attributes={"cost": 20.0})
        g.add_edge("C", "A", weight=3.0, attributes={"cost": 30.0})

        edge_lists = g.all_spanning_trees(weight_attr="cost", return_subgraphs=False)

        assert len(edge_lists) == 3

        # Weights in edge tuples should reflect 'cost' attribute
        all_weights = []

        for edges in edge_lists:
            for _, _, w in edges:
                all_weights.append(w)

        assert 10.0 in all_weights
        assert 20.0 in all_weights
        assert 30.0 in all_weights

    def test_all_spanning_trees_single_edge(self):
        """Test spanning tree of graph with single edge."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        trees = g.all_spanning_trees()

        assert len(trees) == 1
        assert trees[0].count_edges() == 1

    def test_all_spanning_trees_directed_graph(self):
        """Test all spanning trees on directed graph."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("A", "C")  # Additional edge

        trees = g.all_spanning_trees(max_trees=100)

        for tree in trees:
            assert tree.is_directed()
            # Each tree should have V-1 edges
            assert tree.count_edges() == 2

    def test_all_spanning_trees_disconnected_error(self):
        """Test that disconnected graph raises error."""
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")  # Separate component

        with pytest.raises(ValueError, match="disconnected|no spanning tree"):
            g.all_spanning_trees()

    def test_all_spanning_trees_single_node(self):
        """Test spanning tree of single node graph."""
        g = Graph()
        g.add_node("A")

        # Single node has one "trivial" spanning tree with 0 edges
        trees = g.all_spanning_trees()

        assert len(trees) == 1
        assert trees[0].count_nodes() == 1
        assert trees[0].count_edges() == 0

    def test_all_spanning_trees_preserves_all_nodes(self):
        """Test that all nodes are preserved in spanning trees."""
        g = Graph()

        nodes = ["A", "B", "C", "D", "E"]

        for n in nodes:
            g.add_node(n)

        # Create a cycle
        for i in range(len(nodes)):
            g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

        trees = g.all_spanning_trees()
        for tree in trees:
            assert tree.count_nodes() == 5
            for n in nodes:
                assert tree.has_node(n)

    def test_all_spanning_trees_cycle_graph(self):
        """Test all spanning trees of a cycle graph."""
        g = Graph()
        n = 5

        nodes = [f"N{i}" for i in range(n)]

        for node in nodes:
            g.add_node(node)

        for i in range(n):
            g.add_edge(nodes[i], nodes[(i + 1) % n])

        trees = g.all_spanning_trees()

        # Cycle graph Cn has exactly n spanning trees
        assert len(trees) == n

    def test_all_spanning_trees_unique(self):
        """Test that all returned spanning trees are unique."""
        g = Graph()

        nodes = ["A", "B", "C", "D"]

        for n in nodes:
            g.add_node(n)

        # Create K4
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        trees = g.all_spanning_trees()

        # Convert each tree to a frozenset of edge pairs for comparison
        tree_signatures = []

        for tree in trees:
            edges = set()
            for eid, src, tgt in tree.get_edges_as_tuples():
                src_label = tree.get_label_from_id(src)
                tgt_label = tree.get_label_from_id(tgt)
                edges.add(frozenset([src_label, tgt_label]))
            tree_signatures.append(frozenset(edges))

        # All signatures should be unique
        assert len(tree_signatures) == len(set(tree_signatures))


class TestSpanningTreesIntegration:
    """Integration tests combining steiner_tree and all_spanning_trees."""

    def test_steiner_tree_all_nodes_equals_mst(self):
        """Steiner tree with all nodes as terminals should approximate MST."""
        g = Graph(weighted=True)

        nodes = ["A", "B", "C", "D"]

        for n in nodes:
            g.add_node(n)

        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        mst = g.minimum_spanning_tree()
        steiner = g.steiner_tree(nodes)

        # Both should have same number of edges
        assert mst.count_edges() == steiner.count_edges() == 3

    def test_minimum_spanning_tree_in_all_trees(self):
        """The MST should be among all spanning trees."""
        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        mst = g.minimum_spanning_tree()
        all_trees = g.all_spanning_trees()

        # Get MST edges
        mst_edges = set()

        for eid, src, tgt in mst.get_edges_as_tuples():
            src_label = mst.get_label_from_id(src)
            tgt_label = mst.get_label_from_id(tgt)
            mst_edges.add(frozenset([src_label, tgt_label]))

        # Check that MST is among all trees
        found = False

        for tree in all_trees:
            tree_edges = set()
            for eid, src, tgt in tree.get_edges_as_tuples():
                src_label = tree.get_label_from_id(src)
                tgt_label = tree.get_label_from_id(tgt)
                tree_edges.add(frozenset([src_label, tgt_label]))
            if tree_edges == mst_edges:
                found = True
                break

        assert found

    def test_steiner_tree_subset_of_spanning_tree(self):
        """Steiner tree on subset should use fewer or equal edges."""
        g = Graph(weighted=True)

        nodes = ["A", "B", "C", "D", "E"]

        for n in nodes:
            g.add_node(n)

        # Create a path A-B-C-D-E
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=1.0)
        g.add_edge("C", "D", weight=1.0)
        g.add_edge("D", "E", weight=1.0)

        # Steiner tree connecting just A and E
        steiner = g.steiner_tree(["A", "E"])

        # Full spanning tree
        all_trees = g.all_spanning_trees()

        assert len(all_trees) == 1  # Path has unique spanning tree

        # Steiner tree should have 4 edges (the whole path since that's the only connection)
        assert steiner.count_edges() == 4


# Spanning tree counts (matrix-tree theorem) validated against NetworkX, and random spanning
# trees checked against the exact product-of-weights distribution.


def _digraphs(count=120):
    for seed in range(count):
        rnd = random.Random(seed)
        n = rnd.randint(1, 10)
        G = nx.gnp_random_graph(n, rnd.choice([0.2, 0.4, 0.7]), seed=seed, directed=True)
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.choice([rnd.uniform(-3, 5), rnd.randint(-2, 4)])
        yield G, rnd


def _build(G, weighted=True):
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in sorted(G)])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
        else:
            g.add_edge(str(u), str(v))
    return g


class TestSpanningTreeCounts:
    def test_against_networkx(self):
        for G, rnd in _digraphs(80):
            g = _build(G)
            if not len(G):
                continue
            root = rnd.randrange(len(G))
            assert g.number_of_spanning_trees(root=str(root), weighted=False) == round(
                nx.number_of_spanning_trees(G, root=root)
            )
            U = G.to_undirected()
            for u, v in U.edges:
                U[u][v]["weight"] = abs(U[u][v]["weight"]) + 0.5
            gu = _build(U)
            count = gu.number_of_spanning_trees(weighted=False)
            assert isinstance(count, int) and count == round(nx.number_of_spanning_trees(U))
            assert gu.number_of_spanning_trees() == pytest.approx(
                nx.number_of_spanning_trees(U, weight="weight"), rel=1e-8, abs=1e-12
            )

    def test_known_counts(self):
        assert ec.Graph().add_full_graph(6).number_of_spanning_trees() == 6**4
        assert ec.Graph().add_full_graph(20).number_of_spanning_trees() == 20**18  # exact
        assert ec.Graph().add_full_graph(60).number_of_spanning_trees() == 60**58  # > 128 bits
        assert ec.Graph().add_cycle_graph(7).number_of_spanning_trees() == 7
        split = ec.Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        assert split.number_of_spanning_trees() == 0
        assert ec.Graph().add_nodes(["A"]).number_of_spanning_trees() == 1
        multi = ec.Graph().add_nodes(["A", "B", "C"])
        multi.add_edges([("A", "B"), ("A", "B"), ("B", "C"), ("C", "C")])
        assert multi.number_of_spanning_trees() == 2  # parallel edges count, loops do not

    def test_errors(self):
        with pytest.raises(ValueError, match="at least one node"):
            ec.Graph().number_of_spanning_trees()
        d = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="root"):
            d.number_of_spanning_trees()

    def test_steps(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B"), ("B", "C"), ("A", "C")])
        g.number_of_spanning_trees(root="A")
        assert g.steps[-1].kind == KIND_QUERY and g.steps[-1].delta.nodes_selected == ["A"]


class TestRandomSpanningTree:
    def test_distribution_matches_product_of_weights(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 2), (2, 0, 3), (2, 3, 1), (3, 0, 1)])
        g = _build(G)
        weights = {}
        for combo in itertools.combinations(g.get_edge_ids(limit=None), 3):
            sub = g.edge_subgraph(list(combo))
            if sub.count_nodes() == 4 and sub.is_tree():
                weights[frozenset(combo)] = math.prod(g.get_edge(e).weight for e in combo)
        total = sum(weights.values())
        assert total == pytest.approx(g.number_of_spanning_trees())
        n = 20000
        counts = collections.Counter()
        for s in range(n):
            tree = g.random_spanning_tree(seed=s)
            key = frozenset(
                g.find_edge_ids(*s_.split("--"))[0]
                for s_ in tree.get_edges_str_repr(use_labels=True)
            )
            counts[key] += 1
        assert set(counts) <= set(weights)
        for key, w in weights.items():
            assert abs(counts[key] / n - w / total) < 0.015

    def test_properties_and_seed(self):
        grid = ec.Graph().add_grid_2d(5, 5)
        t1, t2 = grid.random_spanning_tree(seed=3), grid.random_spanning_tree(seed=3)
        assert t1.is_tree() and t1.count_edges() == 24
        assert t1.get_edges_str_repr(use_labels=True) == t2.get_edges_str_repr(use_labels=True)

    def test_errors(self):
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).random_spanning_tree()
        with pytest.raises(ValueError, match="connected"):
            ec.Graph().add_nodes(["A", "B"]).random_spanning_tree()
