"""Tests for `modular_decomposition()` and `is_cograph()` (NetworkX PR #7957)."""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _graph(edges, nodes=()):
    """An undirected EdgeCase graph on string labels."""
    labels = list(dict.fromkeys([str(v) for v in nodes] + [str(x) for e in edges for x in e]))
    g = ec.Graph().add_nodes(labels)
    if edges:
        g.add_edges([(str(u), str(v)) for u, v in edges])
    return g


def _from_nx(G, order=None):
    return _graph(G.edges(), order if order is not None else G.nodes())


def _nodes(tree):
    """{tree label: attributes}."""
    return {
        tree.get_label_from_id(n): tree.get_node(n).attributes
        for n in tree.get_node_ids(limit=None)
    }


def _parents(tree):
    """{child label: parent label}."""
    out = {}
    for e in tree.get_edge_ids(limit=None):
        edge = tree.get_edge(e)
        out[tree.get_label_from_id(edge.target)] = tree.get_label_from_id(edge.source)
    return out


def _modules(tree):
    """{frozenset of member labels: (type, parent's member set or None)}."""
    attrs, parents = _nodes(tree), _parents(tree)
    return {
        frozenset(a["nodes"]): (
            a["type"],
            frozenset(attrs[parents[x]]["nodes"]) if x in parents else None,
        )
        for x, a in attrs.items()
    }


def _adjacency(g):
    labels = [g.get_label_from_id(i) for i in g.get_node_ids(limit=None)]
    adj = {v: set() for v in labels}
    for e in g.get_edge_ids(limit=None):
        edge = g.get_edge(e)
        s, t = g.get_label_from_id(edge.source), g.get_label_from_id(edge.target)
        if s != t:
            adj[s].add(t)
            adj[t].add(s)
    return adj


def _brute(g):
    """The strong modules straight from the definition, in the form of `_modules()`."""
    adj = _adjacency(g)
    V = frozenset(adj)

    def is_module(S):
        return all(len({v in adj[x] for v in S}) <= 1 for x in V - S)

    mods = [
        frozenset(S)
        for k in range(1, len(V) + 1)
        for S in itertools.combinations(sorted(V), k)
        if is_module(frozenset(S))
    ]
    strong = [M for M in mods if all(M <= N or N <= M or not M & N for N in mods)]

    def kind(M):
        if len(M) == 1:
            return "leaf"
        H = nx.Graph()
        H.add_nodes_from(M)
        H.add_edges_from((a, b) for a in M for b in adj[a] if b in M)
        if not nx.is_connected(H):
            return "parallel"
        if not nx.is_connected(nx.complement(H)):
            return "series"
        return "prime"

    return {
        M: (kind(M), min((N for N in strong if M < N), key=len, default=None)) for M in strong
    }


def _random_graph(seed, n_max=8):
    rng = random.Random(seed)
    n = rng.randint(1, n_max)
    p = rng.random()
    edges = [(a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < p]
    return _graph(edges, range(n))


def _p4_free(g):
    adj = _adjacency(g)
    for quad in itertools.permutations(adj, 4):
        a, b, c, d = quad
        pairs = {(a, b), (b, c), (c, d)}
        if all((y in adj[x]) == ((x, y) in pairs) for x, y in itertools.combinations(quad, 2)):
            return False
    return True


def _random_cograph(rng, labels):
    """A random cograph on `labels` (unions and joins), as a NetworkX graph."""
    if len(labels) == 1:
        G = nx.Graph()
        G.add_node(labels[0])
        return G
    cut = rng.randint(1, len(labels) - 1)
    A, B = _random_cograph(rng, labels[:cut]), _random_cograph(rng, labels[cut:])
    G = nx.union(A, B)
    if rng.random() < 0.5:
        G.add_edges_from(itertools.product(A, B))
    return G


def _edge_types(tree):
    """The PR's view of the tree: (parent type, child type or leaf label) pairs."""
    attrs = _nodes(tree)

    def name(x):
        return x if attrs[x]["type"] == "leaf" else attrs[x]["type"]

    return {(name(p), name(c)) for c, p in _parents(tree).items()}


# The example from the PR (Goeppel, "Efficient Implementation of Modular Graph
# Decomposition", chapter 3)
PR_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    (4, 5), (4, 6), (4, 7), (5, 8), (6, 8), (7, 8),
]


# =============================================================================
# Known answers (ported from the PR)
# =============================================================================


class TestKnownAnswers:
    def test_pr_example(self):
        tree = _graph(PR_EDGES).modular_decomposition()
        assert _nodes(tree)["M1"]["type"] == "prime"
        assert _edge_types(tree) == {
            ("prime", "series"), ("prime", "parallel"), ("prime", "4"), ("prime", "8"),
            ("series", "0"), ("series", "1"), ("series", "2"), ("series", "3"),
            ("parallel", "5"), ("parallel", "6"), ("parallel", "7"),
        }

    def test_pr_example_labels_and_order(self):
        tree = _graph(PR_EDGES).modular_decomposition()
        assert tree.get_edges_str_repr(use_labels=True) == [
            "M1->M2", "M2->0", "M2->1", "M2->2", "M2->3", "M1->4",
            "M1->M3", "M3->5", "M3->6", "M3->7", "M1->8",
        ]
        attrs = _nodes(tree)
        assert attrs["M2"] == {"type": "series", "nodes": ["0", "1", "2", "3"]}
        assert attrs["M3"] == {"type": "parallel", "nodes": ["5", "6", "7"]}
        assert attrs["4"] == {"type": "leaf", "nodes": ["4"]}

    def test_pivot_independence(self):
        # NetworkX's pivot picker cannot change the tree; here no insertion order can either
        G = nx.Graph(PR_EDGES)
        want = _modules(_from_nx(G).modular_decomposition())
        rng = random.Random(1)
        for _ in range(10):
            order = list(G)
            rng.shuffle(order)
            assert _modules(_from_nx(G, order).modular_decomposition()) == want

    def test_docstring_example(self):
        g = ec.Graph().add_nodes(list("ABCDEFGHI"))
        g.add_edges([(a, b) for a in "ABCD" for b in "ABCDE" if a < b])
        g.add_edges([("E", x) for x in "FGH"] + [(x, "I") for x in "FGH"])
        attrs = _nodes(g.modular_decomposition())
        assert attrs["M1"]["type"] == "prime"
        assert attrs["M2"] == {"type": "series", "nodes": list("ABCD")}
        assert attrs["M3"] == {"type": "parallel", "nodes": list("FGH")}

    def test_cograph_docstring_example(self):
        k23 = _graph([(u, v) for u in "ab" for v in "xyz"])
        path = _graph([("A", "B"), ("B", "C"), ("C", "D")])
        assert (k23.is_cograph(), path.is_cograph()) == (True, False)


# =============================================================================
# Against the definition
# =============================================================================


class TestAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(150))
    def test_strong_modules(self, seed):
        g = _random_graph(seed)
        assert _modules(g.modular_decomposition()) == _brute(g)

    @pytest.mark.parametrize("seed", range(80))
    def test_is_cograph_is_p4_free(self, seed):
        g = _random_graph(500 + seed, n_max=7)
        assert g.is_cograph() == _p4_free(g)

    @pytest.mark.parametrize("seed", range(30))
    def test_random_cographs(self, seed):
        rng = random.Random(seed)
        labels = [f"v{i}" for i in range(rng.randint(1, 40))]
        rng.shuffle(labels)
        G = _random_cograph(rng, labels)
        g = _from_nx(G)
        assert g.is_cograph()
        # The cotree describes the graph: two nodes are adjacent iff they split at a series node
        tree = g.modular_decomposition()
        attrs, parents = _nodes(tree), _parents(tree)

        def ancestors(x):
            out = [x]
            while out[-1] in parents:
                out.append(parents[out[-1]])
            return out

        for a, b in itertools.combinations(labels, 2):
            up = set(ancestors(a))
            lca = next(x for x in ancestors(b) if x in up)
            assert (attrs[lca]["type"] == "series") == G.has_edge(a, b)

    @pytest.mark.parametrize("seed", range(10))
    def test_larger_random_graphs_structure(self, seed):
        G = nx.gnm_random_graph(300, 600 + 300 * seed, seed=seed)
        g = _from_nx(G)
        tree = g.modular_decomposition()
        attrs, parents = _nodes(tree), _parents(tree)
        children = {}
        for c, p in parents.items():
            children.setdefault(p, []).append(c)
        assert sum(a["type"] == "leaf" for a in attrs.values()) == 300
        adj = _adjacency(g)
        for x, a in attrs.items():
            members = set(a["nodes"])
            # every module is a module
            for out in set(adj) - members:
                assert len({v in adj[out] for v in members}) == 1
            if a["type"] != "leaf":
                kids = children[x]
                assert len(kids) >= 2
                assert sorted(v for k in kids for v in attrs[k]["nodes"]) == sorted(members)


# =============================================================================
# Structure and edge cases
# =============================================================================


class TestEdgeCases:
    def test_empty_graph(self):
        tree = ec.Graph().modular_decomposition()
        assert tree.count_nodes() == 0
        assert tree.is_directed()
        assert ec.Graph().is_cograph() is True

    def test_single_node(self):
        tree = _graph([], ["x"]).modular_decomposition()
        assert _nodes(tree) == {"x": {"type": "leaf", "nodes": ["x"]}}
        assert tree.count_edges() == 0

    @pytest.mark.parametrize(
        "G, root",
        [
            (nx.complete_graph(5), "series"),
            (nx.empty_graph(5), "parallel"),
            (nx.path_graph(4), "prime"),
            (nx.cycle_graph(5), "prime"),
            (nx.complete_bipartite_graph(2, 3), "series"),
            (nx.path_graph(3), "series"),
        ],
    )
    def test_root_types(self, G, root):
        tree = _from_nx(G).modular_decomposition()
        attrs = _nodes(tree)
        assert attrs["M1"]["type"] == root
        assert attrs["M1"]["nodes"] == sorted(str(v) for v in G)

    def test_two_level_cotree(self):
        # K_{2,3}: a series root over the parallel modules {a, b} and {x, y, z}
        tree = _graph([(u, v) for u in "ab" for v in "xyz"]).modular_decomposition()
        assert tree.get_edges_str_repr(use_labels=True) == [
            "M1->M2", "M2->a", "M2->b", "M1->M3", "M3->x", "M3->y", "M3->z",
        ]

    def test_preorder_labels(self):
        G = nx.disjoint_union(nx.complete_graph(3), nx.path_graph(4))
        tree = _from_nx(G).modular_decomposition()
        # parallel root, then the series triangle (M2) before the prime path (M3)
        assert tree.get_edges_str_repr(use_labels=True) == [
            "M1->M2", "M2->0", "M2->1", "M2->2", "M1->M3", "M3->3", "M3->4", "M3->5", "M3->6",
        ]

    def test_self_loops_and_parallel_edges_ignored(self):
        plain = _graph(PR_EDGES).modular_decomposition()
        noisy = _graph(PR_EDGES + [(4, 4), (0, 0), (1, 0), (5, 8)]).modular_decomposition()
        assert _modules(noisy) == _modules(plain)

    def test_label_clash(self):
        tree = _graph([("M1", "M2")], ["x"]).modular_decomposition()
        attrs = _nodes(tree)
        assert set(attrs) == {"MM1", "MM2", "M1", "M2", "x"}
        assert attrs["MM1"]["type"] == "parallel"
        assert attrs["MM2"] == {"type": "series", "nodes": ["M1", "M2"]}

    def test_deep_threshold_graph(self):
        # each new node is isolated or dominating: the cotree is a path of depth n - 1
        g = ec.Graph().add_nodes([str(i) for i in range(60)])
        g.add_edges([(str(i), str(j)) for i in range(1, 60, 2) for j in range(i)])
        tree = g.modular_decomposition()
        assert tree.count_nodes() == 60 + 59
        assert g.is_cograph()

    def test_deterministic(self):
        g = _random_graph(7)
        a, b = g.modular_decomposition(), g.modular_decomposition()
        assert _nodes(a) == _nodes(b)
        assert a.get_edges_str_repr(use_labels=True) == b.get_edges_str_repr(use_labels=True)


# =============================================================================
# Validation and steps
# =============================================================================


class TestValidationAndSteps:
    def test_directed_raises(self):
        g = ec.Graph(directed=True).add_nodes(["a", "b"]).add_edges([("a", "b")])
        with pytest.raises(ValueError, match="undirected"):
            g.modular_decomposition()
        with pytest.raises(ValueError, match=r"is_cograph\(\) requires an undirected"):
            g.is_cograph()

    def test_does_not_modify_graph(self):
        g = _graph(PR_EDGES)
        before = (g.count_nodes(), g.count_edges())
        g.modular_decomposition()
        g.is_cograph()
        assert (g.count_nodes(), g.count_edges()) == before

    def test_steps(self):
        g = _graph(PR_EDGES)
        g.modular_decomposition()
        step = g.steps[-1]
        assert step.code == "G.modular_decomposition()"
        assert step.result == "prime"
        assert step.narration == (
            "modular_decomposition(): 3 modules (1 prime, 1 series, 1 parallel) over 9 leaves;"
            " the root is prime."
        )
        assert g.is_cograph() is False
        assert g.steps[-1].narration == "is_cograph(): False."
        one = _graph([], ["x"])
        one.modular_decomposition()
        assert one.steps[-1].narration == (
            "modular_decomposition(): a single node, so the tree is one leaf."
        )
        empty = ec.Graph()
        empty.modular_decomposition()
        assert empty.steps[-1].narration == "modular_decomposition(): the graph is empty."
