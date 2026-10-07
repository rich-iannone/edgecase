"""Tests for DAG methods: ancestors, descendants, topological_generations,
dag_longest_path(_length), condensation and lowest_common_ancestor."""

import itertools
import random
import warnings

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_QUERY, KIND_TRANSFORM

# Validated against NetworkX on random DAGs and checked on hand-built pipelines.


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


def _random_dags():
    for seed in range(15):
        rnd = random.Random(seed)
        D = nx.gnp_random_graph(14, 0.25, seed=seed, directed=True)
        D = nx.DiGraph([(u, v) for u, v in D.edges if u < v])
        D.add_nodes_from(range(14))
        for u, v in D.edges:
            D[u][v]["weight"] = rnd.choice([1, 2, 3])
        for weighted in (False, True):
            yield D, _build(D, weighted), weighted


def _pipeline() -> ec.Graph:
    return (
        ec.Graph(directed=True)
        .add_nodes(["fetch", "compile", "test", "package", "docs"])
        .add_edges(
            [
                ("fetch", "compile"),
                ("compile", "test"),
                ("test", "package"),
                ("compile", "package"),
                ("fetch", "docs"),
            ]
        )
    )


def _two_triangles() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _labels(g: ec.Graph, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


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


class TestDagAgainstNetworkX:
    def test_ancestors_descendants_any_graph(self):
        for G, g, _ in _random_cases():
            for x in list(G.nodes)[:5]:
                assert {_lab(g, k) for k in g.ancestors(str(x))} == nx.ancestors(G, x)
                assert {_lab(g, k) for k in g.descendants(str(x))} == nx.descendants(G, x)

    def test_generations_longest_path_lca(self):
        for D, d, weighted in _random_dags():
            gens = [sorted(_lab(d, x) for x in gen) for gen in d.topological_generations()]
            assert gens == [sorted(x) for x in nx.topological_generations(D)]
            path = [_lab(d, x) for x in d.dag_longest_path()]
            assert all(D.has_edge(a, b) for a, b in zip(path, path[1:]))
            ref = nx.dag_longest_path_length(
                D, weight="weight" if weighted else "unused", default_weight=1
            )
            assert d.dag_longest_path_length() == pytest.approx(ref)
            for a, b in itertools.islice(itertools.combinations(range(14), 2), 0, 91, 4):
                ours = d.lowest_common_ancestor(str(a), str(b))
                if nx.lowest_common_ancestor(D, a, b) is None:
                    assert ours is None
                    continue
                common = (nx.ancestors(D, a) | {a}) & (nx.ancestors(D, b) | {b})
                o = _lab(d, ours)
                assert o in common and not any(c in common for c in D.successors(o))

    def test_condensation(self):
        for G, g, _ in _random_cases():
            if not G.is_directed():
                continue
            ours = g.condensation()
            ref = nx.condensation(G)
            members = sorted(
                sorted(int(x) for x in m) for m in ours.get_node_attrs("members").values()
            )
            assert members == sorted(sorted(ref.nodes[k]["members"]) for k in ref)
            assert ours.count_edges() == ref.number_of_edges()
            assert ours.is_dag()
            # nodes were added in topological order
            assert all(s < t for _, s, t in ours.get_edges_as_tuples())


class TestDag:
    def test_pipeline(self):
        g = _pipeline()
        assert _labels(g, g.ancestors("package")) == ["fetch", "compile", "test"]
        assert _labels(g, g.descendants("compile")) == ["test", "package"]
        gens = [_labels(g, gen) for gen in g.topological_generations()]
        assert gens == [["fetch"], ["compile", "docs"], ["test"], ["package"]]
        assert _labels(g, g.dag_longest_path()) == ["fetch", "compile", "test", "package"]
        assert g.dag_longest_path_length() == 3

    def test_weighted_longest_path(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B", 1.0), ("B", "D", 1.0), ("A", "C", 5.0), ("C", "D", 0.5)])
        assert _labels(g, g.dag_longest_path()) == ["A", "C", "D"]
        assert g.dag_longest_path_length() == 5.5

    def test_cycles_and_undirected_rejected(self):
        cyc = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        loop = ec.Graph(directed=True).add_nodes(["A"]).add_edge("A", "A")
        for g in (cyc, loop):
            for method in ("topological_generations", "dag_longest_path"):
                with pytest.raises(ValueError, match="cycle"):
                    getattr(g, method)()
            with pytest.raises(ValueError, match="cycle"):
                g.lowest_common_ancestor("A", "A")
        und = _two_triangles()
        for method in ("topological_generations", "dag_longest_path", "condensation"):
            with pytest.raises(ValueError, match="directed"):
                getattr(und, method)()

    def test_ancestors_on_undirected_is_component(self):
        g = _two_triangles().add_node("Z")
        assert _labels(g, g.ancestors("A")) == ["B", "C", "D", "E", "F"]
        assert g.descendants("Z") == []

    def test_empty_and_edgeless(self):
        assert ec.Graph(directed=True).dag_longest_path() == []
        assert ec.Graph(directed=True).topological_generations() == []
        g = ec.Graph(directed=True).add_nodes(["A", "B"])
        assert len(g.dag_longest_path()) == 1 and g.dag_longest_path_length() == 0

    def test_lca_taxonomy_and_self_ancestry(self):
        t = ec.Graph(directed=True).add_nodes(["M", "C", "R", "cat", "dog", "mouse"])
        t.add_edges([("M", "C"), ("M", "R"), ("C", "cat"), ("C", "dog"), ("R", "mouse")])
        lab = t.get_label_from_id
        assert lab(t.lowest_common_ancestor("cat", "dog")) == "C"
        assert lab(t.lowest_common_ancestor("cat", "mouse")) == "M"
        assert lab(t.lowest_common_ancestor("C", "dog")) == "C"
        two_roots = ec.Graph(directed=True).add_nodes(["A", "B"])
        assert two_roots.lowest_common_ancestor("A", "B") is None

    def test_condensation_labels_and_attributes(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCDE"))
        g.add_edges([("A", "B"), ("B", "A"), ("B", "C"), ("C", "D"), ("D", "C")])
        dag = g.condensation()
        df = dag.get_node_df()
        assert df["label"].to_list() == ["{A,B}", "{C,D}", "E"]
        assert df["members"].to_list() == [["A", "B"], ["C", "D"], ["E"]]
        assert df["size"].to_list() == [2, 2, 1]
        assert dag.get_edges_str_repr(use_labels=True) == ["{A,B}->{C,D}"]
        assert g.count_nodes() == 5  # original untouched

    def test_dag_steps(self):
        g = _pipeline()
        g.ancestors("package")
        assert g.steps[-1].delta.nodes_selected == ["fetch", "compile", "test"]
        g.topological_generations()
        assert g.steps[-1].delta.node_groups == {
            "fetch": 0,
            "compile": 1,
            "docs": 1,
            "test": 2,
            "package": 3,
        }
        g.dag_longest_path()
        assert g.steps[-1].narration == "Longest path: fetch → compile → test → package (length 3)."
        g.lowest_common_ancestor("test", "docs")
        assert g.steps[-1].delta.nodes_selected == ["fetch"]

    def test_condensation_records_on_both_graphs(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC")).add_edges([("A", "B"), ("B", "A")])
        dag = g.condensation()
        step = g.steps[-1]
        assert step.kind == KIND_TRANSFORM
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 1}
        assert len(dag.steps) == 1 and dag.steps[0].delta.nodes_added == ["{A,B}", "C"]


# Dominators and d-separation, validated against NetworkX.


def _digraph(G):
    g = ec.Graph(directed=True).add_nodes([str(n) for n in G])
    for u, v in G.edges:
        g.add_edge(str(u), str(v))
    return g


class TestDominators:
    def test_against_networkx(self):
        for seed in range(100):
            rnd = random.Random(seed)
            D = nx.gnp_random_graph(rnd.randint(1, 12), 0.25, seed=seed, directed=True)
            g = _digraph(D)
            lab = {g.get_id_from_label(str(n)): n for n in D}
            idom = {lab[a]: lab[b] for a, b in g.immediate_dominators("0").items()}
            assert idom == nx.immediate_dominators(D, 0)
            df = {lab[a]: {lab[x] for x in b} for a, b in g.dominance_frontiers("0").items()}
            assert df == nx.dominance_frontiers(D, 0)

    def test_errors_and_steps(self):
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_path_graph(3).immediate_dominators(1)
        g = ec.Graph(directed=True).add_nodes(["entry", "a", "b", "merge"])
        g.add_edges([("entry", "a"), ("entry", "b"), ("a", "merge"), ("b", "merge")])
        idom = g.immediate_dominators("entry")
        assert idom[g.get_id_from_label("merge")] == g.get_id_from_label("entry")
        assert g.steps[-1].kind == KIND_QUERY
        assert g.dominance_frontiers("entry")[g.get_id_from_label("a")] == {
            g.get_id_from_label("merge")
        }


class TestDSeparation:
    def test_against_networkx(self):
        for seed in range(80):
            rnd = random.Random(seed)
            n = rnd.randint(3, 10)
            D = nx.DiGraph()
            D.add_nodes_from(range(n))
            D.add_edges_from(
                (u, v) for u, v in itertools.combinations(range(n), 2) if rnd.random() < 0.3
            )
            g = _digraph(D)
            for _ in range(10):
                x, y = rnd.sample(range(n), 2)
                rest = [v for v in range(n) if v not in (x, y)]
                z = set(rnd.sample(rest, rnd.randint(0, len(rest))))
                ids = lambda s: [str(v) for v in s]  # noqa: E731
                assert g.is_d_separator(str(x), str(y), ids(z)) == nx.is_d_separator(D, x, y, z)
                assert g.is_minimal_d_separator(str(x), str(y), ids(z)) == (
                    nx.is_minimal_d_separator(D, x, y, z)
                )
                ref = nx.find_minimal_d_separator(D, x, y)
                ours = g.find_minimal_d_separator(str(x), str(y))
                if ref is None:
                    assert ours is None
                else:
                    assert {int(g.get_label_from_id(v)) for v in ours} == ref
                    assert g.is_minimal_d_separator(str(x), str(y), ours)

    def test_examples_and_errors(self):
        g = ec.Graph(directed=True).add_nodes(["talent", "luck", "success"])
        g.add_edges([("talent", "success"), ("luck", "success")])
        assert g.is_d_separator("talent", "luck", [])
        assert not g.is_d_separator("talent", "luck", "success")
        with pytest.raises(ValueError, match="disjoint"):
            g.is_d_separator("talent", "luck", "talent")
        cyclic = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="acyclic"):
            cyclic.is_d_separator("A", "B", [])
        g.find_minimal_d_separator("talent", "luck")
        assert g.steps[-1].delta.node_groups == {"talent": 0, "luck": 1}


def _random_dag(seed, n):
    rnd = random.Random(seed)
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    D.add_edges_from((u, v) for u, v in itertools.combinations(range(n), 2) if rnd.random() < 0.3)
    g = ec.Graph(directed=True).add_nodes([str(x) for x in D])
    for u, v in D.edges:
        g.add_edge(str(u), str(v))
    return D, g


class TestDagExtras:
    """antichains, colliders and v-structures, lexicographic topological order, DAG to
    branching, all-pairs LCA, moral graph and aperiodicity, validated against NetworkX."""

    def test_against_networkx(self):
        warnings.simplefilter("ignore", DeprecationWarning)
        for seed in range(100):
            D, g = _random_dag(seed, random.Random(seed).randint(1, 8))
            lab = lambda x: int(g.get_label_from_id(x))  # noqa: E731
            topo = list(nx.topological_sort(D))
            assert [[lab(x) for x in a] for a in g.antichains([str(x) for x in topo])] == list(
                nx.antichains(D, topo)
            )
            assert sorted(tuple(map(lab, c)) for c in g.colliders()) == sorted(nx.dag.colliders(D))
            assert sorted(tuple(map(lab, c)) for c in g.v_structures()) == sorted(
                nx.dag.v_structures(D)
            )
            ds = nx.relabel_nodes(D, {n: str(n) for n in D})
            assert [g.get_label_from_id(x) for x in g.lexicographical_topological_sort()] == list(
                nx.lexicographical_topological_sort(ds)
            )
            branching, ref = g.dag_to_branching(), nx.dag_to_branching(D)
            assert (branching.count_nodes(), branching.count_edges()) == (
                ref.number_of_nodes(),
                ref.number_of_edges(),
            )
            moral = g.moral_graph()
            assert {
                frozenset(map(int, e.split("--")))
                for e in moral.get_edges_str_repr(use_labels=True)
            } == {frozenset(e) for e in nx.moral_graph(D).edges}
            if len(D):
                ours = {
                    (lab(a), lab(b)): lab(c)
                    for (a, b), c in g.all_pairs_lowest_common_ancestor().items()
                }
                assert set(ours) == set(dict(nx.all_pairs_lowest_common_ancestor(D)))
                anc = {n: nx.ancestors(D, n) | {n} for n in D}
                for (u, v), c in ours.items():
                    common = anc[u] & anc[v]
                    assert c in common and not any(s in common for s in D.successors(c))

    def test_aperiodic_and_errors(self):
        for seed in range(60):
            G = nx.gnp_random_graph(
                random.Random(seed).randint(2, 8), 0.4, seed=seed, directed=True
            )
            if not nx.is_strongly_connected(G):
                continue
            g = ec.Graph(directed=True).add_nodes([str(n) for n in G])
            for u, v in G.edges:
                g.add_edge(str(u), str(v))
            assert g.is_aperiodic() == nx.is_aperiodic(G)
        chain = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="strongly connected"):
            chain.is_aperiodic()
        cyc = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="acyclic"):
            cyc.antichains()
        with pytest.raises(ValueError, match="cycle"):
            cyc.lexicographical_topological_sort()
        assert ec.Graph(directed=True).add_nodes(["b", "a"]).lexicographical_topological_sort(
            key=lambda n: -n
        ) == [2, 1]
        assert cyc.moral_graph().count_edges() == 1
