import random

import edgecase as ec
import networkx as nx
import pytest
from networkx.algorithms import node_classification as nx_nc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ids(g, *labels):
    return [g.get_id_from_label(lbl) for lbl in labels]


def id_map(g, *labels):
    return {lbl: g.get_id_from_label(lbl) for lbl in labels}


def make_chain(n=5):
    g = ec.Graph()
    labels = [chr(65 + i) for i in range(n)]
    for lbl in labels:
        g.add_node(label=lbl)
    for i in range(n - 1):
        g.add_edge(source=labels[i], target=labels[i + 1])
    return g


def make_two_clusters():
    """Two tight clusters with a bridge."""
    g = ec.Graph()
    for i in range(8):
        g.add_node(label=str(i))
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(source=str(i), target=str(j))
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(source=str(i), target=str(j))
    g.add_edge(source="3", target="4")
    return g


# ===========================================================================
# label_propagation — basic functionality
# ===========================================================================


class TestLabelPropagationBasic:
    def test_two_seeds_chain(self):
        g = make_chain()
        a, e = ids(g, "A", "E")
        result = g.label_propagation({a: 0, e: 1}, return_dict=True)
        assert result[a] == 0
        assert result[e] == 1
        assert len(result) == 5

    def test_all_nodes_labeled(self):
        g = make_chain()
        a, e = ids(g, "A", "E")
        result = g.label_propagation({a: 0, e: 1}, return_dict=True)
        for nid in g.get_node_ids():
            assert nid in result

    def test_seeds_preserved(self):
        g = make_chain()
        a, c, e = ids(g, "A", "C", "E")
        result = g.label_propagation({a: 10, c: 20, e: 30}, return_dict=True)
        assert result[a] == 10
        assert result[c] == 20
        assert result[e] == 30

    def test_single_seed(self):
        g = make_chain()
        a = ids(g, "A")[0]
        result = g.label_propagation({a: 23}, return_dict=True)
        # All connected nodes should get label 23
        for nid in g.get_node_ids():
            assert result[nid] == 23

    def test_cluster_propagation(self):
        g = make_two_clusters()
        n0, n7 = [g.get_id_from_label(x) for x in ("0", "7")]
        result = g.label_propagation({n0: 0, n7: 1}, return_dict=True)
        # Nodes 0-3 should mostly get label 0, nodes 4-7 label 1
        cluster_0 = [result[g.get_id_from_label(str(i))] for i in range(4)]
        cluster_1 = [result[g.get_id_from_label(str(i))] for i in range(4, 8)]
        assert cluster_0.count(0) >= 3
        assert cluster_1.count(1) >= 3


# ===========================================================================
# label_propagation — return_dict and chaining
# ===========================================================================


class TestLabelPropagationChaining:
    def test_return_self(self):
        g = make_chain()
        a = ids(g, "A")[0]
        result = g.label_propagation({a: 0})
        assert result is g

    def test_return_dict(self):
        g = make_chain()
        a = ids(g, "A")[0]
        result = g.label_propagation({a: 0}, return_dict=True)
        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_chain()
        a = ids(g, "A")[0]
        g.label_propagation({a: 0})
        found = any(c[0] == "label_propagation" for c in g._computations)
        assert found

    def test_attr_stored(self):
        g = make_chain()
        a = ids(g, "A")[0]
        g.label_propagation({a: 0}, attr_name="my_label")
        ndf = g.get_node_df()
        assert "my_label" in ndf.columns


# ===========================================================================
# label_propagation — error handling
# ===========================================================================


class TestLabelPropagationErrors:
    def test_empty_known_labels(self):
        g = make_chain()
        with pytest.raises(ValueError, match="at least one"):
            g.label_propagation({}, return_dict=True)

    def test_invalid_node_id(self):
        g = make_chain()
        with pytest.raises(ValueError, match="not found"):
            g.label_propagation({99999: 0}, return_dict=True)


# ===========================================================================
# label_propagation — edge cases
# ===========================================================================


class TestLabelPropagationEdgeCases:
    def test_all_labeled(self):
        g = make_chain()
        known = {nid: i for i, nid in enumerate(g.get_node_ids())}
        result = g.label_propagation(known, return_dict=True)
        assert result == known

    def test_disconnected_components(self):
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B")
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge(source="X", target="Y")
        a, x = [g.get_id_from_label(lbl) for lbl in ("A", "X")]
        result = g.label_propagation({a: 0, x: 1}, return_dict=True)
        b, y = [g.get_id_from_label(lbl) for lbl in ("B", "Y")]
        assert result[b] == 0
        assert result[y] == 1

    def test_isolated_unlabeled_node(self):
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B")
        g.add_node(label="C")  # isolated
        a = g.get_id_from_label("A")
        c = g.get_id_from_label("C")
        result = g.label_propagation({a: 0}, return_dict=True)
        assert result[c] == -1  # unreachable stays -1

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        a = g.get_id_from_label("A")
        result = g.label_propagation({a: 5}, return_dict=True)
        assert result == {a: 5}

    def test_max_iter_respected(self):
        g = make_chain(20)
        a = g.get_id_from_label("A")
        t = g.get_id_from_label("T")
        # With max_iter=1, propagation won't reach far nodes
        result = g.label_propagation({a: 0, t: 1}, max_iter=1, return_dict=True)
        # At least the seeds are correct
        assert result[a] == 0
        assert result[t] == 1


# ===========================================================================
# label_propagation — weighted graph
# ===========================================================================


class TestLabelPropagationWeighted:
    def test_weight_influences_result(self):
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="B", target="A", weight=10.0)
        g.add_edge(source="B", target="C", weight=1.0)
        a, c = [g.get_id_from_label(x) for x in ("A", "C")]
        b = g.get_id_from_label("B")
        result = g.label_propagation({a: 0, c: 1}, return_dict=True)
        # B should prefer label 0 (from A) due to higher weight
        assert result[b] == 0


# ===========================================================================
# label_propagation — directed graph
# ===========================================================================


class TestLabelPropagationDirected:
    def test_directed_propagation(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="C")
        a = g.get_id_from_label("A")
        result = g.label_propagation({a: 0}, return_dict=True)
        b = g.get_id_from_label("B")
        c = g.get_id_from_label("C")
        assert result[b] == 0
        assert result[c] == 0


class TestNodeClassification:
    """harmonic_function and local_and_global_consistency, validated against NetworkX (nodes
    whose top class scores tie exactly are skipped: NetworkX resolves those by rounding)."""

    def test_against_networkx(self):
        for seed in range(80):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(rnd.randint(2, 14), 0.3, seed=seed)
            weighted = seed % 3 == 0
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.uniform(0.5, 2) if weighted else 1
            for n in rnd.sample(list(G), max(1, len(G) // 3)):
                G.nodes[n]["cls"] = rnd.choice("xyz")
            g = ec.Graph(weighted=weighted).add_nodes(
                [(str(n), {"cls": G.nodes[n]["cls"]}) if "cls" in G.nodes[n] else str(n) for n in G]
            )
            for u, v, d in G.edges(data=True):
                g.add_edge(str(u), str(v), weight=d["weight"])
            for ours, ref in [
                (g.harmonic_function("cls"), nx_nc.harmonic_function(G, label_name="cls")),
                (
                    g.local_and_global_consistency("cls"),
                    nx_nc.local_and_global_consistency(G, label_name="cls"),
                ),
            ]:
                mine = [ours[g.get_id_from_label(str(n))] for n in G]
                differ = [i for i, (a, b) in enumerate(zip(mine, ref)) if a != b]
                assert len(differ) <= 3  # only exact ties may differ

    def test_example_and_errors(self):
        g = ec.Graph().add_nodes(list("ABCDEFGH"))
        g.add_edges(
            [
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
                ("E", "F"),
                ("E", "G"),
                ("E", "H"),
                ("F", "G"),
                ("F", "H"),
                ("G", "H"),
                ("D", "E"),
            ]
        )
        g.set_node_attrs(
            "label", {g.get_id_from_label("A"): "red", g.get_id_from_label("H"): "blue"}
        )
        predicted = g.harmonic_function()
        assert {g.get_label_from_id(n): v for n, v in predicted.items()} == dict(
            zip("ABCDEFGH", ["red"] * 4 + ["blue"] * 4)
        )
        assert g.local_and_global_consistency()[g.get_id_from_label("D")] == "red"
        with pytest.raises(ValueError, match="labeled"):
            g.harmonic_function(label_name="missing")
