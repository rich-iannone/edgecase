"""Tests for chordal graph methods: is_chordal, chordal_graph_cliques, chordal_graph_treewidth,
complete_to_chordal_graph and find_induced_nodes, validated against NetworkX."""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_TRANSFORM


def _build(G):
    g = ec.Graph().add_nodes([str(n) for n in sorted(G)])
    for u, v in sorted(G.edges()):
        g.add_edge(str(u), str(v))
    return g


def _sorted_graph(G):
    H = nx.Graph()
    H.add_nodes_from(sorted(G))
    H.add_edges_from(sorted(tuple(sorted(e)) for e in G.edges()))
    return H


def _lab(g, nid):
    return int(g.get_label_from_id(nid))


def _random(count):
    for seed in range(count):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(rnd.randint(1, 11), rnd.choice([0.2, 0.35, 0.6]), seed=seed)
        yield seed, _sorted_graph(G)


class TestChordal:
    def test_is_chordal_and_triangulation_match_networkx(self):
        for _, G in _random(200):
            g = _build(G)
            assert g.is_chordal() == nx.is_chordal(G)
            chordal, alpha = g.complete_to_chordal_graph()
            H, ref_alpha = nx.complete_to_chordal_graph(G)
            fill = {
                frozenset(map(int, e.split("--")))
                for e in chordal.get_edges_str_repr(use_labels=True)
            }
            assert fill - {frozenset(e) for e in G.edges()} == (
                {frozenset(e) for e in H.edges()} - {frozenset(e) for e in G.edges()}
            )
            assert chordal.is_chordal()
            if not nx.is_chordal(G):
                assert {_lab(g, n): a for n, a in alpha.items()} == ref_alpha

    def test_cliques_treewidth_and_induced_nodes(self):
        for seed, G in _random(120):
            C = _sorted_graph(nx.complete_to_chordal_graph(G)[0])
            g = _build(C)
            ours = sorted(sorted(_lab(g, n) for n in c) for c in g.chordal_graph_cliques())
            assert ours == sorted(sorted(c) for c in nx.chordal_graph_cliques(C))
            assert g.chordal_graph_treewidth() == nx.chordal_graph_treewidth(C)
            for s, t in list(itertools.combinations(sorted(C), 2))[:5]:
                # Ties between missing edges are broken by node ID here and by Python set order
                # in NetworkX, so the node sets can differ; emptiness and the target cannot
                found = {_lab(g, n) for n in g.find_induced_nodes(str(s), str(t))}
                expected = nx.find_induced_nodes(C, s, t)
                assert bool(found) == bool(expected)
                if found:
                    assert t in found

    def test_examples_and_errors(self):
        square = ec.Graph().add_cycle_graph(4)
        assert not square.is_chordal()
        with pytest.raises(ValueError, match="not chordal"):
            square.chordal_graph_cliques()
        with pytest.raises(ValueError, match="not chordal"):
            square.find_induced_nodes(1, 3)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).is_chordal()
        assert ec.Graph().add_full_graph(6).chordal_graph_treewidth() == 5
        assert ec.Graph().add_star_graph(5).chordal_graph_treewidth() == 1
        ring = ec.Graph().add_cycle_graph(6)
        chordal, _ = ring.complete_to_chordal_graph()
        assert chordal.count_edges() == 9 and ring.count_edges() == 6
        assert ring.steps[-1].kind == KIND_TRANSFORM
