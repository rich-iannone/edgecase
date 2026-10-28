"""Tests for the incident-edge index: `_incident_edges()` and the methods that use it."""

import random

import edgecase as ec
import pytest


def _ids(g, *labels):
    return [g.get_id_from_label(label) for label in labels]


def _incident(g, label, direction):
    """`_incident_edges()` with the other endpoints as labels."""
    node = g.get_id_from_label(label)
    return [(e, g.get_label_from_id(other)) for e, other in g._graph._incident_edges(node,
                                                                                     direction)]


def _brute_force(g, node, direction):
    """What `_incident_edges()` should return, from a scan of every edge."""
    out, inward = [], []
    for e in sorted(g.get_edge_ids(limit=None)):
        x = g.get_edge(e)
        if x.source == node:
            out.append((e, x.target))
        if x.target == node and not (direction == "both" and x.source == node):
            inward.append((e, x.source))
    return {"out": out, "in": inward, "both": out + inward}[direction]


@pytest.fixture
def g():
    # a->b twice (parallel), b->c, c->a, a loop on b
    g = ec.Graph(directed=True).add_nodes(["a", "b", "c"])
    g.add_edges([("a", "b"), ("a", "b"), ("b", "c"), ("c", "a"), ("b", "b")])
    return g


class TestIncidentEdges:
    def test_directions(self, g):
        e1, e2, e3, e4, e5 = sorted(g.get_edge_ids(limit=None))
        assert _incident(g, "a", "out") == [(e1, "b"), (e2, "b")]
        assert _incident(g, "a", "in") == [(e4, "c")]
        assert _incident(g, "a", "both") == [(e1, "b"), (e2, "b"), (e4, "c")]
        assert _incident(g, "b", "out") == [(e3, "c"), (e5, "b")]
        assert _incident(g, "b", "in") == [(e1, "a"), (e2, "a"), (e5, "b")]

    def test_self_loop_listed_once_for_both(self, g):
        e1, e2, e3, _, e5 = sorted(g.get_edge_ids(limit=None))
        assert _incident(g, "b", "both") == [(e3, "c"), (e5, "b"), (e1, "a"), (e2, "a")]

    def test_isolated_node(self, g):
        g.add_node("lonely")
        for direction in ("out", "in", "both"):
            assert _incident(g, "lonely", direction) == []

    def test_undirected_follows_stored_orientation(self):
        g = ec.Graph().add_nodes(["a", "b"])
        g.add_edge("b", "a")
        (e,) = g.get_edge_ids(limit=None)
        assert _incident(g, "a", "out") == []
        assert _incident(g, "a", "in") == [(e, "b")]
        assert _incident(g, "a", "both") == [(e, "b")]
        assert _incident(g, "b", "both") == [(e, "a")]

    def test_errors(self, g):
        with pytest.raises(ValueError, match="Invalid direction 'sideways'"):
            g._graph._incident_edges(g.get_id_from_label("a"), "sideways")
        with pytest.raises(ValueError, match="Node 999 does not exist"):
            g._graph._incident_edges(999, "out")


class TestIndexStaysCurrent:
    """The index is built on first use, so each edit is checked after it has been built."""

    def _check(self, g):
        for node in g.get_node_ids(limit=None):
            for direction in ("out", "in", "both"):
                assert g._graph._incident_edges(node, direction) == _brute_force(g, node,
                                                                                 direction)

    def test_adding_edges(self, g):
        self._check(g)
        g.add_edge("c", "b").add_edge("c", "c")
        self._check(g)

    def test_deleting_edges_and_nodes(self, g):
        self._check(g)
        g.delete_edge_by_id(sorted(g.get_edge_ids(limit=None))[1])
        self._check(g)
        g.delete_node("c")
        self._check(g)

    def test_reversing_edges(self, g):
        self._check(g)
        g.select_edges_by_node(["c"]).rev_edge_dir_ws()
        assert "a->c" in g.get_edges_str_repr(use_labels=True)
        self._check(g)

    def test_failed_batch_is_rolled_back(self, g):
        self._check(g)
        with pytest.raises(ec.NodeNotFoundError):
            g.add_edges([("a", "c"), ("a", "missing")])
        assert g.count_edges() == 5
        self._check(g)


class TestUsers:
    """The selection methods that now read the index give the same answers as a scan."""

    @pytest.mark.parametrize("directed", [True, False])
    def test_random_graphs(self, directed):
        rng = random.Random(7)
        g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(12)])
        g.add_edges([(str(rng.randrange(12)), str(rng.randrange(12))) for _ in range(40)])
        edges = {e: g.get_edge(e) for e in g.get_edge_ids(limit=None)}
        nodes = rng.sample(sorted(g.get_node_ids(limit=None)), 4)

        def touching(keep):
            return sorted(e for e, x in edges.items() if keep(x))

        g.clear_selection().select_edges_by_node(nodes)
        assert sorted(g.get_selected_edge_ids()) == touching(
            lambda x: x.source in nodes or x.target in nodes)

        g.clear_selection().select_nodes_by_id(nodes).trav_out_edge()
        assert sorted(g.get_selected_edge_ids()) == touching(
            lambda x: x.source in nodes or (not directed and x.target in nodes))

        g.clear_selection().select_nodes_by_id(nodes).trav_in_edge()
        assert sorted(g.get_selected_edge_ids()) == touching(
            lambda x: x.target in nodes or (not directed and x.source in nodes))

        chosen = rng.sample(sorted(edges), 8)
        pairs = {(edges[e].source, edges[e].target) for e in chosen}
        g.clear_selection().select_edges_by_id(chosen).trav_reverse_edge()
        assert sorted(g.get_selected_edge_ids()) == touching(
            lambda x: (x.target, x.source) in pairs)

    def test_trav_reverse_edge_with_parallel_edges_and_loops(self, g):
        e1, e2, _, _, e5 = sorted(g.get_edge_ids(limit=None))
        g.add_edge("b", "a")
        back = max(g.get_edge_ids(limit=None))
        g.select_edges_by_id([e1]).trav_reverse_edge()
        assert g.get_selected_edge_ids() == [back]
        g.clear_selection().select_edges_by_id([back]).trav_reverse_edge()
        assert sorted(g.get_selected_edge_ids()) == [e1, e2]
        g.clear_selection().select_edges_by_id([e5]).trav_reverse_edge()
        assert g.get_selected_edge_ids() == [e5]

    def test_unknown_nodes_select_nothing(self, g):
        (a,) = _ids(g, "a")
        g._graph.select_edges_by_node_id([999])
        assert g.get_selected_edge_ids() == []
        g._graph.select_edges_by_node_id([a, 999])
        assert len(g.get_selected_edge_ids()) == 3
