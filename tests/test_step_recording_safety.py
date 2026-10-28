"""Step recording stays consistent with the graph when a builder raises partway.

The calls a compound operation makes record no steps of their own (`_recording_depth` above 0).
An exception between the increment and the decrement used to leave the depth stuck above 0,
which silently turned step recording off for the rest of the graph's life, and the nodes or
edges added before the failure stayed in the graph with no step describing them.
"""

import ast
import inspect

import pytest

import edgecase as ec
from edgecase.graph import Graph


def state(g):
    """Everything an undone addition could leave behind."""
    rg = g._graph
    ids = sorted(g.get_node_ids(limit=None))
    return {
        "nodes": [(i, g.get_label_from_id(i)) for i in ids],
        "edges": list(g.get_edges_as_tuples()),
        "neighbors": {i: list(rg.neighbors(i)) for i in ids},
        "node_order": rg.get_node_creation_order(),
        "edge_order": rg.get_edge_creation_order(),
        "mark": rg._addition_mark(),
        "selected": (sorted(rg.get_selected_node_ids()), sorted(rg.get_selected_edge_ids())),
        "steps": [s.code for s in g._steps],
    }


def base_graph(directed=False):
    g = ec.Graph(directed=directed).add_nodes(["a", "b", "c"])
    g.add_edges([("a", "b"), ("b", "c"), ("c", "c")])
    return g


def assert_recording_works(g):
    assert g._recording_depth == 0
    n = len(g._steps)
    g.add_node("zz")
    assert len(g._steps) == n + 1
    assert g._steps[-1].method == "add_node"


class TestBulkBuilders:
    def test_add_edges_by_id_missing_node(self):
        g = ec.Graph().add_nodes(["a", "b"])
        before = state(g)
        with pytest.raises(ValueError):
            g.add_edges_by_id([(1, 2), (1, 99)])
        assert state(g) == before
        assert_recording_works(g)

    @pytest.mark.parametrize("directed", [False, True])
    def test_add_edges_missing_node(self, directed):
        g = base_graph(directed)
        before = state(g)
        with pytest.raises(ValueError):
            g.add_edges([("a", "c"), ("a", "b"), ("c", "nope")])
        assert state(g) == before
        assert_recording_works(g)

    def test_add_edges_auto_create_rolls_back_created_nodes(self, monkeypatch):
        g = base_graph()
        before = state(g)
        real = Graph.add_edge
        calls = []

        def flaky(self, source, *args, **kwargs):
            calls.append(source)
            if len(calls) == 3:
                raise RuntimeError("boom")
            return real(self, source, *args, **kwargs)

        monkeypatch.setattr(Graph, "add_edge", flaky)
        with pytest.raises(RuntimeError):
            g.add_edges([("x", "y"), ("y", "a"), ("z", "a")], auto_create_nodes=True)
        monkeypatch.undo()
        assert state(g) == before
        assert_recording_works(g)

    def test_add_nodes_failing_partway(self, monkeypatch):
        g = base_graph()
        before = state(g)
        real = Graph.add_node

        def flaky(self, label=None, *args, **kwargs):
            if label == "q3":
                raise RuntimeError("boom")
            return real(self, label, *args, **kwargs)

        monkeypatch.setattr(Graph, "add_node", flaky)
        with pytest.raises(RuntimeError):
            g.add_nodes(["q1", "q2", "q3", "q4"])
        monkeypatch.undo()
        assert state(g) == before
        assert_recording_works(g)

    def test_add_node_with_failing_edge(self, monkeypatch):
        g = base_graph()
        before = state(g)

        def failing(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(Graph, "add_edge", failing)
        with pytest.raises(RuntimeError):
            g.add_node("new", from_node="a")
        monkeypatch.undo()
        assert state(g) == before
        assert not g.has_node("new")
        assert_recording_works(g)

    def test_successful_builds_unchanged(self):
        g = base_graph()
        assert [s.method for s in g._steps] == ["add_nodes", "add_edges"]
        g.add_node("d", from_node="a", to_node="b")
        assert g._steps[-1].delta.edges_added == [("d", "a"), ("b", "d")]
        g.add_edge("p", "q", auto_create_nodes=True)
        assert g.has_node("p") and g.has_node("q")
        assert g._recording_depth == 0


class TestGenerators:
    @pytest.mark.parametrize(
        "build",
        [
            lambda g: g.add_complete_bipartite_graph(3, 4),
            lambda g: g.add_wheel_graph(6),
            lambda g: g.add_gn_graph(12, seed=1),
            lambda g: g.add_barbell_graph(3, 2),
        ],
    )
    def test_failing_generator_leaves_graph_unchanged(self, monkeypatch, build):
        g = base_graph()
        before = state(g)
        real = Graph._add_generated_edges

        def flaky(self, *args, **kwargs):
            real(self, *args, **kwargs)
            raise RuntimeError("boom")

        monkeypatch.setattr(Graph, "_add_generated_edges", flaky)
        with pytest.raises(RuntimeError):
            build(g)
        monkeypatch.undo()
        assert state(g) == before
        # The ID counters rewind, so a retry builds what it would have without the failure
        fresh = build(base_graph())
        build(g)
        assert state(g)["nodes"] == state(fresh)["nodes"]
        assert state(g)["edges"] == state(fresh)["edges"]
        assert_recording_works(g)

    def test_keyboard_interrupt_rolls_back(self, monkeypatch):
        g = base_graph()
        before = state(g)
        real = Graph._add_generated_edges

        def interrupted(self, *args, **kwargs):
            real(self, *args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(Graph, "_add_generated_edges", interrupted)
        with pytest.raises(KeyboardInterrupt):
            g.add_wheel_graph(5)
        monkeypatch.undo()
        assert state(g) == before
        assert_recording_works(g)

    def test_nested_generator_failure(self, monkeypatch):
        """A failure inside a compound nested in another one undoes both levels."""
        g = base_graph()
        before = state(g)

        def failing(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(Graph, "_add_generated_edges", failing)
        with pytest.raises(RuntimeError):
            with g._compound("outer", "G.outer()"):
                g.add_path_graph(4)
                g.add_wheel_graph(5)
        monkeypatch.undo()
        assert state(g) == before
        assert_recording_works(g)

    def test_rollback_keeps_existing_adjacency_order(self):
        g = base_graph()
        before = state(g)
        with pytest.raises(RuntimeError):
            with g._atomic():
                g.add_edges([("a", "b"), ("c", "a"), ("b", "b")])
                g.add_node("n", from_node="a")
                raise RuntimeError("boom")
        assert state(g) == before

    @pytest.mark.parametrize("kind", ["nodes", "edges"])
    def test_rollback_drops_selection_of_removed_items(self, kind):
        g = base_graph()
        with pytest.raises(RuntimeError):
            with g._atomic():
                g.add_nodes(["s1", "s2"]).add_edge("s1", "a")
                if kind == "nodes":
                    g.select_nodes_by_id(g.get_node_ids(limit=None))
                else:
                    g._graph.select_edges_by_id([eid for eid, _, _ in g.get_edges_as_tuples()])
                raise RuntimeError("boom")
        if kind == "nodes":
            assert set(g._graph.get_selected_node_ids()) == set(g.get_node_ids(limit=None))
        else:
            live = {eid for eid, _, _ in g.get_edges_as_tuples()}
            assert set(g._graph.get_selected_edge_ids()) == live


class TestSuppressionRestored:
    def test_trav_out_until_condition_raises(self):
        g = ec.Graph(directed=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b"), ("b", "c")])
        g.select_nodes_by_id([g.get_id_from_label("a")])

        def bad(node_id):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            g.trav_out_until(bad)
        assert_recording_works(g)

    def test_bfs_attr_failure(self, monkeypatch):
        g = base_graph()

        def failing(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(Graph, "set_node_attrs", failing)
        with pytest.raises(RuntimeError):
            g.bfs("a")
        with pytest.raises(RuntimeError):
            g.dfs("a")
        monkeypatch.undo()
        assert_recording_works(g)

    def test_dataframe_init_failure(self):
        import polars as pl

        with pytest.raises(Exception):
            ec.Graph(edges_df=pl.DataFrame({"from": ["a"], "nope": ["b"]}))
        g = ec.Graph(edges_df=pl.DataFrame({"from": ["a"], "to": ["b"]}))
        assert len(g._steps) == 1
        assert_recording_works(g)


def test_depth_only_changed_by_helpers():
    """`_recording_depth` is only changed by the exception-safe context managers."""
    tree = ast.parse(inspect.getsource(Graph))
    allowed = {"_no_recording", "__init__", "__getattr__"}
    offenders = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func.name in allowed:
            continue
        for node in ast.walk(func):
            targets = []
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr == "_recording_depth":
                    offenders.append(f"{func.name}:{node.lineno}")
    assert offenders == []
