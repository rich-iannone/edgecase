"""Tests for the Cypher query planner and matcher (`edgecase.cypher.Plan` and `Matcher`)."""

import itertools
from collections import Counter
from decimal import Decimal
from fractions import Fraction

import edgecase as ec
import numpy as np
import pytest
from edgecase.cypher import (
    Evaluator,
    GraphView,
    Matcher,
    NodePattern,
    Plan,
    conjuncts,
    element_conditions,
    parse_query,
)
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


def matches(g, text, params=None, distinct_nodes=False, label_key="name"):
    """The matches, as dicts, from both engines, which must give the same ones in the same
    order, or raise the same error."""
    query = parse_query(text)
    found = {}
    for engine in ("python", "rust"):
        try:
            matcher = Matcher(query, text, g._graph, params, label_key, distinct_nodes, engine)
            found[engine] = list(matcher.matches())
        except Exception as e:
            found[engine] = e
    python, rust = found["python"], found["rust"]
    if isinstance(python, Exception) or isinstance(rust, Exception):
        assert type(python) is type(rust) and str(python) == str(rust), (python, rust)
        raise rust
    assert python == rust
    return rust


def rows(g, text, params=None, distinct_nodes=False, label_key="name"):
    """The named variables of each match, as tuples of labels (nodes), IDs (edges), tuples of
    IDs (paths) and `None`s, in order."""
    query = parse_query(text)
    names = list(query.variables)
    out = []
    for b in matches(g, text, params, distinct_nodes, label_key):
        out.append(tuple(
            g.get_label_from_id(b[v]) if query.variables[v] == "node" and b[v] is not None
            else b[v] for v in names
        ))
    return out


def plan(text, directed=True):
    return str(Plan(parse_query(text), directed)).split("\n")


@pytest.fixture
def g():
    # a->b twice (parallel), b->c, c->a, a loop on c, d on its own
    g = ec.Graph(directed=True).add_nodes([
        ("a", {"type": "P", "k": 1}), ("b", {"type": ["P", "Q"], "k": 2}),
        ("c", {"type": "Q"}), ("d", {}),
    ])
    g.add_edge("a", "b", rel="X").add_edge("a", "b", rel="Y").add_edge("b", "c", rel="X")
    g.add_edge("c", "a", rel="X", weight=3).add_edge("c", "c", rel="LOOP")
    return g


def edge(g, n):
    """The `n`th edge ID, counting from 1 in creation order."""
    return sorted(g.get_edge_ids(limit=None))[n - 1]


# =============================================================================
# Matching
# =============================================================================


class TestDirections:
    def test_out_and_in(self, g):
        assert sorted(rows(g, "MATCH (x)-->(y)")) == [
            ("a", "b"), ("a", "b"), ("b", "c"), ("c", "a"), ("c", "c")]
        assert sorted(rows(g, "MATCH (y)<--(x)")) == [
            ("a", "c"), ("b", "a"), ("b", "a"), ("c", "b"), ("c", "c")]

    def test_either_way(self, g):
        found = Counter(rows(g, "MATCH (x)--(y)"))
        assert found == Counter({("a", "b"): 2, ("b", "a"): 2, ("b", "c"): 1, ("c", "b"): 1,
                                 ("c", "a"): 1, ("a", "c"): 1, ("c", "c"): 1})

    def test_the_relationship_is_bound(self, g):
        assert sorted(rows(g, "MATCH (x:P)-[r]->(:P)")) == [("a", edge(g, 1)), ("a", edge(g, 2))]

    def test_expanding_against_the_arrow(self, g):
        # the planner starts at `y` (it has a property) and follows the edge backwards
        assert plan("MATCH (x)-[r]->(y {k: 2})") == ["scan y", "expand (y)<-[r]-(x)"]
        assert sorted(rows(g, "MATCH (x)-[r]->(y {k: 2})")) == [
            ("a", edge(g, 1), "b"), ("a", edge(g, 2), "b")]


class TestMultigraphsAndLoops:
    def test_parallel_edges_are_separate_matches(self, g):
        assert rows(g, "MATCH (x {k: 1})-[r]->(y {k: 2})") == [
            ("a", edge(g, 1), "b"), ("a", edge(g, 2), "b")]

    def test_self_loops(self, g):
        assert rows(g, "MATCH (x)-[r]->(x)") == [("c", edge(g, 5))]
        assert rows(g, "MATCH (x)-[r]-(x)") == [("c", edge(g, 5))]
        assert ("c", edge(g, 5), "c") in rows(g, "MATCH (x)-[r]->(y)")

    def test_two_cycles_use_two_edges(self, g):
        g.add_edge("b", "a")
        assert sorted(rows(g, "MATCH (x)-->(y)-->(x)")) == [
            ("a", "b"), ("a", "b"), ("b", "a"), ("b", "a")]

    def test_relationships_are_distinct(self, g):
        # `x, y, x` needs two different edges between x and y: here the parallel pair
        assert Counter(rows(g, "MATCH (x)-[r]-(y)-[s]-(x) WHERE x.k = 1 AND y.k = 2")) == Counter(
            {("a", edge(g, 1), "b", edge(g, 2)): 1, ("a", edge(g, 2), "b", edge(g, 1)): 1})
        # a loop can't be used twice in one match
        assert rows(g, "MATCH (x)-[r]->(x)-[s]->(x)") == []

    def test_triangle(self, g):
        found = rows(g, "MATCH (x)-[:X]->(y)-[:X]->(z)-[:X]->(x)")
        assert sorted(found) == [("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")]


class TestUndirected:
    @pytest.fixture
    def u(self):
        u = ec.Graph(directed=False).add_nodes(["a", "b", "c"])
        u.add_edge("a", "b").add_edge("c", "b").add_edge("c", "c")
        return u

    def test_every_arrow_means_either_way(self, u):
        expected = Counter({("a", "b"): 1, ("b", "a"): 1, ("b", "c"): 1, ("c", "b"): 1,
                            ("c", "c"): 1})
        for text in ("MATCH (x)-->(y)", "MATCH (x)<--(y)", "MATCH (x)--(y)"):
            assert Counter(rows(u, text)) == expected
        assert plan("MATCH (x)-->(y)", directed=False) == ["scan x", "expand (x)-[ 0]-(y)"]

    def test_paths(self, u):
        assert sorted(rows(u, "MATCH (x)--(y)--(z) WHERE x <> y AND y <> z")) == [
            ("a", "b", "c"), ("c", "b", "a")]


class TestSeveralPatterns:
    def test_shared_variables_join(self, g):
        assert sorted(rows(g, "MATCH (x)-[:X]->(y), (y)-[:X]->(z)")) == [
            ("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")]

    def test_relationships_are_distinct_across_patterns(self, g):
        assert sorted(rows(g, "MATCH (x)-[r]->(x), (y)-[s]->(y)")) == []
        assert len(rows(g, "MATCH ()-[r]->(), ()-[s]->()")) == 5 * 4

    def test_cross_product(self, g):
        found = rows(g, "MATCH (x:Q), (y {k: 1})")
        assert sorted(found) == [("b", "a"), ("c", "a")]
        assert len(rows(g, "MATCH (x), (y)")) == 16

    def test_a_lone_node(self, g):
        assert sorted(rows(g, "MATCH (n)")) == [("a",), ("b",), ("c",), ("d",)]
        assert sorted(rows(g, "MATCH (n) WHERE NOT (n:P OR n:Q)")) == [("d",)]


class TestConditions:
    def test_labels_types_and_properties(self, g):
        assert sorted(rows(g, "MATCH (x:P:Q)")) == [("b",)]
        assert sorted(rows(g, "MATCH (x:P|Q)")) == [("a",), ("b",), ("c",)]
        assert sorted(rows(g, "MATCH (x)-[:Y|LOOP]->(y)")) == [("a", "b"), ("c", "c")]
        assert sorted(rows(g, "MATCH (x)-[{weight: 3}]->(y)")) == [("c", "a")]
        assert rows(g, "MATCH (x {name: 'b'})") == [("b",)]
        assert rows(g, "MATCH (x {name: 'b'})", label_key=None) == []

    def test_where(self, g):
        assert sorted(rows(g, "MATCH (x)-[r]->(y) WHERE x.k < y.k OR type(r) = 'LOOP'")) == [
            ("a", edge(g, 1), "b"), ("a", edge(g, 2), "b"), ("c", edge(g, 5), "c")]
        assert rows(g, "MATCH (x)-->(y) WHERE x.k = $k AND y:Q", {"k": 1}) == [("a", "b"),
                                                                              ("a", "b")]

    def test_null_properties_never_match(self, g):
        assert rows(g, "MATCH (x {k: null})") == []
        assert rows(g, "MATCH (x {k: $k})", {"k": None}) == []

    def test_constant_conditions(self, g):
        assert len(rows(g, "MATCH (x) WHERE true")) == 4
        assert rows(g, "MATCH (x) WHERE $on", {"on": False}) == []
        assert rows(g, "MATCH (x) WHERE null") == []

    def test_id_seeds(self, g):
        a, b = g.get_id_from_label("a"), g.get_id_from_label("b")
        assert plan("MATCH (x)-->(y) WHERE id(y) = 2") == ["scan y by id", "expand (y)<-[ 0]-(x)"]
        assert rows(g, f"MATCH (x)-->(y) WHERE id(y) = {b}") == [("a", "b"), ("a", "b")]
        assert rows(g, f"MATCH (x) WHERE {a}.0 = id(x)") == [("a",)]
        assert rows(g, "MATCH (x) WHERE id(x) IN $ids", {"ids": [b, a, a, 99, -1, 1.5, "1"]}) \
            == [("a",), ("b",)]
        for value in ("null", "'1'", "-1", "99", "1.5", "2e30"):
            assert rows(g, f"MATCH (x) WHERE id(x) = {value}") == []
        assert rows(g, "MATCH (x) WHERE id(x) IN null") == []
        with pytest.raises(ValueError, match="IN needs a list, not 1"):
            rows(g, "MATCH (x) WHERE id(x) IN 1")

    def test_relationship_ids(self, g):
        assert rows(g, f"MATCH (x)-[r]->(y) WHERE id(r) = {edge(g, 3)}") == [
            ("b", edge(g, 3), "c")]


class TestDistinctNodes:
    def test_nodes_dont_repeat(self, g):
        g.add_edge("b", "a")
        # one variable is one node, so a 2-cycle still matches
        assert len(rows(g, "MATCH (x)-->(y)-->(x)", distinct_nodes=True)) == 4
        assert all(x != z for x, _, z in rows(g, "MATCH (x)-->(y)-->(z)", distinct_nodes=True))
        found = rows(g, "MATCH (x)-->(y)", distinct_nodes=True)
        assert ("c", "c") not in found and len(found) == 5
        # anonymous nodes count too
        assert sorted(rows(g, "MATCH (x)-->()-->(z)", distinct_nodes=True)) == [
            ("a", "c"), ("a", "c"), ("b", "a"), ("c", "b"), ("c", "b")]

    def test_one_variable_is_one_node(self, g):
        assert rows(g, "MATCH (x)-[r]->(x)", distinct_nodes=True) == [("c", edge(g, 5))]


class TestErrorsAndEmpty:
    def test_empty_graph(self):
        e = ec.Graph()
        assert rows(e, "MATCH (x)-->(y)") == []
        # compiled up front, so it fails without any rows to look at
        with pytest.raises(ValueError, match=r"\$year isn't given"):
            rows(e, "MATCH (x) WHERE x.born = $year")
        with pytest.raises(ValueError, match=r"the function nope\(\) isn't supported"):
            rows(e, "MATCH (x) WHERE nope(x)")

    def test_row_errors_raise_when_met(self, g):
        assert rows(g, "MATCH (x {k: 3}) WHERE x.k") == []  # no row reaches the WHERE
        with pytest.raises(ValueError, match="WHERE needs true, false or null, not 1"):
            rows(g, "MATCH (x) WHERE x.k")

    def test_the_python_engine_is_lazy(self, g):
        text = "MATCH (x), (y), (z)"
        found = Matcher(parse_query(text), text, g._graph, engine="python").matches()
        assert len(list(itertools.islice(found, 3))) == 3

    @pytest.mark.parametrize("limit", [0, 1, 5, 64, 65, None])
    def test_limit_gives_the_first_rows(self, g, limit):
        text = "MATCH (x), (y), (z)"
        query = parse_query(text)
        python = Matcher(query, text, g._graph, engine="python").rows(limit)
        rust = Matcher(query, text, g._graph).rows(limit)
        assert rust == python
        assert len(python) == (64 if limit is None else min(64, limit))

    def test_count_gives_the_number_and_the_first_row(self, g):
        text = "MATCH (x)-->(y)"
        query = parse_query(text)
        for engine in ("python", "rust"):
            matcher = Matcher(query, text, g._graph, engine=engine)
            assert matcher.count() == (5, matcher.rows()[0])
        text = "MATCH (x {k: 5})"
        assert Matcher(parse_query(text), text, g._graph).count() == (0, None)

    def test_unknown_engine(self, g):
        with pytest.raises(ValueError, match="engine must be 'rust' or 'python'"):
            Matcher(parse_query("MATCH (x)"), "MATCH (x)", g._graph, engine="java")

    def test_search_order_is_fixed(self, g):
        text = "MATCH (x)--(y)--(z)"
        assert rows(g, text) == rows(g, text)


class TestVariableLength:
    def test_ranges(self, g):
        e1, e2, e3 = edge(g, 1), edge(g, 2), edge(g, 3)
        assert rows(g, "MATCH (x {k: 1})-[r*2]->(y)") == [("a", (e1, e3), "c"),
                                                          ("a", (e2, e3), "c")]
        assert sorted(rows(g, "MATCH (x {k: 1})-[r*0..1]->(y)")) == [
            ("a", (), "a"), ("a", (e1,), "b"), ("a", (e2,), "b")]
        assert rows(g, "MATCH (x {name: 'd'})-[r*]-(y)") == []
        assert rows(g, "MATCH (x {name: 'd'})-[r*0..]-(y)") == [("d", (), "d")]
        assert len(rows(g, "MATCH (x {k: 1})-[*..2]->(y)")) == 4
        assert len(rows(g, "MATCH (x {k: 1})-[*2..]->(y)")) == len(
            rows(g, "MATCH (x {k: 1})-[*]->(y)")) - 2

    def test_zero_hops_bind_one_node(self, g):
        assert rows(g, "MATCH (x:P)-[*0]->(y:Q)") == [("b", "b")]

    def test_edges_come_in_pattern_order(self, g):
        # the planner starts at `y` and walks backwards, but the path still runs from x to y
        e1, e2, e3, e5 = edge(g, 1), edge(g, 2), edge(g, 3), edge(g, 5)
        assert plan("MATCH (x)-[r*2]->(y {name: 'c'})") == ["scan y", "expand (y)<-[r*2]-(x)"]
        assert sorted(rows(g, "MATCH (x)-[r*2]->(y {name: 'c'})")) == [
            ("a", (e1, e3), "c"), ("a", (e2, e3), "c"), ("b", (e3, e5), "c")]

    def test_types_and_properties_hold_for_every_edge(self, g):
        assert sorted(rows(g, "MATCH (x)-[:X*2]->(y)")) == [("a", "c"), ("b", "a"), ("c", "b")]
        with pytest.raises(ValueError, match=r"expected '\]', found '\*'"):
            rows(g, "MATCH (x)-[r {weight: 3}*1..]->(y)")  # the length comes first
        assert rows(g, "MATCH (x)-[r*1.. {weight: 3}]->(y)") == [("c", (edge(g, 4),), "a")]

    def test_each_edge_is_used_once_per_clause(self, g):
        cycles = rows(g, "MATCH (x)-[r*]->(x)")
        assert cycles and all(len(set(r)) == len(r) for _, r in cycles)
        e1, e2 = edge(g, 1), edge(g, 2)
        assert rows(g, f"MATCH (x)-[s]->(y), (x)-[r*]->(y) WHERE id(s) = {e1}") == [
            ("a", e1, "b", (e2,))]

    def test_distinct_nodes_cover_the_inner_nodes(self, g):
        # the two triangles from each start, and the loop at c
        assert len(rows(g, "MATCH (x)-[r*]->(x)", distinct_nodes=True)) == 7
        # a zero-length path would make y the same node as x
        assert sorted(rows(g, "MATCH (x)-[*0..1]->(y)", distinct_nodes=True)) == [
            ("a", "b"), ("a", "b"), ("b", "c"), ("c", "a")]

    def test_undirected(self):
        u = ec.Graph(directed=False).add_nodes(["a", "b", "c"])
        u.add_edge("a", "b").add_edge("c", "b").add_edge("c", "c")
        assert [len(r) for _, r, _ in rows(u, "MATCH (x {name: 'a'})-[r*]->(y)")] == [1, 2, 3]

    def test_conditions_on_the_path(self, g):
        found = rows(g, "MATCH (x {k: 1})-[r*0..]->(y) WHERE size(r) >= 3")
        assert found and all(len(r) >= 3 for _, r, _ in found)


class TestSeveralClauses:
    def test_clauses_join_on_their_variables(self, g):
        assert rows(g, "MATCH (x {k: 1}) MATCH (x)-[r]->(y)") == [
            ("a", edge(g, 1), "b"), ("a", edge(g, 2), "b")]

    def test_edges_may_repeat_across_clauses(self, g):
        assert len(rows(g, "MATCH (x)-[r]->(y) MATCH (y)<-[s]-(x)")) == 7
        assert len(rows(g, "MATCH (x)-[r]->(y), (y)<-[s]-(x)")) == 2

    def test_distinct_nodes_hold_per_clause(self, g):
        g.add_edge("b", "a")
        found = rows(g, "MATCH (x)-->(y) MATCH (y)-->(z)", distinct_nodes=True)
        assert ("a", "b", "a") in found
        assert ("a", "b", "a") not in rows(g, "MATCH (x)-->(y)-->(z)", distinct_nodes=True)
        # two nodes of an earlier clause must differ when a later one uses both
        assert ("c", "c", edge(g, 5)) in rows(g, "MATCH (x), (y) MATCH (x)-[r]-(y)")
        assert ("c", "c", edge(g, 5)) not in rows(g, "MATCH (x), (y) MATCH (x)-[r]-(y)",
                                                  distinct_nodes=True)

    def test_a_relationship_used_again_is_the_same_edge(self, g):
        e3 = edge(g, 3)
        assert rows(g, f"MATCH (x)-[r]->(y) WHERE id(r) = {e3} MATCH (u)-[r]-(v)") == [
            ("b", e3, "c", "b", "c"), ("b", e3, "c", "c", "b")]

    def test_each_where_belongs_to_its_clause(self, g):
        assert rows(g, "MATCH (x) WHERE x.k = 1 MATCH (x)-->(y) WHERE y:Q") == [("a", "b"),
                                                                                ("a", "b")]


class TestOptional:
    def test_missing_matches_are_null(self, g):
        assert rows(g, "MATCH (x) OPTIONAL MATCH (x)-[r:LOOP]->(y)") == [
            ("a", None, None), ("b", None, None), ("c", edge(g, 5), "c"), ("d", None, None)]

    def test_where_is_part_of_the_optional_match(self, g):
        assert rows(g, "MATCH (x) OPTIONAL MATCH (x)-->(y) WHERE y.k = 2") == [
            ("a", "b"), ("a", "b"), ("b", None), ("c", None), ("d", None)]

    def test_optional_first(self, g):
        assert rows(g, "OPTIONAL MATCH (x:Nope)") == [(None,)]
        assert rows(g, "OPTIONAL MATCH (x:Q)") == [("b",), ("c",)]
        assert rows(g, "OPTIONAL MATCH (x) WHERE $go", {"go": False}) == [(None,)]

    def test_a_null_variable_matches_nothing_later(self, g):
        text = "MATCH (x) OPTIONAL MATCH (x)-[:LOOP]->(y) {} MATCH (y)-->(z)"
        assert rows(g, text.format("")) == [("c", "c", "a"), ("c", "c", "c")]
        assert rows(g, text.format("OPTIONAL")) == [
            ("a", None, None), ("b", None, None), ("c", "c", "a"), ("c", "c", "c"),
            ("d", None, None)]
        # in WHERE, a null variable follows the usual null rules
        assert rows(g, "MATCH (x) OPTIONAL MATCH (x)-[:LOOP]->(y) MATCH (z) WHERE y IS NULL "
                       "AND z.k = 1") == [("a", None, "a"), ("b", None, "a"), ("d", None, "a")]

    def test_optional_paths(self, g):
        assert rows(g, "MATCH (x {name: 'd'}) OPTIONAL MATCH (x)-[r*]->(y)") == [
            ("d", None, None)]


# =============================================================================
# Planning
# =============================================================================


class TestPlan:
    def test_slots(self):
        p = Plan(parse_query("MATCH (a)-[r]->()<-[]-(a), (` 0`)"))
        assert p.slots == {"a": "node", "r": "relationship", " 1": "node", " 2": "relationship",
                           " 0": "node"}

    def test_start_at_the_most_constrained_node(self):
        assert plan("MATCH (a)-->(b:L)-->(c {k: 1})")[0] == "scan c"
        assert plan("MATCH (a)-->(b:L)-->(c)")[0] == "scan b"
        assert plan("MATCH (a)-->(b) WHERE b.k > 1")[0] == "scan b"
        assert plan("MATCH (a {k: 1})-->(b) WHERE id(b) = 3")[0] == "scan b by id"
        assert plan("MATCH (a)-->(b)")[0] == "scan a"  # a tie goes to the first

    def test_cycles_close_as_soon_as_they_can(self):
        assert plan("MATCH (a)-[r]->(b)-[s]->(c)-[t]->(a), (c)-[u]->(d)") == [
            "scan a", "expand (a)-[r]->(b)", "expand (b)-[s]->(c)", "connect (c)-[t]->(a)",
            "expand (c)-[u]->(d)"]

    def test_conditions_go_to_the_first_step_that_can_check_them(self):
        p = Plan(parse_query("MATCH (a:L)-[r:T]->(b) WHERE a.k = 1 AND b.k = a.k AND r.w > 0 "
                             "AND $go"))
        assert [len(step.conditions) for step in p.steps] == [2, 3]
        assert len(p.stages[0].entry) == 1

    def test_variable_length_steps(self):
        assert plan("MATCH (a)-[r*1..3]->(b)") == ["scan a", "expand (a)-[r*1..3]->(b)"]
        assert plan("MATCH (a)-[*]-(b)") == ["scan a", "expand (a)-[ 0*1..]-(b)"]
        assert plan("MATCH (a)-[r]->(b), (a)<-[p*0..2]-(b)") == [
            "scan a", "expand (a)-[r]->(b)", "connect (a)<-[p*0..2]-(b)"]
        p = Plan(parse_query("MATCH (a)-[r:T*2 {w: 1}]->(b)"))
        assert p.paths == {"r": ("a", "b")}
        assert len(p.steps[1].edge_conditions) == 2 and p.steps[1].conditions == []

    def test_clauses(self):
        assert plan("MATCH (a) MATCH (a)-->(b), (c)") == [
            "scan a", "match", "expand (a)-[ 0]->(b)", "scan c"]
        assert plan("MATCH (a) OPTIONAL MATCH (a)-->(b)") == [
            "scan a", "optional match", "expand (a)-[ 0]->(b)"]
        assert plan("OPTIONAL MATCH (a)") == ["optional match", "scan a"]
        p = Plan(parse_query("MATCH (a) MATCH (a:L)-->(b) WHERE a.k = 1 AND $go AND b.k = a.k"))
        later = p.stages[1]
        assert len(later.entry) == 3 and later.new == [" 0", "b"] and later.joined == ["a"]
        assert len(later.steps[0].conditions) == 1

    def test_groups_are_scanned_in_turn(self):
        # the most constrained group goes first
        assert plan("MATCH (a)-->(b), (c)-->(d:L)") == [
            "scan d", "expand (d)<-[ 1]-(c)", "scan a", "expand (a)-[ 0]->(b)"]


# =============================================================================
# Against a brute-force oracle
# =============================================================================


def oracle(g, text, params=None, distinct_nodes=False):
    """Every match of the query, by enumeration (the named variables of each, as a Counter).

    Each clause tries every assignment of nodes and distinct edges to its new elements, and
    every sequence of distinct edges for its variable-length relationships, keeping those that
    fit the arrows and the conditions. An OPTIONAL MATCH that keeps none gives nulls instead.
    """
    query = parse_query(text)
    directed = g.is_directed()
    kinds, clauses, i = {}, [], 0
    for clause in query.clauses:
        before = set(kinds)
        used, rels, conditions, edge_conditions = {}, [], [], {}

        def slot(element):
            nonlocal i
            i += 1
            name = element.variable or f"anon{i}"
            if isinstance(element, NodePattern):
                kind = "node"
            else:
                kind = "relationship" if element.length is None else "relationships"
            kinds.setdefault(name, kind)
            used.setdefault(name, kind)
            if kind == "relationships":
                edge_conditions[name] = element_conditions("anon_edge", element)
            else:
                conditions.extend(element_conditions(name, element))
            return name

        for pattern in clause.patterns:
            left = slot(pattern.nodes[0])
            for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                r = slot(rel)
                right = slot(node)
                rels.append((r, left, right, rel.direction, rel.length))
                left = right
        conditions += conjuncts(clause.where)
        clauses.append((clause.optional, before, used, rels, conditions, edge_conditions))

    evaluator = Evaluator(text, {**kinds, "anon_edge": "relationship"}, params,
                          GraphView(g._graph))
    edges = {e: g.get_edge(e) for e in g.get_edge_ids(limit=None)}
    all_nodes = sorted(g.get_node_ids(limit=None))

    def chains(length, arrow, tests):
        """(start, end) -> [(edges, inner nodes)] for every sequence of distinct edges of the
        right length that chains up along the arrow."""
        low, high = length
        high = len(edges) if high is None else min(high, len(edges))
        found = {}
        for n in range(low, high + 1):
            for path in itertools.permutations(sorted(edges), n):
                if not all(t({"anon_edge": e}) for e in path for t in tests):
                    continue
                for start in all_nodes:
                    node, inner = start, []
                    for e in path:
                        node = _follow(edges[e], node, arrow, directed)
                        if node is None:
                            break
                        inner.append(node)
                    if node is not None:
                        found.setdefault((start, node), []).append((path, inner[:-1]))
        return found

    out = [{}]
    for optional, before, used, rels, conditions, edge_conditions in clauses:
        tests = [evaluator.condition(c) for c in conditions]
        paths = {r: chains(length, arrow, [evaluator.condition(c) for c in edge_conditions[r]])
                 for r, _, _, arrow, length in rels if length is not None}
        new = [s for s in used if s not in before]
        new_nodes = [s for s in new if used[s] == "node"]
        new_rels = [s for s in new if used[s] == "relationship"]
        rows = []
        for row in out:
            extended = []
            if all(row[s] is not None for s in used if s in before):
                for nodes in itertools.product(all_nodes, repeat=len(new_nodes)):
                    for chosen in itertools.permutations(sorted(edges), len(new_rels)):
                        b = row | dict(zip(new_nodes, nodes)) | dict(zip(new_rels, chosen))
                        singles = [r for r, *_, length in rels if length is None]
                        if not all(_fits(edges[b[r]], b[x], b[y], arrow, directed)
                                   for r, x, y, arrow, length in rels if length is None):
                            continue
                        options = [paths[r].get((b[x], b[y]), [])
                                   for r, x, y, _, length in rels if length is not None]
                        for picked in itertools.product(*options):
                            c = dict(b)
                            inner = []
                            for r, (path, nodes_on) in zip(paths, picked):
                                c[r] = path
                                inner += nodes_on
                            used_edges = [c[r] for r in singles] + [
                                e for r in paths for e in c[r]]
                            if len(set(used_edges)) < len(used_edges):
                                continue
                            node_values = [c[s] for s, k in used.items() if k == "node"] + inner
                            if distinct_nodes and len(set(node_values)) < len(node_values):
                                continue
                            if all(t(c) for t in tests):
                                extended.append(c)
            if not extended and optional:
                extended = [row | {s: None for s in new}]
            rows += extended
        out = rows
    return Counter(tuple(b[v] for v in query.variables) for b in out)


def _fits(e, x, y, arrow, directed):
    forward, backward = (e.source, e.target) == (x, y), (e.source, e.target) == (y, x)
    if not directed or arrow == "either":
        return forward or backward
    return forward if arrow == "out" else backward


def _follow(e, node, arrow, directed):
    """The node an edge leads to from `node` along the arrow, or `None`."""
    if e.source == node and (arrow != "in" or not directed):
        return e.target
    if e.target == node and (arrow != "out" or not directed):
        return e.source
    return None


@st.composite
def graphs(draw):
    g = ec.Graph(directed=draw(st.booleans()))
    n = draw(st.integers(2, 4))
    for i in range(n):
        attrs = {}
        labels = draw(st.sampled_from([None, "P", "Q", ["P", "Q"]]))
        if labels is not None:
            attrs["type"] = labels
        k = draw(st.sampled_from([None, 1, 2]))
        if k is not None:
            attrs["k"] = k
        g.add_node(f"n{i}", **attrs)
    for _ in range(draw(st.integers(1, 7))):
        g.add_edge(f"n{draw(st.integers(0, n - 1))}", f"n{draw(st.integers(0, n - 1))}",
                   rel=draw(st.sampled_from(["A", "B"])), weight=draw(st.sampled_from([1, 2])))
    return g


@st.composite
def queries(draw):
    node_names, rel_names, path_names = [], [], []
    count, has_path = 0, False

    def node(names):
        name = draw(st.sampled_from(["a", "b", "c", "d", ""]))
        if name:
            names.append(name)
        # mostly unconstrained, so that most queries have matches
        labels = draw(st.sampled_from(["", "", "", ":P", ":Q", ":P:Q", ":P|Q"]))
        props = draw(st.sampled_from(["", "", "", "", " {k: 1}", " {k: $k}"]))
        return f"({name}{labels}{props})"

    def rel(earlier, in_clause):
        nonlocal count, has_path
        count += 1
        # at most one variable-length relationship, of at most two hops, keeps the oracle fast
        length = "" if has_path else draw(st.sampled_from(
            ["", "", "", "*0..1", "*1..2", "*2", "*..2", "*0..2"]))
        has_path = has_path or bool(length)
        reuse = [r for r in earlier if r not in in_clause]
        if not length and reuse and draw(st.integers(0, 4)) == 0:
            name = draw(st.sampled_from(reuse))
        else:
            name = draw(st.sampled_from([f"r{count}", ""]))
        if name:
            in_clause.append(name)
            (path_names if length else rel_names).append(name)
        types = draw(st.sampled_from(["", "", ":A", ":A|B"]))
        props = draw(st.sampled_from(["", "", "", " {weight: 2}"]))
        arrow = draw(st.sampled_from(["-[{}]->", "<-[{}]-", "-[{}]-"]))
        return arrow.format(name + types + length + props)

    def where():
        templates = ["{n}.k = 1", "{n}.k IS NULL", "{n}:P OR {m}:Q", "NOT {n}:P", "{n} <> {m}",
                     "id({n}) = 2", "id({n}) IN [1, 3]", "{n}.k < {m}.k", "{n} IS NULL"]
        if rel_names:
            templates += ["type({r}) = 'A'", "{r}.weight > 1"]
        if path_names:
            templates += ["size({p}) = 1", "size({p}) > 0 OR {n}.k = 2"]
        if not node_names or not draw(st.booleans()):
            return ""
        parts = []
        for _ in range(draw(st.integers(1, 2))):
            parts.append(draw(st.sampled_from(templates)).format(
                n=draw(st.sampled_from(node_names)), m=draw(st.sampled_from(node_names)),
                r=draw(st.sampled_from(rel_names or ["x"])),
                p=draw(st.sampled_from(path_names or ["x"]))))
        return " WHERE " + draw(st.sampled_from([" AND ", " OR "])).join(parts)

    clauses = []
    for i in range(draw(st.integers(1, 3))):
        earlier, in_clause = list(rel_names), []
        keyword = "MATCH" if i == 0 and draw(st.integers(0, 3)) else draw(
            st.sampled_from(["MATCH", "OPTIONAL MATCH"]))
        patterns = []
        for _ in range(draw(st.integers(1, 2 if i == 0 else 1))):
            text = node(node_names)
            for _ in range(draw(st.integers(0, 2))):
                text += rel(earlier, in_clause) + node(node_names)
            patterns.append(text)
        clauses.append(f"{keyword} " + ", ".join(patterns) + where())
    return " ".join(clauses)


FIXED_QUERIES = [
    "MATCH (a)",
    "MATCH (a)-[r]->(b)",
    "MATCH (a)<-[r]-(b)",
    "MATCH (a)-[r]-(b)",
    "MATCH (a)-[r]->(a)",
    "MATCH (a)-[r]-(a)",
    "MATCH (a)-->(b)-->(c)",
    "MATCH (a)--(b)--(c)",
    "MATCH (a)-->(b)-->(a)",
    "MATCH (a)--(b)--(a)",
    "MATCH (a)-->(b)<--(c)",
    "MATCH (a)-->(b)-->(c)-->(a)",
    "MATCH (a)-[r]->(b), (b)-[s]->(c)",
    "MATCH (a)-[r]->(b), (a)-[s]->(b)",
    "MATCH (a)-->(b), (c)",
    "MATCH ()-[r]->(), ()-[s]-()",
    "MATCH (a:P)-[:A]->(b)",
    "MATCH (a:P|Q)-[r:A|B]-(b:Q)",
    "MATCH (a {k: 1})--(b) WHERE b.k IS NULL OR b.k > a.k",
    "MATCH (a)-[r]->(b) WHERE r.weight > 1 OR a = b",
    "MATCH (a)-->(b)-->(c) WHERE a <> c",
    "MATCH (a)--(b) WHERE id(a) IN [1, 2] AND id(b) = 3",
    "MATCH (a)-[r*]->(b)",
    "MATCH (a)-[r*0..2]-(b)",
    "MATCH (a)-[*2]->(a)",
    "MATCH (a)<-[r:A*1..3]-(b)",
    "MATCH (a)-[r*..2 {weight: 2}]->(b)-[s]->(c)",
    "MATCH (a)-[r*0..]->(b) WHERE size(r) <> 1",
    "MATCH (a)-[s]->(b), (a)-[r*1..2]-(b)",
    "MATCH (a) MATCH (a)-->(b)",
    "MATCH (a)-[r]->(b) MATCH (b)-[s]->(c)",
    "MATCH (a)-[r]->(b) MATCH (c)-[r]-(d)",
    "MATCH (a), (b) MATCH (a)-[r]-(b)",
    "MATCH (a) OPTIONAL MATCH (a)-[r]->(b) WHERE b.k = 1",
    "MATCH (a:P) OPTIONAL MATCH (a)-->(b:Q) OPTIONAL MATCH (b)-->(c)",
    "OPTIONAL MATCH (a:P)-->(b:Q)",
    "MATCH (a) OPTIONAL MATCH (a)-[r*1..2]->(b) WHERE id(b) = 1",
    "MATCH (a) OPTIONAL MATCH (a)-->(b) MATCH (b)-->(c)",
]


@pytest.mark.parametrize("text", FIXED_QUERIES)
@given(g=graphs(), distinct_nodes=st.booleans())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_typical_queries_against_the_oracle(text, g, distinct_nodes):
    query = parse_query(text)
    expected = oracle(g, text, None, distinct_nodes)
    found = Counter(
        tuple(b[v] for v in query.variables)
        for b in matches(g, text, distinct_nodes=distinct_nodes)
    )
    assert found == expected


@given(g=graphs(), text=queries(), distinct_nodes=st.booleans())
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_against_the_oracle(g, text, distinct_nodes):
    query = parse_query(text)
    slots = Plan(query).slots
    assume(sum(k == "node" for k in slots.values()) <= 5)
    params = {"k": 1}
    expected = oracle(g, text, params, distinct_nodes)
    found = Counter(
        tuple(b[v] for v in query.variables)
        for b in matches(g, text, params, distinct_nodes=distinct_nodes)
    )
    assert found == expected, text


@given(g=graphs(), text=queries(), limit=st.sampled_from([None, 0, 1, 3]), sort=st.booleans())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_engines_agree_on_tables(g, text, limit, sort):
    query = parse_query(text)
    slots = Plan(query).slots
    columns = [("slot", name) for name in slots]
    for name, kind in slots.items():
        if kind == "node":
            columns += [("label", name), ("prop", name, "k"), ("prop", name, "name")]
        elif kind == "relationship":
            columns += [("prop", name, "weight"), ("prop", name, "rel")]
    sort = list(query.variables) if sort else None
    tables = [
        Matcher(query, text, g._graph, {"k": 1}, engine=engine).table(columns, sort, limit)
        for engine in ("rust", "python")
    ]
    assert tables[0] == tables[1], text
    n, cells, first = tables[0]
    assert all(len(column) == n for column in cells)
    if limit is not None:
        assert n <= limit


# =============================================================================
# The Rust engine against the Python one
# =============================================================================
# `matches()` runs both engines and checks that they agree, so the tests above check the Rust
# engine too. These aim at the conditions it checks natively, with values it must leave to
# Python.


class _NoEquals:
    """An attribute value whose comparisons raise."""

    def __eq__(self, other):
        raise RuntimeError("can't compare")

    __lt__ = __gt__ = __le__ = __ge__ = __eq__
    __hash__ = object.__hash__


_ODD_VALUES = [
    None, 0, 1, 2, -1, 1.0, 2.5, float("nan"), float("inf"), True, False, "1", "a", "b", "",
    "ab", "ña", "\ud800a",
    [1], [1, 2], ("a",), {"x": 1}, np.int64(1), np.float64(2.5), np.bool_(True), np.str_("a"),
    Fraction(1, 2), Decimal("1"), 2**70, _NoEquals(),
]
_ODD_TYPES = ["P", "Q", ["P", "Q"], ("Q",), [], ["P", 1], 5, None, np.str_("P"), [np.str_("Q")]]


@st.composite
def odd_graphs(draw):
    g = ec.Graph(directed=draw(st.booleans()))
    n = draw(st.integers(1, 4))
    for i in range(n):
        attrs = {}
        for key, pool in (("v", _ODD_VALUES), ("type", _ODD_TYPES),
                          ("name", [None, "n0", "x", 1])):
            value = draw(st.sampled_from(pool))
            if value is not None or draw(st.booleans()):
                attrs[key] = value
        g.add_node(f"n{i}", **attrs)
    for _ in range(draw(st.integers(0, 5))):
        attrs = {}
        rel = draw(st.sampled_from(["A", "B", 1, None, np.str_("A")]))
        if rel is not None:
            attrs["rel"] = rel
        value = draw(st.sampled_from(_ODD_VALUES))
        if value is not None or draw(st.booleans()):
            attrs["v"] = value
        weight = draw(st.sampled_from([None, 1, 2.5, "heavy", 2]))
        if weight in ("heavy", 2):  # a `weight` attribute, on an edge with no weight
            attrs["weight"], weight = weight, None
        ids = sorted(g.get_node_ids(limit=None))
        g._graph.add_edge(ec._core.Edge(
            id=0, source=ids[draw(st.integers(0, n - 1))], target=ids[draw(st.integers(0, n - 1))],
            weight=weight, attributes=attrs or None,
        ))
    return g


_CONSTANTS = ["1", "2", "2.5", "1.0", "-1", "'a'", "'1'", "true", "false", "null", "$p", "$q",
              "[1]", "{x: 1}"]
_CONDITIONS = [
    "{s}.v {op} {c}", "{c} {op} {s}.v", "{s}.v IN [{c}, {c}]", "{s}.v IN $list", "{s}.v IN {c}",
    "{s}.v IS NULL", "{s}.v IS NOT NULL", "exists({s}.v)", "{s} IS NULL", "a:P", "a:P|Q",
    "a:P:Q", "r:A", "r:A|B", "a.type = 'P'", "a.name {op} {c}", "a.name = 'n0'", "r.weight {op} {c}",
    "id(a) {op} id(b)", "id(a) {op} {c}", "{c} {op} id(r)", "id(r) IN $list", "a = b", "a <> b",
    "a {op} b", "a = r", "r <> r", "{s}.v", "{s}.v = {s}.v", "NOT {s}.v = {c}",
    "{s}.v {op} a.v", "id(b) {op} {s}.v", "a.name {op} b.name", "{s}.v STARTS WITH {c}",
    "a.name STARTS WITH {c}", "{c} ENDS WITH {s}.v", "{s}.v CONTAINS a.name",
    "{s}.v {op} {c} OR a:P", "a.v IS NULL OR {s}.v {op} {c}",
    "(a.name = {c} OR r:A) AND (b:Q OR {s}.v CONTAINS {c})",
]


@st.composite
def odd_queries(draw):
    conditions = []
    for _ in range(draw(st.integers(1, 2))):
        conditions.append(draw(st.sampled_from(_CONDITIONS)).format(
            s=draw(st.sampled_from(["a", "b", "r"])),
            op=draw(st.sampled_from(["=", "<>", "<", "<=", ">", ">="])),
            c=draw(st.sampled_from(_CONSTANTS)),
        ))
    pattern = draw(st.sampled_from([
        "MATCH (a)-[r]-(b)", "MATCH (a)-[r]->(b)", "MATCH (a), (b)-[r]->()",
        "MATCH (a) OPTIONAL MATCH (a)-[r]->(b)",
    ]))
    return f"{pattern} WHERE " + " AND ".join(conditions)


@given(g=odd_graphs(), text=odd_queries(), p=st.sampled_from(_ODD_VALUES),
       q=st.sampled_from(_ODD_VALUES), items=st.lists(st.sampled_from(_ODD_VALUES), max_size=3),
       label_key=st.sampled_from(["name", "v", None]), distinct_nodes=st.booleans())
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_engines_agree_on_odd_values(g, text, p, q, items, label_key, distinct_nodes):
    params = {"p": p, "q": q, "list": tuple(items) if len(items) == 2 else items}
    try:
        matches(g, text, params, distinct_nodes, label_key)
    except (ValueError, RuntimeError):
        pass  # `matches()` has checked that both engines raise it


@pytest.mark.parametrize("text", [
    "MATCH (a) WHERE a.v = 1", "MATCH (a) WHERE a.v < 2", "MATCH (a) WHERE a.v IN [1, 'a']",
    "MATCH (a) WHERE a.v = $p", "MATCH (a) WHERE a.v <> 1", "MATCH (a) WHERE a.v >= 1",
])
def test_engines_agree_on_values_rust_cant_read(text):
    g = ec.Graph()
    for i, value in enumerate(_ODD_VALUES):
        g.add_node(f"n{i}", v=value)
    assert matches(g, text, {"p": np.int64(1)})  # (some match, and both engines agree on which)


class TestRustEngine:
    def test_a_python_error_in_a_fallback_propagates(self):
        g = ec.Graph().add_node("a", v=1)
        with pytest.raises(ValueError, match="WHERE needs true, false or null, not 1"):
            matches(g, "MATCH (a) WHERE a.v")
        with pytest.raises(ValueError, match="toLower\\(\\) can't take 1"):
            matches(g, "MATCH (a) WHERE toLower(a.v) = 'x'")

    def test_seed_errors_raise_on_entering_the_clause(self):
        # the scan of `a` finds nothing, but the seed of `b` is worked out first all the same
        g = ec.Graph(directed=True).add_node("x", type="Q")
        with pytest.raises(ValueError, match="toLower\\(\\) can't take 5"):
            matches(g, "MATCH (a:P), (b) WHERE id(b) = toLower(5)")
        assert matches(g, "MATCH (a:P) MATCH (b) WHERE id(b) = toLower(5)") == []

    def test_nulls_from_optional_match(self):
        g = ec.Graph(directed=True).add_nodes(["a", "b"]).add_edge("a", "b")
        found = rows(g, "MATCH (x) OPTIONAL MATCH (x)-[r]->(y) WHERE y.k IS NULL AND id(r) > -1")
        assert found == [("a", edge(g, 1), "b"), ("b", None, None)]

    def test_a_long_path_does_not_overflow_the_stack(self):
        n = 20_000
        g = ec.Graph(directed=True).add_nodes([f"n{i}" for i in range(n)])
        g.add_edges([(f"n{i}", f"n{i + 1}") for i in range(n - 1)])
        text = "MATCH (a)-[r*]->(b) WHERE id(a) = $start AND id(b) = $end"
        query = parse_query(text)
        ids = sorted(g.get_node_ids(limit=None))
        params = {"start": ids[0], "end": ids[-1]}
        (row,) = Matcher(query, text, g._graph, params).rows()
        assert len(row[1]) == n - 1

    def test_kept_results_with_sparse_ids(self):
        # one-element conditions keep their results by ID, in a table, or in a map when the IDs
        # are too far apart for one
        g = ec.Graph(directed=True)
        for i, node_id in enumerate([3, 10**12, 7, 2**40]):
            g._graph.add_node(ec._core.Node(
                id=node_id, label=f"n{i}", attributes={"type": "PQ"[i % 2], "k": i}))
        g.add_edges([("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n0"), ("n0", "n2")])
        for text in ["MATCH (a:P)-->(b)-->(c:P)", "MATCH (a)-->(b)-->(c) WHERE b.k > 0",
                     "MATCH (a)-[r]->(b)<-[s]-(c) WHERE a.name STARTS WITH 'n'"]:
            assert matches(g, text)

    def test_rows_are_tuples_in_slot_order(self, g):
        text = "MATCH (x)-[r*1..2]->(y)-->(z) WHERE id(x) = 1"
        matcher = Matcher(parse_query(text), text, g._graph)
        assert list(matcher.plan.slots) == ["x", "r", "y", " 0", "z"]
        for row in matcher.rows():
            assert isinstance(row, tuple) and isinstance(row[1], tuple) and len(row) == 5
            assert matcher.binding(row) == dict(zip(matcher.plan.slots, row))


class TestNative:
    """Which conditions the Rust search checks itself."""

    def native(self, condition, params=None, g=None):
        text = f"MATCH (a)-[r]->(b)-[p*]->(c) WHERE {condition}"
        g = g or ec.Graph(directed=True)
        matcher = Matcher(parse_query(text), text, g._graph, params)
        (cond,) = [c for s in matcher._program[4] for step in s[4] for c in step[6]]
        return cond[0]

    @pytest.mark.parametrize("condition, expected", [
        ("a:P:Q|R", ("labels", 0, [["P"], ["Q", "R"]])),
        ("r:A", ("labels", 1, [["A"]])),
        ("a.k = 1", ("cmp", ("prop", 0, "k"), "=", 1)),
        ("1 < a.k", ("cmp", ("prop", 0, "k"), ">", 1)),
        ("a.k >= 'x'", ("cmp", ("prop", 0, "k"), ">=", "x")),
        ("id(r) <= $p", ("cmp", ("id", 1), "<=", 2.5)),
        ("a.k = null", ("cmp", ("prop", 0, "k"), "=", None)),
        ("a.k IN [1, $p, null]", ("in", ("prop", 0, "k"), [1, 2.5, None])),
        ("id(a) IN $list", ("in", ("id", 0), [1, 2])),
        ("a.k IS NULL", ("null", 0, "k", False)),
        ("p IS NOT NULL", ("null", 3, None, True)),
        ("exists(r.w)", ("null", 1, "w", True)),
        ("id(a) < id(b)", ("ids", 0, 2, "<")),
        ("a <> b", ("ids", 0, 2, "<>")),
        ("a = r", None),
        ("a < b", None),
        ("a.k = b.k", ("cmp2", ("prop", 0, "k"), "=", ("prop", 2, "k"))),
        ("id(a) > r.k", ("cmp2", ("id", 0), ">", ("prop", 1, "k"))),
        ("a.k = [1]", None),
        ("a.k = $np", None),
        ("a.k IN $odd", None),
        ("a.k STARTS WITH 'x'", ("text", "STARTS WITH", ("prop", 0, "k"), ("const", "x"))),
        ("$p CONTAINS r.k", ("text", "CONTAINS", ("const", 2.5), ("prop", 1, "k"))),
        ("a.k ENDS WITH b.name", ("text", "ENDS WITH", ("prop", 0, "k"), ("prop", 2, "name"))),
        ("a.k STARTS WITH $odd", None),
        ("a:P OR b:Q", ("or", [("labels", 0, [["P"]]), ("labels", 2, [["Q"]])])),
        ("a:P OR (b:Q AND id(a) < 3)",
         ("or", [("labels", 0, [["P"]]),
                 ("and", [("labels", 2, [["Q"]]), ("cmp", ("id", 0), "<", 3)])])),
        ("a:P OR size(p) = 1", None),
        ("a:P XOR b:Q", None),
        ("size(p) = 1", None),
    ])
    def test_forms(self, condition, expected):
        params = {"p": 2.5, "list": (1, 2), "np": np.int64(1), "odd": [1, [2]]}
        assert self.native(condition, params) == expected

    @pytest.mark.parametrize("condition", [
        "a.k < b.k", "a.k = b.k", "id(a) <> b.k", "a.name STARTS WITH 'n'",
        "b.name ENDS WITH a.v OR b.name ENDS WITH '1'", "a.name CONTAINS '1' OR b:P",
        "(a:P AND b.k = 2) OR id(a) > id(b)", "a.missing CONTAINS 'x' OR a.k = 1",
    ])
    def test_plain_values_never_call_python(self, condition):
        g = ec.Graph(directed=True)
        for i in range(4):
            g.add_node(f"n{i}", k=i % 3, type="P" if i % 2 else "Q")
        g.add_edges([("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n0"), ("n1", "n3")])
        text = f"MATCH (a)-->(b) WHERE {condition}"
        matcher = Matcher(parse_query(text), text, g._graph)

        def python(binding):
            raise AssertionError("the condition was left to Python")

        def strip(conds):
            assert all(native is not None for native, _ in conds)
            return [(native, python) for native, _ in conds]

        names, kinds, label_key, distinct, stages = matcher._program
        matcher._program = (names, kinds, label_key, distinct, [
            (optional, strip(entry), new, joined, [
                (*step[:6], strip(step[6]), step[7], strip(step[8]), *step[9:])
                for step in steps
            ])
            for optional, entry, new, joined, steps in stages
        ])
        reference = Matcher(parse_query(text), text, g._graph, engine="python")
        assert matcher.rows() == reference.rows()
        assert reference.rows()  # (the condition holds somewhere)
