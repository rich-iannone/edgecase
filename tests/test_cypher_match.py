"""Tests for the Cypher query planner and matcher (`edgecase.cypher.Plan` and `Matcher`)."""

import itertools
from collections import Counter

import edgecase as ec
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


def rows(g, text, params=None, distinct_nodes=False, label_key="name"):
    """The named variables of each match, as label (nodes) or ID (edges) tuples, in order."""
    query = parse_query(text)
    matcher = Matcher(query, text, g._graph, params, label_key, distinct_nodes)
    names = list(query.variables)
    out = []
    for b in matcher.matches():
        out.append(tuple(
            g.get_label_from_id(b[v]) if query.variables[v] == "node" else b[v] for v in names
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

    def test_matches_is_lazy(self, g):
        text = "MATCH (x), (y), (z)"
        matches = Matcher(parse_query(text), text, g._graph).matches()
        assert len(list(itertools.islice(matches, 3))) == 3

    def test_search_order_is_fixed(self, g):
        text = "MATCH (x)--(y)--(z)"
        assert rows(g, text) == rows(g, text)


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
        assert len(p.constant) == 1

    def test_groups_are_scanned_in_turn(self):
        # the most constrained group goes first
        assert plan("MATCH (a)-->(b), (c)-->(d:L)") == [
            "scan d", "expand (d)<-[ 1]-(c)", "scan a", "expand (a)-[ 0]->(b)"]


# =============================================================================
# Against a brute-force oracle
# =============================================================================


def oracle(g, text, params=None, distinct_nodes=False):
    """Every assignment of nodes and distinct edges to the pattern elements that fits the arrows
    and the conditions, by enumeration (the named variables of each, as a Counter)."""
    query = parse_query(text)
    kinds, rels, conditions = {}, [], []

    def slot(element, i):
        name = element.variable or f"anon{i}"
        kinds.setdefault(name, "node" if isinstance(element, NodePattern) else "relationship")
        conditions.extend(element_conditions(name, element))
        return name

    i = 0
    for pattern in query.patterns:
        left = slot(pattern.nodes[0], i := i + 1)
        for rel, node in zip(pattern.rels, pattern.nodes[1:]):
            r = slot(rel, i := i + 1)
            right = slot(node, i := i + 1)
            rels.append((r, left, right, rel.direction))
            left = right
    conditions += conjuncts(query.where)
    evaluator = Evaluator(text, kinds, params, GraphView(g._graph))
    tests = [evaluator.condition(c) for c in conditions]

    node_slots = [s for s, k in kinds.items() if k == "node"]
    rel_slots = [s for s, k in kinds.items() if k == "relationship"]
    edges = {e: g.get_edge(e) for e in g.get_edge_ids(limit=None)}
    found = Counter()
    for nodes in itertools.product(sorted(g.get_node_ids(limit=None)), repeat=len(node_slots)):
        if distinct_nodes and len(set(nodes)) < len(nodes):
            continue
        for chosen in itertools.permutations(sorted(edges), len(rel_slots)):
            b = dict(zip(node_slots, nodes)) | dict(zip(rel_slots, chosen))
            if all(_fits(edges[b[r]], b[x], b[y], arrow, g.is_directed())
                   for r, x, y, arrow in rels) and all(t(b) for t in tests):
                found[tuple(b[v] for v in query.variables)] += 1
    return found


def _fits(e, x, y, arrow, directed):
    forward, backward = (e.source, e.target) == (x, y), (e.source, e.target) == (y, x)
    if not directed or arrow == "either":
        return forward or backward
    return forward if arrow == "out" else backward


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
    node_names, rel_count = [], 0

    def node():
        name = draw(st.sampled_from(["a", "b", "c", ""]))
        if name:
            node_names.append(name)
        # mostly unconstrained, so that most queries have matches
        labels = draw(st.sampled_from(["", "", "", ":P", ":Q", ":P:Q", ":P|Q"]))
        props = draw(st.sampled_from(["", "", "", "", " {k: 1}", " {k: $k}"]))
        return f"({name}{labels}{props})"

    def rel():
        nonlocal rel_count
        rel_count += 1
        name = draw(st.sampled_from([f"r{rel_count}", ""]))
        types = draw(st.sampled_from(["", "", ":A", ":A|B"]))
        props = draw(st.sampled_from(["", "", "", " {weight: 2}"]))
        arrow = draw(st.sampled_from(["-[{}]->", "<-[{}]-", "-[{}]-"]))
        return arrow.format(name + types + props), name

    patterns, rel_names = [], []
    for _ in range(draw(st.integers(1, 2))):
        text = node()
        for _ in range(draw(st.integers(0, 2))):
            r, name = rel()
            if name:
                rel_names.append(name)
            text += r + node()
        patterns.append(text)
    text = "MATCH " + ", ".join(patterns)

    templates = ["{n}.k = 1", "{n}.k IS NULL", "{n}:P OR {m}:Q", "NOT {n}:P", "{n} <> {m}",
                 "id({n}) = 2", "id({n}) IN [1, 3]", "{n}.k < {m}.k"]
    if rel_names:
        templates += ["type({r}) = 'A'", "{r}.weight > 1"]
    if node_names and draw(st.booleans()):
        parts = []
        for _ in range(draw(st.integers(1, 2))):
            parts.append(draw(st.sampled_from(templates)).format(
                n=draw(st.sampled_from(node_names)), m=draw(st.sampled_from(node_names)),
                r=draw(st.sampled_from(rel_names or ["x"]))))
        text += " WHERE " + draw(st.sampled_from([" AND ", " OR "])).join(parts)
    return text


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
]


@pytest.mark.parametrize("text", FIXED_QUERIES)
@given(g=graphs(), distinct_nodes=st.booleans())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_typical_queries_against_the_oracle(text, g, distinct_nodes):
    query = parse_query(text)
    expected = oracle(g, text, None, distinct_nodes)
    found = Counter(
        tuple(b[v] for v in query.variables)
        for b in Matcher(query, text, g._graph, distinct_nodes=distinct_nodes).matches()
    )
    assert found == expected


@given(g=graphs(), text=queries(), distinct_nodes=st.booleans())
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_against_the_oracle(g, text, distinct_nodes):
    query = parse_query(text)
    slots = Plan(query).slots
    assume(sum(k == "node" for k in slots.values()) <= 4)
    params = {"k": 1}
    expected = oracle(g, text, params, distinct_nodes)
    found = Counter(
        tuple(b[v] for v in query.variables)
        for b in Matcher(query, text, g._graph, params, distinct_nodes=distinct_nodes).matches()
    )
    assert found == expected, text
