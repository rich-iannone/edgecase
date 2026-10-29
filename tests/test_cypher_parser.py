"""Tests for the Cypher parser (`edgecase.cypher`): tokens, patterns, expressions and errors."""

import pytest
from edgecase.cypher import (
    Aggregate,
    BoolOp,
    Call,
    Compare,
    HasLabels,
    IsNull,
    ListExpr,
    Literal,
    Neg,
    NodePattern,
    Not,
    Param,
    Prop,
    RelPattern,
    Var,
    aggregates_in,
    parse_create,
    parse_query,
    substitute,
    walk,
)


def _where(condition):
    """The WHERE expression of a query over `(a)-[r]->(b)`."""
    return parse_query(f"MATCH (a)-[r]->(b) WHERE {condition}").clauses[0].where


# =============================================================================
# Patterns
# =============================================================================


class TestPatterns:
    def test_nodes(self):
        q = parse_query("MATCH (p:Person:Actor {name: 'Tom', born: $year}), (:Movie|Show), ()")
        p, movie, empty = (pattern.nodes[0] for pattern in q.patterns)
        assert p == NodePattern("p", [("Person",), ("Actor",)],
                                {"name": Literal("Tom"), "born": Param("year")})
        assert movie == NodePattern(None, [("Movie", "Show")])
        assert empty == NodePattern()
        assert q.variables == {"p": "node"}

    def test_relationships(self):
        q = parse_query("MATCH (a)-[r:KNOWS|LIKES {since: 2020}]->(b)<-[:R]-(c)-[]-(d)--(e)<--(f)")
        rels = q.patterns[0].rels
        assert rels[0] == RelPattern("r", ["KNOWS", "LIKES"], {"since": Literal(2020)}, "out")
        assert rels[1] == RelPattern(None, ["R"], {}, "in")
        assert [r.direction for r in rels[2:]] == ["either", "either", "in"]
        assert q.variables == {"a": "node", "r": "relationship", "b": "node", "c": "node",
                               "d": "node", "e": "node", "f": "node"}

    def test_old_type_alternation(self):
        q = parse_query("MATCH ()-[:A|:B]->()")
        assert q.patterns[0].rels[0].types == ["A", "B"]

    def test_null_property_is_kept(self):
        q = parse_query("MATCH (a {k: null, l: [1, -2]})")
        assert q.patterns[0].nodes[0].props == {"k": Literal(None), "l": Literal([1, -2])}

    def test_params(self):
        q = parse_query("MATCH (a {x: $x, y: $`odd name`, z: $0})")
        assert list(q.patterns[0].nodes[0].props.values()) == [
            Param("x"), Param("odd name"), Param("0"),
        ]

    def test_repeated_node_variable_is_a_join(self):
        q = parse_query("MATCH (a)-->(b), (b)-->(a)")
        assert q.variables == {"a": "node", "b": "node"}

    def test_create(self):
        (pattern,) = parse_create("CREATE (a:P {k: null})-[r:R]->(b)")
        assert pattern.nodes[0] == NodePattern("a", [("P",)], {"k": Literal(None)})
        assert pattern.rels[0] == RelPattern("r", ["R"], {}, "out")

    @pytest.mark.parametrize(
        "text, length",
        [("*", (1, None)), ("*2", (2, 2)), ("*0", (0, 0)), ("*1..3", (1, 3)), ("*2..", (2, None)),
         ("*..3", (1, 3)), ("*..", (1, None)), ("* 0 .. 2", (0, 2)), ("", None)],
    )
    def test_variable_length(self, text, length):
        q = parse_query(f"MATCH (a)-[r:R{text} {{k: 1}}]->(b)")
        assert q.patterns[0].rels[0] == RelPattern("r", ["R"], {"k": Literal(1)}, "out", length)
        assert q.variables["r"] == ("relationship" if length is None else "relationships")

    def test_anonymous_variable_length(self):
        assert parse_query("MATCH (a)<-[*2]-(b)").patterns[0].rels[0] == RelPattern(
            None, [], {}, "in", (2, 2))

    def test_path_variables(self):
        q = parse_query("MATCH p = (a)-[r*]->(b), (c), q=(d)--(e)")
        assert [pattern.variable for pattern in q.patterns] == ["p", None, "q"]
        assert [pattern.pos for pattern in q.patterns] == [6, 25, 30]
        assert q.variables == {"p": "path", "a": "node", "r": "relationships", "b": "node",
                               "c": "node", "q": "path", "d": "node", "e": "node"}
        assert all(pattern.shortest is None for pattern in q.patterns)

    def test_shortest_paths(self):
        q = parse_query("MATCH p = shortestPath((a)-[*]->(b)), ALLSHORTESTPATHS((c)-[r:R*0..3]-(d))")
        one, every = q.patterns
        assert (one.variable, one.shortest, one.pos) == ("p", "one", 6)
        assert (every.variable, every.shortest) == (None, "all")
        assert every.rels[0] == RelPattern("r", ["R"], {}, "either", (0, 3))
        assert list(q.variables) == ["p", "a", "b", "c", "r", "d"]
        # the function names are only special before `(`
        assert parse_query("MATCH (shortestPath)").variables == {"shortestPath": "node"}

    def test_positions(self):
        q = parse_query("MATCH (a)\n  -[r]->(b)")
        assert q.patterns[0].nodes[0].pos == 6
        assert q.patterns[0].rels[0].pos == 12


# =============================================================================
# Expressions
# =============================================================================


class TestExpressions:
    def test_comparisons(self):
        for op in ("=", "<>", "<", "<=", ">", ">=", "=~"):
            assert _where(f"a.x {op} 1") == Compare(op, Prop(Var("a"), "x"), Literal(1))

    def test_arrows_next_to_comparisons(self):
        # `<-` in an expression is `<` then a minus sign
        assert _where("a.x<-1") == Compare("<", Prop(Var("a"), "x"), Literal(-1))
        assert _where("a.x<>-b.y") == Compare("<>", Prop(Var("a"), "x"),
                                              Neg(Prop(Var("b"), "y")))
        assert _where("a.x>=1") == Compare(">=", Prop(Var("a"), "x"), Literal(1))

    def test_chained_comparison(self):
        assert _where("1 < a.x <= 3") == BoolOp("AND", [
            Compare("<", Literal(1), Prop(Var("a"), "x")),
            Compare("<=", Prop(Var("a"), "x"), Literal(3)),
        ])

    def test_precedence(self):
        x, y, z = (Compare("=", Prop(Var("a"), k), Literal(1)) for k in "xyz")
        assert _where("a.x = 1 OR a.y = 1 AND NOT a.z = 1") == BoolOp("OR", [
            x, BoolOp("AND", [y, Not(z)]),
        ])
        assert _where("(a.x = 1 OR a.y = 1) AND a.z = 1") == BoolOp("AND", [
            BoolOp("OR", [x, y]), z,
        ])
        assert _where("a.x = 1 XOR a.y = 1 OR a.z = 1") == BoolOp("OR", [
            BoolOp("XOR", [x, y]), z,
        ])
        assert _where("a.x = 1 AND a.y = 1 AND a.z = 1") == BoolOp("AND", [x, y, z])

    def test_predicates(self):
        name = Prop(Var("a"), "name")
        assert _where("a.name STARTS WITH 'T'") == Compare("STARTS WITH", name, Literal("T"))
        assert _where("a.name ends with $s") == Compare("ENDS WITH", name, Param("s"))
        assert _where("a.name CONTAINS 'o'") == Compare("CONTAINS", name, Literal("o"))
        assert _where("a.name IN ['x', $y]") == Compare(
            "IN", name, ListExpr([Literal("x"), Param("y")]))
        assert _where("a.name IN $names") == Compare("IN", name, Param("names"))
        assert _where("a.name IS NULL") == IsNull(name)
        assert _where("a.name IS NOT NULL") == IsNull(name, negated=True)
        assert _where("NOT a.name IS NULL") == Not(IsNull(name))

    def test_labels_and_functions(self):
        assert _where("a:Person:Actor") == HasLabels(Var("a"), [("Person",), ("Actor",)])
        assert _where("a:A|B") == HasLabels(Var("a"), [("A", "B")])
        assert _where("toLower(a.name) = 'tom'") == Compare(
            "=", Call("toLower", [Prop(Var("a"), "name")]), Literal("tom"))
        assert _where("id(a) <> id(b)") == Compare("<>", Call("id", [Var("a")]),
                                                   Call("id", [Var("b")]))
        assert _where("type(r) = 'R' AND size(labels(a)) > 0").operands[1] == Compare(
            ">", Call("size", [Call("labels", [Var("a")])]), Literal(0))

    def test_literals(self):
        assert _where("a.x = TRUE").right == Literal(True)
        assert _where("a.x = null").right == Literal(None)
        assert _where("a.x = 'it\\'s'").right == Literal("it's")
        assert _where("a.x = 2.5e1").right == Literal(25.0)
        assert _where("a.x = -(1)").right == Literal(-1)
        assert _where("a.x IN []").right == ListExpr([])

    def test_backticked_variable(self):
        q = parse_query("MATCH (`my var`) WHERE `my var`.`a key` = 1 RETURN `my var`")
        assert q.clauses[0].where == Compare("=", Prop(Var("my var"), "a key"), Literal(1))
        assert q.returns.items[0].name == "my var"

    def test_walk(self):
        found = [e for e in walk(_where("a.x = 1 AND NOT toLower(b.y) IN [r.z]"))
                 if isinstance(e, Var)]
        assert [v.name for v in found] == ["a", "b", "r"]


# =============================================================================
# RETURN
# =============================================================================


class TestClauses:
    def test_several_clauses(self):
        q = parse_query("MATCH (a)-[r]->(b) WHERE a.x = 1 OPTIONAL MATCH (b)-->(c), (d) "
                        "match (c)-[r]->(e) WHERE e.y = d.y RETURN a, e")
        assert [c.optional for c in q.clauses] == [False, True, False]
        assert [len(c.patterns) for c in q.clauses] == [1, 2, 1]
        assert q.clauses[0].where == Compare("=", Prop(Var("a"), "x"), Literal(1))
        assert q.clauses[1].where is None
        assert q.clauses[2].where.right == Prop(Var("d"), "y")
        assert list(q.variables) == ["a", "r", "b", "c", "d", "e"]
        assert len(q.patterns) == 4

    def test_optional_first(self):
        q = parse_query("OPTIONAL MATCH (a) RETURN a")
        assert q.clauses[0].optional and q.clauses[0].pos == 0

    def test_where_sees_earlier_clauses(self):
        q = parse_query("MATCH (a) MATCH (b) WHERE b.x = a.x")
        assert q.clauses[0].where is None

    def test_relationship_reused_in_a_later_clause(self):
        assert parse_query("MATCH ()-[r]->() MATCH ()-[r]-()").variables == {
            "r": "relationship"}


class TestReturn:
    def test_items(self):
        q = parse_query("MATCH (p)-[r]->(m) RETURN p, r, m.title;")
        assert [item.name for item in q.returns.items] == ["p", "r", "m.title"]
        assert q.returns.items[2].expr == Prop(Var("m"), "title")
        assert not q.returns.distinct

    def test_distinct(self):
        assert parse_query("MATCH (p) RETURN DISTINCT p.name").returns.distinct

    def test_no_where_or_return(self):
        q = parse_query("match (p)")
        assert q.clauses[0].where is None and q.returns is None

    def test_names(self):
        # The alias, else `v` or `v.key`, else the expression as written (spacing and case kept)
        q = parse_query("MATCH p = (a)-[r]->(b) RETURN a.x AS ax, length( p ), toLower(a.name), "
                        "[a,  b], -a.x AS n, id(r)")
        assert [item.name for item in q.returns.items] == [
            "ax", "length( p )", "toLower(a.name)", "[a,  b]", "n", "id(r)"]
        assert q.returns.items[1].expr == Call("length", [Var("p")])

    def test_alias_of_a_variable(self):
        q = parse_query("MATCH (a) RETURN a AS node, a.x as x")
        assert [(item.expr, item.name) for item in q.returns.items] == [
            (Var("a"), "node"), (Prop(Var("a"), "x"), "x")]

    def test_star(self):
        q = parse_query("MATCH p = (a)-[r]->(b) RETURN *, a.x")
        assert q.returns.star
        assert [item.name for item in q.returns.items] == ["p", "a", "r", "b", "a.x"]
        # Anonymous parts of the pattern aren't columns
        q = parse_query("MATCH (a)-->() RETURN *")
        assert [item.name for item in q.returns.items] == ["a"]

    def test_distinct_star(self):
        q = parse_query("MATCH (a)-->(b) RETURN DISTINCT *")
        assert q.returns.distinct and [item.name for item in q.returns.items] == ["a", "b"]

    def test_order_by(self):
        q = parse_query("MATCH (a) RETURN a ORDER BY a.x, a.y DESC, a.z ASC, a.w DESCENDING, "
                        "a.v ascending")
        assert [(key.expr, key.descending) for key in q.returns.order] == [
            (Prop(Var("a"), "x"), False), (Prop(Var("a"), "y"), True),
            (Prop(Var("a"), "z"), False), (Prop(Var("a"), "w"), True),
            (Prop(Var("a"), "v"), False)]
        assert q.returns.skip is None and q.returns.limit is None

    def test_order_by_a_column(self):
        # A column name stands for the column's expression
        q = parse_query("MATCH (a) RETURN a.x AS x, a.y AS a ORDER BY x DESC, a, toLower(x)")
        assert [key.expr for key in q.returns.order] == [
            Prop(Var("a"), "x"), Prop(Var("a"), "y"), Call("toLower", [Prop(Var("a"), "x")])]

    def test_order_by_column_not_substituted_twice(self):
        # `a` is replaced by `a.y`, and the `a` inside `a.y` stays the variable
        q = parse_query("MATCH (a) RETURN a.y AS a ORDER BY a.z")
        assert q.returns.order[0].expr == Prop(Prop(Var("a"), "y"), "z")

    def test_order_by_after_distinct(self):
        q = parse_query("MATCH (a) RETURN DISTINCT a.x AS x, a.y ORDER BY x, a.y DESC")
        assert [key.expr for key in q.returns.order] == [Prop(Var("a"), "x"), Prop(Var("a"), "y")]

    def test_skip_and_limit(self):
        q = parse_query("MATCH (a) RETURN a SKIP 2 LIMIT $n")
        assert q.returns.skip == Literal(2) and q.returns.limit == Param("n")
        q = parse_query("MATCH (a) RETURN a ORDER BY a.x LIMIT size([1, 2])")
        assert q.returns.skip is None and q.returns.limit.name == "size"

    def test_keywords_case_insensitive(self):
        q = parse_query("match (a) return distinct a.x as x order by x desc skip 1 limit 2")
        assert q.returns.distinct and q.returns.items[0].name == "x"
        assert q.returns.order[0].descending
        assert q.returns.skip == Literal(1) and q.returns.limit == Literal(2)


class TestAggregates:
    def test_calls(self):
        q = parse_query("MATCH (a)-[r]->(b) RETURN count(*), COUNT(DISTINCT b), collect(a.x), "
                        "Sum(r.weight) AS w, avg(a.x), min(b), max(DISTINCT a.x)")
        assert [item.expr for item in q.returns.items] == [
            Aggregate("count", None), Aggregate("count", Var("b"), True),
            Aggregate("collect", Prop(Var("a"), "x")), Aggregate("sum", Prop(Var("r"), "weight")),
            Aggregate("avg", Prop(Var("a"), "x")), Aggregate("min", Var("b")),
            Aggregate("max", Prop(Var("a"), "x"), True)]
        # the name as written, for the column
        assert [item.name for item in q.returns.items] == [
            "count(*)", "COUNT(DISTINCT b)", "collect(a.x)", "w", "avg(a.x)", "min(b)",
            "max(DISTINCT a.x)"]

    def test_inside_expressions(self):
        q = parse_query("MATCH (a) RETURN a.x, size(collect(a)) AS n, [a.x, count(*)]")
        assert q.returns.items[1].expr == Call("size", [Aggregate("collect", Var("a"))])
        assert aggregates_in(q.returns.items[2].expr) == [Aggregate("count", None)]
        assert aggregates_in(q.returns.items[0].expr) == []

    def test_grouping_keys(self):
        # Beside an aggregation, a column may use the grouping keys, or expressions of them
        for text in ("MATCH (a)-->(b) RETURN a, b.x, [a.y, b.x, count(*)]",
                     "MATCH (a) RETURN a.x, toLower(a.x) = max(a.y)",
                     "MATCH (a) RETURN count(*) AS n, [1, sum(a.x)]"):
            parse_query(text)

    def test_order_by(self):
        # ORDER BY uses the returned columns, by name or by expression, and the aggregations
        # that RETURN returns
        q = parse_query("MATCH (a) RETURN a, count(*) AS n, max(a.x) ORDER BY n DESC, max(a.x), "
                        "a.y, toLower(n)")
        assert [key.expr for key in q.returns.order] == [
            Aggregate("count", None), Aggregate("max", Prop(Var("a"), "x")),
            Prop(Var("a"), "y"), Call("toLower", [Aggregate("count", None)])]

    def test_substitute_inside(self):
        expr = Aggregate("collect", Var("x"), True)
        assert substitute(expr, {"x": Var("y")}) == Aggregate("collect", Var("y"), True)
        assert substitute(Aggregate("count", None), {"x": Var("y")}) == Aggregate("count", None)


class TestWith:
    def test_parts(self):
        # each WITH ends a part, which the next part takes as its `previous`
        q = parse_query("MATCH (a)-->(b) WITH b, count(*) AS n WHERE n > 1 "
                        "MATCH (b)-->(c) RETURN b, n, c")
        first = q.previous
        assert first.previous is None and first.inputs == {}
        assert first.returns.keyword == "WITH"
        assert [item.name for item in first.returns.items] == ["b", "n"]
        assert first.returns.where == Compare(">", Var("n"), Literal(1))
        assert q.returns.keyword == "RETURN" and q.returns.where is None
        assert q.inputs == {"b": "node", "n": "value"}
        assert list(q.variables) == ["b", "n", "c"]

    def test_kinds_of_the_inputs(self):
        # a variable keeps its kind, as does an expression that can only give a node or a
        # relationship; anything else is a value
        q = parse_query("MATCH p = (a)-[r]->(b)-[s*]->(c) WITH p, r, s, a AS x, min(b) AS y, "
                        "a.k AS k, [a] AS l RETURN *")
        assert q.inputs == {"p": "path", "r": "relationship", "s": "relationships",
                            "x": "node", "y": "node", "k": "value", "l": "value"}

    def test_chains_and_star(self):
        q = parse_query("MATCH (a)-->(b) WITH * WITH a WHERE a.k = 1 RETURN a")
        assert q.inputs == {"a": "node"} and q.previous.inputs == {"a": "node", "b": "node"}
        assert q.previous.previous.returns.star

    def test_distinct_order_skip_limit(self):
        q = parse_query("MATCH (a) WITH DISTINCT a.k AS k ORDER BY k DESC SKIP 1 LIMIT $n "
                        "WHERE k > 0 RETURN k")
        w = q.previous.returns
        assert w.distinct and w.order[0].descending
        assert w.skip == Literal(1) and w.limit == Param("n")
        assert w.where == Compare(">", Var("k"), Literal(0))

    def test_a_final_part_without_return(self):
        q = parse_query("MATCH (a) WITH a MATCH (a)-->(b)")
        assert q.returns is None and q.inputs == {"a": "node"}
        assert list(q.variables) == ["a", "b"]

    def test_names_can_be_reused_after_with(self):
        # a variable that WITH drops can be defined again, as a new one
        q = parse_query("MATCH (a)-->(b) WITH a MATCH (b) RETURN a, b")
        assert q.inputs == {"a": "node"} and list(q.variables) == ["a", "b"]

    def test_aggregates_in_with(self):
        q = parse_query("MATCH (a) WITH collect(a) AS all, count(*) AS n RETURN all, n")
        assert q.previous.returns.items[0].expr == Aggregate("collect", Var("a"))


class TestSubstitute:
    def test_replaces_variables(self):
        names = {"x": Prop(Var("a"), "x")}
        expr = parse_query("MATCH (x) WHERE x > 1 AND NOT x IN [x, 2] AND f(x) IS NULL "
                           "RETURN x").clauses[0].where
        result = substitute(expr, names)
        assert not any(isinstance(e, Var) and e.name == "x" for e in walk(result))
        assert sum(e == Prop(Var("a"), "x") for e in walk(result)) == 4

    def test_leaves_others(self):
        expr = Compare("=", Var("b"), Literal(1))
        assert substitute(expr, {"a": Literal(2)}) == expr


# =============================================================================
# Errors
# =============================================================================


class TestErrors:
    @pytest.mark.parametrize(
        "text, message",
        [
            ("", r"starts with MATCH, but the text is empty"),
            ("RETURN 1", r"starts with MATCH, found 'RETURN' \(line 1, column 1\)"),
            ("CREATE (a)", r"CREATE isn't supported: a query only reads the graph"),
            ("OPTIONAL (a)", r"expected MATCH, found '\(' \(line 1, column 10\)"),
            ("MATCH (a) OPTIONAL RETURN a", r"expected MATCH, found 'RETURN'"),
            ("MATCH (a) RETURN a MATCH (b)",
             r"expected ORDER BY, SKIP, LIMIT or the end of the query, found 'MATCH'"),
            ("MATCH (a) ORDER BY a.x", r"ORDER BY needs a RETURN before it \(RETURN \* returns "
                                       r"every variable\) \(line 1, column 11\)"),
            ("MATCH (a) LIMIT 3", r"LIMIT needs a RETURN before it"),
            ("MATCH (a) RETURN a LIMIT 3 SKIP 1", r"ORDER BY, SKIP and LIMIT come after RETURN, "
                                                 r"in that order \(line 1, column 28\)"),
            ("MATCH (a) RETURN a SKIP 1 ORDER BY a", r"ORDER BY, SKIP and LIMIT come after"),
            ("MATCH (a) RETURN a ORDER a", r"expected BY, found 'a'"),
            ("MATCH (a) RETURN a ORDER BY a a", r"expected SKIP, LIMIT or the end of the query, "
                                               r"found 'a'"),
            ("MATCH (a) RETURN a SKIP 1 1", r"expected LIMIT or the end of the query, found '1'"),
            ("MATCH (a) RETURN a LIMIT 1 1", r"expected the end of the query, found '1'"),
            ("MATCH (a) RETURN a SKIP", r"expected an expression, but the text ended"),
            ("MATCH (a) RETURN a LIMIT a.k", r"LIMIT can't use the variable `a`; it takes a "
                                            r"number or a parameter \(line 1, column 26\)"),
            ("MATCH (a) RETURN a SKIP size([a])", r"SKIP can't use the variable `a`"),
            ("MATCH (a) RETURN a ORDER BY b", r"the variable `b` isn't defined in MATCH, or as a "
                                             r"column of RETURN \(line 1, column 29\)"),
            ("MATCH (a) RETURN DISTINCT a.x ORDER BY a.y",
             r"after RETURN DISTINCT, ORDER BY can only use the returned columns, and `a` isn't "
             r"one \(line 1, column 40\)"),
            ("MATCH (a)-->(b) RETURN DISTINCT a ORDER BY b", r"and `b` isn't one"),
            ("MATCH (a) RETURN a ORDER BY count(a)",
             r"ORDER BY can only sort by an aggregation that RETURN returns; return count\(\) as "
             r"a column, and sort by its name \(line 1, column 29\)"),
            ("MATCH (a) RETURN a.x, count(*) ORDER BY max(a.y)", r"return max\(\) as a column"),
            ("MATCH (a)-->(b) RETURN a.x, count(*) ORDER BY b.x",
             r"after an aggregation, ORDER BY can only use the returned columns, and `b` isn't "
             r"one \(line 1, column 47\)"),
            ("MATCH (a) RETURN a.x, count(*) ORDER BY a.y", r"and `a` isn't one"),
            ("MATCH (a)-->(b) RETURN a.x, [b.x, count(*)]",
             r"`b` is used beside an aggregation, so it must be a grouping key: return `b` \(or "
             r"the expression it's in\) as a column of its own \(line 1, column 30\)"),
            ("MATCH (a) RETURN a.x, [a.y, count(*)]", r"`a` is used beside an aggregation"),
            ("MATCH (a) RETURN count(*), a.x = max(a.x)", r"`a` is used beside an aggregation"),
            ("MATCH (a) WHERE count(*) > 1", r"count\(\) is an aggregating function, which only "
                                             r"RETURN, WITH and ORDER BY can use \(line 1, "
                                             r"column 17\)"),
            ("MATCH (a) RETURN a LIMIT count(*)", r"count\(\) is an aggregating function"),
            ("MATCH (a {x: count(*)})", r"expected a value, found 'count'"),
            ("MATCH (a) RETURN count(max(a.x))", r"aggregating functions can't be nested, as max\(\) "
                                                r"is in count\(\) \(line 1, column 24\)"),
            ("MATCH (a) RETURN collect([Sum(a.x)])", r"as Sum\(\) is in collect\(\)"),
            ("MATCH (a) RETURN sum(*)", r"only count\(\) takes \*, as count\(\*\); sum\(\) takes an "
                                        r"expression \(line 1, column 18\)"),
            ("MATCH (a) RETURN count(DISTINCT *)", r"count\(\*\) counts rows, so it can't be "
                                                   r"DISTINCT"),
            ("MATCH (a) RETURN count()", r"count\(\) takes one argument, or \* to count the rows"),
            ("MATCH (a) RETURN min()", r"min\(\) takes one argument \(line 1, column 18\)"),
            ("MATCH (a) RETURN collect(a, a)", r"collect\(\) takes one argument \(line 1, column "
                                               r"27\)"),
            ("MATCH (a) RETURN count(*", r"expected '\)' to close count\(\*\), but the text "
                                         r"ended"),
            ("MATCH (a) RETURN avg(a.x", r"expected '\)' to close avg\(\)"),
            ("MATCH (a) RETURN count(DISTINCT)", r"count\(\) takes one argument, or \*"),
            ("MATCH (a) RETURN a AS", r"expected a column name, but the text ended"),
            ("MATCH (a) RETURN a.x AS a, a", r"the column 'a' is returned twice"),
            ("MATCH (a) RETURN *, a", r"the column 'a' is returned twice \(line 1, column 21\)"),
            ("MATCH () RETURN *", r"RETURN \* returns every named variable, but the query names "
                                  r"none \(line 1, column 10\)"),
            ("MATCH (a) RETURN a, *", r"expected an expression, found '\*'"),
            ("MATCH (a) WITH a", r"a query can't end with WITH; finish it with RETURN \(or "
                                 r"MATCH\) \(line 1, column 11\)"),
            ("MATCH (a) WITH a WHERE a.k = 1", r"a query can't end with WITH"),
            ("WITH 1 AS x RETURN x", r"a query starts with MATCH, found 'WITH'"),
            ("MATCH (a) WITH a.x RETURN 1",
             r"WITH needs a name for each expression that isn't a variable, as in "
             r"WITH a.name AS name \(line 1, column 16\)"),
            ("MATCH (a) WITH count(*) RETURN 1", r"WITH needs a name for each expression"),
            ("MATCH (a)-->(b) WITH a RETURN b",
             r"the variable `b` isn't carried over by the WITH before it; list it in that WITH "
             r"to use it after \(line 1, column 31\)"),
            ("MATCH (a) WITH a.x AS v, a WITH a RETURN v", r"`v` isn't carried over"),
            ("MATCH (a)-->(b) WITH a MATCH (a) WHERE b.k = 1", r"`b` isn't carried over"),
            ("MATCH (a) WITH a RETURN c", r"the variable `c` isn't defined in MATCH or WITH "
                                          r"\(line 1, column 25\)"),
            ("MATCH (a)-->(b) WITH a WHERE b.x = 1 RETURN a",
             r"the WHERE of WITH sees only the columns of WITH, and `b` isn't one; list it in "
             r"the WITH to use it \(line 1, column 30\)"),
            ("MATCH (a) WITH a.x AS v WHERE a.x = 1 RETURN v", r"and `a` isn't one"),
            ("MATCH (a) WITH a WHERE count(*) > 1 RETURN a",
             r"count\(\) is an aggregating function, which only RETURN, WITH and ORDER BY"),
            ("MATCH (a) WITH a.x AS v MATCH (v) RETURN v",
             r"`v` holds a value that WITH carries over, not a node or relationship, so it "
             r"can't be used in a pattern \(line 1, column 31\)"),
            ("MATCH (a) WITH count(*) AS n MATCH ()-[n]->() RETURN n", r"`n` holds a value"),
            ("MATCH p = (a) WITH p MATCH (p) RETURN p", r"`p` is a path variable, so it can't "
                                                       r"also be a node"),
            ("MATCH (a)-[r*]->(b) WITH r MATCH ()-[r]->() RETURN r",
             r"the variable-length relationship `r` can't be used again in a later MATCH"),
            ("MATCH (a) WITH a WHERE a.k > 1 ORDER BY a.k RETURN a",
             r"ORDER BY, SKIP and LIMIT come after WITH, in that order, and before its WHERE "
             r"\(line 1, column 32\)"),
            ("MATCH (a) WITH a LIMIT 1 SKIP 1 RETURN a", r"ORDER BY, SKIP and LIMIT come after"),
            ("MATCH (a) WITH a a", r"expected ORDER BY, SKIP, LIMIT, WHERE, MATCH, WITH or "
                                   r"RETURN, found 'a' \(line 1, column 18\)"),
            ("MATCH (a) WITH a, a RETURN a", r"the column 'a' is listed twice \(line 1, column "
                                             r"19\)"),
            ("MATCH () WITH * RETURN 1", r"WITH \* lists every named variable, but the query "
                                         r"names none"),
            ("MATCH (a) WITH a ORDER BY b RETURN a",
             r"the variable `b` isn't defined in MATCH, or as a column of WITH"),
            ("MATCH (a) WITH a.x AS v, count(*) AS n ORDER BY a.y RETURN v",
             r"after an aggregation, ORDER BY can only use the listed columns, and `a` isn't "
             r"one"),
            ("MATCH (a)-->(b) WITH a.x AS v, [b.x, count(*)] AS w RETURN v",
             r"`b` is used beside an aggregation, so it must be a grouping key: list `b`"),
            ("MATCH (a) WITH a ORDER BY count(*) RETURN a",
             r"ORDER BY can only sort by an aggregation that WITH lists; list count\(\) as a "
             r"column"),
            ("MATCH (a) WITH DISTINCT a.x AS v ORDER BY a.y RETURN v",
             r"after WITH DISTINCT"),
            ("MATCH (a) b", r"expected WHERE, MATCH, WITH, RETURN or the end of the query, found "
                            r"'b'"),
            ("MATCH (a) WHERE a.x = 1 (b)",
             r"expected MATCH, WITH, RETURN or the end of the query, found '\('"),
            ("MATCH (a) RETURN a a", r"expected ORDER BY, SKIP, LIMIT or the end of the query, "
                                     r"found 'a'"),
            ("MATCH (a)-[*3..1]->(b)", r"can't have more than 1 hops and at least 3; write "
                                       r"\*1..3 \(line 1, column 12\)"),
            ("MATCH (a)-[*1.5]->(b)", r"a number of hops is a whole number, not '1.5'"),
            ("MATCH (a)-[*$n]->(b)", r"expected '\]', found '\$n'"),
            ("MATCH (a)-[:A:B]->(b)", r"a single type; write :A\|B"),
            ("MATCH (a)<-[r]->(b)", r"can't point both ways \(line 1, column 10\)"),
            ("MATCH (a)-[r]->(r)", r"`r` is used for both a node and a relationship"
                                   r" \(line 1, column 16\)"),
            ("MATCH (a)-[r]->(b), (b)-[r]->(c)",
             r"relationship variable `r` is used more than once in one MATCH \(line 1, column "
             r"24\)"),
            ("MATCH (a)-[r*]->(b), (b)-[r]->(c)", r"`r` is used more than once in one MATCH"),
            ("MATCH (a)-[r*]->(b) MATCH (b)-[r]->(c)",
             r"the variable-length relationship `r` can't be used again in a later MATCH "
             r"\(line 1, column 30\)"),
            ("MATCH (a)-[r]->(b) MATCH (b)-[r*]->(c)", r"`r` can't be used again"),
            ("MATCH (a)-[r]->(b) MATCH (r)", r"`r` is used for both a node and a relationship"),
            ("MATCH (a) WHERE b.x = 1 MATCH (b)",
             r"`b` is only defined in a later MATCH; a WHERE sees its own MATCH and the ones "
             r"before it \(line 1, column 17\)"),
            ("MATCH (a) WHERE b.x = 1", r"`b` isn't defined in MATCH \(line 1, column 17\)"),
            ("MATCH (a) RETURN c", r"`c` isn't defined in MATCH \(line 1, column 18\)"),
            ("MATCH (a) WHERE a.x =", r"expected an expression, but the text ended"),
            ("MATCH (a) WHERE AND", r"expected an expression, found 'AND'"),
            ("MATCH (a) WHERE a.x IS 1", r"expected NULL, found '1'"),
            ("MATCH (a) WHERE a.x STARTS 'T'", r"expected WITH, found \"'T'\""),
            ("MATCH (a) WHERE (a.x = 1", r"expected '\)' to close the parenthesis"),
            ("MATCH (a) WHERE a.x IN [1, 2", r"expected ',' or '\]' in the list"),
            ("MATCH (a) WHERE f(a, ", r"expected an expression, but the text ended"),
            ("MATCH (a) WHERE a.x = {k: 1}", r"map values aren't supported"),
            ("MATCH (a) WHERE a.x & 1", r"unexpected character '&'"),
            ("MATCH (a) WHERE a.x = $", r"unexpected character '\$'"),
            ("MATCH (a) RETURN a, a", r"the column 'a' is returned twice \(line 1, column 21\)"),
            ("MATCH (a {x: 1, x: 2})", r"'x' is given twice"),
            ("MATCH (a {x: [$y]})", r"expected a value, found '\$y'"),
            ("MATCH p = (a), p = (b)", r"the path variable `p` is defined twice \(line 1, column "
                                       r"16\)"),
            ("MATCH p = (a) MATCH p = (b)", r"the path variable `p` is defined twice"),
            ("MATCH a = (a)", r"`a` is a path variable, so it can't also be a node \(line 1, "
                              r"column 11\)"),
            ("MATCH p = (a) MATCH (b)-[p]-()", r"`p` is a path variable, so it can't also be a "
                                               r"relationship"),
            ("MATCH (a) MATCH a = (b)", r"`a` is used for both a path and a node \(line 1, "
                                        r"column 17\)"),
            ("MATCH ()-[r]->() MATCH r = ()", r"`r` is used for both a path and a relationship"),
            ("MATCH p = q = (a)", r"expected '\(' to start a node, found 'q'"),
            ("MATCH shortestPath((a))", r"shortestPath\(\) takes a pattern with one relationship, "
                                        r"such as \(a\)-\[\*\]-\(b\) \(line 1, column 7\)"),
            ("MATCH allShortestPaths((a)-[*]->(b)-->(c))",
             r"allShortestPaths\(\) takes a pattern with one relationship"),
            ("MATCH shortestPath((a)-->(b))", r"shortestPath\(\) needs a variable-length "
                                              r"relationship, such as -\[\*\]- or "
                                              r"-\[:KNOWS\*..5\]- \(line 1, column 23\)"),
            ("MATCH shortestPath((a)-[*2..]->(b))", r"shortestPath\(\) finds paths of at least 0 "
                                                    r"or 1 relationships, not 2"),
            ("MATCH shortestPath((a)-[*]->(b)", r"expected '\)' to close shortestPath\(\), but "
                                                r"the text ended"),
        ],
    )
    def test_query_errors(self, text, message):
        with pytest.raises(ValueError, match=r"^Cypher: .*" + message):
            parse_query(text)

    def test_aggregate_names_are_still_variables(self):
        returns = parse_query("MATCH (count) RETURN count, count.x").returns
        assert [item.name for item in returns.items] == ["count", "count.x"]

    @pytest.mark.parametrize(
        "text, message",
        [
            ("CREATE (a {k: $v})", r"parameters \(\$name\) can only be used in queries"
                                   r" \(line 1, column 15\)"),
            ("CREATE (a:A|B)", r"plain labels \(:A:B\), not alternatives \(:A\|B\)"
                               r" \(line 1, column 12\)"),
            ("CREATE (a)-[:A|B]->(b)", r"exactly one type \(line 1, column 15\)"),
            ("CREATE (a)-[*]->(b)", r"expected '\]', found '\*'"),
            ("CREATE p = (a)", r"a path variable \(p = \.\.\.\) can only be used in a query "
                               r"\(line 1, column 8\)"),
        ],
    )
    def test_create_errors(self, text, message):
        with pytest.raises(ValueError, match=r"^Cypher: .*" + message):
            parse_create(text)
