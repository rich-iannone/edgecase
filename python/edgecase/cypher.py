"""Cypher: the tokenizer, parser and syntax tree, the evaluator for query expressions, and the
query planner and matcher.

`from_cypher()` builds graphs from the `CREATE` clauses parsed here, and the query methods run the
`MATCH` queries. Parsing checks the syntax and how variables are used. The `Evaluator` gives the
expressions their meaning (which functions exist, how values compare, Cypher's `null` logic),
reading the graph through a `GraphView`. A `Plan` orders the search for a query's matches, and
the `Matcher` carries it out.

This module imports nothing from the rest of EdgeCase, so that both `io` and `graph` can use it.
"""

from __future__ import annotations

import itertools
import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

# =============================================================================
# Tokens
# =============================================================================

# `<-` and `->` aren't tokens of their own: a relationship is read from `<`, `-` and `>`, so that
# `a<-1` in an expression is still `a < -1`
_TOKEN = re.compile(
    r"""
      (?P<space>\s+|//[^\n]*|/\*.*?\*/)
    | (?P<name>[^\W\d]\w*)
    | (?P<quoted>`(?:[^`]|``)*`)
    | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    | (?P<number>(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)
    | (?P<param>\$(?:[^\W\d]\w*|\d+|`(?:[^`]|``)*`))
    | (?P<punct><>|<=|>=|=~|\.\.|[()\[\]{}:,;<>\-.=|*])
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPES = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", "'": "'", '"': '"', "\\": "\\"}

# words that can't be read as a variable inside an expression
_RESERVED = frozenset({
    "AND", "AS", "CONTAINS", "CREATE", "DISTINCT", "ENDS", "IN", "IS", "MATCH", "NOT", "OR",
    "RETURN", "STARTS", "WHERE", "WITH", "XOR",
})

_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=", "=~")

# aggregating functions, which RETURN rejects with a pointer to Polars
_AGGREGATES = frozenset({"avg", "collect", "count", "max", "min", "sum"})

# the shortest-path functions of a pattern, lower-cased -> as documented
_SHORTEST = {"shortestpath": "shortestPath", "allshortestpaths": "allShortestPaths"}


def position_error(fmt: str, text: str, pos: int, message: str) -> ValueError:
    """A `ValueError` for a problem at `pos` in `text`, giving its line and column."""
    line = text.count("\n", 0, pos) + 1
    column = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return ValueError(f"{fmt}: {message} (line {line}, column {column})")


def cypher_error(text: str, pos: int, message: str) -> ValueError:
    """A `ValueError` for a problem at `pos` in the Cypher `text`."""
    return position_error("Cypher", text, pos, message)


# =============================================================================
# Syntax tree
# =============================================================================

# Every node keeps the position of its text, for error messages; positions don't count when
# comparing nodes.


@dataclass
class Literal:
    """A string, number, boolean, `null` or list of those."""

    value: Any
    pos: int = field(default=0, compare=False)


@dataclass
class Param:
    """A query parameter, `$name`."""

    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Var:
    """A variable bound by the pattern."""

    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Prop:
    """A property lookup, `subject.key`."""

    subject: Expr
    key: str
    pos: int = field(default=0, compare=False)


@dataclass
class HasLabels:
    """A label test, `subject:A:B` or `subject:A|B` (the same form as `NodePattern.labels`)."""

    subject: Expr
    labels: list[tuple[str, ...]]
    pos: int = field(default=0, compare=False)


@dataclass
class Call:
    """A function call, `name(args)`; the name is kept as written."""

    name: str
    args: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class ListExpr:
    """A list of expressions, `[a, b]`."""

    items: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class Not:
    operand: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class Neg:
    """Unary minus on anything but a number literal (which is folded into the literal)."""

    operand: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class BoolOp:
    """`AND`, `OR` or `XOR` over two or more operands."""

    op: str
    operands: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class Compare:
    """A binary predicate: `=`, `<>`, `<`, `<=`, `>`, `>=`, `=~`, `IN`, `STARTS WITH`,
    `ENDS WITH` or `CONTAINS`. A chain such as `a < b < c` becomes an `AND` of comparisons."""

    op: str
    left: Expr
    right: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class IsNull:
    """`operand IS NULL`, or `IS NOT NULL` when `negated`."""

    operand: Expr
    negated: bool = False
    pos: int = field(default=0, compare=False)


Expr = Union[Literal, Param, Var, Prop, HasLabels, Call, ListExpr, Not, Neg, BoolOp, Compare,
             IsNull]


@dataclass
class NodePattern:
    """`(variable:Label {key: value})`.

    `labels` is a list of alternatives that must all hold: `:A:B` is `[("A",), ("B",)]` and
    `:A|B` is `[("A", "B")]`. Property values are `Literal`s or `Param`s; `null` values are kept.
    """

    variable: Optional[str] = None
    labels: list[tuple[str, ...]] = field(default_factory=list)
    props: dict[str, Expr] = field(default_factory=dict)
    pos: int = field(default=0, compare=False)


@dataclass
class RelPattern:
    """`-[variable:TYPE|OTHER *min..max {key: value}]->`, with `direction` `"out"` (`->`), `"in"`
    (`<-`) or `"either"` (no arrow). `types` holds the alternatives (empty for any type).

    `length` is `None` for a single relationship, and `(min, max)` for a variable-length one,
    with `max` `None` when there is no upper bound: `*` is `(1, None)`, `*2` is `(2, 2)` and
    `*0..3` is `(0, 3)`. The types and properties then hold for every relationship on the path.
    """

    variable: Optional[str] = None
    types: list[str] = field(default_factory=list)
    props: dict[str, Expr] = field(default_factory=dict)
    direction: str = "out"
    length: Optional[tuple[int, Optional[int]]] = None
    pos: int = field(default=0, compare=False)


@dataclass
class Pattern:
    """A chain of nodes joined by relationships: `rels[k]` joins `nodes[k]` and `nodes[k + 1]`.

    `variable` names the whole path, as in `p = (a)-->(b)`. `shortest` is `"one"` for
    `shortestPath(...)` and `"all"` for `allShortestPaths(...)`, whose pattern is one
    variable-length relationship between two nodes.
    """

    nodes: list[NodePattern]
    rels: list[RelPattern] = field(default_factory=list)
    variable: Optional[str] = None
    shortest: Optional[str] = None
    pos: int = field(default=0, compare=False)


@dataclass
class ReturnItem:
    """A `RETURN` column: a variable or one of its properties, named as `v` or `v.prop`."""

    expr: Expr
    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Return:
    items: list[ReturnItem]
    distinct: bool = False
    pos: int = field(default=0, compare=False)


@dataclass
class Clause:
    """`[OPTIONAL] MATCH patterns [WHERE where]`."""

    patterns: list[Pattern]
    where: Optional[Expr] = None
    optional: bool = False
    pos: int = field(default=0, compare=False)


@dataclass
class Query:
    """One or more `MATCH` clauses, then `[RETURN returns]`.

    `variables` maps each named variable to its kind, in the order they first appear: `"node"`,
    `"relationship"`, `"relationships"` for a variable-length relationship (a list of them), or
    `"path"` for a path variable.
    """

    clauses: list[Clause]
    returns: Optional[Return] = None
    variables: dict[str, str] = field(default_factory=dict)

    @property
    def patterns(self) -> list[Pattern]:
        """The patterns of every clause, in order."""
        return [pattern for clause in self.clauses for pattern in clause.patterns]


def walk(expr: Expr):
    """Yield `expr` and every expression inside it, parents first."""
    yield expr
    if isinstance(expr, (Prop, HasLabels)):
        yield from walk(expr.subject)
    elif isinstance(expr, (Not, Neg, IsNull)):
        yield from walk(expr.operand)
    elif isinstance(expr, Compare):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, (Call, ListExpr, BoolOp)):
        for sub in expr.args if isinstance(expr, Call) else (
            expr.items if isinstance(expr, ListExpr) else expr.operands
        ):
            yield from walk(sub)


# =============================================================================
# Parsing
# =============================================================================


class _TokenParser:
    """The token helpers shared by the text-format parsers (Cypher, DOT)."""

    FORMAT = ""

    def __init__(self, text: str):
        self.text = text
        # (kind, text, position) for every token
        self.tokens: list[tuple[str, Any, int]] = []
        self.i = 0

    def _pos(self) -> int:
        """The position of the next token (the end of the text if there's none)."""
        return self.tokens[self.i][2] if self.i < len(self.tokens) else len(self.text)

    def _fail(self, message: str, pos: Optional[int] = None):
        raise position_error(self.FORMAT, self.text, self._pos() if pos is None else pos, message)

    def _peek(self) -> Optional[str]:
        """The next token's text (`None` at the end)."""
        return self.tokens[self.i][1] if self.i < len(self.tokens) else None

    def _next(self, what: str) -> tuple[str, Any, int]:
        if self.i >= len(self.tokens):
            self._fail(f"expected {what}, but the text ended")
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, punct: str, what: Optional[str] = None) -> None:
        tok = self._next(what or repr(punct))
        if tok[1] != punct:
            self.i -= 1
            self._fail(f"expected {what or repr(punct)}, found {tok[1]!r}")


def _unsupported(word: str) -> Optional[str]:
    """Why a clause can't follow the pattern of a query, if `word` starts one."""
    if word in ("CREATE", "MERGE", "SET", "DELETE", "DETACH", "REMOVE"):
        return f"{word} isn't supported: a query only reads the graph"
    if word in ("ORDER", "SKIP", "LIMIT"):
        return f"{word} isn't supported; sort or slice the table that match() returns instead"
    if word in ("WITH", "UNWIND", "UNION", "CALL"):
        return f"{word} isn't supported"
    return None


class _CypherParser(_TokenParser):
    """A parser for Cypher `CREATE` scripts and `MATCH` queries."""

    FORMAT = "Cypher"

    def __init__(self, text: str):
        super().__init__(text)
        pos = 0
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if m is None:
                if text.startswith("/*", pos):
                    self._fail("a comment that never ends", pos)
                if text[pos] in "'\"`":
                    self._fail(f"the quote {text[pos]} is never closed", pos)
                self._fail(f"unexpected character {text[pos]!r}", pos)
            kind = m.lastgroup
            if kind != "space":
                self.tokens.append((kind, m.group(), pos))
            pos = m.end()

    # --- token helpers --------------------------------------------------------

    def _accept(self, punct: str) -> bool:
        if self._peek() == punct and self.tokens[self.i][0] == "punct":
            self.i += 1
            return True
        return False

    def _at_keyword(self, word: str) -> bool:
        return (self.i < len(self.tokens) and self.tokens[self.i][0] == "name"
                and self.tokens[self.i][1].upper() == word)

    def _accept_keyword(self, word: str) -> bool:
        if self._at_keyword(word):
            self.i += 1
            return True
        return False

    def _expect_keyword(self, word: str) -> None:
        if not self._accept_keyword(word):
            if self.i >= len(self.tokens):
                self._fail(f"expected {word}, but the text ended")
            self._fail(f"expected {word}, found {self._peek()!r}")

    def _at_name(self) -> bool:
        return self.i < len(self.tokens) and self.tokens[self.i][0] in ("name", "quoted")

    def _name(self, what: str) -> str:
        kind, value, _ = self._next(what)
        if kind == "name":
            return value
        if kind == "quoted":
            return value[1:-1].replace("``", "`")
        self.i -= 1
        self._fail(f"expected {what}, found {value!r}")

    # --- clauses --------------------------------------------------------------

    def parse_create(self) -> list[Pattern]:
        patterns = []
        while self.i < len(self.tokens):
            if self._accept(";"):
                continue
            kind, word, _ = self._next("CREATE")
            if kind != "name" or word.upper() != "CREATE":
                self.i -= 1
                hint = ""
                if kind == "name" and word.upper() == "MATCH":
                    hint = "; to query a graph, use Graph.match(), select_match() or count_matches()"
                self._fail(f"only CREATE clauses can build a graph, found {word!r}{hint}")
            patterns.append(self._pattern(create=True))
            while self._accept(","):
                patterns.append(self._pattern(create=True))
        return patterns

    def parse_query(self) -> Query:
        if not (self._at_keyword("MATCH") or self._at_keyword("OPTIONAL")):
            if self.i >= len(self.tokens):
                self._fail("a query starts with MATCH, but the text is empty")
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            self._fail(reason or f"a query starts with MATCH, found {word!r}")
        clauses: list[Clause] = []
        while self._at_keyword("MATCH") or self._at_keyword("OPTIONAL"):
            pos = self._pos()
            optional = self._accept_keyword("OPTIONAL")
            self._expect_keyword("MATCH")
            patterns = [self._pattern(create=False)]
            while self._accept(","):
                patterns.append(self._pattern(create=False))
            where = self._expr() if self._accept_keyword("WHERE") else None
            clauses.append(Clause(patterns, where, optional, pos))
        returns, pos = None, self._pos()
        if self._accept_keyword("RETURN"):
            returns = self._return(pos)
        self._accept(";")
        if self.i < len(self.tokens):
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            if reason is None:
                expected = ("the end of the query" if returns else
                            "MATCH, RETURN or the end of the query" if clauses[-1].where else
                            "WHERE, MATCH, RETURN or the end of the query")
                reason = f"expected {expected}, found {word!r}"
            self._fail(reason)
        query = Query(clauses, returns)
        self._check_variables(query)
        return query

    def _check_variables(self, query: Query) -> None:
        kinds = query.variables
        # every pattern variable, to tell a WHERE that looks ahead from one that has a typo
        anywhere = {part.variable for pattern in query.patterns
                    for part in [*pattern.nodes, *pattern.rels, pattern]}
        for clause in query.clauses:
            in_clause: set[str] = set()
            for pattern in clause.patterns:
                name = pattern.variable
                if name is not None:
                    known = kinds.get(name)
                    if known == "path":
                        self._fail(f"the path variable `{name}` is defined twice", pattern.pos)
                    if known is not None:
                        what = "node" if known == "node" else "relationship"
                        self._fail(f"`{name}` is used for both a path and a {what}", pattern.pos)
                    kinds[name] = "path"
                parts: list[Union[NodePattern, RelPattern]] = [pattern.nodes[0]]
                for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                    parts += [rel, node]
                for part in parts:
                    self._check_part(part, kinds, in_clause)
            if clause.where is not None:
                for sub in walk(clause.where):
                    if isinstance(sub, Var) and sub.name not in kinds:
                        if sub.name in anywhere:
                            self._fail(f"the variable `{sub.name}` is only defined in a later "
                                       "MATCH; a WHERE sees its own MATCH and the ones before it",
                                       sub.pos)
                        self._fail(f"the variable `{sub.name}` isn't defined in MATCH", sub.pos)
        if query.returns is not None:
            for item in query.returns.items:
                for sub in walk(item.expr):
                    if isinstance(sub, Var) and sub.name not in kinds:
                        self._fail(f"the variable `{sub.name}` isn't defined in MATCH", sub.pos)

    def _check_part(self, part: Union[NodePattern, RelPattern], kinds: dict[str, str],
                    in_clause: set[str]) -> None:
        """Record the kind of a pattern element's variable, and check how it's reused."""
        name = part.variable
        if name is None:
            return
        if isinstance(part, NodePattern):
            kind = "node"
        else:
            kind = "relationship" if part.length is None else "relationships"
        known = kinds.get(name)
        if known is None:
            kinds[name] = kind
        elif known == "path":
            what = "node" if kind == "node" else "relationship"
            self._fail(f"`{name}` is a path variable, so it can't also be a {what}", part.pos)
        elif (known == "node") != (kind == "node"):
            self._fail(f"`{name}` is used for both a node and a relationship", part.pos)
        elif kind != "node" and name in in_clause:
            # a row matches each relationship once, so its variable can't repeat in a clause
            self._fail(f"the relationship variable `{name}` is used more than once in one MATCH",
                       part.pos)
        elif "relationships" in (known, kind):
            self._fail(f"the variable-length relationship `{name}` can't be used again in a "
                       "later MATCH", part.pos)
        in_clause.add(name)

    def _return(self, start: int) -> Return:
        distinct = self._accept_keyword("DISTINCT")
        items: list[ReturnItem] = []
        while True:
            pos = self._pos()
            if self._accept("*"):
                self._fail("RETURN * isn't supported; name the variables to return", pos)
            word = self._peek()
            if (self._at_name() and word.lower() in _AGGREGATES and self.i + 1 < len(self.tokens)
                    and self.tokens[self.i + 1][1] == "("):
                self._fail(f"aggregation ({word}()) isn't supported; use count_matches(), or group "
                           "the match() table with Polars", pos)
            expr = self._expr()
            if isinstance(expr, Var):
                name = expr.name
            elif isinstance(expr, Prop) and isinstance(expr.subject, Var):
                name = f"{expr.subject.name}.{expr.key}"
            else:
                self._fail("RETURN can only list variables and their properties (v or v.prop)",
                           pos)
            if self._at_keyword("AS"):
                self._fail("aliases (AS) aren't supported; rename the columns of the match() "
                           "table instead")
            if any(item.name == name for item in items):
                self._fail(f"the column {name!r} is returned twice", pos)
            items.append(ReturnItem(expr, name, pos))
            if not self._accept(","):
                return Return(items, distinct, start)

    # --- patterns -------------------------------------------------------------

    def _pattern(self, create: bool) -> Pattern:
        """A chain of nodes and relationships, and in a query, `p = chain` or
        `[p =] shortestPath(chain)` (or `allShortestPaths`)."""
        start = self._pos()
        variable = None
        if (self._at_name() and self.i + 1 < len(self.tokens)
                and self.tokens[self.i + 1][:2] == ("punct", "=")):
            if create:
                self._fail("a path variable (p = ...) can only be used in a query")
            variable = self._name("a path variable")
            self.i += 1  # the `=`
        word = self._peek() if self._at_name() else None
        if (not create and word is not None and word.lower() in _SHORTEST
                and self.i + 1 < len(self.tokens) and self.tokens[self.i + 1][1] == "("):
            name, pos = _SHORTEST[word.lower()], self._pos()
            self.i += 2  # the name and `(`
            pattern = self._chain(create)
            self._expect(")", f"')' to close {name}()")
            if len(pattern.rels) != 1:
                self._fail(f"{name}() takes a pattern with one relationship, such as "
                           "(a)-[*]-(b)", pos)
            rel = pattern.rels[0]
            if rel.length is None:
                self._fail(f"{name}() needs a variable-length relationship, such as -[*]- or "
                           "-[:KNOWS*..5]-", rel.pos)
            if rel.length[0] > 1:
                self._fail(f"{name}() finds paths of at least 0 or 1 relationships, not "
                           f"{rel.length[0]}", rel.pos)
            pattern.shortest = "all" if name == "allShortestPaths" else "one"
        else:
            pattern = self._chain(create)
        pattern.variable, pattern.pos = variable, start
        return pattern

    def _chain(self, create: bool) -> Pattern:
        pattern = Pattern([self._node(create)])
        while self._peek() in ("-", "<"):
            pattern.rels.append(self._rel(create))
            pattern.nodes.append(self._node(create))
        return pattern

    def _node(self, create: bool) -> NodePattern:
        start = self._pos()
        self._expect("(", "'(' to start a node")
        variable = self._name("a variable") if self._at_name() else None
        labels = self._labels(create)
        props = self._map(create) if self._peek() == "{" else {}
        self._expect(")", "')' to end the node")
        return NodePattern(variable, labels, props, start)

    def _labels(self, create: bool) -> list[tuple[str, ...]]:
        labels = []
        while self._accept(":"):
            group = [self._name("a node label")]
            while self._peek() == "|":
                if create:
                    self._fail("a node in CREATE needs plain labels (:A:B), not alternatives "
                               "(:A|B)")
                self.i += 1
                group.append(self._name("a node label"))
            labels.append(tuple(group))
        return labels

    def _rel(self, create: bool) -> RelPattern:
        start = self._pos()
        back = self._accept("<")
        self._expect("-", "'-' in a relationship")
        rel = RelPattern(pos=start)
        if self._accept("["):
            if self._at_name():
                rel.variable = self._name("a relationship variable")
            if self._accept(":"):
                rel.types.append(self._name("a relationship type"))
                while self._peek() == "|":
                    if create:
                        self._fail("a relationship in CREATE needs exactly one type")
                    self.i += 1
                    self._accept(":")  # the older `:A|:B` spelling
                    rel.types.append(self._name("a relationship type"))
                if self._peek() == ":":
                    self._fail("a relationship has a single type; write :A|B to match either")
            if self._peek() == "*" and not create:
                rel.length = self._length()
            if self._peek() == "{":
                rel.props = self._map(create)
            self._expect("]")
        self._expect("-", "'-' in a relationship")
        forward = self._accept(">")
        if back and forward:
            self._fail("a relationship can't point both ways", start)
        rel.direction = "in" if back else "out" if forward else "either"
        return rel

    def _length(self) -> tuple[int, Optional[int]]:
        """`*`, `*n`, `*min..max`, `*min..` or `*..max`, as `(min, max)`."""
        pos = self._pos()
        self.i += 1  # the `*`
        low = self._hops() if self._at_number() else None
        if self._accept(".."):
            high = self._hops() if self._at_number() else None
            low = 1 if low is None else low
        elif low is None:
            high, low = None, 1
        else:
            high = low
        if high is not None and low > high:
            self._fail(f"a variable-length relationship can't have more than {high} hops and at "
                       f"least {low}; write *{high}..{low}", pos)
        return low, high

    def _at_number(self) -> bool:
        return self.i < len(self.tokens) and self.tokens[self.i][0] == "number"

    def _hops(self) -> int:
        _, value, pos = self._next("a number of hops")
        if not value.isdigit():
            self._fail(f"a number of hops is a whole number, not {value!r}", pos)
        return int(value)

    def _map(self, create: bool) -> dict[str, Expr]:
        self._expect("{")
        out: dict[str, Expr] = {}
        if self._accept("}"):
            return out
        while True:
            pos = self._pos()
            key = self._name("a property name")
            self._expect(":", "':' after the property name")
            if key in out:
                self._fail(f"the property {key!r} is given twice", pos)
            pos = self._pos()
            if self.i < len(self.tokens) and self.tokens[self.i][0] == "param":
                if create:
                    self._fail("parameters ($name) can only be used in queries")
                out[key] = Param(self._param_name(self.tokens[self.i][1]), pos)
                self.i += 1
            else:
                out[key] = Literal(self._value(), pos)
            if self._accept(","):
                continue
            self._expect("}", "',' or '}' in the properties")
            return out

    def _value(self) -> Any:
        kind, value, pos = self._next("a value")
        if kind == "string":
            return self._unquote(value, pos)
        if kind == "number":
            return _number(value)
        if value == "-" and self._peek() is not None and self.tokens[self.i][0] == "number":
            return -self._value()
        if kind == "name" and value.lower() in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[value.lower()]
        if value == "[":
            items = []
            if self._accept("]"):
                return items
            while True:
                items.append(self._value())
                if self._accept(","):
                    continue
                self._expect("]", "',' or ']' in the list")
                return items
        if value == "{":
            self._fail("a property value can't be a map", pos)
        self._fail(f"expected a value, found {value!r}", pos)

    def _unquote(self, token: str, pos: int) -> str:
        body, out, k = token[1:-1], [], 0
        while k < len(body):
            c = body[k]
            if c != "\\":
                out.append(c)
                k += 1
                continue
            e = body[k + 1]
            if e in _ESCAPES:
                out.append(_ESCAPES[e])
                k += 2
            elif e in "uU":
                width = 4 if e == "u" else 8
                digits = body[k + 2:k + 2 + width]
                if not re.fullmatch(r"[0-9a-fA-F]{%d}" % width, digits):
                    self._fail(f"a bad \\{e} escape in a string", pos + 1 + k)
                out.append(chr(int(digits, 16)))
                k += 2 + width
            else:
                self._fail(f"an unknown escape \\{e} in a string", pos + 1 + k)
        return "".join(out)

    @staticmethod
    def _param_name(token: str) -> str:
        name = token[1:]
        return name[1:-1].replace("``", "`") if name.startswith("`") else name

    # --- expressions ----------------------------------------------------------
    #
    # From the loosest binding to the tightest: OR, XOR, AND, NOT, comparisons, the string, list
    # and null predicates, unary minus, then property lookups and label tests.

    def _expr(self) -> Expr:
        return self._bool_op("OR", self._xor)

    def _xor(self) -> Expr:
        return self._bool_op("XOR", self._and)

    def _and(self) -> Expr:
        return self._bool_op("AND", self._not)

    def _bool_op(self, op: str, operand) -> Expr:
        pos = self._pos()
        operands = [operand()]
        while self._accept_keyword(op):
            operands.append(operand())
        return operands[0] if len(operands) == 1 else BoolOp(op, operands, pos)

    def _not(self) -> Expr:
        pos = self._pos()
        if self._accept_keyword("NOT"):
            return Not(self._not(), pos)
        return self._comparison()

    def _comparison(self) -> Expr:
        first = left = self._predicate()
        parts: list[Expr] = []
        while self._peek() in _COMPARISONS and self.tokens[self.i][0] == "punct":
            op, pos = self.tokens[self.i][1], self.tokens[self.i][2]
            self.i += 1
            right = self._predicate()
            parts.append(Compare(op, left, right, pos))
            left = right
        if not parts:
            return first
        return parts[0] if len(parts) == 1 else BoolOp("AND", parts, first.pos)

    def _predicate(self) -> Expr:
        operand = self._unary()
        while True:
            pos = self._pos()
            if self._accept_keyword("IS"):
                negated = self._accept_keyword("NOT")
                self._expect_keyword("NULL")
                operand = IsNull(operand, negated, pos)
            elif self._accept_keyword("IN"):
                operand = Compare("IN", operand, self._unary(), pos)
            elif self._at_keyword("STARTS") or self._at_keyword("ENDS"):
                word = self.tokens[self.i][1].upper()
                self.i += 1
                self._expect_keyword("WITH")
                operand = Compare(f"{word} WITH", operand, self._unary(), pos)
            elif self._accept_keyword("CONTAINS"):
                operand = Compare("CONTAINS", operand, self._unary(), pos)
            else:
                return operand

    def _unary(self) -> Expr:
        pos = self._pos()
        if self._accept("-"):
            operand = self._unary()
            if (isinstance(operand, Literal) and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)):
                return Literal(-operand.value, pos)
            return Neg(operand, pos)
        return self._postfix()

    def _postfix(self) -> Expr:
        expr = self._atom()
        while True:
            pos = self._pos()
            if self._accept("."):
                expr = Prop(expr, self._name("a property name"), expr.pos)
            elif self._peek() == ":":
                expr = HasLabels(expr, self._labels(create=False), pos)
            else:
                return expr

    def _atom(self) -> Expr:
        kind, value, pos = self._next("an expression")
        if kind == "string":
            return Literal(self._unquote(value, pos), pos)
        if kind == "number":
            return Literal(_number(value), pos)
        if kind == "param":
            return Param(self._param_name(value), pos)
        if kind == "quoted":
            return Var(value[1:-1].replace("``", "`"), pos)
        if kind == "name":
            upper = value.upper()
            if upper in ("TRUE", "FALSE", "NULL"):
                return Literal({"TRUE": True, "FALSE": False, "NULL": None}[upper], pos)
            if self._accept("("):
                args: list[Expr] = []
                if not self._accept(")"):
                    args.append(self._expr())
                    while self._accept(","):
                        args.append(self._expr())
                    self._expect(")", "',' or ')' in the arguments")
                return Call(value, args, pos)
            if upper in _RESERVED:
                self._fail(f"expected an expression, found {value!r}", pos)
            return Var(value, pos)
        if value == "(":
            expr = self._expr()
            self._expect(")", "')' to close the parenthesis")
            return expr
        if value == "[":
            items: list[Expr] = []
            if not self._accept("]"):
                items.append(self._expr())
                while self._accept(","):
                    items.append(self._expr())
                self._expect("]", "',' or ']' in the list")
            return ListExpr(items, pos)
        if value == "{":
            self._fail("map values aren't supported in expressions", pos)
        self._fail(f"expected an expression, found {value!r}", pos)


def _number(text: str) -> Union[int, float]:
    return float(text) if any(c in text for c in ".eE") else int(text)


def parse_create(text: str) -> list[Pattern]:
    """Parse a script of `CREATE` clauses (and `;`s) into its patterns, in order."""
    return _CypherParser(text).parse_create()


def parse_query(text: str) -> Query:
    """Parse a query: `[OPTIONAL] MATCH ... [WHERE ...]` clauses, then `[RETURN ...]`."""
    return _CypherParser(text).parse_query()


# =============================================================================
# The data model
# =============================================================================


def labels_of(type_value: Any) -> Optional[list[str]]:
    """The Cypher labels a node's `type` attribute holds: a string is one label and a non-empty
    list of strings is several. Any other value (a number, an empty list) isn't labels, and
    gives `None`: the node has no labels, and `type` is an ordinary property."""
    if isinstance(type_value, str):
        return [type_value]
    if (isinstance(type_value, (list, tuple)) and type_value
            and all(isinstance(t, str) for t in type_value)):
        return list(type_value)
    return None


@dataclass(frozen=True)
class NodeRef:
    """A node as a value in an expression (what a node variable evaluates to)."""

    id: int


@dataclass(frozen=True)
class RelRef:
    """A relationship as a value in an expression."""

    id: int


@dataclass(frozen=True)
class PathRef:
    """A path as a value in an expression: its node IDs, and the IDs of the relationships
    between them (one fewer)."""

    nodes: tuple[int, ...]
    rels: tuple[int, ...]


class GraphView:
    """The node and edge data of a Rust graph as the query evaluator sees it.

    - `n.key` is the attribute, or the node's label when `key` is `label_key` and there is no
      such attribute (`from_cypher()` moves `name` into the label).
    - `r.weight` is the edge weight, or a `weight` attribute when the edge has no weight.
    - Missing attributes are `None` (Cypher's `null`).
    - The labels of a node come from its `type` attribute (see `labels_of()`), and the type of
      a relationship is its `rel` attribute when that's a string.

    Nodes and edges are read once and cached, so a view belongs to one query run.
    """

    def __init__(self, graph: Any, label_key: Optional[str] = "name"):
        self._graph = graph
        self.label_key = label_key
        self._nodes: dict[int, tuple[Optional[str], dict]] = {}
        self._edges: dict[int, tuple[int, int, Optional[float], dict]] = {}

    def node(self, node_id: int) -> tuple[Optional[str], dict]:
        """`(label, attributes)`."""
        found = self._nodes.get(node_id)
        if found is None:
            node = self._graph.get_node(node_id)
            found = self._nodes[node_id] = (node.label, dict(node.attributes or {}))
        return found

    def edge(self, edge_id: int) -> tuple[int, int, Optional[float], dict]:
        """`(source, target, weight, attributes)`."""
        found = self._edges.get(edge_id)
        if found is None:
            edge = self._graph.get_edge(edge_id)
            found = self._edges[edge_id] = (edge.source, edge.target, edge.weight,
                                            dict(edge.attributes or {}))
        return found

    def node_prop(self, node_id: int, key: str) -> Any:
        label, attrs = self.node(node_id)
        if key in attrs:
            return attrs[key]
        return label if key == self.label_key else None

    def edge_prop(self, edge_id: int, key: str) -> Any:
        _, _, weight, attrs = self.edge(edge_id)
        if key == "weight" and weight is not None:
            return weight
        return attrs.get(key)

    def node_labels(self, node_id: int) -> list[str]:
        return labels_of(self.node(node_id)[1].get("type")) or []

    def edge_type(self, edge_id: int) -> Optional[str]:
        rel = self.edge(edge_id)[3].get("rel")
        return rel if isinstance(rel, str) else None


# =============================================================================
# Evaluation
# =============================================================================

# Values follow Cypher's rules: `None` is `null`, and anything that meets a `null` (a missing
# property, say) is `null` too, apart from `IS NULL`, `OR` with a true side and `AND` with a
# false side. `WHERE` keeps a row only when its condition is `true`.


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "map"
    if isinstance(value, NodeRef):
        return "node"
    if isinstance(value, RelRef):
        return "relationship"
    if isinstance(value, PathRef):
        return "path"
    return "other"


def _describe(value: Any) -> str:
    kind = _kind(value)
    if kind in ("node", "relationship", "path"):
        return f"a {kind}"
    if kind == "list" and value and all(isinstance(v, RelRef) for v in value):
        return "a list of relationships"
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _all_equal(pairs) -> Optional[bool]:
    """`true` if every pair is equal, `false` if any isn't, else `null`."""
    result: Optional[bool] = True
    for a, b in pairs:
        same = equals(a, b)
        if same is False:
            return False
        if same is None:
            result = None
    return result


def equals(a: Any, b: Any) -> Optional[bool]:
    """Cypher's `a = b`: `null` if either side is `null`, `false` for different types, and
    numbers compare by value (so `1 = 1.0`). Lists and maps compare item by item."""
    ka, kb = _kind(a), _kind(b)
    if ka == "null" or kb == "null":
        return None
    if ka != kb:
        return False
    if ka == "list":
        return False if len(a) != len(b) else _all_equal(zip(a, b))
    if ka == "map":
        return False if a.keys() != b.keys() else _all_equal((a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:  # an odd attribute value that can't be compared
        return False


def order(a: Any, b: Any) -> Optional[int]:
    """Negative, zero or positive as `a` sorts before, with or after `b`, or `None` when they
    can't be ordered (a `null`, different types, NaN, or types without an order)."""
    ka, kb = _kind(a), _kind(b)
    if ka != kb or ka in ("null", "map", "node", "relationship", "path", "other"):
        return None
    if ka == "list":
        for x, y in zip(a, b):
            c = order(x, y)
            if c is None or c != 0:
                return c
        return len(a) - len(b)
    if ka == "number" and (a != a or b != b):
        return None
    try:
        return bool(a > b) - bool(a < b)
    except Exception:  # an odd attribute value that can't be ordered
        return None


_ORDERINGS = {
    "<": lambda c: c < 0, "<=": lambda c: c <= 0, ">": lambda c: c > 0, ">=": lambda c: c >= 0,
}


def _is_nan(value: Any) -> bool:
    return _kind(value) == "number" and value != value


class Evaluator:
    """Compiles the expressions of one query into Python functions of a binding.

    A binding maps variable names to node or edge IDs, a tuple of edge IDs for a variable-length
    relationship, a pair of tuples `(node IDs, edge IDs)` for a path, or `None` for an
    `OPTIONAL MATCH` that found nothing. `variables` gives the
    kind of each variable (as in `Query.variables`), `params` the values of the
    `$name` parameters, and `view` reads the graph. Errors point at the query `text`.
    """

    def __init__(self, text: str, variables: dict[str, str], params: Optional[dict],
                 view: GraphView):
        self.text = text
        self.variables = variables
        self.params = {} if params is None else params
        self.view = view

    def _fail(self, pos: int, message: str):
        raise cypher_error(self.text, pos, message)

    def condition(self, expr: Expr):
        """A function of a binding that is `True` when `expr` is `true` (`false` and `null` both
        reject the row). A value that isn't a boolean or `null` is an error."""
        run = self.compile(expr)

        def test(binding):
            value = run(binding)
            if value is True:
                return True
            if value is None or value is False:
                return False
            self._fail(expr.pos, f"WHERE needs true, false or null, not {_describe(value)}")
        return test

    def compile(self, expr: Expr):
        """A function of a binding that returns the value of `expr`."""
        method = getattr(self, "_" + type(expr).__name__.lower())
        return method(expr)

    # --- operands -------------------------------------------------------------

    def _literal(self, expr: Literal):
        value = expr.value
        return lambda binding: value

    def _param(self, expr: Param):
        params = self.params
        if expr.name in params:
            value = params[expr.name]
        elif expr.name.isdigit() and int(expr.name) in params:
            value = params[int(expr.name)]
        else:
            self._fail(expr.pos, f"the parameter ${expr.name} isn't given in params=")
        return lambda binding: value

    def _var(self, expr: Var):
        name, kind = expr.name, self.variables[expr.name]
        if kind == "node":
            return lambda binding: None if binding[name] is None else NodeRef(binding[name])
        if kind == "relationships":
            return lambda binding: (None if binding[name] is None
                                    else [RelRef(edge) for edge in binding[name]])
        if kind == "path":
            return lambda binding: None if binding[name] is None else PathRef(*binding[name])
        return lambda binding: None if binding[name] is None else RelRef(binding[name])

    def _listexpr(self, expr: ListExpr):
        items = [self.compile(item) for item in expr.items]
        return lambda binding: [item(binding) for item in items]

    def _prop(self, expr: Prop):
        key, view = expr.key, self.view
        if isinstance(expr.subject, Var):
            # the usual case, without making a NodeRef or RelRef first
            name, kind = expr.subject.name, self.variables[expr.subject.name]
            if kind == "relationships":
                self._fail(expr.pos, f"`{name}` is a variable-length relationship, a list of "
                                     "relationships, so it has no properties; to require a "
                                     "property of every relationship on the path, put it in "
                                     "the pattern, as in -[r*1..3 {key: value}]->")
            if kind == "path":
                self._fail(expr.pos, f"`{name}` is a path, so it has no properties; read them "
                                     "from the nodes and relationships of its pattern")
            if kind == "node":
                return lambda binding: (None if binding[name] is None
                                        else view.node_prop(binding[name], key))
            return lambda binding: (None if binding[name] is None
                                    else view.edge_prop(binding[name], key))
        subject = self.compile(expr.subject)

        def run(binding):
            value = subject(binding)
            if isinstance(value, NodeRef):
                return view.node_prop(value.id, key)
            if isinstance(value, RelRef):
                return view.edge_prop(value.id, key)
            if isinstance(value, dict):
                return value.get(key)
            if value is None:
                return None
            self._fail(expr.pos, f"can't read the property {key!r} of {_describe(value)}")
        return run

    def _haslabels(self, expr: HasLabels):
        subject, groups, view = self.compile(expr.subject), expr.labels, self.view

        def run(binding):
            value = subject(binding)
            if isinstance(value, NodeRef):
                have = view.node_labels(value.id)
            elif isinstance(value, RelRef):
                # a relationship has one type, so `r:A|B` asks whether it is either
                rel_type = view.edge_type(value.id)
                have = [] if rel_type is None else [rel_type]
            elif value is None:
                return None
            else:
                self._fail(expr.pos, f"only nodes and relationships have labels, not "
                                     f"{_describe(value)}")
            return all(any(label in have for label in group) for group in groups)
        return run

    # --- functions ------------------------------------------------------------

    def _call(self, expr: Call):
        name = expr.name.lower()
        if name not in _FUNCTIONS:
            known = ", ".join(f"{f}()" for f in _FUNCTION_NAMES.values())
            self._fail(expr.pos, f"the function {expr.name}() isn't supported (the supported "
                                 f"ones are {known})")
        if len(expr.args) != 1:
            self._fail(expr.pos, f"{_FUNCTION_NAMES[name]}() takes one argument, not "
                                 f"{len(expr.args)}")
        if name == "exists":
            if not isinstance(expr.args[0], Prop):
                self._fail(expr.args[0].pos, "exists() takes a property, such as "
                                             "exists(n.age)")
            prop = self.compile(expr.args[0])
            return lambda binding: prop(binding) is not None
        arg, apply, view = self.compile(expr.args[0]), _FUNCTIONS[name], self.view

        def run(binding):
            value = arg(binding)
            if value is None:
                return None
            result = apply(value, view)
            if result is _WRONG_TYPE:
                self._fail(expr.pos, f"{expr.name}() can't take {_describe(value)}")
            return result
        return run

    # --- operators ------------------------------------------------------------

    def _boolean(self, value: Any, pos: int) -> Optional[bool]:
        if value is None or isinstance(value, bool):
            return value
        self._fail(pos, f"expected true, false or null, not {_describe(value)}")

    def _not(self, expr: Not):
        operand = self.compile(expr.operand)

        def run(binding):
            value = self._boolean(operand(binding), expr.operand.pos)
            return None if value is None else not value
        return run

    def _neg(self, expr: Neg):
        operand = self.compile(expr.operand)

        def run(binding):
            value = operand(binding)
            if value is None:
                return None
            if _kind(value) != "number":
                self._fail(expr.pos, f"can't negate {_describe(value)}")
            return -value
        return run

    def _boolop(self, expr: BoolOp):
        parts = [(self.compile(e), e.pos) for e in expr.operands]
        boolean = self._boolean

        if expr.op == "AND":
            def run(binding):
                result: Optional[bool] = True
                for part, pos in parts:
                    value = boolean(part(binding), pos)
                    if value is False:
                        return False
                    if value is None:
                        result = None
                return result
        elif expr.op == "OR":
            def run(binding):
                result: Optional[bool] = False
                for part, pos in parts:
                    value = boolean(part(binding), pos)
                    if value is True:
                        return True
                    if value is None:
                        result = None
                return result
        else:
            def run(binding):
                result: Optional[bool] = False
                for part, pos in parts:
                    value = boolean(part(binding), pos)
                    if value is None:
                        return None
                    result = result != value
                return result
        return run

    def _isnull(self, expr: IsNull):
        operand, negated = self.compile(expr.operand), expr.negated
        return lambda binding: (operand(binding) is None) != negated

    def _compare(self, expr: Compare):
        left, right, op = self.compile(expr.left), self.compile(expr.right), expr.op
        if op in ("=", "<>"):
            negate = op == "<>"

            def run(binding):
                same = equals(left(binding), right(binding))
                return None if same is None else same != negate
        elif op in _ORDERINGS:
            test = _ORDERINGS[op]

            def run(binding):
                a, b = left(binding), right(binding)
                if _is_nan(a) and _kind(b) == "number" or _is_nan(b) and _kind(a) == "number":
                    return False
                c = order(a, b)
                return None if c is None else test(c)
        elif op == "IN":
            def run(binding):
                value, items = left(binding), right(binding)
                if items is None:
                    return None
                if _kind(items) != "list":
                    self._fail(expr.pos, f"IN needs a list, not {_describe(items)}")
                result: Optional[bool] = False
                for item in items:
                    same = equals(value, item)
                    if same is True:
                        return True
                    if same is None:
                        result = None
                return result
        elif op == "=~":
            if isinstance(expr.right, Literal) and isinstance(expr.right.value, str):
                pattern = self._regex(expr.right.value, expr.right.pos)
                patterns = None
            else:
                pattern, patterns = None, {}

            def run(binding):
                value = left(binding)
                regex = pattern
                if regex is None:
                    source = right(binding)
                    if not isinstance(source, str):
                        return None
                    regex = patterns.get(source)
                    if regex is None:
                        regex = patterns[source] = self._regex(source, expr.right.pos)
                if not isinstance(value, str):
                    return None
                return regex.fullmatch(value) is not None
        else:
            test = {"STARTS WITH": str.startswith, "ENDS WITH": str.endswith,
                    "CONTAINS": str.__contains__}[op]

            def run(binding):
                a, b = left(binding), right(binding)
                if not isinstance(a, str) or not isinstance(b, str):
                    return None
                return test(a, b)
        return run

    def _regex(self, source: str, pos: int):
        try:
            return re.compile(source)
        except re.error as e:
            self._fail(pos, f"the regular expression {source!r} is invalid: {e}")


_WRONG_TYPE = object()


def _fn_id(value, view):
    return value.id if isinstance(value, (NodeRef, RelRef)) else _WRONG_TYPE


def _fn_type(value, view):
    return view.edge_type(value.id) if isinstance(value, RelRef) else _WRONG_TYPE


def _fn_labels(value, view):
    return view.node_labels(value.id) if isinstance(value, NodeRef) else _WRONG_TYPE


def _fn_size(value, view):
    return len(value) if isinstance(value, (str, list, tuple)) else _WRONG_TYPE


def _fn_lower(value, view):
    return value.lower() if isinstance(value, str) else _WRONG_TYPE


def _fn_upper(value, view):
    return value.upper() if isinstance(value, str) else _WRONG_TYPE


def _fn_length(value, view):
    return len(value.rels) if isinstance(value, PathRef) else _WRONG_TYPE


def _fn_nodes(value, view):
    return [NodeRef(n) for n in value.nodes] if isinstance(value, PathRef) else _WRONG_TYPE


def _fn_relationships(value, view):
    return [RelRef(r) for r in value.rels] if isinstance(value, PathRef) else _WRONG_TYPE


# lower-cased name -> implementation; `exists()` is handled by the evaluator itself
_FUNCTIONS = {
    "id": _fn_id, "type": _fn_type, "labels": _fn_labels, "size": _fn_size,
    "tolower": _fn_lower, "toupper": _fn_upper, "length": _fn_length, "nodes": _fn_nodes,
    "relationships": _fn_relationships, "exists": None,
}
# lower-cased name -> the name as documented
_FUNCTION_NAMES = {f.lower(): f for f in ("id", "type", "labels", "size", "toLower", "toUpper",
                                          "length", "nodes", "relationships", "exists")}


def conjuncts(expr: Optional[Expr]) -> list[Expr]:
    """The parts of a top-level `AND` (the whole expression if it isn't one, none for `None`)."""
    if expr is None:
        return []
    if isinstance(expr, BoolOp) and expr.op == "AND":
        return [part for operand in expr.operands for part in conjuncts(operand)]
    return [expr]


def variables_in(expr: Expr) -> set[str]:
    """The variables an expression reads."""
    return {sub.name for sub in walk(expr) if isinstance(sub, Var)}


def element_conditions(variable: str, element: Union[NodePattern, RelPattern]) -> list[Expr]:
    """The conditions a pattern element puts on its variable, as `WHERE` expressions: `(n:A:B)`
    is `n:A AND n:B`, `-[r:A|B]-` is `r:A|B`, and `{k: v}` is `n.k = v`."""
    var = Var(variable, element.pos)
    conditions: list[Expr] = []
    if isinstance(element, NodePattern):
        conditions += [HasLabels(var, [group], element.pos) for group in element.labels]
    elif element.types:
        conditions.append(HasLabels(var, [tuple(element.types)], element.pos))
    for key, value in element.props.items():
        conditions.append(Compare("=", Prop(var, key, value.pos), value, value.pos))
    return conditions


# =============================================================================
# Planning
# =============================================================================


@dataclass
class Step:
    """One step of a `Stage`. It binds one node, or one relationship and the node at its far end.

    - `"scan"` binds `node` to each node of the graph. With a `seed` (the operator and value
      side of an `id(node) = value` or `id(node) IN value` condition), it binds only those IDs.
    - `"expand"` binds `rel` to each edge at the bound node `start`, as
      `_incident_edges(start, direction)` lists them, and `node` to the edge's other end.
    - `"connect"` is the same, for edges whose other end is the already bound `node`.

    A variable-length relationship has `hops` (its `(min, max)`), and binds `rel` to each path
    of edges from `start` instead, each edge passing `edge_conditions` (which read the plan's
    `edge_slot`). With `reverse`, the path runs from `node` to `start` in the pattern, so its
    edges are bound in the opposite order. A `fixed` relationship was bound by an earlier clause,
    so the step only checks that edge.
    - `"shortest"` binds the variable-length `rel` of `shortestPath()` (with `shortest` `"one"`)
      or `allShortestPaths()` (`"all"`) to the shortest paths from the bound node `start` to
      the bound `node` (see `Matcher._extend()`).

    `paths` lists the path variables the step completes, which are bound with its slots.
    `conditions` are checked as soon as the step has bound its slots.
    """

    kind: str
    node: str
    rel: Optional[str] = None
    start: Optional[str] = None
    direction: Optional[str] = None
    seed: Optional[tuple[str, Expr]] = None
    conditions: list[Expr] = field(default_factory=list)
    hops: Optional[tuple[int, Optional[int]]] = None
    edge_conditions: list[Expr] = field(default_factory=list)
    reverse: bool = False
    fixed: bool = False
    paths: list[str] = field(default_factory=list)
    shortest: Optional[str] = None

    def __str__(self) -> str:
        if self.kind == "scan":
            return f"scan {self.node}" + (" by id" if self.seed else "")
        rel = self.rel
        if self.hops is not None:
            low, high = self.hops
            rel += f"*{low}" if low == high else f"*{low}..{'' if high is None else high}"
        arrow = {"out": "-[{}]->", "in": "<-[{}]-", "both": "-[{}]-"}[self.direction]
        kind = "all shortest" if self.shortest == "all" else self.kind
        return f"{kind} ({self.start}){arrow.format(rel)}({self.node})"


@dataclass
class Stage:
    """The plan of one `MATCH` clause.

    `entry` holds the conditions that read only slots bound by earlier clauses (or none at all),
    checked before the clause's `steps`. `new` lists the slots the clause binds, and `joined`
    the slots of earlier clauses that its patterns use again. `paths` lists the path variables
    whose patterns earlier clauses bound entirely, bound before `entry` is checked.
    """

    optional: bool
    steps: list[Step]
    entry: list[Expr]
    new: list[str]
    joined: list[str]
    paths: list[str] = field(default_factory=list)


def _id_seed(condition: Expr, slot: str) -> Optional[tuple[str, Expr]]:
    """`("=", value)` or `("IN", value)` when `condition` is `id(slot) = value` (either way round)
    or `id(slot) IN value`, and `value` reads no variable."""
    if not isinstance(condition, Compare) or condition.op not in ("=", "IN"):
        return None
    sides = [(condition.left, condition.right)]
    if condition.op == "=":
        sides.append((condition.right, condition.left))
    for call, value in sides:
        if (isinstance(call, Call) and call.name.lower() == "id" and call.args == [Var(slot)]
                and not variables_in(value)):
            return condition.op, value
    return None


def _selectivity(conditions: list[Expr], slot: str) -> int:
    """A rough measure of how few nodes the conditions on `slot` alone let through: an ID
    beats a property value, which beats a label, which beats any other test."""
    score = 0
    for condition in conditions:
        if _id_seed(condition, slot):
            score += 100
        elif isinstance(condition, Compare) and condition.op == "=":
            score += 4
        elif isinstance(condition, HasLabels):
            score += 2
        else:
            score += 1
    return score


_FLIP = {"out": "in", "in": "out", "either": "both"}
# a step's direction, followed backwards
_BACK = {"out": "in", "in": "out", "both": "both"}


class Plan:
    """The order in which to bind the slots of a query, and where each condition is checked.

    - **Slots:** a slot is a node or relationship of the patterns. Each named variable is one
      slot, however often it appears. Each anonymous element gets a slot of its own, named so
      that no variable clashes with it (`" 0"`, `" 1"`, ...). `slots` maps every slot to its
      kind (as in `Query.variables`), in the order they first appear. `paths` maps each
      variable-length relationship to its left and right node slots.
    - **Path variables:** a path variable is a slot too, just before those of its pattern, and
      `walks` maps it to the slots it's made of: the first node, then each relationship. It's
      bound by the step that binds the last of those (or before the stage's entry conditions,
      when earlier clauses bound them all).
    - **Shortest paths:** the relationship of `shortestPath()` or `allShortestPaths()` is never
      followed to reach a node. Its two nodes are bound first, like any others, and a
      `"shortest"` step after the clause's other steps finds the paths between them.
    - **Stages:** each `MATCH` clause is a `Stage`, planned with the slots of the clauses before
      it already bound.
    - **Conditions:** the patterns' labels, types and properties (`element_conditions()`) and
      the parts of the clause's `WHERE` (`conjuncts()`). Each is checked at the first step that
      has bound every slot it reads. Those of a variable-length relationship are checked on
      each of its edges instead, through the slot `edge_slot`. Those that read a shortest
      path's relationship (or its path variable) go to its `"shortest"` step, so they decide
      which paths are the shortest.
    - **Order:** each connected group of patterns starts at its most constrained node (by
      `_selectivity()`, ties going to the first), or at a node bound by an earlier clause. It
      then grows along relationships, closing a cycle (`"connect"`) whenever it can, so that
      every later node is reached through an edge rather than by a scan. Groups that share no
      variable are scanned one after another, which gives their cross product.
    - **Directions:** on an undirected graph every relationship is followed with `"both"`, so
      `-->`, `<--` and `--` mean the same.
    """

    def __init__(self, query: Query, directed: bool = True):
        self.slots: dict[str, str] = {}
        self.paths: dict[str, tuple[str, str]] = {}
        self.walks: dict[str, list[str]] = {}
        self.edge_slot = " edge"
        while self.edge_slot in query.variables:
            self.edge_slot += "_"
        self._counter = 0
        self.stages = [self._stage(clause, query, directed) for clause in query.clauses]

    def _stage(self, clause: Clause, query: Query, directed: bool) -> Stage:
        before = set(self.slots)
        used: dict[str, str] = {}  # the slots of this clause, in order
        rels = []  # (slot, left node, right node, arrow, hops)
        conditions: list[Expr] = []
        per_edge: dict[str, list[Expr]] = {}

        def slot(element: Union[NodePattern, RelPattern], kind: str) -> str:
            name = element.variable
            while name is None or (element.variable is None and name in query.variables):
                name, self._counter = f" {self._counter}", self._counter + 1
            self.slots.setdefault(name, kind)
            used.setdefault(name, kind)
            if kind == "relationships":
                per_edge[name] = element_conditions(self.edge_slot, element)
            else:
                conditions.extend(element_conditions(name, element))
            return name

        shortest = []  # the rels of shortestPath() patterns, as in `rels`, then the mode
        for pattern in clause.patterns:
            if pattern.variable is not None:
                self.slots[pattern.variable] = used[pattern.variable] = "path"
            left = slot(pattern.nodes[0], "node")
            walk = [left]
            for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                rel_slot = slot(rel, "relationship" if rel.length is None else "relationships")
                right = slot(node, "node")
                found = (rel_slot, left, right, rel.direction, rel.length)
                if pattern.shortest is None:
                    rels.append(found)
                else:
                    shortest.append((*found, pattern.shortest))
                if rel.length is not None:
                    self.paths[rel_slot] = (left, right)
                walk.append(rel_slot)
                left = right
            if pattern.variable is not None:
                self.walks[pattern.variable] = walk
        conditions += conjuncts(clause.where)

        # the path variables whose patterns earlier clauses bound entirely
        pending = [p for p in used if used[p] == "path"]
        early = [p for p in pending if set(self.walks[p]) <= before]
        pending = [p for p in pending if p not in early]
        ready = before | set(early)

        needs = [variables_in(c) for c in conditions]
        new = [s for s in used if s not in before]
        entry = [c for c, n in zip(conditions, needs) if n <= ready]
        alone = {s: [c for c, n in zip(conditions, needs) if n == {s}] for s in new}
        score = {s: _selectivity(alone[s], s) for s in new}
        score.update({s: 10**9 for s in used if s in before})

        steps: list[Step] = []
        placed = [n <= ready for n in needs]
        bound = set(ready)

        def add(step: Step) -> None:
            bound.update(s for s in (step.node, step.rel) if s is not None)
            for p in list(pending):
                if set(self.walks[p]) <= bound:
                    pending.remove(p)
                    step.paths.append(p)
                    bound.add(p)
            for i, n in enumerate(needs):
                if not placed[i] and n <= bound:
                    placed[i] = True
                    step.conditions.append(conditions[i])
            steps.append(step)

        nodes = [s for s, kind in used.items() if kind == "node"]
        todo = list(rels)
        while todo or len(bound & set(nodes)) < len(nodes):
            ready = [r for r in todo if r[1] in bound or r[2] in bound]
            if not ready:
                start = max((s for s in nodes if s not in bound), key=score.__getitem__)
                seed = next(filter(None, (_id_seed(c, start) for c in alone[start])), None)
                add(Step("scan", start, seed=seed))
                continue
            closing = [r for r in ready if r[1] in bound and r[2] in bound]
            if closing:
                rel = closing[0]
            else:
                rel = max(ready, key=lambda r: score[r[2] if r[1] in bound else r[1]])
            todo.remove(rel)
            rel_slot, left, right, arrow, hops = rel
            if left in bound:
                start, node, direction = left, right, "both" if arrow == "either" else arrow
            else:
                start, node, direction = right, left, _FLIP[arrow]
            add(Step("connect" if closing else "expand", node, rel_slot, start,
                     direction if directed else "both", hops=hops,
                     edge_conditions=per_edge.get(rel_slot, []), reverse=start != left,
                     fixed=rel_slot in before))
        # a shortest path joins two nodes bound by now, and is never followed to reach one
        for rel_slot, left, right, arrow, hops, mode in shortest:
            direction = "both" if arrow == "either" or not directed else arrow
            add(Step("shortest", right, rel_slot, left, direction, hops=hops,
                     edge_conditions=per_edge[rel_slot], shortest=mode))
        return Stage(clause.optional, steps, entry, new, [s for s in used if s in before], early)

    @property
    def steps(self) -> list[Step]:
        """The steps of every stage, in order."""
        return [step for stage in self.stages for step in stage.steps]

    def __str__(self) -> str:
        lines = []
        for i, stage in enumerate(self.stages):
            if i or stage.optional:
                lines.append("optional match" if stage.optional else "match")
            lines += [str(step) for step in stage.steps]
        return "\n".join(lines)


# =============================================================================
# Matching
# =============================================================================

_SCALARS = (bool, int, float, str)
_MIRROR = {"=": "=", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _native(expr: Expr, kinds: dict[str, str], index: dict[str, int],
            evaluator: Evaluator) -> Optional[tuple]:
    """The native form of a condition, which the Rust search checks without calling Python (see
    `src/cypher.rs`), or `None` when it has none. Slots are given by position (`index`).

    The forms are label and type tests on a variable, comparisons of a property or an `id()`
    with a constant scalar or with another property or `id()`, `IN` a list of constant scalars,
    `IS NULL` and `exists()` on a property or variable, comparisons of two IDs (`id(a) <
    id(b)`, or `a = b` for two variables of the same kind), `STARTS WITH`, `ENDS WITH` and
    `CONTAINS` on properties, IDs and constants, and `AND` and `OR` of native forms. A constant
    is a literal or parameter whose value is exactly `None`, a `bool`, an `int`, a `float` or a
    `str`, so that Rust reads it as Python does.
    """

    def single(e: Expr) -> Optional[int]:
        # a node or relationship variable (not a variable-length one)
        if isinstance(e, Var) and kinds[e.name] in ("node", "relationship"):
            return index[e.name]
        return None

    def operand(e: Expr) -> Optional[tuple]:
        if isinstance(e, Prop) and single(e.subject) is not None:
            return ("prop", single(e.subject), e.key)
        if (isinstance(e, Call) and e.name.lower() == "id" and len(e.args) == 1
                and single(e.args[0]) is not None):
            return ("id", single(e.args[0]))
        return None

    def constant(e: Expr) -> tuple[bool, Any]:
        # `(True, value)` for a literal or parameter scalar
        if isinstance(e, (Literal, Param)):
            value = evaluator.compile(e)({})
            if value is None or type(value) in _SCALARS:
                return True, value
        return False, None

    def constants(e: Expr) -> Optional[list]:
        if isinstance(e, ListExpr):
            found = [constant(item) for item in e.items]
            return [v for _, v in found] if all(ok for ok, _ in found) else None
        if isinstance(e, (Literal, Param)):
            value = evaluator.compile(e)({})
            if (type(value) in (list, tuple)
                    and all(v is None or type(v) in _SCALARS for v in value)):
                return list(value)
        return None

    if isinstance(expr, BoolOp) and expr.op in ("AND", "OR"):
        parts = [_native(e, kinds, index, evaluator) for e in expr.operands]
        return None if any(p is None for p in parts) else (expr.op.lower(), parts)
    if isinstance(expr, HasLabels):
        slot = single(expr.subject)
        return None if slot is None else ("labels", slot, [list(g) for g in expr.labels])
    if isinstance(expr, IsNull):
        if isinstance(expr.operand, Var):
            return ("null", index[expr.operand.name], None, expr.negated)
        found = operand(expr.operand)
        if found is not None and found[0] == "prop":
            return ("null", found[1], found[2], expr.negated)
        return None
    if isinstance(expr, Call) and expr.name.lower() == "exists" and len(expr.args) == 1:
        found = operand(expr.args[0])
        if found is not None and found[0] == "prop":
            return ("null", found[1], found[2], True)
        return None
    if not isinstance(expr, Compare):
        return None
    op, left, right = expr.op, expr.left, expr.right
    if op == "IN":
        lhs, items = operand(left), constants(right)
        return None if lhs is None or items is None else ("in", lhs, items)
    if op in ("STARTS WITH", "ENDS WITH", "CONTAINS"):
        terms = []
        for e in (left, right):
            ok, value = constant(e)
            terms.append(("const", value) if ok else operand(e))
        return None if None in terms else ("text", op, *terms)
    if op not in _MIRROR:
        return None
    lhs, rhs = operand(left), operand(right)
    if lhs is not None and rhs is not None:
        if lhs[0] == rhs[0] == "id":
            return ("ids", lhs[1], rhs[1], op)
        return ("cmp2", lhs, op, rhs)
    if lhs is not None or rhs is not None:
        ok, value = constant(right if lhs is not None else left)
        if not ok:
            return None
        return ("cmp", lhs, op, value) if lhs is not None else ("cmp", rhs, _MIRROR[op], value)
    a, b = single(left), single(right)
    if (op in ("=", "<>") and a is not None and b is not None
            and kinds[left.name] == kinds[right.name]):
        return ("ids", a, b, op)
    return None


class Matcher:
    """Finds the matches of a parsed query in a Rust graph, by a backtracking search over its
    `Plan`.

    - **Matches:** `rows()` gives each match as a tuple with a value for every slot of the plan,
      anonymous ones included, in the order of `plan.slots`; `binding()` turns one into a dict
      of slot → value, and `matches()` yields those dicts. A node or relationship slot holds an
      ID, a variable-length relationship a tuple of edge IDs in pattern order, a path variable a
      tuple `(node IDs, edge IDs)` in pattern order, and a slot that an `OPTIONAL MATCH` found
      nothing for holds `None`. Matches come in search order, which is fixed for a given graph
      and query.
    - **Shortest paths:** a `"shortest"` step first measures the distance of each node to the
      far end by a breadth-first search (backwards from it, and only as far as needed), then
      searches the trails from the near end for each length in turn, from the lower bound up,
      following only edges that keep the far end within reach. The first length with a trail
      that passes the step's conditions is the shortest: `allShortestPaths()` gives every such
      trail, `shortestPath()` the first. The distances skip the edges (and, with
      `distinct_nodes=True`, the nodes) that the rest of the clause uses, but not those of the
      trail itself, so they're lower bounds that only prune the search; a longer length is tried
      only while some branch was cut off by its length. When the two ends are the same node,
      this finds the shortest cycle through it.
    - **Clauses:** each `MATCH` extends the matches of the clauses before it, and drops those it
      can't extend. An `OPTIONAL MATCH` keeps them instead, with its new slots `None`. A pattern
      that uses a `None` slot again matches nothing.
    - **Distinctness:** the relationships of one clause are distinct edges, which is Cypher's
      rule, while nodes may repeat. With `distinct_nodes=True`, the nodes of one clause must be
      distinct too (the inner nodes of variable-length paths included), as in a subgraph
      isomorphism. Different clauses may reuse edges and nodes.
    - **When errors are raised:** conditions are compiled when the matcher is made, so a missing
      parameter or an unknown function raises then, even on an empty graph. Errors in values
      (`WHERE n.name`, say) raise when a row meets them.
    - **Engines:** with `engine="rust"`, the search runs in Rust (`src/cypher.rs`), which checks
      the conditions that have a native form (`_native()`) itself and calls the compiled Python
      function for the rest. With `engine="python"`, it runs here, as the reference: both give
      the same matches in the same order, and raise the same errors. The Rust search finds all
      the matches (or the first `limit`) before giving any, while the Python one is lazy.
    """

    def __init__(self, query: Query, text: str, graph: Any, params: Optional[dict] = None,
                 label_key: Optional[str] = "name", distinct_nodes: bool = False,
                 engine: str = "rust"):
        if engine not in ("rust", "python"):
            raise ValueError(f"engine must be 'rust' or 'python', not {engine!r}")
        self.plan = plan = Plan(query, graph.is_directed)
        self.graph = graph
        self.distinct_nodes = distinct_nodes
        self.engine = engine
        self.view = GraphView(graph, label_key)
        kinds = {**plan.slots, plan.edge_slot: "relationship"}
        self.evaluator = evaluator = Evaluator(text, kinds, params, self.view)
        self._stages = [(
            stage,
            [evaluator.condition(c) for c in stage.entry],
            [[evaluator.condition(c) for c in step.conditions] for step in stage.steps],
            [[evaluator.condition(c) for c in step.edge_conditions] for step in stage.steps],
            [step.seed and (step.seed[0], evaluator.compile(step.seed[1]))
             for step in stage.steps],
        ) for stage in plan.stages]
        self._all_nodes: Optional[list[int]] = None
        self._names = list(plan.slots)
        index = {name: i for i, name in enumerate(kinds)}
        self._node_slots = [index[s] for s, kind in plan.slots.items() if kind == "node"]
        self._edge_slots = [(index[s], kind == "relationships")
                            for s, kind in plan.slots.items() if kind in ("relationship",
                                                                          "relationships")]
        self._paths = [(index[rel], index[left]) for rel, (left, _) in plan.paths.items()]
        # each path variable's first node, then its relationships (and whether each is a list)
        self._walks = {p: (walk[0], [(s, plan.slots[s] == "relationships") for s in walk[1:]])
                       for p, walk in plan.walks.items()}
        self._program = self._compile(kinds, index, label_key) if engine == "rust" else None

    def _compile(self, kinds: dict[str, str], index: dict[str, int],
                 label_key: Optional[str]) -> tuple:
        """The plan as a program for the Rust search (`src/cypher.rs`), with slots given by
        their position in `kinds` (the plan's slots, then its `edge_slot`):

        - program: `(names, kinds, label_key, distinct_nodes, stages)`;
        - stage: `(optional, entry, new, joined, steps, paths)`;
        - step: `(kind, node, rel, start, direction, seed, conditions, hops, edge_conditions,
          reverse, fixed, paths)`, where `kind` is `"scan"`, `"expand"`, `"connect"`,
          `"shortest"` or `"all_shortest"`, and `seed` is `None` or a function of no arguments
          that gives the sorted IDs a scan tries;
        - paths: a list of `(path, walk)`, a path variable's slot and those of `Plan.walks`;
        - condition: `(native, fallback)`, a form from `_native()` (or `None`) and the
          compiled function of a binding dict.
        """
        evaluator = self.evaluator

        def conds(exprs, fns):
            return [(_native(e, kinds, index, evaluator), fn) for e, fn in zip(exprs, fns)]

        def paths(names):
            return [(index[p], [index[s] for s in self.plan.walks[p]]) for p in names]

        stages = []
        for stage, entry, checks, edge_checks, seeds in self._stages:
            steps = []
            for step, tests, edge_tests, seed in zip(stage.steps, checks, edge_checks, seeds):
                kind = "all_shortest" if step.shortest == "all" else step.kind
                steps.append((
                    kind, index[step.node],
                    None if step.rel is None else index[step.rel],
                    None if step.start is None else index[step.start],
                    step.direction,
                    None if seed is None else (lambda seed=seed: self._candidates(seed)),
                    conds(step.conditions, tests), step.hops,
                    conds(step.edge_conditions, edge_tests), step.reverse, step.fixed,
                    paths(step.paths),
                ))
            stages.append((stage.optional, conds(stage.entry, entry),
                           [index[s] for s in stage.new], [index[s] for s in stage.joined],
                           steps, paths(stage.paths)))
        return (list(kinds), list(kinds.values()), label_key, self.distinct_nodes, stages)

    def _candidates(self, seed) -> list[int]:
        """The nodes a scan tries: all of them, or those an `id()` condition names."""
        if seed is not None:
            op, value = seed[0], seed[1]({})
            if op == "=" or _kind(value) == "list":
                # anything else is left to the condition itself, which knows how to fail
                ids = {int(v) for v in ([value] if op == "=" else value)
                       if _kind(v) == "number" and math.isfinite(v) and v == int(v)
                       and 0 <= v < 2**64}
                return sorted(i for i in ids if self.graph.has_node_id(i))
            if value is None:
                return []
        if self._all_nodes is None:
            self._all_nodes = sorted(self.graph.get_node_ids())
        return self._all_nodes

    def rows(self, limit: Optional[int] = None) -> list[tuple]:
        """The matches, or the first `limit` of them, as tuples."""
        if limit == 0:
            return []
        if self._program is not None:
            return self.graph._cypher_match(self._program, limit)
        return list(itertools.islice(self._search(), limit))

    def count(self) -> tuple[int, Optional[tuple]]:
        """The number of matches, and the first one (or `None`)."""
        if self._program is not None:
            return self.graph._cypher_match(self._program, None, True)
        n, first = 0, None
        for row in self._search():
            if first is None:
                first = row
            n += 1
        return n, first

    def table(self, columns: list[tuple], sort: Optional[list[str]] = None,
              limit: Optional[int] = None) -> tuple[int, list[list], Optional[tuple]]:
        """The matches, or the first `limit` of them, as columns: `(count, columns, first)`,
        where `first` is the first match in search order (as from `rows()`), or `None`.

        Each column is `("slot", name)`, the slot's values (a list of edge IDs for a
        variable-length relationship, and a dict `{"nodes": [...], "relationships": [...]}` of
        IDs for a path variable), `("label", name)`, the labels of a node slot (or a path's dict
        with the labels of its nodes), or `("prop", name, key)`, a property of a node or
        relationship slot as `GraphView` reads it. With `sort`, the rows are sorted by the IDs
        of those slots (a variable-length relationship compares as its list of edge IDs, a path
        as its node IDs and then its edge IDs, and `None` sorts last), keeping search order for
        ties; without it, they stay in search order. The Rust engine builds the columns without
        making a tuple per match.
        """
        index = {name: i for i, name in enumerate(self._names)}
        if self._program is not None:
            table = (None if sort is None else [index[name] for name in sort],
                     [(c[0], index[c[1]], *c[2:]) for c in columns])
            return self.graph._cypher_match(self._program, limit, False, table)
        rows = self.rows(limit)
        first = rows[0] if rows else None
        if sort is not None:
            keys = [index[name] for name in sort]
            rows.sort(key=lambda row: tuple((1, 0) if row[i] is None else (0, row[i])
                                            for i in keys))
        slots, view = self.plan.slots, self.view
        cells = []
        for column in columns:
            i, kind = index[column[1]], slots[column[1]]
            values = [row[i] for row in rows]
            if kind == "path":
                label = (lambda n: view.node(n)[0]) if column[0] == "label" else (lambda n: n)
                cells.append([None if v is None else
                              {"nodes": [label(n) for n in v[0]], "relationships": list(v[1])}
                              for v in values])
            elif column[0] == "slot":
                cells.append([v if v is None or kind != "relationships" else list(v)
                              for v in values])
            elif column[0] == "label":
                cells.append([None if v is None else view.node(v)[0] for v in values])
            else:
                read = view.node_prop if kind == "node" else view.edge_prop
                cells.append([None if v is None else read(v, column[2]) for v in values])
        return len(rows), cells, first

    def binding(self, row: tuple) -> dict[str, Any]:
        """A match as a dict of slot → value."""
        return dict(zip(self._names, row))

    def matches(self):
        """The matches as dicts (see `binding()`), lazily with the Python engine."""
        rows = self._search() if self._program is None else self.rows()
        for row in rows:
            yield self.binding(row)

    def _search(self):
        """The Python search, yielding each match as a tuple."""
        binding: dict[str, Any] = {}
        stages, names = self._stages, self._names
        last = len(stages)

        def run(s):
            if s == last:
                yield tuple([binding[name] for name in names])
                return
            stage = stages[s][0]
            found = False
            for _ in self._extend(stages[s], binding):
                found = True
                yield from run(s + 1)
            if stage.optional and not found:
                for slot in stage.new:
                    binding[slot] = None
                yield from run(s + 1)

        yield from run(0)

    def _extend(self, compiled, binding: dict[str, Any]):
        """Bind the slots of one stage in every way that matches, yielding after each."""
        stage, entry, checks, edge_checks, seeds = compiled
        self._bind_paths(stage.paths, binding)
        if not all(check(binding) for check in entry):
            return
        slots, distinct = self.plan.slots, self.distinct_nodes
        used_nodes: set[int] = set()
        used_edges: set[int] = set()
        for slot in stage.joined:
            value = binding[slot]
            if value is None:
                return
            if slots[slot] == "node":
                if distinct and value in used_nodes:
                    return
                used_nodes.add(value)
            elif value in used_edges:
                return
            else:
                used_edges.add(value)
        if not distinct:
            used_nodes.clear()
        steps, graph = stage.steps, self.graph
        candidates = [self._candidates(seed) if step.kind == "scan" else None
                      for step, seed in zip(steps, seeds)]
        last = len(steps)
        bind_paths = self._bind_paths

        def search(k):
            if k == last:
                yield
                return
            step, tests = steps[k], checks[k]
            if step.kind == "scan":
                for node in candidates[k]:
                    if distinct and node in used_nodes:
                        continue
                    binding[step.node] = node
                    bind_paths(step.paths, binding)
                    if all(test(binding) for test in tests):
                        used_nodes.add(node)
                        yield from search(k + 1)
                        used_nodes.discard(node)
                return
            if step.kind == "shortest":
                yield from shortest(k, step, tests)
                return
            if step.hops is not None:
                yield from paths(k, step, tests)
                return
            connect = step.kind == "connect"
            target = binding[step.node] if connect else None
            fixed = binding[step.rel] if step.fixed else None
            for edge, other in graph._incident_edges(binding[step.start], step.direction):
                if fixed is not None:
                    if edge != fixed:
                        continue
                elif edge in used_edges:
                    continue
                if connect:
                    if other != target:
                        continue
                elif distinct and other in used_nodes:
                    continue
                else:
                    binding[step.node] = other
                binding[step.rel] = edge
                bind_paths(step.paths, binding)
                if all(test(binding) for test in tests):
                    if fixed is None:
                        used_edges.add(edge)
                    fresh = distinct and not connect
                    if fresh:
                        used_nodes.add(other)
                    yield from search(k + 1)
                    if fixed is None:
                        used_edges.discard(edge)
                    if fresh:
                        used_nodes.discard(other)

        def paths(k, step, tests):
            # a depth-first walk over the paths from `start`, each edge used once per clause
            low, high = step.hops
            connect = step.kind == "connect"
            target = binding[step.node] if connect else None
            edge_tests, probe, edge_slot = edge_checks[k], {}, self.plan.edge_slot
            path: list[int] = []

            def found(node):
                if not connect:
                    binding[step.node] = node
                binding[step.rel] = tuple(reversed(path)) if step.reverse else tuple(path)
                bind_paths(step.paths, binding)
                if all(test(binding) for test in tests):
                    yield from search(k + 1)

            def walk(node, depth):
                if depth >= low:
                    if connect:
                        if node == target:
                            yield from found(node)
                    elif not (distinct and depth == 0):  # the end would be the start
                        yield from found(node)
                if depth == high or (distinct and connect and depth and node == target):
                    return  # (with distinct nodes, a path can end at `target`, not pass it)
                for edge, other in graph._incident_edges(node, step.direction):
                    if edge in used_edges:
                        continue
                    mark = distinct and other not in used_nodes
                    if distinct and not mark and not (connect and other == target):
                        continue
                    probe[edge_slot] = edge
                    if not all(test(probe) for test in edge_tests):
                        continue
                    used_edges.add(edge)
                    path.append(edge)
                    if mark:
                        used_nodes.add(other)
                    yield from walk(other, depth + 1)
                    used_edges.discard(edge)
                    path.pop()
                    if mark:
                        used_nodes.discard(other)

            yield from walk(binding[step.start], 0)

        def shortest(k, step, tests):
            # the paths from `a` to `b` with the fewest edges among those that pass the step's
            # conditions: for each length from the least on, a depth-first walk over the
            # paths of that length, cut short by each node's distance to `b`
            low, high = step.hops
            a, b = binding[step.start], binding[step.node]
            edge_tests, probe, edge_slot = edge_checks[k], {}, self.plan.edge_slot
            one = step.shortest == "one"

            def passes(edge):
                probe[edge_slot] = edge
                return all(test(probe) for test in edge_tests)

            # the distances to `b`, found by a breadth-first search backwards from it, one
            # layer at a time, as far as the walk needs them
            dist, frontier, depth = {b: 0}, [b], 0
            back = _BACK[step.direction]

            def grow():
                nonlocal frontier, depth
                layer = []
                for node in frontier:
                    for edge, other in graph._incident_edges(node, back):
                        if (other in dist or edge in used_edges
                                or (distinct and other in used_nodes and other != a)
                                or not passes(edge)):
                            continue
                        dist[other] = depth + 1
                        layer.append(other)
                frontier, depth = layer, depth + 1

            path: list[int] = []
            state = {"found": False, "cut": False}  # a path passed; a longer one might

            def walk(node, at, length):
                if at == length:
                    state["cut"] = True
                    if node == b:
                        binding[step.rel] = tuple(path)
                        bind_paths(step.paths, binding)
                        if all(test(binding) for test in tests):
                            state["found"] = True
                            yield from search(k + 1)
                    return
                left = length - at - 1  # the edges still to take after this one
                for edge, other in graph._incident_edges(node, step.direction):
                    if edge in used_edges:
                        continue
                    mark = distinct and other not in used_nodes
                    if distinct and not mark and not (left == 0 and other == b):
                        continue  # (`b` itself is used, so a path ends there)
                    d = dist.get(other)
                    if d is None or d > left:
                        if d is not None or frontier:
                            state["cut"] = True
                        continue
                    if not passes(edge):
                        continue
                    used_edges.add(edge)
                    path.append(edge)
                    if mark:
                        used_nodes.add(other)
                    yield from walk(other, at + 1, length)
                    used_edges.discard(edge)
                    path.pop()
                    if mark:
                        used_nodes.discard(other)
                    if one and state["found"]:
                        return

            length = low
            while high is None or length <= high:
                while depth < length and frontier:
                    grow()
                if a not in dist:
                    if not frontier:
                        return  # `b` can't be reached from `a`
                    length += 1
                    continue
                state["found"] = state["cut"] = False
                yield from walk(a, 0, length)
                if state["found"] or not state["cut"]:
                    return
                length += 1

        yield from search(0)

    def _bind_paths(self, paths: list[str], binding: dict[str, Any]) -> None:
        """Bind each of the path variables `paths`, whose slots are all bound."""
        for p in paths:
            binding[p] = self._path_value(p, binding)

    def _path_value(self, p: str, binding: dict[str, Any]) -> Optional[tuple]:
        """The value of path variable `p`: `(node IDs, edge IDs)`, or `None` if a slot of its
        pattern is `None`. Each edge leads from the node before it to its other end."""
        first, parts = self._walks[p]
        node = binding[first]
        if node is None:
            return None
        nodes, edges = [node], []
        for slot, many in parts:
            value = binding[slot]
            if value is None:
                return None
            for edge in value if many else (value,):
                source, target = self.view.edge(edge)[:2]
                node = target if source == node else source
                nodes.append(node)
                edges.append(edge)
        return tuple(nodes), tuple(edges)

    def nodes_of(self, row: tuple) -> list[int]:
        """The nodes of a match: its node slots, then the inner nodes of its variable-length
        paths, skipping `None` slots."""
        nodes = [row[i] for i in self._node_slots if row[i] is not None]
        for rel, left in self._paths:
            edges, node = row[rel], row[left]
            if not edges:
                continue
            for edge in edges[:-1]:
                source, target = self.view.edge(edge)[:2]
                node = target if source == node else source
                nodes.append(node)
        return nodes

    def edges_of(self, row: tuple) -> list[int]:
        """The edges of a match, those of its variable-length paths included, skipping `None`
        slots."""
        edges = []
        for i, path in self._edge_slots:
            value = row[i]
            if value is not None:
                edges.extend(value if path else [value])
        return edges
