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
    """`-[variable:TYPE|OTHER {key: value}]->`, with `direction` `"out"` (`->`), `"in"` (`<-`) or
    `"either"` (no arrow). `types` holds the alternatives (empty for any type)."""

    variable: Optional[str] = None
    types: list[str] = field(default_factory=list)
    props: dict[str, Expr] = field(default_factory=dict)
    direction: str = "out"
    pos: int = field(default=0, compare=False)


@dataclass
class Pattern:
    """A chain of nodes joined by relationships: `rels[k]` joins `nodes[k]` and `nodes[k + 1]`."""

    nodes: list[NodePattern]
    rels: list[RelPattern] = field(default_factory=list)


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
class Query:
    """`MATCH patterns [WHERE where] [RETURN returns]`.

    `variables` maps each named variable to `"node"` or `"relationship"`, in the order they
    first appear.
    """

    patterns: list[Pattern]
    where: Optional[Expr] = None
    returns: Optional[Return] = None
    variables: dict[str, str] = field(default_factory=dict)


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
    if word == "MATCH":
        return "only one MATCH clause is supported; separate the patterns with commas instead"
    if word == "OPTIONAL":
        return "OPTIONAL MATCH isn't supported"
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
                self._fail(f"only CREATE clauses can build a graph, found {word!r}")
            patterns.append(self._pattern(create=True))
            while self._accept(","):
                patterns.append(self._pattern(create=True))
        return patterns

    def parse_query(self) -> Query:
        if not self._accept_keyword("MATCH"):
            if self.i >= len(self.tokens):
                self._fail("a query starts with MATCH, but the text is empty")
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            if reason is None or word.upper() == "MATCH":
                reason = f"a query starts with MATCH, found {word!r}"
            self._fail(reason)
        patterns = [self._pattern(create=False)]
        while self._accept(","):
            patterns.append(self._pattern(create=False))
        where = self._expr() if self._accept_keyword("WHERE") else None
        returns, pos = None, self._pos()
        if self._accept_keyword("RETURN"):
            returns = self._return(pos)
        self._accept(";")
        if self.i < len(self.tokens):
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            if reason is None:
                expected = ("the end of the query" if returns else
                            "RETURN or the end of the query" if where else
                            "WHERE, RETURN or the end of the query")
                reason = f"expected {expected}, found {word!r}"
            self._fail(reason)
        query = Query(patterns, where, returns)
        self._check_variables(query)
        return query

    def _check_variables(self, query: Query) -> None:
        kinds = query.variables
        for pattern in query.patterns:
            parts: list[Union[NodePattern, RelPattern]] = [pattern.nodes[0]]
            for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                parts += [rel, node]
            for part in parts:
                if part.variable is None:
                    continue
                kind = "node" if isinstance(part, NodePattern) else "relationship"
                if part.variable not in kinds:
                    kinds[part.variable] = kind
                elif kinds[part.variable] != kind:
                    self._fail(f"`{part.variable}` is used for both a node and a relationship",
                               part.pos)
                elif kind == "relationship":
                    # a row matches each relationship once, so its variable can't repeat
                    self._fail(f"the relationship variable `{part.variable}` is used more than "
                               "once", part.pos)
        exprs = [query.where] if query.where is not None else []
        if query.returns is not None:
            exprs += [item.expr for item in query.returns.items]
        for expr in exprs:
            for sub in walk(expr):
                if isinstance(sub, Var) and sub.name not in kinds:
                    self._fail(f"the variable `{sub.name}` isn't defined in MATCH", sub.pos)

    def _return(self, start: int) -> Return:
        distinct = self._accept_keyword("DISTINCT")
        items: list[ReturnItem] = []
        while True:
            pos = self._pos()
            if self._accept("*"):
                self._fail("RETURN * isn't supported; name the variables to return", pos)
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
                self._fail(
                    "variable-length relationships (*) aren't supported yet; "
                    "select_nodes_in_neighborhood() or trav_out_until() can follow several hops"
                )
            if self._peek() == "{":
                rel.props = self._map(create)
            self._expect("]")
        self._expect("-", "'-' in a relationship")
        forward = self._accept(">")
        if back and forward:
            self._fail("a relationship can't point both ways", start)
        rel.direction = "in" if back else "out" if forward else "either"
        return rel

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
    """Parse a `MATCH ... [WHERE ...] [RETURN ...]` query."""
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
    return "other"


def _describe(value: Any) -> str:
    kind = _kind(value)
    if kind in ("node", "relationship"):
        return f"a {kind}"
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
    if ka != kb or ka in ("null", "map", "node", "relationship", "other"):
        return None
    if ka == "list":
        for x, y in zip(a, b):
            c = order(x, y)
            if c is None or c != 0:
                return c
        return len(a) - len(b)
    if ka == "number" and (a != a or b != b):
        return None
    return (a > b) - (a < b)


_ORDERINGS = {
    "<": lambda c: c < 0, "<=": lambda c: c <= 0, ">": lambda c: c > 0, ">=": lambda c: c >= 0,
}


def _is_nan(value: Any) -> bool:
    return _kind(value) == "number" and value != value


class Evaluator:
    """Compiles the expressions of one query into Python functions of a binding.

    A binding maps variable names to node or edge IDs. `variables` gives the kind of each
    variable (`"node"` or `"relationship"`, as in `Query.variables`), `params` the values of the
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
        name = expr.name
        if self.variables[name] == "node":
            return lambda binding: NodeRef(binding[name])
        return lambda binding: RelRef(binding[name])

    def _listexpr(self, expr: ListExpr):
        items = [self.compile(item) for item in expr.items]
        return lambda binding: [item(binding) for item in items]

    def _prop(self, expr: Prop):
        key, view = expr.key, self.view
        if isinstance(expr.subject, Var):
            # the usual case, without making a NodeRef or RelRef first
            name = expr.subject.name
            if self.variables[name] == "node":
                return lambda binding: view.node_prop(binding[name], key)
            return lambda binding: view.edge_prop(binding[name], key)
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


# lower-cased name -> implementation; `exists()` is handled by the evaluator itself
_FUNCTIONS = {
    "id": _fn_id, "type": _fn_type, "labels": _fn_labels, "size": _fn_size,
    "tolower": _fn_lower, "toupper": _fn_upper, "exists": None,
}
# lower-cased name -> the name as documented
_FUNCTION_NAMES = {f.lower(): f for f in ("id", "type", "labels", "size", "toLower", "toUpper",
                                          "exists")}


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
    """One step of a `Plan`. It binds one node, or one relationship and the node at its far end.

    - `"scan"` binds `node` to each node of the graph. With a `seed` (the operator and value
      side of an `id(node) = value` or `id(node) IN value` condition), it binds only those IDs.
    - `"expand"` binds `rel` to each edge at the bound node `start`, as
      `_incident_edges(start, direction)` lists them, and `node` to the edge's other end.
    - `"connect"` is the same, for edges whose other end is the already bound `node`.

    `conditions` are checked as soon as the step has bound its slots.
    """

    kind: str
    node: str
    rel: Optional[str] = None
    start: Optional[str] = None
    direction: Optional[str] = None
    seed: Optional[tuple[str, Expr]] = None
    conditions: list[Expr] = field(default_factory=list)

    def __str__(self) -> str:
        if self.kind == "scan":
            return f"scan {self.node}" + (" by id" if self.seed else "")
        arrow = {"out": "-[{}]->", "in": "<-[{}]-", "both": "-[{}]-"}[self.direction]
        return f"{self.kind} ({self.start}){arrow.format(self.rel)}({self.node})"


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


class Plan:
    """The order in which to bind the slots of a `MATCH`, and where each condition is checked.

    - **Slots:** a slot is a node or relationship of the patterns. Each named variable is one
      slot, however often it appears. Each anonymous element gets a slot of its own, named so
      that no variable clashes with it (`" 0"`, `" 1"`, ...). `slots` maps every slot to
      `"node"` or `"relationship"`, in the order they first appear.
    - **Conditions:** the patterns' labels, types and properties (`element_conditions()`) and
      the parts of the `WHERE` (`conjuncts()`). Each is checked at the first step that has bound
      every slot it reads. Those that read no slot are `constant`.
    - **Order:** each connected group of patterns starts at its most constrained node (by
      `_selectivity()`, ties going to the first). It then grows along relationships, closing a
      cycle (`"connect"`) whenever it can, so that every later node is reached through an edge
      rather than by a scan. Groups that share no variable are scanned one after another, which
      gives their cross product.
    - **Directions:** on an undirected graph every relationship is followed with `"both"`, so
      `-->`, `<--` and `--` mean the same.
    """

    def __init__(self, query: Query, directed: bool = True):
        self.slots: dict[str, str] = {}
        rels: list[tuple[str, str, str, str]] = []  # (slot, left node, right node, arrow)
        conditions: list[Expr] = []
        counter = 0

        def slot(element: Union[NodePattern, RelPattern], kind: str) -> str:
            nonlocal counter
            name = element.variable
            while name is None or (element.variable is None and name in query.variables):
                name, counter = f" {counter}", counter + 1
            self.slots.setdefault(name, kind)
            conditions.extend(element_conditions(name, element))
            return name

        for pattern in query.patterns:
            left = slot(pattern.nodes[0], "node")
            for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                rel_slot = slot(rel, "relationship")
                right = slot(node, "node")
                rels.append((rel_slot, left, right, rel.direction))
                left = right
        conditions += conjuncts(query.where)

        needs = [variables_in(c) for c in conditions]
        self.constant = [c for c, n in zip(conditions, needs) if not n]
        alone = {s: [c for c, n in zip(conditions, needs) if n == {s}] for s in self.slots}
        score = {s: _selectivity(alone[s], s) for s in self.slots}

        self.steps: list[Step] = []
        placed = [not n for n in needs]
        bound: set[str] = set()

        def add(step: Step) -> None:
            bound.update(s for s in (step.node, step.rel) if s is not None)
            for i, n in enumerate(needs):
                if not placed[i] and n <= bound:
                    placed[i] = True
                    step.conditions.append(conditions[i])
            self.steps.append(step)

        nodes = [s for s, kind in self.slots.items() if kind == "node"]
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
            rel_slot, left, right, arrow = rel
            if left in bound:
                start, node, direction = left, right, "both" if arrow == "either" else arrow
            else:
                start, node, direction = right, left, _FLIP[arrow]
            add(Step("connect" if closing else "expand", node, rel_slot, start,
                     direction if directed else "both"))

    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)


# =============================================================================
# Matching
# =============================================================================


class Matcher:
    """Finds the matches of a parsed query in a Rust graph, by a backtracking search over its
    `Plan`.

    - **Matches:** `matches()` yields each match as a dict of slot → node or edge ID, covering
      every slot, anonymous ones included. They come in search order, which is fixed for a
      given graph and query.
    - **Distinctness:** the relationships of one match are distinct edges, which is Cypher's
      rule, while nodes may repeat. With `distinct_nodes=True`, the node slots must be distinct
      nodes too, as in a subgraph isomorphism.
    - **When errors are raised:** conditions are compiled when the matcher is made, so a missing
      parameter or an unknown function raises then, even on an empty graph. Errors in values
      (`WHERE n.name`, say) raise when a row meets them.
    """

    def __init__(self, query: Query, text: str, graph: Any, params: Optional[dict] = None,
                 label_key: Optional[str] = "name", distinct_nodes: bool = False):
        self.plan = Plan(query, graph.is_directed)
        self.graph = graph
        self.distinct_nodes = distinct_nodes
        self.evaluator = evaluator = Evaluator(text, self.plan.slots, params,
                                               GraphView(graph, label_key))
        self._constant = [evaluator.condition(c) for c in self.plan.constant]
        self._checks = [[evaluator.condition(c) for c in step.conditions]
                        for step in self.plan.steps]
        self._seeds = [step.seed and (step.seed[0], evaluator.compile(step.seed[1]))
                       for step in self.plan.steps]
        self._all_nodes: Optional[list[int]] = None

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

    def matches(self):
        if not all(check({}) for check in self._constant):
            return
        steps, checks, graph = self.plan.steps, self._checks, self.graph
        candidates = [self._candidates(seed) if step.kind == "scan" else None
                      for step, seed in zip(steps, self._seeds)]
        distinct = self.distinct_nodes
        binding: dict[str, int] = {}
        used_edges: set[int] = set()
        used_nodes: set[int] = set()
        last = len(steps)

        def search(k):
            if k == last:
                yield dict(binding)
                return
            step, tests = steps[k], checks[k]
            if step.kind == "scan":
                for node in candidates[k]:
                    if distinct and node in used_nodes:
                        continue
                    binding[step.node] = node
                    if all(test(binding) for test in tests):
                        used_nodes.add(node)
                        yield from search(k + 1)
                        used_nodes.discard(node)
                return
            connect = step.kind == "connect"
            target = binding[step.node] if connect else None
            for edge, other in graph._incident_edges(binding[step.start], step.direction):
                if edge in used_edges:
                    continue
                if connect:
                    if other != target:
                        continue
                elif distinct and other in used_nodes:
                    continue
                else:
                    binding[step.node] = other
                binding[step.rel] = edge
                if all(test(binding) for test in tests):
                    used_edges.add(edge)
                    fresh = distinct and not connect
                    if fresh:
                        used_nodes.add(other)
                    yield from search(k + 1)
                    used_edges.discard(edge)
                    if fresh:
                        used_nodes.discard(other)

        yield from search(0)
