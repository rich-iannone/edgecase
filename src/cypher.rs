//! The Cypher matcher: the backtracking search that `Matcher` in `cypher.py` runs over a plan.
//!
//! Python parses and plans the query and compiles every condition into a Python function. It
//! hands the plan over as a *program* (see `Matcher._compile()`), where each condition may also
//! carry a *native* form: a label or type test, a comparison of a property or an ID with a
//! constant or another property, `IN` a constant list, `IS NULL`, a comparison of two IDs, a
//! string test (`STARTS WITH`, `ENDS WITH`, `CONTAINS`), or an `AND` or `OR` of native forms. With
//! `table=`, it also builds the sorted columns of `match()`. The search checks a
//! native form here, and calls the Python function only for the other conditions and for values
//! the native form can't decide (an attribute value of an unusual type, say). Either way the
//! answer is the one the Python evaluator gives, and the matches come in the same order as from
//! the Python search, which stays as the reference. That includes the steps of `shortestPath()`
//! and `allShortestPaths()`, whose search (a lazy breadth-first search from the far end, then
//! the trails from the near end for each length in turn) runs the same way in both.

use crate::graph::Graph;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::rc::Rc;

/// The value of a slot during the search
#[derive(Clone)]
enum Val {
    Unset,
    Null,
    Node(u64),
    Edge(u64),
    Path(Vec<u64>),
    /// A path variable: its nodes, and the edges between them
    Walk(Vec<u64>, Vec<u64>),
    /// Any other value a column of `WITH` holds, at this place in `Search::values`
    Value(usize),
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Node,
    Rel,
    Path,
    Walk,
    /// A column of `WITH` that holds any other value
    Value,
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Out,
    In,
    Both,
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn parse(text: &str) -> PyResult<Op> {
        Ok(match text {
            "=" => Op::Eq,
            "<>" => Op::Ne,
            "<" => Op::Lt,
            "<=" => Op::Le,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            _ => return Err(bad_program(&format!("unknown operator {text:?}"))),
        })
    }

    fn compare_op(self) -> CompareOp {
        match self {
            Op::Eq => CompareOp::Eq,
            Op::Ne => CompareOp::Ne,
            Op::Lt => CompareOp::Lt,
            Op::Le => CompareOp::Le,
            Op::Gt => CompareOp::Gt,
            Op::Ge => CompareOp::Ge,
        }
    }

    fn holds<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        }
    }
}

/// The kind of a value, as `_kind()` in `cypher.py` sees it. `Other` is anything the native
/// forms leave to Python.
#[derive(Clone, Copy, PartialEq)]
enum Class {
    Null,
    Bool,
    Number,
    Str,
    List,
    Map,
    Other,
}

fn classify(value: &Bound<'_, PyAny>) -> Class {
    if value.is_none() {
        Class::Null
    } else if value.is_instance_of::<PyBool>() {
        Class::Bool
    } else if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        Class::Number
    } else if value.is_instance_of::<PyString>() {
        Class::Str
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        Class::List
    } else if value.is_instance_of::<PyDict>() {
        Class::Map
    } else {
        Class::Other
    }
}

/// What a comparison reads from the binding
enum Operand {
    /// A property of a node or relationship slot, with `key == label_key` (for nodes) or
    /// `key == "weight"` (for relationships) noted
    Prop {
        slot: usize,
        key: Py<PyString>,
        special: bool,
    },
    /// The ID of a node or relationship slot
    Id(usize),
}

/// An operand of a string test, or a constant
enum Term {
    Operand(Operand),
    Const(Py<PyAny>),
}

#[derive(Clone, Copy)]
enum TextTest {
    StartsWith,
    EndsWith,
    Contains,
}

/// A condition the search can check without Python
enum Native {
    /// `x:A:B|C` for a node, or `r:A|B` for a relationship: every group must hold one label
    Labels(usize, Vec<Vec<Py<PyString>>>),
    /// `operand op constant`, for a constant boolean, number or string
    Compare(Operand, Op, Py<PyAny>, Class),
    /// `operand IN [constants]`, each a boolean, number, string or `null`
    In(Operand, Vec<(Py<PyAny>, Class)>),
    /// `slot IS NULL` (no key) or `slot.key IS NULL`, or with `negated`, `IS NOT NULL`
    IsNull(usize, Option<(Py<PyString>, bool)>, bool),
    /// `operand op operand`, such as `a.x < b.y`, when neither is an ID
    Compare2(Operand, Op, Operand),
    /// `term STARTS WITH term`, `ENDS WITH` or `CONTAINS`, true when both are strings
    Text(TextTest, Term, Term),
    /// `id(a) op id(b)`, also for `a = b` and `a <> b`
    Ids(usize, usize, Op),
    /// `x AND y AND ...`, which a native form decides only when it decides every part
    All(Vec<Native>),
    /// `x OR y OR ...`, likewise
    Any(Vec<Native>),
    /// A condition that is never true, such as `{k: null}`
    Never,
    /// A condition that reads one node or relationship slot only, whose results are kept by
    /// ID in `Search::cache[index]`, since it's often met again for the same node or edge
    Cached(Box<Native>, usize, usize),
}

impl Native {
    /// The one slot this reads, when its result depends only on the node or edge there
    fn single_slot(&self) -> Option<usize> {
        match self {
            Native::Labels(slot, _) | Native::IsNull(slot, Some(_), _) => Some(*slot),
            Native::Compare(Operand::Prop { slot, .. }, ..)
            | Native::In(Operand::Prop { slot, .. }, _) => Some(*slot),
            Native::Text(_, Term::Operand(Operand::Prop { slot, .. }), Term::Const(_))
            | Native::Text(_, Term::Const(_), Term::Operand(Operand::Prop { slot, .. })) => {
                Some(*slot)
            }
            _ => None,
        }
    }
}

struct Cond {
    native: Option<Native>,
    fallback: Py<PyAny>,
}

#[derive(PartialEq)]
enum StepKind {
    Scan,
    Expand,
    Connect,
    Shortest,
    AllShortest,
}

struct Step {
    kind: StepKind,
    node: usize,
    rel: usize,
    start: usize,
    dir: Dir,
    seed: Option<Py<PyAny>>,
    conds: Vec<Cond>,
    hops: Option<(usize, Option<usize>)>,
    edge_conds: Vec<Cond>,
    reverse: bool,
    fixed: bool,
    /// The path variables the step binds, each with the slots of its pattern
    paths: Vec<(usize, Vec<usize>)>,
}

struct Stage {
    optional: bool,
    entry: Vec<Cond>,
    new: Vec<usize>,
    joined: Vec<usize>,
    steps: Vec<Step>,
    /// The path variables bound before `entry` is checked
    paths: Vec<(usize, Vec<usize>)>,
}

struct Program {
    names: Vec<Py<PyString>>,
    kinds: Vec<Kind>,
    label_key: Option<String>,
    /// The slot the conditions on each edge of a variable-length relationship read
    edge_slot: usize,
    distinct_nodes: bool,
    stages: Vec<Stage>,
    /// The number of `Native::Cached` conditions
    caches: usize,
}

/// The kept results of one `Native::Cached` condition: a byte per ID (0 for unknown, 1 for
/// false, 2 for true) when the IDs are dense enough, else a hash map
enum Memo {
    Empty,
    Dense(Vec<u8>),
    Sparse(FxHashMap<u64, bool>),
}

impl Memo {
    fn get(&self, id: u64) -> Option<bool> {
        match self {
            Memo::Empty => None,
            Memo::Dense(bytes) => match bytes.get(id as usize) {
                Some(1) => Some(false),
                Some(2) => Some(true),
                _ => None,
            },
            Memo::Sparse(map) => map.get(&id).copied(),
        }
    }

    fn set(&mut self, graph: &Graph, nodes: bool, id: u64, result: bool) {
        if let Memo::Empty = self {
            let bound = graph.id_bound(nodes);
            *self = if bound <= 4 * graph.element_count(nodes) as u64 + 4096 {
                Memo::Dense(vec![0; bound as usize])
            } else {
                Memo::Sparse(FxHashMap::default())
            };
        }
        match self {
            Memo::Dense(bytes) => {
                if let Some(byte) = bytes.get_mut(id as usize) {
                    *byte = 1 + result as u8;
                }
            }
            Memo::Sparse(map) => {
                map.insert(id, result);
            }
            Memo::Empty => {}
        }
    }
}

/// A column of the table that `run()` gives with `table=` (see `Matcher.table()`)
enum Column {
    /// The value of a slot: an ID, a list of edge IDs, a path's dict of IDs or `None`
    Slot(usize),
    /// The label of a node slot, or a path's dict with the labels of its nodes
    Label(usize),
    /// A property of a node or relationship slot (an `Operand::Prop`)
    Prop(Operand),
    /// The place of the row's input in `inputs`
    Input,
}

/// A slot value kept for the table, with a path as a range of `Table::paths`
#[derive(Clone, Copy)]
enum Cell {
    Null,
    Id(u64),
    Path(usize, usize),
    /// A path variable: its nodes at `paths[i..j]` and its edges at `paths[j..k]`
    Walk(usize, usize, usize),
    /// A `Val::Value`
    Value(usize),
}

/// The matches kept for the table: the values of `slots`, row after row, and the input of
/// each row
struct Table {
    slots: Vec<usize>,
    rows: usize,
    cells: Vec<Cell>,
    paths: Vec<u64>,
    inputs: Vec<usize>,
}

fn bad_program(message: &str) -> PyErr {
    PyValueError::new_err(format!("invalid Cypher match program: {message}"))
}

// =============================================================================
// Reading the program
// =============================================================================

fn parse_program(program: &Bound<'_, PyAny>) -> PyResult<Program> {
    type Fields<'py> = (
        Vec<String>,
        Vec<String>,
        Option<String>,
        bool,
        Vec<Bound<'py, PyAny>>,
    );
    let (names, kinds, label_key, distinct_nodes, stages): Fields = program.extract()?;
    let py = program.py();
    let kinds = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "node" => Ok(Kind::Node),
            "relationship" => Ok(Kind::Rel),
            "relationships" => Ok(Kind::Path),
            "path" => Ok(Kind::Walk),
            "value" => Ok(Kind::Value),
            _ => Err(bad_program(&format!("unknown slot kind {kind:?}"))),
        })
        .collect::<PyResult<Vec<Kind>>>()?;
    if names.is_empty() || names.len() != kinds.len() {
        return Err(bad_program("the slot names and kinds don't match"));
    }
    let reader = Reader {
        kinds: &kinds,
        label_key: label_key.as_deref(),
        caches: Default::default(),
    };
    let stages = stages
        .iter()
        .map(|stage| reader.stage(stage))
        .collect::<PyResult<_>>()?;
    let caches = reader.caches.get();
    Ok(Program {
        edge_slot: names.len() - 1,
        names: names
            .iter()
            .map(|name| PyString::new(py, name).unbind())
            .collect(),
        kinds,
        label_key,
        distinct_nodes,
        stages,
        caches,
    })
}

struct Reader<'k> {
    kinds: &'k [Kind],
    label_key: Option<&'k str>,
    /// The number of `Native::Cached` conditions so far
    caches: std::cell::Cell<usize>,
}

impl Reader<'_> {
    fn slot(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let slot: usize = item.extract()?;
        if slot >= self.kinds.len() {
            return Err(bad_program(&format!("slot {slot} is out of range")));
        }
        Ok(slot)
    }

    fn stage(&self, stage: &Bound<'_, PyAny>) -> PyResult<Stage> {
        type Items<'py> = Vec<Bound<'py, PyAny>>;
        type Fields<'py> = (
            bool,
            Items<'py>,
            Items<'py>,
            Items<'py>,
            Items<'py>,
            Items<'py>,
        );
        let (optional, entry, new, joined, steps, paths): Fields = stage.extract()?;
        Ok(Stage {
            paths: self.paths(&paths)?,
            optional,
            entry: self.conds(&entry)?,
            new: new.iter().map(|s| self.slot(s)).collect::<PyResult<_>>()?,
            joined: joined
                .iter()
                .map(|s| self.slot(s))
                .collect::<PyResult<_>>()?,
            steps: steps
                .iter()
                .map(|step| self.step(step))
                .collect::<PyResult<_>>()?,
        })
    }

    fn step(&self, step: &Bound<'_, PyAny>) -> PyResult<Step> {
        type Fields<'py> = (
            String,
            Bound<'py, PyAny>,
            Option<Bound<'py, PyAny>>,
            Option<Bound<'py, PyAny>>,
            Option<String>,
            Option<Py<PyAny>>,
            Vec<Bound<'py, PyAny>>,
            Option<(usize, Option<usize>)>,
            Vec<Bound<'py, PyAny>>,
            bool,
            bool,
            Vec<Bound<'py, PyAny>>,
        );
        let (kind, node, rel, start, dir, seed, conds, hops, edge_conds, reverse, fixed, paths): Fields =
            step.extract()?;
        let kind = match kind.as_str() {
            "scan" => StepKind::Scan,
            "expand" => StepKind::Expand,
            "connect" => StepKind::Connect,
            "shortest" => StepKind::Shortest,
            "all_shortest" => StepKind::AllShortest,
            _ => return Err(bad_program(&format!("unknown step kind {kind:?}"))),
        };
        let dir = match dir.as_deref() {
            None | Some("both") => Dir::Both,
            Some("out") => Dir::Out,
            Some("in") => Dir::In,
            Some(other) => return Err(bad_program(&format!("unknown direction {other:?}"))),
        };
        let optional_slot = |item: Option<Bound<'_, PyAny>>| -> PyResult<usize> {
            match item {
                Some(item) => self.slot(&item),
                None if kind == StepKind::Scan => Ok(0),
                None => Err(bad_program("only a scan step can do without rel and start")),
            }
        };
        let (rel, start) = (optional_slot(rel)?, optional_slot(start)?);
        if matches!(kind, StepKind::Shortest | StepKind::AllShortest) && hops.is_none() {
            return Err(bad_program("a shortest-path step needs hops"));
        }
        Ok(Step {
            paths: self.paths(&paths)?,
            node: self.slot(&node)?,
            rel,
            start,
            dir,
            seed,
            conds: self.conds(&conds)?,
            hops,
            edge_conds: self.conds(&edge_conds)?,
            reverse,
            fixed,
            kind,
        })
    }

    /// `[(path, walk)]`: a path slot, and its pattern's first node and relationships
    fn paths(&self, paths: &[Bound<'_, PyAny>]) -> PyResult<Vec<(usize, Vec<usize>)>> {
        paths
            .iter()
            .map(|item| {
                let (path, walk): (Bound<'_, PyAny>, Vec<Bound<'_, PyAny>>) = item.extract()?;
                let path = self.slot(&path)?;
                let walk = walk
                    .iter()
                    .map(|s| self.slot(s))
                    .collect::<PyResult<Vec<_>>>()?;
                let kinds: Vec<Kind> = walk.iter().map(|&s| self.kinds[s]).collect();
                if self.kinds[path] != Kind::Walk
                    || kinds.first() != Some(&Kind::Node)
                    || !kinds[1..]
                        .iter()
                        .all(|k| matches!(k, Kind::Rel | Kind::Path))
                {
                    return Err(bad_program("a path is a node, then relationships"));
                }
                Ok((path, walk))
            })
            .collect()
    }

    fn conds(&self, conds: &[Bound<'_, PyAny>]) -> PyResult<Vec<Cond>> {
        conds
            .iter()
            .map(|cond| {
                let (native, fallback): (Option<Bound<'_, PyAny>>, Py<PyAny>) = cond.extract()?;
                let native = native.map(|n| self.native(&n)).transpose()?;
                let native = native.map(|n| match n.single_slot() {
                    Some(slot) => {
                        let index = self.caches.replace(self.caches.get() + 1);
                        Native::Cached(Box::new(n), slot, index)
                    }
                    None => n,
                });
                Ok(Cond { native, fallback })
            })
            .collect()
    }

    fn native(&self, native: &Bound<'_, PyAny>) -> PyResult<Native> {
        let tag: String = native.get_item(0)?.extract()?;
        let item = |i: usize| native.get_item(i);
        Ok(match tag.as_str() {
            "labels" => {
                let slot = self.single(&item(1)?)?;
                let groups: Vec<Vec<Bound<'_, PyString>>> = item(2)?.extract()?;
                let groups = groups
                    .into_iter()
                    .map(|group| group.into_iter().map(Bound::unbind).collect())
                    .collect();
                Native::Labels(slot, groups)
            }
            "cmp" => {
                let value = item(3)?;
                let class = self.constant(&value)?;
                if class == Class::Null {
                    Native::Never
                } else {
                    let op = Op::parse(&item(2)?.extract::<String>()?)?;
                    Native::Compare(self.operand(&item(1)?)?, op, value.unbind(), class)
                }
            }
            "in" => {
                let values: Vec<Bound<'_, PyAny>> = item(2)?.extract()?;
                let values = values
                    .into_iter()
                    .map(|v| Ok((self.constant(&v)?, v)))
                    .map(|found: PyResult<_>| found.map(|(class, v)| (v.unbind(), class)))
                    .collect::<PyResult<_>>()?;
                Native::In(self.operand(&item(1)?)?, values)
            }
            "null" => {
                let slot = self.slot(&item(1)?)?;
                let key: Option<Bound<'_, PyString>> = item(2)?.extract()?;
                let key = match key {
                    Some(key) => {
                        let (key, special) = self.prop_key(slot, key)?;
                        Some((key, special))
                    }
                    None => None,
                };
                Native::IsNull(slot, key, item(3)?.extract()?)
            }
            "ids" => {
                let op = Op::parse(&item(3)?.extract::<String>()?)?;
                Native::Ids(self.single(&item(1)?)?, self.single(&item(2)?)?, op)
            }
            "cmp2" => {
                let op = Op::parse(&item(2)?.extract::<String>()?)?;
                Native::Compare2(self.operand(&item(1)?)?, op, self.operand(&item(3)?)?)
            }
            "text" => {
                let test = match item(1)?.extract::<String>()?.as_str() {
                    "STARTS WITH" => TextTest::StartsWith,
                    "ENDS WITH" => TextTest::EndsWith,
                    "CONTAINS" => TextTest::Contains,
                    other => return Err(bad_program(&format!("unknown string test {other:?}"))),
                };
                Native::Text(test, self.term(&item(2)?)?, self.term(&item(3)?)?)
            }
            "and" | "or" => {
                let parts: Vec<Bound<'_, PyAny>> = item(1)?.extract()?;
                let parts = parts
                    .iter()
                    .map(|part| self.native(part))
                    .collect::<PyResult<_>>()?;
                if tag == "and" {
                    Native::All(parts)
                } else {
                    Native::Any(parts)
                }
            }
            "never" => Native::Never,
            _ => return Err(bad_program(&format!("unknown native condition {tag:?}"))),
        })
    }

    /// `("const", value)`, or an operand
    fn term(&self, term: &Bound<'_, PyAny>) -> PyResult<Term> {
        if term.get_item(0)?.extract::<String>()? == "const" {
            return Ok(Term::Const(term.get_item(1)?.unbind()));
        }
        Ok(Term::Operand(self.operand(term)?))
    }

    /// A node or (single) relationship slot
    fn single(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let slot = self.slot(item)?;
        if matches!(self.kinds[slot], Kind::Path | Kind::Walk | Kind::Value) {
            return Err(bad_program(
                "only a node or relationship slot can be tested natively",
            ));
        }
        Ok(slot)
    }

    fn operand(&self, operand: &Bound<'_, PyAny>) -> PyResult<Operand> {
        let tag: String = operand.get_item(0)?.extract()?;
        let slot = self.single(&operand.get_item(1)?)?;
        match tag.as_str() {
            "prop" => {
                let (key, special) = self.prop_key(slot, operand.get_item(2)?.extract()?)?;
                Ok(Operand::Prop { slot, key, special })
            }
            "id" => Ok(Operand::Id(slot)),
            _ => Err(bad_program(&format!("unknown operand {tag:?}"))),
        }
    }

    fn prop_key(&self, slot: usize, key: Bound<'_, PyString>) -> PyResult<(Py<PyString>, bool)> {
        if matches!(self.kinds[slot], Kind::Path | Kind::Walk | Kind::Value) {
            return Err(bad_program(
                "only a node or relationship slot has properties to read natively",
            ));
        }
        let text = key.to_cow()?;
        let special = match self.kinds[slot] {
            Kind::Node => self.label_key == Some(text.as_ref()),
            _ => text == "weight",
        };
        Ok((key.unbind(), special))
    }

    /// A table column: `("slot", slot)`, `("label", slot)`, `("prop", slot, key)` or
    /// `("input",)`
    fn column(&self, column: &Bound<'_, PyAny>) -> PyResult<Column> {
        let tag: String = column.get_item(0)?.extract()?;
        match tag.as_str() {
            "input" => Ok(Column::Input),
            "slot" => Ok(Column::Slot(self.slot(&column.get_item(1)?)?)),
            "label" => {
                let slot = self.slot(&column.get_item(1)?)?;
                if !matches!(self.kinds[slot], Kind::Node | Kind::Walk) {
                    return Err(bad_program("only a node or path slot has labels"));
                }
                Ok(Column::Label(slot))
            }
            "prop" => Ok(Column::Prop(self.operand(column)?)),
            _ => Err(bad_program(&format!("unknown column {tag:?}"))),
        }
    }

    /// The class of a constant, which must be `null`, a boolean, a number or a string
    fn constant(&self, value: &Bound<'_, PyAny>) -> PyResult<Class> {
        match classify(value) {
            class @ (Class::Null | Class::Bool | Class::Number | Class::Str) => Ok(class),
            _ => Err(bad_program(
                "a native constant must be null, a boolean, a number or a string",
            )),
        }
    }
}

// =============================================================================
// Searching
// =============================================================================

/// One level of the depth-first walk over the paths of a variable-length relationship
struct Frame {
    node: u64,
    /// The position of the next incident edge to try
    next: usize,
    /// Whether stepping onto `node` added it to the used nodes
    marked: bool,
}

struct Search<'a, 'py> {
    py: Python<'py>,
    graph: &'a Graph,
    program: &'a Program,
    binding: Vec<Val>,
    /// The nodes and edges used so far by each stage's clause
    used_nodes: Vec<FxHashSet<u64>>,
    used_edges: Vec<FxHashSet<u64>>,
    /// Whether each stage has found a match for the current row of the stages before it
    found: Vec<bool>,
    all_nodes: Option<Rc<Vec<u64>>>,
    /// The candidates of each seeded scan, by stage and step, worked out on first use
    seeded: Vec<Vec<Option<Rc<Vec<u64>>>>>,
    limit: Option<usize>,
    count_only: bool,
    count: usize,
    rows: Vec<Py<PyTuple>>,
    first: Option<Py<PyTuple>>,
    /// With `table=`, the matches kept for the table, in place of `rows`
    table: Option<Table>,
    /// The values of the `Val::Value` slots of every input
    values: Vec<Py<PyAny>>,
    /// The place in `inputs` of the input being searched
    input: usize,
    /// The results of each `Native::Cached` condition, by node or edge ID
    cache: RefCell<Vec<Memo>>,
    ticks: u32,
}

/// The incident edges of `node` in `dir`, from position `*next` on, as `_incident_edges()`
/// lists them: for `Both`, the out-edges and then the in-edges, each self-loop once
fn next_pair(graph: &Graph, node: u64, dir: Dir, next: &mut usize) -> Option<(u64, u64)> {
    let out = if dir == Dir::In {
        &[][..]
    } else {
        graph.out_pairs(node)
    };
    let inward = if dir == Dir::Out {
        &[][..]
    } else {
        graph.in_pairs(node)
    };
    loop {
        let i = *next;
        *next += 1;
        if i < out.len() {
            return Some(out[i]);
        }
        let pair = *inward.get(i - out.len())?;
        if dir == Dir::Both && pair.1 == node {
            continue; // a self-loop, already listed as an out-edge
        }
        return Some(pair);
    }
}

impl<'a, 'py> Search<'a, 'py> {
    fn node_at(&self, slot: usize) -> PyResult<u64> {
        match self.binding[slot] {
            Val::Node(node) => Ok(node),
            _ => Err(bad_program(
                "a step starts or ends at a slot that holds no node",
            )),
        }
    }

    fn value(&self, val: &Val) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match val {
            Val::Unset | Val::Null => py.None().into_bound(py),
            Val::Node(id) | Val::Edge(id) => (*id).into_pyobject(py)?.into_any(),
            Val::Path(edges) => PyTuple::new(py, edges.iter().copied())?.into_any(),
            Val::Walk(nodes, edges) => (
                PyTuple::new(py, nodes.iter().copied())?,
                PyTuple::new(py, edges.iter().copied())?,
            )
                .into_pyobject(py)?
                .into_any(),
            Val::Value(i) => self.values[*i].bind(py).clone(),
        })
    }

    /// The current match, as a tuple in slot order (without the edge slot)
    fn row(&self) -> PyResult<Py<PyTuple>> {
        let values = self.binding[..self.program.edge_slot]
            .iter()
            .map(|val| self.value(val))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(self.py, values)?.unbind())
    }

    /// Record a match; `true` means the limit is reached
    fn emit(&mut self) -> PyResult<bool> {
        self.count += 1;
        if !self.count_only && self.table.is_none() {
            let row = self.row()?;
            self.rows.push(row);
        } else if self.first.is_none() {
            self.first = Some(self.row()?);
        }
        if let Some(table) = &mut self.table {
            table.rows += 1;
            table.inputs.push(self.input);
            for &slot in &table.slots {
                let cell = match &self.binding[slot] {
                    Val::Node(id) | Val::Edge(id) => Cell::Id(*id),
                    Val::Path(edges) => {
                        let start = table.paths.len();
                        table.paths.extend_from_slice(edges);
                        Cell::Path(start, table.paths.len())
                    }
                    Val::Walk(nodes, edges) => {
                        let start = table.paths.len();
                        table.paths.extend_from_slice(nodes);
                        let middle = table.paths.len();
                        table.paths.extend_from_slice(edges);
                        Cell::Walk(start, middle, table.paths.len())
                    }
                    Val::Value(i) => Cell::Value(*i),
                    Val::Null | Val::Unset => Cell::Null,
                };
                table.cells.push(cell);
            }
        }
        Ok(self.limit.is_some_and(|limit| self.count >= limit))
    }

    /// Let Ctrl-C stop a long search
    fn tick(&mut self) -> PyResult<()> {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % 65536 == 0 {
            self.py.check_signals()?;
        }
        Ok(())
    }

    // --- conditions -----------------------------------------------------------

    fn check_all(&self, conds: &[Cond]) -> PyResult<bool> {
        for cond in conds {
            if !self.check(cond)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn check(&self, cond: &Cond) -> PyResult<bool> {
        if let Some(native) = &cond.native {
            if let Some(result) = self.native(native) {
                return Ok(result);
            }
        }
        let binding = PyDict::new(self.py);
        for (val, name) in self.binding.iter().zip(&self.program.names) {
            if !matches!(val, Val::Unset) {
                binding.set_item(name.bind(self.py), self.value(val)?)?;
            }
        }
        cond.fallback.bind(self.py).call1((binding,))?.is_truthy()
    }

    /// The result of a native condition, or `None` when it's for Python to decide (Python
    /// errors are also left to Python, which raises them the same way)
    fn native(&self, native: &Native) -> Option<bool> {
        let py = self.py;
        match native {
            Native::Labels(slot, groups) => self.has_labels(*slot, groups).ok(),
            Native::Compare(operand, op, constant, class) => {
                let value = self.operand(operand).ok()?;
                compare(&value, *op, constant.bind(py), *class)
            }
            Native::In(operand, constants) => {
                let value = self.operand(operand).ok()?;
                match classify(&value) {
                    Class::Other => None,
                    Class::Null => Some(false),
                    _ => {
                        for (constant, class) in constants {
                            if *class != Class::Null
                                && compare(&value, Op::Eq, constant.bind(py), *class)?
                            {
                                return Some(true);
                            }
                        }
                        Some(false)
                    }
                }
            }
            Native::IsNull(slot, key, negated) => {
                let null = match (&self.binding[*slot], key) {
                    (Val::Null | Val::Unset, _) => true,
                    (_, None) => false,
                    (_, Some((key, special))) => self.prop(*slot, key, *special).ok()?.is_none(),
                };
                Some(null != *negated)
            }
            Native::Compare2(a, op, b) => {
                let (a, b) = (self.operand(a).ok()?, self.operand(b).ok()?);
                // `compare()` takes a scalar on the right, so lists and maps are left to Python
                match classify(&b) {
                    Class::Null => Some(false),
                    class @ (Class::Bool | Class::Number | Class::Str) => {
                        compare(&a, *op, &b, class)
                    }
                    _ => None,
                }
            }
            Native::Text(test, a, b) => {
                let (a, b) = (self.term(a).ok()?, self.term(b).ok()?);
                // as `str.startswith()` and the others, which `null` and non-strings fail
                let (Ok(a), Ok(b)) = (a.downcast::<PyString>(), b.downcast::<PyString>()) else {
                    return Some(false);
                };
                let (a, b) = (a.to_cow().ok()?, b.to_cow().ok()?);
                Some(match test {
                    TextTest::StartsWith => a.starts_with(b.as_ref()),
                    TextTest::EndsWith => a.ends_with(b.as_ref()),
                    TextTest::Contains => a.contains(b.as_ref()),
                })
            }
            // only "is it true" matters, so `false` and `null` parts count alike; every part
            // must be decided, since Python would raise any error one of them meets
            Native::All(parts) => {
                let mut all = true;
                for part in parts {
                    all &= self.native(part)?;
                }
                Some(all)
            }
            Native::Any(parts) => {
                let mut any = false;
                for part in parts {
                    any |= self.native(part)?;
                }
                Some(any)
            }
            Native::Ids(a, b, op) => match (&self.binding[*a], &self.binding[*b]) {
                (Val::Node(x) | Val::Edge(x), Val::Node(y) | Val::Edge(y)) => Some(op.holds(x, y)),
                _ => Some(false),
            },
            Native::Never => Some(false),
            Native::Cached(inner, slot, index) => {
                let (Val::Node(id) | Val::Edge(id)) = self.binding[*slot] else {
                    return self.native(inner);
                };
                if let Some(result) = self.cache.borrow()[*index].get(id) {
                    return Some(result);
                }
                // (undecided results aren't kept, so Python sees each of those)
                let result = self.native(inner)?;
                let nodes = self.program.kinds[*slot] == Kind::Node;
                self.cache.borrow_mut()[*index].set(self.graph, nodes, id, result);
                Some(result)
            }
        }
    }

    fn term(&self, term: &Term) -> PyResult<Bound<'py, PyAny>> {
        match term {
            Term::Operand(operand) => self.operand(operand),
            Term::Const(value) => Ok(value.bind(self.py).clone()),
        }
    }

    fn operand(&self, operand: &Operand) -> PyResult<Bound<'py, PyAny>> {
        match operand {
            Operand::Prop { slot, key, special } => self.prop(*slot, key, *special),
            Operand::Id(slot) => match self.binding[*slot] {
                Val::Node(id) | Val::Edge(id) => Ok(id.into_pyobject(self.py)?.into_any()),
                _ => Ok(self.py.None().into_bound(self.py)),
            },
        }
    }

    fn prop(&self, slot: usize, key: &Py<PyString>, special: bool) -> PyResult<Bound<'py, PyAny>> {
        read_prop(self.graph, self.py, &self.binding[slot], key, special)
    }

    fn has_labels(&self, slot: usize, groups: &[Vec<Py<PyString>>]) -> PyResult<bool> {
        let py = self.py;
        let have: Vec<Bound<'py, PyAny>> = match self.binding[slot] {
            Val::Node(id) => {
                let node = self
                    .graph
                    .node_by_id(id)
                    .ok_or_else(|| bad_program("no such node"))?;
                let value = match &node.attributes {
                    Some(attrs) => attrs.bind(py).get_item("type")?,
                    None => None,
                };
                // as `labels_of()`: a string, or a non-empty list or tuple of strings
                match value {
                    Some(v) if v.is_instance_of::<PyString>() => vec![v],
                    Some(v) if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() => {
                        let items: Vec<Bound<'py, PyAny>> =
                            v.try_iter()?.collect::<PyResult<_>>()?;
                        if items.iter().all(|item| item.is_instance_of::<PyString>()) {
                            items
                        } else {
                            vec![]
                        }
                    }
                    _ => vec![],
                }
            }
            Val::Edge(id) => {
                let edge = self
                    .graph
                    .edge_by_id(id)
                    .ok_or_else(|| bad_program("no such edge"))?;
                let value = match &edge.attributes {
                    Some(attrs) => attrs.bind(py).get_item("rel")?,
                    None => None,
                };
                value
                    .filter(|v| v.is_instance_of::<PyString>())
                    .into_iter()
                    .collect()
            }
            _ => return Ok(false), // null
        };
        for group in groups {
            let mut any = false;
            for label in group {
                for item in &have {
                    if item.eq(label.bind(py))? {
                        any = true;
                        break;
                    }
                }
                if any {
                    break;
                }
            }
            if !any {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // --- the search -------------------------------------------------------------

    fn stage(&mut self, s: usize) -> PyResult<bool> {
        let program = self.program;
        if s == program.stages.len() {
            return self.emit();
        }
        let stage = &program.stages[s];
        self.found[s] = false;
        if self.enter(s)? && self.search(s, 0)? {
            return Ok(true);
        }
        if stage.optional && !self.found[s] {
            for &slot in &stage.new {
                self.binding[slot] = Val::Null;
            }
            return self.stage(s + 1);
        }
        Ok(false)
    }

    /// Check a stage's entry conditions and the slots it shares with earlier stages, and start
    /// its used nodes and edges with those slots
    fn enter(&mut self, s: usize) -> PyResult<bool> {
        let program = self.program;
        let stage = &program.stages[s];
        self.bind_paths(&stage.paths)?;
        if !self.check_all(&stage.entry)? {
            return Ok(false);
        }
        let distinct = program.distinct_nodes;
        self.used_nodes[s].clear();
        self.used_edges[s].clear();
        for &slot in &stage.joined {
            match self.binding[slot] {
                Val::Node(node) => {
                    if !self.used_nodes[s].insert(node) && distinct {
                        return Ok(false);
                    }
                }
                Val::Edge(edge) => {
                    if !self.used_edges[s].insert(edge) {
                        return Ok(false);
                    }
                }
                _ => return Ok(false), // null: a pattern that uses it again matches nothing
            }
        }
        if !distinct {
            self.used_nodes[s].clear();
        }
        // the seeds are worked out on entering the stage, as in the Python search, so that an
        // error in one raises at the same point
        for k in 0..stage.steps.len() {
            if stage.steps[k].seed.is_some() {
                self.candidates(s, k)?;
            }
        }
        Ok(true)
    }

    fn candidates(&mut self, s: usize, k: usize) -> PyResult<Rc<Vec<u64>>> {
        let program = self.program;
        if let Some(seed) = &program.stages[s].steps[k].seed {
            if let Some(found) = &self.seeded[s][k] {
                return Ok(found.clone());
            }
            let ids: Vec<u64> = seed.bind(self.py).call0()?.extract()?;
            let ids = Rc::new(ids);
            self.seeded[s][k] = Some(ids.clone());
            return Ok(ids);
        }
        Ok(self
            .all_nodes
            .get_or_insert_with(|| Rc::new(self.graph.sorted_node_ids()))
            .clone())
    }

    /// Bind the slots of step `k` of stage `s` and those after it in every way that matches;
    /// `true` means the limit is reached
    fn search(&mut self, s: usize, k: usize) -> PyResult<bool> {
        let program = self.program;
        let steps = &program.stages[s].steps;
        if k == steps.len() {
            self.found[s] = true;
            return self.stage(s + 1);
        }
        self.tick()?;
        let step = &steps[k];
        let distinct = program.distinct_nodes;
        if step.kind == StepKind::Scan {
            let candidates = self.candidates(s, k)?;
            for &node in candidates.iter() {
                if distinct && self.used_nodes[s].contains(&node) {
                    continue;
                }
                self.binding[step.node] = Val::Node(node);
                self.bind_paths(&step.paths)?;
                if self.check_all(&step.conds)? {
                    if distinct {
                        self.used_nodes[s].insert(node);
                    }
                    if self.search(s, k + 1)? {
                        return Ok(true);
                    }
                    if distinct {
                        self.used_nodes[s].remove(&node);
                    }
                }
            }
            return Ok(false);
        }
        if matches!(step.kind, StepKind::Shortest | StepKind::AllShortest) {
            return self.shortest(s, k);
        }
        if step.hops.is_some() {
            return self.paths(s, k);
        }
        let graph = self.graph;
        let connect = step.kind == StepKind::Connect;
        let start = self.node_at(step.start)?;
        let target = if connect {
            Some(self.node_at(step.node)?)
        } else {
            None
        };
        let fixed = match (step.fixed, &self.binding[step.rel]) {
            (false, _) => None,
            (true, Val::Edge(edge)) => Some(*edge),
            (true, _) => return Ok(false),
        };
        let mut next = 0;
        while let Some((edge, other)) = next_pair(graph, start, step.dir, &mut next) {
            match fixed {
                Some(fixed) if edge != fixed => continue,
                None if self.used_edges[s].contains(&edge) => continue,
                _ => {}
            }
            match target {
                Some(target) if other != target => continue,
                Some(_) => {}
                None if distinct && self.used_nodes[s].contains(&other) => continue,
                None => self.binding[step.node] = Val::Node(other),
            }
            self.binding[step.rel] = Val::Edge(edge);
            self.bind_paths(&step.paths)?;
            if !self.check_all(&step.conds)? {
                continue;
            }
            let fresh = distinct && !connect;
            if fixed.is_none() {
                self.used_edges[s].insert(edge);
            }
            if fresh {
                self.used_nodes[s].insert(other);
            }
            if self.search(s, k + 1)? {
                return Ok(true);
            }
            if fixed.is_none() {
                self.used_edges[s].remove(&edge);
            }
            if fresh {
                self.used_nodes[s].remove(&other);
            }
        }
        Ok(false)
    }

    /// The paths of a variable-length step: a depth-first walk from its start, using each edge
    /// once per clause, kept on an explicit stack so long paths can't overflow the call stack
    fn paths(&mut self, s: usize, k: usize) -> PyResult<bool> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let (low, high) = step.hops.unwrap_or((1, Some(1)));
        let distinct = program.distinct_nodes;
        let connect = step.kind == StepKind::Connect;
        let target = if connect {
            Some(self.node_at(step.node)?)
        } else {
            None
        };
        let start = self.node_at(step.start)?;
        let mut path: Vec<u64> = Vec::new();
        if self.path_end(s, k, start, &path, low)? {
            return Ok(true);
        }
        let mut stack = vec![Frame {
            node: start,
            next: 0,
            marked: false,
        }];
        while !stack.is_empty() {
            let depth = stack.len() - 1;
            let top = &mut stack[depth];
            let node = top.node;
            // with distinct nodes, a path can end at the target, but not pass through it
            let cut = high == Some(depth) || (distinct && depth > 0 && target == Some(node));
            let pair = if cut {
                None
            } else {
                next_pair(self.graph, node, step.dir, &mut top.next)
            };
            let Some((edge, other)) = pair else {
                let frame = stack.pop().expect("the stack isn't empty");
                if !stack.is_empty() {
                    if let Some(edge) = path.pop() {
                        self.used_edges[s].remove(&edge);
                    }
                    if frame.marked {
                        self.used_nodes[s].remove(&frame.node);
                    }
                }
                continue;
            };
            if self.used_edges[s].contains(&edge) {
                continue;
            }
            let mark = distinct && !self.used_nodes[s].contains(&other);
            if distinct && !mark && target != Some(other) {
                continue;
            }
            self.binding[program.edge_slot] = Val::Edge(edge);
            if !self.check_all(&step.edge_conds)? {
                continue;
            }
            self.used_edges[s].insert(edge);
            path.push(edge);
            if mark {
                self.used_nodes[s].insert(other);
            }
            stack.push(Frame {
                node: other,
                next: 0,
                marked: mark,
            });
            self.tick()?;
            if self.path_end(s, k, other, &path, low)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Try the walk's current path, which ends at `node`, as a match of the step
    fn path_end(
        &mut self,
        s: usize,
        k: usize,
        node: u64,
        path: &[u64],
        low: usize,
    ) -> PyResult<bool> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let connect = step.kind == StepKind::Connect;
        if path.len() < low {
            return Ok(false);
        }
        if connect {
            if self.node_at(step.node)? != node {
                return Ok(false);
            }
        } else if program.distinct_nodes && path.is_empty() {
            return Ok(false); // the end would be the start
        } else {
            self.binding[step.node] = Val::Node(node);
        }
        let mut edges = path.to_vec();
        if step.reverse {
            edges.reverse();
        }
        self.binding[step.rel] = Val::Path(edges);
        self.bind_paths(&step.paths)?;
        if self.check_all(&step.conds)? {
            return self.search(s, k + 1);
        }
        Ok(false)
    }

    /// Bind each path variable of `paths`, whose slots are all bound, as `_path_value()` does
    fn bind_paths(&mut self, paths: &[(usize, Vec<usize>)]) -> PyResult<()> {
        for (path, walk) in paths {
            self.binding[*path] = self.walk_value(walk)?;
        }
        Ok(())
    }

    fn walk_value(&self, walk: &[usize]) -> PyResult<Val> {
        let Val::Node(mut node) = self.binding[walk[0]] else {
            return Ok(Val::Null);
        };
        let (mut nodes, mut edges) = (vec![node], Vec::new());
        for &slot in &walk[1..] {
            let found = match &self.binding[slot] {
                Val::Edge(edge) => std::slice::from_ref(edge),
                Val::Path(path) => &path[..],
                _ => return Ok(Val::Null),
            };
            for &id in found {
                let edge = self
                    .graph
                    .edge_by_id(id)
                    .ok_or_else(|| bad_program("no such edge"))?;
                node = if edge.source == node {
                    edge.target
                } else {
                    edge.source
                };
                nodes.push(node);
                edges.push(id);
            }
        }
        Ok(Val::Walk(nodes, edges))
    }

    /// Whether the edge passes the conditions of step `k` of stage `s` on each edge of its
    /// variable-length relationship
    fn edge_passes(&mut self, s: usize, k: usize, edge: u64) -> PyResult<bool> {
        let program = self.program;
        self.binding[program.edge_slot] = Val::Edge(edge);
        self.check_all(&program.stages[s].steps[k].edge_conds)
    }

    /// The paths of a shortest-path step, as `shortest()` in `Matcher._extend()` finds them:
    /// those from `start` to `node` with the fewest edges among the ones that pass the step's
    /// conditions. For each length from the least on, a depth-first walk tries the paths of
    /// that length, cut short by each node's distance to the end, which a breadth-first search
    /// backwards from the end finds one layer at a time, as far as the walk needs.
    fn shortest(&mut self, s: usize, k: usize) -> PyResult<bool> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let (low, high) = step.hops.unwrap_or((1, None));
        let (a, b) = (self.node_at(step.start)?, self.node_at(step.node)?);
        let back = match step.dir {
            Dir::Out => Dir::In,
            Dir::In => Dir::Out,
            Dir::Both => Dir::Both,
        };
        let distinct = program.distinct_nodes;
        let mut dist: FxHashMap<u64, usize> = FxHashMap::default();
        dist.insert(b, 0);
        let (mut frontier, mut depth) = (vec![b], 0);
        let mut length = low;
        while high.is_none_or(|high| length <= high) {
            while depth < length && !frontier.is_empty() {
                let mut layer = Vec::new();
                for &node in &frontier {
                    let mut next = 0;
                    while let Some((edge, other)) = next_pair(self.graph, node, back, &mut next) {
                        if dist.contains_key(&other)
                            || self.used_edges[s].contains(&edge)
                            || (distinct && other != a && self.used_nodes[s].contains(&other))
                            || !self.edge_passes(s, k, edge)?
                        {
                            continue;
                        }
                        dist.insert(other, depth + 1);
                        layer.push(other);
                    }
                }
                self.tick()?;
                frontier = layer;
                depth += 1;
            }
            if !dist.contains_key(&a) {
                if frontier.is_empty() {
                    return Ok(false); // the end can't be reached from the start
                }
                length += 1;
                continue;
            }
            match self.shortest_walk(s, k, a, b, length, &dist, frontier.is_empty())? {
                None => return Ok(true),
                Some((found, cut)) if found || !cut => return Ok(false),
                Some(_) => length += 1,
            }
        }
        Ok(false)
    }

    /// The walk of `shortest()` over the paths of one `length`, on an explicit stack:
    /// `Some((found, cut))`, whether a path passed the step's conditions and whether a longer
    /// one might, or `None` when the limit is reached
    #[allow(clippy::too_many_arguments)]
    fn shortest_walk(
        &mut self,
        s: usize,
        k: usize,
        a: u64,
        b: u64,
        length: usize,
        dist: &FxHashMap<u64, usize>,
        exhausted: bool,
    ) -> PyResult<Option<(bool, bool)>> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let one = step.kind == StepKind::Shortest;
        let distinct = program.distinct_nodes;
        let mut path: Vec<u64> = Vec::new();
        if length == 0 {
            if a != b {
                return Ok(Some((false, true)));
            }
            return Ok(self.shortest_end(s, k, &path)?.map(|found| (found, true)));
        }
        let (mut found, mut cut) = (false, false);
        let mut stack = vec![Frame {
            node: a,
            next: 0,
            marked: false,
        }];
        while !stack.is_empty() {
            let at = stack.len() - 1;
            let top = &mut stack[at];
            let Some((edge, other)) = next_pair(self.graph, top.node, step.dir, &mut top.next)
            else {
                let frame = stack.pop().expect("the stack isn't empty");
                if !stack.is_empty() {
                    if let Some(edge) = path.pop() {
                        self.used_edges[s].remove(&edge);
                    }
                    if frame.marked {
                        self.used_nodes[s].remove(&frame.node);
                    }
                }
                continue;
            };
            if self.used_edges[s].contains(&edge) {
                continue;
            }
            let left = length - at - 1; // the edges still to take after this one
            let mark = distinct && !self.used_nodes[s].contains(&other);
            if distinct && !mark && !(left == 0 && other == b) {
                continue; // (the end itself is used, so a path ends there)
            }
            match dist.get(&other) {
                Some(&d) if d <= left => {}
                Some(_) => {
                    cut = true;
                    continue;
                }
                None => {
                    cut |= !exhausted;
                    continue;
                }
            }
            if !self.edge_passes(s, k, edge)? {
                continue;
            }
            self.used_edges[s].insert(edge);
            path.push(edge);
            if mark {
                self.used_nodes[s].insert(other);
            }
            if left > 0 {
                stack.push(Frame {
                    node: other,
                    next: 0,
                    marked: mark,
                });
                self.tick()?;
                continue;
            }
            cut = true;
            if other == b {
                match self.shortest_end(s, k, &path)? {
                    None => return Ok(None),
                    Some(passed) => found |= passed,
                }
            }
            self.used_edges[s].remove(&edge);
            path.pop();
            if mark {
                self.used_nodes[s].remove(&other);
            }
            if one && found {
                // undo the rest of the walk
                for &edge in &path {
                    self.used_edges[s].remove(&edge);
                }
                for frame in &stack[1..] {
                    if frame.marked {
                        self.used_nodes[s].remove(&frame.node);
                    }
                }
                break;
            }
        }
        Ok(Some((found, cut)))
    }

    /// Try a path of `shortest()` that reaches the end: whether it passed the step's
    /// conditions, or `None` when the limit is reached
    fn shortest_end(&mut self, s: usize, k: usize, path: &[u64]) -> PyResult<Option<bool>> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        self.binding[step.rel] = Val::Path(path.to_vec());
        self.bind_paths(&step.paths)?;
        if !self.check_all(&step.conds)? {
            return Ok(Some(false));
        }
        Ok(if self.search(s, k + 1)? {
            None
        } else {
            Some(true)
        })
    }
}

/// A property of a node or edge, as `GraphView.node_prop()` and `edge_prop()` read it, with
/// `special` for `key == label_key` (nodes) or `key == "weight"` (edges)
fn read_prop<'py>(
    graph: &Graph,
    py: Python<'py>,
    val: &Val,
    key: &Py<PyString>,
    special: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let none = || py.None().into_bound(py);
    match *val {
        Val::Node(id) => {
            let node = graph
                .node_by_id(id)
                .ok_or_else(|| bad_program("no such node"))?;
            if let Some(attrs) = &node.attributes {
                if let Some(value) = attrs.bind(py).get_item(key.bind(py))? {
                    return Ok(value);
                }
            }
            Ok(match (&node.label, special) {
                (Some(label), true) => PyString::new(py, label).into_any(),
                _ => none(),
            })
        }
        Val::Edge(id) => {
            let edge = graph
                .edge_by_id(id)
                .ok_or_else(|| bad_program("no such edge"))?;
            if let (Some(weight), true) = (edge.weight, special) {
                return Ok(PyFloat::new(py, weight).into_any());
            }
            if let Some(attrs) = &edge.attributes {
                if let Some(value) = attrs.bind(py).get_item(key.bind(py))? {
                    return Ok(value);
                }
            }
            Ok(none())
        }
        _ => Ok(none()),
    }
}

/// `value op constant` by Cypher's rules, or `None` when the value is for Python to judge.
/// `null` is never true, values of different kinds are unequal and unordered, and values of
/// the same kind compare as Python compares them, as `equals()` and `order()` do.
fn compare(
    value: &Bound<'_, PyAny>,
    op: Op,
    constant: &Bound<'_, PyAny>,
    class: Class,
) -> Option<bool> {
    match classify(value) {
        Class::Other => None,
        Class::Null => Some(false),
        kind if kind != class => Some(op == Op::Ne),
        _ => match op {
            Op::Eq | Op::Ne => {
                // as `equals()`, which counts a comparison that fails as unequal
                let same = value.eq(constant).unwrap_or(false);
                Some(same != (op == Op::Ne))
            }
            _ => value
                .rich_compare(constant, op.compare_op())
                .ok()?
                .is_truthy()
                .ok(),
        },
    }
}

/// The order of two table cells of one slot: by ID, a variable-length relationship as its list
/// of edge IDs, a path as its node IDs and then its edge IDs, and nulls last, as the keys of
/// `Matcher.table()` sort
fn cmp_cells(a: Cell, b: Cell, paths: &[u64]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Cell::Null, Cell::Null) => Ordering::Equal,
        (Cell::Null, _) => Ordering::Greater,
        (_, Cell::Null) => Ordering::Less,
        (Cell::Id(x), Cell::Id(y)) => x.cmp(&y),
        (Cell::Path(i, j), Cell::Path(k, l)) => paths[i..j].cmp(&paths[k..l]),
        (Cell::Walk(i, j, k), Cell::Walk(l, m, n)) => paths[i..j]
            .cmp(&paths[l..m])
            .then_with(|| paths[j..k].cmp(&paths[m..n])),
        _ => Ordering::Equal, // a slot holds one kind of value
    }
}

/// The table of `run()` with `table=`: `(count, columns, first)`, each column a list in the
/// order of the inputs and then the `sort` slots (or in search order, without `sort`)
#[allow(clippy::too_many_arguments)]
fn table_result(
    graph: &Graph,
    py: Python,
    program: &Program,
    table: Table,
    sort: Option<Vec<usize>>,
    columns: &[Column],
    first: Option<Py<PyTuple>>,
    held: &[Py<PyAny>],
) -> PyResult<PyObject> {
    let (width, count) = (table.slots.len(), table.rows);
    // where each slot is in a row of cells
    let mut at = vec![0; program.kinds.len()];
    for (i, &slot) in table.slots.iter().enumerate() {
        at[slot] = i;
    }
    let cell = |row: usize, slot: usize| table.cells[row * width + at[slot]];
    let mut order: Vec<usize> = (0..count).collect();
    if let Some(sort) = sort {
        let keys: Vec<usize> = sort.iter().map(|&slot| at[slot]).collect();
        order.sort_by(|&a, &b| {
            let c = table.inputs[a].cmp(&table.inputs[b]);
            if c.is_ne() {
                return c;
            }
            for &k in &keys {
                let c = cmp_cells(
                    table.cells[a * width + k],
                    table.cells[b * width + k],
                    &table.paths,
                );
                if c.is_ne() {
                    return c;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let none = || py.None().into_bound(py);
    let label = |id: u64| -> PyResult<Bound<'_, PyAny>> {
        let node = graph
            .node_by_id(id)
            .ok_or_else(|| bad_program("no such node"))?;
        Ok(match &node.label {
            Some(label) => PyString::new(py, label).into_any(),
            None => none(),
        })
    };
    // a path as `{"nodes": [...], "relationships": [...]}`, its nodes as IDs or labels
    let walk = |i: usize, j: usize, k: usize, labels: bool| -> PyResult<Bound<'_, PyAny>> {
        let nodes = if labels {
            let found = table.paths[i..j]
                .iter()
                .map(|&id| label(id))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, found)?
        } else {
            PyList::new(py, &table.paths[i..j])?
        };
        let dict = PyDict::new(py);
        dict.set_item("nodes", nodes)?;
        dict.set_item("relationships", PyList::new(py, &table.paths[j..k])?)?;
        Ok(dict.into_any())
    };
    let mut out = Vec::with_capacity(columns.len());
    for column in columns {
        let mut values = Vec::with_capacity(count);
        for &row in &order {
            let value = match column {
                Column::Slot(slot) => match cell(row, *slot) {
                    Cell::Null => none(),
                    Cell::Id(id) => id.into_pyobject(py)?.into_any(),
                    Cell::Path(i, j) => PyList::new(py, &table.paths[i..j])?.into_any(),
                    Cell::Walk(i, j, k) => walk(i, j, k, false)?,
                    Cell::Value(i) => held[i].bind(py).clone(),
                },
                Column::Label(slot) => match cell(row, *slot) {
                    Cell::Id(id) => label(id)?,
                    Cell::Walk(i, j, k) => walk(i, j, k, true)?,
                    _ => none(),
                },
                Column::Prop(Operand::Prop { slot, key, special }) => {
                    let val = match (cell(row, *slot), program.kinds[*slot]) {
                        (Cell::Id(id), Kind::Node) => Val::Node(id),
                        (Cell::Id(id), _) => Val::Edge(id),
                        _ => Val::Null,
                    };
                    read_prop(graph, py, &val, key, *special)?
                }
                Column::Prop(Operand::Id(_)) => return Err(bad_program("an ID isn't a column")),
                Column::Input => table.inputs[row].into_pyobject(py)?.into_any(),
            };
            values.push(value);
        }
        out.push(PyList::new(py, values)?);
    }
    let first = first.map_or_else(|| py.None(), |row| row.into_any());
    Ok((count, PyList::new(py, out)?, first)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The values of `inputs` (a list of tuples of values, as a binding holds them, for the first
/// slots), with those of `Val::Value` slots added to `values`
fn read_inputs(
    program: &Program,
    inputs: &Bound<'_, PyAny>,
    values: &mut Vec<Py<PyAny>>,
) -> PyResult<Vec<Vec<Val>>> {
    let rows: Vec<Vec<Bound<'_, PyAny>>> = inputs.extract()?;
    rows.into_iter()
        .map(|row| {
            if row.len() > program.edge_slot {
                return Err(bad_program("an input has more values than there are slots"));
            }
            row.into_iter()
                .zip(&program.kinds)
                .map(|(value, kind)| {
                    if value.is_none() {
                        return Ok(Val::Null);
                    }
                    Ok(match kind {
                        Kind::Node => Val::Node(value.extract()?),
                        Kind::Rel => Val::Edge(value.extract()?),
                        Kind::Path => Val::Path(value.extract()?),
                        Kind::Walk => {
                            let (nodes, edges) = value.extract()?;
                            Val::Walk(nodes, edges)
                        }
                        Kind::Value => {
                            values.push(value.unbind());
                            Val::Value(values.len() - 1)
                        }
                    })
                })
                .collect()
        })
        .collect()
}

pub(crate) fn run(
    graph: &Graph,
    py: Python,
    program: &Bound<'_, PyAny>,
    limit: Option<usize>,
    count_only: bool,
    table: Option<&Bound<'_, PyAny>>,
    inputs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let program = parse_program(program)?;
    let mut values = Vec::new();
    let inputs = match inputs {
        Some(inputs) => read_inputs(&program, inputs, &mut values)?,
        None => vec![Vec::new()],
    };
    // with `table=`, `(sort, columns)`: the slots to sort by (or `None`) and the columns
    let mut spec = None;
    if let Some(table) = table {
        let (sort, columns): (Option<Vec<Bound<'_, PyAny>>>, Vec<Bound<'_, PyAny>>) =
            table.extract()?;
        let reader = Reader {
            kinds: &program.kinds,
            label_key: program.label_key.as_deref(),
            caches: Default::default(),
        };
        let sort = sort
            .map(|slots| {
                slots
                    .iter()
                    .map(|slot| reader.slot(slot))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let columns = columns
            .iter()
            .map(|column| reader.column(column))
            .collect::<PyResult<Vec<_>>>()?;
        let mut slots: Vec<usize> = sort.iter().flatten().copied().collect();
        for column in &columns {
            let slot = match column {
                Column::Slot(slot) | Column::Label(slot) => *slot,
                Column::Prop(Operand::Prop { slot, .. }) | Column::Prop(Operand::Id(slot)) => *slot,
                Column::Input => continue,
            };
            slots.push(slot);
        }
        slots.sort_unstable();
        slots.dedup();
        spec = Some((sort, columns, slots));
    }
    let mut search = Search {
        py,
        graph,
        program: &program,
        binding: vec![Val::Unset; program.kinds.len()],
        used_nodes: vec![FxHashSet::default(); program.stages.len()],
        used_edges: vec![FxHashSet::default(); program.stages.len()],
        found: vec![false; program.stages.len()],
        all_nodes: None,
        seeded: program
            .stages
            .iter()
            .map(|stage| vec![None; stage.steps.len()])
            .collect(),
        limit,
        count_only,
        count: 0,
        rows: Vec::new(),
        first: None,
        table: spec.as_ref().map(|(_, _, slots)| Table {
            slots: slots.clone(),
            rows: 0,
            cells: Vec::new(),
            paths: Vec::new(),
            inputs: Vec::new(),
        }),
        values,
        input: 0,
        cache: RefCell::new((0..program.caches).map(|_| Memo::Empty).collect()),
        ticks: 0,
    };
    if limit != Some(0) {
        for (i, input) in inputs.into_iter().enumerate() {
            search.input = i;
            search.binding.fill(Val::Unset);
            for (slot, val) in input.into_iter().enumerate() {
                search.binding[slot] = val;
            }
            if search.stage(0)? {
                break; // the limit is reached
            }
        }
    }
    if let (Some((sort, columns, _)), Some(table)) = (spec, search.table.take()) {
        let values = std::mem::take(&mut search.values);
        return table_result(
            graph,
            py,
            &program,
            table,
            sort,
            &columns,
            search.first,
            &values,
        );
    }
    if count_only {
        let first = search.first.map_or_else(|| py.None(), |row| row.into_any());
        return Ok((search.count, first).into_pyobject(py)?.into_any().unbind());
    }
    Ok(PyList::new(py, search.rows)?.into_any().unbind())
}
