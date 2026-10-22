import ast
import html
import json
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import polars as pl

from edgecase import _core as rust
from edgecase.graph import Graph
from edgecase.steps import KIND_BUILD_NODES, StepDelta


def from_matrix_market(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a Matrix Market coordinate format file.

    Matrix Market is a standard format for representing sparse matrices and networks. This function
    reads coordinate format files (.mtx) and automatically detects graph properties from the header
    line:

    - Symmetry: 'symmetric' -> undirected, 'general' -> directed
    - Data type: 'pattern' -> unweighted, 'real'/'integer' -> weighted

    The function automatically converts 1-indexed node IDs (Matrix Market standard) to 0-indexed
    node IDs used internally.

    Parameters
    ----------
    filepath
        Path to the Matrix Market (.mtx) file.
    directed
        Whether to create a directed graph. If None (default), automatically detected from file
        header ('symmetric' -> False, 'general' -> True).
    weighted
        Whether the graph is weighted. If None (default), automatically detected from file header
        ('pattern' -> False, 'real'/'integer' -> True).
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph constructed from the Matrix Market file

    Examples
    --------
    ```python
    from edgecase.io import from_matrix_market

    # Example assumes .mtx files exist in your working directory
    # Auto-detect graph properties from file header
    # G = from_matrix_market("soc-karate.mtx")
    # print(f"Directed: {G.is_directed}, Weighted: {G.is_weighted}")
    # Directed: False, Weighted: False

    # Override auto-detection
    # G = from_matrix_market("network.mtx", directed=True, weighted=False)

    # Load with a custom name
    # G = from_matrix_market(
    #     "network.mtx",
    #     graph_name="My Network"
    # )
    # print(G.name)
    # My Network
    ```

    Notes
    -----
    - Matrix Market files use 1-based indexing, which is automatically converted to 0-based indexing
    - Only coordinate format (sparse) is supported
    - Header format: %%MatrixMarket matrix coordinate <type> <symmetry>
      * Symmetry: 'symmetric' -> undirected, 'general' -> directed
      * Type: 'pattern' -> unweighted, 'real'/'integer' -> weighted
    - Parameters `directed` and `weighted` override auto-detection when provided

    """
    filepath = Path(filepath)

    # Parse header to detect graph properties
    detected_directed = None
    detected_weighted = None

    with open(filepath, "r") as f:
        first_line = f.readline().strip()

        # Parse Matrix Market header: %%MatrixMarket matrix coordinate <type> <symmetry>
        if first_line.startswith("%%MatrixMarket"):
            parts = first_line.lower().split()

            # Check symmetry (last part)
            if len(parts) >= 5:
                symmetry = parts[4]
                if symmetry == "symmetric":
                    detected_directed = False
                elif symmetry in ("general", "skew-symmetric"):
                    detected_directed = True

            # Check data type (4th part)
            if len(parts) >= 4:
                data_type = parts[3]
                if data_type == "pattern":
                    detected_weighted = False
                elif data_type in ("real", "integer", "complex"):
                    detected_weighted = True

    # Use provided values or fall back to detected values or defaults
    use_directed = (
        directed
        if directed is not None
        else (detected_directed if detected_directed is not None else False)
    )
    use_weighted = (
        weighted
        if weighted is not None
        else (detected_weighted if detected_weighted is not None else False)
    )

    # Read the file and parse edges
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if line.startswith("%"):
                continue

            parts = line.split()

            # Skip header line (rows cols nonzeros)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                continue

            # Parse edge data
            if len(parts) >= 2:
                # Convert from 1-indexed to 0-indexed
                source = int(parts[0]) - 1
                target = int(parts[1]) - 1

                # Get weight if available and requested
                weight = None
                if use_weighted and len(parts) >= 3:
                    weight = float(parts[2])

                edges.append((source, target, weight))

    # Create edges DataFrame
    edge_data = {"from": [e[0] for e in edges], "to": [e[1] for e in edges]}

    if use_weighted:
        edge_data["weight"] = [e[2] if e[2] is not None else 1.0 for e in edges]

    edges_df = pl.DataFrame(edge_data)

    # Build and return graph
    # Note: nodes_df is None, so nodes will be auto-created from edges
    graph = Graph(
        directed=use_directed, weighted=use_weighted, edges_df=edges_df, graph_name=graph_name
    )

    return graph


# =============================================================================
# Shared helpers
# =============================================================================

# Namespace for the `json` marker EdgeCase puts on GraphML keys and GEXF attributes whose values
# are JSON-encoded (lists, dicts, mixed types), so they read back as the original values
_EC_NS = "https://github.com/rich-iannone/edgecase"
_EC_JSON = f"{{{_EC_NS}}}json"

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

_GEXF_VERSIONS = {
    "1.2draft": ("http://www.gexf.net/1.2draft", "http://www.gexf.net/1.2draft/viz", "1.2"),
    "1.3": ("http://gexf.net/1.3", "http://gexf.net/1.3/viz", "1.3"),
}


def _code(func: str, source: Any) -> str:
    """The code string recorded in a reader's build step, e.g. `from_gml('net.gml')`."""
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, (str, bytes)) and len(source) <= 60:
        return f"{func}({source!r})"
    return f"{func}(<{type(source).__name__}>)"


def _build_graph(
    nodes: list[tuple[Any, Optional[str], dict]],
    edges: list[tuple[Any, Any, Optional[float], dict]],
    directed: bool,
    weighted: bool,
    graph_name: Optional[str],
    code: str,
) -> Graph:
    """
    Build a graph from parsed file contents, recording it as a single step.

    `nodes` holds `(file_key, label, attributes)` in file order and `edges` holds
    `(source_key, target_key, weight, attributes)`. Edges may name keys with no node entry; those
    nodes are created, labeled with the key. A node without a label is labeled with its key. When
    the labels aren't unique (common in Gephi and Cytoscape exports, where the label is only a
    display name), every node is labeled with its key instead and the file's label is kept in a
    `name` attribute (`file_label` if `name` is taken).
    """
    index: dict[Any, int] = {}
    keys: list = []
    labels: list[Optional[str]] = []
    attrs: list[dict] = []
    for key, label, node_attrs in nodes:
        if key in index:
            raise ValueError(f"Node id {key!r} appears more than once in the file")
        index[key] = len(keys)
        keys.append(key)
        labels.append(None if label is None else str(label))
        attrs.append(node_attrs)
    for source, target, _, _ in edges:
        for key in (source, target):
            if key not in index:
                index[key] = len(keys)
                keys.append(key)
                labels.append(None)
                attrs.append({})

    resolved = [str(key) if label is None else label for key, label in zip(keys, labels)]
    if len(set(resolved)) < len(resolved):
        seen: set[str] = set()
        dupe = next(label for label in resolved if label in seen or seen.add(label))
        by_key = [str(key) for key in keys]
        if len(set(by_key)) < len(by_key):
            raise ValueError(f"Node label {dupe!r} is used by more than one node")
        attr = "file_label" if any("name" in a for a in attrs) else "name"
        warnings.warn(
            f"Node label {dupe!r} is used by more than one node, so the file's node ids are used "
            f"as labels instead; the labels from the file are kept in the {attr!r} attribute",
            stacklevel=3,
        )
        for node_attrs, label in zip(attrs, labels):
            if label is not None:
                node_attrs[attr] = label
        resolved = by_key

    graph = Graph(directed=directed, weighted=weighted, graph_name=graph_name)
    add_node = graph._graph.add_node
    ids = [
        add_node(rust.Node(id=0, label=label, attributes=_clean(node_attrs)))
        for label, node_attrs in zip(resolved, attrs)
    ]
    add_edge = graph._graph.add_edge
    pairs = []
    for source, target, weight, edge_attrs in edges:
        s, t = index[source], index[target]
        add_edge(
            rust.Edge(
                id=0,
                source=ids[s],
                target=ids[t],
                weight=weight if weighted else None,
                attributes=_clean(edge_attrs),
            )
        )
        pairs.append((resolved[s], resolved[t]))

    method = code.split("(", 1)[0]
    graph._record_step(
        KIND_BUILD_NODES, method, code, delta=StepDelta(nodes_added=resolved, edges_added=pairs)
    )
    return graph


def _clean(attrs: dict) -> Optional[dict]:
    """Drop `None` values; `None` for an empty dict (what the Rust constructors expect)."""
    kept = {k: v for k, v in attrs.items() if v is not None}
    return kept or None


def _nodes_of(graph: Graph) -> list[tuple[int, Optional[str], dict]]:
    """`(id, label, attributes)` for every node, in ID order."""
    out = []
    for nid in sorted(graph.get_node_ids(limit=None)):
        node = graph.get_node(nid)
        out.append((nid, node.label, dict(node.attributes or {})))
    return out


def _edges_of(graph: Graph) -> list[tuple[int, int, Optional[float], dict]]:
    """`(source_id, target_id, weight, attributes)` for every edge, in ID order."""
    out = []
    for eid, _, _ in sorted(graph._graph.edges()):
        edge = graph.get_edge(eid)
        out.append((edge.source, edge.target, edge.weight, dict(edge.attributes or {})))
    return out


def _node_keys(nodes: list[tuple[int, Optional[str], dict]]) -> dict[int, str]:
    """The id each node gets in a file: its label, or (if unlabeled) a string of its ID."""
    taken = {label for _, label, _ in nodes if label is not None}
    keys: dict[int, str] = {}
    for nid, label, _ in nodes:
        if label is None:
            label = str(nid)
            while label in taken:
                label = f"n{label}"
            taken.add(label)
        keys[nid] = label
    return keys


def _has_parallel_edges(edges: list, directed: bool) -> bool:
    seen = set()
    for source, target, _, _ in edges:
        pair = (source, target) if directed else (min(source, target), max(source, target))
        if pair in seen:
            return True
        seen.add(pair)
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _xy(value: Any) -> Optional[tuple[float, float]]:
    """An `(x, y)` position from a 2- or 3-item sequence of finite numbers, else `None`."""
    if isinstance(value, (list, tuple)) and len(value) in (2, 3) and all(map(_is_number, value)):
        x, y = float(value[0]), float(value[1])
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return None


def _value_type(values: list) -> str:
    """The typed-attribute type for a set of values: boolean, long, double, string or json."""
    if all(isinstance(v, bool) for v in values):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "long"
    if all(_is_number(v) for v in values):
        return "double"
    if all(isinstance(v, str) for v in values):
        return "string"
    return "json"


def _encode_value(value: Any, kind: str) -> str:
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "long":
        return str(value)
    if kind == "double":
        return repr(float(value))
    if kind == "string":
        return value
    return json.dumps(value, default=str)


def _decode_value(text: str, kind: str, is_json: bool = False) -> Any:
    """Decode a typed attribute value (GraphML `attr.type` or GEXF `type`)."""
    if is_json:
        try:
            return json.loads(text)
        except ValueError:
            return text
    kind = kind.lower()
    try:
        if kind == "boolean":
            lowered = text.strip().lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            return text
        if kind in ("int", "long", "integer", "short", "byte"):
            try:
                return int(text)
            except ValueError:
                return float(text)
        if kind in ("float", "double"):
            return float(text)
    except ValueError:
        return text
    return text


def _typed_columns(rows: list[dict]) -> dict[str, str]:
    """Attribute name -> value type, over every non-`None` value in `rows` (names sorted)."""
    values: dict[str, list] = {}
    for row in rows:
        for name, value in row.items():
            if value is not None:
                values.setdefault(name, []).append(value)
    return {name: _value_type(values[name]) for name in sorted(values)}


def _local(tag: str) -> str:
    """An XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    """The `{namespace}` prefix of an XML tag ('' if it has none)."""
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


def _write_xml(root: "ET.Element", filepath: Union[str, Path]) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(Path(filepath), encoding="utf-8", xml_declaration=True)


def _load_json(source: Union[dict, list, str, bytes, Path]) -> Any:
    """JSON data from a dict/list, a JSON string or bytes, or a path to a JSON file."""
    if isinstance(source, (dict, list)):
        return source
    if isinstance(source, bytes):
        return json.loads(source)
    if isinstance(source, str) and source.lstrip()[:1] in ("{", "["):
        return json.loads(source)
    with open(Path(source), encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: dict, filepath: Union[str, Path]) -> None:
    with open(Path(filepath), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


# =============================================================================
# GraphML I/O
# =============================================================================


def from_graphml(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a GraphML file.

    GraphML is an XML-based format used by Cytoscape, Gephi, yEd, igraph, NetworkX and many
    bioinformatics tools. Every `<data>` value is read with the type its `<key>` declares (boolean,
    int, long, float, double or string), and a key's `<default>` fills in for elements without a
    value. Node and edge attributes all come through, `type` and `rel` included.

    A node's label comes from a node key named `label` if there is one, and otherwise from the
    node's `id`. An edge key named `weight` sets edge weights. Only the first `<graph>` element is
    read; nodes and edges in nested graphs are included in it.

    Parameters
    ----------
    filepath
        Path to the GraphML (.graphml) file.
    directed
        Whether to create a directed graph. If `None`, detected from the `edgedefault` attribute of
        the `<graph>` element.
    weighted
        Whether the graph has weights. If `None`, detected from the presence of a `weight` key for
        edges.
    graph_name
        Name for the graph. If `None`, uses the graph's `name` data (as NetworkX writes it), or
        else the `id` attribute of the `<graph>` element.

    Returns
    -------
    Graph
        A graph constructed from the GraphML file, built as a single step.

    Examples
    --------
    Attributes keep their types through a round trip:

    ```python
    import tempfile
    from pathlib import Path

    import edgecase as ec

    graph = ec.karate_club().cmty_louvain(attr_name="community")
    path = Path(tempfile.mkdtemp()) / "karate.graphml"
    ec.to_graphml(graph, path)

    copy = ec.from_graphml(path)

    # Returns (34, 78)
    copy.count_nodes(), copy.count_edges()

    # Returns an int
    copy.get_node("0").attributes["community"]
    ```

    """
    root = ET.parse(Path(filepath)).getroot()
    ns = _namespace(root.tag)

    keys: dict[str, dict] = {}
    for key in root.iter(f"{ns}key"):
        default = key.find(f"{ns}default")
        keys[key.get("id", "")] = {
            "for": key.get("for", "all"),
            "name": key.get("attr.name") or key.get("id", ""),
            "type": key.get("attr.type", "string"),
            "json": key.get(_EC_JSON) == "true",
            "default": None if default is None else default.text,
        }

    graph_elem = root.find(f".//{ns}graph")
    if graph_elem is None:
        raise ValueError("No <graph> element found in GraphML file")

    if directed is None:
        directed = graph_elem.get("edgedefault", "undirected") == "directed"
    if weighted is None:
        weighted = any(k["name"] == "weight" and k["for"] in ("edge", "all") for k in keys.values())

    defaults = {
        domain: {
            k["name"]: _decode_value(k["default"], k["type"], k["json"])
            for k in keys.values()
            if k["default"] is not None and k["for"] in (domain, "all")
        }
        for domain in ("graph", "node", "edge")
    }

    def read_data(elem: "ET.Element", domain: str) -> dict:
        values = dict(defaults[domain])
        for data in elem.findall(f"{ns}data"):
            if len(data):
                # Structured content (e.g. yEd's graphics elements) has no plain value
                continue
            key = keys.get(data.get("key", ""))
            text = data.text or ""
            if key is None:
                values[data.get("key", "")] = text
            elif text.strip() or key["type"] == "string":
                values[key["name"]] = _decode_value(text, key["type"], key["json"])
        return values

    if graph_name is None:
        name = read_data(graph_elem, "graph").get("name")
        graph_name = name if isinstance(name, str) else graph_elem.get("id")

    nodes = []
    for node in graph_elem.iter(f"{ns}node"):
        values = read_data(node, "node")
        label = values.pop("label", None)
        nodes.append((node.get("id", ""), label, values))

    edges = []
    for edge in graph_elem.iter(f"{ns}edge"):
        values = read_data(edge, "edge")
        weight = None
        if _is_number(values.get("weight")):
            weight = float(values.pop("weight"))
        edges.append((edge.get("source", ""), edge.get("target", ""), weight, values))

    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_graphml", filepath)
    )


def to_graphml(
    graph: Graph,
    filepath: Union[str, Path],
) -> None:
    """
    Write a graph to a GraphML file.

    Each node is written with its label as its `id` (unlabeled nodes get their numeric ID), and
    every node and edge attribute is written as typed `<data>`: booleans as `boolean`, integers as
    `long`, other numbers as `double` and strings as `string`. Lists, dicts and attributes with
    mixed types are stored as JSON strings, marked so that `from_graphml()` decodes them again.
    Edge weights go in a `weight` key. The file can be read by NetworkX, igraph, Gephi, Cytoscape
    and yEd.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    """
    nodes = _nodes_of(graph)
    edges = _edges_of(graph)
    node_keys = _node_keys(nodes)
    edge_rows = [
        {**attrs, "weight": weight} if weight is not None else attrs
        for _, _, weight, attrs in edges
    ]

    root = ET.Element(
        "graphml",
        {
            "xmlns": _GRAPHML_NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{_GRAPHML_NS} {_GRAPHML_NS}/1.0/graphml.xsd",
        },
    )

    key_ids: dict[str, dict[str, tuple[str, str]]] = {}
    count = 0
    graph_name = graph.get_graph_name()
    domains = (
        ("graph", [{"name": graph_name}] if graph_name else []),
        ("node", [attrs for _, _, attrs in nodes]),
        ("edge", edge_rows),
    )
    for domain, rows in domains:
        key_ids[domain] = {}
        for name, kind in _typed_columns(rows).items():
            attrib = {
                "id": f"d{count}",
                "for": domain,
                "attr.name": name,
                "attr.type": "string" if kind == "json" else kind,
            }
            if kind == "json":
                attrib["edgecase:json"] = "true"
                root.set("xmlns:edgecase", _EC_NS)
            ET.SubElement(root, "key", attrib)
            key_ids[domain][name] = (f"d{count}", kind)
            count += 1

    def add_data(elem: "ET.Element", values: dict, domain: str) -> None:
        for name, value in values.items():
            if value is not None:
                key_id, kind = key_ids[domain][name]
                ET.SubElement(elem, "data", key=key_id).text = _encode_value(value, kind)

    graph_attrib = {"edgedefault": "directed" if graph.is_directed() else "undirected"}
    if graph_name:
        graph_attrib["id"] = graph_name
    graph_elem = ET.SubElement(root, "graph", graph_attrib)
    if graph_name:
        add_data(graph_elem, {"name": graph_name}, "graph")

    for nid, _, attrs in nodes:
        add_data(ET.SubElement(graph_elem, "node", id=node_keys[nid]), attrs, "node")
    for (source, target, _, _), values in zip(edges, edge_rows):
        elem = ET.SubElement(graph_elem, "edge", source=node_keys[source], target=node_keys[target])
        add_data(elem, values, "edge")

    _write_xml(root, filepath)


# =============================================================================
# GML I/O
# =============================================================================

_GML_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")
_GML_INT = re.compile(r"[+-]?\d+$")
_GML_FLOAT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|[+-]?(INF|NAN)$", re.IGNORECASE)


def from_gml(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a GML (Graph Modelling Language) file.

    GML is a hierarchical ASCII-based format common in network science tools. Every key in a
    `node` or `edge` block other than `id`, `label`, `source` and `target` becomes an attribute:
    numbers stay numbers, HTML entities in strings (`&quot;`, `&#233;`) are decoded, a key repeated
    within a block becomes a list, and a nested block (such as `graphics [ ... ]`) becomes a dict.

    A node's label comes from its `label` key, and otherwise from its `id`. An edge's weight comes
    from its `weight` key or, when no edge in the file has one, from `value`.

    Parameters
    ----------
    filepath
        Path to the GML file.
    directed
        Whether to create a directed graph. If `None`, detected from the `directed` field in the
        `graph` block (0 = undirected, 1 = directed).
    weighted
        Whether the graph has weights. If `None`, detected from the presence of `weight` (or
        `value`) fields on edges.
    graph_name
        Name for the graph. If `None`, uses the `name` (or `label`) field of the `graph` block.

    Returns
    -------
    Graph
        A graph constructed from the GML file, built as a single step.
    """
    with open(Path(filepath), encoding="utf-8") as f:
        content = f.read()

    parsed = _gml_parse(_gml_tokenize(content))

    graph_block = next(
        (val for key, val in parsed if key == "graph" and isinstance(val, list)), None
    )
    if graph_block is None:
        raise ValueError("No 'graph' block found in GML file")

    props = _gml_dict([(k, v) for k, v in graph_block if k not in ("node", "edge")])
    if directed is None:
        directed = props.get("directed") == 1
    if graph_name is None:
        name = props.get("name", props.get("label"))
        graph_name = name if isinstance(name, str) else None

    nodes = []
    for key, block in graph_block:
        if key == "node" and isinstance(block, list):
            values = _gml_dict(block)
            if "id" not in values:
                raise ValueError("A GML node block has no 'id'")
            node_id = values.pop("id")
            label = values.pop("label", None)
            nodes.append((node_id, None if label is None else str(label), values))

    edge_blocks = [block for key, block in graph_block if key == "edge" and isinstance(block, list)]
    weight_key = "weight" if any(k == "weight" for b in edge_blocks for k, _ in b) else "value"
    edges = []
    for block in edge_blocks:
        values = _gml_dict(block)
        if "source" not in values or "target" not in values:
            raise ValueError("A GML edge block needs both a 'source' and a 'target'")
        source, target = values.pop("source"), values.pop("target")
        weight = None
        if _is_number(values.get(weight_key)):
            weight = float(values.pop(weight_key))
        edges.append((source, target, weight, values))

    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)

    return _build_graph(nodes, edges, directed, weighted, graph_name, _code("from_gml", filepath))


def _gml_tokenize(text: str) -> list[str]:
    """Tokenize GML text into a flat list of tokens."""
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == "#":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "[]":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            tokens.append(text[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n[]"#':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _gml_parse(tokens: list[str]) -> list:
    """Parse tokenized GML into nested `(key, value)` lists, with values decoded."""
    pos = 0

    def block(depth: int) -> list:
        nonlocal pos
        items: list = []
        while pos < len(tokens):
            key = tokens[pos]
            if key == "]":
                if depth == 0:
                    raise ValueError("Unexpected ']' in GML file")
                pos += 1
                return items
            if pos + 1 >= len(tokens) or tokens[pos + 1] == "]":
                raise ValueError(f"GML key {key!r} has no value")
            value = tokens[pos + 1]
            pos += 2
            items.append((key, block(depth + 1) if value == "[" else _gml_value(value)))
        if depth:
            raise ValueError("Unclosed '[' in GML file")
        return items

    return block(0)


def _gml_value(token: str) -> Any:
    if token.startswith('"'):
        return html.unescape(token[1:-1])
    if _GML_INT.match(token):
        return int(token)
    if _GML_FLOAT.match(token):
        return float(token)
    return token


def _gml_dict(items: list) -> dict:
    """A block's `(key, value)` pairs as a dict: repeated keys gather into lists."""
    out: dict = {}
    repeated: set = set()
    for key, value in items:
        if isinstance(value, list):
            value = _gml_dict(value)
        if key in repeated:
            out[key].append(value)
        elif key in out:
            out[key] = [out[key], value]
            repeated.add(key)
        else:
            out[key] = value
    return out


def _gml_str(text: str) -> str:
    """A quoted GML string; `&`, `"` and non-ASCII characters become HTML entities."""
    text = text.replace("&", "&amp;").replace('"', "&quot;")
    return '"' + "".join(c if 32 <= ord(c) < 127 else f"&#{ord(c)};" for c in text) + '"'


def _gml_lines(key: str, value: Any, indent: str, reserved: tuple = ()) -> list[str]:
    """GML lines for one attribute (lists become repeated keys, dicts nested blocks)."""
    if value is None:
        return []
    if not _GML_KEY.match(key) or key in reserved:
        raise ValueError(
            f"Attribute name {key!r} can't be written to GML: names must start with a letter and "
            f"hold only letters, digits and underscores, and {', '.join(reserved)} are reserved"
            if reserved
            else f"Attribute name {key!r} can't be written to GML: names must start with a letter "
            "and hold only letters, digits and underscores"
        )
    if isinstance(value, bool):
        return [f"{indent}{key} {int(value)}"]
    if isinstance(value, int):
        return [f"{indent}{key} {value}"]
    if isinstance(value, float):
        if math.isnan(value):
            return [f"{indent}{key} NAN"]
        if math.isinf(value):
            return [f"{indent}{key} {'-' if value < 0 else ''}INF"]
        return [f"{indent}{key} {value!r}"]
    if isinstance(value, dict):
        lines = [f"{indent}{key} ["]
        for k, v in value.items():
            lines.extend(_gml_lines(str(k), v, indent + "  "))
        return lines + [f"{indent}]"]
    if isinstance(value, (list, tuple)):
        return [line for item in value for line in _gml_lines(key, item, indent)]
    return [f"{indent}{key} {_gml_str(str(value))}"]


def to_gml(
    graph: Graph,
    filepath: Union[str, Path],
) -> None:
    """
    Write a graph to a GML file.

    Nodes are written with their numeric IDs as `id` and their labels as `label`; edges with
    their weights as `weight`. Every attribute is written too: numbers and strings as they are,
    booleans as `1` and `0` (GML has no boolean type), lists as repeated keys and dicts as nested
    blocks. A graph with parallel edges is marked `multigraph 1` so NetworkX accepts it.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.

    Raises
    ------
    ValueError
        If an attribute name isn't a valid GML key (a letter followed by letters, digits and
        underscores), or a node attribute is named `id` or `label`, or an edge attribute is named
        `source` or `target`.
    """
    nodes = _nodes_of(graph)
    edges = _edges_of(graph)

    lines = ["graph ["]
    lines.append(f"  directed {1 if graph.is_directed() else 0}")
    if _has_parallel_edges(edges, graph.is_directed()):
        lines.append("  multigraph 1")
    if graph.get_graph_name():
        lines.append(f"  name {_gml_str(graph.get_graph_name())}")

    for nid, label, attrs in nodes:
        lines.append("  node [")
        lines.append(f"    id {nid}")
        if label is not None:
            lines.append(f"    label {_gml_str(label)}")
        for key, value in attrs.items():
            lines.extend(_gml_lines(key, value, "    ", reserved=("id", "label")))
        lines.append("  ]")

    for source, target, weight, attrs in edges:
        lines.append("  edge [")
        lines.append(f"    source {source}")
        lines.append(f"    target {target}")
        values = {**attrs, "weight": weight} if weight is not None else attrs
        for key, value in values.items():
            lines.extend(_gml_lines(key, value, "    ", reserved=("source", "target")))
        lines.append("  ]")

    lines.append("]")

    with open(Path(filepath), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


# =============================================================================
# SIF I/O
# =============================================================================


def from_sif(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a Simple Interaction Format (SIF) file.

    SIF is Cytoscape's native text format. Each line has the form::

        nodeA <interaction_type> nodeB [nodeC ...]

    A node with no interactions can appear alone on a line. Fields are separated by tabs or, in a
    file without tabs, by spaces. Each edge keeps its interaction type in its `rel` attribute, so
    `to_sif()` writes it back.

    Parameters
    ----------
    filepath
        Path to the SIF file.
    directed
        Whether to create a directed graph. Defaults to `False`.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph constructed from the SIF file, built as a single step.
    """
    if directed is None:
        directed = False

    labels: dict[str, None] = {}
    edges = []
    with open(Path(filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()

            source = parts[0].strip()
            labels.setdefault(source)
            if len(parts) < 3:
                continue
            rel = parts[1].strip()
            for target in parts[2:]:
                target = target.strip()
                if target:
                    labels.setdefault(target)
                    edges.append((source, target, None, {"rel": rel}))

    nodes = [(label, label, {}) for label in labels]
    return _build_graph(nodes, edges, directed, False, graph_name, _code("from_sif", filepath))


def to_sif(
    graph: Graph,
    filepath: Union[str, Path],
    interaction_type: str = "interacts_with",
) -> None:
    """
    Write a graph to a Simple Interaction Format (SIF) file.

    Each edge's interaction type is its `rel` attribute, or `interaction_type` for edges without
    one. Edges sharing a source and an interaction type are written on one line, and nodes with no
    edges on lines of their own. SIF has no place for weights or other attributes.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    interaction_type
        The interaction type for edges with no `rel` attribute. Defaults to `"interacts_with"`.
    """
    nodes = _nodes_of(graph)
    keys = _node_keys(nodes)

    groups: dict[tuple[str, str], list[str]] = {}
    connected: set[int] = set()
    for source, target, _, attrs in _edges_of(graph):
        rel = attrs.get("rel")
        rel = interaction_type if rel is None else str(rel)
        groups.setdefault((keys[source], rel), []).append(keys[target])
        connected.update((source, target))

    with open(Path(filepath), "w", encoding="utf-8") as f:
        for (source, rel), targets in groups.items():
            f.write("\t".join([source, rel, *targets]) + "\n")
        for nid, _, _ in nodes:
            if nid not in connected:
                f.write(f"{keys[nid]}\n")


# =============================================================================
# TSV Edge List I/O
# =============================================================================


def from_tsv_edge_list(
    filepath: Union[str, Path],
    source_col: Union[str, int] = 0,
    target_col: Union[str, int] = 1,
    weight_col: Union[str, int, None] = None,
    directed: bool = False,
    graph_name: Optional[str] = None,
    delimiter: str = "\t",
    comment: str = "#",
    skip_rows: int = 0,
    has_header: bool = False,
) -> Graph:
    """
    Create a graph from a tab-separated (or custom-delimited) edge list file.

    Common export format from BioGRID, STRING, IntAct, and other databases. Lines with too few
    columns are skipped, and a weight that isn't a number leaves its edge without a weight.

    Parameters
    ----------
    filepath
        Path to the edge list file.
    source_col
        Column index (0-based int) or header name for source nodes.
    target_col
        Column index (0-based int) or header name for target nodes.
    weight_col
        Column index or header name for edge weights. If `None`, no weights.
    directed
        Whether to create a directed graph.
    graph_name
        Name for the graph.
    delimiter
        Column delimiter. Defaults to tab.
    comment
        Lines starting with this character are skipped.
    skip_rows
        Number of initial lines to skip (before header detection).
    has_header
        If `True`, the first non-skipped, non-comment line is treated as a header row and column
        names can be used for `source_col`, etc.

    Returns
    -------
    Graph
        A graph constructed from the edge list file, built as a single step.
    """
    columns = {"source_col": source_col, "target_col": target_col, "weight_col": weight_col}
    named = [name for name, col in columns.items() if isinstance(col, str)]
    if named and not has_header:
        raise ValueError(f"{named[0]}={columns[named[0]]!r} names a column, so has_header=True")

    def column_index(name: str, header: list[str]) -> Optional[int]:
        col = columns[name]
        if not isinstance(col, str):
            return col
        if col not in header:
            raise ValueError(f"{name}={col!r} isn't in the header row: {header}")
        return header.index(col)

    indices: Optional[tuple] = None if has_header else (source_col, target_col, weight_col)
    labels: dict[str, None] = {}
    edges = []
    with open(Path(filepath), encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= skip_rows:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(comment):
                continue

            # only the line break is trimmed, so a trailing empty field (a missing weight) stays
            cols = line.rstrip("\r\n").split(delimiter)
            if indices is None:
                header = [c.strip() for c in cols]
                indices = tuple(column_index(name, header) for name in columns)
                continue

            src_idx, tgt_idx, wgt_idx = indices
            needed = max(src_idx, tgt_idx, -1 if wgt_idx is None else wgt_idx)
            if len(cols) <= needed:
                continue
            source = cols[src_idx].strip()
            target = cols[tgt_idx].strip()
            if not source or not target:
                continue

            weight = None
            if wgt_idx is not None:
                try:
                    weight = float(cols[wgt_idx].strip())
                except ValueError:
                    pass

            labels.setdefault(source)
            labels.setdefault(target)
            edges.append((source, target, weight, {}))

    nodes = [(label, label, {}) for label in labels]
    weighted = weight_col is not None
    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_tsv_edge_list", filepath)
    )


def to_tsv_edge_list(
    graph: Graph,
    filepath: Union[str, Path],
    delimiter: str = "\t",
    header: bool = True,
    weight: bool = True,
) -> None:
    """
    Write a graph to a tab-separated edge list file.

    Each edge is written as its source and target labels, followed by its weight in a weighted
    graph (left empty for an edge without one). Nodes with no edges and attributes aren't written.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    delimiter
        Column delimiter. Defaults to tab.
    header
        Whether to write a header row.
    weight
        Whether to include a weight column (only if the graph is weighted).
    """
    keys = _node_keys(_nodes_of(graph))
    include_weight = weight and graph.is_weighted()

    with open(Path(filepath), "w", encoding="utf-8") as f:
        if header:
            cols = ["source", "target"]
            if include_weight:
                cols.append("weight")
            f.write(delimiter.join(cols) + "\n")

        for source, target, edge_weight, _ in _edges_of(graph):
            parts = [keys[source], keys[target]]
            if include_weight:
                parts.append("" if edge_weight is None else repr(float(edge_weight)))
            f.write(delimiter.join(parts) + "\n")


# =============================================================================
# GEXF I/O
# =============================================================================


def from_gexf(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a GEXF file.

    GEXF is the XML format of Gephi. Versions 1.1draft, 1.2draft and 1.3 are read. Node and edge
    attributes declared in `<attributes>` blocks are read with their declared types (and their
    defaults); an edge's `label` and `kind` become attributes. Gephi's visual data is read too: a
    `viz:position` becomes a `pos` attribute (so `plot(layout="pos")` reproduces the Gephi
    layout), a `viz:color` becomes a hex `color` and a `viz:size` a `size`, unless the node
    already has an attribute of that name. Time information in dynamic graphs is ignored.

    A node's label comes from its `label`. Gephi doesn't require labels to be unique; when they
    aren't, the node ids become the labels and the file's labels are kept in a `name` attribute.

    Parameters
    ----------
    filepath
        Path to the GEXF (.gexf) file.
    directed
        Whether to create a directed graph. If `None`, the graph is directed when its
        `defaultedgetype` is `directed` or any edge has `type="directed"`.
    weighted
        Whether the graph has weights. If `None`, detected from the presence of `weight` on edges.
    graph_name
        Name for the graph. If `None`, uses the `name` attribute of the `<graph>` element.

    Returns
    -------
    Graph
        A graph constructed from the GEXF file, built as a single step.

    Examples
    --------
    A layout computed in EdgeCase travels to Gephi as node positions, and back again:

    ```python
    import tempfile
    from pathlib import Path

    import edgecase as ec

    graph = ec.karate_club().layout_spring(seed=1)
    path = Path(tempfile.mkdtemp()) / "karate.gexf"
    ec.to_gexf(graph, path)

    copy = ec.from_gexf(path)
    copy.plot(layout="pos")
    ```

    """
    root = ET.parse(Path(filepath)).getroot()
    if _local(root.tag) != "gexf":
        raise ValueError(f"Not a GEXF file: the root element is <{_local(root.tag)}>")
    ns = _namespace(root.tag)

    graph_elem = root.find(f"{ns}graph")
    if graph_elem is None:
        raise ValueError("No <graph> element found in GEXF file")

    declared: dict[str, dict[str, tuple]] = {"node": {}, "edge": {}}
    for block in graph_elem.findall(f"{ns}attributes"):
        cls = block.get("class", "node")
        if cls not in declared:
            continue
        for attribute in block.findall(f"{ns}attribute"):
            default = attribute.find(f"{ns}default")
            aid = attribute.get("id", "")
            declared[cls][aid] = (
                attribute.get("title") or aid,
                attribute.get("type", "string"),
                attribute.get(_EC_JSON) == "true",
                None if default is None else default.text,
            )

    def read_attvalues(elem: "ET.Element", cls: str) -> dict:
        values = {
            title: _decode_value(default, kind, is_json)
            for title, kind, is_json, default in declared[cls].values()
            if default is not None
        }
        container = elem.find(f"{ns}attvalues")
        if container is not None:
            for attvalue in container.findall(f"{ns}attvalue"):
                aid = attvalue.get("for", attvalue.get("id", ""))
                title, kind, is_json, _ = declared[cls].get(aid, (aid, "string", False, None))
                values[title] = _decode_value(attvalue.get("value", ""), kind, is_json)
        return values

    def read_viz(elem: "ET.Element", values: dict) -> None:
        for child in elem:
            name = _local(child.tag)
            try:
                if name == "position" and "pos" not in values:
                    values["pos"] = [float(child.get("x", 0)), float(child.get("y", 0))]
                elif name == "color" and "color" not in values:
                    values["color"] = _gexf_hex(child)
                elif name in ("size", "thickness") and name not in values:
                    values[name] = float(child.get("value", 1))
                elif name == "shape" and "shape" not in values:
                    values["shape"] = child.get("value")
            except ValueError:
                continue

    nodes = []
    for node in graph_elem.iter(f"{ns}node"):
        values = read_attvalues(node, "node")
        read_viz(node, values)
        nodes.append((node.get("id", ""), node.get("label"), values))

    edge_directed = False
    edges = []
    for edge in graph_elem.iter(f"{ns}edge"):
        values = read_attvalues(edge, "edge")
        for name in ("label", "kind"):
            if edge.get(name) is not None:
                values.setdefault(name, edge.get(name))
        read_viz(edge, values)
        edge_directed = edge_directed or edge.get("type") == "directed"
        weight = edge.get("weight")
        edges.append(
            (
                edge.get("source", ""),
                edge.get("target", ""),
                None if weight is None else float(weight),
                values,
            )
        )

    if directed is None:
        directed = graph_elem.get("defaultedgetype", "undirected") == "directed" or edge_directed
    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)
    if graph_name is None:
        graph_name = graph_elem.get("name")

    return _build_graph(nodes, edges, directed, weighted, graph_name, _code("from_gexf", filepath))


def _gexf_hex(elem: "ET.Element") -> str:
    """A `#rrggbb` (or `#rrggbbaa`) color from a `viz:color` element."""
    if elem.get("hex"):
        text = elem.get("hex", "").lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", text):
            raise ValueError(text)
        rgb = "#" + text.lower()
    else:
        rgb = "#" + "".join(f"{int(elem.get(c, 0)):02x}" for c in "rgb")
    alpha = float(elem.get("a", 1))
    return rgb if alpha >= 1 else rgb + f"{round(alpha * 255):02x}"


def to_gexf(
    graph: Graph,
    filepath: Union[str, Path],
    version: str = "1.2draft",
) -> None:
    """
    Write a graph to a GEXF file, for Gephi.

    Nodes are written with their numeric IDs as `id` and their labels as `label`, edges with their
    weights as `weight`. Every node and edge attribute is declared and written as a typed
    attribute value (lists, dicts and mixed types as JSON strings, which `from_gexf()` decodes).
    A node's `pos` attribute (stored by the `layout_*()` methods) is also written as a
    `viz:position`, so Gephi opens the graph in that layout; a hex `color` attribute is written as
    a `viz:color` and a numeric `size` as a `viz:size`.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    version
        The GEXF version: `"1.2draft"` (the default, which NetworkX can also read) or `"1.3"`.
    """
    if version not in _GEXF_VERSIONS:
        raise ValueError(f"version must be one of {sorted(_GEXF_VERSIONS)}, not {version!r}")
    ns, viz_ns, number = _GEXF_VERSIONS[version]

    nodes = _nodes_of(graph)
    edges = _edges_of(graph)

    root = ET.Element("gexf", {"xmlns": ns, "xmlns:viz": viz_ns, "version": number})
    ET.SubElement(ET.SubElement(root, "meta"), "creator").text = "EdgeCase"

    graph_attrib = {
        "defaultedgetype": "directed" if graph.is_directed() else "undirected",
        "mode": "static",
    }
    if graph.get_graph_name():
        graph_attrib["name"] = graph.get_graph_name()
    graph_elem = ET.SubElement(root, "graph", graph_attrib)

    declared: dict[str, dict[str, tuple[str, str]]] = {}
    for cls, rows in (("node", [a for _, _, a in nodes]), ("edge", [a for *_, a in edges])):
        columns = _typed_columns(rows)
        declared[cls] = {}
        if not columns:
            continue
        block = ET.SubElement(graph_elem, "attributes", {"class": cls, "mode": "static"})
        for i, (title, kind) in enumerate(columns.items()):
            attrib = {"id": str(i), "title": title, "type": "string" if kind == "json" else kind}
            if kind == "json":
                attrib["edgecase:json"] = "true"
                root.set("xmlns:edgecase", _EC_NS)
            ET.SubElement(block, "attribute", attrib)
            declared[cls][title] = (str(i), kind)

    def add_attvalues(elem: "ET.Element", values: dict, cls: str) -> None:
        present = [(k, v) for k, v in values.items() if v is not None]
        if present:
            container = ET.SubElement(elem, "attvalues")
            for title, value in present:
                aid, kind = declared[cls][title]
                ET.SubElement(
                    container, "attvalue", {"for": aid, "value": _encode_value(value, kind)}
                )

    nodes_elem = ET.SubElement(graph_elem, "nodes")
    for nid, label, attrs in nodes:
        attrib = {"id": str(nid)}
        if label is not None:
            attrib["label"] = label
        elem = ET.SubElement(nodes_elem, "node", attrib)
        add_attvalues(elem, attrs, "node")
        color = attrs.get("color")
        if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", color):
            rgb = {c: str(int(color[i : i + 2], 16)) for c, i in zip("rgb", (1, 3, 5))}
            if len(color) == 9:
                rgb["a"] = repr(round(int(color[7:9], 16) / 255, 3))
            ET.SubElement(elem, "viz:color", rgb)
        xy = _xy(attrs.get("pos"))
        if xy is not None:
            ET.SubElement(elem, "viz:position", x=repr(xy[0]), y=repr(xy[1]), z="0.0")
        if _is_number(attrs.get("size")) and math.isfinite(attrs["size"]):
            ET.SubElement(elem, "viz:size", value=repr(float(attrs["size"])))

    edges_elem = ET.SubElement(graph_elem, "edges")
    for i, (source, target, weight, attrs) in enumerate(edges):
        attrib = {"id": str(i), "source": str(source), "target": str(target)}
        if weight is not None:
            attrib["weight"] = repr(float(weight))
        add_attvalues(ET.SubElement(edges_elem, "edge", attrib), attrs, "edge")

    _write_xml(root, filepath)


# =============================================================================
# Node-link JSON and Cytoscape.js JSON
# =============================================================================


def from_node_link(
    source: Union[dict, str, bytes, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from node-link JSON data.

    Node-link is the JSON layout used by D3.js and by NetworkX's `node_link_data()`: a `nodes`
    list of objects with an `id`, an `edges` (or `links`) list of objects with a `source` and a
    `target`, and top-level `directed` and `graph` fields. Each node's `id` becomes its label and
    every other field becomes an attribute; an edge's numeric `weight` sets its weight, its `key`
    (NetworkX's multigraph edge key) is dropped, and every other field becomes an attribute.

    Parameters
    ----------
    source
        The data: a dict, a JSON string, or a path to a JSON file.
    directed
        Whether to create a directed graph. If `None`, read from the `directed` field.
    weighted
        Whether the graph has weights. If `None`, detected from numeric `weight` fields on edges.
    graph_name
        Name for the graph. If `None`, uses the `name` in the `graph` field.

    Returns
    -------
    Graph
        A graph constructed from the data, built as a single step.

    Examples
    --------
    ```python
    import edgecase as ec

    data = {
        "directed": True,
        "nodes": [{"id": "a", "type": "gene"}, {"id": "b", "type": "protein"}],
        "edges": [{"source": "a", "target": "b", "rel": "encodes"}],
    }
    graph = ec.from_node_link(data)

    # Returns 'encodes'
    graph.get_edge_df()["rel"][0]
    ```

    """
    data = _load_json(source)
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise ValueError("Node-link data must be an object with a 'nodes' list")
    edge_list = data.get("edges", data.get("links", []))

    if directed is None:
        directed = bool(data.get("directed", False))
    if graph_name is None and isinstance(data.get("graph"), dict):
        name = data["graph"].get("name")
        graph_name = name if isinstance(name, str) else None

    nodes = []
    for node in data["nodes"]:
        values = dict(node)
        if "id" not in values:
            raise ValueError("Every node in node-link data needs an 'id'")
        key = _hashable(values.pop("id"))
        nodes.append((key, str(key), values))

    edges = []
    for edge in edge_list:
        values = dict(edge)
        if "source" not in values or "target" not in values:
            raise ValueError("Every edge in node-link data needs a 'source' and a 'target'")
        src, tgt = _hashable(values.pop("source")), _hashable(values.pop("target"))
        values.pop("key", None)
        weight = float(values.pop("weight")) if _is_number(values.get("weight")) else None
        edges.append((src, tgt, weight, values))

    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)

    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_node_link", source)
    )


def _hashable(value: Any) -> Any:
    """JSON node ids can be lists (NetworkX writes tuple nodes that way); make them tuples."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        raise ValueError(f"A node id can't be an object: {value!r}")
    return value


def to_node_link(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
    edges: str = "edges",
) -> dict:
    """
    Convert a graph to node-link JSON data.

    The result has the layout D3.js and NetworkX's `node_link_graph()` read: `directed`,
    `multigraph` (true when the graph has parallel edges), `graph` (holding the graph's `name`),
    `nodes` (each node's label as `id`, plus its attributes) and `edges` (`source`, `target`,
    `weight` and the edge's attributes).

    Parameters
    ----------
    graph
        The graph to convert.
    filepath
        If given, the data is also written to this file as JSON.
    edges
        The name of the edge list: `"edges"` (the default, as in NetworkX 3.6 and later) or
        `"links"` (as in D3.js and earlier NetworkX).

    Returns
    -------
    dict
        The node-link data.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_path_graph(3)
    data = ec.to_node_link(graph)

    # Returns [{'source': '1', 'target': '2'}, {'source': '2', 'target': '3'}]
    data["edges"]
    ```

    """
    if edges not in ("edges", "links"):
        raise ValueError(f"edges must be 'edges' or 'links', not {edges!r}")
    node_rows = _nodes_of(graph)
    edge_rows = _edges_of(graph)
    keys = _node_keys(node_rows)

    data: dict = {
        "directed": graph.is_directed(),
        "multigraph": _has_parallel_edges(edge_rows, graph.is_directed()),
        "graph": {"name": graph.get_graph_name()} if graph.get_graph_name() else {},
        "nodes": [
            {"id": keys[nid], **{k: v for k, v in attrs.items() if k != "id"}}
            for nid, _, attrs in node_rows
        ],
        edges: [],
    }
    for source, target, weight, attrs in edge_rows:
        record: dict = {"source": keys[source], "target": keys[target]}
        if weight is not None:
            record["weight"] = weight
        reserved = ("source", "target", "weight") if weight is not None else ("source", "target")
        record.update((k, v) for k, v in attrs.items() if k not in reserved)
        data[edges].append(record)

    if filepath is not None:
        _write_json(data, filepath)
    return data


def from_cytoscape(
    source: Union[dict, list, str, bytes, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from Cytoscape.js JSON data.

    This is the JSON that Cytoscape.js uses for its `elements` (and that the Cytoscape desktop
    app exports as `.cyjs`, and NetworkX's `cytoscape_data()` writes). The elements can be an
    object of `nodes` and `edges` lists or one flat list (with a `group` field on each element).
    Each node's `name` becomes its label (its `id` if it has no name), and the other `data` fields
    become attributes. A node's `position` becomes a `pos` attribute, with y flipped so it points
    up as in EdgeCase's layouts. An edge's numeric `weight` sets its weight; its `id` (and
    NetworkX's `key`) are dropped.

    Cytoscape doesn't require node names to be unique. When they aren't, the node ids become the
    labels and the names are kept in a `name` attribute.

    Parameters
    ----------
    source
        The data: a dict, a JSON string, or a path to a JSON file.
    directed
        Whether to create a directed graph. If `None`, read from the top-level `directed` field
        (undirected if there's none).
    weighted
        Whether the graph has weights. If `None`, detected from numeric `weight` fields on edges.
    graph_name
        Name for the graph. If `None`, uses the `name` in the top-level `data` field.

    Returns
    -------
    Graph
        A graph constructed from the data, built as a single step.
    """
    data = _load_json(source)
    elements = data.get("elements", []) if isinstance(data, dict) else data
    if isinstance(elements, dict):
        node_elements = elements.get("nodes", [])
        edge_elements = elements.get("edges", [])
    elif isinstance(elements, list):
        node_elements, edge_elements = [], []
        for element in elements:
            group = element.get("group")
            is_edge = group == "edges" or (group is None and "source" in element.get("data", {}))
            (edge_elements if is_edge else node_elements).append(element)
    else:
        raise ValueError("Cytoscape data must have an 'elements' object or list")

    info = data.get("data", {}) if isinstance(data, dict) else {}
    if isinstance(info, list):
        # NetworkX writes the graph's attributes as a list of pairs
        info = dict(info)
    if directed is None:
        directed = bool(data.get("directed", False)) if isinstance(data, dict) else False
    if graph_name is None:
        name = info.get("name")
        graph_name = name if isinstance(name, str) else None

    nodes = []
    for element in node_elements:
        values = dict(element.get("data", {}))
        if "id" not in values:
            raise ValueError("Every node in Cytoscape data needs a 'data' object with an 'id'")
        key = str(values.pop("id"))
        name = values.pop("name", None)
        if "value" in values and str(values["value"]) in (key, str(name)):
            # NetworkX's copy of the node's own key
            del values["value"]
        position = element.get("position")
        if "pos" not in values and isinstance(position, dict):
            xy = _xy([position.get("x"), position.get("y")])
            if xy is not None:
                values["pos"] = [xy[0], -xy[1]]
        nodes.append((key, None if name is None else str(name), values))

    edges = []
    for element in edge_elements:
        values = dict(element.get("data", {}))
        if "source" not in values or "target" not in values:
            raise ValueError("Every edge in Cytoscape data needs a 'source' and a 'target'")
        src, tgt = str(values.pop("source")), str(values.pop("target"))
        values.pop("id", None)
        values.pop("key", None)
        weight = float(values.pop("weight")) if _is_number(values.get("weight")) else None
        edges.append((src, tgt, weight, values))

    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)

    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_cytoscape", source)
    )


def to_cytoscape(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
) -> dict:
    """
    Convert a graph to Cytoscape.js JSON data.

    The result can be passed to Cytoscape.js as `elements` (or opened in the Cytoscape desktop
    app, or read by NetworkX's `cytoscape_graph()`). Each node's label is its `id` and `name`, and
    its attributes join them in `data`; a node's `pos` attribute becomes its `position` (with y
    flipped, since y points down on screen). Each edge gets an `id`, its `source` and `target`,
    its `weight`, and its attributes (plus a `key` numbering parallel edges, if there are any, as
    NetworkX expects). Attributes named like those fields are left out.

    Parameters
    ----------
    graph
        The graph to convert.
    filepath
        If given, the data is also written to this file as JSON.

    Returns
    -------
    dict
        The Cytoscape.js data: `data` (the graph's name), `directed`, `multigraph` and `elements`.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_path_graph(3).layout_circular()
    data = ec.to_cytoscape(graph)

    # Returns the first node, with its position
    data["elements"]["nodes"][0]
    ```

    """
    node_rows = _nodes_of(graph)
    edge_rows = _edges_of(graph)
    keys = _node_keys(node_rows)

    nodes = []
    for nid, _, attrs in node_rows:
        key = keys[nid]
        xy = _xy(attrs.get("pos"))
        values = {
            k: v
            for k, v in attrs.items()
            if k not in ("id", "name", "value") and not (k == "pos" and xy is not None)
        }
        element: dict = {"data": {"id": key, "value": key, "name": key, **values}}
        if xy is not None:
            element["position"] = {"x": xy[0], "y": -xy[1]}
        nodes.append(element)

    # Element ids are shared between nodes and edges in Cytoscape.js, so keep them apart
    taken = set(keys.values())
    directed = graph.is_directed()
    multigraph = _has_parallel_edges(edge_rows, directed)
    pair_counts: dict[tuple[int, int], int] = {}
    edges = []
    for i, (source, target, weight, attrs) in enumerate(edge_rows):
        edge_id = f"e{i}"
        while edge_id in taken:
            edge_id = f"e{edge_id}"
        record: dict = {"id": edge_id, "source": keys[source], "target": keys[target]}
        if multigraph:
            # NetworkX's `cytoscape_graph()` tells parallel edges apart by `key`
            pair = (source, target) if directed else (min(source, target), max(source, target))
            record["key"] = pair_counts.get(pair, 0)
            pair_counts[pair] = record["key"] + 1
        if weight is not None:
            record["weight"] = weight
        record.update((k, v) for k, v in attrs.items() if k not in record)
        edges.append({"data": record})

    data = {
        "data": {"name": graph.get_graph_name()} if graph.get_graph_name() else {},
        "directed": directed,
        "multigraph": multigraph,
        "elements": {"nodes": nodes, "edges": edges},
    }
    if filepath is not None:
        _write_json(data, filepath)
    return data


# =============================================================================
# Pajek I/O
# =============================================================================

_PAJEK_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


def _pajek_tokens(line: str) -> list[str]:
    return [quoted if bare == "" else bare for quoted, bare in _PAJEK_TOKEN.findall(line)]


def _pajek_number(token: str) -> Union[int, float, None]:
    try:
        return int(token)
    except ValueError:
        pass
    try:
        number = float(token)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _pajek_pairs(tokens: list[str]) -> dict:
    """`key value` pairs, with values read as int, then float, then string."""
    values = {}
    for key, text in zip(tokens[::2], tokens[1::2]):
        number = _pajek_number(text)
        values[key] = text if number is None else number
    return values


def from_pajek(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a Pajek (.net) file.

    Pajek's format lists vertices under `*Vertices n` (each as `id "label"`, optionally followed by
    `x y` or `x y z` coordinates and `key value` parameters) and edges under `*Edges` (undirected)
    or `*Arcs` (directed), as `source target [weight]` lines with optional `key value` parameters.
    `*Edgeslist` and `*Arcslist` lines (a vertex followed by all its neighbors) are read too.
    Coordinates become a `pos` attribute and parameters become attributes. A two-mode network
    (`*Vertices n n1`) gets a `part` attribute: 0 for the first `n1` vertices and 1 for the rest.
    A relation label in a section header (`*Arcs :2 "advice"`) becomes the `rel` attribute of the
    section's edges.

    Vertices that are counted but not listed are labeled with their numbers. Only the first
    network in a file is read, and `*Matrix` sections aren't supported.

    Parameters
    ----------
    filepath
        Path to the Pajek file.
    directed
        Whether to create a directed graph. If `None`, the graph is directed when the file has an
        `*Arcs` or `*Arcslist` section. In a directed graph, each `*Edges` line becomes a pair of
        opposite arcs.
    weighted
        Whether the graph has weights. If `None`, detected from the presence of weights on edges.
    graph_name
        Name for the graph. If `None`, taken from the `*Network` line.

    Returns
    -------
    Graph
        A graph constructed from the Pajek file, built as a single step.
    """
    with open(Path(filepath), encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    count: Optional[int] = None
    first_mode: Optional[int] = None
    vertices: dict[int, tuple[Optional[str], dict]] = {}
    rows: list[tuple[int, int, Optional[float], dict, bool]] = []
    name: Optional[str] = None
    section: Optional[str] = None
    relation: dict = {}
    seen_network = False

    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("*"):
            head, _, rest = line[1:].partition(" ")
            head = head.lower()
            section = None
            if head == "network":
                if seen_network:
                    break
                seen_network = True
                name = rest.strip() or None
            elif head == "vertices":
                parts = rest.split()
                try:
                    count = int(parts[0])
                    first_mode = int(parts[1]) if len(parts) > 1 else None
                except (IndexError, ValueError):
                    raise ValueError(f"Line {number}: bad *Vertices line {raw!r}") from None
                section = "vertices"
            elif head in ("arcs", "edges", "arcslist", "edgeslist"):
                section = head
                relation = {}
                tokens = _pajek_tokens(rest)
                if tokens and tokens[0].startswith(":"):
                    label = tokens[1] if len(tokens) > 1 else tokens[0][1:]
                    relation = {"rel": label}
            elif head == "matrix":
                raise ValueError("Pajek *Matrix sections aren't supported")
            continue
        if section is None:
            continue

        tokens = _pajek_tokens(line)
        try:
            if section == "vertices":
                vid = int(tokens[0])
                label = tokens[1] if len(tokens) > 1 else None
                rest = tokens[2:]
                coords = []
                while rest and len(coords) < 3 and _pajek_number(rest[0]) is not None:
                    coords.append(float(rest.pop(0)))
                values = {}
                if len(coords) >= 2:
                    values["pos"] = coords[:2]
                if len(rest) % 2:
                    values["shape"] = rest.pop(0)
                values.update(_pajek_pairs(rest))
                vertices[vid] = (label, values)
            elif section in ("arcs", "edges"):
                source, target = int(tokens[0]), int(tokens[1])
                rest = tokens[2:]
                weight = None
                if rest and _pajek_number(rest[0]) is not None:
                    weight = float(rest.pop(0))
                values = {**relation, **_pajek_pairs(rest)}
                rows.append((source, target, weight, values, section == "arcs"))
            else:
                source = int(tokens[0])
                for target in tokens[1:]:
                    rows.append((source, int(target), None, dict(relation), section == "arcslist"))
        except (IndexError, ValueError):
            raise ValueError(
                f"Line {number}: can't read {raw!r} in the *{section} section"
            ) from None

    if directed is None:
        directed = any(is_arc for *_, is_arc in rows)
    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _, _ in rows)
    if graph_name is None:
        graph_name = name

    ids = sorted(set(range(1, (count or 0) + 1)) | set(vertices))
    nodes = []
    for vid in ids:
        label, values = vertices.get(vid, (None, {}))
        if first_mode is not None:
            values.setdefault("part", 0 if vid <= first_mode else 1)
        nodes.append((vid, label, values))

    edges = []
    for source, target, weight, values, is_arc in rows:
        edges.append((source, target, weight, values))
        if directed and not is_arc and source != target:
            edges.append((target, source, weight, dict(values)))

    return _build_graph(nodes, edges, directed, weighted, graph_name, _code("from_pajek", filepath))


def _pajek_quote(text: str, always: bool = False) -> str:
    """A Pajek token; quoted when needed (Pajek has no escapes, so `"` becomes `'`)."""
    text = " ".join(text.replace('"', "'").split()) if always or text.strip() != text else text
    if always or not text or any(c.isspace() or c == '"' for c in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def to_pajek(
    graph: Graph,
    filepath: Union[str, Path],
) -> None:
    """
    Write a graph to a Pajek (.net) file.

    Vertices are numbered 1 to n in node ID order and written with their labels (quoted, with any
    `"` replaced by `'`, since Pajek has no escapes). A node's `pos` attribute is written as its
    coordinates, and its other attributes as `key value` parameters. Edges are written under
    `*Arcs` (directed graphs) or `*Edges`, with their weights and attributes. Pajek parameters
    hold single values, so attributes holding lists, dicts or `None` are left out, with a warning;
    booleans are written as `1` and `0`.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    """
    nodes = _nodes_of(graph)
    edges = _edges_of(graph)
    index = {nid: i for i, (nid, _, _) in enumerate(nodes, 1)}
    skipped: set[str] = set()

    def params(values: dict) -> list[str]:
        out = []
        for key, value in values.items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                text = repr(value) if isinstance(value, float) else str(value)
            elif isinstance(value, str):
                text = value
            else:
                skipped.add(key)
                continue
            out += [_pajek_quote(key), _pajek_quote(text)]
        return out

    lines = []
    if graph.get_graph_name():
        lines.append(f"*Network {' '.join(graph.get_graph_name().split())}")
    lines.append(f"*Vertices {len(nodes)}")
    for nid, label, attrs in nodes:
        parts = [str(index[nid]), _pajek_quote(str(nid) if label is None else label, always=True)]
        xy = _xy(attrs.get("pos"))
        if xy is not None:
            parts += [repr(xy[0]), repr(xy[1])]
            attrs = {k: v for k, v in attrs.items() if k != "pos"}
        lines.append(" ".join(parts + params(attrs)))

    lines.append("*Arcs" if graph.is_directed() else "*Edges")
    for source, target, weight, attrs in edges:
        parts = [str(index[source]), str(index[target])]
        if weight is not None:
            parts.append(repr(float(weight)))
        lines.append(" ".join(parts + params(attrs)))

    if skipped:
        warnings.warn(
            "Pajek parameters hold single values, so these attributes weren't written: "
            + ", ".join(sorted(skipped)),
            stacklevel=2,
        )

    with open(Path(filepath), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


# =============================================================================
# graph6 / sparse6
# =============================================================================

_PLUS_63 = bytes((i + 63) & 0xFF for i in range(256))


def _n_to_data(n: int) -> list[int]:
    """The graph6/sparse6 encoding of a node count, as 6-bit values."""
    if n <= 62:
        return [n]
    if n <= 258047:
        return [63, (n >> 12) & 0x3F, (n >> 6) & 0x3F, n & 0x3F]
    if n <= 68719476735:
        return [63, 63] + [(n >> shift) & 0x3F for shift in (30, 24, 18, 12, 6, 0)]
    raise ValueError(f"graph6 and sparse6 hold at most 68719476735 nodes, not {n}")


def _data_to_n(data: bytes) -> tuple[int, bytes]:
    """Split 6-bit values into the node count and the rest."""
    if not data:
        raise ValueError("graph6/sparse6 data is empty")
    if data[0] <= 62:
        return data[0], data[1:]
    if len(data) >= 4 and data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    if len(data) >= 8:
        return sum(data[2 + i] << (30 - 6 * i) for i in range(6)), data[8:]
    raise ValueError("graph6/sparse6 data ends inside the node count")


def _undirected_index(graph: Graph, fmt: str) -> tuple[int, list[tuple[int, int]]]:
    """The node count and `(i, j)` edge positions (0-based, in node ID order)."""
    if graph.is_directed():
        raise ValueError(f"{fmt} holds undirected graphs only; this graph is directed")
    ids = sorted(graph.get_node_ids(limit=None))
    index = {nid: i for i, nid in enumerate(ids)}
    pairs = [(index[s], index[t]) for _, s, t in sorted(graph._graph.edges())]
    return len(ids), pairs


def _six_bit_lines(source: Union[str, bytes, Path], header: bytes) -> list[bytes]:
    """The non-empty lines of graph6/sparse6 input as 6-bit values, headers removed."""
    if isinstance(source, Path) or (isinstance(source, str) and os.path.isfile(source)):
        raw = Path(source).read_bytes()
    elif isinstance(source, str):
        raw = source.encode("ascii", errors="replace")
    else:
        raw = bytes(source)
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith(header):
            line = line[len(header) :]
        if not line:
            continue
        if any(c < 63 or c > 126 for c in line.lstrip(b":;&")):
            raise ValueError(f"Not graph6/sparse6 data: {line[:40]!r}")
        lines.append(line)
    if not lines:
        raise ValueError("No graph6/sparse6 data found")
    return lines


def _numbered_graph(n: int, edges: list[tuple[int, int]], code: str) -> Graph:
    """An undirected graph with nodes labeled '1' to 'n' (edges given 0-based)."""
    nodes = [(i, str(i + 1), {}) for i in range(n)]
    return _build_graph(nodes, [(i, j, None, {}) for i, j in edges], False, False, None, code)


def to_graph6(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
    header: bool = False,
) -> str:
    """
    Encode a graph in graph6 format.

    graph6 is the compact text format of nauty, the House of Graphs and many graph enumeration
    tools: one line of printable ASCII per graph, holding the node count and the upper triangle of
    the adjacency matrix. It stores structure only: node order (by ID) is kept, but labels,
    weights and attributes aren't.

    Parameters
    ----------
    graph
        The graph to encode. It must be undirected, with no self-loops or parallel edges (use
        `to_sparse6()` for those).
    filepath
        If given, the encoding is also written to this file (as one line).
    header
        Whether to start the encoding with the `>>graph6<<` header.

    Returns
    -------
    str
        The graph6 encoding, without a trailing newline.

    Examples
    --------
    ```python
    import edgecase as ec

    # Returns 'Dhc'
    ec.to_graph6(ec.Graph().add_cycle_graph(5))
    ```

    """
    n, pairs = _undirected_index(graph, "graph6")
    bits = bytearray((n * (n - 1) // 2 + 5) // 6)
    for i, j in pairs:
        if i == j:
            raise ValueError("graph6 can't hold self-loops; use to_sparse6() instead")
        if i > j:
            i, j = j, i
        p = j * (j - 1) // 2 + i
        mask = 32 >> (p % 6)
        if bits[p // 6] & mask:
            raise ValueError("graph6 can't hold parallel edges; use to_sparse6() instead")
        bits[p // 6] |= mask
    text = (bytes(_n_to_data(n)) + bytes(bits)).translate(_PLUS_63).decode("ascii")
    if header:
        text = ">>graph6<<" + text
    if filepath is not None:
        with open(Path(filepath), "w", encoding="ascii") as f:
            f.write(text + "\n")
    return text


def from_graph6(source: Union[str, bytes, Path]) -> Union[Graph, list[Graph]]:
    """
    Create a graph from graph6 data.

    The nodes are labeled `"1"` to `"n"` in the order of the adjacency matrix. A file (or string)
    can hold one graph per line, as graph collections like the House of Graphs do; then a list of
    graphs is returned.

    Parameters
    ----------
    source
        The graph6 data, as a string or bytes, or a path to a file holding it. A string naming an
        existing file is read as a path. A `>>graph6<<` header is allowed.

    Returns
    -------
    Graph or list[Graph]
        The graph, or a list of graphs if the input has more than one line.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.from_graph6("Dhc")

    # Returns True
    graph.is_isomorphic(ec.Graph().add_cycle_graph(5))
    ```

    """
    graphs = []
    for line in _six_bit_lines(source, b">>graph6<<"):
        if line[:1] in (b":", b";", b"&"):
            raise ValueError("This is sparse6 or digraph6 data, not graph6")
        n, data = _data_to_n(bytes(c - 63 for c in line))
        expected = (n * (n - 1) // 2 + 5) // 6
        if len(data) != expected:
            raise ValueError(
                f"graph6 data for {n} nodes needs {expected} adjacency characters, not {len(data)}"
            )
        edges = []
        for byte_index, byte in enumerate(data):
            if not byte:
                continue
            for k in range(6):
                if byte & (32 >> k):
                    p = 6 * byte_index + k
                    j = (1 + math.isqrt(8 * p + 1)) // 2
                    if j < n:
                        edges.append((p - j * (j - 1) // 2, j))
        graphs.append(_numbered_graph(n, edges, _code("from_graph6", source)))
    return graphs[0] if len(graphs) == 1 else graphs


def to_sparse6(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
    header: bool = False,
) -> str:
    """
    Encode a graph in sparse6 format.

    sparse6 is graph6's companion for sparse graphs: it lists edges rather than the whole
    adjacency matrix, so it's much shorter for graphs with few edges, and it can hold self-loops
    and parallel edges. Like graph6 it stores structure only (node order by ID is kept).

    Parameters
    ----------
    graph
        The graph to encode. It must be undirected.
    filepath
        If given, the encoding is also written to this file (as one line).
    header
        Whether to start the encoding with the `>>sparse6<<` header.

    Returns
    -------
    str
        The sparse6 encoding (starting with `:`), without a trailing newline.

    Examples
    --------
    ```python
    import edgecase as ec

    # Returns ':DaY_~'
    ec.to_sparse6(ec.Graph().add_cycle_graph(5))
    ```

    """
    n, pairs = _undirected_index(graph, "sparse6")
    k = max(1, (n - 1).bit_length())

    def enc(x: int) -> list[int]:
        return [(x >> (k - 1 - i)) & 1 for i in range(k)]

    bits: list[int] = []
    current = 0
    for v, u in sorted((max(i, j), min(i, j)) for i, j in pairs):
        if v == current:
            bits.append(0)
            bits.extend(enc(u))
        elif v == current + 1:
            current += 1
            bits.append(1)
            bits.extend(enc(u))
        else:
            current = v
            bits.append(1)
            bits.extend(enc(v))
            bits.append(0)
            bits.extend(enc(u))
    if k < 6 and n == (1 << k) and (-len(bits)) % 6 >= k and current < n - 1:
        # Padding with 1s here would read back as an edge to node n - 1, so pad with a 0 first
        bits.append(0)
    bits.extend([1] * ((-len(bits)) % 6))

    data = [int("".join(map(str, bits[i : i + 6])), 2) for i in range(0, len(bits), 6)]
    text = ":" + (bytes(_n_to_data(n)) + bytes(data)).translate(_PLUS_63).decode("ascii")
    if header:
        text = ">>sparse6<<" + text
    if filepath is not None:
        with open(Path(filepath), "w", encoding="ascii") as f:
            f.write(text + "\n")
    return text


def from_sparse6(source: Union[str, bytes, Path]) -> Union[Graph, list[Graph]]:
    """
    Create a graph from sparse6 data.

    The nodes are labeled `"1"` to `"n"`. Self-loops and parallel edges are kept. A file (or
    string) can hold one graph per line; then a list of graphs is returned. Incremental sparse6
    (lines starting with `;`) isn't supported.

    Parameters
    ----------
    source
        The sparse6 data, as a string or bytes, or a path to a file holding it. A string naming an
        existing file is read as a path. A `>>sparse6<<` header is allowed.

    Returns
    -------
    Graph or list[Graph]
        The graph, or a list of graphs if the input has more than one line.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.from_sparse6(":DaY_~")

    # Returns (5, 5)
    graph.count_nodes(), graph.count_edges()
    ```

    """
    graphs = []
    for line in _six_bit_lines(source, b">>sparse6<<"):
        if line[:1] == b";":
            raise ValueError("Incremental sparse6 (starting with ';') isn't supported")
        if line[:1] != b":":
            raise ValueError("sparse6 data starts with ':'")
        n, data = _data_to_n(bytes(c - 63 for c in line[1:]))
        k = max(1, (n - 1).bit_length())

        edges = []
        v = 0
        for b, x in _sparse6_pairs(data, k):
            if b:
                v += 1
            if x >= n or v >= n:
                # The 1s padding the last character can read as an oversized number
                break
            if x > v:
                v = x
            else:
                edges.append((x, v))
        graphs.append(_numbered_graph(n, edges, _code("from_sparse6", source)))
    return graphs[0] if len(graphs) == 1 else graphs


def _sparse6_pairs(data: bytes, k: int):
    """The `(b, x)` pairs of sparse6 edge data: a 1-bit flag and a k-bit number."""
    chunks = iter(data)
    d = 0
    d_len = 0
    while True:
        if d_len < 1:
            try:
                d = next(chunks)
            except StopIteration:
                return
            d_len = 6
        d_len -= 1
        b = (d >> d_len) & 1
        x = d & ((1 << d_len) - 1)
        x_len = d_len
        while x_len < k:
            try:
                d = next(chunks)
            except StopIteration:
                return
            d_len = 6
            x = (x << 6) + d
            x_len += 6
        x >>= x_len - k
        d_len = x_len - k
        yield b, x


# =============================================================================
# NumPy, SciPy sparse and pandas
# =============================================================================


def _require(module: str) -> Any:
    """Import an optional dependency, with an install hint if it's missing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as err:
        package = module.split(".", 1)[0]
        raise ImportError(
            f"This function needs {package}, which isn't installed (`pip install {package}`)"
        ) from err


def _node_order(graph: Graph, nodelist: Optional[Sequence]) -> list[int]:
    """Node IDs in matrix order: `nodelist` (IDs or labels) or else ID order."""
    if nodelist is None:
        return sorted(graph.get_node_ids(limit=None))
    ids = [graph._node_id_or_raise(node) for node in nodelist]
    if len(set(ids)) < len(ids):
        raise ValueError("nodelist names a node more than once")
    return ids


def _node_names(graph: Graph, ids: list[int]) -> list[str]:
    """The name each node gets in an export (its label, or its ID if unlabeled), for `ids`."""
    keys = _node_keys(
        [(nid, graph.get_label_from_id(nid), {}) for nid in graph.get_node_ids(limit=None)]
    )
    return [keys[nid] for nid in ids]


def _matrix_entries(
    graph: Graph,
    ids: list[int],
    weighted: Optional[bool],
    multigraph_weight: str,
) -> tuple[Any, Any, Any]:
    """
    The `(rows, cols, values)` arrays of a graph's adjacency matrix, one entry per nonzero cell.

    An edge counts its weight (1 if it has none or weights aren't used). Parallel edges combine
    by `multigraph_weight` ("sum", "min" or "max"). An undirected edge fills both of its cells, a
    self-loop only its diagonal cell. Edges to nodes outside `ids` are left out.
    """
    np = _require("numpy")
    reduce = {"sum": np.add, "min": np.minimum, "max": np.maximum}.get(multigraph_weight)
    if reduce is None:
        raise ValueError(
            f"multigraph_weight must be 'sum', 'min' or 'max', not {multigraph_weight!r}"
        )
    index = {nid: i for i, nid in enumerate(ids)}
    use_weights = graph._use_weights(weighted)
    directed = graph.is_directed()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for _, source, target, weight in graph._graph._weighted_edges():
        i, j = index.get(source), index.get(target)
        if i is None or j is None:
            continue
        value = weight if use_weights and weight is not None else 1.0
        rows.append(i)
        cols.append(j)
        vals.append(value)
        if not directed and i != j:
            rows.append(j)
            cols.append(i)
            vals.append(value)
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    v = np.asarray(vals, dtype=float)
    if len(r):
        order = np.lexsort((c, r))
        r, c, v = r[order], c[order], v[order]
        starts = np.flatnonzero(np.r_[True, (r[1:] != r[:-1]) | (c[1:] != c[:-1])])
        r, c, v = r[starts], c[starts], reduce.reduceat(v, starts)
    return r, c, v


def _from_entries(
    n: int,
    rows: Any,
    cols: Any,
    vals: Any,
    directed: bool,
    weighted: Optional[bool],
    parallel_edges: bool,
    node_labels: Optional[Sequence],
    graph_name: Optional[str],
    code: str,
) -> Graph:
    """
    Build a graph from the edge cells of a square matrix (already checked for symmetry when
    undirected, and with each undirected edge's cell given only once).
    """
    np = _require("numpy")
    if len(vals) and not np.all(np.isfinite(vals)):
        raise ValueError(
            "The matrix has non-finite entries; pass the value it uses for non-edges as nonedge="
        )
    if node_labels is None:
        labels = [str(i) for i in range(n)]
    else:
        labels = [str(label) for label in node_labels]
        if len(labels) != n:
            raise ValueError(f"node_labels has {len(labels)} labels for {n} nodes")
        if len(set(labels)) < n:
            raise ValueError("node_labels holds the same label more than once")

    if parallel_edges:
        if len(vals) and not (np.all(vals >= 0) and np.all(vals == np.floor(vals))):
            raise ValueError("With parallel_edges=True, each entry must be a whole number of edges")
        weighted = bool(weighted)
        edges = [
            (i, j, 1.0, {})
            for i, j, count in zip(rows.tolist(), cols.tolist(), vals.tolist())
            for _ in range(int(count))
        ]
    else:
        if weighted is None:
            weighted = bool(np.any(vals != 1))
        edges = [
            (i, j, value, {})
            for i, j, value in zip(rows.tolist(), cols.tolist(), vals.astype(float).tolist())
        ]
    nodes = [(i, label, {}) for i, label in enumerate(labels)]
    return _build_graph(nodes, edges, directed, weighted, graph_name, code)


def _from_dense(
    array: Any,
    directed: bool,
    weighted: Optional[bool],
    parallel_edges: bool,
    node_labels: Optional[Sequence],
    nonedge: float,
    graph_name: Optional[str],
    code: str,
) -> Graph:
    np = _require("numpy")
    matrix = np.asarray(array)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"The matrix must be square, not of shape {matrix.shape}")
    if not (np.issubdtype(matrix.dtype, np.number) or matrix.dtype == bool) or np.iscomplexobj(
        matrix
    ):
        raise TypeError(f"The matrix must hold real numbers, not {matrix.dtype}")
    matrix = matrix.astype(float)
    edge = ~np.isnan(matrix) if math.isnan(nonedge) else matrix != nonedge
    if not directed:
        if not (np.array_equal(edge, edge.T) and np.array_equal(matrix[edge], matrix.T[edge])):
            raise ValueError(
                "The matrix isn't symmetric, so it can't describe an undirected graph; "
                "pass directed=True"
            )
        edge = np.triu(edge)
    rows, cols = np.nonzero(edge)
    return _from_entries(
        matrix.shape[0],
        rows,
        cols,
        matrix[rows, cols],
        directed,
        weighted,
        parallel_edges,
        node_labels,
        graph_name,
        code,
    )


def to_numpy_array(
    graph: Graph,
    nodelist: Optional[Sequence[Union[int, str]]] = None,
    weighted: Optional[bool] = None,
    nonedge: float = 0.0,
    multigraph_weight: str = "sum",
    dtype: Any = None,
) -> Any:
    """
    Convert a graph to a NumPy adjacency matrix.

    Entry `[i, j]` holds the weight of the edge from node `i` to node `j` (1 for an edge with no
    weight), or `nonedge=` when there's no such edge. An undirected graph gives a symmetric
    matrix, and a self-loop puts its weight on the diagonal once. Parallel edges are combined:
    by default their weights are summed, so in an unweighted graph the entry counts the edges.
    The result matches NetworkX's `to_numpy_array()`.

    Parameters
    ----------
    graph
        The graph to convert.
    nodelist
        The nodes (IDs or labels) for the rows and columns, in order. Edges to nodes outside it
        are left out. If `None`, every node is used, in ID order.
    weighted
        Whether entries hold edge weights (`True`) or count edges (`False`). If `None`, follows
        the graph's `weighted` setting.
    nonedge
        The value for node pairs with no edge between them, such as `0.0` (the default) or
        `float("inf")` for distance-style matrices.
    multigraph_weight
        How parallel edges combine: `"sum"` (the default), `"min"` or `"max"`.
    dtype
        The NumPy dtype of the result (float by default).

    Returns
    -------
    numpy.ndarray
        The adjacency matrix. Use `get_node_ids()` or `nodelist=` to know which node each row is.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_path_graph(3)

    # Returns array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    ec.to_numpy_array(graph)
    ```

    """
    np = _require("numpy")
    ids = _node_order(graph, nodelist)
    rows, cols, vals = _matrix_entries(graph, ids, weighted, multigraph_weight)
    matrix = np.full((len(ids), len(ids)), nonedge, dtype=float)
    matrix[rows, cols] = vals
    return matrix if dtype is None else matrix.astype(dtype)


def from_numpy_array(
    array: Any,
    directed: bool = False,
    weighted: Optional[bool] = None,
    parallel_edges: bool = False,
    node_labels: Optional[Sequence[Union[int, str]]] = None,
    nonedge: float = 0.0,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a NumPy adjacency matrix.

    Each entry that isn't `nonedge=` (0 by default) becomes an edge from the row's node to the
    column's node, weighted by the entry. For an undirected graph the matrix must be symmetric,
    and each edge is read once from the upper triangle (with the diagonal holding self-loops).

    Parameters
    ----------
    array
        A square array (or anything `numpy.asarray()` accepts) of real numbers.
    directed
        Whether to create a directed graph.
    weighted
        Whether the edges get the entries as weights. If `None`, they do when any edge's entry
        isn't 1.
    parallel_edges
        If `True`, each entry is a count of parallel edges (it must be a whole number), so an
        entry of 3 gives three edges. This is how an unweighted multigraph's matrix reads back.
    node_labels
        Labels for the nodes, in row order. If `None`, the nodes are labeled `"0"`, `"1"`, ...
    nonedge
        The entry value that means "no edge" (it can be `float("nan")` or `float("inf")`).
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph with one node per row, built as a single step.

    Examples
    --------
    ```python
    import numpy as np
    import edgecase as ec

    array = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    graph = ec.from_numpy_array(array, node_labels=["a", "b", "c"])

    # Returns ['a--b', 'b--c']
    graph.get_edges_str_repr(use_labels=True)

    # Returns [2.0, 1.0]
    graph.get_edge_df()["weight"].to_list()
    ```

    """
    return _from_dense(
        array,
        directed,
        weighted,
        parallel_edges,
        node_labels,
        nonedge,
        graph_name,
        _code("from_numpy_array", array),
    )


def to_scipy_sparse_array(
    graph: Graph,
    nodelist: Optional[Sequence[Union[int, str]]] = None,
    weighted: Optional[bool] = None,
    multigraph_weight: str = "sum",
    dtype: Any = None,
    format: str = "csr",
) -> Any:
    """
    Convert a graph to a SciPy sparse adjacency matrix.

    The sparse counterpart of `to_numpy_array()`: entry `[i, j]` holds the weight of the edge from
    node `i` to node `j` (1 for an edge with no weight), with parallel edges combined, and only
    node pairs joined by an edge are stored. The result matches NetworkX's
    `to_scipy_sparse_array()`.

    Parameters
    ----------
    graph
        The graph to convert.
    nodelist
        The nodes (IDs or labels) for the rows and columns, in order. Edges to nodes outside it
        are left out. If `None`, every node is used, in ID order.
    weighted
        Whether entries hold edge weights (`True`) or count edges (`False`). If `None`, follows
        the graph's `weighted` setting.
    multigraph_weight
        How parallel edges combine: `"sum"` (the default), `"min"` or `"max"`.
    dtype
        The NumPy dtype of the result (float by default).
    format
        The sparse format: `"csr"` (the default), `"csc"`, `"coo"`, `"lil"`, `"dok"`, `"bsr"` or
        `"dia"`.

    Returns
    -------
    scipy.sparse.sparray
        The sparse adjacency matrix.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_cycle_graph(100)
    matrix = ec.to_scipy_sparse_array(graph)

    # Returns 200
    matrix.nnz
    ```

    """
    sparse = _require("scipy.sparse")
    ids = _node_order(graph, nodelist)
    rows, cols, vals = _matrix_entries(graph, ids, weighted, multigraph_weight)
    if dtype is not None:
        vals = vals.astype(dtype)
    matrix = sparse.coo_array((vals, (rows, cols)), shape=(len(ids), len(ids)))
    return matrix.asformat(format)


def from_scipy_sparse_array(
    matrix: Any,
    directed: bool = False,
    weighted: Optional[bool] = None,
    parallel_edges: bool = False,
    node_labels: Optional[Sequence[Union[int, str]]] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a SciPy sparse adjacency matrix.

    Each stored, nonzero entry becomes an edge from the row's node to the column's node, weighted
    by the entry (explicitly stored zeros aren't edges, and duplicate entries in COO data are
    summed first). For an undirected graph the matrix must be symmetric, and each edge is read
    once from the upper triangle (with the diagonal holding self-loops).

    Parameters
    ----------
    matrix
        A square SciPy sparse array or matrix, in any format.
    directed
        Whether to create a directed graph.
    weighted
        Whether the edges get the entries as weights. If `None`, they do when any edge's entry
        isn't 1.
    parallel_edges
        If `True`, each entry is a count of parallel edges (it must be a whole number).
    node_labels
        Labels for the nodes, in row order. If `None`, the nodes are labeled `"0"`, `"1"`, ...
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph with one node per row, built as a single step.

    Examples
    --------
    ```python
    import scipy.sparse as sp
    import edgecase as ec

    matrix = sp.csr_array(([1.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))
    graph = ec.from_scipy_sparse_array(matrix, directed=True)

    # Returns 2
    graph.count_edges()
    ```

    """
    np = _require("numpy")
    sparse = _require("scipy.sparse")
    if not sparse.issparse(matrix):
        raise TypeError(f"Expected a SciPy sparse array or matrix, not {type(matrix).__name__}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"The matrix must be square, not of shape {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix.data[:0]):
        raise TypeError(f"The matrix must hold real numbers, not {matrix.dtype}")
    coo = sparse.coo_array(matrix, dtype=float)
    coo.sum_duplicates()
    coo.eliminate_zeros()
    rows, cols, vals = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
    if len(vals) and not np.all(np.isfinite(vals)):
        raise ValueError("The matrix has non-finite entries")
    if not directed:
        csr = coo.tocsr()
        if (csr != csr.T).nnz:
            raise ValueError(
                "The matrix isn't symmetric, so it can't describe an undirected graph; "
                "pass directed=True"
            )
        upper = rows <= cols
        rows, cols, vals = rows[upper], cols[upper], vals[upper]
    order = np.lexsort((cols, rows))
    return _from_entries(
        matrix.shape[0],
        rows[order],
        cols[order],
        vals[order],
        directed,
        weighted,
        parallel_edges,
        node_labels,
        graph_name,
        _code("from_scipy_sparse_array", matrix),
    )


def to_pandas_adjacency(
    graph: Graph,
    nodelist: Optional[Sequence[Union[int, str]]] = None,
    weighted: Optional[bool] = None,
    nonedge: float = 0.0,
    multigraph_weight: str = "sum",
    dtype: Any = None,
) -> Any:
    """
    Convert a graph to a pandas adjacency matrix.

    The same matrix as `to_numpy_array()`, as a DataFrame whose index and columns are the node
    labels (an unlabeled node is named by its ID). The result matches NetworkX's
    `to_pandas_adjacency()`.

    Parameters
    ----------
    graph
        The graph to convert.
    nodelist
        The nodes (IDs or labels) for the rows and columns, in order. Edges to nodes outside it
        are left out. If `None`, every node is used, in ID order.
    weighted
        Whether entries hold edge weights (`True`) or count edges (`False`). If `None`, follows
        the graph's `weighted` setting.
    nonedge
        The value for node pairs with no edge between them.
    multigraph_weight
        How parallel edges combine: `"sum"` (the default), `"min"` or `"max"`.
    dtype
        The NumPy dtype of the values (float by default).

    Returns
    -------
    pandas.DataFrame
        The adjacency matrix, labeled by node.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph(directed=True).add_nodes(["a", "b"]).add_edge("a", "b")

    # Returns 1.0
    ec.to_pandas_adjacency(graph).loc["a", "b"]
    ```

    """
    pd = _require("pandas")
    ids = _node_order(graph, nodelist)
    matrix = to_numpy_array(graph, ids, weighted, nonedge, multigraph_weight, dtype)
    names = _node_names(graph, ids)
    return pd.DataFrame(matrix, index=names, columns=names)


def from_pandas_adjacency(
    df: Any,
    directed: bool = False,
    weighted: Optional[bool] = None,
    parallel_edges: bool = False,
    nonedge: float = 0.0,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a pandas adjacency matrix.

    The index and the columns must hold the same node names (in any order; the columns are put
    in index order). Each node is labeled with its name, in index order, and each entry that
    isn't `nonedge=` becomes an edge as in `from_numpy_array()`.

    Parameters
    ----------
    df
        A square DataFrame of real numbers.
    directed
        Whether to create a directed graph. An undirected graph needs a symmetric matrix.
    weighted
        Whether the edges get the entries as weights. If `None`, they do when any edge's entry
        isn't 1.
    parallel_edges
        If `True`, each entry is a count of parallel edges (it must be a whole number).
    nonedge
        The entry value that means "no edge".
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph built as a single step.

    Examples
    --------
    ```python
    import pandas as pd
    import edgecase as ec

    df = pd.DataFrame([[0, 1], [1, 0]], index=["a", "b"], columns=["a", "b"])

    # Returns 1
    ec.from_pandas_adjacency(df).count_edges()
    ```

    """
    names = list(df.index)
    if len(set(names)) < len(names) or set(names) != set(df.columns) or df.shape[1] != len(names):
        raise ValueError("The index and the columns must hold the same node names, once each")
    return _from_dense(
        df.reindex(columns=names).to_numpy(),
        directed,
        weighted,
        parallel_edges,
        names,
        nonedge,
        graph_name,
        _code("from_pandas_adjacency", df),
    )


def to_pandas_edgelist(
    graph: Graph,
    source: str = "source",
    target: str = "target",
) -> Any:
    """
    Convert a graph to a pandas edge list.

    One row per edge, in edge ID order: the endpoints' labels (an unlabeled node is named by its
    ID) in the `source=` and `target=` columns, a `weight` column when any edge has a weight, and
    one column per edge attribute (missing values are `None`/`NaN`). Parallel edges are separate
    rows. `from_pandas_edgelist()` and NetworkX's `from_pandas_edgelist()` read it back.

    Parameters
    ----------
    graph
        The graph to convert.
    source
        The name of the source column.
    target
        The name of the target column.

    Returns
    -------
    pandas.DataFrame
        The edge list.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_path_graph(3)

    # Returns [['1', '2'], ['2', '3']]
    ec.to_pandas_edgelist(graph).values.tolist()
    ```

    """
    pd = _require("pandas")
    if source == target:
        raise ValueError("source and target must be different column names")
    node_rows = [(nid, graph.get_label_from_id(nid), {}) for nid in graph.get_node_ids(limit=None)]
    keys = _node_keys(node_rows)
    edge_rows = _edges_of(graph)
    columns: dict[str, list] = {source: [], target: []}
    has_weight = any(weight is not None for _, _, weight, _ in edge_rows)
    if has_weight:
        columns["weight"] = []
    reserved = set(columns)
    rows = []
    for s, t, weight, attrs in edge_rows:
        row = {source: keys[s], target: keys[t]}
        if has_weight:
            row["weight"] = weight
        for name, value in attrs.items():
            if name not in reserved:
                row[name] = value
                columns.setdefault(name, [])
        rows.append(row)
    for row in rows:
        for name, values in columns.items():
            values.append(row.get(name))
    return pd.DataFrame(columns)


def _missing(value: Any) -> bool:
    """Whether a DataFrame cell is missing: `None`, `NaN` or pandas' `NA`/`NaT`."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return type(value).__name__ in ("NAType", "NaTType")


def from_pandas_edgelist(
    df: Any,
    source: str = "source",
    target: str = "target",
    edge_attr: Union[bool, str, Sequence[str], None] = None,
    directed: bool = False,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a pandas (or Polars) edge list.

    Each row becomes an edge from the node in the `source=` column to the node in the `target=`
    column; the nodes are created as they first appear and labeled with their values (as
    strings). A numeric `weight` value among the attribute columns sets the edge's weight, and
    missing values (`None`, `NaN`, `NA`) leave that attribute unset.

    Parameters
    ----------
    df
        The edge list, as a pandas or Polars DataFrame.
    source
        The name of the source column.
    target
        The name of the target column.
    edge_attr
        Which other columns become edge attributes: `None` (the default) for none, `True` for
        all of them, or a column name or a list of column names. This follows NetworkX's
        `from_pandas_edgelist()`.
    directed
        Whether to create a directed graph.
    weighted
        Whether the graph has weights. If `None`, it does when any edge gets a weight.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph built as a single step.

    Examples
    --------
    ```python
    import pandas as pd
    import edgecase as ec

    df = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "weight": [2.0, 3.5]})
    graph = ec.from_pandas_edgelist(df, edge_attr="weight")

    # Returns [2.0, 3.5]
    graph.get_edge_df()["weight"].to_list()
    ```

    """
    if edge_attr is True:
        attr_cols = [c for c in df.columns if c not in (source, target)]
    elif edge_attr is None or edge_attr is False:
        attr_cols = []
    elif isinstance(edge_attr, (str, int)):
        attr_cols = [edge_attr]
    else:
        attr_cols = list(edge_attr)
    absent = [c for c in (source, target, *attr_cols) if c not in df.columns]
    if absent:
        raise ValueError(f"The DataFrame has no column {absent[0]!r}")

    columns = [df[c].to_list() for c in (source, target, *attr_cols)]
    names = [str(c) for c in attr_cols]
    nodes: list = []
    seen: set = set()
    edges = []
    for row in zip(*columns):
        src, tgt = row[0], row[1]
        for key in (src, tgt):
            if _missing(key):
                raise ValueError("The source and target columns can't have missing values")
            if key not in seen:
                seen.add(key)
                nodes.append((key, None, {}))
        attrs = {name: value for name, value in zip(names, row[2:]) if not _missing(value)}
        weight = float(attrs.pop("weight")) if _is_number(attrs.get("weight")) else None
        edges.append((src, tgt, weight, attrs))
    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)
    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_pandas_edgelist", df)
    )


# =============================================================================
# Adjacency lists
# =============================================================================


def _text_lines(source: Union[str, bytes, Path, Sequence[str]]) -> list[str]:
    """
    The lines of a text format. A `Path`, or a string with no newline, is a file to read; a
    string with a newline (or bytes) is the text itself; a list holds the lines.
    """
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        with open(Path(source), encoding="utf-8") as f:
            return f.read().splitlines()
    if isinstance(source, bytes):
        return source.decode("utf-8").splitlines()
    if isinstance(source, str):
        return source.splitlines()
    return [line.rstrip("\n") for line in source]


def _split(line: str, delimiter: Optional[str], maxsplit: int = -1) -> list[str]:
    """Split a line on `delimiter` (any whitespace if `None`), dropping empty fields."""
    if delimiter is None:
        return line.split(None, maxsplit)
    return [part.strip() for part in line.split(delimiter, maxsplit) if part.strip()]


def _uncomment(line: str, comments: Optional[str]) -> str:
    if comments:
        p = line.find(comments)
        if p >= 0:
            return line[:p]
    return line


def _token_names(graph: Graph, delimiter: str, fmt: str) -> dict[int, str]:
    """Node ID -> name for a format that separates names with `delimiter`, checking each name."""
    keys = _node_keys(
        [(nid, graph.get_label_from_id(nid), {}) for nid in graph.get_node_ids(limit=None)]
    )
    for name in keys.values():
        bad = (
            not name
            or name != name.strip()
            or "#" in name
            or "\n" in name
            or "\r" in name
            or delimiter in name
            or (not delimiter.strip() and any(c.isspace() for c in name))
        )
        if bad:
            raise ValueError(
                f"The node label {name!r} can't be written to {fmt}: labels there can't be "
                "empty or contain the delimiter, whitespace or '#'"
            )
    return keys


def _adjacency_rows(graph: Graph, edges: list) -> dict[int, list]:
    """
    Node ID -> the edges listed on its line (in edge order), NetworkX-style: a directed edge under
    its source, an undirected edge under whichever endpoint comes first in node ID order.
    `edges` holds `(source, target, ...)` tuples; each row holds `(neighbor, *rest)`.
    """
    rows: dict[int, list] = {nid: [] for nid in sorted(graph.get_node_ids(limit=None))}
    directed = graph.is_directed()
    for source, target, *rest in edges:
        if directed or source <= target:
            rows[source].append((target, *rest))
        else:
            rows[target].append((source, *rest))
    return rows


def _write_text(lines: list[str], filepath: Union[str, Path, None]) -> str:
    text = "".join(line + "\n" for line in lines)
    if filepath is not None:
        with open(Path(filepath), "w", encoding="utf-8") as f:
            f.write(text)
    return text


def to_adjlist(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
    delimiter: str = " ",
) -> str:
    """
    Encode a graph as an adjacency list.

    Each line holds a node followed by its neighbors, as in NetworkX's `write_adjlist()`: nodes
    are listed in ID order by label (an unlabeled node is named by its ID), a directed edge is
    listed under its source, and an undirected edge once, under whichever endpoint comes first.
    Parallel edges repeat the neighbor. The format holds structure only: weights and attributes
    aren't written.

    Parameters
    ----------
    graph
        The graph to encode. Its labels can't be empty or contain the delimiter, whitespace or
        `#` (the comment marker).
    filepath
        If given, the adjacency list is also written to this file.
    delimiter
        The separator between names on a line.

    Returns
    -------
    str
        The adjacency list, one line per node.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_cycle_graph(3)

    # Returns '1 2 3\\n2 3\\n3\\n'
    ec.to_adjlist(graph)
    ```

    """
    if not delimiter:
        raise ValueError("delimiter can't be empty")
    keys = _token_names(graph, delimiter, "an adjacency list")
    rows = _adjacency_rows(graph, [(s, t) for _, s, t in sorted(graph._graph.edges())])
    lines = [
        delimiter.join([keys[nid]] + [keys[nbr] for (nbr,) in nbrs]) for nid, nbrs in rows.items()
    ]
    return _write_text(lines, filepath)


def from_adjlist(
    source: Union[str, bytes, Path, Sequence[str]],
    directed: bool = False,
    delimiter: Optional[str] = None,
    comments: Optional[str] = "#",
    parallel_edges: bool = False,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from an adjacency list.

    Each line holds a node followed by its neighbors, as written by `to_adjlist()` or NetworkX's
    `write_adjlist()`. Every name becomes a node, labeled with the name: first the nodes that start
    a line, in line order, then any that appear only as neighbors. Each neighbor gives an edge
    from the line's node.

    Parameters
    ----------
    source
        A path to the file, the text itself (a string with at least one newline), or a list of
        lines.
    directed
        Whether to create a directed graph.
    delimiter
        The separator between names on a line. If `None` (the default), any run of whitespace.
    comments
        The marker that starts a comment (the rest of the line is ignored), or `None` for none.
    parallel_edges
        If `False` (the default), a node pair listed more than once (in either order, for an
        undirected graph) gives one edge, as NetworkX's `read_adjlist()` does. If `True`, every
        listing gives an edge, as `read_adjlist(create_using=nx.MultiGraph)` does.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph built as a single step.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.from_adjlist("a b c\\nb c\\nd\\n")

    # Returns (4, 3)
    (graph.count_nodes(), graph.count_edges())
    ```

    """
    nodes: list = []
    seen_nodes: set = set()
    edges = []
    seen_pairs: set = set()
    for line in _text_lines(source):
        names = _split(_uncomment(line, comments), delimiter)
        if not names:
            continue
        u = names[0]
        if u not in seen_nodes:
            seen_nodes.add(u)
            nodes.append((u, u, {}))
        for v in names[1:]:
            if not parallel_edges:
                pair = (u, v) if directed or u <= v else (v, u)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            edges.append((u, v, None, {}))
    return _build_graph(nodes, edges, directed, False, graph_name, _code("from_adjlist", source))


def _literal_ok(value: Any) -> bool:
    """Whether `repr(value)` reads back with `ast.literal_eval()`."""
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_literal_ok(v) for v in value)
    if isinstance(value, dict):
        return all(_literal_ok(k) and _literal_ok(v) for k, v in value.items())
    return False


def to_multiline_adjlist(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
    delimiter: str = " ",
) -> str:
    """
    Encode a graph as a multiline adjacency list.

    The layout of NetworkX's `write_multiline_adjlist()`: for each node (in ID order), a line with
    its name and neighbor count, then one line per neighbor with the neighbor's name and the
    edge's data as a Python dict (its `weight`, if set, and its attributes). Edges are listed as
    in `to_adjlist()`, so each undirected edge appears once. Unlike `to_adjlist()`, weights and
    edge attributes are kept; node attributes aren't.

    Parameters
    ----------
    graph
        The graph to encode. Its labels can't be empty or contain the delimiter, whitespace or
        `#`, and its edge attributes must be plain values (numbers, strings, booleans, `None`,
        and lists, tuples and dicts of them).
    filepath
        If given, the adjacency list is also written to this file.
    delimiter
        The separator between a name and what follows it.

    Returns
    -------
    str
        The multiline adjacency list.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph(weighted=True).add_nodes(["a", "b"]).add_edge("a", "b", weight=2.0)

    # Returns "a 1\\nb {'weight': 2.0}\\nb 0\\n"
    ec.to_multiline_adjlist(graph)
    ```

    """
    if not delimiter:
        raise ValueError("delimiter can't be empty")
    keys = _token_names(graph, delimiter, "a multiline adjacency list")
    edge_rows = [(s, t, (w, attrs)) for s, t, w, attrs in _edges_of(graph)]
    lines = []
    for nid, nbrs in _adjacency_rows(graph, edge_rows).items():
        lines.append(f"{keys[nid]}{delimiter}{len(nbrs)}")
        for nbr, (weight, attrs) in nbrs:
            data = {} if weight is None else {"weight": weight}
            data.update((k, v) for k, v in attrs.items() if k not in data)
            if not _literal_ok(data):
                raise ValueError(
                    f"The edge data {data!r} can't be written to a multiline adjacency list, "
                    "which holds plain values only (finite numbers, strings, booleans, None, and "
                    "lists and dicts of them)"
                )
            lines.append(f"{keys[nbr]}{delimiter}{data!r}")
    return _write_text(lines, filepath)


def _edge_data(text: str, comments: Optional[str], line_no: int) -> Any:
    """The edge data after a neighbor's name: a Python literal, possibly with a comment after it."""
    cuts = [len(text)]
    if comments:
        cuts += [m.start() for m in re.finditer(re.escape(comments), text)]
    for cut in sorted(set(cuts), reverse=True):
        candidate = text[:cut].strip()
        if not candidate:
            return {}
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError):
            continue
    raise ValueError(f"Line {line_no}: can't read the edge data {text.strip()!r}")


def from_multiline_adjlist(
    source: Union[str, bytes, Path, Sequence[str]],
    directed: bool = False,
    delimiter: Optional[str] = None,
    comments: Optional[str] = "#",
    parallel_edges: bool = False,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a multiline adjacency list.

    Reads the layout of `to_multiline_adjlist()` and NetworkX's `write_multiline_adjlist()`: a
    line with a node's name and neighbor count, then that many lines, each with a neighbor's name
    and (optionally) the edge's data as a Python dict. A numeric `weight` in the data sets the
    edge's weight and the other entries become edge attributes; data that's a bare number is
    taken as the weight. Nodes are created in the order of their count lines, then any that
    appear only as neighbors. Unlike NetworkX's reader, string values containing spaces or `#`
    are kept intact.

    Parameters
    ----------
    source
        A path to the file, the text itself (a string with at least one newline), or a list of
        lines.
    directed
        Whether to create a directed graph.
    delimiter
        The separator after a name. If `None` (the default), any run of whitespace.
    comments
        The marker that starts a comment, or `None` for none.
    parallel_edges
        If `False` (the default), a node pair listed more than once (in either order, for an
        undirected graph) gives one edge, keeping the first listing's data. If `True`, every
        listing gives an edge.
    weighted
        Whether the graph has weights. If `None`, it does when any edge has a numeric `weight`.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph built as a single step.

    Examples
    --------
    ```python
    import edgecase as ec

    text = "a 2\\nb {'weight': 2.0}\\nc {'kind': 'x y'}\\nb 0\\nc 0\\n"
    graph = ec.from_multiline_adjlist(text)

    # Returns 'x y'
    graph.get_edge_df()["kind"][1]
    ```

    """
    lines = _text_lines(source)
    nodes: list = []
    seen_nodes: set = set()
    edges = []
    seen_pairs: set = set()
    position = 0

    def next_line() -> tuple[int, str]:
        nonlocal position
        while position < len(lines):
            position += 1
            line = lines[position - 1].strip()
            if line and not (comments and line.startswith(comments)):
                return position, line
        return position, ""

    while True:
        line_no, line = next_line()
        if not line:
            break
        parts = _split(_uncomment(line, comments), delimiter)
        if not parts:
            continue
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Line {line_no}: expected a node name and its neighbor count")
        u, count = parts[0], int(parts[1])
        if u not in seen_nodes:
            seen_nodes.add(u)
            nodes.append((u, u, {}))
        for _ in range(count):
            line_no, line = next_line()
            if not line:
                raise ValueError(f"The file ends before all of node {u!r}'s neighbors are listed")
            if delimiter is None:
                fields = line.split(None, 1)
                name, rest = fields[0], fields[1] if len(fields) > 1 else ""
            else:
                name, _, rest = line.partition(delimiter)
            name = name.strip()
            if comments and comments in name:
                name, rest = name[: name.find(comments)].strip(), ""
            if not name:
                raise ValueError(f"Line {line_no}: expected a neighbor's name")
            data = _edge_data(rest, comments, line_no)
            weight = None
            if _is_number(data):
                data, weight = {}, float(data)
            elif not isinstance(data, dict):
                raise ValueError(f"Line {line_no}: the edge data must be a dict, not {data!r}")
            else:
                data = dict(data)
                if _is_number(data.get("weight")):
                    weight = float(data.pop("weight"))
            if not parallel_edges:
                pair = (u, name) if directed or u <= name else (name, u)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            edges.append((u, name, weight, {str(k): v for k, v in data.items()}))

    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)
    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_multiline_adjlist", source)
    )


# =============================================================================
# JSON adjacency and tree data
# =============================================================================


def to_adjacency_data(
    graph: Graph,
    filepath: Union[str, Path, None] = None,
) -> dict:
    """
    Convert a graph to adjacency JSON data.

    The layout of NetworkX's `adjacency_data()`: `directed`, `multigraph` (true when the graph has
    parallel edges), `graph` (a list of `[key, value]` pairs, holding the graph's `name`), `nodes`
    (each node's label as `id`, plus its attributes) and `adjacency`, which holds, for each node in
    the same order, a list of its neighbors: each neighbor's `id` with the edge's `weight` and
    attributes, and in a multigraph the edge's `key` (its index among the parallel edges). An
    undirected edge is listed under both of its endpoints, a self-loop once.

    Parameters
    ----------
    graph
        The graph to convert.
    filepath
        If given, the data is also written to this file as JSON.

    Returns
    -------
    dict
        The adjacency data.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph().add_path_graph(3)

    # Returns [[{'id': '2'}], [{'id': '1'}, {'id': '3'}], [{'id': '2'}]]
    ec.to_adjacency_data(graph)["adjacency"]
    ```

    """
    node_rows = _nodes_of(graph)
    edge_rows = _edges_of(graph)
    keys = _node_keys(node_rows)
    directed = graph.is_directed()
    multigraph = _has_parallel_edges(edge_rows, directed)

    # Each node's neighbors in order of first appearance, with each neighbor's edges grouped
    neighbors: dict[int, dict[int, list[dict]]] = {nid: {} for nid, _, _ in node_rows}
    edge_keys: dict[tuple[int, int], int] = {}
    for source, target, weight, attrs in edge_rows:
        pair = (source, target) if directed else (min(source, target), max(source, target))
        key = edge_keys.get(pair, 0)
        edge_keys[pair] = key + 1
        record: dict = {}
        if weight is not None:
            record["weight"] = weight
        record.update((k, v) for k, v in attrs.items() if k not in ("id", "key", *record))
        if multigraph:
            record["key"] = key
        neighbors[source].setdefault(target, []).append(record)
        if not directed and source != target:
            neighbors[target].setdefault(source, []).append(record)

    data = {
        "directed": directed,
        "multigraph": multigraph,
        "graph": [["name", graph.get_graph_name()]] if graph.get_graph_name() else [],
        "nodes": [
            {"id": keys[nid], **{k: v for k, v in attrs.items() if k != "id"}}
            for nid, _, attrs in node_rows
        ],
        "adjacency": [
            [
                {"id": keys[nbr], **record}
                for nbr, records in neighbors[nid].items()
                for record in records
            ]
            for nid, _, _ in node_rows
        ],
    }
    if filepath is not None:
        _write_json(data, filepath)
    return data


def from_adjacency_data(
    source: Union[dict, str, bytes, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from adjacency JSON data.

    Reads the layout of `to_adjacency_data()` and NetworkX's `adjacency_data()`. Each node's `id`
    becomes its label and its other fields become attributes. Each neighbor entry gives an edge:
    its numeric `weight` sets the edge's weight, its `key` is dropped and its other fields become
    edge attributes. In undirected data, where each edge is listed under both endpoints, a node
    pair gets as many edges as the longer of its two listings.

    Parameters
    ----------
    source
        The data: a dict, a JSON string, or a path to a JSON file.
    directed
        Whether to create a directed graph. If `None`, read from the `directed` field.
    weighted
        Whether the graph has weights. If `None`, detected from numeric `weight` fields.
    graph_name
        Name for the graph. If `None`, uses the `name` in the `graph` field.

    Returns
    -------
    Graph
        A graph built as a single step.

    Examples
    --------
    ```python
    import edgecase as ec

    data = {
        "directed": False,
        "nodes": [{"id": "a"}, {"id": "b"}],
        "adjacency": [[{"id": "b", "weight": 2.0}], [{"id": "a", "weight": 2.0}]],
    }

    # Returns 1
    ec.from_adjacency_data(data).count_edges()
    ```

    """
    data = _load_json(source)
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("nodes"), list)
        or not isinstance(data.get("adjacency"), list)
    ):
        raise ValueError("Adjacency data must be an object with 'nodes' and 'adjacency' lists")
    if len(data["nodes"]) != len(data["adjacency"]):
        raise ValueError("Adjacency data needs one 'adjacency' list per node")

    if directed is None:
        directed = bool(data.get("directed", False))
    if graph_name is None:
        info = data.get("graph", {})
        if isinstance(info, list):
            info = {pair[0]: pair[1] for pair in info if isinstance(pair, list) and len(pair) == 2}
        name = info.get("name") if isinstance(info, dict) else None
        graph_name = name if isinstance(name, str) else None

    nodes = []
    for node in data["nodes"]:
        values = dict(node)
        if "id" not in values:
            raise ValueError("Every node in adjacency data needs an 'id'")
        key = _hashable(values.pop("id"))
        nodes.append((key, str(key), values))

    edges = []
    # Undirected: node pair -> {listing node: its edges}, in order of first appearance
    listings: dict[frozenset, dict[Any, list]] = {}
    for (u, _, _), entries in zip(nodes, data["adjacency"]):
        for entry in entries:
            values = dict(entry)
            if "id" not in values:
                raise ValueError("Every neighbor in adjacency data needs an 'id'")
            v = _hashable(values.pop("id"))
            values.pop("key", None)
            weight = float(values.pop("weight")) if _is_number(values.get("weight")) else None
            edge = (u, v, weight, values)
            if directed:
                edges.append(edge)
            else:
                listings.setdefault(frozenset((u, v)), {}).setdefault(u, []).append(edge)
    for sides in listings.values():
        edges.extend(max(sides.values(), key=len))

    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)
    return _build_graph(
        nodes, edges, directed, weighted, graph_name, _code("from_adjacency_data", source)
    )


def to_tree_data(
    graph: Graph,
    root: Union[int, str],
    filepath: Union[str, Path, None] = None,
    ident: str = "id",
    children: str = "children",
) -> dict:
    """
    Convert a tree to nested JSON tree data.

    The layout of NetworkX's `tree_data()` (and of D3.js hierarchies): the root node as an object
    holding its label under `ident=`, its attributes, and (unless it's a leaf) its children under
    `children=`, each child an object of the same kind. Children are listed in edge ID order.
    Edge weights and attributes aren't kept.

    Parameters
    ----------
    graph
        The tree. A directed graph must be an arborescence with every edge pointing away from
        `root=`; an undirected graph must be a tree (connected, with no cycles), and is oriented
        away from `root=`.
    root
        The root node (ID or label).
    filepath
        If given, the data is also written to this file as JSON.
    ident
        The field that holds each node's label.
    children
        The field that holds each node's children.

    Returns
    -------
    dict
        The tree data, nested from the root.

    Examples
    --------
    ```python
    import edgecase as ec

    graph = ec.Graph(directed=True).add_nodes(["r", "a", "b"])
    graph.add_edges([("r", "a"), ("r", "b")])

    # Returns {'id': 'r', 'children': [{'id': 'a'}, {'id': 'b'}]}
    ec.to_tree_data(graph, "r")
    ```

    """
    if ident == children:
        raise ValueError("ident and children must be different fields")
    root_id = graph._node_id_or_raise(root)
    node_rows = _nodes_of(graph)
    keys = _node_keys(node_rows)
    edges = sorted(graph._graph.edges())
    if len(edges) != len(node_rows) - 1:
        raise ValueError(
            f"The graph isn't a tree: it has {len(node_rows)} nodes and {len(edges)} edges"
        )
    directed = graph.is_directed()
    incident: dict[int, list[tuple[int, int]]] = {nid: [] for nid, _, _ in node_rows}
    for eid, source, target in edges:
        incident[source].append((eid, target))
        if not directed:
            incident[target].append((eid, source))

    kids: dict[int, list[int]] = {root_id: []}
    parent_edge = {root_id: None}
    order = [root_id]
    for nid in order:
        for eid, nbr in incident[nid]:
            if eid == parent_edge[nid]:
                continue
            if nbr in kids:
                raise ValueError(
                    f"The graph isn't a tree rooted at {root!r}: node {keys[nbr]!r} is reached "
                    "more than once"
                )
            kids[nbr] = []
            kids[nid].append(nbr)
            parent_edge[nbr] = eid
            order.append(nbr)
    if len(order) < len(node_rows):
        raise ValueError(f"The graph isn't a tree rooted at {root!r}: not every node is reached")

    records = {}
    for nid, _, attrs in node_rows:
        clash = next((k for k in (ident, children) if k in attrs), None)
        if clash is not None:
            raise ValueError(
                f"Node {keys[nid]!r} has an attribute {clash!r}, which tree data uses for "
                "its own field; pass a different ident= or children="
            )
        records[nid] = {ident: keys[nid], **attrs}
    for nid in order:
        if kids[nid]:
            records[nid][children] = [records[kid] for kid in kids[nid]]
    data = records[root_id]
    if filepath is not None:
        _write_json(data, filepath)
    return data


def from_tree_data(
    source: Union[dict, str, bytes, Path],
    ident: str = "id",
    children: str = "children",
    directed: bool = True,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a tree from nested JSON tree data.

    Reads the layout of `to_tree_data()`, NetworkX's `tree_data()` and D3.js hierarchies: an
    object for the root node, holding its id under `ident=` and its children (objects of the
    same kind) under `children=`. Each node is labeled with its id and gets the object's other
    fields as attributes, and each child gets an edge from its parent. Nodes are added in
    depth-first order, as NetworkX's `tree_graph()` does.

    Parameters
    ----------
    source
        The data: a dict, a JSON string, or a path to a JSON file.
    ident
        The field that holds each node's id.
    children
        The field that holds each node's children.
    directed
        Whether to create a directed tree (edges from parent to child, the default) or an
        undirected one.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        The tree, built as a single step.

    Examples
    --------
    ```python
    import edgecase as ec

    data = {"id": "r", "children": [{"id": "a", "children": [{"id": "b"}]}]}

    # Returns ['r->a', 'a->b']
    ec.from_tree_data(data).get_edges_str_repr(use_labels=True)
    ```

    """
    if ident == children:
        raise ValueError("ident and children must be different fields")
    data = _load_json(source)
    nodes = []
    edges = []
    stack: list[tuple[Any, Any]] = [(data, None)]
    while stack:
        record, parent = stack.pop()
        if not isinstance(record, dict) or ident not in record:
            raise ValueError(f"Every node in tree data must be an object with an {ident!r} field")
        values = dict(record)
        key = _hashable(values.pop(ident))
        kids = values.pop(children, [])
        if not isinstance(kids, list):
            raise ValueError(f"The {children!r} field must be a list")
        nodes.append((key, str(key), values))
        if parent is not None:
            edges.append((parent, key, None, {}))
        stack.extend((kid, key) for kid in reversed(kids))
    return _build_graph(nodes, edges, directed, False, graph_name, _code("from_tree_data", source))


__all__ = [
    "from_matrix_market",
    "from_graphml",
    "to_graphml",
    "from_gml",
    "to_gml",
    "from_sif",
    "to_sif",
    "from_tsv_edge_list",
    "to_tsv_edge_list",
    "from_gexf",
    "to_gexf",
    "from_node_link",
    "to_node_link",
    "from_cytoscape",
    "to_cytoscape",
    "from_pajek",
    "to_pajek",
    "from_graph6",
    "to_graph6",
    "from_sparse6",
    "to_sparse6",
    "to_numpy_array",
    "from_numpy_array",
    "to_scipy_sparse_array",
    "from_scipy_sparse_array",
    "to_pandas_adjacency",
    "from_pandas_adjacency",
    "to_pandas_edgelist",
    "from_pandas_edgelist",
    "to_adjlist",
    "from_adjlist",
    "to_multiline_adjlist",
    "from_multiline_adjlist",
    "to_adjacency_data",
    "from_adjacency_data",
    "to_tree_data",
    "from_tree_data",
]
