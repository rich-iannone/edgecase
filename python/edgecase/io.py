from pathlib import Path
from typing import Optional, Union

import polars as pl

from edgecase.graph import Graph


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

    GraphML is an XML-based format used by Cytoscape, STRING, yEd, and many bioinformatics tools.
    This reader supports the core GraphML spec: `<node>`, `<edge>`, and `<data>` elements with
    `<key>` declarations.

    Parameters
    ----------
    filepath
        Path to the GraphML (.graphml) file.
    directed
        Whether to create a directed graph. If `None`, detected from the `edgedefault` attribute of
        the `<graph>` element.
    weighted
        Whether the graph has weights. If `None`, detected from the presence of a `weight` data key
        on edges.
    graph_name
        Name for the graph. If `None`, uses the `id` attribute of the `<graph>` element.

    Returns
    -------
    Graph
        A graph constructed from the GraphML file.
    """
    import xml.etree.ElementTree as ET

    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    key_defs: dict[str, dict] = {}
    for key_elem in root.iter(f"{ns}key"):
        kid = key_elem.get("id", "")
        key_defs[kid] = {
            "for": key_elem.get("for", ""),
            "name": key_elem.get("attr.name", kid),
            "type": key_elem.get("attr.type", "string"),
        }

    graph_elem = root.find(f".//{ns}graph")
    if graph_elem is None:
        raise ValueError("No <graph> element found in GraphML file")

    if directed is None:
        edgedefault = graph_elem.get("edgedefault", "undirected")
        directed = edgedefault == "directed"

    if graph_name is None:
        graph_name = graph_elem.get("id")

    weight_key_id = None
    for kid, kdef in key_defs.items():
        if kdef["for"] in ("edge", "all") and kdef["name"] == "weight":
            weight_key_id = kid
            break

    if weighted is None:
        weighted = weight_key_id is not None

    g = Graph(directed=directed, weighted=weighted, graph_name=graph_name)

    node_id_map: dict[str, str] = {}
    for node_elem in graph_elem.iter(f"{ns}node"):
        raw_id = node_elem.get("id", "")
        label = raw_id
        for data_elem in node_elem.findall(f"{ns}data"):
            k = data_elem.get("key", "")
            if k in key_defs and key_defs[k]["name"] == "label" and data_elem.text:
                label = data_elem.text
        g.add_node(label=label)
        node_id_map[raw_id] = label

    for edge_elem in graph_elem.iter(f"{ns}edge"):
        src_raw = edge_elem.get("source", "")
        tgt_raw = edge_elem.get("target", "")
        src_label = node_id_map.get(src_raw, src_raw)
        tgt_label = node_id_map.get(tgt_raw, tgt_raw)

        kwargs: dict = {}
        if weighted and weight_key_id is not None:
            for data_elem in edge_elem.findall(f"{ns}data"):
                if data_elem.get("key") == weight_key_id and data_elem.text:
                    kwargs["weight"] = float(data_elem.text)
                    break

        g.add_edge(src_label, tgt_label, **kwargs)

    return g


def to_graphml(
    graph: Graph,
    filepath: Union[str, Path],
) -> None:
    """
    Write a graph to a GraphML file.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    """
    import xml.etree.ElementTree as ET

    filepath = Path(filepath)

    graphml = ET.Element("graphml")
    graphml.set("xmlns", "http://graphml.graphstruct.org/xmlns")

    ET.SubElement(
        graphml,
        "key",
        id="label",
        attrib={
            "for": "node",
            "attr.name": "label",
            "attr.type": "string",
        },
    )

    if graph.is_weighted():
        ET.SubElement(
            graphml,
            "key",
            id="weight",
            attrib={
                "for": "edge",
                "attr.name": "weight",
                "attr.type": "double",
            },
        )

    edgedefault = "directed" if graph.is_directed() else "undirected"
    graph_elem = ET.SubElement(graphml, "graph", edgedefault=edgedefault)

    node_ids = graph.get_node_ids()
    id_to_label: dict[int, str] = {}
    for nid in node_ids:
        label = graph.get_label_from_id(nid)
        if label is None:
            label = str(nid)
        id_to_label[nid] = label
        node_elem = ET.SubElement(graph_elem, "node", id=str(nid))
        data_elem = ET.SubElement(node_elem, "data", key="label")
        data_elem.text = label

    weight_map: dict[tuple[int, int], float] = {}
    if graph.is_weighted():
        edf = graph.get_edge_df()
        for row in edf.iter_rows(named=True):
            src = row.get("source") or row.get("from")
            tgt = row.get("target") or row.get("to")
            w = row.get("weight")
            if src is not None and tgt is not None and w is not None:
                weight_map[(src, tgt)] = w

    edges_raw = graph._graph.edges()
    for eid, src, tgt in edges_raw:
        edge_elem = ET.SubElement(graph_elem, "edge", source=str(src), target=str(tgt))
        if graph.is_weighted():
            w = weight_map.get((src, tgt))
            if w is not None:
                data_elem = ET.SubElement(edge_elem, "data", key="weight")
                data_elem.text = str(w)

    tree = ET.ElementTree(graphml)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="unicode", xml_declaration=True)


# =============================================================================
# GML I/O
# =============================================================================


def from_gml(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a GML (Graph Modelling Language) file.

    GML is a hierarchical ASCII-based format common in network science tools.

    Parameters
    ----------
    filepath
        Path to the GML file.
    directed
        Whether to create a directed graph. If `None`, detected from the `directed` field in the
        `graph` block (0 = undirected, 1 = directed).
    weighted
        Whether the graph has weights. If `None`, detected from the presence of `value` fields on
        edges.
    graph_name
        Name for the graph.

    Returns
    -------
    Graph
        A graph constructed from the GML file.
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    tokens = _gml_tokenize(content)
    parsed = _gml_parse(tokens)

    graph_block = None
    for key, val in parsed:
        if key == "graph":
            graph_block = val
            break

    if graph_block is None:
        raise ValueError("No 'graph' block found in GML file")

    if directed is None:
        for key, val in graph_block:
            if key == "directed":
                directed = int(val) == 1
                break
        if directed is None:
            directed = False

    nodes_raw: list[list] = []
    edges_raw: list[list] = []
    for key, val in graph_block:
        if key == "node":
            nodes_raw.append(val)
        elif key == "edge":
            edges_raw.append(val)

    has_weight = False
    if weighted is None:
        for eblock in edges_raw:
            for key, val in eblock:
                if key == "value" or key == "weight":
                    has_weight = True
                    break
            if has_weight:
                break
        weighted = has_weight

    g = Graph(directed=directed, weighted=weighted, graph_name=graph_name)

    gml_id_to_label: dict[str, str] = {}
    for nblock in nodes_raw:
        nid = None
        label = None
        for key, val in nblock:
            if key == "id":
                nid = str(val)
            elif key == "label":
                label = val.strip('"')
        if nid is not None:
            if label is None:
                label = nid
            g.add_node(label=label)
            gml_id_to_label[nid] = label

    for eblock in edges_raw:
        source = None
        target = None
        weight = None
        for key, val in eblock:
            if key == "source":
                source = str(val)
            elif key == "target":
                target = str(val)
            elif key in ("value", "weight"):
                weight = float(val)
        if source is not None and target is not None:
            src_label = gml_id_to_label.get(source, source)
            tgt_label = gml_id_to_label.get(target, target)
            kwargs = {}
            if weighted and weight is not None:
                kwargs["weight"] = weight
            g.add_edge(src_label, tgt_label, **kwargs)

    return g


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
    """Parse tokenized GML into nested (key, value) lists."""
    result: list = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == "]":
            return result
        if i + 1 < n:
            key = tokens[i]
            i += 1
            if tokens[i] == "[":
                i += 1
                sub_tokens = []
                depth = 1
                while i < n and depth > 0:
                    if tokens[i] == "[":
                        depth += 1
                    elif tokens[i] == "]":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    sub_tokens.append(tokens[i])
                    i += 1
                result.append((key, _gml_parse(sub_tokens)))
            else:
                result.append((key, tokens[i]))
                i += 1
        else:
            i += 1
    return result


def to_gml(
    graph: Graph,
    filepath: Union[str, Path],
) -> None:
    """
    Write a graph to a GML file.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    """
    filepath = Path(filepath)
    lines: list[str] = []
    lines.append("graph [")
    lines.append(f"  directed {1 if graph.is_directed() else 0}")

    node_ids = graph.get_node_ids()
    id_to_label: dict[int, str] = {}
    for nid in node_ids:
        label = graph.get_label_from_id(nid)
        if label is None:
            label = str(nid)
        id_to_label[nid] = label
        lines.append("  node [")
        lines.append(f"    id {nid}")
        lines.append(f'    label "{label}"')
        lines.append("  ]")

    edges_raw = graph._graph.edges()
    if graph.is_weighted():
        edf = graph.get_edge_df()
        weight_map: dict[tuple[int, int], float] = {}
        for row in edf.iter_rows(named=True):
            src = row.get("source") or row.get("from")
            tgt = row.get("target") or row.get("to")
            w = row.get("weight")
            if src is not None and tgt is not None and w is not None:
                weight_map[(src, tgt)] = w

    for eid, src, tgt in edges_raw:
        lines.append("  edge [")
        lines.append(f"    source {src}")
        lines.append(f"    target {tgt}")
        if graph.is_weighted():
            w = weight_map.get((src, tgt))
            if w is not None:
                lines.append(f"    value {w}")
        lines.append("  ]")

    lines.append("]")

    with open(filepath, "w") as f:
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

    A node with no interactions can appear alone on a line.

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
        A graph constructed from the SIF file.
    """
    filepath = Path(filepath)
    if directed is None:
        directed = False

    g = Graph(directed=directed, weighted=False, graph_name=graph_name)

    seen_nodes: set[str] = set()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()

            if len(parts) == 0:
                continue

            source = parts[0]
            if source not in seen_nodes:
                g.add_node(label=source)
                seen_nodes.add(source)

            if len(parts) == 1:
                continue

            if len(parts) >= 3:
                for target in parts[2:]:
                    target = target.strip()
                    if not target:
                        continue
                    if target not in seen_nodes:
                        g.add_node(label=target)
                        seen_nodes.add(target)
                    g.add_edge(source, target)

    return g


def to_sif(
    graph: Graph,
    filepath: Union[str, Path],
    interaction_type: str = "interacts_with",
) -> None:
    """
    Write a graph to a Simple Interaction Format (SIF) file.

    Parameters
    ----------
    graph
        The graph to write.
    filepath
        Output file path.
    interaction_type
        The interaction label placed between source and target nodes. Defaults to
        `"interacts_with"`.
    """
    filepath = Path(filepath)

    node_ids = graph.get_node_ids()
    id_to_label: dict[int, str] = {}
    for nid in node_ids:
        label = graph.get_label_from_id(nid)
        if label is None:
            label = str(nid)
        id_to_label[nid] = label

    adjacency: dict[str, list[str]] = {}
    nodes_with_edges: set[str] = set()
    edges_raw = graph._graph.edges()
    for eid, src, tgt in edges_raw:
        src_label = id_to_label[src]
        tgt_label = id_to_label[tgt]
        adjacency.setdefault(src_label, []).append(tgt_label)
        nodes_with_edges.add(src_label)
        nodes_with_edges.add(tgt_label)

    with open(filepath, "w") as f:
        for src_label, targets in adjacency.items():
            f.write(f"{src_label}\t{interaction_type}\t{chr(9).join(targets)}\n")
        for nid in node_ids:
            label = id_to_label[nid]
            if label not in nodes_with_edges:
                f.write(f"{label}\n")


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

    Common export format from BioGRID, STRING, IntAct, and other databases.

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
        A graph constructed from the edge list file.
    """
    filepath = Path(filepath)
    weighted = weight_col is not None

    g = Graph(directed=directed, weighted=weighted, graph_name=graph_name)

    header_names: Optional[list[str]] = None
    src_idx: int = 0
    tgt_idx: int = 1
    wgt_idx: Optional[int] = None

    seen_nodes: set[str] = set()
    line_num = 0

    with open(filepath, "r") as f:
        for line in f:
            line_num += 1
            if line_num <= skip_rows:
                continue

            stripped = line.strip()
            if not stripped or stripped.startswith(comment):
                continue

            if has_header and header_names is None:
                header_names = stripped.split(delimiter)
                if isinstance(source_col, str):
                    src_idx = header_names.index(source_col)
                else:
                    src_idx = source_col
                if isinstance(target_col, str):
                    tgt_idx = header_names.index(target_col)
                else:
                    tgt_idx = target_col
                if weight_col is not None:
                    if isinstance(weight_col, str):
                        wgt_idx = header_names.index(weight_col)
                    else:
                        wgt_idx = weight_col
                continue

            if header_names is None and line_num == skip_rows + 1:
                if isinstance(source_col, int):
                    src_idx = source_col
                if isinstance(target_col, int):
                    tgt_idx = target_col
                if weight_col is not None and isinstance(weight_col, int):
                    wgt_idx = weight_col

            cols = stripped.split(delimiter)
            max_needed = max(src_idx, tgt_idx)
            if wgt_idx is not None:
                max_needed = max(max_needed, wgt_idx)
            if len(cols) <= max_needed:
                continue

            source = cols[src_idx].strip()
            target = cols[tgt_idx].strip()

            if not source or not target:
                continue

            if source not in seen_nodes:
                g.add_node(label=source)
                seen_nodes.add(source)
            if target not in seen_nodes:
                g.add_node(label=target)
                seen_nodes.add(target)

            kwargs = {}
            if weighted and wgt_idx is not None:
                try:
                    kwargs["weight"] = float(cols[wgt_idx].strip())
                except ValueError:
                    pass

            g.add_edge(source, target, **kwargs)

    return g


def to_tsv_edge_list(
    graph: Graph,
    filepath: Union[str, Path],
    delimiter: str = "\t",
    header: bool = True,
    weight: bool = True,
) -> None:
    """
    Write a graph to a tab-separated edge list file.

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
    filepath = Path(filepath)

    node_ids = graph.get_node_ids()
    id_to_label: dict[int, str] = {}
    for nid in node_ids:
        label = graph.get_label_from_id(nid)
        if label is None:
            label = str(nid)
        id_to_label[nid] = label

    include_weight = weight and graph.is_weighted()

    weight_map: dict[tuple[int, int], float] = {}
    if include_weight:
        edf = graph.get_edge_df()
        for row in edf.iter_rows(named=True):
            src = row.get("source") or row.get("from")
            tgt = row.get("target") or row.get("to")
            w = row.get("weight")
            if src is not None and tgt is not None and w is not None:
                weight_map[(src, tgt)] = w

    edges_raw = graph._graph.edges()

    with open(filepath, "w") as f:
        if header:
            cols = ["source", "target"]
            if include_weight:
                cols.append("weight")
            f.write(delimiter.join(cols) + "\n")

        for eid, src, tgt in edges_raw:
            src_label = id_to_label[src]
            tgt_label = id_to_label[tgt]
            parts = [src_label, tgt_label]
            if include_weight:
                w = weight_map.get((src, tgt), 1.0)
                parts.append(str(w))
            f.write(delimiter.join(parts) + "\n")


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
]
