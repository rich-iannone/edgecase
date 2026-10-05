import json
import math
import random
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional


def _load_vis_network_js() -> str:
    """
    Load vis-network.js content from static directory.

    Returns the complete vis-network JavaScript code for embedding in HTML.
    vis-network is a 672KB library for interactive network visualization
    with physics-based layouts.

    Returns
    -------
    str
        Complete vis-network JavaScript code
    """
    static_dir = Path(__file__).parent / "static"
    vis_network_path = static_dir / "vis-network.min.js"

    if not vis_network_path.exists():
        raise FileNotFoundError(
            f"vis-network.min.js not found at {vis_network_path}. "
            "This file is required for graph visualization."
        )

    return vis_network_path.read_text()


def plot_graph(
    graph,
    output_path: Optional[str] = None,
    open_browser: bool = True,
    width: int = 800,
    height: int = 600,
    node_size: float = 20.0,
    edge_width: float = 2.0,
    layout: str = "force",
    static: bool = False,
    display_inline: Optional[bool] = None,
) -> str:
    """
    Render graph using vis-network.

    Parameters
    ----------
    graph
        EdgeCase graph to visualize
    output_path
        Path to save HTML file. If `None`, creates a temporary file.
    open_browser
        Whether to open in browser automatically. Defaults to `True`.
    width
        Canvas width in pixels. Defaults to 800.
    height
        Canvas height in pixels. Defaults to 600.
    node_size
        Node radius in pixels. Defaults to 20.0.
    edge_width
        Edge line width in pixels. Defaults to 2.0.
    layout
        Layout algorithm: `"force"` (physics-based), `"circular"`, or `"random"`. Defaults to
        `"force"`.
    static
        If `True`, disables physics simulation and interactivity (no node dragging, no animation).
        The graph will be rendered in a fixed position after initial layout. Useful for static
        reports, exports, or embedding in documents. Defaults to `False`.
    display_inline
        Whether to display the HTML inline in an IPython/Jupyter session. If `None` (default),
        display inline whenever an IPython session is active. If `False`, never display inline,
        so `open_browser=` decides whether a browser tab is opened.

    Returns
    -------
    str
        Path to generated HTML file

    Examples
    --------
    ```python
    from edgecase import Graph, plot_graph

    G = Graph()
    G.add_node(label="A")
    G.add_node(label="B")
    G.add_node(label="C")
    G.add_edge(source="A", target="B")
    G.add_edge(source="B", target="C")

    plot_graph(G, layout="force")
    plot_graph(G, layout="force", static=True)  # Static view
    ```
    """
    # Extract graph data (fast - direct from Rust)
    nodes = graph.get_node_ids(limit=None)
    # One entry per edge, so parallel edges (and A->B plus B->A) are all drawn
    edges = graph.get_edges_as_tuples()

    # Compute initial positions based on layout
    if layout == "circular":
        positions = _circular_layout(nodes)
    elif layout == "random":
        positions = _random_layout(nodes)
    else:
        # "force" layout - let vis-network handle it
        positions = None

    html = _generate_vis_network_html(
        graph,
        nodes,
        edges,
        positions,
        width=width,
        height=height,
        node_size=node_size,
        edge_width=edge_width,
        static=static,
    )

    # Save and optionally open
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="edgecase_")
        with open(fd, "w") as f:
            f.write(html)
    else:
        with open(output_path, "w") as f:
            f.write(html)

    # Check if we're in a notebook environment
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None and display_inline is not False:
            # We're in a notebook - display the HTML directly
            # Wrap in an iframe srcdoc to avoid any notebook sanitization issues
            # Add extra space for body padding (20px) + container padding (20px) + borders
            iframe_html = f"""
            <iframe
                srcdoc="{html.replace('"', "&quot;")}"
                width="{width + 100}"
                height="{height + 120}"
                frameborder="0"
                style="border: none; overflow: hidden;">
            </iframe>
            """
            display(HTML(iframe_html))
            # In notebooks, we display inline, so don't open browser or return
            return output_path
    except (ImportError, NameError):
        pass

    # Not in a notebook - use browser if requested
    if open_browser:
        webbrowser.open(f"file://{Path(output_path).absolute()}")

    return output_path


def _circular_layout(nodes: list[str]) -> dict:
    """Arrange nodes in a circle."""
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node] = [math.cos(angle) * 200, math.sin(angle) * 200]
    return positions


def _random_layout(nodes: list[str]) -> dict:
    """Random node positions."""
    return {node: [random.uniform(-200, 200), random.uniform(-200, 200)] for node in nodes}


def _force_directed_layout(
    node_ids: list,
    edges: list,
    iterations: int = 500,
    use_barnes_hut: bool = True,
) -> dict:
    """Compute force-directed layout via the Rust Fruchterman-Reingold engine.

    Parameters
    ----------
    node_ids : list
        Node identifiers (any hashable type).
    edges : list
        Edge tuples `(source, target)`.
    iterations : int
        Number of FR iterations.
    use_barnes_hut : bool
        Whether to use Barnes-Hut approximation.

    Returns
    -------
    dict
        Mapping of node id to `[x, y]` position.
    """
    from edgecase import _core as rust

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    index_edges = []
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            index_edges.append((id_to_idx[src], id_to_idx[tgt]))

    raw = rust.force_directed_layout_rust(len(node_ids), index_edges, iterations, use_barnes_hut)

    return {node_ids[idx]: [x, y] for idx, (x, y) in raw.items()}


def _generate_vis_network_html(
    graph,
    nodes: list[str],
    edges: list[tuple[int, int, int]],
    positions: dict | None,
    width: int,
    height: int,
    node_size: float,
    edge_width: float,
    static: bool = False,
) -> str:
    """Generate standalone HTML with vis-network renderer.

    Parameters
    ----------
    static : bool, default False
        If True, disables physics simulation and interactivity after initial layout.
    """

    # Get graph properties
    is_directed = graph.is_directed()

    # Helper function to calculate luminance and choose contrasting text color
    def get_contrast_color(hex_color: str) -> str:
        """Calculate luminance and return white or dark gray for best contrast."""
        # Remove # if present
        hex_color = hex_color.lstrip("#")

        # Handle short hex format (#RGB -> #RRGGBB)
        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])

        # Parse RGB values
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
        except (ValueError, IndexError):
            return "#fff"  # Default to white if parsing fails

        # Calculate relative luminance (WCAG formula)
        def adjust(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = adjust(r), adjust(g), adjust(b)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Return white for dark colors, dark gray for light colors
        return "#fff" if luminance < 0.5 else "#333"

    # Convert to vis-network format
    vis_nodes = []
    for node in nodes:
        node_obj = graph.get_node(node)

        # Add subtle spacing hack for single-digit labels to improve centering
        label = str(node)
        if len(label) == 1:
            label = "\u2009" + label  # Add thin space (U+2009) to nudge right

        node_dict = {
            "id": str(node),
            "label": label,
        }

        # Optionally add initial positions if provided
        if positions and node in positions:
            pos = positions[node]
            node_dict["x"] = pos[0]
            node_dict["y"] = pos[1]

        # Add custom attributes if present
        if node_obj and node_obj.attributes:
            # Support both 'color' and 'fillcolor' (for compatibility with Graphviz)
            node_color = None
            if "color" in node_obj.attributes:
                node_color = node_obj.attributes["color"]
                node_dict["color"] = node_color
            elif "fillcolor" in node_obj.attributes:
                node_color = node_obj.attributes["fillcolor"]
                node_dict["color"] = node_color

            # Set font color based on background color for contrast
            if node_color:
                node_dict["font"] = {"color": get_contrast_color(node_color)}

            if "size" in node_obj.attributes:
                node_dict["value"] = node_obj.attributes["size"]

            # Store all original attributes for info panel display
            node_dict["attrs"] = dict(node_obj.attributes)
        vis_nodes.append(node_dict)

    vis_edges = []
    # Parallel edges between the same two nodes get alternating curves of growing roundness so
    # they don't draw on top of each other; the first one keeps the default smooth curve
    pair_count: dict[tuple, int] = {}
    for eid, src, tgt in edges:
        edge_obj = graph.get_edge(eid)
        edge_dict = {
            "id": str(eid),
            "from": str(src),
            "to": str(tgt),
        }
        pair = (min(src, tgt), max(src, tgt))
        k = pair_count.get(pair, 0)
        pair_count[pair] = k + 1
        if k > 0 and src != tgt:
            # Flip the curve direction for edges stored the other way round, so CW/CCW refer to
            # the same side of the pair
            clockwise = (k % 2 == 1) == (src < tgt)
            edge_dict["smooth"] = {
                "enabled": True,
                "type": "curvedCW" if clockwise else "curvedCCW",
                "roundness": 0.2 * ((k + 1) // 2),
            }
        # Add custom attributes if present
        if edge_obj and edge_obj.attributes:
            if "color" in edge_obj.attributes:
                edge_dict["color"] = edge_obj.attributes["color"]
            if "weight" in edge_obj.attributes:
                edge_dict["value"] = edge_obj.attributes["weight"]

            # Store all original attributes for info panel display
            edge_dict["attrs"] = dict(edge_obj.attributes)
        vis_edges.append(edge_dict)

    # Load vis-network library inline
    vis_network_js = _load_vis_network_js()

    # Build HTML in parts to avoid f-string escaping issues with the large JS library
    html_start = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EdgeCase Graph Visualization</title>
    <script type="text/javascript">
"""

    html_after_vis = (
        """
    </script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        #viz-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: fit-content;
            margin: 0 auto;
        }
        #mynetwork {
            width: """
        + str(width)
        + """px;
            height: """
        + str(height)
        + """px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #fafafa;
        }
        #controls {
            margin-top: 10px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .zoom-btn {
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 500;
            color: #fff;
            background: #667eea;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .zoom-btn:hover {
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        .zoom-btn:active {
            transform: translateY(0);
        }
        #info-panel {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            max-width: 300px;
            display: none;
            z-index: 1000;
        }
        #info-panel.visible {
            display: block;
        }
        #info-panel .info-title {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            color: #333;
        }
        #info-panel .info-item {
            margin: 4px 0;
            color: #555;
        }
        #info-panel .info-label {
            font-weight: 500;
            color: #666;
        }
        #info-panel .info-value {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            color: #333;
        }
    </style>
</head>
<body>
    <div id="viz-container">
        <div id="info-panel">
            <div id="info-content"></div>
        </div>
        <div id="mynetwork"></div>
        <div id="controls">
            <button class="zoom-btn" onclick="zoomOut()">Zoom Out</button>
            <button class="zoom-btn" onclick="resetZoom()">Reset View</button>
            <button class="zoom-btn" onclick="zoomIn()">Zoom In</button>
        </div>
    </div>

    <script type="text/javascript">
        // Wait for DOM to be fully loaded before initializing
        window.addEventListener('DOMContentLoaded', function() {
            // Check if vis is loaded
            if (typeof vis === 'undefined') {
                console.error('vis-network library not loaded!');
                document.getElementById('mynetwork').innerHTML = '<div style="padding: 20px; color: red;">Error: vis-network library failed to load.</div>';
                return;
            }

            // Graph data
            var nodesData = """
        + json.dumps(vis_nodes)
        + """;
            var edgesData = """
        + json.dumps(vis_edges)
        + """;

            console.log('Nodes:', nodesData.length);
            console.log('Edges:', edgesData.length);

            var nodes = new vis.DataSet(nodesData);
            var edges = new vis.DataSet(edgesData);

            // Graph properties
            var isDirected = """
        + ("true" if is_directed else "false")
        + """;
            var isStatic = """
        + ("true" if static else "false")
        + """;

            // Adaptive settings based on graph size
            var nodeCount = nodesData.length;
            var isLargeGraph = nodeCount > 1000;
            var stabilizationIterations = isLargeGraph ? 100 : 200;
            // Disable physics after stabilization if static mode, otherwise keep running
            var physicsEnabled = !isStatic;

            // Create network
            var container = document.getElementById('mynetwork');

            var data = {
                nodes: nodes,
                edges: edges
            };

            var options = {
                nodes: {
                    shape: 'circle',
                    size: """
        + str(int(node_size))
        + """,
                    font: {
                        size: 12,
                        color: '#fff',
                        face: 'arial',
                        bold: '600',
                        vadjust: 0
                    },
                    scaling: {
                        label: {
                            enabled: false
                        }
                    },
                    borderWidth: 2,
                    color: {
                        border: '#2B7CE9',
                        background: '#97C2FC',
                        highlight: {
                            border: '#2B7CE9',
                            background: '#D2E5FF'
                        },
                        hover: {
                            border: '#2B7CE9',
                            background: '#D2E5FF'
                        }
                    }
                },
                edges: {
                    width: """
        + str(int(edge_width))
        + """,
                    color: {
                        color: '#848484',
                        highlight: '#2B7CE9'
                    },
                    smooth: {
                        type: 'continuous',
                        roundness: 0.5
                    },
                    arrows: {
                        to: {
                            enabled: """
        + ("true" if is_directed else "false")
        + """,
                            scaleFactor: 0.5
                        }
                    }
                },
                physics: {
                    enabled: physicsEnabled,
                    barnesHut: {
                        theta: 0.5,
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0
                    },
                    maxVelocity: 50,
                    minVelocity: 0.1,
                    solver: 'barnesHut',
                    stabilization: {
                        enabled: true,
                        iterations: stabilizationIterations,
                        updateInterval: isLargeGraph ? 50 : 100,
                        fit: true
                    }
                },
                interaction: {
                    dragNodes: !isStatic,
                    dragView: true,
                    zoomView: true,
                    hover: !isLargeGraph && !isStatic,
                    tooltipDelay: 100,
                    zoomSpeed: 0.5,
                    hideEdgesOnDrag: isLargeGraph,
                    hideEdgesOnZoom: isLargeGraph
                }
            };

            var network = new vis.Network(container, data, options);

            // Add click handler for node and edge info
            network.on("click", function(params) {
                var infoPanel = document.getElementById('info-panel');
                var infoContent = document.getElementById('info-content');

                if (params.nodes.length > 0) {
                    // Node clicked
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);

                    var html = '<div class="info-title">Node: ' + nodeId + '</div>';

                    // Show node attributes from the attrs object
                    if (node.attrs) {
                        var keys = Object.keys(node.attrs);
                        for (var i = 0; i < keys.length; i++) {
                            var key = keys[i];

                            // Skip aesthetic attributes
                            if (key === 'size' || key === 'color' || key === 'fillcolor') {
                                continue;
                            }

                            var value = node.attrs[key];

                            // Format floats to 4 decimal places
                            if (typeof value === 'number' && !Number.isInteger(value)) {
                                value = value.toFixed(4);
                            }

                            html += '<div class="info-item"><span class="info-label">' + key + ':</span> <span class="info-value">' + value + '</span></div>';
                        }
                    }

                    infoContent.innerHTML = html;
                    infoPanel.classList.add('visible');
                } else if (params.edges.length > 0) {
                    // Edge clicked
                    var edgeId = params.edges[0];
                    var edge = edges.get(edgeId);

                    var separator = isDirected ? ' → ' : ' — ';
                    var html = '<div class="info-title">Edge: ' + edge.from + separator + edge.to + '</div>';

                    // Show edge attributes from the attrs object
                    if (edge.attrs) {
                        var keys = Object.keys(edge.attrs);
                        for (var i = 0; i < keys.length; i++) {
                            var key = keys[i];

                            // Skip aesthetic attributes
                            if (key === 'color' || key === 'weight') {
                                continue;
                            }

                            var value = edge.attrs[key];

                            // Format floats to 4 decimal places
                            if (typeof value === 'number' && !Number.isInteger(value)) {
                                value = value.toFixed(4);
                            }

                            html += '<div class="info-item"><span class="info-label">' + key + ':</span> <span class="info-value">' + value + '</span></div>';
                        }
                    }

                    infoContent.innerHTML = html;
                    infoPanel.classList.add('visible');
                } else {
                    // Clicked on empty space - hide panel
                    infoPanel.classList.remove('visible');
                }
            });

            // Log when stabilization is done
            network.on("stabilizationIterationsDone", function() {
                console.log("Network stabilized!");
                network.fit();
                // In static mode, stop physics after initial layout
                if (isStatic) {
                    network.setOptions({ physics: { enabled: false } });
                    console.log("Static mode: physics disabled after stabilization");
                }
            });

            // Zoom control functions
            window.zoomIn = function() {
                var scale = network.getScale();
                network.moveTo({
                    scale: scale * 1.2,
                    animation: {
                        duration: 300,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            };

            window.zoomOut = function() {
                var scale = network.getScale();
                network.moveTo({
                    scale: scale / 1.2,
                    animation: {
                        duration: 300,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            };

            window.resetZoom = function() {
                network.fit({
                    animation: {
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            };
        });
    </script>
</body>
</html>"""
    )

    html = html_start + vis_network_js + html_after_vis

    return html


def plot_cosmo(
    graph,
    point_color_by: Optional[str] = None,
    point_size_by: Optional[str] = None,
    link_color_by: Optional[str] = None,
    link_width_by: Optional[str] = None,
):
    """
    Render graph as an interactive Cosmograph widget in a Jupyter notebook.

    Requires the `cosmograph` package (`pip install cosmograph`). The widget
    uses GPU-accelerated WebGL rendering and handles millions of nodes/edges in
    the browser.

    Parameters
    ----------
    graph : Graph
        EdgeCase graph to visualize.
    point_color_by : str, optional
        Node DataFrame column name to use for point colour encoding.
    point_size_by : str, optional
        Node DataFrame column name to use for point size encoding.
    link_color_by : str, optional
        Edge DataFrame column name to use for link colour encoding.
    link_width_by : str, optional
        Edge DataFrame column name to use for link width encoding.

    Returns
    -------
    cosmograph.Cosmograph
        An interactive Cosmograph widget. Assign it to a variable and evaluate
        it in a notebook cell to display it.

    Raises
    ------
    ImportError
        If the `cosmograph` package is not installed.

    Examples
    --------
    Basic usage in a Jupyter notebook:

    ```python
    from edgecase import Graph, plot_cosmo

    G = Graph()
    G.add_node(label="Alice", type="user")
    G.add_node(label="Bob", type="admin")
    G.add_node(label="Carol", type="user")
    G.add_edge(source="Alice", target="Bob")
    G.add_edge(source="Bob", target="Carol")

    widget = plot_cosmo(G)
    widget
    ```

    Colour nodes by type attribute:

    ```python
    widget = plot_cosmo(G, point_color_by="type")
    widget
    ```
    """
    try:
        from cosmograph import cosmo
    except ImportError as e:
        raise ImportError(
            "The 'cosmograph' package is required for plot_cosmo(). "
            "Install it with: pip install cosmograph"
        ) from e

    node_df = graph.get_node_df()
    edge_df = graph.get_edge_df()

    # Convert Polars DataFrames to pandas (cosmograph requires pandas)
    points = node_df.to_pandas()
    links = edge_df.to_pandas()

    kwargs = {
        "point_id_by": "id",
        "link_source_by": "source",
        "link_target_by": "target",
    }

    if point_color_by is not None:
        kwargs["point_color_by"] = point_color_by
    if point_size_by is not None:
        kwargs["point_size_by"] = point_size_by
    if link_color_by is not None:
        kwargs["link_color_by"] = link_color_by
    if link_width_by is not None:
        kwargs["link_width_by"] = link_width_by

    # Include all columns for hover/inspect by default
    point_cols = [c for c in points.columns if c != "id"]
    link_cols = [c for c in links.columns if c not in ("source", "target")]

    if point_cols:
        kwargs["point_include_columns"] = point_cols
    if link_cols:
        kwargs["link_include_columns"] = link_cols

    # Use label column for point labels when present
    if "label" in points.columns:
        kwargs.setdefault("point_label_by", "label")

    return cosmo(points=points, links=links, **kwargs)


# Convenience export
__all__ = ["plot_graph", "plot_cosmo"]
