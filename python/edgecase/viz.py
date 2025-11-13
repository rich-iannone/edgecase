import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional


def _load_viz_js() -> str:
    """
    Load viz.js content from static directory.

    Returns the complete viz.js JavaScript code for embedding in HTML.
    viz.js is a 2MB compiled Graphviz WASM/asm.js library that provides
    production-quality graph layouts (neato, dot, fdp, circo, twopi).

    Returns
    -------
    str
        Complete viz.js JavaScript code
    """
    static_dir = Path(__file__).parent / "static"
    viz_path = static_dir / "viz.js"

    if not viz_path.exists():
        raise FileNotFoundError(
            f"viz.js not found at {viz_path}. "
            "This file is required for Graphviz layout engines (neato, dot, fdp, circo, twopi)."
        )

    return viz_path.read_text()


def plot_graph(
    graph,
    output_path: Optional[str] = None,
    open_browser: bool = True,
    width: int = 800,
    height: int = 600,
    node_size: float = 20.0,
    edge_width: float = 2.0,
    layout: str = "neato",
) -> str:
    """
    Render graph using WebGL (deck.gl-style approach).

    Parameters
    ----------
    graph : Graph
        EdgeCase graph to visualize
    output_path : str, optional
        Path to save HTML file. If None, creates temp file.
    open_browser : bool, default True
        Whether to open in browser automatically
    width : int, default 800
        Canvas width in pixels
    height : int, default 600
        Canvas height in pixels
    node_size : float, default 20.0
        Node radius in pixels
    edge_width : float, default 2.0
        Edge line width in pixels
    layout : str, default "neato"
        Layout algorithm: "neato", "dot", "fdp", "circo", "twopi" (Graphviz engines),
        or "force", "circular", "random" (simple layouts)

    Returns
    -------
    str
        Path to generated HTML file

    Examples
    --------
    >>> from edgecase import Graph, plot_graph
    >>> G = Graph()
    >>> G.add_edge("A", "B")
    >>> G.add_edge("B", "C")
    >>> plot_graph(G, layout="neato")
    """
    # Extract graph data (fast - direct from Rust)
    nodes = graph.get_node_ids()
    edges = []

    # Build edge list efficiently
    seen = set()
    for node in nodes:
        for neighbor in graph.neighbors(node):
            edge_key = tuple(sorted([node, neighbor]))
            if edge_key not in seen:
                edges.append([node, neighbor])
                seen.add(edge_key)

    # Use Graphviz layouts if specified
    if layout in ["neato", "dot", "fdp", "circo", "twopi"]:
        dot_string = _generate_dot_string(graph, nodes, edges)
        viz_js_code = _load_viz_js()
        html = _generate_graphviz_svg_html(
            graph,
            nodes,
            edges,
            dot_string,
            viz_js_code,
            layout,
            width=width,
            height=height,
        )
    # Compute simple layout
    elif layout == "force":
        positions, graph_extent = _force_directed_layout(nodes, edges, iterations=50)
        html = _generate_webgl_html(
            graph,
            nodes,
            edges,
            positions,
            graph_extent,
            width=width,
            height=height,
            node_size=node_size,
            edge_width=edge_width,
        )
    elif layout == "circular":
        positions = _circular_layout(nodes)
        # Calculate extent for circular layout
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        graph_extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) if all_x else 0
        html = _generate_webgl_html(
            graph,
            nodes,
            edges,
            positions,
            graph_extent,
            width=width,
            height=height,
            node_size=node_size,
            edge_width=edge_width,
        )
    else:
        positions = _random_layout(nodes)
        graph_extent = 2.0  # Random layout uses [-1,1] range
        html = _generate_webgl_html(
            graph,
            nodes,
            edges,
            positions,
            graph_extent,
            width=width,
            height=height,
            node_size=node_size,
            edge_width=edge_width,
        )

    # Save and optionally open
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="edgecase_")
        with open(fd, "w") as f:
            f.write(html)
    else:
        with open(output_path, "w") as f:
            f.write(html)

    if open_browser:
        webbrowser.open(f"file://{Path(output_path).absolute()}")

    return output_path


def _force_directed_layout(nodes: list[str], edges: list[list[str]], iterations: int = 500) -> dict:
    """Fruchterman-Reingold layout algorithm (based on igraph implementation).

    Reference: https://igraph.org/r/html/1.3.1/layout_with_fr.html
    """
    import math
    import random

    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: [0.0, 0.0]}

    # Initialize positions randomly in a large space
    random.seed(42)
    # Start with larger initial spacing to avoid local minima
    init_range = math.sqrt(n) * 2.0
    positions = {
        node: [random.uniform(-init_range, init_range), random.uniform(-init_range, init_range)]
        for node in nodes
    }

    # Build adjacency for connected nodes
    adj = {node: set() for node in nodes}
    for src, tgt in edges:
        adj[src].add(tgt)
        adj[tgt].add(src)

    # Layout parameters (igraph FR algorithm)
    # Reduce area to avoid massive scaling compression that causes saddle shapes
    # Use n * 2.0 instead of n * 10.0 for more compact initial layout
    area = n * 2.0
    k = math.sqrt(area / n)  # Optimal distance between nodes

    # Initial temperature (controls movement magnitude)
    temp = k * 2.0

    # Run more iterations for better convergence
    total_iterations = max(iterations, 1000)

    for iteration in range(total_iterations):
        # Calculate repulsive forces between all pairs
        forces = {node: [0.0, 0.0] for node in nodes}

        for i, v in enumerate(nodes):
            for u in nodes[i + 1 :]:
                dx = positions[v][0] - positions[u][0]
                dy = positions[v][1] - positions[u][1]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0.01 else 0.01

                # Repulsive force: f_r(d) = k^2 / d
                force = (k * k) / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force

                forces[v][0] += fx
                forces[v][1] += fy
                forces[u][0] -= fx
                forces[u][1] -= fy

        # Calculate attractive forces for connected nodes
        for src, tgt in edges:
            dx = positions[src][0] - positions[tgt][0]
            dy = positions[src][1] - positions[tgt][1]
            dist_sq = dx * dx + dy * dy
            dist = math.sqrt(dist_sq) if dist_sq > 0.01 else 0.01

            # Attractive force: f_a(d) = d^2 / k
            force = (dist * dist) / k
            fx = (dx / dist) * force
            fy = (dy / dist) * force

            forces[src][0] -= fx
            forces[src][1] -= fy
            forces[tgt][0] += fx
            forces[tgt][1] += fy

        # Update positions with temperature (cooling)
        for node in nodes:
            fx, fy = forces[node]
            force_mag = math.sqrt(fx * fx + fy * fy)

            if force_mag > 0.01:
                # Limit displacement by temperature
                disp = min(force_mag, temp)
                positions[node][0] += (fx / force_mag) * disp
                positions[node][1] += (fy / force_mag) * disp

        # Cool down (exponential cooling for better convergence)
        temp *= 0.99

    # Iterative scaling approach to balance spacing, overlap prevention, and viewport fitting
    if positions:
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Get the range and center
        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0

        # Calculate average edge length in layout coordinates
        total_edge_length = 0.0
        for src, tgt in edges:
            dx = positions[src][0] - positions[tgt][0]
            dy = positions[src][1] - positions[tgt][1]
            total_edge_length += math.sqrt(dx * dx + dy * dy)
        avg_edge_length = total_edge_length / len(edges) if edges else k

        # Debug output
        print("\\n=== Layout Scaling Debug (Python) ===")
        print(f"Number of nodes: {n}")
        print(f"Area: {area:.2f}")
        print(f"k (optimal distance): {k:.4f}")
        print(f"Average edge length in layout: {avg_edge_length:.4f}")
        print(f"Before scaling - X range: [{min_x:.4f}, {max_x:.4f}] = {range_x:.4f}")
        print(f"Before scaling - Y range: [{min_y:.4f}, {max_y:.4f}] = {range_y:.4f}")

        # Target: edges should be about 5 node diameters long in viewport
        # Node size will be 20px, so in viewport coords (assuming 800px canvas),
        # one node diameter ≈ 0.05 viewport units (20/400)
        node_diameter_in_viewport = 0.05
        target_edge_length_in_viewport = node_diameter_in_viewport * 5  # 5 node diameters

        # Minimum readable edge length: 2.5 node diameters (prevent node overlap)
        min_edge_length_in_viewport = node_diameter_in_viewport * 2.5

        # Calculate scale to achieve target edge length
        scale_by_edges = (
            target_edge_length_in_viewport / avg_edge_length if avg_edge_length > 0 else 1.0
        )

        print(f"target_edge_length_in_viewport: {target_edge_length_in_viewport:.4f}")
        print(f"scale_by_edges: {scale_by_edges:.4f}")

        # Apply edge-based scaling and centering
        for node in nodes:
            positions[node][0] = (positions[node][0] - center_x) * scale_by_edges
            positions[node][1] = (positions[node][1] - center_y) * scale_by_edges

        # Check if graph fits in viewport with padding
        all_x_scaled = [positions[node][0] for node in nodes]
        all_y_scaled = [positions[node][1] for node in nodes]
        scaled_range_x = max(all_x_scaled) - min(all_x_scaled)
        scaled_range_y = max(all_y_scaled) - min(all_y_scaled)
        max_scaled_range = max(scaled_range_x, scaled_range_y)

        # Calculate what the edge length would be after shrinking
        potential_shrink = 1.6 / max_scaled_range if max_scaled_range > 1.6 else 1.0
        edge_length_after_shrink = target_edge_length_in_viewport * potential_shrink

        # Only shrink if we can maintain minimum readable edge length
        if max_scaled_range > 1.6 and edge_length_after_shrink >= min_edge_length_in_viewport:
            shrink_factor = 1.6 / max_scaled_range
            print(f"Graph extends beyond viewport, shrinking by {shrink_factor:.4f}")
            print(
                f"Edge length after shrink: {edge_length_after_shrink:.4f} ({edge_length_after_shrink / node_diameter_in_viewport:.2f} diameters)"
            )
            for node in nodes:
                positions[node][0] *= shrink_factor
                positions[node][1] *= shrink_factor
            # Recalculate after shrinking
            all_x_scaled = [positions[node][0] for node in nodes]
            all_y_scaled = [positions[node][1] for node in nodes]
            scaled_range_x = max(all_x_scaled) - min(all_x_scaled)
            scaled_range_y = max(all_y_scaled) - min(all_y_scaled)
        else:
            if max_scaled_range > 1.6:
                print(
                    f"Graph extends beyond viewport ({max_scaled_range:.2f}), but maintaining edge readability"
                )
                print(
                    f"Use pan/zoom to navigate - edges kept at {target_edge_length_in_viewport / node_diameter_in_viewport:.1f} diameters"
                )

        # Calculate final edge lengths and graph extent
        total_scaled_edge_length = 0.0
        for src, tgt in edges:
            dx = positions[src][0] - positions[tgt][0]
            dy = positions[src][1] - positions[tgt][1]
            total_scaled_edge_length += math.sqrt(dx * dx + dy * dy)
        avg_scaled_edge_length = total_scaled_edge_length / len(edges) if edges else 0

        # Calculate final extent for initial zoom
        all_x_final = [positions[node][0] for node in nodes]
        all_y_final = [positions[node][1] for node in nodes]
        final_range_x = max(all_x_final) - min(all_x_final) if all_x_final else 0
        final_range_y = max(all_y_final) - min(all_y_final) if all_y_final else 0
        graph_extent = max(final_range_x, final_range_y)

        print(
            f"After scaling - X range: [{min(all_x_scaled):.4f}, {max(all_x_scaled):.4f}] = {scaled_range_x:.4f}"
        )
        print(
            f"After scaling - Y range: [{min(all_y_scaled):.4f}, {max(all_y_scaled):.4f}] = {scaled_range_y:.4f}"
        )
        print(f"Average edge length after scaling: {avg_scaled_edge_length:.4f}")
        print(
            f"Edge length in node diameters: {avg_scaled_edge_length / node_diameter_in_viewport:.2f}"
        )
        print(f"Graph extent for zoom calculation: {graph_extent:.4f}")
        print("=====================================\\n")

    return positions, graph_extent


def _circular_layout(nodes: list[str]) -> dict:
    """Arrange nodes in a circle."""
    import math

    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node] = [math.cos(angle), math.sin(angle)]
    return positions


def _random_layout(nodes: list[str]) -> dict:
    """Random node positions."""
    import random

    return {node: [random.uniform(-1, 1), random.uniform(-1, 1)] for node in nodes}


def _generate_graphviz_html(
    graph,
    nodes: list[str],
    edges: list[list[str]],
    dot_string: str,
    viz_js_code: str,
    layout_engine: str,
    width: int,
    height: int,
    node_size: float,
    edge_width: float,
) -> str:
    """
    Generate HTML with embedded viz.js for Graphviz layouts.

    This function creates a standalone HTML file that:
    1. Embeds the 2MB viz.js library (Graphviz WASM/asm.js)
    2. Calls Viz() to compute node positions using specified Graphviz engine
    3. Renders the graph using the existing WebGL renderer

    Parameters
    ----------
    graph : Graph
        EdgeCase graph object
    nodes : list[str]
        Node labels
    edges : list[list[str]]
        Edge pairs [source, target]
    dot_string : str
        DOT format representation of graph
    viz_js_code : str
        Complete viz.js JavaScript code (2MB)
    layout_engine : str
        Graphviz engine: 'neato', 'dot', 'fdp', 'circo', or 'twopi'
    width : int
        Canvas width in pixels
    height : int
        Canvas height in pixels
    node_size : float
        Node radius in pixels
    edge_width : float
        Edge line width in pixels

    Returns
    -------
    str
        Complete HTML document with embedded viz.js and WebGL renderer
    """
    import json

    # Get graph properties
    is_directed = graph.is_directed
    graph_type = "directed" if is_directed else "undirected"

    # Escape DOT string for JavaScript
    dot_escaped = json.dumps(dot_string)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EdgeCase Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        #viz-container {{
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: inline-block;
            border-radius: 8px;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        #info {{
            margin-bottom: 10px;
            font-size: 14px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        #canvas-wrapper {{
            position: relative;
            display: inline-block;
        }}
        #canvas {{
            border: 1px solid #ddd;
            background: #f5f5f5;
            cursor: grab;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        #canvas:active {{
            cursor: grabbing;
        }}
        #reset-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            color: #333;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        #reset-btn .icon {{
            font-size: 18px;
            line-height: 1;
        }}
        #reset-btn:hover {{
            background: #fff;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        #stats {{
            margin-top: 10px;
            font-size: 12px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        #zoom-mode-control {{
            margin-top: 15px;
            padding: 10px 15px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 12px;
            color: #fff;
        }}
        #zoom-mode-label {{
            font-weight: 500;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 24px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        .toggle-switch.active {{
            background: rgba(59, 130, 246, 0.6);
        }}
        .toggle-slider {{
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: #fff;
            border-radius: 50%;
            transition: transform 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .toggle-switch.active .toggle-slider {{
            transform: translateX(26px);
        }}
        #zoom-mode-text {{
            min-width: 100px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div id="viz-container">
        <div id="container">
            <div id="info">
                EdgeCase Graph | {graph_type} | {len(nodes)} nodes, {len(edges)} edges | Layout: {layout_engine}
            </div>
            <div id="canvas-wrapper">
                <canvas id="canvas" width="{width}" height="{height}"></canvas>
                <button id="reset-btn" title="Reset view">
                    <span class="icon">⟲</span>
                    <span>Reset</span>
                </button>
                <div id="tooltip"></div>
            </div>
            <div id="stats">Drag to pan • Scroll to zoom • Hover for node info</div>
            <div id="zoom-mode-control">
                <span id="zoom-mode-label">Zoom Mode:</span>
                <div class="toggle-switch" id="zoom-toggle">
                    <div class="toggle-slider"></div>
                </div>
                <span id="zoom-mode-text">Semantic</span>
            </div>
        </div>
    </div>

    <script>
        // Embed viz.js (2MB Graphviz WASM/asm.js library)
        {viz_js_code}
    </script>

    <script>
        // Use Viz.js to compute layout
        const dotString = {dot_escaped};
        const layoutEngine = '{layout_engine}';
        const isDirected = {str(is_directed).lower()};

        // Call Viz() to get positions (returns JSON)
        let vizResult;
        try {{
            vizResult = Viz(dotString, {{
                format: 'json',
                engine: layoutEngine
            }});
        }} catch (error) {{
            console.error('Viz.js layout error:', error);
            alert('Graph layout failed: ' + error.message);
            throw error;
        }}

        // Parse JSON result
        const vizData = JSON.parse(vizResult);

        // Extract node positions from Graphviz output
        // Graphviz uses points (1/72 inch), convert to canvas coordinates
        const positions = {{}};
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        vizData.objects.forEach(obj => {{
            if (obj.pos) {{
                // pos format: "x,y" in points
                const [x, y] = obj.pos.split(',').map(parseFloat);
                // Convert obj.name to string to match our node IDs
                positions[String(obj.name)] = [x, y];
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }}
        }});

        console.log('Viz.js positions:', positions);

        // Calculate extent for zoom
        const graph_extent = Math.max(maxX - minX, maxY - minY);

        // Center positions around origin
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        for (const node in positions) {{
            positions[node][0] -= centerX;
            positions[node][1] -= centerY;
        }}

        // Build node and edge data
        const nodesInfo = {json.dumps([{"id": str(node), "label": str(node), "attributes": graph.get_node(node).attributes if graph.get_node(node) else {}} for node in nodes])};
        const nodeData = nodesInfo.map(nodeInfo => ({{
            id: nodeInfo.id,
            label: nodeInfo.label,
            attributes: nodeInfo.attributes,
            x: positions[nodeInfo.id] ? positions[nodeInfo.id][0] : 0,
            y: positions[nodeInfo.id] ? positions[nodeInfo.id][1] : 0
        }}));

        console.log('Node data:', nodeData);

        const edgeList = {json.dumps([[str(e[0]), str(e[1])] for e in edges])};
        const edgeData = edgeList.map(edge => ({{
            source: edge[0],
            target: edge[1],
            sourceX: positions[edge[0]] ? positions[edge[0]][0] : 0,
            sourceY: positions[edge[0]] ? positions[edge[0]][1] : 0,
            targetX: positions[edge[1]] ? positions[edge[1]][0] : 0,
            targetY: positions[edge[1]] ? positions[edge[1]][1] : 0
        }}));

        console.log('Edge data:', edgeData);

        // Calculate initial zoom
        const initialZoom = (graph_extent > 1.6) ? (1.6 / graph_extent) : 0.7;

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');
        const resetBtn = document.getElementById('reset-btn');
        const zoomToggle = document.getElementById('zoom-toggle');
        const zoomModeText = document.getElementById('zoom-mode-text');

        // Viewport state
        let zoom = initialZoom;
        const initialPanX = 0;
        const initialPanY = 0;
        let panX = 0, panY = 0;
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;
        let hoveredNode = null;
        let zoomMode = 'semantic';  // Semantic zoom for Graphviz layouts (nodes stay constant size)

        // Transform world coords to screen coords
        function toScreen(x, y) {{
            return [
                (x * zoom + panX) * canvas.width / 2 + canvas.width / 2,
                (-y * zoom + panY) * canvas.height / 2 + canvas.height / 2
            ];
        }}

        // Transform screen coords to world coords
        function toWorld(screenX, screenY) {{
            const worldX = ((screenX - canvas.width / 2) / (canvas.width / 2) - panX) / zoom;
            const worldY = -((screenY - canvas.height / 2) / (canvas.height / 2) - panY) / zoom;
            return [worldX, worldY];
        }}

        // Find node at screen position
        function findNodeAt(screenX, screenY) {{
            const [worldX, worldY] = toWorld(screenX, screenY);
            const hitRadius = zoomMode === 'geometric'
                ? {node_size} * 0.015 * zoom
                : {node_size} * 0.015;

            for (const node of nodeData) {{
                const dx = node.x - worldX;
                const dy = node.y - worldY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist <= hitRadius) {{
                    return node;
                }}
            }}
            return null;
        }}

        // Draw arrow for directed edges
        function drawArrow(fromX, fromY, toX, toY, nodeRadius, arrowSize) {{
            const angle = Math.atan2(toY - fromY, toX - fromX);
            const arrowBackDist = arrowSize * 0.866;
            const lineEndX = toX - Math.cos(angle) * (nodeRadius + arrowBackDist);
            const lineEndY = toY - Math.sin(angle) * (nodeRadius + arrowBackDist);
            const arrowTipX = toX - Math.cos(angle) * nodeRadius;
            const arrowTipY = toY - Math.sin(angle) * nodeRadius;

            ctx.beginPath();
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(lineEndX, lineEndY);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(arrowTipX, arrowTipY);
            ctx.lineTo(
                arrowTipX - arrowSize * Math.cos(angle - Math.PI / 6),
                arrowTipY - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            ctx.lineTo(
                arrowTipX - arrowSize * Math.cos(angle + Math.PI / 6),
                arrowTipY - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            ctx.closePath();
            ctx.fillStyle = 'rgba(96, 165, 250, 1.0)';
            ctx.fill();
        }}

        // Render function
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const effectiveNodeSize = zoomMode === 'geometric' ? {node_size} * zoom : {node_size};
            const effectiveEdgeWidth = zoomMode === 'geometric' ? {edge_width} * zoom : {edge_width};
            const effectiveArrowSize = zoomMode === 'geometric' ? 10 * zoom : 10;

            // Draw edges
            ctx.strokeStyle = 'rgba(96, 165, 250, 0.6)';
            ctx.lineWidth = effectiveEdgeWidth;

            for (const edge of edgeData) {{
                const [x1, y1] = toScreen(edge.sourceX, edge.sourceY);
                const [x2, y2] = toScreen(edge.targetX, edge.targetY);

                if (isDirected) {{
                    drawArrow(x1, y1, x2, y2, effectiveNodeSize, effectiveArrowSize);
                }} else {{
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }}
            }}

            // Draw nodes
            for (const node of nodeData) {{
                const [x, y] = toScreen(node.x, node.y);
                const isHovered = hoveredNode && hoveredNode.id === node.id;

                ctx.fillStyle = isHovered ? '#e0f2fe' : '#f0f9ff';
                ctx.beginPath();
                ctx.arc(x, y, effectiveNodeSize, 0, 2 * Math.PI);
                ctx.fill();

                const gradient = ctx.createRadialGradient(
                    x - effectiveNodeSize * 0.3, y - effectiveNodeSize * 0.3, 0,
                    x, y, effectiveNodeSize
                );
                gradient.addColorStop(0, 'rgba(255, 255, 255, 0.6)');
                gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
                gradient.addColorStop(1, 'rgba(147, 197, 253, 0.2)');
                ctx.fillStyle = gradient;
                ctx.fill();

                ctx.strokeStyle = isHovered ? 'rgba(59, 130, 246, 0.5)' : 'rgba(96, 165, 250, 0.4)';
                ctx.lineWidth = isHovered ? 3 : 2;
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(x - effectiveNodeSize * 0.25, y - effectiveNodeSize * 0.25, effectiveNodeSize * 0.4, 0, Math.PI);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.lineWidth = isHovered ? 2 : 1.5;
                ctx.stroke();

                const nodeId = node.id;
                const idLength = nodeId.length;

                let fontSize;
                if (idLength === 1) {{
                    fontSize = effectiveNodeSize * 1.3;
                }} else if (idLength === 2) {{
                    fontSize = effectiveNodeSize * 1.0;
                }} else if (idLength === 3) {{
                    fontSize = effectiveNodeSize * 0.8;
                }} else {{
                    fontSize = effectiveNodeSize * 0.65;
                }}

                ctx.font = `600 ${{fontSize}}px -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif`;
                ctx.fontVariantNumeric = 'tabular-nums';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = isHovered ? '#1e40af' : '#3b82f6';
                ctx.fillText(nodeId, x, y);
            }}
        }}

        function resetView() {{
            zoom = initialZoom;
            panX = initialPanX;
            panY = initialPanY;
            render();
        }}

        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            if (isDragging) {{
                const dx = (e.clientX - lastMouseX) / canvas.width * 2;
                const dy = (e.clientY - lastMouseY) / canvas.height * 2;
                panX += dx;
                panY += dy;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
            }} else {{
                const node = findNodeAt(mouseX, mouseY);
                if (node !== hoveredNode) {{
                    hoveredNode = node;
                    if (node) {{
                        let tooltipText = node.id === node.label
                            ? `Node: ${{node.id}}`
                            : `Node: ${{node.id}}\\nLabel: ${{node.label}}`;

                        // Add attributes to tooltip
                        if (node.attributes && Object.keys(node.attributes).length > 0) {{
                            tooltipText += '\\n';
                            for (const [key, value] of Object.entries(node.attributes)) {{
                                tooltipText += `\\n${{key}}: ${{value}}`;
                            }}
                        }}

                        tooltip.textContent = tooltipText;
                        tooltip.style.whiteSpace = 'pre-line';
                        tooltip.style.left = `${{mouseX + 10}}px`;
                        tooltip.style.top = `${{mouseY + 10}}px`;
                        tooltip.style.opacity = '1';
                    }} else {{
                        tooltip.style.opacity = '0';
                    }}
                    render();
                }}
            }}
        }});

        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});

        canvas.addEventListener('mouseleave', () => {{
            hoveredNode = null;
            tooltip.style.opacity = '0';
            isDragging = false;
            render();
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.95 : 1.05;
            zoom *= zoomFactor;
            zoom = Math.max(0.1, Math.min(10, zoom));
            render();
        }});

        resetBtn.addEventListener('click', resetView);

        zoomToggle.addEventListener('click', () => {{
            zoomMode = zoomMode === 'semantic' ? 'geometric' : 'semantic';
            zoomToggle.classList.toggle('active');
            zoomModeText.textContent = zoomMode === 'semantic' ? 'Semantic' : 'Geometric';
            render();
        }});

        render();
    </script>
</body>
</html>"""

    return html


def _generate_dot_string(graph, nodes: list[str], edges: list[list[str]]) -> str:
    """
    Generate Graphviz DOT format string from graph.

    Converts an EdgeCase graph to DOT format for use with Graphviz layout engines
    (neato, dot, fdp, circo, twopi). The DOT format is a text-based graph description
    language that Graphviz can parse and render.

    Parameters
    ----------
    graph : Graph
        EdgeCase graph object
    nodes : list[str]
        List of node labels
    edges : list[list[str]]
        List of [source, target] edge pairs

    Returns
    -------
    str
        DOT format string suitable for Graphviz rendering

    Examples
    --------
    >>> G = Graph()
    >>> G.add_edge("A", "B")
    >>> dot = _generate_dot_string(G, ["A", "B"], [["A", "B"]])
    >>> print(dot)
    graph {
      "A";
      "B";
      "A" -- "B";
    }
    """
    is_directed = graph.is_directed
    graph_type = "digraph" if is_directed else "graph"
    edge_op = "->" if is_directed else "--"

    lines = [f"{graph_type} {{"]

    # Graph-level attributes
    lines.append("  graph [outputorder=edgesfirst, bgcolor=white];")

    # Node defaults
    lines.append(
        "  node [fontname=Helvetica, fontsize=10, shape=circle, fixedsize=true, width=0.5, style=filled, fillcolor=aliceblue, color=gray50, fontcolor=gray50];"
    )

    # Edge defaults
    if is_directed:
        lines.append(
            "  edge [fontname=Helvetica, fontsize=8, len=1.5, color=gray60, penwidth=1.5, arrowsize=0.5];"
        )
    else:
        lines.append(
            "  edge [fontname=Helvetica, fontsize=8, len=1.5, color=gray60, penwidth=1.5];"
        )

    # Add nodes with individual styling from attributes
    for node in nodes:
        node_obj = graph.get_node(node)
        attrs = node_obj.attributes if node_obj else {}

        # Map common attribute names to Graphviz DOT attributes
        dot_attrs = []

        # Size/width (use 'size' or 'width' attribute)
        if "size" in attrs:
            dot_attrs.append(f"width={attrs['size']}")
        elif "width" in attrs:
            dot_attrs.append(f"width={attrs['width']}")

        # Fill color (use 'fillcolor' or 'fill' attribute)
        if "fillcolor" in attrs:
            dot_attrs.append(f'fillcolor="{attrs["fillcolor"]}"')
        elif "fill" in attrs:
            dot_attrs.append(f'fillcolor="{attrs["fill"]}"')

        # Border color (use 'color' attribute)
        if "color" in attrs:
            dot_attrs.append(f'color="{attrs["color"]}"')

        # Shape (use 'shape' attribute)
        if "shape" in attrs:
            dot_attrs.append(f"shape={attrs['shape']}")

        # Style (use 'style' attribute)
        if "style" in attrs:
            dot_attrs.append(f'style="{attrs["style"]}"')

        # Font color (use 'fontcolor' attribute)
        if "fontcolor" in attrs:
            dot_attrs.append(f'fontcolor="{attrs["fontcolor"]}"')

        # Pen width (use 'penwidth' attribute)
        if "penwidth" in attrs:
            dot_attrs.append(f"penwidth={attrs['penwidth']}")

        # Build node line with attributes
        if dot_attrs:
            attrs_str = ", ".join(dot_attrs)
            lines.append(f'  "{node}" [{attrs_str}];')
        else:
            lines.append(f'  "{node}";')

    # Add edges
    for src, tgt in edges:
        lines.append(f'  "{src}" {edge_op} "{tgt}";')

    lines.append("}")

    return "\n".join(lines)


def _generate_graphviz_svg_html(
    graph,
    nodes: list[str],
    edges: list[list[str]],
    dot_string: str,
    viz_js_code: str,
    layout_engine: str,
    width: int,
    height: int,
) -> str:
    """
    Generate HTML with Graphviz SVG output embedded.

    This approach uses Graphviz's native SVG rendering which includes:
    - Production-quality node sizing and spacing
    - Proper edge routing and arrow rendering
    - Built-in styling that looks professional
    - No need to reimplement rendering logic

    Parameters
    ----------
    graph : Graph
        EdgeCase graph object
    dot_string : str
        DOT format representation of graph
    viz_js_code : str
        Complete viz.js JavaScript code (2MB)
    layout_engine : str
        Graphviz engine: 'neato', 'dot', 'fdp', 'circo', or 'twopi'
    width : int
        Container width in pixels
    height : int
        Container height in pixels

    Returns
    -------
    str
        Complete HTML document with embedded SVG
    """
    graph_type = "Directed" if graph.is_directed else "Undirected"
    num_nodes = len(graph.get_node_ids())
    num_edges = graph.num_edges()

    # Generate unique ID suffix for this graph instance to avoid conflicts in notebooks
    import json
    import time

    unique_id = str(int(time.time() * 1000000) % 1000000)  # Use microseconds modulo 1M

    # Collect node data with attributes
    nodes_data = []
    for node in nodes:
        node_obj = graph.get_node(node)
        nodes_data.append(
            {
                "id": str(node),
                "label": str(node),
                "attributes": node_obj.attributes if node_obj else {},
            }
        )

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges)
    is_directed = "true" if graph.is_directed else "false"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EdgeCase Graph - {layout_engine}</title>
    <style>
        #edgecase-viz-wrapper-{unique_id} {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 50%, #80cbc4 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: {height + 100}px;
            padding: 20px;
            margin: 0;
        }}

        #edgecase-viz-wrapper-{unique_id} #viz-container-{unique_id} {{
            background: linear-gradient(to bottom, #f1f8f7 0%, #e8f5f3 100%);
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 77, 64, 0.15);
            padding: 24px;
            max-width: {width + 48}px;
            border: 1px solid #b2dfdb;
        }}

        #edgecase-viz-wrapper-{unique_id} #info-{unique_id} {{
            text-align: center;
            padding: 12px;
            color: #00695c;
            font-size: 14px;
            margin-bottom: 16px;
            border-bottom: 2px solid #b2dfdb;
            background: rgba(178, 223, 219, 0.2);
            border-radius: 6px;
        }}

        #edgecase-viz-wrapper-{unique_id} #svg-container-{unique_id} {{
            width: {width}px;
            height: {height}px;
            overflow: auto;
            border: 2px solid #80cbc4;
            border-radius: 6px;
            position: relative;
            box-shadow: inset 0 1px 3px rgba(0, 77, 64, 0.1);
        }}

        /* Ambient smoke effect background */
        #edgecase-viz-wrapper-{unique_id} #svg-container-{unique_id}::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(ellipse at 20% 30%, rgba(200, 220, 240, 0.6) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 70%, rgba(240, 210, 230, 0.5) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 80%, rgba(220, 240, 210, 0.5) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 20%, rgba(210, 230, 245, 0.6) 0%, transparent 50%),
                white;
            pointer-events: none;
            z-index: 0;
            animation: smokeMove-{unique_id} 60s ease-in-out infinite;
        }}

        @keyframes smokeMove-{unique_id} {{
            0%, 100% {{
                background-position: 0% 0%, 100% 100%, 50% 100%, 100% 0%;
                background-size: 150% 150%, 140% 140%, 160% 160%, 130% 130%;
            }}
            25% {{
                background-position: 30% 20%, 80% 70%, 40% 90%, 90% 10%;
                background-size: 140% 140%, 160% 160%, 150% 150%, 145% 145%;
            }}
            50% {{
                background-position: 50% 40%, 60% 50%, 30% 70%, 80% 30%;
                background-size: 160% 160%, 150% 150%, 140% 140%, 155% 155%;
            }}
            75% {{
                background-position: 20% 60%, 90% 30%, 60% 80%, 70% 50%;
                background-size: 145% 145%, 135% 135%, 155% 155%, 140% 140%;
            }}
        }}

        #edgecase-viz-wrapper-{unique_id} #svg-container-{unique_id} svg {{
            width: 100%;
            height: 100%;
            position: relative;
            z-index: 1;
        }}

        #edgecase-viz-wrapper-{unique_id} .controls {{
            margin-top: 16px;
            text-align: center;
        }}

        #edgecase-viz-wrapper-{unique_id} button {{
            background: linear-gradient(to bottom, #4db6ac 0%, #26a69a 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin: 0 6px;
            box-shadow: 0 2px 4px rgba(0, 77, 64, 0.2);
            transition: all 0.2s ease;
        }}

        #edgecase-viz-wrapper-{unique_id} button:hover {{
            background: linear-gradient(to bottom, #26a69a 0%, #00897b 100%);
            box-shadow: 0 3px 6px rgba(0, 77, 64, 0.3);
            transform: translateY(-1px);
        }}

        #edgecase-viz-wrapper-{unique_id} button:active {{
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0, 77, 64, 0.2);
        }}

        /* Custom tooltip styling */
        #edgecase-viz-wrapper-{unique_id} #custom-tooltip-{unique_id} {{
            position: fixed;
            background: rgba(0, 77, 64, 0.95);
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
            pointer-events: none;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.15s ease;
            white-space: nowrap;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}

        #edgecase-viz-wrapper-{unique_id} #custom-tooltip-{unique_id}.visible {{
            opacity: 1;
        }}
    </style>
</head>
<body>
    <div id="edgecase-viz-wrapper-{unique_id}">
    <div id="viz-container-{unique_id}">
        <div id="info-{unique_id}">
            EdgeCase Graph | {graph_type} | {num_nodes} nodes, {num_edges} edges | Layout: {layout_engine}
        </div>
        <div id="svg-container-{unique_id}"></div>
        <div class="controls">
            <button id="zoom-in-btn-{unique_id}">Zoom In</button>
            <button id="reset-btn-{unique_id}">Reset</button>
            <button id="zoom-out-btn-{unique_id}">Zoom Out</button>
        </div>
        <div id="custom-tooltip-{unique_id}"></div>
    </div>

    <script>
        (function() {{
            // CRITICAL: Disable AMD/RequireJS detection so viz.js uses global scope
            // Save original define if it exists
            var originalDefine = (typeof define !== 'undefined') ? define : undefined;
            var originalRequire = (typeof require !== 'undefined') ? require : undefined;

            // Temporarily undefine to force viz.js to use global scope
            if (typeof define !== 'undefined') {{
                define = undefined;
            }}
            if (typeof require !== 'undefined' && typeof require.config === 'function') {{
                require = undefined;
            }}

            // Embed viz.js inline - will now use global.Viz = Viz branch
            {viz_js_code}

            // Restore define/require after viz.js loads
            if (originalDefine !== undefined) {{
                define = originalDefine;
            }}
            if (originalRequire !== undefined) {{
                require = originalRequire;
            }}

            // Get Viz reference (should be in global scope now)
            var VizFunc = (typeof Viz !== 'undefined') ? Viz : window.Viz;

            if (!VizFunc) {{
                console.error('ERROR: Viz still not available after neutralizing AMD');
                document.getElementById('svg-container-{unique_id}').innerHTML = '<p style="color: red;">Error: Could not load Graphviz renderer</p>';
                return;
            }}

            const dotString = `{dot_string}`;
            const layoutEngine = '{layout_engine}';
            const svgContainer = document.getElementById('svg-container-{unique_id}');

            // Node data with attributes for tooltips
            const nodesData = {nodes_json};
            const nodesMap = {{}};
            nodesData.forEach(node => {{
                nodesMap[node.id] = node;
            }});

            // Edge data for tooltips: [[source, target], ...]
            const edgesData = {edges_json};
            const isDirected = {is_directed};

            try {{
                // Generate SVG with Graphviz
                const svg = VizFunc(dotString, {{
                    format: 'svg',
                    engine: layoutEngine
                }});

                svgContainer.innerHTML = svg;

                // SVG will auto-scale to container via CSS (width: 100%; height: 100%)
                const svgElement = svgContainer.querySelector('svg');

                // Make SVG background transparent so smoke effect shows through
                const bgPolygon = svgElement.querySelector('g.graph polygon');
                if (bgPolygon) {{
                    bgPolygon.setAttribute('fill', 'none');
                    bgPolygon.setAttribute('fill-opacity', '0');
                }}

                // Create SVG filter for glassy effect with background color amplification
                const defs = svgElement.querySelector('defs') || svgElement.insertBefore(
                    document.createElementNS('http://www.w3.org/2000/svg', 'defs'),
                    svgElement.firstChild
                );

                const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
                filter.setAttribute('id', 'glassEffect-{unique_id}');
                filter.setAttribute('x', '-50%');
                filter.setAttribute('y', '-50%');
                filter.setAttribute('width', '200%');
                filter.setAttribute('height', '200%');

                // Blur the node for soft glassy edges
                const blur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
                blur.setAttribute('in', 'SourceGraphic');
                blur.setAttribute('stdDeviation', '0.5');
                blur.setAttribute('result', 'blurred');

                // Merge blurred with original to keep stroke crisp
                const merge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
                const mergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
                mergeNode1.setAttribute('in', 'blurred');
                const mergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
                mergeNode2.setAttribute('in', 'SourceGraphic');

                merge.appendChild(mergeNode1);
                merge.appendChild(mergeNode2);
                filter.appendChild(blur);
                filter.appendChild(merge);
                defs.appendChild(filter);

                // Apply glassy/liquid effect to nodes
                const nodeGroups = svgElement.querySelectorAll('g.node');
                const nodeEllipses = [];
                nodeGroups.forEach(nodeGroup => {{
                    const ellipse = nodeGroup.querySelector('ellipse');
                    if (ellipse) {{
                        // Store ellipse reference for slider control
                        nodeEllipses.push(ellipse);
                        // Make node fill semi-transparent for glassy effect (start at 0.7 for 2.5x)
                        ellipse.setAttribute('fill-opacity', '0.7');
                        // Apply custom filter to node group (soft blur)
                        nodeGroup.style.filter = 'url(#glassEffect-{unique_id})';
                        // Keep stroke opaque for definition
                        ellipse.setAttribute('stroke-opacity', '1');
                    }}
                }});

            let scale = 1;
            let panX = 0;
            let panY = 0;
            let isDragging = false;
            let startX = 0;
            let startY = 0;

            // Custom tooltip element
            const customTooltip = document.getElementById('custom-tooltip-{unique_id}');

            // Function to setup custom tooltips
            function setupCustomTooltips(svg) {{
                // Remove ALL native SVG title elements (including background)
                svg.querySelectorAll('title').forEach(title => title.remove());

                // Setup custom tooltips for nodes
                const nodes = svg.querySelectorAll('g.node');
                nodes.forEach(node => {{
                    // The first text element usually contains the node label
                    const textElement = node.querySelector('text');
                    const nodeId = textElement ? textElement.textContent.trim() : 'Unknown';

                    // Get node data with attributes
                    const nodeData = nodesMap[nodeId];

                    node.addEventListener('mouseenter', (e) => {{
                        let tooltipText = `Node: ${{nodeId}}`;

                        // Add attributes if available
                        if (nodeData && nodeData.attributes && Object.keys(nodeData.attributes).length > 0) {{
                            for (const [key, value] of Object.entries(nodeData.attributes)) {{
                                tooltipText += `\\n${{key}}: ${{value}}`;
                            }}
                        }}

                        customTooltip.textContent = tooltipText;
                        customTooltip.style.whiteSpace = 'pre-line';
                        customTooltip.classList.add('visible');
                    }});

                    node.addEventListener('mousemove', (e) => {{
                        customTooltip.style.left = (e.clientX + 10) + 'px';
                        customTooltip.style.top = (e.clientY + 10) + 'px';
                    }});

                    node.addEventListener('mouseleave', () => {{
                        customTooltip.classList.remove('visible');
                    }});
                }});

                // Setup custom tooltips for edges
                const edges = svg.querySelectorAll('g.edge');
                edges.forEach((edge, index) => {{
                    // Get edge data (source -> target)
                    let source = '';
                    let target = '';
                    let edgeId = index;  // Use index as edge ID

                    if (index < edgesData.length) {{
                        source = edgesData[index][0];
                        target = edgesData[index][1];
                    }}

                    // Format: Edge: <edge_id> (<source> → <target>) or (<source> — <target>)
                    // Use → (U+2192) for directed, — (U+2014 em dash) for undirected
                    const separator = isDirected ? '→' : '—';
                    const edgeLabel = `Edge: ${{edgeId}} (${{source}} ${{separator}} ${{target}})`;

                    // Create a wider invisible hit area for easier hovering
                    const visiblePath = edge.querySelector('path');
                    if (visiblePath) {{
                        const hitArea = visiblePath.cloneNode(true);
                        hitArea.setAttribute('stroke', 'transparent');
                        hitArea.setAttribute('stroke-width', '12');  // Much wider hit area
                        hitArea.setAttribute('fill', 'none');
                        hitArea.setAttribute('pointer-events', 'stroke');
                        // Insert the hit area before the visible path
                        visiblePath.parentNode.insertBefore(hitArea, visiblePath);

                        // Add hover effect to visible path
                        hitArea.addEventListener('mouseenter', () => {{
                            visiblePath.style.strokeWidth = '2';
                            visiblePath.style.stroke = '#00897b';  // Teal highlight
                        }});

                        hitArea.addEventListener('mouseleave', () => {{
                            visiblePath.style.strokeWidth = '';
                            visiblePath.style.stroke = '';
                        }});
                    }}

                    edge.addEventListener('mouseenter', (e) => {{
                        customTooltip.textContent = edgeLabel;
                        customTooltip.classList.add('visible');
                    }});

                    edge.addEventListener('mousemove', (e) => {{
                        customTooltip.style.left = (e.clientX + 10) + 'px';
                        customTooltip.style.top = (e.clientY + 10) + 'px';
                    }});

                    edge.addEventListener('mouseleave', () => {{
                        customTooltip.classList.remove('visible');
                    }});
                }});
            }}

            // Setup custom tooltips
            setupCustomTooltips(svgElement);

            // Smooth transform with transition
            function updateTransform(smooth = false) {{
                if (smooth) {{
                    svgElement.style.transition = 'transform 0.3s ease-out';
                }} else {{
                    svgElement.style.transition = 'none';
                }}
                svgElement.style.transformOrigin = '0 0';
                svgElement.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
            }}

            // Zoom centered on a specific point
            function zoomAt(clientX, clientY, delta) {{
                const rect = svgContainer.getBoundingClientRect();
                const x = clientX - rect.left;
                const y = clientY - rect.top;

                // Point in SVG space before zoom
                const svgX = (x - panX) / scale;
                const svgY = (y - panY) / scale;

                // Apply zoom
                const newScale = scale * delta;

                // Clamp zoom between 0.1x and 10x
                if (newScale < 0.1 || newScale > 10) {{
                    return;
                }}

                scale = newScale;

                // Adjust pan to keep the point under cursor
                panX = x - svgX * scale;
                panY = y - svgY * scale;

                updateTransform(true);
            }}

            // Zoom controls - smooth button zoom centered on viewport
            document.getElementById('zoom-in-btn-{unique_id}').addEventListener('click', () => {{
                const rect = svgContainer.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                zoomAt(centerX, centerY, 1.2);
            }});

            document.getElementById('zoom-out-btn-{unique_id}').addEventListener('click', () => {{
                const rect = svgContainer.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                zoomAt(centerX, centerY, 1 / 1.2);
            }});

            document.getElementById('reset-btn-{unique_id}').addEventListener('click', () => {{
                scale = 1;
                panX = 0;
                panY = 0;
                updateTransform(true);
            }});

            // Mouse wheel zoom - centered on cursor, gentle sensitivity
            svgContainer.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.97 : 1.03;  // 3% per scroll (gentle zoom)
                zoomAt(e.clientX, e.clientY, delta);
            }});

            // Pan with mouse drag - smooth dragging
            svgContainer.addEventListener('mousedown', (e) => {{
                isDragging = true;
                startX = e.clientX - panX;
                startY = e.clientY - panY;
                svgContainer.style.cursor = 'grabbing';
            }});

            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    panX = e.clientX - startX;
                    panY = e.clientY - startY;
                    updateTransform(false);  // No smooth transition while dragging
                }}
            }});

            document.addEventListener('mouseup', () => {{
                isDragging = false;
                svgContainer.style.cursor = 'grab';
            }});

            svgContainer.style.cursor = 'grab';

            }} catch (error) {{
                console.error('ERROR generating layout:', error);
                svgContainer.innerHTML = '<p style="color: red; padding: 20px;">Error generating layout: ' + error.message + '</p>';
            }}
        }})(); // End of IIFE
    </script>
    </div> <!-- end edgecase-viz-wrapper-{unique_id} -->
</body>
</html>"""

    return html


def _generate_webgl_html(
    graph,
    nodes: list[str],
    edges: list[list[str]],
    positions: dict,
    graph_extent: float,
    width: int,
    height: int,
    node_size: float,
    edge_width: float,
) -> str:
    """Generate standalone HTML with WebGL renderer (deck.gl style)."""

    # Get graph properties
    is_directed = graph.is_directed
    graph_type = "directed" if is_directed else "undirected"

    # Calculate initial zoom to fit graph in viewport
    # Target: graph should fit within 1.6 viewport units (leaving 20% padding)
    if graph_extent > 1.6:
        initial_zoom = 1.6 / graph_extent
    else:
        initial_zoom = 0.7  # Default zoom for small graphs

    # Convert to JSON-serializable format
    # Note: In EdgeCase, node strings are typically labels
    node_data = [
        {
            "id": node,
            "label": node,  # In EdgeCase, nodes() returns labels
            "x": positions[node][0],
            "y": positions[node][1],
        }
        for node in nodes
    ]

    edge_data = [
        {
            "source": src,
            "target": tgt,
            "sourceX": positions[src][0],
            "sourceY": positions[src][1],
            "targetX": positions[tgt][0],
            "targetY": positions[tgt][1],
        }
        for src, tgt in edges
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EdgeCase Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        #viz-container {{
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: inline-block;
            border-radius: 8px;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        #info {{
            margin-bottom: 10px;
            font-size: 14px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        #canvas-wrapper {{
            position: relative;
            display: inline-block;
        }}
        #canvas {{
            border: 1px solid #ddd;
            background: #f5f5f5;
            cursor: grab;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        #canvas:active {{
            cursor: grabbing;
        }}
        #reset-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            color: #333;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        #reset-btn .icon {{
            font-size: 18px;
            line-height: 1;
        }}
        #reset-btn:hover {{
            background: #fff;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        #stats {{
            margin-top: 10px;
            font-size: 12px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        #zoom-mode-control {{
            margin-top: 15px;
            padding: 10px 15px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 12px;
            color: #fff;
        }}
        #zoom-mode-label {{
            font-weight: 500;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 24px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        .toggle-switch.active {{
            background: rgba(59, 130, 246, 0.6);
        }}
        .toggle-slider {{
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: #fff;
            border-radius: 50%;
            transition: transform 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .toggle-switch.active .toggle-slider {{
            transform: translateX(26px);
        }}
        #zoom-mode-text {{
            min-width: 100px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div id="viz-container">
        <div id="container">
            <div id="info">
                EdgeCase Graph | {graph_type} | {len(nodes)} nodes, {len(edges)} edges
            </div>
            <div id="canvas-wrapper">
                <canvas id="canvas" width="{width}" height="{height}"></canvas>
                <button id="reset-btn" title="Reset view">
                    <span class="icon">⟲</span>
                    <span>Reset</span>
                </button>
                <div id="tooltip"></div>
            </div>
            <div id="stats">Drag to pan • Scroll to zoom • Hover for node info</div>
            <div id="zoom-mode-control">
                <span id="zoom-mode-label">Zoom Mode:</span>
                <div class="toggle-switch active" id="zoom-toggle">
                    <div class="toggle-slider"></div>
                </div>
                <span id="zoom-mode-text">Geometric</span>
            </div>
        </div>
    </div>

    <script>
        const nodes = {json.dumps(node_data)};
        const edges = {json.dumps(edge_data)};
        const isDirected = {str(is_directed).lower()};
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');
        const resetBtn = document.getElementById('reset-btn');
        const zoomToggle = document.getElementById('zoom-toggle');
        const zoomModeText = document.getElementById('zoom-mode-text');

        // Debug: Log node positions and edge lengths
        console.log('=== Graph Layout Debug Info ===');
        console.log('Number of nodes:', nodes.length);
        console.log('Number of edges:', edges.length);
        console.log('\\nNode positions:');
        nodes.forEach(node => {{
            console.log(`  Node ${{node.id}}: x=${{node.x.toFixed(4)}}, y=${{node.y.toFixed(4)}}`);
        }});

        // Calculate edge lengths
        console.log('\\nEdge lengths:');
        edges.forEach(edge => {{
            const dx = edge.targetX - edge.sourceX;
            const dy = edge.targetY - edge.sourceY;
            const length = Math.sqrt(dx * dx + dy * dy);
            console.log(`  ${{edge.source}} → ${{edge.target}}: ${{length.toFixed(4)}}`);
        }});

        // Calculate bounding box
        const allX = nodes.map(n => n.x);
        const allY = nodes.map(n => n.y);
        const minX = Math.min(...allX);
        const maxX = Math.max(...allX);
        const minY = Math.min(...allY);
        const maxY = Math.max(...allY);
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        console.log('\\nBounding box:');
        console.log(`  X range: [${{minX.toFixed(4)}}, ${{maxX.toFixed(4)}}] = ${{rangeX.toFixed(4)}}`);
        console.log(`  Y range: [${{minY.toFixed(4)}}, ${{maxY.toFixed(4)}}] = ${{rangeY.toFixed(4)}}`);
        console.log('================================\\n');

        // Viewport state - calculate initial zoom to fit large graphs
        const initialZoom = {initial_zoom:.4f};
        let zoom = initialZoom;
        let panX = 0, panY = 0;
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;
        let hoveredNode = null;
        let zoomMode = 'geometric';  // 'semantic' or 'geometric' - default to geometric

        // Transform world coords to screen coords
        function toScreen(x, y) {{
            return [
                (x * zoom + panX) * canvas.width / 2 + canvas.width / 2,
                (-y * zoom + panY) * canvas.height / 2 + canvas.height / 2
            ];
        }}

        // Transform screen coords to world coords
        function toWorld(screenX, screenY) {{
            const worldX = ((screenX - canvas.width / 2) / (canvas.width / 2) - panX) / zoom;
            const worldY = -((screenY - canvas.height / 2) / (canvas.height / 2) - panY) / zoom;
            return [worldX, worldY];
        }}

        // Find node at screen position
        function findNodeAt(screenX, screenY) {{
            const [worldX, worldY] = toWorld(screenX, screenY);
            // Hit radius adjusts based on zoom mode
            const hitRadius = zoomMode === 'geometric'
                ? {node_size} * 0.015 * zoom
                : {node_size} * 0.015;

            for (const node of nodes) {{
                const dx = node.x - worldX;
                const dy = node.y - worldY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist <= hitRadius) {{
                    return node;
                }}
            }}
            return null;
        }}

        // Draw arrow for directed edges
        function drawArrow(fromX, fromY, toX, toY, nodeRadius, arrowSize) {{
            const angle = Math.atan2(toY - fromY, toX - fromX);

            // Calculate arrowhead back distance (how far back from tip the base is)
            const arrowBackDist = arrowSize * 0.866;  // cos(30°) ≈ 0.866 for proper arrow shape

            // Shorten line to account for both node radius AND arrowhead base
            const lineEndX = toX - Math.cos(angle) * (nodeRadius + arrowBackDist);
            const lineEndY = toY - Math.sin(angle) * (nodeRadius + arrowBackDist);

            // Arrow tip position (at edge of node)
            const arrowTipX = toX - Math.cos(angle) * nodeRadius;
            const arrowTipY = toY - Math.sin(angle) * nodeRadius;

            // Draw line (shortened to meet arrowhead base)
            ctx.beginPath();
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(lineEndX, lineEndY);
            ctx.stroke();

            // Draw arrowhead with proper shape
            ctx.beginPath();
            ctx.moveTo(arrowTipX, arrowTipY);  // Arrow tip
            ctx.lineTo(
                arrowTipX - arrowSize * Math.cos(angle - Math.PI / 6),
                arrowTipY - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            ctx.lineTo(
                arrowTipX - arrowSize * Math.cos(angle + Math.PI / 6),
                arrowTipY - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            ctx.closePath();

            // Use fully opaque fill to match edge color cleanly
            ctx.fillStyle = 'rgba(96, 165, 250, 1.0)';
            ctx.fill();
        }}

        // Render function (60fps with requestAnimationFrame)
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Calculate effective sizes based on zoom mode
            const effectiveNodeSize = zoomMode === 'geometric' ? {node_size} * zoom : {node_size};
            const effectiveEdgeWidth = zoomMode === 'geometric' ? {edge_width} * zoom : {edge_width};
            const effectiveArrowSize = zoomMode === 'geometric' ? 10 * zoom : 10;

            // Draw edges
            ctx.strokeStyle = 'rgba(96, 165, 250, 0.6)';  // Match node border color harmoniously
            ctx.lineWidth = effectiveEdgeWidth;

            for (const edge of edges) {{
                const [x1, y1] = toScreen(edge.sourceX, edge.sourceY);
                const [x2, y2] = toScreen(edge.targetX, edge.targetY);

                if (isDirected) {{
                    drawArrow(x1, y1, x2, y2, effectiveNodeSize, effectiveArrowSize);
                }} else {{
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }}
            }}

            // Draw nodes
            for (const node of nodes) {{
                const [x, y] = toScreen(node.x, node.y);
                const isHovered = hoveredNode && hoveredNode.id === node.id;

                // Node circle with light fill
                ctx.fillStyle = isHovered ? '#e0f2fe' : '#f0f9ff';
                ctx.beginPath();
                ctx.arc(x, y, effectiveNodeSize, 0, 2 * Math.PI);
                ctx.fill();

                // Subtle gradient overlay for depth
                const gradient = ctx.createRadialGradient(
                    x - effectiveNodeSize * 0.3, y - effectiveNodeSize * 0.3, 0,
                    x, y, effectiveNodeSize
                );
                gradient.addColorStop(0, 'rgba(255, 255, 255, 0.6)');
                gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
                gradient.addColorStop(1, 'rgba(147, 197, 253, 0.2)');
                ctx.fillStyle = gradient;
                ctx.fill();

                // Translucent glassy border with shimmer effect
                ctx.strokeStyle = isHovered ? 'rgba(59, 130, 246, 0.5)' : 'rgba(96, 165, 250, 0.4)';
                ctx.lineWidth = isHovered ? 3 : 2;
                ctx.stroke();

                // Top shimmer highlight for glass effect
                ctx.beginPath();
                ctx.arc(x - effectiveNodeSize * 0.25, y - effectiveNodeSize * 0.25, effectiveNodeSize * 0.4, 0, Math.PI);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.lineWidth = isHovered ? 2 : 1.5;
                ctx.stroke();

                // Draw node ID text in center
                const nodeId = node.id;
                const idLength = nodeId.length;

                // Scale font size based on ID length and node size (larger sizes for readability)
                let fontSize;
                if (idLength === 1) {{
                    fontSize = effectiveNodeSize * 1.3;
                }} else if (idLength === 2) {{
                    fontSize = effectiveNodeSize * 1.0;
                }} else if (idLength === 3) {{
                    fontSize = effectiveNodeSize * 0.8;
                }} else {{
                    fontSize = effectiveNodeSize * 0.65;
                }}

                ctx.font = `600 ${{fontSize}}px -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif`;
                ctx.fontVariantNumeric = 'tabular-nums';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = isHovered ? '#1e40af' : '#3b82f6';
                ctx.fillText(nodeId, x, y);
            }}
        }}

        // Reset view to initial state
        function resetView() {{
            zoom = initialZoom;
            panX = initialPanX;
            panY = initialPanY;
            render();
        }}

        // Mouse interaction
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            if (isDragging) {{
                const dx = (e.clientX - lastMouseX) / canvas.width * 2;
                const dy = (e.clientY - lastMouseY) / canvas.height * 2;
                panX += dx;
                panY += dy;  // Changed from -= to += to fix inverted Y panning
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
            }} else {{
                // Check for hover
                const node = findNodeAt(mouseX, mouseY);
                if (node !== hoveredNode) {{
                    hoveredNode = node;
                    if (node) {{
                        // Show ID and label in tooltip (if they differ)
                        const tooltipText = node.id === node.label
                            ? `Node: ${{node.id}}`
                            : `Node: ${{node.id}}\\nLabel: ${{node.label}}`;
                        tooltip.textContent = tooltipText;
                        tooltip.style.whiteSpace = 'pre-line';
                        // Position tooltip relative to canvas
                        tooltip.style.left = `${{mouseX + 10}}px`;
                        tooltip.style.top = `${{mouseY + 10}}px`;
                        tooltip.style.opacity = '1';
                    }} else {{
                        tooltip.style.opacity = '0';
                    }}
                    render();
                }}
            }}
        }});

        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});

        canvas.addEventListener('mouseleave', () => {{
            hoveredNode = null;
            tooltip.style.opacity = '0';
            isDragging = false;
            render();
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            // Reduced sensitivity: 0.95/1.05 instead of 0.9/1.1
            const zoomFactor = e.deltaY > 0 ? 0.95 : 1.05;
            zoom *= zoomFactor;
            zoom = Math.max(0.1, Math.min(10, zoom));
            render();
        }});

        resetBtn.addEventListener('click', resetView);

        // Zoom mode toggle
        zoomToggle.addEventListener('click', () => {{
            zoomMode = zoomMode === 'semantic' ? 'geometric' : 'semantic';
            zoomToggle.classList.toggle('active');
            zoomModeText.textContent = zoomMode === 'semantic' ? 'Semantic' : 'Geometric';
            render();
        }});

        // Initial render
        render();
    </script>
</body>
</html>"""

    return html


# Convenience export
__all__ = ["plot_graph"]
