import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional


def plot_graph(
    graph,
    output_path: Optional[str] = None,
    open_browser: bool = True,
    width: int = 800,
    height: int = 600,
    node_size: float = 20.0,
    edge_width: float = 2.0,
    layout: str = "force",
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
    layout : str, default "force"
        Layout algorithm: "force", "circular", "random"

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
    >>> plot_graph(G)
    """
    # Extract graph data (fast - direct from Rust)
    nodes = graph.nodes()
    edges = []

    # Build edge list efficiently
    seen = set()
    for node in nodes:
        for neighbor in graph.neighbors(node):
            edge_key = tuple(sorted([node, neighbor]))
            if edge_key not in seen:
                edges.append([node, neighbor])
                seen.add(edge_key)

    # Compute layout
    if layout == "force":
        positions = _force_directed_layout(nodes, edges, iterations=50)
    elif layout == "circular":
        positions = _circular_layout(nodes)
    else:
        positions = _random_layout(nodes)

    # Generate HTML with embedded WebGL renderer
    html = _generate_webgl_html(
        graph,
        nodes,
        edges,
        positions,
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
    # Use a large area to avoid cramping
    area = n * 50.0  # Much larger area for natural spreading
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

    # Normalize positions to fit in [-0.9, 0.9] range (leave some padding)
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

        # Use the larger range to maintain aspect ratio
        max_range = max(range_x, range_y)

        # Scale to fit in [-0.9, 0.9] (0.9 to leave padding)
        scale = 1.8 / max_range if max_range > 0 else 1.0

        for node in nodes:
            positions[node][0] = (positions[node][0] - center_x) * scale
            positions[node][1] = (positions[node][1] - center_y) * scale

    return positions


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


def _generate_webgl_html(
    graph,
    nodes: list[str],
    edges: list[list[str]],
    positions: dict,
    width: int,
    height: int,
    node_size: float,
    edge_width: float,
) -> str:
    """Generate standalone HTML with WebGL renderer (deck.gl style)."""

    # Get graph properties
    is_directed = graph.is_directed
    graph_type = "directed" if is_directed else "undirected"

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

        // Viewport state (start zoomed out slightly for padding)
        let zoom = 0.85;
        let panX = 0, panY = 0;
        const initialZoom = 0.85;
        const initialPanX = 0;
        const initialPanY = 0;
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
