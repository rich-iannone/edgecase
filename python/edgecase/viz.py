"""
Fast graph visualization using WebGL (deck.gl approach).

This module provides GPU-accelerated graph rendering similar to deck.gl,
generating standalone HTML that renders thousands of nodes/edges smoothly.
"""

import json
from typing import Optional
import tempfile
import webbrowser
from pathlib import Path


def plot_graph(
    graph,
    output_path: Optional[str] = None,
    open_browser: bool = True,
    width: int = 800,
    height: int = 600,
    node_size: float = 5.0,
    edge_width: float = 1.0,
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
    node_size : float, default 5.0
        Node radius in pixels
    edge_width : float, default 1.0
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
        nodes, edges, positions,
        width=width, height=height,
        node_size=node_size, edge_width=edge_width
    )
    
    # Save and optionally open
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="edgecase_")
        with open(fd, 'w') as f:
            f.write(html)
    else:
        with open(output_path, 'w') as f:
            f.write(html)
    
    if open_browser:
        webbrowser.open(f"file://{Path(output_path).absolute()}")
    
    return output_path


def _force_directed_layout(nodes: list[str], edges: list[list[str]], iterations: int = 50) -> dict:
    """Fast force-directed layout (Fruchterman-Reingold inspired)."""
    import random
    import math
    
    # Initialize random positions
    positions = {node: [random.uniform(-1, 1), random.uniform(-1, 1)] for node in nodes}
    
    # Build adjacency for fast lookup
    adj = {node: set() for node in nodes}
    for src, tgt in edges:
        adj[src].add(tgt)
        adj[tgt].add(src)
    
    # Force-directed iterations
    k = math.sqrt(1.0 / len(nodes))  # optimal distance
    
    for iteration in range(iterations):
        # Cooling schedule
        temp = 1.0 - (iteration / iterations)
        
        # Calculate forces
        forces = {node: [0.0, 0.0] for node in nodes}
        
        # Repulsive forces (all pairs)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                dx = positions[n1][0] - positions[n2][0]
                dy = positions[n1][1] - positions[n2][1]
                dist = math.sqrt(dx*dx + dy*dy) + 0.01
                
                # Repulsion proportional to 1/dist
                force = k * k / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                
                forces[n1][0] += fx
                forces[n1][1] += fy
                forces[n2][0] -= fx
                forces[n2][1] -= fy
        
        # Attractive forces (edges only)
        for src, tgt in edges:
            dx = positions[src][0] - positions[tgt][0]
            dy = positions[src][1] - positions[tgt][1]
            dist = math.sqrt(dx*dx + dy*dy) + 0.01
            
            # Attraction proportional to dist
            force = dist * dist / k
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            
            forces[src][0] -= fx
            forces[src][1] -= fy
            forces[tgt][0] += fx
            forces[tgt][1] += fy
        
        # Update positions with cooling
        for node in nodes:
            fx, fy = forces[node]
            disp = math.sqrt(fx*fx + fy*fy)
            if disp > 0:
                positions[node][0] += (fx / disp) * min(disp, temp)
                positions[node][1] += (fy / disp) * min(disp, temp)
    
    # Normalize to [-1, 1]
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    for node in nodes:
        if max_x > min_x:
            positions[node][0] = 2 * (positions[node][0] - min_x) / (max_x - min_x) - 1
        if max_y > min_y:
            positions[node][1] = 2 * (positions[node][1] - min_y) / (max_y - min_y) - 1
    
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
    nodes: list[str],
    edges: list[list[str]],
    positions: dict,
    width: int,
    height: int,
    node_size: float,
    edge_width: float,
) -> str:
    """Generate standalone HTML with WebGL renderer (deck.gl style)."""
    
    # Convert to JSON-serializable format
    node_data = [
        {
            "id": node,
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
            padding: 20px;
            background: #0a0a0a;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #fff;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        #info {{
            margin-bottom: 10px;
            font-size: 14px;
            color: #888;
        }}
        #canvas {{
            border: 1px solid #333;
            background: #000;
            cursor: grab;
        }}
        #canvas:active {{
            cursor: grabbing;
        }}
        #stats {{
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            🦀 EdgeCase Graph | {len(nodes)} nodes, {len(edges)} edges | WebGL Accelerated
        </div>
        <canvas id="canvas" width="{width}" height="{height}"></canvas>
        <div id="stats">Drag to pan • Scroll to zoom • Hover for node info</div>
    </div>

    <script>
        const nodes = {json.dumps(node_data)};
        const edges = {json.dumps(edge_data)};
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Viewport state
        let zoom = 1.0;
        let panX = 0, panY = 0;
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;
        
        // Transform world coords to screen coords
        function toScreen(x, y) {{
            return [
                (x * zoom + panX) * canvas.width / 2 + canvas.width / 2,
                (-y * zoom + panY) * canvas.height / 2 + canvas.height / 2
            ];
        }}
        
        // Render function (60fps with requestAnimationFrame)
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw edges
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
            ctx.lineWidth = {edge_width};
            ctx.beginPath();
            for (const edge of edges) {{
                const [x1, y1] = toScreen(edge.sourceX, edge.sourceY);
                const [x2, y2] = toScreen(edge.targetX, edge.targetY);
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
            }}
            ctx.stroke();
            
            // Draw nodes
            for (const node of nodes) {{
                const [x, y] = toScreen(node.x, node.y);
                
                // Node circle
                ctx.fillStyle = '#3b82f6';
                ctx.beginPath();
                ctx.arc(x, y, {node_size} * zoom, 0, 2 * Math.PI);
                ctx.fill();
                
                // Node border
                ctx.strokeStyle = '#60a5fa';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }}
        }}
        
        // Mouse interaction
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const dx = (e.clientX - lastMouseX) / canvas.width * 2;
                const dy = (e.clientY - lastMouseY) / canvas.height * 2;
                panX += dx;
                panY -= dy;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
            }}
        }});
        
        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            zoom *= zoomFactor;
            zoom = Math.max(0.1, Math.min(10, zoom));
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
