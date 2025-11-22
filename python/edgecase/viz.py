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
            "This file is required for WebGL visualization."
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
    ```python
    from edgecase import Graph, plot_graph

    G = Graph()
    G.add_node(label="A")
    G.add_node(label="B")
    G.add_node(label="C")
    G.add_edge(source="A", target="B")
    G.add_edge(source="B", target="C")

    plot_graph(G, layout="neato")
    ```
    """
    # Extract graph data (fast - direct from Rust)
    nodes = graph.get_node_ids(limit=None)
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
        # Use fewer iterations for large graphs to improve performance
        iterations = 25 if len(nodes) > 200 else 50
        positions, graph_extent = _force_directed_layout(nodes, edges, iterations=iterations)
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

    # Check if we're in a notebook environment
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            # We're in a notebook - display the HTML directly
            # Wrap in an iframe srcdoc to avoid any notebook sanitization issues
            # Add extra space for body padding (20px) + container padding (20px) + borders on both sides
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


class _QuadTreeNode:
    """Node in a quadtree for Barnes-Hut force calculation.

    Represents a spatial region that may contain multiple particles (nodes).
    Internal nodes represent aggregated mass, leaf nodes contain individual particles.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.center_x = (x_min + x_max) / 2
        self.center_y = (y_min + y_max) / 2
        self.width = x_max - x_min
        self.height = y_max - y_min

        # Mass properties (for aggregation)
        self.mass = 0.0
        self.mass_x = 0.0  # Sum of mass * x
        self.mass_y = 0.0  # Sum of mass * y

        # Child quadrants (NW, NE, SW, SE)
        self.children = [None, None, None, None]

        # Particles in this node (only for leaf nodes)
        self.particles = []

        # Whether this is a leaf node
        self.is_leaf = True

    @property
    def center_of_mass(self):
        """Calculate center of mass for this region."""
        if self.mass == 0:
            return (self.center_x, self.center_y)
        return (self.mass_x / self.mass, self.mass_y / self.mass)


def _build_quadtree(nodes: list[str], positions: dict) -> _QuadTreeNode:
    """Build a quadtree from node positions for Barnes-Hut approximation.

    Parameters
    ----------
    nodes : list[str]
        List of node identifiers
    positions : dict
        Dictionary mapping node IDs to [x, y] positions

    Returns
    -------
    _QuadTreeNode
        Root of the quadtree
    """
    if not nodes:
        return None

    # Find bounding box
    all_x = [positions[node][0] for node in nodes]
    all_y = [positions[node][1] for node in nodes]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding to avoid edge cases
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create root node
    root = _QuadTreeNode(x_min, x_max, y_min, y_max)

    # Insert all particles
    for node_id in nodes:
        x, y = positions[node_id]
        _insert_particle(root, node_id, x, y, 1.0)  # Mass = 1.0 for all nodes

    return root


def _insert_particle(node: _QuadTreeNode, particle_id: str, x: float, y: float, mass: float):
    """Insert a particle into the quadtree.

    Parameters
    ----------
    node : _QuadTreeNode
        Current node to insert into
    particle_id : str
        Identifier for the particle
    x, y : float
        Position of the particle
    mass : float
        Mass of the particle
    """
    # Update mass properties
    node.mass += mass
    node.mass_x += mass * x
    node.mass_y += mass * y

    if node.is_leaf:
        if len(node.particles) == 0:
            # Empty leaf - add particle
            node.particles.append((particle_id, x, y))
        else:
            # Leaf with particle(s) - need to subdivide
            old_particles = node.particles[:]
            node.particles = []
            node.is_leaf = False

            # Create children
            mid_x = (node.x_min + node.x_max) / 2
            mid_y = (node.y_min + node.y_max) / 2

            # Quadrants: NW (0), NE (1), SW (2), SE (3)
            node.children[0] = _QuadTreeNode(node.x_min, mid_x, mid_y, node.y_max)  # NW
            node.children[1] = _QuadTreeNode(mid_x, node.x_max, mid_y, node.y_max)  # NE
            node.children[2] = _QuadTreeNode(node.x_min, mid_x, node.y_min, mid_y)  # SW
            node.children[3] = _QuadTreeNode(mid_x, node.x_max, node.y_min, mid_y)  # SE

            # Re-insert old particles
            for old_id, old_x, old_y in old_particles:
                quadrant = _get_quadrant(node, old_x, old_y)
                _insert_particle(node.children[quadrant], old_id, old_x, old_y, mass)

            # Insert new particle
            quadrant = _get_quadrant(node, x, y)
            _insert_particle(node.children[quadrant], particle_id, x, y, mass)
    else:
        # Internal node - insert into appropriate child
        quadrant = _get_quadrant(node, x, y)
        _insert_particle(node.children[quadrant], particle_id, x, y, mass)


def _get_quadrant(node: _QuadTreeNode, x: float, y: float) -> int:
    """Determine which quadrant a point belongs to.

    Returns
    -------
    int
        Quadrant index: 0=NW, 1=NE, 2=SW, 3=SE
    """
    mid_x = (node.x_min + node.x_max) / 2
    mid_y = (node.y_min + node.y_max) / 2

    if x < mid_x:
        return 0 if y >= mid_y else 2  # West side
    else:
        return 1 if y >= mid_y else 3  # East side


def _calculate_barnes_hut_force(
    particle_id: str, pos: list[float], node: _QuadTreeNode, k: float, theta: float
) -> tuple[float, float]:
    """Calculate repulsive force on a particle using Barnes-Hut approximation.

    Parameters
    ----------
    particle_id : str
        ID of the particle to calculate force for
    pos : list[float]
        Current position [x, y] of the particle
    node : _QuadTreeNode
        Current quadtree node
    k : float
        Optimal distance parameter (from FR algorithm)
    theta : float
        Barnes-Hut approximation parameter (0.5 is typical)

    Returns
    -------
    tuple[float, float]
        Force components (fx, fy)
    """
    import math

    if node is None or node.mass == 0:
        return (0.0, 0.0)

    # Get center of mass for this region
    com_x, com_y = node.center_of_mass

    # Distance to center of mass
    dx = pos[0] - com_x
    dy = pos[1] - com_y
    dist_sq = dx * dx + dy * dy
    dist = math.sqrt(dist_sq) if dist_sq > 0.01 else 0.01

    # Check if we can use this node as aggregate (Barnes-Hut criterion)
    # s/d < theta, where s is the width of the region and d is the distance
    if node.is_leaf or (node.width / dist < theta):
        # Use this node as aggregate (or it's a leaf with one particle)
        if node.is_leaf and len(node.particles) == 1 and node.particles[0][0] == particle_id:
            # Don't calculate force on itself
            return (0.0, 0.0)

        # Repulsive force: f_r(d) = k^2 / d
        force = (k * k) / dist
        fx = (dx / dist) * force * node.mass  # Scale by mass
        fy = (dy / dist) * force * node.mass

        return (fx, fy)
    else:
        # Need to recurse into children for more accuracy
        total_fx = 0.0
        total_fy = 0.0

        for child in node.children:
            if child is not None:
                fx, fy = _calculate_barnes_hut_force(particle_id, pos, child, k, theta)
                total_fx += fx
                total_fy += fy

        return (total_fx, total_fy)


def _force_directed_layout(
    nodes: list[str], edges: list[list[str]], iterations: int = 500
) -> tuple[dict, float]:
    """Fruchterman-Reingold layout algorithm with Rust implementation.

    Uses optimized Rust code with Barnes-Hut quadtree approximation for repulsive forces,
    providing 10-100x speedup over pure Python implementation.

    Parameters
    ----------
    nodes : list[str]
        Node IDs
    edges : list[list[str]]
        Edge list as [source, target] pairs
    iterations : int
        Number of force-directed iterations

    Returns
    -------
    tuple[dict, float]
        (positions dict mapping node_id -> [x, y], graph_extent)
    """
    from edgecase._core import force_directed_layout_rust

    n = len(nodes)
    if n == 0:
        return {}, 0.0
    if n == 1:
        return {nodes[0]: [0.0, 0.0]}, 0.0

    # Call Rust implementation
    print(
        f"Computing force layout with Rust for {n} nodes, {len(edges)} edges, {iterations} iterations..."
    )
    import time

    start_time = time.time()

    # Convert nodes to indices for Rust
    # Filter out self-loops as they don't contribute to layout forces
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_tuples = [(node_to_idx[src], node_to_idx[tgt]) for src, tgt in edges if src != tgt]

    if len(edge_tuples) < len(edges):
        print(f"  Filtered {len(edges) - len(edge_tuples)} self-loop(s) from layout computation")

    # Call Rust function - returns dict mapping node index -> (x, y)
    positions_dict = force_directed_layout_rust(
        node_count=n,
        edges=edge_tuples,
        iterations=iterations,
        use_barnes_hut=n > 100,
    )

    elapsed = time.time() - start_time
    print(f"✓ Layout computed in {elapsed:.2f}s")

    # Convert back to dict with original node IDs
    # positions_dict maps node_index -> (x, y) tuple
    positions = {nodes[i]: list(pos) for i, pos in positions_dict.items()}

    # Calculate extent for zoom calculation
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    range_x = max(all_x) - min(all_x) if all_x else 0
    range_y = max(all_y) - min(all_y) if all_y else 0
    graph_extent = max(range_x, range_y)

    print(f"Graph extent: {graph_extent:.4f}")

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
        #canvas-wrapper::before {{
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
            animation: smokeMoveCanvas 60s ease-in-out infinite;
        }}
        @keyframes smokeMoveCanvas {{
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
                background-size: 150% 150%, 145% 145%, 155% 155%, 140% 140%;
            }}
        }}
        #canvas {{
            position: relative;
            z-index: 1;
            border: 1px solid #ddd;
            background: transparent;
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
        const nodesInfo = {json.dumps([{"id": str(node), "label": str(node), "attributes": (graph.get_node(node).attributes if graph.get_node(node) and graph.get_node(node).attributes else {})} for node in nodes])};
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

            const effectiveNodeSize = zoomMode === 'geometric' ? Math.max(1.5, {node_size} * zoom) : {node_size};
            const effectiveEdgeWidth = zoomMode === 'geometric' ? Math.max(0.5, {edge_width} * zoom) : {edge_width};
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
            // Don't resume simulation if it was stabilized
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
            zoom = Math.max(0.001, Math.min(10, zoom));
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
    ```python
    from edgecase import Graph
    from edgecase.viz import _generate_dot_string

    G = Graph()
    G.add_node(label="A")
    G.add_node(label="B")
    G.add_edge(source="A", target="B")

    # Get node and edge IDs
    node_ids = G.get_node_ids(limit=None)
    edges = [[e[1], e[2]] for e in G.get_edges_as_tuples()]

    dot = _generate_dot_string(G, node_ids, edges)
    print(dot)
    ```
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
        attrs = node_obj.attributes if (node_obj and node_obj.attributes) else {}

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
    num_nodes = len(graph.get_node_ids(limit=None))
    num_edges = graph.count_edges()

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
                "attributes": node_obj.attributes if (node_obj and node_obj.attributes) else {},
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
    is_directed = graph.is_directed()
    graph_type = "directed" if is_directed else "undirected"

    # Calculate initial zoom to fit graph in viewport
    # Use graph extent and edge density to determine appropriate zoom level
    # Dense graphs need zooming IN, sparse graphs need zooming OUT
    if graph_extent > 0 and len(nodes) > 0:
        # Calculate edge density (edges per node)
        edge_density = len(edges) / len(nodes)

        # Base target viewport size on node count
        if len(nodes) > 1000:
            base_target = 0.15
        elif len(nodes) > 200:
            base_target = 0.25
        else:
            base_target = 0.4

        # Adjust based on edge density with refined thresholds
        # Very sparse graphs (density < 1.2): need significant zoom in (trees, stars)
        # Sparse graphs (density < 2): slight zoom in
        # Normal graphs (density 2-4): no adjustment
        # Dense graphs (density > 4): zoom out more
        if edge_density > 6:
            density_multiplier = 0.7  # Dense graphs are compact, zoom out
        elif edge_density > 4:
            density_multiplier = 0.85
        elif edge_density > 2:
            density_multiplier = 1.0  # Normal density
        elif edge_density > 1.2:
            density_multiplier = 1.3  # Sparse but connected
        else:
            density_multiplier = 1.6  # Very sparse (trees, stars) - zoom in

        target_viewport_size = base_target * density_multiplier
        initial_zoom = target_viewport_size / (graph_extent * 2.0)
        # Clamp between reasonable bounds - allow very small zoom
        initial_zoom = max(0.002, min(initial_zoom, 0.5))
    else:
        initial_zoom = 0.1

    print(
        f"Initial zoom calculated: {initial_zoom:.4f} (graph_extent: {graph_extent:.2f}, edge_density: {len(edges) / len(nodes) if len(nodes) > 0 else 0:.2f})"
    )

    # Add diagnostic info for zoom tuning
    diagnostic_info = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "graph_extent": graph_extent,
        "initial_zoom": initial_zoom,
    }

    # Convert to vis-network format
    vis_nodes = []
    for node in nodes:
        if node not in positions:
            print(f"WARNING: Node {node} not in positions dict!")
            continue
        node_obj = graph.get_node(node)
        node_dict = {
            "id": str(node),
            "label": str(node),
        }
        # Add custom attributes if present
        if node_obj and node_obj.attributes:
            # Support both 'color' and 'fillcolor' (for compatibility with Graphviz)
            if "color" in node_obj.attributes:
                node_dict["color"] = node_obj.attributes["color"]
            elif "fillcolor" in node_obj.attributes:
                node_dict["color"] = node_obj.attributes["fillcolor"]
            if "size" in node_obj.attributes:
                node_dict["value"] = node_obj.attributes["size"]
        vis_nodes.append(node_dict)

    print(f"Generated {len(vis_nodes)} nodes for visualization (from {len(nodes)} total)")

    vis_edges = []
    for src, tgt in edges:
        edge_ids = graph.find_edge_ids(src, tgt)
        edge_obj = graph.get_edge(edge_ids[0]) if edge_ids else None
        edge_dict = {
            "from": str(src),
            "to": str(tgt),
        }
        # Add custom attributes if present
        if edge_obj and edge_obj.attributes:
            if "color" in edge_obj.attributes:
                edge_dict["color"] = edge_obj.attributes["color"]
            if "weight" in edge_obj.attributes:
                edge_dict["value"] = edge_obj.attributes["weight"]
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
        #info {
            margin-bottom: 15px;
            font-size: 14px;
            color: #333;
            font-weight: 500;
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
        #stats {
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div id="viz-container">
        <div id="info">
            EdgeCase Graph | """
        + graph_type
        + """ | """
        + str(len(nodes))
        + """ nodes, """
        + str(len(edges))
        + """ edges
        </div>
        <div id="mynetwork"></div>
        <div id="stats">Drag nodes to move • Scroll to zoom • Click and drag background to pan</div>
    </div>

    <script type="text/javascript">
        // Wait for DOM to be fully loaded before initializing
        window.addEventListener('DOMContentLoaded', function() {
            // Check if vis is loaded
            if (typeof vis === 'undefined') {
                console.error('vis-network library not loaded!');
                document.getElementById('mynetwork').innerHTML = '<div style="padding: 20px; color: red;">Error: vis-network library failed to load. Check your internet connection.</div>';
                return;
            } else {
                console.log('vis-network library loaded successfully');
                console.log('vis version:', vis.Network ? 'Network class available' : 'Network class missing');
            }

            // Debug: Log what we're working with
            console.log('Initializing vis-network...');

            // Create nodes and edges datasets
            var nodesData = """
        + json.dumps(vis_nodes)
        + """;
            var edgesData = """
        + json.dumps(vis_edges)
        + """;

            console.log('Nodes:', nodesData.length);
            console.log('Edges:', edgesData.length);
            if (nodesData.length > 0) {
                console.log('Sample node:', nodesData[0]);
            }
            if (edgesData.length > 0) {
                console.log('Sample edge:', edgesData[0]);
            }

            var nodes = new vis.DataSet(nodesData);
            var edges = new vis.DataSet(edgesData);

            // Create network
            var container = document.getElementById('mynetwork');
            console.log('Container:', container);
            console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);

            if (container.offsetWidth === 0 || container.offsetHeight === 0) {
                console.error('Container has zero dimensions!');
            }

            var data = {
                nodes: nodes,
                edges: edges
            };

            var options = {
                nodes: {
                    shape: 'dot',
                    size: 16,
                    font: {
                        size: 14,
                        color: '#333'
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
                    width: 2,
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
                    enabled: true,
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
                        iterations: 200,
                        updateInterval: 100,
                        fit: true
                    }
                },
                interaction: {
                    dragNodes: true,
                    dragView: true,
                    zoomView: true,
                    hover: true,
                    tooltipDelay: 100
                }
            };

            console.log('Creating network with options:', options);

            try {
                var network = new vis.Network(container, data, options);
                console.log('Network created successfully');
            } catch (error) {
                console.error('Error creating network:', error);
                document.getElementById('mynetwork').innerHTML = '<div style="padding: 20px; color: red;">Error creating visualization: ' + error.message + '</div>';
                return;
            }

            console.log('Network created, starting stabilization...');

            // Add click handler for node info
            network.on("click", function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    console.log('Node clicked:', node);
                }
            });

            // Log stabilization progress
            network.on("stabilizationProgress", function(params) {
                var progress = Math.round((params.iterations / params.total) * 100);
                if (progress % 10 === 0) {
                    console.log('Stabilization progress:', progress + '%');
                }
            });

            // Log when stabilization is done
            network.on("stabilizationIterationsDone", function() {
                console.log("Network stabilized!");
                network.setOptions({ physics: false });
                network.fit();
            });

            network.on("afterDrawing", function() {
                console.log("Network drawn");
            });
        }); // End DOMContentLoaded
    </script>
</body>
</html>"""
    )

    # Concatenate the parts (vis_network_js is not in an f-string, so it won't have escaping issues)
    html = html_start + vis_network_js + html_after_vis

    return html


# Convenience export
__all__ = ["plot_graph"]
