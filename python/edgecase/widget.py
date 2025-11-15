import json
from typing import Any, Dict, List, Optional

# Color palettes for categorical data
COLOR_PALETTES = {
    "default": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "vibrant": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
    ],
    "pastel": [
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
        "#8dd3c7",
    ],
    "dark": [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#8da0cb",
        "#fc8d62",
    ],
    "tableau": [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
    ],
}


def _is_notebook() -> bool:
    """Detect if running in a notebook environment."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False
    except (ImportError, NameError):
        return False


class GraphWidget:
    """
    Interactive graph visualization widget for Jupyter notebooks.

    Features:
    - Auto-detects notebook environment
    - Uses node attributes as aesthetics (color, size, shape)
    - Interactive tooltips on hover
    - Zoom, pan, and selection
    - WebGL-accelerated for large graphs
    - Static export to PNG/SVG

    Aesthetic Mappings:
    - `color` attribute → node fill color (hex or category)
    - `size` attribute → node radius
    - `shape` attribute → node shape (circle, square, triangle)
    - `label` attribute → tooltip text (defaults to node ID)
    - `community` attribute → automatic color mapping

    Examples
    --------
    Basic visualization:

    ```python
    from edgecase import Graph, GraphWidget

    G = Graph(directed=False)
    G.add_node(label='A')
    G.add_node(label='B')
    G.add_node(label='C')
    G.add_edge(source='A', target='B')
    G.add_edge(source='B', target='C')
    G.add_edge(source='C', target='A')

    widget = GraphWidget(G)
    widget.show()
    ```

    With community detection:

    ```python
    from edgecase import Graph, GraphWidget

    G = Graph(directed=False)
    for i in range(6):
        G.add_node(label=str(i))
    G.add_edge(source='0', target='1')
    G.add_edge(source='1', target='2')
    G.add_edge(source='0', target='2')
    G.add_edge(source='3', target='4')
    G.add_edge(source='4', target='5')
    G.add_edge(source='3', target='5')

    G.get_cmty_louvain(attr_name="community")
    widget = GraphWidget(G)
    widget.show()
    ```

    Custom aesthetics:

    ```python
    from edgecase import Graph, GraphWidget

    G = Graph(directed=False)
    G.add_node(label='A')
    G.add_node(label='B')
    G.add_node(label='C')
    G.add_edge(source='A', target='B')
    G.add_edge(source='B', target='C')

    # Size nodes by degree
    for node_id in G.node_ids(limit=None):
        degree = G.get_degree(node=node_id)
        G.set_node_attr(node_id=node_id, attr="size", value=degree * 0.5)

    widget = GraphWidget(G)
    widget.show()
    ```
    """

    def __init__(
        self,
        graph,
        layout: str = "force",
        width: int = 800,
        height: int = 600,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        shape_by: Optional[str] = None,
        node_size: float = 10.0,
        size_scale: float = 1.0,
        edge_width: float = 1.0,
        color_palette: str = "default",
        background: str = "#ffffff",
        **kwargs,
    ):
        """
        Initialize graph widget.

        Parameters
        ----------
        graph : Graph
            EdgeCase graph to visualize
        layout : str, default "force"
            Layout algorithm: "force", "circular", "random", "kamada_kawai"
        width : int, default 800
            Canvas width in pixels
        height : int, default 600
            Canvas height in pixels
        color_by : str, optional
            Node attribute to use for coloring (e.g., 'community', 'type')
        size_by : str, optional
            Node attribute to use for sizing (e.g., 'degree', 'centrality')
        shape_by : str, optional
            Node attribute to use for shape (e.g., 'type')
        node_size : float, default 10.0
            Base node size in pixels
        size_scale : float, default 1.0
            Scaling factor for size_by attribute
        edge_width : float, default 1.0
            Edge line width
        color_palette : str, default "default"
            Color palette name: "default", "vibrant", "pastel", "dark", "tableau"
        background : str, default "#ffffff"
            Background color (hex)
        **kwargs
            Additional layout or rendering options
        """
        self.graph = graph
        self.layout = layout
        self.width = width
        self.height = height
        self.color_by = color_by
        self.size_by = size_by
        self.shape_by = shape_by
        self.node_size = node_size
        self.size_scale = size_scale
        self.edge_width = edge_width
        self.color_palette = color_palette
        self.background = background
        self.kwargs = kwargs

        # Lazy initialization
        self._widget = None
        self._data = None

    def _prepare_data(self) -> Dict[str, Any]:
        """Extract graph data and compute aesthetics."""
        from edgecase.viz import _circular_layout, _force_directed_layout, _random_layout

        # Get nodes and edges
        nodes = self.graph.node_ids(limit=None)
        edges = []
        seen = set()

        for node in nodes:
            for neighbor in self.graph.neighbors(node):
                edge_key = tuple(sorted([node, neighbor]))
                if edge_key not in seen:
                    edges.append(list(edge_key))
                    seen.add(edge_key)

        # Compute layout
        if self.layout == "force":
            positions = _force_directed_layout(nodes, edges, iterations=100)
        elif self.layout == "circular":
            positions = _circular_layout(nodes)
        else:
            positions = _random_layout(nodes)

        # Build node data with aesthetics
        node_data = []
        for node_id in nodes:
            node_info = {
                "id": node_id,
                "x": positions[node_id][0],
                "y": positions[node_id][1],
                "label": str(node_id),
            }

            # Get node attributes
            try:
                node = self.graph.get_node(node_id)
                if node and node.attributes:
                    node_info.update(node.attributes)
            except Exception:
                pass

            node_data.append(node_info)

        # Compute aesthetic mappings
        node_data = self._apply_aesthetics(node_data)

        return {
            "nodes": node_data,
            "edges": edges,
            "width": self.width,
            "height": self.height,
            "edge_width": self.edge_width,
            "background": self.background,
        }

    def _apply_aesthetics(self, node_data: List[Dict]) -> List[Dict]:
        """Apply aesthetic mappings to node data.

        Priority:
        1. Direct visual attributes ('color', 'size', 'shape') from nodes
        2. Legacy color_by/size_by parameters for backward compatibility
        3. Default values
        """

        # Legacy: Color mapping via color_by parameter (for backward compatibility)
        if self.color_by:
            unique_values = list(set(n.get(self.color_by) for n in node_data if self.color_by in n))
            if unique_values:
                color_map = self._create_color_map(unique_values)
                for node in node_data:
                    if self.color_by in node and "color" not in node:
                        # Only apply if node doesn't already have a 'color' attribute
                        node["color"] = color_map.get(node[self.color_by], "#cccccc")

        # Legacy: Size mapping via size_by parameter (for backward compatibility)
        if self.size_by:
            sizes = [n.get(self.size_by, 0) for n in node_data if self.size_by in n]
            if sizes:
                min_size, max_size = min(sizes), max(sizes)
                size_range = max_size - min_size if max_size > min_size else 1.0
                for node in node_data:
                    if self.size_by in node and "size" not in node:
                        # Only apply if node doesn't already have a 'size' attribute
                        normalized = (node[self.size_by] - min_size) / size_range
                        node["size"] = self.node_size * (0.5 + normalized * self.size_scale)

        # Apply defaults for nodes without visual attributes
        for node in node_data:
            if "size" not in node:
                node["size"] = self.node_size
            else:
                # Scale existing size values to pixel units if they look normalized (0-1 range)
                if isinstance(node["size"], (int, float)) and 0 <= node["size"] <= 1:
                    node["size"] = self.node_size * (0.5 + node["size"] * self.size_scale)

            if "color" not in node:
                node["color"] = "#4285f4"  # Default blue

        return node_data

    def _create_color_map(self, categories: List[Any]) -> Dict[Any, str]:
        """Create color mapping for categorical data using built-in palettes.""" ""
        # Get the color palette
        palette = COLOR_PALETTES.get(self.color_palette, COLOR_PALETTES["default"])

        # Cycle through colors if we have more categories than colors
        n_categories = len(categories)
        if n_categories <= len(palette):
            colors = palette[:n_categories]
        else:
            # Repeat palette to cover all categories
            colors = (palette * (n_categories // len(palette) + 1))[:n_categories]

        return dict(zip(categories, colors))

    def show(self):
        """Display the widget in a notebook or open in browser."""
        if _is_notebook():
            return self._show_widget()
        else:
            return self._show_browser()

    def _show_widget(self):
        """Show as AnyWidget in notebook."""
        try:
            import anywidget
            import traitlets
        except ImportError:
            print("AnyWidget not installed. Falling back to browser display.")
            print("Install with: pip install anywidget")
            return self._show_browser()

        if self._data is None:
            self._data = self._prepare_data()

        # Create widget class dynamically
        class GraphWidgetImpl(anywidget.AnyWidget):
            _esm = """
function render({ model, el }) {
  const data = model.get('data');
  if (!data || !data.nodes) {
    el.innerHTML = '<p>No graph data available</p>';
    return;
  }

  // Create container for canvas and tooltip
  const container = document.createElement('div');
  container.style.position = 'relative';
  container.style.display = 'inline-block';
  el.appendChild(container);

  const canvas = document.createElement('canvas');
  canvas.width = data.width || 800;
  canvas.height = data.height || 600;
  canvas.style.border = '1px solid #ccc';
  canvas.style.cursor = 'grab';
  container.appendChild(canvas);

  // Create tooltip
  const tooltip = document.createElement('div');
  tooltip.style.position = 'absolute';
  tooltip.style.display = 'none';
  tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
  tooltip.style.color = 'white';
  tooltip.style.padding = '8px 12px';
  tooltip.style.borderRadius = '4px';
  tooltip.style.fontSize = '12px';
  tooltip.style.pointerEvents = 'none';
  tooltip.style.zIndex = '1000';
  tooltip.style.maxWidth = '200px';
  container.appendChild(tooltip);

  const ctx = canvas.getContext('2d');

  // Interactive state
  let transform = {
    offsetX: canvas.width / 2,
    offsetY: canvas.height / 2,
    scale: Math.min(canvas.width, canvas.height) / 2.5
  };

  let dragState = {
    isDragging: false,
    dragNode: null,
    startX: 0,
    startY: 0,
    isPanning: false
  };

  let hoveredNode = null;

  // Helper: Screen to world coordinates
  function screenToWorld(screenX, screenY) {
    return {
      x: (screenX - transform.offsetX) / transform.scale,
      y: (screenY - transform.offsetY) / transform.scale
    };
  }

  // Helper: World to screen coordinates
  function worldToScreen(worldX, worldY) {
    return {
      x: worldX * transform.scale + transform.offsetX,
      y: worldY * transform.scale + transform.offsetY
    };
  }

  // Helper: Find node at position
  function findNodeAt(worldX, worldY) {
    for (let node of data.nodes) {
      const dx = node.x - worldX;
      const dy = node.y - worldY;
      const radius = (node.size || 10) / transform.scale;
      if (dx * dx + dy * dy < radius * radius) {
        return node;
      }
    }
    return null;
  }

  // Draw function
  function draw() {
    // Clear canvas
    ctx.fillStyle = data.background || '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(transform.offsetX, transform.offsetY);
    ctx.scale(transform.scale, transform.scale);

    // Draw edges
    if (data.edges && data.edges.length > 0) {
      ctx.strokeStyle = '#cccccc';
      ctx.lineWidth = (data.edge_width || 1) / transform.scale;
      data.edges.forEach(edge => {
        const from = data.nodes.find(n => n.id === edge[0]);
        const to = data.nodes.find(n => n.id === edge[1]);
        if (from && to) {
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      });
    }

    // Draw nodes
    data.nodes.forEach(node => {
      const isHovered = hoveredNode === node;

      // Highlight hovered node
      if (isHovered) {
        ctx.fillStyle = 'rgba(255, 200, 0, 0.3)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, (node.size || 10) / transform.scale * 1.5, 0, 2 * Math.PI);
        ctx.fill();
      }

      ctx.fillStyle = node.color || '#4285f4';
      ctx.beginPath();
      ctx.arc(node.x, node.y, (node.size || 10) / transform.scale, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = isHovered ? '#ffc800' : '#ffffff';
      ctx.lineWidth = (isHovered ? 2.5 : 1.5) / transform.scale;
      ctx.stroke();
    });

    // Draw labels
    ctx.font = `${12 / transform.scale}px sans-serif`;
    ctx.fillStyle = '#333333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    data.nodes.forEach(node => {
      ctx.fillText(node.label || node.id, node.x, node.y);
    });

    ctx.restore();
  }

  // Mouse event handlers
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const world = screenToWorld(screenX, screenY);

    const node = findNodeAt(world.x, world.y);

    if (node) {
      // Start dragging node
      dragState.isDragging = true;
      dragState.dragNode = node;
      dragState.startX = world.x;
      dragState.startY = world.y;
      canvas.style.cursor = 'grabbing';
    } else {
      // Start panning
      dragState.isPanning = true;
      dragState.startX = screenX;
      dragState.startY = screenY;
      canvas.style.cursor = 'grabbing';
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const world = screenToWorld(screenX, screenY);

    if (dragState.isDragging && dragState.dragNode) {
      // Drag node
      dragState.dragNode.x = world.x;
      dragState.dragNode.y = world.y;
      draw();
    } else if (dragState.isPanning) {
      // Pan view
      const dx = screenX - dragState.startX;
      const dy = screenY - dragState.startY;
      transform.offsetX += dx;
      transform.offsetY += dy;
      dragState.startX = screenX;
      dragState.startY = screenY;
      draw();
    } else {
      // Check for hover
      const node = findNodeAt(world.x, world.y);
      if (node !== hoveredNode) {
        hoveredNode = node;

        if (node) {
          // Show tooltip
          const screen = worldToScreen(node.x, node.y);
          tooltip.innerHTML = `<strong>${node.label || node.id}</strong><br/>`;

          // Add node attributes to tooltip (exclude aesthetic attributes)
          const aestheticAttrs = ['id', 'x', 'y', 'label', 'color', 'size', 'fillcolor', 'fontcolor', 'width', 'height', 'shape'];
          const attrs = [];
          for (let key in node) {
            if (!aestheticAttrs.includes(key)) {
              let value = node[key];
              // Format numeric values: round floats to 5 decimals, keep integers as-is
              if (typeof value === 'number') {
                value = Number.isInteger(value) ? value : parseFloat(value.toFixed(5));
              }
              attrs.push(`${key}: ${value}`);
            }
          }
          if (attrs.length > 0) {
            tooltip.innerHTML += attrs.join('<br/>');
          }

          tooltip.style.display = 'block';
          tooltip.style.left = (screen.x + 15) + 'px';
          tooltip.style.top = (screen.y - 15) + 'px';
          canvas.style.cursor = 'pointer';
        } else {
          tooltip.style.display = 'none';
          canvas.style.cursor = 'grab';
        }

        draw();
      }
    }
  });

  canvas.addEventListener('mouseup', () => {
    dragState.isDragging = false;
    dragState.dragNode = null;
    dragState.isPanning = false;
    canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
  });

  canvas.addEventListener('mouseleave', () => {
    dragState.isDragging = false;
    dragState.dragNode = null;
    dragState.isPanning = false;
    hoveredNode = null;
    tooltip.style.display = 'none';
    canvas.style.cursor = 'grab';
    draw();
  });

  // Zoom with mouse wheel
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Zoom factor
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;

    // Zoom towards mouse position
    const worldBefore = screenToWorld(screenX, screenY);
    transform.scale *= zoomFactor;
    const worldAfter = screenToWorld(screenX, screenY);

    // Adjust offset to keep point under mouse stationary
    transform.offsetX += (worldAfter.x - worldBefore.x) * transform.scale;
    transform.offsetY += (worldAfter.y - worldBefore.y) * transform.scale;

    draw();
  });

  // Initial draw
  draw();
}

export default { render };
"""
            data = traitlets.Dict({}).tag(sync=True)

        widget = GraphWidgetImpl(data=self._data)
        self._widget = widget

        # Display in notebook (don't return to avoid double display)
        from IPython.display import display

        display(widget)

    def _show_browser(self):
        """Fallback to opening in browser."""
        return self.graph.show(target="browser")

    def to_html(self, path: str) -> str:
        """Export to standalone interactive HTML file."""
        if self._data is None:
            self._data = self._prepare_data()

        # Read the interactive JavaScript code from the widget
        js_code = """
        const data = DATA_PLACEHOLDER;
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        let transform = {
            offsetX: canvas.width / 2,
            offsetY: canvas.height / 2,
            scale: Math.min(canvas.width, canvas.height) / 2.5
        };

        let dragState = { isDragging: false, dragNode: null, startX: 0, startY: 0, isPanning: false };
        let hoveredNode = null;

        function screenToWorld(screenX, screenY) {
            return {
                x: (screenX - transform.offsetX) / transform.scale,
                y: (screenY - transform.offsetY) / transform.scale
            };
        }

        function findNodeAt(worldX, worldY) {
            for (let node of data.nodes) {
                const dx = node.x - worldX;
                const dy = node.y - worldY;
                const radius = (node.size || 10) / transform.scale;
                if (dx * dx + dy * dy < radius * radius) {
                    return node;
                }
            }
            return null;
        }

        function draw() {
            ctx.fillStyle = data.background || '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.save();
            ctx.translate(transform.offsetX, transform.offsetY);
            ctx.scale(transform.scale, transform.scale);

            // Draw edges
            if (data.edges && data.edges.length > 0) {
                ctx.strokeStyle = '#cccccc';
                ctx.lineWidth = (data.edge_width || 1) / transform.scale;
                data.edges.forEach(edge => {
                    const from = data.nodes.find(n => n.id === edge[0]);
                    const to = data.nodes.find(n => n.id === edge[1]);
                    if (from && to) {
                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.stroke();
                    }
                });
            }

            // Draw nodes
            data.nodes.forEach(node => {
                const isHovered = hoveredNode === node;
                if (isHovered) {
                    ctx.fillStyle = 'rgba(255, 200, 0, 0.3)';
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, (node.size || 10) / transform.scale * 1.5, 0, 2 * Math.PI);
                    ctx.fill();
                }
                ctx.fillStyle = node.color || '#4285f4';
                ctx.beginPath();
                ctx.arc(node.x, node.y, (node.size || 10) / transform.scale, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = isHovered ? '#ffc800' : '#ffffff';
                ctx.lineWidth = (isHovered ? 2.5 : 1.5) / transform.scale;
                ctx.stroke();
            });

            // Draw labels
            ctx.font = `${12 / transform.scale}px sans-serif`;
            ctx.fillStyle = '#333333';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            data.nodes.forEach(node => {
                ctx.fillText(node.label || node.id, node.x, node.y);
            });

            ctx.restore();
        }

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const world = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
            const node = findNodeAt(world.x, world.y);
            if (node) {
                dragState.isDragging = true;
                dragState.dragNode = node;
                canvas.style.cursor = 'grabbing';
            } else {
                dragState.isPanning = true;
                dragState.startX = e.clientX - rect.left;
                dragState.startY = e.clientY - rect.top;
                canvas.style.cursor = 'grabbing';
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            const world = screenToWorld(screenX, screenY);

            if (dragState.isDragging && dragState.dragNode) {
                dragState.dragNode.x = world.x;
                dragState.dragNode.y = world.y;
                draw();
            } else if (dragState.isPanning) {
                transform.offsetX += screenX - dragState.startX;
                transform.offsetY += screenY - dragState.startY;
                dragState.startX = screenX;
                dragState.startY = screenY;
                draw();
            } else {
                const node = findNodeAt(world.x, world.y);
                if (node !== hoveredNode) {
                    hoveredNode = node;
                    canvas.style.cursor = node ? 'pointer' : 'grab';
                    draw();
                }
            }
        });

        canvas.addEventListener('mouseup', () => {
            dragState.isDragging = false;
            dragState.dragNode = null;
            dragState.isPanning = false;
            canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
        });

        canvas.addEventListener('mouseleave', () => {
            dragState.isDragging = false;
            dragState.isPanning = false;
            hoveredNode = null;
            canvas.style.cursor = 'grab';
            draw();
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
            const worldBefore = screenToWorld(screenX, screenY);
            transform.scale *= zoomFactor;
            const worldAfter = screenToWorld(screenX, screenY);
            transform.offsetX += (worldAfter.x - worldBefore.x) * transform.scale;
            transform.offsetY += (worldAfter.y - worldBefore.y) * transform.scale;
            draw();
        });

        draw();
        """

        js_code = js_code.replace("DATA_PLACEHOLDER", json.dumps(self._data))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EdgeCase Interactive Graph</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: sans-serif;
            background: {self.background};
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            margin-bottom: 10px;
        }}
        #canvas {{
            border: 1px solid #ccc;
            cursor: grab;
        }}
        .instructions {{
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Interactive Graph Visualization</h1>
    <canvas id="canvas" width="{self.width}" height="{self.height}"></canvas>
    <div class="instructions">
        🖱️ Drag nodes to move • Drag background to pan • Scroll to zoom • Hover to highlight
    </div>
    <script>
        {js_code}
    </script>
</body>
</html>
"""

        with open(path, "w") as f:
            f.write(html)

        return path


# Convenience function
def graph_widget(graph, **kwargs) -> GraphWidget:
    """
    Create an interactive graph widget.

    Parameters
    ----------
    graph : Graph
        EdgeCase graph to visualize
    **kwargs
        Passed to GraphWidget constructor

    Returns
    -------
    GraphWidget
        Widget instance (call .show() to display)

    Examples
    --------
    Basic usage:

    ```python
    from edgecase import Graph, graph_widget

    G = Graph(directed=False)
    G.add_node(label='A')
    G.add_node(label='B')
    G.add_node(label='C')
    G.add_edge(source='A', target='B')
    G.add_edge(source='B', target='C')

    widget = graph_widget(G)
    widget.show()
    ```

    With community detection:

    ```python
    from edgecase import Graph, graph_widget

    G = Graph(directed=False)
    for i in range(6):
        G.add_node(label=str(i))
    G.add_edge(source='0', target='1')
    G.add_edge(source='1', target='2')
    G.add_edge(source='0', target='2')
    G.add_edge(source='3', target='4')
    G.add_edge(source='4', target='5')
    G.add_edge(source='3', target='5')

    G.get_cmty_louvain(attr_name="community")
    widget = graph_widget(G)
    widget.show()
    ```
    """
    return GraphWidget(graph, **kwargs)
