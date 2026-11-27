// The EdgeCase viewer: a plot's scene drawn with WebGL 2, to pan and zoom.
//
// `mount(root, buffer, options)` draws the payload in `buffer` (see `src/scene/payload.rs`) in
// the element `root`. The nodes' positions, radii and colors live in textures, read by index;
// an edge is one instance of a triangle strip along its curve, and a node one instance of a
// quad. The source stays ASCII, since a saved file may not say that it's UTF-8.
//
// Edges add their density, -ln(1 - opacity) each, and their density-weighted color into a
// half-float buffer, which a full-screen pass draws over the background as 1 - exp(-density),
// the opacity that SVG strokes of the same opacity stacked up would have. The raster backend
// draws edges the same way. Since the density adds up, a view can be drawn progressively: a
// moving view draws a share of the edges, scaled up to stand for all of them, and the rest are
// added over the frames that follow once it stops. With the log tone, as the raster's, the pass
// shows the density by its logarithm instead, opaque at the view's densest pixel, which a pyramid
// of maxima finds on the GPU.

const TEX_W = 2048; // texels per row; the shaders shift by 11
const BUDGET = 150000; // the edges that a frame of a moving view draws
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 1000;
const CANVAS_MAX = 4096; // the canvas's longer side, in pixels
// The legend folds into a button when it would take more of the box than this
const LEGEND_SHARE = 0.35;
// A node or an edge is under the pointer when it is drawn within this many pixels of it
const HIT = 4;
// The controls' panel is this wide, and docks beside the drawing in a box at least this wide
const PANEL_W = 280;
const DOCK_MIN = 840;

// The buttons' icons, as paths on a 16 by 16 grid, stroked in the text color
const ICONS = {
  in: "M8 3v10M3 8h10",
  out: "M3 8h10",
  fit: "M8 2.5a5.5 5.5 0 1 0 0 11a5.5 5.5 0 1 0 0-11M8 8h.01",
  full: "M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10",
  exit: "M6 2.5V6H2.5M10 2.5V6h3.5M13.5 10H10v3.5M2.5 10H6v3.5",
  legend: "M6.5 4.5h7M6.5 8h7M6.5 11.5h7M3 4.5h.01M3 8h.01M3 11.5h.01",
  settings: "M2.5 4.5h6M12 4.5h1.5M2.5 11.5h1.5M8 11.5h5.5M10 2.5v4M6 9.5v4",
  close: "M4 4l8 8M12 4l-8 8",
  reset: "M3.5 8a4.5 4.5 0 1 0 1.5-3.35M3.5 2.5v3h3",
  search: "M7 2.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9M10.5 10.5l3 3",
  lasso: "M8 2.5c3 0 5.5 1.5 5.5 3.5S11 9.5 8 9.5S2.5 8 2.5 6S5 2.5 8 2.5M5 9.1c-.8.9-.8 2 0 2.6s1.2 1.4.5 2.3",
};
const icon = (path) => `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="${path}" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const el = (tag, className, text) => {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
};

// The payload's header and columns, as views of `buffer`; a column left out of it is a number
export function parse(buffer) {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "ECV1") throw new Error("Not a viewer payload");
  const len = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + len)));
  const column = (name) => {
    const c = header.columns[name];
    if (!c) return header.constants[name] ?? null;
    return new (c.type === "f32" ? Float32Array : Uint32Array)(buffer, c.offset, c.length);
  };
  const names = ["x", "y", "r", "fill", "border", "border_width", "border_opacity", "order",
    "src", "tgt", "color", "width", "opacity", "bow", "loop_size", "level", "base_width", "label_order",
    "label_color", "node_id", "node_id_hi", "base_fill", "base_border", "base_color", "base_label_order",
    "node_key", "edge_key"];
  return { header, ...Object.fromEntries(names.map((name) => [name, column(name)])) };
}

const LINK_VS = `#version 300 es
precision highp float;
precision highp int;
in vec2 a_corner;            // x: t along the edge, y: side, -1 or 1; or x 2 at the arrowhead's corner y
in uint a_src, a_tgt, a_color;
in float a_width, a_opacity, a_bow, a_loop_size;
uniform highp sampler2D u_nodes, u_colors;
uniform vec2 u_scale, u_offset, u_viewport;
uniform float u_unit;        // pixels per unit of the scene's width
uniform float u_radius;      // pixels per unit of the scene's radii
uniform float u_fringe;
uniform bool u_directed;
out vec3 v_color;
out vec3 v_side;             // how far in from each side of the stroke, or of the arrowhead, in pixels
out vec3 v_head;             // on the stroke, how far in from each side of the arrowhead
out float v_gain;
vec4 texel(highp sampler2D t, uint i) { return texelFetch(t, ivec2(int(i & 2047u), int(i >> 11)), 0); }
vec2 quad(vec2 a, vec2 b, vec2 c, float k) { return mix(mix(a, b, k), mix(b, c, k), k); }
vec2 cubic(vec2 a, vec2 b, vec2 c, vec2 d, float k) { return quad(mix(a, b, k), mix(b, c, k), mix(c, d, k), k); }
// How far p is from the line through a and b, on c's side
float inside(vec2 p, vec2 a, vec2 b, vec2 c) {
  vec2 e = b - a;
  float side = e.x * (c.y - a.y) - e.y * (c.x - a.x);
  return sign(side) * (e.x * (p.y - a.y) - e.y * (p.x - a.x)) / max(length(e), 1e-6);
}
// sin(0.45) and cos(0.45), where a self-loop meets its node
const float LOOP_SIN = 0.43496553, LOOP_COS = 0.90044710;
const vec4 HIDDEN = vec4(2.0, 2.0, 2.0, 1.0);
void main() {
  vec4 sn = texel(u_nodes, a_src), tn = texel(u_nodes, a_tgt);
  vec2 half_vp = 0.5 * u_viewport;
  // In pixels from the center, y up
  vec2 a = (sn.xy * u_scale + u_offset) * half_vp, b = (tn.xy * u_scale + u_offset) * half_vp;
  float ra = sn.z * u_radius, rb = tn.z * u_radius;
  // A stroke thinner than a pixel is drawn a pixel wide, as faint as it is thin
  float line = a_width * u_unit;
  vec4 col = texel(u_colors, a_color);
  float gain = col.a * a_opacity;
  v_gain = gain * min(line, 1.0);
  line = max(line, 1.0);
  float w = line + u_fringe;
  // The arrowhead grows with the edge's width, as the SVG's does
  float grow = max(a_width / 1.2, 0.6) * u_unit;
  float head_len = 8.0 * grow, head_wid = 5.0 * grow;
  // The path, as the SVG has it: a self-loop's two cubic curves above its node, p[0] to p[6],
  // or the quadratic curve p[0], p[1], p[2] between the nodes' borders
  bool loop = a_src == a_tgt;
  vec2 p[7];
  vec2 lo, hi;
  if (loop) {
    float lr = 1.2 * ra * a_loop_size, cp = 1.6 * lr;
    vec2 top = a + vec2(0.0, ra + lr);
    p[0] = a + ra * vec2(-LOOP_SIN, LOOP_COS);
    p[1] = p[0] + vec2(-0.3 * cp, cp);
    p[2] = top - vec2(lr, 0.0);
    p[3] = top + vec2(0.0, 0.2 * lr);
    p[4] = top + vec2(lr, 0.0);
    p[6] = a + ra * vec2(LOOP_SIN, LOOP_COS);
    p[5] = p[6] + vec2(0.3 * cp, cp);
    lo = p[0];
    hi = p[0];
    for (int i = 1; i < 7; i++) {
      lo = min(lo, p[i]);
      hi = max(hi, p[i]);
    }
  } else {
    // The control point, out from the middle by the bow's share of the length
    vec2 d = tn.xy - sn.xy;
    if (length(d) < 1e-6) { gl_Position = HIDDEN; return; }
    vec2 m = ((0.5 * (sn.xy + tn.xy) + vec2(-d.y, d.x) * a_bow) * u_scale + u_offset) * half_vp;
    lo = min(min(a, b), m);
    hi = max(max(a, b), m);
    vec2 s = length(m - a) > 1e-6 ? a + normalize(m - a) * ra : a;
    vec2 e = length(b - m) > 1e-6 ? b - normalize(b - m) * rb : b;
    p[0] = s;
    p[1] = a_bow == 0.0 ? 0.5 * (s + e) : m;
    p[2] = e;
  }
  // Edges whose hull is off screen collapse to nothing
  lo -= max(w, u_directed ? head_wid : 0.0);
  hi += max(w, u_directed ? head_wid : 0.0);
  if (v_gain <= 0.0 || any(greaterThan(lo, half_vp)) || any(lessThan(hi, -half_vp))) {
    gl_Position = HIDDEN;
    return;
  }
  v_color = col.rgb;
  vec2 tip = loop ? p[6] : p[2];
  vec2 dir = loop ? p[6] - p[5] : p[2] - p[1];
  bool head = u_directed && length(dir) > 1e-6;
  vec2 q0, q1, q2;
  if (head) {
    vec2 u = normalize(dir);
    vec2 base = tip - u * head_len, across = vec2(-u.y, u.x) * 0.5 * head_wid;
    q0 = tip;
    q1 = base + across;
    q2 = base - across;
  }
  if (a_corner.x > 1.5) {
    // The arrowhead, grown about its incenter so that its sides fade out over a pixel
    if (!head) { gl_Position = HIDDEN; return; }
    float l0 = length(q1 - q2), l1 = length(q2 - q0), l2 = length(q0 - q1);
    float around = l0 + l1 + l2;
    vec2 center = (l0 * q0 + l1 * q1 + l2 * q2) / around;
    float inner = head_len * head_wid / around;
    int k = int(a_corner.y);
    vec2 q = k == 0 ? q0 : k == 1 ? q1 : q2;
    vec2 pt = center + (q - center) * (inner + 1.0) / inner;
    v_side = vec3(inside(pt, q0, q1, q2), inside(pt, q1, q2, q0), inside(pt, q2, q0, q1));
    v_head = vec3(-1e4);
    v_gain = gain;
    gl_Position = vec4(pt / half_vp, 0.0, 1.0);
    return;
  }
  float k = a_corner.x;
  vec2 pt, tan;
  if (loop) {
    if (k < 0.5) {
      pt = cubic(p[0], p[1], p[2], p[3], 2.0 * k);
      tan = quad(p[1] - p[0], p[2] - p[1], p[3] - p[2], 2.0 * k);
    } else if (k > 0.5) {
      pt = cubic(p[3], p[4], p[5], p[6], 2.0 * k - 1.0);
      tan = quad(p[4] - p[3], p[5] - p[4], p[6] - p[5], 2.0 * k - 1.0);
    } else {
      // Where the two meet, at an angle
      pt = p[3];
      tan = normalize(p[3] - p[2]) + normalize(p[4] - p[3]);
    }
  } else {
    // A straight edge stops at its arrowhead's base, and a curve runs on to the tip
    if (head && a_bow == 0.0) k *= max(0.0, 1.0 - head_len / length(2.0 * dir));
    pt = quad(p[0], p[1], p[2], k);
    tan = mix(p[1] - p[0], p[2] - p[1], k);
  }
  float tl = length(tan);
  vec2 side = tl > 0.0 ? vec2(-tan.y, tan.x) / tl : vec2(0.0, 1.0);
  float dist = a_corner.y * 0.5 * w;
  v_side = vec3(0.5 * line + dist, 0.5 * line - dist, 1e4);
  pt += side * dist;
  // The stroke gives way where the arrowhead covers it, so that the two are drawn as one shape,
  // as the SVG's group draws them
  v_head = head
    ? vec3(inside(pt, q0, q1, q2), inside(pt, q1, q2, q0), inside(pt, q2, q0, q1))
    : vec3(-1e4);
  gl_Position = vec4(pt / half_vp, 0.0, 1.0);
}`;

const LINK_FS = `#version 300 es
precision highp float;
in vec3 v_color, v_side, v_head;
in float v_gain;
uniform bool u_density;
out vec4 o;
float cover(vec3 s) { return clamp(0.5 + min(min(s.x, s.y), s.z), 0.0, 1.0); }
void main() {
  float a = v_gain * cover(v_side) * (1.0 - cover(v_head));
  if (a <= 0.0) discard;
  // The density, or else the stroke over what's under it
  float d = u_density ? -log(1.0 - min(a, 0.999)) : a;
  o = vec4(v_color * d, d);
}`;

const POINT_VS = `#version 300 es
precision highp float;
in vec2 a_quad;
in uint a_index;
uniform highp sampler2D u_nodes, u_styles, u_colors;
uniform vec2 u_scale, u_offset, u_viewport;
uniform float u_unit;        // pixels per unit of the scene's radii
out vec2 v_uv;
flat out vec3 v_fill, v_border;
flat out float v_r, v_fill_gain, v_bw, v_border_gain;
vec4 texel(highp sampler2D t, uint i) { return texelFetch(t, ivec2(int(i & 2047u), int(i >> 11)), 0); }
void main() {
  vec4 nd = texel(u_nodes, a_index), st = texel(u_styles, a_index);
  // Hidden by the legend
  if (st.w > 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec4 fill = texel(u_colors, uint(nd.w)), border = texel(u_colors, uint(st.x));
  // A node or border thinner than a pixel is drawn a pixel across, as faint as it is small
  float r = nd.z * u_unit;
  v_fill_gain = fill.a * (r < 0.5 ? pow(r / 0.5, 2.0) : 1.0);
  v_r = max(r, 0.5);
  float bw = st.y * u_unit / 2.0;
  v_border_gain = bw <= 0.0 ? 0.0 : st.z * border.a * (bw < 0.5 ? bw / 0.5 : 1.0);
  v_bw = max(bw, 0.5);
  v_fill = fill.rgb;
  v_border = border.rgb;
  v_uv = a_quad * (v_r + v_bw + 1.0);
  vec2 c = nd.xy * u_scale + u_offset;
  gl_Position = vec4(c + v_uv / (0.5 * u_viewport), 0.0, 1.0);
}`;

// Shaded, a node is the gradient of `src/scene/shade.rs`, as the SVG and the PNG draw it
const POINT_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
flat in vec3 v_fill, v_border;
flat in float v_r, v_fill_gain, v_bw, v_border_gain;
uniform bool u_shade;
out vec4 o;
void main() {
  float d = length(v_uv);
  float af = clamp(v_r + 0.5 - d, 0.0, 1.0) * v_fill_gain;
  float ab = v_border_gain * clamp(v_bw + 0.5 - abs(d - v_r), 0.0, 1.0);
  if (af <= 0.0 && ab <= 0.0) discard;
  vec3 fill = v_fill;
  if (u_shade) {
    // From the highlight, up and to the left (y runs up), lighter there and darker past it
    float t = min(length(v_uv - vec2(-0.35, 0.35) * v_r) / (1.5 * v_r), 1.0);
    fill = t < 0.45
      ? mix(mix(v_fill, vec3(1.0), 0.5), v_fill, t / 0.45)
      : mix(v_fill, 0.65 * v_fill, (t - 0.45) / 0.55);
  }
  vec4 c = vec4(fill * af, af);
  c = vec4(v_border * ab, ab) + c * (1.0 - ab);
  o = c;
}`;

const TONE_VS = `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// The density as opacity: as SVG strokes stack up, or with the log tone (`u_lone` above 0), so
// up to a lone edge's density and from there by the density's logarithm, opaque at the view's
// densest pixel (or past it), as the raster's `ToneMap` has it
const TONE_FS = `#version 300 es
precision highp float;
uniform sampler2D u_density, u_peak;
uniform vec2 u_size;
uniform float u_gain, u_lone;
out vec4 o;
void main() {
  vec4 d = u_gain * texture(u_density, gl_FragCoord.xy / u_size);
  if (d.a <= 0.0) discard;
  float a = 1.0 - exp(-d.a);
  if (u_lone > 0.0 && d.a > u_lone) {
    float top = max(max(u_gain * texelFetch(u_peak, ivec2(0), 0).r, 3.912), u_lone * 1.01);
    float lone_a = 1.0 - exp(-u_lone);
    a = lone_a + (1.0 - lone_a) * min(log(d.a / u_lone) / log(top / u_lone), 1.0);
  }
  o = vec4(d.rgb / d.a * a, a);
}`;

// A level of the pyramid of maxima: each pixel the most of a 4 by 4 block of the level below,
// whose density is its alpha (the first) or its red
const PEAK_FS = `#version 300 es
precision highp float;
uniform sampler2D u_from;
uniform ivec2 u_from_size;
uniform bool u_first;
out vec4 o;
void main() {
  ivec2 base = 4 * ivec2(gl_FragCoord.xy);
  float most = 0.0;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      vec4 t = texelFetch(u_from, min(base + ivec2(i, j), u_from_size - 1), 0);
      most = max(most, u_first ? t.a : t.r);
    }
  }
  o = vec4(most, 0.0, 0.0, 1.0);
}`;

function compile(gl, vs, fs) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const u = {};
  const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const name = gl.getActiveUniform(prog, i).name;
    u[name] = gl.getUniformLocation(prog, name);
  }
  const a = (name) => gl.getAttribLocation(prog, name);
  return { prog, u, a };
}

// A float texture of `rows(i)`, four floats for each of `n` entries
function table(gl, n, fill) {
  const rows = Math.max(1, Math.ceil(n / TEX_W));
  const data = new Float32Array(TEX_W * rows * 4);
  for (let i = 0; i < n; i++) fill(data, 4 * i, i);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, rows, 0, gl.RGBA, gl.FLOAT, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  return tex;
}

const at = (col, i) => (typeof col === "number" ? col : col[i]);

const isCurved = (data) => typeof data.bow !== "number" || data.bow !== 0;

// A self-loop above its node at (x, y), of radius r and drawn `size` times the usual size, as
// `self_loop_points()` in `src/scene/geometry.rs` has it (y runs down): where it leaves the node,
// then the control points and the end of each of its two cubic curves, as [x0, y0, x1, ...]
const LOOP_SIN = 0.43496553411123023, LOOP_COS = 0.9004471023526769;
function loopPoints(x, y, r, size) {
  const lr = 1.2 * r * size, cp = 1.6 * lr, top = y - r - lr;
  const sx = x - r * LOOP_SIN, ex = x + r * LOOP_SIN, ey = y - r * LOOP_COS;
  return [sx, ey, sx - 0.3 * cp, ey - cp, x - lr, top, x, top - 0.2 * lr, x + lr, top, ex + 0.3 * cp, ey - cp, ex, ey];
}

const hasLoops = (data) => {
  const { src, tgt } = data, m = data.header.edges;
  for (let j = 0; j < m; j++) if (at(src, j) === at(tgt, j)) return true;
  return false;
};

// The drawing: `draw(view, count)` draws a view with the first `count` edges, `refine(chunk)`
// adds the next ones, `setColors(colors, bg)` changes the color table and the background,
// `update(names)` reads the columns of `data` with those names again, `setShade(on)` shades the
// nodes as spheres or not, and `setTone(tone)` tones the edges, "over" or "log" (which needs
// the density buffer, as `density` says)
function renderer(canvas, data) {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const { header } = data;
  const n = header.nodes, m = header.edges;
  const density = gl.getExtension("EXT_color_buffer_float") !== null;

  const nodeTable = () => table(gl, n, (t, j, i) => t.set([at(data.x, i), at(data.y, i), at(data.r, i), at(data.fill, i)], j));
  let nodes = nodeTable();
  // The last is 1 for a node hidden
  const styleTable = () => table(gl, n, (t, j, i) => t.set([at(data.border, i), at(data.border_width, i), at(data.border_opacity, i), at(data.hidden, i)], j));
  let styles = styleTable();
  let colors = null;

  // Edges: a strip per instance, along a quadratic curve (or a self-loop's two cubic ones) in
  // pieces, then for a directed graph its arrowhead, after a turn of empty triangles. A long
  // thin triangle costs as much as the box around it, so a straight edge is split too
  const link = compile(gl, LINK_VS, LINK_FS);
  const loops = hasLoops(data);
  let segments = 0, corners = 0;
  const linkVao = gl.createVertexArray();
  const stripBuf = gl.createBuffer();
  gl.bindVertexArray(linkVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, stripBuf);
  gl.enableVertexAttribArray(link.a("a_corner"));
  gl.vertexAttribPointer(link.a("a_corner"), 2, gl.FLOAT, false, 0, 0);
  function stripFor() {
    segments = isCurved(data) || loops ? 16 : 8;
    const head = header.directed ? [1, 1, 2, 0, 2, 0, 2, 1, 2, 2] : [];
    const strip = new Float32Array((segments + 1) * 4 + head.length);
    for (let s = 0; s <= segments; s++) strip.set([s / segments, -1, s / segments, 1], 4 * s);
    strip.set(head, (segments + 1) * 4);
    corners = strip.length / 2;
    gl.bindBuffer(gl.ARRAY_BUFFER, stripBuf);
    gl.bufferData(gl.ARRAY_BUFFER, strip, gl.STATIC_DRAW);
  }
  stripFor();
  // A column per instance, or one value for all
  const linkCols = [];
  for (const [name, int] of [["src", 1], ["tgt", 1], ["color", 1], ["width", 0], ["opacity", 0], ["bow", 0], ["loop_size", 0]]) {
    const loc = link.a("a_" + name);
    if (loc >= 0) linkCols.push({ name, loc, int, buf: null, value: 0 });
  }
  function edgeColumn(c) {
    const col = data[c.name];
    gl.bindVertexArray(linkVao);
    if (typeof col === "number") {
      if (c.buf) {
        gl.disableVertexAttribArray(c.loc);
        gl.deleteBuffer(c.buf);
        c.buf = null;
      }
      c.value = col;
    } else {
      if (!c.buf) {
        c.buf = gl.createBuffer();
        gl.enableVertexAttribArray(c.loc);
        gl.vertexAttribDivisor(c.loc, 1);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, c.buf);
      gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    }
    gl.bindVertexArray(null);
  }
  for (const c of linkCols) edgeColumn(c);
  // WebGL 2 has no base instance, so a range of edges starts where the columns are pointed.
  // The values for all are the context's, not the vertex array's, so they are set each time
  function pointEdges(from) {
    for (const { loc, buf, int, value } of linkCols) {
      if (buf) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        if (int) gl.vertexAttribIPointer(loc, 1, gl.UNSIGNED_INT, 0, from * 4);
        else gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, from * 4);
      } else if (int) {
        gl.vertexAttribI4ui(loc, value, 0, 0, 0);
      } else {
        gl.vertexAttrib1f(loc, value);
      }
    }
  }

  // Nodes: a quad per instance, in the scene's order
  const point = compile(gl, POINT_VS, POINT_FS);
  const pointVao = gl.createVertexArray();
  gl.bindVertexArray(pointVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(point.a("a_quad"));
  gl.vertexAttribPointer(point.a("a_quad"), 2, gl.FLOAT, false, 0, 0);
  const orderBuf = gl.createBuffer();
  const orderFor = () => {
    gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
    gl.bufferData(gl.ARRAY_BUFFER, data.order ?? Uint32Array.from({ length: n }, (_, i) => i), gl.STATIC_DRAW);
  };
  orderFor();
  gl.enableVertexAttribArray(point.a("a_index"));
  gl.vertexAttribIPointer(point.a("a_index"), 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(point.a("a_index"), 1);
  gl.bindVertexArray(null);

  const tone = compile(gl, TONE_VS, TONE_FS);
  const peak = compile(gl, TONE_VS, PEAK_FS);
  const emptyVao = gl.createVertexArray();
  // The density buffer, and for the log tone the pyramid of its maxima down to a pixel
  let fbo = null, fboTex = null, fboSize = [0, 0], levels = [];
  function texture(format, w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, format, w, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb, w, h };
  }
  function target(w, h) {
    if (fboSize[0] === w && fboSize[1] === h) return;
    for (const t of [{ tex: fboTex, fb: fbo }, ...levels]) {
      if (t.fb) { gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.tex); }
    }
    ({ tex: fboTex, fb: fbo } = texture(gl.RGBA16F, w, h));
    levels = [];
    let lw = w, lh = h;
    do {
      lw = Math.ceil(lw / 4);
      lh = Math.ceil(lh / 4);
      levels.push(texture(gl.R16F, lw, lh));
    } while (lw > 1 || lh > 1);
    fboSize = [w, h];
  }
  // The view's densest pixel, into the pyramid's last level
  function reduce(v) {
    gl.disable(gl.BLEND);
    gl.useProgram(peak.prog);
    gl.bindVertexArray(emptyVao);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(peak.u.u_from, 0);
    let from = { tex: fboTex, w: v.w, h: v.h };
    for (const level of levels) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, level.fb);
      gl.viewport(0, 0, level.w, level.h);
      gl.bindTexture(gl.TEXTURE_2D, from.tex);
      gl.uniform2i(peak.u.u_from_size, from.w, from.h);
      gl.uniform1i(peak.u.u_first, from.tex === fboTex ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      from = level;
    }
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.BLEND);
    return from.tex;
  }

  let bg = [1, 1, 1], alphas = [];
  let drawn = 0, view = null, shade = false, log = false;
  // A lone edge's density, for the log tone: the median edge's, as the raster has it, over up to
  // 4096 of the edges evenly spread; kept until the unit, the widths, the opacities or the
  // colors change
  let lone = null;
  function loneAt(unit) {
    if (lone && lone.unit === unit) return lone.value;
    const step = Math.max(1, Math.floor(m / 4096)), ds = [];
    for (let j = 0; j < m; j += step) {
      const gain = (alphas[at(data.color, j)] ?? 1) * at(data.opacity, j) * Math.min(at(data.width, j) * unit, 1);
      if (gain > 0) ds.push(-Math.log(1 - Math.min(gain, 0.999)));
    }
    ds.sort((a, b) => a - b);
    lone = { unit, value: Math.max(ds.length ? ds[ds.length >> 1] : 0, 1e-3) };
    return lone.value;
  }

  function bind(prog, v) {
    gl.useProgram(prog.prog);
    gl.uniform2f(prog.u.u_scale, ...v.scale);
    gl.uniform2f(prog.u.u_offset, ...v.offset);
    gl.uniform2f(prog.u.u_viewport, v.w, v.h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, nodes);
    gl.uniform1i(prog.u.u_nodes, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, colors);
    gl.uniform1i(prog.u.u_colors, 1);
  }

  // Edges [from, to), into the density buffer (cleared first with `clear`), or onto the canvas
  function edges(v, from, to, clear) {
    if (density) {
      target(v.w, v.h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      if (clear) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
      gl.blendFunc(gl.ONE, gl.ONE);
    } else {
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }
    gl.viewport(0, 0, v.w, v.h);
    gl.enable(gl.BLEND);
    bind(link, v);
    gl.uniform1f(link.u.u_unit, v.unit);
    gl.uniform1f(link.u.u_radius, v.radius);
    gl.uniform1f(link.u.u_fringe, 1.5);
    gl.uniform1i(link.u.u_directed, header.directed ? 1 : 0);
    gl.uniform1i(link.u.u_density, density ? 1 : 0);
    gl.bindVertexArray(linkVao);
    pointEdges(from);
    if (to > from) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, corners, to - from);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    drawn = clear ? to : drawn + (to - from);
  }

  // The density over the background, as if every edge were drawn, then the nodes
  function finish(v) {
    if (density) {
      const top = log && m ? reduce(v) : null;
      gl.viewport(0, 0, v.w, v.h);
      gl.clearColor(...bg, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(tone.prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fboTex);
      gl.uniform1i(tone.u.u_density, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, top ?? fboTex);
      gl.uniform1i(tone.u.u_peak, 1);
      gl.uniform1f(tone.u.u_lone, top ? loneAt(v.unit) : 0);
      gl.uniform2f(tone.u.u_size, v.w, v.h);
      gl.uniform1f(tone.u.u_gain, m / Math.max(drawn, 1));
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      gl.viewport(0, 0, v.w, v.h);
    }
    bind(point, v);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, styles);
    gl.uniform1i(point.u.u_styles, 2);
    gl.uniform1f(point.u.u_unit, v.radius);
    gl.uniform1i(point.u.u_shade, shade ? 1 : 0);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(pointVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.bindVertexArray(null);
  }

  return {
    gl,
    density,
    edgeCount: m,
    setColors(table4, background) {
      if (colors) gl.deleteTexture(colors);
      colors = table(gl, table4.length, (t, j, i) => t.set(table4[i], j));
      alphas = table4.map((c) => c[3]);
      lone = null;
      bg = background;
    },
    setShade(on) {
      shade = on;
    },
    setTone(name) {
      log = density && name === "log";
    },
    update(names) {
      if (["width", "opacity", "color"].some((name) => names.includes(name))) lone = null;
      if (names.includes("r") || names.includes("fill")) {
        gl.deleteTexture(nodes);
        nodes = nodeTable();
      }
      if (["border", "border_width", "border_opacity", "hidden"].some((name) => names.includes(name))) {
        gl.deleteTexture(styles);
        styles = styleTable();
      }
      if (names.includes("order")) orderFor();
      for (const c of linkCols) if (names.includes(c.name)) edgeColumn(c);
      if (names.includes("bow")) stripFor();
    },
    // `v`: the canvas's size in pixels, the scale and offset from the scene's units to clip
    // space, and the pixels per unit of edge width and of node radius
    draw(v, count = m) {
      view = v;
      if (!density) {
        gl.viewport(0, 0, v.w, v.h);
        gl.clearColor(...bg, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        count = m;
      }
      edges(v, 0, Math.min(m, count), true);
      finish(v);
    },
    // Adds the next `chunk` edges to the view, and redraws it; false once every edge is in
    refine(chunk) {
      if (!density || !view || drawn >= m) return false;
      edges(view, drawn, Math.min(m, drawn + chunk), false);
      finish(view);
      return true;
    },
  };
}

// The labels, as text on a canvas over the drawing, by the rule that `src/scene/labels.rs` has
// for the SVG: up to `state.limit` of the nodes' labels in view (null for all of them), taken in
// order of priority (the payload's `label_order`), each one that doesn't overlap one taken
// before it, then the edges' labels likewise, the selected edges first (`v.edgeFirst`), then the
// highlighted ones, each along its curve at its middle where the edge is long enough for it.
// Nodes that the legend hides (`data.hidden`), and edges drawn without opacity, have none. They
// keep their size in pixels as the view zooms, in `state.size` for the nodes' and `EDGE_FONT` for
// the edges', ringed by the background
const EDGE_FONT = 10;
const LABEL_GAP = 2;
function labelLayer(layer, data, font) {
  const { header } = data;
  const n = header.nodes, m = header.edges;
  const info = header.labels;
  const state = { show: info ? info.show : false, limit: info ? info.limit : null, size: info ? info.font_size : 12 };
  const ctx = layer.getContext("2d");
  let texts = null;
  // The labels' widths, measured once at each font size
  let nodeW = null, edgeW = null, measured = "";
  // What the last frame drew: the nodes whose labels it drew, the edges', and their boxes
  const drawn = { nodes: [], edges: [], boxes: [] };
  function draw(v) {
    if (layer.width !== v.w || layer.height !== v.h) [layer.width, layer.height] = [v.w, v.h];
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, v.w, v.h);
    drawn.nodes = [];
    drawn.edges = [];
    drawn.boxes = [];
    if (!texts || !(v.cw > 0 && v.ch > 0)) return;
    ctx.setTransform(v.w / v.cw, 0, 0, v.h / v.ch, 0, 0);
    Object.assign(ctx, { lineJoin: "round", lineWidth: 3, strokeStyle: v.bg, textAlign: "center" });
    // The boxes taken, by the cells of a grid over the view that they cover
    const cell = 64, cols = Math.ceil(v.cw / cell) + 2, rows = Math.ceil(v.ch / cell) + 2;
    const taken = new Map();
    const cells = (b, f) => {
      const c0 = Math.max(-1, Math.floor(b[0] / cell)), c1 = Math.min(cols, Math.floor(b[2] / cell));
      const r0 = Math.max(-1, Math.floor(b[1] / cell)), r1 = Math.min(rows, Math.floor(b[3] / cell));
      for (let c = c0; c <= c1; c++) for (let r = r0; r <= r1; r++) if (f((c + 1) * (rows + 2) + r + 1) === false) return false;
      return true;
    };
    const clear = (b) => cells(b, (key) => !(taken.get(key) ?? []).some((o) =>
      b[0] < o[2] + LABEL_GAP && o[0] < b[2] + LABEL_GAP && b[1] < o[3] + LABEL_GAP && o[1] < b[3] + LABEL_GAP));
    const take = (b) => {
      drawn.boxes.push(b);
      cells(b, (key) => { const list = taken.get(key); if (list) list.push(b); else taken.set(key, [b]); });
    };
    const limit = state.limit;
    const X = (i) => v.cw / 2 + (at(data.x, i) - v.cx) * v.px;
    const Y = (i) => v.ch / 2 + (at(data.y, i) - v.cy) * v.px;

    // The nodes', under them
    if (state.show && texts.nodes) {
      const size = state.size;
      ctx.font = `${size}px ${font}`;
      if (measured !== ctx.font) {
        nodeW = new Float32Array(n).fill(-1);
        measured = ctx.font;
      }
      const placed = [];
      for (let q = 0; q < n && (limit === null || placed.length < limit); q++) {
        const k = at(data.label_order, q);
        const text = texts.nodes[k];
        if (!text || at(data.hidden, k) > 0) continue;
        const top = Y(k) + at(data.r, k) * v.grow + 2;
        if (top > v.ch || top + 1.25 * size < 0) continue;
        if (nodeW[k] < 0) nodeW[k] = ctx.measureText(text).width;
        const x = X(k), half = nodeW[k] / 2;
        if (x + half < 0 || x - half > v.cw) continue;
        const b = [x - half, top, x + half, top + 1.25 * size];
        if (limit !== null && !clear(b)) continue;
        take(b);
        placed.push([k, x, top + size]);
      }
      // The first taken over the rest
      for (let p = placed.length - 1; p >= 0; p--) {
        const [k, x, y] = placed[p];
        ctx.strokeText(texts.nodes[k], x, y);
        ctx.fillStyle = v.colors[at(data.label_color, k)];
        ctx.fillText(texts.nodes[k], x, y);
      }
      drawn.nodes = placed.map(([k]) => k);
    }

    // The edges', the selected first, then the last drawn, which are the highlighted ones
    if (texts.edges && m) {
      ctx.font = `${EDGE_FONT}px ${font}`;
      edgeW ??= new Float32Array(m).fill(-1);
      const placed = [], hh = 0.625 * EDGE_FONT;
      const edge = (j) => {
        const row = v.edgeRow ? v.edgeRow[j] : j;
        const text = texts.edges[row];
        const s = at(data.src, j), t = at(data.tgt, j);
        if (!text || s === t || at(data.opacity, j) <= 0) return;
        // The middle of its curve (see `hover()`), and the curve's direction there, the chord's
        const sx = X(s), sy = Y(s), tx = X(t), ty = Y(t), bow = at(data.bow, j);
        const mx = (sx + tx) / 2 - (ty - sy) * bow / 2, my = (sy + ty) / 2 + (tx - sx) * bow / 2;
        if (mx < 0 || mx > v.cw || my < 0 || my > v.ch) return;
        if (edgeW[row] < 0) edgeW[row] = ctx.measureText(text).width;
        const w = edgeW[row];
        if (Math.hypot(tx - sx, ty - sy) < w + (at(data.r, s) + at(data.r, t)) * v.grow + 8) return;
        // Read from left to right
        let a = Math.atan2(ty - sy, tx - sx);
        if (a > Math.PI / 2) a -= Math.PI;
        else if (a < -Math.PI / 2) a += Math.PI;
        const cos = Math.abs(Math.cos(a)), sin = Math.abs(Math.sin(a));
        const hx = (cos * w) / 2 + sin * hh, hy = (sin * w) / 2 + cos * hh;
        const b = [mx - hx, my - hy, mx + hx, my + hy];
        if (limit !== null && !clear(b)) return;
        take(b);
        placed.push([text, mx, my, a, j]);
      };
      const first = v.edgeFirst ?? [], skip = new Set(first);
      for (let q = 0; q < first.length && (limit === null || placed.length < limit); q++) edge(first[q]);
      for (let j = m - 1; j >= 0 && (limit === null || placed.length < limit); j--) if (!skip.has(j)) edge(j);
      ctx.textBaseline = "middle";
      ctx.fillStyle = v.ink;
      const base = ctx.getTransform();
      for (const [text, x, y, a] of placed) {
        ctx.translate(x, y);
        ctx.rotate(a);
        ctx.strokeText(text, 0, 0);
        ctx.globalAlpha = 0.7;
        ctx.fillText(text, 0, 0);
        ctx.globalAlpha = 1;
        ctx.setTransform(base);
      }
      ctx.textBaseline = "alphabetic";
      drawn.edges = placed.map((p) => p[4]);
    }
  }
  return { state, draw, drawn, setTexts(t) { texts = t; } };
}

// A computed CSS color, "rgb(r, g, b)" or "rgba(...)", from 0 to 1
function cssRgb(color) {
  const v = (color.match(/[\d.]+/g) ?? [255, 255, 255]).slice(0, 3).map(Number);
  return v.map((c) => c / 255);
}

// `--ec-side` is the width that the docked panel takes from the drawing
const STYLE = `
.ec-viewer { position: relative; overflow: hidden; user-select: none; -webkit-user-select: none; }
.ec-viewer.ec-docked.ec-panel-open { --ec-side: ${PANEL_W}px; }
.ec-viewer [hidden] { display: none !important; }
.ec-viewer canvas { position: absolute; inset: 0; width: calc(100% - var(--ec-side, 0px)); height: 100%; display: block; touch-action: pan-x pan-y; outline: none; }
.ec-viewer canvas.ec-text { pointer-events: none; }
.ec-viewer canvas.ec-grab { cursor: grab; }
.ec-viewer canvas.ec-grabbing { cursor: grabbing; }
.ec-viewer .ec-legend-layer { position: absolute; top: 0; right: var(--ec-side, 0px) !important; pointer-events: none; }
.ec-viewer button { display: flex; align-items: center; justify-content: center; width: 26px; height: 26px; padding: 0; border: 0; border-radius: 6px; color: inherit; background: color-mix(in srgb, currentColor 14%, transparent); cursor: pointer; opacity: 0.75; }
.ec-viewer button:hover, .ec-viewer button:focus-visible { opacity: 1; }
.ec-viewer button svg { display: block; }
.ec-viewer .ec-controls { position: absolute; left: 8px; top: 8px; display: flex; flex-direction: column; gap: 4px; }
.ec-viewer .ec-controls .ec-apart { margin-top: 6px; }
.ec-viewer .ec-controls [aria-expanded=true], .ec-viewer .ec-controls [aria-pressed=true] { opacity: 1; background: color-mix(in srgb, currentColor 26%, transparent); }
.ec-viewer .ec-legend-toggle { position: absolute; right: calc(var(--ec-side, 0px) + 8px); top: 8px; display: none; }
.ec-viewer.ec-legend-folded .ec-legend-toggle { display: flex; }
.ec-viewer.ec-legend-folded .ec-legend-layer, .ec-viewer.ec-legend-off .ec-legend-layer, .ec-viewer.ec-legend-off .ec-legend-toggle { display: none; }
.ec-viewer.ec-legend-folded.ec-legend-open:not(.ec-legend-off) .ec-legend-layer { display: block; top: 40px !important; right: calc(var(--ec-side, 0px) + 8px) !important; pointer-events: auto !important; max-width: calc(100% - var(--ec-side, 0px) - 48px); max-height: calc(100% - 48px); overflow: auto; background: inherit; border-radius: 8px; box-shadow: 0 0 0 1px color-mix(in srgb, currentColor 15%, transparent), 0 4px 12px rgba(0, 0, 0, 0.15); }
.ec-viewer.ec-legend-folded.ec-legend-open .ec-legend-toggle { opacity: 1; }
.ec-viewer:fullscreen, .ec-viewer.ec-expanded { width: 100vw !important; height: 100vh !important; max-width: none !important; aspect-ratio: auto !important; margin: 0 !important; }
.ec-viewer.ec-expanded { position: fixed !important; inset: 0; z-index: 2147483000; }
html.ec-expanded-page, html.ec-expanded-page body { overflow: hidden !important; }
.ec-viewer .ec-mark { position: absolute; inset: 0; width: calc(100% - var(--ec-side, 0px)); height: 100%; pointer-events: none; overflow: visible; }
.ec-viewer .ec-mark path { fill: none; stroke: currentColor; stroke-linecap: round; }
.ec-viewer .ec-mark path.ec-lasso { stroke-width: 1.5; stroke-dasharray: 4 3; stroke-linejoin: round; fill: color-mix(in srgb, currentColor 6%, transparent); }
.ec-viewer .ec-search { position: absolute; left: 38px; top: 8px; width: 26px; height: 26px; border-radius: 6px; background: color-mix(in srgb, currentColor 14%, transparent); opacity: 0.75; overflow: hidden; transition: width 0.15s ease; }
.ec-viewer .ec-search:hover { opacity: 1; }
.ec-viewer .ec-search:focus-within { width: min(200px, calc(100% - var(--ec-side, 0px) - 46px)); opacity: 1; background: inherit; box-shadow: 0 0 0 1.5px color-mix(in srgb, currentColor 60%, transparent); }
.ec-viewer .ec-search svg { position: absolute; left: 5px; top: 5px; pointer-events: none; }
.ec-viewer .ec-search input { position: absolute; inset: 0; width: 100%; height: 100%; box-sizing: border-box; margin: 0; padding: 0 8px 0 26px; border: 0; outline: none; background: transparent; color: inherit; font-size: 12px; cursor: pointer; }
.ec-viewer .ec-search:focus-within input { cursor: text; }
.ec-viewer .ec-search input::placeholder { color: inherit; opacity: 0.6; }
.ec-viewer .ec-search:not(:focus-within) input::placeholder { color: transparent; }
.ec-viewer .ec-results { position: absolute; left: 38px; top: 38px; width: min(200px, calc(100% - var(--ec-side, 0px) - 46px)); box-sizing: border-box; margin: 0; padding: 4px; list-style: none; background: inherit; border-radius: 8px; box-shadow: 0 0 0 1px color-mix(in srgb, currentColor 15%, transparent), 0 4px 12px rgba(0, 0, 0, 0.15); font-size: 12px; line-height: 1.4; text-align: left; }
.ec-viewer .ec-results li { padding: 4px 6px; border-radius: 4px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ec-viewer .ec-results li[aria-selected=true] { background: color-mix(in srgb, currentColor 14%, transparent); }
.ec-viewer .ec-results .ec-dim { margin-left: 6px; opacity: 0.6; }
.ec-viewer .ec-results .ec-none { cursor: default; opacity: 0.7; }
.ec-viewer .ec-legend-layer .ec-key { pointer-events: auto; cursor: pointer; outline: none; }
.ec-viewer .ec-legend-layer .ec-key.ec-off { opacity: 0.35; }
.ec-viewer .ec-legend-layer .ec-key:focus-visible rect { stroke: currentColor; stroke-width: 1.5; }
.ec-viewer .ec-tip { position: absolute; left: 0; top: 0; display: none; padding: 6px 10px; border-radius: 6px; font-size: 12px; line-height: 1.4; pointer-events: none; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18); max-width: min(260px, 100%); box-sizing: border-box; overflow-wrap: anywhere; }
.ec-viewer .ec-tip.ec-on { display: block; }
.ec-viewer .ec-hint { position: absolute; left: calc((100% - var(--ec-side, 0px)) / 2); bottom: 10px; transform: translateX(-50%); padding: 4px 10px; border-radius: 6px; font: 12px system-ui, sans-serif; color: inherit; background: color-mix(in srgb, currentColor 14%, transparent); pointer-events: none; opacity: 0; transition: opacity 0.3s; white-space: nowrap; }
.ec-viewer .ec-hint.ec-on { opacity: 1; }
.ec-viewer .ec-message { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font: 13px system-ui, sans-serif; }
.ec-viewer .ec-panel { position: absolute; top: 0; right: 0; bottom: 0; width: min(${PANEL_W}px, calc(100% - 40px)); box-sizing: border-box; display: flex; flex-direction: column; background: inherit; color: inherit; font-size: 12px; line-height: 1.4; text-align: left; user-select: text; -webkit-user-select: text; box-shadow: -6px 0 18px rgba(0, 0, 0, 0.18), 0 0 0 1px color-mix(in srgb, currentColor 15%, transparent); transform: translateX(calc(100% + 24px)); visibility: hidden; transition: transform 0.2s ease, visibility 0s linear 0.2s; }
.ec-viewer.ec-panel-open .ec-panel { transform: none; visibility: visible; transition: transform 0.2s ease, visibility 0s; }
.ec-viewer.ec-docked .ec-panel { width: ${PANEL_W}px; box-shadow: -1px 0 0 color-mix(in srgb, currentColor 15%, transparent); }
@media (prefers-reduced-motion: reduce) { .ec-viewer .ec-panel, .ec-viewer.ec-panel-open .ec-panel, .ec-viewer .ec-search { transition: none; } }
.ec-viewer .ec-panel button { width: auto; height: auto; padding: 4px 8px; font: inherit; background: none; opacity: 1; }
.ec-viewer .ec-panel :focus-visible { outline: 2px solid currentColor; outline-offset: 1px; }
.ec-viewer .ec-panel-head { display: flex; align-items: flex-end; gap: 4px; padding: 6px 6px 0 8px; border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
.ec-viewer .ec-tabs { display: flex; flex: 1; gap: 2px; min-width: 0; overflow-x: auto; }
.ec-viewer .ec-panel [role=tab] { padding: 6px 8px; border-radius: 6px 6px 0 0; border-bottom: 2px solid transparent; opacity: 0.7; white-space: nowrap; }
.ec-viewer .ec-panel [role=tab][aria-selected=true] { opacity: 1; font-weight: 600; border-bottom-color: currentColor; }
.ec-viewer .ec-panel [role=tab].ec-changed::after { content: ""; display: inline-block; width: 5px; height: 5px; margin-left: 4px; border-radius: 50%; background: currentColor; vertical-align: 2px; }
.ec-viewer .ec-panel .ec-close { align-self: center; width: 26px; height: 26px; padding: 0; opacity: 0.75; }
.ec-viewer .ec-tabpanel { flex: 1; min-height: 0; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 14px; }
.ec-viewer .ec-field { display: flex; flex-direction: column; gap: 4px; }
.ec-viewer .ec-field-head { display: flex; align-items: center; gap: 6px; min-height: 22px; }
.ec-viewer .ec-field-head .ec-name { flex: 1; }
.ec-viewer .ec-changed > .ec-field-head .ec-name { font-weight: 600; }
.ec-viewer .ec-changed > .ec-field-head .ec-name::before { content: ""; display: inline-block; width: 6px; height: 6px; margin-right: 6px; border-radius: 50%; background: currentColor; vertical-align: 1px; }
.ec-viewer .ec-field output { font-variant-numeric: tabular-nums; opacity: 0.8; }
.ec-viewer .ec-panel .ec-reset { width: 22px; height: 22px; padding: 0; opacity: 0.75; }
.ec-viewer .ec-field input[type=range] { width: 100%; margin: 0; accent-color: currentColor; }
.ec-viewer .ec-field input[type=checkbox], .ec-viewer .ec-field input[type=radio] { margin: 0; accent-color: currentColor; }
.ec-viewer .ec-choices { display: flex; flex-wrap: wrap; gap: 4px 14px; }
.ec-viewer .ec-choices label { display: flex; align-items: center; gap: 5px; }
.ec-viewer .ec-note { margin: 0; font-size: 11px; opacity: 0.7; }
.ec-viewer .ec-more summary { cursor: pointer; opacity: 0.8; }
.ec-viewer .ec-more[open] summary { margin-bottom: 10px; }
.ec-viewer .ec-more .ec-field + .ec-field { margin-top: 14px; }
.ec-viewer .ec-panel-foot .ec-note { flex-basis: 100%; }
.ec-viewer .ec-panel-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 12px; border-top: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
.ec-viewer .ec-panel .ec-copy { padding: 6px 10px; font-weight: 600; background: color-mix(in srgb, currentColor 12%, transparent); }
.ec-viewer .ec-status { opacity: 0.8; }
.ec-viewer .ec-code { width: 100%; min-height: 6em; box-sizing: border-box; padding: 6px; font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: inherit; background: transparent; border: 1px solid color-mix(in srgb, currentColor 25%, transparent); border-radius: 6px; resize: vertical; }
`;

// The theme, "light" or "dark", or a CSS variable that holds it: the automatic theme's, whose
// value follows the page's color scheme by the same rules as the SVG's colors
function themeOf(root, theme) {
  const name = /^var\((--[\w-]+)\)$/.exec(theme ?? "");
  return name ? getComputedStyle(root).getPropertyValue(name[1]).trim() : theme;
}

// options:
// - theme, the plot's (see `themeOf()`), and scheme, a variable that holds the theme that the
//   root's colors are in, which the theme control switches by a class on the root, `ec-light`,
//   `ec-dark` or `ec-auto` (see `_Colors.scoped()` in `svg_renderer.py`)
// - font, the panel's and the tooltips', and tip, the tooltips' colors (bg and fg)
// - call, the `plot()` call, for "Copy as code": `args`, [name, Python source] for each argument
//   given, and `params`, the parameters' names in order; and moves, true where `plot()` lays
//   the nodes out for their size
// - tone, the edges' tone that `plot()` was given (`value`: "auto", "over" or "log") and what
//   "auto" is for this plot (`auto`), and shade, true for nodes shaded as spheres
// The plot's background is `root`'s
export function mount(root, buffer, options = {}) {
  const data = parse(buffer);
  const { header } = data;
  // The nodes that the legend hides, none at first
  data.hidden = 0;
  if (!document.getElementById("ec-viewer-style")) {
    const style = document.createElement("style");
    style.id = "ec-viewer-style";
    style.textContent = STYLE;
    document.head.appendChild(style);
  }
  const canvas = document.createElement("canvas");
  canvas.tabIndex = 0;
  canvas.className = "ec-grab";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `Graph of ${header.nodes} nodes and ${header.edges} edges. Drag to pan; zoom with the buttons, the keys + and -, or the wheel once clicked; F for full screen; hover over a node or an edge for its details; click one to select it, with Shift to add it, Shift and drag to lasso nodes, Esc to clear; / to search the nodes; the Settings button opens the plot's controls`);
  root.prepend(canvas);
  // A moved view draws `BUDGET` edges at once, so they go in a random order, to be a fair share;
  // `edgeRow` is each one's place in the scene's order, its row of the tooltips
  const edgeRow = header.edges > BUDGET ? shuffleEdges(data) : null;
  // The selection, drawn as `highlight=` would draw what is selected, which "Copy as code"
  // writes: the nodes chosen, and the edges (j, in the drawing order), whose ends are highlighted
  // with them as `highlight_edges=` has them. An edge is chosen with the others between its ends,
  // as `highlight_edges=[(u, v)]` takes them. And the legend's swatches turned off (their places
  // plus 1, as the payload's "node_key" and "edge_key" count them), whose nodes and edges are
  // hidden, with the edges of the nodes hidden. Without a selection, the payload's own
  // highlighting is drawn
  const sel = header.select;
  const chosen = { nodes: new Set(), edges: new Set() };
  const off = new Set();
  // The columns as the payload has them (in the shuffled order, where the edges are shuffled)
  const orig = {
    r: data.r, width: data.width, opacity: data.opacity, bow: data.bow, fill: data.fill, border: data.border,
    border_width: data.border_width, border_opacity: data.border_opacity, order: data.order, color: data.color,
    label_color: data.label_color, label_order: data.label_order, hidden: data.hidden,
  };
  // The columns that the selection and the legend give, the edges' levels with a selection, those
  // hidden, and the selected edges, whose labels come first
  let styled = { ...orig, edgeLevel: null, edgeHidden: null, edgeFirst: null };
  const r = renderer(canvas, data);
  if (!r) {
    canvas.remove();
    const msg = document.createElement("div");
    msg.className = "ec-message";
    msg.textContent = "This plot needs WebGL 2, which this browser doesn't have.";
    root.append(msg);
    return null;
  }
  const textCanvas = document.createElement("canvas");
  textCanvas.className = "ec-text";
  textCanvas.setAttribute("aria-hidden", "true");
  canvas.after(textCanvas);
  const words = labelLayer(textCanvas, data, options.font ?? "system-ui, sans-serif");
  let labelTexts = null;

  // The camera: the scene point at the center, and the zoom. At zoom 1 the scene's frame fits
  // the canvas as the SVG's viewBox fits its box
  const [vx, vy, vw, vh] = header.viewbox;
  const home = { cx: vx + vw / 2, cy: vy + vh / 2, zoom: 1 };
  const cam = { ...home };
  const fit = (w, h) => Math.min(w / vw, h / vh);
  function view() {
    // At most `CANVAS_MAX` pixels along either side, scaled alike, so that the drawing keeps its
    // shape on a large screen
    const [cw, ch] = [canvas.clientWidth, canvas.clientHeight];
    const dpr = Math.min(window.devicePixelRatio || 1, CANVAS_MAX / Math.max(cw, ch, 1));
    const w = Math.max(1, Math.round(cw * dpr));
    const h = Math.max(1, Math.round(ch * dpr));
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
    const unit = fit(w, h);
    const px = unit * cam.zoom;
    const scale = [px / (w / 2), -px / (h / 2)];
    // Nodes grow with the square root of the zoom, so that zooming in opens up space between
    // them; edges keep their width
    return { w, h, scale, offset: [-cam.cx * scale[0], -cam.cy * scale[1]], unit, radius: unit * Math.sqrt(cam.zoom) };
  }

  // The theme's color table, and the background, which change with the page's color scheme,
  // and the text color, the edge labels'
  let shown = null;
  function applyTheme() {
    const dark = themeOf(root, options.scheme ?? options.theme) === "dark";
    const { backgroundColor: bg, color: ink } = getComputedStyle(root);
    if (shown && shown.dark === dark && shown.bg === bg && shown.ink === ink) return false;
    const table4 = dark ? header.colors.dark : header.colors.light;
    const colors = table4.map(([cr, cg, cb, ca]) => `rgba(${Math.round(cr * 255)}, ${Math.round(cg * 255)}, ${Math.round(cb * 255)}, ${ca})`);
    shown = { dark, bg, ink, colors };
    r.setColors(table4, cssRgb(bg));
    return true;
  }
  applyTheme();
  const tones = options.tone ?? { value: "auto", auto: "over" };
  const toneOf = (value) => (value === "auto" ? tones.auto : value);
  r.setTone(toneOf(tones.value));
  r.setShade(Boolean(options.shade));

  // A moved view draws `BUDGET` edges at once, and the rest over the frames that follow
  let pending = false, refining = 0;
  const redraw = () => {
    if (pending) return;
    pending = true;
    cancelAnimationFrame(refining);
    requestAnimationFrame(() => {
      pending = false;
      r.draw(view(), BUDGET);
      drawText();
      if (pointer) hover();
      const more = () => { if (r.refine(BUDGET)) refining = requestAnimationFrame(more); };
      refining = requestAnimationFrame(more);
    });
  };
  // The labels over the drawing, once their texts are read
  function drawText() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight, unit = fit(cw, ch);
    words.draw({
      w: canvas.width, h: canvas.height, cw, ch, cx: cam.cx, cy: cam.cy, px: unit * cam.zoom,
      grow: unit * Math.sqrt(cam.zoom), colors: shown.colors, bg: shown.bg, ink: shown.ink, edgeRow,
      edgeFirst: styled.edgeFirst,
    });
  }
  r.draw(view());
  drawText();
  readPacked(root, "labels").then((texts) => {
    labelTexts = texts;
    words.setTexts(texts);
    redraw();
  });

  // From a point on the page to the scene's units
  const toScene = (ex, ey) => {
    const b = canvas.getBoundingClientRect();
    const px = fit(b.width, b.height) * cam.zoom;
    return [cam.cx + (ex - b.left - b.width / 2) / px, cam.cy + (ey - b.top - b.height / 2) / px];
  };
  // Zoom by `f` about a point on the page, or the center
  function zoomBy(f, ex, ey) {
    const b = canvas.getBoundingClientRect();
    if (ex === undefined) [ex, ey] = [b.left + b.width / 2, b.top + b.height / 2];
    const [wx, wy] = toScene(ex, ey);
    cam.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, cam.zoom * f));
    const [nx, ny] = toScene(ex, ey);
    cam.cx += wx - nx;
    cam.cy += wy - ny;
    redraw();
  }
  function panBy(dx, dy) {
    const b = canvas.getBoundingClientRect();
    const px = fit(b.width, b.height) * cam.zoom;
    cam.cx -= dx / px;
    cam.cy -= dy / px;
    redraw();
  }
  function reset() {
    Object.assign(cam, home);
    redraw();
  }

  // Hover: the node or the edge under the pointer, outlined, with its tooltip, the SVG's, from
  // the tables of them, read on the first hover. A touch shows them with a tap
  const NS = "http://www.w3.org/2000/svg";
  const mark = document.createElementNS(NS, "svg");
  mark.setAttribute("class", "ec-mark");
  mark.setAttribute("aria-hidden", "true");
  const outline = document.createElementNS(NS, "path");
  mark.append(outline);
  textCanvas.after(mark);
  const tip = document.createElement("div");
  tip.className = "ec-tip";
  tip.setAttribute("role", "tooltip");
  const look = options.tip ?? {};
  Object.assign(tip.style, { background: look.bg ?? "", color: look.fg ?? "", fontFamily: options.font ?? "" });
  root.append(tip);
  // Made again when the controls change the drawing
  let pick = picker(data, isCurved(data) ? 16 : 1);
  // The pointer, on the page, and what it's over; the tooltips once read (null if there are
  // none), and the element whose tooltip is shown
  let pointer = null, hovered = null, tips, reading = null, filled = "";
  let queued = false;
  function hoverAt(ex, ey) {
    pointer = ex === undefined ? null : [ex, ey];
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      hover();
    });
  }
  // What is drawn under a point on the page
  function under(ex, ey) {
    const b = canvas.getBoundingClientRect();
    if (!(b.width > 0)) return null;
    const unit = fit(b.width, b.height);
    const [x, y] = toScene(ex, ey);
    return pick(x, y, { px: unit * cam.zoom, radius: unit * Math.sqrt(cam.zoom), unit, thin: b.width / Math.max(canvas.width, 1) });
  }
  function hover() {
    const b = canvas.getBoundingClientRect();
    const unit = fit(b.width, b.height), px = unit * cam.zoom, grow = unit * Math.sqrt(cam.zoom);
    const thin = b.width / Math.max(canvas.width, 1);
    hovered = pointer && down.size === 0 ? under(...pointer) : null;
    if (!hovered) {
      outline.removeAttribute("d");
      tip.classList.remove("ec-on");
      return;
    }
    // The outline: a ring around the node, or the edge drawn over itself
    const X = (sx) => (b.width / 2 + (sx - cam.cx) * px).toFixed(1);
    const Y = (sy) => (b.height / 2 + (sy - cam.cy) * px).toFixed(1);
    let key;
    if (hovered.node !== undefined) {
      const i = hovered.node;
      const ring = Math.max(at(data.r, i) * grow, thin / 2) + 2.5;
      const cx = b.width / 2 + (at(data.x, i) - cam.cx) * px;
      outline.setAttribute("d", `M${(cx - ring).toFixed(1)} ${Y(at(data.y, i))}a${ring} ${ring} 0 1 0 ${2 * ring} 0a${ring} ${ring} 0 1 0 ${-2 * ring} 0`);
      outline.setAttribute("stroke-width", "1.5");
      key = `n${i}`;
    } else {
      const j = hovered.edge, s = at(data.src, j), t = at(data.tgt, j);
      const sx = at(data.x, s), sy = at(data.y, s), tx = at(data.x, t), ty = at(data.y, t);
      if (s === t) {
        const p = loopPoints(+X(sx), +Y(sy), at(data.r, s) * grow, at(data.loop_size, j)).map((c) => c.toFixed(1));
        outline.setAttribute("d", `M${p[0]} ${p[1]}C${p.slice(2, 8).join(" ")}C${p.slice(8).join(" ")}`);
      } else {
        const bow = at(data.bow, j);
        const cx = (sx + tx) / 2 - (ty - sy) * bow, cy = (sy + ty) / 2 + (tx - sx) * bow;
        outline.setAttribute("d", `M${X(sx)} ${Y(sy)}Q${X(cx)} ${Y(cy)} ${X(tx)} ${Y(ty)}`);
      }
      outline.setAttribute("stroke-width", String(Math.max(at(data.width, j) * unit, 1) + 1));
      key = `e${j}`;
    }
    if (tips === undefined) {
      reading ??= readPacked(root, "tips").then((t) => {
        tips = t;
        hover();
      });
      return;
    }
    if (!tips) return;
    // Its rows, as text, since the values are the graph's own
    if (key !== filled) {
      filled = key;
      const rows = hovered.node !== undefined
        ? tipRow(tips.nodes, hovered.node)
        : tipRow(tips.edges, edgeRow ? edgeRow[hovered.edge] : hovered.edge);
      tip.replaceChildren(...rows.map(([k, v]) => {
        const line = document.createElement("div");
        const name = document.createElement("b");
        name.textContent = k;
        line.append(name, `: ${v}`);
        return line;
      }));
    }
    if (!tip.childElementCount) {
      tip.classList.remove("ec-on");
      return;
    }
    // Next to the pointer but inside the drawing, flipped to its left near the right edge
    tip.classList.add("ec-on");
    const box = root.getBoundingClientRect();
    const [w, h] = [tip.offsetWidth, tip.offsetHeight];
    let x = pointer[0] - box.left + 12;
    if (x + w > b.width) x = pointer[0] - box.left - 12 - w;
    tip.style.left = `${Math.max(0, Math.min(x, b.width - w))}px`;
    tip.style.top = `${Math.max(0, Math.min(pointer[1] - box.top - 10, box.height - h))}px`;
  }

  // A node's ID, as Python writes it
  const idOf = (i) => String((BigInt(at(data.node_id_hi, i)) << 32n) | BigInt(at(data.node_id, i)));
  function relook() {
    const n = header.nodes, m = header.edges;
    let hidden = 0, edgeHidden = null;
    if (off.size) {
      hidden = new Float32Array(n);
      for (let i = 0; i < n; i++) if (off.has(at(data.node_key, i))) hidden[i] = 1;
      edgeHidden = new Uint8Array(m);
      for (let j = 0; j < m; j++) {
        if (off.has(at(data.edge_key, j)) || hidden[at(data.src, j)] || hidden[at(data.tgt, j)]) edgeHidden[j] = 1;
      }
    }
    styled = { ...orig, hidden, edgeLevel: null, edgeHidden, edgeFirst: null };
    if (chosen.nodes.size || chosen.edges.size) {
      const level = new Uint8Array(n), edgeLevel = new Uint8Array(m);
      for (const i of chosen.nodes) level[i] = 2;
      for (const j of chosen.edges) {
        edgeLevel[j] = 2;
        level[at(data.src, j)] = level[at(data.tgt, j)] = 2;
      }
      // By the builder's rules (see `SceneBuilder.scene()`), from the columns without highlighting
      const fill = data.base_fill ?? orig.fill, border = data.base_border ?? orig.border;
      const color = data.base_color ?? orig.color, priority = data.base_label_order ?? orig.label_order;
      // Items of level `first`, then the others, each in their order
      const apart = (count, item, first) => {
        const out = new Uint32Array(count);
        let p = 0;
        for (const want of [first, 2 - first]) {
          for (let q = 0; q < count; q++) {
            const k = item(q);
            if (level[k] === want) out[p++] = k;
          }
        }
        return out;
      };
      Object.assign(styled, {
        fill: Uint32Array.from(level, (l, i) => (l ? (sel.node_colors ? at(fill, i) : sel.accent) : sel.muted)),
        border: Uint32Array.from(level, (l, i) => (l ? (sel.node_colors ? at(border, i) : sel.accent_border) : sel.muted_border)),
        border_width: 1.5,
        border_opacity: 0.3,
        // Highlighted nodes are drawn over the muted ones, and labelled first
        order: apart(n, (q) => q, 0),
        label_color: orig.label_color === null ? null : Uint32Array.from(level, (l) => (l ? sel.label : sel.muted_label)),
        label_order: priority === null ? null : apart(n, (q) => at(priority, q), 2),
        // A highlighted edge keeps a color that edge_color_by= gives it, or else takes the accent
        color: Uint32Array.from(edgeLevel, (l, j) => {
          const c = at(color, j);
          return l ? (!sel.edge_colors || c === sel.edge ? sel.accent : c) : sel.edge;
        }),
        edgeLevel,
        edgeFirst: [...chosen.edges],
      });
    }
    restyle();
  }
  function choose(nodes, edges) {
    chosen.nodes = new Set(nodes);
    chosen.edges = new Set(edges);
    relook();
  }
  // A click chooses what is under it, or with `add` (Shift, Command or Ctrl) adds it to the
  // selection or takes it out; on nothing, it clears the selection
  function clickAt(ex, ey, add) {
    if (!sel) return;
    const hit = under(ex, ey);
    if (!hit) {
      if (!add && (chosen.nodes.size || chosen.edges.size)) choose([], []);
      return;
    }
    if (hit.node !== undefined) {
      const nodes = new Set(add ? chosen.nodes : []);
      if (add && nodes.has(hit.node)) nodes.delete(hit.node);
      else nodes.add(hit.node);
      choose(nodes, add ? chosen.edges : []);
      return;
    }
    const s = at(data.src, hit.edge), t = at(data.tgt, hit.edge);
    const edges = new Set(add ? chosen.edges : []), had = add && edges.has(hit.edge);
    for (let j = 0; j < header.edges; j++) {
      if (at(data.src, j) !== s || at(data.tgt, j) !== t) continue;
      if (had) edges.delete(j);
      else edges.add(j);
    }
    choose(add ? chosen.nodes : [], edges);
  }

  // The lasso: a loop drawn with Shift held, or in lasso mode, whose nodes (those shown) join the
  // selection. Its points are in pixels of the canvas's box
  let lasso = null, lassoOn = false;
  const lassoPath = document.createElementNS(NS, "path");
  lassoPath.setAttribute("class", "ec-lasso");
  mark.append(lassoPath);
  function lassoTo(ex, ey) {
    const b = canvas.getBoundingClientRect(), p = [ex - b.left, ey - b.top], last = lasso[lasso.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 3) return;
    lasso.push(p);
    lassoPath.setAttribute("d", `M${lasso.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join("L")}Z`);
  }
  // Ends the lasso; true if it was a loop
  function lassoEnd(take) {
    const b = canvas.getBoundingClientRect();
    const loop = lasso.map(([x, y]) => toScene(x + b.left, y + b.top));
    lasso = null;
    lassoPath.removeAttribute("d");
    if (!take || loop.length < 3) return false;
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const [x, y] of loop) [x0, y0, x1, y1] = [Math.min(x0, x), Math.min(y0, y), Math.max(x1, x), Math.max(y1, y)];
    const nodes = new Set(chosen.nodes);
    for (let i = 0; i < header.nodes; i++) {
      const x = at(data.x, i), y = at(data.y, i);
      if (x < x0 || x > x1 || y < y0 || y > y1 || at(data.hidden, i) > 0) continue;
      let inside = false;
      for (let a = 0, z = loop.length - 1; a < loop.length; z = a++) {
        const [xa, ya] = loop[a], [xz, yz] = loop[z];
        if (ya > y !== yz > y && x < ((xz - xa) * (y - ya)) / (yz - ya) + xa) inside = !inside;
      }
      if (inside) nodes.add(i);
    }
    choose(nodes, chosen.edges);
    return true;
  }
  function setLasso(on) {
    lassoOn = on;
    lassoButton.setAttribute("aria-pressed", String(on));
    // A touch draws the loop rather than scrolling the page
    canvas.style.cursor = on ? "crosshair" : "";
    canvas.style.touchAction = on ? "none" : "";
  }

  // Flies to node i, close enough to draw it at least 8 pixels across
  let flight = 0;
  function flyTo(i) {
    cancelAnimationFrame(flight);
    const unit = fit(canvas.clientWidth, canvas.clientHeight), size = at(data.r, i) * unit;
    const zoom = size > 0 ? Math.min(ZOOM_MAX, Math.max(cam.zoom, (8 / size) ** 2)) : cam.zoom;
    const from = { ...cam }, to = { cx: at(data.x, i), cy: at(data.y, i), zoom };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      Object.assign(cam, to);
      redraw();
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const u = Math.min(1, (now - t0) / 450), e = u < 0.5 ? 4 * u ** 3 : 1 - (2 - 2 * u) ** 3 / 2;
      cam.cx = from.cx + (to.cx - from.cx) * e;
      cam.cy = from.cy + (to.cy - from.cy) * e;
      cam.zoom = from.zoom * (to.zoom / from.zoom) ** e;
      redraw();
      if (u < 1) flight = requestAnimationFrame(step);
    };
    flight = requestAnimationFrame(step);
  }

  // Pointers: one drags, two pinch. A touch scrolls the page unless two fingers are down. The
  // wheel zooms once the plot has been clicked (`active`), until the pointer leaves it. A press
  // that doesn't move is a click, or a tap
  const down = new Map();
  let pinch = null, active = false;
  let tap = null, press = null;
  canvas.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    cancelAnimationFrame(flight);
    down.set(e.pointerId, [e.clientX, e.clientY]);
    tap = down.size === 1 ? [e.clientX, e.clientY] : null;
    press = down.size === 1 ? { x: e.clientX, y: e.clientY, add: e.shiftKey || e.metaKey || e.ctrlKey, moved: false } : null;
    if (lasso) lassoEnd(false);
    if (sel && down.size === 1 && (e.shiftKey || lassoOn)) {
      const b = canvas.getBoundingClientRect();
      lasso = [[e.clientX - b.left, e.clientY - b.top]];
    }
    hoverAt();
    if (e.pointerType !== "touch" || down.size === 2 || lasso) canvas.setPointerCapture(e.pointerId);
    if (e.pointerType !== "touch" && !lasso) canvas.className = "ec-grabbing";
    active = true;
    pinch = null;
  });
  const up = (e) => {
    if (!down.delete(e.pointerId)) return;
    pinch = null;
    canvas.className = "ec-grab";
    const click = e.type === "pointerup" && press && !press.moved;
    if (lasso) {
      if (!lassoEnd(e.type === "pointerup") && click) clickAt(e.clientX, e.clientY, press.add);
    } else if (click) {
      clickAt(e.clientX, e.clientY, press.add);
    }
    press = null;
    // A mouse hovers again where it was let go; a touch that didn't move is a tap
    if (e.type === "pointerup" && (e.pointerType !== "touch" || (tap && Math.hypot(e.clientX - tap[0], e.clientY - tap[1]) < 10))) {
      hoverAt(e.clientX, e.clientY);
    }
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch") hoverAt(); });
  canvas.addEventListener("pointermove", (e) => {
    const last = down.get(e.pointerId);
    if (!last) {
      if (e.pointerType !== "touch") hoverAt(e.clientX, e.clientY);
      return;
    }
    if (tap && Math.hypot(e.clientX - tap[0], e.clientY - tap[1]) >= 10) tap = null;
    if (press && Math.hypot(e.clientX - press.x, e.clientY - press.y) >= (e.pointerType === "touch" ? 10 : 5)) press.moved = true;
    if (lasso) {
      lassoTo(e.clientX, e.clientY);
      return;
    }
    if (down.size === 1 && e.pointerType !== "touch") {
      panBy(e.clientX - last[0], e.clientY - last[1]);
    } else if (down.size === 2) {
      down.set(e.pointerId, [e.clientX, e.clientY]);
      const [[x1, y1], [x2, y2]] = [...down.values()];
      const now = { d: Math.hypot(x2 - x1, y2 - y1), x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
      if (pinch) {
        panBy(now.x - pinch.x, now.y - pinch.y);
        if (pinch.d > 0) zoomBy(now.d / pinch.d, now.x, now.y);
      }
      pinch = now;
      return;
    }
    down.set(e.pointerId, [e.clientX, e.clientY]);
  });
  canvas.addEventListener("dblclick", (e) => zoomBy(e.shiftKey ? 0.5 : 2, e.clientX, e.clientY));

  // The wheel zooms once the plot has been clicked, or with Ctrl or Command (as a trackpad's pinch
  // sends it), and otherwise scrolls the page, with a hint
  let hintTimer = 0;
  const hint = document.createElement("div");
  hint.className = "ec-hint";
  hint.textContent = `Click the plot to zoom with the wheel, or hold ${/Mac|iPhone|iPad/.test(navigator.platform) ? "\u2318" : "Ctrl"}`;
  root.append(hint);
  root.addEventListener("pointerleave", () => { active = false; });
  canvas.addEventListener("wheel", (e) => {
    if (!active && !isFull() && !e.ctrlKey && !e.metaKey) {
      hint.classList.add("ec-on");
      clearTimeout(hintTimer);
      hintTimer = setTimeout(() => hint.classList.remove("ec-on"), 1500);
      return;
    }
    e.preventDefault();
    cancelAnimationFrame(flight);
    const step = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    zoomBy(Math.exp(-e.deltaY * step * (e.ctrlKey ? 0.01 : 0.002)), e.clientX, e.clientY);
  }, { passive: false });

  canvas.addEventListener("keydown", (e) => {
    // Esc leaves lasso mode, or clears the selection, unless it's closing the panel
    if (e.key === "Escape" && sel && !isOpen() && (lassoOn || chosen.nodes.size || chosen.edges.size)) {
      e.preventDefault();
      e.stopPropagation();
      if (lassoOn) setLasso(false);
      else choose([], []);
      return;
    }
    const keys = {
      "+": () => zoomBy(1.5), "=": () => zoomBy(1.5), "-": () => zoomBy(1 / 1.5), "0": reset,
      ArrowLeft: () => panBy(40, 0), ArrowRight: () => panBy(-40, 0),
      ArrowUp: () => panBy(0, 40), ArrowDown: () => panBy(0, -40),
      f: toggleFull, F: toggleFull,
    };
    if (keys[e.key] && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      keys[e.key]();
    }
  });

  function button(path, title, act) {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = icon(path);
    b.title = title;
    b.setAttribute("aria-label", title);
    b.addEventListener("click", act);
    return b;
  }
  const controls = document.createElement("div");
  controls.className = "ec-controls";
  const fullButton = button(ICONS.full, "Full screen (F)", () => toggleFull());
  const lassoButton = button(ICONS.lasso, "Lasso (or Shift and drag)", () => setLasso(!lassoOn));
  lassoButton.setAttribute("aria-pressed", "false");
  const panelButton = button(ICONS.settings, "Settings", () => setOpen(!isOpen()));
  controls.append(
    button(ICONS.in, "Zoom in", () => zoomBy(1.5)),
    button(ICONS.out, "Zoom out", () => zoomBy(1 / 1.5)),
    button(ICONS.fit, "Fit the graph (0)", reset),
    ...(sel ? [lassoButton] : []),
    fullButton,
    panelButton,
  );
  (sel ? lassoButton : fullButton).classList.add("ec-apart");
  root.append(controls);
  const search = sel ? searchPill() : null;
  // "/" goes to the search
  root.addEventListener("keydown", (e) => {
    if (e.key !== "/" || !search || e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    e.preventDefault();
    search.input.focus();
  });

  // The search: a pill by the buttons that widens on focus, for a node by its ID or its label,
  // giving those that are what is typed, then those that start with it, then those that hold it,
  // up to 8 (leaving out those the legend hides). Choosing one flies to it and selects it
  function searchPill() {
    const uid = root.id || `ec-viewer-${Math.random().toString(36).slice(2)}`;
    const box = el("div", "ec-search");
    box.innerHTML = icon(ICONS.search);
    const input = el("input");
    Object.assign(input, { type: "text", placeholder: "Search nodes", spellcheck: false, autocomplete: "off" });
    input.style.fontFamily = options.font ?? "";
    const list = el("ul", "ec-results");
    list.id = `${uid}-results`;
    list.style.fontFamily = options.font ?? "";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Nodes found");
    list.hidden = true;
    for (const [k, v] of [["role", "combobox"], ["aria-label", "Search the nodes by ID or label (/)"],
      ["aria-autocomplete", "list"], ["aria-expanded", "false"], ["aria-controls", list.id]]) input.setAttribute(k, v);
    box.append(input);
    root.append(box, list);
    // The IDs and the labels in lower case, once wanted
    let ids = null, labels = null, found = [], current = -1;
    function find(q) {
      q = q.trim().toLowerCase();
      if (!q) return [];
      const n = header.nodes;
      ids ??= Array.from({ length: n }, (_, i) => idOf(i).toLowerCase());
      if (!labels && labelTexts?.nodes) labels = labelTexts.nodes.map((t) => (t ?? "").toLowerCase());
      const tiers = [[], [], []];
      for (let i = 0; i < n && tiers[0].length < 8; i++) {
        if (at(data.hidden, i) > 0) continue;
        let best = 3;
        for (const text of labels ? [ids[i], labels[i]] : [ids[i]]) {
          if (text === q) best = 0;
          else if (best > 1 && text.startsWith(q)) best = 1;
          else if (best > 2 && text.includes(q)) best = 2;
        }
        if (best < 3 && tiers[best].length < 8) tiers[best].push(i);
      }
      return tiers.flat().slice(0, 8);
    }
    function mark() {
      [...list.children].forEach((li, k) => li.setAttribute("aria-selected", String(k === current)));
      if (current >= 0) input.setAttribute("aria-activedescendant", `${uid}-result${current}`);
      else input.removeAttribute("aria-activedescendant");
    }
    function render() {
      found = find(input.value);
      current = found.length ? 0 : -1;
      list.replaceChildren(...found.map((i, k) => {
        const li = el("li");
        li.id = `${uid}-result${k}`;
        li.setAttribute("role", "option");
        const id = idOf(i), label = labelTexts?.nodes?.[i];
        if (label && label !== id) li.append(label, el("span", "ec-dim", id));
        else li.append(id);
        // Chosen before the input loses the focus
        li.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          choice(k);
        });
        return li;
      }));
      if (!found.length && input.value.trim()) {
        const none = el("li", "ec-none", "No nodes found");
        none.setAttribute("role", "option");
        none.setAttribute("aria-disabled", "true");
        list.append(none);
      }
      list.hidden = !list.childElementCount;
      input.setAttribute("aria-expanded", String(!list.hidden));
      mark();
    }
    function close() {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }
    function choice(k) {
      const i = found[k];
      if (i === undefined) return;
      close();
      canvas.focus({ preventScroll: true });
      flyTo(i);
      choose([i], []);
    }
    input.addEventListener("input", render);
    input.addEventListener("focus", () => { if (input.value.trim()) render(); });
    input.addEventListener("blur", close);
    // Its keys are its own, Esc included
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if ((e.key === "ArrowDown" || e.key === "ArrowUp") && found.length) {
        e.preventDefault();
        current = (current + (e.key === "ArrowDown" ? 1 : -1) + found.length) % found.length;
        mark();
      } else if (e.key === "Enter") {
        e.preventDefault();
        choice(Math.max(current, 0));
      } else if (e.key === "Escape") {
        e.preventDefault();
        if (input.value) {
          input.value = "";
          render();
        } else {
          canvas.focus({ preventScroll: true });
        }
      }
    });
    return { input, find, choose: (i) => { found = [i]; choice(0); } };
  }

  // Full screen: the plot's box, or where the page doesn't allow that (as in some notebooks'
  // frames and VS Code's webviews), an overlay over the whole window. The camera keeps its
  // center, and zoom 1 fits the larger box
  const fullElement = () => document.fullscreenElement ?? document.webkitFullscreenElement;
  const isFull = () => fullElement() === root || root.classList.contains("ec-expanded");
  function syncFull() {
    const full = isFull();
    fullButton.innerHTML = icon(full ? ICONS.exit : ICONS.full);
    fullButton.title = full ? "Leave full screen (F)" : "Full screen (F)";
    fullButton.setAttribute("aria-label", fullButton.title);
  }
  function expand(on) {
    root.classList.toggle("ec-expanded", on);
    document.documentElement.classList.toggle("ec-expanded-page", on);
    syncFull();
  }
  async function toggleFull() {
    if (fullElement() === root) {
      (document.exitFullscreen ?? document.webkitExitFullscreen).call(document);
    } else if (root.classList.contains("ec-expanded")) {
      expand(false);
    } else {
      const request = root.requestFullscreen ?? root.webkitRequestFullscreen;
      let done = false;
      if (request && (document.fullscreenEnabled ?? document.webkitFullscreenEnabled)) {
        try {
          await request.call(root);
          done = true;
        } catch (_) {
          // Refused, so the overlay instead
        }
      }
      if (!done) expand(true);
    }
    canvas.focus({ preventScroll: true });
  }
  for (const name of ["fullscreenchange", "webkitfullscreenchange"]) document.addEventListener(name, syncFull);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && root.classList.contains("ec-expanded")) expand(false);
  });

  // The legend keeps its size, at the top right, and folds into a button where it would take
  // too much of the box
  const legend = root.querySelector(".ec-legend-layer");
  let legendButton = null;
  if (legend) {
    legendButton = button(ICONS.legend, "Legend", () => {
      const open = root.classList.toggle("ec-legend-open");
      legendButton.setAttribute("aria-expanded", String(open));
    });
    legendButton.className = "ec-legend-toggle";
    legendButton.setAttribute("aria-expanded", "false");
    root.append(legendButton);
  }
  function fold() {
    if (!legend) return;
    const svg = legend.querySelector("svg");
    const [lw, lh] = [svg.width.baseVal.value, svg.height.baseVal.value];
    const folded = lw > LEGEND_SHARE * canvas.clientWidth || lh > canvas.clientHeight - 16;
    root.classList.toggle("ec-legend-folded", folded);
    if (!folded) {
      root.classList.remove("ec-legend-open");
      legendButton.setAttribute("aria-expanded", "false");
    }
  }

  const { setOpen, isOpen, panelControls, restyle, syncNote } = controlPanel();

  // The legend's swatches show or hide what they stand for, or with Alt (Option) show it alone of
  // its block, or the whole block again where it was alone
  const keys = legend && sel ? [...legend.querySelectorAll(".ec-key")] : [];
  const place = (g) => Number(g.dataset.key) + 1;
  function toggleKey(key, alone) {
    const g = keys[key - 1];
    if (!g) return;
    if (alone) {
      const block = keys.filter((k) => k.dataset.block === g.dataset.block).map(place);
      const only = !off.has(key) && block.every((k) => k === key || off.has(k));
      for (const k of block) {
        if (only || k === key) off.delete(k);
        else off.add(k);
      }
    } else if (off.has(key)) {
      off.delete(key);
    } else {
      off.add(key);
    }
    for (const k of keys) {
      const on = !off.has(place(k));
      k.setAttribute("aria-checked", String(on));
      k.classList.toggle("ec-off", !on);
    }
    syncNote();
    relook();
  }
  for (const g of keys) {
    g.addEventListener("click", (e) => toggleKey(place(g), e.altKey));
    g.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      toggleKey(place(g), e.altKey);
    });
  }
  // The panel docks where the box is wide, and the drawing narrows to make room
  function layout() {
    root.classList.toggle("ec-docked", root.clientWidth >= DOCK_MIN);
    fold();
    hoverAt();
    redraw();
  }
  layout();
  new ResizeObserver(layout).observe(root);
  // The page's or the system's color scheme can change under the automatic theme, as the theme
  // control can make it
  const recolor = () => { if (applyTheme()) redraw(); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
  const watch = new MutationObserver(recolor);
  for (const e of [document.documentElement, document.body]) watch.observe(e, { attributes: true });

  // The controls: a panel of what `plot()` would draw with other arguments, the tabs in its
  // terms, and "Copy as code", which writes the call. A control redraws what the call would
  // draw, from what the payload's encodings started from (`header.style`): every radius, width
  // and opacity is the start's times a factor, or by the builder's rules where edges are
  // highlighted (see `SceneBuilder.scene()`), and a control back at its start draws the payload's
  // own columns. The frame stays the reader's, though `plot()` fits its frame to the nodes' size
  // and the legend
  function controlPanel() {
    const style = header.style;
    const call = options.call ?? { args: [], params: [] };
    const given = new Map(call.args);
    const n = header.nodes, m = header.edges;
    const startTheme = /^var\(/.test(options.theme ?? "") ? "auto" : options.theme ?? "light";
    const list = [];
    if (style) {
      list.push(
        { name: "node_size", tab: "Nodes", label: "Size", kind: "log", start: style.node_size, low: 0.05, sentinel: 16,
          note: options.moves ? "plot() places the nodes for their size, so the copied code's layout differs a little" : null },
        { name: "edge_opacity", tab: "Edges", label: "Opacity", kind: "linear", start: style.edge_opacity,
          sentinel: given.has("edge_color_by") ? 0.4 : null },
        { name: "edge_width", tab: "Edges", label: "Width", kind: "log", start: style.edge_width, low: 0.5, sentinel: 1.2 },
        { name: "curved_edges", tab: "Edges", label: "Curved", kind: "check", start: style.curved, code: (v) => (v ? "True" : null) },
      );
    }
    list.push({ name: "shade", tab: "Nodes", label: "Shaded", kind: "check", start: Boolean(options.shade), code: (v) => (v ? "True" : null) });
    // The log tone needs the density buffer
    if (r.density && m) {
      const name = (t) => (t === "log" ? "Log" : "Over");
      list.push({ name: "edge_tone", tab: "Edges", more: true, label: "Tone", kind: "choice", start: tones.value,
        choices: [["auto", `Auto (${name(tones.auto)})`], ["over", "Over"], ["log", "Log"]],
        code: (v) => (v === "auto" ? null : JSON.stringify(v)) });
    }
    // The labels: whether they're shown, and the most of them, by steps up to the nodes' count
    const info = header.labels;
    if (info) {
      list.push({ name: "show_labels", tab: "Labels", label: "Labels", kind: "check", start: info.show, code: (v) => (v ? null : "False") });
      const most = given.has("max_labels") ? Number(given.get("max_labels")) : null;
      const steps = [5, 10, 20, 30, 50];
      for (let k = 100; steps[steps.length - 1] < n; k *= 10) steps.push(k, 2 * k, 5 * k);
      while (steps.length > 1 && steps[steps.length - 2] >= n) steps.pop();
      if (most !== null && !steps.includes(most)) steps.push(most);
      steps.sort((a, b) => a - b);
      list.push({ name: "max_labels", tab: "Labels", label: "At most", kind: "steps", start: most, steps: [null, ...steps],
        text: (v) => (v === null ? `Auto (${info.auto === null ? "all" : info.auto})` : String(v)), code: (v) => (v === null ? null : String(v)) });
    }
    list.push({ name: "theme", tab: "View", label: "Theme", kind: "choice", start: startTheme,
      choices: [["auto", "Auto"], ["light", "Light"], ["dark", "Dark"]], code: (v) => (v === "auto" ? null : JSON.stringify(v)) });
    if (legend) list.push({ name: "legend", tab: "View", more: true, label: "Legend", kind: "check", start: true, code: (v) => (v ? null : "False") });
    for (const c of list) c.value = c.start;
    const byName = Object.fromEntries(list.map((c) => [c.name, c]));

    // Redraws with the controls' values, the selection and the legend's filters (`styled`)
    const times = (col, count, k) => (typeof col === "number" ? col * k : Float32Array.from(col, (v) => v * k));
    const each = (count, f) => Float32Array.from({ length: count }, (_, j) => f(j));
    function restyle() {
      if (!style) return;
      const v = (name) => byName[name].value;
      const changed = [];
      const set = (name, col) => {
        if (data[name] === col) return;
        data[name] = col;
        changed.push(name);
      };
      const k = v("edge_width") / style.edge_width, o = v("edge_opacity");
      // The edges' levels, the selection's or else the payload's own, and their widths before
      const { edgeLevel, edgeHidden } = styled;
      const level = edgeLevel ?? data.level, base = data.base_width ?? orig.width;
      // The labels' font follows the node size, as `plot()`'s does
      if (info) words.state.size = v("node_size") === style.node_size ? info.font_size : Math.max(8, Math.min(13, 0.7 * v("node_size")));
      set("r", v("node_size") === style.node_size ? orig.r : times(orig.r, n, v("node_size") / style.node_size));
      for (const name of ["fill", "border", "border_width", "border_opacity", "order", "label_color", "label_order", "color", "hidden"]) {
        set(name, styled[name]);
      }
      if (level === null) {
        set("width", k === 1 ? orig.width : times(orig.width, m, k));
      } else {
        // Muted, soft and accent edges
        set("width", k === 1 && !edgeLevel ? orig.width : each(m, (j) => {
          const w = at(base, j) * k, l = at(level, j);
          return l === 0 ? w : l === 1 ? Math.max(1.3 * w, 1) : Math.max(2 * w, 1.5);
        }));
      }
      // The payload's own where the opacity and the levels are; none for an edge hidden
      const own = o === style.edge_opacity && !edgeLevel;
      const opacity = (j) => {
        if (own) return at(orig.opacity, j);
        if (level === null) return o;
        const l = at(level, j);
        return l === 0 ? 0.5 * o : Math.max(o, l === 1 ? 0.5 : 0.9);
      };
      if (edgeHidden) set("opacity", each(m, (j) => (edgeHidden[j] ? 0 : opacity(j))));
      else set("opacity", own ? orig.opacity : level === null ? o : each(m, opacity));
      const curve = v("curved_edges");
      set("bow", curve === style.curved ? orig.bow : typeof orig.bow === "number" ? orig.bow + (curve ? 0.2 : -0.2) : each(m, (j) => orig.bow[j] + (curve ? 0.2 : -0.2)));
      if (!changed.length) return;
      r.update(changed);
      // The grids of the picker hold the nodes' sizes and the edges' curves
      if (changed.includes("r") || changed.includes("bow")) pick = picker(data, isCurved(data) ? 16 : 1);
      else pick.refresh();
      redraw();
    }
    function apply(c) {
      if (c.name === "theme") {
        root.classList.remove("ec-light", "ec-dark", "ec-auto");
        if (c.value !== c.start) root.classList.add(`ec-${c.value}`);
        recolor();
      } else if (c.name === "legend") {
        root.classList.toggle("ec-legend-off", !c.value);
      } else if (c.name === "shade") {
        r.setShade(c.value);
        redraw();
      } else if (c.name === "edge_tone") {
        r.setTone(toneOf(c.value));
        redraw();
      } else if (c.name === "show_labels" || c.name === "max_labels") {
        const most = byName.max_labels.value;
        Object.assign(words.state, { show: byName.show_labels.value, limit: most === null ? info.auto : most });
        redraw();
      } else {
        restyle();
      }
    }

    // The panel: tabs of fields, and a footer
    const uid = root.id || `ec-viewer-${Math.random().toString(36).slice(2)}`;
    const panel = el("div", "ec-panel");
    panel.id = `${uid}-panel`;
    panel.setAttribute("role", "region");
    panel.setAttribute("aria-label", "Plot controls");
    panel.style.fontFamily = options.font ?? "system-ui, sans-serif";
    panelButton.setAttribute("aria-expanded", "false");
    panelButton.setAttribute("aria-controls", panel.id);
    const head = el("div", "ec-panel-head");
    const tablist = el("div", "ec-tabs");
    tablist.setAttribute("role", "tablist");
    tablist.setAttribute("aria-label", "Controls");
    const closer = button(ICONS.close, "Close the controls", () => setOpen(false));
    closer.classList.add("ec-close");
    head.append(tablist, closer);
    panel.append(head);
    // A tab shows once a control is in it
    const names = ["Nodes", "Edges", "Labels", "Filter", "Layout", "View"].filter((t) => list.some((c) => c.tab === t));
    const tabs = [], pages = [];
    let current = 0;
    names.forEach((name, i) => {
      const tab = el("button", "", name);
      tab.type = "button";
      tab.id = `${uid}-tab${i}`;
      tab.setAttribute("role", "tab");
      const page = el("div", "ec-tabpanel");
      page.id = `${uid}-page${i}`;
      page.setAttribute("role", "tabpanel");
      page.setAttribute("aria-labelledby", tab.id);
      page.tabIndex = 0;
      tab.setAttribute("aria-controls", page.id);
      tab.addEventListener("click", () => select(i));
      tablist.append(tab);
      panel.append(page);
      tabs.push(tab);
      pages.push(page);
      // Its common choices, then the rest
      const more = el("details", "ec-more");
      more.append(el("summary", "", "More"));
      for (const c of list.filter((c) => c.tab === name)) (c.more ? more : page).append(field(c, `${uid}-${c.name}`));
      if (more.childElementCount > 1) page.append(more);
    });
    function select(i, focus) {
      current = i;
      tabs.forEach((tab, j) => {
        tab.setAttribute("aria-selected", String(i === j));
        tab.tabIndex = i === j ? 0 : -1;
        pages[j].hidden = i !== j;
      });
      if (focus) tabs[i].focus();
    }
    select(0);
    tablist.addEventListener("keydown", (e) => {
      const i = tabs.indexOf(document.activeElement);
      const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (i < 0 || to === undefined) return;
      e.preventDefault();
      select((to + tabs.length) % tabs.length, true);
    });

    // A field: its name, its value, a reset button while it's changed, and its input
    function field(c, id) {
      const box = el("div", "ec-field");
      const top = el("div", "ec-field-head");
      const name = el(c.kind === "choice" ? "span" : "label", "ec-name", c.label);
      const reset = button(ICONS.reset, `Reset ${c.label}`, () => setValue(c, c.start));
      reset.classList.add("ec-reset");
      reset.hidden = true;
      box.append(top);
      let sync;
      if (c.kind === "check") {
        const input = el("input");
        input.type = "checkbox";
        input.id = id;
        name.htmlFor = id;
        input.addEventListener("change", () => setValue(c, input.checked));
        top.append(input, name, reset);
        sync = () => { input.checked = c.value; };
      } else if (c.kind === "choice") {
        name.id = `${id}-name`;
        const group = el("div", "ec-choices");
        group.setAttribute("role", "radiogroup");
        group.setAttribute("aria-labelledby", name.id);
        const inputs = c.choices.map(([value, text]) => {
          const label = el("label");
          const input = el("input");
          input.type = "radio";
          input.name = id;
          input.value = value;
          input.addEventListener("change", () => { if (input.checked) setValue(c, value); });
          label.append(input, text);
          group.append(label);
          return input;
        });
        top.append(name, reset);
        box.append(group);
        sync = () => { for (const input of inputs) input.checked = input.value === c.value; };
      } else {
        // A slider, in quarter steps of a doubling for sizes and widths, from a quarter to eight
        // times the start, or in hundredths for opacity
        const input = el("input");
        input.type = "range";
        input.id = id;
        name.htmlFor = id;
        const out = el("output");
        out.htmlFor = id;
        let toValue, toInput;
        if (c.kind === "log") {
          const lo = Math.max(-8, Math.ceil(4 * Math.log2(c.low / c.start) - 1e-9));
          Object.assign(input, { min: String(Math.min(lo, 0)), max: "12", step: "1" });
          toValue = (k) => (k === 0 ? c.start : Math.max(c.low, Math.round(c.start * 2 ** (k / 4) * 100) / 100));
          toInput = (v) => String(Math.round(4 * Math.log2(v / c.start)));
        } else if (c.kind === "steps") {
          Object.assign(input, { min: "0", max: String(c.steps.length - 1), step: "1" });
          toValue = (i) => c.steps[i];
          toInput = (v) => String(c.steps.indexOf(v));
        } else {
          Object.assign(input, { min: "0.01", max: "1", step: "0.01" });
          toValue = (x) => (Math.abs(x - c.start) < 0.005 ? c.start : Math.round(x * 100) / 100);
          toInput = (v) => String(v);
        }
        input.addEventListener("input", () => setValue(c, toValue(Number(input.value)), true));
        top.append(name, out, reset);
        box.append(input);
        const say = c.text ?? show;
        sync = (typing) => {
          if (!typing) input.value = toInput(c.value);
          out.textContent = say(c.value);
          input.setAttribute("aria-valuetext", say(c.value));
        };
      }
      const note = c.note ? el("p", "ec-note", c.note) : null;
      if (note) box.append(note);
      c.sync = (typing) => {
        sync(typing);
        const changed = c.value !== c.start;
        box.classList.toggle("ec-changed", changed);
        reset.hidden = !changed;
        if (note) note.hidden = !changed;
        const tab = tabs[names.indexOf(c.tab)];
        tab.classList.toggle("ec-changed", list.some((d) => d.tab === c.tab && d.value !== d.start));
      };
      return box;
    }
    // A value that `plot()` would take as "automatic" moves off it
    function setValue(c, value, typing) {
      if (value === c.sentinel && value !== c.start) value = Math.round((value + 0.01) * 100) / 100;
      const was = c.value;
      c.value = value;
      c.sync(typing);
      if (value !== was) apply(c);
    }
    for (const c of list) c.sync();

    // "Copy as code": the call, with the changed controls' arguments and the selection, in the
    // parameters' order, on one line where it fits in 88 characters. The legend's filters are the
    // view's, which no argument of `plot()` has
    function code() {
      const args = new Map(given);
      for (const c of list) {
        if (c.value === c.start) continue;
        const source = c.code ? c.code(c.value) : pyNumber(c.value);
        if (source === null) args.delete(c.name);
        else args.set(c.name, source);
      }
      if (chosen.nodes.size || chosen.edges.size) {
        args.delete("highlight");
        args.delete("highlight_edges");
        const byId = (a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);
        const ids = [...chosen.nodes].map(idOf).sort(byId);
        if (ids.length) args.set("highlight", ids.length === 1 ? ids[0] : `{${ids.join(", ")}}`);
        const ends = new Map();
        for (const j of chosen.edges) {
          let [a, b] = [idOf(at(data.src, j)), idOf(at(data.tgt, j))];
          // An undirected edge's ends in the order of their IDs
          if (!header.directed && byId(a, b) > 0) [a, b] = [b, a];
          ends.set(`${a} ${b}`, [a, b]);
        }
        const pairs = [...ends.values()].sort((p, q) => byId(p[0], q[0]) || byId(p[1], q[1]));
        if (pairs.length) args.set("highlight_edges", `[${pairs.map(([a, b]) => `(${a}, ${b})`).join(", ")}]`);
      }
      const rank = (name) => { const i = call.params.indexOf(name); return i < 0 ? Infinity : i; };
      const parts = [...args].sort((a, b) => rank(a[0]) - rank(b[0])).map(([k, v]) => `${k}=${v}`);
      const line = `g.plot(${parts.join(", ")})`;
      return line.length <= 88 ? line : `g.plot(\n${parts.map((p) => `    ${p},\n`).join("")})`;
    }
    const foot = el("div", "ec-panel-foot");
    const copy = el("button", "ec-copy", "Copy as code");
    copy.type = "button";
    const status = el("span", "ec-status");
    status.setAttribute("role", "status");
    const text = el("textarea", "ec-code");
    text.readOnly = true;
    text.hidden = true;
    text.setAttribute("aria-label", "The plot() call");
    let said = 0;
    copy.addEventListener("click", async () => {
      const source = code();
      let done = false;
      try {
        await navigator.clipboard.writeText(source);
        done = true;
      } catch (_) {
        // Not allowed here (as in some notebooks' frames), so in a box to copy from
      }
      text.value = source;
      text.hidden = done;
      if (!done) {
        text.focus();
        text.select();
        try { done = document.execCommand("copy"); } catch (_) { done = false; }
      }
      status.textContent = done ? "Copied" : "Copy the code from the box";
      clearTimeout(said);
      said = setTimeout(() => { status.textContent = ""; }, 2500);
    });
    const filtered = el("p", "ec-note", "The legend's filters aren't in the code");
    filtered.hidden = true;
    const syncNote = () => { filtered.hidden = off.size === 0; };
    foot.append(copy, status, filtered, text);
    panel.append(foot);
    root.append(panel);

    // Opened by its button, closed by its own, by Esc, or in a small box by a click outside it
    let shownOpen = false;
    function setOpen(on) {
      if (shownOpen === on) return;
      shownOpen = on;
      root.classList.toggle("ec-panel-open", on);
      panelButton.setAttribute("aria-expanded", String(on));
      if (on) tabs[current].focus({ preventScroll: true });
      else if (panel.contains(document.activeElement)) panelButton.focus({ preventScroll: true });
      if (!on) text.hidden = true;
      layout();
    }
    root.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || !shownOpen) return;
      e.stopPropagation();
      setOpen(false);
      panelButton.focus({ preventScroll: true });
    });
    document.addEventListener("pointerdown", (e) => {
      if (shownOpen && !root.classList.contains("ec-docked") && !panel.contains(e.target) && !panelButton.contains(e.target)) setOpen(false);
    }, true);
    const panelControls = {
      values: () => Object.fromEntries(list.map((c) => [c.name, c.value])),
      set(name, value) { if (byName[name]) setValue(byName[name], value); },
      code,
    };
    return { setOpen, isOpen: () => shownOpen, panelControls, restyle, syncNote };
  }

  return {
    camera: cam, redraw, reset, renderer: r, data, labels: words, controls: { ...panelControls, open: setOpen },
    selection: { nodes: () => [...chosen.nodes], edges: () => [...chosen.edges], choose, clickAt, setLasso },
    legend: { off: () => [...off], toggle: toggleKey },
    search,
    flyTo,
  };
}

// A value as it reads in Python
const show = (v) => String(Math.round(v * 100) / 100);
const pyNumber = (v) => (/[.e]/.test(String(v)) ? String(v) : `${v}.0`);

// The edges in a random (but fixed) order, so that the first k of them are a fair share. Gives
// the order, each edge's place in the scene's drawing order
function shuffleEdges(data) {
  const m = data.header.edges;
  const perm = new Uint32Array(m);
  for (let i = 0; i < m; i++) perm[i] = i;
  let s = 0x9e3779b9;
  for (let i = m - 1; i > 0; i--) {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    const j = (s >>> 0) % (i + 1);
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (const name of ["src", "tgt", "color", "width", "opacity", "bow", "loop_size", "level", "base_width", "base_color", "edge_key"]) {
    const col = data[name];
    if (col === null || typeof col === "number") continue;
    const out = new col.constructor(m);
    for (let i = 0; i < m; i++) out[i] = col[perm[i]];
    data[name] = out;
  }
  return perm;
}

// A grid of `cols` by `rows` cells over `frame`, of the `count` items that `visit(i, add)` puts
// in the cells that `add(x0, y0, x1, y1)` boxes cover (points outside the frame are in its edge
// cells). Gives `each(x0, y0, x1, y1, f)`, which calls `f` on the items of the cells a box
// covers, once for each time an item was added to one
function grid(frame, cols, rows, count, visit) {
  const [fx, fy, fw, fh] = frame;
  const cw = fw / cols || 1, ch = fh / rows || 1;
  const ix = (x) => Math.min(cols - 1, Math.max(0, Math.floor((x - fx) / cw)));
  const iy = (y) => Math.min(rows - 1, Math.max(0, Math.floor((y - fy) / ch)));
  const cells = cols * rows;
  // The cells' lists follow one another in `items`, cell k's from `start[k]`: counted, then filled
  const start = new Uint32Array(cells + 1);
  let items = null, fill = null, item = 0;
  const add = (x0, y0, x1, y1) => {
    const a = ix(x0), b = ix(x1), c = iy(y0), d = iy(y1);
    for (let y = c; y <= d; y++) {
      for (let x = a; x <= b; x++) {
        const k = y * cols + x;
        if (items) items[fill[k]++] = item;
        else start[k + 1]++;
      }
    }
  };
  for (item = 0; item < count; item++) visit(item, add);
  for (let k = 0; k < cells; k++) start[k + 1] += start[k];
  fill = start.slice(0, cells);
  items = new Uint32Array(start[cells]);
  for (item = 0; item < count; item++) visit(item, add);
  return (x0, y0, x1, y1, f) => {
    const a = ix(x0), b = ix(x1), c = iy(y0), d = iy(y1);
    for (let y = c; y <= d; y++) {
      for (let x = a; x <= b; x++) {
        const k = y * cols + x;
        for (let p = start[k]; p < start[k + 1]; p++) f(items[p]);
      }
    }
  };
}

// What is drawn at a point. `pick(x, y, v)` gives the node or the edge drawn nearest the scene
// point (x, y), as `{ node: i }` or `{ edge: j }` (j in the drawing order), if one is within
// `HIT` pixels of it, or null. A node drawn over the point comes first, the one on top if
// several are. `v` holds the pixels per unit of the scene (`px`), of node radius (`radius`) and
// of edge width (`unit`), and the thinnest line drawn (`thin`, the smallest node is half as
// wide), all in CSS pixels.
//
// It looks in grids over the scene, built when first wanted: of the nodes' centers, and of the
// edges' curves, in straight pieces about as long as a cell is wide. Curves are taken in
// `segments` pieces, as the shader draws them, and self-loops, which grow with the nodes as the
// view zooms, in `LOOP_PIECES`, each in the box of the largest it's drawn. Nodes hidden and edges
// drawn without opacity aren't picked. `pick.refresh()` takes in new colors, widths, opacities
// and drawing order of the nodes, which don't move anything in the grids
const LOOP_PIECES = 16;
function picker(data, segments) {
  const { header } = data;
  const n = header.nodes, m = header.edges;
  const [, , vw, vh] = header.viewbox;
  const frame = header.viewbox;
  const pts = new Float64Array(2 * (Math.max(segments, LOOP_PIECES) + 1));
  // Edge j's curve, into `pts`, with its node's radius `f` times the scene's for a self-loop; the
  // pieces it's in, 0 for one that isn't drawn
  function curve(j, f) {
    const s = at(data.src, j), t = at(data.tgt, j);
    const sx = at(data.x, s), sy = at(data.y, s), tx = at(data.x, t), ty = at(data.y, t);
    if (s === t) {
      const q = loopPoints(sx, sy, at(data.r, s) * f, at(data.loop_size, j));
      const half = LOOP_PIECES / 2;
      for (let h = 0; h < 2; h++) {
        for (let k = 0; k <= half; k++) {
          const u = k / half, a = (1 - u) ** 3, b = 3 * u * (1 - u) ** 2, c = 3 * u * u * (1 - u), d = u ** 3;
          for (let e = 0; e < 2; e++) {
            const o = 6 * h + e;
            pts[2 * (h * half + k) + e] = a * q[o] + b * q[o + 2] + c * q[o + 4] + d * q[o + 6];
          }
        }
      }
      return LOOP_PIECES;
    }
    const bow = at(data.bow, j);
    const cx = (sx + tx) / 2 - (ty - sy) * bow, cy = (sy + ty) / 2 + (tx - sx) * bow;
    for (let k = 0; k <= segments; k++) {
      const u = k / segments, a = (1 - u) * (1 - u), b = 2 * u * (1 - u), c = u * u;
      pts[2 * k] = a * sx + b * cx + c * tx;
      pts[2 * k + 1] = a * sy + b * cy + c * ty;
    }
    return segments;
  }
  const side = (size) => [
    Math.max(1, Math.min(1024, Math.ceil(vw / size))),
    Math.max(1, Math.min(1024, Math.ceil(vh / size))),
  ];

  let nodes = null, rank = null, rmax = 0;
  function ranks() {
    rank = new Uint32Array(n);
    for (let j = 0; j < n; j++) rank[data.order ? data.order[j] : j] = j;
  }
  function nodeGrid() {
    ranks();
    for (let i = 0; i < n; i++) rmax = Math.max(rmax, at(data.r, i));
    // A few nodes to a cell
    const [cols, rows] = side(1.5 * Math.sqrt((vw * vh) / Math.max(n, 1)));
    nodes = grid(frame, cols, rows, n, (i, add) => {
      const x = at(data.x, i), y = at(data.y, i);
      add(x, y, x, y);
    });
  }

  let edges = null, seen = null, stamp = 0, wmax = 0;
  function widest() {
    wmax = 0;
    for (let j = 0; j < m; j++) wmax = Math.max(wmax, at(data.width, j));
  }
  function edgeGrid() {
    // Cells about as wide as half the average edge is long, so that an edge is in a few
    let length = 0;
    widest();
    for (let j = 0; j < m; j++) {
      if (at(data.src, j) === at(data.tgt, j)) continue;
      const count = curve(j);
      for (let k = 0; k < count; k++) {
        length += Math.hypot(pts[2 * k + 2] - pts[2 * k], pts[2 * k + 3] - pts[2 * k + 1]);
      }
    }
    const size = Math.max(length / (2 * Math.max(m, 1)), Math.sqrt(vw * vh) / 1024);
    const [cols, rows] = side(size);
    const cell = Math.min(vw / cols, vh / rows) || 1;
    // The smallest zoom draws the nodes the largest, twice the scene's radii
    const largest = 1 / Math.sqrt(ZOOM_MIN);
    edges = grid(frame, cols, rows, m, (j, add) => {
      const s = at(data.src, j);
      if (s === at(data.tgt, j)) {
        if (!curve(j, largest)) return;
        const xs = pts.filter((_, k) => k % 2 === 0), ys = pts.filter((_, k) => k % 2 === 1);
        add(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(at(data.y, s), ...ys));
        return;
      }
      const count = curve(j);
      for (let k = 0; k < count; k++) {
        const x0 = pts[2 * k], y0 = pts[2 * k + 1], x1 = pts[2 * k + 2], y1 = pts[2 * k + 3];
        const parts = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / cell));
        for (let p = 0; p < parts; p++) {
          const a = p / parts, b = (p + 1) / parts;
          const ax = x0 + (x1 - x0) * a, ay = y0 + (y1 - y0) * a;
          const bx = x0 + (x1 - x0) * b, by = y0 + (y1 - y0) * b;
          add(Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by));
        }
      }
    });
    seen = new Uint32Array(m);
  }

  // The distance from (x, y) to the curve in `pts`, in `count` pieces
  function distance(x, y, count) {
    let best = Infinity;
    for (let k = 0; k < count; k++) {
      const x0 = pts[2 * k], y0 = pts[2 * k + 1];
      const dx = pts[2 * k + 2] - x0, dy = pts[2 * k + 3] - y0;
      const len = dx * dx + dy * dy;
      const u = len > 0 ? Math.min(1, Math.max(0, ((x - x0) * dx + (y - y0) * dy) / len)) : 0;
      best = Math.min(best, Math.hypot(x - x0 - u * dx, y - y0 - u * dy));
    }
    return best;
  }

  function pick(x, y, v) {
    if (!nodes) nodeGrid();
    // How far outside its disc the point is, in pixels, for the nodes near it
    let node = -1, gap = Infinity, over = -1;
    let reach = (Math.max(rmax * v.radius, v.thin / 2) + HIT) / v.px;
    nodes(x - reach, y - reach, x + reach, y + reach, (i) => {
      if (at(data.hidden, i) > 0) return;
      const d = Math.hypot(at(data.x, i) - x, at(data.y, i) - y) * v.px - Math.max(at(data.r, i) * v.radius, v.thin / 2);
      if (d <= 0 && (over < 0 || rank[i] > rank[over])) over = i;
      if (d < gap) [node, gap] = [i, d];
    });
    if (over >= 0) return { node: over };
    if (!edges) edgeGrid();
    // And outside its stroke, for the edges, each looked at once
    let edge = -1, egap = Infinity;
    stamp = stamp === 0xffffffff ? 1 : stamp + 1;
    if (stamp === 1) seen.fill(0);
    reach = (Math.max(wmax * v.unit, v.thin) / 2 + HIT) / v.px;
    edges(x - reach, y - reach, x + reach, y + reach, (j) => {
      if (seen[j] === stamp) return;
      seen[j] = stamp;
      if (at(data.opacity, j) <= 0) return;
      const count = curve(j, v.radius / v.px);
      if (!count) return;
      const d = distance(x, y, count) * v.px - Math.max(at(data.width, j) * v.unit, v.thin) / 2;
      if (d < egap) [edge, egap] = [j, d];
    });
    if (Math.min(gap, egap) > HIT) return null;
    return gap <= egap ? { node } : { edge };
  }
  pick.refresh = () => {
    if (nodes) ranks();
    if (edges) widest();
  };
  return pick;
}

// What the plot's gzipped script `ec-<name>` holds: the tooltips' tables (`tips`) or the labels'
// texts (`labels`), each as `{ nodes, edges }`, or null where there is none or the browser can't
// read it
async function readPacked(root, name) {
  const script = root.querySelector(`script.ec-${name}`);
  if (!script || typeof DecompressionStream === "undefined") return null;
  try {
    const text = script.textContent.trim();
    const bytes = Uint8Array.fromBase64
      ? Uint8Array.fromBase64(text)
      : Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  } catch (_) {
    return null;
  }
}

// Row i of a tooltip table (see `Tips::table()` in `src/scene/tips.rs`), as [key, text] pairs
function tipRow(table, i) {
  const out = [];
  for (let k = 0; k < table.k.length; k++) {
    const v = table.c[k][i];
    if (v === undefined) continue;
    const text = Array.isArray(v) ? v.join(", ") : v !== null && typeof v === "object" ? JSON.stringify(v) : String(v);
    out.push([String(table.k[k]), text]);
  }
  return out;
}
