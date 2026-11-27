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
// added over the frames that follow once it stops.

const TEX_W = 2048; // texels per row; the shaders shift by 11
const BUDGET = 150000; // the edges that a frame of a moving view draws
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 1000;

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
    "src", "tgt", "color", "width", "opacity", "bow"];
  return { header, ...Object.fromEntries(names.map((name) => [name, column(name)])) };
}

const LINK_VS = `#version 300 es
precision highp float;
precision highp int;
in vec2 a_corner;            // x: t along the edge, y: side, -1 or 1
in uint a_src, a_tgt, a_color;
in float a_width, a_opacity, a_bow;
uniform highp sampler2D u_nodes, u_colors;
uniform vec2 u_scale, u_offset, u_viewport;
uniform float u_unit;        // pixels per unit of the scene's width
uniform float u_fringe;
out vec3 v_color;
out float v_dist, v_line, v_gain;
vec4 texel(highp sampler2D t, uint i) { return texelFetch(t, ivec2(int(i & 2047u), int(i >> 11)), 0); }
void main() {
  vec2 s = texel(u_nodes, a_src).xy, t = texel(u_nodes, a_tgt).xy;
  // Self-loops aren't drawn yet
  if (a_src == a_tgt) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  // The control point, out from the middle by the bow's share of the length, as the SVG has it
  vec2 d = t - s;
  vec2 c = 0.5 * (s + t) + vec2(-d.y, d.x) * a_bow;
  vec2 half_vp = 0.5 * u_viewport;
  vec2 a = (s * u_scale + u_offset) * half_vp, b = (t * u_scale + u_offset) * half_vp;
  vec2 m = (c * u_scale + u_offset) * half_vp;
  // A stroke thinner than a pixel is drawn a pixel wide, as faint as it is thin
  float line = a_width * u_unit;
  vec4 col = texel(u_colors, a_color);
  v_gain = col.a * a_opacity * min(line, 1.0);
  line = max(line, 1.0);
  float w = line + u_fringe;
  // Edges whose curve's hull is off screen collapse to nothing
  vec2 lo = min(min(a, b), m) - w, hi = max(max(a, b), m) + w;
  if (v_gain <= 0.0 || any(greaterThan(lo, half_vp)) || any(lessThan(hi, -half_vp))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float k = a_corner.x;
  vec2 p = mix(mix(a, m, k), mix(m, b, k), k);
  vec2 tan = (1.0 - k) * (m - a) + k * (b - m);
  float tl = length(tan);
  vec2 side = tl > 0.0 ? vec2(-tan.y, tan.x) / tl : vec2(0.0, 1.0);
  v_dist = a_corner.y * 0.5 * w;
  v_line = line;
  v_color = col.rgb;
  gl_Position = vec4((p + side * v_dist) / half_vp, 0.0, 1.0);
}`;

const LINK_FS = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_dist, v_line, v_gain;
uniform bool u_density;
out vec4 o;
void main() {
  float a = v_gain * clamp(0.5 * v_line + 0.5 - abs(v_dist), 0.0, 1.0);
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
    // Shaded as a sphere lit from the upper left (y runs down), more so the larger it is drawn
    vec2 q = v_uv / v_r;
    vec3 n = vec3(q, sqrt(max(0.0, 1.0 - dot(q, q))));
    vec3 l = normalize(vec3(-0.45, 0.55, 0.7));
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(reflect(-l, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
    vec3 lit = v_fill * (0.55 + 0.55 * diff) + 0.25 * spec;
    fill = mix(v_fill, min(lit, 1.0), smoothstep(1.5, 4.0, v_r));
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

const TONE_FS = `#version 300 es
precision highp float;
uniform sampler2D u_density;
uniform vec2 u_size;
uniform float u_gain;
out vec4 o;
void main() {
  vec4 d = u_gain * texture(u_density, gl_FragCoord.xy / u_size);
  if (d.a <= 0.0) discard;
  float a = 1.0 - exp(-d.a);
  o = vec4(d.rgb / d.a * a, a);
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

// The drawing: `draw(view, count)` draws a view with the first `count` edges, `refine(chunk)`
// adds the next ones, and `setColors(colors, bg)` changes the color table and the background.
// With `shade`, nodes are shaded as spheres; the SVG and PNG draw them flat
function renderer(canvas, data, shade) {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const { header } = data;
  const n = header.nodes, m = header.edges;
  const density = gl.getExtension("EXT_color_buffer_float") !== null;

  const nodes = table(gl, n, (t, j, i) => t.set([at(data.x, i), at(data.y, i), at(data.r, i), at(data.fill, i)], j));
  const styles = table(gl, n, (t, j, i) => t.set([at(data.border, i), at(data.border_width, i), at(data.border_opacity, i), 0], j));
  let colors = null;

  // Edges: a strip per instance, along a quadratic curve in pieces. A long thin triangle costs
  // as much as the box around it, so a straight edge is split too
  const link = compile(gl, LINK_VS, LINK_FS);
  const curved = typeof data.bow !== "number" || data.bow !== 0;
  const segments = curved ? 16 : 8;
  const strip = new Float32Array((segments + 1) * 4);
  for (let s = 0; s <= segments; s++) strip.set([s / segments, -1, s / segments, 1], 4 * s);
  const linkVao = gl.createVertexArray();
  gl.bindVertexArray(linkVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, strip, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(link.a("a_corner"));
  gl.vertexAttribPointer(link.a("a_corner"), 2, gl.FLOAT, false, 0, 0);
  // A column per instance, or one value for all
  const perEdge = [], constant = [];
  for (const [name, int] of [["src", 1], ["tgt", 1], ["color", 1], ["width", 0], ["opacity", 0], ["bow", 0]]) {
    const loc = link.a("a_" + name), col = data[name];
    if (loc < 0) continue;
    if (typeof col === "number") {
      constant.push({ loc, value: col, int });
      continue;
    }
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribDivisor(loc, 1);
    perEdge.push({ loc, buf, int });
  }
  // WebGL 2 has no base instance, so a range of edges starts where the columns are pointed.
  // The values for all are the context's, not the vertex array's, so they are set each time
  function pointEdges(from) {
    for (const { loc, buf, int } of perEdge) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      if (int) gl.vertexAttribIPointer(loc, 1, gl.UNSIGNED_INT, 0, from * 4);
      else gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, from * 4);
    }
    for (const { loc, value, int } of constant) {
      if (int) gl.vertexAttribI4ui(loc, value, 0, 0, 0);
      else gl.vertexAttrib1f(loc, value);
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
  const order = data.order ?? Uint32Array.from({ length: n }, (_, i) => i);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, order, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(point.a("a_index"));
  gl.vertexAttribIPointer(point.a("a_index"), 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(point.a("a_index"), 1);
  gl.bindVertexArray(null);

  const tone = compile(gl, TONE_VS, TONE_FS);
  const emptyVao = gl.createVertexArray();
  let fbo = null, fboTex = null, fboSize = [0, 0];
  function target(w, h) {
    if (fboSize[0] === w && fboSize[1] === h) return;
    if (fbo) { gl.deleteFramebuffer(fbo); gl.deleteTexture(fboTex); }
    fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, fboTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fboSize = [w, h];
  }

  let bg = [1, 1, 1];
  let drawn = 0, view = null;

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
    gl.uniform1f(link.u.u_fringe, 1.5);
    gl.uniform1i(link.u.u_density, density ? 1 : 0);
    gl.bindVertexArray(linkVao);
    pointEdges(from);
    if (to > from) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, (segments + 1) * 2, to - from);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    drawn = clear ? to : drawn + (to - from);
  }

  // The density over the background, as if every edge were drawn, then the nodes
  function finish(v) {
    gl.viewport(0, 0, v.w, v.h);
    if (density) {
      gl.clearColor(...bg, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(tone.prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fboTex);
      gl.uniform1i(tone.u.u_density, 0);
      gl.uniform2f(tone.u.u_size, v.w, v.h);
      gl.uniform1f(tone.u.u_gain, m / Math.max(drawn, 1));
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
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
      bg = background;
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

// A computed CSS color, "rgb(r, g, b)" or "rgba(...)", from 0 to 1
function cssRgb(color) {
  const v = (color.match(/[\d.]+/g) ?? [255, 255, 255]).slice(0, 3).map(Number);
  return v.map((c) => c / 255);
}

const STYLE = `
.ec-viewer { position: relative; overflow: hidden; user-select: none; -webkit-user-select: none; }
.ec-viewer canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: pan-x pan-y; outline: none; }
.ec-viewer canvas.ec-grab { cursor: grab; }
.ec-viewer canvas.ec-grabbing { cursor: grabbing; }
.ec-viewer .ec-legend-layer { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.ec-viewer .ec-controls { position: absolute; left: 8px; top: 8px; display: flex; flex-direction: column; gap: 4px; }
.ec-viewer .ec-controls button { width: 26px; height: 26px; padding: 0; border: 0; border-radius: 6px; font: 15px/26px system-ui, sans-serif; color: inherit; background: color-mix(in srgb, currentColor 14%, transparent); cursor: pointer; opacity: 0.75; }
.ec-viewer .ec-controls button:hover { opacity: 1; }
.ec-viewer .ec-hint { position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%); padding: 4px 10px; border-radius: 6px; font: 12px system-ui, sans-serif; color: inherit; background: color-mix(in srgb, currentColor 14%, transparent); pointer-events: none; opacity: 0; transition: opacity 0.3s; white-space: nowrap; }
.ec-viewer .ec-hint.ec-on { opacity: 1; }
.ec-viewer .ec-message { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font: 13px system-ui, sans-serif; }
`;

// The theme, "light" or "dark", or a CSS variable that holds it: the automatic theme's, whose
// value follows the page's color scheme by the same rules as the SVG's colors
function themeOf(root, theme) {
  const name = /^var\((--[\w-]+)\)$/.exec(theme ?? "");
  return name ? getComputedStyle(root).getPropertyValue(name[1]).trim() : theme;
}

// options: theme (see `themeOf()`), and shade, for nodes shaded as spheres. The plot's
// background is `root`'s
export function mount(root, buffer, options = {}) {
  const data = parse(buffer);
  const { header } = data;
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
  canvas.setAttribute("aria-label", `Graph of ${header.nodes} nodes and ${header.edges} edges. Drag to pan; zoom with the buttons, the keys + and -, or the wheel once clicked`);
  root.prepend(canvas);
  // A moved view draws `BUDGET` edges at once, so they go in a random order, to be a fair share
  if (header.edges > BUDGET) shuffleEdges(data);
  const r = renderer(canvas, data, options.shade);
  if (!r) {
    canvas.remove();
    const msg = document.createElement("div");
    msg.className = "ec-message";
    msg.textContent = "This plot needs WebGL 2, which this browser doesn't have.";
    root.append(msg);
    return null;
  }

  // The camera: the scene point at the center, and the zoom. At zoom 1 the scene's frame fits
  // the canvas as the SVG's viewBox fits its box
  const [vx, vy, vw, vh] = header.viewbox;
  const home = { cx: vx + vw / 2, cy: vy + vh / 2, zoom: 1 };
  const cam = { ...home };
  const fit = (w, h) => Math.min(w / vw, h / vh);
  function view() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.min(4096, Math.round(canvas.clientWidth * dpr)));
    const h = Math.max(1, Math.min(4096, Math.round(canvas.clientHeight * dpr)));
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
    const unit = fit(w, h);
    const px = unit * cam.zoom;
    const scale = [px / (w / 2), -px / (h / 2)];
    // Nodes grow with the square root of the zoom, so that zooming in opens up space between
    // them; edges keep their width
    return { w, h, scale, offset: [-cam.cx * scale[0], -cam.cy * scale[1]], unit, radius: unit * Math.sqrt(cam.zoom) };
  }

  // The theme's color table, and the background, which change with the page's color scheme
  let shown = null;
  function applyTheme() {
    const dark = themeOf(root, options.theme) === "dark";
    const bg = getComputedStyle(root).backgroundColor;
    if (shown && shown.dark === dark && shown.bg === bg) return false;
    shown = { dark, bg };
    r.setColors(dark ? header.colors.dark : header.colors.light, cssRgb(bg));
    return true;
  }
  applyTheme();

  // A moved view draws `BUDGET` edges at once, and the rest over the frames that follow
  let pending = false, refining = 0;
  const redraw = () => {
    if (pending) return;
    pending = true;
    cancelAnimationFrame(refining);
    requestAnimationFrame(() => {
      pending = false;
      r.draw(view(), BUDGET);
      const more = () => { if (r.refine(BUDGET)) refining = requestAnimationFrame(more); };
      refining = requestAnimationFrame(more);
    });
  };
  r.draw(view());

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

  // Pointers: one drags, two pinch. A touch scrolls the page unless two fingers are down. The
  // wheel zooms once the plot has been clicked (`active`), until the pointer leaves it
  const down = new Map();
  let pinch = null, active = false;
  canvas.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    down.set(e.pointerId, [e.clientX, e.clientY]);
    if (e.pointerType !== "touch" || down.size === 2) canvas.setPointerCapture(e.pointerId);
    if (e.pointerType !== "touch") canvas.className = "ec-grabbing";
    active = true;
    pinch = null;
  });
  const up = (e) => {
    down.delete(e.pointerId);
    pinch = null;
    canvas.className = "ec-grab";
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointermove", (e) => {
    const last = down.get(e.pointerId);
    if (!last) return;
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
    if (!active && !e.ctrlKey && !e.metaKey) {
      hint.classList.add("ec-on");
      clearTimeout(hintTimer);
      hintTimer = setTimeout(() => hint.classList.remove("ec-on"), 1500);
      return;
    }
    e.preventDefault();
    const step = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    zoomBy(Math.exp(-e.deltaY * step * (e.ctrlKey ? 0.01 : 0.002)), e.clientX, e.clientY);
  }, { passive: false });

  canvas.addEventListener("keydown", (e) => {
    const keys = {
      "+": () => zoomBy(1.5), "=": () => zoomBy(1.5), "-": () => zoomBy(1 / 1.5), "0": reset,
      ArrowLeft: () => panBy(40, 0), ArrowRight: () => panBy(-40, 0),
      ArrowUp: () => panBy(0, 40), ArrowDown: () => panBy(0, -40),
    };
    if (keys[e.key] && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      keys[e.key]();
    }
  });

  const controls = document.createElement("div");
  controls.className = "ec-controls";
  for (const [label, title, act] of [["+", "Zoom in", () => zoomBy(1.5)], ["\u2212", "Zoom out", () => zoomBy(1 / 1.5)], ["\u2922", "Fit the graph", reset]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.setAttribute("aria-label", title);
    b.addEventListener("click", act);
    controls.append(b);
  }
  root.append(controls);

  new ResizeObserver(() => redraw()).observe(root);
  // The page's or the system's color scheme can change under the automatic theme
  if (options.theme !== themeOf(root, options.theme)) {
    const recolor = () => { if (applyTheme()) redraw(); };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);
    const watch = new MutationObserver(recolor);
    for (const el of [document.documentElement, document.body]) watch.observe(el, { attributes: true });
  }
  return { camera: cam, redraw, reset, renderer: r };
}

// The edges in a random (but fixed) order, so that the first k of them are a fair share
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
  for (const name of ["src", "tgt", "color", "width", "opacity", "bow"]) {
    const col = data[name];
    if (typeof col === "number") continue;
    const out = new col.constructor(m);
    for (let i = 0; i < m; i++) out[i] = col[perm[i]];
    data[name] = out;
  }
}
