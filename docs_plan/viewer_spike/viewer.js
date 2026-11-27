// The W4 spike: instanced points and links in WebGL2, from positions computed in Rust.
//
// Node positions, color indexes and radii live in one RGBA32F texture, read by index with
// texelFetch. A link is one instance of a strip: two vertices for a straight link, or
// 2 * (segments + 1) along a quadratic curve, widened to the line width in screen space. A point
// is one instance of a quad, shaded as a sphere in the fragment shader.
//
// By default the links add their density, -ln(1 - alpha) each, and their density-weighted color
// into a half-float buffer, which a full-screen pass tone-maps over the background, as the
// raster backend does: 1 - exp(-density). `density: false` blends each link straight onto the
// canvas instead (additive on dark, "over" on light).
//
// Since the density adds up, a view can be drawn progressively: a first frame with `budget`
// links, scaled up to stand for all of them, then the rest added over the following frames.

const TEX_W = 2048; // nodes per texture row; the shaders shift by 11
const PALETTE_MAX = 16;

const LINK_VS = `#version 300 es
precision highp float;
precision highp int;
in vec2 a_corner;            // x: t along the link, y: side, -1 or 1
in uvec2 a_ends;             // per instance: source and target
uniform highp sampler2D u_nodes;
uniform vec2 u_scale, u_offset, u_viewport;
uniform float u_width;       // quad width in pixels, the line plus the anti-aliasing fringe
uniform float u_curve;       // control point offset, as a share of the link's length
uniform vec3 u_palette[${PALETTE_MAX}];
out vec3 v_color;
out float v_dist;            // pixels from the center line
vec4 node(uint i) { return texelFetch(u_nodes, ivec2(int(i & 2047u), int(i >> 11)), 0); }
void main() {
  vec4 s = node(a_ends.x), t = node(a_ends.y);
  vec2 half_vp = 0.5 * u_viewport;
  vec2 a = (s.xy * u_scale + u_offset) * half_vp, b = (t.xy * u_scale + u_offset) * half_vp;
  vec2 d = b - a;
  float len = length(d);
  vec2 nrm = len > 0.0 ? vec2(-d.y, d.x) / len : vec2(0.0, 1.0);
  vec2 c = 0.5 * (a + b) + nrm * len * u_curve;
  // Links whose curve's hull is off screen collapse to nothing
  vec2 lo = min(min(a, b), c) - u_width, hi = max(max(a, b), c) + u_width;
  if (any(greaterThan(lo, half_vp)) || any(lessThan(hi, -half_vp))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float k = a_corner.x;
  vec2 p = mix(mix(a, c, k), mix(c, b, k), k);
  vec2 tan = (1.0 - k) * (c - a) + k * (b - c);
  float tl = length(tan);
  vec2 side = tl > 0.0 ? vec2(-tan.y, tan.x) / tl : nrm;
  v_dist = a_corner.y * 0.5 * u_width;
  gl_Position = vec4((p + side * v_dist) / half_vp, 0.0, 1.0);
  v_color = u_palette[int(s.z)];
}`;

const LINK_FS = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_dist;
uniform float u_line, u_alpha;
out vec4 o;
void main() {
  float a = u_alpha * clamp(0.5 * u_line + 0.5 - abs(v_dist), 0.0, 1.0);
  o = vec4(v_color * a, a);
}`;

const POINT_VS = `#version 300 es
precision highp float;
in vec2 a_quad;
uniform highp sampler2D u_nodes;
uniform vec2 u_scale, u_offset, u_viewport;
uniform float u_radius;
uniform vec3 u_palette[${PALETTE_MAX}];
out vec2 v_uv;
out vec3 v_color;
flat out float v_r;
void main() {
  int i = gl_InstanceID;
  vec4 nd = texelFetch(u_nodes, ivec2(i & 2047, i >> 11), 0);
  vec2 half_vp = 0.5 * u_viewport;
  float r = u_radius * nd.w;
  vec2 c = nd.xy * u_scale + u_offset;
  v_uv = a_quad * (r + 1.0);
  v_r = r;
  v_color = u_palette[int(nd.z)];
  gl_Position = vec4(c + v_uv / half_vp, 0.0, 1.0);
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
in vec3 v_color;
flat in float v_r;
out vec4 o;
void main() {
  float d = length(v_uv);
  float cov = clamp(v_r + 0.5 - d, 0.0, 1.0);
  if (cov <= 0.0) discard;
  vec2 q = v_uv / max(v_r, 1e-3);
  vec3 n = vec3(q, sqrt(max(0.0, 1.0 - dot(q, q))));
  vec3 l = normalize(vec3(-0.45, 0.55, 0.7));
  float diff = max(dot(n, l), 0.0);
  float spec = pow(max(dot(reflect(-l, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
  vec3 col = v_color * (0.35 + 0.75 * diff) + 0.3 * spec;
  o = vec4(col * cov, cov);
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
    const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, "");
    u[name] = gl.getUniformLocation(prog, name);
  }
  return { prog, u };
}

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

export async function loadPayload(base) {
  const [header, buf] = await Promise.all([
    fetch(base + ".json").then((r) => r.json()),
    fetch(base + ".bin").then((r) => r.arrayBuffer()),
  ]);
  return { header, buf };
}

// opts: curved (bool), segments, straightSegments, dark (bool), density (bool), res, fringe,
// sample, budget, links and points (bool, to time each alone), dpr, lineWidth (CSS px), alpha, radius (CSS px)
export function createViewer(canvas, { header, buf }, opts = {}) {
  const dpr = opts.dpr ?? window.devicePixelRatio ?? 1;
  const dark = opts.dark ?? true;
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false });
  if (!gl) throw new Error("WebGL2 is not available");
  const n = header.nodes, m = header.edges;
  const xy = new Float32Array(buf, header.xy[0], n * 2);
  const ends = new Uint32Array(buf, header.ends[0], m * 2);
  const color = new Uint8Array(buf, header.color[0], n);

  // The node texture: x, y, color index, radius factor
  const rows = Math.ceil(n / TEX_W);
  const texel = new Float32Array(TEX_W * rows * 4);
  for (let i = 0; i < n; i++) {
    texel[4 * i] = xy[2 * i];
    texel[4 * i + 1] = xy[2 * i + 1];
    texel[4 * i + 2] = color[i];
    texel[4 * i + 3] = 1;
  }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, rows, 0, gl.RGBA, gl.FLOAT, texel);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);

  const palette = new Float32Array(PALETTE_MAX * 3);
  header.palette.slice(0, PALETTE_MAX).forEach((h, i) => palette.set(hexToRgb(h), 3 * i));

  // Links: a strip per instance
  const link = compile(gl, LINK_VS, LINK_FS);
  // Straight links can be split too: a long thin triangle is binned into every tile its bounding
  // box touches, so pieces of a link cost less than the whole of it
  const segments = opts.curved ? (opts.segments ?? 16) : (opts.straightSegments ?? 8);
  const strip = new Float32Array((segments + 1) * 4);
  for (let s = 0; s <= segments; s++) strip.set([s / segments, -1, s / segments, 1], 4 * s);
  const linkVao = gl.createVertexArray();
  gl.bindVertexArray(linkVao);
  const corner = gl.getAttribLocation(link.prog, "a_corner");
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, strip, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(corner);
  gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
  const endsLoc = gl.getAttribLocation(link.prog, "a_ends");
  const endsBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, endsBuf);
  gl.bufferData(gl.ARRAY_BUFFER, ends, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(endsLoc);
  gl.vertexAttribIPointer(endsLoc, 2, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(endsLoc, 1);

  // Points: a quad per instance
  const point = compile(gl, POINT_VS, POINT_FS);
  const pointVao = gl.createVertexArray();
  gl.bindVertexArray(pointVao);
  const quad = gl.getAttribLocation(point.prog, "a_quad");
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(quad);
  gl.vertexAttribPointer(quad, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  const cam = { zoom: 1, cx: 0, cy: 0 };
  let ex = 1e-6, ey = 1e-6;
  for (let i = 0; i < n; i++) {
    ex = Math.max(ex, Math.abs(xy[2 * i]));
    ey = Math.max(ey, Math.abs(xy[2 * i + 1]));
  }
  // Pixels per world unit: the zoom times the scale that fits the graph to 95% of the canvas
  const pxPerUnit = (w, h) => 0.95 * Math.min(w / 2 / ex, h / 2 / ey) * cam.zoom;
  const lineCss = opts.lineWidth ?? 1;
  const alpha = opts.alpha ?? Math.min(0.4, Math.max(0.03, 2e4 / m));
  const density = (opts.density ?? true) && gl.getExtension("EXT_color_buffer_float") !== null;
  const tone = compile(gl, TONE_VS, TONE_FS);
  const emptyVao = gl.createVertexArray();
  let fbo = null, fboTex = null, fboSize = [0, 0];
  function densityTarget(w, h) {
    if (fboSize[0] === w && fboSize[1] === h) return;
    if (fbo) { gl.deleteFramebuffer(fbo); gl.deleteTexture(fboTex); }
    fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, fboTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fboSize = [w, h];
  }

  function resize() {
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
  }

  // The camera's uniforms, and the density buffer's size and line weight
  function view() {
    resize();
    const vw = canvas.width, vh = canvas.height;
    const px = pxPerUnit(vw, vh);
    const scale = [px / (vw / 2), px / (vh / 2)];
    const offset = [-cam.cx * scale[0], -cam.cy * scale[1]];
    // The density buffer can be drawn at a share of the canvas's resolution (res), and is then
    // scaled up by the tone pass. Thin lines draw 1 px wide at their width's share of the alpha
    const res = density ? (opts.res ?? 1) : 1;
    const dw = Math.round(vw * res), dh = Math.round(vh * res);
    const line = Math.max(lineCss * dpr * res, 1);
    const share = alpha * Math.min(lineCss * dpr * res, 1);
    return { vw, vh, scale, offset, dw, dh, line, a: density ? -Math.log(1 - share) : share };
  }

  // Links [from, to) of the payload's order; into the density buffer, cleared first if `clear`
  let drawn = 0;
  function linkPass(v, from, to, clear) {
    gl.enable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    if (density) {
      densityTarget(v.dw, v.dh);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, v.dw, v.dh);
      if (clear) {
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
      gl.blendFunc(gl.ONE, gl.ONE);
    } else {
      // Additive on dark, "over" on light
      gl.viewport(0, 0, v.vw, v.vh);
      if (dark) gl.blendFunc(gl.ONE, gl.ONE);
      else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.useProgram(link.prog);
    gl.uniform1i(link.u.u_nodes, 0);
    gl.uniform2f(link.u.u_scale, ...v.scale);
    gl.uniform2f(link.u.u_offset, ...v.offset);
    gl.uniform2f(link.u.u_viewport, v.dw, v.dh);
    gl.uniform1f(link.u.u_width, v.line + (opts.fringe ?? 2));
    gl.uniform1f(link.u.u_line, v.line);
    gl.uniform1f(link.u.u_curve, opts.curved ? 0.15 : 0);
    gl.uniform1f(link.u.u_alpha, v.a);
    gl.uniform3fv(link.u.u_palette, palette);
    gl.bindVertexArray(linkVao);
    // WebGL2 has no base instance, so the range starts where the attribute is pointed
    gl.bindBuffer(gl.ARRAY_BUFFER, endsBuf);
    gl.vertexAttribIPointer(endsLoc, 2, gl.UNSIGNED_INT, 0, from * 8);
    if ((opts.links ?? true) && to > from) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, (segments + 1) * 2, to - from);
    drawn = clear ? to : drawn + (to - from);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // The background, the density tone-mapped as if all links were drawn, and the points over it
  function composite(v) {
    gl.viewport(0, 0, v.vw, v.vh);
    if (density) {
      const bg = dark ? [0.07, 0.07, 0.08] : [0.98, 0.98, 0.98];
      gl.clearColor(...bg, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(tone.prog);
      gl.bindTexture(gl.TEXTURE_2D, fboTex);
      gl.uniform1i(tone.u.u_density, 0);
      gl.uniform2f(tone.u.u_size, v.vw, v.vh);
      // The links drawn so far stand in for all of them: a shuffled order makes them a sample
      gl.uniform1f(tone.u.u_gain, m / Math.max(drawn, 1));
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindTexture(gl.TEXTURE_2D, tex);
    }
    // Points grow with the square root of the zoom, so a zoomed-in view opens up space
    const r0 = opts.radius ?? Math.min(6, Math.max(0.75, Math.sqrt((canvas.clientWidth * canvas.clientHeight) / n) * 0.12));
    gl.useProgram(point.prog);
    gl.uniform1i(point.u.u_nodes, 0);
    gl.uniform2f(point.u.u_scale, ...v.scale);
    gl.uniform2f(point.u.u_offset, ...v.offset);
    gl.uniform2f(point.u.u_viewport, v.vw, v.vh);
    gl.uniform1f(point.u.u_radius, r0 * dpr * Math.sqrt(cam.zoom));
    gl.uniform3fv(point.u.u_palette, palette);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(pointVao);
    if (opts.points ?? true) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.bindVertexArray(null);
  }

  // A frame of a new view, with the first `count` links (all by default, or opts.sample of them)
  function draw(count = Math.ceil(m * Math.min(1, opts.sample ?? 1))) {
    const v = view();
    count = Math.min(m, count);
    if (!density) {
      gl.viewport(0, 0, v.vw, v.vh);
      const bg = dark ? [0.07, 0.07, 0.08] : [0.98, 0.98, 0.98];
      gl.clearColor(...bg, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      count = m;
    }
    linkPass(v, 0, count, true);
    composite(v);
  }

  // Adds the next `chunk` links to the view, and redraws; false once every link is in
  function refine(chunk) {
    if (!density || drawn >= m) return false;
    const v = view();
    linkPass(v, drawn, Math.min(m, drawn + chunk), false);
    composite(v);
    return true;
  }

  // Waits for the GPU to finish the frame, by reading back one pixel
  const pixel = new Uint8Array(4);
  function sync() {
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  }

  // A moved view draws `budget` links at once, and the rest over the frames that follow
  const budget = opts.budget ?? 150_000;
  let pending = false, refining = 0;
  const redraw = () => {
    if (pending) return;
    pending = true;
    cancelAnimationFrame(refining);
    requestAnimationFrame(() => {
      pending = false;
      draw(Math.min(m, budget));
      const more = () => { if (refine(budget)) refining = requestAnimationFrame(more); };
      refining = requestAnimationFrame(more);
    });
  };
  // Pan by dragging, zoom about the pointer with the wheel
  const toWorld = (ex, ey) => {
    const r = canvas.getBoundingClientRect();
    const px = pxPerUnit(r.width, r.height);
    return [cam.cx + (ex - r.left - r.width / 2) / px, cam.cy - (ey - r.top - r.height / 2) / px];
  };
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = canvas.getBoundingClientRect();
    const px = pxPerUnit(r.width, r.height);
    cam.cx -= (e.clientX - drag[0]) / px;
    cam.cy += (e.clientY - drag[1]) / px;
    drag = [e.clientX, e.clientY];
    redraw();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    const f = Math.exp(-e.deltaY * 0.002);
    cam.zoom = Math.min(500, Math.max(0.2, cam.zoom * f));
    const [nx, ny] = toWorld(e.clientX, e.clientY);
    cam.cx += wx - nx;
    cam.cy += wy - ny;
    redraw();
  }, { passive: false });

  return { gl, cam, draw, refine, sync, redraw };
}
