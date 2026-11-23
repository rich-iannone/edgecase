// The homepage hero's animated backdrop: a network drifting across the page, its nodes gathered in
// a few loose communities. Now and then a breadth-first search spreads out from one of them,
// lighting each edge and node as it reaches them, hop by hop (and a click on the backdrop starts
// one from the nearest node). In dark mode the network glows. It's faint behind the logo and
// tagline, paused out of view, and still with reduced motion.
(() => {
  "use strict";

  const hero = document.querySelector(".gd-hero");
  if (!hero) return;
  const canvas = document.createElement("canvas");
  canvas.id = "ec-backdrop";
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  hero.prepend(canvas);

  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const root = document.documentElement;
  const isDark = () =>
    root.classList.contains("quarto-dark") ||
    document.body.classList.contains("quarto-dark") ||
    root.getAttribute("data-bs-theme") === "dark";

  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const pick = (items) => items[(Math.random() * items.length) | 0];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const smooth = (lo, hi, x) => {
    const k = clamp((x - lo) / (hi - lo), 0, 1);
    return k * k * (3 - 2 * k);
  };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  // (A standard normal deviate)
  const normal = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

  // The canvas's size (in CSS pixels); the logo and tagline's box in it, which is kept clear; and
  // the page's text over it (the lead, the calls to action, the stats and the sidebar), kept nearly so
  let W = 0;
  let H = 0;
  let dpr = 1;
  const focus = { left: 0, right: 0, top: 0, bottom: 0 };
  let texts = [];
  const TEXTS = [
    "#quarto-document-content > p:first-of-type",
    ".ec-cta",
    ".ec-stats",
    "#quarto-margin-sidebar",
  ];

  // How much of a thing to draw at a point: less near the logo and tagline, and over text
  const gap = (r, x, y) => Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
  const clearing = (x, y, over = 0.3) => {
    let k = 0.1 + 0.9 * smooth(0, 130, gap(focus, x, y));
    for (const r of texts) k = Math.min(k, over + (1 - over) * smooth(0, 90, gap(r, x, y)));
    return k;
  };

  function layout() {
    const box = hero.getBoundingClientRect();
    const nav = document.getElementById("quarto-header");
    const navHeight = nav ? nav.offsetHeight : 0;
    const heroTop = box.top + window.scrollY;
    // From under the navbar to below the hero (where it fades out), across the page
    const top = navHeight - heroTop;
    const width = root.clientWidth;
    const height = Math.round(heroTop - navHeight + box.height + Math.min(380, window.innerHeight * 0.45));
    canvas.style.left = `${-box.left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // (The logo shown, for the theme, and the tagline)
    const parts = Array.from(hero.querySelectorAll(".gd-hero-logo img, .gd-hero-tagline"))
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0);
    if (parts.length > 0) {
      focus.left = Math.min(...parts.map((r) => r.left));
      focus.right = Math.max(...parts.map((r) => r.right));
      focus.top = Math.min(...parts.map((r) => r.top)) - box.top - top;
      focus.bottom = Math.max(...parts.map((r) => r.bottom)) - box.top - top;
    } else {
      focus.left = width * 0.3;
      focus.right = width * 0.7;
      focus.top = height * 0.2;
      focus.bottom = height * 0.5;
    }
    texts = TEXTS.map((selector) => document.querySelector(selector))
      .filter((el) => el !== null)
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0)
      .map((r) => ({ left: r.left, right: r.right, top: r.top - box.top - top, bottom: r.bottom - box.top - top }));

    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const resized = Math.abs(width - W) > W * 0.1 || Math.abs(height - H) > H * 0.1;
    if (width !== W || height !== H || ratio !== dpr) {
      W = width;
      H = height;
      dpr = ratio;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    return resized;
  }

  /* ── The network ───────────────────────────────────────────────────────────────────────────── */

  // The communities' colors (the site's orange, teal and violet), the color of the nodes in none,
  // and the search's color, for each theme
  const THEMES = {
    light: {
      groups: [[234, 88, 12], [8, 145, 178], [124, 58, 237]],
      plain: [100, 116, 139],
      accent: [234, 88, 12],
      node: 0.8,
      edge: 0.42,
      glow: 0,
    },
    dark: {
      groups: [[249, 115, 22], [34, 211, 238], [167, 139, 250]],
      plain: [139, 148, 158],
      accent: [251, 146, 60],
      node: 0.95,
      edge: 0.5,
      glow: 0.6,
    },
  };
  const theme = () => (isDark() ? THEMES.dark : THEMES.light);

  const HOP = 0.42; // (seconds for the search to cross an edge)
  const AFTERGLOW = 2.2; // (seconds for what it lit to fade)
  const MAX_HOPS = 6;

  let reach = 100;
  let nodes = [];
  let clusters = [];
  let edges = [];
  let waves = [];
  let nextWave = 0;

  function initNetwork() {
    const scale = clamp(W / 1400, 0.7, 1.1);
    reach = 96 * scale;
    const count = clamp(Math.round((W * H) / 7600), 36, 170);
    const k = clamp(Math.round(W / 330), 3, 7);
    clusters = [];
    for (let i = 0; i < k; i++) {
      clusters.push({
        x: ((i + rand(0.3, 0.7)) * W) / k,
        y: H * (i % 2 ? rand(0.5, 0.72) : rand(0.2, 0.42)),
        group: i % 3,
        phase: rand(0, Math.PI * 2),
      });
    }
    const spread = Math.min(W / k, H) * 0.32;
    nodes = [];
    for (let i = 0; i < count; i++) {
      // (Most in a community; some on their own, anywhere)
      const loose = Math.random() < 0.26;
      const home = loose ? null : pick(clusters);
      nodes.push({
        home,
        group: home ? home.group : -1,
        ox: home ? normal() * spread : rand(0, W),
        oy: home ? normal() * spread * 0.8 : rand(0, H * 0.85),
        ax: rand(8, 26),
        ay: rand(6, 18),
        fx: rand(0.05, 0.13),
        fy: rand(0.04, 0.11),
        px: rand(0, Math.PI * 2),
        py: rand(0, Math.PI * 2),
        r: rand(2, 4.4) * scale,
        x: 0,
        y: 0,
      });
    }
  }

  function move(t) {
    for (const node of nodes) {
      let x = node.ox;
      let y = node.oy;
      if (node.home) {
        x += node.home.x + 22 * Math.sin(t * 0.045 + node.home.phase);
        y += node.home.y + 14 * Math.cos(t * 0.038 + node.home.phase);
      }
      node.x = x + node.ax * Math.sin(t * node.fx + node.px);
      node.y = y + node.ay * Math.cos(t * node.fy + node.py);
    }
    // The edges: between nodes near each other (nearer still if they're in different communities)
    edges = [];
    const far = reach * reach;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > far) continue;
        const same = a.group >= 0 && a.group === b.group;
        const limit = same ? reach : reach * 0.6;
        const d = Math.sqrt(d2);
        if (d < limit) edges.push([i, j, d / limit, same]);
      }
    }
  }

  // A breadth-first search from a node, over the edges as they are now
  function startWave(t, source = null) {
    const adj = nodes.map(() => []);
    for (const [i, j] of edges) {
      adj[i].push(j);
      adj[j].push(i);
    }
    if (source === null) {
      const choices = nodes
        .map((node, i) => i)
        .filter((i) => adj[i].length >= 2 && clearing(nodes[i].x, nodes[i].y) > 0.6 && nodes[i].x > 0 && nodes[i].x < W);
      if (choices.length === 0) return;
      source = pick(choices);
    }
    const level = new Map([[source, 0]]);
    const parent = new Map();
    const queue = [source];
    let depth = 0;
    while (queue.length > 0 && level.size < 48) {
      const u = queue.shift();
      const next = level.get(u) + 1;
      if (next > MAX_HOPS) continue;
      for (const v of adj[u]) {
        if (level.has(v)) continue;
        level.set(v, next);
        parent.set(v, u);
        depth = Math.max(depth, next);
        queue.push(v);
      }
    }
    // (Mostly in the site's orange; now and then in the source's community's color)
    const group = nodes[source].group;
    const tone = group >= 0 && Math.random() < 0.35 ? group : -1;
    waves.push({ start: t, source, level, parent, depth, tone });
  }

  // Soft glows, drawn from a sprite for each color
  const sprites = new Map();
  function glow(color, x, y, radius, alpha) {
    if (alpha <= 0.004) return;
    const key = color.join(",");
    let sprite = sprites.get(key);
    if (!sprite) {
      sprite = document.createElement("canvas");
      sprite.width = sprite.height = 64;
      const g = sprite.getContext("2d");
      const fill = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      fill.addColorStop(0, rgba(color, 1));
      fill.addColorStop(0.35, rgba(color, 0.35));
      fill.addColorStop(1, rgba(color, 0));
      g.fillStyle = fill;
      g.fillRect(0, 0, 64, 64);
      sprites.set(key, sprite);
    }
    ctx.globalAlpha = alpha;
    ctx.drawImage(sprite, x - radius, y - radius, radius * 2, radius * 2);
    ctx.globalAlpha = 1;
  }

  function dot(x, y, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // A soft wash of the communities' colors, behind the network
  const WASH = [
    [0, 0.16, 0.3],
    [1, 0.84, 0.28],
    [2, 0.58, 0.82],
  ];

  function drawNetwork(t) {
    const c = theme();
    const dark = isDark();
    const color = (group) => (group >= 0 ? c.groups[group] : c.plain);

    WASH.forEach(([group, fx, fy], k) => {
      const x = W * (fx + 0.05 * Math.sin(t * 0.07 + k * 2));
      const y = H * (fy + 0.06 * Math.cos(t * 0.06 + k));
      const radius = Math.max(W, H) * 0.42;
      const wash = ctx.createRadialGradient(x, y, 0, x, y, radius);
      wash.addColorStop(0, rgba(c.groups[group], dark ? 0.1 : 0.06));
      wash.addColorStop(1, rgba(c.groups[group], 0));
      ctx.fillStyle = wash;
      ctx.fillRect(0, 0, W, H);
    });

    // The edges, fainter the longer they are (and tinted within a community)
    ctx.lineCap = "round";
    ctx.lineWidth = 1.2;
    for (const [i, j, k, same] of edges) {
      const a = nodes[i];
      const b = nodes[j];
      const alpha = c.edge * Math.pow(1 - k, 0.9) * clearing((a.x + b.x) / 2, (a.y + b.y) / 2);
      if (alpha < 0.01) continue;
      ctx.strokeStyle = rgba(same ? color(a.group) : c.plain, alpha);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // The nodes
    if (dark) ctx.globalCompositeOperation = "lighter";
    for (const node of nodes) {
      const room = clearing(node.x, node.y);
      if (dark) glow(color(node.group), node.x, node.y, node.r * 5, c.glow * 0.45 * room);
      dot(node.x, node.y, node.r, rgba(color(node.group), c.node * room));
    }
    ctx.globalCompositeOperation = "source-over";

    // The searches: each edge lit as the search crosses it, then the node it reaches
    if (t >= nextWave) {
      if (waves.length < 2) startWave(t);
      nextWave = t + rand(2.6, 4.4);
    }
    waves = waves.filter((wave) => t - wave.start < wave.depth * HOP + AFTERGLOW + 0.3);
    if (dark) ctx.globalCompositeOperation = "lighter";
    for (const wave of waves) {
      const tone = wave.tone >= 0 ? c.groups[wave.tone] : c.accent;
      for (const [v, hops] of wave.level) {
        if (hops === 0) continue;
        const a = nodes[wave.parent.get(v)];
        const b = nodes[v];
        const t0 = wave.start + (hops - 1) * HOP;
        const k = (t - t0) / HOP;
        if (k <= 0) continue;
        const reached = Math.min(1, k);
        const fade = 1 - smooth(0, AFTERGLOW, t - t0 - HOP);
        const room = clearing((a.x + b.x) / 2, (a.y + b.y) / 2, 0.15);
        const x = a.x + (b.x - a.x) * reached;
        const y = a.y + (b.y - a.y) * reached;
        // (A wide, faint stroke under a fine one, for a glow)
        ctx.strokeStyle = rgba(tone, (dark ? 0.22 : 0.12) * fade * room);
        ctx.lineWidth = dark ? 6 : 5;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.strokeStyle = rgba(tone, (dark ? 0.95 : 0.8) * fade * room);
        ctx.lineWidth = 1.6;
        ctx.stroke();
        // (The search on its way along the edge)
        if (k < 1) {
          glow(tone, x, y, 12, (dark ? 0.8 : 0.45) * room);
          dot(x, y, 2.4, rgba(tone, room));
        }
      }
      for (const [v, hops] of wave.level) {
        const age = t - (wave.start + hops * HOP);
        if (age < 0) continue;
        const node = nodes[v];
        const fade = 1 - smooth(0, AFTERGLOW + 0.4, age);
        const pop = 1 + 0.9 * Math.exp(-age * 5);
        const room = clearing(node.x, node.y, 0.15);
        glow(tone, node.x, node.y, node.r * 7 * pop, (dark ? 0.7 : 0.35) * fade * room);
        dot(node.x, node.y, (node.r + 1.2) * pop, rgba(tone, 0.95 * fade * room));
        // (The source, with a ring that spreads from it)
        if (hops === 0 && age < 1.6) {
          ctx.strokeStyle = rgba(tone, 0.6 * (1 - age / 1.6) * room);
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r + 4 + age * 34, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    ctx.globalCompositeOperation = "source-over";

    // The pointer, as a node of its own, joined to the nodes near it
    if (pointer) {
      const room = clearing(pointer.x, pointer.y, 0.15);
      ctx.lineWidth = 1.2;
      for (const node of nodes) {
        const d = Math.hypot(node.x - pointer.x, node.y - pointer.y);
        if (d > reach * 1.25) continue;
        ctx.strokeStyle = rgba(c.accent, 0.55 * (1 - d / (reach * 1.25)) * room);
        ctx.beginPath();
        ctx.moveTo(pointer.x, pointer.y);
        ctx.lineTo(node.x, node.y);
        ctx.stroke();
      }
      dot(pointer.x, pointer.y, 3, rgba(c.accent, 0.8 * room));
    }
  }

  /* ── Running it ────────────────────────────────────────────────────────────────────────────── */

  let t = 0;
  let last = 0;
  let frame = 0;
  let inView = true;
  let ready = false;
  let pointer = null;

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    move(t);
    drawNetwork(t);
  }

  const running = () => inView && !motion.matches && !document.hidden;

  function tick(now) {
    frame = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    t += dt;
    draw();
    if (running()) frame = requestAnimationFrame(tick);
  }

  function start() {
    if (!ready) return;
    if (running()) {
      if (frame === 0) {
        last = 0;
        frame = requestAnimationFrame(tick);
      }
    } else {
      // (Still: one frame, as it is)
      if (frame !== 0) cancelAnimationFrame(frame);
      frame = 0;
      draw();
    }
  }

  let pending = 0;
  function refit() {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      if (layout() || !ready) {
        initNetwork();
        // (With a search already under way)
        move(t);
        waves = [];
        startWave(t - 1.3);
        nextWave = t + 1.5;
      }
      ready = true;
      canvas.classList.add("ec-on");
      start();
    });
  }

  // The pointer (a mouse's), where it's over the backdrop
  window.addEventListener(
    "pointermove",
    (event) => {
      if (event.pointerType !== "mouse") return;
      const r = canvas.getBoundingClientRect();
      const x = event.clientX - r.left;
      const y = event.clientY - r.top;
      pointer = x >= 0 && x <= r.width && y >= 0 && y <= r.height * 0.8 ? { x, y } : null;
    },
    { passive: true },
  );
  document.addEventListener("pointerleave", () => {
    pointer = null;
  });

  // A click on the backdrop (not on a link, a button or code) starts a search from the nearest node
  document.addEventListener("pointerdown", (event) => {
    if (!ready || event.button !== 0) return;
    if (event.target.closest("a, button, input, select, textarea, label, summary, pre, code, .cell, .ec-cta, svg, img")) return;
    const r = canvas.getBoundingClientRect();
    const x = event.clientX - r.left;
    const y = event.clientY - r.top;
    if (x < 0 || x > r.width || y < 0 || y > r.height * 0.8) return;
    let best = -1;
    let bestD = 160;
    nodes.forEach((node, i) => {
      const d = Math.hypot(node.x - x, node.y - y);
      if (d < bestD) {
        best = i;
        bestD = d;
      }
    });
    if (best < 0) return;
    if (waves.length >= 3) waves.shift();
    startWave(t, best);
    if (!running()) draw();
  });

  // (Started a little way into its life, so that it's under way)
  t = 20;
  refit();
  if ("ResizeObserver" in window) new ResizeObserver(refit).observe(hero);
  window.addEventListener("resize", refit);
  motion.addEventListener("change", start);
  document.addEventListener("visibilitychange", start);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      inView = entries[entries.length - 1].isIntersecting;
      start();
    }).observe(canvas);
  }
  // (A theme's toggle: redrawn at once, if it's still)
  const redraw = () => {
    if (ready && !running()) draw();
  };
  new MutationObserver(redraw).observe(root, { attributes: true, attributeFilter: ["class", "data-bs-theme"] });
  new MutationObserver(redraw).observe(document.body, { attributes: true, attributeFilter: ["class"] });
})();
