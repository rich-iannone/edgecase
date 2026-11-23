// The homepage's showcase: the copy buttons, the stats that count up, the stepper that builds an
// analysis one step at a time, the tour of Cypher queries (each lighting up what it matches in the
// graph), the marquee of method diagrams, and the benchmark race
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // Copy text, and show that it's copied on the button (its icon, for a moment)
  function copyText(button, text) {
    const done = () => {
      const icon = button.querySelector(".bi");
      button.classList.add("ec-copied");
      if (icon) icon.className = "bi bi-check2";
      setTimeout(() => {
        button.classList.remove("ec-copied");
        if (icon) icon.className = "bi bi-clipboard";
      }, 1400);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => {});
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        done();
      } catch (_) {}
      area.remove();
    }
  }

  function setUpCopyButtons() {
    for (const button of document.querySelectorAll("[data-copy]")) {
      button.addEventListener("click", () => copyText(button, button.dataset.copy));
    }
  }

  // Run `callback` once, the first time `el` comes into view (or at once, without the observer)
  function whenSeen(el, callback, margin = "0px 0px -15% 0px") {
    if (!("IntersectionObserver" in window)) {
      callback();
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          callback();
        }
      },
      { rootMargin: margin },
    );
    observer.observe(el);
  }

  // Keep track of whether `el` is across the middle of the window, however tall it is
  function watchInView(el, onChange) {
    if (!("IntersectionObserver" in window)) {
      onChange(true);
      return;
    }
    new IntersectionObserver(
      (entries) => {
        for (const entry of entries) onChange(entry.isIntersecting);
      },
      { rootMargin: "-30% 0px -30% 0px" },
    ).observe(el);
  }

  /* ── The stats, counting up ─────────────────────────────────────────────────────────────────── */

  function setUpStats(stats) {
    const numbers = Array.from(stats.querySelectorAll("[data-count]"));
    const format = (n) => Math.round(n).toLocaleString("en-US");
    if (reduceMotion.matches) return;
    for (const el of numbers) el.textContent = `0${el.dataset.suffix || ""}`;
    whenSeen(stats, () => {
      const begin = performance.now();
      const DURATION = 1500;
      const step = (now) => {
        const k = Math.min(1, (now - begin) / DURATION);
        const eased = 1 - Math.pow(1 - k, 3);
        for (const el of numbers) {
          el.textContent = `${format(Number(el.dataset.count) * eased)}${el.dataset.suffix || ""}`;
        }
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ── Tabs over stacked panels (for the stepper and the tour) ───────────────────────────────── */

  // The lines of a code cell's code, as elements, each with the line break after it (so that a line
  // can be folded away; the code's text, which the copy button copies, is as it was)
  function codeLines(cell) {
    const lines = Array.from(cell.querySelectorAll("pre code > span"));
    for (const line of lines) {
      const after = line.nextSibling;
      if (after && after.nodeType === Node.TEXT_NODE && after.data === "\n") line.appendChild(after);
    }
    return lines;
  }

  // Fold the runs of lines that aren't new (but the line before and after each new one) into a
  // line that says how many there are, which unfolds them when clicked
  function foldUnchanged(lines, added) {
    const keep = lines.map((_, k) => added[k] || added[k - 1] || added[k + 1]);
    let k = 0;
    while (k < lines.length) {
      if (keep[k]) {
        k++;
        continue;
      }
      let end = k;
      while (end < lines.length && !keep[end]) end++;
      const run = lines.slice(k, end);
      if (run.length >= 3) {
        const [mark, ...rest] = run;
        mark.classList.add("ec-fold-mark");
        mark.dataset.fold = `${run.length} unchanged lines`;
        mark.tabIndex = 0;
        mark.setAttribute("role", "button");
        mark.setAttribute("aria-label", `Show ${run.length} unchanged lines`);
        for (const line of rest) line.classList.add("ec-folded");
        const unfold = () => {
          mark.classList.remove("ec-fold-mark");
          mark.removeAttribute("role");
          mark.removeAttribute("tabindex");
          mark.removeAttribute("aria-label");
          for (const line of rest) line.classList.remove("ec-folded");
        };
        mark.addEventListener("click", unfold);
        mark.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            unfold();
          }
        });
      }
      k = end;
    }
  }

  // Which lines of `next` aren't in `prev` (by the longest common subsequence of the two)
  function newLines(prev, next) {
    const m = prev.length;
    const n = next.length;
    const lcs = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        lcs[i][j] = prev[i] === next[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
      }
    }
    const added = new Array(n).fill(true);
    let i = 0;
    let j = 0;
    while (i < m && j < n) {
      if (prev[i] === next[j]) {
        added[j] = false;
        i++;
        j++;
      } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
        i++;
      } else {
        j++;
      }
    }
    return added;
  }

  // Mark the lines of a step's code that are new since the step before it, fold the rest, and
  // put the output first, then the code that made it. Returns the step's lines (as text).
  function markNewCode(step, previous) {
    const cell = step.querySelector(".cell");
    if (!cell) return previous;
    const lines = codeLines(cell);
    const text = lines.map((line) => line.textContent.replace(/\s+$/, ""));
    const added = newLines(previous, text);
    let count = 0;
    lines.forEach((line, k) => {
      if (added[k] && text[k] !== "" && previous.length > 0) {
        line.classList.add("ec-new");
        count++;
      }
    });
    if (previous.length > 0) foldUnchanged(lines, added);
    const code = cell.querySelector(".cell-code");
    if (code && count > 0) {
      const badge = document.createElement("span");
      badge.className = "ec-new-count";
      badge.textContent = `${count} new line${count === 1 ? "" : "s"}`;
      code.appendChild(badge);
    }
    const output = cell.querySelector(".cell-output-display");
    if (code && output) cell.insertBefore(output, code);
    return text;
  }

  // Turn the `.ec-step` children of `box` into panels stacked in one place, with a tab for each
  // (named by its `.ec-step-name`), autoplaying through them while in view until the reader takes
  // over. `options.diff` marks each step's new code; `options.loop` goes back to the first after the
  // last; `options.onShow` is called with each step's index as it's shown.
  function setUpTabs(box, options) {
    const steps = Array.from(box.querySelectorAll(":scope > .ec-step"));
    if (steps.length === 0) return;

    const tabs = document.createElement("div");
    tabs.className = options.tabsClass;
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", options.label);

    const viewport = document.createElement("div");
    viewport.className = "ec-panels";
    const stack = document.createElement("div");
    stack.className = "ec-stack";
    viewport.appendChild(stack);

    const id = box.id || options.id;
    let previous = [];

    const buttons = steps.map((step, index) => {
      // The step's name (its first paragraph) goes on its tab
      const nameEl = step.querySelector(".ec-step-name");
      const name = nameEl ? nameEl.textContent.trim() : `Step ${index + 1}`;
      if (nameEl) {
        const para = nameEl.closest("p");
        (para && para.parentElement === step ? para : nameEl).remove();
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = options.tabClass;
      button.id = `${id}-tab-${index}`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `${id}-panel-${index}`);
      const number = options.numbered ? `<span class="ec-tab-num">${index + 1}</span>` : "";
      button.innerHTML = `${number}<span class="ec-tab-name"></span><span class="ec-tab-progress" aria-hidden="true"></span>`;
      button.querySelector(".ec-tab-name").textContent = name;
      tabs.appendChild(button);

      step.id = `${id}-panel-${index}`;
      step.setAttribute("role", "tabpanel");
      step.setAttribute("aria-labelledby", button.id);
      step.classList.add("ec-panel");
      if (options.diff) previous = markNewCode(step, previous);

      stack.appendChild(step);
      return button;
    });

    box.prepend(tabs);
    box.append(viewport);

    let active = -1;
    let observer = null;

    function fitHeight() {
      if (active < 0) return;
      viewport.style.height = `${steps[active].offsetHeight}px`;
    }

    function scrollToNew(step) {
      const pre = step.querySelector(".cell-code pre");
      const first = step.querySelector(".ec-new");
      if (!pre || !first) return;
      const top = first.offsetTop - pre.offsetTop - 12;
      pre.scrollTo({ top: Math.max(0, top), behavior: reduceMotion.matches ? "auto" : "smooth" });
    }

    function show(index, focus = false) {
      if (index === active) return;
      active = index;
      steps.forEach((step, k) => {
        const on = k === index;
        step.classList.toggle("ec-active", on);
        step.inert = !on;
        step.setAttribute("aria-hidden", on ? "false" : "true");
        buttons[k].setAttribute("aria-selected", on ? "true" : "false");
        buttons[k].tabIndex = on ? 0 : -1;
        buttons[k].classList.toggle("ec-done", k < index);
      });
      if (focus) buttons[index].focus();
      if (observer) {
        observer.disconnect();
        observer.observe(steps[index]);
      }
      fitHeight();
      if (options.diff) scrollToNew(steps[index]);
      if (options.onShow) options.onShow(index);
    }

    // Autoplay: through the steps while they're in view, until the reader takes over
    const DWELL = options.dwell || 5200;
    let timer = null;
    let playing = !reduceMotion.matches;
    let visible = false;

    function stopPlaying() {
      playing = false;
      box.classList.remove("ec-playing");
      clearTimeout(timer);
      timer = null;
    }

    function schedule() {
      clearTimeout(timer);
      timer = null;
      if (!playing || !visible) {
        box.classList.remove("ec-playing");
        return;
      }
      if (active >= steps.length - 1 && !options.loop) {
        stopPlaying();
        return;
      }
      // (Restarting the tab's progress bar)
      box.classList.remove("ec-playing");
      void box.offsetWidth;
      box.style.setProperty("--ec-dwell", `${DWELL}ms`);
      box.classList.add("ec-playing");
      timer = setTimeout(() => {
        show((active + 1) % steps.length);
        schedule();
      }, DWELL);
    }

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        stopPlaying();
        show(index);
      });
    });

    tabs.addEventListener("keydown", (event) => {
      const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      let next = null;
      if (event.key in keys) next = (active + keys[event.key] + steps.length) % steps.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = steps.length - 1;
      if (next === null) return;
      event.preventDefault();
      stopPlaying();
      show(next, true);
    });

    // Working with what's shown (or its code) stops the autoplay too
    for (const el of [viewport, ...(options.stopOn || [])]) {
      el.addEventListener("pointerdown", stopPlaying);
      el.addEventListener("focusin", stopPlaying);
    }

    if ("ResizeObserver" in window) observer = new ResizeObserver(fitHeight);
    window.addEventListener("resize", fitHeight);

    box.classList.add("ec-ready");
    show(0);

    watchInView(box, (inView) => {
      visible = inView;
      schedule();
    });
  }

  /* ── The tour of Cypher queries ────────────────────────────────────────────────────────────── */

  const attrsOf = (el) => {
    try {
      return JSON.parse(el.getAttribute("data-attrs") || "{}");
    } catch (_) {
      return {};
    }
  };

  // Get each plot's parts ready to be lit: its edges (the visible group before each edge's hit
  // target), its nodes and their labels
  function prepareStage(stage) {
    const svgs = Array.from(stage.querySelectorAll("svg")).filter((svg) => svg.querySelector("[data-node]"));
    for (const svg of svgs) {
      for (const hit of svg.querySelectorAll("[data-edge]")) {
        const shown = hit.previousElementSibling;
        if (shown && shown.tagName.toLowerCase() === "g") {
          shown.classList.add("ec-edge");
          shown.dataset.edgeId = String(attrsOf(hit).id);
        }
      }
      const labels = new Map();
      for (const circle of svg.querySelectorAll("circle[data-node]")) {
        circle.classList.add("ec-node");
        const label = attrsOf(circle).label;
        if (label != null) labels.set(String(label), circle.dataset.node);
      }
      for (const text of svg.querySelectorAll("text")) {
        const node = labels.get(text.textContent.trim());
        if (node !== undefined) {
          text.classList.add("ec-label");
          text.dataset.node = node;
        }
      }
    }
    return svgs;
  }

  // Light up the nodes and edges a query matched (and dim the rest); the edges are drawn in, one
  // after another, in the order given
  function highlight(svgs, hits) {
    const nodes = new Set((hits.nodes || []).map(String));
    const order = (hits.edges || []).map(String);
    const edges = new Set(order);
    for (const svg of svgs) {
      svg.classList.add("ec-lit");
      for (const el of svg.querySelectorAll(".ec-node, .ec-label")) {
        el.classList.toggle("ec-hit", nodes.has(el.dataset.node));
      }
      for (const g of svg.querySelectorAll(".ec-edge")) {
        const on = edges.has(g.dataset.edgeId);
        g.classList.toggle("ec-hit", on);
        if (!on || reduceMotion.matches || typeof g.animate !== "function") continue;
        const delay = 120 + order.indexOf(g.dataset.edgeId) * 110;
        for (const line of g.querySelectorAll("line, path")) {
          const length = typeof line.getTotalLength === "function" ? line.getTotalLength() : 0;
          if (!length) continue;
          line.animate(
            [
              { strokeDasharray: `${length}`, strokeDashoffset: `${length}` },
              { strokeDasharray: `${length}`, strokeDashoffset: "0" },
            ],
            { duration: 420, delay, easing: "ease-out", fill: "backwards" },
          );
        }
        for (const head of g.querySelectorAll("polygon")) {
          head.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: delay + 330, fill: "backwards" });
        }
      }
    }
  }

  function setUpTour(tour) {
    const stage = tour.querySelector(".ec-stage");
    const queries = tour.querySelector(".ec-queries");
    if (!stage || !queries) return;
    const svgs = prepareStage(stage);
    const steps = Array.from(queries.querySelectorAll(":scope > .ec-step"));
    const hits = steps.map((step) => {
      const data = step.querySelector("script.ec-hits");
      try {
        return data ? JSON.parse(data.textContent) : {};
      } catch (_) {
        return {};
      }
    });
    setUpTabs(queries, {
      id: "ec-tour",
      label: "Queries",
      tabsClass: "ec-chips",
      tabClass: "ec-chip",
      numbered: false,
      loop: true,
      dwell: 5600,
      stopOn: [stage],
      onShow: (index) => highlight(svgs, hits[index]),
    });
    // (The chips go above the graph, and the panels below it)
    const chips = queries.querySelector(":scope > .ec-chips");
    if (chips) tour.prepend(chips);
  }

  /* ── The marquee of method diagrams ─────────────────────────────────────────────────────────── */

  function setUpMarquee(marquee) {
    for (const track of marquee.querySelectorAll(".ec-marquee-track")) {
      // (A second copy of the tiles after the first, so the row loops without a seam)
      const copy = Array.from(track.children).map((tile) => {
        const clone = tile.cloneNode(true);
        clone.setAttribute("aria-hidden", "true");
        clone.tabIndex = -1;
        return clone;
      });
      track.append(...copy);
      const seconds = Math.max(40, track.scrollWidth / 2 / 38);
      track.style.setProperty("--ec-marquee-time", `${seconds}s`);
    }
    marquee.classList.add("ec-ready");
  }

  /* ── The benchmark race ─────────────────────────────────────────────────────────────────────── */

  const LIBRARIES = [
    { key: "edgecase", name: "EdgeCase" },
    { key: "igraph", name: "igraph" },
    { key: "networkx", name: "NetworkX" },
  ];

  const formatTime = (s) => {
    if (s < 1) {
      const ms = s * 1000;
      return `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
    }
    return `${s.toFixed(2)} s`;
  };

  const formatRatio = (r) => (r >= 10 ? `${Math.round(r)}×` : `${r.toFixed(1)}×`);

  function setUpRace(race) {
    let bench;
    try {
      bench = JSON.parse(race.dataset.bench);
    } catch (_) {
      return;
    }
    const rows = bench.rows;
    const short = (task) => task.replace(/^Build from an edge list$/, "Build").replace(/ decomposition$/, "");

    const chips = document.createElement("div");
    chips.className = "ec-chips ec-race-chips";
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", "Tasks");
    const buttons = rows.map((row, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ec-chip";
      button.textContent = short(row.task);
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        stopPlaying();
        run(index);
      });
      chips.appendChild(button);
      return button;
    });

    const stage = document.createElement("div");
    stage.className = "ec-race-stage";
    stage.innerHTML = `<div class="ec-race-head"><span class="ec-race-task"></span><span class="ec-race-graph"></span></div>`;
    const lanes = LIBRARIES.map((lib) => {
      const lane = document.createElement("div");
      lane.className = "ec-lane";
      lane.dataset.lib = lib.key;
      lane.innerHTML =
        `<span class="ec-lane-name">${lib.name}</span>` +
        `<span class="ec-lane-track"><span class="ec-lane-bar"></span></span>` +
        `<span class="ec-lane-time">–</span>`;
      stage.appendChild(lane);
      return {
        lib,
        lane,
        bar: lane.querySelector(".ec-lane-bar"),
        time: lane.querySelector(".ec-lane-time"),
      };
    });
    const summary = document.createElement("p");
    summary.className = "ec-race-summary";
    summary.setAttribute("aria-live", "polite");
    stage.appendChild(summary);

    const table = race.querySelector(".ec-race-table");
    race.prepend(chips, stage);
    if (table) race.appendChild(table);

    const RACE = 3000; // (milliseconds for the slowest to finish)
    let current = -1;
    let frame = 0;
    let timer = null;
    let playing = !reduceMotion.matches;
    let visible = false;
    let begun = false;

    function stopPlaying() {
      playing = false;
      clearTimeout(timer);
      timer = null;
    }

    function finish(row) {
      const ec = row.edgecase;
      const parts = [];
      for (const other of ["networkx", "igraph"]) {
        const name = other === "networkx" ? "NetworkX" : "igraph";
        const r = row[other] / ec;
        if (r >= 1.05) parts.push(`<strong>${formatRatio(r)} faster</strong> than ${name}`);
        else if (r > 1 / 1.05) parts.push(`about as fast as ${name}`);
        else parts.push(`${formatRatio(1 / r)} slower than ${name}`);
      }
      summary.innerHTML = `EdgeCase is ${parts.join(" and ")} here.`;
      summary.classList.add("ec-shown");
    }

    function run(index) {
      current = index;
      const row = rows[index];
      buttons.forEach((button, k) => button.setAttribute("aria-pressed", k === index ? "true" : "false"));
      stage.querySelector(".ec-race-task").textContent = row.task;
      stage.querySelector(".ec-race-graph").textContent = `on ${row.graph}`;
      summary.classList.remove("ec-shown");
      const slowest = Math.max(...LIBRARIES.map((lib) => row[lib.key]));
      for (const lane of lanes) {
        lane.lane.classList.remove("ec-done");
        lane.lane.title = `${lane.lib.name}: ${formatTime(row[lane.lib.key])}`;
      }
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      const begin = performance.now();
      const draw = (now) => {
        const k = reduceMotion.matches ? 1 : Math.min(1, (now - begin) / RACE);
        const elapsed = k * slowest;
        for (const lane of lanes) {
          const total = row[lane.lib.key];
          const shown = Math.min(elapsed, total);
          lane.bar.style.width = `${(shown / slowest) * 100}%`;
          lane.time.textContent = formatTime(shown);
          if (elapsed >= total) lane.lane.classList.add("ec-done");
        }
        if (k < 1) {
          frame = requestAnimationFrame(draw);
        } else {
          finish(row);
          schedule();
        }
      };
      frame = requestAnimationFrame(draw);
    }

    function schedule() {
      clearTimeout(timer);
      timer = null;
      if (!playing || !visible) return;
      timer = setTimeout(() => run((current + 1) % rows.length), 2600);
    }

    race.addEventListener("pointerdown", (event) => {
      if (!event.target.closest(".ec-chip")) stopPlaying();
    });

    // (It starts at the task given, the first time it's in view)
    const first = Math.max(0, rows.findIndex((row) => row.task === race.dataset.start));
    watchInView(race, (inView) => {
      visible = inView;
      if (inView && !begun) {
        begun = true;
        run(first);
      } else if (!inView) {
        clearTimeout(timer);
        timer = null;
      } else if (current >= 0 && summary.classList.contains("ec-shown")) {
        schedule();
      }
    });
    // (Before then, it shows that task at its start)
    buttons[first].setAttribute("aria-pressed", "true");
    stage.querySelector(".ec-race-task").textContent = rows[first].task;
    stage.querySelector(".ec-race-graph").textContent = `on ${rows[first].graph}`;
  }

  // The step player: it starts playing the first time it's in view (the one shown, for the theme),
  // unless it's been played already
  function setUpPlayer(box) {
    if (reduceMotion.matches) return;
    let touched = false;
    box.addEventListener("pointerdown", () => {
      touched = true;
    });
    whenSeen(box, () => {
      if (touched) return;
      const play = Array.from(box.querySelectorAll('.step-controls button[id$="-play"]')).find(
        (button) => button.offsetParent !== null,
      );
      if (play) play.click();
    }, "0px 0px -35% 0px");
  }

  function init() {
    setUpCopyButtons();
    document.querySelectorAll(".ec-stats").forEach(setUpStats);
    document.querySelectorAll(".ec-stepper").forEach((stepper) =>
      setUpTabs(stepper, {
        id: "ec-stepper",
        label: "The steps of the analysis",
        tabsClass: "ec-tabs",
        tabClass: "ec-tab",
        numbered: true,
        diff: true,
      }),
    );
    document.querySelectorAll(".ec-tour").forEach(setUpTour);
    document.querySelectorAll(".ec-marquee").forEach(setUpMarquee);
    document.querySelectorAll(".ec-race").forEach(setUpRace);
    document.querySelectorAll(".ec-player").forEach(setUpPlayer);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
