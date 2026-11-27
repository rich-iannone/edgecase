// cosmos.gl on the same payload, camera path and timing as ours.html. Bundled by esbuild into
// cosmos_bundle.js (see bench.py).
import { Graph } from "@cosmos.gl/graph";
import { params, run, gpuName } from "./bench.js";

const t0 = performance.now();
const base = "data/" + (params.get("data") ?? "wiki_vote");
const [header, buf] = await Promise.all([
  fetch(base + ".json").then((r) => r.json()),
  fetch(base + ".bin").then((r) => r.arrayBuffer()),
]);
const t1 = performance.now();
const n = header.nodes, m = header.edges;
const xy = new Float32Array(buf, header.xy[0], n * 2);
const ends = new Uint32Array(buf, header.ends[0], m * 2);
const color = new Uint8Array(buf, header.color[0], n);
const rgb = header.palette.map((h) => {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
});
// cosmos.gl takes links and colors as Float32Array
const links = Float32Array.from(ends);
const colors = new Float32Array(n * 4);
for (let i = 0; i < n; i++) colors.set([...rgb[color[i]], 1], 4 * i);

const div = document.getElementById("g");
const radius = Math.min(6, Math.max(0.75, Math.sqrt((innerWidth * innerHeight) / n) * 0.12));
const graph = new Graph(div, {
  enableSimulation: false,
  enableRenderLoop: false,
  transitionDuration: 0,
  fitViewOnInit: true,
  fitViewDelay: 0,
  fitViewDuration: 0,
  fitViewPadding: 0.05,
  rescalePositions: true,
  backgroundColor: "#121214",
  pointDefaultSize: 2 * radius,
  linkDefaultWidth: 1,
  linkOpacity: Math.min(0.4, Math.max(0.03, 2e4 / m)),
  linkColorInterpolateFromEndpoints: true,
  curvedLinks: params.get("curved") === "1",
  attribution: "",
});
await graph.ready;
graph.setPointPositions(xy.slice());
graph.setPointColors(colors);
graph.setLinks(links);
graph.render(undefined, 0);
const t2 = performance.now();

const canvas = div.querySelector("canvas");
const gl = canvas.getContext("webgl2");
const pixel = new Uint8Array(4);
const sync = () => {
  const bound = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  gl.bindFramebuffer(gl.FRAMEBUFFER, bound);
};
const draw = () => graph.renderOneFrame();
draw();
sync();
const t3 = performance.now();
// The fit runs as a d3 transition, which applies on the next timer tick even at duration 0, so
// it comes after the load is timed
graph.fitView(0, 0.05);
await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
draw();
sync();
const z0 = graph.getZoomLevel();
const result = {
  renderer: "cosmos.gl",
  gpu: gpuName(gl),
  canvas: [canvas.width, canvas.height],
  load: { fetch: +(t1 - t0).toFixed(1), setup: +(t2 - t1).toFixed(1), first_frame: +(t3 - t2).toFixed(1), total: +(t3 - t0).toFixed(1) },
  zoom0: z0,
};
if (params.get("bench") === "1") {
  Object.assign(result, run({ setZoom: (z) => graph.setZoomLevel(z0 * z, 0, false), draw, sync }));
}
window.__result = result;
