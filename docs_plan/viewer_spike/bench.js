// The harness both pages share: the same camera path, and the same timing. Each frame is drawn
// and then waited for by reading back a pixel, so its time is the whole CPU and GPU cost of the
// frame, without the display's refresh capping it. (Timing from requestAnimationFrame says
// nothing in headless Chrome, which runs it unthrottled whatever the GPU is doing.)
//
// The camera zooms from the fitted view to 16x and back, about the center.

export const params = new URLSearchParams(location.search);

export function zoomAt(k, frames) {
  return Math.exp(Math.log(16) * (1 - Math.cos((2 * Math.PI * k) / frames)) / 2);
}

export function stats(ms) {
  const s = [...ms].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: +q(0.5).toFixed(2), p95: +q(0.95).toFixed(2), fps: +(1000 / q(0.5)).toFixed(1) };
}

// setZoom(z) moves the camera, draw() draws a frame, sync() waits for the GPU
export function run({ setZoom, draw, sync }, frames = 120) {
  // Warm up: shader compiles and first uploads aren't part of a frame's cost
  for (let k = 0; k < 5; k++) { setZoom(zoomAt(k, frames)); draw(); sync(); }
  const ms = [];
  for (let k = 0; k < frames; k++) {
    const t = performance.now();
    setZoom(zoomAt(k, frames));
    draw();
    sync();
    ms.push(performance.now() - t);
  }
  setZoom(1);
  draw();
  sync();
  return { frame: stats(ms) };
}

export function gpuName(gl) {
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}
