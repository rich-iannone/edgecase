"""Time ours.html against cosmos.html in Chrome with the GPU, and screenshot both.

    uv run --no-project --with playwright python docs_plan/viewer_spike/bench.py [--headed]

Run export.py first, and bundle cosmos_page.js:

    npx esbuild cosmos_page.js --bundle --format=esm --minify --outfile=cosmos_bundle.js
"""

import functools
import http.server
import json
import statistics
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SHOTS = HERE / "shots"
FLAGS = [
    "--use-angle=metal",
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--disable-gpu-vsync",
    "--disable-frame-rate-limit",
]
VIEWPORT = {"width": 1400, "height": 1000}
DPR = 2
LOAD_RUNS = 3
# renderer, query. Ours: straight links in 8 pieces (the default), and in one; without the
# density pass; a moving view's frames (150k links, the rest added once it stops); curved.
# cosmos.gl: straight and curved
CASES = [
    ("ours", "curved=0"),
    ("ours", "curved=0&seg=1"),
    ("ours", "curved=0&density=0"),
    ("ours", "curved=0&motion=1"),
    ("ours", "curved=1"),
    ("cosmos", "curved=0"),
    ("cosmos", "curved=1"),
]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve():
    handler = functools.partial(Quiet, directory=str(HERE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"


def page_result(browser, url, shot=None):
    context = browser.new_context(viewport=VIEWPORT, device_scale_factor=DPR)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(url)
    try:
        page.wait_for_function("window.__result !== undefined", timeout=300_000)
    except Exception:
        raise RuntimeError(f"{url}: {errors}")
    result = page.evaluate("window.__result")
    if shot:
        page.screenshot(path=str(shot))
    context.close()
    return result


def main():
    SHOTS.mkdir(exist_ok=True)
    base = serve()
    rows = []
    headless = "--headed" not in sys.argv
    with sync_playwright() as p:

        def launch():
            return p.chromium.launch(channel="chrome", headless=headless, args=FLAGS)

        for data in ("wiki_vote", "rand_1m"):
            for renderer, query in CASES:
                url = f"{base}/{renderer}.html?data={data}&{query}"
                # Each load in a new browser, so that no shader or HTTP cache is warm
                loads = []
                for _ in range(LOAD_RUNS):
                    browser = launch()
                    loads.append(page_result(browser, url)["load"])
                    browser.close()
                browser = launch()
                tag = query.replace("&", "_").replace("=", "")
                r = page_result(browser, url + "&bench=1", SHOTS / f"{renderer}_{data}_{tag}.png")
                browser.close()
                r["load"] = {k: statistics.median(run[k] for run in loads) for k in loads[0]}
                r.update(data=data, query=query)
                rows.append(r)
                print(json.dumps(r), flush=True)
    (HERE / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
