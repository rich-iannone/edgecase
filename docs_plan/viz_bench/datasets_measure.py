"""Measure plot() on each bundled dataset, each case in its own subprocess."""

import gzip
import json
import resource
import subprocess
import sys
import time
import warnings

DATASETS = [
    "florentine_families", "davis_southern_women", "karate_club", "dolphins", "les_miserables",
    "polbooks", "football", "jazz", "celegans_neural", "celegans", "power_grid", "ca_grqc",
    "internet", "gnutella", "pgp", "wiki_vote", "physics", "condmat", "enron",
]


def one(name: str, labels: bool) -> dict:
    import edgecase as ec

    t = time.perf_counter()
    g = getattr(ec, name)()
    load_s = time.perf_counter() - t
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        p = g.plot(seed=1, show_labels=labels)
        html = p._repr_html_()
        plot_s = time.perf_counter() - t
        t = time.perf_counter()
        png = p._repr_png_()
        png_s = time.perf_counter() - t
    data = png[0] if isinstance(png, tuple) else png
    raw = html.encode()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "graph": name,
        "labels": labels,
        "n": g.count_nodes(),
        "m": g.count_edges(),
        "backend": "png" if "<image" in html else "svg",
        "load_s": round(load_s, 2),
        "plot_s": round(plot_s, 2),
        "png_s": round(png_s, 2),
        "html_mb": round(len(raw) / 1e6, 3),
        "html_gz_mb": round(len(gzip.compress(raw, 6)) / 1e6, 3),
        "elements": html.count("<") - html.count("</"),
        "png_kb": round(len(data) / 1e3),
        "rss_mb": round(peak / 1e6 if sys.platform == "darwin" else peak / 1e3),
        "warnings": [str(w.message)[:80] for w in caught],
    }


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], sys.argv[3] == "1")))
        sys.exit(0)
    out = open(sys.argv[1], "w")
    for labels in (False, True):
        for name in DATASETS:
            try:
                r = subprocess.run([sys.executable, __file__, "--one", name, str(int(labels))],
                                   capture_output=True, text=True, timeout=900)
                line = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else json.dumps(
                    {"graph": name, "labels": labels, "error": r.stderr.strip().splitlines()[-1:]})
            except subprocess.TimeoutExpired:
                line = json.dumps({"graph": name, "labels": labels, "error": "timeout"})
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
