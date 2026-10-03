"""Enrichment and network-biology methods: annotated gene networks.

Template: `compute`. Left: a small gene network (genes `G1` … `G8`) with the
annotation groups as node colors (gene set X `g2`, gene set Y `g3`,
unannotated genes plain ink) and the tested module behind a soft hull.
Right: a card or table with the counts, fold enrichment and p-value
(or the method's own numbers), every value what edgecase returns.

Shared layouts so siblings can be compared directly:

- `NET`: two four-gene modules, G1–G4 and G5–G8, joined by the bridge
  G4–G5. Gene set X = {G1, G2, G3}, Y = {G6, G7, G8}; the bridge genes G4 and
  G5 are unannotated. Used by `subgraph_enrichment`,
  `module_annotation_enrichment`, `network_proximity`, `gwas_pipeline` and
  (with edge weights) `soft_threshold_power`. Louvain numbers the modules
  1 (G1–G4) and 0 (G5–G8).
- `topological_overlap` uses the left module plus G5 (the same positions), so
  its 5 × 5 matrix stays small enough to read.
- `differential_edges` compares two conditions, so it uses `binary` instead:
  the same five genes in `g` and `other`, edges only in `g` (removed) in
  `accent`, edges only in `other` (added) in `g2`, as in the set-ops family.
"""

from core import Scene, binary, compute, figure, place, ring

UNIT = 80  # one density for the whole family

NET = {"G1": (0, 0.65), "G2": (0.8, 0), "G3": (0.8, 1.3), "G4": (1.6, 0.65),
       "G5": (2.6, 0.65), "G6": (3.4, 0), "G7": (3.4, 1.3), "G8": (4.2, 0.65)}
NET_EDGES = [("G1", "G2"), ("G1", "G3"), ("G2", "G3"), ("G2", "G4"), ("G3", "G4"),
             ("G4", "G5"), ("G5", "G6"), ("G5", "G7"), ("G6", "G7"), ("G6", "G8"),
             ("G7", "G8")]
MOD_1 = ["G1", "G2", "G3", "G4"]  # Louvain community 1
MOD_0 = ["G5", "G6", "G7", "G8"]  # Louvain community 0

# gene-set annotation: node attribute "set"
SETS = {"G1": "g2", "G2": "g2", "G3": "g2", "G6": "g3", "G7": "g3", "G8": "g3"}


def _net(s: Scene, box, *, styles=None, edges=None, **kw) -> None:
    """The shared gene network; nodes colored by gene set unless `styles` says otherwise."""
    s.graph(place(NET, box, unit=UNIT), NET_EDGES if edges is None else edges,
            styles=SETS if styles is None else styles, **kw)


# --- enrichment tests -------------------------------------------------------


@figure("subgraph_enrichment", "enrichment", "Subgraph enrichment",
        "An eight-gene network of two modules joined by the bridge G4 to G5; genes G1, G2, G3 "
        "belong to gene set X (teal) and G6, G7, G8 to set Y (violet). The tested module G1 to "
        "G4 sits in a tinted region and holds 3 of its 4 genes in set X, which has only 3 "
        "members among all 8 genes. The card gives k = 3 of n = 4, K = 3 of N = 8, fold "
        "enrichment 2.0 and hypergeometric p-value 0.071.")
def subgraph_enrichment(s: Scene) -> None:
    left, right = compute(s, "g", 'g.subgraph_enrichment(m, "set", "X")')
    _net(s, left)
    s.hull(MOD_1, "accent")
    s.card_centered(right, ["module  k = 3 of n = 4", "graph   K = 3 of N = 8",
                            ("fold    2.0", "accent"), ("p       0.071", "accent")],
                    title="set = X")


@figure("module_annotation_enrichment", "enrichment", "Module annotation enrichment",
        "The eight-gene network split into its two Louvain modules, G1 to G4 (module 1) and "
        "G5 to G8 (module 0), each in a tinted region, with gene set X in teal and set Y in "
        "violet. Every module is tested against every set: module 1 holds all 3 X genes and "
        "module 0 all 3 Y genes, each a 2.0-fold enrichment with Bonferroni-adjusted p-value "
        "0.29 over the four tests, while the other two pairs have fold 0 and p 1.")
def module_annotation_enrichment(s: Scene) -> None:
    left, right = compute(s, "g", 'g.module_annotation_enrichment(c, "set")')
    _net(s, left)
    s.hull(MOD_1, "accent")
    s.hull(MOD_0, "accent")
    rows = [["module", "set", "k / n", "fold", "p adj"],
            ["1", "X", "3 / 4", "2.0", "0.29"],
            ["1", "Y", "0 / 4", "0.0", "1.00"],
            ["0", "X", "0 / 4", "0.0", "1.00"],
            ["0", "Y", "3 / 4", "2.0", "0.29"]]
    s.table_centered(right, rows, [50, 34, 50, 40, 44],
                     highlight={**{(1, j): "g2" for j in range(5)},
                                **{(4, j): "g3" for j in range(5)}})


# --- distances between gene sets ----------------------------------------------


@figure("network_proximity", "enrichment", "Network proximity",
        "The eight-gene network with gene set A = G1, G2 in teal and set B = G6, G8 in violet. "
        "The closest pair is G2 and G6, three steps apart through the bridge G4 to G5, drawn "
        "in the accent color. The card reports that all 4 cross-set pairs are reachable and "
        "the closest-pair proximity is 3.")
def network_proximity(s: Scene) -> None:
    left, right = compute(s, "g", "g.network_proximity(a, b)")
    path = {("G2", "G4"), ("G4", "G5"), ("G5", "G6")}
    _net(s, left, styles={"G1": "g2", "G2": "g2", "G6": "g3", "G8": "g3"},
         edges=[(a, b, {"style": "accent"} if (a, b) in path else {}) for a, b in NET_EDGES])
    s.card_centered(right, ["min d(a, b) = d(G2, G6)", "reachable  4 of 4 pairs",
                            ("proximity  3.0", "accent")], title="method = closest")


# --- pipeline -----------------------------------------------------------------


@figure("gwas_pipeline", "enrichment", "GWAS network pipeline",
        "The eight-gene network after the pipeline: raw scores of 4.0 on G1, G2, G3 and 0.5 "
        "elsewhere are smoothed over the network, so every gene is shaded by its smoothed "
        "score and the signal leaks onto G4. Louvain then splits the network into module 1, "
        "G1 to G4, and module 0, G5 to G8, drawn as gray regions. The table "
        "ranks the modules by mean smoothed score, 3.05 for module 1 and 0.57 for module 0.")
def gwas_pipeline(s: Scene) -> None:
    left, right = compute(s, "g", "g.gwas_pipeline(scores)")
    smoothed = {"G1": 3.23, "G2": 3.69, "G3": 3.69, "G4": 1.6, "G5": 0.71, "G6": 0.57,
                "G7": 0.57, "G8": 0.44}
    top = max(smoothed.values())
    _net(s, left, styles={}, heat={k: v / top for k, v in smoothed.items()})
    # neutral hulls: an accent region would wash out the heat shading
    s.hull(MOD_1, "ink", opacity=0.06)
    s.hull(MOD_0, "ink", opacity=0.06)
    rows = [["module", "genes", "mean"], ["1", "G1–G4", "3.05"], ["0", "G5–G8", "0.57"]]
    s.table_centered(right, rows, [56, 64, 52],
                     highlight={(1, j): "accent" for j in range(3)})


# --- WGCNA ----------------------------------------------------------------------

# edge weights (co-expression strengths) in NET_EDGES order
WEIGHTS = [0.4, 0.2, 0.8, 0.5, 0.8, 0.5, 0.2, 0.3, 0.3, 0.2, 0.2]


@figure("soft_threshold_power", "enrichment", "Soft-threshold power",
        "The eight-gene network with a co-expression weight on every edge. Each candidate "
        "power raises the weights to that power and fits a scale-free line to the degree "
        "distribution; the table lists the signed fit R squared and mean connectivity for "
        "powers 1 to 6, rising from -0.35 to 0.93. Power 4 is the first with R squared at "
        "least 0.8 (0.90), so it is the recommended power, highlighted in the accent color.")
def soft_threshold_power(s: Scene) -> None:
    left, right = compute(s, "g", "g.soft_threshold_power(return_dict=True)")
    _net(s, left, styles={},
         edges=[(a, b, {"weight": w}) for (a, b), w in zip(NET_EDGES, WEIGHTS)])
    rows = [["power", "R²", "mean k"], ["1", "-0.35", "1.10"], ["2", "0.10", "0.57"],
            ["3", "0.67", "0.36"], ["4", "0.90", "0.25"], ["5", "0.93", "0.18"],
            ["6", "0.93", "0.14"], ["…", "…", "…"]]
    s.table_centered(right, rows, [56, 56, 64],
                     highlight={(4, j): "accent" for j in range(3)})


TOM = {k: NET[k] for k in ["G1", "G2", "G3", "G4", "G5"]}
TOM_EDGES = NET_EDGES[:6]


@figure("topological_overlap", "enrichment", "Topological overlap matrix",
        "A five-gene network: a diamond G1 to G4 with G5 hanging off G4. G1 and G4 are not "
        "linked but share the two neighbors G2 and G3, drawn in the accent color, so their "
        "topological overlap is 2 over min(2, 3) + 1, that is 0.67. The table is the full "
        "symmetric matrix with 1 on the diagonal and the G1, G4 entries highlighted.")
def topological_overlap(s: Scene) -> None:
    left, right = compute(s, "g", "g.topological_overlap(return_dict=True)")
    wit = {("G1", "G2"), ("G1", "G3"), ("G2", "G4"), ("G3", "G4")}
    s.graph(place(TOM, left, unit=UNIT),
            [(a, b, {"style": "accent"} if (a, b) in wit else {}) for a, b in TOM_EDGES],
            styles={"G1": "soft", "G4": "soft", "G2": "accent", "G3": "accent"})
    names = list(TOM)
    tom = [[1.0, 1.0, 1.0, 0.67, 0.0], [1.0, 1.0, 1.0, 0.67, 0.5], [1.0, 1.0, 1.0, 0.67, 0.5],
           [0.67, 0.67, 0.67, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0, 1.0]]
    rows = [[""] + names] + [[a] + [f"{v:.2f}".rstrip("0").rstrip(".") if v in (0, 1) else
                                    f"{v:.2f}" for v in r] for a, r in zip(names, tom)]
    s.table_centered(right, rows, [34] + [40] * 5,
                     highlight={(1, 4): "accent", (4, 1): "accent"})


# --- two conditions -------------------------------------------------------------

DIFF = ring(["G1", "G2", "G3", "G4", "G5"], radius=0.85)
DIFF_SHARED = [("G1", "G2"), ("G2", "G3"), ("G3", "G4"), ("G5", "G1"), ("G1", "G3")]
DIFF_REMOVED = ("G4", "G5")  # only in g
DIFF_ADDED = ("G3", "G5")  # only in other


@figure("differential_edges", "enrichment", "Differential edges",
        "The same five genes in two conditions. g has a five-gene ring with the chord G1 to "
        "G3; in other, the edge G4 to G5 (accent) is gone, drawn dashed, and a new edge G3 to "
        "G5 (teal) appears. The card lists G3, G5 as added and G4, G5 as removed, in the same "
        "colors.")
def differential_edges(s: Scene) -> None:
    a, b, c = binary(s, "vs", ("g", "other", "g.differential_edges(other)"))
    s.graph(place(DIFF, a, unit=UNIT), DIFF_SHARED + [(*DIFF_REMOVED, {"style": "accent"})],
            prefix="g.")
    s.graph(place(DIFF, b, unit=UNIT),
            DIFF_SHARED + [(*DIFF_REMOVED, {"style": "removed"}), (*DIFF_ADDED, {"style": "g2"})],
            prefix="h.")
    s.card_centered(c, [("added   [(G3, G5)]", "g2"), ("removed [(G4, G5)]", "accent")])
