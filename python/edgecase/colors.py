# Sequential color schemes (single-hue, multi-hue)
# Good for continuous data, ordered data
SEQUENTIAL_PALETTES = {
    # Blues
    "Blues": [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ],
    "BuGn": [
        "#f7fcfd",
        "#e5f5f9",
        "#ccece6",
        "#99d8c9",
        "#66c2a4",
        "#41ae76",
        "#238b45",
        "#006d2c",
        "#00441b",
    ],
    "BuPu": [
        "#f7fcfd",
        "#e0ecf4",
        "#bfd3e6",
        "#9ebcda",
        "#8c96c6",
        "#8c6bb1",
        "#88419d",
        "#810f7c",
        "#4d004b",
    ],
    "GnBu": [
        "#f7fcf0",
        "#e0f3db",
        "#ccebc5",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#0868ac",
        "#084081",
    ],
    "Greens": [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",
    ],
    "Greys": [
        "#ffffff",
        "#f0f0f0",
        "#d9d9d9",
        "#bdbdbd",
        "#969696",
        "#737373",
        "#525252",
        "#252525",
        "#000000",
    ],
    "Oranges": [
        "#fff5eb",
        "#fee6ce",
        "#fdd0a2",
        "#fdae6b",
        "#fd8d3c",
        "#f16913",
        "#d94801",
        "#a63603",
        "#7f2704",
    ],
    "OrRd": [
        "#fff7ec",
        "#fee8c8",
        "#fdd49e",
        "#fdbb84",
        "#fc8d59",
        "#ef6548",
        "#d7301f",
        "#b30000",
        "#7f0000",
    ],
    "PuBu": [
        "#fff7fb",
        "#ece7f2",
        "#d0d1e6",
        "#a6bddb",
        "#74a9cf",
        "#3690c0",
        "#0570b0",
        "#045a8d",
        "#023858",
    ],
    "PuBuGn": [
        "#fff7fb",
        "#ece2f0",
        "#d0d1e6",
        "#a6bddb",
        "#67a9cf",
        "#3690c0",
        "#02818a",
        "#016c59",
        "#014636",
    ],
    "PuRd": [
        "#f7f4f9",
        "#e7e1ef",
        "#d4b9da",
        "#c994c7",
        "#df65b0",
        "#e7298a",
        "#ce1256",
        "#980043",
        "#67001f",
    ],
    "Purples": [
        "#fcfbfd",
        "#efedf5",
        "#dadaeb",
        "#bcbddc",
        "#9e9ac8",
        "#807dba",
        "#6a51a3",
        "#54278f",
        "#3f007d",
    ],
    "RdPu": [
        "#fff7f3",
        "#fde0dd",
        "#fcc5c0",
        "#fa9fb5",
        "#f768a1",
        "#dd3497",
        "#ae017e",
        "#7a0177",
        "#49006a",
    ],
    "Reds": [
        "#fff5f0",
        "#fee0d2",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#a50f15",
        "#67000d",
    ],
    "YlGn": [
        "#ffffe5",
        "#f7fcb9",
        "#d9f0a3",
        "#addd8e",
        "#78c679",
        "#41ab5d",
        "#238443",
        "#006837",
        "#004529",
    ],
    "YlGnBu": [
        "#ffffd9",
        "#edf8b1",
        "#c7e9b4",
        "#7fcdbb",
        "#41b6c4",
        "#1d91c0",
        "#225ea8",
        "#253494",
        "#081d58",
    ],
    "YlOrBr": [
        "#ffffe5",
        "#fff7bc",
        "#fee391",
        "#fec44f",
        "#fe9929",
        "#ec7014",
        "#cc4c02",
        "#993404",
        "#662506",
    ],
    "YlOrRd": [
        "#ffffcc",
        "#ffeda0",
        "#fed976",
        "#feb24c",
        "#fd8d3c",
        "#fc4e2a",
        "#e31a1c",
        "#bd0026",
        "#800026",
    ],
}

# Diverging color schemes
# Good for data with a meaningful midpoint (e.g., positive/negative)
DIVERGING_PALETTES = {
    "BrBG": [
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#f5f5f5",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
    ],
    "PiYG": [
        "#c51b7d",
        "#de77ae",
        "#f1b6da",
        "#fde0ef",
        "#f7f7f7",
        "#e6f5d0",
        "#b8e186",
        "#7fbc41",
        "#4d9221",
    ],
    "PRGn": [
        "#762a83",
        "#9970ab",
        "#c2a5cf",
        "#e7d4e8",
        "#f7f7f7",
        "#d9f0d3",
        "#a6dba0",
        "#5aae61",
        "#1b7837",
    ],
    "PuOr": [
        "#b35806",
        "#e08214",
        "#fdb863",
        "#fee0b6",
        "#f7f7f7",
        "#d8daeb",
        "#b2abd2",
        "#8073ac",
        "#542788",
    ],
    "RdBu": [
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
    ],
    "RdGy": [
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#ffffff",
        "#e0e0e0",
        "#bababa",
        "#878787",
        "#4d4d4d",
    ],
    "RdYlBu": [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#ffffbf",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
    ],
    "RdYlGn": [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#d9ef8b",
        "#a6d96a",
        "#66bd63",
        "#1a9850",
    ],
    "Spectral": [
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
    ],
}

# Qualitative color schemes
# Good for categorical data with no inherent order
QUALITATIVE_PALETTES = {
    "Accent": [
        "#7fc97f",
        "#beaed4",
        "#fdc086",
        "#ffff99",
        "#386cb0",
        "#f0027f",
        "#bf5b17",
        "#666666",
    ],
    "Dark2": [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ],
    "Paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ],
    "Pastel1": [
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
    ],
    "Pastel2": [
        "#b3e2cd",
        "#fdcdac",
        "#cbd5e8",
        "#f4cae4",
        "#e6f5c9",
        "#fff2ae",
        "#f1e2cc",
        "#cccccc",
    ],
    "Set1": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ],
    "Set2": [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ],
    "Set3": [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    ],
}

# Viridis color scales (perceptually uniform, colorblind-friendly)
VIRIDIS_PALETTES = {
    "viridis": [
        "#440154",
        "#482777",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ],
    "magma": [
        "#000004",
        "#180f3d",
        "#451077",
        "#721f81",
        "#9e2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9668",
        "#feca8d",
        "#fcfdbf",
    ],
    "inferno": [
        "#000004",
        "#1b0c41",
        "#4a0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fb9b06",
        "#f7d13d",
        "#fcffa4",
    ],
    "plasma": [
        "#0d0887",
        "#47039f",
        "#7301a8",
        "#9c179e",
        "#bd3786",
        "#d8576b",
        "#ed7953",
        "#fb9f3a",
        "#fdca26",
        "#f0f921",
    ],
    "cividis": [
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a69c75",
        "#c4b56c",
        "#e1ce55",
        "#fde725",
    ],
}

# Legacy palettes from widget.py for backwards compatibility
LEGACY_PALETTES = {
    "default": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "vibrant": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
    ],
    "pastel": [
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
        "#8dd3c7",
    ],
    "dark": [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#8da0cb",
        "#fc8d62",
    ],
    "tableau": [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
    ],
}

# Combined palette dictionary for easy lookup
COLOR_PALETTES = {
    **SEQUENTIAL_PALETTES,
    **DIVERGING_PALETTES,
    **QUALITATIVE_PALETTES,
    **VIRIDIS_PALETTES,
    **LEGACY_PALETTES,
}

# Palette categories for documentation
PALETTE_CATEGORIES = {
    "sequential": list(SEQUENTIAL_PALETTES.keys()),
    "diverging": list(DIVERGING_PALETTES.keys()),
    "qualitative": list(QUALITATIVE_PALETTES.keys()),
    "viridis": list(VIRIDIS_PALETTES.keys()),
    "legacy": list(LEGACY_PALETTES.keys()),
}


def get_palette(name: str) -> list[str]:
    """Get a color palette by name.

    Parameters
    ----------
    name : str
        Palette name (e.g., 'Greens', 'RdBu', 'viridis', 'Set1')

    Returns
    -------
    list[str]
        List of hex color strings

    Raises
    ------
    ValueError
        If palette name not found
    """
    if name in COLOR_PALETTES:
        return COLOR_PALETTES[name]
    raise ValueError(
        f"Unknown palette '{name}'. Available palettes: {', '.join(sorted(COLOR_PALETTES.keys()))}"
    )


def list_palettes(category: str = None) -> list[str]:
    """List available color palettes, optionally filtered by category.

    Parameters
    ----------
    category : str, optional
        Palette category ('sequential', 'diverging', 'qualitative', 'viridis', 'legacy')
        If None, returns all palettes

    Returns
    -------
    list[str]
        Sorted list of palette names
    """
    if category is None:
        return sorted(COLOR_PALETTES.keys())
    if category in PALETTE_CATEGORIES:
        return sorted(PALETTE_CATEGORIES[category])
    raise ValueError(
        f"Unknown category '{category}'. Available categories: {', '.join(PALETTE_CATEGORIES.keys())}"
    )
