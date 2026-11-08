# Downloading Small Datasets for Visualization Testing

This guide helps you download small to medium-sized datasets (100-4000 nodes) for testing graph visualization.

## Datasets Overview

| Dataset      | Nodes  | Edges  | Type       | Source            |
| ------------ | ------ | ------ | ---------- | ----------------- |
| dolphins     | 62     | 159    | Undirected | NetworkRepository |
| polbooks     | 105    | 441    | Undirected | NetworkRepository |
| football     | 115    | 613    | Undirected | NetworkRepository |
| jazz         | 198    | 2,742  | Undirected | NetworkRepository |
| celegans     | 297    | 2,148  | Directed   | NetworkRepository |
| power_grid   | 4,941  | 6,594  | Undirected | NetworkRepository |
| erdos_collab | 6,927  | 11,850 | Undirected | SNAP              |
| pgp          | 10,680 | 24,316 | Directed   | NetworkRepository |
| internet_as  | 22,963 | 48,436 | Undirected | SNAP              |

## Download Instructions

### Option 1: Using NetworkRepository (Recommended)

```bash
cd python/edgecase/data/

# Dolphins
curl -O http://networkrepository.com/soc-dolphins.mtx
mv soc-dolphins.mtx dolphins.mtx
gzip -9 dolphins.mtx

# Political Books
curl -O http://networkrepository.com/polbooks.mtx
gzip -9 polbooks.mtx

# College Football
curl -O http://networkrepository.com/fb-pages-food.mtx  # or football.mtx
mv fb-pages-food.mtx football.mtx
gzip -9 football.mtx

# Jazz
curl -O http://networkrepository.com/bn-jazz.mtx
mv bn-jazz.mtx jazz.mtx
gzip -9 jazz.mtx

# C. Elegans
curl -O http://networkrepository.com/bn-celegans.mtx
mv bn-celegans.mtx celegans.mtx
gzip -9 celegans.mtx

# Power Grid
curl -O http://networkrepository.com/power-1138-bus.mtx
mv power-1138-bus.mtx power_grid.mtx
gzip -9 power_grid.mtx

# Erdős Collaboration
curl -O http://networkrepository.com/ca-erdos-992.mtx
mv ca-erdos-992.mtx erdos_collab.mtx
gzip -9 erdos_collab.mtx

# PGP Trust Network
curl -O http://networkrepository.com/tech-pgp.mtx
mv tech-pgp.mtx pgp.mtx
gzip -9 pgp.mtx

# Internet AS
curl -O http://networkrepository.com/tech-as-topology.mtx
mv tech-as-topology.mtx internet_as.mtx
gzip -9 internet_as.mtx
```

### Option 2: Using SNAP Datasets

Some datasets are available from SNAP:

```bash
# If the above don't work, try SNAP:
cd python/edgecase/data/

# Download and convert to Matrix Market format
# (SNAP uses edge list format, need conversion)
```

### Option 3: Alternative Sources

If NetworkRepository links are broken, try:

- **KONECT**: http://konect.cc/
- **Mark Newman's website**: http://www-personal.umich.edu/~mejn/netdata/
- **UCI Network Data Repository**: https://networkdata.ics.uci.edu/

## Verification

After downloading, verify file sizes:

```bash
cd python/edgecase/data/
ls -lh *.mtx.gz

# Expected compressed sizes:
# dolphins.mtx.gz       ~1 KB
# polbooks.mtx.gz       ~2 KB
# football.mtx.gz       ~2 KB
# jazz.mtx.gz           ~8 KB
# celegans.mtx.gz       ~7 KB
# power_grid.mtx.gz     ~25 KB
# erdos_collab.mtx.gz   ~40 KB
# pgp.mtx.gz            ~75 KB
# internet_as.mtx.gz    ~150 KB
```

## Usage

Once downloaded and compressed:

```python
from edgecase import (
    dolphins, football, polbooks, jazz, celegans,
    power_grid, erdos_collab, pgp, internet_as
)

# Load for visualization testing - small scale (< 300 nodes)
g1 = dolphins()      # 62 nodes - tiny
g2 = polbooks()      # 105 nodes - small
g3 = football()      # 115 nodes - small
g4 = jazz()          # 198 nodes - small-medium
g5 = celegans()      # 297 nodes - medium

# Medium scale (1000-5000 nodes)
g6 = power_grid()    # 4,941 nodes
g7 = erdos_collab()  # 6,927 nodes

# Larger medium scale (10k-25k nodes)
g8 = pgp()           # 10,680 nodes
g9 = internet_as()   # 22,963 nodes

# All load via generic interface too
g = load_dataset("power_grid")
```

## Notes

- All datasets are compressed with gzip -9 for minimal package size
- Total compressed size: ~310 KB (excellent for bundling)
- These complement karate (34 nodes) to provide comprehensive visualization test coverage
- Range: 34 → 62 → 105 → 115 → 198 → 297 → 4941 → 6927 → 10680 → 22963 nodes
