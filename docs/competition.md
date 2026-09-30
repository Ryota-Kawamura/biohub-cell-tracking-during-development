# The competition

Source: the [competition page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development), as of Sep 2026.

## Task

From 3D + time fluorescence microscopy of zebrafish embryos:

1. **detect** every cell at every time point (nodes),
2. **link** cells between consecutive time points (edges),
3. **identify divisions** (a node with two children), reconstructing lineages.

## Timeline (UTC)

| | |
|---|---|
| Start | 2026-06-29 |
| Entry and team merger deadline | 2026-09-22 23:59 |
| **Final submission deadline** | **2026-09-29 23:59** |

Code competition: submissions are notebooks, no internet, at most 12 hours on
CPU or GPU, output `submission.csv`. Five submissions per day; two can be selected
for the final ranking.

## Data

- One sample is a short video stored as **Zarr v3**, array `(T, Z, Y, X)`, typically
  `(100, 64, 256, 256)` uint16, one chunk per time point (~840 MB uncompressed).
- Voxel size **z = 1.625 µm, y = x = 0.40625 µm** (anisotropic).
- Ground truth (train only) is **.geff**, a Zarr-based graph format: node ids, integer
  voxel centroids, and `(source, target)` edges.
- **Annotations are sparse**: not every cell is labelled.
- Train has 199 samples from only two embryos (`44b6`, `6bba`, 86 GB). The test set is
  embryo-disjoint, and the hidden test set is about the same size as train.

## Metric

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

- Predicted and true nodes are matched per frame by optimal bipartite matching on
  scaled centroid distance, **up to 7 µm**.
- A predicted edge is a true positive only if both ends match true nodes that are
  linked in the ground truth. Over-predicting the node count is penalised ("adjusted").
- A division is a node with out-degree ≥ 2. A July 2026 exploit (synthetic far-away
  forks) was patched and all submissions rescored; far-away forks now count as false positives.
- The score can exceed 1.0.

## Submission format

One CSV mixing node rows and edge rows, grouped by dataset:

```csv
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
2,6bba,edge,-1,-1,-1,-1,-1,1,2
```

## Leaderboard shape

The public LB uses 29% of the test set. It was saturated when I joined: the top 100 teams
sat between 0.974 and 0.950, and about 390 teams were tied at exactly 0.947 (the score of
the best public notebook). Ties are ordered by submission time, so a single +0.001 was
worth hundreds of places.
