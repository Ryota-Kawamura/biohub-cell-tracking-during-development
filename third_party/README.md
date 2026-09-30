# third_party/

External code under redistributable licenses, pinned to a specific commit.
**Do not edit these files.** To update, re-import and change the commit hash below.

## tracking-cellmot (official scorer and baseline)

- Source: https://github.com/royerlab/kaggle-cell-tracking-competition
- License: **BSD-3-Clause** (`tracking-cellmot/LICENSE`)
- Commit: `075fc5f5a52d11077f9dc2b074644618f26939e2` ("Merge pull request #2 from royerlab/metrics-fix")
- Imported: 2026-09-20

The organisers' scorer, identical to the one used on the leaderboard, including the
July 2026 patch to the division metric.

| Path | Contents |
|---|---|
| `src/tracking_cellmot/metrics.py` | edge Jaccard and aggregation |
| `src/tracking_cellmot/division_metrics.py` | division Jaccard (patched) |
| `src/tracking_cellmot/io.py` | Zarr / geff I/O |
| `src/tracking_cellmot/models/` | TemporalUNet3D / SimpleNodeTransformer |
| `scripts/evaluate.py` | scores predicted .geff against ground-truth .geff |
| `scripts/csv_to_geffs.py` | submission.csv → .geff |
| `scripts/geffs_to_csv.py` | .geff → submission.csv |
| `metrics.md` | official description of the metric |

Scoring never reads the images (ground truth is .geff, voxel scale comes from the .zarr
metadata), so it runs on CPU without using GPU quota.
