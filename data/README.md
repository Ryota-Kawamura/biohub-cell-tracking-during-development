# data/

The competition data is not committed (redistribution terms, and about 86 GB
for train). Kaggle notebooks see it mounted at
`/kaggle/input/biohub-cell-tracking-during-development/`, so this project never
needs a local copy.

To inspect a single file locally:

```bash
kaggle competitions files biohub-cell-tracking-during-development
kaggle competitions download biohub-cell-tracking-during-development -f <path> -p data/
```

Layout once downloaded:

```
data/
├── train/   # {embryo}_{fov}.zarr (images) + {embryo}_{fov}.geff (ground truth)
├── test/    # .zarr only; replaced by the hidden test set when a submission is scored
└── sample_submission.csv
```
