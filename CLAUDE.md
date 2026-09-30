# Working rules

Rules for working in this repository (followed by both me and the coding assistant).
Competition details: [docs/competition.md](docs/competition.md).

## Never

- Commit anything under `data/` (redistribution terms and size).
- Commit credentials (`~/.kaggle/credentials.json`, `access_token`, `kaggle.json`).
  The pre-commit hook in `scripts/hooks/` blocks them; enable it with
  `git config core.hooksPath scripts/hooks`.
- Reimplement the metric. Use the official scorer in `third_party/tracking-cellmot`.
- Submit a notebook with the built-in validator or sweep on. It fits the public test
  but can exceed 12 hours on the hidden test (this happened with Geometric Fusion).

## Submission notebooks

- No internet: no `pip install` from PyPI, no downloads. Dependencies come as wheels
  in an attached Kaggle dataset.
- Must finish in 12 hours on a hidden test about the size of train; design for 8–9.
- Output is `submission.csv`.
- Comments, docstrings and prints in notebooks are in English.

## Experiments

- All computation runs in Kaggle notebooks, not locally.
- A variant is an entry in `experiments/configs/batchNN.json` (base notebook + `BIOHUB_*`
  overrides), launched with `scripts/batch.py launch`. Submissions go only through
  `scripts/cycle.py`.
- Limits: 5 submissions per UTC day, never the same slug twice, no new submissions after
  2026-09-28 15:00 UTC. Check GPU with `scripts/gpu.py`.
- Single-parameter tweaks have been refuted many times on this pipeline; prefer changes
  that improve edges without changing the node count.
- Do not change the final selection without asking.
- Record every result in `submissions/manifest.csv` and a short note in `docs/journal.md`.
  Then rerun `python scripts/plot_results.py` so the README chart stays current.

## Data pitfalls

- Voxels are anisotropic: z = 1.625 µm, y = x = 0.40625 µm. Scale before any distance.
- Train has only two embryos; CV is stratified by embryo and scores are reported per embryo.
- A sample is ~840 MB uncompressed; read one time point at a time.
