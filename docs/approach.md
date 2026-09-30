# Approach

## Starting point: fork the strongest public pipeline

With 9 days left and 30 GPU hours a week, training a new model was not realistic. The
public notebooks had converged on one stack:

```
detection   TemporalUNet3D (3D U-Net with temporal context), TTA, two seeds fused
candidates  distance gate (~10 µm) + top-k parents
linking     ILP over edge / appearance / disappearance / division costs (tracksdata + SCIP)
post-proc   gap closing, Hungarian motion relink, safe divisions, short-track filter,
            DeepCenter veto (a second 3D U-Net that confirms synthetic nodes)
```

I forked the best one, Harmonic Fusion (public 0.947), unchanged, and confirmed it
reproduced 0.947 and finished the hidden test inside the 12-hour limit. Dependencies come
as wheels inside a Kaggle dataset, installed offline with `pip install --no-index`.

Every setting of this notebook is read from `BIOHUB_*` environment variables, so
experiments are just environment overrides on top of a fixed notebook.

## Measuring before tuning

The official scorer is published and never reads the images, so scoring can run on CPU.
The plan was: cache the raw detection + ILP graphs once on GPU, then sweep post-processing
settings cheaply.

- **CV split** ([harness/splits.py](../harness/splits.py)): train has only two embryos,
  so an embryo-disjoint split would be 2-fold. Instead: stratify by embryo, split by sample
  (tune 48 / confirm 32), and always report per-embryo scores to catch settings that only
  help one embryo.
- **Why 48 samples:** 8 samples held only 12 true divisions, so one event moved division
  Jaccard by 0.08.
- **CPU sweeps failed on speed:** the DeepCenter veto is a 3D U-Net and took 7.2 h per
  configuration on CPU (46 min on GPU). The cache idea moved to GPU instead.

## Divisions looked like the opportunity, and were not

On 8 samples the baseline had edge Jaccard 0.926 but division Jaccard only 0.231
(3 TP / 1 FP / 9 FN), and division counts 0.1× in the score. Reading the official
`division_metrics.py`:

```python
fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks
```

Extra forks near a true division (`considered`) are always false positives, and the
bipartite matching gives each true division one predicted fork, so crowding the area
can knock out a correct one. Sweeps confirmed it: tightening the division gates lost true
positives, loosening them added false positives. The defaults were already at the optimum.

A second lesson came from the same sweep: the forked notebook's first cell **reassigned
about 47 settings unconditionally**, so my early overrides were silently ignored.
`scripts/batch.py` now rewrites those assignments into `setdefault` and updates the
notebook's config-drift guard for each override.

## What the leaderboard taught

The offline proxy (48 samples) turned out not to predict the LB for changes this small,
so I switched to probing the LB directly: 5 variants a day, each a light run
(validator and sweep off, ~26 min of GPU).

| Batch | Idea | LB |
|---|---|---|
| baseline | Harmonic Fusion, sweep off | 0.946 |
| b01 | division divergence 1.0 / 3.0 / 4.5, relink radius 5.5 | 0.941–0.946 |
| b02 | detection threshold ±0.005, ILP division weight | 0.945–0.946 |
| b03 | relink learned-bonus ×2.5 / ×5, relink off | 0.945–0.946 |
| b04 | Geometric Fusion's selected combination, light rebuild | 0.947 |

(v0001, the untouched fork with its built-in sweep, scored 0.947.)

- **Single parameters do not move this plateau.** Every one tied or lost. A public
  Discussion (Justin CH123) reported the same for 10 more, and observed that **the loss
  tracked the change in predicted node count**, not offline edge scores; a change of four
  nodes still cost 0.001.
- **The relink stage is not where the points are.** It replaces the ILP edges, so it looked
  like the place to inject edge probabilities without touching node count. Turning it off
  entirely left the score unchanged.
- **The 0.948 public notebook did not reproduce.** Its identical output scored 0.947 for me,
  most likely a rounding boundary.

The remaining experiments (b05, b06) split Geometric Fusion's combination into its
edge-only part (relink radius and velocity weight: node count unchanged) and its
node-removing part (leaf pruning), on both pipelines.

## Runtime is part of the design

The hidden test is about 50 times larger than the 4 public test samples. The unmodified
Geometric Fusion notebook took 4 h on the public test, 3.4 h of which was a fixed-cost
validator and sweep, and it **timed out** on the hidden test. A light rebuild with the
sweep's choice fixed produced byte-identical output in 26 minutes. Every submission now
uses the light form, and a scored submission is safe to select because its public and
private scores come from the same completed run.

## Final selection

Two 0.947 submissions from different pipelines: v0001 (Harmonic Fusion) and b04-glight
(Geometric Fusion). With 71% of the test still private and the top of the board this
compressed, two different pipelines at the same public score hedge better than two
variants of one.
