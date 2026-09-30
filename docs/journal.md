# Journal

Short, dated notes on what was done and learned. Times are UTC.

## 2026-09-20 — Set-up and baseline

- Joined 9 days before the deadline. All computation runs in Kaggle notebooks; locally
  there is only code and git.
- Surveyed public notebooks. The best public line was Harmonic Fusion (0.947, Apache 2.0);
  forked it as `notebooks/infer`, unchanged except for one settings cell.
- Vendored the official scorer (`third_party/tracking-cellmot`). It never reads images,
  so scoring can run on CPU.
- v0001 (the fork as-is): 1 h 55 min on GPU T4 x2, of which only 9 minutes was test
  prediction; most was the built-in validator and parameter sweep.
- Submission was blocked until **identity verification** (Persona) was done; phone
  verification alone is not enough. Worth doing on day one of any competition.
- `kaggle kernels push` always runs the notebook. Wrote `scripts/push_kernel.py` to use
  the API's QUICK_SAVE instead, so edits do not cost GPU time.
- Two background pollers refreshing the same OAuth token invalidated each other. Moved to
  an API token and one poller, and made the poll loop stop only on explicit
  COMPLETE / ERROR states.
- sweep01 on 8 samples: loosening or tightening the division gates both lost.
  **Found that the fork's first cell reassigns ~47 settings unconditionally**, so the
  sweep had silently used defaults; fixed by rewriting them to `setdefault`.

## 2026-09-21 — The leaderboard is a plateau

- v0001 scored **0.947**, the same as its source. Around 390 teams were tied at exactly
  0.947 and ties are ordered by submission time, so +0.001 would move hundreds of places.
- sweep02 cached raw graphs for 48 samples. On 48 samples the proxy level differs a lot
  from 8 samples (0.907 vs 0.949), and division FPs appear that 8 samples hid.
- The CPU port of the sweep worked functionally but took 7.2 h per configuration
  (the DeepCenter veto is a 3D U-Net). Moved the cached sweep back to GPU.
- GPU accounting: submission reruns on the hidden test do not use the weekly quota,
  and QUICK_SAVE does not start a run.

## 2026-09-22 — Probing the LB directly

- batch01: the controlled baseline (sweep off) is 0.946; v0001's 0.947 is likely a
  rounding boundary. DIVERGE 3.0, predicted +0.0005 by the proxy, scored 0.945.
  **The proxy cannot rank changes this small.**
- A public Discussion (Justin CH123) reported 10 single-parameter changes that all lost,
  with the loss tracking the change in predicted node count. Together with our results,
  single-parameter tuning is a dead end; edge-only changes are the direction left.
- Wrote `scripts/batch.py` (parallel variants, one kernel per slug) and made submission
  idempotent after a duplicate submit wasted two slots.
- Measured the GPU quota in the Kaggle UI and started a ledger (`scripts/gpu.py`); the
  estimate was within 0.3 h. Heavy runs (sweeps) had used 70% of the week.

## 2026-09-23 — Relink and automation

- batch02 (detection threshold, ILP division weight): 0.945–0.946.
- batch03 (relink): learned bonus ×2.5 → 0.945, ×5 → 0.946, **relink off → 0.946**.
  The relink stage does not show at three decimals.
- Wrote `scripts/cycle.py` so the loop runs unattended: collect scores, submit finished
  kernels (5/day, no duplicates, stop before the cutoff), report quota and GPU.
- Added a pre-commit hook that blocks Kaggle credentials.

## 2026-09-24 — Geometric Fusion and the 12-hour limit

- The unmodified Geometric Fusion notebook (public 0.948) **timed out** on the hidden
  test: 4 h on the public test, but 3.4 h of that is a fixed validator + sweep, and the
  hidden test is ~50 times larger.
- `b04-glight`, the same notebook with its sweep's choice fixed and the sweep off,
  produced identical output in 26 minutes and scored **0.947**. The public 0.948 did not
  reproduce.
- Found the weekly GPU reset (Saturday ~00:13 UTC) in the account menu; before it,
  the reserve for final runs is no longer needed.
- `batch.py` can now build variants from either pipeline (`"base": "geofusion"`).
- Launched b05 (Geometric Fusion: relink-only change; pipeline with no overrides) and
  b06 (Geometric Fusion without leaf pruning; the relink-only change on Harmonic Fusion).
- Rewrote the repository in English as a portfolio: consolidated docs, removed unused
  scaffolding, and stopped tracking third-party code that had come back in kernel outputs.
- Prepared `notebooks/public/biohub-cell-tracking-light-pipeline` for publishing: the b04-glight
  configuration with an English introduction, credits and license notes. Saved to Kaggle as
  private without running it.
- Scheduled one run of the public notebook after the weekly GPU reset
  (`experiments/configs/pending_runs.json`; `cycle.py` lists it when due).
- Added README figures: raw data, sparse ground-truth tracks and a division, rendered by
  `notebooks/viz` on a CPU kernel, plus an LB chart from `scripts/plot_results.py`.

## 2026-09-24 (later) — Where Geometric Fusion's +0.001 comes from

| Variant | LB |
|---|---|
| Geometric Fusion pipeline, no overrides (b05-gbase) | 0.946 |
| + relink-only part of its combo (b05-gedge) | 0.946 |
| + full combo (b04-glight) | 0.947 |

- The pipeline itself is no better than Harmonic Fusion (both 0.946).
- The relink-only part does nothing, again. Whatever is worth the +0.001 (or the rounding
  boundary) is in the node-changing part: leaf pruning, wider division gates, DeepCenter threshold.
- b06-gnoleaf (combo without leaf pruning) will say whether leaf pruning is the piece.

## 2026-09-24 (late) — Two public notebooks now score 0.953

- The goal changed to a medal. The live LB (3,878 teams): bronze cutoff is rank 387 at 0.948, but
  0.948 is a 128-team tie ordered by submission time, so a new 0.948 would land around rank 439.
  Safely inside bronze on the public LB needs 0.949 (silver cutoff 0.952).
- Checked whether the 0.948 → 0.947 gap was run-to-run noise: our geofusion and glight outputs are
  byte-identical, so runs are deterministic and resubmitting cannot help.
- Listing by score found `kunaldesale2408/biohub-cell-tracking` and `anvithpothula/biohub-x138`,
  both **public 0.953**. The first (Apache 2.0, 24 min) adds a learned coordinate-refinement
  head (`anvithpothula/biohub-v1284-head-s075`) to Harmonic Fusion. Forked unmodified as
  `biohub-k953` and started it; `cycle.py` submits it when the day's quota resets.
- Lesson: the daily scout listed these notebooks as known, but I never opened their score.
  Titles and vote counts do not show the score; the page does.
- b06: glight without leaf pruning 0.947 (pruning not needed); the relink-only change on
  Harmonic Fusion 0.946. The combo's +0.001 sits in the wider division gates / DeepCenter threshold,
  and is small next to the +0.006 of the coordinate-refinement notebooks.
- `anvithpothula/biohub-x138` is line-for-line the same code as the k953 source, so it is not forked.
- `batch.py` can now build from k953 (`"base": "k953"`; a header cell is inserted). Launched b08 for
  tomorrow's quota: k953 + Geometric Fusion's helpful pieces (velocity 0.25, wider division gates,
  DeepCenter safe-div 0.15), and k953 + the division part only.
- b08 v1 failed in 15 s: k953's settings carry trailing comments and the setdefault rewrite pulled
  them inside the call. Fixed the rewrite, and batch.py now parses every cell before pushing.
  Relaunched as v2.

## 2026-09-25 — Submitting the 0.953 line

- 00:03 UTC: submitted k953 (unmodified 0.953 notebook), b08-kcombo and b08-kdiv (3/5).
  The last two slots wait for these scores (~07:00 UTC), leaving time to run and submit today.
- Scout: two new notebooks, neither useful (`amanatar/optimized-biohub-max-score` scores 0.910,
  `denpugovkin/biohub-exp003-two-embryos` has no score). The top of the score-sorted list is unchanged.
- 06:56 UTC: **k953 scored 0.953** (public rank 151 of 3,898). b08-kcombo also 0.953.
  Cutoffs moved fast as teams fork the public notebook: bronze 0.948 -> 0.951, silver 0.953
  (218 teams tied at 0.953). Private is 71% of the test, so nothing is settled.
- The coordinate-refinement head shifts each detected centre by at most 2 um toward the learned
  target (about 10% closer in its author's held-out check). A regression like this tends to
  under-shoot, and scaling the shift changes positions only, not node count. b09 tries the shift
  x1.5 and x0.5 via a code patch (`"patches"` in batch configs; each anchor must match exactly once).

## 2026-09-25 (later) — Retraining the refinement head on more data

- The 0.953 gain comes from the V1284 head, which its author trained on 20 movies (4,136 pairs);
  their 4-movie version was worse, so data volume matters. Train has 199 movies.
- The notebook already has a `capture` mode that saves detector centres and the 224-d features
  the head uses. Built a capture run from k953 (`experiments/configs/capture01.json`): predict on a
  slice of train instead of test, stop after prediction, tar the features. Named outside `batch*`
  so `cycle.py` never submits it.
- `notebooks/headtrain`: pairs captured centres with ground truth (Hungarian, 3 um), checks the
  coordinate convention against the ground truth first, trains the same head architecture, and
  compares none / public head / ours on held-out movies (4 folds). Only a head that beats the
  public one on held-out movies gets used.
- Smoke test first (8 movies, today's leftover GPU); the full capture (~120 movies) runs after the
  weekly GPU reset.
- `batch.py` gained `patches`, `truncate_after` and `append_cells` for these non-submission runs.
- b08-kdiv 0.953 too: the Geometric Fusion settings add nothing on top of the refinement head.
  Submitted b09 (refinement shift x1.5 / x0.5); 5/5 used today.
- Capture smoke test (8 movies): 17 min of prediction, 800 frames, 256 MB. The head-training notebook
  runs end to end. Held-out residual: none 1.496 um, public head 1.389 um, ours (8 movies, 1,327
  pairs) 1.436 um. As the author found, a few movies are not enough. The full capture (199 movies,
  two kernels of ~3.5 h) is scheduled right after the weekly GPU reset.
- Bug caught before it cost anything: the submissions API returns 20 entries by default. With 25
  submissions, the oldest looked unsubmitted and cycle.py queued b01-base and b01-div10 again (only the
  exhausted daily quota stopped it). Now requests 200 per page and treats every scored slug as submitted.
- b09: refinement shift x1.5 -> 0.949, x0.5 -> 0.945 (x1.0 is 0.953). The score is very sensitive
  to how the head moves centres, and the public head sits near the optimum. A retrained head can
  move the LB a lot in either direction, so it is only submitted if it clearly wins on held-out
  movies, and a proven 0.953 stays in the final selection.
- Public rank slipped 151 -> 186 as more teams fork the 0.953 notebook: 256 teams now share 0.953
  (ranks 183-438), and both the silver (195) and bronze (390) cutoffs sit inside that tie.

## 2026-09-26 — Retrained refinement head

Captured detector features on all 199 train movies (cap-a, cap-b) and retrained the V1284
head (224→32→3, same bound). Held-out residual by movie, 4 folds: no refinement 1.509 µm,
public head 1.300 µm, ours 1.062 µm (better in every fold). Detections matched the ground
truth best with a +0.375-cell centre offset on y and x. Launched `biohub-b10-khead`: k953 with our head.

**b10-khead: 0.943** (k953 0.953). The retrained head beats the public one on held-out residual (1.06 vs 1.30 um), yet the LB drops by 0.010, below the unrefined pipeline (~0.946). The most likely cause is a convention mismatch: the head was trained against the zyx_ds4_center hypothesis (+0.375 cell on y/x), while the notebook's downstream conversion assumes the public head's frame. The LB, not the residual, decides; k953 stays in the final selection.

**Head retrained in the notebook's output frame.** The notebook writes ds coords x (1,4,4) with no centre offset, so headtrain v3 learns targets in that frame (zyx_ds4). Held-out residual: none 1.46, public 1.33, ours 1.06 um. The public head is no better in this frame than in the centred one (1.33 vs 1.30), so the frame mismatch is not confirmed as the cause of 0.943. Launched as biohub-b11-khead4 to let the LB decide.

**b11-khead4: 0.953**, tying k953 (b10-khead in the centred frame: 0.943). The frame did matter: training targets must be in the frame the notebook writes (ds coords x4, no centre offset). The 1.06 vs 1.33 um residual gain does not show at three decimals on the public LB.

**Final selection set to biohub-k953 and biohub-b11-khead4** (both 0.953 public): the proven public head and our head with the lower held-out residual.

## 2026-09-27 15:10 UTC — last scout before the cutoff
- One new public notebook (mtoshidesu/testbiohub-lf-dctta020-sectta1, best public 0.947). Its 8-view detection TTA, DeepCenter TTA and edge-feature TTA are already in k953; the only difference is the secondary edge-TTA weight (1.0 vs 0.75), a single knob on a weaker stack. Not worth a slot.
- Recent discussions (relink stage at the 0.947 plateau, noisy ground truth, "improve every part") give no concrete technique to add. No new submission; b12-khead4combo is still being scored.
- b12-khead4combo (b11 head + kcombo linking settings): 0.953, a tie. Final selection stays k953 + b11-khead4. Public LB at this point: rank 394/3955 at 0.953 (397 teams tied at 0.953; bronze line rank 395).
- Last-shot batch (lottery, spare slots and GPU would otherwise go unused): b13-secetta10 (secondary edge-TTA weight 1.0, from the new public notebook), b13-bonus075 (learned relink bonus 0.75; raising it lost in b03), and b14-headens5 (5-seed ensemble of the retrained head, launched once headtrain v4 finishes). Final selection changes only with approval and only above 0.953.
- b13-secetta10 and b13-bonus075 submitted. headtrain v4: 5-seed ensemble train residual 0.992 vs single 1.006 um; b14-headens5 launched.
- b13-secetta10 0.953 and b13-bonus075 0.953: both ties. b14-headens5 still scoring.
- b14-headens5: 0.951 (-0.002). Submission window closed at 15:00 UTC. Final selection kept as k953 + b11-khead4 (user confirmed the recommendation).
- Last-shot round (user asked to use every slot): cutoff moved to 09-29 05:00 UTC (scoring ~19 h, deadline 23:59). Head-shift scale sweep on b11: b15 x1.25/x1.5 launched; b16 x2.0/x0.8, b17 x1.25+bonus075 / x1.25+secetta10, b18 x1.5+secetta10 queued two at a time.
- b15-shift125/150 failed: the module rejects displacements above 2 um. Added a patch raising that check to 4 um in batches 15-18; relaunched as b15-shift125f/150f.
- b16-shift200/shift080 finished on GPU; launched b17-s125bonus075 / b17-s125secetta (22:40 UTC). b15-shift125f/150f still awaiting scores; today's 5 submission slots are used up (day resets 00:00 UTC).
- b15-shift125f: 0.948, b15-shift150f: 0.947 (both below 0.953; scaling the head shift up hurts). b18 cannot be submitted (5/5 used, cutoff 05:00 UTC); b16/b17 still awaiting scores.
- Last-shot sweep done: every head-shift scale away from 1.0 loses (x0.8 0.951, x1.25 0.948, x1.5 0.947, x2.0 0.930). The trained scale is already the optimum. Final selection unchanged: k953 + b11-khead4 (0.953).

## 2026-09-30 — final standings
- Private LB 0.924, rank 183 of 4,017, Silver medal (confirmed by Kaggle). Selected: b11-khead4 (private 0.924) and k953 (0.917).
- The retrained head (b11) tied k953 on public but beat it by +0.007 on private, matching the held-out residual gain (1.06 vs 1.33 um).
- Best unselected: b12-khead4combo 0.929 (public tie at 0.953, so no signal to pick it).
