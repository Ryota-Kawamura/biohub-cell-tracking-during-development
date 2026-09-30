"""Cross-validation splits.

The test set is embryo-disjoint from train, but train has only two embryos
(44b6 and 6bba). An embryo-disjoint split would be a 2-fold "tune on one,
check on the other", far too coarse for 20+ post-processing parameters.

The compromise:

- stratify by embryo, then split by sample, so both tune and confirm contain
  both embryos;
- always report the score per embryo as well, to catch settings that only
  help one of them (those are the ones likely to break on unseen embryos);
- check the chosen setting on confirm exactly once, so confirm never turns
  into a second tuning set.

Samples are sorted by name and shuffled with a fixed seed, so the split is
reproducible.
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLITS_PATH = ROOT / "experiments" / "configs" / "splits.json"

SEED = 42

# Divisions are rare: 8 samples held only 12 ground-truth divisions, so one
# event moved division Jaccard by 0.08. Enlarging tune/confirm to 80 samples
# costs about 3.1 GPU hours to cache once; scoring afterwards runs on CPU.
N_TUNE_PER_EMBRYO = 24
N_CONFIRM_PER_EMBRYO = 16


def embryo_of(sample: str) -> str:
    """`44b6_0113de3b` -> `44b6` (the folder name starts with the embryo id)."""
    return sample.split("_")[0]


def build(samples: list[str]) -> dict:
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for name in sorted(samples):
        by_embryo[embryo_of(name)].append(name)

    rng = random.Random(SEED)
    tune: list[str] = []
    confirm: list[str] = []
    pool: list[str] = []

    for embryo in sorted(by_embryo):
        names = by_embryo[embryo][:]
        rng.shuffle(names)
        tune += names[:N_TUNE_PER_EMBRYO]
        confirm += names[N_TUNE_PER_EMBRYO : N_TUNE_PER_EMBRYO + N_CONFIRM_PER_EMBRYO]
        pool += names[N_TUNE_PER_EMBRYO + N_CONFIRM_PER_EMBRYO :]

    return {
        "seed": SEED,
        "counts": {k: len(v) for k, v in by_embryo.items()},
        "tune": sorted(tune),
        "confirm": sorted(confirm),
        "pool": sorted(pool),
    }


def load() -> dict:
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
