"""Thin wrapper around the official scorer.

Scoring is delegated to `third_party/tracking-cellmot` (published by the
organisers, BSD-3). It never reads the images, so it runs on CPU.

    score = adjusted_edge_jaccard + 0.1 * division_jaccard

Usage:

    from harness.score import score_dir
    result = score_dir(pred_dir="preds", gt_dir="/kaggle/input/.../train")
    print(result["overall"], result["by_embryo"])
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = ROOT / "third_party" / "tracking-cellmot"


def _ensure_path() -> None:
    for p in (VENDOR / "src", VENDOR / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def score_dir(
    pred_dir: str | pathlib.Path,
    gt_dir: str | pathlib.Path,
    max_distance: float = 7.0,
) -> dict:
    """Score every .geff present in both pred_dir and gt_dir.

    Returns the per-embryo breakdown next to the overall score, to spot
    settings that only help one embryo (see harness/splits.py).
    """
    _ensure_path()
    from evaluate import evaluate_pairs  # type: ignore[import-not-found]
    from tracking_cellmot.metrics import summarise  # type: ignore[import-not-found]

    from harness.splits import embryo_of

    rows, skipped = evaluate_pairs(pred_dir, gt_dir, max_distance=max_distance)

    by_embryo: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get("dataset") or row.get("name") or "")
        by_embryo.setdefault(embryo_of(name), []).append(row)

    return {
        "overall": summarise(rows),
        "by_embryo": {k: summarise(v) for k, v in sorted(by_embryo.items())},
        "n_samples": len(rows),
        "skipped": skipped,
        "rows": rows,
    }
