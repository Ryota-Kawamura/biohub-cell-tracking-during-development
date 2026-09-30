"""Plot every scored submission from submissions/manifest.csv.

    python scripts/plot_results.py      # writes docs/images/lb_results.png
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "submissions" / "manifest.csv"
OUT = ROOT / "docs" / "images" / "lb_results.png"

BASELINE = 0.946
COLOURS = {"infer": "#4c78a8", "geofusion": "#f58518"}


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    scored = [r for r in rows if r["public_lb"].replace(".", "", 1).isdigit()]
    failed = [r for r in rows if not r["public_lb"].replace(".", "", 1).isdigit()]

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.2))
    labels = []
    for i, r in enumerate(scored + failed):
        y = len(scored) + len(failed) - 1 - i
        change = r["change"] if len(r["change"]) <= 44 else r["change"][:42] + "..."
        labels.append((y, f"{r['submission']}: {change}"))
        if r in scored:
            score = float(r["public_lb"])
            ax.plot([BASELINE, score], [y, y], color="#bbbbbb", lw=1, zorder=1)
            ax.scatter(score, y, s=46, color=COLOURS.get(r["base"], "grey"), zorder=2)
            ax.text(score + 0.0002, y, f"{score:.3f}", va="center", fontsize=8)
        else:
            ax.text(0.9392, y, f"{r['public_lb']}: exceeded the 12 h limit, no score", va="center", fontsize=8, color="#c0392b")

    ax.axvline(BASELINE, color="#888888", ls="--", lw=1)
    ax.text(BASELINE, -1.1, "baseline 0.946", fontsize=8, color="#555555", ha="center")
    ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=8)
    ax.set_xlim(0.939, 0.9495)
    ax.set_ylim(-1.5, len(rows) - 0.4)
    ax.set_xlabel("public leaderboard score (3 decimals)")
    ax.set_title("Every submission against the baseline", fontsize=10)
    for base, colour in COLOURS.items():
        ax.scatter([], [], color=colour, label=f"base: {base}")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(scored)} scored, {len(failed)} without score)")


if __name__ == "__main__":
    main()
