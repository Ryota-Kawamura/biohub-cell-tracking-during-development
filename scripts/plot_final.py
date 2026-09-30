"""Figures for the final standings (README).

Reads experiments/results/final_scores.csv (public and private score of every
submission, saved after the competition closed) and writes:

  docs/images/public_vs_private.png   every submission, public against private
  docs/images/coordinate_head.png     held-out error of the coordinate head, and
                                      the shift-scale sweep on the leaderboard

    python scripts/plot_final.py
"""
import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # our retrained head
ORANGE = "#eb6834"    # public notebook it was built on
GRAY = "#b9b8b2"      # everything else

SELECTED = {"biohub-k953", "biohub-b11-khead4"}


def uses_head(slug):
    """Batches 10-18 all run k953 with the retrained coordinate head."""
    import re
    m = re.match(r"biohub-b(\d+)-", slug)
    return bool(m) and 10 <= int(m.group(1)) <= 18

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def load():
    rows = list(csv.DictReader(open(ROOT / "experiments/results/final_scores.csv", encoding="utf-8")))
    for r in rows:
        r["public"], r["private"] = float(r["public"]), float(r["private"])
    return rows


def public_vs_private(rows):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    groups = [
        ("other experiments", GRAY, lambda s: not uses_head(s) and s != "biohub-k953"),
        ("public notebook (k953)", ORANGE, lambda s: s == "biohub-k953"),
        ("with our retrained head", BLUE, uses_head),
    ]
    for label, color, pick in groups:
        pts = [r for r in rows if pick(r["slug"])]
        ax.scatter([r["public"] for r in pts], [r["private"] for r in pts], s=70, color=color,
                   edgecolor=SURFACE, linewidth=2, label=label, zorder=3)
    for r in rows:
        if r["slug"] in SELECTED or r["slug"] == "biohub-b12-khead4combo":
            name = {"biohub-k953": "k953 (selected)", "biohub-b11-khead4": "b11 (selected, final 0.924)",
                    "biohub-b12-khead4combo": "b12 (best private, 0.929)"}[r["slug"]]
            xy_text = {"biohub-k953": (0.9523, 0.9135), "biohub-b11-khead4": (0.9445, 0.9275),
                       "biohub-b12-khead4combo": (0.9523, 0.9315)}[r["slug"]]
            ax.annotate(name, (r["public"], r["private"]), xytext=xy_text,
                        ha="right", va="center", fontsize=10, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axvline(0.953, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(0.9532, 0.8905, "public 0.953:\n7 submissions tie,\nprivate spreads\n0.917 – 0.929", fontsize=9.5, color=INK2, va="bottom")
    ax.set_xlabel("Public LB score")
    ax.set_ylabel("Private LB score")
    ax.set_title("Public ties, private separates", loc="left", fontsize=14, color=INK, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="center left", fontsize=10)
    ax.set_xlim(0.927, 0.9575)
    ax.set_ylim(0.888, 0.934)
    fig.tight_layout()
    fig.savefig(IMG / "public_vs_private.png")
    plt.close(fig)


def coordinate_head(rows):
    by = {r["slug"]: r for r in rows}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})

    # Held-out error by movie (4 folds, 199 training movies, 111k detection-GT pairs).
    names = ["no refinement", "public head", "retrained head"]
    vals = [1.458, 1.328, 1.063]
    bars = a.bar(names, vals, width=0.55, color=[GRAY, ORANGE, BLUE], edgecolor=SURFACE, linewidth=2, zorder=3)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=11)
    a.set_ylim(0, 1.7)
    a.set_ylabel("mean distance to ground truth (µm)")
    a.set_title("Held-out centre error", loc="left", fontsize=13, color=INK, fontweight="bold")
    a.grid(axis="x", visible=False)

    scales = [0.8, 1.0, 1.25, 1.5, 2.0]
    slugs = ["biohub-b16-shift080", "biohub-b11-khead4", "biohub-b15-shift125f", "biohub-b15-shift150f", "biohub-b16-shift200"]
    for key, color, label in (("public", INK2, "public"), ("private", BLUE, "private")):
        ys = [by[s][key] for s in slugs]
        b.plot(scales, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        b.text(2.04, ys[-1], label, color=color, va="center", fontsize=10)
    b.annotate("trained scale is the optimum", (1.0, by["biohub-b11-khead4"]["private"]), xytext=(1.12, 0.905),
               fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    b.set_xticks(scales, [f"×{s:g}" for s in scales])
    b.set_xlim(0.7, 2.25)
    b.set_xlabel("head shift, scaled after training")
    b.set_ylabel("LB score")
    b.set_title("Scaling the learned shift", loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(w_pad=3)
    fig.savefig(IMG / "coordinate_head.png")
    plt.close(fig)


def timeline(rows):
    """Private score of every submission in the order it was made."""
    import datetime as dt
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)

    def lineage(s):
        if uses_head(s):
            return BLUE
        if s.startswith(("biohub-k953", "biohub-b08", "biohub-b09")):
            return ORANGE
        return GRAY

    when = [dt.datetime.fromisoformat(r["submitted_utc"]) for r in rows]
    for w, r in zip(when, rows):
        ax.scatter(w, r["private"], s=60, color=lineage(r["slug"]), edgecolor=SURFACE, linewidth=2, zorder=3)
    best, xs, ys = 0.0, [], []
    for w, r in sorted(zip(when, rows), key=lambda p: p[0]):
        best = max(best, r["private"])
        xs.append(w); ys.append(best)
    ax.step(xs, ys, where="post", color=INK2, lw=1.2, zorder=2)
    ax.text(xs[-1], ys[-1] + 0.0012, "best so far", color=INK2, fontsize=9.5, ha="right")
    for label, color in (("Harmonic / Geometric Fusion forks", GRAY), ("0.953 public notebook (k953)", ORANGE),
                         ("k953 + retrained head", BLUE)):
        ax.scatter([], [], s=60, color=color, label=label)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.set_ylim(0.885, 0.935)
    ax.set_ylabel("Private LB score")
    ax.xaxis.set_major_formatter(__import__("matplotlib.dates", fromlist=["x"]).DateFormatter("%b %d"))
    ax.set_title("37 submissions in 9 days", loc="left", fontsize=14, color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(IMG / "timeline.png")
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    public_vs_private(rows)
    coordinate_head(rows)
    timeline(rows)
    print("wrote figures to", IMG)
