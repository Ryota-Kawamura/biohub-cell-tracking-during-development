"""Illustrations for the Kaggle write-up and the README.

  docs/images/banner.png      result summary (rank, scores, gain)
  docs/images/pipeline.png    the four pipeline stages, with the one I changed
  docs/images/frame_bug.png   why the first retrained head scored lower

    python scripts/plot_writeup.py
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

IMG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "images"

SURFACE = "#fcfcfb"
PANEL = "#f0efec"
INK = "#0b0b0b"
INK2 = "#52514e"
LINE = "#d6d5d0"
BLUE = "#2a78d6"
BLUE_SOFT = "#e3eefb"
ORANGE = "#eb6834"
SILVER = "#8e9196"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def box(ax, x, y, w, h, face, edge=None, lw=0, r=0.03):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge or face, linewidth=lw))


def banner():
    fig = plt.figure(figsize=(12, 3.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    ax.text(0.5, 2.95, "BIOHUB – CELL TRACKING DURING DEVELOPMENT", fontsize=11, color=INK2,
            fontweight="bold")
    ax.text(0.5, 2.35, "Retraining one small head, measured by held-out movie", fontsize=19, color=INK,
            fontweight="bold")
    tiles = [("181st", "of 3,947 teams", SILVER, "silver medal"),
             ("0.924", "private LB", BLUE, "final score"),
             ("+0.007", "private vs. base", BLUE, "public: tied at 0.953"),
             ("1.33 → 1.06", "µm centre error", BLUE, "held-out, by movie")]
    x = 0.5
    for big, small, color, note in tiles:
        box(ax, x, 0.35, 2.6, 1.55, PANEL, r=0.12)
        ax.add_patch(Rectangle((x, 0.35), 0.07, 1.55, facecolor=color, edgecolor="none"))
        ax.text(x + 0.3, 1.3, big, fontsize=22, color=INK, fontweight="bold", va="center")
        ax.text(x + 0.3, 0.83, small, fontsize=11, color=INK2, va="center")
        ax.text(x + 0.3, 0.55, note, fontsize=9.5, color=INK2, va="center", style="italic")
        x += 2.8
    fig.savefig(IMG / "banner.png")
    plt.close(fig)


def pipeline():
    fig = plt.figure(figsize=(12, 2.9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 2.9); ax.axis("off")
    stages = [("Detection", "TemporalUNet3D\n2 seeds · 8-view TTA", False),
              ("Centre refinement", "224 → 32 → 3 MLP\nshift ≤ 2 µm", True),
              ("Linking", "candidate edges → ILP\ntracksdata + SCIP", False),
              ("Post-processing", "gap closing · relink\ndivisions · DeepCenter", False)]
    w, gap, y, h = 2.55, 0.43, 0.75, 1.55
    x = 0.4
    for i, (title, body, mine) in enumerate(stages):
        box(ax, x, y, w, h, BLUE_SOFT if mine else PANEL, edge=BLUE if mine else None,
            lw=2 if mine else 0, r=0.12)
        ax.text(x + w / 2, y + h - 0.38, title, ha="center", fontsize=13, fontweight="bold",
                color=BLUE if mine else INK)
        ax.text(x + w / 2, y + 0.55, body, ha="center", va="center", fontsize=10, color=INK2,
                linespacing=1.5)
        tag = "retrained on 199 movies" if mine else "public, unchanged"
        ax.text(x + w / 2, y - 0.3, tag, ha="center", fontsize=10,
                color=BLUE if mine else INK2, fontweight="bold" if mine else "normal")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.06, y + h / 2), xytext=(x + w + 0.06, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5, mutation_scale=14))
        x += w + gap
    ax.text(0.4, 2.62, "The pipeline: one stage changed", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(IMG / "pipeline.png")
    plt.close(fig)


def frame_bug():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=150)
    titles = [("Notebook output frame", "index × 4, no offset", ORANGE),
              ("Frame I trained in first", "voxel centre, +0.375 voxel (1.5 px)", BLUE)]
    for ax, (title, sub, color) in zip(axes, titles):
        ax.set_xlim(-1.1, 4.1); ax.set_ylim(-1.4, 4.1); ax.set_aspect("equal"); ax.axis("off")
        # 4 x 4 full-resolution pixels (centres at 0..3) make one downsampled voxel
        for i in range(5):
            v = i - 0.5
            ax.plot([-0.5, 3.5], [v, v], color=LINE, lw=1); ax.plot([v, v], [-0.5, 3.5], color=LINE, lw=1)
        ax.add_patch(Rectangle((-0.5, -0.5), 4, 4, fill=False, edgecolor=INK2, lw=2))
        ax.text(1.5, 3.75, title, ha="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(1.5, -1.0, sub, ha="center", fontsize=11, color=INK2)
    # one downsampled voxel = 4 × 4 full-resolution pixels
    axes[0].scatter([0], [0], s=180, color=ORANGE, zorder=5, edgecolor=SURFACE, linewidth=2)
    axes[0].annotate("where the pipeline\nwrites the node", (0, 0), xytext=(0.9, 1.6), fontsize=10,
                     color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[1].scatter([0], [0], s=120, color=ORANGE, alpha=0.35, zorder=4)
    axes[1].scatter([1.5], [1.5], s=180, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=2)
    axes[1].annotate("", xy=(1.42, 1.42), xytext=(0.08, 0.08),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    axes[1].text(1.9, 0.3, "constant offset\nin every target\n→ 0.943 public", fontsize=10, color=INK)
    fig.suptitle("One downsampled voxel (4 × 4 pixels in y, x)", fontsize=11, color=INK2, y=0.06)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(IMG / "frame_bug.png")
    plt.close(fig)


if __name__ == "__main__":
    banner(); pipeline(); frame_bug()
    print("wrote banner.png, pipeline.png, frame_bug.png")
