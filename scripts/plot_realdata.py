"""Real-data figures for the Kaggle write-up, drawn from the figdata.npz that
notebooks/writeupfigs saves (download it into artifacts/writeupfigs/ first).

  docs/images/real_pipeline.png    one training frame: input, detections, tracks, and a zoom
                                   before / after the public and the retrained head
  docs/images/cell_over_time.png   one annotated cell over five frames
  docs/images/heldout_errors.png   held-out error per movie and per detection
  docs/images/target_frame.png     the training targets in the two coordinate frames

    kaggle kernels output ryotakawamura94/biohub-writeupfigs -p artifacts/writeupfigs
    python scripts/plot_realdata.py [example index]
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "images"
D = np.load(ROOT / "artifacts" / "writeupfigs" / "figdata.npz")
EX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#b9b8b2"
# brighter versions of the same hues for marks drawn on dark microscopy images
ON_IMG = {"gt": "#3ddc84", "det": "#ffffff", "public": "#ff8a4c", "ours": "#5ab0ff"}
PX = 0.40625   # um per full-resolution pixel in y and x

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11})

mip = D[f"ex{EX}_mip"].astype(np.float32)
ts = list(D[f"ex{EX}_ts"])
tc = int(D[f"ex{EX}_tc"])
det = D[f"ex{EX}_det"]          # t, det zyx, public zyx, ours zyx (um)
gt = D[f"ex{EX}_gt"]            # id, t, zyx (um)
edges = D[f"ex{EX}_edges"]
movie = str(D["examples"][EX])
W = mip.shape[-1]


def at(arr, t):
    return arr[arr[:, 0 if arr is det else 1] == t]


def frame(t):
    d = det[det[:, 0] == t]
    g = gt[gt[:, 1] == t]
    return d[:, 1:4], d[:, 4:7], d[:, 7:10], g[:, 2:5], g[:, 0].astype(np.int64)


def match(a, b, max_um=3.0):
    if not len(a) or not len(b):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= max_um, d, 1e6))
    keep = d[r, c] <= max_um
    return np.stack([r[keep], c[keep]], 1)


def errors(t, box=None):
    raw, pub, ours, g, _ = frame(t)
    pr = match(raw, g)
    if box is not None:
        y0, x0, s = box
        gy, gx = g[pr[:, 1], 1] / PX, g[pr[:, 1], 2] / PX
        pr = pr[(gy >= y0) & (gy < y0 + s) & (gx >= x0) & (gx < x0 + s)]
    return {k: np.linalg.norm(p[pr[:, 0]] - g[pr[:, 1]], axis=1) for k, p in
            (("none", raw), ("public", pub), ("ours", ours))}


def img(ax, t, box=None):
    ax.imshow(mip[ts.index(t)], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if box is None:
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
    else:
        y0, x0, s = box
        ax.set_xlim(x0 - 0.5, x0 + s - 0.5); ax.set_ylim(y0 + s - 0.5, y0 - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def gt_circles(ax, g, size):
    ax.scatter(g[:, 2] / PX, g[:, 1] / PX, s=size, facecolors="none", edgecolors=ON_IMG["gt"],
               linewidths=1.6, zorder=4)


def scalebar(ax, x0, y0, um, color="white"):
    ax.plot([x0, x0 + um / PX], [y0, y0], color=color, lw=3, solid_capstyle="butt", zorder=6)
    ax.text(x0 + um / PX / 2, y0 - 2.2 * (ax.get_ylim()[0] - ax.get_ylim()[1]) / 100, f"{um:g} µm",
            color=color, ha="center", va="bottom", fontsize=9, zorder=6)


def best_box(size=56):
    """The size x size window with the most detector-matched annotated cells at tc."""
    raw, _, _, g, _ = frame(tc)
    pr = match(raw, g)
    gy, gx = g[pr[:, 1], 1] / PX, g[pr[:, 1], 2] / PX
    best = (-1, 0, 0)
    for y0 in range(4, W - size - 3, 4):
        for x0 in range(4, W - size - 3, 4):
            n = int(((gy >= y0 + 4) & (gy < y0 + size - 4) & (gx >= x0 + 4) & (gx < x0 + size - 4)).sum())
            best = max(best, (n, y0, x0))
    return best[1], best[2], size


def real_pipeline():
    raw, pub, ours, g, gid = frame(tc)
    pr = match(raw, g)
    err = errors(tc)
    # the six annotated cells with the largest detector error in this frame
    e_raw = np.linalg.norm(raw[pr[:, 0]] - g[pr[:, 1]], axis=1)
    pick = pr[np.argsort(-e_raw)[:6]]

    fig = plt.figure(figsize=(13.2, 7.9), dpi=150)
    gs = fig.add_gridspec(2, 6, height_ratios=[2.05, 1], hspace=0.42, wspace=0.08)
    a, b, c = (fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3))

    img(a, tc)
    scalebar(a, 10, W - 12, 20)
    a.set_title(f"(a) Input: one frame, max-projection over z\n{movie}, t = {tc}, "
                f"(T, Z, Y, X) = {tuple(int(v) for v in D[f'ex{EX}_shape'])}", fontsize=10.5)

    img(b, tc)
    b.scatter(raw[:, 2] / PX, raw[:, 1] / PX, s=6, c="#ff5a5a", linewidths=0, zorder=3)
    gt_circles(b, g, 60)
    for n, (i, j) in enumerate(pick, 1):
        b.text(g[j, 2] / PX + 6, g[j, 1] / PX - 6, str(n), color="#ffd23f", fontsize=10,
               fontweight="bold", zorder=6)
    b.set_title(f"(b) Detector: {len(raw)} centres in 3D (red)\n"
                f"ground truth labels only {len(g)} cells (green)", fontsize=10.5)

    img(c, tc)
    parent = {int(dd): int(s) for s, dd in edges}
    pos = {int(r[0]): (int(r[1]), r[3] / PX, r[4] / PX) for r in gt}
    cmap = plt.get_cmap("tab10")
    for i, nid in enumerate(gid):
        pts, cur = [], int(nid)
        while cur in pos and pos[cur][0] >= tc - 10:
            pts.append(pos[cur][1:]); cur = parent.get(cur, -1)
        if len(pts) > 1:
            q = np.array(pts)
            c.plot(q[:, 1], q[:, 0], "-", color=cmap(i % 10), lw=1.6, alpha=0.95)
        c.scatter([pts[0][1]], [pts[0][0]], s=10, c="white", zorder=4)
    c.set_title("(c) Linking (public, unchanged) must recover\nthe annotated tracks, here over t−10 … t",
                fontsize=10.5)

    half = 14   # 28 px = 11 µm crops
    for n, (i, j) in enumerate(pick):
        ax = fig.add_subplot(gs[1, n])
        cy, cx = g[j, 1] / PX, g[j, 2] / PX
        y0 = int(np.clip(round(cy) - half, 0, W - 2 * half)); x0 = int(np.clip(round(cx) - half, 0, W - 2 * half))
        img(ax, tc, (y0, x0, 2 * half))
        ax.scatter([cx], [cy], s=520, facecolors="none", edgecolors=ON_IMG["gt"], linewidths=1.8, zorder=4)
        ax.scatter([cx], [cy], s=12, c=ON_IMG["gt"], zorder=8)
        for key, col in (("public", pub), ("ours", ours)):
            ax.annotate("", xy=(col[i, 2] / PX, col[i, 1] / PX), xytext=(raw[i, 2] / PX, raw[i, 1] / PX),
                        arrowprops=dict(arrowstyle="-|>", color=ON_IMG[key], lw=1.6, mutation_scale=9,
                                        shrinkA=0, shrinkB=0), zorder=6)
            ax.scatter([col[i, 2] / PX], [col[i, 1] / PX], s=30, c=ON_IMG[key], zorder=7)
        ax.scatter([raw[i, 2] / PX], [raw[i, 1] / PX], s=90, marker="+", c=ON_IMG["det"], linewidths=2, zorder=5)
        d = [np.linalg.norm(v[i] - g[j]) for v in (raw, pub, ours)]
        ax.text(0.04, 0.96, f"{n + 1}", transform=ax.transAxes, ha="left", va="top", fontsize=11,
                color="#ffd23f", fontweight="bold", zorder=9)
        ax.text(0.5, -0.06, f"{d[0]:.2f} → {d[1]:.2f} → {d[2]:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=INK)
    fig.text(0.5, 0.395, "(d) Centre refinement, the only stage I changed", ha="center", fontsize=11,
             color=INK, fontweight="bold")
    fig.text(0.5, 0.372, "The six annotated cells with the largest detector error in this frame "
             "(11 µm crops). Under each: 3D error in µm, detector → public head → retrained head",
             ha="center", fontsize=10, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ON_IMG["gt"], mew=2, ms=10, label="ground truth"),
               plt.Line2D([], [], marker="+", ls="", color=INK2, mew=2, ms=10, label="detector"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="public head (20 movies)"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="retrained head (199 movies, held-out fold)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.045, f"Over all {len(err['none'])} matched cells in this frame: "
             f"{err['none'].mean():.2f} → {err['public'].mean():.2f} → {err['ours'].mean():.2f} µm. "
             "Projections hide z; errors are 3D.", ha="center", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.11)
    fig.savefig(IMG / "real_pipeline.png")
    plt.close(fig)
    return pick


def cell_over_time():
    # the annotated cell present in the most frames of the window ending at tc + 1
    window = [t for t in ts if tc - 3 <= t <= tc + 1]
    ids_by_t = [set(gt[gt[:, 1] == t][:, 0].astype(np.int64)) for t in window]
    parent = {int(dd): int(s) for s, dd in edges}
    # follow tracks backwards from tc + 1
    best = None
    for nid in ids_by_t[-1]:
        chain, cur = [nid], nid
        while len(chain) < len(window) and cur in parent:
            cur = parent[cur]; chain.append(cur)
        if len(chain) == len(window):
            chain = chain[::-1]
            # prefer cells where all three variants are matched in every frame
            e = []
            for t, cid in zip(window, chain):
                raw, pub, ours, g, gid = frame(t)
                j = np.where(gid == cid)[0][0]
                k = np.linalg.norm(raw - g[j], axis=1).argmin()
                e.append((np.linalg.norm(raw[k] - g[j]), np.linalg.norm(pub[k] - g[j]),
                          np.linalg.norm(ours[k] - g[j])))
            e = np.array(e)
            if e[:, 0].max() < 3.0:
                score = e[:, 0].mean()        # a typical cell: rank by raw error only
                cand = (abs(score - 1.4), chain, e)
                best = cand if best is None or cand[0] < best[0] else best
    _, chain, e = best
    half = 20
    fig, axes = plt.subplots(1, len(window), figsize=(3.0 * len(window), 4.0), dpi=150)
    for ax, t, cid, ee in zip(axes, window, chain, e):
        raw, pub, ours, g, gid = frame(t)
        j = np.where(gid == cid)[0][0]
        cy, cx = g[j, 1] / PX, g[j, 2] / PX
        img(ax, t, (int(cy) - half, int(cx) - half, 2 * half))
        k = np.linalg.norm(raw - g[j], axis=1).argmin()
        ax.scatter([cx], [cy], s=900, facecolors="none", edgecolors=ON_IMG["gt"], linewidths=2, zorder=4)
        ax.scatter([raw[k, 2] / PX], [raw[k, 1] / PX], s=120, marker="+", c=ON_IMG["det"], linewidths=2, zorder=5)
        ax.scatter([pub[k, 2] / PX], [pub[k, 1] / PX], s=60, c=ON_IMG["public"], zorder=6)
        ax.scatter([ours[k, 2] / PX], [ours[k, 1] / PX], s=60, c=ON_IMG["ours"], zorder=7)
        ax.scatter([cx], [cy], s=14, c=ON_IMG["gt"], zorder=8)
        ax.set_title(f"t = {t}", fontsize=11)
        ax.text(0.5, -0.05, f"{ee[0]:.2f} → {ee[1]:.2f} → {ee[2]:.2f} µm", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ON_IMG["gt"], mew=2, ms=11, label="ground truth"),
               plt.Line2D([], [], marker="+", ls="", color=INK2, mew=2, ms=10, label="detector"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="public head"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="retrained head")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("One annotated cell over five frames (16 µm crops). "
                 "Under each frame: 3D error of detector → public head → retrained head",
                 fontsize=11, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(IMG / "cell_over_time.png")
    plt.close(fig)


def heldout_errors():
    pub, ours = D["per_movie_public"], D["per_movie_ours"]
    pe = D["pair_err"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
    lo, hi = min(pub.min(), ours.min()) - 0.05, max(pub.max(), ours.max()) + 0.05
    a.plot([lo, hi], [lo, hi], color=GRAY, lw=1.2, zorder=1)
    a.scatter(pub, ours, s=26, c=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_xlabel("public head: mean error (µm)"); a.set_ylabel("retrained head: mean error (µm)")
    n_better = int((ours < pub).sum())
    a.set_title(f"Per movie, held-out: retrained head better on {n_better} of {len(pub)}", loc="left")
    a.text(hi - 0.03, lo + 0.04, "below the line =\nretrained head better", ha="right", va="bottom",
           fontsize=9.5, color=INK2)
    a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)

    xs = np.linspace(0, 3, 301)
    for col, name, color in ((2, "retrained head", BLUE), (1, "public head", ORANGE), (0, "detector only", GRAY)):
        v = np.sort(pe[:, col])
        cdf = np.searchsorted(v, xs, side="right") / len(v)
        b.plot(xs, cdf, color=color, lw=2.4 if col else 2.0, label=f"{name}: mean {v.mean():.2f} µm")
    b.legend(frameon=False, loc="lower right", fontsize=10, handlelength=1.6)
    b.set_xlim(0, 3); b.set_ylim(0, 1.0)
    b.set_xlabel("3D distance to the ground-truth centre (µm)"); b.set_ylabel("share of detections")
    b.set_title(f"Per detection, held-out ({len(pe):,} pairs from {len(pub)} movies)", loc="left")
    b.grid(color=GRID, lw=0.8); b.set_axisbelow(True)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=3)
    fig.savefig(IMG / "heldout_errors.png")
    plt.close(fig)


def target_frame():
    step = PX                                       # targets lie on a 0.406 um lattice
    Y = D["targets"][:, 1:3].ravel()               # y and x components (um)
    off = float(D["centre_offset_um"][1])
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=150)
    for vals, color, label, w, dx in ((Y - off, ORANGE, "voxel-centre frame: my first targets", 0.17, 0),
                                      (Y, BLUE, "output frame (index × 4): the fix", 0.17, 0)):
        k = np.round(vals / step * 2) / 2              # lattice index (half steps for the offset frame)
        u, n = np.unique(k, return_counts=True)
        keep = np.abs(u * step) <= 2.5
        ax.bar(u[keep] * step + dx, n[keep] / len(vals), width=w, color=color, alpha=0.9, label=label)
    ymax = ax.get_ylim()[1]
    for v, c in ((Y.mean(), BLUE), ((Y - off).mean(), ORANGE)):
        ax.axvline(v, color=c, lw=1.4, ls="--")
    ax.annotate("", xy=((Y - off).mean(), ymax * 0.97), xytext=(Y.mean(), ymax * 0.97),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(((Y - off).mean() + Y.mean()) / 2, ymax * 1.0, f"{off:.2f} µm (1.5 px)",
            ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, ymax * 1.1)
    ax.set_xlabel("training target in y and x: ground truth − detection (µm); dashed: mean")
    ax.set_yticks([])
    ax.set_title("The same 111k pairs in two coordinate frames: a constant 0.61 µm shift in every target",
                 loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "target_frame.png")
    plt.close(fig)


if __name__ == "__main__":
    box = real_pipeline(); cell_over_time(); heldout_errors(); target_frame()
    print("example", EX, movie, "t", tc, "box", box)
    for k, v in errors(tc).items():
        print(f"  frame t={tc} {k:6s} {v.mean():.3f} um ({len(v)} cells)")
