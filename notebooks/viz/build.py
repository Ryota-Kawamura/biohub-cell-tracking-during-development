"""Build the visualisation notebook (kept as a script so the cells stay reviewable).

    python notebooks/viz/build.py
    python scripts/push_kernel.py -p notebooks/viz --run     # CPU only
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# Biohub cell tracking: what the data looks like

Renders the figures for the project README: the raw 3D + time volume, the sparse
ground-truth tracks, and one cell division. CPU only.
""")

code(r'''
# zarr v3 is not in the Kaggle image; install it offline from the support pack wheels.
import glob
import subprocess
import sys

wheel_dirs = sorted({p.rsplit("/", 1)[0] for p in glob.glob("/kaggle/input/**/wheels/zarr-*.whl", recursive=True)})
print("wheel dirs:", wheel_dirs)
cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet"]
for d in wheel_dirs:
    cmd += ["--find-links", d]
cmd += ["zarr>=3", "donfig", "google-crc32c", "numcodecs", "typing-extensions"]
print(subprocess.run(cmd, capture_output=True, text=True).stderr[-500:])
''')

code(r'''
import pathlib

import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

print("zarr", zarr.__version__)
TRAIN = next(p for p in pathlib.Path("/kaggle/input").rglob("train") if p.is_dir() and any(p.glob("*.geff")))
OUT = pathlib.Path("/kaggle/working")
print("train dir:", TRAIN)
''')

code(r'''
def load_graph(geff_path):
    g = zarr.open_group(str(geff_path), mode="r")
    ids = np.asarray(g["nodes/ids"][:])
    props = {k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")}
    edges = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    return ids, props, edges


# Pick a sample with several ground-truth divisions (a node with two children).
best = None
for geff in sorted(TRAIN.glob("*.geff"))[:60]:
    ids, props, edges = load_graph(geff)
    src, counts = np.unique(edges[:, 0], return_counts=True)
    n_div = int((counts >= 2).sum())
    if best is None or n_div > best[1]:
        best = (geff, n_div, len(ids))
geff_path, n_div, n_nodes = best
name = geff_path.stem
print(f"sample {name}: {n_nodes} annotated nodes, {n_div} divisions")
ids, props, edges = load_graph(geff_path)
''')

code(r'''
vol = zarr.open_array(str(TRAIN / f"{name}.zarr" / "0"), mode="r")
T, Z, Y, X = vol.shape
print("volume shape (T, Z, Y, X):", vol.shape, vol.dtype)

# Max-intensity projection over z, one frame at a time to keep memory low.
mip = np.stack([np.asarray(vol[t]).max(axis=0) for t in range(T)]).astype(np.float32)
lo, hi = np.percentile(mip, [1, 99.7])
mip = np.clip((mip - lo) / (hi - lo), 0, 1)

index = {int(i): k for k, i in enumerate(ids)}
T_ = props["t"].astype(int)
Yc = props["y"].astype(float)
Xc = props["x"].astype(float)
parent = {int(d): int(s) for s, d in edges}
children = {}
for s, d in edges:
    children.setdefault(int(s), []).append(int(d))
''')

code(r'''
# Figure 1: the raw input at three time points.
frames = [0, T // 2, T - 1]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, t in zip(axes, frames):
    ax.imshow(mip[t], cmap="magma")
    ax.set_title(f"t = {t}")
    ax.axis("off")
fig.suptitle(f"Raw input, max-intensity projection over z  ({T} frames of {Z} x {Y} x {X} voxels)", y=0.98)
fig.tight_layout()
fig.savefig(OUT / "fig_data.png", dpi=110, bbox_inches="tight")
plt.close(fig)
''')

code(r'''
def trail(node_id, length):
    """Positions of a node and up to `length` ancestors, newest first."""
    pts = []
    cur = node_id
    for _ in range(length + 1):
        k = index.get(cur)
        if k is None:
            break
        pts.append((Xc[k], Yc[k]))
        cur = parent.get(cur)
        if cur is None:
            break
    return np.array(pts)


def draw_tracks(ax, t, length=12):
    ax.imshow(mip[t], cmap="gray")
    here = np.where(T_ == t)[0]
    for k in here:
        p = trail(int(ids[k]), length)
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], "-", color="#35d0ff", lw=1.0, alpha=0.9)
    ax.scatter(Xc[here], Yc[here], s=9, c="#ffcc33", edgecolors="none")
    ax.set_xlim(0, X)
    ax.set_ylim(Y, 0)
    ax.axis("off")


# Figure 2: sparse ground-truth tracks on top of the image.
t_mid = int(np.bincount(T_).argmax())
fig, ax = plt.subplots(figsize=(6.5, 6.5))
draw_tracks(ax, t_mid)
ax.set_title(f"Ground truth at t = {t_mid}: annotated cells (yellow) and their last 12 steps (blue).\n"
             "Annotations are sparse: most visible cells carry no label.", fontsize=9)
fig.savefig(OUT / "fig_tracks.png", dpi=110, bbox_inches="tight")
plt.close(fig)
''')

code(r'''
# Figure 3: one division, the mother and its two daughters across five frames.
mothers = [m for m, ks in children.items() if len(ks) >= 2 and m in index]
mothers.sort(key=lambda m: abs(T_[index[m]] - T // 2))
m = mothers[0]
km = index[m]
tm = int(T_[km])
cy, cx = Yc[km], Xc[km]


def lineage_colour(nid):
    """Red for the mother and her ancestors, blue for her descendants."""
    cur, steps = nid, 0
    while cur is not None and steps < 5:
        if cur == m:
            return "#ff5a5a" if steps == 0 else "#35d0ff"
        cur = parent.get(cur)
        steps += 1
    cur = m
    for _ in range(5):
        cur = parent.get(cur)
        if cur == nid:
            return "#ff5a5a"
    return None


half = 28
y0, x0 = int(max(cy - half, 0)), int(max(cx - half, 0))
fig, axes = plt.subplots(1, 5, figsize=(13, 3.0))
for ax, dt in zip(axes, range(-2, 3)):
    t = min(max(tm + dt, 0), T - 1)
    ax.imshow(mip[t, y0:y0 + 2 * half, x0:x0 + 2 * half], cmap="gray")
    for k in np.where(T_ == t)[0]:
        c = lineage_colour(int(ids[k]))
        if c:
            ax.scatter([Xc[k] - x0], [Yc[k] - y0], s=40, c=c)
    ax.set_title(f"t = {t}" + ("  (division)" if t == tm else ""), fontsize=9)
    ax.axis("off")
fig.suptitle("A cell division: the mother (red) splits into two daughters (blue)", y=1.02)
fig.savefig(OUT / "fig_division.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("division at t =", tm, "children:", children[m])
''')

code(r'''
# Animation: 40 frames around the busiest time point, tracks drawn as they grow.
start = int(np.clip(t_mid - 20, 0, T - 40))
fig, ax = plt.subplots(figsize=(4.2, 4.2))
fig.subplots_adjust(0, 0, 1, 1)


def update(i):
    ax.clear()
    draw_tracks(ax, start + i, length=8)
    ax.text(6, 14, f"t = {start + i}", color="white", fontsize=9)


anim = animation.FuncAnimation(fig, update, frames=40)
anim.save(OUT / "tracks.gif", writer=animation.PillowWriter(fps=6), dpi=70)
plt.close(fig)
for f in sorted(OUT.glob("*.png")) + sorted(OUT.glob("*.gif")):
    print(f.name, round(f.stat().st_size / 1e6, 2), "MB")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "biohub-viz.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "ryotakawamura94/biohub-viz", "title": "biohub-viz", "code_file": "biohub-viz.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false",
        "enable_internet": "false", "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("built")
