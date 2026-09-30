"""Build the head-training notebook (CPU).

Trains the V1284 coordinate-refinement head on features captured by
`capture*.json` runs, validating by movie against the public head.

    python notebooks/headtrain/build.py <capture-kernel-slug> [<capture-kernel-slug> ...]
    python scripts/push_kernel.py -p notebooks/headtrain --run
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
captures = sys.argv[1:] or ["biohub-cap-smoke"]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# V1284 head: retrain on more movies

Pairs captured detector centres with ground-truth nodes, trains the same small head
(224 -> 32 -> 3, output bounded to 2 um), and compares it by held-out movie with the
public head. CPU only.
""")

code(r'''
import glob, subprocess, sys
wheel_dirs = sorted({p.rsplit("/", 1)[0] for p in glob.glob("/kaggle/input/**/wheels/zarr-*.whl", recursive=True)})
cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet"]
for d in wheel_dirs:
    cmd += ["--find-links", d]
cmd += ["zarr>=3", "donfig", "google-crc32c", "numcodecs", "typing-extensions"]
print(subprocess.run(cmd, capture_output=True, text=True).stderr[-300:])
''')

code(r'''
import io, json, pathlib, tarfile
import numpy as np
import torch
import zarr
from scipy.optimize import linear_sum_assignment

torch.manual_seed(0)
np.random.seed(0)
TRAIN = next(p for p in pathlib.Path("/kaggle/input").rglob("train") if p.is_dir() and any(p.glob("*.geff")))
TARS = sorted(pathlib.Path("/kaggle/input").rglob("capture.tar"))
HEAD_PUBLIC = sorted(pathlib.Path("/kaggle/input").rglob("biohub-v1284-head-s075/v1284_head.pt"))[0]
WORK = pathlib.Path("/kaggle/working")
print("train:", TRAIN, "| capture archives:", [str(t) for t in TARS])

frames = {}  # (movie, t) -> (coords, features)
for tar_path in TARS:
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".npz"):
                continue
            parts = pathlib.PurePosixPath(m.name).parts
            movie, t = parts[-2], int(parts[-1][:-4])
            z = np.load(io.BytesIO(tar.extractfile(m).read()))
            frames[(movie, t)] = (z["coords"], z["features"])
movies = sorted({k[0] for k in frames})
print(len(movies), "movies,", len(frames), "frames")
c0, f0 = next(iter(frames.values()))
print("coords", c0.shape, c0.dtype, c0[:3].tolist(), "| features", f0.shape)
''')

code(r'''
def load_gt(movie):
    g = zarr.open_group(str(TRAIN / f"{movie}.geff"), mode="r")
    p = {k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")}
    return p

VOX = np.array([1.625, 0.40625, 0.40625])
# The feature grid is isotropic at 1.625 um; which axis is which, and whether the
# first coordinate column is time, is checked against the ground truth below.
def det_um(coords, hyp):
    zyx = coords[:, 1:4].astype(np.float64)
    if hyp == "zyx_ds4":
        return zyx * 1.625
    if hyp == "zyx_ds4_center":
        return (zyx + np.array([0, 0.375, 0.375])) * 1.625
    if hyp == "zyx_full":
        return zyx * VOX
    raise ValueError(hyp)

def match(det, gt, max_um):
    if not len(det) or not len(gt):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= max_um, d, 1e6))
    keep = d[r, c] <= max_um
    return np.stack([r[keep], c[keep]], 1)

check = movies[:2]
for hyp in ("zyx_ds4", "zyx_ds4_center", "zyx_full"):
    n_pairs, dist = 0, []
    for movie in check:
        gt = load_gt(movie)
        gt_um = np.stack([gt["z"], gt["y"], gt["x"]], 1) * VOX
        for (mv, t), (coords, _) in frames.items():
            if mv != movie:
                continue
            sel = gt["t"] == t
            det = det_um(coords, hyp)
            pr = match(det, gt_um[sel], 4.0)
            n_pairs += len(pr)
            dist += list(np.linalg.norm(det[pr[:, 0]] - gt_um[sel][pr[:, 1]], axis=1))
    print(f"{hyp:16s} pairs {n_pairs:6d}  median dist {np.median(dist) if dist else float('nan'):.3f} um")
''')

code(r'''
HYP = "zyx_ds4"   # replaced below if another hypothesis matches clearly better
scores = {}
for hyp in ("zyx_ds4", "zyx_ds4_center", "zyx_full"):
    pairs, dist = 0, []
    for movie in movies[:2]:
        gt = load_gt(movie)
        gt_um = np.stack([gt["z"], gt["y"], gt["x"]], 1) * VOX
        for (mv, t), (coords, _) in frames.items():
            if mv == movie:
                sel = gt["t"] == t
                det = det_um(coords, hyp)
                pr = match(det, gt_um[sel], 4.0)
                pairs += len(pr)
                dist += list(np.linalg.norm(det[pr[:, 0]] - gt_um[sel][pr[:, 1]], axis=1))
    scores[hyp] = (pairs, float(np.median(dist)) if dist else 9.9)
HYP = max(scores, key=lambda h: (scores[h][0], -scores[h][1]))
print("best match", HYP, scores[HYP])
# The notebook writes ds coords * (1, 4, 4) with no centre offset, so the head
# must learn targets in that frame, whatever frame matches the GT best.
HYP = "zyx_ds4"
print("training in", HYP, scores[HYP])

X, Y, G = [], [], []
for mi, movie in enumerate(movies):
    gt = load_gt(movie)
    gt_um = np.stack([gt["z"], gt["y"], gt["x"]], 1) * VOX
    for (mv, t), (coords, feats) in frames.items():
        if mv != movie:
            continue
        sel = gt["t"] == t
        det = det_um(coords, HYP)
        pr = match(det, gt_um[sel], 3.0)
        if not len(pr):
            continue
        X.append(feats[pr[:, 0]]); Y.append(gt_um[sel][pr[:, 1]] - det[pr[:, 0]]); G += [mi] * len(pr)
X = np.concatenate(X).astype(np.float32); Y = np.concatenate(Y).astype(np.float32); G = np.array(G)
print("pairs:", len(X), "| mean |target| um:", float(np.linalg.norm(Y, axis=1).mean()))
''')

code(r'''
def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight); torch.nn.init.zeros_(head[-1].bias)
    return head

def bounded(head, x):
    d = head(x)
    return 2.0 * d / (1.0 + torch.linalg.vector_norm(d, dim=-1, keepdim=True))

def residual(pred, Y):
    return float(np.linalg.norm(Y - pred, axis=1).mean())

pub = torch.load(HEAD_PUBLIC, map_location="cpu", weights_only=True)
pub_head = make_head(); pub_head.load_state_dict(pub["state_dict"]); pub_head.eval()

def predict(head, mean, scale, Xs):
    with torch.no_grad():
        return bounded(head, (torch.as_tensor(Xs) - mean) / scale).numpy()

def train_head(Xtr, Ytr, epochs=300, wd=1e-4):
    mean = torch.as_tensor(Xtr.mean(0)); scale = torch.as_tensor(Xtr.std(0) + 1e-6)
    head = make_head(); opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=wd)
    xt = (torch.as_tensor(Xtr) - mean) / scale; yt = torch.as_tensor(Ytr)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 1024):
            b = perm[i:i + 1024]
            loss = torch.nn.functional.smooth_l1_loss(bounded(head, xt[b]), yt[b], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head, mean, scale

# Held-out by movie: 4 folds.
folds = np.arange(len(movies)) % 4
res = {"none": [], "public": [], "ours": []}
for k in range(4):
    va = np.isin(G, np.where(folds == k)[0]); tr = ~va
    if va.sum() == 0 or tr.sum() == 0:
        continue
    h, m, s = train_head(X[tr], Y[tr])
    res["none"].append(residual(np.zeros_like(Y[va]), Y[va]))
    res["public"].append(residual(predict(pub_head, pub["mean"], pub["scale"], X[va]), Y[va]))
    res["ours"].append(residual(predict(h, m, s, X[va]), Y[va]))
for k, v in res.items():
    print(f"{k:7s} mean residual {np.mean(v):.4f} um  (per fold {np.round(v, 4).tolist()})")
''')

code(r'''
# Final head on all pairs, saved in the format the notebook loads.
h, m, s = train_head(X, Y)
torch.save({"state_dict": h.state_dict(), "mean": m, "scale": s}, WORK / "v1284_head.pt")
json.dump({"movies": len(movies), "pairs": int(len(X)), "hypothesis": HYP,
           "cv_residual_um": {k: float(np.mean(v)) for k, v in res.items()}},
          open(WORK / "head_report.json", "w"), indent=2)
print(open(WORK / "head_report.json").read())
''')

code(r'''
# Seed ensemble packed into one wider head: hidden units side by side, output layer
# scaled by 1/K, so the raw outputs are averaged before the 2 um bound.
# mean/scale come from the same X for every seed, so they are shared.
K = 5
heads = []
for seed in range(K):
    torch.manual_seed(100 + seed)
    hk, m, s = train_head(X, Y)
    heads.append(hk)
W1 = torch.cat([h[0].weight.data for h in heads]); b1 = torch.cat([h[0].bias.data for h in heads])
W2 = torch.cat([h[2].weight.data for h in heads], 1) / K; b2 = sum(h[2].bias.data for h in heads) / K
torch.save({"state_dict": {"0.weight": W1, "0.bias": b1, "2.weight": W2, "2.bias": b2},
            "mean": m, "scale": s}, WORK / "v1284_head_ens5.pt")
wide = torch.nn.Sequential(torch.nn.Linear(224, 32 * K), torch.nn.SiLU(), torch.nn.Linear(32 * K, 3))
wide.load_state_dict({"0.weight": W1, "0.bias": b1, "2.weight": W2, "2.bias": b2}); wide.eval()
print("ensemble train residual", residual(predict(wide, m, s, X), Y), "| single", residual(predict(h, m, s, X), Y))
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "biohub-headtrain.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "ryotakawamura94/biohub-headtrain", "title": "biohub-headtrain", "code_file": "biohub-headtrain.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1", "anvithpothula/biohub-v1284-head-s075"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [f"ryotakawamura94/{c}" for c in captures], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("built with captures:", captures)
