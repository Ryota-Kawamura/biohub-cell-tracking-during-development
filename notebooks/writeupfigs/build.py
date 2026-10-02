"""Build the notebook that extracts real-data material for the write-up figures.

It retrains the coordinate head by held-out movie (same code and folds as
notebooks/headtrain), then saves a small figdata.npz: image crops, detections before
and after each head, ground truth and per-movie errors. The figures themselves are
drawn locally by scripts/plot_realdata.py, so styling never needs a Kaggle rerun.

    python notebooks/writeupfigs/build.py
    python scripts/push_kernel.py -p notebooks/writeupfigs --run     # CPU only
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
# Biohub 181st place: real-data material for the write-up figures

Retrains the coordinate-refinement head by held-out movie (4 folds, as in `biohub-headtrain`)
and saves `figdata.npz`: max-projection crops of a few training movies, the detector centres
before and after the public and the retrained head, the ground truth, and per-movie errors.
CPU only.
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
import io, json, pathlib, tarfile, time
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
if len(TARS) < 2:
    # The capture output is in Version 1 of each capture notebook; later versions only
    # added a description and have no output.
    # Attaching another version inside a batch session is refused, so download it over HTTP.
    import os
    os.environ["DISABLE_KAGGLE_CACHE"] = "true"
    import kagglehub
    def fetch(slug, tries=4):
        for i in range(tries):
            try:
                return pathlib.Path(kagglehub.notebook_output_download(
                    f"ryotakawamura94/{slug}/versions/1", path="capture.tar", force_download=i > 0))
            except Exception as e:   # large downloads sometimes end early
                print(slug, "attempt", i + 1, "failed:", type(e).__name__)
        raise RuntimeError(slug)
    TARS = [fetch(s) for s in ("biohub-cap-a", "biohub-cap-b")]
print("train:", TRAIN, "| capture archives:", [str(t) for t in TARS])
assert len(TARS) == 2, "both capture archives are needed"

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
''')

code(r'''
VOX = np.array([1.625, 0.40625, 0.40625])
DS = 1.625                       # feature grid spacing in um, all three axes
CENTRE = np.array([0, 0.375, 0.375]) * DS   # voxel-centre offset (1.5 full-res px in y, x)

def load_gt(movie):
    g = zarr.open_group(str(TRAIN / f"{movie}.geff"), mode="r")
    p = {k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")}
    p["ids"] = np.asarray(g["nodes/ids"][:])
    p["edges"] = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    return p

def det_um(coords):
    # The pipeline writes ds coords * (1, 4, 4) with no centre offset.
    return coords[:, 1:4].astype(np.float64) * DS

def match(det, gt, max_um):
    if not len(det) or not len(gt):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= max_um, d, 1e6))
    keep = d[r, c] <= max_um
    return np.stack([r[keep], c[keep]], 1)

GT = {mv: load_gt(mv) for mv in movies}
X, Y, G, T_ = [], [], [], []
for mi, movie in enumerate(movies):
    gt = GT[movie]
    gt_um = np.stack([gt["z"], gt["y"], gt["x"]], 1) * VOX
    for (mv, t), (coords, feats) in frames.items():
        if mv != movie:
            continue
        sel = gt["t"] == t
        det = det_um(coords)
        pr = match(det, gt_um[sel], 3.0)
        if not len(pr):
            continue
        X.append(feats[pr[:, 0]]); Y.append(gt_um[sel][pr[:, 1]] - det[pr[:, 0]])
        G += [mi] * len(pr); T_ += [t] * len(pr)
X = np.concatenate(X).astype(np.float32); Y = np.concatenate(Y).astype(np.float32)
G = np.array(G); T_ = np.array(T_)
print("pairs:", len(X))
''')

code(r'''
def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight); torch.nn.init.zeros_(head[-1].bias)
    return head

def bounded(head, x):
    d = head(x)
    return 2.0 * d / (1.0 + torch.linalg.vector_norm(d, dim=-1, keepdim=True))

pub = torch.load(HEAD_PUBLIC, map_location="cpu", weights_only=True)
pub_head = make_head(); pub_head.load_state_dict(pub["state_dict"]); pub_head.eval()

def predict(head, mean, scale, Xs):
    with torch.no_grad():
        return bounded(head, (torch.as_tensor(np.asarray(Xs, np.float32)) - mean) / scale).numpy()

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

folds = np.arange(len(movies)) % 4
fold_heads = {}
P_pub = predict(pub_head, pub["mean"], pub["scale"], X)
P_ours = np.zeros_like(Y)
for k in range(4):
    t0 = time.time()
    va = np.isin(G, np.where(folds == k)[0])
    fold_heads[k] = train_head(X[~va], Y[~va])
    P_ours[va] = predict(*fold_heads[k], X[va])
    print(f"fold {k}: {time.time() - t0:.0f}s")

err = {"none": np.linalg.norm(Y, axis=1),
       "public": np.linalg.norm(Y - P_pub, axis=1),
       "ours": np.linalg.norm(Y - P_ours, axis=1)}
for k, v in err.items():
    print(f"{k:7s} held-out mean error {v.mean():.4f} um")
per_movie = {k: np.array([v[G == i].mean() for i in range(len(movies))]) for k, v in err.items()}
print("movies where the retrained head is better:", int((per_movie["ours"] < per_movie["public"]).sum()), "/", len(movies))
''')

code(r'''
# Example movies: the frames with the most annotated cells, one movie per fold.
def busiest(movie):
    t, n = np.unique(GT[movie]["t"], return_counts=True)
    ok = (t >= 12) & (t <= t.max() - 3)
    return int(t[ok][n[ok].argmax()]), int(n[ok].max())

cands = sorted(movies, key=lambda mv: -busiest(mv)[1])
examples = []
for k in range(4):
    mv = next(m for m in cands if folds[movies.index(m)] == k)
    examples.append(mv)
print("examples:", [(mv, busiest(mv)) for mv in examples])

out = {"movies": np.array(movies), "per_movie_none": per_movie["none"],
       "per_movie_public": per_movie["public"], "per_movie_ours": per_movie["ours"],
       "pair_err": np.stack([err["none"], err["public"], err["ours"]], 1).astype(np.float32),
       "targets": Y.astype(np.float32), "pred_public": P_pub.astype(np.float32),
       "pred_ours": P_ours.astype(np.float32), "pair_movie": G, "pair_t": T_,
       "centre_offset_um": CENTRE, "examples": np.array(examples)}

for j, mv in enumerate(examples):
    tc, _ = busiest(mv)
    vol = zarr.open_array(str(TRAIN / f"{mv}.zarr" / "0"), mode="r")
    ts = list(range(max(tc - 10, 0), min(tc + 3, vol.shape[0] - 1) + 1))
    mip = np.stack([np.asarray(vol[t]).max(axis=0) for t in ts]).astype(np.float32)
    lo, hi = np.percentile(mip, [1, 99.7])
    out[f"ex{j}_mip"] = np.clip((mip - lo) / (hi - lo), 0, 1).astype(np.float16)
    out[f"ex{j}_ts"] = np.array(ts); out[f"ex{j}_tc"] = tc
    out[f"ex{j}_shape"] = np.array(vol.shape)
    head_k = fold_heads[folds[movies.index(mv)]]
    rows = []   # t, det zyx (um), public zyx (um), ours zyx (um)
    for t in ts:
        coords, feats = frames[(mv, t)]
        det = det_um(coords)
        rows.append(np.column_stack([np.full(len(det), t), det,
                                     det + predict(pub_head, pub["mean"], pub["scale"], feats),
                                     det + predict(*head_k, feats)]))
    out[f"ex{j}_det"] = np.concatenate(rows).astype(np.float32)
    gt = GT[mv]
    out[f"ex{j}_gt"] = np.column_stack([gt["ids"], gt["t"], np.stack([gt["z"], gt["y"], gt["x"]], 1) * VOX])
    out[f"ex{j}_edges"] = gt["edges"]

np.savez_compressed(WORK / "figdata.npz", **out)
json.dump({"pairs": int(len(X)), "movies": len(movies),
           "heldout_mean_um": {k: float(v.mean()) for k, v in err.items()},
           "movies_better": int((per_movie["ours"] < per_movie["public"]).sum()),
           "examples": examples}, open(WORK / "figdata.json", "w"), indent=2)
print(open(WORK / "figdata.json").read())
print(round((WORK / "figdata.npz").stat().st_size / 1e6, 2), "MB")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "biohub-writeupfigs.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", HERE / "biohub-writeupfigs.ipynb")
