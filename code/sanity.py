#!/usr/bin/env python3
"""Gates that must pass before any real training run.

  1. snapshot    fit ONE CFD snapshot; error on all cells must be small
  2. consistency the training loss and the plotting/eval path must give the same error
                 on the same points (catches train/eval mismatches)
  3. trajectory  fit every snapshot of ONE run; error on all cells must be small

Writes <out>/sanity/{summary.json, snapshot.png, trajectory.gif} and exits 1 if a gate fails.

    python sanity.py --config configs/pilot.json
    python sanity.py --config configs/pilot.json --set sanity.snapshot_steps=500   # quick look
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from parc3d import config, viz
from parc3d.core import FIELDS, encode_frame, evaluate_frames, field_loss, nrmse, predict
from parc3d.data import Dataset
from parc3d.trainers import train_autoencoder

args = config.parser(__doc__).parse_args()
cfg = config.load(args.config, args.set)
s = cfg["sanity"]
out = Path(cfg["out"]) / "sanity"
out.mkdir(parents=True, exist_ok=True)
config.save(cfg, out / "config.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

ds = Dataset(cfg, device, cases=[s["run"]], need_splits=False)
run = ds.runs[0]
k = s["frame"] % run.T
n_ctx = cfg["ae"]["n_context"][1]
summary = dict(run=run.name, cells=run.N, snapshots=run.T)
fmt = lambda v: " ".join(f"{f} {x:.4f}" for f, x in zip(FIELDS, v))

# ---------------- gate 1: one snapshot ----------------
print(f"\n=== GATE 1: fit snapshot t = {run.times[k]:.3f} s of {run.name} ===", flush=True)
model, _ = train_autoencoder(cfg, ds, [(0, k)], [(0, k)], out / "snapshot",
                             steps=s["snapshot_steps"], eval_every=max(1, s["snapshot_steps"] // 5))
model.eval()
gt = run.frame(k)
z = encode_frame(model, ds, 0, k, n_ctx)
pred = predict(model, ds.norm, z, gt[:, :3], ds.fs[0][k])
e1 = nrmse(pred, gt[:, 3:7], ds.norm.u_ref)
g1 = bool(e1.mean() < s["snapshot_pass"])
summary["gate1_snapshot"] = dict(nrmse=e1.round(4).tolist(), mean=float(e1.mean()), threshold=s["snapshot_pass"], passed=g1)
print(f"GATE 1 {'PASS' if g1 else 'FAIL'}: NRMSE all {run.N:,} cells: {fmt(e1)} | mean {e1.mean():.4f} "
      f"(threshold {s['snapshot_pass']})", flush=True)
sl = viz.Slice(run.xyz)
viz.save_png(out / "snapshot.png", sl, sl.grid(predict(model, ds.norm, z, sl.pts, ds.fs[0][k])), sl.cfd(gt),
             f"Sanity gate 1 · fit to this one snapshot · {run.name} · t = {run.times[k] * 1000:.0f} ms · z = 0")

# ---------------- gate 2: training loss == eval path ----------------
print("\n=== GATE 2: training loss vs evaluation path on identical points ===", flush=True)
with torch.no_grad():
    gen = torch.Generator(device=device).manual_seed(123)
    q = ds.query(0, k, 8192, gen)[None]
    loss_train_path = field_loss(model, ds.norm, z, q, ds.fs[0][k:k + 1]).item()
    p_eval = torch.from_numpy(predict(model, ds.norm, z, q[0, :, :3].cpu().numpy(), ds.fs[0][k])).to(device)
    r_eval = ds.norm.fields(p_eval, ds.fs[0][k])
    r_true = ds.norm.fields(q[0, :, 3:7], ds.fs[0][k])
    loss_eval_path = torch.mean((r_eval - r_true) ** 2).item()
rel = abs(loss_train_path - loss_eval_path) / max(loss_train_path, 1e-12)
g2 = bool(rel < 1e-3)
summary["gate2_consistency"] = dict(train_path=loss_train_path, eval_path=loss_eval_path, rel_diff=rel, passed=g2)
print(f"GATE 2 {'PASS' if g2 else 'FAIL'}: train-path MSE {loss_train_path:.6f}  eval-path MSE "
      f"{loss_eval_path:.6f}  (rel diff {rel:.2e})", flush=True)

# ---------------- gate 3: one trajectory ----------------
print(f"\n=== GATE 3: fit all {run.T} snapshots of {run.name} ===", flush=True)
items = [(0, kk) for kk in range(run.T)]
mon = [(0, int(kk)) for kk in np.unique(np.linspace(0, run.T - 1, 4).round().astype(int))]
model, _ = train_autoencoder(cfg, ds, items, mon, out / "trajectory",
                             steps=s["trajectory_steps"], eval_every=max(1, s["trajectory_steps"] // 5))
model.eval()
rows = evaluate_frames(model, ds, mon, n_ctx)
e3 = np.mean([r["nrmse"] for r in rows], 0)
g3 = bool(e3.mean() < s["trajectory_pass"])
summary["gate3_trajectory"] = dict(nrmse=e3.round(4).tolist(), mean=float(e3.mean()),
                                   threshold=s["trajectory_pass"], passed=g3)
print(f"GATE 3 {'PASS' if g3 else 'FAIL'}: mean NRMSE over 4 snapshots, all cells: {fmt(e3)} | mean "
      f"{e3.mean():.4f} (threshold {s['trajectory_pass']})", flush=True)

frames = np.unique(np.linspace(0, run.T - 1, min(run.T, 60)).round().astype(int))
P = np.stack([sl.grid(predict(model, ds.norm, encode_frame(model, ds, 0, kk, n_ctx), sl.pts, ds.fs[0][kk]))
              for kk in frames])
G = np.stack([sl.cfd(run.frame(kk)) for kk in frames])
yaw = np.degrees(np.arctan2(run.fs[:, 0], run.fs[:, 1]))
viz.save_gif(out / "trajectory.gif", sl, P, G,
             [f"Sanity gate 3 · fit to this run · {run.name} · reconstruction · z = 0\n"
              f"t = {run.times[kk] * 1000:.0f} ms · inflow {np.linalg.norm(run.fs[kk]):.1f} m/s, yaw {yaw[kk]:.1f}°"
              for kk in frames])

(out / "summary.json").write_text(json.dumps(summary, indent=1))
ok = g1 and g2 and g3
print(f"\nSANITY {'PASSED' if ok else 'FAILED'} -> {out}", flush=True)
sys.exit(0 if ok else 1)
