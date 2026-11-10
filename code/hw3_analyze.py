#!/usr/bin/env python3
"""HW3 analysis: collect the supernode_alpha experiments into tables and figures.

    python hw3_analyze.py --config configs/hw3.json --root /scratch/crv4jj/parc3d_v2/runs/hw3

Reads <root>/alpha_*/ae/log.csv, <root>/alpha_*/eval/{metrics.csv, mesh_independence.json} and the
Slurm logs, and writes <root>/summary/:
  results.md / results.json   test NRMSE by group, unseen-mesh gap, mesh agreement, cost
  training_curves.png         reconstruction loss and val NRMSE vs step
  mesh_gap.png                seen-mesh vs never-seen-mesh NRMSE per alpha
  supernodes.png              where the 1024 supernodes land on each mesh (the mechanism)
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from parc3d import config
from parc3d.model import Encoder
from parc3d.viz import plt

FIELDS = ("u", "v", "w", "p")
HELD = ("s03_ramp", "s11_yawstep", "s19_gust")
GROUPS = {
    "seen mesh m00, new schedules": [f"B_m00_{s}" for s in HELD],
    "never-seen mesh m02, new schedules": [f"B_m02_{s}" for s in HELD],
    "never-seen mesh m02, training schedules": [f"B_m02_{s}" for s in ("s00_ramp", "s08_yawstep", "s16_gust")],
}
SEEN, UNSEEN = list(GROUPS)[:2]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default="configs/hw3.json")
ap.add_argument("--root", default="/scratch/crv4jj/parc3d_v2/runs/hw3")
ap.add_argument("--logs", default="logs")
ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
args = ap.parse_args()
cfg = config.load(args.config, args.set)
root = Path(args.root)
out = root / "summary"
out.mkdir(parents=True, exist_ok=True)
runs = sorted(root.glob("alpha_*"), key=lambda p: float(p.name.split("_")[1]))
alphas = [float(p.name.split("_")[1]) for p in runs]
print("experiments:", [p.name for p in runs])


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ---------------- test metrics ----------------
res = {}
for a, d in zip(alphas, runs):
    r = dict(groups={}, gap={})
    rows = read_csv(d / "eval" / "metrics.csv")
    for g, names in GROUPS.items():
        sel = [x for x in rows if x["run"] in names]
        r["groups"][g] = {f: float(np.mean([float(x[f"nrmse_{f}"]) for x in sel])) for f in FIELDS}
    for f in FIELDS:
        r["gap"][f] = r["groups"][UNSEEN][f] / r["groups"][SEEN][f]
    mi = json.loads((d / "eval" / "mesh_independence.json").read_text())
    pairs = [m for m in mi if m["a"].startswith("B_m00") and m["b"].startswith("B_m02")]
    r["mesh_agreement_m00_vs_m02"] = {f: float(np.mean([m["pred_diff_rms_over_ref"][f] for m in pairs]))
                                     for f in FIELDS} if pairs else None
    log = read_csv(d / "ae" / "log.csv")
    best = min(log, key=lambda x: float(x["val_nrmse"]))
    r["best_step"], r["best_val_nrmse"] = int(best["step"]), float(best["val_nrmse"])
    r["final_rec_loss"] = float(log[-1]["rec"])
    r["log"] = log
    # cost, from the Slurm log of this alpha
    r["s_per_step"] = r["train_minutes"] = None
    for lf in sorted(Path(args.logs).glob(f"hw3_a{d.name.split('_')[1]}-*.out")):
        s = lf.read_text(errors="replace")
        sp = re.findall(r"([\d.]+) s/step", s)
        tm = re.findall(r"autoencoder done in ([\d.]+) min", s)
        if sp:
            r["s_per_step"] = float(np.median([float(x) for x in sp]))
        if tm:
            r["train_minutes"] = float(tm[-1])
    res[a] = r

# ---------------- mechanism: where do the supernodes land on each mesh? ----------------
lo, hi = np.array(cfg["physics"]["bounds"], np.float32).T
semi = np.array([0.20, 0.50, 0.15], np.float32)
data_root = Path(cfg["data"]["root"])
meshes = ("m00", "m01", "m02", "m03")
n_ctx, S = cfg["ae"]["n_context"][1], cfg["model"]["n_supernodes"]
mech = {"cells": {}, "near_cells": {}, "near_super": {a: {} for a in alphas}}
rng = np.random.default_rng(0)
for m in meshes:
    xyz = np.load(data_root / f"B_{m}_s00_ramp" / "xyz.npy").astype(np.float32)
    rho = np.sqrt((((xyz) / semi) ** 2).sum(1))
    mech["cells"][m] = int(len(xyz))
    mech["near_cells"][m] = float((rho < 3).mean())
    for a in alphas:
        enc = Encoder({**cfg["model"], "supernode_alpha": a}).eval()
        fr = []
        for rep in range(5):
            pts = xyz[rng.choice(len(xyz), n_ctx, replace=False)]
            xn = torch.from_numpy(2 * (pts - lo) / (hi - lo) - 1)[None]
            xs = xn[0, :S] if a == 0 else enc.pick_supernodes(xn, S)[0]
            ps = lo + (xs.numpy() + 1) / 2 * (hi - lo)
            fr.append(float((np.sqrt(((ps / semi) ** 2).sum(1)) < 3).mean()))
        mech["near_super"][a][m] = float(np.mean(fr))
for a in alphas:
    v = np.array([mech["near_super"][a][m] for m in meshes])
    res[a]["near_body_supernodes"] = mech["near_super"][a]
    res[a]["near_body_spread_across_meshes"] = float(v.max() - v.min())

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a in alphas:
    log = res[a]["log"]
    st = [int(x["step"]) for x in log]
    ax[0].plot(st, [float(x["rec"]) for x in log], label=f"alpha = {a:g}")
    ax[1].plot(st, [float(x["val_nrmse"]) for x in log], label=f"alpha = {a:g}")
ax[0].set_yscale("log"); ax[0].set_title("training reconstruction loss (MSE, normalized)")
ax[1].set_title("validation NRMSE (all cells, mesh m01, new schedules)")
for x in ax:
    x.set_xlabel("step"); x.legend()
fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False)
w = 0.8 / len(GROUPS)
for j, f in enumerate(FIELDS):
    for gi, g in enumerate(GROUPS):
        ax[j].bar(np.arange(len(alphas)) + (gi - 1) * w, [res[a]["groups"][g][f] for a in alphas], w, label=g)
    ax[j].set_xticks(range(len(alphas)), [f"α={a:g}" for a in alphas]); ax[j].set_title(f"{f} NRMSE")
ax[0].legend(fontsize=7)
fig.suptitle("Test reconstruction error: seen mesh vs never-seen mesh"); fig.tight_layout()
fig.savefig(out / "mesh_gap.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
w = 0.8 / (len(alphas) + 1)
ax.bar(np.arange(4) - 0.4 + w / 2, [mech["near_cells"][m] for m in meshes], w, label="all cells", color="0.6")
for ai, a in enumerate(alphas):
    ax.bar(np.arange(4) - 0.4 + w * (ai + 1.5), [mech["near_super"][a][m] for m in meshes], w,
           label=f"supernodes, α = {a:g}")
ax.set_xticks(range(4), [f"{m}\n{mech['cells'][m]:,} cells" for m in meshes])
ax.set_ylabel("fraction within 3 body radii"); ax.legend(fontsize=8)
ax.set_title("Where the 1024 supernodes land on each mesh")
fig.tight_layout(); fig.savefig(out / "supernodes.png", dpi=150); plt.close(fig)

# ---------------- tables ----------------
lines = ["# HW3 results: density-tempered supernode sampling", "",
         "Test NRMSE on every cell (every 10th snapshot). Lower is better.", ""]
lines += ["| α | group | u | v | w | p |", "|---|---|---|---|---|---|"]
for a in alphas:
    for g in GROUPS:
        lines.append(f"| {a:g} | {g} | " + " | ".join(f"{res[a]['groups'][g][f]:.4f}" for f in FIELDS) + " |")
lines += ["", "Unseen-mesh gap = NRMSE(m02) / NRMSE(m00) on the same new schedules (1.0 = no gap).", "",
          "| α | u | v | w | p | best val NRMSE (step) | near-body supernode spread across meshes | s/step | train min |",
          "|---|---|---|---|---|---|---|---|---|"]
for a in alphas:
    r = res[a]
    lines.append(f"| {a:g} | " + " | ".join(f"{r['gap'][f]:.2f}" for f in FIELDS)
                 + f" | {r['best_val_nrmse']:.4f} ({r['best_step']}) | {r['near_body_spread_across_meshes']:.3f}"
                 + f" | {r['s_per_step'] or float('nan'):.3f} | {r['train_minutes'] or float('nan'):.1f} |")
if all(res[a]["mesh_agreement_m00_vs_m02"] for a in alphas):
    lines += ["", "Prediction agreement between m00 and m02 inputs at the shared probes "
              "(RMS difference / reference scale; lower = more mesh-independent).", "",
              "| α | u | v | w | p |", "|---|---|---|---|---|"]
    for a in alphas:
        lines.append(f"| {a:g} | " + " | ".join(f"{res[a]['mesh_agreement_m00_vs_m02'][f]:.4f}" for f in FIELDS) + " |")
lines += ["", "Cells per mesh and fraction within 3 body radii: " +
          ", ".join(f"{m} {mech['cells'][m]:,} ({mech['near_cells'][m]:.3f})" for m in meshes)]
(out / "results.md").write_text("\n".join(lines) + "\n")
for a in alphas:
    res[a].pop("log")
(out / "results.json").write_text(json.dumps({"alphas": res, "mechanism": mech}, indent=1, default=str))
print("\n".join(lines))
print("\nwrote", out)
