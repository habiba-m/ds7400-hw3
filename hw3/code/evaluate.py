#!/usr/bin/env python3
"""Evaluate on all cells, at the shared probes, and make the z = 0 GIF / PNG.

  recon    encoder sees the CFD at the same t (autoencoder quality)
  rollout  encoder sees ONLY the first snapshot; the propagator steps the tokens through the
           whole inflow schedule (the end goal)

    python evaluate.py --config configs/pilot.json                       # recon + rollout on test runs
    python evaluate.py --config configs/pilot.json --runs B_m00_gust --modes recon
Writes <out>/eval/: metrics.csv, probe_metrics.csv, mesh_independence.json,
<run>_<mode>.gif, <run>_<mode>_last.png, <run>_nrmse_vs_time.png
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from parc3d import config, viz
from parc3d.viz import plt
from parc3d.core import FIELDS, encode_frame, load_ae, load_prop, nrmse, predict
from parc3d.data import Dataset
from parc3d.trainers import rollout

ap = config.parser(__doc__)
ap.add_argument("--ae", default=None, help="default <out>/ae/best.pt")
ap.add_argument("--dyn", default=None, help="default <out>/dyn/best.pt (rollout is skipped if missing)")
ap.add_argument("--runs", default="", help="comma list (default: test split)")
ap.add_argument("--modes", default="recon,rollout")
ap.add_argument("--no-gif", action="store_true")
ap.add_argument("--gif-frames", type=int, default=80, help="max frames per GIF")
ap.add_argument("--frame-stride", type=int, default=1, help="score every n-th snapshot (metrics only)")
args = ap.parse_args()
cfg = config.load(args.config, args.set)
out = Path(cfg["out"]) / "eval"
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae_path = Path(args.ae or Path(cfg["out"]) / "ae" / "best.pt")
ae, norm, ck = load_ae(ae_path, device)
print(f"autoencoder {ae_path} (step {ck['step']})", flush=True)
modes = [m for m in args.modes.split(",") if m]
prop = None
if "rollout" in modes:
    dyn_path = Path(args.dyn or Path(cfg["out"]) / "dyn" / "best.pt")
    if dyn_path.exists():
        prop, dck = load_prop(dyn_path, device)
        print(f"propagator {dyn_path} (step {dck['step']})", flush=True)
    else:
        print(f"no propagator at {dyn_path}: skipping rollout", flush=True)
        modes.remove("rollout")

ds = Dataset(cfg, device, normalizer=norm)
ids = [ds.index[n] for n in args.runs.split(",") if n] if args.runs else ds.splits["test"]
n_ctx = cfg["ae"]["n_context"][1]
probe_xyz = np.load(cfg["data"]["probes"]) if Path(cfg["data"]["probes"]).exists() else None


@torch.no_grad()
def tokens(i, mode, ks):
    """(len(ks), L, d) tokens at snapshots ks of run i (a rollout still steps through all of them)."""
    if mode == "recon":
        return torch.cat([encode_frame(ae, ds, i, k, n_ctx) for k in ks])
    z0 = encode_frame(ae, ds, i, 0, n_ctx)
    return rollout(prop, z0, ds.cond[i][None])[0][list(ks)]


rows, prow, probe_pred = [], [], {}
for i in ids:
    r = ds.runs[i]
    split = ds.split_name(i)
    sl = None if args.no_gif else viz.Slice(r.xyz)
    yaw = np.degrees(np.arctan2(r.fs[:, 0], r.fs[:, 1]))
    curves = {}
    ks = list(range(0, r.T, args.frame_stride))
    for mode in modes:
        Z = tokens(i, mode, ks)
        e = []
        for j, k in enumerate(ks):
            gt = r.frame(k)
            m = nrmse(predict(ae, norm, Z[j:j + 1], gt[:, :3], ds.fs[i][k]), gt[:, 3:7], norm.u_ref)
            e.append(m)
            rows.append(dict(run=r.name, split=split, mode=mode, k=k, t=float(r.times[k]),
                             **{f"nrmse_{f}": float(x) for f, x in zip(FIELDS, m)}))
        curves[mode] = np.array(e)
        print(f"{r.name} ({split}) {mode:7s}: mean NRMSE all cells " +
              " ".join(f"{f} {x:.4f}" for f, x in zip(FIELDS, curves[mode].mean(0))), flush=True)

        if probe_xyz is not None and r.probe_values is not None:
            pp = np.stack([predict(ae, norm, Z[j:j + 1], probe_xyz, ds.fs[i][k]) for j, k in enumerate(ks)])
            probe_pred[(r.name, mode)] = pp
            pm = np.mean([nrmse(pp[j], np.asarray(r.probe_values[k]), norm.u_ref) for j, k in enumerate(ks)], 0)
            prow.append(dict(run=r.name, split=split, mode=mode, **{f"nrmse_{f}": float(x) for f, x in zip(FIELDS, pm)}))

        if sl is not None:
            js = np.unique(np.linspace(0, len(ks) - 1, min(len(ks), args.gif_frames)).round().astype(int))
            frames = [ks[j] for j in js]
            P = np.stack([sl.grid(predict(ae, norm, Z[j:j + 1], sl.pts, ds.fs[i][ks[j]])) for j in js])
            G = np.stack([sl.cfd(r.frame(k)) for k in frames])
            what = ("reconstruction: encoder sees the CFD at each t" if mode == "recon"
                    else f"rollout from the CFD at t = {r.times[0] * 1000:.0f} ms only")
            titles = [f"{r.name} ({split}) · z = 0 · {what}\nt = {r.times[k] * 1000:.0f} ms · inflow "
                      f"{np.linalg.norm(r.fs[k]):.1f} m/s, yaw {yaw[k]:.1f}°" for k in frames]
            viz.save_gif(out / f"{r.name}_{mode}.gif", sl, P, G, titles)
            viz.save_png(out / f"{r.name}_{mode}_last.png", sl, P[-1], G[-1], titles[-1])
            print("  wrote", out / f"{r.name}_{mode}.gif", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    for j, f in enumerate(FIELDS):
        for mode, c in curves.items():
            axes[j].plot(r.times[ks], c[:, j], label=mode)
        axes[j].set_title(f); axes[j].set_xlabel("t [s]")
    axes[0].set_ylabel("NRMSE (all cells)"); axes[0].legend()
    fig.suptitle(f"{r.name} ({split})"); fig.tight_layout()
    fig.savefig(out / f"{r.name}_nrmse_vs_time.png", dpi=150); plt.close(fig)

for name, data in (("metrics.csv", rows), ("probe_metrics.csv", prow)):
    if data:
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(data[0])); w.writeheader(); w.writerows(data)

# mesh independence: same flow, different input mesh, predictions compared at identical probes
mi = []
for (na, mode), pa in probe_pred.items():
    for (nb, mb), pb in probe_pred.items():
        ra, rb = ds.runs[ds.index[na]], ds.runs[ds.index[nb]]
        if mode == mb and na < nb and ra.flow == rb.flow and ra.mesh != rb.mesh and pa.shape == pb.shape:
            ref = np.array([norm.u_ref] * 3 + [0.5 * norm.u_ref ** 2])
            d = np.sqrt(np.nanmean((pa - pb) ** 2, axis=(0, 1))) / ref
            mi.append(dict(a=na, b=nb, mode=mode, pred_diff_rms_over_ref=dict(zip(FIELDS, d.round(5).tolist()))))
(out / "mesh_independence.json").write_text(json.dumps(mi, indent=1))
print("\nresults in", out, flush=True)
