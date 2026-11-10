#!/usr/bin/env python3
"""Stage 2: train the latent propagator on top of a frozen autoencoder (rollout).

    python train_dyn.py --config configs/pilot.json            # uses <out>/ae/best.pt
Writes <out>/dyn/{best.pt, last.pt, log.csv}. best.pt = lowest NRMSE of a FULL rollout of
the val runs from their first snapshot, on all cells.
"""
from pathlib import Path

import torch

from parc3d import config
from parc3d.core import load_ae
from parc3d.data import Dataset
from parc3d.trainers import train_propagator

ap = config.parser(__doc__)
ap.add_argument("--ae", default=None, help="autoencoder checkpoint (default <out>/ae/best.pt)")
args = ap.parse_args()
cfg = config.load(args.config, args.set)
out = Path(cfg["out"]) / "dyn"
config.save(cfg, out / "config.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

ae_path = Path(args.ae or Path(cfg["out"]) / "ae" / "best.pt")
ae, norm, ck = load_ae(ae_path, device)
print(f"autoencoder {ae_path} (step {ck['step']}, val NRMSE {ck['val_nrmse']:.4f})", flush=True)
ds = Dataset(cfg, device, normalizer=norm)              # same normalization as the autoencoder
train = [i for i in ds.splits["train"] if ds.runs[i].T > 1]
val = [i for i in ds.splits["val"] if ds.runs[i].T > 1]
if not train or not val:
    raise SystemExit("propagator needs transient (Dataset B) runs in both train and val")
train_propagator(cfg, ds, ae, train, val, out)
